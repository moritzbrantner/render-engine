// One walk of the scene a frame, and from it the nodes each pass hands three.js (#168).
//
// `renderer.render(scene, camera)` walks the whole scene graph (`projectObject`): every node,
// drawn or not, is visited, layer-tested and, when it draws, frustum-tested and brought up to
// date, once per sun cascade, per spot or point-light view and per main draw. Here the frame walks
// the graph once and keeps what draws; a pass then gives three, as the scene's children, only the
// nodes that draw something in it: those with a visible material (a shadow view gives everything
// that casts nothing there an invisible one, the main pass's opaque and blended halves hide each
// other's materials) whose bounding sphere, cached by the walk, meets the pass's frustum. three
// stays the drawer and still tests, sorts and draws what it is given, so a pass's CPU cost follows
// what it draws, not the size of the graph.
//
// What three decides per camera stays three's: an `LOD` and a `Group` with a render order are
// listed whole, with their subtrees, and walked by three as before. A listed node's own children
// are listed themselves, so for the frame's passes it has none (`enter`, `leave`).

import { Frustum, Line, Matrix4, Mesh, Points, WebGLCoordinateSystem } from 'three';

const NO_CHILDREN = Object.freeze([]);

// Kinds of listed node.
const DRAWN = 0;
const LIGHT = 1;
const TREE = 2;

// three's own frustum tests, by the bounding sphere `sphereOf` reads.
const SPHERE_TESTS = new Set([Mesh.prototype.intersectsFrustum, Line.prototype.intersectsFrustum, Points.prototype.intersectsFrustum]);

/** Whether any of a node's materials draws now. */
function draws(material) {
  if (!Array.isArray(material)) return material ? material.visible : true;
  for (let i = 0; i < material.length; i++) if (material[i]?.visible) return true;
  return false;
}

export class PassLists {
  /** How many nodes the last `walk` visited (a hidden node counts, what is below it does not). */
  nodes = 0;
  /** Every node with a material and that material, in walk order. */
  materials = [];
  /** The visible InstancedMeshes that draw an instance and cast (`ShadowCulling`). */
  instanced = [];
  /** The visible BatchedMeshes that cast (`ShadowCulling`). */
  batched = [];

  #objects = [];
  #kinds = [];
  // Per listed node its world bounding sphere (x, y, z, radius); radius -1 without one.
  #spheres = new Float64Array(0);
  // Per listed node whether a shadow view draws it with another geometry (`userData.shadowGeometry`).
  #proxied = [];
  #count = 0;
  // The scene's own children, and whether its passes take them as they are.
  #children = null;
  #whole = false;
  // Listed nodes with children: [node, its children].
  #aside = [];
  #list = [];
  #frustum = new Frustum();
  #clip = new Matrix4();
  #planes = new Float64Array(24);
  #notCaster = () => false;
  #all = false;
  #material = () => {};

  /**
   * Walks `scene` once: `material(node, its material)` runs for each visited node that has one,
   * and the visible ones that draw are kept for `list`. A hidden node's subtree draws nothing and
   * is not visited (#177), except below an `LOD`, which shows and hides its levels per camera;
   * with `all` every node is (`Lighting.prepare` compiles hidden objects' materials too).
   * `notCaster(node)` says which InstancedMeshes and BatchedMeshes cast nothing.
   */
  walk(scene, material, notCaster, all = false) {
    this.#all = all;
    this.nodes = 0;
    this.materials.length = 0;
    this.instanced.length = 0;
    this.batched.length = 0;
    this.#objects.length = 0;
    this.#kinds.length = 0;
    this.#proxied.length = 0;
    this.#count = 0;
    this.#material = material;
    this.#notCaster = notCaster;
    // An LOD shows and hides its own children per camera.
    this.#whole = scene.isLOD === true;
    this.#visit(scene, true, true, true);
    if (this.#spheres.length < this.#count * 4) this.#spheres = new Float64Array(this.#count * 8);
    for (let i = 0; i < this.#count; i++) this.#sphereOf(i);
  }

  #visit(node, visible, tree, root) {
    this.nodes++;
    visible &&= node.visible !== false;
    if (!visible && !tree && !this.#all) return;
    const material = node.material;
    if (material) {
      this.materials.push([node, material]);
      this.#material(node, material);
      if (visible && node.isInstancedMesh && node.count > 0 && !this.#notCaster(node)) this.instanced.push(node);
      if (visible && node.isBatchedMesh && !this.#notCaster(node)) this.batched.push(node);
    }
    // Most nodes are plain ones (transforms, bones): three's own `type` settles them at once.
    const plain = !material && (node.type === 'Object3D' || node.type === 'Bone');
    if (visible && !tree && !root && !plain) {
      // A render order on a Group reaches its subtree where the Group's layers meet the camera's.
      const kind =
        node.isLOD || (node.isGroup && node.renderOrder !== 0)
          ? TREE
          : node.isLight
            ? LIGHT
            : node.isMesh || node.isLine || node.isPoints || node.isSprite
              ? DRAWN
              : -1;
      if (kind >= 0) {
        this.#objects.push(node);
        this.#kinds.push(kind);
        this.#proxied.push(kind === DRAWN && Boolean(node.userData.shadowGeometry));
        this.#count++;
        tree = kind === TREE;
      }
    }
    const children = node.children;
    const inside = root ? this.#whole : tree;
    for (let i = 0; i < children.length; i++) this.#visit(children[i], visible, inside, false);
  }

  // The sphere three's frustum test gives node `i` (`Frustum.intersectsObject`), a little larger;
  // none for what three tests otherwise or has yet to compute.
  #sphereOf(i) {
    const o = this.#objects[i];
    const s = this.#spheres;
    s[i * 4 + 3] = -1;
    if (this.#kinds[i] !== DRAWN || !o.frustumCulled || !SPHERE_TESTS.has(o.intersectsFrustum)) return;
    const sphere = o.boundingSphere !== undefined ? o.boundingSphere : o.geometry?.boundingSphere;
    if (!sphere) return;
    const m = o.matrixWorld.elements;
    const { x, y, z } = sphere.center;
    s[i * 4] = m[0] * x + m[4] * y + m[8] * z + m[12];
    s[i * 4 + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    s[i * 4 + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
    const scale = Math.sqrt(
      Math.max(m[0] * m[0] + m[1] * m[1] + m[2] * m[2], m[4] * m[4] + m[5] * m[5] + m[6] * m[6], m[8] * m[8] + m[9] * m[9] + m[10] * m[10]),
    );
    s[i * 4 + 3] = sphere.radius * scale * 1.000001 + 1e-6;
  }

  /** Before the frame's passes: the listed nodes give up their children, which are listed too. */
  enter(scene) {
    this.#children = scene.children;
    if (this.#whole) return;
    for (let i = 0; i < this.#count; i++) {
      const o = this.#objects[i];
      if (this.#kinds[i] === TREE || o.children.length === 0) continue;
      this.#aside.push([o, o.children]);
      o.children = NO_CHILDREN;
    }
  }

  /** After the passes: the scene and every listed node have their children back. */
  leave(scene) {
    if (this.#children) scene.children = this.#children;
    this.#children = null;
    for (const [o, children] of this.#aside) o.children = children;
    this.#aside.length = 0;
  }

  /**
   * The children `scene` takes for a pass through `camera` (its matrices set), between `enter`
   * and `leave`: the listed nodes that draw now and meet its frustum, then `extra`'s. A `shadow`
   * view takes no three.js lights (its materials read none) and tests no node drawn with a
   * shadow proxy, whose sphere may differ.
   */
  list(camera, shadow, extra = null) {
    if (this.#whole) {
      if (!extra) return this.#children;
      return [...this.#children, extra];
    }
    this.#clip.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.#frustum.setFromProjectionMatrix(this.#clip, WebGLCoordinateSystem, camera.reversedDepth);
    const p = this.#planes;
    this.#frustum.planes.forEach((plane, k) => {
      p[k * 4] = plane.normal.x;
      p[k * 4 + 1] = plane.normal.y;
      p[k * 4 + 2] = plane.normal.z;
      p[k * 4 + 3] = plane.constant;
    });
    const list = this.#list;
    list.length = 0;
    const s = this.#spheres;
    const objects = this.#objects;
    const kinds = this.#kinds;
    for (let i = 0; i < this.#count; i++) {
      const o = objects[i];
      const kind = kinds[i];
      if (kind === DRAWN) {
        // A shadow view's culling hides the InstancedMeshes it draws nothing of.
        if (o.visible === false || !draws(o.material)) continue;
        const r = s[i * 4 + 3];
        if (r >= 0 && !(shadow && this.#proxied[i])) {
          const x = s[i * 4];
          const y = s[i * 4 + 1];
          const z = s[i * 4 + 2];
          if (
            p[0] * x + p[1] * y + p[2] * z + p[3] < -r ||
            p[4] * x + p[5] * y + p[6] * z + p[7] < -r ||
            p[8] * x + p[9] * y + p[10] * z + p[11] < -r ||
            p[12] * x + p[13] * y + p[14] * z + p[15] < -r ||
            p[16] * x + p[17] * y + p[18] * z + p[19] < -r ||
            p[20] * x + p[21] * y + p[22] * z + p[23] < -r
          ) {
            continue;
          }
        }
      } else if (kind === LIGHT && shadow) {
        continue;
      }
      list.push(o);
    }
    if (extra && extra.children.length > 0) list.push(extra);
    return list;
  }
}
