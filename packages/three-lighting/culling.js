// Per-view instance culling for the shadow passes, render-engine's (#30) on three.js. three culls
// a plain Mesh per view by its bounding sphere, but draws an InstancedMesh whole in every view
// (or culls it by one sphere around all its instances), so each sun cascade and light view would
// redraw the whole field. Here, per shadow view, an InstancedMesh caster:
//
// - draws nothing when no instance's bounding sphere meets the view's frustum (the sphere around
//   them all settles most meshes at once);
// - straddling the frustum with at least PER_INSTANCE_MIN instances, draws only the instances
//   inside, through a stand-in InstancedMesh kept per mesh and view (same geometry, material and
//   world matrix; the shadow materials read no per-instance attribute but the matrix). Each
//   stand-in keeps its own vertex array and uploads its matrices only when its instances change,
//   and the mesh itself is never touched for the main pass;
// - with fewer instances, draws them all: the ones outside cost the GPU less than sorting them
//   out costs the CPU, and clipped (or clamped onto the map's cleared depth) they cast nothing.
//
// The spheres are conservative (the geometry's sphere scaled by the largest axis of the object
// and of the instance), so no instance that reaches a view is dropped and the shadows stay the
// same. They are cached per mesh and recomputed only when its instance matrices, count, world
// matrix or geometry change; a view's per-instance results are kept while its planes and the
// mesh stay the same.

import { DynamicDrawUsage, Frustum, Group, InstancedMesh, Matrix4 } from 'three';

/** Below this many instances a straddling InstancedMesh draws whole, not instance by instance. */
export const PER_INSTANCE_MIN = 64;

const maxScale = (m, o) =>
  Math.sqrt(
    Math.max(
      m[o] * m[o] + m[o + 1] * m[o + 1] + m[o + 2] * m[o + 2],
      m[o + 4] * m[o + 4] + m[o + 5] * m[o + 5] + m[o + 6] * m[o + 6],
      m[o + 8] * m[o + 8] + m[o + 9] * m[o + 9] + m[o + 10] * m[o + 10],
    ),
  );

// The spheres are stored in f32: round them out a little.
const pad = (r) => r * 1.0001 + 1e-4;

// Whether sphere `i` of `spheres` (x, y, z, radius) meets the six planes `p` (normal, constant).
const meets = (p, spheres, i) => {
  const x = spheres[i * 4];
  const y = spheres[i * 4 + 1];
  const z = spheres[i * 4 + 2];
  const r = -spheres[i * 4 + 3];
  return (
    p[0] * x + p[1] * y + p[2] * z + p[3] >= r &&
    p[4] * x + p[5] * y + p[6] * z + p[7] >= r &&
    p[8] * x + p[9] * y + p[10] * z + p[11] >= r &&
    p[12] * x + p[13] * y + p[14] * z + p[15] >= r &&
    p[16] * x + p[17] * y + p[18] * z + p[19] >= r &&
    p[20] * x + p[21] * y + p[22] * z + p[23] >= r
  );
};

export class ShadowCulling {
  /** @param {import('three').Material} hidden the material a culled mesh with children takes */
  constructor(hidden) {
    this.hidden = hidden;
    this.#instanced = [];
    this.#states = [];
    this.#state = new WeakMap();
    this.#frustum = new Frustum();
    this.#clip = new Matrix4();
    this.#planes = new Float64Array(24);
    this.#indices = new Uint32Array(0);
    // Per view its last planes and a stamp that changes with them.
    this.#viewPlanes = [];
    this.#viewStamps = [];
    // Meshes this view hid: [mesh, own material, or null when it turned invisible].
    this.#hidden = [];
    // The stand-ins this view draws, in the scene while the shadow passes run.
    this.#standIns = new Group();
    this.#standIns.name = 'three-lighting shadow stand-ins';
    this.#standIns.matrixAutoUpdate = false;
    this.#standIns.matrixWorldAutoUpdate = false;
  }

  #instanced;
  #states;
  #state;
  #frustum;
  #clip;
  #planes;
  #indices;
  #viewPlanes;
  #viewStamps;
  #hidden;
  #standIns;

  /** Starts this frame's shadow passes over `scene` with its visible InstancedMeshes that draw any instance. */
  begin(scene, instanced) {
    this.#instanced = instanced;
    this.#states = instanced.map((mesh) => this.#spheres(mesh));
    scene.add(this.#standIns);
  }

  /** Ends the shadow passes: every mesh as it was, the stand-ins out of the scene. */
  end() {
    this.reset();
    this.#standIns.removeFromParent();
  }

  #spheres(mesh) {
    let s = this.#state.get(mesh);
    if (!s) {
      s = {
        spheres: new Float32Array(0),
        all: new Float64Array(4),
        revision: 0,
        version: -1,
        count: -1,
        attribute: null,
        geometry: null,
        world: new Matrix4(),
        views: [],
      };
      this.#state.set(mesh, s);
      mesh.addEventListener('dispose', () => {
        for (const v of s.views) v?.standIn?.dispose();
        this.#state.delete(mesh);
      });
    }
    const instances = mesh.instanceMatrix;
    const geometry = mesh.geometry;
    if (
      s.attribute === instances &&
      s.version === instances.version &&
      s.count === mesh.count &&
      s.geometry === geometry &&
      s.world.equals(mesh.matrixWorld)
    ) {
      return s;
    }
    if (geometry.boundingSphere === null) geometry.computeBoundingSphere();
    const { center, radius } = geometry.boundingSphere;
    const m = mesh.matrixWorld.elements;
    const objectScale = maxScale(m, 0);
    const a = instances.array;
    const n = mesh.count;
    if (s.spheres.length < n * 4) s.spheres = new Float32Array(instances.count * 4);
    const spheres = s.spheres;
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
      const o = i * 16;
      // The sphere centre through the instance, then through the object.
      const x = a[o] * center.x + a[o + 4] * center.y + a[o + 8] * center.z + a[o + 12];
      const y = a[o + 1] * center.x + a[o + 5] * center.y + a[o + 9] * center.z + a[o + 13];
      const z = a[o + 2] * center.x + a[o + 6] * center.y + a[o + 10] * center.z + a[o + 14];
      const c = [m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14]];
      for (let k = 0; k < 3; k++) {
        spheres[i * 4 + k] = c[k];
        lo[k] = Math.min(lo[k], c[k]);
        hi[k] = Math.max(hi[k], c[k]);
      }
      spheres[i * 4 + 3] = pad(radius * objectScale * maxScale(a, o));
    }
    // One sphere around them all, so a view can take or drop the whole mesh at once.
    const c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    let reach = 0;
    for (let i = 0; i < n; i++) {
      const dx = spheres[i * 4] - c[0];
      const dy = spheres[i * 4 + 1] - c[1];
      const dz = spheres[i * 4 + 2] - c[2];
      reach = Math.max(reach, Math.sqrt(dx * dx + dy * dy + dz * dz) + spheres[i * 4 + 3]);
    }
    s.all.set([c[0], c[1], c[2], pad(reach)]);
    s.attribute = instances;
    s.version = instances.version;
    s.count = n;
    s.geometry = geometry;
    s.world.copy(mesh.matrixWorld);
    // Every view's results are stale.
    s.revision++;
    return s;
  }

  // A culled mesh without children turns invisible, which three skips before any other work;
  // one with children only draws with the hidden material.
  #hide(mesh) {
    if (mesh.children.length === 0) {
      this.#hidden.push([mesh, null]);
      mesh.visible = false;
      return;
    }
    this.#hidden.push([mesh, mesh.material]);
    mesh.material = Array.isArray(mesh.material) ? mesh.material.map(() => this.hidden) : this.hidden;
  }

  // The instances of `mesh`, straddling view `index`, it draws there: none, all, or (a view
  // record with a stand-in) some. Kept while neither the mesh nor the view's planes change.
  #classify(mesh, s, index, stamp) {
    const v = (s.views[index] ??= { result: null, revision: -1, stamp: -1, standIn: null, capacity: 0, indices: null, count: -1, filled: -1 });
    if (v.revision === s.revision && v.stamp === stamp) return v.result;
    v.revision = s.revision;
    v.stamp = stamp;
    const p = this.#planes;
    const n = s.count;
    if (this.#indices.length < n) this.#indices = new Uint32Array(n);
    const indices = this.#indices;
    let visible = 0;
    for (let i = 0; i < n; i++) if (meets(p, s.spheres, i)) indices[visible++] = i;
    if (visible === n || visible === 0) {
      v.result = visible === n;
      return v.result;
    }
    const capacity = mesh.instanceMatrix.count;
    if (!v.standIn || v.capacity !== capacity) {
      v.standIn?.dispose();
      const standIn = new InstancedMesh(mesh.geometry, mesh.material, capacity);
      standIn.name = `${mesh.name} (shadow view ${index})`;
      standIn.instanceMatrix.setUsage(DynamicDrawUsage);
      standIn.frustumCulled = false;
      standIn.matrixAutoUpdate = false;
      standIn.matrixWorldAutoUpdate = false;
      v.standIn = standIn;
      v.capacity = capacity;
      v.indices = new Uint32Array(capacity);
      v.count = -1;
    }
    v.result = v;
    // The same instances as this view's last upload: the matrices hold.
    let same = v.filled === s.revision && v.count === visible;
    for (let j = 0; same && j < visible; j++) same = v.indices[j] === indices[j];
    if (same) return v;
    const from = mesh.instanceMatrix.array;
    const attribute = v.standIn.instanceMatrix;
    const to = attribute.array;
    for (let j = 0; j < visible; j++) to.set(from.subarray(indices[j] * 16, indices[j] * 16 + 16), j * 16);
    v.indices.set(indices.subarray(0, visible));
    v.count = visible;
    v.filled = s.revision;
    v.standIn.count = visible;
    attribute.clearUpdateRanges();
    attribute.addUpdateRange(0, visible * 16);
    attribute.needsUpdate = true;
    return v;
  }

  /**
   * Before shadow view `index` renders through `camera` (its culling camera, matrices set): each
   * InstancedMesh draws only what of it meets the camera's frustum.
   */
  view(index, camera) {
    this.reset();
    this.#clip.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.#frustum.setFromProjectionMatrix(this.#clip, camera.coordinateSystem, camera.reversedDepth);
    const p = this.#planes;
    this.#frustum.planes.forEach((plane, k) => {
      p[k * 4] = plane.normal.x;
      p[k * 4 + 1] = plane.normal.y;
      p[k * 4 + 2] = plane.normal.z;
      p[k * 4 + 3] = plane.constant;
    });
    const last = (this.#viewPlanes[index] ??= new Float64Array(24).fill(NaN));
    let moved = false;
    for (let k = 0; k < 24 && !moved; k++) moved = last[k] !== p[k];
    if (moved) {
      last.set(p);
      this.#viewStamps[index] = (this.#viewStamps[index] ?? 0) + 1;
    }
    const stamp = this.#viewStamps[index];

    const standIns = this.#standIns.children;
    const instanced = this.#instanced;
    for (let i = 0; i < instanced.length; i++) {
      const mesh = instanced[i];
      const s = this.#states[i];
      // The sphere around all instances first: wholly outside drops the mesh, wholly inside
      // keeps it whole.
      const all = s.all;
      const ar = all[3];
      let inside = true;
      let outside = false;
      for (let k = 0; k < 24; k += 4) {
        const d = p[k] * all[0] + p[k + 1] * all[1] + p[k + 2] * all[2] + p[k + 3];
        if (d < -ar) {
          outside = true;
          break;
        }
        if (d < ar) inside = false;
      }
      if (inside) continue;
      if (!outside && s.count < PER_INSTANCE_MIN) {
        let any = false;
        for (let j = 0; j < s.count && !any; j++) any = meets(p, s.spheres, j);
        if (any) continue;
        outside = true;
      }
      const result = outside ? false : this.#classify(mesh, s, index, stamp);
      if (result === true) continue;
      if (result) {
        const standIn = result.standIn;
        standIn.geometry = mesh.geometry;
        standIn.material = mesh.material;
        standIn.matrixWorld.copy(mesh.matrixWorld);
        standIn.layers.mask = mesh.layers.mask;
        standIn.renderOrder = mesh.renderOrder;
        standIns.push(standIn);
      }
      this.#hide(mesh);
    }
  }

  /** Gives every mesh back its visibility and material, and drops this view's stand-ins. */
  reset() {
    for (const [mesh, material] of this.#hidden) {
      if (material === null) mesh.visible = true;
      else mesh.material = material;
    }
    this.#hidden.length = 0;
    this.#standIns.children.length = 0;
  }
}
