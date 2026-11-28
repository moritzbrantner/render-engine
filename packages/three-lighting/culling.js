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
//
// A BatchedMesh caster is culled the same way (#169), whatever three's own per-instance culling
// is set to (`perObjectFrustumCulled`; three rebuilds its one draw list, sorts it and uploads it
// again in every view it culls in): per shadow view it draws nothing, itself, or, straddling the
// view, only the instances inside through a stand-in: the mesh itself with a draw list and
// indirect texture of that view's (three's private `_multiDraw*` and `_indirectTexture`), built
// and uploaded only when the instances in the view change. The mesh's own list is not touched.

import { DataTexture, DynamicDrawUsage, Frustum, Group, InstancedMesh, Matrix4, RedIntegerFormat, Sphere, UnsignedIntType } from 'three';

const RANGE = {};
const SPHERE = new Sphere();
const NOTHING = () => {};

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

// Whether sphere `i` of `spheres` (x, y, z, radius) meets the six planes `p` (normal, constant)
// and has at least radius `min`.
const meets = (p, spheres, i, min = 0) => {
  if (spheres[i * 4 + 3] < min) return false;
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

  /**
   * How many instance bounding spheres the last `view` tested against its frustum: none where a
   * mesh's sphere around all its instances settled it, or its last result for that view holds.
   */
  tests = 0;

  /** The group of the stand-ins the current view draws, for a caller that lists what it draws. */
  get standIns() {
    return this.#standIns;
  }

  /**
   * Starts this frame's shadow passes with the visible InstancedMeshes that draw any instance.
   * With `scene` the stand-ins join it until `end`; otherwise the caller draws `standIns`.
   */
  begin(instanced, scene = null, batched = []) {
    this.#instanced = instanced;
    this.#states = instanced.map((mesh) => this.#spheres(mesh));
    this.#batched = batched;
    this.#batchStates = batched.map((mesh) => this.#batchSpheres(mesh));
    scene?.add(this.#standIns);
  }

  // The visible BatchedMeshes that cast, and their states.
  #batched = [];
  #batchStates = [];

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
        smallest: 0,
        largest: 0,
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
    let smallest = Infinity;
    let largest = 0;
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
      const r = pad(radius * objectScale * maxScale(a, o));
      spheres[i * 4 + 3] = r;
      smallest = Math.min(smallest, r);
      largest = Math.max(largest, r);
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
    // The instances' smallest and largest radii, so a size limit can take or drop the whole mesh.
    s.smallest = n > 0 ? Math.fround(smallest) : 0;
    s.largest = Math.fround(largest);
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
  #classify(mesh, s, index, stamp, min) {
    const v = (s.views[index] ??= { result: null, revision: -1, stamp: -1, min: -1, standIn: null, capacity: 0, indices: null, count: -1, filled: -1 });
    if (v.revision === s.revision && v.stamp === stamp && v.min === min) return v.result;
    v.revision = s.revision;
    v.stamp = stamp;
    v.min = min;
    const p = this.#planes;
    const n = s.count;
    if (this.#indices.length < n) this.#indices = new Uint32Array(n);
    const indices = this.#indices;
    let visible = 0;
    for (let i = 0; i < n; i++) if (meets(p, s.spheres, i, min)) indices[visible++] = i;
    this.tests += n;
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
   * InstancedMesh draws only what of it meets the camera's frustum, and of that only the instances
   * whose bounding sphere has at least radius `min` (projected-size culling; 0 keeps all).
   */
  view(index, camera, min = 0) {
    this.reset();
    this.tests = 0;
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
      // keeps it whole, unless some instances are too small there. All too small drops it.
      const all = s.all;
      const ar = all[3];
      let inside = s.smallest >= min;
      let outside = s.largest < min;
      for (let k = 0; k < 24 && !outside; k += 4) {
        const d = p[k] * all[0] + p[k + 1] * all[1] + p[k + 2] * all[2] + p[k + 3];
        if (d < -ar) {
          outside = true;
          break;
        }
        if (d < ar) inside = false;
      }
      // Outside one plane is outside, whatever the planes before said.
      if (inside && !outside) continue;
      if (!outside && s.count < PER_INSTANCE_MIN) {
        let any = false;
        let j = 0;
        for (; j < s.count && !any; j++) any = meets(p, s.spheres, j, min);
        this.tests += j;
        if (any) continue;
        outside = true;
      }
      const result = outside ? false : this.#classify(mesh, s, index, stamp, min);
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

    const batched = this.#batched;
    for (let i = 0; i < batched.length; i++) {
      const mesh = batched[i];
      const s = this.#batchStates[i];
      const all = s.all;
      const ar = all[3];
      let inside = s.smallest >= min;
      let outside = s.count === 0 || s.largest < min;
      for (let k = 0; k < 24 && !outside; k += 4) {
        const d = p[k] * all[0] + p[k + 1] * all[1] + p[k + 2] * all[2] + p[k + 3];
        if (d < -ar) {
          outside = true;
          break;
        }
        if (d < ar) inside = false;
      }
      if (inside && !outside) continue;
      const result = outside ? false : this.#classifyBatch(mesh, s, index, stamp, min);
      if (result === true) continue;
      if (result) standIns.push(result.standIn);
      this.#hide(mesh);
    }
  }

  // The world bounding spheres of `mesh`'s drawn instances (active and visible), with each one's
  // id and draw range; recomputed when its matrices, its world matrix, or what it draws change.
  #batchSpheres(mesh) {
    let s = this.#state.get(mesh);
    if (!s) {
      s = {
        spheres: new Float32Array(0),
        ids: new Uint32Array(0),
        starts: new Int32Array(0),
        counts: new Int32Array(0),
        // Per instance id: its geometry + 1 when drawn (0 otherwise), that geometry's range and sphere.
        codes: new Int32Array(0),
        bounds: [],
        count: 0,
        all: new Float64Array(4),
        smallest: 0,
        largest: 0,
        revision: 0,
        matrices: null,
        version: -1,
        world: new Matrix4(),
        views: [],
      };
      this.#state.set(mesh, s);
      const dispose = () => {
        for (const v of s.views) v?.texture?.dispose();
        s.views.length = 0;
        this.#state.delete(mesh);
      };
      mesh.addEventListener('dispose', dispose);
      mesh.geometry.addEventListener('dispose', dispose);
    }
    const info = mesh._instanceInfo;
    const matrices = mesh._matricesTexture;
    const moved = s.matrices !== matrices || s.version !== matrices.version || !s.world.equals(mesh.matrixWorld);
    // three sets the flag whenever an instance or a geometry is added, removed, shown, hidden
    // or replaced, and clears it at its own next draw of the whole list: while it is set, compare.
    if (!moved && s.revision > 0 && s.codes.length === info.length * 3 && !(mesh._visibilityChanged && this.#batchChanged(mesh, s))) {
      return s;
    }
    const n = info.length;
    if (s.codes.length !== n * 3) {
      s.codes = new Int32Array(n * 3);
      s.spheres = new Float32Array(n * 4);
      s.ids = new Uint32Array(n);
      s.starts = new Int32Array(n);
      s.counts = new Int32Array(n);
    }
    s.bounds.length = n;
    const index = mesh.geometry.index;
    const bytes = index === null ? 1 : index.array.BYTES_PER_ELEMENT;
    const w = mesh.matrixWorld.elements;
    const objectScale = maxScale(w, 0);
    const a = matrices.image.data;
    const spheres = s.spheres;
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    let smallest = Infinity;
    let largest = 0;
    let count = 0;
    for (let i = 0; i < n; i++) {
      const drawn = info[i].visible && info[i].active;
      const geometry = drawn ? info[i].geometryIndex : -1;
      s.codes[i * 3] = geometry + 1;
      s.bounds[i] = null;
      if (!drawn) continue;
      const range = mesh.getGeometryRangeAt(geometry, RANGE);
      s.codes[i * 3 + 1] = range.start;
      s.codes[i * 3 + 2] = range.count;
      const { center, radius } = mesh.getBoundingSphereAt(geometry, SPHERE);
      s.bounds[i] = mesh._geometryInfo[geometry].boundingSphere;
      const o = i * 16;
      const x = a[o] * center.x + a[o + 4] * center.y + a[o + 8] * center.z + a[o + 12];
      const y = a[o + 1] * center.x + a[o + 5] * center.y + a[o + 9] * center.z + a[o + 13];
      const z = a[o + 2] * center.x + a[o + 6] * center.y + a[o + 10] * center.z + a[o + 14];
      const c = [w[0] * x + w[4] * y + w[8] * z + w[12], w[1] * x + w[5] * y + w[9] * z + w[13], w[2] * x + w[6] * y + w[10] * z + w[14]];
      for (let k = 0; k < 3; k++) {
        spheres[count * 4 + k] = c[k];
        lo[k] = Math.min(lo[k], c[k]);
        hi[k] = Math.max(hi[k], c[k]);
      }
      const r = pad(radius * objectScale * maxScale(a, o));
      spheres[count * 4 + 3] = r;
      smallest = Math.min(smallest, r);
      largest = Math.max(largest, r);
      s.ids[count] = i;
      s.starts[count] = range.start * bytes;
      s.counts[count] = range.count;
      count++;
    }
    const c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
    let reach = 0;
    for (let i = 0; i < count; i++) {
      const dx = spheres[i * 4] - c[0];
      const dy = spheres[i * 4 + 1] - c[1];
      const dz = spheres[i * 4 + 2] - c[2];
      reach = Math.max(reach, Math.sqrt(dx * dx + dy * dy + dz * dz) + spheres[i * 4 + 3]);
    }
    s.all.set(count > 0 ? [c[0], c[1], c[2], pad(reach)] : [0, 0, 0, 0]);
    s.smallest = count > 0 ? Math.fround(smallest) : 0;
    s.largest = Math.fround(largest);
    s.count = count;
    s.matrices = matrices;
    s.version = matrices.version;
    s.world.copy(mesh.matrixWorld);
    s.revision++;
    return s;
  }

  // Whether what `mesh` draws (which instances, each one's geometry, range and sphere) differs
  // from what `s` holds.
  #batchChanged(mesh, s) {
    const info = mesh._instanceInfo;
    const geometries = mesh._geometryInfo;
    const codes = s.codes;
    for (let i = 0; i < info.length; i++) {
      const drawn = info[i].visible && info[i].active;
      if (!drawn) {
        if (codes[i * 3] !== 0) return true;
        continue;
      }
      const g = geometries[info[i].geometryIndex];
      if (codes[i * 3] !== info[i].geometryIndex + 1 || codes[i * 3 + 1] !== g.start || codes[i * 3 + 2] !== g.count || s.bounds[i] !== g.boundingSphere) {
        return true;
      }
    }
    return false;
  }

  // The instances of the BatchedMesh `mesh`, straddling view `index`, it draws there: none, all,
  // or (a view record with a stand-in) some. Kept while neither the mesh nor the view's planes
  // change.
  #classifyBatch(mesh, s, index, stamp, min) {
    const v = (s.views[index] ??= { result: null, revision: -1, stamp: -1, min: -1, standIn: null, texture: null, starts: null, counts: null, size: 0 });
    if (v.revision === s.revision && v.stamp === stamp && v.min === min) return v.result;
    v.revision = s.revision;
    v.stamp = stamp;
    v.min = min;
    const p = this.#planes;
    const n = s.count;
    if (this.#indices.length < n) this.#indices = new Uint32Array(n);
    const indices = this.#indices;
    let visible = 0;
    for (let i = 0; i < n; i++) if (meets(p, s.spheres, i, min)) indices[visible++] = i;
    this.tests += n;
    if (visible === n || visible === 0) {
      v.result = visible === n;
      return v.result;
    }
    // three's own list has room for every instance; the indirect texture is as large as its.
    const size = mesh._indirectTexture.image.width;
    const capacity = mesh._multiDrawStarts.length;
    if (!v.standIn || v.size !== size || v.starts.length !== capacity) {
      v.texture?.dispose();
      v.texture = new DataTexture(new Uint32Array(size * size), size, size, RedIntegerFormat, UnsignedIntType);
      v.starts = new Int32Array(capacity);
      v.counts = new Int32Array(capacity);
      v.size = size;
      // The mesh itself with another draw list: everything else (geometry, the material a view
      // gives it, matrices, colours) is read through to the mesh. It draws its list as it is, so
      // neither three's per-instance culling and sorting nor a subclass's hook runs for it.
      const standIn = Object.create(mesh);
      standIn.name = `${mesh.name} (shadow view ${index})`;
      standIn.visible = true;
      standIn.children = [];
      standIn.frustumCulled = false;
      standIn.perObjectFrustumCulled = false;
      standIn.sortObjects = false;
      standIn._visibilityChanged = false;
      standIn.onBeforeRender = NOTHING;
      standIn.onAfterRender = NOTHING;
      standIn._indirectTexture = v.texture;
      standIn._multiDrawStarts = v.starts;
      standIn._multiDrawCounts = v.counts;
      standIn._multiDrawCount = -1;
      v.standIn = standIn;
    }
    v.result = v;
    const data = v.texture.image.data;
    let same = v.standIn._multiDrawCount === visible;
    for (let j = 0; same && j < visible; j++) same = data[j] === s.ids[indices[j]] && v.starts[j] === s.starts[indices[j]] && v.counts[j] === s.counts[indices[j]];
    if (same) return v;
    for (let j = 0; j < visible; j++) {
      const k = indices[j];
      data[j] = s.ids[k];
      v.starts[j] = s.starts[k];
      v.counts[j] = s.counts[k];
    }
    v.standIn._multiDrawCount = visible;
    v.texture.needsUpdate = true;
    return v;
  }

  /**
   * For `Lighting.prepare`, in place of `view`: one stand-in for each mesh a view may draw
   * through stand-ins whose program differs from the mesh's own (stand-ins carry no instance
   * colours or morphs), so compiling the view compiles that program too. `reset` drops them.
   */
  standInVariants() {
    this.reset();
    const standIns = this.#standIns.children;
    for (let i = 0; i < this.#instanced.length; i++) {
      const mesh = this.#instanced[i];
      if (this.#states[i].count < PER_INSTANCE_MIN) continue;
      if (mesh.instanceColor === null && mesh.morphTexture === null) continue;
      let standIn = this.#variants.get(mesh);
      if (!standIn) {
        standIn = new InstancedMesh(mesh.geometry, mesh.material, 1);
        standIn.frustumCulled = false;
        standIn.matrixAutoUpdate = false;
        standIn.matrixWorldAutoUpdate = false;
        this.#variants.set(mesh, standIn);
      }
      standIn.geometry = mesh.geometry;
      standIn.material = mesh.material;
      standIns.push(standIn);
    }
  }

  // Per mesh, the stand-in standInVariants compiles.
  #variants = new WeakMap();

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
