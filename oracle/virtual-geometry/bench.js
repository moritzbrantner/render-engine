// Development-only adapter using public Three.js APIs. No private batching fields.
import * as T from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { KTX2Loader } from 'three/addons/loaders/KTX2Loader.js';

const q = new URLSearchParams(location.search);
const name = q.get('asset') || 'crag.kit.outcrop-buttress-1';
const kit = q.get('kit') || 'crag';
const sceneInput = q.get('scene') ? await fetch(q.get('scene')).then(r => { if (!r.ok) throw new Error('Scene unavailable'); return r.json(); }) : null;
if (sceneInput && sceneInput.asset !== name) throw new Error('Scene asset mismatch');
const root = `/target/virtual-geometry/inputs/${kit}/scenes/${name}/`;
const hierarchy = `/target/virtual-geometry/${q.get('hierarchy') || kit}/${name}/`;
const renderer = new T.WebGLRenderer({ canvas: document.querySelector('canvas'), antialias: true, preserveDrawingBuffer: true });
renderer.setSize(1280, 720, false);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = T.PCFShadowMap;
renderer.toneMapping = T.ACESFilmicToneMapping;
const gl = renderer.getContext();
const timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');
const debug = gl.getExtension('WEBGL_debug_renderer_info');
const environment = { renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : 'unavailable', userAgent: navigator.userAgent,
  multiDraw: !!gl.getExtension('WEBGL_multi_draw'), gpuTimer: !!timer, resolution: [1280, 720], three: T.REVISION };
const ktx = new KTX2Loader().setTranscoderPath('/target/virtual-geometry/basis/').detectSupport(renderer);
const loader = new GLTFLoader().setKTX2Loader(ktx);
const startLoad = performance.now();
const [source, meta, bin] = await Promise.all([loader.loadAsync(root + 'default.gltf'), fetch(hierarchy + name + '.json').then(r => r.json()), fetch(hierarchy + name + '.bin').then(r => r.arrayBuffer())]);
let original;
source.scene.updateMatrixWorld(true);
source.scene.traverse(o => { if (o.isMesh) { if (original) throw new Error('One material/primitive required for this experiment'); original = o; } });
if (!original || original.isSkinnedMesh) throw new Error('Static single-primitive asset required');
const material = original.material;
const geometry = original.geometry.clone().applyMatrix4(original.matrixWorld);
const CT = { Float32Array, Int8Array, Int32Array, Uint32Array };
const sec = k => { const s = meta.sections[k]; return new CT[s.type](bin, s.offset, s.bytes / CT[s.type].BYTES_PER_ELEMENT); };
const p = sec('positions'), normals = sec('normals'), uv = sec('uvs'), indices = sec('indices'), cu = sec('clusterU'), gf = sec('groupF'), mo = sec('memberOff'), mi = sec('memberIds');
const consumer = new Int32Array(meta.clusters);
for (let g = 0; g < meta.groups; g++) for (let i = mo[g]; i < mo[g + 1]; i++) consumer[mi[i]] = g;
// Carry original material attributes through the reference's reordered vertices.
const attributeKey = (pos, tex) => [...pos, ...tex].join(',');
const sourceAttributes = new Map();
const gp = geometry.attributes.position, guv = geometry.attributes.uv, gn = geometry.attributes.normal;
if (!geometry.attributes.tangent) geometry.computeTangents();
const gt = geometry.attributes.tangent, gc = geometry.attributes.color;
for (let i = 0; i < gp.count; i++) {
  const key = attributeKey([gp.getX(i), gp.getY(i), gp.getZ(i)], [guv.getX(i), guv.getY(i)]);
  if (!sourceAttributes.has(key)) sourceAttributes.set(key, []);
  sourceAttributes.get(key).push(i);
}
const clusters = [];
for (let c = 0; c < meta.clusters; c++) {
  const remap = new Map(), pp = [], nn = [], uu = [], tt = [], cc = [], ii = [];
  for (let j = cu[c * 4]; j < cu[c * 4] + cu[c * 4 + 1]; j++) {
    const index = indices[j];
    if (!remap.has(index)) {
      remap.set(index, remap.size); pp.push(...p.subarray(index * 3, index * 3 + 3));
      const n = Array.from(normals.subarray(index * 4, index * 4 + 3), n => n / 127);
      const candidates = sourceAttributes.get(attributeKey(Array.from(p.subarray(index * 3, index * 3 + 3)), Array.from(uv.subarray(index * 2, index * 2 + 2))));
      if (!candidates) throw new Error('Cluster vertex has no matching source attributes');
      const dot = i => gn.getX(i) * n[0] + gn.getY(i) * n[1] + gn.getZ(i) * n[2];
      const original = candidates.reduce((best, i) => dot(i) > dot(best) ? i : best);
      nn.push(gn.getX(original), gn.getY(original), gn.getZ(original));
      uu.push(...uv.subarray(index * 2, index * 2 + 2));
      tt.push(gt.getX(original), gt.getY(original), gt.getZ(original), gt.getW(original));
      if (gc) cc.push(gc.getX(original), gc.getY(original), gc.getZ(original));
    }
    ii.push(remap.get(index));
  }
  const g = new T.BufferGeometry();
  g.setAttribute('position', new T.Float32BufferAttribute(pp, 3));
  g.setAttribute('normal', new T.Float32BufferAttribute(nn, 3));
  g.setAttribute('uv', new T.Float32BufferAttribute(uu, 2));
  g.setAttribute('tangent', new T.Float32BufferAttribute(tt, 4));
  if (gc) g.setAttribute('color', new T.Float32BufferAttribute(cc, 3));
  g.setIndex(ii); g.computeBoundingSphere(); clusters.push(g);
}
const lods = [];
for (const level of ['lod1', 'lod2', 'lod3', 'lod4']) {
  const response = await fetch(root + level + '.gltf');
  if (response.ok) {
    const gltf = await loader.parseAsync(await response.text(), root);
    gltf.scene.updateMatrixWorld(true);
    gltf.scene.traverse(o => { if (o.isMesh) lods.push(o.geometry.clone().applyMatrix4(o.matrixWorld)); });
  }
}
const loadMs = performance.now() - startLoad;
const scene = new T.Scene(); scene.background = new T.Color(0.15, 0.15, 0.15);
scene.add(new T.HemisphereLight(0xffffff, 0x555555, 1));
const sun = new T.DirectionalLight(0xffffff, 3); sun.position.set(30, 70, 40); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048); Object.assign(sun.shadow.camera, { left: -150, right: 150, top: 150, bottom: -150, near: 1, far: 500 }); scene.add(sun); scene.add(sun.target);
const camera = new T.PerspectiveCamera(50, 1280 / 720, 0.1, 1500);
const tick = () => new Promise(resolve => requestAnimationFrame(resolve));
const stats = input => { const v = input.slice().sort((a, b) => a - b); return { median: v[Math.round((v.length - 1) * .5)], p95: v[Math.round((v.length - 1) * .95)] }; };
const rows = [];
let active;

function setup(mode, count) {
  if (active) { scene.remove(active); active.dispose(); }
  const side = Math.ceil(Math.sqrt(count));
  const matrices = sceneInput ? sceneInput.transforms.map(values => new T.Matrix4().fromArray(Array.from(new Float32Array(values)))) : Array.from({ length: count }, (_, i) => new T.Matrix4().makeTranslation((i % side - (side - 1) / 2) * 16, 0, (Math.floor(i / side) - (side - 1) / 2) * 16));
  if (matrices.length !== count) throw new Error('Scene placement count mismatch');
  const cluster = mode.startsWith('cluster');
  let ids;
  if (cluster) {
    const vertices = clusters.reduce((n, g) => n + g.attributes.position.count, 0);
    const indexCount = clusters.reduce((n, g) => n + g.index.count, 0);
    active = new T.BatchedMesh(count * clusters.length, vertices, indexCount, material);
    active.sortObjects = false;
    const geometries = clusters.map(g => active.addGeometry(g));
    ids = matrices.map(m => geometries.map(g => { const id = active.addInstance(g); active.setMatrixAt(id, m); return id; }));
  } else {
    const g = mode === 'source' ? geometry : lods[Number(mode.slice(3)) - 1];
    active = new T.InstancedMesh(g, material, count);
    matrices.forEach((m, i) => active.setMatrixAt(i, m));
    active.instanceMatrix.needsUpdate = true;
  }
  active.castShadow = true; active.receiveShadow = true; active.frustumCulled = false; scene.add(active);
  return { matrices, ids, count, mode };
}

function select(state, frame) {
  const a = frame * Math.PI * 2 / 240, radius = Math.sqrt(state.count) * 9 + 15;
  camera.position.set(Math.sin(a) * radius, 18, Math.cos(a) * radius); camera.lookAt(0, 3, 0); camera.updateMatrixWorld();
  if (sceneInput) {
    const pose = sceneInput.frames[frame % sceneInput.frames.length];
    camera.position.fromArray(pose.eye); camera.lookAt(new T.Vector3().fromArray(pose.target)); camera.updateMatrixWorld();
  }
  if (!state.ids) return;
  const errorPx = Number(state.mode.split('-')[1]);
  const K = 720 / (2 * Math.tan(50 * Math.PI / 360));
  const errors = new Float64Array(meta.groups), center = new T.Vector3();
  state.matrices.forEach((matrix, i) => {
    const scale = matrix.getMaxScaleOnAxis();
    for (let g = 0; g < meta.groups; g++) {
      const o = g * 5; center.set(gf[o], gf[o + 1], gf[o + 2]).applyMatrix4(matrix);
      const d = center.distanceTo(camera.position) - gf[o + 3] * scale;
      errors[g] = d <= 1e-6 ? Infinity : gf[o + 4] * scale * K / d;
    }
    for (let c = 0; c < meta.clusters; c++) {
      const producer = cu[c * 4 + 2];
      active.setVisibleAt(state.ids[i][c], (producer < 0 || errors[producer] <= errorPx) && (consumer[c] === 0 || errors[consumer[c]] > errorPx));
    }
  });
}

async function bench({ mode = 'cluster-2', count = 64, round = 0 } = {}) {
  const state = setup(mode, count);
  for (let i = 0; i < 60; i++) { select(state, i); renderer.render(scene, camera); await tick(); }
  // Without timer queries (Chrome on Android), sync_ms is the time from selection until a
  // 1-pixel readPixels returns: CPU plus the GPU work it waits for. Not GPU time.
  const cpu = [], selection = [], queries = [], triangles = [], draws = [], sync = [], pixel = new Uint8Array(4);
  renderer.info.autoReset = false;
  for (let frame = 0; frame < 240; frame++) {
    await tick(); renderer.info.reset();
    const query = timer ? gl.createQuery() : null;
    if (query) gl.beginQuery(timer.TIME_ELAPSED_EXT, query);
    const start = performance.now(); select(state, frame); selection.push(performance.now() - start);
    renderer.render(scene, camera); cpu.push(performance.now() - start);
    if (query) { gl.endQuery(timer.TIME_ELAPSED_EXT); queries.push(query); }
    else { gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel); sync.push(performance.now() - start); }
    triangles.push(renderer.info.render.triangles); draws.push(renderer.info.render.calls);
  }
  const gpu = [];
  for (const query of queries) {
    for (let wait = 0; !gl.getQueryParameter(query, gl.QUERY_RESULT_AVAILABLE); wait++) {
      if (wait > 600) throw new Error('GPU timestamp timeout');
      await tick();
    }
    if (gl.getParameter(timer.GPU_DISJOINT_EXT)) throw new Error('Disjoint GPU timing; discard run');
    gpu.push(gl.getQueryParameter(query, gl.QUERY_RESULT) / 1e6); gl.deleteQuery(query);
  }
  const row = { mode, count, round, cpu_ms: stats(cpu), selection_ms: stats(selection), gpu_ms: gpu.length ? stats(gpu) : null, sync_ms: sync.length ? stats(sync) : null,
    raw: { cpu_ms: cpu, selection_ms: selection, gpu_ms: gpu, sync_ms: sync, submitted_triangles_all_views: triangles, renderer_calls_all_views: draws } };
  rows.push(row); document.querySelector('#status').textContent = JSON.stringify({ environment, asset: name, mode, count, cpu: row.cpu_ms, gpu: row.gpu_ms }, null, 2);
  return { ...row, raw: undefined };
}
window.vg = { environment, sceneInput, bench, counts: sceneInput ? [sceneInput.transforms.length] : [1, 64, 256], modes: ['source', ...lods.map((_, i) => `lod${i + 1}`), 'cluster-1', 'cluster-2', 'cluster-4'],
  report: () => ({ environment, asset: name, scene: sceneInput, load_ms: loadMs, loading: 'Local HTTP, mixed asset/hierarchy/LOD loading and transcode; not comparable cold download', resources: performance.getEntriesByType('resource').map(r => ({ name: r.name, transferSize: r.transferSize, encodedBodySize: r.encodedBodySize, duration: r.duration })), rows }),
  capture: (mode, count = 64, frame = 0) => { const state = setup(mode, count); select(state, frame); renderer.render(scene, camera); return renderer.domElement.toDataURL('image/png'); } };
document.querySelector('#status').textContent = JSON.stringify({ environment, asset: name, ready: true }, null, 2);
window.vgReady = true;
