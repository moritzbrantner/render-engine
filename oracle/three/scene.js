// Builds one exported oracle scenario with packages/three-lighting and renders it: the image
// (and, where the browser exposes GPU timer queries, an advisory GPU time) for render.mjs.

import {
  BatchedMesh,
  Bone,
  BufferAttribute,
  BufferGeometry,
  Color,
  DetachedBindMode,
  InstancedMesh,
  Matrix4,
  Mesh,
  PerspectiveCamera,
  RGBAFormat,
  SRGBColorSpace,
  Scene,
  Skeleton,
  SkinnedMesh,
  UnsignedByteType,
  WebGLRenderTarget,
  WebGLRenderer,
} from 'three';
import {
  Lighting,
  LitMaterial,
  ParticleEmitter,
  loadCubemap,
  loadToneMappingLuts,
  textureFromData,
} from '../../packages/three-lighting/index.js';

/**
 * Renders the exported scenario at `base` (`<base>.three.json` + `.three.bin`) and returns
 * `{ png (base64), renderer, gpuMs }`. `timedFrames` frames are timed for the advisory GPU time
 * (render.mjs passes the oracle's count; 0 skips timing). Also the WebGL 2 side of the browser
 * backend selection (`oracle/webgpu-smoke`, docs/webgl2-fallback.md).
 */
export async function renderScenario(base, timedFrames = 0) {
  const doc = await (await fetch(`${base}.three.json`)).json();
  const bin = await (await fetch(`${base}.three.bin`)).arrayBuffer();
  const { width, height } = doc;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const renderer = new WebGLRenderer({ canvas, antialias: false, alpha: false });
  renderer.setPixelRatio(1);
  renderer.setSize(width, height, false);
  const gl = renderer.getContext();
  const debug = gl.getExtension('WEBGL_debug_renderer_info');
  const rendererName = debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);

  const f32 = (r) => new Float32Array(bin, r.offset, r.count);
  const geometries = doc.meshes.map((m) => {
    const g = new BufferGeometry();
    g.setAttribute('position', new BufferAttribute(f32(m.positions), 3));
    g.setAttribute('normal', new BufferAttribute(f32(m.normals), 3));
    if (m.uvs) g.setAttribute('uv', new BufferAttribute(f32(m.uvs), 2));
    if (m.tangents) g.setAttribute('tangent', new BufferAttribute(f32(m.tangents), 4));
    if (m.colors) g.setAttribute('color', new BufferAttribute(f32(m.colors), 4));
    if (m.joints) {
      g.setAttribute('skinIndex', new BufferAttribute(new Uint16Array(bin, m.joints.offset, m.joints.count), 4));
      g.setAttribute('skinWeight', new BufferAttribute(f32(m.weights), 4));
    }
    g.setIndex(new BufferAttribute(new Uint32Array(bin, m.indices.offset, m.indices.count), 1));
    return g;
  });
  // Every texture once, with the engine's mip chain and sampler.
  const textures = doc.textures.map((t) =>
    textureFromData({ ...t, levels: t.levels.map((l) => new Uint8Array(bin, l.offset, l.length)) }),
  );
  const TEXTURES = ['baseColorTexture', 'emissiveTexture', 'metallicRoughnessTexture', 'normalMapTexture', 'occlusionTexture'];
  // Each skin as a three Skeleton whose bones sit at the joints' mesh-local transforms (outside
  // the scene graph, so nothing recomputes them): its palette is render-engine's.
  const matrices = (r) => {
    const data = f32(r);
    return Array.from({ length: r.count / 16 }, (_, i) => new Matrix4().fromArray(data, i * 16));
  };
  const skeletons = doc.skins.map((s) => {
    const bones = matrices(s.modelFromJoint).map((m) => {
      const bone = new Bone();
      bone.matrixAutoUpdate = false;
      bone.matrixWorldAutoUpdate = false;
      bone.matrixWorld.copy(m);
      return bone;
    });
    return new Skeleton(bones, matrices(s.inverseBindposes));
  });
  const scene = new Scene();
  const materials = new Map();
  // Unskinned opaque objects sharing a mesh and a material draw as one InstancedMesh, as the
  // engine instances them; the rest as one Mesh each.
  const batches = new Map();
  for (const o of doc.objects) {
    const key = JSON.stringify(o.material);
    const params = { ...o.material };
    for (const name of TEXTURES) params[name] = params[name] == null ? null : textures[params[name]];
    if (!materials.has(key)) materials.set(key, new LitMaterial(params));
    const material = materials.get(key);
    const skinned = o.skin != null;
    if (!skinned && material.alphaMode === 'opaque') {
      const batchKey = `${o.mesh}:${key}`;
      if (!batches.has(batchKey)) batches.set(batchKey, { geometry: geometries[o.mesh], material, matrices: [] });
      batches.get(batchKey).matrices.push(o.matrix);
      continue;
    }
    const mesh = new (skinned ? SkinnedMesh : Mesh)(geometries[o.mesh], material);
    mesh.matrixAutoUpdate = false;
    mesh.matrix.fromArray(o.matrix);
    if (skinned) {
      mesh.bindMode = DetachedBindMode;
      mesh.bind(skeletons[o.skin], new Matrix4());
    }
    scene.add(mesh);
  }
  const instance = new Matrix4();
  // Single objects sharing a material and an attribute layout draw as one BatchedMesh (white
  // instance colours), as a web consumer batches static scenery; a lone one as a Mesh.
  const singles = new Map();
  for (const { geometry, material, matrices } of batches.values()) {
    if (matrices.length === 1) {
      const layout = `${material.uuid}:${Object.keys(geometry.attributes).sort().join(',')}`;
      if (!singles.has(layout)) singles.set(layout, { material, draws: [] });
      singles.get(layout).draws.push({ geometry, matrix: matrices[0] });
      continue;
    }
    const mesh = new InstancedMesh(geometry, material, matrices.length);
    matrices.forEach((m, i) => mesh.setMatrixAt(i, instance.fromArray(m)));
    mesh.computeBoundingSphere();
    scene.add(mesh);
  }
  const white = new Color(1, 1, 1);
  for (const { material, draws } of singles.values()) {
    if (draws.length === 1) {
      const mesh = new Mesh(draws[0].geometry, material);
      mesh.matrixAutoUpdate = false;
      mesh.matrix.fromArray(draws[0].matrix);
      scene.add(mesh);
      continue;
    }
    const unique = [...new Set(draws.map((d) => d.geometry))];
    const vertices = unique.reduce((n, g) => n + g.attributes.position.count, 0);
    const indices = unique.reduce((n, g) => n + g.index.count, 0);
    const mesh = new BatchedMesh(draws.length, vertices, indices, material);
    const ids = new Map(unique.map((g) => [g, mesh.addGeometry(g)]));
    for (const { geometry, matrix } of draws) {
      const id = mesh.addInstance(ids.get(geometry));
      mesh.setMatrixAt(id, instance.fromArray(matrix));
      mesh.setColorAt(id, white);
    }
    scene.add(mesh);
  }

  const c = doc.camera;
  const camera = new PerspectiveCamera((c.fovY * 180) / Math.PI, width / height, c.near, 10_000);
  camera.position.set(...c.eye);
  camera.lookAt(...c.target);

  const lighting = new Lighting(await loadToneMappingLuts(), {
    shadowMapSize: doc.shadowMapSize,
    pointShadowMapSize: doc.pointShadowMapSize,
  });
  lighting.pointLights = doc.pointLights;
  lighting.spotLights = doc.spotLights;
  lighting.clearColor = doc.clearColor;
  lighting.ev100 = c.ev100;
  lighting.tonemapping = c.tonemapping;
  lighting.shadowFilter = c.shadowFilter;
  lighting.sun = doc.sun;
  lighting.moon = doc.moon;
  lighting.sky = doc.sky;
  lighting.ambient = doc.ambient;
  lighting.fog = doc.fog;
  if (doc.environment) {
    const [diffuse, specular] = await Promise.all([
      loadCubemap(`/${doc.environment.diffuse}`),
      loadCubemap(`/${doc.environment.specular}`),
    ]);
    lighting.environment = { diffuse, specular, intensity: doc.environment.intensity };
  }

  // An sRGB target like Bevy's output, encoded on store by the device.
  const target = new WebGLRenderTarget(width, height, {
    type: UnsignedByteType,
    format: RGBAFormat,
    colorSpace: SRGBColorSpace,
    depthBuffer: false,
  });
  const pixels = new Uint8Array(width * height * 4);
  const image = new ImageData(width, height);
  const png = document.createElement('canvas');
  png.width = width;
  png.height = height;
  // One frame as a base64 PNG, rows top-down (GL rows run bottom-up).
  const capture = () => {
    lighting.render(renderer, scene, camera, target);
    renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);
    const row = width * 4;
    for (let y = 0; y < height; y++) {
      image.data.set(pixels.subarray((height - 1 - y) * row, (height - y) * row), y * row);
    }
    png.getContext('2d').putImageData(image, 0, 0);
    const dataUrl = png.toDataURL('image/png');
    return dataUrl.slice(dataUrl.indexOf(',') + 1);
  };
  if (doc.particles) return { renderer: rendererName, ...runParticles(doc.particles, scene, textures, capture) };
  const frame = capture();

  // Advisory GPU time: the whole Lighting.render per frame, median of timedFrames.
  let gpuMs = null;
  const timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');
  if (timer && timedFrames > 0) {
    const queries = [];
    for (let i = 0; i < timedFrames; i++) {
      const q = gl.createQuery();
      gl.beginQuery(timer.TIME_ELAPSED_EXT, q);
      lighting.render(renderer, scene, camera, target);
      gl.endQuery(timer.TIME_ELAPSED_EXT);
      queries.push(q);
    }
    gl.finish();
    const times = [];
    for (const q of queries) {
      for (let tries = 0; tries < 1000 && !gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE); tries++) {
        await new Promise((r) => setTimeout(r, 5));
      }
      if (gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE) && !gl.getParameter(timer.GPU_DISJOINT_EXT)) {
        times.push(gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6);
      }
    }
    times.sort((a, b) => a - b);
    if (times.length) gpuMs = times[Math.floor(times.length / 2)];
  }

  return { png: frame, renderer: rendererName, gpuMs };
}

/**
 * A particle scenario's emitters on `particles.js`, driven as the oracle drives render-engine's
 * (`run_engine_particles`): added inactive, stepped until Bevy's first spawn update, then each
 * by its driver (its spawner; a burst on frame 1; consumer-fed states after every step). Returns
 * the plate (no particles yet), the captures, and per emitter and frame the spawn and alive
 * counts.
 */
function runParticles(p, scene, textures, capture) {
  const emitters = p.emitters.map((e) => {
    const texture = e.desc.texture && { ...e.desc.texture, texture: textures[e.desc.texture.texture] };
    const emitter = new ParticleEmitter({ ...e.desc, texture, active: false });
    emitter.setTransform(new Matrix4().makeTranslation(...e.translation));
    scene.add(emitter);
    return emitter;
  });
  const plate = capture();
  for (let i = 1; i < p.firstSpawnUpdate; i++) for (const e of emitters) e.step(p.dt);
  emitters.forEach((e, i) => e.setEmitter({ ...e.desc, active: p.emitters[i].driver === 'spawner' }));
  const last = Math.max(...p.captures.map((c) => c.frame));
  const spawnCounts = emitters.map(() => []);
  const alive = emitters.map(() => []);
  const captures = [];
  for (let frame = 1; frame <= last; frame++) {
    emitters.forEach((e, i) => {
      if (p.emitters[i].driver === 'trigger' && frame === 1) e.burst(p.emitters[i].burst, e.transform);
    });
    for (const e of emitters) e.step(p.dt);
    emitters.forEach((e, i) => {
      const x = p.emitters[i];
      if (x.driver === 'snapshot') e.setParticles(x.snapshot[frame - 1]);
      spawnCounts[i].push(x.driver === 'snapshot' ? x.spawns[frame - 1] : e.spawnCount);
      alive[i].push(e.aliveCount);
    });
    for (const c of p.captures) if (c.frame === frame) captures.push({ label: c.label, png: capture() });
  }
  return { plate, captures, spawnCounts, alive };
}
