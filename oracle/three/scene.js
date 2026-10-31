// Builds one exported oracle scenario with packages/three-lighting, renders it and hands the
// image (and, where the browser exposes GPU timer queries, an advisory GPU time) to render.mjs.

import {
  BufferAttribute,
  BufferGeometry,
  Mesh,
  PerspectiveCamera,
  RGBAFormat,
  SRGBColorSpace,
  Scene,
  UnsignedByteType,
  WebGLRenderTarget,
  WebGLRenderer,
} from 'three';
import { Lighting, LitMaterial, loadCubemap, loadToneMappingLuts } from '../../packages/three-lighting/index.js';

// Frames timed for the advisory GPU time (render.mjs passes the oracle's count; 0 skips timing).
const TIMED_FRAMES = Number(new URLSearchParams(location.search).get('frames') ?? 0);

async function main() {
  const base = new URLSearchParams(location.search).get('scene');
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
    g.setIndex(new BufferAttribute(new Uint32Array(bin, m.indices.offset, m.indices.count), 1));
    return g;
  });
  const scene = new Scene();
  const materials = new Map();
  for (const o of doc.objects) {
    const key = JSON.stringify(o.material);
    if (!materials.has(key)) materials.set(key, new LitMaterial(o.material));
    const mesh = new Mesh(geometries[o.mesh], materials.get(key));
    mesh.matrixAutoUpdate = false;
    mesh.matrix.fromArray(o.matrix);
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
  lighting.render(renderer, scene, camera, target);
  const pixels = new Uint8Array(width * height * 4);
  renderer.readRenderTargetPixels(target, 0, 0, width, height, pixels);

  // GL rows run bottom-up.
  const image = new ImageData(width, height);
  const row = width * 4;
  for (let y = 0; y < height; y++) {
    image.data.set(pixels.subarray((height - 1 - y) * row, (height - y) * row), y * row);
  }
  const png = document.createElement('canvas');
  png.width = width;
  png.height = height;
  png.getContext('2d').putImageData(image, 0, 0);
  const dataUrl = png.toDataURL('image/png');

  // Advisory GPU time: the whole Lighting.render per frame, median of TIMED_FRAMES.
  let gpuMs = null;
  const timer = gl.getExtension('EXT_disjoint_timer_query_webgl2');
  if (timer && TIMED_FRAMES > 0) {
    const queries = [];
    for (let i = 0; i < TIMED_FRAMES; i++) {
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

  window.oracleResult = { png: dataUrl.slice(dataUrl.indexOf(',') + 1), renderer: rendererName, gpuMs };
}

main().catch((e) => {
  window.oracleResult = { error: `${e.message}\n${e.stack}` };
});
