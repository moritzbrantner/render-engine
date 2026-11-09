// Runs one oracle scenario on the engine compiled to wasm32 in the browser's WebGPU and leaves
// the result in `window.smokeResult` for run.mjs. Query: scene=<name>, mode=render|time|present|
// particles, frames=n, count=1 (GPU API calls and bytes per frame), spawn, frame (particles).
import init, { Smoke } from './pkg/webgpu_smoke.js';

const q = new URLSearchParams(location.search);

const counters = {};
function countCalls() {
  for (const type of ['GPUDevice', 'GPUQueue', 'GPUCommandEncoder', 'GPURenderPassEncoder', 'GPUComputePassEncoder', 'GPUBuffer', 'GPUTexture']) {
    const proto = window[type].prototype;
    for (const name of Object.getOwnPropertyNames(proto)) {
      const d = Object.getOwnPropertyDescriptor(proto, name);
      if (typeof d.value !== 'function' || name === 'constructor') continue;
      const original = d.value;
      proto[name] = function (...args) {
        const key = `${type}.${name}`;
        counters[key] = (counters[key] ?? 0) + 1;
        if (type === 'GPUQueue' && (name === 'writeBuffer' || name === 'writeTexture')) {
          const data = args[name === 'writeBuffer' ? 2 : 1];
          counters['bytesUploaded'] = (counters['bytesUploaded'] ?? 0) + (data?.byteLength ?? data?.length ?? 0);
        }
        return original.apply(this, args);
      };
    }
  }
}

function base64(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

async function main() {
  if (q.get('count') === '1') countCalls();
  const t0 = performance.now();
  await init();
  const wasmMs = performance.now() - t0;
  const smoke = await Smoke.create();
  const info = JSON.parse(smoke.info());
  const startupMs = performance.now() - t0;
  const scene = q.get('scene');
  const mode = q.get('mode') ?? 'render';
  const result = { wasmMs, startupMs, info, mode, scene };
  if (mode === 'render' || mode === 'particles') {
    const t1 = performance.now();
    if (q.get('count') === '1') for (const k in counters) delete counters[k];
    const f = mode === 'render' ? await smoke.render(scene) : await smoke.particles(scene, Number(q.get('spawn')), Number(q.get('frame')));
    result.firstFrameMs = performance.now() - t1;
    result.timeToFirstFrameMs = performance.now() - t0;
    if (q.get('count') === '1') result.calls = { ...counters };
    result.width = f.width;
    result.height = f.height;
    result.uploadMs = f.upload_ms;
    result.prepareMs = f.prepare_ms;
    result.rgba = base64(f.rgba);
  } else if (mode === 'time') {
    result.timing = JSON.parse(await smoke.time(scene, Number(q.get('frames') ?? 300)));
  } else if (mode === 'present') {
    const canvas = document.getElementById('surface');
    smoke.present(scene, canvas);
    const snapshot = document.createElement('canvas');
    snapshot.width = canvas.width;
    snapshot.height = canvas.height;
    const ctx = snapshot.getContext('2d');
    ctx.drawImage(canvas, 0, 0);
    const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
    result.width = canvas.width;
    result.height = canvas.height;
    result.rgba = base64(new Uint8Array(image.data.buffer));
  }
  window.smokeResult = result;
}

main().catch((e) => {
  window.smokeResult = { error: String(e?.stack ?? e) };
});
