// The showcase page (#128): shooter frames rendered live by render-engine compiled to wasm, on
// WebGPU through render-engine-web, behind the browser backend selection of
// packages/three-lighting/backend.js. The shooter frames have no three.js export yet, so this page
// offers no WebGL 2 backend: without WebGPU it shows NoGraphicsBackendError's message.
//
// Serve the repository root and open /web/showcase/ (build first: scripts/build-web.sh).
// Query: frame=<name> (or scene=<name>), view=orbit|eye|<orbit degrees>, backend=webgpu.
// task=capture|time|startup|lose|resize|status runs one evidence step for
// oracle/webgpu-smoke/run.mjs (mode `showcase`) and leaves its result in window.smokeResult.

import { NoGraphicsBackendError, selectBackend } from '../../packages/three-lighting/backend.js';

const moduleAt = performance.now();
const FRAMES = ['tick1422', 'tick1782'];
const q = new URLSearchParams(location.search);
const task = q.get('task');
const $ = (id) => document.getElementById(id);
const status = $('status');
const canvas = $('view');
if (task === 'capture' || task === 'lose') document.body.classList.add('capture');

const frame = q.get('frame') ?? q.get('scene') ?? FRAMES[0];
const names = FRAMES.includes(frame) ? FRAMES : [frame, ...FRAMES];

function onEvent(kind, json) {
  const detail = JSON.parse(json);
  if (kind === 'lost') {
    status.removeAttribute('role');
    status.textContent = `GPU device lost (${detail.reason}): creating a new one…`;
  } else if (kind === 'ready' && detail.generation > 1) {
    status.textContent = `Recovered on a new device (${detail.generation - 1} lost so far): ${detail.adapter}`;
  } else if (kind === 'failed') {
    status.setAttribute('role', 'alert');
    status.textContent = `The GPU device was lost and could not be re-created: ${detail.error}`;
  }
}

async function startWebGpu() {
  const t0 = performance.now();
  const wasm = await import('./pkg/shooter_showcase.js');
  await wasm.default();
  const t1 = performance.now();
  const frames = await Promise.all(
    names.map(async (name) => {
      const response = await fetch(new URL(`../../tests/fixtures/shooter/${name}.json`, import.meta.url));
      if (!response.ok) throw new Error(`frame ${name}: HTTP ${response.status}`);
      return response.text();
    }),
  );
  const t2 = performance.now();
  const showcase = await wasm.ShooterShowcase.start(canvas, names, frames, onEvent);
  showcase.select(names.indexOf(frame));
  showcase.set_view(q.get('view') ?? 'orbit');
  return { showcase, startup: { moduleAt, wasmMs: t1 - t0, fetchMs: t2 - t1, startedAt: performance.now() } };
}

const nextFrame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));
async function frames(n) {
  for (let i = 0; i < n; i++) await nextFrame();
}
const stats = (showcase) => JSON.parse(showcase.stats());

function base64(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

/** Draws a frame now and reads the canvas back in the same task. */
function capture(showcase) {
  if (!showcase.render_now()) throw new Error('no frame to capture');
  const copy = document.createElement('canvas');
  copy.width = canvas.width;
  copy.height = canvas.height;
  const ctx = copy.getContext('2d');
  ctx.drawImage(canvas, 0, 0);
  return { width: copy.width, height: copy.height, rgba: ctx.getImageData(0, 0, copy.width, copy.height).data };
}

async function runTask(showcase, startup) {
  await frames(3);
  const s = stats(showcase);
  const result = { task, frame, view: s.view, stats: s };
  if (task === 'capture') {
    const image = capture(showcase);
    Object.assign(result, { tag: `showcase-${s.view}`, width: image.width, height: image.height, rgba: base64(image.rgba) });
  } else if (task === 'time') {
    const [width, height] = (q.get('size') ?? '1280x720').split('x').map(Number);
    result.timing = JSON.parse(await showcase.time(width, height, Number(q.get('frames') ?? 300)));
    result.tag = `showcase-time-${s.view}`;
  } else if (task === 'startup') {
    // From navigation start (performance's time origin), in milliseconds.
    result.startup = {
      moduleAt: startup.moduleAt,
      wasmMs: startup.wasmMs,
      fetchMs: startup.fetchMs,
      adapterMs: s.adapterMs,
      deviceMs: s.deviceMs,
      rendererMs: s.rendererMs,
      worldMs: s.worldMs,
      firstFrameAt: s.firstFrameAt,
    };
    result.tag = 'showcase-startup';
  } else if (task === 'lose') {
    const before = capture(showcase).rgba;
    // destroy: the device's own destroy(); crash: the harness crashes the GPU process (Chromium).
    if (q.get('how') === 'crash') window.smokeCrashGpu = true;
    else showcase.lose_device();
    const deadline = performance.now() + 20_000;
    // Timers, not animation frames: a browser can stop serving animation frames after a loss
    // (Firefox 159 did, #135), and the check below must see that rather than hang.
    const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    while (showcase.generation() < 2 && performance.now() < deadline) await sleep(50);
    // Frames keep coming after the recovery: the page's own animation frames and the loop's
    // drawn frames, 30 of each within 10 s (the new device's first frames compile pipelines).
    const drawnAtRecovery = stats(showcase).drawn;
    const recoveredAt = performance.now();
    let animationFrames = 0;
    const count = () => {
      animationFrames++;
      if (animationFrames < 30) requestAnimationFrame(count);
    };
    requestAnimationFrame(count);
    const loopFrames = () => stats(showcase).drawn - drawnAtRecovery;
    while ((animationFrames < 30 || loopFrames() < 30) && performance.now() < recoveredAt + 10_000) await sleep(50);
    const framesAfterMs = performance.now() - recoveredAt;
    const after = stats(showcase);
    const image = capture(showcase);
    const same = image.rgba.length === before.length && image.rgba.every((v, i) => v === before[i]);
    Object.assign(result, {
      tag: `showcase-${q.get('how') === 'crash' ? 'crash' : 'lose'}-${s.view}`,
      generation: showcase.generation(),
      losses: after.losses,
      errorsAfter: after.errors,
      status: status.textContent,
      sameImageAfterRecovery: same,
      animationFramesAfter: animationFrames,
      loopFramesAfter: after.drawn - drawnAtRecovery,
      framesAfterMs,
      width: image.width,
      height: image.height,
      rgba: base64(image.rgba),
    });
  } else if (task === 'resize') {
    const sizes = [{ css: 'full', ...pick(s) }];
    for (const [w, h] of [[640, 360], [333, 517]]) {
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      await frames(3);
      sizes.push({ css: `${w}x${h}`, ...pick(stats(showcase)), canvas: `${canvas.width}x${canvas.height}` });
    }
    Object.assign(result, { tag: 'showcase-resize', devicePixelRatio: devicePixelRatio, sizes, errors: stats(showcase).errors });
  }
  return result;
}
const pick = (s) => ({ width: s.width, height: s.height });

function showStats(showcase, backend) {
  const s = stats(showcase);
  const fps = s.frameIntervalMs > 0 ? (1000 / s.frameIntervalMs).toFixed(0) : '–';
  if (!s.adapter) {
    $('stats').textContent = `${backend} · no device (lost ${s.losses}×) · re-creating…`;
    return;
  }
  $('stats').textContent = [
    `${backend} · ${s.adapter} · device ${s.generation}`,
    `${s.width}×${s.height} px (devicePixelRatio ${devicePixelRatio}) · ${fps} fps`,
    `${s.frame}, ${s.view}: ${s.instances} instances · ${s.pointLights} point lights`,
    `CPU prepare ${s.prepareMs.toFixed(3)} · encode ${s.encodeMs.toFixed(3)} · submit ${s.submitMs.toFixed(3)} ms`,
    s.errors ? `GPU errors: ${s.errors} (${s.lastError})` : '',
  ]
    .filter(Boolean)
    .join('\n');
}

async function main() {
  const forced = q.get('backend');
  const result = { mode: 'showcase', task };
  try {
    const chosen = await selectBackend({ webgpu: startWebGpu }, forced ? { backends: [forced] } : {});
    const { showcase, startup } = chosen.value;
    window.showcase = showcase; // for the console and the harness
    status.textContent = 'Rendering with WebGPU (render-engine)';
    for (const [i, name] of names.entries()) $('frame').add(new Option(name, String(i), false, name === frame));
    $('frame').onchange = (e) => showcase.select(Number(e.target.value));
    $('camera').value = q.get('view') === 'eye' ? 'eye' : 'orbit';
    $('camera').onchange = (e) => showcase.set_view(e.target.value);
    $('lose').onclick = () => showcase.lose_device();
    $('controls').hidden = false;
    setInterval(() => showStats(showcase, 'WebGPU'), 500);
    result.backend = chosen.backend;
    result.attempts = chosen.attempts;
    if (task && task !== 'status') Object.assign(result, await runTask(showcase, startup));
  } catch (e) {
    if (!(e instanceof NoGraphicsBackendError)) throw e;
    status.setAttribute('role', 'alert');
    status.textContent = e.message;
    Object.assign(result, { backend: null, attempts: e.attempts });
  }
  result.status = status.textContent;
  window.smokeResult = result;
}

main().catch((e) => {
  status.setAttribute('role', 'alert');
  status.textContent = String(e?.message ?? e);
  window.smokeResult = { error: String(e?.stack ?? e) };
});
