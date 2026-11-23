// Drives the WebGPU smoke in headless Chromium, with the same Playwright the oracle's three.js
// side uses (`oracle/three/render.mjs`).
//
//   node oracle/webgpu-smoke/run.mjs <out dir> <mode> <scenario> [key=value ...]
//
// Modes: render, present, time, particles, meshes (#116), probe, select. Writes <out>/<scenario>.<tag>.rgba (or
// .png) and .json, where tag is webgpu (render, particles), select.<backend> (select: select.webgpu,
// select.webgl2 or select.null when neither started) or the mode; SMOKE_TAG overrides it. A
// particle run writes <scenario>.<capture>.<tag>.rgba per capture and the plate, and reads the
// oracle's <scenario>.bevy.json from <out>; the page reads the glTF files a scenario needs from
// the repository's assets/ (#129). Set SMOKE_FALLBACK=1 for Chromium's software
// adapter (SwiftShader), SMOKE_DEV=1 for --enable-webgpu-developer-features (unquantised
// timestamps; measurement only), SMOKE_BROWSER=firefox for Playwright's Firefox, SMOKE_DPR=n for a
// device pixel ratio of n.
//
// Mode `showcase` opens the browser showcase (web/showcase/, #128) instead, with <scenario> as the
// shooter frame and its `task=` (capture, time, startup, lose [how=crash], resize, status); the page names
// the tag (showcase-<view>, ...).

import { createServer } from 'node:http';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, firefox } from 'playwright';

// The repository root, so the `select` mode can load packages/three-lighting, three.js and the
// oracle's three.js scene; /out/ is the output directory (the scenario exports it reads).
const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const [outDir, mode, scene, ...rest] = process.argv.slice(2);
if (!outDir || !mode || !scene) {
  console.error('usage: node oracle/webgpu-smoke/run.mjs <out dir> <mode> <scenario> [key=value ...]');
  process.exit(2);
}
const out = resolve(outDir);
mkdirSync(out, { recursive: true });
const types = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json' };
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const [base, rel] = path.startsWith('/out/') ? [out, path.slice(5)] : [root, path.slice(1)];
  const file = normalize(join(base, rel));
  if (!file.startsWith(base)) {
    res.writeHead(403).end();
    return;
  }
  try {
    const body = readFileSync(file);
    res.writeHead(200, {
      'content-type': types[extname(file)] ?? 'application/octet-stream',
      // Cross-origin isolation: performance.now() at 5 µs instead of 100 µs in Chromium.
      'cross-origin-opener-policy': 'same-origin',
      'cross-origin-embedder-policy': 'require-corp',
    });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
// SMOKE_WEBGPU=0 and SMOKE_WEBGL=0 switch the APIs off, for the `select` mode's fallback and
// error paths. Without WebGPU the GL device stays the same (Vulkan on, so ANGLE does not drop to
// SwiftShader, whose sRGB stores round differently from the oracle's adapter). Chromium 159 offers a
// hardware adapter on Linux without --enable-unsafe-webgpu, and only disabling WebGPUService too
// takes it away.
const args =
  process.env.SMOKE_WEBGPU === '0'
    ? ['--enable-features=Vulkan', '--disable-features=WebGPU,WebGPUService', '--use-vulkan=native', '--ignore-gpu-blocklist', '--enable-gpu']
    : ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-vulkan=native', '--ignore-gpu-blocklist', '--enable-gpu'];
if (process.env.SMOKE_WEBGL === '0') args.push('--disable-webgl');
if (process.env.SMOKE_FALLBACK === '1') args.push('--use-webgpu-adapter=swiftshader');
if (process.env.SMOKE_DEV === '1') args.push('--enable-webgpu-developer-features');
// SMOKE_ANGLE=gl: WebGL through ANGLE on the system's OpenGL rather than Vulkan (on WSL with
// GALLIUM_DRIVER=d3d12, the Windows GPU through D3D12, which no Vulkan driver reaches). WebGPU
// keeps its Vulkan adapter.
if (process.env.SMOKE_ANGLE === 'gl') args.push('--use-angle=gl');
// SMOKE_BROWSER=firefox: Playwright's Firefox, the same switches as prefs (WebGPU is behind
// `dom.webgpu.enabled` there; the Chromium-only switches do not apply).
const browser =
  process.env.SMOKE_BROWSER === 'firefox'
    ? await firefox.launch({
        headless: true,
        firefoxUserPrefs: {
          'dom.webgpu.enabled': process.env.SMOKE_WEBGPU !== '0',
          // Firefox's driver blocklist disables WebGPU on Mesa older than 25.0.4 (its bug 1979007),
          // llvmpipe included; the gate wants lavapipe, the oracle's adapter, whatever the Mesa.
          // Without a display Firefox has no WebGPU at all ("disabled by blocklist"): run it
          // under Xvfb where there is none (CI: xvfb-run).
          'gfx.webgpu.ignore-blocklist': true,
          ...(process.env.SMOKE_WEBGL === '0' ? { 'webgl.disabled': true } : {}),
        },
      })
    : await chromium.launch({ headless: true, args });
const page = await browser.newPage({ deviceScaleFactor: Number(process.env.SMOKE_DPR ?? 1) });
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));
const query = new URLSearchParams([['scene', scene], ['mode', mode], ...rest.map((kv) => kv.split('='))]);
const t0 = Date.now();
const path = mode === 'showcase' ? '/web/showcase/index.html' : '/oracle/webgpu-smoke/index.html';
await page.goto(`http://127.0.0.1:${server.address().port}${path}?${query}`);
// The showcase's `task=lose how=crash`: a real device loss, by crashing Chromium's GPU process
// once the page asks for it.
if (mode === 'showcase' && query.get('how') === 'crash') {
  await page.waitForFunction(() => window.smokeCrashGpu || window.smokeResult, null, { polling: 100, timeout: 120_000 });
  if (await page.evaluate(() => window.smokeCrashGpu)) {
    const cdp = await browser.newBrowserCDPSession();
    await cdp.send('Browser.crashGpuProcess');
  }
}
const handle = await page
  // Polled on a timer: animation frames can stop (Firefox after a destroyed device).
  .waitForFunction(() => window.smokeResult, null, { polling: 100, timeout: Number(process.env.SMOKE_TIMEOUT_MS ?? 300_000) })
  .catch((e) => {
    console.error(`webgpu smoke: no result (${e.message})\n${logs.join('\n')}`);
    process.exit(1);
  });
const result = await handle.jsonValue();
result.navigationMs = Date.now() - t0;
result.browser = browser.version();
result.logs = logs;
await browser.close();
server.close();
if (result.error) {
  console.error(`webgpu smoke failed: ${result.error}\n${logs.join('\n')}`);
  process.exit(1);
}
const tag =
  process.env.SMOKE_TAG ??
  result.tag ??
  (mode === 'render' || mode === 'particles' ? 'webgpu' : mode === 'select' ? `select.${result.backend}` : mode);
if (result.rgba) writeFileSync(join(out, `${scene}.${tag}.rgba`), Buffer.from(result.rgba, 'base64'));
if (result.png) writeFileSync(join(out, `${scene}.${tag}.png`), Buffer.from(result.png, 'base64'));
// Particle runs: the plate and each capture, <scenario>.<label>.<tag>.rgba.
for (const { label, rgba } of result.images ?? []) writeFileSync(join(out, `${scene}.${label}.${tag}.rgba`), Buffer.from(rgba, 'base64'));
delete result.rgba;
delete result.png;
delete result.images;
writeFileSync(join(out, `${scene}.${tag}.json`), JSON.stringify(result, null, 1));
console.log(JSON.stringify(result));
