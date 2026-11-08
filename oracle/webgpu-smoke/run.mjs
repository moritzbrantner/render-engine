// Drives the WebGPU smoke in headless Chromium, with the same Playwright the oracle's three.js
// side uses (`oracle/three/render.mjs`).
//
//   node oracle/webgpu-smoke/run.mjs <out dir> <mode> <scenario> [key=value ...]
//
// Modes: render, present, time, particles. Writes <out>/<scenario>.<tag>.rgba and .json, where
// tag is webgpu (render), present or the mode. Set SMOKE_FALLBACK=1 for Chromium's software
// adapter (SwiftShader), SMOKE_DEV=1 for --enable-webgpu-developer-features (unquantised
// timestamps; measurement only).

import { createServer } from 'node:http';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const here = resolve(fileURLToPath(new URL('.', import.meta.url)));
const [outDir, mode, scene, ...rest] = process.argv.slice(2);
if (!outDir || !mode || !scene) {
  console.error('usage: node oracle/webgpu-smoke/run.mjs <out dir> <mode> <scenario> [key=value ...]');
  process.exit(2);
}
const out = resolve(outDir);
mkdirSync(out, { recursive: true });
const types = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' };
const server = createServer((req, res) => {
  const file = normalize(join(here, decodeURIComponent(new URL(req.url, 'http://x').pathname)));
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
const args = ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-vulkan=native', '--ignore-gpu-blocklist', '--enable-gpu'];
if (process.env.SMOKE_FALLBACK === '1') args.push('--use-webgpu-adapter=swiftshader');
if (process.env.SMOKE_DEV === '1') args.push('--enable-webgpu-developer-features');
const browser = await chromium.launch({ headless: true, args });
const page = await browser.newPage();
const logs = [];
page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));
const query = new URLSearchParams([['scene', scene], ['mode', mode], ...rest.map((kv) => kv.split('='))]);
const t0 = Date.now();
await page.goto(`http://127.0.0.1:${server.address().port}/index.html?${query}`);
const handle = await page.waitForFunction(() => window.smokeResult, null, { timeout: Number(process.env.SMOKE_TIMEOUT_MS ?? 300_000) });
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
const tag = process.env.SMOKE_TAG ?? (mode === 'render' ? 'webgpu' : mode);
if (result.rgba) writeFileSync(join(out, `${scene}.${tag}.rgba`), Buffer.from(result.rgba, 'base64'));
delete result.rgba;
writeFileSync(join(out, `${scene}.${tag}.json`), JSON.stringify(result, null, 1));
console.log(JSON.stringify(result));
