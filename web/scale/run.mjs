// Opens web/scale/ (#172) in Playwright's Chromium or Firefox on the hardware GPU and prints the
// page's JSON result.
//
//   node web/scale/run.mjs [chromium|firefox] [frames]
//
// Chromium runs with --enable-webgpu-developer-features (unquantised timestamps), as
// scripts/web-showcase.sh times it.

import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, firefox } from 'playwright';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const [which = 'chromium', frames = '300'] = process.argv.slice(2);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' };
const server = createServer((req, res) => {
  const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const file = normalize(join(root, path.endsWith('/') ? `${path}index.html` : path));
  if (!file.startsWith(root)) return res.writeHead(403).end();
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
const browser =
  which === 'firefox'
    ? await firefox.launch({ headless: true, firefoxUserPrefs: { 'dom.webgpu.enabled': true, 'gfx.webgpu.ignore-blocklist': true } })
    : await chromium.launch({
        headless: true,
        args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-vulkan=native', '--ignore-gpu-blocklist', '--enable-gpu', '--use-angle=vulkan', '--enable-webgpu-developer-features'],
      });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
await page.goto(`http://127.0.0.1:${server.address().port}/web/scale/?frames=${frames}`);
await page.waitForFunction(() => globalThis.result !== undefined, null, { timeout: 300_000 });
const result = await page.evaluate(() => globalThis.result);
console.log(JSON.stringify({ browser: which, version: browser.version(), ...result, pageErrors: errors }));
await browser.close();
server.close();
process.exit(result.error ? 1 : 0);
