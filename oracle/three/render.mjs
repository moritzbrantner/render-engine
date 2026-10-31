// The oracle's three.js side: renders exported scenarios (`<name>.three.json` + `.three.bin`)
// with packages/three-lighting in headless Chromium and writes `<name>.three.png` (and
// `<name>.three.err` on failure).
//
//   node oracle/three/render.mjs <out dir> <scenario>...
//
// THREE_FRAMES=n also times n frames (EXT_disjoint_timer_query_webgl2, advisory).
//
// WebGL runs through ANGLE on Vulkan, on the same kind of adapter as the Bevy side: the oracle
// sets THREE_VK_ICD to lavapipe's ICD on a software adapter (CI), so the device's f16 and sRGB
// stores round like Bevy's. THREE_BACKEND=swiftshader uses Chromium's SwiftShader instead (its
// sRGB store rounds differently from lavapipe's, about one 8-bit step on a sixth of the pixels).

import { createServer } from 'node:http';
import { readFile, writeFile, rm } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const [outDir, ...names] = process.argv.slice(2);
if (!outDir || names.length === 0) {
  console.error('usage: node oracle/three/render.mjs <out dir> <scenario>...');
  process.exit(2);
}
const out = resolve(outDir);

const types = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.json': 'application/json',
  '.bin': 'application/octet-stream',
  '.ktx2': 'application/octet-stream',
};

const server = createServer(async (req, res) => {
  const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const [base, rel] = path.startsWith('/out/') ? [out, path.slice(5)] : [root, path.slice(1)];
  const file = normalize(join(base, rel));
  if (!file.startsWith(base)) {
    res.writeHead(403).end();
    return;
  }
  try {
    const body = await readFile(file);
    res.writeHead(200, { 'content-type': types[extname(file)] ?? 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;

const backend = process.env.THREE_BACKEND ?? 'vulkan';
const vulkan = ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=vulkan', '--enable-features=Vulkan', '--use-vulkan=native'];
const browser = await chromium.launch({
  headless: true,
  args: {
    swiftshader: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--use-gl=angle'],
    vulkan,
  }[backend],
  env: { ...process.env, ...(process.env.THREE_VK_ICD ? { VK_ICD_FILENAMES: process.env.THREE_VK_ICD } : {}) },
});

let failed = false;
for (const name of names) {
  const page = await browser.newPage();
  const logs = [];
  page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
  page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));
  await rm(join(out, `${name}.three.err`), { force: true });
  await rm(join(out, `${name}.three.png`), { force: true });
  try {
    const frames = Number(process.env.THREE_FRAMES ?? 0);
    await page.goto(`${origin}/oracle/three/index.html?scene=${encodeURIComponent(`/out/${name}`)}&frames=${frames}`);
    const result = await page.waitForFunction(() => window.oracleResult, null, { timeout: Number(process.env.THREE_TIMEOUT_MS ?? 600_000) });
    const { png, error, renderer, gpuMs } = await result.jsonValue();
    if (error) throw new Error(error);
    await writeFile(join(out, `${name}.three.png`), Buffer.from(png, 'base64'));
    await writeFile(join(out, `${name}.three.meta.json`), JSON.stringify({ renderer, gpuMs }));
    console.log(`three: ${name} rendered (${renderer})`);
    if (process.env.THREE_DEBUG === '1') console.log(logs.join('\n'));
  } catch (e) {
    failed = true;
    const message = `${e.message}\n${logs.join('\n')}`;
    await writeFile(join(out, `${name}.three.err`), message);
    console.error(`three: ${name} failed: ${message}`);
  }
  await page.close();
}
await browser.close();
server.close();
process.exit(failed ? 1 : 0);
