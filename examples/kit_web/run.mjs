// Renders the asset-tooling kit bursts on WebGL 2 (main.js) in headless Chromium and compares
// each with the native `kit_showcase --plain <scene>` image (docs/particle-kit-showcase.md).
// Evidence, not a gate.
//
//   bash scripts/fetch-kits.sh
//   cargo run --release --locked --example kit_showcase -- --plain combat_bursts   # and zoo_bursts
//   node examples/kit_web/run.mjs [scene...]          default: combat_bursts zoo_bursts
//
// Writes target/kit-showcase/<scene>.web.png. KIT_VK_ICD points ANGLE's Vulkan at an ICD (the
// oracle's three.js side passes lavapipe's on a software adapter). KIT_ANGLE=gl runs ANGLE on
// the system's OpenGL instead (on WSL with GALLIUM_DRIVER=d3d12: the Windows GPU through D3D12).

import { createServer } from 'node:http';
import { readFile, writeFile } from 'node:fs/promises';
import { inflateSync } from 'node:zlib';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const out = join(root, 'target/kit-showcase');
const scenes = process.argv.slice(2).length ? process.argv.slice(2) : ['combat_bursts', 'zoo_bursts'];
const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json' };
const server = createServer(async (req, res) => {
  const file = normalize(join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname)));
  if (!file.startsWith(root)) return res.writeHead(403).end();
  try {
    const body = await readFile(file);
    res.writeHead(200, { 'content-type': types[extname(file)] ?? 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404).end();
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const browser = await chromium.launch({
  headless: true,
  args:
    process.env.KIT_ANGLE === 'gl'
      ? ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=gl']
      : ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=vulkan', '--enable-features=Vulkan', '--use-vulkan=native'],
  env: { ...process.env, ...(process.env.KIT_VK_ICD ? { VK_ICD_FILENAMES: process.env.KIT_VK_ICD } : {}) },
});

/** RGBA8 pixels of an 8-bit RGB or RGBA PNG. */
function pixels(buf) {
  let pos = 8;
  let w;
  let h;
  let bpp;
  const idat = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString('ascii', pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + len);
    if (type === 'IHDR') [w, h, bpp] = [data.readUInt32BE(0), data.readUInt32BE(4), data[9] === 6 ? 4 : 3];
    if (type === 'IDAT') idat.push(data);
    pos += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const stride = w * bpp;
  const rows = Buffer.alloc(stride * h);
  const rgba = Buffer.alloc(w * h * 4);
  for (let y = 0; y < h; y++) {
    const f = raw[y * (stride + 1)];
    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? rows[y * stride + x - bpp] : 0;
      const b = y ? rows[(y - 1) * stride + x] : 0;
      const c = x >= bpp && y ? rows[(y - 1) * stride + x - bpp] : 0;
      let v = raw[y * (stride + 1) + 1 + x];
      if (f === 1) v += a;
      else if (f === 2) v += b;
      else if (f === 3) v += (a + b) >> 1;
      else if (f === 4) {
        const p = a + b - c;
        const [pa, pb, pc] = [Math.abs(p - a), Math.abs(p - b), Math.abs(p - c)];
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      rows[y * stride + x] = v & 255;
    }
    for (let x = 0; x < w; x++) for (let k = 0; k < 4; k++) rgba[(y * w + x) * 4 + k] = k < bpp ? rows[y * stride + x * bpp + k] : 255;
  }
  return { w, h, rgba };
}

let failed = false;
for (const scene of scenes) {
  const page = await browser.newPage();
  const logs = [];
  page.on('console', (m) => logs.push(`${m.type()}: ${m.text()}`));
  page.on('pageerror', (e) => logs.push(`pageerror: ${e.message}`));
  await page.goto(`http://127.0.0.1:${server.address().port}/examples/kit_web/index.html?scene=${scene}`);
  const result = await (await page.waitForFunction(() => window.kitResult, null, { timeout: 300_000 })).jsonValue();
  await page.close();
  if (result.error) {
    failed = true;
    console.error(`${scene}: ${result.error}\n${logs.join('\n')}`);
    continue;
  }
  const web = Buffer.from(result.png, 'base64');
  await writeFile(join(out, `${scene}.web.png`), web);
  // Against the native image: how many pixels differ, by how much, and how much of the frame
  // each side covers with particles (differs from the plain background).
  const native = pixels(await readFile(join(out, `${scene}.plain.png`)));
  const ours = pixels(web);
  const bg = [...native.rgba.subarray(0, 3)];
  let [over2, sum, coveredNative, coveredWeb] = [0, 0, 0, 0];
  const n = native.w * native.h;
  for (let i = 0; i < n * 4; i += 4) {
    let d = 0;
    for (let k = 0; k < 3; k++) d = Math.max(d, Math.abs(native.rgba[i + k] - ours.rgba[i + k]));
    sum += d;
    if (d > 2) over2++;
    if ([0, 1, 2].some((k) => Math.abs(native.rgba[i + k] - bg[k]) > 2)) coveredNative++;
    if ([0, 1, 2].some((k) => Math.abs(ours.rgba[i + k] - bg[k]) > 2)) coveredWeb++;
  }
  const pct = (v) => `${((100 * v) / n).toFixed(2)} %`;
  console.log(
    `${scene}: ${result.renderer}; alive ${result.alive.join('/')}; covered native ${pct(coveredNative)}, web ${pct(coveredWeb)}; ` +
      `pixels differing by more than 2 steps ${pct(over2)}, mean max-channel difference ${(sum / n).toFixed(3)} steps`,
  );
}
await browser.close();
server.close();
process.exit(failed ? 1 : 0);
