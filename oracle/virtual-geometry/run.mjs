// Run after starting a repository-root HTTP server. GPU runs must be serialized.
// node oracle/virtual-geometry/run.mjs http://127.0.0.1:PORT crag ASSET OUTPUT
import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';
const [origin, kit, asset, out, hierarchy = kit] = process.argv.slice(2);
if (!origin || !kit || !asset || !out) throw new Error('Expected origin, kit, asset, output');
await fs.mkdir(out, { recursive: true });
const reportPath = path.join(out, 'report.json');
let previous;
try { previous = JSON.parse(await fs.readFile(reportPath, 'utf8')); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
if (previous && previous.asset !== asset) throw new Error('Output belongs to another asset; choose a new directory');
const browser = await chromium.launch({ headless: true, args: ['--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=vulkan', '--enable-features=Vulkan', '--use-vulkan=native'] });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  page.on('pageerror', e => console.error(e));
  await page.goto(`${origin}/oracle/virtual-geometry/?kit=${encodeURIComponent(kit)}&asset=${encodeURIComponent(asset)}&hierarchy=${encodeURIComponent(hierarchy)}`);
  await page.waitForFunction(() => window.vgReady, { timeout: 120000 });
  const environment = await page.evaluate(() => window.vg.environment);
  console.log(environment);
  if (previous && JSON.stringify(previous.environment) !== JSON.stringify(environment)) throw new Error('Incomparable browser environment; choose a new output directory');
  const completed = previous?.rows.slice() || [];
  const modes = await page.evaluate(() => window.vg.modes);
  for (const count of [1, 64, 256]) {
    for (let round = 0; round < 2; round++) {
      const order = round ? modes.slice().reverse() : modes;
      for (const mode of order) {
        if (!completed.some(r => r.mode === mode && r.count === count && r.round === round)) {
          console.log(await page.evaluate(o => window.vg.bench(o), { mode, count, round }));
          const current = await page.evaluate(() => window.vg.report());
          completed.push(current.rows.at(-1));
        } else console.log(`reuse ${mode}, ${count}, round ${round}`);
        const report = await page.evaluate(() => window.vg.report());
        report.rows = completed;
        report.loading = 'Latest page load only; frame samples may span resumed invocations on the same reported environment';
        await fs.writeFile(reportPath + '.tmp', JSON.stringify(report, null, 2));
        await fs.rename(reportPath + '.tmp', reportPath);
        if (!round) for (const frame of [0, 60, 120, 180]) {
          const capturePath = path.join(out, `${count}-${mode}-${frame}.png`);
          try { await fs.access(capturePath); continue; }
          catch (error) { if (error.code !== 'ENOENT') throw error; }
          const png = await page.evaluate(o => window.vg.capture(o.mode, o.count, o.frame), { mode, count, frame });
          await fs.writeFile(capturePath, Buffer.from(png.split(',')[1], 'base64'));
        }
      }
    }
  }
} finally {
  await browser.close();
}
