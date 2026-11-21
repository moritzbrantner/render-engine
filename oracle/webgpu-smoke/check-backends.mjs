// Browser regression check of the backend selection (docs/webgl2-fallback.md): runs run.mjs's
// `select` mode in headless Chromium configurations and asserts which backend each one starts.
// The image check is `compare select` (scripts/browser-backends.sh runs both).
//
//   node oracle/webgpu-smoke/check-backends.mjs <out dir>
//
// Needs the oracle's exports of sun_shadows and textured_props in <out dir> and the wasm build
// (pkg/). The WebGPU case needs a WebGPU adapter in headless Chromium; SKIP_WEBGPU=1 skips it.

import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const run = fileURLToPath(new URL('./run.mjs', import.meta.url));
const [out] = process.argv.slice(2);
if (!out) {
  console.error('usage: node oracle/webgpu-smoke/check-backends.mjs <out dir>');
  process.exit(2);
}

function select(scene, env, ...args) {
  const stdout = execFileSync('node', [run, out, 'select', scene, ...args], {
    env: { ...process.env, SMOKE_TIMEOUT_MS: '300000', ...env },
    encoding: 'utf8',
    maxBuffer: 1 << 26,
  });
  return JSON.parse(stdout);
}

const cases = [
  {
    name: 'WebGPU available: the engine renders',
    skip: process.env.SKIP_WEBGPU === '1',
    run: () => select('sun_shadows', {}),
    expect: (r) => r.backend === 'webgpu' && r.attempts.length === 0 && r.width === 1280,
  },
  {
    name: 'WebGPU initialisation fails (glTF files withheld from the wasm): falls back to WebGL 2',
    skip: process.env.SKIP_WEBGPU === '1',
    run: () => select('textured_props', {}, 'assets=0'),
    expect: (r) => r.backend === 'webgl2' && r.attempts[0]?.backend === 'webgpu' && r.attempts[0].reason.startsWith('initialisation failed'),
  },
  {
    name: 'No WebGPU: WebGL 2 renders (sun_shadows)',
    run: () => select('sun_shadows', { SMOKE_WEBGPU: '0' }),
    expect: (r) => r.backend === 'webgl2' && r.attempts[0]?.backend === 'webgpu' && r.status === 'Rendering with WebGL 2 (three-lighting)',
  },
  {
    name: 'No WebGPU: WebGL 2 renders (textured_props)',
    run: () => select('textured_props', { SMOKE_WEBGPU: '0' }),
    expect: (r) => r.backend === 'webgl2',
  },
  {
    name: 'Neither: the page shows the error with both reasons',
    run: () => select('sun_shadows', { SMOKE_WEBGPU: '0', SMOKE_WEBGL: '0' }),
    expect: (r) =>
      r.backend === null &&
      r.status.startsWith('No supported graphics backend.') &&
      r.status.includes('WebGPU: ') &&
      r.status.includes('WebGL 2: ') &&
      r.attempts.length === 2,
  },
];

let failed = 0;
for (const c of cases) {
  if (c.skip) {
    console.log(`skip  ${c.name}`);
    continue;
  }
  let result;
  try {
    result = c.run();
  } catch (e) {
    failed++;
    console.log(`FAIL  ${c.name}: ${e.message}`);
    continue;
  }
  const ok = c.expect(result);
  if (!ok) failed++;
  const detail = result.attempts?.map((a) => `${a.backend}: ${a.reason}`).join('; ');
  console.log(`${ok ? 'ok   ' : 'FAIL '} ${c.name} -> ${result.backend}${detail ? ` (${detail})` : ''}`);
}
process.exit(failed ? 1 : 0);
