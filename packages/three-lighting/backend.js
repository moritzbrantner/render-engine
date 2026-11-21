// Browser backend selection (docs/webgl2-fallback.md): WebGPU first (render-engine compiled to
// wasm), then WebGL 2 (this package, three.js), else an error that says why each one failed.
//
// render-engine's wgpu renderer does not run on wgpu's WebGL 2 backend (storage buffers, compute,
// cube-array and multisampled depth textures; the probe in the doc), so the WebGL 2 side is this
// package's port of the same shaders, gated against the same Bevy oracle (D8). This module only
// decides; it imports nothing, so a page can choose before it loads either renderer.

/** Backends in the order they are tried. */
export const BACKENDS = Object.freeze(['webgpu', 'webgl2']);

/**
 * WebGL 2 extensions the port needs on every scene: its HDR targets are half-float render
 * targets, which core WebGL 2 cannot render to. (`OES_texture_float_linear` is needed only under
 * a sky; the sky reports its own error.)
 */
export const WEBGL2_EXTENSIONS = Object.freeze(['EXT_color_buffer_float']);

const LABEL = { webgpu: 'WebGPU', webgl2: 'WebGL 2' };

/** Neither backend could start. `attempts` lists `{ backend, reason }` in the order tried. */
export class NoGraphicsBackendError extends Error {
  constructor(attempts) {
    const lines = attempts.map((a) => `${LABEL[a.backend] ?? a.backend}: ${a.reason}`);
    super(
      `No supported graphics backend. ${lines.join('; ')}. ` +
        'Use a current browser with hardware acceleration enabled (WebGPU, or WebGL 2 with float render targets).',
    );
    this.name = 'NoGraphicsBackendError';
    this.attempts = attempts;
  }
}

const message = (e) => (e instanceof Error ? e.message : String(e));

/**
 * Whether WebGPU can give an adapter. `{ ok: true }` or `{ ok: false, reason }`. The adapter is
 * not kept: wgpu requests its own.
 */
export async function probeWebGpu(nav = globalThis.navigator) {
  if (!nav?.gpu) {
    return { ok: false, reason: 'navigator.gpu is missing (no WebGPU in this browser, or not a secure context)' };
  }
  try {
    const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { ok: false, reason: 'no WebGPU adapter (WebGPU disabled or the GPU is blocklisted)' };
    return { ok: true };
  } catch (e) {
    return { ok: false, reason: `requestAdapter failed: ${message(e)}` };
  }
}

/**
 * Whether a WebGL 2 context with `extensions` can be created on a fresh canvas from
 * `createCanvas`. `{ ok: true }` or `{ ok: false, reason }`. The probe context is released.
 */
export function probeWebGl2(createCanvas = () => globalThis.document.createElement('canvas'), extensions = WEBGL2_EXTENSIONS) {
  let gl;
  try {
    gl = createCanvas().getContext('webgl2');
  } catch (e) {
    return { ok: false, reason: `getContext('webgl2') failed: ${message(e)}` };
  }
  if (!gl) return { ok: false, reason: 'no WebGL 2 context (WebGL disabled, or the GPU is blocklisted)' };
  const missing = extensions.filter((name) => !gl.getExtension(name));
  gl.getExtension('WEBGL_lose_context')?.loseContext();
  if (missing.length) return { ok: false, reason: `missing ${missing.join(', ')}` };
  return { ok: true };
}

/**
 * Starts the first backend that both probes and initialises: `init.webgpu()` and
 * `init.webgl2()` are the caller's async initialisers (an initialiser that throws falls through
 * to the next backend, like a failed probe). Returns `{ backend, value, attempts }`, where `value`
 * is the initialiser's result and `attempts` the backends that failed before it. Throws
 * `NoGraphicsBackendError` when none starts.
 *
 * Options: `backends` restricts and orders the candidates (default `BACKENDS`; a page's
 * `?backend=webgl2` maps here), `navigator` and `createCanvas` replace the browser's (tests).
 */
export async function selectBackend(init, { backends = BACKENDS, navigator = globalThis.navigator, createCanvas, extensions } = {}) {
  const probes = {
    webgpu: () => probeWebGpu(navigator),
    webgl2: () => probeWebGl2(createCanvas, extensions),
  };
  const attempts = [];
  for (const backend of backends) {
    if (!probes[backend]) throw new TypeError(`unknown backend ${backend}; expected one of ${BACKENDS.join(', ')}`);
    if (!init[backend]) {
      attempts.push({ backend, reason: 'not offered by this page' });
      continue;
    }
    const probe = await probes[backend]();
    if (!probe.ok) {
      attempts.push({ backend, reason: probe.reason });
      continue;
    }
    try {
      return { backend, value: await init[backend](), attempts };
    } catch (e) {
      attempts.push({ backend, reason: `initialisation failed: ${message(e)}` });
    }
  }
  throw new NoGraphicsBackendError(attempts);
}
