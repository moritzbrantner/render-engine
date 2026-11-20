// Browser backend selection (backend.js): WebGPU, then WebGL 2, then a clear error. Runs without
// a GPU or a browser: stand-in navigators and canvases.
import { describe, expect, test } from 'bun:test';
import { NoGraphicsBackendError, probeWebGl2, probeWebGpu, selectBackend } from './backend.js';

const webgpu = (requestAdapter = async () => ({})) => ({ gpu: { requestAdapter } });
const noWebgpu = {};

function canvas({ context = true, extensions = ['EXT_color_buffer_float'], throws = null } = {}) {
  const made = { lost: false };
  const gl = {
    getExtension(name) {
      if (name === 'WEBGL_lose_context') return { loseContext: () => (made.lost = true) };
      return extensions.includes(name) ? {} : null;
    },
  };
  const create = () => ({
    getContext(kind) {
      if (throws) throw new Error(throws);
      return kind === 'webgl2' && context ? gl : null;
    },
  });
  create.made = made;
  return create;
}

const inits = (calls = []) => ({
  webgpu: async () => (calls.push('webgpu'), 'engine'),
  webgl2: async () => (calls.push('webgl2'), 'three'),
});

describe('selection', () => {
  test('prefers WebGPU when it has an adapter', async () => {
    const calls = [];
    const r = await selectBackend(inits(calls), { navigator: webgpu(), createCanvas: canvas() });
    expect(r).toEqual({ backend: 'webgpu', value: 'engine', attempts: [] });
    expect(calls).toEqual(['webgpu']);
  });

  test('falls back to WebGL 2 without navigator.gpu', async () => {
    const calls = [];
    const r = await selectBackend(inits(calls), { navigator: noWebgpu, createCanvas: canvas() });
    expect(r.backend).toBe('webgl2');
    expect(r.value).toBe('three');
    expect(r.attempts).toEqual([{ backend: 'webgpu', reason: expect.stringContaining('navigator.gpu is missing') }]);
    expect(calls).toEqual(['webgl2']);
  });

  test('falls back to WebGL 2 when there is no adapter', async () => {
    const r = await selectBackend(inits(), { navigator: webgpu(async () => null), createCanvas: canvas() });
    expect(r.backend).toBe('webgl2');
    expect(r.attempts[0].reason).toContain('no WebGPU adapter');
  });

  test('falls back to WebGL 2 when requestAdapter rejects', async () => {
    const nav = webgpu(async () => {
      throw new Error('GPU process crashed');
    });
    const r = await selectBackend(inits(), { navigator: nav, createCanvas: canvas() });
    expect(r.backend).toBe('webgl2');
    expect(r.attempts[0].reason).toBe('requestAdapter failed: GPU process crashed');
  });

  test('`backends` forces and orders the candidates', async () => {
    const calls = [];
    const r = await selectBackend(inits(calls), { backends: ['webgl2'], navigator: webgpu(), createCanvas: canvas() });
    expect(r.backend).toBe('webgl2');
    expect(calls).toEqual(['webgl2']);
    await expect(selectBackend(inits(), { backends: ['webgl1'] })).rejects.toThrow(TypeError);
  });

  test('a backend the page does not offer is skipped with a reason', async () => {
    const r = await selectBackend({ webgl2: async () => 'three' }, { navigator: webgpu(), createCanvas: canvas() });
    expect(r.backend).toBe('webgl2');
    expect(r.attempts).toEqual([{ backend: 'webgpu', reason: 'not offered by this page' }]);
  });
});

describe('initialisation failure', () => {
  test('a WebGPU initialiser that throws falls through to WebGL 2', async () => {
    const r = await selectBackend(
      {
        webgpu: async () => {
          throw new Error('device lost during pipeline creation');
        },
        webgl2: async () => 'three',
      },
      { navigator: webgpu(), createCanvas: canvas() },
    );
    expect(r.backend).toBe('webgl2');
    expect(r.attempts).toEqual([{ backend: 'webgpu', reason: 'initialisation failed: device lost during pipeline creation' }]);
  });

  test('neither backend: one error naming both reasons', async () => {
    const error = await selectBackend(inits(), { navigator: noWebgpu, createCanvas: canvas({ context: false }) }).catch((e) => e);
    expect(error).toBeInstanceOf(NoGraphicsBackendError);
    expect(error.attempts.map((a) => a.backend)).toEqual(['webgpu', 'webgl2']);
    expect(error.message).toStartWith('No supported graphics backend. WebGPU: navigator.gpu is missing');
    expect(error.message).toContain('WebGL 2: no WebGL 2 context');
  });

  test('both initialisers throw: both failures are reported', async () => {
    const fail = (m) => async () => {
      throw new Error(m);
    };
    const error = await selectBackend({ webgpu: fail('wasm'), webgl2: fail('shader') }, { navigator: webgpu(), createCanvas: canvas() }).catch((e) => e);
    expect(error).toBeInstanceOf(NoGraphicsBackendError);
    expect(error.message).toContain('WebGPU: initialisation failed: wasm');
    expect(error.message).toContain('WebGL 2: initialisation failed: shader');
  });
});

describe('probes', () => {
  test('WebGPU probe without a navigator', async () => {
    expect((await probeWebGpu(undefined)).ok).toBe(false);
  });

  test('WebGL 2 probe needs float render targets and releases its context', () => {
    const ok = canvas();
    expect(probeWebGl2(ok)).toEqual({ ok: true });
    expect(ok.made.lost).toBe(true);
    expect(probeWebGl2(canvas({ extensions: [] }))).toEqual({ ok: false, reason: 'missing EXT_color_buffer_float' });
  });

  test('WebGL 2 probe survives a throwing getContext', () => {
    expect(probeWebGl2(canvas({ throws: 'blocked' }))).toEqual({ ok: false, reason: "getContext('webgl2') failed: blocked" });
  });
});
