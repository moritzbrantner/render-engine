/** Browser backends in the order `selectBackend` tries them. */
export type Backend = 'webgpu' | 'webgl2';

export const BACKENDS: readonly Backend[];
/** WebGL 2 extensions the port needs on every scene (half-float HDR render targets). */
export const WEBGL2_EXTENSIONS: readonly string[];

export interface Attempt {
  backend: Backend;
  reason: string;
}

export type Probe = { ok: true } | { ok: false; reason: string };

/** Neither backend could start; the message names each backend's reason. */
export class NoGraphicsBackendError extends Error {
  readonly attempts: Attempt[];
}

export function probeWebGpu(navigator?: Navigator | { gpu?: unknown }): Promise<Probe>;
export function probeWebGl2(createCanvas?: () => HTMLCanvasElement | OffscreenCanvas, extensions?: readonly string[]): Probe;

/**
 * Starts the first backend that probes and initialises (WebGPU, then WebGL 2 by default). An
 * initialiser that throws falls through to the next backend. Throws `NoGraphicsBackendError`
 * when none starts.
 */
export function selectBackend<G, W>(
  init: { webgpu?: () => Promise<G>; webgl2?: () => Promise<W> },
  options?: {
    backends?: readonly Backend[];
    navigator?: Navigator | { gpu?: unknown };
    createCanvas?: () => HTMLCanvasElement | OffscreenCanvas;
    extensions?: readonly string[];
  },
): Promise<
  | { backend: 'webgpu'; value: G; attempts: Attempt[] }
  | { backend: 'webgl2'; value: W; attempts: Attempt[] }
>;
