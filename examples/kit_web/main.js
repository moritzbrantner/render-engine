// The asset-tooling kit bursts on WebGL 2 (docs/particle-kit-showcase.md): the scene's emitters
// as `kit_showcase --plain <scene>` draws them natively, built here the way a web consumer would
// (the kit's descriptor presets mapped as `kit_showcase.rs`'s `burst()` maps them) and drawn by
// packages/three-lighting's `ParticleEmitter`. Query: scene=combat_bursts|zoo_bursts. Leaves
// `{ png (base64), renderer }` in `window.kitResult` for run.mjs.

import { Matrix4, PerspectiveCamera, RGBAFormat, SRGBColorSpace, Scene, UnsignedByteType, WebGLRenderTarget, WebGLRenderer } from 'three';
import { Lighting, ParticleEmitter, loadToneMappingLuts, textureFromData } from '/packages/three-lighting/index.js';

const WIDTH = 1280;
const HEIGHT = 720;
// kit_showcase's timestep, warm-up and `--plain` background.
const DT = 1 / 60;
const PLAIN_BACKGROUND = [0.02, 0.02, 0.02];

// The emitters of each scene, as kit_showcase's `build()` lists them: (effect, preset, blend).
const SCENES = {
  combat_bursts: {
    kit: 'combat-effects',
    effects: [
      ['spark', 'standard', 'add'],
      ['block-spark', 'heavy', 'add'],
      ['impact-dust', 'standard', 'alpha'],
      ['trail', 'standard', 'alpha'],
    ],
  },
  zoo_bursts: {
    kit: 'zoo-effects',
    effects: [
      ['dust', 'standard', 'alpha'],
      ['leaf', 'standard', 'alpha'],
      ['splash', 'standard', 'alpha'],
    ],
  },
};

/** An 8-bit RGBA, non-interlaced PNG's pixels (the kits' atlases), without the canvas's premultiplication. */
async function decodePng(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let pos = 8;
  let width = 0;
  let height = 0;
  const idat = [];
  while (pos < bytes.length) {
    const length = view.getUint32(pos);
    const type = String.fromCharCode(...bytes.subarray(pos + 4, pos + 8));
    const data = bytes.subarray(pos + 8, pos + 8 + length);
    if (type === 'IHDR') {
      width = view.getUint32(pos + 8);
      height = view.getUint32(pos + 12);
      if (data[8] !== 8 || data[9] !== 6 || data[12] !== 0) throw new Error('expected an 8-bit RGBA, non-interlaced PNG');
    }
    if (type === 'IDAT') idat.push(data);
    pos += 12 + length;
  }
  const raw = new Uint8Array(await new Response(new Blob(idat).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer());
  const stride = width * 4;
  const out = new Uint8Array(stride * height);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let x = 0; x < stride; x++) {
      const a = x >= 4 ? out[y * stride + x - 4] : 0;
      const b = y > 0 ? out[(y - 1) * stride + x] : 0;
      const c = x >= 4 && y > 0 ? out[(y - 1) * stride + x - 4] : 0;
      let v = line[x];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c;
        const [pa, pb, pc] = [Math.abs(p - a), Math.abs(p - b), Math.abs(p - c)];
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
      out[y * stride + x] = v & 255;
    }
  }
  return { width, height, rgba: out };
}

/** kit_showcase's `box_mips`: a 2×2 box filter down to 1×1, rounded. */
function boxMips(rgba, width, height) {
  const levels = [rgba];
  let [w, h] = [width, height];
  while (w > 1 || h > 1) {
    const [nw, nh] = [Math.max(1, w >> 1), Math.max(1, h >> 1)];
    const prev = levels[levels.length - 1];
    const next = new Uint8Array(nw * nh * 4);
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        for (let c = 0; c < 4; c++) {
          let sum = 0;
          for (const [dx, dy] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
            sum += prev[(Math.min(y * 2 + dy, h - 1) * w + Math.min(x * 2 + dx, w - 1)) * 4 + c];
          }
          next[(y * nw + x) * 4 + c] = (sum + 2) >> 2;
        }
      }
    }
    levels.push(next);
    [w, h] = [nw, nh];
  }
  return levels;
}

/** The kit's flipbook effects by key (kit_showcase's `load_effects`). */
async function loadEffects(kit) {
  const root = `/target/kits/${kit}`;
  const json = async (path) => (await fetch(`${root}/${path}`)).json();
  const [manifest, descriptor] = await Promise.all([json('manifest.json'), json(`${kit}.descriptor.json`)]);
  const effects = {};
  for (const e of descriptor.effects) {
    const asset = manifest.assets.find((a) => a.key === e.key);
    const png = await decodePng(new Uint8Array(await (await fetch(`${root}/${asset.path}`)).arrayBuffer()));
    // Straight-alpha sRGB atlases, linear filtering, clamped (cells never wrap).
    const texture = textureFromData({
      width: png.width,
      height: png.height,
      srgb: true,
      levels: boxMips(png.rgba, png.width, png.height),
      sampler: { addressModeU: 'clampToEdge', addressModeV: 'clampToEdge', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' },
    });
    effects[e.key.replace(/\.flipbook$/, '')] = {
      texture,
      grid: [e.atlas.columns, e.atlas.rows],
      frames: e.atlas.frameCount,
      scale: e.scale,
      presets: e.presets,
    };
  }
  return effects;
}

/** kit_showcase's `burst()`: an asset-tooling burst preset as an emitter repeating every `period` s. */
function burst(fx, preset, period, blend, seed) {
  const p = fx.presets[preset];
  return {
    capacity: p.count * Math.max(1, Math.ceil(p.lifetime / period)) * 2,
    spawner: { count: p.count, spawnDuration: p.emitDuration, period, cycles: 0 },
    seed,
    init: [
      { kind: 'positionSphere', center: [0, 0, 0], radius: 0.04, dimension: 'volume' },
      { kind: 'velocityCone', direction: p.direction, spread: p.spread, speed: { min: p.speed[0], max: p.speed[1] } },
      { kind: 'lifetime', value: p.lifetime },
      { kind: 'color', value: [1, 1, 1, 1] },
    ],
    accel: p.gravity,
    sizeOverLife: [
      [0, [fx.scale, fx.scale]],
      [1, [fx.scale * p.endScale, fx.scale * p.endScale]],
    ],
    colorOverLife: p.fadeOut ? { gradient: [[0, [1, 1, 1, 1]], [1, [1, 1, 1, 0]]], blend: 'overwrite' } : null,
    orient: 'cameraPlane',
    texture: { texture: fx.texture, mapping: 'modulate', flipbook: { grid: fx.grid, frames: fx.frames } },
    roundness: 0,
    depthFade: blend === 'alpha' ? 0.15 : 0,
    blend,
  };
}

async function main() {
  const name = new URLSearchParams(location.search).get('scene');
  const spec = SCENES[name];
  if (!spec) throw new Error(`no kit scene ${name}`);
  const effects = await loadEffects(spec.kit);

  const canvas = document.createElement('canvas');
  const renderer = new WebGLRenderer({ canvas, antialias: false, alpha: false });
  renderer.setPixelRatio(1);
  renderer.setSize(WIDTH, HEIGHT, false);
  const scene = new Scene();
  const n = spec.effects.length;
  const emitters = spec.effects.map(([key, preset, blend], i) => {
    const e = new ParticleEmitter(burst(effects[key], preset, 0.6, blend, 100 + i));
    e.setTransform(new Matrix4().makeTranslation((i - (n - 1) / 2) * 1.1, 0.25, 0));
    scene.add(e);
    return e;
  });
  // kit_showcase's warm-up: four bursts and 0.12 s into the fifth, at its timestep.
  const steps = Math.round((0.6 * 4 + 0.12) / DT);
  for (let i = 0; i < steps; i++) for (const e of emitters) e.step(DT);

  // kit_showcase's `daylight` camera; the plain scene has no sun or sky.
  const camera = new PerspectiveCamera(50, WIDTH / HEIGHT, 0.1, 10_000);
  camera.position.set(0, 1, 2.6);
  camera.lookAt(0, 0.4, 0);
  const lighting = new Lighting(await loadToneMappingLuts());
  lighting.clearColor = PLAIN_BACKGROUND;
  lighting.ev100 = 14;
  lighting.tonemapping = 'tonyMcMapface';
  lighting.ambient = { color: [1, 1, 1], brightness: 0 };

  const target = new WebGLRenderTarget(WIDTH, HEIGHT, { type: UnsignedByteType, format: RGBAFormat, colorSpace: SRGBColorSpace, depthBuffer: false });
  lighting.render(renderer, scene, camera, target);
  const pixels = new Uint8Array(WIDTH * HEIGHT * 4);
  renderer.readRenderTargetPixels(target, 0, 0, WIDTH, HEIGHT, pixels);
  const image = new ImageData(WIDTH, HEIGHT);
  const row = WIDTH * 4;
  for (let y = 0; y < HEIGHT; y++) image.data.set(pixels.subarray((HEIGHT - 1 - y) * row, (HEIGHT - y) * row), y * row);
  const out = document.createElement('canvas');
  out.width = WIDTH;
  out.height = HEIGHT;
  out.getContext('2d').putImageData(image, 0, 0);
  const url = out.toDataURL('image/png');
  const gl = renderer.getContext();
  const debug = gl.getExtension('WEBGL_debug_renderer_info');
  window.kitResult = {
    png: url.slice(url.indexOf(',') + 1),
    renderer: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER),
    alive: emitters.map((e) => e.aliveCount),
  };
}

main().catch((e) => {
  window.kitResult = { error: `${e.message}\n${e.stack}` };
});
