// Particles on WebGL 2 (render-engine#126): render-engine's emitters (`src/particles.rs`),
// simulated on the CPU the way its GPU simulation (`src/shaders/particle_sim.wgsl`) replays
// bevy_hanabi, and drawn as camera-facing quads like `src/shaders/particles.wgsl`, in
// `Lighting`'s HDR pass after the blended geometry.
//
// The replay is exact where the engine's is (technical decisions D9): the spawner's per-step
// count, the effect's seed chain (`StdRng::seed_from_u64(seed).random::<u32>()`, ChaCha12), the
// per-particle stream (`pcg_hash(slot ^ seed)`), the dead-slot stack and the init and update
// passes. The engine's update threads reap and append in an order its atomics decide; here they
// run in alive-list order, one valid order of the same pass.
//
// Not on WebGL 2: lit particles (`lighting`), fogged particles (`fog`) and soft particles
// (`depthFade`). An emitter that asks for them draws unlit, unfogged and with hard edges, and
// lists them in `emitter.unsupported`.

import {
  AddEquation,
  CustomBlending,
  DoubleSide,
  DynamicDrawUsage,
  Float32BufferAttribute,
  GLSL3,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
  Matrix4,
  Mesh,
  OneFactor,
  OneMinusSrcAlphaFactor,
  ShaderMaterial,
  SrcAlphaFactor,
  Vector2,
  Vector3,
  ZeroFactor,
} from 'three';

const f32 = Math.fround;

// ---------------------------------------------------------------------------------------------
// Random streams.

/** hanabi's `pcg_hash` (`vfx_common.wesl`), on u32. */
export function pcgHash(input) {
  const state = (Math.imul(input >>> 0, 747796405) + 2891336453) >>> 0;
  const word = Math.imul(((state >>> ((state >>> 28) + 4)) ^ state) >>> 0, 277803737) >>> 0;
  return ((word >>> 22) ^ word) >>> 0;
}

const bits = new Uint32Array(1);
const asFloat = new Float32Array(bits.buffer);

/** hanabi's `to_float01`: 23 random mantissa bits as a float in [0, 1). */
export function toFloat01(u) {
  bits[0] = (u & 0x007fffff) | 0x3f800000;
  return f32(asFloat[0] - 1);
}

/** The per-particle stream of `particle_sim.wgsl` (`frand`, `frand3`, `frand4`). */
class Stream {
  constructor(seed) {
    this.seed = seed >>> 0;
  }

  frand() {
    this.seed = pcgHash(this.seed);
    return toFloat01(pcgHash(this.seed));
  }

  frand3() {
    const out = [];
    for (let i = 0; i < 3; i++) {
      this.seed = pcgHash(this.seed);
      out.push(toFloat01(this.seed));
    }
    return out;
  }

  frand4() {
    const r0 = pcgHash(this.seed);
    const r1 = pcgHash(r0);
    const r2 = pcgHash(r1);
    this.seed = r2;
    return [
      toFloat01(r0),
      toFloat01((((r0 & 0xff000000) >>> 8) | (r1 & 0x0000ffff)) >>> 0),
      toFloat01((((r1 & 0xffff0000) >>> 8) | (r2 & 0x000000ff)) >>> 0),
      toFloat01(r2 >>> 8),
    ];
  }
}

const U64 = (1n << 64n) - 1n;

/** rand_core's `pcg32`, which `seed_from_u64` fills the seed with. */
function pcg32(state) {
  state.v = (state.v * 0x5851f42d4c957f2dn + 0xa17654e46fbe17f3n) & U64;
  const s = state.v;
  const xorshifted = Number((((s >> 18n) ^ s) >> 27n) & 0xffffffffn);
  const rot = Number(s >> 59n);
  return ((xorshifted >>> rot) | (xorshifted << ((32 - rot) & 31))) >>> 0;
}

const rotl = (x, n) => ((x << n) | (x >>> (32 - n))) >>> 0;

function quarter(x, a, b, c, d) {
  x[a] = (x[a] + x[b]) >>> 0;
  x[d] = rotl(x[d] ^ x[a], 16);
  x[c] = (x[c] + x[d]) >>> 0;
  x[b] = rotl(x[b] ^ x[c], 12);
  x[a] = (x[a] + x[b]) >>> 0;
  x[d] = rotl(x[d] ^ x[a], 8);
  x[c] = (x[c] + x[d]) >>> 0;
  x[b] = rotl(x[b] ^ x[c], 7);
}

/**
 * The effect seed of the step after `seed` (hanabi `compile_effects`; render-engine
 * `particles::next_seed`): `StdRng::seed_from_u64(seed).random::<u32>()`, the first word of
 * ChaCha12's first block keyed by eight `pcg32` outputs, counter and nonce zero.
 */
export function nextSeed(seed) {
  const state = { v: BigInt(seed >>> 0) };
  const input = new Uint32Array(16);
  input.set([0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]);
  for (let i = 0; i < 8; i++) input[4 + i] = pcg32(state);
  const x = Array.from(input);
  for (let round = 0; round < 6; round++) {
    quarter(x, 0, 4, 8, 12);
    quarter(x, 1, 5, 9, 13);
    quarter(x, 2, 6, 10, 14);
    quarter(x, 3, 7, 11, 15);
    quarter(x, 0, 5, 10, 15);
    quarter(x, 1, 6, 11, 12);
    quarter(x, 2, 7, 8, 13);
    quarter(x, 3, 4, 9, 14);
  }
  return (x[0] + input[0]) >>> 0;
}

// ---------------------------------------------------------------------------------------------
// Description.

/** render-engine `Spawner`: `count` particles over `spawnDuration` of every `period`, `cycles` times (0 = forever). */
export const Spawner = Object.freeze({
  /** `count` particles once, on the first step (hanabi `SpawnerSettings::once`). */
  once: (count) => ({ count, spawnDuration: 0, period: 0, cycles: 1 }),
  /** A steady rate; fractions carry over between steps (hanabi `SpawnerSettings::rate`). */
  rate: (perSecond) => ({ count: perSecond, spawnDuration: 1, period: 1, cycles: 0 }),
  /** `count` particles at once every `period` seconds, forever (hanabi `SpawnerSettings::burst`). */
  burst: (count, period) => ({ count, spawnDuration: 0, period, cycles: 0 }),
});

/** Most init modifiers per emitter, and keys per gradient (render-engine's limits). */
export const MAX_INIT_MODIFIERS = 8;
export const MAX_GRADIENT_KEYS = 8;

const ORIENT = { cameraPlane: 0, alongVelocity: 1 };
const MAPPING = { modulate: 1, modulateRgb: 2, modulateOpacityFromR: 3 };
const COLOR_BLEND = { overwrite: 0, multiply: 1, add: 2 };

/**
 * An emitter description with the engine's defaults filled in (`src/particles.rs`, `Emitter`).
 * Vectors are arrays; a `Value` is a constant (number or array) or `{ min, max }`.
 */
export function emitterDescription(desc) {
  if (!(desc.capacity > 0)) throw new RangeError('emitter capacity must be positive');
  if (!desc.spawner) throw new TypeError('emitter needs a spawner (Spawner.once/rate/burst)');
  const d = {
    seed: 0,
    init: [],
    accel: [0, 0, 0],
    sizeOverLife: null,
    colorOverLife: null,
    orient: 'cameraPlane',
    texture: null,
    roundness: 0,
    depthFade: 0,
    lighting: null,
    fog: false,
    blend: 'alpha',
    active: true,
    ...desc,
  };
  if (d.init.length > MAX_INIT_MODIFIERS) throw new RangeError(`at most ${MAX_INIT_MODIFIERS} init modifiers`);
  for (const g of [d.sizeOverLife, d.colorOverLife?.gradient]) {
    if (g && (g.length === 0 || g.length > MAX_GRADIENT_KEYS)) throw new RangeError(`gradients take 1 to ${MAX_GRADIENT_KEYS} keys`);
  }
  if (!(d.orient in ORIENT)) throw new TypeError(`orient: ${d.orient}`);
  if (d.blend !== 'add' && d.blend !== 'alpha') throw new TypeError(`blend: ${d.blend}`);
  if (d.texture && !(d.texture.mapping in MAPPING)) throw new TypeError(`texture mapping: ${d.texture.mapping}`);
  if (d.colorOverLife && !(d.colorOverLife.blend in COLOR_BLEND)) throw new TypeError(`colour blend: ${d.colorOverLife.blend}`);
  return d;
}

/** A gradient's value at normalised age `t` (hanabi's generated function: clamped, `mix`). */
export function sampleGradient(keys, t) {
  if (keys.length === 1 || t <= keys[0][0]) return keys[0][1];
  for (let i = 1; i < keys.length; i++) {
    const [ta, va] = keys[i - 1];
    const [tb, vb] = keys[i];
    if (t <= tb) {
      const s = f32((t - ta) / (tb - ta));
      return va.map((a, k) => f32(a * (1 - s) + vb[k] * s));
    }
  }
  return keys[keys.length - 1][1];
}

/** Gradient keys as `{ times, values }` Float32Arrays, `n` components per value. */
function flatten(keys, n) {
  if (!keys) return null;
  return { times: Float32Array.from(keys, (k) => k[0]), values: Float32Array.from(keys.flatMap((k) => k[1])), n };
}

/** `sampleGradient` into `out[at..at + n]`, without allocating (the per-particle path). */
function sampleInto(g, t, out, at) {
  const { times, values, n } = g;
  const k = times.length;
  let i = 1;
  if (k > 1 && t > times[0]) {
    while (i < k && t > times[i]) i++;
    if (i < k) {
      const s = f32((t - times[i - 1]) / (times[i] - times[i - 1]));
      for (let c = 0; c < n; c++) out[at + c] = values[(i - 1) * n + c] * (1 - s) + values[i * n + c] * s;
      return;
    }
    for (let c = 0; c < n; c++) out[at + c] = values[(k - 1) * n + c];
    return;
  }
  for (let c = 0; c < n; c++) out[at + c] = values[c];
}

// ---------------------------------------------------------------------------------------------
// Simulation.

/** hanabi's `EffectSpawner` cycle state (render-engine `SpawnerState`), in f32. */
class SpawnerState {
  cycleTime = 0;
  completedCycles = 0;
  sampledPeriod = 0;
  sampledSpawnDuration = 0;
  sampledCount = 0;
  remainder = 0;

  tick(s, dt) {
    const forever = s.cycles === 0;
    if (!forever && this.completedCycles >= s.cycles) return 0;
    dt = f32(dt);
    for (;;) {
      if (this.sampledPeriod === 0) {
        this.sampledPeriod = f32(s.cycles === 1 ? Math.max(s.spawnDuration, f32(1e-12)) : s.period);
        this.sampledSpawnDuration = f32(s.spawnDuration);
        this.sampledCount = f32(Math.max(s.count, 0));
      }
      const newTime = f32(this.cycleTime + dt);
      if (this.cycleTime <= this.sampledSpawnDuration) {
        const instant = this.sampledSpawnDuration < Math.max(f32(1e-5), f32(dt / 100));
        const add = instant
          ? this.sampledCount
          : f32(
              this.sampledCount *
                Math.min(1, Math.max(0, f32(f32(Math.min(newTime, this.sampledSpawnDuration) - this.cycleTime) / this.sampledSpawnDuration))),
            );
        this.remainder = f32(this.remainder + add);
      }
      this.cycleTime = newTime;
      if (this.cycleTime >= this.sampledPeriod) {
        dt = f32(this.cycleTime - this.sampledPeriod);
        this.cycleTime = 0;
        this.completedCycles++;
        this.sampledPeriod = 0;
        if (!forever && this.completedCycles >= s.cycles) break;
      } else {
        break;
      }
    }
    const count = Math.floor(this.remainder);
    this.remainder = f32(this.remainder - count);
    return count;
  }
}

const value = (v, draw) => (typeof v === 'object' && !Array.isArray(v) ? draw(v.min, v.max) : v);
const scalar = (v, r) => value(v, (min, max) => f32(min + r.frand() * (max - min)));
const vector = (v, r, rand) =>
  value(v, (min, max) => {
    const x = rand.call(r);
    return min.map((a, i) => f32(x[i] * (max[i] - a) + a));
  });

/** glam's `Vec3::any_orthonormal_pair` (Duff et al. 2017) around unit `d`. */
function orthonormalPair([x, y, z]) {
  const sign = z >= 0 ? 1 : -1;
  const k = -1 / (sign + z);
  const b = x * y * k;
  return [
    [1 + sign * x * x * k, sign * b, -sign * x],
    [b, sign + y * y * k, -y],
  ];
}

const normalize = (v) => {
  const l = Math.hypot(...v);
  return v.map((c) => c / l);
};

// Particle state layout per slot (as `particle_sim.wgsl`'s `Particle`).
const STRIDE = 13;
const [POS, AGE, VEL, LIFE, COLOR, SIZE] = [0, 3, 4, 7, 8, 12];

const vertexShader = /* glsl */ `
uniform vec3 cameraRight;
uniform vec3 cameraUp;
uniform vec3 eye;
uniform int orient;
uniform int flipbookColumns;
uniform vec2 flipbookScale;
in vec3 particlePosition;
in float particleSprite;
in vec3 particleVelocity;
in vec2 particleSize;
in vec4 particleColor;
out vec4 vColor;
out vec2 vUv;

void main() {
  vec2 corner = position.xy;
  vec3 axisX = cameraRight;
  vec3 axisY = cameraUp;
  if (orient == 1) {
    vec3 dir = normalize(particlePosition - eye);
    axisX = normalize(particleVelocity);
    axisY = cross(dir, axisX);
  }
  vec2 vpos = corner * particleSize;
  vec3 world = particlePosition + axisX * vpos.x + axisY * vpos.y;
  gl_Position = projectionMatrix * viewMatrix * vec4(world, 1.0);
  vColor = particleColor;
  vec2 uv = vec2(corner.x + 0.5, 0.5 - corner.y);
  if (flipbookColumns > 0) {
    int sprite = int(particleSprite + 0.5);
    vec2 ij = vec2(float(sprite % flipbookColumns), float(sprite / flipbookColumns));
    uv = (ij + uv) * flipbookScale;
  }
  vUv = uv;
}
`;

const fragmentShader = /* glsl */ `
precision highp float;
uniform sampler2D map;
uniform int mapping;
uniform float roundness;
in vec4 vColor;
in vec2 vUv;
layout(location = 0) out highp vec4 fragColor;

void main() {
  vec4 color = vColor;
  if (mapping > 0) {
    vec4 tex = texture(map, vUv);
    if (mapping == 1) color = color * tex;
    else if (mapping == 2) color = vec4(color.rgb * tex.rgb, color.a);
    else color.a = color.a * tex.r;
  }
  if (roundness > 0.0) {
    float n = 2.0 / roundness;
    if (pow(abs(1.0 - 2.0 * vUv.x), n) + pow(abs(1.0 - 2.0 * vUv.y), n) > 1.0) discard;
  }
  fragColor = color;
}
`;

// Two triangles over hanabi's unit quad.
const CORNERS = [-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0];

const right = new Vector3();
const up = new Vector3();
const back = new Vector3();

/**
 * One render-engine emitter on WebGL 2: add it to the scene that `Lighting.render` draws, call
 * `step(dt)` at the game's simulation rate. Its particles draw after the blended geometry, back to
 * front by the emitter's position with the other emitters, unlit, additive (`blend: 'add'`) or
 * alpha-blended into the HDR target, depth-tested but not written.
 *
 * The description is render-engine's `Emitter` in camelCase (`emitterDescription`); a texture
 * is `{ texture: THREE.Texture, mapping, flipbook: { grid: [columns, rows], frames } }`.
 */
export class ParticleEmitter extends Mesh {
  constructor(desc) {
    const geometry = new InstancedBufferGeometry();
    geometry.setAttribute('position', new Float32BufferAttribute(CORNERS, 3));
    const material = new ShaderMaterial({
      glslVersion: GLSL3,
      vertexShader,
      fragmentShader,
      uniforms: {
        cameraRight: { value: new Vector3() },
        cameraUp: { value: new Vector3() },
        eye: { value: new Vector3() },
        orient: { value: 0 },
        flipbookColumns: { value: 0 },
        flipbookScale: { value: new Vector2(1, 1) },
        map: { value: null },
        mapping: { value: 0 },
        roundness: { value: 0 },
      },
      transparent: true,
      // Depth tested like the lit geometry (Lighting's main pass uses three's depth convention).
      depthWrite: false,
      side: DoubleSide,
      blending: CustomBlending,
      blendEquation: AddEquation,
      blendEquationAlpha: AddEquation,
    });
    super(geometry, material);
    this.isParticleEmitter = true;
    this.frustumCulled = false;
    // After the blended geometry; among emitters, back to front by position (Lighting's sort).
    this.renderOrder = 1;
    /** Where new particles spawn (`World::set_emitter_transform`). */
    this.transform = new Matrix4();
    this.spawner = new SpawnerState();
    this.bursts = [];
    /** Particles the spawner and bursts asked for in the last step. */
    this.spawnCount = 0;
    this.setEmitter(desc);
    this.seed = this.desc.seed >>> 0;
  }

  /**
   * Replaces the description (`World::set_emitter`): alive particles keep their attributes and
   * the new modifiers apply to them; a different capacity drops them.
   */
  setEmitter(desc) {
    const d = emitterDescription(desc);
    if (d.capacity !== this.desc?.capacity) this.#allocate(d.capacity);
    this.desc = d;
    this.sizeKeys = flatten(d.sizeOverLife, 2);
    this.colorKeys = flatten(d.colorOverLife?.gradient, 4);
    this.colorBlend = d.colorOverLife ? COLOR_BLEND[d.colorOverLife.blend] : -1;
    this.frames = d.texture?.flipbook?.frames ?? 0;
    this.unsupported = [d.lighting && 'lighting', d.fog && 'fog', d.depthFade > 0 && 'depthFade'].filter(Boolean);
    const u = this.material.uniforms;
    u.orient.value = ORIENT[d.orient];
    u.roundness.value = d.roundness;
    u.mapping.value = d.texture ? MAPPING[d.texture.mapping] : 0;
    u.map.value = d.texture?.texture ?? null;
    const flipbook = d.texture?.flipbook;
    u.flipbookColumns.value = flipbook ? flipbook.grid[0] : 0;
    if (flipbook) u.flipbookScale.value.set(1 / flipbook.grid[0], 1 / flipbook.grid[1]);
    const m = this.material;
    m.blendSrc = SrcAlphaFactor;
    if (d.blend === 'add') {
      // hanabi `AlphaMode::Add`: colour scaled by alpha onto the target, alpha kept.
      Object.assign(m, { blendDst: OneFactor, blendSrcAlpha: ZeroFactor, blendDstAlpha: OneFactor });
    } else {
      Object.assign(m, { blendDst: OneMinusSrcAlphaFactor, blendSrcAlpha: OneFactor, blendDstAlpha: OneMinusSrcAlphaFactor });
    }
    m.needsUpdate = true;
  }

  /** Where new particles spawn; alive ones stay where they are (world-space simulation). */
  setTransform(worldFromEmitter) {
    this.transform.copy(worldFromEmitter);
    this.position.setFromMatrixPosition(worldFromEmitter);
    this.updateMatrix();
  }

  /** `count` particles at `worldFromEmitter` on the next step, active or not (`World::burst`). */
  burst(count, worldFromEmitter = this.transform) {
    if (count > 0) this.bursts.push([count, worldFromEmitter.clone()]);
  }

  /** Particles alive after the last step (or `setParticles`). */
  get aliveCount() {
    return this.alive;
  }

  #allocate(capacity) {
    this.state = new Float32Array(capacity * STRIDE);
    this.dead = Uint32Array.from({ length: capacity }, (_, i) => i);
    this.current = new Uint32Array(capacity);
    this.next = new Uint32Array(capacity);
    this.alive = 0;
    const g = this.geometry;
    const attribute = (name, size) => {
      const a = new InstancedBufferAttribute(new Float32Array(capacity * size), size);
      a.setUsage(DynamicDrawUsage);
      g.setAttribute(name, a);
    };
    attribute('particlePosition', 3);
    attribute('particleSprite', 1);
    attribute('particleVelocity', 3);
    attribute('particleSize', 2);
    attribute('particleColor', 4);
    g.instanceCount = 0;
    const a = g.attributes;
    this.out = {
      position: a.particlePosition.array,
      velocity: a.particleVelocity.array,
      size: a.particleSize.array,
      color: a.particleColor.array,
      sprite: a.particleSprite.array,
    };
  }

  /**
   * Replaces the particles with world-space states (`World::set_particles`): `{ position,
   * velocity, age, lifetime, size, color }`, defaults as `ParticleState::default`. Those whose
   * age has reached their lifetime are left out, and those beyond the capacity.
   */
  setParticles(states) {
    const cap = this.desc.capacity;
    this.dead = Uint32Array.from({ length: cap }, (_, i) => i);
    let n = 0;
    for (const p of states) {
      if (n >= cap) break;
      const age = p.age ?? 0;
      const lifetime = p.lifetime ?? Infinity;
      if (!(age < lifetime)) continue;
      const s = this.state;
      const o = n * STRIDE;
      s.set(p.position ?? [0, 0, 0], o + POS);
      s[o + AGE] = age;
      s.set(p.velocity ?? [0, 0, 0], o + VEL);
      s[o + LIFE] = lifetime;
      s.set(p.color ?? [1, 1, 1, 1], o + COLOR);
      s[o + SIZE] = p.size ?? 1;
      this.current[n] = n;
      this.#quad(n, n);
      n++;
    }
    this.alive = n;
    this.#upload(n);
  }

  /** One simulation step of `dt` seconds (`World::step_particles` for this emitter). */
  step(dt) {
    const d = this.desc;
    const spawn = d.active ? this.spawner.tick(d.spawner, dt) : 0;
    const seed = this.seed;
    this.seed = nextSeed(seed);
    // A burst per entry at its transform, then the spawner's, each taking the next dead rows.
    const entries = [...this.bursts, [spawn, this.transform]];
    this.bursts = [];
    this.spawnCount = entries.reduce((n, [count]) => n + count, 0);
    let rowOffset = 0;
    for (const [count, transform] of entries) {
      for (let i = 0; i < count; i++) {
        const row = this.alive + rowOffset + i;
        if (row >= d.capacity) break;
        const slot = this.dead[row];
        this.#init(slot, new Stream(pcgHash((slot ^ seed) >>> 0)), transform);
        this.current[row] = slot;
      }
      rowOffset += count;
    }
    const n = Math.min(this.alive + this.spawnCount, d.capacity);
    // Update: age, accelerate, move, reap (back on the dead stack) or keep and draw.
    dt = f32(dt);
    const s = this.state;
    const [ax, ay, az] = d.accel;
    let alive = n;
    let drawn = 0;
    for (let j = 0; j < n; j++) {
      const slot = this.current[j];
      const o = slot * STRIDE;
      s[o + AGE] = s[o + AGE] + dt;
      s[o + VEL] = s[o + VEL] + f32(ax * dt);
      s[o + VEL + 1] = s[o + VEL + 1] + f32(ay * dt);
      s[o + VEL + 2] = s[o + VEL + 2] + f32(az * dt);
      for (let k = 0; k < 3; k++) s[o + POS + k] = s[o + POS + k] + f32(s[o + VEL + k] * dt);
      if (s[o + AGE] >= s[o + LIFE]) {
        alive--;
        this.dead[alive] = slot;
        continue;
      }
      this.next[drawn] = slot;
      this.#quad(drawn, slot);
      drawn++;
    }
    this.alive = alive;
    [this.current, this.next] = [this.next, this.current];
    this.#upload(drawn);
  }

  #init(slot, r, transform) {
    const p = { position: [0, 0, 0], velocity: [0, 0, 0], age: 0, lifetime: 1, size: 1, color: [1, 1, 1, 1] };
    for (const m of this.desc.init) {
      switch (m.kind) {
        case 'positionSphere': {
          const radius = m.dimension === 'surface' ? m.radius : f32(Math.pow(r.frand(), 1 / 3) * m.radius);
          const theta = f32(r.frand() * 2 * Math.PI);
          const z = f32(r.frand() * 2 - 1);
          const sinphi = Math.sin(Math.acos(z));
          const dir = [sinphi * Math.cos(theta), sinphi * Math.sin(theta), z];
          p.position = m.center.map((c, i) => f32(c + radius * dir[i]));
          break;
        }
        case 'position':
          p.position = vector(m.value, r, r.frand3);
          break;
        case 'velocitySphere': {
          const speed = scalar(m.speed, r);
          p.velocity = normalize(p.position.map((c, i) => c - m.center[i])).map((c) => f32(c * speed));
          break;
        }
        case 'velocity':
          p.velocity = vector(m.value, r, r.frand3);
          break;
        case 'velocityCone': {
          const axis = normalize(m.direction);
          const c = f32(1 - r.frand() * (1 - f32(Math.cos(m.spread))));
          const phi = f32(r.frand() * 2 * Math.PI);
          const speed = scalar(m.speed, r);
          const s = Math.sqrt(Math.max(1 - c * c, 0));
          const [t1, t2] = orthonormalPair(axis);
          p.velocity = axis.map((a, i) => f32((t1[i] * (s * Math.cos(phi)) + t2[i] * (s * Math.sin(phi)) + a * c) * speed));
          break;
        }
        case 'lifetime':
          p.lifetime = scalar(m.value, r);
          break;
        case 'age':
          p.age = scalar(m.value, r);
          break;
        case 'size':
          p.size = scalar(m.value, r);
          break;
        case 'color':
          p.color = vector(m.value, r, r.frand4);
          break;
        default:
          throw new TypeError(`init modifier ${m.kind}`);
      }
    }
    const e = transform.elements;
    const [x, y, z] = p.position;
    const [vx, vy, vz] = p.velocity;
    const o = slot * STRIDE;
    const s = this.state;
    for (let i = 0; i < 3; i++) {
      s[o + POS + i] = e[i] * x + e[4 + i] * y + e[8 + i] * z + e[12 + i];
      s[o + VEL + i] = e[i] * vx + e[4 + i] * vy + e[8 + i] * vz;
    }
    s[o + AGE] = p.age;
    s[o + LIFE] = p.lifetime;
    s.set(p.color, o + COLOR);
    s[o + SIZE] = p.size;
  }

  /** Instance `j` from slot `slot`, through the render modifiers (`render_attributes`). */
  #quad(j, slot) {
    const s = this.state;
    const out = this.out;
    const o = slot * STRIDE;
    const t = f32(s[o + AGE] / s[o + LIFE]);
    for (let k = 0; k < 3; k++) {
      out.position[j * 3 + k] = s[o + POS + k];
      out.velocity[j * 3 + k] = s[o + VEL + k];
    }
    if (this.sizeKeys) {
      sampleInto(this.sizeKeys, t, out.size, j * 2);
    } else {
      out.size[j * 2] = s[o + SIZE];
      out.size[j * 2 + 1] = s[o + SIZE];
    }
    const color = out.color;
    if (this.colorKeys) {
      sampleInto(this.colorKeys, t, color, j * 4);
      if (this.colorBlend === COLOR_BLEND.multiply) for (let k = 0; k < 4; k++) color[j * 4 + k] *= s[o + COLOR + k];
      else if (this.colorBlend === COLOR_BLEND.add) for (let k = 0; k < 4; k++) color[j * 4 + k] += s[o + COLOR + k];
    } else {
      for (let k = 0; k < 4; k++) color[j * 4 + k] = s[o + COLOR + k];
    }
    out.sprite[j] = this.frames > 0 ? Math.trunc(f32(t * this.frames)) : 0;
  }

  #upload(count) {
    const g = this.geometry;
    g.instanceCount = count;
    for (const name of ['particlePosition', 'particleSprite', 'particleVelocity', 'particleSize', 'particleColor']) {
      const a = g.attributes[name];
      a.clearUpdateRanges();
      if (count > 0) a.addUpdateRange(0, count * a.itemSize);
      a.needsUpdate = true;
    }
  }

  onBeforeRender(renderer, scene, camera) {
    const u = this.material.uniforms;
    camera.matrixWorld.extractBasis(right, up, back);
    u.cameraRight.value.copy(right).normalize();
    u.cameraUp.value.copy(up).normalize();
    u.eye.value.setFromMatrixPosition(camera.matrixWorld);
  }

  dispose() {
    this.geometry.dispose();
    this.material.dispose();
  }
}
