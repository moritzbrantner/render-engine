// The WebGL 2 particle simulation (particles.js) against render-engine's: the same random
// streams, spawner counts, slot allocation and update. Runs without a GPU.
import { describe, expect, test } from 'bun:test';
import { Matrix4 } from 'three';
import { ParticleEmitter, Spawner, nextSeed, pcgHash, sampleGradient, toFloat01 } from './particles.js';

const DT = 1 / 64;

describe('random streams', () => {
  test('pcg_hash matches hanabi', () => {
    // render-engine `pcg_hash_matches_hanabi`.
    expect(pcgHash(0)).toBe(129_708_002);
    expect(pcgHash(1)).toBe(2_831_084_092);
    expect(toFloat01(0x3f80_0000)).toBe(0);
    expect(toFloat01(0xffff_ffff)).toBeLessThan(1);
  });

  test('the seed chain matches render-engine next_seed', () => {
    // render-engine `seed_chain_vectors` pins the same values: change both together.
    const vectors = [
      [0, 3_442_241_407],
      [1, 3_543_144_545],
      [7, 1_789_459_390],
      [42, 572_990_626],
      [0xdead_beef, 3_547_977_055],
      [0xffff_ffff, 2_331_729_548],
    ];
    for (const [seed, next] of vectors) expect(nextSeed(seed)).toBe(next);
  });
});

describe('spawner and gradients', () => {
  // render-engine `rate_spawner_carries_fractions_and_burst_spawns_once`, through emitters.
  const counts = (spawner, steps) => {
    const e = new ParticleEmitter({ capacity: 1024, spawner, init: [{ kind: 'lifetime', value: 100 }] });
    return Array.from({ length: steps }, () => (e.step(DT), e.spawnCount));
  };

  test('rate carries fractions, once spawns once', () => {
    expect(counts(Spawner.rate(96), 6)).toEqual([1, 2, 1, 2, 1, 2]);
    expect(counts(Spawner.rate(96), 64).reduce((a, b) => a + b)).toBe(96);
    expect(counts(Spawner.once(96), 3)).toEqual([96, 0, 0]);
  });

  test('gradients interpolate and clamp', () => {
    const g = [
      [0, [0.1]],
      [1, [0.02]],
    ];
    expect(sampleGradient(g, -1)).toEqual([0.1]);
    expect(sampleGradient(g, 0.5)[0]).toBeCloseTo(0.06, 6);
    expect(sampleGradient(g, 2)).toEqual([0.02]);
  });
});

describe('simulation', () => {
  const sparks = (extra = {}) =>
    new ParticleEmitter({
      capacity: 64,
      spawner: Spawner.once(8),
      seed: 0x5eed,
      init: [
        { kind: 'positionSphere', center: [0, 0, 0], radius: 0.5, dimension: 'volume' },
        { kind: 'velocitySphere', center: [0, 0, 0], speed: 2 },
        { kind: 'lifetime', value: 4 * DT },
      ],
      accel: [0, -9.81, 0],
      ...extra,
    });

  test('replays: the same seed gives the same particles, another seed others', () => {
    const [a, b, c] = [sparks(), sparks(), sparks({ seed: 1 })];
    for (const e of [a, b, c]) e.step(DT);
    const positions = (e) => Array.from(e.geometry.attributes.particlePosition.array.subarray(0, 24));
    expect(positions(a)).toEqual(positions(b));
    expect(positions(a)).not.toEqual(positions(c));
    // Spawn i takes slot i: its stream is pcg_hash(i ^ seed), the first spawn step's seed.
    expect(a.aliveCount).toBe(8);
    expect(a.geometry.instanceCount).toBe(8);
  });

  test('a particle spawned in a step is aged and moved in that step', () => {
    const e = new ParticleEmitter({
      capacity: 4,
      spawner: Spawner.once(1),
      init: [{ kind: 'velocity', value: [1, 0, 0] }],
      accel: [0, -2, 0],
    });
    e.step(DT);
    const p = e.geometry.attributes.particlePosition.array;
    expect(p[0]).toBeCloseTo(DT, 7);
    expect(p[1]).toBeCloseTo(-2 * DT * DT, 7);
  });

  test('reaps at the lifetime and reuses the slot from the dead stack', () => {
    const e = sparks({ spawner: Spawner.rate(64) });
    for (let i = 0; i < 4; i++) e.step(DT);
    // One per step, lifetime four steps: the first dies on step 4.
    expect(e.aliveCount).toBe(3);
    expect(e.dead[3]).toBe(0);
    e.step(DT);
    // The next spawn takes dead row 3: slot 0 again.
    expect(e.aliveCount).toBe(3);
    expect(Array.from(e.current.subarray(0, 3))).toContain(0);
  });

  test('spawns beyond the capacity are dropped', () => {
    const e = sparks({ capacity: 5, spawner: Spawner.once(8) });
    e.step(DT);
    expect(e.spawnCount).toBe(8);
    expect(e.aliveCount).toBe(5);
  });

  test('bursts spawn at their own transform, inactive spawner or not', () => {
    const e = new ParticleEmitter({ capacity: 16, spawner: Spawner.rate(64), active: false, init: [{ kind: 'lifetime', value: 1 }] });
    e.burst(2, new Matrix4().makeTranslation(10, 0, 0));
    e.burst(3, new Matrix4().makeTranslation(0, 0, -5));
    e.step(DT);
    expect(e.spawnCount).toBe(5);
    expect(e.aliveCount).toBe(5);
    const p = e.geometry.attributes.particlePosition.array;
    expect([p[0], p[3], p[6], p[9], p[12]]).toEqual([10, 10, 0, 0, 0]);
    expect(p[14]).toBe(-5);
    e.step(DT);
    expect(e.spawnCount).toBe(0);
  });

  test('the seed chain advances whether the spawner runs or not', () => {
    const [a, b] = [sparks({ active: false }), sparks({ active: false })];
    a.step(DT);
    expect(a.seed).toBe(nextSeed(0x5eed));
    b.step(DT);
    b.step(DT);
    expect(b.seed).toBe(nextSeed(nextSeed(0x5eed)));
  });

  test('setParticles replaces the particles through the render modifiers', () => {
    const e = sparks({
      sizeOverLife: [
        [0, [1, 2]],
        [1, [3, 4]],
      ],
      colorOverLife: { gradient: [[0, [1, 0, 0, 1]]], blend: 'multiply' },
    });
    e.setParticles([
      { position: [1, 2, 3], age: 0.5, lifetime: 1, color: [0.5, 1, 1, 0.5] },
      { position: [0, 0, 0], age: 2, lifetime: 1 },
      { position: [4, 5, 6] },
    ]);
    expect(e.aliveCount).toBe(2);
    const a = e.geometry.attributes;
    expect(Array.from(a.particlePosition.array.subarray(0, 6))).toEqual([1, 2, 3, 4, 5, 6]);
    expect(Array.from(a.particleSize.array.subarray(0, 2))).toEqual([2, 3]);
    expect(Array.from(a.particleColor.array.subarray(0, 4))).toEqual([0.5, 0, 0, 0.5]);
    // Later steps animate them; spawns take the slots after them.
    e.step(DT);
    expect(e.aliveCount).toBe(2 + 8);
  });

  test('flipbook sprite runs once over the life', () => {
    const e = new ParticleEmitter({
      capacity: 1,
      spawner: Spawner.once(1),
      init: [{ kind: 'lifetime', value: 8 * DT }],
      texture: { texture: null, mapping: 'modulate', flipbook: { grid: [2, 2], frames: 4 } },
    });
    const sprites = Array.from({ length: 7 }, () => (e.step(DT), e.geometry.attributes.particleSprite.array[0]));
    expect(sprites).toEqual([0, 1, 1, 2, 2, 3, 3]);
  });

  test('lighting, fog and soft edges are reported as unsupported', () => {
    expect(sparks({ lighting: { perceptualRoughness: 1, metallic: 0, reflectance: 0 }, fog: true, depthFade: 0.2 }).unsupported).toEqual([
      'lighting',
      'fog',
      'depthFade',
    ]);
    expect(sparks().unsupported).toEqual([]);
  });

  test('descriptions are validated', () => {
    expect(() => new ParticleEmitter({ capacity: 0, spawner: Spawner.once(1) })).toThrow(RangeError);
    expect(() => new ParticleEmitter({ capacity: 1 })).toThrow(TypeError);
    expect(() => sparks({ blend: 'multiply' })).toThrow(TypeError);
  });
});
