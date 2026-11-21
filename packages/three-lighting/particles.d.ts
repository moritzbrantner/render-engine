import type { Matrix4, Mesh, Texture } from 'three';

type Vec2 = [number, number];
type Vec3 = [number, number, number];
type Vec4 = [number, number, number, number];

/** A constant, or uniform between `min` and `max` per particle (render-engine `Value`). */
export type Value<T> = T | { min: T; max: T };

/** render-engine `Spawner`: `count` particles over `spawnDuration` of every `period`, `cycles` times (0 = forever). */
export interface SpawnerSettings {
  count: number;
  spawnDuration: number;
  period: number;
  cycles: number;
}

export const Spawner: {
  once(count: number): SpawnerSettings;
  rate(perSecond: number): SpawnerSettings;
  burst(count: number, period: number): SpawnerSettings;
};

/** render-engine `InitModifier`, applied in order. */
export type InitModifier =
  | { kind: 'positionSphere'; center: Vec3; radius: number; dimension: 'volume' | 'surface' }
  | { kind: 'position'; value: Value<Vec3> }
  | { kind: 'velocitySphere'; center: Vec3; speed: Value<number> }
  | { kind: 'velocity'; value: Value<Vec3> }
  | { kind: 'velocityCone'; direction: Vec3; spread: number; speed: Value<number> }
  | { kind: 'lifetime'; value: Value<number> }
  | { kind: 'age'; value: Value<number> }
  | { kind: 'size'; value: Value<number> }
  | { kind: 'color'; value: Value<Vec4> };

/** Keys `[normalised age, value]`, sorted by age. */
export type Gradient<T> = [number, T][];

/** render-engine `Emitter` in camelCase. Omitted fields take the engine's defaults. */
export interface EmitterDescription {
  capacity: number;
  spawner: SpawnerSettings;
  seed?: number;
  init?: InitModifier[];
  accel?: Vec3;
  sizeOverLife?: Gradient<Vec2> | null;
  colorOverLife?: { gradient: Gradient<Vec4>; blend: 'overwrite' | 'multiply' | 'add' } | null;
  orient?: 'cameraPlane' | 'alongVelocity';
  texture?: {
    texture: Texture | null;
    mapping: 'modulate' | 'modulateRgb' | 'modulateOpacityFromR';
    flipbook?: { grid: [number, number]; frames: number } | null;
  } | null;
  roundness?: number;
  /**
   * Soft particles: metres over which a particle fades out in front of the opaque scene (0:
   * hard). `Lighting.render` resolves the opaque draws' depth for it.
   */
  depthFade?: number;
  /** Lit particles: each quad shaded as a StandardMaterial under `Lighting`'s lights. */
  lighting?: { perceptualRoughness: number; metallic: number; reflectance: number } | null;
  /** Fogged particles: `Lighting.fog`, lit or not. */
  fog?: boolean;
  blend?: 'add' | 'alpha';
  active?: boolean;
}

/** render-engine `ParticleState`, world space. */
export interface ParticleState {
  position?: Vec3;
  velocity?: Vec3;
  age?: number;
  lifetime?: number;
  size?: number;
  color?: Vec4;
}

export const MAX_INIT_MODIFIERS: number;
export const MAX_GRADIENT_KEYS: number;

export function pcgHash(input: number): number;
export function toFloat01(u: number): number;
/** render-engine `particles::next_seed`. */
export function nextSeed(seed: number): number;
export function emitterDescription(desc: EmitterDescription): Required<EmitterDescription>;
export function sampleGradient<T extends number[]>(keys: Gradient<T>, t: number): T;

/**
 * One render-engine emitter on WebGL 2, simulated on the CPU like the engine's GPU simulation
 * and drawn in `Lighting`'s HDR pass after the blended geometry.
 */
export class ParticleEmitter extends Mesh {
  constructor(desc: EmitterDescription);
  readonly isParticleEmitter: true;
  readonly desc: Required<EmitterDescription>;
  /** Where new particles spawn (`World::set_emitter_transform`). */
  readonly transform: Matrix4;
  /** The effect seed the next step uses. */
  readonly seed: number;
  /** Particles the spawner and bursts asked for in the last step. */
  readonly spawnCount: number;
  readonly aliveCount: number;
  setEmitter(desc: EmitterDescription): void;
  setTransform(worldFromEmitter: Matrix4): void;
  burst(count: number, worldFromEmitter?: Matrix4): void;
  setParticles(states: ParticleState[]): void;
  step(dt: number): void;
  dispose(): void;
}
