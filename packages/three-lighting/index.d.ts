import type {
  CubeTexture,
  Data3DTexture,
  PerspectiveCamera,
  Scene,
  ShaderMaterial,
  WebGLRenderer,
  WebGLRenderTarget,
} from 'three';

export type Vec3 = [number, number, number];
export type Vec4 = [number, number, number, number];

/** Bevy `CascadeShadowConfigBuilder` and the sun's shadow biases. */
export interface SunShadows {
  numCascades: number;
  minDistance: number;
  firstCascadeFarBound: number;
  maxDistance: number;
  overlapProportion: number;
  /** World-space offset towards the light (m). */
  depthBias: number;
  /** Offset along the surface normal, in shadow texels. */
  normalBias: number;
}

export const DEFAULT_SUN_SHADOWS: Readonly<SunShadows>;
export function singleCascade(maxDistance: number): SunShadows;
export function farBounds(shadows: SunShadows): number[];
export function cascadeRanges(shadows: SunShadows): [number, number][];
export function fitCascade(
  near: number,
  far: number,
  mapSize: number,
  sunDirection: Vec3,
  fovY: number,
  aspect: number,
  worldFromView: ArrayLike<number>,
): { clipFromWorld: number[]; cascadeFromWorld: number[]; texelSize: number; diameter: number; depthRange: number };

/** Bevy `DirectionalLight`. */
export interface Sun {
  /** Direction the light travels. */
  direction: Vec3;
  /** Linear RGB. */
  color: Vec3;
  /** Lux. */
  illuminance: number;
  shadows: SunShadows | null;
}

/** Bevy `GlobalAmbientLight`. */
export interface Ambient {
  color: Vec3;
  /** cd/m². */
  brightness: number;
}

/** Bevy `EnvironmentMapLight`: prefiltered cube maps from `loadCubemap`. */
export interface EnvironmentMap {
  diffuse: CubeTexture;
  specular: CubeTexture;
  /** cd/m² scale. */
  intensity: number;
}

export type FogFalloffValue =
  | { type: 'linear'; start: number; end: number }
  | { type: 'exponential'; density: number }
  | { type: 'exponentialSquared'; density: number }
  | { type: 'atmospheric'; extinction: Vec3; inscattering: Vec3 };

export const KOSCHMIEDER_CONTRAST_THRESHOLD: number;
export const FogFalloff: {
  linear(start: number, end: number): FogFalloffValue;
  exponential(density: number): FogFalloffValue;
  exponentialSquared(density: number): FogFalloffValue;
  atmospheric(extinction: Vec3, inscattering: Vec3): FogFalloffValue;
  fromVisibility(visibility: number): FogFalloffValue;
  fromVisibilitySquared(visibility: number): FogFalloffValue;
  fromVisibilityColors(visibility: number, extinction: Vec4, inscattering: Vec4): FogFalloffValue;
};

/** Bevy `DistanceFog`. */
export interface Fog {
  /** Linear RGBA; alpha scales how much fog is blended in. */
  color: Vec4;
  /** Sun inscattering colour, alpha its strength (0 = off). */
  directionalLightColor?: Vec4;
  directionalLightExponent?: number;
  falloff: FogFalloffValue;
}

export type ToneMapping = 'tonyMcMapface' | 'agx' | 'acesFitted';
export type ShadowFilter = 'gaussian' | 'hardware2x2';

export interface ToneMappingLuts {
  tonyMcMapface: Data3DTexture;
  agx: Data3DTexture;
}

export function exposureFromEv100(ev100: number): number;
export function decodeKtx2(bytes: Uint8Array): Promise<{
  vkFormat: number;
  faceCount: number;
  levels: { width: number; height: number; depth: number; faces: (Uint32Array | Uint16Array)[] }[];
}>;
export function loadLut(url: string): Promise<Data3DTexture>;
export function loadCubemap(url: string): Promise<CubeTexture>;
export function loadToneMappingLuts(urls?: { tonyMcMapface?: string; agx?: string }): Promise<ToneMappingLuts>;

export interface LitMaterialParameters {
  /** Linear RGB. */
  baseColor?: Vec3;
  alpha?: number;
  metallic?: number;
  perceptualRoughness?: number;
  /** Dielectric specular reflectance, 0.5 = 4% F0. */
  reflectance?: number;
  /** Emissive radiance, cd/m². */
  emissive?: Vec3;
}

/** Bevy `StandardMaterial` (untextured subset), shaded by `Lighting`. */
export class LitMaterial extends ShaderMaterial {
  constructor(parameters?: LitMaterialParameters);
  readonly isLitMaterial: true;
}

export class Lighting {
  constructor(luts: ToneMappingLuts, options?: { shadowMapSize?: number });
  /** Linear RGB where nothing is drawn: tone mapped, not exposed. */
  clearColor: Vec3;
  ev100: number;
  tonemapping: ToneMapping;
  shadowFilter: ShadowFilter;
  sun: Sun | null;
  ambient: Ambient;
  environment: EnvironmentMap | null;
  fog: Fog | null;
  readonly shadowMapSize: number;
  /** Shadow cascades, MSAA HDR pass, tone mapping and sRGB output into `target` (null: canvas). */
  render(
    renderer: WebGLRenderer,
    scene: Scene,
    camera: PerspectiveCamera,
    target?: WebGLRenderTarget | null,
    width?: number,
    height?: number,
  ): void;
  dispose(): void;
}
