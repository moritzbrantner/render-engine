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

/** Shadows of a point or spot light (Bevy `shadow_depth_bias`, `shadow_normal_bias`, `shadow_map_near_z`, `soft_shadows_enabled`). */
export interface LightShadows {
  /** World-space offset towards the light (m). */
  depthBias: number;
  /** Offset along the surface normal, in shadow texels. */
  normalBias: number;
  /** Near plane of the shadow projection (m); casters closer to the light cast nothing. */
  nearZ: number;
  /** Percentage-closer soft shadows with a penumbra from the light's radius (Bevy PCSS); otherwise `Lighting.shadowFilter`. */
  soft: boolean;
}

/** Bevy's `PointLight` shadow defaults. */
export const POINT_LIGHT_SHADOWS: Readonly<LightShadows>;
/** Bevy's `SpotLight` shadow defaults. */
export const SPOT_LIGHT_SHADOWS: Readonly<LightShadows>;

/** Bevy `PointLight`. */
export interface PointLight {
  position: Vec3;
  /** Linear RGB. */
  color: Vec3;
  /** Luminous power, lumens. */
  intensity: number;
  /** Distance where the light has faded out (m; Bevy's smooth window over 1/d²). */
  range: number;
  /** Radius of the light's sphere (m): widens highlights and soft-shadow penumbrae. */
  radius?: number;
  /** Cube-map shadows, or null for none. At most 4 shadowed point lights per frame. */
  shadows: LightShadows | null;
}

/** Bevy `SpotLight`. Shadows: a perspective map at `Lighting.shadowMapSize`, at most 4 per frame. */
export interface SpotLight extends PointLight {
  /** Direction the light points. */
  direction: Vec3;
  /** Radians off the axis where the falloff starts. */
  innerAngle: number;
  /** Radians off the axis where the light ends (below π/2). */
  outerAngle: number;
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
  /**
   * `shadowMapSize`: texels per side of each sun cascade and spot map (Bevy
   * `DirectionalLightShadowMap`, default 2048); `pointShadowMapSize`: per point-light cube face
   * (Bevy `PointLightShadowMap`, default 1024).
   */
  constructor(luts: ToneMappingLuts, options?: { shadowMapSize?: number; pointShadowMapSize?: number });
  /** Linear RGB where nothing is drawn: tone mapped, not exposed. */
  clearColor: Vec3;
  ev100: number;
  tonemapping: ToneMapping;
  /** Filter of the sun's shadows and of point and spot shadows that are not soft. */
  shadowFilter: ShadowFilter;
  sun: Sun | null;
  /** Shaded before the spot lights and the sun (Bevy's order), shadow casters first. */
  pointLights: PointLight[];
  spotLights: SpotLight[];
  ambient: Ambient;
  environment: EnvironmentMap | null;
  fog: Fog | null;
  readonly shadowMapSize: number;
  readonly pointShadowMapSize: number;
  /** Shadow cascades, spot maps and point-light cube faces, MSAA HDR pass, tone mapping and sRGB output into `target` (null: canvas). */
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
