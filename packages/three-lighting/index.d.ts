import type {
  CubeTexture,
  Data3DTexture,
  DataTexture,
  Material,
  PerspectiveCamera,
  Scene,
  ShaderMaterial,
  Texture,
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

/** An unshadowed directional light: the moon (Bevy `DirectionalLight` without shadow maps). */
export interface Moon {
  /** Direction the light travels. */
  direction: Vec3;
  /** Linear RGB. */
  color: Vec3;
  /** Lux. */
  illuminance: number;
}

/** Bevy `lux::RAW_SUNLIGHT`: sunlight above the atmosphere, the sun's illuminance under a sky. */
export const RAW_SUNLIGHT: number;

/** The visible disc of a directional light in the sky (Bevy `SunDisk`). */
export interface SunDiskValue {
  /** Angular diameter in radians. */
  angularSize: number;
  /** Brightness multiplier; 0 hides the disc. */
  intensity: number;
}
export const SunDisk: {
  readonly EARTH: Readonly<SunDiskValue>;
  readonly MOON: Readonly<SunDiskValue>;
  readonly OFF: Readonly<SunDiskValue>;
};

/**
 * Bevy's physically based atmosphere (`Atmosphere::earth`, Hillaire's LUTs) with the sun and moon
 * discs (render-engine `Sky`).
 */
export interface Sky {
  /** Average albedo of the planet's surface, for light reflected into the sky. */
  groundAlbedo: Vec3;
  sunDisk: SunDiskValue;
  moonDisk: SunDiskValue;
  /**
   * Ambient and reflected light from the sky, filtered every frame like Bevy's
   * `GeneratedEnvironmentMapLight` (cd/m² scale); null leaves it to `ambient` and `environment`.
   */
  environmentIntensity: number | null;
}

/** render-engine `Sky::default()`. */
export const DEFAULT_SKY: Readonly<Sky>;

/** Sun and moon directions for a time of day (render-engine `DayCycle`); directions point towards the light. */
export class DayCycle {
  constructor(options?: { noonElevation?: number; moonLagHours?: number });
  /** Radians above the horizon at noon (default 60°). */
  noonElevation: number;
  /** Hours the moon trails the sun (default 12). */
  moonLagHours: number;
  toSun(hours: number): Vec3;
  toMoon(hours: number): Vec3;
  /** The hour between noon and midnight at which the sun stands `elevation` radians above the horizon. */
  afternoonHour(elevation: number): number;
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

export type AddressMode = 'repeat' | 'clampToEdge' | 'mirrorRepeat';
export type FilterMode = 'linear' | 'nearest';

/** How a texture is sampled (glTF `sampler`, render-engine `Sampler`). */
export interface Sampler {
  addressModeU: AddressMode;
  addressModeV: AddressMode;
  magFilter: FilterMode;
  minFilter: FilterMode;
  mipmapFilter: FilterMode;
}

/** glTF's default: repeat, trilinear. */
export const DEFAULT_SAMPLER: Readonly<Sampler>;

/**
 * A material texture as render-engine uploads one (`TextureData`): RGBA8 texels, level 0 first,
 * each level half the previous one (one level: no mipmaps, none generated). `srgb` for base
 * colour and emissive, linear for normal, metallic-roughness and occlusion. Rows run top-down
 * from v = 0, as in glTF.
 */
export function textureFromData(data: {
  width: number;
  height: number;
  srgb: boolean;
  levels: Uint8Array[];
  sampler?: Sampler;
}): DataTexture;

/** Bevy `AlphaMode`. */
export type AlphaMode = 'opaque' | 'mask' | 'blend';

export interface LitMaterialParameters {
  /** Linear RGB (glTF `baseColorFactor`). */
  baseColor?: Vec3;
  alpha?: number;
  metallic?: number;
  perceptualRoughness?: number;
  /** Dielectric specular reflectance, 0.5 = 4% F0. */
  reflectance?: number;
  /** Emissive radiance, cd/m². */
  emissive?: Vec3;
  /** sRGB colour texture; its alpha feeds `alphaMode`. Textures sample the mesh's `uv` (TEXCOORD_0). */
  baseColorTexture?: Texture | null;
  /** sRGB, multiplies `emissive`. */
  emissiveTexture?: Texture | null;
  /** Linear; green scales roughness, blue scales metallic (glTF). */
  metallicRoughnessTexture?: Texture | null;
  /** Linear tangent-space normal map (OpenGL convention, +Y up); needs the mesh's MikkTSpace `tangent` attribute. */
  normalMapTexture?: Texture | null;
  /** Linear; red darkens ambient and environment diffuse light. */
  occlusionTexture?: Texture | null;
  /**
   * 'opaque' (default); 'mask': fragments and shadows cut out below `alphaCutoff`; 'blend':
   * straight alpha, drawn after opaque and masked meshes from back to front by object origin,
   * without depth writes, shadows cut out below 0.05.
   */
  alphaMode?: AlphaMode;
  /** glTF `alphaCutoff` for 'mask' (default 0.5). */
  alphaCutoff?: number;
  /** Draws back faces too, with the normal flipped (glTF `doubleSided`). */
  doubleSided?: boolean;
  /** Outputs the base colour as is, without lighting or exposure (glTF `KHR_materials_unlit`). */
  unlit?: boolean;
}

/**
 * Bevy `StandardMaterial` (the glTF core subset render-engine covers), shaded by `Lighting`. The
 * mesh's `color` attribute (glTF `COLOR_0`) multiplies the base colour. On a `SkinnedMesh` it
 * skins on the GPU like the engine (four influences, linear blend), shadows included. On an
 * `InstancedMesh` each instance takes its `instanceMatrix` and its linear `instanceColor`
 * (`setColorAt`) times the base colour, shadows included. Parameters are fixed at construction.
 */
export class LitMaterial extends ShaderMaterial {
  constructor(parameters?: LitMaterialParameters);
  /** The LitMaterial for a material three's GLTFLoader made (standard, physical or unlit basic). */
  static fromGltf(material: Material): LitMaterial;
  readonly isLitMaterial: true;
  readonly alphaMode: AlphaMode;
  readonly doubleSided: boolean;
  readonly unlit: boolean;
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
  /**
   * Alpha where nothing is drawn (default 1). Below 1 the page shows through a canvas created
   * with `alpha: true`; the output is premultiplied, like three's default canvas.
   */
  clearAlpha: number;
  ev100: number;
  tonemapping: ToneMapping;
  /** Filter of the sun's shadows and of point and spot shadows that are not soft. */
  shadowFilter: ShadowFilter;
  sun: Sun | null;
  /** Shaded with the sun (unshadowed lights first, Bevy's order); its disc shows under a sky. */
  moon: Moon | null;
  /**
   * The atmosphere: LUTs every frame, the sky where nothing is drawn, aerial perspective over the
   * scene, the sun and moon filtered by its transmittance and, with `environmentIntensity`, the
   * ambient light (then `environment` must be null). Needs `EXT_color_buffer_float` and
   * `OES_texture_float_linear`.
   */
  sky: Sky | null;
  /** Shaded before the spot lights and the sun (Bevy's order), shadow casters first. */
  pointLights: PointLight[];
  spotLights: SpotLight[];
  ambient: Ambient;
  environment: EnvironmentMap | null;
  fog: Fog | null;
  readonly shadowMapSize: number;
  readonly pointShadowMapSize: number;
  /**
   * Sky LUTs and environment cubes, shadow cascades, spot maps and point-light cube faces, MSAA
   * HDR pass, tone mapping and sRGB output into `target` (null: canvas). The HDR pass sorts like
   * Bevy (opaque, then masked, then blended back to front), replacing any custom opaque or
   * transparent sort while it draws; with a sky it draws the opaque and masked materials, the
   * sky, then the blended (`transparent`) ones, toggling `material.visible` meanwhile. It
   * renders upside down, as wgpu stores a frame (projection y negated, winding inverted), and
   * the output pass flips it back: a custom shader in the scene that reads `gl_FragCoord.y`
   * sees it mirrored (counting down from the top row). Only objects drawn with a `LitMaterial`
   * cast shadows; others (sprites, overlays) are drawn into the HDR pass as three draws them,
   * tone mapped but not exposed.
   */
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
