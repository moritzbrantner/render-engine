// render-engine's lighting on three.js (WebGLRenderer, GLSL ES 3.0): Bevy's PBR BRDF, sun
// cascades with Bevy's fitting and filters, point and spot lights, ambient and image-based light
// from prefiltered cube maps, Bevy's atmosphere with the sun and moon and the ambient light it
// generates (sky.js), distance fog and Bevy's tone mappers. Scene values use Bevy's physical
// units (lux, cd/m², EV100, linear RGB), so a frame matches render-engine's and Bevy's (the
// oracle's `three` side).
//
// The consumer owns the WebGLRenderer, the Scene and the camera. Meshes use `LitMaterial`;
// `Lighting.render` draws the sky's LUTs, the shadow maps, the 4× MSAA HDR main pass (with the
// sky), tone mapping and the sRGB output.

import {
  BackSide,
  ClampToEdgeWrapping,
  Color,
  CubeDepthTexture,
  CubeTexture,
  Data3DTexture,
  DataTexture,
  DepthTexture,
  DoubleSide,
  FloatType,
  FrontSide,
  GLSL3,
  GreaterEqualCompare,
  GreaterEqualDepth,
  HalfFloatType,
  LinearFilter,
  LinearMipmapLinearFilter,
  LinearMipmapNearestFilter,
  LinearSRGBColorSpace,
  Matrix4,
  MirroredRepeatWrapping,
  NearestFilter,
  NearestMipmapLinearFilter,
  NearestMipmapNearestFilter,
  NoBlending,
  OrthographicCamera,
  PerspectiveCamera,
  RGBAFormat,
  RedFormat,
  RepeatWrapping,
  RGBFormat,
  SRGBColorSpace,
  ShaderMaterial,
  UnsignedByteType,
  UnsignedInt5999Type,
  UnsignedIntType,
  Vector2,
  Vector3,
  Vector4,
  WebGLCubeRenderTarget,
  WebGLRenderTarget,
} from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { read as readKtx2 } from 'three/addons/libs/ktx-parse.module.js';
import { ZSTDDecoder } from 'three/addons/libs/zstddec.module.js';
import { cascadeRanges, fitCascade } from './cascades.js';
import { ShadowCulling } from './culling.js';
import { ENVIRONMENT_MIPS, INNER_RADIUS, OUTER_RADIUS, SkyPasses, worldToAtmosphere } from './sky.js';
import {
  fullscreenVertex,
  litFragment,
  litVertex,
  localShadowFragment,
  localShadowVertex,
  MAX_CASCADES,
  MAX_POINT_SHADOWS,
  MAX_SPOT_SHADOWS,
  outputFragment,
  shadowFragment,
  shadowVertex,
  tonemapFragment,
} from './shaders.js';

export { DEFAULT_SUN_SHADOWS, singleCascade, farBounds, cascadeRanges, fitCascade } from './cascades.js';

/** Bevy `Exposure::exposure()` for an EV100. */
export function exposureFromEv100(ev100) {
  return Math.pow(2, -ev100) / 1.2;
}

/** Bevy `lux::RAW_SUNLIGHT`: sunlight above the atmosphere, the sun's illuminance under a sky. */
export const RAW_SUNLIGHT = 130_000;

/** The visible disc of a directional light in the sky (Bevy `SunDisk`): radians, brightness. */
export const SunDisk = Object.freeze({
  /** Bevy `SunDisk::EARTH`: the sun seen from Earth (about 32′). */
  EARTH: Object.freeze({ angularSize: 0.00930842, intensity: 1 }),
  /** The moon seen from Earth (about 31′). */
  MOON: Object.freeze({ angularSize: 0.00904, intensity: 1 }),
  /** No disc (Bevy `SunDisk::OFF`). */
  OFF: Object.freeze({ angularSize: 0, intensity: 0 }),
});

/**
 * render-engine's `Sky::default()`: Bevy's Earth atmosphere with its ground albedo, the sun and
 * moon discs, and the sky's environment-map light at intensity 1 (`environmentIntensity: null`
 * leaves ambient light to `Lighting.ambient` and `Lighting.environment`).
 */
export const DEFAULT_SKY = Object.freeze({
  groundAlbedo: Object.freeze([0.3, 0.3, 0.3]),
  sunDisk: SunDisk.EARTH,
  moonDisk: SunDisk.MOON,
  environmentIntensity: 1,
});

/**
 * Sun and moon directions for a time of day (render-engine `DayCycle`). The game owns the clock;
 * this only maps an hour to directions. The sun rises in the east (+X) at 06:00, culminates
 * towards the south (+Z) at `noonElevation` at 12:00 and sets in the west at 18:00; the moon
 * follows the same path `moonLagHours` behind it. Directions point towards the light (negate
 * them for `Lighting.sun.direction`).
 */
export class DayCycle {
  constructor({ noonElevation = (60 * Math.PI) / 180, moonLagHours = 12 } = {}) {
    this.noonElevation = noonElevation;
    this.moonLagHours = moonLagHours;
  }

  /** Unit direction towards the sun at `hours` (any value; the path repeats every 24 h). */
  toSun(hours) {
    const angle = ((hours - 6) / 12) * Math.PI;
    const [c, s] = [Math.cos(angle), Math.sin(angle)];
    return [c, Math.sin(this.noonElevation) * s, Math.cos(this.noonElevation) * s];
  }

  /** Unit direction towards the moon at `hours`. */
  toMoon(hours) {
    return this.toSun(hours - this.moonLagHours);
  }

  /**
   * The hour between noon and midnight at which the sun stands `elevation` radians above (or,
   * negative, below) the horizon; clamped to noon and midnight.
   */
  afternoonHour(elevation) {
    const s = Math.min(Math.max(Math.sin(elevation) / Math.sin(this.noonElevation), -1), 1);
    return 6 + (12 * (Math.PI - Math.asin(s))) / Math.PI;
  }
}

/** Bevy `REVISED_KOSCHMIEDER_CONTRAST_THRESHOLD`. */
export const KOSCHMIEDER_CONTRAST_THRESHOLD = 0.05;

const koschmieder = (visibility, threshold) => -Math.log(threshold) / visibility;

/** Fog falloffs (Bevy `FogFalloff`). */
export const FogFalloff = {
  linear: (start, end) => ({ type: 'linear', start, end }),
  exponential: (density) => ({ type: 'exponential', density }),
  exponentialSquared: (density) => ({ type: 'exponentialSquared', density }),
  /** @param {number[]} extinction linear RGB @param {number[]} inscattering linear RGB */
  atmospheric: (extinction, inscattering) => ({ type: 'atmospheric', extinction, inscattering }),
  fromVisibility: (visibility) =>
    FogFalloff.exponential(koschmieder(visibility, KOSCHMIEDER_CONTRAST_THRESHOLD)),
  fromVisibilitySquared: (visibility) =>
    FogFalloff.exponentialSquared(
      Math.sqrt(koschmieder(visibility, KOSCHMIEDER_CONTRAST_THRESHOLD) / visibility),
    ),
  /** @param {number[]} extinction linear RGBA @param {number[]} inscattering linear RGBA */
  fromVisibilityColors: (visibility, extinction, inscattering) => {
    const k = koschmieder(visibility, KOSCHMIEDER_CONTRAST_THRESHOLD);
    const E = Math.E;
    return FogFalloff.atmospheric(
      [0, 1, 2].map((i) => Math.pow(1 - extinction[i], E) * k * Math.pow(extinction[3], E)),
      [0, 1, 2].map((i) => Math.pow(inscattering[i], E) * k * Math.pow(inscattering[3], E)),
    );
  },
};

const TONEMAPPING = { acesFitted: 0, tonyMcMapface: 1, agx: 2 };

// Negates clip-space y: the main pass renders upside down (Lighting.render).
const FLIP_Y = new Matrix4().makeScale(1, -1, 1);

let zstd;
async function decoder() {
  zstd ??= (async () => {
    const d = new ZSTDDecoder();
    await d.init();
    return d;
  })();
  return zstd;
}

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}

const VK_FORMAT_R16G16B16A16_SFLOAT = 97;
const VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 = 123;

/**
 * Decodes an uncompressed KTX2 file (zstd or no supercompression) into per-level, per-face
 * texel arrays: `E5B9G9R9_UFLOAT_PACK32` (Uint32Array) or `R16G16B16A16_SFLOAT` (Uint16Array).
 */
export async function decodeKtx2(bytes) {
  const container = readKtx2(bytes);
  const { vkFormat, pixelWidth, pixelHeight, pixelDepth, faceCount } = container;
  const texel = { [VK_FORMAT_E5B9G9R9_UFLOAT_PACK32]: 4, [VK_FORMAT_R16G16B16A16_SFLOAT]: 8 }[vkFormat];
  if (!texel) throw new Error(`unsupported KTX2 vkFormat ${vkFormat}`);
  const zstdDecoder = container.supercompressionScheme === 2 ? await decoder() : null;
  if (container.supercompressionScheme !== 0 && !zstdDecoder) {
    throw new Error(`unsupported KTX2 supercompression ${container.supercompressionScheme}`);
  }
  const levels = container.levels.map((level, i) => {
    const data = zstdDecoder
      ? zstdDecoder.decode(level.levelData, level.uncompressedByteLength)
      : level.levelData;
    const width = Math.max(1, pixelWidth >> i);
    const height = Math.max(1, pixelHeight >> i);
    const depth = Math.max(1, pixelDepth >> i);
    const faceBytes = width * height * depth * texel;
    const faces = Array.from({ length: faceCount }, (_, face) => {
      const slice = data.slice(face * faceBytes, (face + 1) * faceBytes);
      return vkFormat === VK_FORMAT_E5B9G9R9_UFLOAT_PACK32
        ? new Uint32Array(slice.buffer)
        : new Uint16Array(slice.buffer);
    });
    return { width, height, depth, faces };
  });
  return { vkFormat, levels, faceCount };
}

function formatOf(vkFormat) {
  return vkFormat === VK_FORMAT_E5B9G9R9_UFLOAT_PACK32
    ? { format: RGBFormat, type: UnsignedInt5999Type }
    : { format: RGBAFormat, type: HalfFloatType };
}

/** A 3D tone-mapping LUT (Bevy's own KTX2 files), trilinear and clamped like Bevy's sampler. */
export async function loadLut(url) {
  const { vkFormat, levels } = await decodeKtx2(await fetchBytes(url));
  const { width, height, depth, faces } = levels[0];
  const { format, type } = formatOf(vkFormat);
  const texture = new Data3DTexture(faces[0], width, height, depth);
  Object.assign(texture, {
    format,
    type,
    minFilter: LinearFilter,
    magFilter: LinearFilter,
    wrapS: ClampToEdgeWrapping,
    wrapT: ClampToEdgeWrapping,
    wrapR: ClampToEdgeWrapping,
    generateMipmaps: false,
    unpackAlignment: 1,
    colorSpace: LinearSRGBColorSpace,
    needsUpdate: true,
  });
  return texture;
}

/**
 * A prefiltered cube map (render-engine's `bake_ibl` output, Bevy's `EnvironmentMapLight`
 * layout): faces in +X, −X, +Y, −Y, +Z, −Z order, every mip level, uploaded unflipped.
 */
export async function loadCubemap(url) {
  const { vkFormat, levels, faceCount } = await decodeKtx2(await fetchBytes(url));
  if (faceCount !== 6) throw new Error(`${url} is not a cube map`);
  const { format, type } = formatOf(vkFormat);
  const level = (l) =>
    Array.from({ length: 6 }, (_, face) => new DataTexture(l.faces[face], l.width, l.height, format, type));
  const cube = new CubeTexture(level(levels[0]));
  // three.js reads an uncompressed cube's mips (base level excluded) as `mipmaps[j].image[face]`.
  cube.mipmaps = levels.slice(1).map((l) => ({ image: level(l) }));
  Object.assign(cube, {
    format,
    type,
    minFilter: levels.length > 1 ? LinearMipmapLinearFilter : LinearFilter,
    magFilter: LinearFilter,
    generateMipmaps: false,
    flipY: false,
    unpackAlignment: 1,
    colorSpace: LinearSRGBColorSpace,
    needsUpdate: true,
  });
  cube.userData.mipLevels = levels.length;
  return cube;
}

/**
 * Bevy's tone-mapping LUTs, shipped with this package (`src/luts`). Bundlers that move the
 * package's files can pass the URLs they serve the two KTX2 files at.
 */
export async function loadToneMappingLuts({
  tonyMcMapface = new URL('../../src/luts/tony_mc_mapface.ktx2', import.meta.url).href,
  agx = new URL('../../src/luts/AgX-default_contrast.ktx2', import.meta.url).href,
} = {}) {
  const [tony, agxLut] = await Promise.all([loadLut(tonyMcMapface), loadLut(agx)]);
  return { tonyMcMapface: tony, agx: agxLut };
}

/** Uniforms every LitMaterial shares with its Lighting (filled per frame). */
function frameUniforms() {
  return {
    exposure: { value: 1 },
    hasSun: { value: 0 },
    directionToLight: { value: new Vector3() },
    sunColor: { value: new Vector3() },
    ambientColor: { value: new Vector3() },
    sunShadows: { value: 0 },
    gaussianFilter: { value: 1 },
    shadowParams: { value: new Vector4() },
    cascadeClipFromWorld: { value: Array.from({ length: MAX_CASCADES }, () => new Matrix4()) },
    cascadeParams: { value: Array.from({ length: MAX_CASCADES }, () => new Vector4()) },
    shadowMapSize: { value: 2048 },
    shadowMap0: { value: null },
    shadowMap1: { value: null },
    shadowMap2: { value: null },
    shadowMap3: { value: null },
    environment: { value: new Vector3() },
    diffuseEnvironmentMap: { value: null },
    specularEnvironmentMap: { value: null },
    fogColor: { value: new Vector4() },
    fogDirectionalLightColor: { value: new Vector4() },
    fogBe: { value: new Vector4() },
    fogBi: { value: new Vector4() },
    localLights: { value: [] },
    // The moon: direction to it (w 1 when there is one), colour × illuminance.
    moonDirectionToLight: { value: new Vector4() },
    moonColor: { value: new Vector3() },
    // The sky's atmosphere: x planet radius, y atmosphere radius, z 1 with a sky, w 1 when the
    // moon is shaded before the sun (Bevy's directional-light order); the sun and moon disc
    // sizes; its transmittance LUT.
    worldToAtmosphere: { value: worldToAtmosphere() },
    atmosphere: { value: new Vector4(INNER_RADIUS, OUTER_RADIUS, 0, 0) },
    disks: { value: new Vector2() },
    atmosphereTransmittance: { value: null },
    ...Object.fromEntries(
      ['pointShadowMap', 'pointShadowDepth', 'spotShadowMap', 'spotShadowDepth'].flatMap((name) =>
        Array.from({ length: 4 }, (_, i) => [`${name}${i}`, { value: null }]),
      ),
    ),
  };
}

/** Bevy's `PointLight` shadow defaults. */
export const POINT_LIGHT_SHADOWS = Object.freeze({ depthBias: 0.08, normalBias: 0.6, nearZ: 0.1, soft: false });
/** Bevy's `SpotLight` shadow defaults. */
export const SPOT_LIGHT_SHADOWS = Object.freeze({ depthBias: 0.02, normalBias: 1.8, nearZ: 0.1, soft: false });

/** Bevy's point-light cube faces (`CUBE_MAP_FACES`): look direction and up per cube face. */
const CUBE_FACES = [
  [[1, 0, 0], [0, 1, 0]],
  [[-1, 0, 0], [0, 1, 0]],
  [[0, 1, 0], [0, 0, 1]],
  [[0, -1, 0], [0, 0, -1]],
  [[0, 0, -1], [0, 1, 0]],
  [[0, 0, 1], [0, 1, 0]],
];

/** Bevy's right-handed infinite reverse-Z perspective projection (column-major). */
function perspectiveInfiniteReverse(fovY, aspect, near) {
  const f = 1 / Math.tan(fovY / 2);
  return new Matrix4().set(f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, 0, near, 0, 0, -1, 0);
}

/** Bevy `maths::orthonormalize`: a right-handed basis whose z axis is `z`. */
function orthonormalize(z) {
  // `1.copysign(z.z)`: −0 counts as negative.
  const sign = z.z > 0 || Object.is(z.z, 0) ? 1 : -1;
  const a = -1 / (sign + z.z);
  const b = z.x * z.y * a;
  return [new Vector3(1 + sign * z.x * z.x * a, sign * b, -sign * z.x), new Vector3(b, sign + z.y * z.y * a, -z.y), z];
}

/** The rotation of Bevy `Transform::looking_to(direction, up)` as [right, up, back]. */
function lookingTo(direction, up) {
  const back = new Vector3(...direction).normalize().negate();
  const right = new Vector3(...up).cross(back).normalize();
  return [right, back.clone().cross(right), back];
}

/** World-from-view for a light at `position` with basis columns `[x, y, z]`. */
function worldFromView(position, [x, y, z]) {
  return new Matrix4().makeBasis(x, y, z).setPosition(...position);
}

/**
 * Bevy's `extract_lights` and `prepare_lights` for point and spot lights (render-engine
 * `LocalLights`): the packed uniform data, point lights then spot lights with the shadow casters
 * first, and one shadow view per spot map and point-light cube face.
 */
function packLocalLights(pointLights, spotLights, directionalSize, pointSize) {
  const data = [];
  const pointViews = [];
  const spotViews = [];
  const softShadows = [...pointLights, ...spotLights].some((l) => l.shadows?.soft);
  const base = (l) => {
    const k = l.intensity / (4 * Math.PI);
    return [
      new Vector4(),
      new Vector4(l.color[0] * k, l.color[1] * k, l.color[2] * k, 1 / (l.range * l.range)),
      new Vector4(...l.position, l.radius ?? 0),
      new Vector4(),
      new Vector4(),
    ];
  };
  const shadowed = (light, s, texelSize, radius, index) => {
    // √2 for the worst-case diagonal texel offset, as in Bevy.
    light[3].set(s.depthBias, s.normalBias * texelSize * Math.SQRT2, s.nearZ, s.soft ? radius : 0);
    light[4].y = index;
    light[4].z = 1;
  };
  const casterFirst = (lights) => [...lights.filter((l) => l.shadows), ...lights.filter((l) => !l.shadows)];

  for (const l of casterFirst(pointLights)) {
    const light = base(l);
    const nearZ = l.shadows?.nearZ ?? POINT_LIGHT_SHADOWS.nearZ;
    const projection = perspectiveInfiniteReverse(Math.PI / 2, 1, nearZ);
    const p = projection.elements;
    light[0].set(p[10], p[11], p[14], p[15]);
    if (l.shadows) {
      const cube = pointViews.length / 6;
      if (cube >= MAX_POINT_SHADOWS) throw new Error(`at most ${MAX_POINT_SHADOWS} shadowed point lights`);
      shadowed(light, l.shadows, 2 / pointSize, l.radius ?? 0, cube);
      for (const [direction, up] of CUBE_FACES) {
        const world = worldFromView(l.position, lookingTo(direction, up));
        pointViews.push({ world, clipFromWorld: projection.clone().multiply(world.clone().invert()), fovY: Math.PI / 2, near: nearZ, range: l.range });
      }
    }
    data.push(...light);
  }
  for (const l of casterFirst(spotLights)) {
    const light = base(l);
    const outer = Math.min(l.outerAngle, Math.PI / 2 - 1e-4);
    const inner = Math.min(l.innerAngle, outer);
    const direction = new Vector3(...l.direction).normalize();
    const cosOuter = Math.cos(outer);
    const spotScale = 1 / Math.max(Math.cos(inner) - cosOuter, 1e-4);
    light[0].set(direction.x, direction.z, spotScale, -cosOuter * spotScale);
    light[4].x = Math.tan(outer);
    light[4].w = direction.y < 0 || Object.is(direction.y, -0) ? 1 : 0;
    if (l.shadows) {
      const k = spotViews.length;
      if (k >= MAX_SPOT_SHADOWS) throw new Error(`at most ${MAX_SPOT_SHADOWS} shadowed spot lights`);
      // Spot maps follow the cascades (shader layers 4 and up), at the cascades' size.
      shadowed(light, l.shadows, (2 * Math.tan(outer)) / directionalSize, l.radius ?? 0, MAX_CASCADES + k);
      // Bevy spot_light_world_from_view / spot_light_clip_from_view.
      const world = worldFromView(l.position, orthonormalize(direction.clone().negate()));
      const projection = perspectiveInfiniteReverse(2 * outer, 1, l.shadows.nearZ);
      spotViews.push({ world, clipFromWorld: projection.clone().multiply(world.clone().invert()), fovY: 2 * outer, near: l.shadows.nearZ, range: l.range });
    }
    data.push(...light);
  }
  return {
    data,
    pointViews,
    spotViews,
    defines: {
      LIT_POINT_LIGHTS: pointLights.length,
      LIT_SPOT_LIGHTS: spotLights.length,
      LIT_POINT_SHADOWS: pointViews.length / 6,
      LIT_SPOT_SHADOWS: spotViews.length,
      ...(softShadows ? { LIT_SOFT_SHADOWS: '' } : {}),
    },
  };
}

const NO_LOCAL_LIGHTS = { LIT_POINT_LIGHTS: 0, LIT_SPOT_LIGHTS: 0, LIT_POINT_SHADOWS: 0, LIT_SPOT_SHADOWS: 0 };

const WRAP = { repeat: RepeatWrapping, clampToEdge: ClampToEdgeWrapping, mirrorRepeat: MirroredRepeatWrapping };

/** three's minification filter for render-engine's `min_filter` and `mipmap_filter`. */
function minFilterOf(min, mipmap, mipmapped) {
  if (!mipmapped) return min === 'nearest' ? NearestFilter : LinearFilter;
  if (min === 'nearest') return mipmap === 'nearest' ? NearestMipmapNearestFilter : NearestMipmapLinearFilter;
  return mipmap === 'nearest' ? LinearMipmapNearestFilter : LinearMipmapLinearFilter;
}

/** glTF's default sampler (render-engine `Sampler::default`): repeat, trilinear. */
export const DEFAULT_SAMPLER = Object.freeze({
  addressModeU: 'repeat',
  addressModeV: 'repeat',
  magFilter: 'linear',
  minFilter: 'linear',
  mipmapFilter: 'linear',
});

/**
 * A material texture as render-engine uploads one (`TextureData`): RGBA8 texels, level 0 first,
 * each level half the previous one, with its sampler. One level means no mipmaps; none are
 * generated. `srgb` for colour (base colour, emissive), linear for data (normal,
 * metallic-roughness, occlusion), as glTF prescribes. Rows run top-down from v = 0, as in glTF.
 */
export function textureFromData({ width, height, srgb, levels, sampler = DEFAULT_SAMPLER }) {
  const texture = new DataTexture(levels[0], width, height, RGBAFormat, UnsignedByteType);
  const mipmapped = levels.length > 1;
  texture.mipmaps = mipmapped
    ? levels.map((data, i) => ({ data, width: Math.max(1, width >> i), height: Math.max(1, height >> i) }))
    : [];
  Object.assign(texture, {
    wrapS: WRAP[sampler.addressModeU],
    wrapT: WRAP[sampler.addressModeV],
    magFilter: sampler.magFilter === 'nearest' ? NearestFilter : LinearFilter,
    minFilter: minFilterOf(sampler.minFilter, sampler.mipmapFilter, mipmapped),
    generateMipmaps: false,
    flipY: false,
    unpackAlignment: 1,
    colorSpace: srgb ? SRGBColorSpace : LinearSRGBColorSpace,
    needsUpdate: true,
  });
  return texture;
}

/** Bevy `AlphaMode`: 'opaque', 'mask' (with `alphaCutoff`) or 'blend'. */
const ALPHA_MODES = ['opaque', 'mask', 'blend'];

const TEXTURE_DEFINES = {
  baseColorTexture: 'LIT_BASE_COLOR_TEXTURE',
  emissiveTexture: 'LIT_EMISSIVE_TEXTURE',
  metallicRoughnessTexture: 'LIT_METALLIC_ROUGHNESS_TEXTURE',
  occlusionTexture: 'LIT_OCCLUSION_TEXTURE',
  normalMapTexture: 'LIT_NORMAL_MAP',
};

/**
 * Bevy `StandardMaterial` (the glTF core subset render-engine covers): linear base colour and
 * alpha, metallic, perceptual roughness, reflectance (0.5 = 4% F0), emissive radiance (cd/m²),
 * each times its texture (`TEXCOORD_0`, three's `uv`); the mesh's `color` attribute (glTF
 * `COLOR_0`) multiplies the base colour. Normal maps need the mesh's MikkTSpace `tangent`
 * attribute. Alpha modes sort and draw like Bevy's phases (opaque, then masked, then blended back
 * to front by object origin) and masked or blended materials cut their shadows out. On a
 * `SkinnedMesh` it skins on the GPU (Bevy's four-influence linear blend), shadows included.
 * The parameters are fixed at construction.
 */
export class LitMaterial extends ShaderMaterial {
  constructor({
    baseColor = [1, 1, 1],
    alpha = 1,
    metallic = 0,
    perceptualRoughness = 0.5,
    reflectance = 0.5,
    emissive = [0, 0, 0],
    baseColorTexture = null,
    emissiveTexture = null,
    metallicRoughnessTexture = null,
    normalMapTexture = null,
    occlusionTexture = null,
    alphaMode = 'opaque',
    alphaCutoff = 0.5,
    doubleSided = false,
    unlit = false,
  } = {}) {
    if (!ALPHA_MODES.includes(alphaMode)) throw new Error(`unknown alphaMode ${alphaMode}`);
    const textures = { baseColorTexture, emissiveTexture, metallicRoughnessTexture, normalMapTexture, occlusionTexture };
    const defines = { ...NO_LOCAL_LIGHTS };
    for (const [name, define] of Object.entries(TEXTURE_DEFINES)) {
      if (textures[name]) defines[define] = '';
    }
    if (alphaMode === 'mask') defines.LIT_ALPHA_MASK = '';
    if (alphaMode === 'blend') defines.LIT_ALPHA_BLEND = '';
    if (doubleSided) defines.LIT_DOUBLE_SIDED = '';
    if (unlit) defines.LIT_UNLIT = '';
    super({
      glslVersion: GLSL3,
      vertexShader: litVertex,
      fragmentShader: litFragment,
      uniforms: {
        baseColor: { value: new Vector4(baseColor[0], baseColor[1], baseColor[2], alpha) },
        emissive: { value: new Vector3(...emissive) },
        metallic: { value: metallic },
        perceptualRoughness: { value: perceptualRoughness },
        reflectance: { value: reflectance },
        // Masked: the material's cutoff. Blended: the shadow pass's (Bevy's 0.05).
        alphaCutoff: { value: alphaMode === 'blend' ? 0.05 : alphaMode === 'mask' ? alphaCutoff : 0 },
        ...Object.fromEntries(Object.entries(textures).map(([name, t]) => [name, { value: t }])),
        ...frameUniforms(),
      },
      defines,
      side: doubleSided ? DoubleSide : FrontSide,
      // Blended: straight alpha over what is behind, depth tested but not written, one pass
      // for both faces (Bevy's Transparent3d).
      transparent: alphaMode === 'blend',
      depthWrite: alphaMode !== 'blend',
      forceSinglePass: true,
    });
    // Attributes the mesh does not have: no tangents, white vertex colours.
    this.defaultAttributeValues = { ...this.defaultAttributeValues, tangent: [0, 0, 0, 0], color: [1, 1, 1, 1] };
    this.isLitMaterial = true;
    this.toneMapped = false;
    this.alphaMode = alphaMode;
    this.doubleSided = doubleSided;
    this.unlit = unlit;
  }

  /**
   * The LitMaterial for a material three's GLTFLoader made: `MeshStandardMaterial` (or
   * `MeshPhysicalMaterial`, whose extensions are dropped) or, for `KHR_materials_unlit`,
   * `MeshBasicMaterial`. Maps glTF's factors and textures the way Bevy's glTF loader does.
   */
  static fromGltf(material) {
    const color = material.color ? [material.color.r, material.color.g, material.color.b] : [1, 1, 1];
    const alphaMode = material.transparent ? 'blend' : material.alphaTest > 0 ? 'mask' : 'opaque';
    const common = {
      baseColor: color,
      alpha: material.opacity ?? 1,
      baseColorTexture: material.map ?? null,
      alphaMode,
      alphaCutoff: alphaMode === 'mask' ? material.alphaTest : 0.5,
      doubleSided: material.side === DoubleSide,
    };
    if (material.isMeshBasicMaterial) return new LitMaterial({ ...common, unlit: true });
    const e = material.emissive;
    const strength = material.emissiveIntensity ?? 1;
    return new LitMaterial({
      ...common,
      metallic: material.metalness ?? 0,
      perceptualRoughness: material.roughness ?? 1,
      metallicRoughnessTexture: material.roughnessMap ?? material.metalnessMap ?? null,
      normalMapTexture: material.normalMap ?? null,
      occlusionTexture: material.aoMap ?? null,
      emissive: e ? [e.r * strength, e.g * strength, e.b * strength] : [0, 0, 0],
      emissiveTexture: material.emissiveMap ?? null,
    });
  }
}

/** Opaque, then alpha-masked draws (Bevy's Opaque3d and AlphaMask3d), each three's way. */
function opaqueSort(a, b) {
  const phase = (item) => (item.material.alphaMode === 'mask' ? 1 : 0);
  return (
    phase(a) - phase(b) ||
    a.groupOrder - b.groupOrder ||
    a.renderOrder - b.renderOrder ||
    a.material.id - b.material.id ||
    (a.materialVariant ?? 0) - (b.materialVariant ?? 0) ||
    a.z - b.z ||
    a.id - b.id
  );
}

/** Blended draws from back to front by object origin in view space (Bevy's Transparent3d). */
function transparentSort(camera) {
  const v = camera.matrixWorldInverse.elements;
  const viewZ = (item) => {
    const m = item.object.matrixWorld.elements;
    return v[2] * m[12] + v[6] * m[13] + v[10] * m[14] + v[14];
  };
  return (a, b) =>
    a.groupOrder - b.groupOrder || a.renderOrder - b.renderOrder || viewZ(a) - viewZ(b) || a.id - b.id;
}

/** Turns a define on or off, recompiling when it changes. */
function setDefine(material, name, on) {
  if (on === Object.hasOwn(material.defines, name)) return;
  if (on) material.defines[name] = '';
  else delete material.defines[name];
  material.needsUpdate = true;
}

/** A reverse-Z comparison depth texture with Bevy's linear shadow sampler. */
function comparisonDepth(depthTexture) {
  depthTexture.compareFunction = GreaterEqualCompare;
  depthTexture.minFilter = LinearFilter;
  depthTexture.magFilter = LinearFilter;
  return depthTexture;
}

function shadowTarget(size) {
  const depthTexture = comparisonDepth(new DepthTexture(size, size, FloatType));
  return new WebGLRenderTarget(size, size, { depthBuffer: true, depthTexture, type: HalfFloatType });
}

// A spot or point-light map: a comparison depth texture plus the depth in an R32F colour target
// (linear, for the soft-shadow blocker search).
const LOCAL_COLOR = {
  type: FloatType,
  format: RedFormat,
  minFilter: LinearFilter,
  magFilter: LinearFilter,
  generateMipmaps: false,
  colorSpace: LinearSRGBColorSpace,
};

function spotShadowTarget(size) {
  const depthTexture = comparisonDepth(new DepthTexture(size, size, FloatType));
  return new WebGLRenderTarget(size, size, { ...LOCAL_COLOR, depthBuffer: true, depthTexture });
}

function pointShadowTarget(size) {
  const target = new WebGLCubeRenderTarget(size, { ...LOCAL_COLOR, depthBuffer: true });
  // three.js allocates cube depth as DEPTH_COMPONENT24 (Bevy: Depth32Float).
  target.depthTexture = comparisonDepth(new CubeDepthTexture(size, UnsignedIntType));
  return target;
}

/**
 * The frame's lights, sky colour, exposure and tone mapping, and the passes that render a scene
 * with them. Fields mirror render-engine's `Scene` and `Camera` (Bevy units).
 */
export class Lighting {
  /**
   * @param {{ tonyMcMapface: Data3DTexture, agx: Data3DTexture }} luts from loadToneMappingLuts()
   */
  constructor(luts, { shadowMapSize = 2048, pointShadowMapSize = 1024 } = {}) {
    this.luts = luts;
    /** Linear RGB written where nothing is drawn; tone mapped, not exposed (Bevy ClearColor). */
    this.clearColor = [0, 0, 0];
    /**
     * Alpha written where nothing is drawn: 1 is opaque; below 1 the page shows through a canvas
     * created with `alpha: true` (premultiplied, three's default).
     */
    this.clearAlpha = 1;
    this.ev100 = 9.7;
    /** 'tonyMcMapface' (Bevy's default), 'agx' or 'acesFitted'. */
    this.tonemapping = 'tonyMcMapface';
    /** 'gaussian' (Bevy's default) or 'hardware2x2'. */
    this.shadowFilter = 'gaussian';
    /**
     * `{ direction, color, illuminance, shadows }`: direction the light travels, linear colour,
     * lux, and `DEFAULT_SUN_SHADOWS`-shaped cascades or null.
     */
    this.sun = null;
    /** `{ color, brightness }` (cd/m²). */
    this.ambient = { color: [1, 1, 1], brightness: 80 };
    /** `{ diffuse, specular, intensity }`: cube maps from loadCubemap, cd/m² scale. */
    this.environment = null;
    /** `{ color: rgba, directionalLightColor: rgba, directionalLightExponent, falloff }`. */
    this.fog = null;
    /**
     * `{ position, color, intensity, range, radius, shadows }`: lumens, metres, and
     * `POINT_LIGHT_SHADOWS`-shaped cube-map shadows or null.
     */
    this.pointLights = [];
    /**
     * `{ position, direction, color, intensity, range, radius, innerAngle, outerAngle, shadows }`:
     * the direction the light points, lumens over the full sphere like a point light's, radians
     * off the axis, and `SPOT_LIGHT_SHADOWS`-shaped shadows or null.
     */
    this.spotLights = [];
    /**
     * `{ direction, color, illuminance }`: an unshadowed directional light, the moon (lux; its
     * disc shows under a sky).
     */
    this.moon = null;
    /**
     * `DEFAULT_SKY`-shaped: Bevy's physically based atmosphere (transmittance, multiscattering,
     * sky-view and aerial-view LUTs every frame) with the sun and moon discs, the sky drawn where
     * nothing else is and aerial perspective over the scene. With `environmentIntensity` it is
     * also the ambient light: rendered into a cube and filtered like Bevy's
     * `GeneratedEnvironmentMapLight` every frame (then `environment` must be null). Under a sky
     * the sun's illuminance is the light above the atmosphere (`RAW_SUNLIGHT`). Needs
     * `EXT_color_buffer_float` and `OES_texture_float_linear`.
     */
    this.sky = null;

    this.uniforms = frameUniforms();
    this.shadowMapSize = shadowMapSize;
    this.pointShadowMapSize = pointShadowMapSize;
    this.spotShadowTargets = [];
    this.pointShadowTargets = [];
    this.localShadowCamera = new PerspectiveCamera();
    this.localShadowMaterial = new ShaderMaterial({
      glslVersion: GLSL3,
      vertexShader: localShadowVertex,
      fragmentShader: localShadowFragment,
      uniforms: { lightClip: { value: new Matrix4() }, flipY: { value: 1 } },
      depthFunc: GreaterEqualDepth,
      side: FrontSide,
    });
    // Per LitMaterial that cuts its shadow out or draws both faces: { cascade, local }.
    this.shadowVariants = new WeakMap();
    // Objects without a LitMaterial skip the shadow passes.
    this.noShadow = new ShaderMaterial({ visible: false });
    // Per shadow view, an InstancedMesh draws only its instances inside (culling.js).
    this.shadowCulling = new ShadowCulling(this.noShadow);
    this.shadowTargets = Array.from({ length: MAX_CASCADES }, () => shadowTarget(shadowMapSize));
    this.shadowCamera = new OrthographicCamera();
    this.shadowMaterial = new ShaderMaterial({
      glslVersion: GLSL3,
      vertexShader: shadowVertex,
      fragmentShader: shadowFragment,
      uniforms: { cascadeClip: { value: new Matrix4() } },
      depthFunc: GreaterEqualDepth,
      colorWrite: false,
      side: FrontSide,
    });
    this.tonemapQuad = new FullScreenQuad(
      new ShaderMaterial({
        glslVersion: GLSL3,
        vertexShader: fullscreenVertex,
        fragmentShader: tonemapFragment,
        uniforms: {
          hdrTexture: { value: null },
          lutTexture: { value: null },
          method: { value: 1 },
          // three's ACESFilmicToneMapping divides by 0.6; Bevy's fit does not.
          toneMappingExposure: { value: 0.6 },
        },
        depthTest: false,
        depthWrite: false,
        blending: NoBlending,
      }),
    );
    this.outputQuad = new FullScreenQuad(
      new ShaderMaterial({
        glslVersion: GLSL3,
        vertexShader: fullscreenVertex,
        fragmentShader: outputFragment,
        uniforms: { postTexture: { value: null }, encodeSrgb: { value: 1 } },
        depthTest: false,
        depthWrite: false,
        blending: NoBlending,
      }),
    );
    this.size = [0, 0];
  }

  #targets(width, height) {
    if (this.size[0] === width && this.size[1] === height) return;
    this.hdrTarget?.dispose();
    this.postTarget?.dispose();
    const options = {
      type: HalfFloatType,
      format: RGBAFormat,
      minFilter: NearestFilter,
      magFilter: NearestFilter,
      colorSpace: LinearSRGBColorSpace,
    };
    // The depth resolves into a texture after each draw, for the sky's aerial perspective.
    this.hdrTarget = new WebGLRenderTarget(width, height, {
      ...options,
      samples: 4,
      depthBuffer: true,
      depthTexture: new DepthTexture(width, height, UnsignedIntType),
    });
    this.postTarget = new WebGLRenderTarget(width, height, { ...options, depthBuffer: false });
    this.size = [width, height];
  }

  /** Cascades for this frame, fitted to `camera` (a PerspectiveCamera with updated matrices). */
  cascades(camera, aspect) {
    const shadows = this.sun?.shadows;
    if (!shadows) return [];
    const fovY = (camera.fov * Math.PI) / 180;
    return cascadeRanges(shadows).map(([near, far]) => ({
      far,
      ...fitCascade(near, far, this.shadowMapSize, this.sun.direction, fovY, aspect, camera.matrixWorld.elements),
    }));
  }

  #updateUniforms(camera, cascades, local) {
    const u = this.uniforms;
    u.localLights.value = local.data;
    for (let i = 0; i < MAX_POINT_SHADOWS; i++) {
      const t = this.pointShadowTargets[i];
      u[`pointShadowMap${i}`].value = t?.depthTexture ?? null;
      u[`pointShadowDepth${i}`].value = t?.texture ?? null;
    }
    for (let i = 0; i < MAX_SPOT_SHADOWS; i++) {
      const t = this.spotShadowTargets[i];
      u[`spotShadowMap${i}`].value = t?.depthTexture ?? null;
      u[`spotShadowDepth${i}`].value = t?.texture ?? null;
    }
    u.exposure.value = exposureFromEv100(this.ev100);
    const sun = this.sun;
    u.hasSun.value = sun ? 1 : 0;
    if (sun) {
      const d = new Vector3(...sun.direction).normalize().negate();
      u.directionToLight.value.copy(d);
      u.sunColor.value.set(...sun.color).multiplyScalar(sun.illuminance);
    }
    u.ambientColor.value.set(...this.ambient.color).multiplyScalar(this.ambient.brightness);
    u.sunShadows.value = cascades.length ? 1 : 0;
    u.gaussianFilter.value = this.shadowFilter === 'gaussian' ? 1 : 0;
    u.shadowMapSize.value = this.shadowMapSize;
    if (cascades.length) {
      const s = sun.shadows;
      u.shadowParams.value.set(s.depthBias, s.normalBias * Math.SQRT2, s.overlapProportion, cascades.length);
    }
    cascades.forEach((c, i) => {
      u.cascadeClipFromWorld.value[i].fromArray(c.clipFromWorld);
      u.cascadeParams.value[i].set(c.far, c.texelSize, 0, 0);
    });
    for (let i = 0; i < MAX_CASCADES; i++) {
      u[`shadowMap${i}`].value = this.shadowTargets[i].depthTexture;
    }
    const moon = this.moon;
    u.moonDirectionToLight.value.set(0, 0, 0, 0);
    u.moonColor.value.set(0, 0, 0);
    if (moon) {
      const d = new Vector3(...moon.direction).normalize().negate();
      u.moonDirectionToLight.value.set(d.x, d.y, d.z, 1);
      u.moonColor.value.set(...moon.color).multiplyScalar(moon.illuminance);
    }
    const sky = this.sky;
    u.atmosphere.value.z = sky ? 1 : 0;
    u.atmosphere.value.w = sun?.shadows ? 1 : 0;
    u.disks.value.set(sky?.sunDisk.angularSize ?? 0, sky?.moonDisk.angularSize ?? 0);
    u.atmosphereTransmittance.value = sky ? this.skyPasses.transmittance.texture : null;
    const skyEnvironment = sky?.environmentIntensity ?? null;
    if (this.environment && skyEnvironment !== null) {
      throw new Error('a sky with environment light replaces Lighting.environment; set one of them');
    }
    const env =
      this.environment ??
      (skyEnvironment === null
        ? null
        : {
            intensity: skyEnvironment,
            diffuse: this.skyPasses.diffuse.texture,
            specular: this.skyPasses.specular.texture,
            mipLevels: ENVIRONMENT_MIPS,
          });
    const mips = env ? (env.mipLevels ?? env.specular.userData.mipLevels) : 1;
    u.environment.value.set(env ? env.intensity : 0, mips - 1, env ? 1 : 0);
    u.diffuseEnvironmentMap.value = env?.diffuse ?? null;
    u.specularEnvironmentMap.value = env?.specular ?? null;
    const fog = this.fog;
    if (fog) {
      const fl = fog.falloff;
      const be = {
        linear: () => [fl.start, fl.end, 0],
        exponential: () => [fl.density, 0, 0],
        exponentialSquared: () => [fl.density, 0, 0],
        atmospheric: () => fl.extinction,
      }[fl.type]();
      const bi = {
        linear: [0, 0, 0, 1],
        exponential: [0, 0, 0, 2],
        exponentialSquared: [0, 0, 0, 3],
        atmospheric: [...(fl.inscattering ?? [0, 0, 0]), 4],
      }[fl.type];
      u.fogColor.value.set(...fog.color);
      u.fogDirectionalLightColor.value.set(...(fog.directionalLightColor ?? [0, 0, 0, 0]));
      u.fogBe.value.set(be[0], be[1], be[2], fog.directionalLightExponent ?? 8);
      u.fogBi.value.set(...bi);
    } else {
      u.fogBi.value.set(0, 0, 0, 0);
    }
  }

  #renderShadows(renderer, scene, cascades, local) {
    const cam = this.shadowCamera;
    const state = renderer.state;
    if (!this.shadowTargetsReady) {
      // Unused cascades stay bound to the shader: give every depth texture its comparison
      // sampler state (set when three.js first sets up the target) and a cleared map.
      state.buffers.depth.setClear(0);
      for (const target of this.shadowTargets) {
        renderer.setRenderTarget(target);
        renderer.clear(false, true, false);
      }
      this.shadowTargetsReady = true;
    }
    // Bevy's shadow pipeline where the browser allows it: zero-to-one clip space and unclipped
    // depth clamped by the rasterizer (EXT_clip_control + EXT_depth_clamp).
    const ext = (name) => (renderer.extensions.has(name) ? renderer.extensions.get(name) : null);
    const clipControl = ext('EXT_clip_control');
    const depthClamp = ext('EXT_depth_clamp');
    const native = Boolean(clipControl && depthClamp);
    if (native !== Boolean(this.shadowMaterial.defines.NATIVE_DEPTH)) {
      if (native) this.shadowMaterial.defines.NATIVE_DEPTH = '';
      else delete this.shadowMaterial.defines.NATIVE_DEPTH;
      this.shadowMaterial.needsUpdate = true;
    }
    // Spot and point views clip depth like Bevy's, so they need only the zero-to-one clip space.
    const localNative = Boolean(clipControl);
    if (localNative !== Boolean(this.localShadowMaterial.defines.NATIVE_DEPTH)) {
      if (localNative) this.localShadowMaterial.defines.NATIVE_DEPTH = '';
      else delete this.localShadowMaterial.defines.NATIVE_DEPTH;
      this.localShadowMaterial.needsUpdate = true;
    }
    const gl = renderer.getContext();
    if (clipControl) clipControl.clipControlEXT(clipControl.LOWER_LEFT_EXT, clipControl.ZERO_TO_ONE_EXT);
    if (native) gl.enable(depthClamp.DEPTH_CLAMP_EXT);
    // Every caster draws with a shadow material: the shared one, or its LitMaterial's variant
    // when that cuts out by alpha or draws both faces. three culls a plain mesh per view by its
    // sphere; culling.js culls the instances of each visible InstancedMesh.
    const casters = [];
    scene.traverse((o) => {
      if (o.material) casters.push([o, o.material]);
    });
    // Only drawn meshes: a stand-in would draw a mesh whose parent is hidden.
    const instanced = [];
    scene.traverseVisible((o) => {
      if (o.isInstancedMesh && o.count > 0) instanced.push(o);
    });
    const culling = this.shadowCulling;
    culling.begin(scene, instanced);
    const assign = (local) => {
      const pick = (m) => this.#shadowMaterial(m, local, native, localNative);
      for (const [o, m] of casters) o.material = Array.isArray(m) ? m.map(pick) : pick(m);
    };
    assign(false);
    cascades.forEach((c, i) => {
      // Culling camera: the cascade's x/y extent, unbounded towards the sun (pancaked casters).
      const half = c.diameter / 2;
      cam.matrixWorldInverse.fromArray(c.cascadeFromWorld);
      cam.matrixWorld.copy(cam.matrixWorldInverse).invert();
      cam.projectionMatrix.makeOrthographic(-half, half, half, -half, -1e7, c.depthRange, cam.coordinateSystem, cam.reversedDepth);
      cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
      cam.matrixAutoUpdate = false;
      cam.matrixWorldAutoUpdate = false;
      culling.view(i, cam);
      this.shadowMaterial.uniforms.cascadeClip.value.fromArray(c.clipFromWorld);
      renderer.setRenderTarget(this.shadowTargets[i]);
      state.buffers.depth.setClear(0);
      renderer.clear(false, true, false);
      renderer.render(scene, cam);
    });
    culling.reset();
    if (native) gl.disable(depthClamp.DEPTH_CLAMP_EXT);
    if (local.spotViews.length || local.pointViews.length) {
      this.#renderLocalShadows(renderer, scene, local, () => assign(true));
    }
    culling.end();
    for (const [o, m] of casters) o.material = m;
    if (clipControl) {
      const reversed = state.buffers.depth.getReversed();
      clipControl.clipControlEXT(
        clipControl.LOWER_LEFT_EXT,
        reversed ? clipControl.ZERO_TO_ONE_EXT : clipControl.NEGATIVE_ONE_TO_ONE_EXT,
      );
    }
    state.buffers.depth.setClear(1);
  }

  /**
   * The shadow material `material`'s casters draw with: a LitMaterial's own variant when it is
   * masked, blended or double-sided (sharing its uniforms), otherwise the shared one.
   */
  #shadowMaterial(material, local, native, localNative) {
    // Objects drawn with any other material (sprites, overlays, a backdrop) cast no shadow.
    if (!material.isLitMaterial) return this.noShadow;
    const shared = local ? this.localShadowMaterial : this.shadowMaterial;
    if (material.alphaMode === 'opaque' && !material.doubleSided) return shared;
    let variants = this.shadowVariants.get(material);
    if (!variants) {
      const discard = material.alphaMode !== 'opaque';
      const variant = (base) => {
        const m = base.clone();
        // Clones copy uniforms; share the pass's and the material's instead.
        m.uniforms = {
          ...base.uniforms,
          baseColor: material.uniforms.baseColor,
          alphaCutoff: material.uniforms.alphaCutoff,
          baseColorTexture: material.uniforms.baseColorTexture,
        };
        m.defines = { ...base.defines };
        if (discard) m.defines.LIT_ALPHA_DISCARD = '';
        if (discard && material.uniforms.baseColorTexture.value) m.defines.LIT_BASE_COLOR_TEXTURE = '';
        return m;
      };
      variants = { cascade: variant(this.shadowMaterial), local: variant(this.localShadowMaterial) };
      this.shadowVariants.set(material, variants);
      material.addEventListener('dispose', () => {
        variants.cascade.dispose();
        variants.local.dispose();
        this.shadowVariants.delete(material);
      });
    }
    const m = local ? variants.local : variants.cascade;
    setDefine(m, 'NATIVE_DEPTH', local ? localNative : native);
    m.side = material.doubleSided ? DoubleSide : shared.side;
    return m;
  }

  /**
   * Spot maps, then point-light cube faces (render-engine's pass order). `assign` gives the
   * casters their local shadow materials, whose faces follow the shared material's side.
   */
  #renderLocalShadows(renderer, scene, local, assign) {
    const state = renderer.state;
    const material = this.localShadowMaterial;
    const cam = this.localShadowCamera;
    cam.matrixAutoUpdate = false;
    cam.matrixWorldAutoUpdate = false;
    renderer.setClearColor(new Color(0, 0, 0), 0);
    state.buffers.color.setMask(true);
    let index = MAX_CASCADES;
    const pass = (view, target, face) => {
      // Culling camera: the light's frustum out to its range (Bevy also culls casters beyond it).
      cam.fov = (view.fovY * 180) / Math.PI;
      cam.aspect = 1;
      cam.near = view.near;
      cam.far = view.range;
      cam.updateProjectionMatrix();
      cam.matrixWorld.copy(view.world);
      cam.matrixWorldInverse.copy(view.world).invert();
      this.shadowCulling.view(index++, cam);
      material.uniforms.lightClip.value.copy(view.clipFromWorld);
      renderer.setRenderTarget(target, face);
      state.buffers.depth.setClear(0);
      renderer.clear(true, true, false);
      renderer.render(scene, cam);
    };
    material.uniforms.flipY.value = 1;
    material.side = FrontSide;
    assign();
    local.spotViews.forEach((view, i) => pass(view, this.spotShadowTargets[i]));
    // Cube faces render upside down (see localShadowVertex), which flips the winding.
    material.uniforms.flipY.value = -1;
    material.side = BackSide;
    assign();
    local.pointViews.forEach((view, i) => pass(view, this.pointShadowTargets[Math.floor(i / 6)], i % 6));
  }

  /** The sky's LUTs and environment cubes, created with the first sky. */
  get skyPasses() {
    this.#sky ??= new SkyPasses();
    return this.#sky;
  }

  #sky = null;

  /**
   * The sky's directional lights in Bevy's order (`prepare_lights` sorts unshadowed lights
   * first, then by spawn order, sun before moon), each with its disc.
   */
  #skyLights() {
    const light = (l, disk) => ({
      directionToLight: new Vector3(...l.direction).normalize().negate(),
      color: l.color.map((c) => c * l.illuminance),
      diskSize: disk.angularSize,
      diskIntensity: disk.intensity,
    });
    const sun = this.sun ? light(this.sun, this.sky.sunDisk) : null;
    const moon = this.moon ? light(this.moon, this.sky.moonDisk) : null;
    const ordered = this.sun?.shadows ? [moon, sun] : [sun, moon];
    return ordered.filter(Boolean);
  }

  #renderSky(renderer, camera, aspect) {
    this.skyPasses.render(renderer, {
      groundAlbedo: this.sky.groundAlbedo,
      lights: this.#skyLights(),
      worldFromView: camera.matrixWorld,
      clipFromView: perspectiveInfiniteReverse((camera.fov * Math.PI) / 180, aspect, camera.near),
      exposure: exposureFromEv100(this.ev100),
      environment: (this.sky.environmentIntensity ?? null) !== null,
    });
  }

  /**
   * Renders `scene` from `camera` into `target` (null: the canvas), sized `width × height`.
   */
  render(renderer, scene, camera, target = null, width = target?.width, height = target?.height) {
    if (width === undefined) {
      const size = renderer.getDrawingBufferSize(new Vector3());
      [width, height] = [size.x, size.y];
    }
    this.#targets(width, height);
    const autoClear = renderer.autoClear;
    const clearColor = renderer.getClearColor(new Color());
    const clearAlpha = renderer.getClearAlpha();
    renderer.autoClear = false;

    scene.updateMatrixWorld();
    camera.updateMatrixWorld();
    // The world matrices hold for every pass of this frame: three need not update them per pass.
    const matrixWorldAutoUpdate = scene.matrixWorldAutoUpdate;
    scene.matrixWorldAutoUpdate = false;
    try {
      this.#renderPasses(renderer, scene, camera, target, width, height);
    } finally {
      scene.matrixWorldAutoUpdate = matrixWorldAutoUpdate;
      renderer.setClearColor(clearColor, clearAlpha);
      renderer.autoClear = autoClear;
    }
  }

  #renderPasses(renderer, scene, camera, target, width, height) {
    const cascades = this.cascades(camera, width / height);
    const local = packLocalLights(this.pointLights, this.spotLights, this.shadowMapSize, this.pointShadowMapSize);
    while (this.spotShadowTargets.length < local.spotViews.length) {
      this.spotShadowTargets.push(spotShadowTarget(this.shadowMapSize));
    }
    while (this.pointShadowTargets.length < local.pointViews.length / 6) {
      this.pointShadowTargets.push(pointShadowTarget(this.pointShadowMapSize));
    }
    if (this.sky) this.#renderSky(renderer, camera, width / height);
    this.#updateUniforms(camera, cascades, local);
    const definesKey = JSON.stringify(local.defines);
    const sceneMaterials = new Set();
    scene.traverse((o) => {
      const materials = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of materials) {
        if (m) sceneMaterials.add(m);
        if (!m?.isLitMaterial) continue;
        // The frame uniforms are shared objects whose values #updateUniforms sets: a material
        // takes them once.
        if (m.uniforms.exposure !== this.uniforms.exposure) Object.assign(m.uniforms, this.uniforms);
        // Light counts size the shader's loops and samplers: a change recompiles, as in three.
        if (m.userData.localLights !== definesKey) {
          delete m.defines.LIT_SOFT_SHADOWS;
          Object.assign(m.defines, local.defines);
          m.userData.localLights = definesKey;
          m.needsUpdate = true;
        }
      }
    });

    this.#renderShadows(renderer, scene, cascades, local);

    renderer.setRenderTarget(this.hdrTarget);
    // Premultiplied, so a transparent clear leaves nothing behind for the page.
    const alpha = Math.min(1, Math.max(0, this.clearAlpha));
    renderer.setClearColor(new Color(...this.clearColor).multiplyScalar(alpha), alpha);
    // The shadow material left colour writes off.
    renderer.state.buffers.color.setMask(true);
    renderer.clear(true, true, false);
    renderer.setOpaqueSort(opaqueSort);
    renderer.setTransparentSort(transparentSort(camera));
    // WebGL stores NDC y = -1 in row 0 where wgpu stores +1, which mirrors the frame and its 4×
    // MSAA sample pattern relative to Bevy's. So the main pass renders upside down (projection y
    // negated, winding inverted so culling and gl_FrontFacing hold) and stores the HDR frame top
    // row first, like wgpu; the output pass flips it back. three derives the winding from each
    // object's world matrix alone, so the inversion wraps state.setMaterial for this pass.
    const projection = camera.projectionMatrix;
    const upright = projection.clone();
    const state = renderer.state;
    const setMaterial = state.setMaterial;
    const drawScene = () => {
      try {
        projection.premultiply(FLIP_Y);
        state.setMaterial = (material, frontFaceCW) => setMaterial(material, !frontFaceCW);
        renderer.render(scene, camera);
      } finally {
        state.setMaterial = setMaterial;
        projection.copy(upright);
      }
    };
    if (this.sky) {
      // Like Bevy's main pass with a sky: the opaque and alpha-masked draws, the sky over them
      // (reading their depth), then the blended draws.
      const visible = new Map([...sceneMaterials].map((m) => [m, m.visible]));
      try {
        for (const [m, v] of visible) m.visible = v && !m.transparent;
        drawScene();
        this.skyPasses.composite(renderer, this.hdrTarget, this.hdrTarget.depthTexture, camera.projectionMatrixInverse);
        for (const [m, v] of visible) m.visible = v && m.transparent;
        drawScene();
      } finally {
        for (const [m, v] of visible) m.visible = v;
      }
    } else {
      drawScene();
    }
    renderer.setOpaqueSort(null);
    renderer.setTransparentSort(null);

    const tm = this.tonemapQuad.material.uniforms;
    tm.hdrTexture.value = this.hdrTarget.texture;
    tm.method.value = TONEMAPPING[this.tonemapping];
    // Like Bevy, operators without a LUT still bind one (the small AgX LUT).
    tm.lutTexture.value = this.tonemapping === 'tonyMcMapface' ? this.luts.tonyMcMapface : this.luts.agx;
    renderer.setRenderTarget(this.postTarget);
    this.tonemapQuad.render(renderer);

    this.outputQuad.material.uniforms.postTexture.value = this.postTarget.texture;
    // An sRGB target (SRGB8_ALPHA8) encodes on store, like Bevy's output; otherwise the shader does.
    this.outputQuad.material.uniforms.encodeSrgb.value = target?.texture.colorSpace === SRGBColorSpace ? 0 : 1;
    renderer.setRenderTarget(target);
    this.outputQuad.render(renderer);
  }

  dispose() {
    for (const t of [...this.shadowTargets, ...this.spotShadowTargets, ...this.pointShadowTargets]) t.dispose();
    this.hdrTarget?.dispose();
    this.postTarget?.dispose();
    this.shadowMaterial.dispose();
    this.localShadowMaterial.dispose();
    this.noShadow.dispose();
    this.tonemapQuad.material.dispose();
    this.outputQuad.material.dispose();
    this.#sky?.dispose();
  }
}
