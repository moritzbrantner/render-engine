// render-engine's lighting on three.js (WebGLRenderer, GLSL ES 3.0): Bevy's PBR BRDF, sun
// cascades with Bevy's fitting and filters, ambient and image-based light from prefiltered cube
// maps, distance fog and Bevy's tone mappers. Scene values use Bevy's physical units (lux, cd/m²,
// EV100, linear RGB), so a frame matches render-engine's and Bevy's (the oracle's `three` side).
//
// The consumer owns the WebGLRenderer, the Scene and the camera. Meshes use `LitMaterial`;
// `Lighting.render` draws the sun's shadow cascades, the 4× MSAA HDR main pass, tone mapping and
// the sRGB output.

import {
  BackSide,
  ClampToEdgeWrapping,
  Color,
  CubeDepthTexture,
  CubeTexture,
  Data3DTexture,
  DataTexture,
  DepthTexture,
  FloatType,
  FrontSide,
  GLSL3,
  GreaterEqualCompare,
  GreaterEqualDepth,
  HalfFloatType,
  LinearFilter,
  LinearMipmapLinearFilter,
  LinearSRGBColorSpace,
  Matrix4,
  NearestFilter,
  NoBlending,
  OrthographicCamera,
  PerspectiveCamera,
  RGBAFormat,
  RedFormat,
  RGBFormat,
  SRGBColorSpace,
  ShaderMaterial,
  UnsignedInt5999Type,
  UnsignedIntType,
  Vector3,
  Vector4,
  WebGLCubeRenderTarget,
  WebGLRenderTarget,
} from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { read as readKtx2 } from 'three/addons/libs/ktx-parse.module.js';
import { ZSTDDecoder } from 'three/addons/libs/zstddec.module.js';
import { cascadeRanges, fitCascade } from './cascades.js';
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
    viewportHeight: { value: 1 },
    localLights: { value: [] },
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

/**
 * Bevy `StandardMaterial` (untextured subset): linear base colour, metallic, perceptual
 * roughness, reflectance (0.5 = 4% F0) and emissive radiance (cd/m²).
 */
export class LitMaterial extends ShaderMaterial {
  constructor({
    baseColor = [1, 1, 1],
    alpha = 1,
    metallic = 0,
    perceptualRoughness = 0.5,
    reflectance = 0.5,
    emissive = [0, 0, 0],
  } = {}) {
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
        ...frameUniforms(),
      },
      defines: { ...NO_LOCAL_LIGHTS },
      side: FrontSide,
    });
    this.isLitMaterial = true;
    this.toneMapped = false;
  }
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
    this.hdrTarget = new WebGLRenderTarget(width, height, { ...options, samples: 4, depthBuffer: true });
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

  #updateUniforms(camera, cascades, local, height) {
    const u = this.uniforms;
    u.viewportHeight.value = height;
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
    const env = this.environment;
    u.environment.value.set(env ? env.intensity : 0, env ? env.specular.userData.mipLevels - 1 : 0, env ? 1 : 0);
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
    scene.overrideMaterial = this.shadowMaterial;
    cascades.forEach((c, i) => {
      // Culling camera: the cascade's x/y extent, unbounded towards the sun (pancaked casters).
      const half = c.diameter / 2;
      cam.matrixWorldInverse.fromArray(c.cascadeFromWorld);
      cam.matrixWorld.copy(cam.matrixWorldInverse).invert();
      cam.projectionMatrix.makeOrthographic(-half, half, half, -half, -1e7, c.depthRange);
      cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
      cam.matrixAutoUpdate = false;
      cam.matrixWorldAutoUpdate = false;
      this.shadowMaterial.uniforms.cascadeClip.value.fromArray(c.clipFromWorld);
      renderer.setRenderTarget(this.shadowTargets[i]);
      state.buffers.depth.setClear(0);
      renderer.clear(false, true, false);
      renderer.render(scene, cam);
    });
    if (native) gl.disable(depthClamp.DEPTH_CLAMP_EXT);
    this.#renderLocalShadows(renderer, scene, local);
    if (clipControl) {
      const reversed = state.buffers.depth.getReversed();
      clipControl.clipControlEXT(
        clipControl.LOWER_LEFT_EXT,
        reversed ? clipControl.ZERO_TO_ONE_EXT : clipControl.NEGATIVE_ONE_TO_ONE_EXT,
      );
    }
    state.buffers.depth.setClear(1);
    scene.overrideMaterial = null;
  }

  /** Spot maps, then point-light cube faces (render-engine's pass order). */
  #renderLocalShadows(renderer, scene, local) {
    if (!local.spotViews.length && !local.pointViews.length) return;
    const state = renderer.state;
    const material = this.localShadowMaterial;
    const cam = this.localShadowCamera;
    cam.matrixAutoUpdate = false;
    cam.matrixWorldAutoUpdate = false;
    scene.overrideMaterial = material;
    renderer.setClearColor(new Color(0, 0, 0), 0);
    state.buffers.color.setMask(true);
    const pass = (view, target, face) => {
      // Culling camera: the light's frustum out to its range (Bevy also culls casters beyond it).
      cam.fov = (view.fovY * 180) / Math.PI;
      cam.aspect = 1;
      cam.near = view.near;
      cam.far = view.range;
      cam.updateProjectionMatrix();
      cam.matrixWorld.copy(view.world);
      cam.matrixWorldInverse.copy(view.world).invert();
      material.uniforms.lightClip.value.copy(view.clipFromWorld);
      renderer.setRenderTarget(target, face);
      state.buffers.depth.setClear(0);
      renderer.clear(true, true, false);
      renderer.render(scene, cam);
    };
    material.uniforms.flipY.value = 1;
    material.side = FrontSide;
    local.spotViews.forEach((view, i) => pass(view, this.spotShadowTargets[i]));
    // Cube faces render upside down (see localShadowVertex), which flips the winding.
    material.uniforms.flipY.value = -1;
    material.side = BackSide;
    local.pointViews.forEach((view, i) => pass(view, this.pointShadowTargets[Math.floor(i / 6)], i % 6));
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
    const cascades = this.cascades(camera, width / height);
    const local = packLocalLights(this.pointLights, this.spotLights, this.shadowMapSize, this.pointShadowMapSize);
    while (this.spotShadowTargets.length < local.spotViews.length) {
      this.spotShadowTargets.push(spotShadowTarget(this.shadowMapSize));
    }
    while (this.pointShadowTargets.length < local.pointViews.length / 6) {
      this.pointShadowTargets.push(pointShadowTarget(this.pointShadowMapSize));
    }
    this.#updateUniforms(camera, cascades, local, height);
    const definesKey = JSON.stringify(local.defines);
    scene.traverse((o) => {
      const materials = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of materials) {
        if (!m?.isLitMaterial) continue;
        Object.assign(m.uniforms, this.uniforms);
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
    renderer.setClearColor(new Color(...this.clearColor), 1);
    // The shadow material left colour writes off.
    renderer.state.buffers.color.setMask(true);
    renderer.clear(true, true, false);
    renderer.render(scene, camera);

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

    renderer.setClearColor(clearColor, clearAlpha);
    renderer.autoClear = autoClear;
  }

  dispose() {
    for (const t of [...this.shadowTargets, ...this.spotShadowTargets, ...this.pointShadowTargets]) t.dispose();
    this.hdrTarget?.dispose();
    this.postTarget?.dispose();
    this.shadowMaterial.dispose();
    this.localShadowMaterial.dispose();
    this.tonemapQuad.material.dispose();
    this.outputQuad.material.dispose();
  }
}
