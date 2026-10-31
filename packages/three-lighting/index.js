// render-engine's lighting on three.js (WebGLRenderer, GLSL ES 3.0): Bevy's PBR BRDF, sun
// cascades with Bevy's fitting and filters, ambient and image-based light from prefiltered cube
// maps, distance fog and Bevy's tone mappers. Scene values use Bevy's physical units (lux, cd/m²,
// EV100, linear RGB), so a frame matches render-engine's and Bevy's (the oracle's `three` side).
//
// The consumer owns the WebGLRenderer, the Scene and the camera. Meshes use `LitMaterial`;
// `Lighting.render` draws the sun's shadow cascades, the 4× MSAA HDR main pass, tone mapping and
// the sRGB output.

import {
  ClampToEdgeWrapping,
  Color,
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
  RGBAFormat,
  RGBFormat,
  SRGBColorSpace,
  ShaderMaterial,
  UnsignedInt5999Type,
  Vector3,
  Vector4,
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
  MAX_CASCADES,
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
  };
}

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
      side: FrontSide,
    });
    this.isLitMaterial = true;
    this.toneMapped = false;
  }
}

function shadowTarget(size) {
  const depthTexture = new DepthTexture(size, size, FloatType);
  depthTexture.compareFunction = GreaterEqualCompare;
  depthTexture.minFilter = LinearFilter;
  depthTexture.magFilter = LinearFilter;
  return new WebGLRenderTarget(size, size, { depthBuffer: true, depthTexture, type: HalfFloatType });
}

/**
 * The frame's lights, sky colour, exposure and tone mapping, and the passes that render a scene
 * with them. Fields mirror render-engine's `Scene` and `Camera` (Bevy units).
 */
export class Lighting {
  /**
   * @param {{ tonyMcMapface: Data3DTexture, agx: Data3DTexture }} luts from loadToneMappingLuts()
   */
  constructor(luts, { shadowMapSize = 2048 } = {}) {
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

    this.uniforms = frameUniforms();
    this.shadowMapSize = shadowMapSize;
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

  #updateUniforms(camera, cascades) {
    const u = this.uniforms;
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

  #renderShadows(renderer, scene, cascades) {
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
    const gl = renderer.getContext();
    if (native) {
      clipControl.clipControlEXT(clipControl.LOWER_LEFT_EXT, clipControl.ZERO_TO_ONE_EXT);
      gl.enable(depthClamp.DEPTH_CLAMP_EXT);
    }
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
    if (native) {
      const reversed = state.buffers.depth.getReversed();
      clipControl.clipControlEXT(
        clipControl.LOWER_LEFT_EXT,
        reversed ? clipControl.ZERO_TO_ONE_EXT : clipControl.NEGATIVE_ONE_TO_ONE_EXT,
      );
      gl.disable(depthClamp.DEPTH_CLAMP_EXT);
    }
    state.buffers.depth.setClear(1);
    scene.overrideMaterial = null;
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
    this.#updateUniforms(camera, cascades);
    scene.traverse((o) => {
      const materials = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of materials) if (m?.isLitMaterial) Object.assign(m.uniforms, this.uniforms);
    });

    this.#renderShadows(renderer, scene, cascades);

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
    for (const t of this.shadowTargets) t.dispose();
    this.hdrTarget?.dispose();
    this.postTarget?.dispose();
    this.shadowMaterial.dispose();
    this.tonemapQuad.material.dispose();
    this.outputQuad.material.dispose();
  }
}
