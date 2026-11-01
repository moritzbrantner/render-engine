// The sky passes (render-engine `src/sky.rs`): Bevy 0.20's atmosphere (`bevy_pbr::atmosphere`,
// Hillaire's LUT method) and the runtime environment-map filter its
// `AtmosphereEnvironmentMapLight` drives (`GeneratedEnvironmentMapLight`), on WebGL 2.
//
// Each frame, before the shadow and main passes: the transmittance, multiscattering, sky-view and
// aerial-view LUTs, then, with sky ambient light, the sky rendered into a 128² cube, its mip
// chain, the GGX-prefiltered specular cube (one pass per mip) and the 32² irradiance
// cube. After the opaque draws `composite` blends the sky and the aerial perspective into the
// HDR frame; the main pass samples the transmittance LUT to filter the sun and moon light.
// Every pass is a fullscreen fragment pass (WebGL 2 has no compute), so the sky needs
// EXT_color_buffer_float (float targets) and OES_texture_float_linear (the Rgba32Float medium
// LUTs, as in Bevy).

import {
  AddEquation,
  BufferGeometry,
  ClampToEdgeWrapping,
  CustomBlending,
  DataTexture,
  Float32BufferAttribute,
  FloatType,
  GLSL3,
  HalfFloatType,
  LinearFilter,
  LinearMipmapLinearFilter,
  LinearSRGBColorSpace,
  Matrix4,
  Mesh,
  NearestFilter,
  NoBlending,
  OneFactor,
  OrthographicCamera,
  RGBAFormat,
  ShaderMaterial,
  SrcAlphaFactor,
  Vector2,
  Vector3,
  Vector4,
  WebGL3DRenderTarget,
  WebGLArrayRenderTarget,
  WebGLCubeRenderTarget,
  WebGLRenderTarget,
  ZeroFactor,
} from 'three';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import {
  aerialMarchFragment,
  aerialStoreFragment,
  copyFaceFragment,
  copyLevelFragment,
  downsampleFragment,
  environmentFragment,
  irradianceFragment,
  multiscatteringFragment,
  radianceFragment,
  renderSkyFragment,
  skyVertex,
  skyViewFragment,
  transmittanceFragment,
} from './sky-shaders.js';

/** Bevy `Atmosphere::earth`: the planet centre this far below the world origin. */
export const INNER_RADIUS = 6_360_000;
export const OUTER_RADIUS = 6_460_000;
const TRANSMITTANCE_SIZE = [256, 128];
const MULTISCATTERING_SIZE = [32, 32];
const SKY_VIEW_SIZE = [400, 200];
const AERIAL_VIEW_SIZE = 32;
/** Bevy `AtmosphereEnvironmentMapLight::default().size` and the diffuse cube's size. */
const ENVIRONMENT_SIZE = 128;
export const ENVIRONMENT_MIPS = Math.log2(ENVIRONMENT_SIZE) + 1;
const IRRADIANCE_SIZE = 32;
/** `ScatteringMedium::earth(256, 256)`. */
const FALLOFF_RESOLUTION = 256;
const PHASE_RESOLUTION = 256;

/**
 * Bevy's `ScatteringMedium::earth(256, 256)` LUTs (`GpuScatteringMedium::prepare_asset`):
 * absorption and scattering density per falloff parameter (two rows), and scattering × phase per
 * falloff and (non-linearly mapped) `neg_LdotV`.
 */
function earthMedium() {
  const FRAC_4_PI = 0.25 / Math.PI;
  const FRAC_3_16_PI = 0.1875 / Math.PI;
  const terms = [
    {
      absorption: [0, 0, 0],
      scattering: [5.802e-6, 13.558e-6, 33.1e-6],
      falloff: (p) => exponential(8 / 60, p),
      phase: (x) => FRAC_3_16_PI * (1 + x * x),
    },
    {
      absorption: [3.996e-6, 3.996e-6, 3.996e-6],
      scattering: [0.444e-6, 0.444e-6, 0.444e-6],
      falloff: (p) => exponential(1.2 / 60, p),
      phase: (x) => {
        const g = 0.8;
        const denom = 1 + g * g - 2 * g * x;
        return (FRAC_4_PI * (1 - g * g)) / (denom * Math.sqrt(denom));
      },
    },
    {
      absorption: [0.65e-6, 1.881e-6, 0.085e-6],
      scattering: [0, 0, 0],
      falloff: (p) => Math.max(1 - Math.abs(p - 0.75) / (0.5 * 0.3), 0),
      phase: () => FRAC_4_PI,
    },
  ];
  function exponential(scale, p) {
    const s = -1 / scale;
    const expS = Math.exp(s);
    return (Math.exp((1 - p) * s) - expS) / (1 - expS);
  }
  const n = FALLOFF_RESOLUTION;
  const density = new Float32Array(n * 2 * 4);
  for (let i = 0; i < n; i++) {
    const p = (i + 0.5) / n;
    for (const t of terms) {
      const f = t.falloff(p);
      for (let c = 0; c < 3; c++) {
        density[i * 4 + c] += t.absorption[c] * f;
        density[(n + i) * 4 + c] += t.scattering[c] * f;
      }
    }
  }
  const scattering = new Float32Array(n * PHASE_RESOLUTION * 4);
  for (let rawI = 0; rawI < n * PHASE_RESOLUTION; rawI++) {
    // Bevy indexes the falloff with `raw_i % phase_resolution`; both are 256 here.
    const i = rawI % PHASE_RESOLUTION;
    const j = Math.floor(rawI / PHASE_RESOLUTION);
    const f = (i + 0.5) / FALLOFF_RESOLUTION;
    const s = (2 * (j + 0.5)) / PHASE_RESOLUTION - 1;
    const negLdotV = Math.sign(s) * (1 - Math.pow(1 - Math.abs(s), 2));
    for (const t of terms) {
      const k = t.phase(negLdotV) * t.falloff(f);
      for (let c = 0; c < 3; c++) scattering[rawI * 4 + c] += t.scattering[c] * k;
    }
  }
  const texture = (data, height) => {
    const t = new DataTexture(data, n, height, RGBAFormat, FloatType);
    Object.assign(t, {
      minFilter: LinearFilter,
      magFilter: LinearFilter,
      wrapS: ClampToEdgeWrapping,
      wrapT: ClampToEdgeWrapping,
      generateMipmaps: false,
      flipY: false,
      colorSpace: LinearSRGBColorSpace,
      needsUpdate: true,
    });
    return t;
  };
  return { density: texture(density, 2), scattering: texture(scattering, PHASE_RESOLUTION) };
}

/** Bevy's `AtmosphereSampler` (bilinear, clamped) on an Rgba16Float LUT. */
const LUT = {
  type: HalfFloatType,
  format: RGBAFormat,
  minFilter: LinearFilter,
  magFilter: LinearFilter,
  generateMipmaps: false,
  depthBuffer: false,
  colorSpace: LinearSRGBColorSpace,
};

/** A target whose texture has every mip level allocated, rendered level by level. */
function mipmapped(target, levels) {
  target.texture.mipmaps = Array.from({ length: levels }, () => ({}));
  return target;
}

/** Bevy `extract_atmosphere`: world to atmosphere space, the planet centre at the origin. */
export function worldToAtmosphere() {
  return new Matrix4().makeTranslation(0, INNER_RADIUS, 0);
}

/**
 * The atmosphere LUTs and the sky's environment cubes of one `Lighting`. `render` updates them
 * for a frame; `composite` blends the sky into the HDR frame.
 */
export class SkyPasses {
  constructor() {
    const medium = earthMedium();
    this.medium = medium;
    this.transmittance = new WebGLRenderTarget(...TRANSMITTANCE_SIZE, LUT);
    this.multiscattering = new WebGLRenderTarget(...MULTISCATTERING_SIZE, LUT);
    this.skyView = new WebGLRenderTarget(...SKY_VIEW_SIZE, LUT);
    this.aerialView = new WebGL3DRenderTarget(AERIAL_VIEW_SIZE, AERIAL_VIEW_SIZE, AERIAL_VIEW_SIZE, LUT);
    // The aerial march's running sums between slices (f32, like the compute pass's registers).
    const state = { ...LUT, type: FloatType, minFilter: NearestFilter, magFilter: NearestFilter, count: 2 };
    this.aerialState = [0, 1].map(() => new WebGLRenderTarget(AERIAL_VIEW_SIZE, AERIAL_VIEW_SIZE, state));
    // The sky cube as the filter reads it (Bevy's intermediate 2D array, one layer per face,
    // trilinear), a scratch array for its downsampling, the specular and the diffuse cube.
    const filtered = { ...LUT, minFilter: LinearMipmapLinearFilter, depth: 6 };
    this.source = mipmapped(new WebGLArrayRenderTarget(ENVIRONMENT_SIZE, ENVIRONMENT_SIZE, 6, filtered), ENVIRONMENT_MIPS);
    this.scratch = mipmapped(
      new WebGLArrayRenderTarget(ENVIRONMENT_SIZE, ENVIRONMENT_SIZE, 6, { ...LUT, minFilter: NearestFilter, magFilter: NearestFilter }),
      ENVIRONMENT_MIPS,
    );
    this.specular = mipmapped(new WebGLCubeRenderTarget(ENVIRONMENT_SIZE, filtered), ENVIRONMENT_MIPS);
    this.diffuse = new WebGLCubeRenderTarget(IRRADIANCE_SIZE, LUT);
    // One level of the specular or diffuse cube, its six faces side by side.
    this.atlas = new WebGLRenderTarget(6 * ENVIRONMENT_SIZE, ENVIRONMENT_SIZE, {
      ...LUT,
      minFilter: NearestFilter,
      magFilter: NearestFilter,
    });

    const v4 = () => [new Vector4(), new Vector4()];
    this.uniforms = {
      groundAlbedo: { value: new Vector3() },
      worldToAtmosphere: { value: worldToAtmosphere() },
      worldFromAtmosphere: { value: new Matrix4() },
      atmosphereFromWorld: { value: new Matrix4() },
      worldFromView: { value: new Matrix4() },
      viewFromClip: { value: new Matrix4() },
      viewPosition: { value: new Vector3() },
      viewExposure: { value: 1 },
      lightColor: { value: v4() },
      lightDirection: { value: v4() },
      lightDisk: { value: v4() },
      lightCount: { value: 0 },
      mediumDensityLut: { value: medium.density },
      mediumScatteringLut: { value: medium.scattering },
      transmittanceLut: { value: this.transmittance.texture },
      multiscatteringLut: { value: this.multiscattering.texture },
      skyViewLut: { value: this.skyView.texture },
      aerialViewLut: { value: this.aerialView.texture },
    };
    const material = (fragmentShader, own = {}, extra = {}) =>
      new ShaderMaterial({
        glslVersion: GLSL3,
        vertexShader: skyVertex,
        fragmentShader,
        uniforms: { ...this.uniforms, ...own },
        depthTest: false,
        depthWrite: false,
        blending: NoBlending,
        ...extra,
      });
    const filter = (fragmentShader, own = {}) =>
      material(fragmentShader, { inputTexture: { value: null }, face: { value: 0 }, size: { value: 0 }, ...own });
    this.materials = {
      transmittance: material(transmittanceFragment),
      multiscattering: material(multiscatteringFragment),
      skyView: material(skyViewFragment),
      aerialMarch: material(aerialMarchFragment, {
        slice: { value: 0 },
        totalIn: { value: null },
        throughputIn: { value: null },
      }),
      aerialStore: material(aerialStoreFragment, { totalIn: { value: null } }),
      environment: material(environmentFragment, { face: { value: 0 }, size: { value: ENVIRONMENT_SIZE } }),
      downsample: filter(downsampleFragment, { mipLevel: { value: 0 } }),
      copyLevel: filter(copyLevelFragment, { mipLevel: { value: 0 } }),
      copyFace: material(copyFaceFragment, { atlas: { value: null }, face: { value: 0 }, size: { value: 0 } }),
      radiance: filter(radianceFragment, { sampleCount: { value: 0 }, roughnessLevel: { value: 0 } }),
      irradiance: filter(irradianceFragment, { sampleCount: { value: 1024 } }),
    };
    // Bevy's render_sky blend `inscattering + dst × transmittance` with the transmittance per
    // channel (dual-source blending). WebGL 2 has no second blend source, so it runs once per
    // colour channel with only that channel written, the channel's transmittance in alpha and
    // the blend `src + dst × src alpha`: one blend per channel, like Bevy's. The frame's alpha
    // is kept, as in Bevy.
    this.compositeMaterial = material(
      renderSkyFragment,
      {
        sceneDepth: { value: null },
        cameraViewFromClip: { value: new Matrix4() },
        viewportSize: { value: new Vector2() },
        channel: { value: 0 },
      },
      {
        blending: CustomBlending,
        blendEquation: AddEquation,
        blendSrc: OneFactor,
        blendDst: SrcAlphaFactor,
        blendEquationAlpha: AddEquation,
        blendSrcAlpha: ZeroFactor,
        blendDstAlpha: OneFactor,
      },
    );
    // The composite's fullscreen triangle sets the channel's write mask around its draw (three's
    // materials only switch all colour writes); three's cached mask stays all-on.
    this.compositeMesh = new Mesh(this.#fullscreenTriangle(), this.compositeMaterial);
    this.compositeMesh.frustumCulled = false;
    this.compositeMesh.onBeforeRender = (renderer) => {
      const c = this.compositeMaterial.uniforms.channel.value;
      renderer.getContext().colorMask(c === 0, c === 1, c === 2, false);
    };
    this.compositeMesh.onAfterRender = (renderer) => renderer.getContext().colorMask(true, true, true, true);
    this.compositeCamera = new OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quad = new FullScreenQuad(this.materials.transmittance);
  }

  /** A fullscreen pass into `target`'s `layer` (or cube face) and `mip`, or its `w × h` corner. */
  #draw(renderer, material, target, layer = 0, mip = 0, w = Math.max(1, target.width >> mip), h = Math.max(1, target.height >> mip)) {
    target.viewport.set(0, 0, w, h);
    renderer.setRenderTarget(target, layer, mip);
    this.quad.material = material;
    this.quad.render(renderer);
  }

  /**
   * Updates the LUTs (and, with `environment`, the environment cubes) for a frame.
   * @param lights the directional lights in Bevy's order: `{ directionToLight, color, diskSize,
   *   diskIntensity }`, colour × illuminance
   */
  render(renderer, { groundAlbedo, lights, worldFromView, clipFromView, exposure, environment }) {
    const u = this.uniforms;
    u.groundAlbedo.value.set(...groundAlbedo);
    u.worldFromView.value.copy(worldFromView);
    u.viewFromClip.value.copy(clipFromView).invert();
    const camWorld = new Vector3().setFromMatrixPosition(worldFromView);
    u.viewPosition.value.copy(camWorld);
    u.viewExposure.value = exposure;
    // Bevy `prepare_atmosphere_transforms`.
    const camPos = camWorld.clone().applyMatrix4(u.worldToAtmosphere.value);
    const atmoY = camPos.lengthSq() > 0 ? camPos.clone().normalize() : new Vector3(0, 1, 0);
    const horizontal = (ref) => ref.clone().sub(atmoY.clone().multiplyScalar(atmoY.dot(ref)));
    let atmoZ = horizontal(new Vector3(0, 0, -1));
    atmoZ = atmoZ.lengthSq() > 0 ? atmoZ.normalize() : horizontal(new Vector3(0, -1, 0)).normalize();
    const atmoX = atmoY.clone().cross(atmoZ).normalize();
    u.worldFromAtmosphere.value.makeBasis(atmoX, atmoY, atmoZ).setPosition(camWorld);
    u.atmosphereFromWorld.value.copy(u.worldFromAtmosphere.value).transpose();
    u.lightCount.value = Math.min(lights.length, 2);
    lights.slice(0, 2).forEach((l, i) => {
      u.lightColor.value[i].set(...l.color, 1);
      u.lightDirection.value[i].set(l.directionToLight.x, l.directionToLight.y, l.directionToLight.z, 0);
      u.lightDisk.value[i].set(l.diskSize, l.diskIntensity, 0, 0);
    });

    const m = this.materials;
    this.#draw(renderer, m.transmittance, this.transmittance);
    this.#draw(renderer, m.multiscattering, this.multiscattering);
    this.#draw(renderer, m.skyView, this.skyView);
    for (let slice = 0; slice < AERIAL_VIEW_SIZE; slice++) {
      const [from, to] = slice % 2 ? this.aerialState : [...this.aerialState].reverse();
      m.aerialMarch.uniforms.slice.value = slice;
      m.aerialMarch.uniforms.totalIn.value = from.textures[0];
      m.aerialMarch.uniforms.throughputIn.value = from.textures[1];
      this.#draw(renderer, m.aerialMarch, to);
      m.aerialStore.uniforms.totalIn.value = to.textures[0];
      this.#draw(renderer, m.aerialStore, this.aerialView, slice);
    }
    if (!environment) return;

    for (let face = 0; face < 6; face++) {
      m.environment.uniforms.face.value = face;
      this.#draw(renderer, m.environment, this.source, face);
    }
    // Bevy `prepare_generated_environment_map_bind_groups`: radiance roughness `mip / 7` with
    // `32 · 2^⌊4·roughness⌋` samples, irradiance 1024 samples.
    for (let mip = 1; mip < ENVIRONMENT_MIPS; mip++) {
      for (let face = 0; face < 6; face++) {
        m.downsample.uniforms.inputTexture.value = this.source.texture;
        m.downsample.uniforms.face.value = face;
        m.downsample.uniforms.mipLevel.value = mip;
        this.#draw(renderer, m.downsample, this.scratch, face, mip);
        m.copyLevel.uniforms.inputTexture.value = this.scratch.texture;
        m.copyLevel.uniforms.face.value = face;
        m.copyLevel.uniforms.mipLevel.value = mip;
        this.#draw(renderer, m.copyLevel, this.source, face, mip);
      }
    }
    // Each level's six faces in one atlas draw (as Bevy dispatches them together), then copied
    // into the cube.
    const toCube = (cube, size, mip) => {
      const c = m.copyFace.uniforms;
      c.atlas.value = this.atlas.texture;
      c.size.value = size;
      for (let face = 0; face < 6; face++) {
        c.face.value = face;
        this.#draw(renderer, m.copyFace, cube, face, mip);
      }
    };
    const r = m.radiance.uniforms;
    r.inputTexture.value = this.source.texture;
    for (let mip = 0; mip < ENVIRONMENT_MIPS; mip++) {
      const roughness = mip / (ENVIRONMENT_MIPS - 1);
      const size = ENVIRONMENT_SIZE >> mip;
      r.roughnessLevel.value = roughness;
      r.sampleCount.value = 32 * 2 ** Math.floor(roughness * 4);
      r.size.value = size;
      this.#draw(renderer, m.radiance, this.atlas, 0, 0, 6 * size, size);
      toCube(this.specular, size, mip);
    }
    const d = m.irradiance.uniforms;
    d.inputTexture.value = this.source.texture;
    d.size.value = IRRADIANCE_SIZE;
    this.#draw(renderer, m.irradiance, this.atlas, 0, 0, 6 * IRRADIANCE_SIZE, IRRADIANCE_SIZE);
    toCube(this.diffuse, IRRADIANCE_SIZE, 0);
  }

  /**
   * Blends the sky and aerial perspective into `target` (the current MSAA HDR target, its frame
   * stored top row first) after the opaque draws, whose resolved depth is `depthTexture`.
   */
  composite(renderer, target, depthTexture, cameraProjectionInverse) {
    const u = this.compositeMaterial.uniforms;
    u.sceneDepth.value = depthTexture;
    u.cameraViewFromClip.value.copy(cameraProjectionInverse);
    u.viewportSize.value.set(target.width, target.height);
    renderer.setRenderTarget(target);
    renderer.state.buffers.color.setMask(true);
    for (let channel = 0; channel < 3; channel++) {
      u.channel.value = channel;
      renderer.render(this.compositeMesh, this.compositeCamera);
    }
  }

  #fullscreenTriangle() {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new Float32BufferAttribute([-1, 3, 0, -1, -1, 0, 3, -1, 0], 3));
    return geometry;
  }

  dispose() {
    for (const t of [
      this.transmittance,
      this.multiscattering,
      this.skyView,
      this.aerialView,
      ...this.aerialState,
      this.source,
      this.scratch,
      this.specular,
      this.diffuse,
      this.atlas,
    ]) {
      t.dispose();
    }
    this.medium.density.dispose();
    this.medium.scattering.dispose();
    for (const material of [...Object.values(this.materials), this.compositeMaterial]) material.dispose();
    this.compositeMesh.geometry.dispose();
    this.quad.dispose();
  }
}
