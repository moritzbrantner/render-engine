// The sky passes (render-engine `src/sky.rs`): Bevy 0.20's atmosphere (`bevy_pbr::atmosphere`,
// Hillaire's LUT method) and the runtime environment-map filter its
// `AtmosphereEnvironmentMapLight` drives (`GeneratedEnvironmentMapLight`), on WebGL 2.
//
// Each frame, before the shadow and main passes, the LUTs whose inputs changed (mmorpg#1115: each
// is redrawn when any input changed at all, so the image is the one redrawing every frame gives):
// the transmittance LUT once (it depends only on the medium), the multiscattering LUT with the
// ground albedo, the sky-view LUT with the camera position or a light, and the aerial-view LUT,
// several slices per pass into as many of its layers (multiple render targets), also with the
// camera's rotation. Then, with sky ambient light and only when the sky changed, the sky rendered
// into a 128² cube, its mip chain, the GGX-prefiltered specular cube (one pass per mip) and the
// 32² irradiance cube. After the opaque draws `composite` blends the sky and the aerial perspective into the
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
/**
 * The most aerial-view slices one pass marches: one colour attachment each beside the two of the
 * running sums. WebGL 2 guarantees four draw buffers (two slices a pass, 16 passes); desktop GPUs
 * have eight (six slices a pass, 6 passes).
 */
const AERIAL_SLICES_PER_PASS = 6;
/** Bevy `AtmosphereEnvironmentMapLight::default().size` and the diffuse cube's size. */
const ENVIRONMENT_SIZE = 128;
export const ENVIRONMENT_MIPS = Math.log2(ENVIRONMENT_SIZE) + 1;
const IRRADIANCE_SIZE = 32;
/**
 * The environment cubes are re-filtered when a sky light turns by more than this (radians) or the
 * camera's distance from the planet centre moves by more than `ENVIRONMENT_ALTITUDE_TOLERANCE`
 * metres since the last filter, and when the ground albedo or a light's colour or disc changes at
 * all. Bevy re-filters every frame; within these tolerances the cubes barely change.
 */
export const ENVIRONMENT_TURN_TOLERANCE = (0.25 * Math.PI) / 180;
export const ENVIRONMENT_ALTITUDE_TOLERANCE = 100;
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

/** Whether a LUT drawn for the inputs `was` (null: not drawn) holds what `now` would draw. */
function sameKey(was, now) {
  return was !== null && was.length === now.length && was.every((x, i) => x === now[i]);
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
      // Its slices per pass follow the context's draw buffers (`#aerialPasses`).
      aerialMarch: null,
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
    this.#material = material;
    /** How many times `render` has filtered the environment cubes. */
    this.environmentFilters = 0;
    /** How many fullscreen passes `render` has drawn into the four atmosphere LUTs. */
    this.lutPasses = 0;
  }

  #material;

  /**
   * The inputs each LUT was last drawn for: the GL texture of the transmittance LUT (a new one, as
   * after a lost context, holds none of them), the multiscattering LUT's ground albedo, the sky
   * view's and the aerial view's inputs (null: not drawn).
   */
  #drawn = { texture: undefined, multiscattering: null, skyView: null, aerialView: null };

  /**
   * The aerial march's passes: each pass's first slice, slice count and framebuffer (the running
   * sums' target it writes and its slices' layers of the 3D LUT), built for the GL textures in
   * `textures`; `target` is the render target that binds them (`setRenderTargetFramebuffer`).
   */
  #aerial = null;

  /** The inputs the environment cubes were last filtered for (null: not filtered). */
  #filtered = null;

  /** Whether the cubes filtered for `#filtered` still stand for `inputs`. */
  #environmentCurrent(inputs) {
    const was = this.#filtered;
    if (!was || was.lights.length !== inputs.lights.length) return false;
    const same = (a, b) => a.every((x, i) => x === b[i]);
    if (!same(was.groundAlbedo, inputs.groundAlbedo)) return false;
    if (Math.abs(was.radius - inputs.radius) > ENVIRONMENT_ALTITUDE_TOLERANCE) return false;
    const cosTurn = Math.cos(ENVIRONMENT_TURN_TOLERANCE);
    return was.lights.every((l, i) => {
      const n = inputs.lights[i];
      return (
        same(l.color, n.color) &&
        l.diskSize === n.diskSize &&
        l.diskIntensity === n.diskIntensity &&
        l.direction.dot(n.direction) >= cosTurn
      );
    });
  }

  /** A fullscreen pass into `target`'s `layer` (or cube face) and `mip`, or its `w × h` corner. */
  #draw(renderer, material, target, layer = 0, mip = 0, w = Math.max(1, target.width >> mip), h = Math.max(1, target.height >> mip)) {
    target.viewport.set(0, 0, w, h);
    renderer.setRenderTarget(target, layer, mip);
    this.quad.material = material;
    this.quad.render(renderer);
  }

  /** One fullscreen pass into an atmosphere LUT. */
  #lutPass(renderer, material, target, layer = 0) {
    this.lutPasses++;
    this.#draw(renderer, material, target, layer);
  }

  /** The aerial-view LUT: its march over every slice, `#aerialPasses`' passes in order. */
  #marchAerialView(renderer) {
    const { passes, target } = this.#aerialPasses(renderer);
    const material = this.materials.aerialMarch;
    const u = material.uniforms;
    this.quad.material = material;
    for (const pass of passes) {
      u.firstSlice.value = pass.first;
      u.sliceCount.value = pass.count;
      u.totalIn.value = pass.from.textures[0];
      u.throughputIn.value = pass.from.textures[1];
      renderer.setRenderTargetFramebuffer(target, pass.framebuffer);
      renderer.setRenderTarget(target);
      this.lutPasses++;
      this.quad.render(renderer);
    }
  }

  /**
   * The aerial march's passes for `renderer`'s context, built on first use and again for new GL
   * textures (a restored context). Each pass marches as many slices as the context has draw
   * buffers beside the running sums' two (at most `AERIAL_SLICES_PER_PASS`): its framebuffer has
   * the sums' next target at attachments 0 and 1 and its slices' layers of the 3D LUT after them.
   * three binds one layer of a 3D target at a time, so the framebuffers are the GL context's own,
   * bound through `setRenderTargetFramebuffer`.
   */
  #aerialPasses(renderer) {
    const gl = renderer.getContext();
    const glTexture = (t) => renderer.properties.get(t).__webglTexture;
    for (const t of [this.aerialView, ...this.aerialState]) renderer.initRenderTarget(t);
    const textures = [this.aerialView.texture, ...this.aerialState.flatMap((s) => s.textures)].map(glTexture);
    const was = this.#aerial;
    if (was?.gl === gl && was.textures.every((t, i) => t === textures[i])) return was;
    this.#deleteAerialFramebuffers();

    const buffers = Math.min(gl.getParameter(gl.MAX_DRAW_BUFFERS), gl.getParameter(gl.MAX_COLOR_ATTACHMENTS));
    const perPass = Math.min(AERIAL_SLICES_PER_PASS, buffers - 2);
    if (was?.perPass !== perPass) {
      this.materials.aerialMarch?.dispose();
      this.materials.aerialMarch = this.#material(aerialMarchFragment(perPass), {
        firstSlice: { value: 0 },
        sliceCount: { value: 0 },
        totalIn: { value: null },
        throughputIn: { value: null },
      });
    }
    const passes = [];
    for (let first = 0, i = 0; first < AERIAL_VIEW_SIZE; first += perPass, i++) {
      const count = Math.min(perPass, AERIAL_VIEW_SIZE - first);
      const to = this.aerialState[(i + 1) % 2];
      const framebuffer = gl.createFramebuffer();
      renderer.state.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      to.textures.forEach((t, k) => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + k, gl.TEXTURE_2D, glTexture(t), 0));
      for (let k = 0; k < count; k++) {
        gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2 + k, textures[0], 0, first + k);
      }
      gl.drawBuffers(Array.from({ length: 2 + count }, (_, k) => gl.COLOR_ATTACHMENT0 + k));
      passes.push({ first, count, from: this.aerialState[i % 2], framebuffer });
    }
    // three binds the next target itself.
    renderer.state.bindFramebuffer(gl.FRAMEBUFFER, null);
    const target = was?.target ?? new WebGLRenderTarget(AERIAL_VIEW_SIZE, AERIAL_VIEW_SIZE, { depthBuffer: false });
    this.#aerial = { gl, textures, perPass, passes, target };
    return this.#aerial;
  }

  #deleteAerialFramebuffers() {
    const was = this.#aerial;
    if (!was) return;
    for (const { framebuffer } of was.passes) if (was.gl.isFramebuffer(framebuffer)) was.gl.deleteFramebuffer(framebuffer);
    was.passes = [];
  }

  /**
   * Updates the LUTs (and, with `environment`, the environment cubes when their inputs changed)
   * for a frame.
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
    // A new GL texture (the first frame, or a restored context) holds no LUT nor cube yet.
    renderer.initRenderTarget(this.transmittance);
    const texture = renderer.properties.get(this.transmittance.texture).__webglTexture;
    if (texture !== this.#drawn.texture) {
      this.#drawn = { texture, multiscattering: null, skyView: null, aerialView: null };
      this.#filtered = null;
      this.#lutPass(renderer, m.transmittance, this.transmittance);
    }
    // Each LUT is redrawn when any of its inputs changed at all (exact comparison), so a kept LUT
    // holds what this frame would draw.
    const multiscatteringKey = [...groundAlbedo];
    const lightKey = lights
      .slice(0, 2)
      .flatMap((l) => [l.directionToLight.x, l.directionToLight.y, l.directionToLight.z, ...l.color, l.diskSize, l.diskIntensity]);
    const skyViewKey = [...multiscatteringKey, ...camWorld.toArray(), ...lightKey];
    const aerialViewKey = [...skyViewKey, ...worldFromView.elements, ...clipFromView.elements];
    if (!sameKey(this.#drawn.multiscattering, multiscatteringKey)) {
      this.#drawn.multiscattering = multiscatteringKey;
      this.#lutPass(renderer, m.multiscattering, this.multiscattering);
    }
    if (!sameKey(this.#drawn.skyView, skyViewKey)) {
      this.#drawn.skyView = skyViewKey;
      this.#lutPass(renderer, m.skyView, this.skyView);
    }
    if (!sameKey(this.#drawn.aerialView, aerialViewKey)) {
      this.#drawn.aerialView = aerialViewKey;
      this.#marchAerialView(renderer);
    }
    if (!environment) {
      this.#filtered = null;
      return;
    }
    // The cubes sample the sky-view LUT at the camera's distance from the planet centre.
    const inputs = {
      groundAlbedo: [...groundAlbedo],
      radius: camPos.length(),
      lights: lights.slice(0, 2).map((l) => ({
        direction: new Vector3().copy(l.directionToLight).normalize(),
        color: [...l.color],
        diskSize: l.diskSize,
        diskIntensity: l.diskIntensity,
      })),
    };
    if (this.#environmentCurrent(inputs)) return;
    this.#filtered = inputs;
    this.environmentFilters++;

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

  /**
   * Compiles every program `render` and `composite` draw with, drawing nothing
   * (`Lighting.prepare`): `compile(object, camera)` on each pass's mesh. They all draw into
   * render targets, so one bound target stands for theirs.
   */
  compile(renderer, compile) {
    // The aerial march's program depends on the context's draw buffers.
    this.#aerialPasses(renderer);
    renderer.setRenderTarget(this.transmittance);
    for (const material of Object.values(this.materials)) {
      this.quad.material = material;
      // FullScreenQuad.render draws this mesh.
      compile(this.quad._mesh, this.compositeCamera);
    }
    compile(this.compositeMesh, this.compositeCamera);
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
    for (const material of [...Object.values(this.materials), this.compositeMaterial]) material?.dispose();
    this.#deleteAerialFramebuffers();
    this.#aerial = null;
    this.compositeMesh.geometry.dispose();
    this.quad.dispose();
  }
}
