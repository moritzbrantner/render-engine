# Technical decisions

Current direction, owner-approved 2026-10-30. Source code and the oracle prove what is
implemented; GitHub issues track remaining work.

## D1 — Bevy is the oracle and the performance baseline

render-engine provides lighting and particles for the owner's games. It is built against Bevy
the way [physics-engine](https://github.com/moritzbrantner/physics-engine) is built against
Rapier:

- **Lighting:** `bevy_pbr` (Bevy 0.20.1) is the visual reference.
- **Particles:** `bevy_hanabi` (GPU particles) is the reference. Its latest release (0.19.0) is
  built on Bevy 0.19, and its Bevy 0.20 port (djeedai/bevy_hanabi#576) is only on `main`. The
  oracle therefore pins a `main` commit (`d277555d4a`) and moves to the first 0.20 release once it ships (D9).

Both stay on their latest stable releases. Bevy is a dev/oracle dependency only (`oracle/`); the
engine itself never depends on Bevy, and the games do not ship it.

## D2 — Our own API, shaped by the consumers

The engine is a small wgpu renderer, not a Bevy clone. Its scene description
(`render_engine::scene`) uses Bevy's physical conventions so the oracle can translate it field by
field: metres, directional illuminance in lux, ambient brightness in cd/m², exposure as EV100,
linear RGB colours, the metallic/roughness/reflectance material model and reverse-Z depth.

Consumers, in priority order:

1. **mmorpg native client** (`moenarch-games/mmorpg`, `crates/mmorpg-client`): wgpu 30 plus the
   `three-d-*` crates from `moritzbrantner/3d-lab`. The engine stays on the same wgpu major so it
   can render into the client's device and passes. First feature consumer: mmorpg#529 (Greyhaven
   sun/sky, shadows, ambient/IBL).
2. **mmorpg web client** (three.js 0.186), through a three.js port (see D8).

## D3 — The gate

Every parity scenario renders headless on both engines. The oracle (`oracle/`) fails a scenario
when:

| Gate | Default | Where enforced |
| --- | --- | --- |
| Mean [NVIDIA FLIP](https://research.nvidia.com/publication/2020-07_FLIP) error, render-engine vs Bevy | ≤ **0.005** | everywhere (CI + local) |
| 99th-percentile per-pixel FLIP error | ≤ **0.05** | everywhere (CI + local) |
| Median GPU frame time, render-engine vs Bevy | ≤ **1.10×** Bevy (at most 10% slower) | hardware adapters only |

- FLIP comes from [`nv-flip`](https://crates.io/crates/nv-flip) (gfx-rs bindings to NVIDIA's
  BSD-licensed reference implementation), LDR mode at the default 67 pixels per degree.
- The p99 term exists because the whole-image mean dilutes local defects: on `sun_shadows`, a
  renderer with shadows switched off scores mean 0.0104 but p99 0.748. Calibration on the first
  scenario (RTX 3060 Ti): parity 0.0005 / p99 0.0008; Lambert instead of Burley diffuse 0.0180;
  no multiscatter compensation 0.0128; no normal bias 0.0014 / 0.0059 (passes with the 2×2
  filter). The wide Gaussian filter is more sensitive: on `sun_cascades`, missing Bevy's √2
  normal-bias factor scored 0.0048 / p99 0.052 (fails); with it, 0.0004 / 0.0029, and
  `sun_shadows` went from 0.0005 to 0.0000.
- GPU time is measured with timestamp queries, the same quantity on both sides: the sum of a
  frame's pass spans. Bevy's is the sum of the top-level `RenderDiagnosticsPlugin` spans
  (`render/<span>/elapsed_gpu`: mesh preprocessing, shadow views, main pass, upscaling, …);
  render-engine writes a begin and an end timestamp on every pass. Gaps between passes, where a
  shared GPU runs other processes' work, count on neither side.
- Measurement protocol (#28): 300 frames per side and round; render-engine renders 0.5 s of
  untimed frames first and then submits frames back to back in batches of 64. The rounds
  alternate (Bevy, render-engine, Bevy, render-engine) and each side's time is the median of its
  fastest round, since other work on the GPU and lower power states only slow a round down. A
  scenario still over the margin gets up to two more rounds before it fails. Before #28,
  render-engine submitted one frame and waited for it; the mostly idle GPU then dropped to a
  lower power state part-way through some runs and the same frame timed exactly 4× slower
  (`sun_cascades` 0.20 → 0.80 ms within one run), which made the local gate flip between 0.7×
  and 4× from run to run without any code change. Bevy, when the host's CPU is saturated, shows
  the same 3.3× step (0.27 → 0.88–0.92 ms).
- A scenario's thresholds may only be loosened in the PR that adds it, with the measured numbers
  and the reason in that PR.

## D4 — How CI renders

GitHub-hosted runners have no GPU. CI installs Mesa's **lavapipe** (`mesa-vulkan-drivers`) and
sets `WGPU_FORCE_FALLBACK_ADAPTER=1`; Bevy and render-engine both honour that variable (and
`WGPU_ADAPTER_NAME`), so both sides run on the same software Vulkan device.

- On lavapipe the **image gates are enforced**; GPU times are printed and marked advisory, because
  CPU-rasterised timings say nothing about GPU cost.
- On a hardware adapter (the owner's RTX 3060 Ti) **both gates are enforced**. Run the oracle
  locally before merging work that touches render passes or shaders, and put its table in the
  PR.
- The images agree across devices with the same gate; there is no per-device rule. Every f16
  rounding happens at the same stage as on Bevy, in the device's own `Rgba16Float` store
  (main pass, tone-mapping target, then a blit into the output). Emulating the tone-mapped f16
  store with `pack2x16float` instead (until #22) rounded to nearest where lavapipe's store does
  not: about 5% of pixels came out exactly one 8-bit step brighter than Bevy, never darker, for
  p99 up to 0.045 on lavapipe against 0.0007 on the RTX 3060 Ti. Since #22 every scenario
  scores mean 0.0000 and p99 ≤ 0.0005 on lavapipe and ≤ 0.0007 on the RTX 3060 Ti, and a fog
  density 10% off still fails on lavapipe (mean 0.021–0.026, p99 0.062–0.077).

## D5 — Pinned Bevy configuration per scenario

The Bevy side configures each scenario explicitly instead of relying on Bevy defaults, so a Bevy
upgrade that changes a default shows up as a deliberate scenario change:

- an `Hdr` camera with the scenario's `Tonemapping` (`AcesFitted` for the shadow scenarios,
  `TonyMcMapface` and `AgX` with Bevy's `tonemapping_luts` in `tonemap_range`),
  `DebandDither::Disabled`, `Msaa::Sample4`, default `ColorGrading`, an `Rgba8UnormSrgb` image
  target. render-engine embeds Bevy's own LUT files (`src/luts/`, licences kept there) and
  tone maps into an `Rgba16Float` target that it blits into the output, like Bevy's
  tonemapping and upscaling nodes;
- sun shadows: the scenario's full `CascadeShadowConfigBuilder` (cascade count, minimum and
  maximum distance, first cascade far bound, overlap), its `ShadowFilteringMethod` on the camera
  (`Hardware2x2` or `Gaussian`), `DirectionalLightShadowMap` 2048 and Bevy's depth/normal biases
  (0.02 m, 1.8 texels). `sun_shadows` keeps one cascade with 2×2 PCF; `sun_cascades` uses
  Bevy's defaults (4 cascades, 10 m / 150 m, overlap 0.2, Gaussian). Shadow maps are rendered
  with unclipped depth like Bevy's directional shadow views (`DEPTH_CLIP_CONTROL`; on adapters
  without it, Bevy's fragment-depth emulation). Clamping per vertex instead bends the stored
  depth of large ground triangles that reach towards a low sun; it hid the self-shadowing Bevy
  shows in the fog inscattering lookup (`fog_*_sun` scored 0.014 / 0.20 before);
- `GlobalAmbientLight` for ambient (brightness 0 in the IBL scenarios);
- image-based light: an `EnvironmentMapLight` on the camera (intensity in cd/m², no rotation)
  whose diffuse and specular `Image`s are built from the same decoded texels render-engine
  uploads, with Bevy's default (trilinear, clamped) sampler. The maps are prefiltered once,
  offline, from a committed CC0 HDRI (`assets/ibl/`, provenance in its README) by the oracle's
  `bake_ibl`: the glTF IBL Sampler's split-sum layout that Bevy documents (GGX specular with
  perceptual roughness `mip / (mips - 1)`, Lambertian diffuse), written as zstd `rgb9e5` KTX2 by
  KTX-Software. Both sides therefore sample identical cube maps; render-engine owns the
  sampling and the BRDF terms. These scenarios do not use Bevy's runtime
  `GeneratedEnvironmentMapLight`; the sky does (below);
- distance fog: a `DistanceFog` on the camera with the scenario's colour, falloff and
  directional-light colour/exponent (`fog_modes`: every `FogFalloff`, each without and with sun
  inscattering, clear colour = fog colour). render-engine ports `apply_fog` and `fog.wesl`
  term by term on the exposed colour, including the inscattering's shadow lookup with the view
  direction as normal. Volumetric fog is out of scope;
- point and spot lights: `PointLight`/`SpotLight` entities with the scene's lumens, range,
  radius, cone angles and Bevy's shadow biases (point 0.08 m / 0.6 texels, spot 0.02 m /
  1.8 texels, near plane 0.1 m), `DirectionalLightShadowMap` 2048 (spot maps share its array,
  after the sun cascades) and `PointLightShadowMap` 1024, and the `experimental_pbr_pcss`
  feature so `point_spot_soft` can switch on soft shadows. Bevy assigns lights to clusters;
  render-engine loops over the whole list in Bevy's order (point lights, then spot lights, each
  with the shadow casters first, then the sun). Outside its range a light contributes exactly
  zero in both, so the sums agree; clustering waits until the GPU-time gate asks for it. The
  temporal shadow filter (it needs TAA) and light textures are out of scope;
- consumer meshes and textured materials: `textured_props` reads CC0 glTF props
  (`assets/props/`) once in the oracle (`content.rs`, following Bevy's glTF loader: base
  colour, normal, metallic-roughness, occlusion and emissive textures, alpha mode, double-sided,
  `COLOR_0`, MikkTSpace tangents through the same `bevy_mikktspace` crate when a normal map needs
  them) and gives both sides the same vertex attributes, texels, box-filtered mip chains and
  samplers. Bevy gets `Mesh`, `Image` and `StandardMaterial` assets built from that data
  rather than `bevy_gltf`'s scene spawner, which keeps the translation field by field like
  every other scenario. The props use alpha mask (in the main and the shadow pass), a blended
  sphere and a mirrored instance. render-engine ports `pbr_input_from_standard_material`,
  `apply_normal_mapping`, `alpha_discard` and `prepass_alpha_discard`. It normalises normals
  and tangents per vertex and leaves the interpolated normal unnormalised for the shadow
  biases, as Bevy does. Mirrored instances get a clockwise pipeline instead of Bevy's
  `winding_corrected_front_facing`; for double-sided materials the two agree, and single-sided
  mirrored meshes keep their outside faces;
- time-of-day sky: an `Atmosphere::earth` with `ScatteringMedium::earth(256, 256)` (placed by
  Bevy's hook 6 360 km below the origin), `AtmosphereSettings::default()` (LUT mode) and an
  `AtmosphereEnvironmentMapLight` (128², intensity 1) on the camera, a `SunDisk` on each
  directional light, raw sunlight (`lux::RAW_SUNLIGHT`) and a second, unshadowed directional
  light for the moon (`sky_day_cycle_*`: sun at 60°, 15°, 2° and −10° on one `DayCycle`, the
  moon in the night view). render-engine ports the atmosphere passes term by term
  (`src/shaders/sky/`): the CPU-built medium LUTs (`Rgba32Float`, so a sky needs
  `FLOAT32_FILTERABLE`), the transmittance, multiscattering, sky-view and aerial-view LUTs every
  frame, the fullscreen `render_sky` pass between the opaque and the blended draws (reading the
  MSAA depth's first sample, dual-source blending where the device has it, as Bevy chooses),
  and the sun and moon transmittance and visible-disc share in the main pass. Ambient light
  follows the sky through Bevy's own runtime filter: the sky-view LUT rendered into a 128² cube,
  its mip chain, the GGX radiance filter per mip (hash noise; the oracle has no
  `bluenoise_texture`) and the 1024-sample irradiance cube, all ported from
  `GeneratedEnvironmentMapLight`, so both sides filter identically. The mip chain matters:
  Bevy's single-pass downsampler samples the first level at the centre of each block's first
  texel (`(uv + 0.5) * inverse_input_size`), i.e. it point-samples rather than averages; a box
  filter there scored mean 0.0015–0.0062 / p99 0.024–0.034, the port of it 0.0001 / ≤ 0.0034.
  Bevy filters one frame behind (its filter runs before the atmosphere passes); render-engine
  filters the current frame's sky, which agrees once Bevy has warmed up and does not lag in a
  running day cycle. Bevy's raymarched mode, volumetric fog and stars are out of scope;
- frustum culling (#28): Bevy's CPU culling (`check_visibility` and the light views'), with
  GPU occlusion culling off as Bevy has it by default. Each mesh's local `Aabb` is computed
  once when the `World` uploads it; per frame, every instance's box under its model matrix is
  tested against each view's half spaces (`ViewFrustum::intersects_obb`): the camera without
  its far plane, each sun cascade without its near plane (casters between the sun and the
  cascade still cast; depth is pancaked), spot maps and point-light cube faces with their far
  plane at the light's range after Bevy's light-sphere test. Every view gets its own draws over
  a run of visible instance indices, so a view only draws what it can see. Both sides share
  shape meshes and materials between objects (as a game shares handles), so Bevy's batching and
  ours see the same groups. `many_instances` (22 400 props on a 400 × 350 m field, four
  cascades) measures it;
- GPU skinning (#34): `skinned_figure` gives Bevy the same `Mesh` with
  `ATTRIBUTE_JOINT_INDEX`/`ATTRIBUTE_JOINT_WEIGHT`, a `SkinnedMesh` whose joints are
  unparented entities at `world_from_local * model_from_joint` and the file's inverse bind
  matrices, so Bevy's `joint_global * inverse_bindpose` equals render-engine's
  `world_from_local * palette[j]`. The pose is set on joint transforms (no `AnimationPlayer`,
  so no clock to match). render-engine ports `skin_model` and `skin_normals` (inverse transpose
  of the skinned matrix, tangents through the matrix itself); its palette is mesh-local so the
  instance transform keeps placing the figure, where Bevy's joints are world-space and replace
  the mesh entity's transform. Culling uses per-joint bind-pose boxes like Bevy's
  `SkinnedMeshBounds`; Bevy only does that with `DynamicSkinnedMeshBounds`, so its figures get
  `NoFrustumCulling` (they are always in view). Mesh streams are split (positions, normals,
  surface attributes, joints and weights) so a CPU-side pose can rewrite positions and normals
  alone; Bevy interleaves, which does not change the image.

Later scenarios lift these one at a time (temporal filtering).

## D6 — Order: lighting first, then particles

Lighting scenarios come first because mmorpg#529 is waiting on them: cascaded shadows,
point/spot lights, ambient/IBL, time-of-day sky, fog. Then the mmorpg native integration and the
web path. The particle track (bevy_hanabi oracle, emitters, GPU simulation, lit/soft particles)
starts after that. Its oracle side and parity rules are in D9.

## D7 — Ownership boundary

render-engine owns render passes, lights, shadows, sky/fog/ambient, tone mapping and (later)
particle simulation and rendering. Games own scene content, ECS, asset loading, the game loop,
gameplay and the day clock that drives the sun. The consumer owns the wgpu device, queue and
frame. It uploads meshes, textures and materials once into a `World` (stable handles,
incremental `set_*` updates), describes each frame as a `Scene` (camera and lights) plus a list
of instances (mesh, material, transform), and has `Renderer::encode` record into its encoder and
colour target. Its UI draws on top afterwards. The engine keeps its MSAA, HDR, depth and
shadow targets in a `Viewport`. The headless path uses the same calls. See
[consumer integration](consumer-integration.md).

## D8 — Web path: a three.js port gated by the same oracle

Owner decision 2026-10-31 (#10, option B). The web client gets parity through a **three.js
port** of the lighting, using three.js built-ins where they pass the gate and custom TSL/shader
chunks where they don't. Each web scenario is checked against the **same Bevy oracle
scenarios** through a headless Chromium side with the same FLIP thresholds (D3); GPU time is
advisory there. Option A (render-engine compiled to WASM/WebGPU) stays available later for
particles only, if three.js GPU particles cannot meet the gate. Scope of the first step: #11.

As built (#11):

- **Package.** `packages/three-lighting`, published from the repository root as
  `@moritzbrantner/three-lighting` (consumers pin `github:moritzbrantner/render-engine#<commit>`,
  like 3d-lab's renderer). It targets three.js 0.186's `WebGLRenderer` (the mmorpg web client's
  renderer), so the custom parts are GLSL ES 3.0 `ShaderMaterial`s ported term for term from
  `lit.wgsl` and `tonemap.wgsl`, not TSL (TSL needs the WebGPU/node renderer).
- **Built-ins that pass:** `ShaderMaterial`, multisampled `WebGLRenderTarget`s, `DepthTexture`
  comparison sampling for the 2×2 PCF taps, three's `FullScreenQuad`, its KTX2 libraries
  (`ktx-parse`, `zstddec`) for the IBL cube maps and LUTs, and its `ACESFilmicToneMapping` (the
  same Stephen Hill fit as Bevy's, with `toneMappingExposure` 0.6 to cancel its 1/0.6).
- **Custom, because the built-ins differ from Bevy by more than the gate:**
  `MeshStandardMaterial` (Lambert instead of Burley diffuse, no multiscatter compensation, no
  specular ambient term; D3 measured Lambert alone at FLIP 0.018), three's `CSM` addon (its own
  splits and fitting; Bevy snaps cascades to texels and rounds their diameter), `PMREMGenerator`
  (its own cube-UV filtering, not Bevy's split-sum maps; the port samples the same prefiltered
  KTX2 maps as render-engine), `Fog`/`FogExp2` (view depth and a smoothstep, not distance),
  `AgXToneMapping` (an analytic fit, not Bevy's LUT) and Tony McMapface (not in three.js). `Sky`
  is not used yet; the sky is #39.
- **Shadow pass like Bevy's pipeline:** with `EXT_clip_control` and `EXT_depth_clamp` the cascades
  render in zero-to-one clip space with rasterizer-clamped (pancaked) reverse-Z depth into
  `DEPTH_COMPONENT32F`; without them the depth is remapped and clamped per fragment. The sun-glow
  fog term samples the shadow with the view direction as normal, which sits on the edge of
  self-shadowing; only the native pipeline reproduces Bevy's result there.
- **Point and spot lights (#38)** port render-engine's passes rather than three's `PointLight`/
  `SpotLight` shadows (their own falloff, projection and filters): Bevy's clustered order (points,
  then spots, casters first, then the sun), range window, sphere-light specular and cone falloff;
  spot maps at the cascades' size, point lights in a `WebGLCubeRenderTarget` with a
  `CubeDepthTexture` (three allocates it as `DEPTH_COMPONENT24`; Bevy uses 32-bit float). Both
  render with Bevy's infinite reverse-Z projection and clipped depth; cube faces render upside
  down, because GL stores a face's NDC y = −1 in row 0 where wgpu stores +1. WebGL 2 cannot
  linearly sample a depth texture without comparison, so the local passes also write their depth
  into an R32F target that the soft-shadow (PCSS) blocker search reads. Shader macros are
  `LIT_*`: three.js rewrites `NUM_POINT_LIGHTS` and its other light counts in every shader.
- **Oracle side.** `oracle --no-three` skips it; otherwise the oracle exports every lighting
  scenario the port covers (`oracle/src/three_side.rs`: scene in Bevy units plus vertex data) and
  renders them in one headless Chromium run (`oracle/three/render.mjs`, Playwright) through
  ANGLE on Vulkan, into an sRGB target like Bevy's output. On a software adapter Chromium gets
  lavapipe's ICD, the device Bevy runs on: lavapipe and SwiftShader round sRGB stores
  differently (one 8-bit step on about a sixth of the pixels, FLIP mean 0.023 on
  `sun_shadows`), so only the same device compares the shading. Each scenario reports
  `three.status`: passed, failed (fails the run), a known gap (`three_side::GAPS`, with its
  issue) or not ported (with the issue of the missing capability). GPU time is advisory
  (`EXT_disjoint_timer_query_webgl2` over the whole `Lighting.render`, where the browser has it).

## D9 — Particle parity: seeded replay, then statistics

Decided in #12. Particles are random, so an image comparison only works while both sides draw the
same random numbers for the same particles. Each particle scenario (`oracle/src/particles.rs`)
therefore runs at a **fixed timestep** of exactly 1/64 s. Every age and time is a multiple of a
power of two and adds up exactly in `f32`, so a fixed lifetime ends on the same frame on both
sides. Each scenario also has a **fixed seed** and fixed capture frames. The Bevy side is
Bevy 0.20.1 with `bevy_hanabi` (D1). The spawners start only once every effect's pipelines are
ready, and the frame before the first spawn is captured as the empty background ("plate").
Each capture is judged by one of two tiers:

| Tier | Captures | Gate |
| --- | --- | --- |
| **Exact** (seeded replay) | before any particle slot is recycled (until the first particle dies) | the lighting FLIP gate (D3): mean ≤ 0.005, p99 ≤ 0.05 |
| **Statistical** | after slots are recycled | per scenario: alpha coverage, spatial histogram and colour histogram distances under the scenario's `StatThresholds` |
| **Population**, every frame | all | alive-particle count and age statistics identical per emitter and frame |

- **Why the exact tier ends at the first death.** hanabi seeds each particle's random stream
  with `pcg_hash(slot ^ spawner_seed)` (`vfx_init.wesl`). `spawner_seed` is a CPU-side value
  that changes every frame (`StdRng::seed_from_u64(previous)` from the effect's `prng_seed`), and
  `slot` is the particle's index in the effect's buffer. Until a particle dies, a frame's spawns
  take the next free slots in a fixed order. render-engine can then reproduce the slot, the
  seed chain, `pcg_hash`, `to_float01` and the order in which the init modifiers draw from the
  stream, and get the same particles. Once particles die, the update pass returns their slots
  through GPU atomics, so which new particle gets which slot depends on hanabi's thread
  scheduling. render-engine cannot reproduce that, even though Bevy happens to repeat it
  bit-for-bit on the RTX 3060 Ti and on lavapipe (`--calibrate`, "rerun" rows).
- **Blending.** Additive blending does not depend on the draw order, and hanabi draws in its
  alive-list order, which also comes from atomics. hanabi does not sort an alpha-blended
  effect's particles (only ribbons), so that order shows over a blended background, and it
  changes from run to run: a Bevy rerun of `particles_alpha` scores FLIP p99 0.037–0.065 and
  colour 0.007 against itself. Alpha-blended scenarios are therefore judged by the statistical
  tier at every capture (#14). Between effects, Bevy's transparent phase sorts hanabi's
  effects back to front by their transform, additive and alpha-blended alike (hanabi queues
  both into `Transparent3d`); render-engine does the same (#15; before it drew the additive
  emitters first, which only `smoke_and_sparks`, sparks in front of smoke, tells apart).
- **Statistics** (`ImageStats`, both images against their own plate): *alpha coverage* is the
  share of pixels where some channel changed by more than 2/255, compared as the relative
  difference. The *spatial histogram* adds up, per cell of a 16 × 9 grid, each pixel's largest
  channel change. The *colour histogram* counts covered pixels over a 4 × 4 × 4 RGB grid. Both
  histograms are normalised and compared by total-variation distance (half the L1 distance).
- **Population.** hanabi spawns on the CPU (`EffectSpawner::spawn_count`, deterministic at a fixed
  timestep), and its init pass runs before the update pass in the same frame. A particle spawned
  on frame `s` therefore has age `(k − s + 1)·dt` after frame `k` and is alive while that is below
  its lifetime (`particles::population`). The report has every emitter's spawned and alive counts
  and its mean and maximum age per frame. Reading hanabi's GPU counter back is not needed: the spawn counts and
  fixed lifetimes determine it.
- **Thresholds per scenario, from calibration.** `oracle --calibrate` runs each particle
  scenario on Bevy against itself: a rerun, eight other seeds and gravity × 1.25. The
  statistical thresholds are about 1.5× the largest distance any other seed reaches, and must
  still reject the gravity change. For `particles_fountain` at 2.0 s, seed noise reaches
  coverage 0.107, spatial 0.104 and colour 0.083, so the gate is 0.16 / 0.15 / 0.12. Gravity ×
  1.25 scores spatial 0.203, which fails. The coverage and colour distances stay within seed
  noise under that change; the spatial histogram catches it. In the exact tier, every other seed
  scores FLIP mean ≥ 0.023 and p99 ≥ 0.92, so the FLIP gate rejects any difference in the random
  stream. The numbers are the same on the RTX 3060 Ti and on lavapipe to three decimals.
  `particles_100k` (#14): seed noise at 3.0 s reaches 0.0041 / 0.0146 / 0.0119, so the gate is
  0.006 / 0.022 / 0.018; gravity × 1.25 scores 0.072 / 0.173 / 0.134. `particles_alpha`: seed
  noise over both captures reaches 0.042 / 0.046 / 0.064, so the gate is 0.065 / 0.07 / 0.095;
  buoyancy × 1.25 scores coverage 0.087 at 1.0 s and 0.139 / spatial 0.087 at 3.0 s.
  `particles_textured` (#15): seed noise at 2.5 s reaches 0.061 / 0.068 / 0.074, so the gate is
  0.092 / 0.10 / 0.11. Gravity × 1.25 scores 0.028 / 0.064 / 0.022 there, inside seed noise (the
  sparks' arcs move by a few pixels), so here the exact capture is what rejects it (FLIP mean
  0.0155, p99 0.53; other seeds ≥ 0.019 / 0.57). `smoke_and_sparks` (#15): seed noise over both
  captures reaches 0.061 / 0.090 / 0.061, so the gate is 0.09 / 0.135 / 0.09; gravity and
  buoyancy × 1.25 score spatial 0.160 at 1.0 s.
- **Engine side.** `render_engine::particles` replays hanabi step by step. On the CPU
  (`World::step_particles`): `EffectSpawner::tick` for the spawn count, and the effect seed set
  to `prng_seed` when the emitter is added and advanced by `StdRng::seed_from_u64(seed).random()`
  after every step, active or not. On the GPU (#14, `particle_sim.wgsl`, one compute pass per
  step for all emitters, which share one slot pool like hanabi's slabs): `init`, one thread per
  spawn, takes the slot in dead-stack row `alive + i` (`dead_index` starts as `0, 1, 2, …`),
  then `pcg_hash(slot ^ seed)`, `frand`/`frand3`/`frand4` and the init modifiers in their listed
  order; `prepare` caps the alive counts at capacity and writes the update dispatch and the draw
  arguments; `update`, one thread per alive particle, ages, reaps, does `v += a·dt` and
  `p += v·dt` (the particles just spawned included), returns a dead slot at the decremented
  alive count through an atomic and appends a survivor to the next alive list through another,
  writing its quad (size and colour over life, the flipbook's sprite index, the velocity) for
  the render pass. Each emitter is one
  indirect draw. Like hanabi, which slot a recycled spawn takes then depends on the atomics.
  hanabi compiles an effect, and sets its seed, on app update 1, so the Bevy side reports the
  update on which its spawners started (`first_spawn_update`). The engine side adds its emitters
  inactive and steps them `first_spawn_update − 1` times before activating them, so both seed
  chains are at the same link on the first spawn. With the chain off by one step,
  `particles_fountain@t0500` scores FLIP mean 0.024 and p99 0.94, so the gate checks the chain.
  Particles are drawn in the main pass after the blended instances, as hanabi's quads (the
  unit quad, 4× MSAA, depth tested but not written), all emitters back to front. The render
  modifiers follow hanabi's generated code: `OrientModifier` (`ParallelCameraDepthPlane`, or
  `AlongVelocity`: x along the velocity, y = view ray × x), the size over life as an (x, y)
  pair, `FlipbookModifier`'s uv offset (`(ij + uv) · 1/grid`, the scale computed on the CPU),
  then in the fragment stage `ParticleTextureModifier` (its three sample mappings, the
  texture's own sampler) and `RoundModifier`'s mask. hanabi leaves the sprite index to the
  effect; render-engine's flipbook runs once over the particle's life, `i32(age / lifetime ·
  frames)` after the age step, which the oracle gives hanabi as a `SetAttributeModifier
  (SPRITE_INDEX)` in the update pass.
- **Lighting, fog and soft particles: out of scope** (#15). At the pinned `d277555d4a`,
  hanabi's render pass (`vfx_render.wesl`) binds only the view, the simulation parameters, the
  particle buffers and the effect's own material textures. It reads no light, shadow,
  environment or fog data, and no built-in modifier shades a particle: `RenderContext::
  needs_normal` exists ("for lighting effects") but nothing sets it, and the README's feature list
  has no lighting entry. Bevy's `DistanceFog` lives in `bevy_pbr`'s mesh shaders, which hanabi
  does not use, so particles are not fogged, and Bevy's exposure (applied in those shaders) does
  not scale them either. There is no soft-particle (depth fade) modifier: the pass cannot reach
  the view's depth or prepass textures (a material slot takes an asset `Image`, and the
  fragment stage has no frag-coord expression), so particles meet geometry with the hard edge of
  the depth test (`smoke_and_sparks` shows it where the smoke meets the ground and the box).
  Collision with the depth buffer is unchecked on the README's roadmap. render-engine therefore
  renders particles unlit, unfogged and without depth fade, like its oracle; the lit set around
  `particles_textured` and `smoke_and_sparks` (sun with a shadow cascade, ambient light, a point
  light) lights the geometry only. Lit or soft particles would need another oracle or a hanabi
  version that has them; they are not built without one.
- **Recycling in practice.** On the RTX 3060 Ti and on lavapipe, both sides' atomics happen to
  recycle slots in the same order for `particles_fountain` and `particles_100k`, so their late
  captures also match nearly exactly (FLIP mean ≤ 0.0002, statistics 0.000). The statistical
  tier still gates them, since other GPUs or scenarios need not schedule the same way.
- **GPU time** of a particle scenario covers simulation plus rendering on both sides, in steady
  state after the last capture. render-engine: each timed frame steps the world (its compute
  pass is timed like any other pass) and renders. Bevy: the frames after the last capture (300
  locally), the median of the top-level render diagnostic spans plus the median of hanabi's
  compute work. hanabi writes no timestamps of its own and records all its compute passes in one
  system of Bevy's `RenderGraph` schedule (`bevy_hanabi::render::simulate`), so the oracle adds
  a system right before and one right after it that each write a timestamp into their command
  encoder (`bevy_side::HanabiTimer`, found by system name, which needs Bevy's `debug`
  feature); Bevy submits the encoders in system order. Rounds alternate as for lighting, each
  side keeping its fastest, and the local gate applies as for lighting.

## Standing rules

- **Experimental API:** pick the cleanest API for the consumers; no deprecation windows, shims or
  back-compat layers.
- **Lean harness:** no attestation, fingerprints or evidence packaging; the oracle prints a table,
  writes the PNGs, the FLIP heat map and `report.json`.
- **Free sources only:** HDRIs and textures used by scenarios are CC0 (e.g. Poly Haven).
