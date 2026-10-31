# Technical decisions

Current direction, owner-approved 2026-10-30. Source code and the oracle prove what is
implemented; GitHub issues track remaining work.

## D1 — Bevy is the oracle and the performance baseline

render-engine provides lighting and particles for the owner's games. It is built against Bevy
the way [physics-engine](https://github.com/moritzbrantner/physics-engine) is built against
Rapier:

- **Lighting:** `bevy_pbr` (Bevy 0.20.1) is the visual reference.
- **Particles:** `bevy_hanabi` (0.19.0, GPU particles) is the reference once the particle track
  starts.

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
- GPU time is measured with timestamp queries: 200 frames per side, median. render-engine times
  its whole frame (first pass start to last pass end). Bevy's time is the sum of the top-level
  `RenderDiagnosticsPlugin` spans (`render/<span>/elapsed_gpu`: mesh preprocessing, shadow
  views, main pass, upscaling, …), i.e. Bevy's real GPU work for the same frame.
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
  sampling and the BRDF terms. Bevy's runtime `GeneratedEnvironmentMapLight` is not used;
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
  temporal shadow filter (it needs TAA) and light textures are out of scope.

Later scenarios lift these one at a time (temporal filtering, sky).

## D6 — Order: lighting first, then particles

Lighting scenarios come first because mmorpg#529 is waiting on them: cascaded shadows,
point/spot lights, ambient/IBL, time-of-day sky, fog. Then the mmorpg native integration and the
web path. The particle track (bevy_hanabi oracle, emitters, GPU simulation, lit/soft particles)
starts after that.

## D7 — Ownership boundary

render-engine owns render passes, lights, shadows, sky/fog/ambient, tone mapping and (later)
particle simulation and rendering. Games own scene content, ECS, asset loading, the game loop,
gameplay and the day clock that drives the sun. The engine exposes coarse per-frame data (scene,
lights, instances) and renders into a target the consumer provides.

## D8 — Web path: a three.js port gated by the same oracle

Owner decision 2026-10-31 (#10, option B). The web client gets parity through a **three.js
port** of the lighting, using three.js built-ins where they pass the gate and custom TSL/shader
chunks where they don't. Each web scenario is checked against the **same Bevy oracle
scenarios** through a headless Chromium side with the same FLIP thresholds (D3); GPU time is
advisory there. Option A (render-engine compiled to WASM/WebGPU) stays available later for
particles only, if three.js GPU particles cannot meet the gate. Scope of the first step: #11.

## Standing rules

- **Experimental API:** pick the cleanest API for the consumers; no deprecation windows, shims or
  back-compat layers.
- **Lean harness:** no attestation, fingerprints or evidence packaging; the oracle prints a table,
  writes the PNGs, the FLIP heat map and `report.json`.
- **Free sources only:** HDRIs and textures used by scenarios are CC0 (e.g. Poly Haven).
