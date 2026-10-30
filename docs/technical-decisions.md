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
2. **mmorpg web client** (three.js 0.186). How the web client gets parity is an open decision
   (see the `needs-decision` web-path issue).

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
  no multiscatter compensation 0.0128; no normal bias 0.0014 / 0.0059 (passes — a sub-texel
  bias change is not a parity break).
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
- The images agree across devices: `sun_shadows` scores mean 0.0005 on both lavapipe and the
  RTX 3060 Ti.

## D5 — Pinned Bevy configuration per scenario

The Bevy side configures each scenario explicitly instead of relying on Bevy defaults, so a Bevy
upgrade that changes a default shows up as a deliberate scenario change:

- tone mapping `AcesFitted` (analytic, no LUT), `DebandDither::Disabled`, `Msaa::Sample4`, a
  non-HDR `Rgba8UnormSrgb` target;
- sun shadows: one cascade, `ShadowFilteringMethod::Hardware2x2`, `DirectionalLightShadowMap`
  2048, Bevy's depth/normal biases (0.02 m, 1.8 texels);
- `GlobalAmbientLight` for ambient; no environment map yet.

Later scenarios lift these one at a time (cascades, Gaussian/temporal filtering, IBL, TonyMcMapface
tone mapping, HDR targets, fog, sky).

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

## Standing rules

- **Experimental API:** pick the cleanest API for the consumers; no deprecation windows, shims or
  back-compat layers.
- **Lean harness:** no attestation, fingerprints or evidence packaging; the oracle prints a table,
  writes the PNGs, the FLIP heat map and `report.json`.
- **Free sources only:** HDRIs and textures used by scenarios are CC0 (e.g. Poly Haven).
