# Issue drafts for moritzbrantner/render-engine

The repository did not exist when the bootstrap was finished (2026-10-31), so these are drafts.
File them in this order once the repository exists. `Blocked by` uses the draft numbers (D1…);
turn them into native `blockedBy` links when filing. Every parity issue follows the gate in
`docs/technical-decisions.md` D3: mean FLIP ≤ 0.005, p99 FLIP ≤ 0.05 (CI on lavapipe and local),
median GPU frame time ≤ 1.10× Bevy (local GPU), with the scenario added to `oracle/`.

Order: lighting (D1–D6), consumer API and mmorpg native integration (D7–D8), web path (D9–D10),
particles (D11–D14). D15 is housekeeping and can go any time.

---

## D1 — Lighting: cascaded sun shadows

Blocked by: none

**Outcome.** The sun casts stable shadows over a gameplay-sized view: Bevy's default cascade
layout (4 cascades, first far bound 10 m, maximum 150 m, 20% overlap with blending) and Bevy's
default `ShadowFilteringMethod::Gaussian` (Castaño 13-tap) next to the existing Hardware2x2 path.

**Scenario.** `sun_cascades`: a 200 m ground with rows of boxes/pillars receding from a low
camera, sun at ~35° elevation so shadows cross cascade boundaries.

**Notes.** Port `build_directional_light_cascades`/`calculate_cascade` per cascade (already done
for one), a depth texture array, per-cascade shadow passes, cascade selection and blending in the
fragment shader. `SunShadows` grows a cascade config (`num_cascades`, `first_cascade_far_bound`,
`overlap_proportion`) and a filter enum. Time both sides; Bevy's shadow cost grows with
cascades too.

---

## D2 — Lighting: point and spot lights with shadows

Blocked by: none (after D1 in order)

**Outcome.** Point and spot lights (Bevy `PointLight`, `SpotLight`: intensity in lumens, range,
radius, inner/outer angles) with shadows (cube maps for point lights, a perspective map for spot
lights), for torches, lanterns and interiors.

**Scenario.** `point_spot`: a dusk courtyard — low ambient, 3 point lights (one shadowed) and
2 spot lights (one shadowed) over PBR boxes.

**Notes.** Bevy uses clustered forward shading. Start with a plain forward loop over a bounded
light list (consumer scenes have tens of lights) and only add clustering if the GPU-time gate
asks for it. Match Bevy's `getDistanceAttenuation`, the spot falloff and the point/spot shadow
biases.

---

## D3 — Lighting: HDR target and tone mapping

Blocked by: none

**Outcome.** An HDR main pass (`Rgba16Float`) followed by a tone-mapping pass, with Bevy's
`TonyMcMapface` (the Bevy default, 3D LUT) and `AgX` besides the current in-shader ACES fitted,
plus exposure from EV100. This is the base for sky, IBL and the day/night range mmorpg#529 asks
for ("day, dusk, night and dawn are all readable").

**Scenario.** `tonemap_range`: the `sun_shadows` scene at noon exposure and at a dim exposure,
each with TonyMcMapface and AgX, on an HDR Bevy camera (`Hdr`).

**Notes.** Bevy's TonyMcMapface LUT ships in `bevy_core_pipeline` (`tonemapping_luts` feature,
KTX2, from h3r2tic/tony-mc-mapface); confirm its licence, then load the same LUT so the gate
compares the operator, not a re-derived curve. Keep `DebandDither` off on both sides (or port it).

---

## D4 — Lighting: ambient / image-based lighting

Blocked by: D3

**Outcome.** Image-based ambient light so PBR materials read correctly: Bevy
`EnvironmentMapLight` (prefiltered diffuse + specular cube maps) from a CC0 HDRI, with intensity
in cd/m², alongside the existing uniform `GlobalAmbientLight`.

**Scenario.** `ibl_materials`: a row of spheres/boxes sweeping roughness × metallic under a CC0
Poly Haven outdoor HDRI (e.g. a 2k overcast sky), no sun; a second view with sun + IBL.

**Notes.** Use Bevy's own filtering path (`GeneratedEnvironmentMapLight` or prefiltered KTX2
maps produced once and committed/fetched from a CC0 source) so both sides sample identical
cube maps; render-engine owns the sampling and BRDF integration (split sum with the same F_AB /
DFG approach as Bevy). Add a `Sphere` shape to the scene API.

---

## D5 — Lighting: time-of-day sky

Blocked by: D3, D4

**Outcome.** A physically based sky whose sun direction, colour and intensity follow a
time-of-day input, with a matching ambient probe, for Greyhaven's day clock (mmorpg#529): noon,
golden hour, dusk, night (moon as a second dim directional light).

**Scenario.** `sky_day_cycle`: the same ground + boxes at four sun elevations (60°, 15°, 2°,
-10° with moon), each compared with Bevy's `Atmosphere` (Hillaire LUTs) +
`AtmosphereEnvironmentMapLight`.

**Notes.** Port Bevy's atmosphere LUT passes (transmittance, multiscattering, sky-view,
aerial-view) or a subset that passes the gate; sun colour comes from transmittance, as in Bevy.
The engine takes sun/moon directions as input; the game owns the clock.

---

## D6 — Lighting: fog

Blocked by: D3 (D5 for atmospheric fog colouring)

**Outcome.** Bevy `DistanceFog` (linear, exponential, exponential-squared, atmospheric with
extinction/inscattering colours and directional sun glow) for depth and mood in Greyhaven.

**Scenario.** `fog_modes`: a long receding row of pillars under each fog falloff, with and
without sun glow.

**Notes.** Volumetric fog (`VolumetricFog`) is out of scope until a consumer asks for it.

---

## D7 — Consumer API: meshes, materials and rendering into the consumer's frame

Blocked by: none

**Outcome.** The API the mmorpg native client needs to hand its world to render-engine:
arbitrary indexed meshes (position/normal/uv/tangent), instancing, `StandardMaterial` textures
(base colour, normal, metallic-roughness, occlusion, emissive), alpha mask/blend, and
`Renderer::encode` into a consumer-owned device, encoder, colour and depth target (no internal
MSAA/target ownership when the consumer provides them). Persistent GPU resources with
incremental updates instead of re-uploading the scene each frame.

**Scenario.** `textured_props`: a few CC0 glTF props (Kenney/Quaternius/Poly Haven) loaded on the
Bevy side via `bevy_gltf` and on the engine side via the same glTF data.

**Notes.** Shape the API after how `crates/mmorpg-client` and the `three-d-*` crates hold
geometry today; no ECS in the engine.

---

## D8 — mmorpg native: render Greyhaven lighting with render-engine

Blocked by: D1, D4, D5, D7 (cross-repo: delivers the native half of moenarch-games/mmorpg#529)

**Outcome.** The mmorpg native client (`crates/mmorpg-client`, wgpu 30) renders Greyhaven with
render-engine's sun/moon, cascaded shadows, IBL/ambient, sky and tone mapping, driven by the
existing world clock, within the client's current frame budget. render-engine is a git
dependency pinned by rev, like the 3d-lab crates.

**Notes.** Implementation PR lives in mmorpg (one issue there, linked to mmorpg#529); render-engine
issues only for API gaps found during integration. Default tuning numbers (sun illuminance per
phase, EV100, ambient/IBL intensity, cascade distances) are picked and listed in the mmorpg PR.
Native/web parity follows the decision in D9.

---

## D9 — `needs-decision`: web path — WASM+WebGPU render-engine or a three.js port

Blocked by: none (label `needs-decision`)

### Owner questions

The mmorpg web client uses three.js 0.186. How does it get the same lighting as native?

- **A. Compile render-engine to WASM + WebGPU** and render the web client's world with it
  (replacing three.js rendering, or running it beside three.js on a shared canvas).
- **B. Port the lighting to three.js**, using three.js built-ins where they pass
  (`MeshStandardMaterial`/`MeshPhysicalMaterial`, CSM-style cascades, PMREM for IBL, `Sky`,
  fog, `ACESFilmic`/`AgX` tone mapping) and custom TSL/shader chunks where they don't, and gate
  it against the **same oracle scenarios**: the oracle gains a third side that renders each
  scenario in headless Chromium and compares it to Bevy with the same FLIP thresholds (GPU time
  advisory).

**Recommendation: B (three.js port, gated by the same scenarios).** The web client's scene
graph, materials, loaders and gameplay rendering are three.js; A means replacing that renderer
or compositing two renderers with shared depth, which is a rewrite of the web client rather
than a lighting feature. B keeps WebGL2 as a fallback (WebGPU is still not available
everywhere, and render-engine's WGSL/compute paths have no WebGL2 path), reuses off-the-shelf
three.js features, and the oracle keeps both ports honest: a scenario counts as done on web only
when the three.js side passes the same FLIP gate. Cost: lighting maths exists twice (Rust/WGSL
and three.js), contained by the shared scenarios. A stays available later for particles if
three.js GPU particles cannot meet the gate.

---

## D10 — Web path: implement the decision from D9

Blocked by: D9 (scenarios D1–D6 for coverage)

**Outcome.** Per the D9 answer. With the recommended option: a `web/` three.js lighting module
consumed by the mmorpg web client, plus an oracle `--side three` that renders every lighting
scenario in headless Chromium (Playwright, SwiftShader/WebGL2 on CI) and gates it against Bevy
with the D3 thresholds.

---

## D11 — Particles: bevy_hanabi oracle

Blocked by: D8 (lighting track first)

**Outcome.** The oracle can run particle scenarios: a `bevy_hanabi` 0.19 side (Bevy 0.20),
fixed timestep, fixed seed, screenshot at fixed simulation times, GPU time per frame (simulation
+ render), and render-engine's side. Gates per D3 on the frames, plus particle-count/age
statistics read back from both sides.

**Notes.** Hanabi's GPU RNG must be reproducible for image parity: port its PCG hash and seeding
so identical seeds give identical spawns, or gate stochastic scenarios on statistics + FLIP of
deterministic ones. Decide in the PR and record it in technical decisions.

---

## D12 — Particles: emitters

Blocked by: D11

**Outcome.** Spawners (rate, burst, once), spawn shapes (point, sphere, circle, cone),
initial velocity/size/colour/lifetime, size and colour over lifetime gradients — the hanabi
modifiers the games use for torches, spell impacts, footstep dust and ambient motes.

**Scenario.** `emitters`: one emitter per shape/spawner at fixed times.

---

## D13 — Particles: GPU simulation

Blocked by: D12

**Outcome.** Compute-shader simulation with indirect dispatch/draw: gravity, linear drag,
accelerations, kill by age/AABB, capacity ≥ 100k particles per effect, many effects per frame,
GPU time within the gate against hanabi.

**Scenario.** `particles_100k`: 100k-particle fountain plus 50 small effects.

---

## D14 — Particles: lit and soft particles

Blocked by: D13, D4

**Outcome.** Particles lit by the sun/ambient/IBL from the lighting track, soft particles (depth
fade against the opaque depth), alpha-blended and additive modes, camera-facing and
velocity-aligned billboards, flipbook textures (CC0).

**Scenario.** `smoke_and_sparks`: lit smoke intersecting the ground and boxes, additive sparks.

---

## D15 — Pages: publish the oracle report

Blocked by: none

**Outcome.** A Pages workflow (reusable-workflows `deploy-pages`, as in physics-engine) that
publishes `pages-dist/` (site + last `main` oracle run from the `oracle-report` artifact) so the
Bevy/engine/FLIP images are viewable without downloading artifacts.
