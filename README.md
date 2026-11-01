# render-engine

Lighting and particles for the owner's games, built on wgpu and measured against
[Bevy](https://bevy.org).

The games' native client renders with wgpu 30 and the `three-d-*` crates; its web client uses
three.js. render-engine provides the lighting those clients need — sun and shadows first, then
point/spot lights, ambient/IBL, sky and fog — and particle emitters, through its own small API,
and checks every capability against Bevy: same scene, perceptual image difference (NVIDIA FLIP)
and GPU frame time. See [technical decisions](docs/technical-decisions.md).

## Shape of the engine

- **Consumer-owned frame:** the game keeps its wgpu device, window and UI. Meshes (glTF
  attributes), textures and `StandardMaterial`s are uploaded once into a `World` under stable
  handles. Each frame the game passes a list of instances (mesh, material, transform), and the
  renderer encodes into the game's encoder and colour target. See
  [consumer integration](docs/consumer-integration.md).
- **Scene in physical units:** `render_engine::scene` describes the camera, sun, point and spot
  lights, ambient light, environment map and fog in Bevy's units (lux, lumens, cd/m², EV100,
  linear colour), so each scenario has an exact Bevy counterpart.
- **wgpu renderer:** cascaded sun shadows (Bevy's cascade fitting and defaults: up to 4
  cascades with blended overlap, Gaussian 13-tap or hardware 2×2 PCF filtering) and a forward
  PBR pass (Filament/Bevy BRDF, 4× MSAA) into an `Rgba16Float` HDR target, then a tone-mapping
  pass with Bevy's operators and LUTs (Tony McMapface by default, AgX, ACES fitted). Image-based
  ambient light from prefiltered cube maps and Bevy's distance fog (linear, exponential,
  exponential squared, atmospheric, with shadowed sun inscattering). Point and spot lights
  (Bevy's range window, sphere-light specular and cone falloff) in a plain forward loop, with
  cube-map shadows for point lights and perspective maps for spot lights, filtered like the
  sun's or with soft shadows (PCSS). Bevy's frustum culling for the camera and every shadow
  view, so each pass only draws the instances it can see. GPU skinning like Bevy's
  `SkinnedMesh` (a joint palette per instance, four influences per vertex), and per-frame
  position/normal updates for meshes posed on the CPU. Same wgpu major as the mmorpg native
  client.
- **Bevy as oracle:** `oracle/` renders every parity scenario on Bevy (`bevy_pbr`) and on
  render-engine, headless. It fails when mean FLIP > 0.005 or p99 FLIP > 0.05, or — on a GPU —
  when render-engine's GPU frame time (median of its fastest timing round) is more than 10%
  slower than Bevy's.
- **Particles:** `render_engine::particles` emitters (burst and rate spawners; position,
  velocity, lifetime, size and colour init modifiers; acceleration; size and colour over life;
  textures and flipbooks; camera-facing or velocity-aligned quads; additive or alpha blending)
  live in the `World`, are stepped by the game and drawn in the main pass, depth tested against
  the scene. The simulation runs in compute shaders (100k particles in `particles_100k`) and
  replays `bevy_hanabi`'s spawner, seed chain, per-particle random stream and slot allocation.
  Like hanabi's, particles are unlit, unfogged and have no depth fade (soft particles).
- **Particle oracle:** particle scenarios run on Bevy with `bevy_hanabi` and on render-engine at a
  fixed timestep and seed. Captures before any particle dies need an exact image match (FLIP).
  Later captures need matching coverage, spatial and colour statistics, and spawn and alive
  counts must match on every frame
  ([D9](docs/technical-decisions.md#d9--particle-parity-seeded-replay-then-statistics)).
- **CI without a GPU:** CI renders both sides on lavapipe (software Vulkan); the image gate is
  enforced there, GPU time is advisory. Locally on a GPU both gates apply.
- **Report site:** <https://moritzbrantner.github.io/render-engine/>, published from CI on
  `main`. `site/` renders the oracle's own `report.json`: an overview by feature with the FLIP
  numbers of the CI run, and per scenario render-engine | Bevy | FLIP side by side, swipe and
  toggle. The GPU-time table comes from the latest committed local run
  (`bash scripts/record-local-gpu.sh` on the RTX 3060 Ti writes `docs/oracle-local-gpu.json`).
  Preview locally with `bash scripts/build-pages.sh` and any static server on `pages-dist/`.

## Example

The consumer owns the device and the frame: content goes into a `World` once, and every frame
is a `Scene` (camera and lights) plus a list of instances, encoded into the consumer's target
(see [consumer integration](docs/consumer-integration.md)). `Headless` runs the same path
offscreen:

```rust,no_run
use glam::{Mat4, Vec2, Vec3, Vec4};
use render_engine::{headless::Headless, material::*, mesh::*, renderer::Viewport, scene::*, world::*};

let gpu = Headless::new()?;
let mut world = World::new(gpu.renderer());
let ground = world.add_mesh(&MeshData::shape(Shape::Plane(Vec2::splat(20.0))));
let cube = world.add_mesh(&MeshData::shape(Shape::Cuboid(Vec3::ONE)));
let white = world.add_material(&StandardMaterial::default());

let scene = Scene {
    camera: Camera { eye: Vec3::new(6.0, 5.0, 8.0), target: Vec3::ZERO, fov_y: 0.785, near: 0.1, ev100: 10.5, shadow_filter: ShadowFilter::Gaussian, tonemapping: Tonemapping::TonyMcMapface },
    clear_color: Vec3::new(0.3, 0.5, 0.8),
    sun: Some(Sun {
        direction: Vec3::new(0.75, -1.0, 0.25).normalize(),
        color: Vec3::ONE,
        illuminance: 10_000.0,
        shadows: Some(SunShadows::default()),
    }),
    point_lights: vec![],
    spot_lights: vec![],
    shadow_maps: ShadowMapSizes::default(),
    ambient: Ambient { color: Vec3::ONE, brightness: 1500.0 },
    environment: None,
    fog: Some(Fog { color: Vec4::new(0.3, 0.5, 0.8, 1.0), falloff: FogFalloff::from_visibility(60.0), ..Fog::default() }),
};
let instances = [
    Instance { mesh: ground, material: white, world_from_local: Mat4::IDENTITY },
    Instance { mesh: cube, material: white, world_from_local: Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0)) },
];
let mut viewport = Viewport::new(gpu.renderer(), 1280, 720);
let frame = gpu.render(&mut viewport, &world, &scene, &instances)?;
# Ok::<(), render_engine::headless::Error>(())
```

## Current parity

| Scenario | Covers | FLIP mean / p99 | GPU ms (engine / Bevy), RTX 3060 Ti | lavapipe ms (advisory) |
| --- | --- | --- | --- | --- |
| `sun_shadows` | ground, PBR boxes (dielectric, metallic), sun + one-cascade 2×2 PCF shadow, ambient | 0.0000 / 0.0000 | 0.069 / 0.196 (0.35×) | 10.7 / 15.2 |
| `sun_cascades` | 200 m corridor, 4 cascades (Bevy defaults) with overlap blending, Gaussian filter | 0.0001 / 0.0023 | 0.183 / 0.263 (0.70×) | 35.7 / 18.0 |
| `fog_modes` (8 views) | 180 m double row of pillars towards a low sun; linear, exponential, exponential squared and atmospheric fog, each without and with shadowed sun inscattering | ≤ 0.0000 / 0.0007 | 0.16–0.25 / 0.25–0.30 (≤ 0.89×) | 22–30 / 16–33 |
| `point_spot` (3 views) | dusk courtyard: 3 point lights (one with cube shadows, one whose range ends in view), 2 spot lights (one shadowed from above, one pointing up a wall); Gaussian, 2×2 PCF on 512/256 maps, and soft shadows (PCSS) close up | ≤ 0.0000 / 0.0009 | 0.23–0.42 / 0.74–0.92 (≤ 0.45×) | 23–48 / 26–53 |
| `textured_props` | CC0 glTF props through the consumer API: base-colour, normal and metallic-roughness textures, vertex colours, alpha-masked double-sided bushes (one mirrored) with cut-out shadows, a blended sphere, one crate mesh drawn twice; cascaded sun + IBL | 0.0000 / 0.0007 | 0.323 / 0.573 (0.56×) | 76.4 / 82.2 |
| `skinned_figure` | CC0 rigged character (7281 vertices, 65 joints, normal-mapped) skinned on the GPU in two poses, one figure mirrored, both casting cascaded sun shadows; IBL | 0.0000 / 0.0002 | 0.191 / 0.425 (0.45×) | 115 / 173 |

## Web: three.js

The mmorpg web client (three.js 0.186) gets the same lighting from
[`packages/three-lighting`](packages/three-lighting) (`@moritzbrantner/three-lighting`, pinned as
`github:moritzbrantner/render-engine#<commit>`; see [consumer integration](docs/consumer-integration.md#web-client-threejs)
and [D8](docs/technical-decisions.md#d8--web-path-a-threejs-port-gated-by-the-same-oracle)). The
oracle renders each lighting scenario it covers in headless Chromium and holds it to the same
FLIP gate against Bevy. CI run on lavapipe (ANGLE on Vulkan), FLIP mean / p99 against Bevy:

| Scenario | three.js vs Bevy | Result |
| --- | --- | --- |
| `sun_shadows` | 0.0003 / 0.0083 | pass |
| `sun_cascades` | 0.0003 / 0.0070 | pass |
| `tonemap_range` (4 views) | ≤ 0.0004 / 0.0107 | pass |
| `ibl_materials`, `ibl_materials_sun` | ≤ 0.0010 / 0.0304 | pass |
| `fog_modes` (8 views) | ≤ 0.0001 / 0.0026 | pass |
| `point_spot` (3 views: Gaussian, 2×2 PCF, PCSS) | ≤ 0.0006 / 0.0153 | pass |
| `many_instances` | 0.0058 / 0.0869 | known gap, #42 |
| `textured_props` | 0.0016 / 0.0438 | pass |
| `skinned_figure` | 0.0009 / 0.0224 | pass |
| `sky_day_cycle` (4 views) | – | not ported, #39 |

## Development

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
cargo run --release --locked --manifest-path oracle/Cargo.toml        # Bevy oracle, both gates on a GPU
WGPU_FORCE_FALLBACK_ADAPTER=1 cargo run --release --locked --manifest-path oracle/Cargo.toml   # as CI (lavapipe)
bash scripts/build-pages.sh                                            # site + last oracle run → pages-dist/
bun install                                                            # three.js side: three + Playwright (needs node)
```

[AGENTS.md](AGENTS.md) lists the full command set and repository rules.
