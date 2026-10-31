# render-engine

Lighting and (next) particles for the owner's games, built on wgpu and measured against
[Bevy](https://bevy.org).

The games' native client renders with wgpu 30 and the `three-d-*` crates; its web client uses
three.js. render-engine provides the lighting those clients need — sun and shadows first, then
point/spot lights, ambient/IBL, sky and fog — and later GPU particles, through its own small API,
and checks every capability against Bevy: same scene, perceptual image difference (NVIDIA FLIP)
and GPU frame time. See [technical decisions](docs/technical-decisions.md).

## Shape of the engine

- **Scene in physical units:** `render_engine::scene` describes camera, sun, point and spot
  lights, ambient light and PBR meshes in Bevy's units (lux, lumens, cd/m², EV100, linear
  colour), so each scenario has an exact Bevy counterpart.
- **wgpu renderer:** cascaded sun shadows (Bevy's cascade fitting and defaults: up to 4
  cascades with blended overlap, Gaussian 13-tap or hardware 2×2 PCF filtering) and a forward
  PBR pass (Filament/Bevy BRDF, 4× MSAA) into an `Rgba16Float` HDR target, then a tone-mapping
  pass with Bevy's operators and LUTs (Tony McMapface by default, AgX, ACES fitted). Image-based
  ambient light from prefiltered cube maps and Bevy's distance fog (linear, exponential,
  exponential squared, atmospheric, with shadowed sun inscattering). Point and spot lights
  (Bevy's range window, sphere-light specular and cone falloff) in a plain forward loop, with
  cube-map shadows for point lights and perspective maps for spot lights, filtered like the
  sun's or with soft shadows (PCSS). Same wgpu major as the mmorpg native client.
- **Bevy as oracle:** `oracle/` renders every parity scenario on Bevy (`bevy_pbr`) and on
  render-engine, headless. It fails when mean FLIP > 0.005 or p99 FLIP > 0.05, or — on a GPU —
  when render-engine's median GPU frame time is more than 10% slower than Bevy's.
- **CI without a GPU:** CI renders both sides on lavapipe (software Vulkan); the image gate is
  enforced there, GPU time is advisory. Locally on a GPU both gates apply.
- **Report page:** `site/` shows each scenario's Bevy image, render-engine image and FLIP error
  map from the last oracle run.

## Example

```rust,no_run
use glam::{Vec2, Vec3, Vec4};
use render_engine::{headless::Headless, scene::*};

let scene = Scene {
    camera: Camera { eye: Vec3::new(6.0, 5.0, 8.0), target: Vec3::ZERO, fov_y: 0.785, near: 0.1, ev100: 10.5, shadow_filter: ShadowFilter::Gaussian, tonemapping: Tonemapping::TonyMcMapface },
    clear_color: Vec3::new(0.3, 0.5, 0.8),
    sun: Some(Sun {
        direction: Vec3::new(0.75, -1.0, 0.25).normalize(),
        color: Vec3::ONE,
        illuminance: 10_000.0,
        shadows: Some(SunShadows::default()),
    }),
    ambient: Ambient { color: Vec3::ONE, brightness: 1500.0 },
    environment: None,
    fog: Some(Fog { color: Vec4::new(0.3, 0.5, 0.8, 1.0), falloff: FogFalloff::from_visibility(60.0), ..Fog::default() }),
    meshes: vec![
        MeshInstance::new(Shape::Plane(Vec2::splat(20.0)), Vec3::ZERO, Material::default()),
        MeshInstance::new(Shape::Cuboid(Vec3::ONE), Vec3::new(0.0, 0.5, 0.0), Material::default()),
    ],
};
let gpu = Headless::new()?;
let frame = gpu.render(&gpu.prepare(&scene, 1280, 720))?;
# Ok::<(), render_engine::headless::Error>(())
```

## Current parity

| Scenario | Covers | FLIP mean / p99 | GPU ms (engine / Bevy), RTX 3060 Ti | lavapipe ms (advisory) |
| --- | --- | --- | --- | --- |
| `sun_shadows` | ground, PBR boxes (dielectric, metallic), sun + one-cascade 2×2 PCF shadow, ambient | 0.0000 / 0.0000 | 0.069 / 0.196 (0.35×) | 10.7 / 15.2 |
| `sun_cascades` | 200 m corridor, 4 cascades (Bevy defaults) with overlap blending, Gaussian filter | 0.0001 / 0.0023 | 0.183 / 0.263 (0.70×) | 35.7 / 18.0 |
| `fog_modes` (8 views) | 180 m double row of pillars towards a low sun; linear, exponential, exponential squared and atmospheric fog, each without and with shadowed sun inscattering | ≤ 0.0000 / 0.0007 | 0.16–0.25 / 0.25–0.30 (≤ 0.89×) | 22–30 / 16–33 |
| `point_spot` (3 views) | dusk courtyard: 3 point lights (one with cube shadows, one whose range ends in view), 2 spot lights (one shadowed from above, one pointing up a wall); Gaussian, 2×2 PCF on 512/256 maps, and soft shadows (PCSS) close up | ≤ 0.0000 / 0.0009 | 0.23–0.42 / 0.74–0.92 (≤ 0.45×) | 23–48 / 26–53 |

## Development

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked
cargo run --release --locked --manifest-path oracle/Cargo.toml        # Bevy oracle, both gates on a GPU
WGPU_FORCE_FALLBACK_ADAPTER=1 cargo run --release --locked --manifest-path oracle/Cargo.toml   # as CI (lavapipe)
bash scripts/build-pages.sh                                            # site + last oracle run → pages-dist/
```

[AGENTS.md](AGENTS.md) lists the full command set and repository rules.
