# Integrating render-engine into a consumer

render-engine draws into a frame the consumer owns. The consumer keeps its window, wgpu device,
queue, surface, game loop, asset loading and UI. The engine keeps its own render targets: a 4×
MSAA HDR colour target, its depth buffer, shadow maps and the tone-mapping targets. It writes
the finished, tone-mapped frame into the consumer's colour view. The example below is written for
the mmorpg native client (`crates/mmorpg-client`), the first consumer.

## Objects

| Type | Lifetime | Role |
| --- | --- | --- |
| `renderer::Renderer` | one per device and output format | pipelines, samplers, tone-mapping LUTs |
| `world::World` | one per device, for as long as the content lives | meshes, textures, materials and particle emitters behind stable handles (`MeshId`, `TextureId`, `MaterialId`, `EmitterId`) |
| `renderer::Viewport` | one per output surface | the engine's targets for one size, plus per-frame buffers (instances, lights, shadow maps) that it reuses and grows |
| `scene::Scene` | rebuilt every frame (plain values) | camera, sun, moon, sky, point and spot lights, ambient light, environment map, fog |
| `world::Instance` | a list rebuilt every frame | `{ mesh, material, world_from_local }`, one per drawn copy |

## Setup

```rust
// The consumer's own device. Request DEPTH_CLIP_CONTROL where the adapter has it, so sun
// shadows use unclipped depth natively (otherwise a fragment-shader fallback is used), and
// DUAL_SOURCE_BLENDING (the sky's per-channel transmittance, else its mean). A `Scene::sky`
// needs FLOAT32_FILTERABLE (Bevy's Rgba32Float medium LUTs); `prepare` panics without it.
let features = adapter.features()
    & (wgpu::Features::DEPTH_CLIP_CONTROL
        | wgpu::Features::DUAL_SOURCE_BLENDING
        | wgpu::Features::FLOAT32_FILTERABLE);
let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
    required_features: features,
    ..Default::default()
}).await?;
let format = /* the sRGB surface format, as today */;

let renderer = Renderer::new(&device, &queue, format);
let mut world = World::new(&renderer);
let mut viewport = Viewport::new(&renderer, config.width, config.height);
```

`Renderer` and `World` keep clones of the device and queue, so later calls do not take them.

## Uploading content once

Load glTF the way the client does today (`gltf::import_slice`), but hand the engine the
attributes and textures instead of baking colours into vertices:

```rust
let mut mesh = MeshData {
    positions, normals,              // POSITION, NORMAL
    uvs,                             // TEXCOORD_0; empty if absent
    tangents,                        // TANGENT; empty if absent
    colors,                          // COLOR_0 as linear RGBA; empty if absent
    indices,
};
if has_normal_map && mesh.tangents.is_empty() {
    mesh.generate_tangents()?;       // MikkTSpace, identical to Bevy's
}
let mesh_id = world.add_mesh(&mesh);

let base = world.add_texture(&TextureData {
    width, height,
    format: wgpu::TextureFormat::Rgba8UnormSrgb,   // colour; Rgba8Unorm for normal/MR/occlusion
    levels,                                         // level 0 first; the engine never makes mips
    sampler: Sampler::default(),                    // glTF default: repeat, trilinear
});
let material = world.add_material(&StandardMaterial {
    base_color: Vec4::from(pbr.base_color_factor()),
    base_color_texture: Some(base),
    metallic_roughness_texture: Some(mr),           // G = roughness, B = metallic
    normal_map_texture: Some(normal),
    alpha_mode: AlphaMode::Mask(0.5),
    double_sided: material.double_sided(),
    ..StandardMaterial::default()
});
```

`oracle/src/content.rs` (`Assets::load_gltf`) is a complete glTF-to-engine mapping that follows
Bevy's glTF loader. Copy it as a starting point.

Handles stay valid until they are removed. In the client, the `resource_key → GpuMesh` cache in
`graphics/frame.rs` becomes `resource_key → MeshId`. Unlike the current `Weak` pointer check,
nothing is uploaded again unless the consumer asks for it:

- `world.set_mesh(id, &data)` rewrites a mesh in place when its vertex and index counts are
  unchanged. Use it for CPU-skinned meshes every frame, until GPU skinning lands.
- `world.set_material(id, &material)` changes a material's colours or textures.
- `remove_mesh`, `remove_texture` and `remove_material` free a handle. A removed texture stays
  alive for as long as a material still uses it.

The `Material` fields used today map as follows:

| Today | `StandardMaterial` |
| --- | --- |
| `color` | `base_color.xyz` |
| `opacity` | `base_color.w`; set `alpha_mode: AlphaMode::Blend` when it is below 1 |
| `emissive` | `emissive` |
| `unlit` | `unlit` |
| `double_sided` | `double_sided` |

Procedural `mmorpg-presentation` geometry keeps its per-vertex colours in `MeshData::colors`.

## Every frame

```rust
let frame = surface.get_current_texture()?;
let view = frame.texture.create_view(&Default::default());

// The game's clock (Greyhaven's day cycle) gives the hour; the engine only needs directions.
let cycle = DayCycle { noon_elevation: 60f32.to_radians(), moon_lag_hours: 12.0 };
let scene = Scene {
    camera: Camera { eye, target, fov_y, near: 0.2, ev100, shadow_filter, tonemapping },
    clear_color: Vec3::ZERO,
    sun: Some(Sun { direction: -cycle.to_sun(hour), color: Vec3::ONE, illuminance: RAW_SUNLIGHT, shadows }),
    moon: Some(Moon { direction: -cycle.to_moon(hour), color: moon_tint, illuminance: 0.3 }),
    sky: Some(Sky::default()),       // sky, aerial perspective and the sky's ambient light
    ambient: Ambient { color: Vec3::ONE, brightness: 0.0 },
    environment: None,                // the sky's own environment light replaces it
    point_lights, spot_lights, shadow_maps: ShadowMapSizes::default(), fog: None,
};
let instances: Vec<Instance> = draws.iter().map(|d| Instance {
    mesh: mesh_ids[&d.resource_key],
    material: d.material_id,
    world_from_local: d.model,          // a negative determinant (mirroring) is handled
}).collect();

renderer.prepare(&mut viewport, &world, &scene, &instances);
let mut encoder = device.create_command_encoder(&Default::default());
renderer.encode(&mut encoder, &viewport, &world, &view, None);
// The client's UI goes on top, as today: same view, LoadOp::Load, no depth.
egui_paint(&mut encoder, &view);
queue.submit([encoder.finish()]);
frame.present();
```

- **Order of draws.** `prepare` sorts the instances: opaque, then alpha-masked, then blended
  from back to front by instance origin (Bevy's phases). The consumer does not sort them.
- **Culling.** `prepare` frustum-culls every instance per view (the camera, each sun cascade,
  spot map and point-light cube face) against its mesh's bounds, which the `World` computes
  from the positions on `add_mesh`/`set_mesh`. List the whole scene; the engine drops what a
  view cannot see.
- **Resize.** Call `viewport.resize(&renderer, width, height)` together with
  `surface.configure`.
- **Units.** `Scene` uses Bevy's physical units: illuminance in lux, light power in lumens,
  ambient light in cd/m², exposure as EV100. The client's day cycle (`Environment::at`) has to
  produce these instead of today's unitless intensities.
- **Sky and day cycle.** With `Scene::sky`, the sun's illuminance is its light above the
  atmosphere (`RAW_SUNLIGHT`): the atmosphere's transmittance reddens and dims it towards the
  horizon and the disc's share above the horizon fades it out, so the client sets directions,
  not colour curves. The moon is a second, unshadowed directional light with its own disc. The
  sky also lights the scene: each frame it is filtered into diffuse and specular cube maps (Bevy
  `AtmosphereEnvironmentMapLight`), so leave `ambient` at 0 and `environment` at `None`. Only
  exposure is left to the client; the oracle's `sky_day_cycle_*` views use EV100 14.5 at noon,
  13.5 with the sun at 15°, 11 at 2° and 3.5 at night (sun 10° below the horizon).
- **Depth.** The engine's depth buffer is reverse-Z and 4× MSAA, and stays internal. Picking
  keeps working because it uses CPU rays (`PickRay`) and does not read the depth buffer.
- **Headless.** `headless::Headless::render(&mut viewport, &world, &scene, &instances)` runs the
  same `prepare` and `encode` into an offscreen target and reads the pixels back. Tests and the
  oracle use it. For GPU timing, pass `FrameTimestamps { query_set, first }` to `encode`: it
  writes a begin and an end timestamp on each pass (`viewport.timestamp_queries()` in all);
  the frame's GPU time is the sum of the spans.

## Particles

Emitters live in the `World` next to the meshes, under `EmitterId` handles. The game steps them
at its simulation rate; every `prepare` draws all alive particles as camera-facing quads in the
main pass, after the blended instances.

```rust
use render_engine::particles::*;

let sparks = world.add_emitter(&Emitter {
    capacity: 256,
    spawner: Spawner::rate(96.0),            // or Spawner::once(n), Spawner::burst(n, period)
    seed: 0x5eed_0001,
    // Applied in order; each random value draws from the particle's stream in this order.
    init: vec![
        InitModifier::PositionSphere { center: Vec3::ZERO, radius: 0.05, dimension: ShapeDimension::Volume },
        InitModifier::Velocity(Value::Uniform { min: Vec3::new(-0.4, 5.6, -0.8), max: Vec3::new(1.2, 6.4, 0.8) }),
        InitModifier::Lifetime(Value::Const(1.5)),
    ],
    accel: Vec3::new(0.0, -9.81, 0.0),
    size_over_life: Some(Gradient::new(vec![(0.0, 0.12), (1.0, 0.04)])),
    color_over_life: Some(ColorOverLife {
        gradient: Gradient::new(vec![(0.0, Vec4::new(1.5, 2.5, 4.0, 1.0)), (1.0, Vec4::new(0.1, 0.3, 0.8, 0.0))]),
        blend: ColorBlend::Overwrite,
    }),
    roundness: 1.0,                          // 0 = square quads, 1 = discs
    blend: ParticleBlend::Add,               // or ParticleBlend::Alpha (sorted back to front)
    active: true,
});
world.set_emitter_transform(sparks, torch.world_from_local);   // where new particles spawn

// In the game loop, once per simulation tick:
world.step_particles(dt);
```

- **Units.** Positions in metres, velocities in m/s, lifetimes in seconds. Colours are linear
  HDR and are written as they are, without the camera's exposure (like hanabi's).
- **Stopping.** `set_emitter(id, &Emitter { active: false, ..desc })` stops spawning and lets
  alive particles finish; `remove_emitter` drops them at once.
- **Statistics.** `world.emitter_stats(id)` has the last step's spawn count and alive count.
- **Simulation space.** Particles live in world space: moving the emitter moves where new ones
  spawn, not the ones already alive.
- **Cost.** The simulation runs on the CPU, which suits thousands of particles; GPU simulation
  for 100k is #14.

## Not covered yet

- Stars, clouds and moon phases: the sky is Bevy's clear atmosphere; the moon is a full disc.
- GPU skinning. Skin on the CPU and call `set_mesh`.
- Clustered lights. Every point and spot light is evaluated for every fragment.
- glTF features outside the core metallic-roughness model: texture transforms, a second UV set,
  clearcoat, transmission.
