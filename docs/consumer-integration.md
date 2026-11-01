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
| `world::World` | one per device, for as long as the content lives | meshes, textures, materials, joint palettes and particle emitters behind stable handles (`MeshId`, `TextureId`, `MaterialId`, `SkinId`, `EmitterId`) |
| `renderer::Viewport` | one per output surface | the engine's targets for one size, plus per-frame buffers (instances, lights, shadow maps) that it reuses and grows |
| `scene::Scene` | rebuilt every frame (plain values) | camera, sun, moon, sky, point and spot lights, ambient light, environment map, fog |
| `world::Instance` | a list rebuilt every frame | `{ mesh, material, world_from_local, skin }`, one per drawn copy |

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
    joints, weights,                 // JOINTS_0, WEIGHTS_0 for skinned meshes; else empty
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

- `world.set_mesh(id, &data)` rewrites every stream of a mesh and its indices, in place when
  the vertex and index counts are unchanged. Use it when the topology or the surface
  attributes change, not for a new pose.
- `world.set_mesh_positions_normals(id, &positions, &normals)` rewrites only a mesh's positions
  and normals (two buffer writes, nothing re-interleaved, no index rewrite). It is the
  per-frame call for meshes still posed on the CPU; see [Skinned meshes](#skinned-meshes) for
  the GPU path.
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

## Skinned meshes

The engine skins on the GPU the way Bevy does (`SkinnedMesh`): the mesh keeps its bind pose,
each instance names a joint palette, and the vertex shaders (main and shadow passes) blend
four joint matrices per vertex. A new pose uploads one 64-byte matrix per joint instead of the
mesh.

```rust
// Once per mesh: the bind pose with JOINTS_0 / WEIGHTS_0.
let body = world.add_mesh(&MeshData { joints, weights, ..bind_pose });
// Once per figure: a palette, then a new pose each frame.
let skin = world.add_skin(&palette);
world.set_skin(skin, &palette);
// Every instance of a mesh with joints names a skin; instances may share one.
Instance { mesh: body, material, world_from_local: unit_transform, skin: Some(skin) }
```

`palette[j]` is joint `j`'s transform relative to the skinned mesh times its inverse bind
matrix, `local_from_joint * inverse_bind[j]`. That is glTF's joint matrix without the
mesh node's inverse global transform, and Bevy's `joint_global * inverse_bindpose` without the
world part: `world_from_local` still places the figure, so the palette stays the same wherever
the unit stands. Culling follows the pose: each mesh keeps a bind-pose box per joint (Bevy
`SkinnedMeshBounds`), and `prepare` unions them under the palette. `oracle/src/content.rs`
(`Assets::load_skin`) computes a palette from a glTF skin; the `skinned_figure` oracle scenario
renders one against Bevy.

**Switching the mmorpg native client** (`crates/mmorpg-presentation` CPU skinning,
`graphics/frame.rs` uploads):

1. Upload each skinned unit mesh once in its bind pose with `joints` and `weights`, keyed by
   resource as other meshes are; drop the per-unit posed copies.
2. Give each moving unit a `SkinId` (`add_skin` when it spawns, `remove_skin` when it
   despawns).
3. Each frame, where the client now evaluates the animation and skins vertices, stop after the
   joint matrices: build the palette (`model_from_joint * inverse_bind`, which the CPU skinning
   loop already computes per joint) and call `set_skin`. Delete the vertex loop and the
   `set_mesh` call.
4. Put `skin: Some(id)` on the unit's instances and keep `world_from_local` as the unit's
   transform; every other instance gets `skin: None`.

Until step 3 lands, a CPU-skinned unit should call `set_mesh_positions_normals` instead of
`set_mesh`. Measured with `cargo run --release --example mesh_upload` (89 figures, 110 k
vertices, RTX 3060 Ti, including submit and the GPU taking the writes): `set_mesh` 2.48 ms
per frame (27.9 µs per mesh), `set_mesh_positions_normals` 1.01 ms (11.3 µs), `set_skin` with
65-joint palettes plus the whole `prepare` 0.39 ms (4.4 µs), of which `prepare` alone takes
0.09 ms. GPU skinning also removes the client's own CPU skinning (about 6 ms per frame in
mmorpg#588).

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
    skin: d.skin_id,                    // Some for skinned meshes, else None
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
  from the positions on `add_mesh`, `set_mesh` and `set_mesh_positions_normals` (skinned
  meshes: per joint, under the current palette). List the whole scene; the engine drops what a
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

Emitters live in the `World` next to the meshes, under `EmitterId` handles; their particles live
on the GPU. The game steps them at its simulation rate by recording compute passes into its
encoder; every frame then draws all alive particles as quads in the main pass, after the
blended instances.

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
    // (x, y) in metres; with `Orient::AlongVelocity`, x is the streak's length.
    size_over_life: Some(Gradient::new(vec![(0.0, Vec2::new(0.24, 0.03)), (1.0, Vec2::new(0.08, 0.02))])),
    color_over_life: Some(ColorOverLife {
        gradient: Gradient::new(vec![(0.0, Vec4::new(1.5, 2.5, 4.0, 1.0)), (1.0, Vec4::new(0.1, 0.3, 0.8, 0.0))]),
        blend: ColorBlend::Overwrite,
    }),
    orient: Orient::AlongVelocity,           // or Orient::CameraPlane
    // A texture in the same world, optionally a flipbook run once over each particle's life.
    texture: Some(ParticleTexture {
        texture: spark_texture,
        mapping: TextureMapping::Modulate,
        flipbook: None,                      // Some(Flipbook { grid: UVec2::new(2, 2), frames: 4 })
    }),
    roundness: 0.0,                          // 0 = square quads, 1 = discs
    blend: ParticleBlend::Add,               // or ParticleBlend::Alpha (emitters back to front)
    active: true,
});
world.set_emitter_transform(sparks, torch.world_from_local);   // where new particles spawn

// In the game loop, once per simulation tick, into the frame's encoder before `encode`:
world.step_particles(&mut encoder, dt, None);
```

- **Units.** Positions in metres, velocities in m/s, lifetimes in seconds. Colours are linear
  HDR and are written as they are, without the camera's exposure (like hanabi's).
- **Stopping.** `set_emitter(id, &Emitter { active: false, ..desc })` stops spawning and lets
  alive particles finish; `remove_emitter` drops them at once.
- **Stepping.** `step_particles` ticks the spawners on the CPU and records one compute pass
  (spawn and init, then age, reap and move) for all emitters at once. Several steps can go into
  one encoder; a frame rendered without a step shows the last step's particles again. Pass
  `FrameTimestamps` to time the pass (two queries).
- **Statistics.** `world.spawn_count(id)` is the last step's spawn count (CPU).
  `world.read_alive_particles()` reads every emitter's alive count back from the GPU and waits,
  for tests and tools.
- **Limits.** At most 8 init modifiers per emitter and 8 keys per gradient.
- **Simulation space.** Particles live in world space: moving the emitter moves where new ones
  spawn, not the ones already alive.
- **Draw order.** Emitters are drawn back to front by their transform's position, additive and
  alpha-blended alike (as Bevy sorts hanabi effects); inside an emitter the order is the GPU's
  alive-list order, as in hanabi.
- **Unlit, unfogged, hard edges.** Particles are not lit by the scene's lights, fog does not
  apply to them, and they are cut hard where they meet geometry (depth tested, no depth fade):
  bevy_hanabi renders them this way, and it is the oracle (technical decisions D9). Tint an
  emitter's colours yourself for the time of day.
- **Cost.** `particles_100k` (about 100k alive in one fountain plus 50 small effects) runs on
  the GPU; see the oracle report for its timings against hanabi.

## Web client (three.js)

The web client gets the same lighting through `@moritzbrantner/three-lighting`
(`packages/three-lighting`, technical decisions D8), a three.js 0.186 `WebGLRenderer` port of the
shaders above, gated against Bevy by the oracle's three.js side. Pin it like the other
`github:` dependencies:

```json
"@moritzbrantner/three-lighting": "github:moritzbrantner/render-engine#<commit>"
```

The consumer keeps its renderer, scene and camera; meshes use `LitMaterial`, and
`Lighting.render` replaces `renderer.render`:

```js
import { FogFalloff, Lighting, LitMaterial, loadCubemap, loadToneMappingLuts, POINT_LIGHT_SHADOWS, singleCascade } from '@moritzbrantner/three-lighting';

const lighting = new Lighting(await loadToneMappingLuts());
lighting.ev100 = 10.5;
lighting.tonemapping = 'tonyMcMapface';              // or 'agx', 'acesFitted'
lighting.clearColor = [0.3, 0.48, 0.75];             // linear, tone mapped, not exposed
lighting.sun = { direction: [0.6, -0.8, 0.2], color: [1, 0.95, 0.88], illuminance: 10_000, shadows: singleCascade(30) };
lighting.ambient = { color: [0.75, 0.85, 1], brightness: 1500 };
lighting.environment = { diffuse: await loadCubemap(diffuseUrl), specular: await loadCubemap(specularUrl), intensity: 2000 };
lighting.fog = { color: [0.4, 0.48, 0.58, 1], falloff: FogFalloff.fromVisibility(120) };
lighting.pointLights = [{ position: [0, 2.2, 0], color: [1, 0.82, 0.6], intensity: 8000, range: 12, radius: 0.08, shadows: POINT_LIGHT_SHADOWS }];
lighting.spotLights = [{ position: [4.5, 5, -3.5], direction: [-5, -5, 3.7], color: [1, 0.95, 0.85], intensity: 60_000, range: 15, radius: 0.1,
  innerAngle: 0.35, outerAngle: 0.56, shadows: null }];   // SPOT_LIGHT_SHADOWS, or { ...SPOT_LIGHT_SHADOWS, soft: true }

scene.add(new Mesh(geometry, new LitMaterial({ baseColor: [0.7, 0.08, 0.06], perceptualRoughness: 0.45 })));
lighting.render(renderer, scene, camera);           // or (renderer, scene, camera, renderTarget)
```

Values are Bevy's units, as in `Scene`. `Lighting.render` draws the sun's cascades (Bevy's
fitting, 2048² maps, Gaussian or 2×2 filtering), the spot maps (beside the cascades, at their
size) and point-light cube maps (1024² faces) with the same filters or soft shadows (PCSS), up to
four shadowed lights of each kind, a 4× MSAA `HalfFloat` HDR pass, the tone
mapping and the sRGB output. Environment maps are the same prefiltered KTX2 cube maps the
native client loads (`assets/ibl`); the tone-mapping LUTs ship with the package. The port
covers what the oracle's three.js side passes; the report lists the rest with its issues.

The HDR pass renders upside down, as wgpu stores a frame, so its MSAA samples match Bevy's; the
output pass flips it back and the canvas or target comes out upright. Other three.js materials
in the scene still render correctly, since the projection and the winding flip together. The
one visible effect: a custom shader in the scene that reads `gl_FragCoord.y` sees it mirrored,
counting down from the top row.

### Textured glTF materials (web)

`LitMaterial` takes the same textured `StandardMaterial` as the engine: base-colour, emissive,
metallic-roughness, occlusion and normal-map textures (three `Texture`s sampling the mesh's `uv`),
the mesh's `color` attribute times the base colour, `alphaMode` `'opaque'`, `'mask'` (with
`alphaCutoff`) or `'blend'`, `doubleSided` and `unlit`. `Lighting.render` draws opaque, then
masked, then blended meshes (back to front by object origin, like Bevy), and masked and blended
materials cut their shadows out. For meshes from three's `GLTFLoader`, convert each material:

```js
gltf.scene.traverse((o) => {
  if (o.isMesh) o.material = LitMaterial.fromGltf(o.material);
});
```

`fromGltf` maps `MeshStandardMaterial` (glTF metallic-roughness) and `MeshBasicMaterial`
(`KHR_materials_unlit`) the way Bevy's glTF loader does; the loader's textures keep their glTF
samplers, and KTX2 textures keep their own mip chains. Normal maps need MikkTSpace tangents: use
the file's `TANGENT`, or generate them as the engine does (`computeMikkTSpaceTangents` from
`three/addons/utils/BufferGeometryUtils.js`). `textureFromData` uploads RGBA8 texels with an
explicit mip chain and sampler, as the engine's `TextureData` does. Not mapped: `normalTexture.scale`,
texture transforms, a second UV set (occlusion on `TEXCOORD_1`) and the physical extensions,
which the engine does not cover either.

### Instanced meshes, other materials and a page background (web)

`LitMaterial` draws three's `InstancedMesh` like the engine's instances: each instance takes its
`instanceMatrix`, and its `setColorAt` colour (linear) multiplies the base colour, in the main
pass and every shadow pass. three draws plain meshes one draw call each, so keep repeated props
instanced. Each shadow view (sun cascade, spot map, point-light cube face) culls instances like
the engine: three culls a plain mesh by its bounding sphere, and an `InstancedMesh` draws there
nothing when no instance's bounding sphere meets the view, or, with at least 64 instances
straddling it, only the instances that do (through a stand-in with a compacted copy of
`instanceMatrix`). The spheres are cached and recomputed when the world matrix, the geometry,
the instance matrices (`instanceMatrix.version`) or `count` change. Draw calls, not triangles,
bound a busy browser frame (each costs CPU in three and in the browser's GPU process, in every
view), so fewer, larger `InstancedMesh`es pay off. Only `LitMaterial` objects cast shadows; anything else in the scene (sprites,
overlays) is drawn into the HDR pass as three draws it, tone mapped but not exposed. A page that
paints its own backdrop behind a transparent canvas (`new WebGLRenderer({ alpha: true })`) sets
`lighting.clearAlpha = 0`: the output is premultiplied where nothing is drawn.

### Skinned meshes (web)

`LitMaterial` skins three's `SkinnedMesh` on the GPU with the engine's linear blend skinning (four
influences from `skinIndex`/`skinWeight`, the blended palette replacing the model matrix, normals
through its inverse transpose), in the main pass and every shadow pass. three's palette is
`bone.matrixWorld × boneInverse` per joint, the engine's `local_from_joint × joint_from_bind`, with
three's `bindMatrix` around it; both bind modes work. A glTF figure from `GLTFLoader` needs only
the material conversion above: its bones sit in the scene, an `AnimationMixer` (or the game) poses
them, and three updates the skeleton once per frame. three culls a `SkinnedMesh` by a bounding
sphere it computes on the CPU from the first pose it sees; a figure whose pose moves far from that
calls `mesh.computeBoundingSphere()` after posing, or sets `frustumCulled = false`.

### Sky and day cycle (web)

`Lighting.sky` is the engine's `Scene::sky`: Bevy's atmosphere with the sun and moon discs,
aerial perspective over the scene and, with `environmentIntensity`, the sky's own ambient light
(it replaces `Lighting.environment`). `Lighting.moon` is the unshadowed second directional light.
The sun's illuminance is the light above the atmosphere (`RAW_SUNLIGHT`); the atmosphere dims and
reddens it, so the game only moves the sun and adjusts exposure. `DayCycle` maps an hour to sun
and moon directions like the engine's:

```js
import { DayCycle, DEFAULT_SKY, DEFAULT_SUN_SHADOWS, RAW_SUNLIGHT } from '@moritzbrantner/three-lighting';

const cycle = new DayCycle({ noonElevation: (60 * Math.PI) / 180, moonLagHours: 12 });
const negate = ([x, y, z]) => [-x, -y, -z];
lighting.sky = DEFAULT_SKY;                       // or { ...DEFAULT_SKY, environmentIntensity: null }
lighting.environment = null;                      // the sky's ambient light replaces it
lighting.ambient = { color: [1, 1, 1], brightness: 0 };
lighting.clearColor = [0, 0, 0];
// every frame, from the game's clock:
lighting.sun = { direction: negate(cycle.toSun(hours)), color: [1, 1, 1], illuminance: RAW_SUNLIGHT, shadows: DEFAULT_SUN_SHADOWS };
lighting.moon = { direction: negate(cycle.toMoon(hours)), color: [0.8, 0.85, 1], illuminance: 0.3 };
lighting.ev100 = 14.5;                            // the oracle's views: 14.5 noon … 3.5 night
```

Each frame `Lighting.render` recomputes the atmosphere LUTs, renders the sky into a 128² cube and
filters it into the specular and diffuse cubes (Bevy's `GeneratedEnvironmentMapLight`), all as
fullscreen WebGL 2 passes (about 220 small draws; the oracle's sky frames take 3.1–3.7 ms of GPU
time in all on an RTX 3060 Ti, Bevy's 1.6–1.7 ms); then the main pass draws opaque and masked
meshes, the sky over them, and the blended meshes. It needs `EXT_color_buffer_float` and
`OES_texture_float_linear`, which desktop and most mobile browsers have. Use the camera's real
near plane; its far plane must lie beyond the scene, since the sky fills what the depth buffer
leaves at the far plane.

## Not covered yet

- Stars, clouds and moon phases: the sky is Bevy's clear atmosphere; the moon is a full disc.
- Morph targets, and more than four joint influences per vertex (`JOINTS_1`).
- Clustered lights. Every point and spot light is evaluated for every fragment.
- glTF features outside the core metallic-roughness model: texture transforms, a second UV set,
  clearcoat, transmission.
