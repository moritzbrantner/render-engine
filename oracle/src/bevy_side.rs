//! The Bevy side of a scenario: a headless Bevy app renders the scene into an image, saves a
//! screenshot after warm-up, then records per-frame GPU time from `RenderDiagnosticsPlugin`
//! timestamp spans.
//!
//! Each Bevy app runs in its own process (`oracle --bevy-side <scenario>`), because a Bevy
//! renderer cannot be created twice in one process.

use crate::content::Geometry;
use crate::particles::{self, Emitter, ParticleScenario, Spawn, Velocity as ParticleVelocity};
use crate::scenarios::Scenario;
use bevy::time::TimeUpdateStrategy;
use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    camera::{Exposure, Hdr, RenderTarget},
    diagnostic::DiagnosticsStore,
    light::{
        Atmosphere, AtmosphereEnvironmentMapLight, CascadeShadowConfigBuilder,
        DirectionalLightShadowMap, EnvironmentMapLight, GlobalAmbientLight, PointLightShadowMap,
        ShadowFilteringMethod, SunDisk, atmosphere::ScatteringMedium,
    },
    pbr::AtmosphereSettings,
    prelude::*,
    render::{
        RenderPlugin,
        diagnostic::RenderDiagnosticsPlugin,
        render_resource::{
            Extent3d, TextureDataOrder, TextureFormat, TextureViewDescriptor, TextureViewDimension,
        },
        renderer::{RenderContext, RenderGraph, RenderGraphSystems, RenderQueue},
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
        view::{
            DebandDither, Msaa, Tonemapping,
            screenshot::{Screenshot, ScreenshotCaptured},
        },
    },
    window::ExitCondition,
};
use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::NoFrustumCulling,
    image::{ImageSampler, ImageSamplerDescriptor},
    mesh::{
        Indices, PrimitiveTopology, VertexAttributeValues,
        skinning::{SkinnedMesh, SkinnedMeshInverseBindposes},
    },
    render::render_resource::Face,
};
use bevy_hanabi as hanabi;
use render_engine::particles::ParticleBlend;
use render_engine::{
    material::{AlphaMode as EngineAlphaMode, TextureData},
    mesh::{MeshData, Shape},
    scene::{
        Cubemap, FogFalloff as EngineFogFalloff, LightShadows, ShadowFilter,
        Tonemapping as EngineTonemapping,
    },
};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Duration,
};

/// Frames rendered before the screenshot so pipelines and shadow maps are settled.
const WARMUP_FRAMES: u32 = 30;

/// What the Bevy run produced.
#[derive(Default)]
pub struct BevyRun {
    pub rgba: Option<(u32, u32, Vec<u8>)>,
    /// GPU milliseconds per timed frame (sum of the top-level render diagnostic spans).
    pub gpu_ms: Vec<f64>,
    /// Top-level span paths that made up the sum (for the report).
    pub spans: Vec<String>,
}

#[derive(Resource, Clone)]
struct Shared(Arc<Mutex<BevyRun>>);

#[derive(Resource)]
struct Plan {
    frame: u32,
    timed_frames: u32,
}

fn v3(v: render_engine::glam::Vec3) -> Vec3 {
    Vec3::from_array(v.to_array())
}

fn linear(v: render_engine::glam::Vec3) -> Color {
    Color::linear_rgb(v.x, v.y, v.z)
}

/// The same texels as render-engine uploads, as a cube `Image` with Bevy's default sampler (what
/// Bevy's KTX2 loader produces from the file).
fn cube_image(cube: &Cubemap) -> Image {
    let mut image = Image {
        data: Some(cube.levels.concat()),
        data_order: TextureDataOrder::MipMajor,
        texture_view_descriptor: Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::Cube),
            ..default()
        }),
        ..default()
    };
    let desc = &mut image.texture_descriptor;
    desc.size = Extent3d {
        width: cube.size,
        height: cube.size,
        depth_or_array_layers: 6,
    };
    desc.mip_level_count = cube.levels.len() as u32;
    desc.format = cube.format;
    image
}

/// The same vertex attributes and indices render-engine uploads, as a Bevy `Mesh`.
fn mesh(m: &MeshData) -> Mesh {
    let arrays3 =
        |v: &[render_engine::glam::Vec3]| v.iter().map(|v| v.to_array()).collect::<Vec<_>>();
    let arrays4 =
        |v: &[render_engine::glam::Vec4]| v.iter().map(|v| v.to_array()).collect::<Vec<_>>();
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, arrays3(&m.positions))
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, arrays3(&m.normals))
    .with_inserted_indices(Indices::U32(m.indices.clone()));
    if !m.uvs.is_empty() {
        let uvs: Vec<[f32; 2]> = m.uvs.iter().map(|v| v.to_array()).collect();
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    }
    if !m.tangents.is_empty() {
        mesh.insert_attribute(Mesh::ATTRIBUTE_TANGENT, arrays4(&m.tangents));
    }
    if !m.colors.is_empty() {
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, arrays4(&m.colors));
    }
    if !m.joints.is_empty() {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(m.joints.clone()),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, arrays4(&m.weights));
    }
    mesh
}

fn mat4(m: render_engine::glam::Mat4) -> Mat4 {
    Mat4::from_cols_array(&m.to_cols_array())
}

/// The same texels, mip chain and sampler render-engine uploads, as a Bevy `Image`.
fn texture_image(t: &TextureData) -> Image {
    let s = t.sampler;
    let mut image = Image {
        data: Some(t.levels.concat()),
        data_order: TextureDataOrder::MipMajor,
        sampler: ImageSampler::Descriptor(ImageSamplerDescriptor::from(
            render_engine::wgpu::SamplerDescriptor {
                address_mode_u: s.address_mode_u,
                address_mode_v: s.address_mode_v,
                mag_filter: s.mag_filter,
                min_filter: s.min_filter,
                mipmap_filter: s.mipmap_filter,
                ..default()
            },
        )),
        ..default()
    };
    let desc = &mut image.texture_descriptor;
    desc.size = Extent3d {
        width: t.width,
        height: t.height,
        depth_or_array_layers: 1,
    };
    desc.mip_level_count = t.levels.len() as u32;
    desc.format = t.format;
    image
}

/// A windowless Bevy app that renders as fast as it can, with GPU timestamp spans.
fn headless_app() -> App {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .build()
            // bevy_hanabi enables Bevy's `x11` feature, which brings in winit; the oracle has no
            // window, and winit fails without a display (CI).
            .disable::<bevy::winit::WinitPlugin>()
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                close_when_requested: false,
                ..default()
            })
            .set(RenderPlugin {
                render_creation: RenderCreation::Automatic(Box::new(WgpuSettings {
                    // Timestamp spans need these; adapters without them record CPU time only.
                    features: WgpuFeatures::TIMESTAMP_QUERY
                        | WgpuFeatures::TIMESTAMP_QUERY_INSIDE_ENCODERS
                        | WgpuFeatures::TIMESTAMP_QUERY_INSIDE_PASSES,
                    ..default()
                })),
                synchronous_pipeline_compilation: true,
                ..default()
            }),
    )
    .add_plugins((
        ScheduleRunnerPlugin::run_loop(Duration::ZERO),
        RenderDiagnosticsPlugin,
    ));
    app
}

/// Sum of the frame's top-level GPU spans (`render/<span>/elapsed_gpu`) and their paths, or
/// `None` before the first spans arrive.
fn frame_gpu_ms(store: &DiagnosticsStore) -> Option<(f64, Vec<String>)> {
    let mut total = 0.0;
    let mut spans = Vec::new();
    for d in store.iter() {
        let path = d.path().as_str();
        let parts: Vec<&str> = path.split('/').collect();
        // Only top-level spans (`render/<span>/elapsed_gpu`); nested spans are inside them.
        if parts.len() == 3
            && parts[0] == "render"
            && parts[2] == "elapsed_gpu"
            && let Some(latest) = d.measurement()
        {
            // Spans that share a name in one frame (the early and the late
            // `per_view_shadow_pass`) arrive as several measurements stamped with the same
            // sync time; `measurement()` alone would keep only the last of them.
            let ms: f64 = d
                .measurements()
                .filter(|m| m.time == latest.time)
                .map(|m| m.value)
                .sum();
            total += ms;
            spans.push(path.to_string());
        }
    }
    (!spans.is_empty()).then_some((total, spans))
}

pub fn run(scenario: Scenario, timed_frames: u32) -> BevyRun {
    let shared = Shared(Arc::new(Mutex::new(BevyRun::default())));
    let (width, height) = (scenario.width, scenario.height);
    let Scenario {
        scene,
        objects,
        assets,
        ..
    } = scenario;

    let mut app = headless_app();
    app.insert_resource(shared.clone())
        .insert_resource(Plan {
            frame: 0,
            timed_frames,
        })
        .insert_resource(GlobalAmbientLight {
            color: linear(scene.ambient.color),
            brightness: scene.ambient.brightness,
            ..default()
        })
        .insert_resource(DirectionalLightShadowMap {
            size: scene.shadow_maps.directional as usize,
        })
        .insert_resource(PointLightShadowMap {
            size: scene.shadow_maps.point as usize,
        });

    app.add_systems(
        Startup,
        move |mut commands: Commands,
              mut images: ResMut<Assets<Image>>,
              mut meshes: ResMut<Assets<Mesh>>,
              mut materials: ResMut<Assets<StandardMaterial>>,
              mut media: ResMut<Assets<ScatteringMedium>>,
              mut bindposes: ResMut<Assets<SkinnedMeshInverseBindposes>>| {
            let target = images.add(Image::new_target_texture(
                width,
                height,
                TextureFormat::Rgba8UnormSrgb,
                None,
            ));
            commands.insert_resource(TargetImage(target.clone()));
            let cam = scene.camera;
            let mut camera = commands.spawn((
                Camera3d::default(),
                Camera {
                    clear_color: ClearColorConfig::Custom(linear(scene.clear_color)),
                    ..default()
                },
                RenderTarget::Image(target.into()),
                Projection::Perspective(PerspectiveProjection {
                    fov: cam.fov_y,
                    near: cam.near,
                    ..default()
                }),
                Exposure { ev100: cam.ev100 },
                Hdr,
                match cam.tonemapping {
                    EngineTonemapping::AcesFitted => Tonemapping::AcesFitted,
                    EngineTonemapping::TonyMcMapface => Tonemapping::TonyMcMapface,
                    EngineTonemapping::AgX => Tonemapping::AgX,
                },
                DebandDither::Disabled,
                Msaa::Sample4,
                match cam.shadow_filter {
                    ShadowFilter::Hardware2x2 => ShadowFilteringMethod::Hardware2x2,
                    ShadowFilter::Gaussian => ShadowFilteringMethod::Gaussian,
                },
                Transform::from_translation(v3(cam.eye)).looking_at(v3(cam.target), Vec3::Y),
            ));
            if let Some(env) = &scene.environment {
                camera.insert(EnvironmentMapLight {
                    diffuse_map: images.add(cube_image(&env.diffuse)),
                    specular_map: images.add(cube_image(&env.specular)),
                    intensity: env.intensity,
                    ..default()
                });
            }
            if let Some(fog) = scene.fog {
                let rgba = |v: render_engine::glam::Vec4| Color::linear_rgba(v.x, v.y, v.z, v.w);
                camera.insert(DistanceFog {
                    color: rgba(fog.color),
                    directional_light_color: rgba(fog.directional_light_color),
                    directional_light_exponent: fog.directional_light_exponent,
                    falloff: match fog.falloff {
                        EngineFogFalloff::Linear { start, end } => {
                            FogFalloff::Linear { start, end }
                        }
                        EngineFogFalloff::Exponential { density } => {
                            FogFalloff::Exponential { density }
                        }
                        EngineFogFalloff::ExponentialSquared { density } => {
                            FogFalloff::ExponentialSquared { density }
                        }
                        EngineFogFalloff::Atmospheric {
                            extinction,
                            inscattering,
                        } => FogFalloff::Atmospheric {
                            extinction: v3(extinction),
                            inscattering: v3(inscattering),
                        },
                    },
                });
            }
            let disk = |d: render_engine::scene::SunDisk| SunDisk {
                angular_size: d.angular_size,
                intensity: d.intensity,
            };
            if let Some(sky) = scene.sky {
                camera.insert(AtmosphereSettings::default());
                if let Some(intensity) = sky.environment_intensity {
                    camera.insert(AtmosphereEnvironmentMapLight {
                        intensity,
                        ..default()
                    });
                }
                // Bevy's Earth atmosphere, placed by its hook 6 360 km below the origin.
                commands.spawn(Atmosphere {
                    ground_albedo: v3(sky.ground_albedo),
                    ..Atmosphere::earth(media.add(ScatteringMedium::earth(256, 256)))
                });
            }
            if let Some(sun) = scene.sun {
                let mut light = commands.spawn((
                    DirectionalLight {
                        color: linear(sun.color),
                        illuminance: sun.illuminance,
                        shadow_maps_enabled: sun.shadows.is_some(),
                        ..default()
                    },
                    Transform::default().looking_to(v3(sun.direction), Vec3::Y),
                ));
                if let Some(sky) = scene.sky {
                    light.insert(disk(sky.sun_disk));
                }
                if let Some(s) = sun.shadows {
                    light.insert((CascadeShadowConfigBuilder {
                        num_cascades: s.num_cascades,
                        minimum_distance: s.min_distance,
                        maximum_distance: s.max_distance,
                        first_cascade_far_bound: s.first_cascade_far_bound,
                        overlap_proportion: s.overlap_proportion,
                    }
                    .build(),));
                    light.entry::<DirectionalLight>().and_modify(move |mut l| {
                        l.shadow_depth_bias = s.depth_bias;
                        l.shadow_normal_bias = s.normal_bias;
                    });
                }
            }
            // Spawned after the sun: Bevy orders unshadowed directional lights first, then by
            // entity.
            if let Some(moon) = scene.moon {
                let mut light = commands.spawn((
                    DirectionalLight {
                        color: linear(moon.color),
                        illuminance: moon.illuminance,
                        shadow_maps_enabled: false,
                        ..default()
                    },
                    Transform::default().looking_to(v3(moon.direction), Vec3::Y),
                ));
                if let Some(sky) = scene.sky {
                    light.insert(disk(sky.moon_disk));
                }
            }
            // Shadow settings default to Bevy's for lights without shadows.
            let shadows = |s: Option<LightShadows>, default: LightShadows| s.unwrap_or(default);
            for l in &scene.point_lights {
                let s = shadows(l.shadows, LightShadows::POINT);
                commands.spawn((
                    PointLight {
                        color: linear(l.color),
                        intensity: l.intensity,
                        range: l.range,
                        radius: l.radius,
                        shadow_maps_enabled: l.shadows.is_some(),
                        shadow_depth_bias: s.depth_bias,
                        shadow_normal_bias: s.normal_bias,
                        shadow_map_near_z: s.near_z,
                        soft_shadows_enabled: s.soft,
                        ..default()
                    },
                    Transform::from_translation(v3(l.position)),
                ));
            }
            for l in &scene.spot_lights {
                let s = shadows(l.shadows, LightShadows::SPOT);
                commands.spawn((
                    SpotLight {
                        color: linear(l.color),
                        intensity: l.intensity,
                        range: l.range,
                        radius: l.radius,
                        inner_angle: l.inner_angle,
                        outer_angle: l.outer_angle,
                        shadow_maps_enabled: l.shadows.is_some(),
                        shadow_depth_bias: s.depth_bias,
                        shadow_normal_bias: s.normal_bias,
                        shadow_map_near_z: s.near_z,
                        soft_shadows_enabled: s.soft,
                        ..default()
                    },
                    Transform::from_translation(v3(l.position)).with_rotation(
                        Quat::from_rotation_arc(Vec3::NEG_Z, v3(l.direction).normalize()),
                    ),
                ));
            }
            let mesh_handles: Vec<Handle<Mesh>> =
                assets.meshes.iter().map(|m| meshes.add(mesh(m))).collect();
            let image_handles: Vec<Handle<Image>> = assets
                .textures
                .iter()
                .map(|t| images.add(texture_image(t)))
                .collect();
            // Shapes and materials are shared between objects, as a game shares its handles.
            let mut shapes = std::collections::HashMap::new();
            let mut shared_materials = std::collections::HashMap::new();
            for m in &objects {
                let mesh = match m.geometry {
                    Geometry::Mesh(i) => mesh_handles[i].clone(),
                    Geometry::Shape(shape) => shapes
                        .entry(format!("{shape:?}"))
                        .or_insert_with(|| match shape {
                            Shape::Cuboid(size) => meshes.add(Cuboid::new(size.x, size.y, size.z)),
                            Shape::Plane(size) => {
                                meshes.add(Plane3d::default().mesh().size(size.x, size.y))
                            }
                            Shape::Sphere {
                                radius,
                                sectors,
                                stacks,
                            } => meshes.add(Sphere::new(radius).mesh().uv(sectors, stacks)),
                        })
                        .clone(),
                };
                let mat = m.material;
                let texture = |i: Option<usize>| i.map(|i| image_handles[i].clone());
                let c = mat.base_color;
                let material = shared_materials
                    .entry(format!("{mat:?}"))
                    .or_insert_with(|| {
                        materials.add(StandardMaterial {
                            base_color: Color::linear_rgba(c.x, c.y, c.z, mat.alpha),
                            base_color_texture: texture(mat.base_color_texture),
                            emissive: LinearRgba::rgb(
                                mat.emissive.x,
                                mat.emissive.y,
                                mat.emissive.z,
                            ),
                            emissive_texture: texture(mat.emissive_texture),
                            metallic: mat.metallic,
                            perceptual_roughness: mat.perceptual_roughness,
                            metallic_roughness_texture: texture(mat.metallic_roughness_texture),
                            reflectance: mat.reflectance,
                            normal_map_texture: texture(mat.normal_map_texture),
                            occlusion_texture: texture(mat.occlusion_texture),
                            alpha_mode: match mat.alpha_mode {
                                EngineAlphaMode::Opaque => AlphaMode::Opaque,
                                EngineAlphaMode::Mask(cutoff) => AlphaMode::Mask(cutoff),
                                EngineAlphaMode::Blend => AlphaMode::Blend,
                            },
                            double_sided: mat.double_sided,
                            // What Bevy's glTF loader sets for `doubleSided`.
                            cull_mode: (!mat.double_sided).then_some(Face::Back),
                            unlit: mat.unlit,
                            ..default()
                        })
                    })
                    .clone();
                let transform = Transform {
                    translation: v3(m.translation),
                    rotation: Quat::from_array(m.rotation.to_array()),
                    scale: v3(m.scale),
                };
                let mut entity =
                    commands.spawn((Mesh3d(mesh), MeshMaterial3d(material), transform));
                if let Some(skin) = m.skin {
                    // Bevy's joints are world-space entities: the figure's transform times each
                    // joint's pose, without a hierarchy. The skinned model matrix replaces the
                    // mesh entity's; that one only places the bounds, so culling is off.
                    let skin = &assets.skins[skin];
                    let world_from_local = m.transform();
                    let joints = skin
                        .model_from_joint
                        .iter()
                        .map(|j| {
                            entity
                                .commands()
                                .spawn(Transform::from_matrix(mat4(world_from_local * *j)))
                                .id()
                        })
                        .collect();
                    let inverse_bindposes = bindposes.add(SkinnedMeshInverseBindposes::from(
                        skin.inverse_bindposes
                            .iter()
                            .map(|m| mat4(*m))
                            .collect::<Vec<_>>(),
                    ));
                    entity.insert((
                        SkinnedMesh {
                            inverse_bindposes,
                            joints,
                        },
                        NoFrustumCulling,
                    ));
                }
            }
        },
    )
    .add_systems(Update, drive);

    app.run();
    let mut run = shared.0.lock().unwrap();
    std::mem::take(&mut *run)
}

#[derive(Resource)]
struct TargetImage(Handle<Image>);

fn drive(
    mut commands: Commands,
    mut plan: ResMut<Plan>,
    target: Option<Res<TargetImage>>,
    shared: Res<Shared>,
    store: Res<DiagnosticsStore>,
    mut exit: MessageWriter<AppExit>,
) {
    plan.frame += 1;
    let Some(target) = target else { return };
    if plan.frame == WARMUP_FRAMES {
        let shared = shared.clone();
        commands.spawn(Screenshot::image(target.0.clone())).observe(
            move |captured: On<ScreenshotCaptured>| {
                let image = &captured.image;
                let size = image.size();
                let rgba = image
                    .clone()
                    .try_into_dynamic()
                    .expect("screenshot converts to an image")
                    .to_rgba8()
                    .into_raw();
                shared.0.lock().unwrap().rgba = Some((size.x, size.y, rgba));
            },
        );
    }
    // Diagnostics lag a few frames behind; record after the screenshot settled.
    if plan.frame > WARMUP_FRAMES + 10 {
        let mut run = shared.0.lock().unwrap();
        if let Some((total, spans)) = frame_gpu_ms(&store) {
            run.gpu_ms.push(total);
            run.spans = spans;
        }
        let done = run.gpu_ms.len() as u32 >= plan.timed_frames
            || plan.frame > WARMUP_FRAMES + 10 + plan.timed_frames * 2;
        if done && run.rgba.is_some() {
            exit.write(AppExit::Success);
        }
    }
}

/// Runs the Bevy side in a child process and reads its PNG and timings back.
pub fn run_in_child(
    name: &str,
    dir: &std::path::Path,
    timed_frames: u32,
) -> Result<(PathBuf, Vec<f64>, Vec<String>), String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let status = std::process::Command::new(exe)
        .args(["--bevy-side", name, "--out"])
        .arg(dir)
        .args(["--frames", &timed_frames.to_string()])
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("bevy side exited with {status}"));
    }
    let json = std::fs::read_to_string(dir.join(format!("{name}.bevy.json")))
        .map_err(|e| e.to_string())?;
    let value: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
    let gpu_ms = value["gpu_ms"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
        .unwrap_or_default();
    let spans = value["spans"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    Ok((dir.join(format!("{name}.bevy.png")), gpu_ms, spans))
}

/// What a particle scenario's Bevy run produced.
#[derive(Default)]
pub struct ParticleRun {
    /// The view one frame before the first spawn (the empty background).
    pub plate: Option<Vec<u8>>,
    /// `(simulation frame, RGBA8)` per capture.
    pub captures: Vec<(u32, Vec<u8>)>,
    /// Particles spawned per emitter and simulation frame (hanabi's `EffectSpawner::spawn_count`).
    pub spawns: Vec<Vec<u32>>,
    /// GPU milliseconds per timed frame (sum of the top-level render diagnostic spans), over
    /// the frames after the last capture.
    pub gpu_ms: Vec<f64>,
    /// GPU milliseconds of hanabi's compute passes per timed frame ([`HanabiTimer`]).
    pub sim_ms: Vec<f64>,
    pub spans: Vec<String>,
    /// The app update (1 = the first) on which the spawners started. hanabi sets each effect's
    /// seed to its `prng_seed` when it compiles the effect on update 1 and advances the seed
    /// chain once per update after that, so the first spawn uses the chain's
    /// `first_spawn_update - 1`-th successor.
    pub first_spawn_update: u32,
}

#[derive(Resource, Clone)]
struct ParticleShared(Arc<Mutex<ParticleRun>>);

#[derive(Resource)]
struct ParticlePlan {
    /// Updates since the app started.
    updates: u32,
    frame: u32,
    /// Simulation frame (1 = the frame the spawners start); 0 while warming up.
    sim_frame: u32,
    captures: Vec<u32>,
    emitters: usize,
    /// Frames timed after the last capture (0: exit right after it).
    timed_frames: u32,
}

impl ParticlePlan {
    fn last_capture(&self) -> u32 {
        self.captures.iter().copied().max().unwrap_or(0)
    }

    /// Whether this simulation frame's GPU time counts: after the last capture, once the
    /// diagnostics (a few frames behind) show frames from after it.
    fn timed(&self) -> bool {
        let first = self.last_capture() + TIMING_LAG;
        self.timed_frames > 0
            && self.sim_frame > first
            && self.sim_frame <= first + self.timed_frames
    }

    fn done(&self) -> bool {
        let extra = if self.timed_frames > 0 {
            TIMING_LAG + self.timed_frames
        } else {
            0
        };
        self.sim_frame > self.last_capture() + extra
    }
}

/// Frames by which GPU timings trail the frame that records them.
const TIMING_LAG: u32 = 4;

/// Timestamps around hanabi's compute passes, which write none of their own: hanabi records all
/// of them (spawner fill, init, indirect, update, sort) in one system of Bevy's `RenderGraph`
/// schedule (`bevy_hanabi::render::simulate`). [`hanabi_timer_begin`] runs right before it and
/// [`hanabi_timer_end`] right after it, each writing a timestamp into its own command encoder;
/// Bevy submits the encoders in system order, so the span covers exactly hanabi's work. The span
/// is read back after the frame's submission and kept in `spans` while it records.
#[derive(Resource)]
struct HanabiTimer {
    query_set: Mutex<Option<render_engine::wgpu::QuerySet>>,
    spans: Arc<HanabiSpans>,
}

/// hanabi's GPU milliseconds per frame, kept while `record` is set (the timed frames).
#[derive(Default)]
struct HanabiSpans {
    record: std::sync::atomic::AtomicBool,
    ms: Mutex<Vec<f64>>,
}

/// The main world's handle on [`HanabiSpans`], to switch recording on for the timed frames.
#[derive(Resource)]
struct HanabiSpansHandle(Arc<HanabiSpans>);

impl std::ops::Deref for HanabiSpansHandle {
    type Target = HanabiSpans;
    fn deref(&self) -> &HanabiSpans {
        &self.0
    }
}

fn hanabi_timer_begin(mut ctx: RenderContext, timer: Res<HanabiTimer>) {
    let inside_encoders = render_engine::wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    if !ctx.render_device().features().contains(inside_encoders) {
        return;
    }
    let mut query_set = timer.query_set.lock().unwrap();
    let device = ctx.render_device().wgpu_device().clone();
    let query_set = query_set.get_or_insert_with(|| {
        device.create_query_set(&render_engine::wgpu::QuerySetDescriptor {
            label: Some("hanabi timer"),
            ty: render_engine::wgpu::QueryType::Timestamp,
            count: 2,
        })
    });
    ctx.command_encoder().write_timestamp(query_set, 0);
}

fn hanabi_timer_end(mut ctx: RenderContext, timer: Res<HanabiTimer>, queue: Res<RenderQueue>) {
    use render_engine::wgpu;
    let query_set = timer.query_set.lock().unwrap();
    let Some(query_set) = query_set.as_ref() else {
        return;
    };
    let device = ctx.render_device().wgpu_device().clone();
    let buffer = |usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hanabi timer"),
            size: 16,
            usage,
            mapped_at_creation: false,
        })
    };
    let resolve = buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
    let readback = buffer(wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ);
    let encoder = ctx.command_encoder();
    encoder.write_timestamp(query_set, 1);
    encoder.resolve_query_set(query_set, 0..2, &resolve, 0);
    encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, 16);
    let period = f64::from(queue.get_timestamp_period());
    let spans = timer.spans.clone();
    let mapped = readback.clone();
    encoder.map_buffer_on_submit(&readback, wgpu::MapMode::Read, .., move |r| {
        if r.is_ok()
            && spans.record.load(std::sync::atomic::Ordering::Relaxed)
            && let Ok(view) = mapped.slice(..).get_mapped_range()
        {
            let tick = |i: usize| u64::from_le_bytes(view[i * 8..i * 8 + 8].try_into().unwrap());
            spans
                .ms
                .lock()
                .unwrap()
                .push(tick(1).wrapping_sub(tick(0)) as f64 * period / 1e6);
        }
    });
}

/// Brackets hanabi's `simulate` system with [`HanabiTimer`]'s, if the device can write
/// timestamps between passes.
fn add_hanabi_timer(app: &mut App, spans: Arc<HanabiSpans>) {
    let render_app = app.sub_app_mut(bevy::render::RenderApp);
    let simulate = render_app
        .world()
        .resource::<bevy::ecs::schedule::Schedules>()
        .get(RenderGraph)
        .and_then(|s| {
            s.graph().systems.iter().find_map(|(_, system, _)| {
                let name = system.name().to_string();
                (name.contains("hanabi") && name.ends_with("::simulate"))
                    .then(|| system.default_system_sets()[0])
            })
        })
        .expect("bevy_hanabi's simulate system is in the RenderGraph schedule");
    render_app
        .insert_resource(HanabiTimer {
            query_set: Mutex::new(None),
            spans,
        })
        .add_systems(
            RenderGraph,
            (
                hanabi_timer_begin
                    .in_set(RenderGraphSystems::Render)
                    .before(simulate),
                hanabi_timer_end
                    .in_set(RenderGraphSystems::Render)
                    .after(simulate)
                    .before(bevy::core_pipeline::schedule::camera_driver),
            ),
        );
}

/// Variations of a particle scenario for calibrating the statistics gate (`oracle --calibrate`).
#[derive(Clone, Copy, Debug)]
pub struct ParticleVariant {
    /// Replaces the scenario's PRNG seed.
    pub seed: Option<u32>,
    /// Scales every emitter's gravity.
    pub gravity_scale: f32,
}

impl Default for ParticleVariant {
    fn default() -> Self {
        ParticleVariant {
            seed: None,
            gravity_scale: 1.0,
        }
    }
}

/// The emitter as a hanabi effect: the same spawner, init, update and render modifiers, in the
/// order `Emitter::engine` gives render-engine (the init modifiers draw from the particle's
/// random stream in this order: position, then velocity).
fn effect_asset(e: &Emitter, seed: u32, gravity_scale: f32) -> hanabi::EffectAsset {
    use hanabi::{Attribute, SetAttributeModifier};
    let w = hanabi::ExprWriter::new();
    let init_position = hanabi::SetPositionSphereModifier {
        center: w.lit(v3(e.position)).expr(),
        radius: w.lit(e.radius).expr(),
        dimension: hanabi::ShapeDimension::Volume,
    };
    enum InitVelocity {
        Radial(hanabi::SetVelocitySphereModifier),
        Jitter(SetAttributeModifier),
    }
    let init_velocity = match e.velocity {
        ParticleVelocity::Radial { speed } => {
            InitVelocity::Radial(hanabi::SetVelocitySphereModifier {
                center: w.lit(v3(e.position)).expr(),
                speed: w.lit(speed).expr(),
            })
        }
        ParticleVelocity::Jitter { base, spread } => {
            let r = w.rand(hanabi::VectorType::VEC3F);
            let v = r * w.lit(v3(spread * 2.0)) + w.lit(v3(base - spread));
            InitVelocity::Jitter(SetAttributeModifier::new(Attribute::VELOCITY, v.expr()))
        }
    };
    let init_age = SetAttributeModifier::new(Attribute::AGE, w.lit(0.0f32).expr());
    let init_lifetime = SetAttributeModifier::new(Attribute::LIFETIME, w.lit(e.lifetime).expr());
    let gravity = hanabi::AccelModifier::new(w.lit(v3(e.gravity * gravity_scale)).expr());
    let round = hanabi::RoundModifier {
        roundness: w.lit(1.0f32).expr(),
    };
    let mut color = hanabi::Gradient::new();
    for &(t, c) in &e.color {
        color.add_key(t, Vec4::from_array(c.to_array()));
    }
    let mut size = hanabi::Gradient::new();
    for &(t, s) in &e.size {
        size.add_key(t, Vec3::splat(s));
    }
    let spawner = match e.spawn {
        Spawn::Burst { count } => hanabi::SpawnerSettings::once((count as f32).into()),
        Spawn::Rate { per_second } => hanabi::SpawnerSettings::rate(per_second.into()),
    };
    let mut asset = hanabi::EffectAsset::new(e.capacity, spawner, w.finish())
        .with_name(e.name)
        .with_alpha_mode(match e.blend {
            ParticleBlend::Add => hanabi::AlphaMode::Add,
            ParticleBlend::Alpha => hanabi::AlphaMode::Blend,
        })
        .with_simulation_condition(hanabi::SimulationCondition::Always)
        .init(init_position);
    asset = match init_velocity {
        InitVelocity::Radial(m) => asset.init(m),
        InitVelocity::Jitter(m) => asset.init(m),
    };
    asset.prng_seed = seed;
    asset
        .init(init_age)
        .init(init_lifetime)
        .update(gravity)
        .render(hanabi::OrientModifier::new(
            hanabi::OrientMode::ParallelCameraDepthPlane,
        ))
        .render(hanabi::ColorOverLifetimeModifier {
            gradient: color,
            blend: hanabi::ColorBlendMode::Overwrite,
            mask: hanabi::ColorBlendMask::RGBA,
        })
        .render(hanabi::SizeOverLifetimeModifier {
            gradient: size,
            screen_space_size: false,
        })
        .render(round)
}

/// Renders a particle scenario on Bevy + bevy_hanabi at the fixed timestep `particles::DT`.
/// After warm-up (and once every effect's pipelines are ready) it captures the empty plate,
/// starts all spawners on the next frame and captures each `Capture::frame`.
pub fn run_particles(
    scenario: ParticleScenario,
    variant: ParticleVariant,
    timed_frames: u32,
) -> ParticleRun {
    let shared = ParticleShared(Arc::new(Mutex::new(ParticleRun {
        spawns: vec![Vec::new(); scenario.emitters.len()],
        ..default()
    })));
    let (width, height) = (scenario.width, scenario.height);
    let mut app = headless_app();
    app.add_plugins(hanabi::HanabiPlugin)
        // hanabi seeds its CPU sampler from the OS; the scenarios only use constant CPU values,
        // but keep it fixed anyway.
        .insert_resource(hanabi::Random(rand_pcg::Pcg32::new(
            0xcafe_f00d_d15e_a5e5,
            0,
        )))
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            particles::DT as f64,
        )))
        .insert_resource(shared.clone())
        .insert_resource(ParticlePlan {
            frame: 0,
            sim_frame: 0,
            updates: 0,
            captures: scenario.captures.iter().map(|c| c.frame).collect(),
            emitters: scenario.emitters.len(),
            timed_frames,
        });
    let seed = variant.seed.unwrap_or(scenario.seed);
    app.add_systems(
        Startup,
        move |mut commands: Commands,
              mut images: ResMut<Assets<Image>>,
              mut effects: ResMut<Assets<hanabi::EffectAsset>>| {
            let target = images.add(Image::new_target_texture(
                width,
                height,
                TextureFormat::Rgba8UnormSrgb,
                None,
            ));
            commands.insert_resource(TargetImage(target.clone()));
            let cam = scenario.camera;
            commands.spawn((
                Camera3d::default(),
                Camera {
                    clear_color: ClearColorConfig::Custom(linear(scenario.clear_color)),
                    ..default()
                },
                RenderTarget::Image(target.into()),
                Projection::Perspective(PerspectiveProjection {
                    fov: cam.fov_y,
                    near: cam.near,
                    ..default()
                }),
                Exposure { ev100: cam.ev100 },
                Hdr,
                match cam.tonemapping {
                    EngineTonemapping::AcesFitted => Tonemapping::AcesFitted,
                    EngineTonemapping::TonyMcMapface => Tonemapping::TonyMcMapface,
                    EngineTonemapping::AgX => Tonemapping::AgX,
                },
                DebandDither::Disabled,
                Msaa::Sample4,
                Transform::from_translation(v3(cam.eye)).looking_at(v3(cam.target), Vec3::Y),
            ));
            for (i, e) in scenario.emitters.iter().enumerate() {
                let asset = effect_asset(e, seed.wrapping_add(i as u32), variant.gravity_scale);
                let spawner = hanabi::EffectSpawner::new(&asset.spawner).with_active(false);
                commands.spawn((
                    hanabi::ParticleEffect::new(effects.add(asset)),
                    spawner,
                    EmitterIndex(i),
                    Transform::from_translation(v3(e.translation)),
                ));
            }
        },
    )
    .add_systems(Update, drive_particles)
    .add_systems(Last, record_frame);
    let hanabi_spans = Arc::new(HanabiSpans::default());
    app.insert_resource(HanabiSpansHandle(hanabi_spans.clone()));
    add_hanabi_timer(&mut app, hanabi_spans.clone());

    app.run();
    let mut run = shared.0.lock().unwrap();
    run.sim_ms = std::mem::take(&mut *hanabi_spans.ms.lock().unwrap());
    std::mem::take(&mut *run)
}

#[derive(Component)]
struct EmitterIndex(usize);

fn screenshot(commands: &mut Commands, target: &TargetImage, shared: &ParticleShared, frame: u32) {
    let shared = shared.clone();
    commands.spawn(Screenshot::image(target.0.clone())).observe(
        move |captured: On<ScreenshotCaptured>| {
            let rgba = captured
                .image
                .clone()
                .try_into_dynamic()
                .expect("screenshot converts to an image")
                .to_rgba8()
                .into_raw();
            let mut run = shared.0.lock().unwrap();
            if frame == 0 {
                run.plate = Some(rgba);
            } else {
                run.captures.push((frame, rgba));
            }
        },
    );
}

fn drive_particles(
    mut commands: Commands,
    mut plan: ResMut<ParticlePlan>,
    target: Option<Res<TargetImage>>,
    shared: Res<ParticleShared>,
    compiled: Query<&hanabi::CompiledParticleEffect>,
    mut spawners: Query<&mut hanabi::EffectSpawner>,
    mut exit: MessageWriter<AppExit>,
) {
    plan.frame += 1;
    plan.updates += 1;
    let Some(target) = target else { return };
    if plan.sim_frame == 0 {
        let ready = compiled.iter().filter(|c| c.is_ready()).count() == plan.emitters;
        if plan.frame == WARMUP_FRAMES && !ready {
            // Keep warming up until every effect can spawn.
            plan.frame -= 1;
        } else if plan.frame == WARMUP_FRAMES {
            screenshot(&mut commands, &target, &shared, 0);
        } else if plan.frame > WARMUP_FRAMES {
            spawners.iter_mut().for_each(|mut s| s.active = true);
            plan.sim_frame = 1;
            shared.0.lock().unwrap().first_spawn_update = plan.updates;
        }
    } else {
        plan.sim_frame += 1;
    }
    if plan.sim_frame == 0 {
        return;
    }
    if plan.captures.contains(&plan.sim_frame) {
        screenshot(&mut commands, &target, &shared, plan.sim_frame);
    }
    let run = shared.0.lock().unwrap();
    if plan.done() && run.plate.is_some() && run.captures.len() == plan.captures.len() {
        exit.write(AppExit::Success);
    }
}

/// After hanabi ticked the spawners (`PostUpdate`): this simulation frame's spawn counts, and
/// the latest frame's GPU time.
fn record_frame(
    plan: Res<ParticlePlan>,
    shared: Res<ParticleShared>,
    store: Res<DiagnosticsStore>,
    spawners: Query<(&hanabi::EffectSpawner, &EmitterIndex)>,
    hanabi_spans: Res<HanabiSpansHandle>,
) {
    if plan.sim_frame == 0 {
        return;
    }
    let mut run = shared.0.lock().unwrap();
    hanabi_spans
        .record
        .store(plan.timed(), std::sync::atomic::Ordering::Relaxed);
    if plan.timed()
        && let Some((total, spans)) = frame_gpu_ms(&store)
    {
        run.gpu_ms.push(total);
        run.spans = spans;
    }
    if plan.sim_frame <= plan.last_capture() {
        for (s, i) in &spawners {
            run.spawns[i.0].push(s.spawn_count);
        }
    }
}

/// Runs a particle scenario's Bevy side in a child process (`--bevy-side` with the variant's
/// `--seed`/`--gravity-scale`, timing `timed_frames` frames after the last capture) and reads its
/// output back from `dir`.
pub fn run_particles_in_child(
    scenario: &ParticleScenario,
    dir: &std::path::Path,
    variant: ParticleVariant,
    timed_frames: u32,
) -> Result<ParticleRun, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut cmd = std::process::Command::new(exe);
    cmd.args(["--bevy-side", scenario.name, "--out"])
        .arg(dir)
        .args(["--frames", &timed_frames.to_string()]);
    if let Some(seed) = variant.seed {
        cmd.args(["--seed", &seed.to_string()]);
    }
    cmd.args(["--gravity-scale", &variant.gravity_scale.to_string()]);
    let status = cmd.status().map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("bevy side exited with {status}"));
    }
    let read = |file: String| -> Result<Vec<u8>, String> {
        Ok(image::open(dir.join(&file))
            .map_err(|e| format!("{file}: {e}"))?
            .to_rgba8()
            .into_raw())
    };
    let json = std::fs::read_to_string(dir.join(format!("{}.bevy.json", scenario.name)))
        .map_err(|e| e.to_string())?;
    let value: serde_json::Value = serde_json::from_str(&json).map_err(|e| e.to_string())?;
    let numbers = |v: &serde_json::Value| -> Vec<f64> {
        v.as_array()
            .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default()
    };
    Ok(ParticleRun {
        plate: Some(read(format!("{}.plate.bevy.png", scenario.name))?),
        captures: scenario
            .captures
            .iter()
            .map(|c| {
                Ok((
                    c.frame,
                    read(format!("{}.{}.bevy.png", scenario.name, c.label()))?,
                ))
            })
            .collect::<Result<_, String>>()?,
        spawns: value["spawns"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|e| numbers(e).into_iter().map(|n| n as u32).collect())
                    .collect()
            })
            .unwrap_or_default(),
        gpu_ms: numbers(&value["gpu_ms"]),
        sim_ms: numbers(&value["sim_ms"]),
        first_spawn_update: value["first_spawn_update"].as_u64().unwrap_or(0) as u32,
        spans: value["spans"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
    })
}
