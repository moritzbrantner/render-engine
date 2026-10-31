//! The Bevy side of a scenario: a headless Bevy app renders the scene into an image, saves a
//! screenshot after warm-up, then records per-frame GPU time from `RenderDiagnosticsPlugin`
//! timestamp spans.
//!
//! Each Bevy app runs in its own process (`oracle --bevy-side <scenario>`), because a Bevy
//! renderer cannot be created twice in one process.

use crate::scenarios::Scenario;
use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    camera::{Exposure, Hdr, RenderTarget},
    diagnostic::DiagnosticsStore,
    light::{
        CascadeShadowConfigBuilder, DirectionalLightShadowMap, EnvironmentMapLight,
        GlobalAmbientLight, ShadowFilteringMethod,
    },
    prelude::*,
    render::{
        RenderPlugin,
        diagnostic::RenderDiagnosticsPlugin,
        render_resource::{
            Extent3d, TextureDataOrder, TextureFormat, TextureViewDescriptor, TextureViewDimension,
        },
        settings::{RenderCreation, WgpuFeatures, WgpuSettings},
        view::{
            DebandDither, Msaa, Tonemapping,
            screenshot::{Screenshot, ScreenshotCaptured},
        },
    },
    window::ExitCondition,
};
use render_engine::scene::{
    Cubemap, FogFalloff as EngineFogFalloff, ShadowFilter, Shape, Tonemapping as EngineTonemapping,
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

pub fn run(scenario: Scenario, timed_frames: u32) -> BevyRun {
    let shared = Shared(Arc::new(Mutex::new(BevyRun::default())));
    let scene = scenario.scene.clone();
    let (width, height) = (scenario.width, scenario.height);

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
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
    ))
    .insert_resource(shared.clone())
    .insert_resource(Plan {
        frame: 0,
        timed_frames,
    })
    .insert_resource(GlobalAmbientLight {
        color: linear(scene.ambient.color),
        brightness: scene.ambient.brightness,
        ..default()
    });
    if let Some(shadows) = scene.sun.and_then(|s| s.shadows) {
        app.insert_resource(DirectionalLightShadowMap {
            size: shadows.map_size as usize,
        });
    }

    app.add_systems(
        Startup,
        move |mut commands: Commands,
              mut images: ResMut<Assets<Image>>,
              mut meshes: ResMut<Assets<Mesh>>,
              mut materials: ResMut<Assets<StandardMaterial>>| {
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
            for m in &scene.meshes {
                let mesh = match m.shape {
                    Shape::Cuboid(size) => meshes.add(Cuboid::new(size.x, size.y, size.z)),
                    Shape::Plane(size) => {
                        meshes.add(Plane3d::default().mesh().size(size.x, size.y))
                    }
                    Shape::Sphere {
                        radius,
                        sectors,
                        stacks,
                    } => meshes.add(Sphere::new(radius).mesh().uv(sectors, stacks)),
                };
                let mat = m.material;
                commands.spawn((
                    Mesh3d(mesh),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: linear(mat.base_color),
                        metallic: mat.metallic,
                        perceptual_roughness: mat.perceptual_roughness,
                        reflectance: mat.reflectance,
                        ..default()
                    })),
                    Transform {
                        translation: v3(m.translation),
                        rotation: Quat::from_array(m.rotation.to_array()),
                        scale: v3(m.scale),
                    },
                ));
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
        let mut total = 0.0;
        let mut spans = Vec::new();
        for d in store.iter() {
            let path = d.path().as_str();
            let parts: Vec<&str> = path.split('/').collect();
            // Only top-level spans (`render/<span>/elapsed_gpu`); nested spans are inside them.
            if parts.len() == 3
                && parts[0] == "render"
                && parts[2] == "elapsed_gpu"
                && let Some(ms) = d.measurement()
            {
                total += ms.value;
                spans.push(path.to_string());
            }
        }
        let mut run = shared.0.lock().unwrap();
        if !spans.is_empty() {
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
