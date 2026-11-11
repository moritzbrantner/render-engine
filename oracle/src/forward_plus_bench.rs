//! Issue #72: bounded evaluation of Forward+ clustered lighting against the reference path (every
//! point and spot light for every fragment), on a small workload matrix (see
//! `docs/forward-plus-experiment.md`). Diagnostic only: it does not touch the regular scenario
//! gates, and the per-fragment loop stays the renderer's default and fallback.

use crate::{
    Options, bevy_side,
    content::{self, Geometry, Material, MeshInstance},
    flip,
    gpu_driven_bench::{Round, read, timed_run},
    save_png, scenarios,
    scenarios::{Feature, Scenario},
};
use render_engine::{
    glam::{Quat, Vec2, Vec3},
    headless::Headless,
    material::AlphaMode,
    mesh::Shape,
    renderer::{ClusterConfig, LightAssignment, Viewport},
    scene::{
        Ambient, Camera, LightShadows, PointLight, Scene, ShadowFilter, ShadowMapSizes, SpotLight,
        Sun, SunShadows, Tonemapping,
    },
    wgpu,
    world::{Instance, World},
};
use serde_json::{Value, json};
use std::{path::Path, process::ExitCode};

/// A deterministic value in `0..1` for `(index, salt)`: a cheap hash, so the scenes need no
/// random state and Bevy's child process rebuilds the same content.
fn rand01(index: usize, salt: u32) -> f32 {
    let mut x = (index as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(salt.wrapping_mul(40_503));
    x ^= x >> 15;
    x = x.wrapping_mul(2_246_822_519);
    x ^= x >> 13;
    x = x.wrapping_mul(3_266_489_917);
    x ^= x >> 16;
    x as f32 / u32::MAX as f32
}

fn hue(h: f32) -> Vec3 {
    let c = |n: f32| {
        let k = (n + h * 6.0) % 6.0;
        1.0 - (k.min(4.0 - k).clamp(0.0, 1.0))
    };
    Vec3::new(c(5.0), c(3.0), c(1.0))
}

fn material(base_color: Vec3, perceptual_roughness: f32) -> Material {
    Material {
        base_color,
        perceptual_roughness,
        ..Material::default()
    }
}

/// An outdoor town-edge scene in the Greyhaven style: sun with cascaded shadows, masked foliage
/// (the CC0 `Bush_Common` prop), a blended water plane and `lights` local lights (point lights,
/// every fourth a spot light) strung along the path.
fn outdoor(name: &'static str, lights: usize) -> Scenario {
    let mut assets = content::Assets::default();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets/props");
    let bush = assets.load_gltf(&dir.join("Bush_Common.gltf"));
    let mut objects = vec![MeshInstance::new(
        Shape::Plane(Vec2::splat(200.0)),
        Vec3::ZERO,
        material(Vec3::new(0.18, 0.28, 0.12), 0.9),
    )];
    // Foliage in two bands either side of the path, on a jittered grid.
    for i in 0..160 {
        let side = if i % 2 == 0 { -1.0 } else { 1.0 };
        let x = side * (6.0 + 22.0 * rand01(i, 1));
        let z = 22.0 - 2.5 * (i / 2) as f32 + 3.0 * rand01(i, 2);
        let yaw = 360.0 * rand01(i, 3);
        let scale = 0.8 + 0.8 * rand01(i, 4);
        for p in &bush {
            objects.push(MeshInstance {
                geometry: Geometry::Mesh(p.mesh),
                translation: Vec3::new(x, 0.0, z),
                rotation: Quat::from_rotation_y(yaw.to_radians()),
                scale: Vec3::splat(scale),
                material: p.material,
                skin: None,
            });
        }
    }
    // A few huts.
    for i in 0..8 {
        let x = if i % 2 == 0 { -16.0 } else { 16.0 };
        let z = 14.0 - 12.0 * (i / 2) as f32;
        objects.push(MeshInstance::new(
            Shape::Cuboid(Vec3::new(5.0, 3.5, 6.0)),
            Vec3::new(x, 1.75, z),
            material(Vec3::new(0.46, 0.40, 0.32), 0.85),
        ));
    }
    // The water: blended, shading the local lights like any other fragment.
    objects.push(MeshInstance::new(
        Shape::Plane(Vec2::new(16.0, 90.0)),
        Vec3::new(0.0, 0.05, -20.0),
        Material {
            base_color: Vec3::new(0.08, 0.2, 0.28),
            alpha: 0.6,
            perceptual_roughness: 0.1,
            alpha_mode: AlphaMode::Blend,
            ..Material::default()
        },
    ));
    let (mut points, mut spots) = (Vec::new(), Vec::new());
    for i in 0..lights {
        let side = if i % 2 == 0 { -1.0 } else { 1.0 };
        let position = Vec3::new(
            side * (3.0 + 3.0 * rand01(i, 5)),
            1.5 + 2.5 * rand01(i, 6),
            26.0 - 80.0 * i as f32 / lights.max(1) as f32,
        );
        let color = hue(0.05 + 0.08 * rand01(i, 7));
        if i % 4 == 3 {
            spots.push(SpotLight {
                position,
                direction: Vec3::new(-side * 0.3, -1.0, 0.2 * (rand01(i, 8) - 0.5)),
                color,
                intensity: 30_000.0,
                range: 16.0,
                radius: 0.0,
                inner_angle: 15f32.to_radians(),
                outer_angle: 35f32.to_radians(),
                shadows: None,
            });
        } else {
            points.push(PointLight {
                position,
                color,
                intensity: 25_000.0,
                range: 12.0,
                radius: 0.05,
                shadows: None,
            });
        }
    }
    let scene = Scene {
        camera: Camera {
            eye: Vec3::new(1.5, 3.0, 34.0),
            target: Vec3::new(0.0, 1.5, 0.0),
            fov_y: 50f32.to_radians(),
            near: 0.1,
            ev100: 10.5,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::TonyMcMapface,
        },
        clear_color: Vec3::new(0.55, 0.65, 0.78),
        sun: Some(Sun {
            direction: Vec3::new(0.4, -1.0, -0.5).normalize(),
            color: Vec3::new(1.0, 0.92, 0.82),
            illuminance: 6_000.0,
            shadows: Some(SunShadows::default()),
        }),
        moon: None,
        sky: None,
        ambient: Ambient {
            color: Vec3::new(0.6, 0.7, 0.9),
            brightness: 400.0,
        },
        point_lights: points,
        spot_lights: spots,
        shadow_maps: ShadowMapSizes::default(),
        environment: None,
        fog: None,
        overlay: Vec::new(),
    };
    Scenario {
        name,
        capability: "Forward+ evaluation: outdoor",
        feature: Feature::PointSpot,
        width: 1280,
        height: 720,
        scene,
        objects,
        assets,
    }
}

/// A dense underground hall lit only by `lights` overlapping point and spot lights, some casting
/// shadows. Intensity falls with the count so the frame stays comparably exposed.
pub(crate) fn dungeon(
    name: &'static str,
    lights: usize,
    casters: usize,
    width: u32,
    height: u32,
) -> Scenario {
    let stone = material(Vec3::new(0.50, 0.47, 0.42), 0.8);
    let mut objects = vec![
        MeshInstance::new(
            Shape::Plane(Vec2::splat(80.0)),
            Vec3::ZERO,
            material(Vec3::new(0.42, 0.40, 0.36), 0.9),
        ),
        MeshInstance::new(
            Shape::Cuboid(Vec3::new(80.0, 0.5, 80.0)),
            Vec3::new(0.0, 7.25, 0.0),
            stone,
        ),
    ];
    for (x, z, sx, sz) in [
        (0.0, -40.0, 80.0, 1.0),
        (0.0, 40.0, 80.0, 1.0),
        (-40.0, 0.0, 1.0, 80.0),
        (40.0, 0.0, 1.0, 80.0),
    ] {
        objects.push(MeshInstance::new(
            Shape::Cuboid(Vec3::new(sx, 7.0, sz)),
            Vec3::new(x, 3.5, z),
            stone,
        ));
    }
    // Pillars on a grid, then scattered crates and spheres.
    for ix in -5..=5 {
        for iz in -5..=5 {
            objects.push(MeshInstance::new(
                Shape::Cuboid(Vec3::new(1.2, 7.0, 1.2)),
                Vec3::new(ix as f32 * 7.0, 3.5, iz as f32 * 7.0),
                stone,
            ));
        }
    }
    for i in 0..400 {
        let x = -36.0 + 72.0 * rand01(i, 21);
        let z = -36.0 + 72.0 * rand01(i, 22);
        let color = Vec3::new(
            0.2 + 0.5 * rand01(i, 23),
            0.2 + 0.4 * rand01(i, 24),
            0.2 + 0.3 * rand01(i, 25),
        );
        let mat = material(color, 0.3 + 0.6 * rand01(i, 26));
        objects.push(if i % 2 == 0 {
            MeshInstance::new(
                Shape::Cuboid(Vec3::splat(0.5 + 0.5 * rand01(i, 27))),
                Vec3::new(x, 0.45, z),
                mat,
            )
        } else {
            MeshInstance::new(
                Shape::Sphere {
                    radius: 0.35,
                    sectors: 16,
                    stacks: 8,
                },
                Vec3::new(x, 0.35, z),
                mat,
            )
        });
    }
    let scale = (256.0 / lights.max(1) as f32).clamp(0.25, 4.0);
    let (mut points, mut spots) = (Vec::new(), Vec::new());
    for i in 0..lights {
        let position = Vec3::new(
            -36.0 + 72.0 * rand01(i, 31),
            1.0 + 5.0 * rand01(i, 32),
            -36.0 + 72.0 * rand01(i, 33),
        );
        let color = hue(rand01(i, 34)).lerp(Vec3::ONE, 0.25);
        let range = 8.0 + 6.0 * rand01(i, 35);
        if i % 4 == 3 {
            spots.push(SpotLight {
                position,
                direction: Vec3::new(rand01(i, 36) - 0.5, -1.0, rand01(i, 37) - 0.5),
                color,
                intensity: 6_000.0 * scale,
                range,
                radius: 0.0,
                inner_angle: 20f32.to_radians(),
                outer_angle: 40f32.to_radians(),
                shadows: (spots.len() < casters / 2).then_some(LightShadows::SPOT),
            });
        } else {
            points.push(PointLight {
                position,
                color,
                intensity: 3_000.0 * scale,
                range,
                radius: 0.05,
                shadows: (points.len() < casters - casters / 2).then_some(LightShadows::POINT),
            });
        }
    }
    let scene = Scene {
        camera: Camera {
            // Between two pillar columns (the pillars stand on multiples of 7 m).
            eye: Vec3::new(3.5, 2.2, 38.0),
            target: Vec3::new(3.5, 2.0, 0.0),
            fov_y: 50f32.to_radians(),
            near: 0.1,
            ev100: 5.0,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::TonyMcMapface,
        },
        clear_color: Vec3::splat(0.01),
        sun: None,
        moon: None,
        sky: None,
        ambient: Ambient {
            color: Vec3::new(0.5, 0.45, 0.7),
            brightness: 1.5,
        },
        point_lights: points,
        spot_lights: spots,
        shadow_maps: ShadowMapSizes::default(),
        environment: None,
        fog: None,
        overlay: Vec::new(),
    };
    Scenario {
        name,
        capability: "Forward+ evaluation: dense lights",
        feature: Feature::PointSpot,
        width,
        height,
        scene,
        objects,
        assets: Default::default(),
    }
}

/// The workload matrix: both scene types with low and high light counts, the light-count ladder
/// for the crossover, shadow casters, and a 1080p row.
fn scenes() -> Vec<Scenario> {
    vec![
        // The pure cost of the path: no local lights to assign or loop over.
        outdoor("outdoor_0", 0),
        outdoor("outdoor_low", 4),
        outdoor("outdoor_high", 64),
        dungeon("dungeon_4", 4, 0, 1280, 720),
        dungeon("dungeon_16", 16, 0, 1280, 720),
        dungeon("dungeon_64", 64, 0, 1280, 720),
        dungeon("dungeon_256", 256, 0, 1280, 720),
        dungeon("dungeon_1024", 1024, 0, 1280, 720),
        dungeon("dungeon_64_shadowed", 64, 4, 1280, 720),
        dungeon("dungeon_256_1080p", 256, 0, 1920, 1080),
    ]
}

/// The scenes compared directly with Bevy (found by name in the Bevy child process).
pub fn bevy_scenes() -> Vec<Scenario> {
    vec![
        outdoor("outdoor_low", 4),
        outdoor("outdoor_high", 64),
        dungeon("dungeon_16", 16, 0, 1280, 720),
        dungeon("dungeon_256", 256, 0, 1280, 720),
    ]
}

type Paths = (LightAssignment, &'static str);

const REFERENCE: Paths = (LightAssignment::PerFragment, "per_fragment");

fn clustered() -> Paths {
    (
        LightAssignment::Clustered(ClusterConfig::default()),
        "clustered",
    )
}

/// One full run of a path on a scene with the reference's timing protocol.
fn timed(
    gpu: &Headless,
    world: &World,
    instances: &[Instance],
    scenario: &Scenario,
    (assignment, _): Paths,
    frames: usize,
) -> Result<Round, String> {
    let configure = |viewport: &mut Viewport| {
        viewport
            .set_light_assignment(gpu.renderer(), assignment)
            .map_err(|e| e.to_string())
    };
    timed_run(
        gpu,
        world,
        instances,
        &scenario.scene,
        &|_, _| {},
        &configure,
        (scenario.width, scenario.height),
        frames,
    )
}

/// One frame with fragment shader statistics; `Some(invocations)` where the device has them.
fn fragment_invocations(
    gpu: &Headless,
    world: &World,
    instances: &[Instance],
    scenario: &Scenario,
    assignment: LightAssignment,
) -> Result<Option<u64>, String> {
    if !gpu
        .device
        .features()
        .contains(wgpu::Features::PIPELINE_STATISTICS_QUERY)
    {
        return Ok(None);
    }
    let renderer = gpu.renderer();
    let mut viewport = Viewport::new(renderer, scenario.width, scenario.height);
    viewport
        .set_light_assignment(renderer, assignment)
        .map_err(|e| e.to_string())?;
    let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("fragment statistics"),
        ty: wgpu::QueryType::PipelineStatistics(
            wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS,
        ),
        count: 2,
    });
    viewport.set_fragment_statistics(Some(query_set.clone()));
    let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("statistics resolve"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("statistics readback"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("statistics target"),
        size: wgpu::Extent3d {
            width: scenario.width,
            height: scenario.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: render_engine::headless::COLOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target = target.create_view(&Default::default());
    renderer.prepare(&mut viewport, world, &scenario.scene, instances);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    renderer.encode(&mut encoder, &viewport, world, &target, None);
    encoder.resolve_query_set(&query_set, 0..1, &resolve, 0);
    encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, 16);
    gpu.queue.submit([encoder.finish()]);
    let bytes = read(gpu, &readback);
    Ok(Some(u64::from_ne_bytes(
        bytes[..8].try_into().expect("8 bytes"),
    )))
}

/// `(shaded fragments, lights looped over)` from one diagnostic frame on the clustered path.
fn cluster_diagnostic(
    gpu: &Headless,
    world: &World,
    instances: &[Instance],
    scenario: &Scenario,
) -> Result<(u64, u64), String> {
    let renderer = gpu.renderer();
    let mut viewport = Viewport::new(renderer, scenario.width, scenario.height);
    viewport
        .set_light_assignment(
            renderer,
            LightAssignment::Clustered(ClusterConfig {
                diagnostic: true,
                ..ClusterConfig::default()
            }),
        )
        .map_err(|e| e.to_string())?;
    gpu.render(&mut viewport, world, &scenario.scene, instances)
        .map_err(|e| e.to_string())?;
    renderer
        .read_cluster_diagnostic(&viewport)
        .ok_or_else(|| "no diagnostic".to_string())
}

fn summary(r: &Round, name: &str) -> Value {
    let lit: f64 = r
        .passes
        .iter()
        .filter(|(k, _)| k.starts_with("lit"))
        .map(|(_, v)| *v)
        .sum();
    let shadow: f64 = r
        .passes
        .iter()
        .filter(|(k, _)| k.as_str() == "shadow")
        .map(|(_, v)| *v)
        .sum();
    json!({
        "path": name,
        "cpu": {
            "prepare_ms": r.prepare_ms,
            "encode_and_finish_ms": r.encode_ms,
            "submit_ms": r.submit_ms,
            "total_ms": r.prepare_ms + r.encode_ms + r.submit_ms,
        },
        "gpu_ms": r.gpu_ms,
        "gpu_lit_ms": lit,
        "gpu_shadow_pass_median_ms": shadow,
        "gpu_passes_ms": r.passes,
        "work": {
            "cluster_tests": r.work.light_cluster_tests,
            "cluster_entries": r.work.light_cluster_entries,
            "cluster_overflows": r.work.cluster_overflows,
            "upload_bytes": r.work.upload_bytes,
            "draws": r.work.draws,
            "allocations": r.work.allocations,
        },
        "memory": {
            "gpu_buffer_bytes": r.memory.gpu_buffer_bytes,
            "clustered_lighting_bytes": r.memory.clustered_lighting_bytes,
            "cpu_scratch_bytes": r.memory.cpu_scratch_bytes,
        },
    })
}

/// A frame on both paths: FLIP over the pair, exact equality, images when they differ.
fn compare_frame(
    gpu: &Headless,
    world: &World,
    instances: &[Instance],
    scene: &Scene,
    (width, height): (u32, u32),
    assignment: LightAssignment,
    dir: Option<(&Path, &str)>,
) -> Result<Value, String> {
    let renderer = gpu.renderer();
    let mut a = Viewport::new(renderer, width, height);
    a.set_light_assignment(renderer, LightAssignment::PerFragment)
        .map_err(|e| e.to_string())?;
    let mut b = Viewport::new(renderer, width, height);
    b.set_light_assignment(renderer, assignment)
        .map_err(|e| e.to_string())?;
    let reference = gpu
        .render(&mut a, world, scene, instances)
        .map_err(|e| e.to_string())?;
    let clustered = gpu
        .render(&mut b, world, scene, instances)
        .map_err(|e| e.to_string())?;
    let metrics = flip(width, height, &reference.rgba, &clustered.rgba);
    let identical = reference.rgba == clustered.rgba;
    let parity = metrics.mean <= crate::FLIP_MEAN_MAX && metrics.p99 <= crate::FLIP_P99_MAX;
    if let Some((dir, name)) = dir
        && !identical
    {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        save_png(
            &dir.join(format!("{name}.reference.png")),
            width,
            height,
            &reference.rgba,
        )?;
        save_png(
            &dir.join(format!("{name}.clustered.png")),
            width,
            height,
            &clustered.rgba,
        )?;
        save_png(
            &dir.join(format!("{name}.flip.png")),
            width,
            height,
            &metrics.heatmap,
        )?;
    }
    Ok(json!({
        "flip_mean": metrics.mean,
        "flip_p99": metrics.p99,
        "parity": parity,
        "byte_identical": identical,
        "cluster_overflows": b.work().cluster_overflows,
        "cluster_tests": b.work().light_cluster_tests,
    }))
}

/// Every oracle lighting scenario through both paths.
fn validate_scenarios(gpu: &Headless, out: &Path, failed: &mut bool) -> Vec<Value> {
    let dir = out.join("scenarios");
    let mut results = Vec::new();
    for scenario in scenarios::all() {
        let (world, instances, _) =
            content::upload(gpu.renderer(), &scenario.assets, &scenario.objects);
        match compare_frame(
            gpu,
            &world,
            &instances,
            &scenario.scene,
            (scenario.width, scenario.height),
            clustered().0,
            Some((&dir, scenario.name)),
        ) {
            Ok(v) => {
                if v["parity"] != true {
                    eprintln!("{}: clustered differs: {v}", scenario.name);
                    *failed = true;
                }
                println!(
                    "validate {:<26} flip mean={:.5} p99={:.4} identical={}",
                    scenario.name,
                    v["flip_mean"].as_f64().unwrap_or(0.0),
                    v["flip_p99"].as_f64().unwrap_or(0.0),
                    v["byte_identical"]
                );
                results.push(json!({"scenario": scenario.name, "result": v}));
            }
            Err(e) => {
                eprintln!("{}: {e}", scenario.name);
                *failed = true;
            }
        }
    }
    results
}

/// Camera cuts, near-plane crossings, light mutation and removal, lights behind the camera, a
/// cut through a light's volume, an empty light list, and forced overflow, all against the
/// per-fragment reference.
fn validate_sequence(gpu: &Headless, out: &Path, failed: &mut bool) -> Vec<Value> {
    let dir = out.join("sequence");
    let scenario = dungeon("sequence", 64, 4, 1280, 720);
    let (world, instances, _) =
        content::upload(gpu.renderer(), &scenario.assets, &scenario.objects);
    let size = (scenario.width, scenario.height);
    let mut results = Vec::new();
    let mut step = |name: &str, scene: &Scene, assignment, failed: &mut bool| match compare_frame(
        gpu,
        &world,
        &instances,
        scene,
        size,
        assignment,
        Some((&dir, name)),
    ) {
        Ok(v) => {
            if v["parity"] != true {
                eprintln!("sequence {name}: {v}");
                *failed = true;
            }
            println!(
                "sequence {name:<24} flip mean={:.5} p99={:.4} identical={} overflows={}",
                v["flip_mean"].as_f64().unwrap_or(0.0),
                v["flip_p99"].as_f64().unwrap_or(0.0),
                v["byte_identical"],
                v["cluster_overflows"]
            );
            results.push(json!({"step": name, "result": v}));
        }
        Err(e) => {
            eprintln!("sequence {name}: {e}");
            *failed = true;
        }
    };
    let cfg = clustered().0;
    let mut scene = scenario.scene.clone();
    step("default", &scene, cfg, failed);
    // A hard cut close to a light, with the near plane through its volume.
    let light = scene.point_lights[5].position;
    scene.camera.eye = light + Vec3::new(0.3, 0.0, 0.3);
    scene.camera.target = light + Vec3::new(3.0, -1.0, -2.0);
    step("cut_inside_light", &scene, cfg, failed);
    // Near-plane crossing: the eye inside a crate at the floor.
    scene.camera.eye = Vec3::new(-20.0, 0.3, 10.0);
    scene.camera.target = Vec3::new(-18.0, 0.5, -5.0);
    step("near_plane", &scene, cfg, failed);
    // A cut high over the hall, looking down.
    scene.camera.eye = Vec3::new(10.0, 6.0, 30.0);
    scene.camera.target = Vec3::new(-5.0, 0.0, -10.0);
    step("cut_high", &scene, cfg, failed);
    // Lights moved, some behind the camera.
    let mut moved = scenario.scene.clone();
    for (i, l) in moved.point_lights.iter_mut().enumerate().take(30) {
        l.position.z += 45.0 * rand01(i, 41);
    }
    step("lights_moved", &moved, cfg, failed);
    // Lights removed: half of them and every spot light.
    let mut fewer = scenario.scene.clone();
    fewer.point_lights.truncate(fewer.point_lights.len() / 2);
    fewer.spot_lights.clear();
    step("lights_removed", &fewer, cfg, failed);
    // No local lights at all.
    let mut none = scenario.scene.clone();
    none.point_lights.clear();
    none.spot_lights.clear();
    step("no_lights", &none, cfg, failed);
    // Forced overflow: 1 and 2 lights per cluster. Every overflowing cluster shades every
    // light: same output, the reference's cost there.
    for cap in [1u32, 2] {
        let name = format!("overflow_cap_{cap}");
        step(
            &name,
            &scenario.scene,
            LightAssignment::Clustered(ClusterConfig {
                max_lights_per_cluster: cap,
                ..ClusterConfig::default()
            }),
            failed,
        );
    }
    // Different grids.
    for (x, y, z) in [(32, 18, 48), (8, 4, 8), (1, 1, 1)] {
        let name = format!("grid_{x}x{y}x{z}");
        step(
            &name,
            &scenario.scene,
            LightAssignment::Clustered(ClusterConfig {
                x,
                y,
                z,
                ..ClusterConfig::default()
            }),
            failed,
        );
    }
    results
}

/// The reference frame of `scenario` against Bevy's, FLIP with D3's thresholds.
fn compare_bevy(gpu: &Headless, out: &Path, scenario: &Scenario) -> Result<Value, String> {
    let dir = out.join("bevy");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let (bevy_png, _, _) = bevy_side::run_in_child(scenario.name, &dir, 1)?;
    let bevy = image::open(&bevy_png)
        .map_err(|e| e.to_string())?
        .to_rgba8();
    let (world, instances, _) =
        content::upload(gpu.renderer(), &scenario.assets, &scenario.objects);
    let renderer = gpu.renderer();
    let mut viewport = Viewport::new(renderer, scenario.width, scenario.height);
    viewport
        .set_light_assignment(renderer, LightAssignment::PerFragment)
        .map_err(|e| e.to_string())?;
    let reference = gpu
        .render(&mut viewport, &world, &scenario.scene, &instances)
        .map_err(|e| e.to_string())?;
    let mut clustered_view = Viewport::new(renderer, scenario.width, scenario.height);
    clustered_view
        .set_light_assignment(renderer, clustered().0)
        .map_err(|e| e.to_string())?;
    let clustered = gpu
        .render(&mut clustered_view, &world, &scenario.scene, &instances)
        .map_err(|e| e.to_string())?;
    let score = |rgba: &[u8], tag: &str| -> Result<Value, String> {
        let f = flip(scenario.width, scenario.height, bevy.as_raw(), rgba);
        save_png(
            &dir.join(format!("{}.{tag}.png", scenario.name)),
            scenario.width,
            scenario.height,
            rgba,
        )?;
        save_png(
            &dir.join(format!("{}.{tag}.flip.png", scenario.name)),
            scenario.width,
            scenario.height,
            &f.heatmap,
        )?;
        Ok(json!({
            "flip_mean": f.mean,
            "flip_p99": f.p99,
            "parity": f.mean <= crate::FLIP_MEAN_MAX && f.p99 <= crate::FLIP_P99_MAX,
        }))
    };
    Ok(json!({
        "scene": scenario.name,
        "reference_vs_bevy": score(&reference.rgba, "reference")?,
        "clustered_vs_bevy": score(&clustered.rgba, "clustered")?,
    }))
}

fn fingerprint(gpu: &Headless) -> Value {
    json!({
        "adapter": gpu.adapter.name,
        "device_type": format!("{:?}", gpu.adapter.device_type),
        "backend": format!("{:?}", gpu.adapter.backend),
        "driver": gpu.adapter.driver,
        "driver_info": gpu.adapter.driver_info,
        "clustered_supported": gpu.renderer().supports_clustered_lighting(),
        "pipeline_statistics": gpu.device.features().contains(wgpu::Features::PIPELINE_STATISTICS_QUERY),
        "max_storage_buffers_per_shader_stage": gpu.device.limits().max_storage_buffers_per_shader_stage,
    })
}

pub fn run(options: &Options) -> ExitCode {
    let gpu = match Headless::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("Forward+ benchmark adapter: {e}");
            return ExitCode::FAILURE;
        }
    };
    let info = fingerprint(&gpu);
    println!("Forward+ #72: {info}");
    let out = options.out.join("forward_plus");
    let names = options.only.clone();
    let frames = options.frames as usize;
    let rounds = options.gpu_rounds as usize;
    let mut failed = false;
    let mut cases = Vec::new();
    for scenario in scenes() {
        if !names.is_empty() && !names.iter().any(|n| n == scenario.name) {
            continue;
        }
        let (world, instances, _) =
            content::upload(gpu.renderer(), &scenario.assets, &scenario.objects);
        let lights = scenario.scene.point_lights.len() + scenario.scene.spot_lights.len();
        let mut runs: [Vec<Round>; 2] = [Vec::new(), Vec::new()];
        let paths = [REFERENCE, clustered()];
        for round in 0..rounds {
            let order = if round % 2 == 0 { [0, 1] } else { [1, 0] };
            for i in order {
                match timed(&gpu, &world, &instances, &scenario, paths[i], frames) {
                    Ok(r) => runs[i].push(r),
                    Err(e) => {
                        eprintln!("{}/{}: {e}", scenario.name, paths[i].1);
                        failed = true;
                    }
                }
            }
        }
        let spread: Vec<Value> = (0..2)
            .map(|i| {
                json!({
                    "gpu_ms": runs[i].iter().map(|r| r.gpu_ms).collect::<Vec<_>>(),
                    "prepare_ms": runs[i].iter().map(|r| r.prepare_ms).collect::<Vec<_>>(),
                })
            })
            .collect();
        for r in &mut runs {
            r.sort_by(|a, b| a.gpu_ms.total_cmp(&b.gpu_ms));
        }
        let [refs, clus] = &runs;
        let (Some(reference), Some(cluster_round)) = (refs.first(), clus.first()) else {
            failed = true;
            continue;
        };
        let fragments = |assignment| {
            fragment_invocations(&gpu, &world, &instances, &scenario, assignment).unwrap_or(None)
        };
        let (ref_invocations, cluster_invocations) = (
            fragments(LightAssignment::PerFragment),
            fragments(clustered().0),
        );
        let diag = match cluster_diagnostic(&gpu, &world, &instances, &scenario) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("{} diagnostic: {e}", scenario.name);
                None
            }
        };
        let compare = compare_frame(
            &gpu,
            &world,
            &instances,
            &scenario.scene,
            (scenario.width, scenario.height),
            clustered().0,
            Some((&out.join(scenario.name), "frame")),
        );
        let compare = match compare {
            Ok(v) => {
                if v["parity"] != true {
                    eprintln!("{}: clustered differs: {v}", scenario.name);
                    failed = true;
                }
                v
            }
            Err(e) => {
                eprintln!("{}: {e}", scenario.name);
                failed = true;
                json!(null)
            }
        };
        let pixels = u64::from(scenario.width) * u64::from(scenario.height);
        let (diag_fragments, diag_lights) = diag.unwrap_or((0, 0));
        let lights_per_fragment = diag_lights as f64 / diag_fragments.max(1) as f64;
        println!(
            "{:<20} L={lights:<5} gpu {:>7.3} → {:>7.3} ms (lit {:>7.3} → {:>7.3}) cpu prepare {:>6.3} → {:>6.3} encode {:>6.3} → {:>6.3} | invocations/px {:.2} | lights/frag {:.2} vs {lights} | list {} KB",
            scenario.name,
            reference.gpu_ms,
            cluster_round.gpu_ms,
            reference
                .passes
                .iter()
                .filter(|(k, _)| k.starts_with("lit"))
                .map(|(_, v)| *v)
                .sum::<f64>(),
            cluster_round
                .passes
                .iter()
                .filter(|(k, _)| k.starts_with("lit"))
                .map(|(_, v)| *v)
                .sum::<f64>(),
            reference.prepare_ms,
            cluster_round.prepare_ms,
            reference.encode_ms,
            cluster_round.encode_ms,
            ref_invocations.unwrap_or(0) as f64 / pixels as f64,
            lights_per_fragment,
            cluster_round.memory.clustered_lighting_bytes / 1024,
        );
        cases.push(json!({
            "scene": scenario.name,
            "lights": lights,
            "instances": instances.len(),
            "resolution": [scenario.width, scenario.height],
            "reference": summary(reference, "per_fragment"),
            "clustered": summary(cluster_round, "clustered"),
            "rounds": { "reference": spread[0], "clustered": spread[1] },
            "fragment_shader_invocations": {
                "reference": ref_invocations,
                "clustered": cluster_invocations,
                "pixels": pixels,
            },
            "diagnostic": {
                "shaded_fragments": diag_fragments,
                "lights_evaluated": diag_lights,
                "lights_per_shaded_fragment_clustered": lights_per_fragment,
                "lights_per_shaded_fragment_reference": lights,
            },
            "compare": compare,
        }));
    }

    let validation = validate_scenarios(&gpu, &out, &mut failed);
    let sequence = validate_sequence(&gpu, &out, &mut failed);
    // The Bevy comparison: only when the matrix ran, to keep `--only` runs short.
    let mut bevy = Vec::new();
    for scenario in bevy_scenes() {
        if !names.is_empty() && !names.iter().any(|n| n == scenario.name) {
            continue;
        }
        match compare_bevy(&gpu, &out, &scenario) {
            Ok(v) => {
                println!("bevy {} {v}", scenario.name);
                bevy.push(v);
            }
            Err(e) => {
                eprintln!("bevy {}: {e}", scenario.name);
                failed = true;
            }
        }
    }

    let _ = std::fs::create_dir_all(&out);
    let report = json!({
        "experiment": "render-engine#72",
        "environment": info,
        "frames_per_run": frames,
        "rounds": rounds,
        "cases": cases,
        "validation_scenarios": validation,
        "validation_sequence": sequence,
        "bevy": bevy,
        "notes": "Fastest round by GPU pass sum; CPU medians of that round.",
    });
    let path = out.join("report.json");
    match serde_json::to_vec_pretty(&report) {
        Ok(bytes) => {
            if let Err(e) = std::fs::write(&path, bytes) {
                eprintln!("{}: {e}", path.display());
                failed = true;
            }
        }
        Err(e) => {
            eprintln!("cannot serialize report: {e}");
            failed = true;
        }
    }
    println!("forward+ report: {}", path.display());
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
