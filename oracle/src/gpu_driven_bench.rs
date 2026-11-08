//! Issue #71: bounded experiment comparing the reference CPU frustum culling + instancing path
//! with the opt-in `Culling::GpuIndirect` path, on identical geometry, materials, lights and
//! shadows (see `docs/gpu-driven-experiment.md`). Diagnostic only: it does not touch the regular
//! scenario gates, and the CPU path stays the renderer's default and fallback.

use crate::{Options, content, flip, save_png, scenarios};
use render_engine::{
    glam::{Quat, Vec3},
    headless::Headless,
    material::StandardMaterial,
    mesh::{MeshData, Shape},
    renderer::{Culling, FrameTimestamps, PrepareWork, Viewport, ViewportMemory},
    scene::{LightShadows, PointLight, Scene, SpotLight},
    wgpu,
    world::{Instance, World},
};
use scenarios::Scenario;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, process::ExitCode, time::Instant};

/// Untimed frames each run renders first (GPU clocks, buffer growth).
const WARMUP_FRAMES: usize = 60;
/// Frames submitted between timestamp readbacks.
const BATCH: u32 = 32;
/// Frames at which a moving or cutting view is captured for the image comparison.
const CAPTURE_FRAMES: [usize; 6] = [0, 29, 30, 31, 32, 60];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum View {
    /// The scenario's own camera: a wedge of the field.
    Wedge,
    /// Elevated over the field: most instances visible.
    High,
    /// At the field's edge looking away: few instances visible.
    Low,
    /// A deterministic orbit, a new camera every frame.
    Moving,
    /// Hard cut between `High` and `Low` every 30 frames.
    Cuts,
}

impl View {
    const ALL: [Self; 5] = [Self::Wedge, Self::High, Self::Low, Self::Moving, Self::Cuts];

    fn id(self) -> &'static str {
        match self {
            Self::Wedge => "wedge",
            Self::High => "high",
            Self::Low => "low",
            Self::Moving => "moving",
            Self::Cuts => "cuts",
        }
    }

    fn apply(self, scene: &mut Scene, default_eye: Vec3, default_target: Vec3, frame: usize) {
        let high = (Vec3::new(0.0, 90.0, 60.0), Vec3::new(0.0, 0.0, -140.0));
        let low = (Vec3::new(0.0, 4.0, 14.0), Vec3::new(0.0, 4.0, 100.0));
        let (eye, target) = match self {
            Self::Wedge => (default_eye, default_target),
            Self::High => high,
            Self::Low => low,
            Self::Moving => {
                let t = frame as f32 * 0.05;
                (
                    Vec3::new(120.0 * t.sin(), 30.0, -100.0 + 150.0 * t.cos()),
                    Vec3::new(0.0, 0.0, -140.0),
                )
            }
            Self::Cuts => {
                if (frame / 30).is_multiple_of(2) {
                    high
                } else {
                    low
                }
            }
        };
        scene.camera.eye = eye;
        scene.camera.target = target;
    }
}

/// A bench scene: a name, the instances and the scene description.
fn scenes() -> Vec<Scenario> {
    let many = scenarios::many_instances;
    let mut out = Vec::new();

    let mut small = many();
    small.name = "small_1k";
    small.objects = small
        .objects
        .into_iter()
        .enumerate()
        .filter_map(|(i, o)| (i == 0 || (i - 1) % 22 == 0).then_some(o))
        .collect();
    out.push(small);

    let mut base = many();
    base.name = "instances_22k";
    out.push(base);

    let mut dense = many();
    dense.name = "dense_90k";
    dense.objects = repeated(
        dense.objects,
        &[(0.0, 0.0), (1.25, 0.0), (0.0, 1.25), (1.25, 1.25)],
        1.0,
    );
    out.push(dense);

    let mut denser = many();
    denser.name = "dense_202k";
    let steps = [0.0, 0.833, 1.667];
    let offsets: Vec<(f32, f32)> = steps
        .iter()
        .flat_map(|&x| steps.iter().map(move |&z| (x, z)))
        .collect();
    denser.objects = repeated(denser.objects, &offsets, 0.6);
    out.push(denser);

    // Many mesh/material runs: 22.4k instances over 3 meshes × 256 materials (the draw-bound
    // case: per-run costs, and one indirect draw per run and view on the GPU path).
    let mut groups = many();
    groups.name = "groups_22k";
    for (i, object) in groups.objects.iter_mut().enumerate().skip(1) {
        let h = (i % 256) as f32 / 256.0;
        object.material.base_color = Vec3::new(0.2 + 0.7 * h, 0.8 - 0.6 * h, 0.5);
    }
    out.push(groups);
    out
}

/// Keeps the ground (index 0) once and repeats every other object at each offset, scaled.
fn repeated(
    objects: Vec<content::MeshInstance>,
    offsets: &[(f32, f32)],
    scale: f32,
) -> Vec<content::MeshInstance> {
    let mut out = Vec::with_capacity(objects.len() * offsets.len());
    let mut iter = objects.into_iter();
    out.extend(iter.next());
    for object in iter {
        for &(dx, dz) in offsets {
            let mut copy = object;
            copy.translation += Vec3::new(dx, 0.0, dz);
            copy.translation.y *= scale;
            copy.scale *= scale;
            out.push(copy);
        }
    }
    out
}

struct Round {
    prepare_ms: f64,
    encode_ms: f64,
    submit_ms: f64,
    gpu_ms: f64,
    gpu_interval_ms: f64,
    passes: BTreeMap<String, f64>,
    work: PrepareWork,
    kept_per_view: Vec<u32>,
    memory: ViewportMemory,
}

fn median(mut v: Vec<f64>) -> f64 {
    assert!(!v.is_empty());
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn median_passes(samples: BTreeMap<String, Vec<f64>>) -> BTreeMap<String, f64> {
    samples.into_iter().map(|(k, v)| (k, median(v))).collect()
}

/// Reads and maps `buffer`, blocking.
fn read(gpu: &Headless, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("map callback").expect("map");
    let bytes = slice.get_mapped_range().expect("mapped").to_vec();
    buffer.unmap();
    bytes
}

/// One timed run of one scene, view and culling mode: per-frame CPU stages and GPU timestamps.
#[allow(clippy::too_many_arguments)]
fn timed_run(
    gpu: &Headless,
    world: &World,
    instances: &[Instance],
    scene: &Scene,
    view: View,
    default: (Vec3, Vec3),
    culling: Culling,
    (width, height): (u32, u32),
    frames: usize,
) -> Result<Round, String> {
    let renderer = gpu.renderer();
    let mut viewport = Viewport::new(renderer, width, height);
    viewport
        .set_culling(renderer, culling)
        .map_err(|e| e.to_string())?;
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench target"),
        size: wgpu::Extent3d {
            width,
            height,
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
    let mut scene = scene.clone();

    // Warm up: prepares and submits with no timestamps.
    let prepare = |viewport: &mut Viewport, scene: &mut Scene, frame: usize| {
        view.apply(scene, default.0, default.1, frame);
        renderer.prepare(viewport, world, scene, instances);
    };
    let warm_start = Instant::now();
    let mut frame = 0;
    // As `Headless::gpu_frame_times`: long enough for the GPU to leave its idle clocks.
    while frame < WARMUP_FRAMES || warm_start.elapsed() < render_engine::headless::WARMUP {
        prepare(&mut viewport, &mut scene, frame);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        renderer.encode(&mut encoder, &viewport, world, &target, None);
        gpu.queue.submit([encoder.finish()]);
        frame += 1;
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());

    let per_frame = viewport.timestamp_queries();
    let labels = viewport.pass_labels();
    let batch = (wgpu::QUERY_SET_MAX_QUERIES / per_frame).clamp(1, BATCH);
    let count = per_frame * batch;
    let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("bench timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count,
    });
    let size = u64::from(count) * 8;
    let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench resolve"),
        size,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let period = f64::from(gpu.queue.get_timestamp_period());

    let mut prepare_ms = Vec::with_capacity(frames);
    let mut encode_ms = Vec::with_capacity(frames);
    let mut submit_ms = Vec::with_capacity(frames);
    let mut gpu_ms = Vec::with_capacity(frames);
    let mut interval_ms = Vec::with_capacity(frames);
    let mut pass_samples: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut done = 0usize;
    while done < frames {
        let n = batch.min((frames - done) as u32);
        for i in 0..n {
            let t0 = Instant::now();
            prepare(&mut viewport, &mut scene, frame);
            let t1 = Instant::now();
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            renderer.encode(
                &mut encoder,
                &viewport,
                world,
                &target,
                Some(FrameTimestamps {
                    query_set: &query_set,
                    first: i * per_frame,
                }),
            );
            if i + 1 == n {
                encoder.resolve_query_set(&query_set, 0..n * per_frame, &resolve, 0);
                encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, size);
            }
            let commands = encoder.finish();
            let t2 = Instant::now();
            gpu.queue.submit([commands]);
            let t3 = Instant::now();
            prepare_ms.push((t1 - t0).as_secs_f64() * 1e3);
            encode_ms.push((t2 - t1).as_secs_f64() * 1e3);
            submit_ms.push((t3 - t2).as_secs_f64() * 1e3);
            frame += 1;
        }
        let bytes = read(gpu, &readback);
        let ticks: Vec<u64> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u64::from_ne_bytes(*b))
            .collect();
        for f in ticks.chunks_exact(per_frame as usize).take(n as usize) {
            let mut total = 0.0;
            for (label, pair) in labels.iter().zip(f.as_chunks::<2>().0) {
                let ms = pair[1].wrapping_sub(pair[0]) as f64 * period / 1e6;
                total += ms;
                pass_samples.entry(label.to_string()).or_default().push(ms);
            }
            gpu_ms.push(total);
            interval_ms.push(f[f.len() - 1].wrapping_sub(f[0]) as f64 * period / 1e6);
        }
        done += n as usize;
    }

    // Per-view instance counts: the GPU path reads back the last frame, the CPU path's are in
    // `work`.
    let kept_per_view = renderer.read_gpu_culled(&viewport).unwrap_or_default();
    Ok(Round {
        prepare_ms: median(prepare_ms),
        encode_ms: median(encode_ms),
        submit_ms: median(submit_ms),
        gpu_ms: median(gpu_ms),
        gpu_interval_ms: median(interval_ms),
        passes: median_passes(pass_samples),
        work: viewport.work(),
        kept_per_view,
        memory: viewport.memory(),
    })
}

fn summary(round: &Round) -> Value {
    let work = round.work;
    json!({
        "cpu": {
            "prepare_ms": round.prepare_ms,
            "encode_and_finish_ms": round.encode_ms,
            "submit_ms": round.submit_ms,
            "total_ms": round.prepare_ms + round.encode_ms + round.submit_ms,
        },
        "gpu_ms": round.gpu_ms,
        "gpu_passes_ms": round.passes,
        // Includes wgpu's indirect-validation dispatches between passes; the pass sum does not.
        "gpu_interval_ms": round.gpu_interval_ms,
        "work": {
            "instances": work.instances,
            "views": work.views,
            "cpu_bounds_tests": work.bounds_tests,
            "gpu_bounds_tests": work.gpu_bounds_tests,
            "cpu_visible": work.visible,
            "cpu_shadow_visible": work.shadow_visible,
            "gpu_kept_per_view": round.kept_per_view,
            "draws": work.draws,
            "shadow_draws": work.shadow_draws,
            "upload_bytes": work.upload_bytes,
            "allocations": work.allocations,
        },
        "memory": {
            "gpu_buffer_bytes": round.memory.gpu_buffer_bytes,
            "gpu_culling_bytes": round.memory.gpu_culling_bytes,
            "cpu_scratch_bytes": round.memory.cpu_scratch_bytes,
        },
    })
}

/// Renders a single frame on each path; FLIP over the pair, exact equality also reported.
fn compare_frame(
    gpu: &Headless,
    world: &World,
    instances: &[Instance],
    scene: &Scene,
    cpu_view: &mut Viewport,
    gpu_view: &mut Viewport,
    dir: Option<(&Path, &str)>,
) -> Result<Value, String> {
    let a = gpu
        .render(cpu_view, world, scene, instances)
        .map_err(|e| e.to_string())?;
    let b = gpu
        .render(gpu_view, world, scene, instances)
        .map_err(|e| e.to_string())?;
    let metrics = flip(a.width, a.height, &a.rgba, &b.rgba);
    let equal = a.rgba == b.rgba;
    let parity = metrics.mean <= crate::FLIP_MEAN_MAX && metrics.p99 <= crate::FLIP_P99_MAX;
    if let Some((dir, name)) = dir
        && !equal
    {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        save_png(
            &dir.join(format!("{name}.cpu.png")),
            a.width,
            a.height,
            &a.rgba,
        )?;
        save_png(
            &dir.join(format!("{name}.gpu.png")),
            b.width,
            b.height,
            &b.rgba,
        )?;
        save_png(
            &dir.join(format!("{name}.flip.png")),
            a.width,
            a.height,
            &metrics.heatmap,
        )?;
    }
    let renderer = gpu.renderer();
    let counts = renderer.read_gpu_culled(gpu_view);
    let gpu_work = gpu_view.work();
    let cpu_work = cpu_view.work();
    let kept_gpu = counts
        .as_ref()
        .map(|c| c.iter().sum::<u32>() + gpu_work.visible + gpu_work.shadow_visible);
    let kept_cpu = cpu_work.visible + cpu_work.shadow_visible;
    Ok(json!({
        "flip_mean": metrics.mean,
        "flip_p99": metrics.p99,
        "parity": parity,
        "byte_identical": equal,
        "kept_cpu": kept_cpu,
        "kept_gpu": kept_gpu,
        "kept_equal": kept_gpu == Some(kept_cpu),
    }))
}

fn pair(gpu: &Headless, width: u32, height: u32) -> Result<(Viewport, Viewport), String> {
    let renderer = gpu.renderer();
    let cpu = Viewport::new(renderer, width, height);
    let mut gpu_view = Viewport::new(renderer, width, height);
    gpu_view
        .set_culling(renderer, Culling::GpuIndirect)
        .map_err(|e| e.to_string())?;
    Ok((cpu, gpu_view))
}

/// Every lighting scenario of the oracle through both paths (one frame each): opaque, masked,
/// blended and skinned materials, shadows, fog, sky and environment. The reference is the CPU
/// path, which the regular oracle holds to Bevy.
fn validate_scenarios(gpu: &Headless, out: &Path, failed: &mut bool) -> Vec<Value> {
    let dir = out.join("gpu_driven").join("scenarios");
    let mut results = Vec::new();
    for scenario in scenarios::all() {
        let (world, instances, _) =
            content::upload(gpu.renderer(), &scenario.assets, &scenario.objects);
        let r = pair(gpu, scenario.width, scenario.height).and_then(|(mut c, mut g)| {
            compare_frame(
                gpu,
                &world,
                &instances,
                &scenario.scene,
                &mut c,
                &mut g,
                Some((&dir, scenario.name)),
            )
        });
        match r {
            Ok(v) => {
                if v["parity"] != true || v["kept_equal"] != true {
                    eprintln!(
                        "{}: GPU path differs from the reference: {v}",
                        scenario.name
                    );
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

/// Near-plane crossings, cuts, mutation, removal, a mesh bounds edit and local-light shadows on
/// `instances_22k` with a point and a spot light, every step compared to the reference.
fn validate_sequence(gpu: &Headless, out: &Path, failed: &mut bool) -> Vec<Value> {
    let dir = out.join("gpu_driven").join("sequence");
    let mut scenario = scenarios::many_instances();
    scenario.scene.point_lights.push(PointLight {
        position: Vec3::new(-6.0, 5.0, -20.0),
        range: 40.0,
        intensity: 800_000.0,
        shadows: Some(LightShadows::POINT),
        ..PointLight::default()
    });
    scenario.scene.spot_lights.push(SpotLight {
        position: Vec3::new(8.0, 8.0, -10.0),
        direction: Vec3::new(-0.4, -1.0, -0.8),
        range: 50.0,
        intensity: 800_000.0,
        shadows: Some(LightShadows::SPOT),
        ..SpotLight::default()
    });
    let (mut world, mut instances, _) =
        content::upload(gpu.renderer(), &scenario.assets, &scenario.objects);
    // An extra edit target: a small cuboid just outside the default view's right edge.
    let small = MeshData::shape(Shape::Cuboid(Vec3::ONE));
    let edit_mesh = world.add_mesh(&small);
    let edit_material = world.add_material(&StandardMaterial::default());
    instances.push(Instance {
        mesh: edit_mesh,
        material: edit_material,
        world_from_local: render_engine::glam::Mat4::from_translation(Vec3::new(40.0, 0.6, -8.0)),
        skin: None,
    });
    let (mut cpu_view, mut gpu_view) = match pair(gpu, scenario.width, scenario.height) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            *failed = true;
            return Vec::new();
        }
    };
    let mut results = Vec::new();
    let mut step =
        |name: &str, world: &World, instances: &[Instance], scene: &Scene, failed: &mut bool| {
            match compare_frame(
                gpu,
                world,
                instances,
                scene,
                &mut cpu_view,
                &mut gpu_view,
                Some((&dir, name)),
            ) {
                Ok(v) => {
                    if v["parity"] != true || v["kept_equal"] != true {
                        eprintln!("sequence {name}: {v}");
                        *failed = true;
                    }
                    println!(
                        "sequence {name:<16} flip mean={:.5} p99={:.4} identical={}",
                        v["flip_mean"].as_f64().unwrap_or(0.0),
                        v["flip_p99"].as_f64().unwrap_or(0.0),
                        v["byte_identical"]
                    );
                    results.push(json!({"step": name, "result": v}));
                }
                Err(e) => {
                    eprintln!("sequence {name}: {e}");
                    *failed = true;
                }
            }
        };

    let mut scene = scenario.scene.clone();
    step("default", &world, &instances, &scene, failed);
    // A hard cut into the middle of the field, close enough that props cross the near plane.
    scene.camera.eye = Vec3::new(0.0, 0.8, -60.0);
    scene.camera.target = Vec3::new(5.0, 0.8, -90.0);
    step("cut_near_plane", &world, &instances, &scene, failed);
    // Cut to a high, wide view.
    scene.camera.eye = Vec3::new(0.0, 80.0, 40.0);
    scene.camera.target = Vec3::new(0.0, 0.0, -140.0);
    step("cut_high", &world, &instances, &scene, failed);
    // Back to the default camera for the mutation sequence.
    scene = scenario.scene.clone();
    step("cut_back", &world, &instances, &scene, failed);
    // Moving instances in and out of view.
    for (i, instance) in instances.iter_mut().enumerate().take(200).skip(1) {
        let moved = Vec3::new(i as f32 * 0.2 - 20.0, 2.0, -10.0);
        instance.world_from_local = render_engine::glam::Mat4::from_rotation_translation(
            Quat::from_rotation_y(i as f32),
            moved,
        );
    }
    step("mutation", &world, &instances, &scene, failed);
    // Removal shrinks the list.
    instances.drain(200..1200);
    step("removal", &world, &instances, &scene, failed);
    // Mesh bounds grow: the cuboid's bounds cross into the view.
    let mut grown = small.clone();
    grown.positions.iter_mut().for_each(|p| *p *= 80.0);
    world.set_mesh_positions_normals(edit_mesh, &grown.positions, &grown.normals);
    step("mesh_bounds", &world, &instances, &scene, failed);
    results
}

fn table(name: &str, view: View, cpu: &Round, gpu: &Round) {
    let total = |r: &Round| r.prepare_ms + r.encode_ms + r.submit_ms;
    println!(
        "{name:<14} {:<7} cpu total {:>7.3} → {:>7.3} ms (prepare {:>6.3}→{:>6.3}, encode {:>6.3}→{:>6.3}) | gpu {:>6.3} → {:>6.3} ms (cull {:.3}, interval {:.3}→{:.3}) | upload {:>7}→{:>7} B | draws {}→{}",
        view.id(),
        total(cpu),
        total(gpu),
        cpu.prepare_ms,
        gpu.prepare_ms,
        cpu.encode_ms,
        gpu.encode_ms,
        cpu.gpu_ms,
        gpu.gpu_ms,
        gpu.passes.get("gpu culling").copied().unwrap_or(0.0),
        cpu.gpu_interval_ms,
        gpu.gpu_interval_ms,
        cpu.work.upload_bytes,
        gpu.work.upload_bytes,
        cpu.work.draws + cpu.work.shadow_draws,
        gpu.work.draws + gpu.work.shadow_draws,
    );
}

pub fn run(options: &Options) -> ExitCode {
    let gpu = match Headless::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("GPU-driven benchmark adapter: {e}");
            return ExitCode::FAILURE;
        }
    };
    let software = gpu.adapter.device_type == wgpu::DeviceType::Cpu;
    let features = gpu.device.features();
    let info = json!({
        "adapter": gpu.adapter.name,
        "device_type": format!("{:?}", gpu.adapter.device_type),
        "backend": format!("{:?}", gpu.adapter.backend),
        "driver": gpu.adapter.driver,
        "driver_info": gpu.adapter.driver_info,
        "gpu_culling_supported": gpu.renderer().supports_gpu_culling(),
        "feature_indirect_first_instance": features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
        "adapter_multi_draw_indirect_count": gpu.adapter_features.contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT),
        "adapter_indirect_first_instance": gpu.adapter_features.contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
        "max_storage_buffers_per_shader_stage": gpu.device.limits().max_storage_buffers_per_shader_stage,
    });
    println!("GPU-driven #71: {info}");

    let names: Vec<String> = options.only.clone();
    let run_frames = options.frames as usize;
    let rounds = options.gpu_rounds as usize;
    let mut failed = false;
    let mut cases = Vec::new();
    for scenario in scenes() {
        if !names.is_empty() && !names.iter().any(|n| n == scenario.name) {
            continue;
        }
        let (world, instances, _) =
            content::upload(gpu.renderer(), &scenario.assets, &scenario.objects);
        let default = (scenario.scene.camera.eye, scenario.scene.camera.target);
        let mut views = Vec::new();
        for view in View::ALL {
            let mut results = [Vec::new(), Vec::new()];
            for round in 0..rounds {
                let order = if round % 2 == 0 {
                    [Culling::Cpu, Culling::GpuIndirect]
                } else {
                    [Culling::GpuIndirect, Culling::Cpu]
                };
                for culling in order {
                    match timed_run(
                        &gpu,
                        &world,
                        &instances,
                        &scenario.scene,
                        view,
                        default,
                        culling,
                        (scenario.width, scenario.height),
                        run_frames,
                    ) {
                        Ok(r) => results[culling as usize].push(r),
                        Err(e) => {
                            eprintln!("{}/{}: {e}", scenario.name, view.id());
                            failed = true;
                        }
                    }
                }
            }
            let fastest = |runs: &mut Vec<Round>| -> Option<Round> {
                runs.sort_by(|a, b| a.gpu_ms.total_cmp(&b.gpu_ms));
                if runs.is_empty() {
                    None
                } else {
                    Some(runs.remove(0))
                }
            };
            let [mut cpu_rounds, mut gpu_rounds] = results;
            let spread = |runs: &[Round]| {
                json!({
                    "gpu_ms": runs.iter().map(|r| r.gpu_ms).collect::<Vec<_>>(),
                    "prepare_ms": runs.iter().map(|r| r.prepare_ms).collect::<Vec<_>>(),
                    "encode_ms": runs.iter().map(|r| r.encode_ms).collect::<Vec<_>>(),
                })
            };
            let round_spread = json!({
                "cpu_path": spread(&cpu_rounds),
                "gpu_path": spread(&gpu_rounds),
            });
            let (Some(cpu), Some(gpu_round)) = (fastest(&mut cpu_rounds), fastest(&mut gpu_rounds))
            else {
                continue;
            };
            table(scenario.name, view, &cpu, &gpu_round);

            // The same frames on both paths: a capture covers the first frame, the frames
            // around the cut (29/30 are `High`/`Low` in `Cuts`) and the moving orbit's.
            let mut captures = Vec::new();
            match pair(&gpu, scenario.width, scenario.height) {
                Ok((mut cpu_view, mut gpu_view)) => {
                    let dir = options.out.join("gpu_driven").join(scenario.name);
                    let mut scene = scenario.scene.clone();
                    for frame in CAPTURE_FRAMES {
                        view.apply(&mut scene, default.0, default.1, frame);
                        let name = format!("{}.f{frame}", view.id());
                        match compare_frame(
                            &gpu,
                            &world,
                            &instances,
                            &scene,
                            &mut cpu_view,
                            &mut gpu_view,
                            Some((&dir, &name)),
                        ) {
                            Ok(v) => {
                                if v["parity"] != true || v["kept_equal"] != true {
                                    eprintln!("{}/{name}: GPU path differs: {v}", scenario.name);
                                    failed = true;
                                }
                                captures.push(json!({"frame": frame, "compare": v}));
                            }
                            Err(e) => {
                                eprintln!("{}/{name}: {e}", scenario.name);
                                failed = true;
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("{e}");
                    failed = true;
                }
            }
            views.push(json!({
                "view": view.id(),
                "cpu_path": summary(&cpu),
                "gpu_path": summary(&gpu_round),
                "rounds": round_spread,
                "captures": captures,
            }));
        }
        cases.push(json!({
            "scene": scenario.name,
            "instances": instances.len(),
            "views": views,
        }));
    }
    let scenarios_result = validate_scenarios(&gpu, &options.out, &mut failed);
    let sequence_result = validate_sequence(&gpu, &options.out, &mut failed);

    let out = options.out.join("gpu_driven");
    let _ = std::fs::create_dir_all(&out);
    let report = json!({
        "experiment": "render-engine#71",
        "environment": info,
        "software_timing_advisory": software,
        "frames_per_run": run_frames,
        "rounds": rounds,
        "resolution": [1280, 720],
        "validation_scenarios": scenarios_result,
        "validation_sequence": sequence_result,
        "cases": cases,
        "notes": "Fastest of the rounds by GPU pass-sum median; CPU times are medians of that round. gpu_interval_ms includes wgpu's indirect-validation dispatches.",
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
    println!("gpu-driven report: {}", path.display());
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
