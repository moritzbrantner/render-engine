//! Issue #59: isolated Bevy 0.20.1 occlusion experiment. No production Hi-Z,
//! changes to regular scenario gates, or benchmark-specific dependency.

use crate::{Options, bevy_side, flip, median, save_png, scenarios};
use bevy_side::{OcclusionMode as Mode, OcclusionView as View};
use render_engine::{headless::Headless, renderer::Viewport};
use serde_json::{Value, json};
use std::{path::Path, process::ExitCode};

fn scenario(name: &str) -> Option<scenarios::Scenario> {
    match name {
        "many_instances" => Some(scenarios::many_instances()),
        "occluded_instances" => Some(scenarios::occluded_instances()),
        "occluded_instances_50" => Some(scenarios::occluded_instances_density(2)),
        "occluded_instances_25" => Some(scenarios::occluded_instances_density(4)),
        _ => None,
    }
}

fn mode(id: &str) -> Option<Mode> {
    Mode::ALL.into_iter().find(|m| m.id() == id)
}

fn view(id: &str) -> Option<View> {
    View::ALL.into_iter().find(|v| v.id() == id)
}

fn directory(out: &Path, name: &str, view: View) -> std::path::PathBuf {
    out.join("occlusion").join(name).join(view.id())
}

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}

fn measured(value: &Value) -> Option<f64> {
    median(&numbers(value))
}

fn pass_medians(value: &Value) -> Value {
    let mut result = serde_json::Map::new();
    if let Some(passes) = value.as_object() {
        for (name, samples) in passes {
            if let Some(ms) = measured(samples) {
                result.insert(name.clone(), json!(ms));
            }
        }
    }
    Value::Object(result)
}

/// A measured *bracket*, not a fitted universal scene-complexity threshold.
/// The open-field control is deliberately excluded because it has a different
/// occlusion fraction, even though it has the same number of objects.
fn crossover(cases: &[Value]) -> Value {
    let mut observations: Vec<(u64, bool)> = cases
        .iter()
        .filter(|case| case["scenario"].as_str().unwrap_or("").starts_with("occluded_"))
        .filter_map(|case| {
            let stationary = case["views"].as_array()?.iter()
                .find(|v| v["view"] == "stationary")?;
            let valid = stationary["visibility"].as_array()
                .is_some_and(|a| !a.is_empty() && a.iter().all(|c| c["flip"]["parity"] == true));
            if !valid { return None; }
            let depth = stationary["hiz_over_depth"].as_f64()?;
            let total = stationary["hiz_over_baseline"].as_f64()?;
            Some((case["logical_objects"].as_u64()?, depth < 1.0 && total < 1.0))
        })
        .collect();
    observations.sort_by_key(|(objects, _)| *objects);
    let first_win = observations.iter().position(|(_, wins)| *wins);
    json!({
        "tested": observations.iter().map(|(objects, wins)| json!({
            "logical_objects": objects,
            "hiz_wins_depth_and_baseline_with_flip_parity": wins,
        })).collect::<Vec<_>>(),
        "smallest_observed_win": first_win.map(|i| observations[i].0),
        "previous_tested_count": first_win.and_then(|i| i.checked_sub(1)).map(|i| observations[i].0),
        "note": "This only brackets density under the fixed occluder layout; use same-hardware raw round timings and more points before claiming a crossover.",
    })
}

/// Internal child mode. Each Bevy renderer runs in a fresh process, as the
/// existing oracle requires. Keep raw samples and snapshots for inspection.
pub fn child(
    name: &str,
    mode_id: &str,
    view_id: &str,
    round: u32,
    options: &Options,
) -> Result<(), String> {
    let scene = scenario(name).ok_or_else(|| format!("unknown scene {name}"))?;
    let mode = mode(mode_id).ok_or_else(|| format!("unknown mode {mode_id}"))?;
    let view = view(view_id).ok_or_else(|| format!("unknown view {view_id}"))?;
    let dir = directory(&options.out, name, view);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let (width, height) = (scene.width, scene.height);
    let run = bevy_side::run_occlusion_bench(scene, options.frames, mode, view);
    if run.gpu_ms.is_empty() {
        return Err("Bevy supplied no GPU timestamp diagnostics".into());
    }
    if run.captures.len() != 7 {
        return Err(format!("expected 7 captures, got {}", run.captures.len()));
    }
    for (frame, image) in &run.captures {
        save_png(
            &dir.join(format!("{mode_id}.r{round}.f{frame}.png")),
            width,
            height,
            image,
        )?;
    }
    let result = json!({
        "mode": mode_id,
        "view": view_id,
        "round": round,
        "occlusion_supported": run.occlusion_supported,
        "gpu_ms": run.gpu_ms,
        "cpu_frame_interval_ms": run.cpu_frame_ms,
        "cpu_render_span_ms": run.cpu_render_span_ms,
        "pipeline_counts": run.pipeline_counts,
        "pass_gpu_ms": run.pass_gpu_ms,
        "spans": run.spans,
        "captures": run.captures.iter().map(|(f, _)| f).collect::<Vec<_>>(),
    });
    std::fs::write(
        dir.join(format!("{mode_id}.r{round}.json")),
        serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

fn run_child(
    out: &Path,
    name: &str,
    mode: Mode,
    view: View,
    round: u32,
    frames: u32,
) -> Result<Value, String> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let status = std::process::Command::new(exe)
        .arg("--occlusion-child")
        .args([name, mode.id(), view.id(), &round.to_string()])
        .arg("--out")
        .arg(out)
        .args(["--frames", &frames.to_string()])
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err(format!("{name}/{} /{} round {round}: {status}", view.id(), mode.id()));
    }
    let path = directory(out, name, view).join(format!("{}.r{round}.json", mode.id()));
    serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

fn image_flip(out: &Path, name: &str, view: View, mode: Mode, frame: u32) -> Result<Value, String> {
    let dir = directory(out, name, view);
    let baseline = image::open(dir.join(format!("baseline.r0.f{frame}.png")))
        .map_err(|e| e.to_string())?
        .to_rgba8();
    let candidate = image::open(dir.join(format!("{}.r0.f{frame}.png", mode.id())))
        .map_err(|e| e.to_string())?
        .to_rgba8();
    if baseline.dimensions() != candidate.dimensions() {
        return Err("capture dimensions differ".into());
    }
    let metrics = flip(
        baseline.width(),
        baseline.height(),
        baseline.as_raw(),
        candidate.as_raw(),
    );
    save_png(
        &dir.join(format!("{}.f{frame}.flip.png", mode.id())),
        baseline.width(),
        baseline.height(),
        &metrics.heatmap,
    )?;
    Ok(json!({
        "frame": frame,
        "mean": metrics.mean,
        "p99": metrics.p99,
        "parity": metrics.mean <= crate::FLIP_MEAN_MAX && metrics.p99 <= crate::FLIP_P99_MAX,
    }))
}

/// Compare against Render Engine's present frustum-culled, *unculled by Hi-Z*
/// output, independently of the three Bevy modes. Only the stationary camera
/// can be compared exactly without knowing the screenshot callback's latency.
fn engine_control(gpu: &Headless, scene: &scenarios::Scenario, out: &Path, frames: u32) -> Value {
    let (mut world, instances, _) =
        crate::content::upload(gpu.renderer(), &scene.assets, &scene.objects);
    let mut viewport = Viewport::new(gpu.renderer(), scene.width, scene.height);
    let render = gpu.render(&mut viewport, &world, &scene.scene, &instances);
    let timing = gpu.gpu_frame_times(
        &mut viewport,
        &mut world,
        &scene.scene,
        &instances,
        frames as usize,
        None,
    );
    let mut result = json!({
        "gpu_ms": timing.ok().and_then(|t| median(&t.total)),
        "logical_objects": scene.objects.len(),
    });
    if let Ok(render) = render {
        let path = directory(out, scene.name, View::Stationary);
        let _ = std::fs::create_dir_all(&path);
        let _ = save_png(
            &path.join("engine.png"),
            render.width,
            render.height,
            &render.rgba,
        );
        if let Ok(bevy) = image::open(path.join("baseline.r0.f30.png")) {
            let bevy = bevy.to_rgba8();
            if (bevy.width(), bevy.height()) == (render.width, render.height) {
                let metrics = flip(render.width, render.height, bevy.as_raw(), &render.rgba);
                result["bevy_engine_flip_mean"] = json!(metrics.mean);
                result["bevy_engine_flip_p99"] = json!(metrics.p99);
            }
        }
    }
    result
}

pub fn run(options: &Options) -> ExitCode {
    if options.frames == 0 {
        eprintln!("--occlusion-bench requires --frames greater than zero");
        return ExitCode::FAILURE;
    }
    let gpu = match Headless::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("occlusion benchmark adapter: {e}");
            return ExitCode::FAILURE;
        }
    };
    let software = gpu.adapter.device_type == render_engine::wgpu::DeviceType::Cpu;
    println!(
        "Occlusion #59: {} ({:?}, {:?}); timings {}",
        gpu.adapter.name,
        gpu.adapter.device_type,
        gpu.adapter.backend,
        if software { "advisory (software rasterizer)" } else { "hardware GPU" },
    );
    let mut names = vec!["many_instances", "occluded_instances"];
    if options.occlusion_sweep {
        names.extend(["occluded_instances_25", "occluded_instances_50"]);
    }
    if options.only.iter().any(|n| !names.contains(&n.as_str())) {
        eprintln!("--occlusion-bench supports --only many_instances|occluded_instances");
        return ExitCode::FAILURE;
    }
    let mut report = Vec::new();
    let mut failed = false;
    for name in names {
        if !options.only.is_empty() && !options.only.iter().any(|n| n == name) {
            continue;
        }
        let Some(scene) = scenario(name) else { continue };
        let mut views = Vec::new();
        for view in View::ALL {
            // Stationary is sufficient for the scaling sweep; motion/cut are
            // exercised on both full-size control and occlusion-heavy cases.
            if name.ends_with("_25") || name.ends_with("_50") {
                if view != View::Stationary { continue; }
            }
            let mut rounds = [Vec::new(), Vec::new(), Vec::new()];
            for round in 0..options.occlusion_rounds {
                let order = if round % 2 == 0 {
                    Mode::ALL
                } else {
                    [Mode::Hiz, Mode::Depth, Mode::Baseline]
                };
                for mode in order {
                    match run_child(&options.out, name, mode, view, round, options.frames) {
                        Ok(run) => rounds[mode as usize].push(run),
                        Err(e) => {
                            eprintln!("{e}");
                            failed = true;
                        }
                    }
                }
            }
            let summary = Mode::ALL.map(|mode| {
                let fastest = rounds[mode as usize]
                    .iter()
                    .filter_map(|v| measured(&v["gpu_ms"]).map(|ms| (ms, v)))
                    .min_by(|(a, _), (b, _)| a.total_cmp(b));
                match fastest {
                    Some((ms, raw)) => json!({
                        "mode": mode.id(),
                        "occlusion_supported": raw["occlusion_supported"],
                        "gpu_ms": ms,
                        "cpu_frame_interval_ms": measured(&raw["cpu_frame_interval_ms"]),
                        "cpu_render_span_ms": measured(&raw["cpu_render_span_ms"]),
                        "pipeline_counts": pass_medians(&raw["pipeline_counts"]),
                        "gpu_pass_ms": pass_medians(&raw["pass_gpu_ms"]),
                        "gpu_round_medians_ms": rounds[mode as usize].iter().map(|r| measured(&r["gpu_ms"])).collect::<Vec<_>>(),
                    }),
                    None => Value::Null,
                }
            });
            let mut captures = Vec::new();
            if rounds.iter().all(|r| !r.is_empty()) {
                for mode in [Mode::Depth, Mode::Hiz] {
                    for frame in [30, 31, 32, 33, 34, 36, 44] {
                        match image_flip(&options.out, name, view, mode, frame) {
                            Ok(metrics) => {
                                if metrics["parity"] != true {
                                    failed = true;
                                }
                                captures.push(json!({"mode":mode.id(), "flip":metrics}));
                            }
                            Err(e) => {
                                eprintln!("{name}/{:?}/{:?}/f{frame}: {e}", view, mode);
                                failed = true;
                            }
                        }
                    }
                }
            }
            let base_ms = summary[0]["gpu_ms"].as_f64();
            let depth_ms = summary[1]["gpu_ms"].as_f64();
            let hiz_ms = if summary[2]["occlusion_supported"] == true {
                summary[2]["gpu_ms"].as_f64()
            } else {
                println!("{name}/{}: Bevy GPU occlusion unsupported; no Hi-Z performance claim", view.id());
                None
            };
            println!(
                "{name:<20} {:<12} base={base_ms:?} depth={depth_ms:?} hiz={hiz_ms:?} hiz/depth={:?}",
                view.id(),
                hiz_ms.zip(depth_ms).map(|(h, d)| h / d),
            );
            views.push(json!({
                "view":view.id(),
                "modes":summary,
                "hiz_over_depth":hiz_ms.zip(depth_ms).map(|(h,d)| h/d),
                "hiz_over_baseline":hiz_ms.zip(base_ms).map(|(h,b)| h/b),
                "depth_over_baseline":depth_ms.zip(base_ms).map(|(d,b)| d/b),
                "visibility": captures,
            }));
        }
        // Captures must already exist before this independent engine comparison.
        let engine = engine_control(&gpu, &scene, &options.out, options.frames);
        if engine["bevy_engine_flip_mean"].is_null() {
            eprintln!("{name}: independent Render Engine image comparison unavailable");
            failed = true;
        } else if engine["bevy_engine_flip_mean"].as_f64().unwrap_or(f64::INFINITY)
            > f64::from(crate::FLIP_MEAN_MAX)
            || engine["bevy_engine_flip_p99"].as_f64().unwrap_or(f64::INFINITY)
                > f64::from(crate::FLIP_P99_MAX)
        {
            eprintln!("{name}: existing Render Engine / Bevy FLIP limits exceeded");
            failed = true;
        }
        report.push(json!({
            "scenario": name,
            "logical_objects":scene.objects.len(),
            "engine_control":engine,
            "views":views,
        }));
    }
    let report_path = options.out.join("occlusion").join("report.json");
    let result = json!({
        "experiment":"render-engine#59",
        "adapter":gpu.adapter.name,
        "device_type":format!("{:?}",gpu.adapter.device_type),
        "backend":format!("{:?}",gpu.adapter.backend),
        "software_timing_advisory":software,
        "frames":options.frames,
        "rounds":options.occlusion_rounds,
        "cpu_metric":"main-thread frame interval, NOT isolated submission cost",
        "submitted_work":"GPU indirect draw/instance count unavailable; shader and clipper invocations are reported when supported; logical object count is NOT submitted work",
        "crossover": crossover(&report),
        "cases":report,
    });
    let report_bytes = match serde_json::to_vec_pretty(&result) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("cannot serialize occlusion report: {e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = std::fs::write(&report_path, report_bytes) {
        eprintln!("{}: {e}", report_path.display());
        failed = true;
    }
    println!("occlusion report: {}", report_path.display());
    if failed { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}
