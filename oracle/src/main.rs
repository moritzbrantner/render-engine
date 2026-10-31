//! Bevy oracle (technical decisions D4). For every parity scenario it renders the scene on Bevy
//! and on render-engine headless, compares the images with NVIDIA FLIP and compares GPU frame
//! time. It fails when the mean FLIP error exceeds `FLIP_MEAN_MAX` or its 99th percentile exceeds
//! `FLIP_P99_MAX`, or — on a hardware adapter — when render-engine's GPU frame time is more than
//! `GPU_MARGIN` × Bevy's.
//!
//! GPU frame time is the sum of a frame's pass spans on both sides (Bevy's top-level render
//! diagnostic spans; our per-pass timestamps), measured in alternating rounds, each side's time
//! being its fastest round's median frame.
//!
//! CI runs it on a software adapter (`WGPU_FORCE_FALLBACK_ADAPTER=1`, lavapipe): the image gate is
//! enforced there, GPU times are printed as advisory. Locally on a GPU both gates apply.
//!
//! ```text
//! oracle                         every scenario, both gates
//! oracle --list                  scenario names
//! oracle --only sun_shadows      run only these scenarios
//! oracle --no-timing             image gate only
//! oracle --out DIR               write <scenario>.{bevy,engine,flip}.png and report.json (default target/oracle)
//! ```

mod bevy_side;
mod content;
mod scenarios;

use render_engine::{headless::Headless, renderer::Viewport};
use std::{path::PathBuf, process::ExitCode};

/// Mean FLIP error (0 = identical, 1 = maximally different) above which a scenario fails.
const FLIP_MEAN_MAX: f32 = 0.005;
/// 99th-percentile per-pixel FLIP error above which a scenario fails.
const FLIP_P99_MAX: f32 = 0.05;
/// render-engine may be at most this much slower than Bevy (median GPU ms per frame).
const GPU_MARGIN: f64 = 1.10;
/// Frames timed per side and round.
const TIMED_FRAMES: u32 = 300;
/// Timing rounds per scenario on a hardware adapter (each side once per round, alternating).
const TIMING_ROUNDS: u32 = 2;
/// Extra rounds for a scenario still over the margin before it fails.
const RETRIES: u32 = 2;

struct Options {
    only: Vec<String>,
    timing: bool,
    list: bool,
    out: PathBuf,
    bevy_side: Option<String>,
    frames: u32,
}

fn options() -> Result<Options, String> {
    let mut o = Options {
        only: Vec::new(),
        timing: true,
        list: false,
        out: PathBuf::from("target/oracle"),
        bevy_side: None,
        frames: TIMED_FRAMES,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--only" => o
                .only
                .push(args.next().ok_or("--only needs a scenario name")?),
            "--no-timing" => o.timing = false,
            "--list" => o.list = true,
            "--out" => o.out = args.next().ok_or("--out needs a directory")?.into(),
            "--bevy-side" => o.bevy_side = Some(args.next().ok_or("--bevy-side needs a name")?),
            "--frames" => {
                o.frames = args
                    .next()
                    .and_then(|n| n.parse().ok())
                    .ok_or("--frames needs a number")?
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(o)
}

/// The fastest of several rounds' medians.
fn fastest(rounds: &[Option<f64>]) -> Option<f64> {
    rounds.iter().flatten().copied().reduce(f64::min)
}

fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    Some(v[v.len() / 2])
}

fn save_png(path: &std::path::Path, w: u32, h: u32, rgba: &[u8]) -> Result<(), String> {
    image::save_buffer(path, rgba, w, h, image::ExtendedColorType::Rgba8).map_err(|e| e.to_string())
}

/// Child process: render one scenario on Bevy, write `<name>.bevy.png` and `<name>.bevy.json`.
fn bevy_child(name: &str, o: &Options) -> Result<(), String> {
    let scenario = scenarios::all()
        .into_iter()
        .find(|s| s.name == name)
        .ok_or_else(|| format!("unknown scenario {name}"))?;
    let run = bevy_side::run(scenario, o.frames);
    let (w, h, rgba) = run.rgba.ok_or("Bevy produced no screenshot")?;
    std::fs::create_dir_all(&o.out).map_err(|e| e.to_string())?;
    save_png(&o.out.join(format!("{name}.bevy.png")), w, h, &rgba)?;
    let json = serde_json::json!({ "gpu_ms": run.gpu_ms, "spans": run.spans });
    std::fs::write(o.out.join(format!("{name}.bevy.json")), json.to_string())
        .map_err(|e| e.to_string())
}

struct Flip {
    mean: f32,
    /// 99th-percentile per-pixel error: catches local defects (a missing shadow) that the
    /// whole-image mean dilutes.
    p99: f32,
    heatmap: Vec<u8>,
}

fn flip(w: u32, h: u32, reference: &[u8], test: &[u8]) -> Flip {
    let rgb = |rgba: &[u8]| -> Vec<u8> {
        rgba.as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect()
    };
    let reference = nv_flip::FlipImageRgb8::with_data(w, h, &rgb(reference));
    let test = nv_flip::FlipImageRgb8::with_data(w, h, &rgb(test));
    let error = nv_flip::flip(reference, test, nv_flip::DEFAULT_PIXELS_PER_DEGREE);
    let mut pool = nv_flip::FlipPool::from_image(&error);
    let heatmap = error
        .apply_color_lut(&nv_flip::magma_lut())
        .to_vec()
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect();
    Flip {
        mean: pool.mean(),
        p99: pool.get_percentile(0.99, false),
        heatmap,
    }
}

fn main() -> ExitCode {
    let o = match options() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    if let Some(name) = &o.bevy_side {
        return match bevy_child(name, &o) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("bevy side: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let scenarios = scenarios::all();
    if o.list {
        for s in &scenarios {
            println!("{:<24} {}", s.name, s.capability);
        }
        return ExitCode::SUCCESS;
    }
    if let Some(unknown) = o
        .only
        .iter()
        .find(|n| !scenarios.iter().any(|s| s.name == n.as_str()))
    {
        eprintln!("unknown scenario {unknown} (see --list)");
        return ExitCode::FAILURE;
    }
    if let Err(e) = std::fs::create_dir_all(&o.out) {
        eprintln!("cannot create {}: {e}", o.out.display());
        return ExitCode::FAILURE;
    }

    let gpu = match Headless::new() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("render-engine: {e}");
            return ExitCode::FAILURE;
        }
    };
    let software = gpu.adapter.device_type == render_engine::wgpu::DeviceType::Cpu;
    let gate_timing = o.timing && !software;
    println!(
        "adapter: {} ({:?}, {:?}); GPU-time gate: {}",
        gpu.adapter.name,
        gpu.adapter.device_type,
        gpu.adapter.backend,
        if gate_timing {
            format!("enforced (≤ {GPU_MARGIN:.2}× Bevy)")
        } else {
            "advisory".into()
        }
    );
    println!(
        "{:<24} {:<20} {:>9} {:>9} {:>12} {:>12} {:>7}  result",
        "scenario", "capability", "FLIP mean", "FLIP p99", "engine ms", "bevy ms", "ratio"
    );

    let mut failed = false;
    let mut report = Vec::new();
    for s in scenarios
        .iter()
        .filter(|s| o.only.is_empty() || o.only.iter().any(|n| n == s.name))
    {
        let frames = if o.timing { o.frames } else { 1 };
        let (bevy_png, bevy_ms, spans) = match bevy_side::run_in_child(s.name, &o.out, frames) {
            Ok(r) => r,
            Err(e) => {
                println!("{:<24} {:<20} bevy failed: {e}", s.name, s.capability);
                failed = true;
                continue;
            }
        };
        let bevy = match image::open(&bevy_png) {
            Ok(i) => i.to_rgba8(),
            Err(e) => {
                println!(
                    "{:<24} {:<20} cannot read {}: {e}",
                    s.name,
                    s.capability,
                    bevy_png.display()
                );
                failed = true;
                continue;
            }
        };
        let (world, instances) = content::upload(gpu.renderer(), &s.assets, &s.objects);
        let mut viewport = Viewport::new(gpu.renderer(), s.width, s.height);
        let ours = match gpu.render(&mut viewport, &world, &s.scene, &instances) {
            Ok(f) => f,
            Err(e) => {
                println!(
                    "{:<24} {:<20} render-engine failed: {e}",
                    s.name, s.capability
                );
                failed = true;
                continue;
            }
        };
        if (bevy.width(), bevy.height()) != (ours.width, ours.height) {
            println!("{:<24} {:<20} size mismatch", s.name, s.capability);
            failed = true;
            continue;
        }
        let f = flip(ours.width, ours.height, bevy.as_raw(), &ours.rgba);
        let write = [
            (format!("{}.engine.png", s.name), &ours.rgba),
            (format!("{}.flip.png", s.name), &f.heatmap),
        ];
        for (file, rgba) in write {
            if let Err(e) = save_png(&o.out.join(&file), ours.width, ours.height, rgba) {
                eprintln!("cannot write {file}: {e}");
            }
        }

        // GPU time, in rounds: Bevy (the child above), then ours, then again. Each side's time
        // is its fastest round's median: other work on a shared GPU and a GPU that dropped to
        // a lower power state only ever slow a round down. A scenario still over the margin
        // gets up to `RETRIES` more rounds before it fails.
        let mut bevy_rounds = vec![median(&bevy_ms)];
        let mut engine_rounds = Vec::new();
        let is_slow = |e: Option<f64>, b: Option<f64>| matches!((e, b), (Some(e), Some(b)) if e > b * GPU_MARGIN);
        while o.timing && engine_rounds.len() < (TIMING_ROUNDS + RETRIES) as usize {
            engine_rounds.push(
                gpu.gpu_frame_times(
                    &mut viewport,
                    &world,
                    &s.scene,
                    &instances,
                    o.frames as usize,
                )
                .ok()
                .and_then(|t| median(&t)),
            );
            let n = engine_rounds.len() as u32;
            let more = gate_timing
                && (n < TIMING_ROUNDS
                    || (n < TIMING_ROUNDS + RETRIES
                        && is_slow(fastest(&engine_rounds), fastest(&bevy_rounds))));
            if !more {
                break;
            }
            match bevy_side::run_in_child(s.name, &o.out, frames) {
                Ok((_, ms, _)) => bevy_rounds.push(median(&ms)),
                Err(_) => break,
            }
        }
        let rounds = engine_rounds.len();
        let engine_ms = fastest(&engine_rounds);
        let bevy_ms = if o.timing {
            fastest(&bevy_rounds)
        } else {
            None
        };
        let image_ok = f.mean <= FLIP_MEAN_MAX && f.p99 <= FLIP_P99_MAX;
        let slow = is_slow(engine_ms, bevy_ms);
        let timing_missing = gate_timing && (engine_ms.is_none() || bevy_ms.is_none());
        let result = if !image_ok {
            format!("IMAGE DIFFERS (FLIP mean > {FLIP_MEAN_MAX} or p99 > {FLIP_P99_MAX})")
        } else if timing_missing {
            "NO GPU TIMESTAMPS".into()
        } else if slow && gate_timing {
            format!("SLOWER THAN BEVY (> {GPU_MARGIN:.2}×)")
        } else if slow {
            "ok (slower, advisory)".into()
        } else {
            "ok".into()
        };
        failed |= !image_ok || timing_missing || (slow && gate_timing);
        let fmt = |v: Option<f64>| v.map_or("-".to_string(), |v| format!("{v:.3}"));
        let ratio = match (engine_ms, bevy_ms) {
            (Some(e), Some(b)) => format!("{:.2}", e / b),
            _ => "-".into(),
        };
        println!(
            "{:<24} {:<20} {:>9.4} {:>9.4} {:>12} {:>12} {:>7}  {result}",
            s.name,
            s.capability,
            f.mean,
            f.p99,
            fmt(engine_ms),
            fmt(bevy_ms),
            ratio
        );
        report.push(serde_json::json!({
            "scenario": s.name,
            "capability": s.capability,
            "feature": s.feature.id(),
            "width": ours.width,
            "height": ours.height,
            "flip_mean": f.mean,
            "flip_p99": f.p99,
            "engine_gpu_ms": engine_ms,
            "bevy_gpu_ms": bevy_ms,
            "timing_rounds": rounds,
            "bevy_spans": spans,
            "result": result,
            "passed": image_ok && !timing_missing && !(slow && gate_timing),
        }));
    }
    let report = serde_json::json!({
        "adapter": gpu.adapter.name,
        "software": software,
        "flip_mean_max": FLIP_MEAN_MAX,
        "flip_p99_max": FLIP_P99_MAX,
        "gpu_margin": GPU_MARGIN,
        "gpu_gate": gate_timing,
        "frames": if o.timing { o.frames } else { 1 },
        "features": scenarios::Feature::ALL.map(|f| serde_json::json!({
            "id": f.id(),
            "label": f.label(),
            "prs": f.prs(),
            "decision": f.decision(),
        })),
        "scenarios": report,
    });
    if let Err(e) = std::fs::write(o.out.join("report.json"), format!("{report:#}")) {
        eprintln!("cannot write report.json: {e}");
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::flip;

    #[test]
    fn flip_separates_identical_from_different_images() {
        let (w, h) = (64, 64);
        let grey = vec![128u8; w * h * 4];
        let mut half = grey.clone();
        for px in half.as_chunks_mut::<4>().0.iter_mut().take(w * h / 2) {
            px[..3].copy_from_slice(&[255, 0, 0]);
        }
        assert_eq!(flip(w as u32, h as u32, &grey, &grey).mean, 0.0);
        assert!(flip(w as u32, h as u32, &grey, &half).mean > super::FLIP_MEAN_MAX);
    }
}
