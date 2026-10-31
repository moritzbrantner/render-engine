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
//! oracle --calibrate             particle scenarios only: Bevy against itself (rerun, other seed,
//!                                gravity ×1.25), to calibrate the particle gates (D9)
//! ```
//!
//! Particle scenarios (`particles.rs`, D9) are judged per capture: exact captures by the FLIP
//! gate, statistical captures by `particles::StatDistance`, every frame by the alive count.

mod bevy_side;
mod content;
mod particles;
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
/// Frames timed per side on a software adapter, where GPU time is only advisory.
const SOFTWARE_TIMED_FRAMES: u32 = 20;
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
    calibrate: bool,
    variant: bevy_side::ParticleVariant,
}

fn options() -> Result<Options, String> {
    let mut o = Options {
        only: Vec::new(),
        timing: true,
        list: false,
        out: PathBuf::from("target/oracle"),
        bevy_side: None,
        frames: TIMED_FRAMES,
        calibrate: false,
        variant: bevy_side::ParticleVariant::default(),
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
            "--calibrate" => o.calibrate = true,
            "--seed" => {
                o.variant.seed = Some(
                    args.next()
                        .and_then(|n| n.parse().ok())
                        .ok_or("--seed needs a number")?,
                )
            }
            "--gravity-scale" => {
                o.variant.gravity_scale = args
                    .next()
                    .and_then(|n| n.parse().ok())
                    .ok_or("--gravity-scale needs a number")?
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
    if let Some(scenario) = particles::all().into_iter().find(|s| s.name == name) {
        return bevy_particles_child(scenario, o);
    }
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

/// Child process for a particle scenario: `<name>.plate.bevy.png`, `<name>.<capture>.bevy.png`
/// and `<name>.bevy.json` (spawns per emitter and frame, GPU times).
fn bevy_particles_child(s: particles::ParticleScenario, o: &Options) -> Result<(), String> {
    let (name, w, h) = (s.name, s.width, s.height);
    let captures = s.captures.clone();
    let run = bevy_side::run_particles(s, o.variant);
    std::fs::create_dir_all(&o.out).map_err(|e| e.to_string())?;
    let plate = run.plate.ok_or("Bevy produced no plate")?;
    save_png(&o.out.join(format!("{name}.plate.bevy.png")), w, h, &plate)?;
    for c in &captures {
        let (_, rgba) = run
            .captures
            .iter()
            .find(|(f, _)| *f == c.frame)
            .ok_or_else(|| format!("no capture at frame {}", c.frame))?;
        save_png(
            &o.out.join(format!("{name}.{}.bevy.png", c.label())),
            w,
            h,
            rgba,
        )?;
    }
    let json =
        serde_json::json!({ "spawns": run.spawns, "gpu_ms": run.gpu_ms, "spans": run.spans });
    std::fs::write(o.out.join(format!("{name}.bevy.json")), json.to_string())
        .map_err(|e| e.to_string())
}

/// One particle capture compared against the reference: the FLIP numbers always, the statistics
/// always, and the gate of its tier.
struct ParticleCompare {
    flip: Flip,
    stats: particles::StatDistance,
    passed: bool,
}

fn compare_particles(
    s: &particles::ParticleScenario,
    tier: particles::Tier,
    reference: (&[u8], &[u8]),
    test: (&[u8], &[u8]),
) -> ParticleCompare {
    let (w, h) = (s.width, s.height);
    let flip = flip(w, h, reference.1, test.1);
    let stats = particles::StatDistance::new(
        &particles::ImageStats::new(w, h, reference.1, reference.0),
        &particles::ImageStats::new(w, h, test.1, test.0),
    );
    let passed = match tier {
        particles::Tier::Exact => flip.mean <= FLIP_MEAN_MAX && flip.p99 <= FLIP_P99_MAX,
        particles::Tier::Statistical => stats.passes(s.stat_max),
    };
    ParticleCompare {
        flip,
        stats,
        passed,
    }
}

/// Alive count and ages per frame and emitter, as JSON for the report.
fn population_json(s: &particles::ParticleScenario, spawns: &[Vec<u32>]) -> serde_json::Value {
    serde_json::Value::Array(
        s.emitters
            .iter()
            .zip(spawns)
            .map(|(e, spawns)| {
                let p = particles::population(spawns, e.lifetime, e.capacity);
                serde_json::json!({
                    "emitter": e.name,
                    "lifetime": e.lifetime,
                    "capacity": e.capacity,
                    "spawned": spawns,
                    "alive": p.iter().map(|p| p.alive).collect::<Vec<_>>(),
                    "mean_age": p.iter().map(|p| p.mean_age).collect::<Vec<_>>(),
                    "max_age": p.iter().map(|p| p.max_age).collect::<Vec<_>>(),
                })
            })
            .collect(),
    )
}

/// The particle scenarios: Bevy + bevy_hanabi as the reference; render-engine's side is not
/// implemented yet (#13), so each scenario is reported as a known gap and does not fail the run.
fn particle_scenarios(o: &Options) -> (Vec<serde_json::Value>, bool) {
    let mut report = Vec::new();
    let mut failed = false;
    for s in particles::all()
        .iter()
        .filter(|s| o.only.is_empty() || o.only.iter().any(|n| n == s.name))
    {
        let run = match bevy_side::run_particles_in_child(s, &o.out, Default::default()) {
            Ok(r) => r,
            Err(e) => {
                println!("{:<24} {:<20} bevy failed: {e}", s.name, "particles");
                failed = true;
                continue;
            }
        };
        let population = population_json(s, &run.spawns);
        for c in &s.captures {
            let alive: u32 = s
                .emitters
                .iter()
                .zip(&run.spawns)
                .map(|(e, sp)| {
                    particles::population(sp, e.lifetime, e.capacity)
                        .get(c.frame as usize - 1)
                        .map_or(0, |p| p.alive)
                })
                .sum();
            println!(
                "{:<24} {:<20} {:>9} {:>9} {:>12} {:>12} {:>7}  KNOWN GAP: engine side (#13); t = {:.3} s ({}), Bevy alive {alive}",
                format!("{}@{}", s.name, c.label()),
                "particles",
                "-",
                "-",
                "-",
                "-",
                "-",
                c.time(),
                c.tier.id(),
            );
        }
        let bevy_ms = median(&run.gpu_ms);
        report.push(serde_json::json!({
            "scenario": s.name,
            "capability": s.capability,
            "feature": "particles",
            "width": s.width,
            "height": s.height,
            "dt": particles::DT,
            "seed": s.seed,
            "stat_max": {
                "coverage": s.stat_max.coverage,
                "spatial": s.stat_max.spatial,
                "colour": s.stat_max.colour,
            },
            "decision": "d9--particle-parity-seeded-replay-then-statistics",
            "captures": s.captures.iter().zip(&run.captures).map(|(c, (_, rgba))| serde_json::json!({
                "label": c.label(),
                "frame": c.frame,
                "time": c.time(),
                "tier": c.tier.id(),
                "bevy_coverage": particles::ImageStats::new(
                    s.width,
                    s.height,
                    rgba,
                    run.plate.as_deref().unwrap_or_default(),
                ).coverage,
                "flip_mean": null,
                "flip_p99": null,
                "stats": null,
                "passed": null,
            })).collect::<Vec<_>>(),
            "population": population,
            "engine_gpu_ms": null,
            "bevy_gpu_ms": bevy_ms,
            "bevy_spans": run.spans,
            "known_gap": 13,
            "result": "known gap: render-engine side not implemented yet (#13)",
            "passed": null,
        }));
    }
    (report, failed)
}

/// Other seeds `--calibrate` compares each particle scenario against.
const CALIBRATION_SEEDS: u32 = 8;

/// `--calibrate`: each particle scenario on Bevy against itself, to show what the gates accept
/// (a rerun, other seeds) and reject (gravity ×1.25).
fn calibrate(o: &Options) -> ExitCode {
    println!(
        "{:<32} {:<14} {:<12} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7}  result",
        "capture",
        "variant",
        "tier",
        "FLIP mean",
        "FLIP p99",
        "coverage",
        "spatial",
        "colour",
        "alive"
    );
    for s in particles::all()
        .iter()
        .filter(|s| o.only.is_empty() || o.only.iter().any(|n| n == s.name))
    {
        let dir = |tag: &str| o.out.join("calibrate").join(tag);
        let base = match bevy_side::run_particles_in_child(s, &dir("base"), Default::default()) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("{}: {e}", s.name);
                return ExitCode::FAILURE;
            }
        };
        let mut variants = vec![("rerun".to_string(), bevy_side::ParticleVariant::default())];
        variants.extend((1..=CALIBRATION_SEEDS).map(|k| {
            (
                format!("seed+{k}"),
                bevy_side::ParticleVariant {
                    seed: Some(s.seed.wrapping_add(k)),
                    gravity_scale: 1.0,
                },
            )
        }));
        variants.push((
            "gravity×1.25".to_string(),
            bevy_side::ParticleVariant {
                seed: None,
                gravity_scale: 1.25,
            },
        ));
        for (tag, variant) in variants {
            let tag = tag.as_str();
            let run = match bevy_side::run_particles_in_child(s, &dir(tag), variant) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{} {tag}: {e}", s.name);
                    return ExitCode::FAILURE;
                }
            };
            let same_population =
                population_json(s, &base.spawns) == population_json(s, &run.spawns);
            let (bp, rp) = (base.plate.as_deref(), run.plate.as_deref());
            for (c, ((_, b), (_, r))) in s
                .captures
                .iter()
                .zip(base.captures.iter().zip(&run.captures))
            {
                let cmp = compare_particles(
                    s,
                    c.tier,
                    (bp.unwrap_or_default(), b),
                    (rp.unwrap_or_default(), r),
                );
                println!(
                    "{:<32} {:<14} {:<12} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>7}  {}",
                    format!("{}@{}", s.name, c.label()),
                    tag,
                    c.tier.id(),
                    cmp.flip.mean,
                    cmp.flip.p99,
                    cmp.stats.coverage,
                    cmp.stats.spatial,
                    cmp.stats.colour,
                    if same_population { "same" } else { "DIFF" },
                    if cmp.passed && same_population {
                        "accepted"
                    } else {
                        "rejected"
                    },
                );
            }
        }
    }
    ExitCode::SUCCESS
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
    let particle_names: Vec<&str> = particles::all().iter().map(|s| s.name).collect();
    if o.list {
        for s in &scenarios {
            println!("{:<24} {}", s.name, s.capability);
        }
        for s in particles::all() {
            println!("{:<24} {}", s.name, s.capability);
        }
        return ExitCode::SUCCESS;
    }
    if let Some(unknown) = o.only.iter().find(|n| {
        !scenarios.iter().any(|s| s.name == n.as_str()) && !particle_names.contains(&n.as_str())
    }) {
        eprintln!("unknown scenario {unknown} (see --list)");
        return ExitCode::FAILURE;
    }
    if let Err(e) = std::fs::create_dir_all(&o.out) {
        eprintln!("cannot create {}: {e}", o.out.display());
        return ExitCode::FAILURE;
    }
    if o.calibrate {
        return calibrate(&o);
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
    // Advisory software timings need only a few frames (CPU rasterisation is slow).
    let o = Options {
        frames: if software {
            o.frames.min(SOFTWARE_TIMED_FRAMES)
        } else {
            o.frames
        },
        ..o
    };
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
    let (particle_report, particles_failed) = particle_scenarios(&o);
    failed |= particles_failed;
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
        "particles": particle_report,
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
