//! Native half of the WebGPU smoke (`docs/webgpu-browser-assessment.md`).
//!
//! ```text
//! compare images <out dir> <scenario>[:<tag>]...  FLIP of <scenario>.<tag> (default webgpu)
//!                                           against the oracle's PNGs; fails outside D3's
//!                                           gates against Bevy
//! compare native <out dir> <scenario>...    the same CPU/GPU protocol on native render-engine
//! compare particles <out dir> <scenario>[:<tag>]...  the browser's particle runs judged as
//!                                           the oracle judges them (D9); fails outside the gates
//! compare soft <out dir> <metres> <scenario>...  soft particles: the browser's run against the
//!                                           same natively; fails outside D3's gates
//! compare diagnostics <out dir> <scenario>...  the browser's GPU-culling and clustered
//!                                           readbacks against native; fails if they differ
//! compare list lighting|particles -           the scenario names the other modes take
//! compare select <out dir> <scenario>...    FLIP of the `select` mode's WebGL 2 fallback; fails
//!                                           outside D3's gates against Bevy
//! compare showcase <out dir> <frame>:<view>...  FLIP of the browser showcase's captures
//!                                           (web/showcase, #128) against native render-engine;
//!                                           fails outside D3's gates
//! compare showcase-native <out dir> <frame>:<view>...  the `time` protocol natively on a
//!                                           showcase view
//! ```
//!
//! A showcase `<view>` is `eye` or an orbit angle in degrees, as the page's `view=` takes it.

#[allow(dead_code)]
#[path = "../../../../examples/shooter_frame/fixture.rs"]
mod fixture;
#[allow(dead_code)]
#[path = "../../../../web/examples/shooter_showcase/view.rs"]
mod view;

#[allow(dead_code)]
#[path = "../../../src/content.rs"]
mod content;
#[allow(dead_code)]
#[path = "../../../src/forward_plus_scenes.rs"]
mod forward_plus_scenes;
#[allow(dead_code)]
#[path = "../../../src/particles.rs"]
mod particles;
#[allow(dead_code)]
#[path = "../../../src/scenarios.rs"]
mod scenarios;

#[allow(dead_code)]
#[path = "../particle_run.rs"]
mod particle_run;
#[allow(dead_code)]
#[path = "../scenario_list.rs"]
mod scenario_list;

use render_engine::{
    headless::Headless,
    renderer::Viewport,
    scene::Scene,
    wgpu,
    world::{Instance, World},
};
use std::path::{Path, PathBuf};
use std::time::Instant;
use view::View;

fn flip(w: u32, h: u32, reference: &[u8], test: &[u8]) -> (f32, f32) {
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
    (pool.mean(), pool.get_percentile(0.99, false))
}

fn png(path: &Path) -> Option<(u32, u32, Vec<u8>)> {
    let image = image::open(path).ok()?.into_rgba8();
    Some((image.width(), image.height(), image.into_raw()))
}

/// `<scenario>[:<tag>]`: the browser capture `<scenario>.<tag>.rgba` (tag `webgpu` by default;
/// `run.mjs` takes another from `SMOKE_TAG`, e.g. for `culling=gpu`) against the oracle's PNGs.
/// False when a capture is missing or outside D3's gates against Bevy.
fn images(out: &Path, specs: &[String]) -> bool {
    let mut ok = true;
    println!(
        "scenario | capture | vs native engine (mean / p99 / differing px) | vs Bevy | vs three.js | native engine vs Bevy"
    );
    for spec in specs {
        let (name, tag) = spec.split_once(':').unwrap_or((spec, "webgpu"));
        let Some((w, h, engine)) = png(&out.join(format!("{name}.engine.png"))) else {
            println!("{name} | {tag} | no engine PNG (run the oracle first) | FAIL");
            ok = false;
            continue;
        };
        let Ok(rgba) = std::fs::read(out.join(format!("{name}.{tag}.rgba"))) else {
            println!("{name} | {tag} | no capture | FAIL");
            ok = false;
            continue;
        };
        image::save_buffer(
            out.join(format!("{name}.{tag}.png")),
            &rgba,
            w,
            h,
            image::ExtendedColorType::Rgba8,
        )
        .expect("png");
        let differing = engine
            .as_chunks::<4>()
            .0
            .iter()
            .zip(rgba.as_chunks::<4>().0)
            .filter(|(a, b)| a != b)
            .count();
        let (m, p) = flip(w, h, &engine, &rgba);
        let against = |file: &str, test: &[u8]| {
            png(&out.join(format!("{name}.{file}.png"))).map(|(_, _, r)| flip(w, h, &r, test))
        };
        let cell =
            |v: Option<(f32, f32)>| v.map_or("-".to_string(), |(m, p)| format!("{m:.4} / {p:.4}"));
        let bevy = against("bevy", &rgba);
        let pass = bevy.is_some_and(|(m, p)| m <= FLIP_MEAN_MAX && p <= FLIP_P99_MAX);
        ok &= pass;
        println!(
            "{name} | {tag} | {m:.4} / {p:.4} / {differing} | {} {} | {} | {}",
            cell(bevy),
            if pass { "ok" } else { "FAIL" },
            cell(against("three", &rgba)),
            cell(against("bevy", &engine)),
        );
    }
    ok
}

/// D3's gates (`oracle/src/main.rs`).
const FLIP_MEAN_MAX: f32 = 0.005;
const FLIP_P99_MAX: f32 = 0.05;

/// The `select` mode's WebGL 2 frame (`<scenario>.select.webgl2.png`, the three.js port) against
/// Bevy, native render-engine, the browser's WebGPU frame (`select.webgpu.rgba`, when the same
/// scenario was also selected onto WebGPU) and the oracle's own three.js frame. False when a frame
/// is missing or the WebGL 2 frame fails D3's gates against Bevy.
fn select(out: &Path, names: &[String]) -> bool {
    let mut ok = true;
    println!(
        "scenario | WebGL 2 vs Bevy (mean / p99) | vs native engine | vs WebGPU | vs oracle three.js (differing px)"
    );
    for name in names {
        let Some((w, h, webgl2)) = png(&out.join(format!("{name}.select.webgl2.png"))) else {
            println!("{name}: no select.webgl2 capture");
            ok = false;
            continue;
        };
        let Some((_, _, bevy)) = png(&out.join(format!("{name}.bevy.png"))) else {
            println!("{name}: no Bevy PNG (run the oracle first)");
            ok = false;
            continue;
        };
        let (mean, p99) = flip(w, h, &bevy, &webgl2);
        let pass = mean <= FLIP_MEAN_MAX && p99 <= FLIP_P99_MAX;
        ok &= pass;
        let cell = |reference: Option<Vec<u8>>| {
            reference.map_or("-".to_string(), |r| {
                let (m, p) = flip(w, h, &r, &webgl2);
                format!("{m:.4} / {p:.4}")
            })
        };
        let file = |tag: &str| png(&out.join(format!("{name}.{tag}.png"))).map(|(_, _, r)| r);
        let webgpu = std::fs::read(out.join(format!("{name}.select.webgpu.rgba"))).ok();
        let three = file("three");
        let differing = three.as_ref().map_or("-".to_string(), |t| {
            t.as_chunks::<4>()
                .0
                .iter()
                .zip(webgl2.as_chunks::<4>().0)
                .filter(|(a, b)| a != b)
                .count()
                .to_string()
        });
        println!(
            "{name} | {mean:.4} / {p99:.4} {} | {} | {} | {} ({differing})",
            if pass { "ok" } else { "FAIL" },
            cell(file("engine")),
            cell(webgpu),
            cell(three.clone()),
        );
    }
    ok
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The protocol of `Smoke::time`, natively, with an offscreen target of the same format.
fn native(name: &str, gpu: &Headless, frames: usize) -> Result<(), String> {
    let s = scenario_list::find(name).ok_or("no scenario")?;
    let (mut world, instances, textures) = content::upload(gpu.renderer(), &s.assets, &s.objects);
    let scene = content::engine_scene(&s.scene, &s.assets, &textures);
    let mut viewport = Viewport::new(gpu.renderer(), s.width, s.height);
    time_native(
        name,
        gpu,
        &mut viewport,
        &mut world,
        &scene,
        &instances,
        frames,
    )
}

/// GPU time by `Headless::gpu_frame_times`, then `prepare`, `encode` and submit per frame.
fn time_native(
    name: &str,
    gpu: &Headless,
    viewport: &mut Viewport,
    world: &mut World,
    scene: &Scene,
    instances: &[Instance],
    frames: usize,
) -> Result<(), String> {
    gpu.render(viewport, world, scene, instances)
        .map_err(|e| e.to_string())?;
    let gpu_times = gpu
        .gpu_frame_times(viewport, world, scene, instances, frames, None)
        .map_err(|e| e.to_string())?;
    let (width, height) = viewport.size();
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
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
    let view = target.create_view(&Default::default());
    let (mut prepare, mut encode, mut submit) = (Vec::new(), Vec::new(), Vec::new());
    for f in 0..frames {
        let t0 = Instant::now();
        gpu.renderer().prepare(viewport, world, scene, instances);
        let t1 = Instant::now();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        gpu.renderer()
            .encode(&mut encoder, viewport, world, &view, None);
        let buffer = encoder.finish();
        let t2 = Instant::now();
        gpu.queue.submit([buffer]);
        let t3 = Instant::now();
        prepare.push((t1 - t0).as_secs_f64() * 1e3);
        encode.push((t2 - t1).as_secs_f64() * 1e3);
        submit.push((t3 - t2).as_secs_f64() * 1e3);
        if f % 64 == 63 {
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
        }
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    println!(
        "{name}: adapter {} gpu {:.3} ms prepare {:.3} encode {:.3} submit {:.3}",
        gpu.adapter.name,
        median(gpu_times.total),
        median(prepare),
        median(encode),
        median(submit)
    );
    Ok(())
}

/// A showcase frame and view (`tick1422:eye`) built natively: its world and scene at
/// `width` × `height`, as the page builds them.
fn showcase_frame(
    gpu: &Headless,
    spec: &str,
    width: u32,
    height: u32,
) -> Result<(String, View, World, Scene), String> {
    let (frame, view) = spec
        .split_once(':')
        .ok_or_else(|| format!("{spec}: expected <frame>:<view>"))?;
    let view = View::parse(view).ok_or_else(|| format!("{spec}: unknown view"))?;
    let json = fixture::FIXTURES
        .iter()
        .find(|(name, _)| *name == frame)
        .ok_or_else(|| format!("{spec}: no fixture {frame}"))?
        .1;
    let mut world = World::new(gpu.renderer());
    let mut scene = fixture::build(&mut world, json, view::BUILD_ASPECT).scene;
    scene.camera = view.camera(&scene.camera, width as f32 / height as f32);
    Ok((frame.to_string(), view, world, scene))
}

/// The showcase's canvas captures (`<frame>.showcase-<view>.rgba`, and the frames after device
/// loss and recovery, `showcase-lose-<view>` and `showcase-crash-<view>`, where present) against
/// native render-engine.
/// False when a capture is missing or outside D3's gates.
fn showcase(out: &Path, specs: &[String]) -> bool {
    let gpu = Headless::new().expect("adapter");
    let mut ok = true;
    println!("frame:view | capture | vs native engine (mean / p99 / differing px) | gate");
    for spec in specs {
        let (w, h) = (1280, 720);
        let (frame, view, world, scene) = match showcase_frame(&gpu, spec, w, h) {
            Ok(f) => f,
            Err(e) => {
                println!("{e}");
                ok = false;
                continue;
            }
        };
        let mut viewport = Viewport::new(gpu.renderer(), w, h);
        let engine = gpu
            .render(&mut viewport, &world, &scene, &[])
            .expect("native frame")
            .rgba;
        let base = format!("{frame}.showcase-{}", view.tag());
        image::save_buffer(
            out.join(format!("{base}.engine.png")),
            &engine,
            w,
            h,
            image::ExtendedColorType::Rgba8,
        )
        .expect("png");
        for tag in [
            format!("showcase-{}", view.tag()),
            format!("showcase-lose-{}", view.tag()),
            format!("showcase-crash-{}", view.tag()),
        ] {
            let Ok(rgba) = std::fs::read(out.join(format!("{frame}.{tag}.rgba"))) else {
                if tag == format!("showcase-{}", view.tag()) {
                    println!("{spec} | {tag} | no capture | FAIL");
                    ok = false;
                }
                continue;
            };
            if rgba.len() != engine.len() {
                println!("{spec} | {tag} | capture is not {w}x{h} | FAIL");
                ok = false;
                continue;
            }
            image::save_buffer(
                out.join(format!("{frame}.{tag}.png")),
                &rgba,
                w,
                h,
                image::ExtendedColorType::Rgba8,
            )
            .expect("png");
            let differing = engine
                .as_chunks::<4>()
                .0
                .iter()
                .zip(rgba.as_chunks::<4>().0)
                .filter(|(a, b)| a != b)
                .count();
            let (m, p) = flip(w, h, &engine, &rgba);
            let pass = m <= FLIP_MEAN_MAX && p <= FLIP_P99_MAX;
            ok &= pass;
            println!(
                "{spec} | {tag} | {m:.4} / {p:.4} / {differing} | {}",
                if pass { "ok" } else { "FAIL" }
            );
        }
    }
    ok
}

/// Bevy's run of a particle scenario, from the oracle's `<name>.bevy.json`.
fn bevy_run(out: &Path, name: &str) -> Result<particle_run::BevyRun, String> {
    let file = out.join(format!("{name}.bevy.json"));
    let json: serde_json::Value = std::fs::read(&file)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .ok_or_else(|| format!("{}: missing (run the oracle first)", file.display()))?;
    let spawns = serde_json::from_value(json["spawns"].clone()).map_err(|e| e.to_string())?;
    let first_spawn_update = json["first_spawn_update"]
        .as_u64()
        .ok_or("first_spawn_update")?;
    Ok(particle_run::BevyRun {
        first_spawn_update: first_spawn_update as u32,
        spawns,
    })
}

fn particle_scenario(name: &str) -> Option<particles::ParticleScenario> {
    particles::all().into_iter().find(|s| s.name == name)
}

/// The browser's particle runs (`run.mjs <out> particles <name>`: `<name>.<capture>.<tag>.rgba`,
/// the plate, and `<name>.<tag>.json` with its spawn and alive counts) judged as the oracle
/// judges render-engine (D9): each capture by its tier against Bevy, every frame by the counts.
/// Also FLIP against native render-engine's capture. False when anything fails.
fn particle_images(out: &Path, specs: &[String]) -> bool {
    let mut ok = true;
    println!(
        "scenario@capture | tier | vs Bevy (mean / p99; stats coverage / spatial / colour) | gate | vs native engine (mean / p99 / differing px) | counts"
    );
    for spec in specs {
        let (name, tag) = spec.split_once(':').unwrap_or((spec, "webgpu"));
        let Some(s) = particle_scenario(name) else {
            println!("{name}: no particle scenario");
            ok = false;
            continue;
        };
        let (w, h) = (s.width, s.height);
        let counts_ok = std::fs::read(out.join(format!("{name}.{tag}.json")))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .and_then(|j| j["particles"]["populationOk"].as_bool())
            .unwrap_or(false);
        ok &= counts_ok;
        let read = |label: &str, side: &str| png(&out.join(format!("{name}.{label}.{side}.png")));
        let browser =
            |label: &str| std::fs::read(out.join(format!("{name}.{label}.{tag}.rgba"))).ok();
        let (Some((_, _, bevy_plate)), Some(plate)) = (read("plate", "bevy"), browser("plate"))
        else {
            println!("{name} | {tag} | no plate | FAIL");
            ok = false;
            continue;
        };
        for c in &s.captures {
            let label = c.label();
            let (Some((_, _, bevy)), Some(rgba)) = (read(&label, "bevy"), browser(&label)) else {
                println!("{name}@{label} | {tag} | missing capture | FAIL");
                ok = false;
                continue;
            };
            image::save_buffer(
                out.join(format!("{name}.{label}.{tag}.png")),
                &rgba,
                w,
                h,
                image::ExtendedColorType::Rgba8,
            )
            .expect("png");
            let (m, p) = flip(w, h, &bevy, &rgba);
            let stats = particles::StatDistance::new(
                &particles::ImageStats::new(w, h, &bevy, &bevy_plate),
                &particles::ImageStats::new(w, h, &rgba, &plate),
            );
            let pass = match c.tier {
                particles::Tier::Exact => m <= FLIP_MEAN_MAX && p <= FLIP_P99_MAX,
                particles::Tier::Statistical => stats.passes(s.stat_max),
            };
            ok &= pass;
            let native = read(&label, "engine").map_or("-".to_string(), |(_, _, e)| {
                let (m, p) = flip(w, h, &e, &rgba);
                let differing = e
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(rgba.as_chunks::<4>().0)
                    .filter(|(a, b)| a != b)
                    .count();
                format!("{m:.4} / {p:.4} / {differing}")
            });
            println!(
                "{name}@{label} | {} | {m:.4} / {p:.4}; {:.3} / {:.3} / {:.3} | {} | {native} | {}",
                c.tier.id(),
                stats.coverage,
                stats.spatial,
                stats.colour,
                if pass { "ok" } else { "FAIL" },
                if counts_ok { "ok" } else { "DIFFER" },
            );
        }
    }
    ok
}

/// Soft particles (`Emitter::depth_fade`): the scenarios run natively with `fade` metres on every
/// emitter (`particle_run`, Bevy's spawns), against the browser's run of the same
/// (`run.mjs <out> particles <name> soft=<fade>` with `SMOKE_TAG=webgpu-soft`). There is no Bevy
/// counterpart for a soft emitter here, so the gate is D3's against native render-engine. Also
/// counts the pixels the fade changed natively against the hard run, so a fade that does
/// nothing shows. False when anything fails.
fn soft(out: &Path, fade: f32, names: &[String]) -> bool {
    let gpu = Headless::new().expect("adapter");
    let mut ok = true;
    println!(
        "scenario@capture | browser soft vs native soft (mean / p99 / differing px) | gate | native soft vs hard (differing px)"
    );
    for name in names {
        let (Some(s), Ok(bevy)) = (particle_scenario(name), bevy_run(out, name)) else {
            println!("{name}: no particle scenario or Bevy run");
            ok = false;
            continue;
        };
        let (w, h) = (s.width, s.height);
        let native = |fade| {
            pollster::block_on(particle_run::run(
                gpu.renderer(),
                &gpu.device,
                &gpu.queue,
                &s,
                &bevy,
                fade,
                async |viewport, world, scene, instances| {
                    gpu.render(viewport, world, scene, instances)
                        .map(|f| f.rgba)
                        .map_err(|e| e.to_string())
                },
            ))
        };
        let (soft_run, hard_run) = match (native(Some(fade)), native(None)) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => {
                println!("{name}: native run failed: {e}");
                ok = false;
                continue;
            }
        };
        for ((frame, engine), (_, hard)) in soft_run.captures.iter().zip(&hard_run.captures) {
            let label = s
                .captures
                .iter()
                .find(|c| c.frame == *frame)
                .expect("captured frame")
                .label();
            image::save_buffer(
                out.join(format!("{name}.{label}.engine-soft.png")),
                engine,
                w,
                h,
                image::ExtendedColorType::Rgba8,
            )
            .expect("png");
            let differing = |a: &[u8], b: &[u8]| {
                a.as_chunks::<4>()
                    .0
                    .iter()
                    .zip(b.as_chunks::<4>().0)
                    .filter(|(a, b)| a != b)
                    .count()
            };
            let faded = differing(engine, hard);
            let Ok(rgba) = std::fs::read(out.join(format!("{name}.{label}.webgpu-soft.rgba")))
            else {
                println!("{name}@{label} | no browser capture | FAIL | {faded}");
                ok = false;
                continue;
            };
            let (m, p) = flip(w, h, engine, &rgba);
            let pass = m <= FLIP_MEAN_MAX && p <= FLIP_P99_MAX && faded > 0;
            ok &= pass;
            println!(
                "{name}@{label} | {m:.4} / {p:.4} / {} | {} | {faded}",
                differing(engine, &rgba),
                if pass { "ok" } else { "FAIL" }
            );
        }
    }
    ok
}

/// The GPU readbacks of `run.mjs <out> render <name> culling=gpu lights=diagnostic` with
/// `SMOKE_TAG=webgpu-gpu` (`<name>.webgpu-gpu.json`: per-view instances kept by
/// `Culling::GpuIndirect`, the clustered counters) against the same frame natively. False when
/// the culled counts differ or the browser has no cluster counters.
fn diagnostics(out: &Path, names: &[String]) -> bool {
    use render_engine::renderer::{ClusterConfig, Culling, LightAssignment};
    let gpu = Headless::new().expect("adapter");
    let mut ok = true;
    println!(
        "scenario | GPU-culled instances per view: native | browser | cluster (fragments, lights looped): native | browser | culled counts equal"
    );
    for name in names {
        let Some(s) = scenario_list::find(name) else {
            println!("{name}: no scenario");
            ok = false;
            continue;
        };
        let (world, instances, _) = content::upload(gpu.renderer(), &s.assets, &s.objects);
        let mut viewport = Viewport::new(gpu.renderer(), s.width, s.height);
        viewport
            .set_culling(gpu.renderer(), Culling::GpuIndirect)
            .expect("GPU culling");
        viewport
            .set_light_assignment(
                gpu.renderer(),
                LightAssignment::Clustered(ClusterConfig {
                    diagnostic: true,
                    ..ClusterConfig::default()
                }),
            )
            .expect("clustered");
        gpu.render(&mut viewport, &world, &s.scene, &instances)
            .expect("frame");
        let culled = pollster::block_on(gpu.renderer().read_gpu_culled(&viewport));
        let cluster = pollster::block_on(gpu.renderer().read_cluster_diagnostic(&viewport));
        let native_culled = serde_json::json!(culled);
        let native_cluster = serde_json::json!(cluster.map(|(f, n)| [f, n]));
        let browser = std::fs::read(out.join(format!("{name}.webgpu-gpu.json")))
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
        let (browser_culled, browser_cluster) =
            browser
                .as_ref()
                .map_or((serde_json::Value::Null, serde_json::Value::Null), |j| {
                    (
                        j["diagnostics"]["gpuCulledPerView"].clone(),
                        j["diagnostics"]["clusterFragmentsLights"].clone(),
                    )
                });
        // The cluster counters count fragment invocations, so they only match where both run
        // the same early depth test (native has `SHADER_EARLY_DEPTH_TEST`, WebGPU does not):
        // reported, not gated.
        let same = browser_culled == native_culled && !browser_cluster.is_null();
        ok &= same;
        println!(
            "{name} | {native_culled} | {browser_culled} | {native_cluster} | {browser_cluster} | {}",
            if same { "ok" } else { "FAIL" }
        );
    }
    ok
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mode, out, names @ ..] = args.as_slice() else {
        eprintln!(
            "usage: compare (images|native|select|showcase|showcase-native) <out dir> <scenario>..."
        );
        std::process::exit(2);
    };
    let out = PathBuf::from(out);
    match mode.as_str() {
        "images" => {
            if !images(&out, names) {
                std::process::exit(1);
            }
        }
        "list" => {
            // `compare list lighting|particles -`: the scenario names, one per line.
            let names: Vec<&str> = match out.to_str() {
                Some("lighting") => scenarios::all().iter().map(|s| s.name).collect(),
                Some("particles") => particles::all().iter().map(|s| s.name).collect(),
                _ => std::process::exit(2),
            };
            println!("{}", names.join("\n"));
        }
        "particles" => {
            if !particle_images(&out, names) {
                std::process::exit(1);
            }
        }
        "soft" => {
            let [fade, names @ ..] = names else {
                std::process::exit(2);
            };
            if !soft(
                &out,
                fade.parse().expect("soft <out> <metres> <scenario>..."),
                names,
            ) {
                std::process::exit(1);
            }
        }
        "diagnostics" => {
            if !diagnostics(&out, names) {
                std::process::exit(1);
            }
        }
        "select" => {
            if !select(&out, names) {
                std::process::exit(1);
            }
        }
        "native" => {
            let gpu = Headless::new().expect("adapter");
            for name in names {
                if let Err(e) = native(name, &gpu, 300) {
                    eprintln!("{name}: {e}");
                }
            }
        }
        "showcase" => {
            if !showcase(&out, names) {
                std::process::exit(1);
            }
        }
        "showcase-native" => {
            let gpu = Headless::new().expect("adapter");
            for spec in names {
                let result =
                    showcase_frame(&gpu, spec, 1280, 720).and_then(|(_, _, mut world, scene)| {
                        let mut viewport = Viewport::new(gpu.renderer(), 1280, 720);
                        time_native(spec, &gpu, &mut viewport, &mut world, &scene, &[], 300)
                    });
                if let Err(e) = result {
                    eprintln!("{spec}: {e}");
                }
            }
        }
        _ => std::process::exit(2),
    }
}
