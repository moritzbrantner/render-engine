//! Native half of the WebGPU smoke (`docs/webgpu-browser-assessment.md`).
//!
//! ```text
//! compare images <out dir> <scenario>...    FLIP of <scenario>.webgpu against the oracle's PNGs
//! compare native <out dir> <scenario>...    the same CPU/GPU protocol on native render-engine
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

fn images(out: &Path, names: &[String]) {
    println!(
        "scenario | vs native engine (mean / p99 / differing px) | vs Bevy | vs three.js | native engine vs Bevy"
    );
    for name in names {
        let Some((w, h, engine)) = png(&out.join(format!("{name}.engine.png"))) else {
            println!("{name}: no engine PNG");
            continue;
        };
        let Ok(rgba) = std::fs::read(out.join(format!("{name}.webgpu.rgba"))) else {
            println!("{name}: no webgpu capture");
            continue;
        };
        image::save_buffer(
            out.join(format!("{name}.webgpu.png")),
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
        let cell = |file: &str, against: &[u8]| {
            png(&out.join(format!("{name}.{file}.png"))).map_or("-".to_string(), |(_, _, r)| {
                let (m, p) = flip(w, h, &r, against);
                format!("{m:.4} / {p:.4}")
            })
        };
        println!(
            "{name} | {m:.4} / {p:.4} / {differing} | {} | {} | {}",
            cell("bevy", &rgba),
            cell("three", &rgba),
            cell("bevy", &engine)
        );
    }
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
    let (mut world, instances, _) = content::upload(gpu.renderer(), &s.assets, &s.objects);
    let mut viewport = Viewport::new(gpu.renderer(), s.width, s.height);
    time_native(
        name,
        gpu,
        &mut viewport,
        &mut world,
        &s.scene,
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
        "images" => images(&out, names),
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
