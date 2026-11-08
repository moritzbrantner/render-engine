//! Native half of the WebGPU smoke (`docs/webgpu-browser-assessment.md`).
//!
//! ```text
//! compare images <out dir> <scenario>...    FLIP of <scenario>.webgpu against the oracle's PNGs
//! compare native <out dir> <scenario>...    the same CPU/GPU protocol on native render-engine
//! ```

#[allow(dead_code)]
#[path = "../../../src/content.rs"]
mod content;
#[allow(dead_code)]
#[path = "../../../src/particles.rs"]
mod particles;
#[allow(dead_code)]
#[path = "../../../src/scenarios.rs"]
mod scenarios;

#[path = "../scenario_list.rs"]
mod scenario_list;

use render_engine::{headless::Headless, renderer::Viewport, wgpu};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn flip(w: u32, h: u32, reference: &[u8], test: &[u8]) -> (f32, f32) {
    let rgb = |rgba: &[u8]| -> Vec<u8> {
        rgba.chunks_exact(4)
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
            .chunks_exact(4)
            .zip(rgba.chunks_exact(4))
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

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The protocol of `Smoke::time`, natively, with an offscreen target of the same format.
fn native(name: &str, gpu: &Headless, frames: usize) -> Result<(), String> {
    let s = scenario_list::find(name).ok_or("no scenario")?;
    let (mut world, instances, _) = content::upload(gpu.renderer(), &s.assets, &s.objects);
    let mut viewport = Viewport::new(gpu.renderer(), s.width, s.height);
    gpu.render(&mut viewport, &world, &s.scene, &instances)
        .map_err(|e| e.to_string())?;
    let gpu_times = gpu
        .gpu_frame_times(
            &mut viewport,
            &mut world,
            &s.scene,
            &instances,
            frames,
            None,
        )
        .map_err(|e| e.to_string())?;
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: s.width,
            height: s.height,
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
        gpu.renderer()
            .prepare(&mut viewport, &world, &s.scene, &instances);
        let t1 = Instant::now();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        gpu.renderer()
            .encode(&mut encoder, &viewport, &world, &view, None);
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [mode, out, names @ ..] = args.as_slice() else {
        eprintln!("usage: compare (images|native) <out dir> <scenario>...");
        std::process::exit(2);
    };
    let out = PathBuf::from(out);
    match mode.as_str() {
        "images" => images(&out, names),
        "native" => {
            let gpu = Headless::new().expect("adapter");
            for name in names {
                if let Err(e) = native(name, &gpu, 300) {
                    eprintln!("{name}: {e}");
                }
            }
        }
        _ => std::process::exit(2),
    }
}
