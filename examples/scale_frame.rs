//! Native frame cost of the Greyhaven-sized scene (`scale_scene/scene.rs`, #172), by the protocol
//! the browser's `scale_bench` uses (`render_engine_web::timing::frame_times`): medians over 300
//! frames at 1280×720 of `prepare` (the moving units listed each frame), `encode` plus `finish`,
//! `submit`, the wall time per frame with the waits for the GPU, and the passes' GPU time, with the frame's `PrepareWork`. Prints one JSON line.
//!
//! `cargo run --release --example scale_frame`

#[path = "scale_scene/scene.rs"]
mod scene;

use render_engine::{headless::Headless, renderer::Viewport, wgpu, world::World};
use std::time::Instant;

const FRAMES: usize = 300;
const BATCH: usize = 64;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let gpu = Headless::new().expect("a wgpu adapter");
    let renderer = gpu.renderer();
    let (device, queue) = (renderer.device(), renderer.queue());
    let mut world = World::new(renderer);
    let built = scene::build(&mut world);
    let mut viewport = Viewport::new(renderer, 1280, 720);
    let units = built.units(0);

    let gpu_ms = gpu
        .gpu_frame_times(
            &mut viewport,
            &mut world,
            &built.scene,
            &units,
            FRAMES,
            None,
        )
        .map(|t| median(t.total))
        .ok();

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("timing target"),
        size: wgpu::Extent3d {
            width: 1280,
            height: 720,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let (mut prepare, mut encode, mut submit) = (Vec::new(), Vec::new(), Vec::new());
    let mut timed = Instant::now();
    for f in 0..FRAMES + BATCH {
        if f == BATCH {
            timed = Instant::now();
        }
        let units = built.units(f);
        let t0 = Instant::now();
        renderer.prepare(&mut viewport, &world, &built.scene, &units);
        let t1 = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer.encode(&mut encoder, &viewport, &world, &view, None);
        let buffer = encoder.finish();
        let t2 = Instant::now();
        queue.submit([buffer]);
        let t3 = Instant::now();
        // The first batch warms up.
        if f >= BATCH {
            prepare.push((t1 - t0).as_secs_f64() * 1e3);
            encode.push((t2 - t1).as_secs_f64() * 1e3);
            submit.push((t3 - t2).as_secs_f64() * 1e3);
        }
        if f % BATCH == BATCH - 1 {
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("device poll");
        }
    }
    let wall_ms = timed.elapsed().as_secs_f64() * 1e3 / FRAMES as f64;
    let w = viewport.work();
    println!(
        "{{\"adapter\":\"{}\",\"frames\":{FRAMES},\"gpuMs\":{},\"prepareMs\":{:.3},\"encodeMs\":{:.3},\"submitMs\":{:.3},\"wallMs\":{wall_ms:.3},\"instances\":{},\"views\":{},\"visible\":{},\"shadowVisible\":{},\"draws\":{},\"shadowDraws\":{},\"triangles\":{},\"shadowTriangles\":{}}}",
        gpu.adapter.name,
        gpu_ms.map_or("null".into(), |g| format!("{g:.3}")),
        median(prepare),
        median(encode),
        median(submit),
        w.instances,
        w.views,
        w.visible,
        w.shadow_visible,
        w.draws,
        w.shadow_draws,
        w.cpu_triangles,
        w.cpu_shadow_triangles,
    );
}
