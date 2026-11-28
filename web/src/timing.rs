//! Frame cost in the browser: the async counterpart of `Headless::gpu_frame_times` and the
//! WebGPU smoke's `time` (#75), so a page reports the same GPU, `prepare`, `encode` and submit
//! numbers as native render-engine on the same content (D10's evidence rows).

use crate::{Gpu, now};
use render_engine::{
    renderer::{FrameTimestamps, Viewport},
    scene::Scene,
    wgpu,
    world::{Instance, World},
};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

/// Untimed frames first, so the GPU is at its working clocks.
const WARMUP_MS: f64 = 500.0;
/// Frames submitted back to back between waits.
const BATCH_FRAMES: u32 = 64;

/// Medians over the timed frames, in milliseconds. `gpu_ms` is `None` without the
/// `timestamp-query` feature.
#[derive(Clone, Copy, Debug)]
pub struct FrameTimes {
    pub frames: u32,
    pub gpu_ms: Option<f64>,
    pub prepare_ms: f64,
    pub encode_ms: f64,
    pub submit_ms: f64,
    /// Wall time per frame over the timed batches, each batch's wait for the GPU included: what
    /// frames submitted back to back cost the whole pipeline (main thread, the browser's GPU
    /// process and the GPU).
    pub wall_ms: f64,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
}

/// Maps `buffer` and copies it out, awaiting the browser.
pub async fn read_buffer(buffer: &wgpu::Buffer) -> Result<Vec<u8>, JsValue> {
    let slice = buffer.slice(..);
    let done = js_sys::Promise::new(&mut |resolve: js_sys::Function, reject: js_sys::Function| {
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = match r {
                Ok(()) => resolve.call0(&JsValue::NULL),
                Err(e) => reject.call1(&JsValue::NULL, &JsValue::from_str(&e.to_string())),
            };
        });
    });
    JsFuture::from(done).await?;
    let bytes = slice
        .get_mapped_range()
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .to_vec();
    buffer.unmap();
    Ok(bytes)
}

/// Renders `frames` frames of `world` under `scene` into an offscreen target of the viewport's
/// size, one submission per frame in back-to-back batches after a warm-up, as natively. GPU time
/// is the sum of the frame's pass timestamp spans; CPU times are `performance.now()` around
/// `prepare`, `encode` plus `finish`, and `submit`.
pub async fn frame_times(
    gpu: &Gpu,
    viewport: &mut Viewport,
    world: &World,
    scene: &Scene,
    instances: &[Instance],
    frames: u32,
) -> Result<FrameTimes, JsValue> {
    let renderer = &gpu.renderer;
    let device = &gpu.device;
    renderer.prepare(viewport, world, scene, instances);
    let (width, height) = viewport.size();
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("timing target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.color_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let timestamps = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let per_frame = viewport.timestamp_queries();
    let batch = (wgpu::QUERY_SET_MAX_QUERIES / per_frame).clamp(1, BATCH_FRAMES);
    let size = u64::from(per_frame * batch) * 8;
    let stamps = timestamps.then(|| {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: per_frame * batch,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        (query_set, resolve)
    });
    // Also the fence between batches when there are no timestamps.
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("timestamp readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let period = f64::from(gpu.queue.get_timestamp_period());

    let warmup = now();
    while now() - warmup < WARMUP_MS {
        for _ in 0..batch {
            let mut encoder = device.create_command_encoder(&Default::default());
            renderer.encode(&mut encoder, viewport, world, &view, None);
            gpu.queue.submit([encoder.finish()]);
        }
        let fence = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("warm-up fence"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        read_buffer(&fence).await?;
    }

    let (mut gpu_ms, mut prepare, mut encode, mut submit) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let timed = now();
    while (prepare.len() as u32) < frames {
        let n = batch.min(frames - prepare.len() as u32);
        for f in 0..n {
            let t0 = now();
            renderer.prepare(viewport, world, scene, instances);
            let t1 = now();
            let mut encoder = device.create_command_encoder(&Default::default());
            let frame_stamps = stamps.as_ref().map(|(query_set, _)| FrameTimestamps {
                query_set,
                first: f * per_frame,
            });
            renderer.encode(&mut encoder, viewport, world, &view, frame_stamps);
            if f + 1 == n
                && let Some((query_set, resolve)) = &stamps
            {
                encoder.resolve_query_set(query_set, 0..n * per_frame, resolve, 0);
                encoder.copy_buffer_to_buffer(resolve, 0, &readback, 0, size);
            }
            let buffer = encoder.finish();
            let t2 = now();
            gpu.queue.submit([buffer]);
            let t3 = now();
            prepare.push(t1 - t0);
            encode.push(t2 - t1);
            submit.push(t3 - t2);
        }
        let bytes = read_buffer(&readback).await?;
        if stamps.is_some() {
            let ticks: Vec<u64> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_le_bytes(*b))
                .collect();
            for f in ticks.chunks_exact(per_frame as usize).take(n as usize) {
                let spans: u64 = f
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| p[1].wrapping_sub(p[0]))
                    .sum();
                gpu_ms.push(spans as f64 * period / 1e6);
            }
        }
    }
    Ok(FrameTimes {
        frames,
        gpu_ms: timestamps.then(|| median(gpu_ms)),
        prepare_ms: median(prepare),
        encode_ms: median(encode),
        submit_ms: median(submit),
        wall_ms: (now() - timed) / f64::from(frames.max(1)),
    })
}
