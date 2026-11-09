//! A wasm32 smoke of render-engine in Chromium WebGPU (`docs/webgpu-browser-assessment.md`, #75):
//! the oracle's scenario content, the engine's own `Renderer`, nothing else. It compiles the
//! oracle's `content.rs`, `scenarios.rs` and `particles.rs` by path instead of copying a scene.
#![cfg(target_arch = "wasm32")]

#[allow(dead_code)]
#[path = "../../src/content.rs"]
mod content;
#[allow(dead_code)]
#[path = "../../src/particles.rs"]
mod particles;
#[allow(dead_code)]
#[path = "../../src/scenarios.rs"]
mod scenarios;

mod scenario_list;

use js_sys::{Function, Promise};
use render_engine::{
    headless::COLOR_FORMAT,
    renderer::{FrameTimestamps, Renderer, Viewport},
    scene::Scene,
    wgpu,
    world::{Instance, World},
};

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

const WARMUP_MS: f64 = 500.0;
const BATCH_FRAMES: u32 = 64;

fn now() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .expect("performance.now")
        .now()
}

fn err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

async fn read(buffer: &wgpu::Buffer) -> Result<Vec<u8>, JsValue> {
    let slice = buffer.slice(..);
    let done = Promise::new(&mut |resolve: Function, reject: Function| {
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = match r {
                Ok(()) => resolve.call0(&JsValue::NULL),
                Err(e) => reject.call1(&JsValue::NULL, &err(e)),
            };
        });
    });
    JsFuture::from(done).await?;
    let bytes = slice.get_mapped_range().expect("mapped").to_vec();
    buffer.unmap();
    Ok(bytes)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[wasm_bindgen]
pub struct Smoke {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    device_ms: f64,
    renderer_ms: f64,
}

/// One scenario on the GPU: its uploaded content and viewport.
struct Loaded {
    world: World,
    instances: Vec<Instance>,
    scene: Scene,
    viewport: Viewport,
    width: u32,
    height: u32,
    upload_ms: f64,
}

#[wasm_bindgen]
impl Smoke {
    /// Adapter, device and pipelines; the same features `Headless::new` asks for.
    pub async fn create() -> Result<Smoke, JsValue> {
        let start = now();
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(err)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("render-engine webgpu smoke"),
                required_features: adapter.features()
                    & (wgpu::Features::TIMESTAMP_QUERY
                        | wgpu::Features::DEPTH_CLIP_CONTROL
                        | wgpu::Features::DUAL_SOURCE_BLENDING
                        | wgpu::Features::FLOAT32_FILTERABLE
                        | wgpu::Features::INDIRECT_FIRST_INSTANCE),
                ..Default::default()
            })
            .await
            .map_err(err)?;
        let device_ms = now() - start;
        let start = now();
        let renderer = Renderer::new(&device, &queue, COLOR_FORMAT);
        Ok(Smoke {
            instance,
            adapter,
            device,
            queue,
            renderer,
            device_ms,
            renderer_ms: now() - start,
        })
    }

    /// `{name, features, deviceMs, rendererMs}` as JSON.
    pub fn info(&self) -> String {
        let i = self.adapter.get_info();
        format!(
            r#"{{"backend":"{:?}","vendor":{},"name":"{}","driver":"{}","features":"{:?}","deviceMs":{},"rendererMs":{}}}"#,
            i.backend,
            i.vendor,
            i.name,
            i.driver_info,
            self.device.features(),
            self.device_ms,
            self.renderer_ms
        )
    }

    fn load(&self, name: &str) -> Result<Loaded, JsValue> {
        let s = scenario_list::find(name).ok_or_else(|| err(format!("no scenario {name}")))?;
        let start = now();
        let (world, instances, _) = content::upload(&self.renderer, &s.assets, &s.objects);
        let viewport = Viewport::new(&self.renderer, s.width, s.height);
        Ok(Loaded {
            world,
            instances,
            scene: s.scene,
            viewport,
            width: s.width,
            height: s.height,
            upload_ms: now() - start,
        })
    }

    fn target(&self, width: u32, height: u32) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("smoke target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    /// One frame to RGBA8 (the path `Headless::render` takes). JSON stages plus the pixels.
    pub async fn render(&self, name: &str) -> Result<Rendered, JsValue> {
        let mut l = self.load(name)?;
        let t0 = now();
        self.renderer
            .prepare(&mut l.viewport, &l.world, &l.scene, &l.instances);
        let prepare_ms = now() - t0;
        let rgba = self.capture(&mut l).await?;
        Ok(Rendered {
            rgba,
            width: l.width,
            height: l.height,
            upload_ms: l.upload_ms,
            prepare_ms,
        })
    }

    async fn capture(&self, l: &mut Loaded) -> Result<Vec<u8>, JsValue> {
        let target = self.target(l.width, l.height);
        let view = target.create_view(&Default::default());
        let row = (l.width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(row * l.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.renderer
            .encode(&mut encoder, &l.viewport, &l.world, &view, None);
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            target.size(),
        );
        self.queue.submit([encoder.finish()]);
        let bytes = read(&readback).await?;
        let mut rgba = Vec::with_capacity((l.width * l.height * 4) as usize);
        for y in 0..l.height {
            let start = (y * row) as usize;
            rgba.extend_from_slice(&bytes[start..start + (l.width * 4) as usize]);
        }
        Ok(rgba)
    }

    /// A particle capture: `frame` simulation steps of the scenario's emitters, then the frame
    /// the oracle captures (`run_engine_particles`, without its alive-count readback, which
    /// blocks). Pixels only.
    pub async fn particles(
        &self,
        name: &str,
        first_spawn_update: u32,
        frame: u32,
    ) -> Result<Rendered, JsValue> {
        let s = particles::all()
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| err(format!("no particle scenario {name}")))?;
        let (mut world, instances, textures) =
            content::upload(&self.renderer, &s.assets, &s.objects);
        let emitters: Vec<_> = s
            .emitters
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let desc = e.engine(s.seed.wrapping_add(i as u32), &textures);
                let id = world.add_emitter(&desc);
                world.set_emitter_transform(
                    id,
                    render_engine::glam::Mat4::from_translation(e.translation),
                );
                (id, desc)
            })
            .collect();
        let step = |world: &mut World| {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            world.step_particles(&mut encoder, particles::DT, None);
            self.queue.submit([encoder.finish()]);
        };
        for _ in 1..first_spawn_update {
            step(&mut world);
        }
        for (id, desc) in &emitters {
            world.set_emitter(
                *id,
                &render_engine::particles::Emitter {
                    active: true,
                    ..desc.clone()
                },
            );
        }
        for _ in 0..frame {
            step(&mut world);
        }
        let mut l = Loaded {
            world,
            instances,
            viewport: Viewport::new(&self.renderer, s.width, s.height),
            scene: s.scene,
            width: s.width,
            height: s.height,
            upload_ms: 0.0,
        };
        self.renderer
            .prepare(&mut l.viewport, &l.world, &l.scene, &l.instances);
        Ok(Rendered {
            rgba: self.capture(&mut l).await?,
            width: l.width,
            height: l.height,
            upload_ms: 0.0,
            prepare_ms: 0.0,
        })
    }

    /// GPU and CPU medians over `frames` frames (`Headless::gpu_frame_times`' protocol: warm-up,
    /// back-to-back batches, one submission per frame). JSON.
    pub async fn time(&self, name: &str, frames: u32) -> Result<String, JsValue> {
        let mut l = self.load(name)?;
        let target = self.target(l.width, l.height);
        let view = target.create_view(&Default::default());
        self.renderer
            .prepare(&mut l.viewport, &l.world, &l.scene, &l.instances);
        let per_frame = l.viewport.timestamp_queries();
        let batch = (wgpu::QUERY_SET_MAX_QUERIES / per_frame).clamp(1, BATCH_FRAMES);
        let count = per_frame * batch;
        let query_set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let size = u64::from(count) * 8;
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let period = f64::from(self.queue.get_timestamp_period());

        let warmup = now();
        while now() - warmup < WARMUP_MS {
            for _ in 0..batch {
                let mut encoder = self.device.create_command_encoder(&Default::default());
                self.renderer
                    .encode(&mut encoder, &l.viewport, &l.world, &view, None);
                self.queue.submit([encoder.finish()]);
            }
            // Wait for the GPU between batches, so warm-up frames do not pile up.
            let probe = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("warmup fence"),
                size: 8,
                usage: wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            read(&probe).await?;
        }

        let (mut gpu, mut prepare, mut encode, mut submit) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        while (gpu.len() as u32) < frames {
            let n = batch.min(frames - gpu.len() as u32);
            for f in 0..n {
                let t0 = now();
                self.renderer
                    .prepare(&mut l.viewport, &l.world, &l.scene, &l.instances);
                let t1 = now();
                let mut encoder = self.device.create_command_encoder(&Default::default());
                let stamps = FrameTimestamps {
                    query_set: &query_set,
                    first: f * per_frame,
                };
                self.renderer
                    .encode(&mut encoder, &l.viewport, &l.world, &view, Some(stamps));
                if f + 1 == n {
                    encoder.resolve_query_set(&query_set, 0..n * per_frame, &resolve, 0);
                    encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, size);
                }
                let buffer = encoder.finish();
                let t2 = now();
                self.queue.submit([buffer]);
                let t3 = now();
                prepare.push(t1 - t0);
                encode.push(t2 - t1);
                submit.push(t3 - t2);
            }
            let bytes = read(&readback).await?;
            let ticks: Vec<u64> = bytemuck::pod_collect_to_vec(&bytes);
            for f in ticks.chunks_exact(per_frame as usize).take(n as usize) {
                let spans: u64 = f.chunks_exact(2).map(|p| p[1].wrapping_sub(p[0])).sum();
                gpu.push(spans as f64 * period / 1e6);
            }
        }
        Ok(format!(
            r#"{{"frames":{},"gpuMs":{},"prepareMs":{},"encodeMs":{},"submitMs":{}}}"#,
            gpu.len(),
            median(gpu),
            median(prepare),
            median(encode),
            median(submit)
        ))
    }

    /// One frame presented to `canvas` through a wgpu surface, then left for the caller to read
    /// within the same task (WebGPU canvas contents are only guaranteed until it ends).
    pub fn present(&self, name: &str, canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
        let mut l = self.load(name)?;
        canvas.set_width(l.width);
        canvas.set_height(l.height);
        let surface = self
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(err)?;
        let caps = surface.get_capabilities(&self.adapter);
        let format = COLOR_FORMAT.remove_srgb_suffix();
        if !caps.formats.contains(&format) {
            return Err(err(format!("canvas offers {:?}", caps.formats)));
        }
        surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: l.width,
                height: l.height,
                present_mode: wgpu::PresentMode::Fifo,
                desired_maximum_frame_latency: 2,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                view_formats: vec![COLOR_FORMAT],
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );
        let wgpu::CurrentSurfaceTexture::Success(frame) = surface.get_current_texture() else {
            return Err(err("no surface texture"));
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(COLOR_FORMAT),
            ..Default::default()
        });
        self.renderer
            .prepare(&mut l.viewport, &l.world, &l.scene, &l.instances);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.renderer
            .encode(&mut encoder, &l.viewport, &l.world, &view, None);
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
        Ok(())
    }
}

#[wasm_bindgen]
pub struct Rendered {
    rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub upload_ms: f64,
    pub prepare_ms: f64,
}

#[wasm_bindgen]
impl Rendered {
    #[wasm_bindgen(getter)]
    pub fn rgba(&self) -> Vec<u8> {
        self.rgba.clone()
    }
}
