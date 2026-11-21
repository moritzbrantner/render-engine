//! A wasm32 smoke of render-engine in a browser's WebGPU (`docs/webgpu-browser-assessment.md`,
//! #75, #129): the oracle's scenario content, the engine's own `Renderer`, nothing else. It
//! compiles the oracle's `content.rs`, `scenarios.rs` and `particles.rs` by path instead of
//! copying a scene; the page fetches the glTF files they read ([`provide_asset`]).
#![cfg(target_arch = "wasm32")]

#[allow(dead_code)]
#[path = "../../src/content.rs"]
mod content;
#[allow(dead_code)]
#[path = "../../src/forward_plus_scenes.rs"]
mod forward_plus_scenes;
#[allow(dead_code)]
#[path = "../../src/particles.rs"]
mod particles;
#[allow(dead_code)]
#[path = "../../src/scenarios.rs"]
mod scenarios;

mod particle_run;
mod scenario_list;

use js_sys::{Function, Promise};
use render_engine::{
    headless::COLOR_FORMAT,
    renderer::{ClusterConfig, Culling, FrameTimestamps, LightAssignment, Renderer, Viewport},
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

/// wgpu reports validation errors by panicking; without this the page only sees `unreachable`.
fn panic_to_console() {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&JsValue::from_str(&info.to_string()))
    }));
}

/// Appends to `globalThis.probeLog`: a GL call can abort the module (the multisampled depth
/// texture), and then only the page still has what the probe found.
fn page_log(line: &str) {
    if let Ok(list) = js_sys::Reflect::get(&js_sys::global(), &"probeLog".into())
        && let Some(list) = list.dyn_ref::<js_sys::Array>()
    {
        list.push(&JsValue::from_str(line));
    }
}

fn set(o: &js_sys::Object, key: &str, value: impl Into<JsValue>) {
    js_sys::Reflect::set(o, &key.into(), &value.into()).expect("plain object");
}

/// The WebGL 2 feasibility probe of `docs/webgl2-fallback.md`: wgpu's GL backend on `canvas`,
/// its adapter's downlevel flags and limits, then `Renderer::new` and one `sun_shadows` frame
/// with every validation error collected instead of panicking on the first (also into the page's
/// `probeLog`). Returns `{adapter, missingDownlevel, limits, errors}` if the frame completes.
#[wasm_bindgen]
pub async fn probe_webgl2(canvas: web_sys::HtmlCanvasElement) -> Result<js_sys::Object, JsValue> {
    panic_to_console();
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::GL;
    let instance = wgpu::Instance::new(descriptor);
    // A WebGL 2 adapter comes from a canvas's context.
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .map_err(err)?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(err)?;
    let out = js_sys::Object::new();
    let info = adapter.get_info();
    set(
        &out,
        "adapter",
        format!("{:?} {} ({})", info.backend, info.name, info.driver_info),
    );
    let downlevel = adapter.get_downlevel_capabilities();
    let missing = wgpu::DownlevelFlags::all() - downlevel.flags;
    set(&out, "missingDownlevel", format!("{missing:?}"));
    let l = adapter.limits();
    let limits = js_sys::Object::new();
    for (k, v) in [
        (
            "maxStorageBuffersPerShaderStage",
            l.max_storage_buffers_per_shader_stage,
        ),
        (
            "maxStorageTexturesPerShaderStage",
            l.max_storage_textures_per_shader_stage,
        ),
        (
            "maxComputeWorkgroupsPerDimension",
            l.max_compute_workgroups_per_dimension,
        ),
        (
            "maxSampledTexturesPerShaderStage",
            l.max_sampled_textures_per_shader_stage,
        ),
        (
            "maxUniformBufferBindingSize",
            l.max_uniform_buffer_binding_size as u32,
        ),
        ("maxTextureDimension2D", l.max_texture_dimension_2d),
        ("maxBindGroups", l.max_bind_groups),
    ] {
        set(&limits, k, v);
    }
    page_log(&format!(
        "adapter {:?} {}; missing downlevel flags {missing:?}; limits {}",
        info.backend,
        info.name,
        js_sys::JSON::stringify(&limits)
            .map(String::from)
            .unwrap_or_default()
    ));
    set(&out, "limits", limits);
    // The adapter's own limits: `Limits::default()` (WebGPU's) is refused outright, since it
    // asks for compute workgroups and storage buffers WebGL 2 does not have.
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("render-engine webgl2 probe"),
            required_limits: l,
            ..Default::default()
        })
        .await
        .map_err(err)?;
    let errors = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let sink = errors.clone();
    device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| {
        // The source chain's innermost line names the missing capability.
        let text = e.to_string();
        let line = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(" | ");
        page_log(&line);
        sink.lock().expect("single-threaded").push(line);
    }));
    let renderer = Renderer::new(&device, &queue, COLOR_FORMAT);
    let s = scenario_list::find("sun_shadows").expect("sun_shadows");
    let (world, instances, _) = content::upload(&renderer, &s.assets, &s.objects);
    let mut viewport = Viewport::new(&renderer, s.width, s.height);
    renderer.prepare(&mut viewport, &world, &s.scene, &instances);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("probe target"),
        size: wgpu::Extent3d {
            width: s.width,
            height: s.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let mut encoder = device.create_command_encoder(&Default::default());
    renderer.encode(&mut encoder, &viewport, &world, &view, None);
    queue.submit([encoder.finish()]);
    let list = js_sys::Array::new();
    for e in errors.lock().expect("single-threaded").iter() {
        list.push(&JsValue::from_str(e));
    }
    set(&out, "errors", list);
    Ok(out)
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
        panic_to_console();
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
        if let Some(missing) = scenario_list::files(name)
            .into_iter()
            .find(|f| !content::asset_provided(f))
        {
            return Err(err(format!(
                "assets/{missing} was not provided by the page"
            )));
        }
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
    /// `culling` is `cpu` (the default) or `gpu` (`Culling::GpuIndirect`); `lights` is
    /// `clustered` (the default), `per-fragment` or `diagnostic` (clustered with its counters).
    /// The GPU path's per-view instance counts and the clustered counters are read back with
    /// the engine's async readbacks into `Rendered::diagnostics`.
    pub async fn render(
        &self,
        name: &str,
        culling: &str,
        lights: &str,
    ) -> Result<Rendered, JsValue> {
        let mut l = self.load(name)?;
        match culling {
            "" | "cpu" => {}
            "gpu" => l
                .viewport
                .set_culling(&self.renderer, Culling::GpuIndirect)
                .map_err(err)?,
            c => return Err(err(format!("culling={c}: cpu or gpu"))),
        }
        let assignment = match lights {
            "" | "clustered" => LightAssignment::default(),
            "per-fragment" => LightAssignment::PerFragment,
            "diagnostic" => LightAssignment::Clustered(ClusterConfig {
                diagnostic: true,
                ..ClusterConfig::default()
            }),
            a => {
                return Err(err(format!(
                    "lights={a}: clustered, per-fragment or diagnostic"
                )));
            }
        };
        l.viewport
            .set_light_assignment(&self.renderer, assignment)
            .map_err(err)?;
        let t0 = now();
        self.renderer
            .prepare(&mut l.viewport, &l.world, &l.scene, &l.instances);
        let prepare_ms = now() - t0;
        let rgba = self.capture(&mut l).await?;
        let culled = self.renderer.read_gpu_culled(&l.viewport).await;
        let cluster = self.renderer.read_cluster_diagnostic(&l.viewport).await;
        let work = l.viewport.work();
        Ok(Rendered {
            rgba,
            width: l.width,
            height: l.height,
            upload_ms: l.upload_ms,
            prepare_ms,
            diagnostics: format!(
                r#"{{"culling":"{:?}","lightAssignment":"{:?}","cpuVisible":{},"cpuShadowVisible":{},"gpuCulledPerView":{},"clusterFragmentsLights":{}}}"#,
                l.viewport.culling(),
                l.viewport.light_assignment(),
                work.visible,
                work.shadow_visible,
                culled.map_or("null".into(), |c| format!("{c:?}")),
                cluster.map_or("null".into(), |(f, n)| format!("[{f},{n}]")),
            ),
        })
    }

    async fn capture(&self, l: &mut Loaded) -> Result<Vec<u8>, JsValue> {
        self.capture_frame(&l.viewport, &l.world).await
    }

    /// Encodes the prepared `viewport` into an offscreen target and reads it back as RGBA8.
    async fn capture_frame(&self, viewport: &Viewport, world: &World) -> Result<Vec<u8>, JsValue> {
        let (width, height) = viewport.size();
        let target = self.target(width, height);
        let view = target.create_view(&Default::default());
        let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(row * height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.renderer
            .encode(&mut encoder, viewport, world, &view, None);
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
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            let start = (y * row) as usize;
            rgba.extend_from_slice(&bytes[start..start + (width * 4) as usize]);
        }
        Ok(rgba)
    }

    /// #116: `frames` rendered frames of the scenario, then `meshes` small meshes of different
    /// sizes, each with its own material, added to its world around the camera's target, then
    /// the frame with them. The upload that failed on SwiftShader after many small allocations
    /// (`mappedAtCreation`) is the point; the pixels show the meshes drew.
    pub async fn many_meshes(
        &self,
        name: &str,
        frames: u32,
        meshes: u32,
    ) -> Result<Rendered, JsValue> {
        use render_engine::{
            glam::{Mat4, Vec3, Vec4},
            material::StandardMaterial,
            mesh::{MeshData, Shape},
        };
        let mut l = self.load(name)?;
        for _ in 0..frames {
            self.renderer
                .prepare(&mut l.viewport, &l.world, &l.scene, &l.instances);
            self.capture(&mut l).await?;
        }
        let start = now();
        let target = l.scene.camera.target;
        for i in 0..meshes {
            let mesh = l.world.add_mesh(&MeshData::shape(Shape::Sphere {
                radius: 0.1,
                sectors: 3 + i % 20,
                stacks: 2 + i % 11,
            }));
            let material = l.world.add_material(&StandardMaterial {
                base_color: Vec4::new(1.0, 0.2 + (i % 5) as f32 * 0.2, 0.2, 1.0),
                ..StandardMaterial::default()
            });
            let (x, z) = ((i % 15) as f32 * 0.25 - 1.75, (i / 15) as f32 * 0.25 - 1.25);
            l.instances.push(Instance {
                mesh,
                material,
                world_from_local: Mat4::from_translation(target + Vec3::new(x, 0.5, z)),
                skin: None,
            });
        }
        let upload_ms = now() - start;
        let t0 = now();
        self.renderer
            .prepare(&mut l.viewport, &l.world, &l.scene, &l.instances);
        let prepare_ms = now() - t0;
        Ok(Rendered {
            rgba: self.capture(&mut l).await?,
            width: l.width,
            height: l.height,
            upload_ms,
            prepare_ms,
            diagnostics: String::new(),
        })
    }

    /// A D9 particle scenario as the oracle runs it (`particle_run`): Bevy's first spawn update
    /// and its spawn counts per emitter and frame (`spawns`, emitter after emitter, each
    /// `spawns.len() / emitters` frames long, from the oracle's `<name>.bevy.json`), the
    /// plate, every capture, and each frame's alive counts read back with the async
    /// `World::read_alive_particles`. `depth_fade` (metres, when not NaN) makes every emitter
    /// soft.
    pub async fn particles(
        &self,
        name: &str,
        first_spawn_update: u32,
        spawns: Vec<u32>,
        depth_fade: f64,
    ) -> Result<ParticleCaptures, JsValue> {
        let s = particles::all()
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| err(format!("no particle scenario {name}")))?;
        let n = s.emitters.len().max(1);
        let frames = spawns.len() / n;
        let bevy = particle_run::BevyRun {
            first_spawn_update,
            spawns: spawns.chunks(frames.max(1)).map(<[u32]>::to_vec).collect(),
        };
        let fade = (!depth_fade.is_nan()).then_some(depth_fade as f32);
        let run = particle_run::run(
            &self.renderer,
            &self.device,
            &self.queue,
            &s,
            &bevy,
            fade,
            async |viewport, world, scene, instances| {
                self.renderer.prepare(viewport, world, scene, instances);
                self.capture_frame(viewport, world)
                    .await
                    .map_err(|e| format!("{e:?}"))
            },
        )
        .await
        .map_err(err)?;
        let population_ok = particle_run::population_ok(&s, &bevy, &run);
        let labels: Vec<String> = run
            .captures
            .iter()
            .map(|(frame, _)| {
                s.captures
                    .iter()
                    .find(|c| c.frame == *frame)
                    .expect("captured frame")
                    .label()
            })
            .collect();
        let json = format!(
            r#"{{"populationOk":{population_ok},"labels":{labels:?},"spawnCounts":{:?},"alive":{:?}}}"#,
            run.spawn_counts, run.alive
        );
        Ok(ParticleCaptures {
            json,
            width: s.width,
            height: s.height,
            plate: run.plate,
            captures: run.captures.into_iter().map(|(_, rgba)| rgba).collect(),
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
                let spans: u64 = f
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| p[1].wrapping_sub(p[0]))
                    .sum();
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
    diagnostics: String,
}

#[wasm_bindgen]
impl Rendered {
    #[wasm_bindgen(getter)]
    pub fn rgba(&self) -> Vec<u8> {
        self.rgba.clone()
    }
}

#[wasm_bindgen]
impl Rendered {
    /// JSON: the viewport's culling and light assignment, its CPU visible counts, and the GPU
    /// readbacks (per-view instances kept by `Culling::GpuIndirect`, the clustered counters).
    #[wasm_bindgen(getter)]
    pub fn diagnostics(&self) -> String {
        self.diagnostics.clone()
    }
}

/// A particle run's plate, captures (in the scenario's capture order) and counts.
#[wasm_bindgen]
pub struct ParticleCaptures {
    json: String,
    pub width: u32,
    pub height: u32,
    plate: Vec<u8>,
    captures: Vec<Vec<u8>>,
}

#[wasm_bindgen]
impl ParticleCaptures {
    /// `{populationOk, labels, spawnCounts, alive}`.
    #[wasm_bindgen(getter)]
    pub fn json(&self) -> String {
        self.json.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn plate(&self) -> Vec<u8> {
        self.plate.clone()
    }

    pub fn capture(&self, i: usize) -> Vec<u8> {
        self.captures[i].clone()
    }
}

/// The files under `assets/` that scenario `name` reads; the page fetches each and hands it to
/// [`provide_asset`] before rendering it (the wasm reads no files).
#[wasm_bindgen]
pub fn asset_files(name: &str) -> Vec<String> {
    scenario_list::files(name)
        .into_iter()
        .map(String::from)
        .collect()
}

/// The bytes of `assets/<path>`, fetched by the page.
#[wasm_bindgen]
pub fn provide_asset(path: &str, bytes: Vec<u8>) {
    content::provide_asset(path, bytes);
}
