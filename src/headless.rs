//! Offscreen rendering for the oracle, tests and tools: render a scene to RGBA8 pixels and time
//! frames with GPU timestamps.
//!
//! Adapter selection honours the same environment variables as Bevy, so both oracle sides land on
//! the same device: `WGPU_FORCE_FALLBACK_ADAPTER=1` picks a software adapter (lavapipe on Linux)
//! and `WGPU_ADAPTER_NAME` picks an adapter whose name contains the value.

use crate::{
    renderer::{FrameTimestamps, PreparedScene, Renderer},
    scene::Scene,
};

/// The colour format of offscreen frames (tone mapped and sRGB-encoded, like a Bevy image target).
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[derive(Debug)]
pub enum Error {
    NoAdapter(String),
    Device(wgpu::RequestDeviceError),
    NoTimestamps,
    Readback(wgpu::BufferAsyncError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NoAdapter(why) => write!(f, "no matching GPU adapter: {why}"),
            Error::Device(e) => write!(f, "cannot create device: {e}"),
            Error::NoTimestamps => write!(f, "adapter does not support timestamp queries"),
            Error::Readback(e) => write!(f, "GPU readback failed: {e}"),
        }
    }
}

impl std::error::Error for Error {}

/// A device plus the renderer, rendering into an offscreen texture.
pub struct Headless {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::AdapterInfo,
    renderer: Renderer,
    timestamps: bool,
}

/// One RGBA8 frame, rows top to bottom, sRGB-encoded.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl Headless {
    pub fn new() -> Result<Self, Error> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let force_fallback_adapter = std::env::var("WGPU_FORCE_FALLBACK_ADAPTER")
            .is_ok_and(|v| !(v.is_empty() || v == "0" || v == "false"));
        let adapter = match std::env::var("WGPU_ADAPTER_NAME") {
            Ok(name) => {
                let name = name.to_lowercase();
                pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
                    .into_iter()
                    .find(|a| a.get_info().name.to_lowercase().contains(&name))
                    .ok_or_else(|| Error::NoAdapter(format!("no adapter named like {name}")))?
            }
            Err(_) => pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter,
                compatible_surface: None,
                apply_limit_buckets: false,
            }))
            .map_err(|e| Error::NoAdapter(e.to_string()))?,
        };
        let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("render-engine headless"),
            // Unclipped shadow-map depth (see `Renderer::new`) where the adapter has it.
            required_features: adapter.features()
                & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::DEPTH_CLIP_CONTROL),
            ..Default::default()
        }))
        .map_err(Error::Device)?;
        let renderer = Renderer::new(&device, &queue, COLOR_FORMAT);
        Ok(Self {
            device,
            queue,
            adapter: adapter.get_info(),
            renderer,
            timestamps,
        })
    }

    pub fn prepare(&self, scene: &Scene, width: u32, height: u32) -> PreparedScene {
        self.renderer
            .prepare(&self.device, &self.queue, scene, width, height)
    }

    fn target(&self, width: u32, height: u32) -> wgpu::Texture {
        self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen target"),
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

    /// Renders one frame and reads it back.
    pub fn render(&self, scene: &PreparedScene) -> Result<Frame, Error> {
        let (width, height) = scene.size();
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
        self.renderer.encode(&mut encoder, scene, &view, None);
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
        let bytes = self.read(&readback)?;
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            let start = (y * row) as usize;
            rgba.extend_from_slice(&bytes[start..start + (width * 4) as usize]);
        }
        Ok(Frame {
            width,
            height,
            rgba,
        })
    }

    /// Renders `frames` frames and returns each frame's GPU time in milliseconds (first pass
    /// start to last pass end, from timestamp queries).
    pub fn gpu_frame_times(&self, scene: &PreparedScene, frames: usize) -> Result<Vec<f64>, Error> {
        if !self.timestamps {
            return Err(Error::NoTimestamps);
        }
        let (width, height) = scene.size();
        let target = self.target(width, height);
        let view = target.create_view(&Default::default());
        let query_set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("frame timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp readback"),
            size: 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let period = f64::from(self.queue.get_timestamp_period());
        let mut times = Vec::with_capacity(frames);
        for _ in 0..frames {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.renderer.encode(
                &mut encoder,
                scene,
                &view,
                Some(FrameTimestamps {
                    query_set: &query_set,
                }),
            );
            encoder.resolve_query_set(&query_set, 0..2, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, 16);
            self.queue.submit([encoder.finish()]);
            let bytes = self.read(&readback)?;
            let ticks: [u64; 2] = bytemuck::pod_read_unaligned(&bytes[..16]);
            times.push(ticks[1].wrapping_sub(ticks[0]) as f64 * period / 1e6);
        }
        Ok(times)
    }

    fn read(&self, buffer: &wgpu::Buffer) -> Result<Vec<u8>, Error> {
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .expect("map_async callback dropped")
            .map_err(Error::Readback)?;
        let bytes = slice.get_mapped_range().expect("buffer is mapped").to_vec();
        buffer.unmap();
        Ok(bytes)
    }
}
