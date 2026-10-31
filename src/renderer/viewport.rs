//! Per-size targets and per-frame GPU state, reused and grown from frame to frame.

use super::{DEPTH_FORMAT, HDR_FORMAT, Renderer, SAMPLE_COUNT, gpu::FrameUniform};
use crate::{
    scene::{Cubemap, Tonemapping},
    sky::SkyTargets,
    world::{MaterialId, MeshId},
};
use std::{ops::Range, sync::Arc};

/// Stride of the per-view shadow uniforms (dynamic-offset alignment).
pub(super) const SHADOW_VIEW_STRIDE: u64 = 256;

/// The main-pass phase of a draw, in drawing order (Bevy's `Opaque3d`, `AlphaMask3d`,
/// `Transparent3d`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Phase {
    Opaque = 0,
    Mask = 1,
    Blend = 2,
    /// Past the last phase (an exclusive range bound).
    End = 3,
}

/// The engine's own render targets for one output size plus the per-frame GPU state (frame
/// uniforms, instances, lights, shadow maps), reused and grown as needed from frame to frame.
pub struct Viewport {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) targets: Targets,
    /// The tone-mapping bind group for the operator it was built for.
    pub(super) tonemap: Option<(Tonemapping, wgpu::BindGroup)>,
    pub(super) frame_buffer: wgpu::Buffer,
    pub(super) instance_buffer: Growable,
    pub(super) light_buffer: Growable,
    pub(super) shadow_view_buffer: Growable,
    pub(super) frame_group: Option<wgpu::BindGroup>,
    pub(super) shadow_view_group: Option<wgpu::BindGroup>,
    pub(super) lighting_group: Option<wgpu::BindGroup>,
    /// Sun cascades, then spot maps (Bevy keeps both in one array).
    pub(super) directional_shadows: ShadowArray,
    pub(super) point_shadows: ShadowArray,
    pub(super) environment: Option<UploadedEnvironment>,
    /// The sky's LUTs and environment cubes while the scene has a sky.
    pub(super) sky: Option<SkyTargets>,
    /// Every shadow pass in order: sun cascades, spot maps, point-light cube faces.
    pub(super) shadow_passes: Vec<ShadowPass>,
    /// Main-pass draws in order; shadow passes draw the same list.
    pub(super) draws: Vec<Draw>,
    pub(super) clear_color: wgpu::Color,
    pub(super) tonemapping: Tonemapping,
}

pub(super) struct Targets {
    pub(super) msaa_view: wgpu::TextureView,
    pub(super) depth_view: wgpu::TextureView,
    /// The resolved HDR frame, read by the tone-mapping pass.
    pub(super) hdr_view: wgpu::TextureView,
    /// The tone-mapped frame (Rgba16Float, like Bevy's main texture), read by the blit pass.
    pub(super) post_view: wgpu::TextureView,
    pub(super) blit_group: wgpu::BindGroup,
}

/// A buffer that is rewritten every frame and replaced by a larger one when it runs out.
pub(super) struct Growable {
    pub(super) buffer: wgpu::Buffer,
    pub(super) label: &'static str,
    pub(super) usage: wgpu::BufferUsages,
}

impl Growable {
    pub(super) fn new(
        device: &wgpu::Device,
        label: &'static str,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let usage = usage | wgpu::BufferUsages::COPY_DST;
        Self {
            buffer: Self::create(device, label, usage, 256),
            label,
            usage,
        }
    }

    fn create(
        device: &wgpu::Device,
        label: &str,
        usage: wgpu::BufferUsages,
        size: u64,
    ) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Writes `bytes` (never empty), returning whether the buffer was replaced.
    pub(super) fn write(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
    ) -> bool {
        let size = bytes.len() as u64;
        let grown = size > self.buffer.size();
        if grown {
            self.buffer = Self::create(device, self.label, self.usage, size.next_power_of_two());
        }
        queue.write_buffer(&self.buffer, 0, bytes);
        grown
    }
}

/// A depth texture array of shadow maps with a view per layer.
pub(super) struct ShadowArray {
    pub(super) size: u32,
    pub(super) layers: u32,
    /// The whole array as the main pass samples it (2D array or cube array).
    pub(super) view: wgpu::TextureView,
    pub(super) layer_views: Vec<wgpu::TextureView>,
}

impl ShadowArray {
    pub(super) fn new(
        device: &wgpu::Device,
        label: &str,
        size: u32,
        layers: u32,
        cube: bool,
    ) -> Self {
        let texture = shadow_maps(device, label, size, layers);
        Self {
            size,
            layers,
            view: texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(if cube {
                    wgpu::TextureViewDimension::CubeArray
                } else {
                    wgpu::TextureViewDimension::D2Array
                }),
                ..Default::default()
            }),
            layer_views: (0..layers)
                .map(|layer| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: layer,
                        array_layer_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect(),
        }
    }
}

pub(super) struct UploadedEnvironment {
    pub(super) diffuse: Arc<Cubemap>,
    pub(super) specular: Arc<Cubemap>,
    pub(super) diffuse_view: wgpu::TextureView,
    pub(super) specular_view: wgpu::TextureView,
}

/// One depth-only pass into a shadow-map layer.
pub(super) struct ShadowPass {
    /// A spot or point-light view (perspective, clipped depth) rather than a sun cascade.
    pub(super) local: bool,
    /// Layer in the directional array, or in the point array for point-light faces.
    pub(super) layer: u32,
    pub(super) point: bool,
    /// Index of the view's matrix in the shadow-view buffer.
    pub(super) view: u32,
}

/// Consecutive instances sharing a pipeline, material and mesh.
pub(super) struct Draw {
    pub(super) face: usize,
    pub(super) phase: Phase,
    pub(super) mesh: MeshId,
    pub(super) material: MaterialId,
    pub(super) instances: Range<u32>,
}

impl Viewport {
    /// Targets for a `width`×`height` output; call [`Renderer::prepare`] before each frame.
    pub fn new(renderer: &Renderer, width: u32, height: u32) -> Self {
        let device = &renderer.device;
        Self {
            width,
            height,
            targets: Targets::new(renderer, width, height),
            tonemap: None,
            frame_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("frame"),
                size: std::mem::size_of::<FrameUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            instance_buffer: Growable::new(device, "instances", wgpu::BufferUsages::STORAGE),
            light_buffer: Growable::new(device, "lights", wgpu::BufferUsages::STORAGE),
            shadow_view_buffer: Growable::new(device, "shadow views", wgpu::BufferUsages::UNIFORM),
            frame_group: None,
            shadow_view_group: None,
            lighting_group: None,
            directional_shadows: ShadowArray::new(device, "directional shadow maps", 1, 1, false),
            point_shadows: ShadowArray::new(device, "point shadow maps", 1, 6, true),
            environment: None,
            sky: None,
            shadow_passes: Vec::new(),
            draws: Vec::new(),
            clear_color: wgpu::Color::BLACK,
            tonemapping: Tonemapping::default(),
        }
    }

    /// Recreates the size-dependent targets (after a window resize).
    pub fn resize(&mut self, renderer: &Renderer, width: u32, height: u32) {
        if (width, height) != (self.width, self.height) {
            self.width = width;
            self.height = height;
            self.targets = Targets::new(renderer, width, height);
            self.tonemap = None;
            if let Some(sky) = &mut self.sky {
                sky.depth_changed();
            }
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl Targets {
    pub(super) fn new(renderer: &Renderer, width: u32, height: u32) -> Self {
        let device = &renderer.device;
        let texture = |label, format, sample_count, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let attachment = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let sampled = attachment | wgpu::TextureUsages::TEXTURE_BINDING;
        let post_view = texture("tone-mapped colour", HDR_FORMAT, 1, sampled);
        let blit_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit"),
            layout: &renderer.blit_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&post_view),
            }],
        });
        Self {
            msaa_view: texture("msaa hdr colour", HDR_FORMAT, SAMPLE_COUNT, attachment),
            // The sky pass reads the depth of the opaque draws.
            depth_view: texture("depth", DEPTH_FORMAT, SAMPLE_COUNT, sampled),
            hdr_view: texture("hdr colour", HDR_FORMAT, 1, sampled),
            post_view,
            blit_group,
        }
    }
}

/// A depth texture array for shadow maps.
fn shadow_maps(device: &wgpu::Device, label: &str, size: u32, layers: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}
