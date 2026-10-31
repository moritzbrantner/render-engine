//! The wgpu renderer: one shadow pass per sun cascade, spot light and point-light cube face, one
//! forward PBR pass with 4× MSAA into an HDR target (opaque, then alpha-masked, then blended
//! instances back to front), a tone-mapping pass into an Rgba16Float target and a blit into the
//! consumer's colour target (Bevy's tonemapping and upscaling nodes).
//!
//! [`Renderer`] holds the pipelines, a [`World`] the consumer's meshes, textures and materials,
//! and a [`Viewport`] the per-size targets and per-frame buffers. Each frame:
//! [`Renderer::prepare`] writes the frame's camera, lights and instances into the viewport, then
//! [`Renderer::encode`] records the passes into the consumer's encoder, ending in its target.

mod encode;
mod gpu;
mod lights;
mod pipelines;
mod prepare;
mod projection;
mod textures;
mod viewport;

pub use encode::FrameTimestamps;
pub use projection::{Cascade, look_at};
pub use viewport::Viewport;

use crate::sky::SkyPipelines;

/// MSAA sample count of the main pass (Bevy's default `Msaa::Sample4`).
pub const SAMPLE_COUNT: u32 = 4;
/// Format of the main pass (Bevy's `ViewTarget::TEXTURE_FORMAT_HDR`).
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Fixed GPU state on the consumer's device: pipelines, layouts and samplers.
pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    frame_layout: wgpu::BindGroupLayout,
    shadow_view_layout: wgpu::BindGroupLayout,
    lighting_layout: wgpu::BindGroupLayout,
    material_layout: wgpu::BindGroupLayout,
    tonemap_layout: wgpu::BindGroupLayout,
    blit_layout: wgpu::BindGroupLayout,
    /// Copies the tone-mapped Rgba16Float frame into the output (Bevy's upscaling blit).
    blit_pipeline: wgpu::RenderPipeline,
    /// Indexed by [`pipelines::shadow_pipeline_index`]: sun cascades (orthographic, unclipped depth)
    /// and spot/point views (perspective, clipped depth, as in Bevy), per face variant, opaque
    /// or cut out by alpha.
    shadow_pipelines: Vec<wgpu::RenderPipeline>,
    /// Indexed by `face * 3 + phase`.
    lit_pipelines: Vec<wgpu::RenderPipeline>,
    /// Indexed by [`pipelines::tonemap_index`].
    tonemap_pipelines: [wgpu::RenderPipeline; 3],
    shadow_sampler: wgpu::Sampler,
    /// Bevy's non-comparison shadow sampler for the soft-shadow blocker search.
    shadow_linear_sampler: wgpu::Sampler,
    /// Bevy's default image sampler (trilinear, clamped), which Bevy uses for environment maps.
    environment_sampler: wgpu::Sampler,
    /// A black 1² cube bound when the scene has no environment map.
    no_environment: wgpu::TextureView,
    tony_mc_mapface_lut: wgpu::TextureView,
    agx_lut: wgpu::TextureView,
    lut_sampler: wgpu::Sampler,
    /// The atmosphere's LUT, environment and fullscreen passes.
    sky: SkyPipelines,
    /// A 1² transmittance LUT bound when the scene has no sky.
    no_transmittance: wgpu::TextureView,
}

impl Renderer {
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn material_layout(&self) -> &wgpu::BindGroupLayout {
        &self.material_layout
    }
}
