//! Per-size targets and per-frame GPU state, reused and grown from frame to frame.

use super::{
    DEPTH_FORMAT, HDR_FORMAT, Renderer, SAMPLE_COUNT,
    gpu::{FrameUniform, InstanceData},
    gpu_culling::GpuCulling,
};
use crate::{
    cull::Frustum,
    particle_pass::ParticleFrame,
    scene::{Cubemap, Tonemapping},
    sky::SkyTargets,
    world::{MaterialId, MeshId},
};
use glam::Vec3;
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
    /// The joint palettes of this frame's skins, one run per skin.
    pub(super) joint_buffer: Growable,
    /// Every view's culled instance indices, one run per draw.
    pub(super) visible_buffer: Growable,
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
    /// Every view's draws: the main pass's (`main_draws`), then each shadow pass's.
    pub(super) draws: Vec<Draw>,
    pub(super) main_draws: Range<usize>,
    /// Scratch for sorting the frame's instances, kept between frames with its allocation.
    pub(super) draw_keys: Vec<DrawKey>,
    /// Scratch for the instance buffer's contents, kept likewise.
    pub(super) instance_data: Vec<InstanceData>,
    pub(super) clear_color: wgpu::Color,
    pub(super) tonemapping: Tonemapping,
    pub(super) particles: ParticleFrame,
    /// What the last `prepare` did (see [`PrepareWork`]).
    pub(super) work: PrepareWork,
    /// The opt-in GPU compaction path's state; `None` is the reference CPU path.
    pub(super) gpu_culling: Option<GpuCulling>,
}

/// Where a viewport's frustum culling runs. The reference path is [`Culling::Cpu`]; see
/// `docs/gpu-driven-experiment.md`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Culling {
    /// Per view on the CPU, in `prepare`: the reference path and the fallback.
    #[default]
    Cpu,
    /// Opaque and alpha-masked unskinned instances are tested and compacted per view by a
    /// compute pass and drawn with one indexed indirect draw per run. Blended and skinned
    /// instances stay on the CPU path in the same frame. Experimental.
    GpuIndirect,
}

/// [`Viewport::set_culling`]: the device lacks what [`Culling::GpuIndirect`] needs, and the
/// viewport stays on [`Culling::Cpu`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuCullingUnsupported;

impl std::fmt::Display for GpuCullingUnsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GPU culling needs INDIRECT_FIRST_INSTANCE and 5 storage buffers per shader stage"
        )
    }
}

impl std::error::Error for GpuCullingUnsupported {}

/// What a viewport's per-frame state retains, from [`Viewport::memory`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ViewportMemory {
    /// Per-frame GPU buffers both paths have: frame, instances, joints, visible indices, lights
    /// and shadow views (at their high-water size).
    pub gpu_buffer_bytes: u64,
    /// GPU buffers only [`Culling::GpuIndirect`] adds: group and view tables, indirect arguments.
    pub gpu_culling_bytes: u64,
    /// CPU scratch vectors kept between frames (draw keys, instance data, draws, shadow passes).
    pub cpu_scratch_bytes: u64,
}

/// The CPU-side work the last [`Renderer::prepare`] did for a viewport, counted where it is
/// done. Deterministic for the same inputs, so tests hold `prepare` to the work contracts in
/// `docs/consumer-integration.md` ("Work contracts").
///
/// With `N` instances, `V` views (the camera plus every shadow view) and `L` shadowed point or
/// spot lights, `prepare` is a full pass over the frame's instance list, not an incremental
/// one: it tests `N · V` boxes plus `N · L` light spheres and uploads `N` instances, whatever
/// changed since the last frame. What the views keep only drives the visible-index upload and
/// the draws.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrepareWork {
    /// Instances listed (`N`).
    pub instances: u32,
    /// Views culled: the camera plus every sun cascade, spot map and point-light cube face.
    pub views: u32,
    /// Instance bounds tested on the CPU: one per instance and view, plus one sphere test per
    /// instance and shadowed point or spot light. With [`Culling::GpuIndirect`], only the
    /// blended and skinned instances that stay on the CPU.
    pub bounds_tests: u64,
    /// Instance and view pairs the [`Culling::GpuIndirect`] compute pass tests (the GPU's
    /// share of the bounds tests); zero on the CPU path.
    pub gpu_bounds_tests: u64,
    /// Instances the camera keeps on the CPU (`k`). With [`Culling::GpuIndirect`] the compacted
    /// runs are only known after the GPU runs ([`Renderer::read_gpu_culled`]), so this counts
    /// only the instances culled on the CPU.
    pub visible: u32,
    /// Instances kept on the CPU, summed over the shadow views.
    pub shadow_visible: u32,
    /// Draw calls the main pass will submit. With [`Culling::GpuIndirect`], one indirect draw
    /// per eligible run, emptied by culling or not.
    pub draws: u32,
    /// Draw calls the shadow passes will submit, summed.
    pub shadow_draws: u32,
    /// Bytes written to GPU buffers and textures: the frame uniform, instances, joint
    /// palettes, lights, shadow views, visible indices, the [`Culling::GpuIndirect`] tables
    /// and environment cube maps. The sky's and the particles' fixed-size uniforms are not
    /// counted.
    pub upload_bytes: u64,
    /// GPU buffers and textures created: grown per-frame buffers, resized shadow-map arrays,
    /// uploaded environment maps and the sky's targets. Zero once the frame's shape is steady.
    pub allocations: u32,
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
pub(crate) struct Growable {
    pub(crate) buffer: wgpu::Buffer,
    label: &'static str,
    usage: wgpu::BufferUsages,
}

impl Growable {
    pub(crate) fn new(
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
    pub(crate) fn write(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
    ) -> bool {
        self.write_at(device, queue, 0, bytes, bytes.len() as u64)
    }

    /// Writes `bytes` at `offset`, growing the buffer to hold at least `size` bytes first.
    /// Returns whether the buffer was replaced.
    pub(crate) fn write_at(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        offset: u64,
        bytes: &[u8],
        size: u64,
    ) -> bool {
        let grown = size > self.buffer.size();
        if grown {
            self.buffer = Self::create(device, self.label, self.usage, size.next_power_of_two());
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.buffer, offset, bytes);
        }
        grown
    }

    pub(crate) fn size(&self) -> u64 {
        self.buffer.size()
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
    /// The instances the pass keeps after culling.
    pub(super) frustum: Frustum,
    /// A local light's position and range: Bevy first drops casters outside this sphere.
    pub(super) sphere: Option<(Vec3, f32)>,
    /// This pass's range of `Viewport::draws`.
    pub(super) draws: Range<usize>,
}

/// What places an instance in the frame's draw order, with what its draw and instance data
/// need from the first pass over the instances.
pub(super) struct DrawKey {
    pub(super) phase: Phase,
    pub(super) face: usize,
    pub(super) material: MaterialId,
    pub(super) mesh: MeshId,
    pub(super) view_z: f32,
    pub(super) skinned: bool,
    /// The skin's first joint in the joint buffer.
    pub(super) joint_offset: u32,
    /// Index in the consumer's instance list.
    pub(super) index: u32,
}

/// Consecutive instances sharing a pipeline, material and mesh: in the sorted instance list
/// while grouping, then in the visible-index buffer once culled for a view.
#[derive(Clone)]
pub(super) struct Draw {
    pub(super) face: usize,
    pub(super) phase: Phase,
    pub(super) skinned: bool,
    pub(super) mesh: MeshId,
    pub(super) material: MaterialId,
    pub(super) instances: Range<u32>,
    /// With [`Culling::GpuIndirect`]: on a group, its row in the GPU tables (`None` when it
    /// stays on the CPU); on a culled draw, its slot in the indirect buffer.
    pub(super) gpu: Option<u32>,
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
            joint_buffer: Growable::new(device, "joints", wgpu::BufferUsages::STORAGE),
            visible_buffer: Growable::new(device, "visible instances", wgpu::BufferUsages::STORAGE),
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
            main_draws: 0..0,
            draw_keys: Vec::new(),
            instance_data: Vec::new(),
            clear_color: wgpu::Color::BLACK,
            tonemapping: Tonemapping::default(),
            particles: ParticleFrame::new(device),
            work: PrepareWork::default(),
            gpu_culling: None,
        }
    }

    /// Selects where frustum culling runs from the next [`Renderer::prepare`] on. Fails, and
    /// leaves the viewport on [`Culling::Cpu`], where the device lacks what
    /// [`Culling::GpuIndirect`] needs ([`Renderer::supports_gpu_culling`]).
    pub fn set_culling(
        &mut self,
        renderer: &Renderer,
        culling: Culling,
    ) -> Result<(), GpuCullingUnsupported> {
        self.gpu_culling = None;
        if culling == Culling::GpuIndirect {
            if !renderer.supports_gpu_culling() {
                return Err(GpuCullingUnsupported);
            }
            self.gpu_culling = Some(GpuCulling::new(&renderer.device));
        }
        Ok(())
    }

    /// Where this viewport's culling runs.
    pub fn culling(&self) -> Culling {
        if self.gpu_culling.is_some() {
            Culling::GpuIndirect
        } else {
            Culling::Cpu
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

    /// What the last [`Renderer::prepare`] into this viewport did.
    pub fn work(&self) -> PrepareWork {
        self.work
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Timestamp queries [`Renderer::encode`] writes for this viewport's last prepared frame: a
    /// begin and an end per pass.
    pub fn timestamp_queries(&self) -> u32 {
        2 * self.pass_labels().len() as u32
    }

    /// One label per timed pass, in the order [`Renderer::encode`] writes their timestamps: the
    /// sky's compute passes, the GPU culling pass, the shadow passes, the main pass (three parts
    /// with a sky), tone mapping and the blit.
    pub fn pass_labels(&self) -> Vec<&'static str> {
        let sky_passes = self.sky.as_ref().map_or(0, |s| s.compute_passes()) as usize;
        let mut labels = vec!["sky"; sky_passes];
        if self.gpu_culling.is_some() {
            labels.push("gpu culling");
        }
        labels.extend(self.shadow_passes.iter().map(|_| "shadow"));
        if self.sky.is_some() {
            labels.extend(["lit opaque", "render sky", "lit transparent"]);
        } else {
            labels.push("lit");
        }
        labels.extend(["tonemap", "blit"]);
        labels
    }

    /// What this viewport's per-frame state retains at the moment (high-water buffer sizes
    /// and scratch capacities), for profiling and the D10 memory row.
    pub fn memory(&self) -> ViewportMemory {
        ViewportMemory {
            gpu_buffer_bytes: self.frame_buffer.size()
                + self.instance_buffer.size()
                + self.joint_buffer.size()
                + self.visible_buffer.size()
                + self.light_buffer.size()
                + self.shadow_view_buffer.size(),
            gpu_culling_bytes: self
                .gpu_culling
                .as_ref()
                .map_or(0, GpuCulling::buffer_bytes),
            cpu_scratch_bytes: (self.draw_keys.capacity() * std::mem::size_of::<DrawKey>()
                + self.instance_data.capacity() * std::mem::size_of::<InstanceData>()
                + self.draws.capacity() * std::mem::size_of::<Draw>()
                + self.shadow_passes.capacity() * std::mem::size_of::<ShadowPass>())
                as u64,
        }
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
