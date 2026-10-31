//! The wgpu renderer: one shadow pass per sun cascade, spot light and point-light cube face, one
//! forward PBR pass with 4× MSAA into an HDR target (opaque, then alpha-masked, then blended
//! instances back to front, then particles), a tone-mapping pass into an Rgba16Float target and
//! a blit into the consumer's colour target (Bevy's tonemapping and upscaling nodes).
//!
//! [`Renderer`] holds the pipelines, a [`World`] the consumer's meshes, textures and materials,
//! and a [`Viewport`] the per-size targets and per-frame buffers. Each frame:
//! [`Renderer::prepare`] writes the frame's camera, lights and instances into the viewport, then
//! [`Renderer::encode`] records the passes into the consumer's encoder, ending in its target.

use crate::{
    cull::{Frustum, Obb},
    material::AlphaMode,
    particle_pass::{ParticleFrame, ParticlePipelines},
    particle_sim::SimPipelines,
    scene::{Cubemap, FogFalloff, LightShadows, MAX_CASCADES, Scene, ShadowFilter, Tonemapping},
    sky::{SkyPipelines, SkyTargets},
    world::{Instance, MaterialId, MeshId, SkinId, World, streams},
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, UVec4, Vec3, Vec4};
use std::{collections::HashMap, ops::Range, sync::Arc};
use wgpu::util::DeviceExt;

/// MSAA sample count of the main pass (Bevy's default `Msaa::Sample4`).
pub const SAMPLE_COUNT: u32 = 4;
/// Format of the main pass (Bevy's `ViewTarget::TEXTURE_FORMAT_HDR`).
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const TONY_MC_MAPFACE_LUT: &[u8] = include_bytes!("luts/tony_mc_mapface.ktx2");
const AGX_LUT: &[u8] = include_bytes!("luts/AgX-default_contrast.ktx2");
/// Stride of the per-view shadow uniforms (dynamic-offset alignment).
const SHADOW_VIEW_STRIDE: u64 = 256;
/// `GpuLight::flags`: the light samples its shadow map.
const LIGHT_SHADOWS: u32 = 1;
/// `GpuLight::flags`: the spot light points downwards (Bevy `SPOT_LIGHT_Y_NEGATIVE`).
const LIGHT_SPOT_Y_NEGATIVE: u32 = 2;
/// Bevy's point-light cube faces (`CUBE_MAP_FACES`): look target and up per cube layer.
const CUBE_FACES: [(Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::Y),
    (Vec3::NEG_X, Vec3::Y),
    (Vec3::Y, Vec3::Z),
    (Vec3::NEG_Y, Vec3::NEG_Z),
    (Vec3::NEG_Z, Vec3::Y),
    (Vec3::Z, Vec3::Y),
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    clip_from_world: Mat4,
    view_from_world: Mat4,
    cascade_clip_from_world: [Mat4; MAX_CASCADES],
    /// Per cascade: x far bound (m), y texel size (m).
    cascade_params: [Vec4; MAX_CASCADES],
    eye: Vec4,
    direction_to_light: Vec4,
    sun_color: Vec4,
    ambient_color: Vec4,
    /// x depth bias (m), y normal bias (texels), z overlap proportion, w cascade count.
    shadow: Vec4,
    /// x exposure, y 1 when the sun casts shadows, z 1 when there is a sun, w 1 for Gaussian
    /// filtering.
    flags: Vec4,
    /// x environment-map intensity (cd/m²), y smallest specular mip level, z 1 when there is an
    /// environment map.
    environment: Vec4,
    /// Distance fog (Bevy `GpuFog`): linear RGBA colour.
    fog_color: Vec4,
    /// Sun inscattering colour, alpha = strength (0 = off).
    fog_directional_light_color: Vec4,
    /// xyz: Bevy `be` (linear: start, end; exponential: density; atmospheric: extinction),
    /// w: directional-light exponent.
    fog_be: Vec4,
    /// xyz: Bevy `bi` (atmospheric inscattering), w: mode (0 off, 1 linear, 2 exponential,
    /// 3 exponential squared, 4 atmospheric).
    fog_bi: Vec4,
    /// x point lights, y spot lights (the spot lights follow the point lights in the list).
    light_counts: UVec4,
    /// Unit direction towards the moon, w 1 when there is a moon.
    moon_direction_to_light: Vec4,
    moon_color: Vec4,
    world_to_atmosphere: Mat4,
    /// x planet radius, y atmosphere radius, z 1 with a sky, w 1 when the moon is shaded first.
    atmosphere: Vec4,
    /// x sun disc, y moon disc angular size.
    disks: Vec4,
}

/// One point or spot light, laid out like Bevy's `GpuClusteredLight`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLight {
    /// Point: the cube-face projection's lower-right 2×2 terms. Spot: direction x and z, spot
    /// scale, spot offset.
    light_custom_data: Vec4,
    /// rgb colour × intensity / 4π, w 1 / range².
    color_inverse_square_range: Vec4,
    position_radius: Vec4,
    flags: u32,
    shadow_depth_bias: f32,
    /// Premultiplied by the shadow texel size at 1 m and √2, like Bevy.
    shadow_normal_bias: f32,
    shadow_map_near_z: f32,
    spot_light_tan_angle: f32,
    /// The light radius with soft shadows, else 0.
    soft_shadow_size: f32,
    /// Point: cube index in the cube array. Spot: layer in the directional shadow array.
    shadow_index: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceData {
    world_from_local: Mat4,
    // mat3x3<f32> in WGSL: three columns padded to 16 bytes.
    normal_from_local: [Vec4; 3],
    /// x 1 when the mesh has tangents, y 1 when the model matrix mirrors, z the skin's first
    /// joint in the joint buffer.
    flags: UVec4,
}

/// Which faces a pipeline rasterises: bit 0 winds mirrored instances clockwise, bit 1 draws back
/// faces (double-sided materials).
const FACE_VARIANTS: usize = 4;

fn face_variant(mirrored: bool, double_sided: bool) -> usize {
    usize::from(mirrored) | usize::from(double_sided) << 1
}

/// Index into `Renderer::shadow_pipelines`, in the order `Renderer::new` creates them.
fn shadow_pipeline_index(skinned: bool, local: bool, face: usize, masked: bool) -> usize {
    ((usize::from(skinned) * 2 + usize::from(local)) * FACE_VARIANTS + face) * 2
        + usize::from(masked)
}

/// Index into `Renderer::lit_pipelines`, in the order `Renderer::new` creates them.
fn lit_pipeline_index(skinned: bool, face: usize, phase: Phase) -> usize {
    (usize::from(skinned) * FACE_VARIANTS + face) * 3 + phase as usize
}

fn primitive(face: usize) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        front_face: if face & 1 != 0 {
            wgpu::FrontFace::Cw
        } else {
            wgpu::FrontFace::Ccw
        },
        cull_mode: (face & 2 == 0).then_some(wgpu::Face::Back),
        ..Default::default()
    }
}

/// The main-pass phase of a draw, in drawing order (Bevy's `Opaque3d`, `AlphaMask3d`,
/// `Transparent3d`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Phase {
    Opaque = 0,
    Mask = 1,
    Blend = 2,
    /// Past the last phase (an exclusive range bound).
    End = 3,
}

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
    /// Indexed by [`shadow_pipeline_index`]: sun cascades (orthographic, unclipped depth)
    /// and spot/point views (perspective, clipped depth, as in Bevy), per face variant, opaque
    /// or cut out by alpha.
    shadow_pipelines: Vec<wgpu::RenderPipeline>,
    /// Indexed by `face * 3 + phase`.
    lit_pipelines: Vec<wgpu::RenderPipeline>,
    /// Indexed by [`tonemap_index`].
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
    /// Camera-facing particle quads (`particle_pass`).
    particles: ParticlePipelines,
    /// The particle simulation's compute pipelines, shared by its worlds (`particle_sim`).
    sim: SimPipelines,
}

/// The tone-mapping shader's `METHOD` constant.
fn tonemap_index(t: Tonemapping) -> usize {
    match t {
        Tonemapping::AcesFitted => 0,
        Tonemapping::TonyMcMapface => 1,
        Tonemapping::AgX => 2,
    }
}

/// Where a frame writes GPU timestamps: a begin and an end query per pass, in pass order from
/// index `first` on ([`Viewport::timestamp_queries`] of them). Summing the passes' spans gives
/// the frame's GPU time the way Bevy's render diagnostics sum their top-level spans, without the
/// gaps between passes where the GPU runs other processes' work.
pub struct FrameTimestamps<'a> {
    pub query_set: &'a wgpu::QuerySet,
    pub first: u32,
}

/// Hands out each pass's pair of timestamp queries.
pub(crate) struct Stamps<'a> {
    query_set: Option<&'a wgpu::QuerySet>,
    next: std::cell::Cell<u32>,
}

impl<'a> Stamps<'a> {
    fn pair(&self) -> Option<(&'a wgpu::QuerySet, u32)> {
        self.query_set.map(|q| {
            let i = self.next.get();
            self.next.set(i + 2);
            (q, i)
        })
    }

    fn render(&self) -> Option<wgpu::RenderPassTimestampWrites<'a>> {
        self.pair()
            .map(|(query_set, i)| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(i),
                end_of_pass_write_index: Some(i + 1),
            })
    }

    pub(crate) fn compute(&self) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
        self.pair()
            .map(|(query_set, i)| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(i),
                end_of_pass_write_index: Some(i + 1),
            })
    }
}

/// The engine's own render targets for one output size plus the per-frame GPU state (frame
/// uniforms, instances, lights, shadow maps), reused and grown as needed from frame to frame.
pub struct Viewport {
    width: u32,
    height: u32,
    targets: Targets,
    /// The tone-mapping bind group for the operator it was built for.
    tonemap: Option<(Tonemapping, wgpu::BindGroup)>,
    frame_buffer: wgpu::Buffer,
    instance_buffer: Growable,
    /// The joint palettes of this frame's skins, one run per skin.
    joint_buffer: Growable,
    /// Every view's culled instance indices, one run per draw.
    visible_buffer: Growable,
    light_buffer: Growable,
    shadow_view_buffer: Growable,
    frame_group: Option<wgpu::BindGroup>,
    shadow_view_group: Option<wgpu::BindGroup>,
    lighting_group: Option<wgpu::BindGroup>,
    /// Sun cascades, then spot maps (Bevy keeps both in one array).
    directional_shadows: ShadowArray,
    point_shadows: ShadowArray,
    environment: Option<UploadedEnvironment>,
    /// The sky's LUTs and environment cubes while the scene has a sky.
    sky: Option<SkyTargets>,
    /// Every shadow pass in order: sun cascades, spot maps, point-light cube faces.
    shadow_passes: Vec<ShadowPass>,
    /// Every view's draws: the main pass's (`main_draws`), then each shadow pass's.
    draws: Vec<Draw>,
    main_draws: Range<usize>,
    clear_color: wgpu::Color,
    tonemapping: Tonemapping,
    particles: ParticleFrame,
}

struct Targets {
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    /// The resolved HDR frame, read by the tone-mapping pass.
    hdr_view: wgpu::TextureView,
    /// The tone-mapped frame (Rgba16Float, like Bevy's main texture), read by the blit pass.
    post_view: wgpu::TextureView,
    blit_group: wgpu::BindGroup,
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
struct ShadowArray {
    size: u32,
    layers: u32,
    /// The whole array as the main pass samples it (2D array or cube array).
    view: wgpu::TextureView,
    layer_views: Vec<wgpu::TextureView>,
}

impl ShadowArray {
    fn new(device: &wgpu::Device, label: &str, size: u32, layers: u32, cube: bool) -> Self {
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

struct UploadedEnvironment {
    diffuse: Arc<Cubemap>,
    specular: Arc<Cubemap>,
    diffuse_view: wgpu::TextureView,
    specular_view: wgpu::TextureView,
}

/// One depth-only pass into a shadow-map layer.
struct ShadowPass {
    /// A spot or point-light view (perspective, clipped depth) rather than a sun cascade.
    local: bool,
    /// Layer in the directional array, or in the point array for point-light faces.
    layer: u32,
    point: bool,
    /// Index of the view's matrix in the shadow-view buffer.
    view: u32,
    /// The instances the pass keeps after culling.
    frustum: Frustum,
    /// A local light's position and range: Bevy first drops casters outside this sphere.
    sphere: Option<(Vec3, f32)>,
    /// This pass's range of `Viewport::draws`.
    draws: Range<usize>,
}

/// Consecutive instances sharing a pipeline, material and mesh: in the sorted instance list
/// while grouping, then in the visible-index buffer once culled for a view.
#[derive(Clone)]
struct Draw {
    face: usize,
    phase: Phase,
    skinned: bool,
    mesh: MeshId,
    material: MaterialId,
    instances: Range<u32>,
}

impl Renderer {
    /// Creates the pipelines on the consumer's device for frames written to `color_format` (tone
    /// mapped; an sRGB format encodes the output) and uploads the tone-mapping LUTs. Unclipped
    /// sun shadow depth uses `DEPTH_CLIP_CONTROL` where the device has it enabled; the sky
    /// blends with `DUAL_SOURCE_BLENDING` where enabled (as Bevy does) and needs
    /// `FLOAT32_FILTERABLE`.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/lit.wgsl"));
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Every skinned instance's joint palette (Bevy `joint_matrices`).
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let cube = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::Cube,
                multisampled: false,
            },
            count: None,
        };
        // Main pass: lights, shadow maps and the environment map.
        let lighting_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lighting"),
            entries: &[
                cube(3),
                cube(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::CubeArray,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // A material: its uniform, then five texture/sampler pairs (base colour, emissive,
        // metallic-roughness, occlusion, normal map).
        let mut material_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for i in 0..5 {
            material_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 1 + 2 * i,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
            material_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2 + 2 * i,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &material_entries,
        });
        // A shadow pass picks its view's matrix through a dynamic offset.
        let shadow_view_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow view"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                }],
            });
        // Positions, normals and the surface attributes; skinned meshes add their joints and
        // weights (see `world::streams`).
        let stream = |stride: usize, attributes| wgpu::VertexBufferLayout {
            array_stride: stride as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes,
        };
        let unskinned_streams = [
            Some(stream(12, &streams::POSITIONS[..])),
            Some(stream(12, &streams::NORMALS[..])),
            Some(stream(
                std::mem::size_of::<crate::world::SurfaceVertex>(),
                &streams::SURFACE[..],
            )),
        ];
        let mut skinned_streams = unskinned_streams.to_vec();
        skinned_streams.push(Some(stream(
            std::mem::size_of::<crate::world::SkinVertex>(),
            &streams::SKIN[..],
        )));
        let vertex_streams = |skinned: bool| -> &[Option<wgpu::VertexBufferLayout<'_>>] {
            if skinned {
                &skinned_streams
            } else {
                &unskinned_streams
            }
        };
        // Reverse Z everywhere: nearer is larger.
        let depth = |write: bool| wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(write),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow"),
                bind_group_layouts: &[
                    Some(&frame_layout),
                    Some(&shadow_view_layout),
                    Some(&material_layout),
                ],
                immediate_size: 0,
            });
        // Unclipped depth like Bevy's directional shadow views: natively where the device has
        // `DEPTH_CLIP_CONTROL`, else clamped in a fragment shader (see `lit.wgsl`).
        let unclipped_depth = device
            .features()
            .contains(wgpu::Features::DEPTH_CLIP_CONTROL);
        let mut shadow_pipelines = Vec::new();
        for (skinned, local) in [(false, false), (false, true), (true, false), (true, true)] {
            for face in 0..FACE_VARIANTS {
                for masked in [false, true] {
                    let emulated = !local && !unclipped_depth;
                    let fragment = match (emulated, masked) {
                        (false, false) => None,
                        (false, true) => Some("shadow_masked_fs"),
                        (true, false) => Some("shadow_emulated_fs"),
                        (true, true) => Some("shadow_emulated_masked_fs"),
                    };
                    shadow_pipelines.push(device.create_render_pipeline(
                        &wgpu::RenderPipelineDescriptor {
                            label: Some(if local { "local shadow" } else { "sun shadow" }),
                            layout: Some(&shadow_pipeline_layout),
                            vertex: wgpu::VertexState {
                                module: &shader,
                                entry_point: Some(match (emulated, skinned) {
                                    (false, false) => "shadow_vs",
                                    (true, false) => "shadow_emulated_vs",
                                    (false, true) => "shadow_skinned_vs",
                                    (true, true) => "shadow_emulated_skinned_vs",
                                }),
                                compilation_options: Default::default(),
                                buffers: vertex_streams(skinned),
                            },
                            primitive: wgpu::PrimitiveState {
                                unclipped_depth: !local && unclipped_depth,
                                ..primitive(face)
                            },
                            depth_stencil: Some(depth(true)),
                            multisample: wgpu::MultisampleState::default(),
                            fragment: fragment.map(|entry_point| wgpu::FragmentState {
                                module: &shader,
                                entry_point: Some(entry_point),
                                compilation_options: Default::default(),
                                targets: &[],
                            }),
                            multiview_mask: None,
                            cache: None,
                        },
                    ));
                }
            }
        }

        let lit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lit"),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(&lighting_layout),
                Some(&material_layout),
            ],
            immediate_size: 0,
        });
        let mut lit_pipelines = Vec::new();
        for (skinned, face) in [false, true]
            .into_iter()
            .flat_map(|s| (0..FACE_VARIANTS).map(move |f| (s, f)))
        {
            for phase in [Phase::Opaque, Phase::Mask, Phase::Blend] {
                let blend = phase == Phase::Blend;
                lit_pipelines.push(device.create_render_pipeline(
                    &wgpu::RenderPipelineDescriptor {
                        label: Some("lit"),
                        layout: Some(&lit_pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: Some(if skinned { "lit_skinned_vs" } else { "lit_vs" }),
                            compilation_options: Default::default(),
                            buffers: vertex_streams(skinned),
                        },
                        primitive: primitive(face),
                        // Blended fragments test depth but do not write it (Bevy).
                        depth_stencil: Some(depth(!blend)),
                        multisample: wgpu::MultisampleState {
                            count: SAMPLE_COUNT,
                            ..Default::default()
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: Some(if phase == Phase::Mask {
                                "lit_masked_fs"
                            } else {
                                "lit_fs"
                            }),
                            compilation_options: Default::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: HDR_FORMAT,
                                blend: blend.then_some(wgpu::BlendState::ALPHA_BLENDING),
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        multiview_mask: None,
                        cache: None,
                    },
                ));
            }
        }

        let tonemap_shader =
            device.create_shader_module(wgpu::include_wgsl!("shaders/tonemap.wgsl"));
        let tonemap_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let tonemap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tonemap"),
                bind_group_layouts: &[Some(&tonemap_layout)],
                immediate_size: 0,
            });
        let tonemap_pipelines = [0.0, 1.0, 2.0].map(|method| {
            let constants = [("METHOD", method)];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("tonemap"),
                layout: Some(&tonemap_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &tonemap_shader,
                    entry_point: Some("tonemap_vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &tonemap_shader,
                    entry_point: Some("tonemap_fs"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &[Some(HDR_FORMAT.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("blit"),
                    bind_group_layouts: &[Some(&blit_layout)],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &tonemap_shader,
                entry_point: Some("tonemap_vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &tonemap_shader,
                entry_point: Some("blit_fs"),
                compilation_options: Default::default(),
                targets: &[Some(color_format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        // Bevy's tone-mapping LUT sampler: trilinear, clamped.
        let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tonemapping lut"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Hardware 2×2 PCF; the Gaussian filter combines 9 of these taps.
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sun shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::GreaterEqual),
            ..Default::default()
        });
        let shadow_linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Bevy `ImageSamplerDescriptor::linear()`.
        let environment_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("environment map"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let no_environment = upload_cube(
            device,
            queue,
            &Cubemap {
                size: 1,
                format: wgpu::TextureFormat::Rgba16Float,
                levels: vec![vec![0; 6 * 8]],
            },
        );

        Self {
            device: device.clone(),
            queue: queue.clone(),
            frame_layout,
            shadow_view_layout,
            lighting_layout,
            material_layout,
            tonemap_layout,
            blit_layout,
            blit_pipeline,
            shadow_pipelines,
            lit_pipelines,
            tonemap_pipelines,
            shadow_sampler,
            shadow_linear_sampler,
            environment_sampler,
            no_environment,
            tony_mc_mapface_lut: load_lut(device, queue, TONY_MC_MAPFACE_LUT),
            agx_lut: load_lut(device, queue, AGX_LUT),
            lut_sampler,
            sky: SkyPipelines::new(device, queue),
            particles: ParticlePipelines::new(device),
            sim: SimPipelines::new(device),
            no_transmittance: device
                .create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("no transmittance"),
                        size: wgpu::Extent3d::default(),
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    &[0; 8],
                )
                .create_view(&Default::default()),
        }
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn material_layout(&self) -> &wgpu::BindGroupLayout {
        &self.material_layout
    }

    pub(crate) fn sim_pipelines(&self) -> &SimPipelines {
        &self.sim
    }

    /// Writes one frame's camera, lights and instances into `viewport`. Only buffers whose
    /// contents changed size are reallocated; meshes, textures and materials stay in `world`.
    pub fn prepare(
        &self,
        viewport: &mut Viewport,
        world: &World,
        scene: &Scene,
        instances: &[Instance],
    ) {
        let (device, queue) = (&self.device, &self.queue);
        let aspect = viewport.width as f32 / viewport.height as f32;
        let cam = &scene.camera;
        let view_from_world = look_at(cam.eye, cam.target);
        let clip_from_view = perspective_infinite_reverse(cam.fov_y, aspect, cam.near);

        let shadows = scene.sun.and_then(|s| s.shadows.map(|cfg| (s, cfg)));
        let world_from_view = view_from_world.inverse();
        let cascades: Vec<(f32, Cascade)> = shadows.map_or_else(Vec::new, |(sun, cfg)| {
            cfg.cascade_ranges()
                .into_iter()
                .map(|(near, far)| {
                    let c = Cascade::fit(
                        near,
                        far,
                        scene.shadow_maps.directional,
                        sun.direction,
                        cam.fov_y,
                        aspect,
                        world_from_view,
                    );
                    (far, c)
                })
                .collect()
        });
        let mut cascade_clip_from_world = [Mat4::IDENTITY; MAX_CASCADES];
        let mut cascade_params = [Vec4::ZERO; MAX_CASCADES];
        for (i, (far, c)) in cascades.iter().enumerate() {
            cascade_clip_from_world[i] = c.clip_from_world;
            cascade_params[i] = Vec4::new(*far, c.texel_size, 0.0, 0.0);
        }

        let lights = LocalLights::new(scene, cascades.len());

        assert!(
            !(scene.environment.is_some()
                && scene.sky.is_some_and(|s| s.environment_intensity.is_some())),
            "a sky with environment light replaces Scene::environment; set one of them"
        );
        let (sky_lights, moon_first) = crate::sky::directional_lights(scene);
        let sky_environment = scene.sky.and_then(|s| s.environment_intensity);
        let sun = scene.sun;
        let frame = FrameUniform {
            clip_from_world: clip_from_view * view_from_world,
            view_from_world,
            cascade_clip_from_world,
            cascade_params,
            eye: cam.eye.extend(1.0),
            direction_to_light: sun.map_or(Vec4::ZERO, |s| (-s.direction.normalize()).extend(0.0)),
            sun_color: sun.map_or(Vec4::ZERO, |s| (s.color * s.illuminance).extend(1.0)),
            ambient_color: (scene.ambient.color * scene.ambient.brightness).extend(1.0),
            shadow: shadows.map_or(Vec4::ZERO, |(_, cfg)| {
                Vec4::new(
                    cfg.depth_bias,
                    // Bevy scales the normal bias by √2 for the worst-case diagonal texel offset.
                    cfg.normal_bias * std::f32::consts::SQRT_2,
                    cfg.overlap_proportion,
                    cascades.len() as f32,
                )
            }),
            flags: Vec4::new(
                cam.exposure(),
                f32::from(u8::from(shadows.is_some())),
                f32::from(u8::from(sun.is_some())),
                f32::from(u8::from(cam.shadow_filter == ShadowFilter::Gaussian)),
            ),
            environment: match (&scene.environment, sky_environment) {
                (Some(e), _) => {
                    Vec4::new(e.intensity, (e.specular.levels.len() - 1) as f32, 1.0, 0.0)
                }
                (None, Some(intensity)) => {
                    Vec4::new(intensity, (crate::sky::SPECULAR_MIPS - 1) as f32, 1.0, 0.0)
                }
                (None, None) => Vec4::ZERO,
            },
            fog_color: scene.fog.map_or(Vec4::ZERO, |f| f.color),
            fog_directional_light_color: scene
                .fog
                .map_or(Vec4::ZERO, |f| f.directional_light_color),
            fog_be: scene.fog.map_or(Vec4::ZERO, |f| {
                let be = match f.falloff {
                    FogFalloff::Linear { start, end } => Vec3::new(start, end, 0.0),
                    FogFalloff::Exponential { density }
                    | FogFalloff::ExponentialSquared { density } => Vec3::new(density, 0.0, 0.0),
                    FogFalloff::Atmospheric { extinction, .. } => extinction,
                };
                be.extend(f.directional_light_exponent)
            }),
            fog_bi: scene.fog.map_or(Vec4::ZERO, |f| match f.falloff {
                FogFalloff::Linear { .. } => Vec4::new(0.0, 0.0, 0.0, 1.0),
                FogFalloff::Exponential { .. } => Vec4::new(0.0, 0.0, 0.0, 2.0),
                FogFalloff::ExponentialSquared { .. } => Vec4::new(0.0, 0.0, 0.0, 3.0),
                FogFalloff::Atmospheric { inscattering, .. } => inscattering.extend(4.0),
            }),
            light_counts: UVec4::new(lights.point_count, lights.spot_count, 0, 0),
            moon_direction_to_light: scene
                .moon
                .map_or(Vec4::ZERO, |m| (-m.direction.normalize()).extend(1.0)),
            moon_color: scene
                .moon
                .map_or(Vec4::ZERO, |m| (m.color * m.illuminance).extend(1.0)),
            world_to_atmosphere: crate::sky::world_to_atmosphere(),
            atmosphere: Vec4::new(
                crate::sky::INNER_RADIUS,
                crate::sky::OUTER_RADIUS,
                f32::from(u8::from(scene.sky.is_some())),
                f32::from(u8::from(moon_first)),
            ),
            disks: scene.sky.map_or(Vec4::ZERO, |s| {
                Vec4::new(s.sun_disk.angular_size, s.moon_disk.angular_size, 0.0, 0.0)
            }),
        };
        queue.write_buffer(&viewport.frame_buffer, 0, bytemuck::bytes_of(&frame));

        // Instances, grouped into draws: opaque, then alpha-masked (each by pipeline, material
        // and mesh), then blended from back to front by origin (Bevy's transparent sort).
        struct Sorted {
            phase: Phase,
            face: usize,
            material: MaterialId,
            mesh: MeshId,
            view_z: f32,
            obb: Obb,
            data: InstanceData,
            skinned: bool,
        }
        // Each skin's palette once, in order of first use (Bevy `prepare_skins`).
        let mut joints: Vec<Mat4> = Vec::new();
        let mut skin_offsets: HashMap<SkinId, u32> = HashMap::new();
        let mut sorted: Vec<Sorted> = instances
            .iter()
            .map(|inst| {
                let material = &world.material(inst.material).material;
                let mesh = world.mesh(inst.mesh);
                let m = inst.world_from_local;
                let linear = Mat3::from_mat4(m);
                let mirrored = linear.determinant() < 0.0;
                let normal = linear.inverse().transpose();
                let skinned = mesh.skin.is_some();
                let (joint_offset, local_aabb) = match (skinned, inst.skin) {
                    (false, None) => (0, mesh.aabb),
                    (true, Some(skin)) => {
                        let palette = world.skin(skin);
                        let offset = *skin_offsets.entry(skin).or_insert_with(|| {
                            joints.extend_from_slice(palette);
                            (joints.len() - palette.len()) as u32
                        });
                        let aabb = mesh
                            .joint_bounds
                            .iter()
                            .filter_map(|(j, b)| Some(b.transformed(*palette.get(*j as usize)?)))
                            .reduce(|a, b| a.union(&b))
                            .unwrap_or(mesh.aabb);
                        (offset, aabb)
                    }
                    (true, None) => panic!("an instance of a skinned mesh needs a skin"),
                    (false, Some(_)) => panic!("a skin on an instance of an unskinned mesh"),
                };
                Sorted {
                    skinned,
                    phase: match material.alpha_mode {
                        AlphaMode::Opaque => Phase::Opaque,
                        AlphaMode::Mask(_) => Phase::Mask,
                        AlphaMode::Blend => Phase::Blend,
                    },
                    face: face_variant(mirrored, material.double_sided),
                    material: inst.material,
                    mesh: inst.mesh,
                    view_z: (view_from_world * m.w_axis).z,
                    obb: Obb::new(local_aabb, m),
                    data: InstanceData {
                        world_from_local: m,
                        normal_from_local: [
                            normal.x_axis.extend(0.0),
                            normal.y_axis.extend(0.0),
                            normal.z_axis.extend(0.0),
                        ],
                        flags: UVec4::new(
                            u32::from(mesh.has_tangents),
                            u32::from(mirrored),
                            joint_offset,
                            0,
                        ),
                    },
                }
            })
            .collect();
        sorted.sort_by(|a, b| {
            a.phase.cmp(&b.phase).then_with(|| {
                if a.phase == Phase::Blend {
                    a.view_z.total_cmp(&b.view_z)
                } else {
                    (a.face, a.material, a.mesh).cmp(&(b.face, b.material, b.mesh))
                }
            })
        });
        // The boxes on their own, compact for the per-view culling loops.
        let obbs: Vec<Obb> = sorted.iter().map(|s| s.obb).collect();
        let mut groups: Vec<Draw> = Vec::new();
        for (i, s) in sorted.iter().enumerate() {
            let i = i as u32;
            match groups.last_mut() {
                Some(d)
                    if (d.phase, d.face, d.material, d.mesh)
                        == (s.phase, s.face, s.material, s.mesh) =>
                {
                    d.instances.end = i + 1;
                }
                _ => groups.push(Draw {
                    face: s.face,
                    phase: s.phase,
                    skinned: s.skinned,
                    mesh: s.mesh,
                    material: s.material,
                    instances: i..i + 1,
                }),
            }
        }
        let mut instance_data: Vec<InstanceData> = sorted.iter().map(|s| s.data).collect();
        if instance_data.is_empty() {
            instance_data.push(InstanceData::zeroed());
        }
        if viewport
            .instance_buffer
            .write(device, queue, bytemuck::cast_slice(&instance_data))
        {
            viewport.frame_group = None;
        }
        if joints.is_empty() {
            joints.push(Mat4::IDENTITY);
        }
        if viewport
            .joint_buffer
            .write(device, queue, bytemuck::cast_slice(&joints))
        {
            viewport.frame_group = None;
        }

        let mut lighting_dirty = viewport.lighting_group.is_none();
        lighting_dirty |=
            viewport
                .light_buffer
                .write(device, queue, bytemuck::cast_slice(&lights.gpu));

        // Like Bevy, spot maps share the sun's array (and its size), after the cascades.
        let directional_layers = (cascades.len() + lights.spot_views.len()) as u32;
        let directional = if directional_layers > 0 {
            (scene.shadow_maps.directional, directional_layers)
        } else {
            (1, 1)
        };
        let shadow_arrays = &mut viewport.directional_shadows;
        if (shadow_arrays.size, shadow_arrays.layers) != directional {
            *shadow_arrays = ShadowArray::new(
                device,
                "directional shadow maps",
                directional.0,
                directional.1,
                false,
            );
            lighting_dirty = true;
        }
        let cube_count = (lights.point_views.len() / 6) as u32;
        let point = if cube_count > 0 {
            (scene.shadow_maps.point, 6 * cube_count)
        } else {
            (1, 6)
        };
        let point_arrays = &mut viewport.point_shadows;
        if (point_arrays.size, point_arrays.layers) != point {
            *point_arrays = ShadowArray::new(device, "point shadow maps", point.0, point.1, true);
            lighting_dirty = true;
        }

        // Shadow views: sun cascades, then spot maps, then point-light faces.
        viewport.shadow_passes.clear();
        let mut view_matrices = Vec::new();
        for (i, (_, c)) in cascades.iter().enumerate() {
            viewport.shadow_passes.push(ShadowPass {
                local: false,
                layer: i as u32,
                point: false,
                view: view_matrices.len() as u32,
                frustum: Frustum::cascade(c.clip_from_world),
                sphere: None,
                draws: 0..0,
            });
            view_matrices.push(c.clip_from_world);
        }
        let local_views = lights
            .spot_views
            .iter()
            .enumerate()
            .map(|(i, v)| ((cascades.len() + i) as u32, false, v))
            .chain(
                lights
                    .point_views
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i as u32, true, v)),
            );
        for (layer, point, v) in local_views {
            viewport.shadow_passes.push(ShadowPass {
                local: true,
                layer,
                point,
                view: view_matrices.len() as u32,
                frustum: v.frustum,
                sphere: Some(v.sphere),
                draws: 0..0,
            });
            view_matrices.push(v.clip_from_world);
        }

        // Frustum culling per view (Bevy `check_visibility` and the shadow views' culling):
        // each view keeps its own draws over a run of visible instance indices.
        viewport.draws.clear();
        let mut visible: Vec<u32> = Vec::new();
        let mut cull = |draws: &mut Vec<Draw>, sees: &dyn Fn(usize) -> bool| {
            let start = draws.len();
            for group in &groups {
                let first = visible.len() as u32;
                visible.extend(group.instances.clone().filter(|&i| sees(i as usize)));
                let end = visible.len() as u32;
                if end > first {
                    draws.push(Draw {
                        instances: first..end,
                        ..group.clone()
                    });
                }
            }
            start..draws.len()
        };
        let camera = Frustum::camera(clip_from_view * view_from_world);
        viewport.main_draws = cull(&mut viewport.draws, &|i| camera.intersects(&obbs[i]));
        // A light's sphere test runs once for all its views (six cube faces).
        let mut in_range: Option<((Vec3, f32), Vec<bool>)> = None;
        for pass in &mut viewport.shadow_passes {
            let draws = match pass.sphere {
                None => cull(&mut viewport.draws, &|i| pass.frustum.intersects(&obbs[i])),
                Some(sphere) => {
                    if in_range.as_ref().is_none_or(|(s, _)| *s != sphere) {
                        let near = obbs
                            .iter()
                            .map(|o| o.intersects_sphere(sphere.0, sphere.1))
                            .collect();
                        in_range = Some((sphere, near));
                    }
                    let near = &in_range.as_ref().expect("filled above").1;
                    cull(&mut viewport.draws, &|i| {
                        near[i] && pass.frustum.intersects(&obbs[i])
                    })
                }
            };
            pass.draws = draws;
        }
        if visible.is_empty() {
            visible.push(0);
        }
        if viewport
            .visible_buffer
            .write(device, queue, bytemuck::cast_slice(&visible))
        {
            viewport.frame_group = None;
        }
        let mut view_bytes = vec![0u8; SHADOW_VIEW_STRIDE as usize * view_matrices.len().max(1)];
        for (i, m) in view_matrices.iter().enumerate() {
            let at = i * SHADOW_VIEW_STRIDE as usize;
            view_bytes[at..at + 64].copy_from_slice(bytemuck::bytes_of(m));
        }
        if viewport
            .shadow_view_buffer
            .write(device, queue, &view_bytes)
        {
            viewport.shadow_view_group = None;
        }

        // Environment maps are uploaded once per distinct pair of cube maps.
        let cached = match (&scene.environment, &viewport.environment) {
            (Some(e), Some(u)) => {
                Arc::ptr_eq(&e.diffuse, &u.diffuse) && Arc::ptr_eq(&e.specular, &u.specular)
            }
            (None, None) => true,
            _ => false,
        };
        if !cached {
            viewport.environment = scene.environment.as_ref().map(|e| UploadedEnvironment {
                diffuse: e.diffuse.clone(),
                specular: e.specular.clone(),
                diffuse_view: upload_cube(device, queue, &e.diffuse),
                specular_view: upload_cube(device, queue, &e.specular),
            });
            lighting_dirty = true;
        }

        match (&scene.sky, viewport.sky.is_some()) {
            (Some(_), false) => {
                viewport.sky = Some(SkyTargets::new(device, &self.sky));
                lighting_dirty = true;
            }
            (None, true) => {
                viewport.sky = None;
                lighting_dirty = true;
            }
            _ => {}
        }
        if let (Some(sky), Some(targets)) = (&scene.sky, &mut viewport.sky) {
            targets.prepare(
                device,
                queue,
                &self.sky,
                sky,
                &sky_lights,
                cam,
                // The camera transform itself, as Bevy's view uniform has it.
                Mat4::from_translation(cam.eye)
                    * Mat4::from_mat3(looking_to(cam.target - cam.eye, Vec3::Y)),
                clip_from_view,
                &viewport.targets.depth_view,
            );
        }

        if lighting_dirty {
            viewport.lighting_group = Some(self.lighting_group(viewport));
        }
        if viewport.frame_group.is_none() {
            viewport.frame_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("frame"),
                layout: &self.frame_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: viewport.frame_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: viewport.instance_buffer.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: viewport.visible_buffer.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: viewport.joint_buffer.buffer.as_entire_binding(),
                    },
                ],
            }));
        }
        if viewport.shadow_view_group.is_none() {
            viewport.shadow_view_group =
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shadow view"),
                    layout: &self.shadow_view_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &viewport.shadow_view_buffer.buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(64),
                        }),
                    }],
                }));
        }
        if viewport.tonemap.as_ref().map(|(t, _)| *t) != Some(cam.tonemapping) {
            viewport.tonemap = Some((
                cam.tonemapping,
                self.tonemap_group(viewport, cam.tonemapping),
            ));
        }
        viewport.tonemapping = cam.tonemapping;
        viewport.particles.prepare(
            device,
            queue,
            &self.particles,
            world,
            clip_from_view * view_from_world,
            world_from_view,
        );
        let c = scene.clear_color;
        viewport.clear_color = wgpu::Color {
            r: c.x.into(),
            g: c.y.into(),
            b: c.z.into(),
            a: 1.0,
        };
    }

    fn lighting_group(&self, viewport: &Viewport) -> wgpu::BindGroup {
        let (diffuse_view, specular_view) = match (&viewport.environment, &viewport.sky) {
            (Some(e), _) => (&e.diffuse_view, &e.specular_view),
            (None, Some(sky)) => (&sky.diffuse_view, &sky.specular_view),
            (None, None) => (&self.no_environment, &self.no_environment),
        };
        let transmittance = viewport
            .sky
            .as_ref()
            .map_or(&self.no_transmittance, |s| &s.transmittance_view);
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting"),
            layout: &self.lighting_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(diffuse_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(specular_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.environment_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &viewport.directional_shadows.view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: viewport.light_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&viewport.point_shadows.view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(transmittance),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.sky.lut_sampler),
                },
            ],
        })
    }

    fn tonemap_group(&self, viewport: &Viewport, tonemapping: Tonemapping) -> wgpu::BindGroup {
        // Like Bevy, operators without a LUT still bind one (the small AgX LUT).
        let lut = match tonemapping {
            Tonemapping::TonyMcMapface => &self.tony_mc_mapface_lut,
            Tonemapping::AcesFitted | Tonemapping::AgX => &self.agx_lut,
        };
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap"),
            layout: &self.tonemap_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&viewport.targets.hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(lut),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.lut_sampler),
                },
            ],
        })
    }

    /// Records the frame last prepared into `viewport`, ending in `target`: a view of the colour
    /// format the renderer was created with, `viewport`-sized, fully overwritten. Draw the UI
    /// into `target` afterwards (`LoadOp::Load`).
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
        world: &World,
        target: &wgpu::TextureView,
        timestamps: Option<FrameTimestamps<'_>>,
    ) {
        let frame_group = viewport
            .frame_group
            .as_ref()
            .expect("Renderer::prepare before encode");
        let draw = |pass: &mut wgpu::RenderPass<'_>,
                    pipelines: &[wgpu::RenderPipeline],
                    draws: Range<usize>,
                    phases: Range<Phase>,
                    index: &dyn Fn(&Draw) -> usize| {
            pass.set_bind_group(0, frame_group, &[]);
            let (mut mesh_bound, mut material_bound) = (None, None);
            for d in viewport.draws[draws]
                .iter()
                .filter(|d| phases.contains(&d.phase))
            {
                pass.set_pipeline(&pipelines[index(d)]);
                if material_bound != Some(d.material) {
                    pass.set_bind_group(2, &world.material(d.material).group, &[]);
                    material_bound = Some(d.material);
                }
                let mesh = world.mesh(d.mesh);
                if mesh_bound != Some(d.mesh) {
                    pass.set_vertex_buffer(0, mesh.positions.slice(..));
                    pass.set_vertex_buffer(1, mesh.normals.slice(..));
                    pass.set_vertex_buffer(2, mesh.surface.slice(..));
                    if let Some(skin) = &mesh.skin {
                        pass.set_vertex_buffer(3, skin.slice(..));
                    }
                    pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                    mesh_bound = Some(d.mesh);
                }
                pass.draw_indexed(0..mesh.index_count, 0, d.instances.clone());
            }
        };
        let stamps = Stamps {
            query_set: timestamps.as_ref().map(|t| t.query_set),
            next: timestamps.as_ref().map_or(0, |t| t.first).into(),
        };
        if let Some(sky) = &viewport.sky {
            sky.encode_luts(encoder, &self.sky, &stamps);
        }

        for shadow in &viewport.shadow_passes {
            let maps = if shadow.point {
                &viewport.point_shadows
            } else {
                &viewport.directional_shadows
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &maps.layer_views[shadow.layer as usize],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: stamps.render(),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(
                1,
                viewport.shadow_view_group.as_ref(),
                &[shadow.view * SHADOW_VIEW_STRIDE as u32],
            );
            draw(
                &mut pass,
                &self.shadow_pipelines,
                shadow.draws.clone(),
                Phase::Opaque..Phase::End,
                &|d: &Draw| {
                    shadow_pipeline_index(d.skinned, shadow.local, d.face, d.phase != Phase::Opaque)
                },
            );
        }

        // The main pass. With a sky it is split like Bevy's: opaque and alpha-masked draws, the
        // sky over them (reading their depth), then the blended draws.
        let targets = &viewport.targets;
        let part = |label, clear, depth, resolve| MainPass {
            label,
            clear,
            depth,
            resolve,
        };
        let clear_color = viewport.clear_color;
        let lit_index = |d: &Draw| lit_pipeline_index(d.skinned, d.face, d.phase);
        match &viewport.sky {
            None => {
                let mut pass = begin_main_pass(
                    encoder,
                    targets,
                    clear_color,
                    part("lit", true, true, true),
                    stamps.render(),
                );
                pass.set_bind_group(1, viewport.lighting_group.as_ref(), &[]);
                draw(
                    &mut pass,
                    &self.lit_pipelines,
                    viewport.main_draws.clone(),
                    Phase::Opaque..Phase::End,
                    &lit_index,
                );
                viewport.particles.draw(&mut pass, &self.particles, world);
            }
            Some(sky) => {
                let mut pass = begin_main_pass(
                    encoder,
                    targets,
                    clear_color,
                    part("lit opaque", true, true, false),
                    stamps.render(),
                );
                pass.set_bind_group(1, viewport.lighting_group.as_ref(), &[]);
                draw(
                    &mut pass,
                    &self.lit_pipelines,
                    viewport.main_draws.clone(),
                    Phase::Opaque..Phase::Blend,
                    &lit_index,
                );
                drop(pass);
                let mut pass = begin_main_pass(
                    encoder,
                    targets,
                    clear_color,
                    part("render sky", false, false, false),
                    stamps.render(),
                );
                sky.draw(&mut pass, &self.sky);
                drop(pass);
                let mut pass = begin_main_pass(
                    encoder,
                    targets,
                    clear_color,
                    part("lit transparent", false, true, true),
                    stamps.render(),
                );
                pass.set_bind_group(1, viewport.lighting_group.as_ref(), &[]);
                draw(
                    &mut pass,
                    &self.lit_pipelines,
                    viewport.main_draws.clone(),
                    Phase::Blend..Phase::End,
                    &lit_index,
                );
                viewport.particles.draw(&mut pass, &self.particles, world);
            }
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tonemap"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.post_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: stamps.render(),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.tonemap_pipelines[tonemap_index(viewport.tonemapping)]);
        pass.set_bind_group(0, viewport.tonemap.as_ref().map(|(_, g)| g), &[]);
        pass.draw(0..3, 0..1);
        drop(pass);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: stamps.render(),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, &targets.blit_group, &[]);
        pass.draw(0..3, 0..1);
    }
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
            clear_color: wgpu::Color::BLACK,
            tonemapping: Tonemapping::default(),
            particles: ParticleFrame::new(device),
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

    /// Timestamp queries [`Renderer::encode`] writes for this viewport's last prepared frame: a
    /// begin and an end per pass.
    pub fn timestamp_queries(&self) -> u32 {
        let sky = self.sky.as_ref().map_or(0, |s| s.compute_passes());
        // The main pass is three with a sky; then tone mapping and the blit.
        let main = if self.sky.is_some() { 3 } else { 1 };
        2 * (sky + self.shadow_passes.len() as u32 + main + 2)
    }
}

impl Targets {
    fn new(renderer: &Renderer, width: u32, height: u32) -> Self {
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

/// How one part of the main pass treats the MSAA colour and depth targets.
struct MainPass {
    label: &'static str,
    /// Clear colour and depth (else load them).
    clear: bool,
    /// Attach the depth target.
    depth: bool,
    /// Resolve into the HDR frame and discard the MSAA targets (else store them for the next
    /// part).
    resolve: bool,
}

fn begin_main_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    targets: &'a Targets,
    clear_color: wgpu::Color,
    p: MainPass,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'a>>,
) -> wgpu::RenderPass<'a> {
    let store = if p.resolve {
        wgpu::StoreOp::Discard
    } else {
        wgpu::StoreOp::Store
    };
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(p.label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &targets.msaa_view,
            depth_slice: None,
            resolve_target: p.resolve.then_some(&targets.hdr_view),
            ops: wgpu::Operations {
                load: if p.clear {
                    wgpu::LoadOp::Clear(clear_color)
                } else {
                    wgpu::LoadOp::Load
                },
                store,
            },
        })],
        depth_stencil_attachment: p.depth.then_some(wgpu::RenderPassDepthStencilAttachment {
            view: &targets.depth_view,
            depth_ops: Some(wgpu::Operations {
                load: if p.clear {
                    wgpu::LoadOp::Clear(0.0)
                } else {
                    wgpu::LoadOp::Load
                },
                store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// Uploads one of Bevy's tone-mapping LUTs: a zstd-supercompressed single-level 3D KTX2 file.
fn load_lut(device: &wgpu::Device, queue: &wgpu::Queue, ktx2: &[u8]) -> wgpu::TextureView {
    let lut = crate::ktx::decode(ktx2).expect("embedded LUT is a supported KTX2 file");
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("tonemapping lut"),
            size: wgpu::Extent3d {
                width: lut.width,
                height: lut.height,
                depth_or_array_layers: lut.depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: lut.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &lut.levels[0],
    );
    texture.create_view(&Default::default())
}

/// Uploads a cube map with all its mip levels.
fn upload_cube(device: &wgpu::Device, queue: &wgpu::Queue, cube: &Cubemap) -> wgpu::TextureView {
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("environment map"),
            size: wgpu::Extent3d {
                width: cube.size,
                height: cube.size,
                depth_or_array_layers: 6,
            },
            mip_level_count: cube.levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: cube.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::MipMajor,
        &cube.levels.concat(),
    );
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    })
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

/// The scene's point and spot lights as the shader reads them, with their shadow views.
struct LocalLights {
    /// Point lights, then spot lights, each with the shadow casters first (Bevy's clustered
    /// light order). Never empty: a zeroed light stands in for none.
    gpu: Vec<GpuLight>,
    point_count: u32,
    spot_count: u32,
    /// Per spot shadow map, in layer order after the sun cascades.
    spot_views: Vec<LightView>,
    /// Per point-light cube face, six per shadow-casting point light.
    point_views: Vec<LightView>,
}

struct LightView {
    clip_from_world: Mat4,
    frustum: Frustum,
    /// The light's position and range: Bevy first drops casters outside this sphere.
    sphere: (Vec3, f32),
}

impl LightView {
    fn new(clip_from_world: Mat4, position: Vec3, forward: Vec3, range: f32) -> Self {
        Self {
            clip_from_world,
            frustum: Frustum::light(clip_from_world, position, forward, range),
            sphere: (position, range),
        }
    }
}

impl LocalLights {
    /// Bevy's `extract_lights` and `prepare_lights` for point and spot lights. `cascades` is the
    /// number of sun cascade layers ahead of the spot maps.
    fn new(scene: &Scene, cascades: usize) -> Self {
        use std::f32::consts::{FRAC_PI_2, PI, SQRT_2};
        let sizes = scene.shadow_maps;
        let base = |position: Vec3, color: Vec3, intensity: f32, range: f32, radius: f32| {
            GpuLight {
                // Luminous power to luminous intensity (lm/sr), for spot lights too.
                color_inverse_square_range: (color * (intensity / (4.0 * PI)))
                    .extend(1.0 / (range * range)),
                position_radius: position.extend(radius),
                ..GpuLight::zeroed()
            }
        };
        let shadowed = |light: &mut GpuLight, s: LightShadows, texel_size: f32, radius: f32| {
            light.flags |= LIGHT_SHADOWS;
            light.shadow_depth_bias = s.depth_bias;
            // √2 for the worst-case diagonal texel offset, as in Bevy.
            light.shadow_normal_bias = s.normal_bias * texel_size * SQRT_2;
            light.shadow_map_near_z = s.near_z;
            light.soft_shadow_size = if s.soft { radius } else { 0.0 };
        };

        let mut gpu = Vec::new();
        let mut point_views = Vec::new();
        let mut spot_views = Vec::new();
        let mut points: Vec<_> = scene.point_lights.iter().collect();
        points.sort_by_key(|l| l.shadows.is_none());
        for l in points {
            let mut light = base(l.position, l.color, l.intensity, l.range, l.radius);
            let near_z = l.shadows.map_or(LightShadows::POINT.near_z, |s| s.near_z);
            let projection = perspective_infinite_reverse(FRAC_PI_2, 1.0, near_z);
            light.light_custom_data = Vec4::new(
                projection.z_axis.z,
                projection.z_axis.w,
                projection.w_axis.z,
                projection.w_axis.w,
            );
            if let Some(s) = l.shadows {
                shadowed(&mut light, s, 2.0 / sizes.point as f32, l.radius);
                light.shadow_index = (point_views.len() / 6) as u32;
                // Axis-aligned faces: the light's translation only.
                for (target, up) in CUBE_FACES {
                    let world_from_view = Mat4::from_translation(l.position)
                        * Mat4::from_mat3(looking_to(target, up));
                    point_views.push(LightView::new(
                        projection * world_from_view.inverse(),
                        l.position,
                        target,
                        l.range,
                    ));
                }
            }
            gpu.push(light);
        }
        let point_count = gpu.len() as u32;

        let mut spots: Vec<_> = scene.spot_lights.iter().collect();
        spots.sort_by_key(|l| l.shadows.is_none());
        for l in spots {
            let mut light = base(l.position, l.color, l.intensity, l.range, l.radius);
            let outer = l.outer_angle.min(FRAC_PI_2 - 1e-4);
            let inner = l.inner_angle.min(outer);
            let direction = l.direction.normalize();
            if direction.y.is_sign_negative() {
                light.flags |= LIGHT_SPOT_Y_NEGATIVE;
            }
            let cos_outer = outer.cos();
            let spot_scale = 1.0 / (inner.cos() - cos_outer).max(1e-4);
            light.light_custom_data = Vec4::new(
                direction.x,
                direction.z,
                spot_scale,
                -cos_outer * spot_scale,
            );
            light.spot_light_tan_angle = outer.tan();
            if let Some(s) = l.shadows {
                let texel_size = 2.0 * outer.tan() / sizes.directional as f32;
                shadowed(&mut light, s, texel_size, l.radius);
                light.shadow_index = (cascades + spot_views.len()) as u32;
                // Bevy `spot_light_world_from_view` / `spot_light_clip_from_view`.
                let world_from_view = Mat4::from_translation(l.position)
                    * Mat4::from_mat3(orthonormalize(-direction));
                let clip_from_view = perspective_infinite_reverse(2.0 * outer, 1.0, s.near_z);
                spot_views.push(LightView::new(
                    clip_from_view * world_from_view.inverse(),
                    l.position,
                    direction,
                    l.range,
                ));
            }
            gpu.push(light);
        }
        let spot_count = gpu.len() as u32 - point_count;
        if gpu.is_empty() {
            gpu.push(GpuLight::zeroed());
        }
        Self {
            gpu,
            point_count,
            spot_count,
            spot_views,
            point_views,
        }
    }
}

/// Bevy's `orthonormalize`: a right-handed basis whose z axis is `z` (the shaders rebuild the
/// same basis from the spot direction).
fn orthonormalize(z: Vec3) -> Mat3 {
    let sign = 1f32.copysign(z.z);
    let a = -1.0 / (sign + z.z);
    let b = z.x * z.y * a;
    let x = Vec3::new(1.0 + sign * z.x * z.x * a, sign * b, -sign * z.x);
    let y = Vec3::new(b, sign + z.y * z.y * a, -z.y);
    Mat3::from_cols(x, y, z)
}

/// The rotation of Bevy `Transform::looking_to(direction, up)`.
fn looking_to(direction: Vec3, up: Vec3) -> Mat3 {
    let back = -direction.normalize();
    let right = up.cross(back).normalize();
    Mat3::from_cols(right, back.cross(right), back)
}

/// Right-handed view matrix with +Y up (the inverse of Bevy `Transform::looking_at(target, Y)`).
pub fn look_at(eye: Vec3, target: Vec3) -> Mat4 {
    (Mat4::from_translation(eye) * Mat4::from_mat3(looking_to(target - eye, Vec3::Y))).inverse()
}

/// Bevy's right-handed infinite reverse-Z perspective projection.
pub(crate) fn perspective_infinite_reverse(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let h = 1.0 / (0.5 * fov_y).tan();
    Mat4::from_cols(
        Vec4::new(h / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, h, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, -1.0),
        Vec4::new(0.0, 0.0, near, 0.0),
    )
}

/// One sun shadow cascade, fitted to the camera frustum slice `near..far` (view distances)
/// exactly like Bevy's `calculate_cascade` (diameter rounded up to whole metres, centre snapped
/// to texels).
#[derive(Clone, Copy, Debug)]
pub struct Cascade {
    pub clip_from_world: Mat4,
    pub texel_size: f32,
}

impl Cascade {
    pub fn fit(
        near: f32,
        far: f32,
        map_size: u32,
        sun_direction: Vec3,
        fov_y: f32,
        aspect: f32,
        world_from_view: Mat4,
    ) -> Self {
        // Light orientation of Bevy `Transform::looking_to(direction, Vec3::Y)`.
        let back = -sun_direction.normalize();
        let right = Vec3::Y.cross(back).normalize();
        let up = back.cross(right);
        let world_from_light = Mat4::from_mat3(Mat3::from_cols(right, up, back));
        let light_from_view = world_from_light.transpose() * world_from_view;

        let t = (fov_y * 0.5).tan();
        let corners = [-near, -far].map(|z| {
            let a = z.abs() * t;
            [
                Vec3::new(a * aspect, -a, z),
                Vec3::new(a * aspect, a, z),
                Vec3::new(-a * aspect, a, z),
                Vec3::new(-a * aspect, -a, z),
            ]
        });
        let corners: [Vec3; 8] = [corners[0], corners[1]].concat().try_into().unwrap();
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for c in corners {
            let p = light_from_view.transform_point3(c);
            min = min.min(p);
            max = max.max(p);
        }
        let body = (corners[0] - corners[6]).length_squared();
        let far = (corners[4] - corners[6]).length_squared();
        let diameter = body.max(far).sqrt().ceil();
        let texel_size = diameter / map_size as f32;
        let centre = Vec3::new(
            (0.5 * (min.x + max.x) / texel_size).floor() * texel_size,
            (0.5 * (min.y + max.y) / texel_size).floor() * texel_size,
            max.z,
        );
        let t = world_from_light.transpose();
        let cascade_from_world =
            Mat4::from_cols(t.x_axis, t.y_axis, t.z_axis, (-centre).extend(1.0));
        let r = (max.z - min.z).recip();
        let clip_from_cascade = Mat4::from_cols(
            Vec4::new(2.0 / diameter, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 / diameter, 0.0, 0.0),
            Vec4::new(0.0, 0.0, r, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 1.0),
        );
        Self {
            clip_from_world: clip_from_cascade * cascade_from_world,
            texel_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascade_covers_the_view_slice() {
        let eye = Vec3::new(6.0, 5.0, 8.0);
        let world_from_view = look_at(eye, Vec3::ZERO).inverse();
        let c = Cascade::fit(
            0.1,
            30.0,
            2048,
            Vec3::new(-0.4, -1.0, -0.3),
            45f32.to_radians(),
            16.0 / 9.0,
            world_from_view,
        );
        // A point the camera looks at, inside the shadow range, lands inside the cascade.
        let p = c.clip_from_world.project_point3(Vec3::ZERO);
        assert!(
            p.x.abs() < 1.0 && p.y.abs() < 1.0 && (0.0..=1.0).contains(&p.z),
            "{p}"
        );
        assert!(c.texel_size > 0.0 && c.texel_size < 0.05);
    }

    #[test]
    fn default_cascades_follow_bevy_spacing() {
        let ranges = crate::scene::SunShadows::default().cascade_ranges();
        let expected = [(0.1, 10.0), (8.0, 24.662), (19.73, 60.822), (48.66, 150.0)];
        assert_eq!(ranges.len(), expected.len());
        for ((near, far), (en, ef)) in ranges.into_iter().zip(expected) {
            assert!(
                (near - en).abs() < 0.01 && (far - ef).abs() < 0.01,
                "{near}..{far}"
            );
        }
    }
}
