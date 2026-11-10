//! Draws a [`World`]'s particles in the main pass, after the blended instances: one instanced
//! quad per alive particle, read from the quads the last simulation step wrote
//! (`particle_sim`), one indirect draw per emitter. Emitters are drawn back to front by their
//! position, additive and alpha-blended alike, as Bevy's transparent phase sorts hanabi's
//! effects.
//!
//! Soft emitters ([`Emitter::depth_fade`]) need the opaque scene's depth while they draw, so a
//! frame with one draws every particle in a pass of its own that binds the depth target
//! read-only and also reads it as a texture (`fragment_soft`). Without one the particles stay in
//! the main pass, as before.
//!
//! Lit and fogged emitters ([`Emitter::lighting`], [`Emitter::fog`]) draw with pipelines compiled
//! from `lit.wgsl`, `particles.wgsl` and `particles_shaded.wgsl` together: they bind the lit
//! pass's frame and lighting groups (0 and 1) and their own at 2 and 3. Unlit, unfogged emitters
//! keep hanabi's shader alone.

use crate::{
    particle_sim::{COUNTER_BYTES, QUAD_BYTES},
    particles::{Emitter, Orient, ParticleBlend, TextureMapping},
    renderer::{DEPTH_FORMAT, Growable, HDR_FORMAT, SAMPLE_COUNT},
    world::{TextureId, World},
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec4};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewUniform {
    clip_from_world: Mat4,
    right: Vec4,
    up: Vec4,
    eye: Vec4,
    /// The camera's near plane (x): reverse-Z infinite depth `d` is `near / view distance`.
    params: Vec4,
}

/// `EMITTER_LIT` and `EMITTER_FOG` in `particles.wgsl`.
const EMITTER_LIT: u32 = 1;
const EMITTER_FOG: u32 = 2;

/// Per drawn emitter, at a dynamic offset of `EMITTER_STRIDE` (`Emitter` in `particles.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmitterUniform {
    roundness: f32,
    orient: u32,
    mapping: u32,
    flipbook_columns: u32,
    flipbook_scale: Vec2,
    depth_fade: f32,
    flags: u32,
    /// Lit particles: perceptual roughness, metallic, reflectance.
    surface: Vec4,
}

impl EmitterUniform {
    fn new(e: &Emitter) -> Self {
        let flipbook = e.texture.and_then(|t| t.flipbook);
        let lighting = e.lighting.unwrap_or_default();
        Self {
            roundness: e.roundness,
            orient: match e.orient {
                Orient::CameraPlane => 0,
                Orient::AlongVelocity => 1,
            },
            mapping: match e.texture.map(|t| t.mapping) {
                None => 0,
                Some(TextureMapping::Modulate) => 1,
                Some(TextureMapping::ModulateRgb) => 2,
                Some(TextureMapping::ModulateOpacityFromR) => 3,
            },
            flipbook_columns: flipbook.map_or(0, |f| f.grid.x),
            // hanabi's `FLIPBOOK_SCALE`, computed on the CPU.
            flipbook_scale: flipbook.map_or(Vec2::ONE, |f| {
                Vec2::new(1.0 / f.grid.x as f32, 1.0 / f.grid.y as f32)
            }),
            depth_fade: e.depth_fade,
            flags: if e.lighting.is_some() { EMITTER_LIT } else { 0 }
                | if e.fog { EMITTER_FOG } else { 0 },
            surface: Vec4::new(
                lighting.perceptual_roughness,
                lighting.metallic,
                lighting.reflectance,
                0.0,
            ),
        }
    }
}

/// `min_uniform_buffer_offset_alignment`'s largest allowed value.
const EMITTER_STRIDE: u64 = 256;

/// A particle pipeline's variant: whether it reads the scene depth (soft: the particle pass,
/// whose depth attachment is read-only) and whether it is lit or fogged (shaded).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Variant {
    soft: bool,
    shaded: bool,
    blend: usize,
}

impl Variant {
    fn index(self) -> usize {
        usize::from(self.soft) * 4 + usize::from(self.shaded) * 2 + self.blend
    }
}

/// Every [`Variant`]'s pipeline, by [`Variant::index`].
pub(crate) struct ParticlePipelines {
    /// The view and emitter uniforms (group 0 unshaded, 2 shaded).
    layout: wgpu::BindGroupLayout,
    /// `layout` plus the scene depth, for soft emitters.
    soft_layout: wgpu::BindGroupLayout,
    /// Each world texture's particle bind group is built with this layout (`World`); group 1
    /// unshaded, 3 shaded.
    pub texture_layout: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::RenderPipeline>,
}

fn blend_index(b: ParticleBlend) -> usize {
    match b {
        ParticleBlend::Add => 0,
        ParticleBlend::Alpha => 1,
    }
}

/// The shaded particles' module: `lit.wgsl`, then `particles.wgsl` with its groups moved from 0
/// and 1 to 2 and 3, then `particles_shaded.wgsl`.
fn shaded_source() -> String {
    let particles = include_str!("shaders/particles.wgsl")
        .replace("@group(0)", "@group(2)")
        .replace("@group(1)", "@group(3)");
    [
        include_str!("shaders/lit.wgsl"),
        &particles,
        include_str!("shaders/particles_shaded.wgsl"),
    ]
    .join("\n")
}

impl ParticlePipelines {
    /// `frame_layout` and `lighting_layout` are the lit pass's groups 0 and 1, which shaded
    /// particles bind too.
    pub fn new(
        device: &wgpu::Device,
        frame_layout: &wgpu::BindGroupLayout,
        lighting_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/particles.wgsl"));
        let shaded_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shaded particles"),
            source: wgpu::ShaderSource::Wgsl(shaded_source().into()),
        });
        let uniform = |binding, has_dynamic_offset, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset,
                min_binding_size: None,
            },
            count: None,
        };
        let view_entries = [
            uniform(0, false, wgpu::ShaderStages::VERTEX_FRAGMENT),
            uniform(1, true, wgpu::ShaderStages::VERTEX_FRAGMENT),
        ];
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle view"),
            entries: &view_entries,
        });
        let soft_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soft particle view"),
            entries: &[
                view_entries[0],
                view_entries[1],
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                },
            ],
        });
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle texture"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // `Quad` in `particle_sim.wgsl`.
        let attributes = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 16,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 32,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 48,
                shader_location: 4,
            },
        ];
        // hanabi `AlphaMode::Add`: colour scaled by alpha onto the target, alpha kept.
        let add = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let blends = [add, wgpu::BlendState::ALPHA_BLENDING];
        let mut pipelines = Vec::with_capacity(8);
        for soft in [false, true] {
            let view = if soft { &soft_layout } else { &layout };
            for shaded in [false, true] {
                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("particles"),
                        bind_group_layouts: &if shaded {
                            vec![
                                Some(frame_layout),
                                Some(lighting_layout),
                                Some(view),
                                Some(&texture_layout),
                            ]
                        } else {
                            vec![Some(view), Some(&texture_layout)]
                        },
                        immediate_size: 0,
                    });
                let module = if shaded { &shaded_shader } else { &shader };
                let fragment = match (shaded, soft) {
                    (false, false) => "fragment",
                    (false, true) => "fragment_soft",
                    (true, false) => "fragment_shaded",
                    (true, true) => "fragment_shaded_soft",
                };
                for blend in blends {
                    pipelines.push(device.create_render_pipeline(
                        &wgpu::RenderPipelineDescriptor {
                            label: Some(fragment),
                            layout: Some(&pipeline_layout),
                            vertex: wgpu::VertexState {
                                module,
                                entry_point: Some("vertex"),
                                compilation_options: Default::default(),
                                buffers: &[Some(wgpu::VertexBufferLayout {
                                    array_stride: QUAD_BYTES,
                                    step_mode: wgpu::VertexStepMode::Instance,
                                    attributes: &attributes,
                                })],
                            },
                            primitive: wgpu::PrimitiveState::default(),
                            // Like hanabi's transparent particles: test depth, never write it.
                            depth_stencil: Some(wgpu::DepthStencilState {
                                format: DEPTH_FORMAT,
                                depth_write_enabled: Some(false),
                                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                                stencil: wgpu::StencilState::default(),
                                bias: wgpu::DepthBiasState::default(),
                            }),
                            multisample: wgpu::MultisampleState {
                                count: SAMPLE_COUNT,
                                ..Default::default()
                            },
                            fragment: Some(wgpu::FragmentState {
                                module,
                                entry_point: Some(fragment),
                                compilation_options: Default::default(),
                                targets: &[Some(wgpu::ColorTargetState {
                                    format: HDR_FORMAT,
                                    blend: Some(blend),
                                    write_mask: wgpu::ColorWrites::ALL,
                                })],
                            }),
                            multiview_mask: None,
                            cache: None,
                        },
                    ));
                }
            }
        }
        Self {
            layout,
            soft_layout,
            texture_layout,
            pipelines,
        }
    }
}

/// One drawn emitter: its pipeline, its id (draw arguments), its slot range (quads) and its
/// texture (its bind group lives with the texture in the world).
struct EmitterDraw {
    variant: Variant,
    id: u32,
    base: u32,
    capacity: u32,
    texture: Option<TextureId>,
}

/// One viewport's particle draws for the frame.
pub(crate) struct ParticleFrame {
    view_buffer: wgpu::Buffer,
    emitter_buffer: Growable,
    group: Option<wgpu::BindGroup>,
    /// `group` plus the scene depth, for soft emitters; rebuilt after the depth target or the
    /// emitter buffer changes.
    soft_group: Option<wgpu::BindGroup>,
    draws: Vec<EmitterDraw>,
}

impl ParticleFrame {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            view_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("particle view"),
                size: std::mem::size_of::<ViewUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            emitter_buffer: Growable::new(device, "particle emitters", wgpu::BufferUsages::UNIFORM),
            group: None,
            soft_group: None,
            draws: Vec::new(),
        }
    }

    /// Whether this frame draws a soft emitter, and so draws its particles in their own pass
    /// (`Renderer::encode`).
    pub fn soft(&self) -> bool {
        self.draws.iter().any(|d| d.variant.soft)
    }

    /// Drops the bind group that reads the old depth target (after a resize).
    pub fn depth_changed(&mut self) {
        self.soft_group = None;
    }

    /// Orders this frame's emitters of `world` as seen through `clip_from_world` from a camera
    /// whose world-space axes are the columns of `world_from_view`, with its near plane at
    /// `near`; soft emitters read `depth_view`.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipelines: &ParticlePipelines,
        world: &World,
        clip_from_world: Mat4,
        world_from_view: Mat4,
        near: f32,
        depth_view: &wgpu::TextureView,
    ) {
        self.draws.clear();
        let forward = -world_from_view.z_axis.truncate();
        let eye = world_from_view.w_axis.truncate();
        // (view depth, draw, uniform); emitters not stepped yet have nothing to draw.
        let mut emitters: Vec<(f32, EmitterDraw, EmitterUniform)> = world
            .emitter_ids()
            .filter_map(|(id, e)| {
                let depth = (e.transform.w_axis.truncate() - eye).dot(forward);
                let draw = EmitterDraw {
                    variant: Variant {
                        soft: e.desc.depth_fade > 0.0,
                        shaded: e.desc.lighting.is_some() || e.desc.fog,
                        blend: blend_index(e.desc.blend),
                    },
                    id,
                    base: e.base?,
                    capacity: e.desc.capacity,
                    texture: e.desc.texture.map(|t| t.texture),
                };
                Some((depth, draw, EmitterUniform::new(&e.desc)))
            })
            .collect();
        // Back to front, additive and alpha-blended alike, as Bevy's transparent phase sorts
        // hanabi's effects (a stable sort: ties in id order).
        emitters.sort_by(|a, b| b.0.total_cmp(&a.0));
        if emitters.is_empty() {
            return;
        }
        let mut uniforms = vec![0u8; emitters.len() * EMITTER_STRIDE as usize];
        for (i, (_, _, uniform)) in emitters.iter().enumerate() {
            let at = i * EMITTER_STRIDE as usize;
            uniforms[at..at + std::mem::size_of::<EmitterUniform>()]
                .copy_from_slice(bytemuck::bytes_of(uniform));
        }
        let replaced = self.emitter_buffer.write(device, queue, &uniforms);
        self.draws = emitters.into_iter().map(|(_, d, _)| d).collect();
        queue.write_buffer(
            &self.view_buffer,
            0,
            bytemuck::bytes_of(&ViewUniform {
                clip_from_world,
                right: world_from_view.x_axis,
                up: world_from_view.y_axis,
                eye: world_from_view.w_axis,
                params: Vec4::new(near, 0.0, 0.0, 0.0),
            }),
        );
        if replaced {
            self.group = None;
            self.soft_group = None;
        }
        let entries = |soft: bool| {
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.view_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.emitter_buffer.buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<EmitterUniform>() as u64),
                    }),
                },
            ];
            if soft {
                entries.push(wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                });
            }
            entries
        };
        if self.soft() && self.soft_group.is_none() {
            self.soft_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("soft particle view"),
                layout: &pipelines.soft_layout,
                entries: &entries(true),
            }));
        }
        if self.group.is_none() {
            self.group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle view"),
                layout: &pipelines.layout,
                entries: &entries(false),
            }));
        }
    }

    /// Draws the prepared emitters' particles into the main pass (MSAA HDR colour and depth), or,
    /// with a soft emitter, into the particle pass (depth read-only). Shaded emitters bind the
    /// lit pass's `frame` and `lighting` groups.
    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        pipelines: &ParticlePipelines,
        world: &World,
        frame: &wgpu::BindGroup,
        lighting: Option<&wgpu::BindGroup>,
    ) {
        let (quads, args) = world.particle_pool().draw_buffers();
        // Emitters are drawn in depth order; consecutive ones often share a pipeline and a
        // texture, so only changes are recorded (#85).
        let mut pipeline = None;
        let mut texture = None;
        for (i, d) in self.draws.iter().enumerate() {
            let v = d.variant;
            // Shaded pipelines put the particle groups after the lit pass's two.
            let first = if v.shaded { 2 } else { 0 };
            if pipeline != Some(v) {
                pass.set_pipeline(&pipelines.pipelines[v.index()]);
                if v.shaded {
                    pass.set_bind_group(0, frame, &[]);
                    pass.set_bind_group(1, lighting, &[]);
                }
                if pipeline.is_none_or(|p: Variant| p.shaded != v.shaded) {
                    texture = None;
                }
                pipeline = Some(v);
            }
            let view = if v.soft {
                &self.soft_group
            } else {
                &self.group
            };
            pass.set_bind_group(first, view.as_ref(), &[(i as u64 * EMITTER_STRIDE) as u32]);
            if texture != Some(d.texture) {
                pass.set_bind_group(first + 1, world.particle_texture(d.texture), &[]);
                texture = Some(d.texture);
            }
            let start = u64::from(d.base) * QUAD_BYTES;
            pass.set_vertex_buffer(
                0,
                quads.slice(start..start + u64::from(d.capacity) * QUAD_BYTES),
            );
            pass.draw_indirect(args, u64::from(d.id) * COUNTER_BYTES);
        }
    }
}
