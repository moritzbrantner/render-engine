//! Draws a [`World`]'s particles in the main pass, after the blended instances: one instanced
//! quad per alive particle, read from the quads the last simulation step wrote
//! (`particle_sim`), one indirect draw per emitter. Emitters are drawn back to front by their
//! position, additive and alpha-blended alike, as Bevy's transparent phase sorts hanabi's
//! effects.

use crate::{
    particle_sim::{COUNTER_BYTES, QUAD_BYTES},
    particles::{Emitter, Orient, ParticleBlend, TextureMapping},
    renderer::{DEPTH_FORMAT, Growable, HDR_FORMAT, SAMPLE_COUNT},
    world::World,
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
}

/// Per drawn emitter, at a dynamic offset of `EMITTER_STRIDE` (`Emitter` in `particles.wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmitterUniform {
    roundness: f32,
    orient: u32,
    mapping: u32,
    flipbook_columns: u32,
    flipbook_scale: Vec2,
    _pad: Vec2,
}

impl EmitterUniform {
    fn new(e: &Emitter) -> Self {
        let flipbook = e.texture.and_then(|t| t.flipbook);
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
            _pad: Vec2::ZERO,
        }
    }
}

/// `min_uniform_buffer_offset_alignment`'s largest allowed value.
const EMITTER_STRIDE: u64 = 256;

/// Indexed by [`blend_index`].
pub(crate) struct ParticlePipelines {
    layout: wgpu::BindGroupLayout,
    texture_layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::RenderPipeline; 2],
}

fn blend_index(b: ParticleBlend) -> usize {
    match b {
        ParticleBlend::Add => 0,
        ParticleBlend::Alpha => 1,
    }
}

impl ParticlePipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/particles.wgsl"));
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle view"),
            entries: &[
                uniform(0, false, wgpu::ShaderStages::VERTEX),
                uniform(1, true, wgpu::ShaderStages::VERTEX_FRAGMENT),
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particles"),
            bind_group_layouts: &[Some(&layout), Some(&texture_layout)],
            immediate_size: 0,
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
        let pipeline = |blend: wgpu::BlendState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("particles"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
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
                    module: &shader,
                    entry_point: Some("fragment"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(blend),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
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
        Self {
            pipelines: [pipeline(add), pipeline(wgpu::BlendState::ALPHA_BLENDING)],
            layout,
            texture_layout,
        }
    }
}

/// One drawn emitter: its pipeline, its id (draw arguments), its slot range (quads) and its
/// texture.
struct EmitterDraw {
    blend: usize,
    id: u32,
    base: u32,
    capacity: u32,
    texture: wgpu::BindGroup,
}

/// One viewport's particle draws for the frame.
pub(crate) struct ParticleFrame {
    view_buffer: wgpu::Buffer,
    emitter_buffer: Growable,
    group: Option<wgpu::BindGroup>,
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
            draws: Vec::new(),
        }
    }

    /// Orders this frame's emitters of `world` as seen through `clip_from_world` from a camera
    /// whose world-space axes are the columns of `world_from_view`.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipelines: &ParticlePipelines,
        world: &World,
        clip_from_world: Mat4,
        world_from_view: Mat4,
    ) {
        self.draws.clear();
        let forward = -world_from_view.z_axis.truncate();
        let eye = world_from_view.w_axis.truncate();
        // (view depth, draw, uniform); emitters not stepped yet have nothing to draw.
        let mut emitters: Vec<(f32, EmitterDraw, EmitterUniform)> = world
            .emitter_ids()
            .filter_map(|(id, e)| {
                let depth = (e.transform.w_axis.truncate() - eye).dot(forward);
                let (view, sampler) = world.particle_texture(e.desc.texture.map(|t| t.texture));
                let draw = EmitterDraw {
                    blend: blend_index(e.desc.blend),
                    id,
                    base: e.base?,
                    capacity: e.desc.capacity,
                    texture: device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("particle texture"),
                        layout: &pipelines.texture_layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(view),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(sampler),
                            },
                        ],
                    }),
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
            }),
        );
        if self.group.is_none() || replaced {
            self.group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle view"),
                layout: &pipelines.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.view_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.emitter_buffer.buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(
                                std::mem::size_of::<EmitterUniform>() as u64
                            ),
                        }),
                    },
                ],
            }));
        }
    }

    /// Draws the prepared emitters' particles into the main pass (MSAA HDR colour and depth).
    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        pipelines: &ParticlePipelines,
        world: &World,
    ) {
        let (quads, args) = world.particle_pool().draw_buffers();
        for (i, d) in self.draws.iter().enumerate() {
            pass.set_pipeline(&pipelines.pipelines[d.blend]);
            pass.set_bind_group(
                0,
                self.group.as_ref(),
                &[(i as u64 * EMITTER_STRIDE) as u32],
            );
            pass.set_bind_group(1, &d.texture, &[]);
            let start = u64::from(d.base) * QUAD_BYTES;
            pass.set_vertex_buffer(
                0,
                quads.slice(start..start + u64::from(d.capacity) * QUAD_BYTES),
            );
            pass.draw_indirect(args, u64::from(d.id) * COUNTER_BYTES);
        }
    }
}
