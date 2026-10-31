//! Draws a [`World`]'s particles in the main pass, after the blended instances: one instanced
//! quad per alive particle, read from the quads the last simulation step wrote
//! (`particle_sim`), one indirect draw per emitter. Additive emitters come first, then the
//! alpha-blended ones back to front by their position, as Bevy's transparent phase sorts
//! hanabi's effects.

use crate::{
    particle_sim::{COUNTER_BYTES, QUAD_BYTES},
    particles::ParticleBlend,
    renderer::{DEPTH_FORMAT, Growable, HDR_FORMAT, SAMPLE_COUNT},
    world::World,
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewUniform {
    clip_from_world: Mat4,
    right: Vec4,
    up: Vec4,
}

/// Per drawn emitter, at a dynamic offset of `EMITTER_STRIDE`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EmitterUniform {
    roundness: f32,
    _pad: [f32; 3],
}

/// `min_uniform_buffer_offset_alignment`'s largest allowed value.
const EMITTER_STRIDE: u64 = 256;

/// Indexed by [`blend_index`].
pub(crate) struct ParticlePipelines {
    layout: wgpu::BindGroupLayout,
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
                uniform(1, true, wgpu::ShaderStages::FRAGMENT),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particles"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4];
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
        }
    }
}

/// One drawn emitter: its pipeline, its id (draw arguments) and its slot range (quads).
struct EmitterDraw {
    blend: usize,
    id: u32,
    base: u32,
    capacity: u32,
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
        // (view depth, draw, roundness); emitters not stepped yet have nothing to draw.
        let mut emitters: Vec<(f32, EmitterDraw, f32)> = world
            .emitter_ids()
            .filter_map(|(id, e)| {
                let depth = (e.transform.w_axis.truncate() - eye).dot(forward);
                let draw = EmitterDraw {
                    blend: blend_index(e.desc.blend),
                    id,
                    base: e.base?,
                    capacity: e.desc.capacity,
                };
                Some((depth, draw, e.desc.roundness))
            })
            .collect();
        // Additive first (in id order), then alpha-blended back to front.
        let alpha = |d: &EmitterDraw| d.blend == blend_index(ParticleBlend::Alpha);
        emitters.sort_by(|a, b| {
            alpha(&a.1).cmp(&alpha(&b.1)).then_with(|| {
                if alpha(&a.1) {
                    b.0.total_cmp(&a.0)
                } else {
                    std::cmp::Ordering::Equal
                }
            })
        });
        if emitters.is_empty() {
            return;
        }
        let mut uniforms = vec![0u8; emitters.len() * EMITTER_STRIDE as usize];
        for (i, (_, _, roundness)) in emitters.iter().enumerate() {
            let at = i * EMITTER_STRIDE as usize;
            uniforms[at..at + std::mem::size_of::<EmitterUniform>()].copy_from_slice(
                bytemuck::bytes_of(&EmitterUniform {
                    roundness: *roundness,
                    _pad: [0.0; 3],
                }),
            );
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
            let start = u64::from(d.base) * QUAD_BYTES;
            pass.set_vertex_buffer(
                0,
                quads.slice(start..start + u64::from(d.capacity) * QUAD_BYTES),
            );
            pass.draw_indirect(args, u64::from(d.id) * COUNTER_BYTES);
        }
    }
}
