//! Draws a [`World`]'s particles in the main pass, after the blended instances: one instanced
//! quad per alive particle, additive emitters first, then the alpha-blended particles of all
//! emitters together, back to front.

use crate::{
    particles::ParticleBlend,
    renderer::{DEPTH_FORMAT, Growable, HDR_FORMAT, SAMPLE_COUNT},
    world::World,
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};
use std::ops::Range;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewUniform {
    clip_from_world: Mat4,
    right: Vec4,
    up: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParticleInstance {
    position_size: [f32; 4],
    color: [f32; 4],
    roundness: f32,
}

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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particle view"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particles"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let attributes = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32];
        let pipeline = |blend: wgpu::BlendState| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("particles"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vertex"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<ParticleInstance>() as u64,
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

/// One viewport's particle instances and draws for the frame.
pub(crate) struct ParticleFrame {
    view_buffer: wgpu::Buffer,
    instances: Growable,
    group: Option<wgpu::BindGroup>,
    /// `(blend_index, instance range)`.
    draws: Vec<(usize, Range<u32>)>,
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
            instances: Growable::new(device, "particle instances", wgpu::BufferUsages::VERTEX),
            group: None,
            draws: Vec::new(),
        }
    }

    /// Writes this frame's particles of `world` as seen through `clip_from_world` from a
    /// camera whose world-space axes are the columns of `world_from_view`.
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
        let mut additive = Vec::new();
        // (view depth, instance) of every alpha-blended particle.
        let mut blended = Vec::new();
        let forward = -world_from_view.z_axis.truncate();
        let eye = world_from_view.w_axis.truncate();
        for e in world.emitters() {
            for &slot in &e.alive {
                let p = &e.particles[slot as usize];
                let (size, color) = e.appearance(p);
                let instance = ParticleInstance {
                    position_size: p.position.extend(size).to_array(),
                    color: color.to_array(),
                    roundness: e.desc.roundness,
                };
                match e.desc.blend {
                    ParticleBlend::Add => additive.push(instance),
                    ParticleBlend::Alpha => {
                        blended.push(((p.position - eye).dot(forward), instance))
                    }
                }
            }
        }
        blended.sort_by(|a, b| b.0.total_cmp(&a.0));
        let n_add = additive.len() as u32;
        let mut all = additive;
        all.extend(blended.into_iter().map(|(_, i)| i));
        if all.is_empty() {
            return;
        }
        if n_add > 0 {
            self.draws.push((blend_index(ParticleBlend::Add), 0..n_add));
        }
        if all.len() as u32 > n_add {
            self.draws
                .push((blend_index(ParticleBlend::Alpha), n_add..all.len() as u32));
        }
        self.instances
            .write(device, queue, bytemuck::cast_slice(&all));
        queue.write_buffer(
            &self.view_buffer,
            0,
            bytemuck::bytes_of(&ViewUniform {
                clip_from_world,
                right: world_from_view.x_axis,
                up: world_from_view.y_axis,
            }),
        );
        if self.group.is_none() {
            self.group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle view"),
                layout: &pipelines.layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.view_buffer.as_entire_binding(),
                }],
            }));
        }
    }

    /// Draws the prepared particles into the main pass (MSAA HDR colour and depth).
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, pipelines: &ParticlePipelines) {
        if self.draws.is_empty() {
            return;
        }
        pass.set_bind_group(0, self.group.as_ref(), &[]);
        pass.set_vertex_buffer(0, self.instances.buffer.slice(..));
        for (blend, range) in &self.draws {
            pass.set_pipeline(&pipelines.pipelines[*blend]);
            pass.draw(0..6, range.clone());
        }
    }
}
