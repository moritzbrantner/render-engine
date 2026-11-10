//! The opt-in GPU frustum culling and indirect path ([`Culling::GpuIndirect`], issue #71; see
//! `docs/gpu-driven-experiment.md`). Off by default; [`Culling::Cpu`] stays the reference.
//!
//! Each frame `prepare` writes a group table (mesh/material/face runs of opaque or masked
//! unskinned instances, with their box), a view table (each view's frustum planes and light
//! sphere) and one `DrawIndexedIndirect` per run and view into a storage/indirect buffer. A
//! single compute pass tests each instance against each view (a 2D dispatch: `x` the instance,
//! `y` the view), appends the survivors to the view's run in the visible-index buffer and bumps
//! the run's instance count. The render passes then issue one `draw_indexed_indirect` per run
//! and view, in the CPU path's order, empty runs included.
//!
//! The runs occupy the start of the visible-index buffer; blended and skinned instances are
//! culled on the CPU and their indices follow, so a frame can mix both.
//!
//! [`Culling::GpuIndirect`]: super::Culling::GpuIndirect
//! [`Culling::Cpu`]: super::Culling::Cpu

use super::{Growable, Renderer, encode::Stamps, viewport::Viewport};
use crate::cull::{Aabb, Frustum};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

/// Stride of one indirect draw in the indirect buffer (`DrawIndexedIndirect`).
pub(super) const INDIRECT_STRIDE: u64 = 20;
/// Instances per workgroup (the shader's `@workgroup_size(64)`).
const WORKGROUP: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GroupGpu {
    center: Vec4,
    half_extents: Vec4,
}

/// A group the compute pass culls: its table row plus what `prepare` needs for its draws.
pub(super) struct GroupRow {
    gpu: GroupGpu,
    index_count: u32,
    /// Instances in the run (filled while grouping).
    pub(super) count: u32,
}

impl GroupRow {
    pub(super) fn new(aabb: Aabb, index_count: u32) -> Self {
        Self {
            gpu: GroupGpu {
                center: aabb.center.extend(0.0),
                half_extents: aabb.half_extents.extend(0.0),
            },
            index_count,
            count: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ViewGpu {
    planes: [Vec4; 6],
    sphere: Vec4,
    plane_count: u32,
    has_sphere: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsGpu {
    group_count: u32,
    instance_count: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

/// The compute pass's pipeline and per-frame tables on one viewport.
pub(super) struct GpuCulling {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    groups: Growable,
    views: Growable,
    indirect: Growable,
    bind_group: Option<wgpu::BindGroup>,
    /// Workgroups: `(ceil(instances / 64), views)`.
    dispatch: (u32, u32),
    /// `(views, rows per view)` of the last prepare, for [`Renderer::read_gpu_culled`].
    shape: (u32, u32),
}

impl Renderer {
    /// Whether the GPU culling path is available on this device: `INDIRECT_FIRST_INSTANCE`
    /// (each run is drawn from its first index in the visible-index buffer) and five storage
    /// buffers in one compute stage.
    pub fn supports_gpu_culling(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
            && self.device.limits().max_storage_buffers_per_shader_stage >= 5
    }

    /// Per view (the camera first), how many instances the GPU pass kept in the frame last
    /// submitted for `viewport`; `None` on the CPU path. Reads back and waits: for tests and
    /// tools, not the frame loop.
    pub fn read_gpu_culled(&self, viewport: &Viewport) -> Option<Vec<u32>> {
        let gpu = viewport.gpu_culling.as_ref()?;
        let (views, rows) = gpu.shape;
        let size = u64::from(views * rows) * INDIRECT_STRIDE;
        if size == 0 {
            return Some(vec![0; views as usize]);
        }
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu culling readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&gpu.indirect.buffer, 0, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok()?.ok()?;
        let bytes = slice.get_mapped_range().expect("mapped").to_vec();
        staging.unmap();
        let args: &[IndirectArgs] = bytemuck::cast_slice(&bytes);
        Some(
            args.chunks(rows as usize)
                .map(|view| view.iter().map(|a| a.instance_count).sum())
                .collect(),
        )
    }
}

impl GpuCulling {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/gpu_cull.wgsl"));
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu culling"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                storage(5, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu culling"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gpu culling"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cull"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu culling params"),
            size: std::mem::size_of::<ParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            params,
            groups: Growable::new(device, "gpu culling groups", wgpu::BufferUsages::STORAGE),
            views: Growable::new(device, "gpu culling views", wgpu::BufferUsages::STORAGE),
            indirect: Growable::new(
                device,
                "gpu culling indirect args",
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_SRC,
            ),
            bind_group: None,
            dispatch: (0, 0),
            shape: (0, 0),
        }
    }

    pub(super) fn indirect_buffer(&self) -> &wgpu::Buffer {
        &self.indirect.buffer
    }

    pub(super) fn buffer_bytes(&self) -> u64 {
        self.params.size() + self.groups.size() + self.views.size() + self.indirect.size()
    }

    pub(super) fn invalidate_bind_group(&mut self) {
        self.bind_group = None;
    }

    /// Writes this frame's tables. Counted in `work` like the other uploads.
    pub(super) fn write_tables(
        viewport: &mut Viewport,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        rows: &[GroupRow],
        clip_from_world: Mat4,
    ) {
        let views: Vec<ViewGpu> =
            std::iter::once(view_row(Frustum::camera(clip_from_world).planes(), None))
                .chain(
                    viewport
                        .shadow_passes
                        .iter()
                        .map(|p| view_row(p.frustum.planes(), p.sphere)),
                )
                .collect();
        let view_count = views.len() as u32;
        let row_count = rows.len() as u32;
        let instance_count = viewport.instance_rows;
        let per_view: u32 = rows.iter().map(|r| r.count).sum();
        let mut args = Vec::with_capacity(views.len() * rows.len());
        for view in 0..view_count {
            let mut first_instance = view * per_view;
            for row in rows {
                args.push(IndirectArgs {
                    index_count: row.index_count,
                    instance_count: 0,
                    first_index: 0,
                    base_vertex: 0,
                    first_instance,
                });
                first_instance += row.count;
            }
        }
        if args.is_empty() {
            args.push(IndirectArgs::zeroed());
        }
        let group_rows: Vec<GroupGpu> = if rows.is_empty() {
            vec![GroupGpu::zeroed()]
        } else {
            rows.iter().map(|r| r.gpu).collect()
        };
        let gpu = viewport
            .gpu_culling
            .as_mut()
            .expect("write_tables only with the GPU path on");
        let work = &mut viewport.work;
        let mut upload = |buffer: &mut Growable, bytes: &[u8]| {
            work.upload_bytes += bytes.len() as u64;
            let grown = buffer.write(device, queue, bytes);
            work.allocations += u32::from(grown);
            grown
        };
        let grown = upload(&mut gpu.groups, bytemuck::cast_slice(&group_rows))
            | upload(&mut gpu.views, bytemuck::cast_slice(&views))
            | upload(&mut gpu.indirect, bytemuck::cast_slice(&args));
        if grown {
            gpu.bind_group = None;
        }
        let params = ParamsGpu {
            group_count: row_count,
            instance_count,
            _pad: [0; 2],
        };
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        work.upload_bytes += std::mem::size_of::<ParamsGpu>() as u64;
        gpu.dispatch = (instance_count.div_ceil(WORKGROUP), view_count);
        gpu.shape = (view_count, row_count);
    }

    /// Creates the bind group if a buffer it binds was replaced.
    pub(super) fn rebuild_bind_group(viewport: &mut Viewport, device: &wgpu::Device) {
        let instances = &viewport.instance_buffer.buffer;
        let visible = &viewport.visible_buffer.buffer;
        let gpu = viewport
            .gpu_culling
            .as_mut()
            .expect("rebuild_bind_group only with the GPU path on");
        if gpu.bind_group.is_some() {
            return;
        }
        gpu.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu culling"),
            layout: &gpu.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: gpu.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gpu.groups.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gpu.views.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: visible.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: gpu.indirect.buffer.as_entire_binding(),
                },
            ],
        }));
    }

    /// Records the culling pass.
    pub(super) fn encode(&self, encoder: &mut wgpu::CommandEncoder, stamps: &Stamps<'_>) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu culling"),
            timestamp_writes: stamps.compute(),
        });
        let (x, y) = self.dispatch;
        if x > 0 && y > 0 {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, self.bind_group.as_ref(), &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
    }
}

fn view_row(planes: &[Vec4], sphere: Option<(Vec3, f32)>) -> ViewGpu {
    let mut array = [Vec4::ZERO; 6];
    array[..planes.len()].copy_from_slice(planes);
    ViewGpu {
        planes: array,
        sphere: sphere.map_or(Vec4::ZERO, |(p, r)| p.extend(r)),
        plane_count: planes.len() as u32,
        has_sphere: u32::from(sphere.is_some()),
        _pad: [0; 2],
    }
}
