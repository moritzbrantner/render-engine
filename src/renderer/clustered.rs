//! The Forward+ light assignment ([`LightAssignment::Clustered`], issues #72 and #94; see
//! `docs/forward-plus-experiment.md`), the default wherever the device supports it. The
//! reference path, [`LightAssignment::PerFragment`], evaluates every point and spot light for
//! every fragment and stays available as the reference and the fallback.
//!
//! Each frame `prepare` assigns the lights to a grid of view-space clusters on the CPU (tiles over
//! the screen × exponential depth slices, Bevy's `assign_objects_to_clusters` shape: a sphere test
//! per cluster, and a cone test for spot lights), writing a table of `(offset, points, spots)` per
//! cluster and one list of light indices, each cluster's lights in the light buffer's order. The
//! lit pipelines are compiled from `lit.wgsl` with its light loop replaced by a loop over the
//! fragment's cluster (`clustered_lights.wgsl`), bound as a fourth bind group. There is no depth
//! prepass and no compute: a cluster is a volume, so blended and alpha-masked fragments look it
//! up exactly like opaque ones.
//!
//! The assignment is conservative (lights padded, clusters bounded by boxes over their frusta)
//! and a light outside a cluster contributes exactly zero to every fragment in it, so the output
//! is the reference path's.
//!
//! **Overflow** is deterministic and keeps the output: a cluster whose list would exceed
//! [`ClusterConfig::max_lights_per_cluster`] points at a run of every light (the first entries of
//! the index list), which is the reference path's per-fragment loop for that cluster alone. The
//! count is in [`PrepareWork::cluster_overflows`](super::PrepareWork).
//!
//! Shadow maps are unchanged: a shadow-casting light is a list entry like any other, and renders
//! its maps whether or not any cluster holds it.

use super::{
    Growable, Renderer,
    gpu::GpuLight,
    lights::LocalLights,
    pipelines,
    viewport::{PrepareWork, Viewport},
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, UVec4, Vec3, Vec4};

/// The grid and the overflow limit of [`LightAssignment::Clustered`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterConfig {
    /// Tiles across the screen.
    pub x: u32,
    /// Tiles down the screen.
    pub y: u32,
    /// Depth slices, the first covering `0..FIRST_SLICE_DEPTH` metres.
    pub z: u32,
    /// Longest list a cluster keeps; a longer one overflows to every light (no output change,
    /// the reference path's cost in that cluster). Bounds the list memory:
    /// `x · y · z · max_lights_per_cluster` indices.
    pub max_lights_per_cluster: u32,
    /// Compile the lists with counters of the shaded fragments and the lights they looped over
    /// ([`Renderer::read_cluster_diagnostic`]): measurement only, it perturbs what it measures.
    pub diagnostic: bool,
}

impl Default for ClusterConfig {
    /// Bevy's cluster grid at 16:9 (`ClusterConfig::default`: 24 depth slices), a limit that the
    /// bench's densest scene does not reach.
    fn default() -> Self {
        Self {
            x: 16,
            y: 9,
            z: 24,
            max_lights_per_cluster: 256,
            diagnostic: false,
        }
    }
}

/// How a viewport's fragments find their point and spot lights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightAssignment {
    /// Every light for every fragment: the reference path, and the fallback where the device
    /// lacks what [`LightAssignment::Clustered`] needs.
    PerFragment,
    /// The fragment's cluster's list: the default; see the module docs.
    Clustered(ClusterConfig),
}

impl Default for LightAssignment {
    /// Clustered on Bevy's grid ([`ClusterConfig::default`]).
    fn default() -> Self {
        Self::Clustered(ClusterConfig::default())
    }
}

/// [`Viewport::set_light_assignment`]: the device lacks what the clustered path needs and the
/// viewport stays on [`LightAssignment::PerFragment`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusteredLightingUnsupported;

impl std::fmt::Display for ClusteredLightingUnsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "clustered lighting needs 4 bind groups and 4 storage buffers per fragment stage"
        )
    }
}

impl std::error::Error for ClusteredLightingUnsupported {}

/// The far edge of depth slice 0 (m): Bevy's `first_slice_depth` default.
const FIRST_SLICE_DEPTH: f32 = 5.0;
/// `(offset, points, spots)` per cluster.
const CLUSTER_ROW: usize = 3;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ClusterParams {
    grid: UVec4,
    factors: Vec4,
}

/// A light as the assignment tests it, in view space (negative z in front of the camera).
#[derive(Clone, Copy)]
struct ViewLight {
    position: Vec3,
    range: f32,
    /// Spot axis in view space, sine and cosine of the outer angle.
    cone: Option<(Vec3, f32, f32)>,
}

/// The clustered path's per-viewport state.
pub(super) struct ClusteredLights {
    pub(super) config: ClusterConfig,
    pub(super) pipelines: Vec<wgpu::RenderPipeline>,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    table: Growable,
    indices: Growable,
    diagnostic: Option<wgpu::Buffer>,
    pub(super) group: Option<wgpu::BindGroup>,
    /// Scratch: `(cluster, light)` pairs of this frame's assignment, kept with its allocation.
    pairs: Vec<(u32, u32)>,
    rows: Vec<u32>,
    list: Vec<u32>,
}

impl Renderer {
    /// Whether [`LightAssignment::Clustered`] is available on this device: a fourth bind group
    /// and four storage buffers in the fragment stage (instances, lights, cluster table and
    /// list).
    pub fn supports_clustered_lighting(&self) -> bool {
        let limits = self.device.limits();
        limits.max_bind_groups >= 4 && limits.max_storage_buffers_per_shader_stage >= 4
    }

    /// With [`ClusterConfig::diagnostic`]: the summed `(shaded fragments, lights they looped
    /// over)` since the last [`Renderer::prepare`]. Reads back and waits: for tools only.
    pub fn read_cluster_diagnostic(&self, viewport: &Viewport) -> Option<(u64, u64)> {
        let buffer = viewport.clustered.as_ref()?.diagnostic.as_ref()?;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster diagnostic readback"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, 8);
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
        let counts: &[u32] = bytemuck::cast_slice(&bytes);
        Some((u64::from(counts[0]), u64::from(counts[1])))
    }
}

impl ClusteredLights {
    pub(super) fn new(
        renderer: &Renderer,
        config: ClusterConfig,
    ) -> Result<Self, ClusteredLightingUnsupported> {
        if !renderer.supports_clustered_lighting()
            || config.x == 0
            || config.y == 0
            || config.z == 0
        {
            return Err(ClusteredLightingUnsupported);
        }
        let device = &renderer.device;
        let ClusteredPipelines { layout, pipelines } = if config.diagnostic {
            ClusteredPipelines::new(renderer, true)
        } else {
            renderer
                .clustered_pipelines
                .get_or_init(|| ClusteredPipelines::new(renderer, false))
                .clone()
        };
        let diagnostic = config.diagnostic.then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cluster diagnostic"),
                size: 8,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        Ok(Self {
            config,
            pipelines,
            layout,
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cluster params"),
                size: std::mem::size_of::<ClusterParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            table: Growable::new(device, "cluster table", wgpu::BufferUsages::STORAGE),
            indices: Growable::new(device, "cluster light indices", wgpu::BufferUsages::STORAGE),
            diagnostic,
            group: None,
            pairs: Vec::new(),
            rows: Vec::new(),
            list: Vec::new(),
        })
    }
}

/// The clustered path's bind group layout and lit pipelines. The default (non-diagnostic) set is
/// compiled once per [`Renderer`], by the first viewport that needs it, and shared.
#[derive(Clone)]
pub(super) struct ClusteredPipelines {
    layout: wgpu::BindGroupLayout,
    pipelines: Vec<wgpu::RenderPipeline>,
}

impl ClusteredPipelines {
    fn new(renderer: &Renderer, diagnostic: bool) -> Self {
        let device = &renderer.device;
        let storage =
            |binding, read_only, ty_size: Option<wgpu::BufferSize>| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: ty_size,
                },
                count: None,
            };
        let mut entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage(1, true, None),
            storage(2, true, None),
        ];
        if diagnostic {
            entries.push(storage(3, false, None));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("clusters"),
            entries: &entries,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lit clustered"),
            source: wgpu::ShaderSource::Wgsl(
                clustered_shader(
                    diagnostic,
                    device
                        .features()
                        .contains(wgpu::Features::SHADER_EARLY_DEPTH_TEST),
                )
                .into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lit clustered"),
            bind_group_layouts: &[
                Some(&renderer.frame_layout),
                Some(&renderer.lighting_layout),
                Some(&renderer.material_layout),
                Some(&layout),
            ],
            immediate_size: 0,
        });
        let pipelines = pipelines::lit_pipelines(device, &pipeline_layout, &shader);
        Self { layout, pipelines }
    }
}

impl ClusteredLights {
    pub(super) fn memory(&self) -> u64 {
        self.params.size() + self.table.size() + self.indices.size()
    }

    /// Bytes of CPU scratch retained between frames.
    pub(super) fn cpu_memory(&self) -> u64 {
        (self.pairs.capacity() * std::mem::size_of::<(u32, u32)>()
            + (self.rows.capacity() + self.list.capacity()) * std::mem::size_of::<u32>())
            as u64
    }

    /// Assigns this frame's lights and writes the table, the list and (when a buffer was
    /// replaced) the bind group.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lights: &LocalLights,
        view_from_world: Mat4,
        (width, height): (u32, u32),
        fov_y: f32,
        work: &mut PrepareWork,
    ) {
        let view_lights: Vec<ViewLight> = view_lights(lights, view_from_world);
        let grid = self.config;
        let aspect = width as f32 / height as f32;
        let tan_y = (fov_y * 0.5).tan();
        let tan_x = tan_y * aspect;
        // Bevy's `MaxClusterableObjectRange`: the slices end at the furthest light's reach.
        let far = view_lights
            .iter()
            .map(|l| -l.position.z + l.range)
            .fold(FIRST_SLICE_DEPTH * 1.001, f32::max);
        let log_factor = if grid.z > 1 {
            (grid.z - 1) as f32 / (far / FIRST_SLICE_DEPTH).ln()
        } else {
            0.0
        };
        let slice_of = |depth: f32| -> u32 {
            if depth <= FIRST_SLICE_DEPTH || grid.z == 1 {
                0
            } else {
                (1 + (log_factor * (depth / FIRST_SLICE_DEPTH).ln()) as u32).min(grid.z - 1)
            }
        };
        // Slice `k ≥ 1` covers `edges[k - 1]..edges[k]`; slice 0 starts at the eye.
        let edges: Vec<f32> = (0..grid.z)
            .map(|s| {
                if grid.z > 1 {
                    FIRST_SLICE_DEPTH
                        * ((far / FIRST_SLICE_DEPTH).ln() * s as f32 / (grid.z - 1) as f32).exp()
                } else {
                    far
                }
            })
            .collect();
        let slice_bounds = |slice: u32| -> (f32, f32) {
            if slice == 0 {
                (0.0, edges[0])
            } else {
                (edges[slice as usize - 1], edges[slice as usize])
            }
        };
        // Conservative clamp from a sphere's extent over a slice to a tile range. The tile's
        // extent at depth `d` is `(ndc · d · tan)`, so the NDC bound of an offset `s` is
        // largest where `d` is smallest in magnitude when `s` has the matching sign.
        let tile_range = |lo: f32, hi: f32, near: f32, far_d: f32, tan: f32, tiles: u32| {
            let ndc = |s: f32, d: f32| s / (d.max(1e-4) * tan);
            let ndc_lo = if lo < 0.0 {
                ndc(lo, near)
            } else {
                ndc(lo, far_d)
            };
            let ndc_hi = if hi > 0.0 {
                ndc(hi, near)
            } else {
                ndc(hi, far_d)
            };
            let to_tile = |n: f32| ((n.clamp(-1.0, 1.0) + 1.0) * 0.5 * tiles as f32) as i64;
            let a = to_tile(ndc_lo).clamp(0, i64::from(tiles) - 1) as u32;
            let b = to_tile(ndc_hi).clamp(0, i64::from(tiles) - 1) as u32;
            (a, b)
        };

        self.pairs.clear();
        let mut tests = 0u64;
        for (index, light) in view_lights.iter().enumerate() {
            let depth = -light.position.z;
            let r = light.range * 1.001 + 0.01;
            if depth + r < 0.0 {
                continue;
            }
            let s0 = slice_of((depth - r).max(0.0));
            let s1 = slice_of(depth + r);
            for slice in s0..=s1 {
                let (d0, d1) = slice_bounds(slice);
                let (cx, cy) = (light.position.x, light.position.y);
                let (tx0, tx1) = tile_range(cx - r, cx + r, d0, d1, tan_x, grid.x);
                // Tile rows run down from the top of the screen.
                let (ty_lo, ty_hi) = tile_range(cy - r, cy + r, d0, d1, tan_y, grid.y);
                let (ty0, ty1) = (grid.y - 1 - ty_hi, grid.y - 1 - ty_lo);
                for ty in ty0..=ty1 {
                    for tx in tx0..=tx1 {
                        tests += 1;
                        let (min, max) = cluster_aabb(tx, ty, d0, d1, grid, tan_x, tan_y);
                        if !sphere_hits_aabb(light.position, r, min, max) {
                            continue;
                        }
                        if let Some((axis, sin, cos)) = light.cone
                            && !cone_hits_aabb(light.position, axis, sin, cos, r, min, max)
                        {
                            continue;
                        }
                        let cluster = (slice * grid.y + ty) * grid.x + tx;
                        self.pairs.push((cluster, index as u32));
                    }
                }
            }
        }

        let cluster_count = (grid.x * grid.y * grid.z) as usize;
        // The all-lights run comes first; an overflowing cluster points at it.
        let all = lights.point_count + lights.spot_count;
        self.rows.clear();
        self.rows.resize(cluster_count * CLUSTER_ROW, 0);
        self.list.clear();
        self.list.extend(0..all);
        // `pairs` is in light order for each cluster (the outer loop): a stable sort by cluster
        // keeps every cluster's lights in the buffer's order.
        let mut counts = vec![0u32; cluster_count];
        for &(cluster, _) in &self.pairs {
            counts[cluster as usize] += 1;
        }
        let cap = grid.max_lights_per_cluster;
        let mut overflows = 0u32;
        let mut fill: Vec<u32> = vec![0; cluster_count];
        for (cluster, &count) in counts.iter().enumerate() {
            let row = cluster * CLUSTER_ROW;
            if count > cap {
                overflows += 1;
                self.rows[row + 1] = lights.point_count;
                self.rows[row + 2] = lights.spot_count;
            } else {
                self.rows[row] = self.list.len() as u32;
                fill[cluster] = self.list.len() as u32;
                self.list.extend(std::iter::repeat_n(0, count as usize));
            }
        }
        for &(cluster, light) in &self.pairs {
            let c = cluster as usize;
            if counts[c] > cap {
                continue;
            }
            self.list[fill[c] as usize] = light;
            fill[c] += 1;
            let is_point = light < lights.point_count;
            self.rows[c * CLUSTER_ROW + if is_point { 1 } else { 2 }] += 1;
        }

        work.light_cluster_tests = tests;
        work.light_cluster_entries = self.pairs.len() as u64;
        work.cluster_overflows = overflows;

        let params = ClusterParams {
            grid: UVec4::new(grid.x, grid.y, grid.z, 0),
            factors: Vec4::new(
                grid.x as f32 / width as f32,
                grid.y as f32 / height as f32,
                FIRST_SLICE_DEPTH,
                log_factor,
            ),
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        // Never empty: storage bindings need at least one element.
        let mut written = |buffer: &mut Growable, data: &[u32]| -> bool {
            let empty = [0u32];
            let bytes: &[u8] = bytemuck::cast_slice(if data.is_empty() { &empty } else { data });
            work.upload_bytes += bytes.len() as u64;
            let grown = buffer.write(device, queue, bytes);
            work.allocations += u32::from(grown);
            grown
        };
        let rows = std::mem::take(&mut self.rows);
        let list = std::mem::take(&mut self.list);
        let table_grown = written(&mut self.table, &rows);
        let indices_grown = written(&mut self.indices, &list);
        self.rows = rows;
        self.list = list;
        if let Some(diag) = &self.diagnostic {
            queue.write_buffer(diag, 0, &[0u8; 8]);
        }
        if table_grown || indices_grown || self.group.is_none() {
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.table.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.indices.buffer.as_entire_binding(),
                },
            ];
            if let Some(diag) = &self.diagnostic {
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: diag.as_entire_binding(),
                });
            }
            self.group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("clusters"),
                layout: &self.layout,
                entries: &entries,
            }));
        }
    }
}

/// The view-space lights the assignment tests, from what the shader reads, so both agree.
fn view_lights(lights: &LocalLights, view_from_world: Mat4) -> Vec<ViewLight> {
    let rotation = Mat3::from_mat4(view_from_world);
    (0..(lights.point_count + lights.spot_count) as usize)
        .map(|i| {
            let light: &GpuLight = &lights.gpu[i];
            let position = view_from_world.transform_point3(light.position_radius.truncate());
            let range = 1.0 / light.color_inverse_square_range.w.sqrt();
            let cone = (i as u32 >= lights.point_count).then(|| {
                let x = light.light_custom_data.x;
                let z = light.light_custom_data.y;
                let mut y = (1.0 - x * x - z * z).max(0.0).sqrt();
                if light.flags & super::lights::LIGHT_SPOT_Y_NEGATIVE != 0 {
                    y = -y;
                }
                let cos = 1.0 / (1.0 + light.spot_light_tan_angle.powi(2)).sqrt();
                let sin = light.spot_light_tan_angle * cos;
                (rotation * Vec3::new(x, y, z), sin, cos)
            });
            ViewLight {
                position,
                range,
                cone,
            }
        })
        .collect()
}

/// A cluster's view-space box: the frustum of its tile between its depths.
fn cluster_aabb(
    tx: u32,
    ty: u32,
    d0: f32,
    d1: f32,
    grid: ClusterConfig,
    tan_x: f32,
    tan_y: f32,
) -> (Vec3, Vec3) {
    let ndc_x = |t: u32| -1.0 + 2.0 * t as f32 / grid.x as f32;
    let ndc_y = |t: u32| 1.0 - 2.0 * t as f32 / grid.y as f32;
    // A frustum edge spreads with depth, so each bound is at the depth that widens it.
    let widest = |lo: f32, hi: f32| {
        (
            if lo < 0.0 { lo * d1 } else { lo * d0 },
            if hi > 0.0 { hi * d1 } else { hi * d0 },
        )
    };
    let (x0, x1) = widest(ndc_x(tx) * tan_x, ndc_x(tx + 1) * tan_x);
    let (y0, y1) = widest(ndc_y(ty + 1) * tan_y, ndc_y(ty) * tan_y);
    (Vec3::new(x0, y0, -d1), Vec3::new(x1, y1, -d0))
}

fn sphere_hits_aabb(center: Vec3, radius: f32, min: Vec3, max: Vec3) -> bool {
    let closest = center.clamp(min, max);
    (center - closest).length_squared() <= radius * radius
}

/// Bevy's spot-light cluster test (`cluster_add_light_cull`): the cluster's bounding sphere
/// against the cone, with the sphere's front and back culled along the axis.
fn cone_hits_aabb(
    apex: Vec3,
    axis: Vec3,
    sin: f32,
    cos: f32,
    range: f32,
    min: Vec3,
    max: Vec3,
) -> bool {
    let center = (min + max) * 0.5;
    let sphere_radius = (max - min).length() * 0.5;
    let v = center - apex;
    let v1 = v.dot(axis);
    let closest = cos * (v.length_squared() - v1 * v1).max(0.0).sqrt() - v1 * sin;
    let angle_cull = closest > sphere_radius;
    let front_cull = v1 > sphere_radius + range;
    let back_cull = v1 < -sphere_radius;
    !(angle_cull || front_cull || back_cull)
}

/// `lit.wgsl` with its light loop replaced by the clustered lookup. With `diagnostic`, the
/// lookup also counts the shaded fragments and the lights they loop over (a read-write storage
/// buffer; with `early_depth` (`SHADER_EARLY_DEPTH_TEST`) `lit_fs` forces its early depth test
/// so opaque overdraw is the reference's).
fn clustered_shader(diagnostic: bool, early_depth: bool) -> String {
    let source = include_str!("../shaders/lit.wgsl");
    let begin = source
        .find("// local-light-loop-begin")
        .expect("lit.wgsl has the light loop's begin marker");
    let end = source
        .find("// local-light-loop-end")
        .expect("lit.wgsl has the light loop's end marker");
    let mut out = String::with_capacity(source.len() + 2048);
    out.push_str(&source[..begin]);
    out.push_str("direct += clustered_local_lights(in, world_normal, surface);\n    ");
    out.push_str(&source[end..]);
    out.push_str(include_str!("../shaders/clustered_lights.wgsl"));
    if diagnostic {
        if early_depth {
            out = out.replace(
                "@fragment\nfn lit_fs(",
                "@early_depth_test(force) @fragment\nfn lit_fs(",
            );
        }
        out = out
            .replace(
                "    // cluster-diagnostic-hook\n",
                "    atomicAdd(&cluster_diagnostic.fragments, 1u);\n    atomicAdd(&cluster_diagnostic.lights, spot_end - offset);\n",
            );
        out.push_str(
            "struct ClusterDiagnostic { fragments: atomic<u32>, lights: atomic<u32> }\n\
             @group(3) @binding(3) var<storage, read_write> cluster_diagnostic: ClusterDiagnostic;\n",
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_loop_markers_are_replaced() {
        assert!(
            !clustered_shader(false, true).contains("for (var i = 0u; i < point_count; i++)"),
            "the per-fragment loop is gone"
        );
        assert!(clustered_shader(false, true).contains("clustered_local_lights(in"));
        assert!(clustered_shader(true, true).contains("atomicAdd"));
        assert!(!clustered_shader(false, true).contains("atomicAdd"));
        assert!(!clustered_shader(true, false).contains("early_depth_test"));
    }

    #[test]
    fn cone_test_keeps_the_axis_and_drops_behind() {
        let (sin, cos) = (30f32.to_radians().sin(), 30f32.to_radians().cos());
        let axis = Vec3::NEG_Z;
        // A small cluster on the axis, in front.
        let (min, max) = (Vec3::new(-0.5, -0.5, -6.0), Vec3::new(0.5, 0.5, -5.0));
        assert!(cone_hits_aabb(Vec3::ZERO, axis, sin, cos, 20.0, min, max));
        // Behind the apex.
        let (min, max) = (Vec3::new(-0.5, -0.5, 5.0), Vec3::new(0.5, 0.5, 6.0));
        assert!(!cone_hits_aabb(Vec3::ZERO, axis, sin, cos, 20.0, min, max));
        // Off to the side, far outside the cone.
        let (min, max) = (Vec3::new(10.0, -0.5, -6.0), Vec3::new(11.0, 0.5, -5.0));
        assert!(!cone_hits_aabb(Vec3::ZERO, axis, sin, cos, 20.0, min, max));
    }
}
