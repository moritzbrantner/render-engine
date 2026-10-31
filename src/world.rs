//! The consumer's GPU-resident content: meshes, textures and materials uploaded once and kept
//! under stable handles until removed. Each frame the consumer lists [`Instance`]s (a mesh, a
//! material and a transform) and the renderer draws them; nothing else is uploaded again.
//! Particle emitters live here too: the consumer advances them with [`World::step_particles`]
//! and the renderer draws every alive particle.

use crate::{
    cull::Aabb,
    material::{AlphaMode, StandardMaterial, TextureData},
    mesh::MeshData,
    particle_sim::ParticlePool,
    particles::{Emitter, EmitterState},
    renderer::{FrameTimestamps, Renderer},
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

/// A mesh in a [`World`]; valid until [`World::remove_mesh`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MeshId(u32);

/// A texture in a [`World`]; valid until [`World::remove_texture`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TextureId(u32);

/// A material in a [`World`]; valid until [`World::remove_material`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MaterialId(u32);

/// A particle emitter in a [`World`]; valid until [`World::remove_emitter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EmitterId(u32);

/// A joint palette in a [`World`]; valid until [`World::remove_skin`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SkinId(u32);

/// One drawn copy of a mesh this frame.
#[derive(Clone, Copy, Debug)]
pub struct Instance {
    pub mesh: MeshId,
    pub material: MaterialId,
    /// Model matrix. A negative determinant (mirroring) flips the winding, which the renderer
    /// accounts for.
    pub world_from_local: Mat4,
    /// The joint palette a skinned mesh (one with [`MeshData::joints`]) is posed with; `None`
    /// for every other mesh. Instances may share a skin.
    pub skin: Option<SkinId>,
}

/// The vertex streams as the shaders read them: positions (location 0) and normals (1) on
/// their own, so a new pose rewrites just those, then [`SurfaceVertex`] and, for skinned
/// meshes, [`SkinVertex`].
pub(crate) mod streams {
    pub const POSITIONS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];
    pub const NORMALS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![1 => Float32x3];
    pub const SURFACE: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![2 => Float32x2, 3 => Float32x4, 4 => Float32x4];
    pub const SKIN: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![5 => Uint16x4, 6 => Float32x4];
}

/// The attributes a pose leaves alone: uv, tangent and colour.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SurfaceVertex {
    uv: [f32; 2],
    tangent: [f32; 4],
    color: [f32; 4],
}

/// `JOINTS_0` and `WEIGHTS_0`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SkinVertex {
    joints: [u16; 4],
    weights: [f32; 4],
}

/// `MaterialUniform::flags` (mirrored in `lit.wgsl`).
pub(crate) mod flags {
    pub const BASE_COLOR_TEXTURE: u32 = 1;
    pub const EMISSIVE_TEXTURE: u32 = 2;
    pub const METALLIC_ROUGHNESS_TEXTURE: u32 = 4;
    pub const OCCLUSION_TEXTURE: u32 = 8;
    pub const NORMAL_MAP: u32 = 16;
    pub const DOUBLE_SIDED: u32 = 32;
    pub const UNLIT: u32 = 64;
    pub const ALPHA_MASK: u32 = 128;
    pub const ALPHA_BLEND: u32 = 256;
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialUniform {
    base_color: Vec4,
    emissive: Vec4,
    metallic: f32,
    perceptual_roughness: f32,
    reflectance: f32,
    /// Mask: the material's cutoff. Blend: the shadow-pass cutoff (Bevy
    /// `PREMULTIPLIED_ALPHA_CUTOFF`).
    alpha_cutoff: f32,
    flags: u32,
    _pad: [u32; 3],
}

pub(crate) struct GpuMesh {
    pub positions: wgpu::Buffer,
    pub normals: wgpu::Buffer,
    pub surface: wgpu::Buffer,
    /// Skinned meshes only.
    pub skin: Option<wgpu::Buffer>,
    pub indices: wgpu::Buffer,
    pub vertex_count: u32,
    pub index_count: u32,
    pub has_tangents: bool,
    /// Local bounds for frustum culling, computed per upload.
    pub aabb: Aabb,
    /// Skinned meshes: the bind-pose box of the vertices each joint moves (Bevy
    /// `SkinnedMeshBounds`), by joint index.
    pub joint_bounds: Vec<(u16, Aabb)>,
}

struct GpuTexture {
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
}

pub(crate) struct GpuMaterial {
    pub material: StandardMaterial,
    pub group: wgpu::BindGroup,
}

/// Slot storage with stable indices; removed slots are reused.
struct Slots<T> {
    items: Vec<Option<T>>,
    free: Vec<u32>,
}

impl<T> Slots<T> {
    fn new() -> Self {
        Self {
            items: Vec::new(),
            free: Vec::new(),
        }
    }

    fn insert(&mut self, item: T) -> u32 {
        if let Some(i) = self.free.pop() {
            self.items[i as usize] = Some(item);
            i
        } else {
            self.items.push(Some(item));
            (self.items.len() - 1) as u32
        }
    }

    fn remove(&mut self, i: u32) -> Option<T> {
        let item = self.items.get_mut(i as usize)?.take();
        if item.is_some() {
            self.free.push(i);
        }
        item
    }

    fn get(&self, i: u32) -> Option<&T> {
        self.items.get(i as usize)?.as_ref()
    }
}

/// Meshes, textures and materials on one device. Handles stay valid until removed; removing a
/// texture that a material still uses keeps it alive for that material.
pub struct World {
    device: wgpu::Device,
    queue: wgpu::Queue,
    material_layout: wgpu::BindGroupLayout,
    /// Bound for every texture slot a material leaves empty (the shader skips it).
    white: GpuTexture,
    meshes: Slots<GpuMesh>,
    textures: Slots<GpuTexture>,
    materials: Slots<GpuMaterial>,
    skins: Slots<Vec<Mat4>>,
    emitters: Slots<EmitterState>,
    particles: ParticlePool,
}

impl World {
    /// An empty world on the renderer's device.
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();
        let white = upload_texture(
            &device,
            &queue,
            &TextureData {
                width: 1,
                height: 1,
                format: wgpu::TextureFormat::Rgba8Unorm,
                levels: vec![vec![255; 4]],
                sampler: Default::default(),
            },
        );
        Self {
            material_layout: renderer.material_layout().clone(),
            device,
            queue,
            white,
            meshes: Slots::new(),
            textures: Slots::new(),
            materials: Slots::new(),
            skins: Slots::new(),
            emitters: Slots::new(),
            particles: ParticlePool::new(renderer.device(), renderer.sim_pipelines()),
        }
    }

    /// Uploads a mesh. Missing uvs and tangents read as zero, missing colours as white.
    pub fn add_mesh(&mut self, mesh: &MeshData) -> MeshId {
        let gpu = self.gpu_mesh(mesh);
        MeshId(self.meshes.insert(gpu))
    }

    /// Replaces a mesh's data under the same handle, rewriting its buffers in place when the
    /// vertex and index counts and the skinning are unchanged. For a new pose of the same
    /// topology, [`World::set_mesh_positions_normals`] (CPU skinning) or a skinned mesh and
    /// [`World::set_skin`] (GPU skinning) upload far less.
    pub fn set_mesh(&mut self, id: MeshId, mesh: &MeshData) {
        let streams = Streams::new(mesh);
        let old = self.meshes.get(id.0).expect("set_mesh on a removed mesh");
        let in_place = old.vertex_count as usize == mesh.positions.len()
            && old.index_count as usize == mesh.indices.len()
            && old.skin.is_some() == streams.skin.is_some();
        if !in_place {
            let gpu = self.gpu_mesh(mesh);
            self.meshes.items[id.0 as usize] = Some(gpu);
            return;
        }
        let slot = self.meshes.items[id.0 as usize]
            .as_mut()
            .expect("checked above");
        let q = &self.queue;
        q.write_buffer(&slot.positions, 0, bytemuck::cast_slice(&mesh.positions));
        q.write_buffer(&slot.normals, 0, bytemuck::cast_slice(&mesh.normals));
        q.write_buffer(&slot.surface, 0, bytemuck::cast_slice(&streams.surface));
        if let (Some(buffer), Some(skin)) = (&slot.skin, &streams.skin) {
            q.write_buffer(buffer, 0, bytemuck::cast_slice(skin));
        }
        q.write_buffer(&slot.indices, 0, bytemuck::cast_slice(&mesh.indices));
        slot.has_tangents = !mesh.tangents.is_empty();
        slot.aabb = Aabb::from_points(&mesh.positions);
        slot.joint_bounds = joint_bounds(mesh);
    }

    /// Rewrites only a mesh's positions and normals, one per vertex, keeping its other
    /// attributes and its indices: a CPU-skinned or morphed pose of an unchanged topology. Two
    /// buffer writes of 12 bytes per vertex each, no re-interleaving. Skinned meshes take their
    /// poses through [`World::set_skin`] instead (their per-joint bounds stay those of the
    /// first upload).
    pub fn set_mesh_positions_normals(&mut self, id: MeshId, positions: &[Vec3], normals: &[Vec3]) {
        let slot = self.meshes.items[id.0 as usize]
            .as_mut()
            .expect("set_mesh_positions_normals on a removed mesh");
        let n = slot.vertex_count as usize;
        assert!(
            positions.len() == n && normals.len() == n,
            "{} positions and {} normals for a mesh of {n} vertices",
            positions.len(),
            normals.len()
        );
        self.queue
            .write_buffer(&slot.positions, 0, bytemuck::cast_slice(positions));
        self.queue
            .write_buffer(&slot.normals, 0, bytemuck::cast_slice(normals));
        slot.aabb = Aabb::from_points(positions);
    }

    fn gpu_mesh(&self, mesh: &MeshData) -> GpuMesh {
        let streams = Streams::new(mesh);
        let buffer = |label, contents: &[u8], usage| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: usage | wgpu::BufferUsages::COPY_DST,
                })
        };
        let vertex = wgpu::BufferUsages::VERTEX;
        GpuMesh {
            positions: buffer(
                "mesh positions",
                bytemuck::cast_slice(&mesh.positions),
                vertex,
            ),
            normals: buffer("mesh normals", bytemuck::cast_slice(&mesh.normals), vertex),
            surface: buffer(
                "mesh surface",
                bytemuck::cast_slice(&streams.surface),
                vertex,
            ),
            skin: streams
                .skin
                .as_ref()
                .map(|s| buffer("mesh skin", bytemuck::cast_slice(s), vertex)),
            indices: buffer(
                "mesh indices",
                bytemuck::cast_slice(&mesh.indices),
                wgpu::BufferUsages::INDEX,
            ),
            vertex_count: mesh.positions.len() as u32,
            index_count: mesh.indices.len() as u32,
            has_tangents: !mesh.tangents.is_empty(),
            aabb: Aabb::from_points(&mesh.positions),
            joint_bounds: joint_bounds(mesh),
        }
    }

    pub fn remove_mesh(&mut self, mesh: MeshId) {
        self.meshes.remove(mesh.0);
    }

    /// A joint palette for skinned meshes: per joint, the mesh-local matrix that moves a
    /// bind-pose vertex with that joint, `local_from_joint * joint_from_bind` (glTF: the joint's
    /// transform relative to the skinned mesh times its inverse bind matrix). The instance's
    /// `world_from_local` then places the skinned mesh, so a palette is independent of where
    /// the figure stands. Bevy's equivalent is `SkinnedMesh` with joint entities.
    pub fn add_skin(&mut self, joints: &[Mat4]) -> SkinId {
        SkinId(self.skins.insert(joints.to_vec()))
    }

    /// A skin's next pose: the whole palette, uploaded by the next [`Renderer::prepare`] that
    /// draws it.
    pub fn set_skin(&mut self, id: SkinId, joints: &[Mat4]) {
        let palette = self.skins.items[id.0 as usize]
            .as_mut()
            .expect("set_skin on a removed skin");
        palette.clear();
        palette.extend_from_slice(joints);
    }

    pub fn remove_skin(&mut self, id: SkinId) {
        self.skins.remove(id.0);
    }

    pub(crate) fn skin(&self, id: SkinId) -> &[Mat4] {
        self.skins.get(id.0).expect("instance of a removed skin")
    }

    /// Uploads a texture with all its mip levels.
    pub fn add_texture(&mut self, texture: &TextureData) -> TextureId {
        TextureId(
            self.textures
                .insert(upload_texture(&self.device, &self.queue, texture)),
        )
    }
    pub fn remove_texture(&mut self, texture: TextureId) {
        self.textures.remove(texture.0);
    }

    /// Adds a material; its textures must be in this world.
    pub fn add_material(&mut self, material: &StandardMaterial) -> MaterialId {
        let gpu = self.gpu_material(material);
        MaterialId(self.materials.insert(gpu))
    }

    /// Replaces a material in place; instances using it pick up the change next frame.
    pub fn set_material(&mut self, id: MaterialId, material: &StandardMaterial) {
        let gpu = self.gpu_material(material);
        let slot = self
            .materials
            .items
            .get_mut(id.0 as usize)
            .and_then(Option::as_mut)
            .expect("set_material on a removed material");
        *slot = gpu;
    }

    pub fn remove_material(&mut self, material: MaterialId) {
        self.materials.remove(material.0);
    }

    /// Adds a particle emitter with no particles yet. Its seed chain starts with the next
    /// [`World::step_particles`].
    pub fn add_emitter(&mut self, emitter: &Emitter) -> EmitterId {
        EmitterId(self.emitters.insert(EmitterState::new(emitter)))
    }

    /// Replaces an emitter's description (e.g. to stop its spawner with `active: false`). Alive
    /// particles keep their attributes and keep moving; the new update and render modifiers
    /// apply to them. A different capacity drops them.
    pub fn set_emitter(&mut self, id: EmitterId, emitter: &Emitter) {
        let e = self.emitters_mut(id);
        if emitter.capacity != e.desc.capacity
            && let Some(base) = e.base.take()
        {
            self.particles.free(base);
        }
        self.emitters_mut(id).desc = emitter.clone();
    }

    /// Where new particles spawn: init positions and velocities are transformed by it. Alive
    /// particles stay where they are (the simulation runs in world space).
    pub fn set_emitter_transform(&mut self, id: EmitterId, world_from_emitter: Mat4) {
        self.emitters_mut(id).transform = world_from_emitter;
    }

    pub fn remove_emitter(&mut self, id: EmitterId) {
        if let Some(base) = self.emitters.remove(id.0).and_then(|e| e.base) {
            self.particles.free(base);
        }
    }

    /// Particles the emitter's spawner asked for in the last step (hanabi
    /// `EffectSpawner::spawn_count`); spawns beyond its capacity are dropped on the GPU.
    pub fn spawn_count(&self, id: EmitterId) -> u32 {
        self.emitters
            .get(id.0)
            .expect("spawn_count on a removed emitter")
            .spawn_count
    }

    /// Every emitter's alive-particle count after the last submitted
    /// [`World::step_particles`]. Reads the GPU counters back and waits for them, so it is meant
    /// for tests and tools, not for every frame of a game.
    pub fn read_alive_particles(&self) -> Vec<(EmitterId, u32)> {
        let counts = self.particles.read_alive(&self.queue);
        self.emitter_ids()
            .map(|(id, e)| {
                let n = e.base.and_then(|_| counts.get(id as usize).copied());
                (EmitterId(id), n.unwrap_or(0))
            })
            .collect()
    }

    /// Records one simulation step of `dt` seconds for every emitter into `encoder`: the
    /// spawners tick on the CPU, then a compute pass spawns, initialises, ages, reaps and moves
    /// the particles on the GPU. Call it at the game's simulation rate (a fixed step keeps runs
    /// repeatable), before [`Renderer::encode`] of the frame that shows the result; several
    /// steps may go into one encoder. With `timestamps`, the pass writes its begin and end
    /// timestamps at `first` and `first + 1`.
    pub fn step_particles(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        timestamps: Option<FrameTimestamps<'_>>,
    ) {
        let emitters = self
            .emitters
            .items
            .iter_mut()
            .enumerate()
            .filter_map(|(i, e)| Some((i as u32, e.as_mut()?)));
        self.particles.step(encoder, emitters, dt, timestamps);
    }

    fn emitters_mut(&mut self, id: EmitterId) -> &mut EmitterState {
        self.emitters
            .items
            .get_mut(id.0 as usize)
            .and_then(Option::as_mut)
            .expect("emitter is removed")
    }

    /// `(id, emitter)` of every emitter.
    pub(crate) fn emitter_ids(&self) -> impl Iterator<Item = (u32, &EmitterState)> {
        self.emitters
            .items
            .iter()
            .enumerate()
            .filter_map(|(i, e)| Some((i as u32, e.as_ref()?)))
    }

    pub(crate) fn particle_pool(&self) -> &ParticlePool {
        &self.particles
    }

    pub(crate) fn mesh(&self, id: MeshId) -> &GpuMesh {
        self.meshes.get(id.0).expect("instance uses a removed mesh")
    }

    pub(crate) fn material(&self, id: MaterialId) -> &GpuMaterial {
        self.materials
            .get(id.0)
            .expect("instance uses a removed material")
    }

    fn gpu_material(&self, m: &StandardMaterial) -> GpuMaterial {
        let mut f = 0;
        let mut slot = |texture: Option<TextureId>, flag: u32| match texture {
            Some(id) => {
                f |= flag;
                self.textures
                    .get(id.0)
                    .expect("material uses a removed texture")
            }
            None => &self.white,
        };
        let base_color = slot(m.base_color_texture, flags::BASE_COLOR_TEXTURE);
        let emissive = slot(m.emissive_texture, flags::EMISSIVE_TEXTURE);
        let metallic_roughness = slot(
            m.metallic_roughness_texture,
            flags::METALLIC_ROUGHNESS_TEXTURE,
        );
        let occlusion = slot(m.occlusion_texture, flags::OCCLUSION_TEXTURE);
        let normal_map = slot(m.normal_map_texture, flags::NORMAL_MAP);
        if m.double_sided {
            f |= flags::DOUBLE_SIDED;
        }
        if m.unlit {
            f |= flags::UNLIT;
        }
        let alpha_cutoff = match m.alpha_mode {
            AlphaMode::Opaque => 0.0,
            AlphaMode::Mask(cutoff) => {
                f |= flags::ALPHA_MASK;
                cutoff
            }
            AlphaMode::Blend => {
                f |= flags::ALPHA_BLEND;
                0.05
            }
        };
        let uniform = MaterialUniform {
            base_color: m.base_color,
            emissive: m.emissive.extend(1.0),
            metallic: m.metallic,
            perceptual_roughness: m.perceptual_roughness,
            reflectance: m.reflectance,
            alpha_cutoff,
            flags: f,
            _pad: [0; 3],
        };
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }];
        for (i, t) in [
            base_color,
            emissive,
            metallic_roughness,
            occlusion,
            normal_map,
        ]
        .into_iter()
        .enumerate()
        {
            entries.push(wgpu::BindGroupEntry {
                binding: 1 + 2 * i as u32,
                resource: wgpu::BindingResource::TextureView(&t.view),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 2 + 2 * i as u32,
                resource: wgpu::BindingResource::Sampler(&t.sampler),
            });
        }
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material"),
            layout: &self.material_layout,
            entries: &entries,
        });
        GpuMaterial {
            material: *m,
            group,
        }
    }
}

fn upload_texture(device: &wgpu::Device, queue: &wgpu::Queue, t: &TextureData) -> GpuTexture {
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("material texture"),
            size: wgpu::Extent3d {
                width: t.width,
                height: t.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: t.levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: t.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::MipMajor,
        &t.levels.concat(),
    );
    let s = t.sampler;
    GpuTexture {
        view: texture.create_view(&Default::default()),
        sampler: device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material texture"),
            address_mode_u: s.address_mode_u,
            address_mode_v: s.address_mode_v,
            mag_filter: s.mag_filter,
            min_filter: s.min_filter,
            mipmap_filter: s.mipmap_filter,
            ..Default::default()
        }),
    }
}

/// A mesh's interleaved streams besides positions and normals (which upload as they are).
struct Streams {
    surface: Vec<SurfaceVertex>,
    skin: Option<Vec<SkinVertex>>,
}

impl Streams {
    fn new(mesh: &MeshData) -> Self {
        let n = mesh.positions.len();
        assert_eq!(mesh.normals.len(), n, "mesh needs one normal per position");
        for (name, len) in [
            ("uvs", mesh.uvs.len()),
            ("tangents", mesh.tangents.len()),
            ("colors", mesh.colors.len()),
            ("joints", mesh.joints.len()),
        ] {
            assert!(
                len == 0 || len == n,
                "mesh {name}: {len} values for {n} positions"
            );
        }
        let skinned = !mesh.joints.is_empty();
        assert_eq!(
            mesh.weights.len(),
            if skinned { n } else { 0 },
            "mesh weights: one per position with joints, none without"
        );
        Self {
            surface: (0..n)
                .map(|i| SurfaceVertex {
                    uv: mesh.uvs.get(i).map_or([0.0; 2], |v| v.to_array()),
                    tangent: mesh.tangents.get(i).map_or([0.0; 4], |v| v.to_array()),
                    color: mesh.colors.get(i).map_or([1.0; 4], |v| v.to_array()),
                })
                .collect(),
            skin: skinned.then(|| {
                mesh.joints
                    .iter()
                    .zip(&mesh.weights)
                    .map(|(j, w)| SkinVertex {
                        joints: *j,
                        weights: w.to_array(),
                    })
                    .collect()
            }),
        }
    }
}

/// Bevy `SkinnedMeshBounds::from_mesh`: per joint, the box around the bind-pose positions it
/// has a non-zero weight on.
fn joint_bounds(mesh: &MeshData) -> Vec<(u16, Aabb)> {
    let mut points: std::collections::BTreeMap<u16, Vec<Vec3>> = Default::default();
    for ((p, j), w) in mesh.positions.iter().zip(&mesh.joints).zip(&mesh.weights) {
        for k in 0..4 {
            if w[k] != 0.0 {
                points.entry(j[k]).or_default().push(*p);
            }
        }
    }
    points
        .into_iter()
        .map(|(j, p)| (j, Aabb::from_points(&p)))
        .collect()
}
