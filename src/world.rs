//! The consumer's GPU-resident content: meshes, textures and materials uploaded once and kept
//! under stable handles until removed. [`Instance`]s (a mesh, a material and a transform) reach
//! the renderer two ways: retained in the world ([`World::add_instance`], drawn every frame until
//! removed and rewritten only when edited) or listed per frame to [`Renderer::prepare`]. Nothing
//! else is uploaded again.
//! Particle emitters live here too: the consumer advances them with [`World::step_particles`]
//! and the renderer draws every alive particle.

use crate::{
    cull::Aabb,
    material::{AlphaMode, StandardMaterial, TextureData},
    mesh::MeshData,
    particle_sim::ParticlePool,
    particles::{Emitter, EmitterState, ParticleState},
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

/// A retained instance in a [`World`]; drawn every frame until [`World::remove_instance`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstanceId(u32);

/// One drawn copy of a mesh: listed for one frame, or retained in a [`World`].
#[derive(Clone, Copy, Debug, PartialEq)]
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
    /// The [`World`] tick of its last upload or edit; retained instances of it are derived
    /// again when it is newer than a viewport's copy.
    pub changed: u64,
}

struct GpuTexture {
    /// Unique per upload (a [`World`] tick; 0 for the white fallback), so a material can tell
    /// whether a reused [`TextureId`] still names the texture its bind group holds.
    serial: u64,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// The texture and its sampler for the particle pass, built once with the texture rather
    /// than per emitter and frame (#85).
    particle: wgpu::BindGroup,
}

pub(crate) struct GpuMaterial {
    pub material: StandardMaterial,
    pub group: wgpu::BindGroup,
    /// The uniform the bind group holds; `set_material` writes it in place while the
    /// textures stay the same.
    uniform: wgpu::Buffer,
    /// [`GpuTexture::serial`] of each texture slot the bind group holds.
    textures: [u64; 5],
    /// The [`World`] tick of its last build or rewrite, like [`GpuMesh::changed`].
    pub changed: u64,
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
    particle_layout: wgpu::BindGroupLayout,
    /// Bound for every texture slot a material leaves empty (the shader skips it).
    white: GpuTexture,
    meshes: Slots<GpuMesh>,
    textures: Slots<GpuTexture>,
    materials: Slots<GpuMaterial>,
    skins: Slots<Vec<Mat4>>,
    emitters: Slots<EmitterState>,
    particles: ParticlePool,
    instances: Slots<Instance>,
    /// Per instance slot, the tick of its last add, set or removal.
    instance_ticks: Vec<u64>,
    /// Counts the edits retained instances depend on (instances, meshes, materials); every
    /// such edit stamps what it changed with the new value.
    tick: u64,
    /// The tick of the last mesh or material edit.
    content_tick: u64,
    /// Tells worlds apart, so a viewport's retained copy follows the world it is given.
    id: u64,
    work: WorldWork,
}

/// GPU work a [`World`] has done for the consumer's edits since it was created, counted where
/// it is done. Subtract two readings for the work between them: a single edit costs the
/// edited mesh, texture or material alone, whatever else the world holds, and
/// [`Renderer::prepare`] never adds to it (see "Work contracts" in
/// `docs/consumer-integration.md`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorldWork {
    /// Meshes whose GPU buffers were created: `add_mesh`, and `set_mesh` when the vertex or
    /// index count or the skinning changed.
    pub mesh_builds: u64,
    /// Meshes rewritten in their buffers: `set_mesh` of the same topology and
    /// `set_mesh_positions_normals`.
    pub mesh_rewrites: u64,
    /// Textures uploaded (`add_texture`).
    pub texture_uploads: u64,
    /// Materials built (`add_material`, and `set_material` when the textures changed).
    pub material_builds: u64,
    /// Materials rewritten in their uniform buffers: `set_material` with the same textures.
    pub material_rewrites: u64,
    /// Retained instances added, changed or removed (`add_instance`, `set_instance` with a
    /// different value, `remove_instance`). They upload nothing here: the next
    /// [`Renderer::prepare`] of each viewport writes the edited instances alone.
    pub instance_edits: u64,
    /// Bytes written to the GPU by all of the above.
    pub upload_bytes: u64,
}

impl std::ops::Sub for WorldWork {
    type Output = Self;

    fn sub(self, earlier: Self) -> Self {
        Self {
            mesh_builds: self.mesh_builds - earlier.mesh_builds,
            mesh_rewrites: self.mesh_rewrites - earlier.mesh_rewrites,
            texture_uploads: self.texture_uploads - earlier.texture_uploads,
            material_builds: self.material_builds - earlier.material_builds,
            material_rewrites: self.material_rewrites - earlier.material_rewrites,
            instance_edits: self.instance_edits - earlier.instance_edits,
            upload_bytes: self.upload_bytes - earlier.upload_bytes,
        }
    }
}

impl World {
    /// An empty world on the renderer's device.
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();
        let particle_layout = renderer.particle_texture_layout().clone();
        let white = upload_texture(
            &device,
            &queue,
            &particle_layout,
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
            particle_layout,
            device,
            queue,
            white,
            meshes: Slots::new(),
            textures: Slots::new(),
            materials: Slots::new(),
            skins: Slots::new(),
            emitters: Slots::new(),
            particles: ParticlePool::new(renderer.device(), renderer.sim_pipelines()),
            instances: Slots::new(),
            instance_ticks: Vec::new(),
            tick: 0,
            content_tick: 0,
            id: {
                static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
            work: WorldWork::default(),
        }
    }

    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    fn touch_content(&mut self) -> u64 {
        self.content_tick = self.next_tick();
        self.content_tick
    }

    /// Adds an instance that is drawn every frame, in every viewport, until removed, beside
    /// the frame's listed instances (Bevy: a mesh entity in the retained render world). A
    /// viewport writes it to the GPU once and again only after it changes, so static scenery
    /// costs no instance upload per frame. Its mesh, material and skin must outlive it.
    pub fn add_instance(&mut self, instance: Instance) -> InstanceId {
        let slot = self.instances.insert(instance);
        self.stamp_instance(slot);
        InstanceId(slot)
    }

    /// Replaces a retained instance (a new transform, mesh, material or skin); the next
    /// [`Renderer::prepare`] draws the new value. Setting the value it already has costs
    /// nothing.
    pub fn set_instance(&mut self, id: InstanceId, instance: Instance) {
        let slot = self.instances.items[id.0 as usize]
            .as_mut()
            .expect("set_instance on a removed instance");
        if *slot != instance {
            *slot = instance;
            self.stamp_instance(id.0);
        }
    }

    /// A retained instance's current value.
    pub fn instance(&self, id: InstanceId) -> Instance {
        *self
            .instances
            .get(id.0)
            .expect("instance() of a removed instance")
    }

    /// Stops drawing a retained instance; its handle may be reused.
    pub fn remove_instance(&mut self, id: InstanceId) {
        if self.instances.remove(id.0).is_some() {
            self.stamp_instance(id.0);
        }
    }

    fn stamp_instance(&mut self, slot: u32) {
        let tick = self.next_tick();
        let slot = slot as usize;
        if self.instance_ticks.len() <= slot {
            self.instance_ticks.resize(slot + 1, 0);
        }
        self.instance_ticks[slot] = tick;
        self.work.instance_edits += 1;
    }

    /// Every retained-instance slot (`None` where removed) with the tick of its last edit.
    pub(crate) fn instance_slots(&self) -> impl Iterator<Item = (Option<&Instance>, u64)> {
        self.instances
            .items
            .iter()
            .map(Option::as_ref)
            .zip(self.instance_ticks.iter().copied())
    }

    /// The retained instance in `slot`.
    pub(crate) fn instance_at(&self, slot: u32) -> Option<&Instance> {
        self.instances.get(slot)
    }

    /// Retained-instance slots, the removed ones included.
    pub(crate) fn instance_slot_count(&self) -> usize {
        self.instances.items.len()
    }

    /// Tells this world apart from every other one.
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// The tick of the last edit retained instances depend on.
    pub(crate) fn tick(&self) -> u64 {
        self.tick
    }

    /// The tick of the last mesh or material edit.
    pub(crate) fn content_tick(&self) -> u64 {
        self.content_tick
    }

    /// The work done for the consumer's edits so far.
    pub fn work(&self) -> WorldWork {
        self.work
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
        let changed = self.touch_content();
        let slot = self.meshes.items[id.0 as usize]
            .as_mut()
            .expect("checked above");
        slot.changed = changed;
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
        self.work.mesh_rewrites += 1;
        self.work.upload_bytes += streams.bytes(mesh);
    }

    /// Rewrites only a mesh's positions and normals, one per vertex, keeping its other
    /// attributes and its indices: a CPU-skinned or morphed pose of an unchanged topology. Two
    /// buffer writes of 12 bytes per vertex each, no re-interleaving. Skinned meshes take their
    /// poses through [`World::set_skin`] instead (their per-joint bounds stay those of the
    /// first upload).
    pub fn set_mesh_positions_normals(&mut self, id: MeshId, positions: &[Vec3], normals: &[Vec3]) {
        let changed = self.touch_content();
        let slot = self.meshes.items[id.0 as usize]
            .as_mut()
            .expect("set_mesh_positions_normals on a removed mesh");
        slot.changed = changed;
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
        self.work.mesh_rewrites += 1;
        self.work.upload_bytes += (size_of_val(positions) + size_of_val(normals)) as u64;
    }

    fn gpu_mesh(&mut self, mesh: &MeshData) -> GpuMesh {
        let changed = self.touch_content();
        let streams = Streams::new(mesh);
        self.work.mesh_builds += 1;
        self.work.upload_bytes += streams.bytes(mesh);
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
            changed,
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
        self.work.texture_uploads += 1;
        self.work.upload_bytes += texture.levels.iter().map(|l| l.len() as u64).sum::<u64>();
        let mut gpu = upload_texture(&self.device, &self.queue, &self.particle_layout, texture);
        gpu.serial = self.next_tick();
        TextureId(self.textures.insert(gpu))
    }
    pub fn remove_texture(&mut self, texture: TextureId) {
        self.textures.remove(texture.0);
    }

    /// Adds a material; its textures must be in this world.
    pub fn add_material(&mut self, material: &StandardMaterial) -> MaterialId {
        let gpu = self.gpu_material(material);
        MaterialId(self.materials.insert(gpu))
    }

    /// Replaces a material in place; instances using it pick up the change next frame. With
    /// the same textures it only writes the material's uniform buffer, so per-frame tints
    /// create no GPU objects; other texture choices build it again.
    pub fn set_material(&mut self, id: MaterialId, material: &StandardMaterial) {
        let (uniform, textures) = self.material_uniform(material);
        let textures = textures.map(|t| t.serial);
        let gpu = self
            .materials
            .items
            .get(id.0 as usize)
            .and_then(Option::as_ref)
            .expect("set_material on a removed material");
        if gpu.textures != textures {
            self.materials.items[id.0 as usize] = Some(self.gpu_material(material));
            return;
        }
        self.queue
            .write_buffer(&gpu.uniform, 0, bytemuck::bytes_of(&uniform));
        self.work.material_rewrites += 1;
        self.work.upload_bytes += size_of::<MaterialUniform>() as u64;
        let changed = self.touch_content();
        let gpu = self.materials.items[id.0 as usize].as_mut().unwrap();
        gpu.material = *material;
        gpu.changed = changed;
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

    /// Triggers a burst (an impact, a muzzle flash): the next [`World::step_particles`] spawns
    /// `count` particles at `world_from_emitter` through the emitter's init modifiers, on top of
    /// what its spawner spawns, active or not. Each burst of a step spawns at its own transform;
    /// the emitter's own transform is left alone. Spawns beyond the capacity are dropped.
    pub fn burst(&mut self, id: EmitterId, count: u32, world_from_emitter: Mat4) {
        if count > 0 {
            self.emitters_mut(id)
                .bursts
                .push((count, world_from_emitter));
        }
    }

    /// Records into `encoder` replacing the emitter's particles with `particles`, world-space
    /// states from outside the engine (physics debris, server-authoritative projectiles): the
    /// frame encoded next draws them through the emitter's render modifiers
    /// ([`ParticleState`]). Those whose age has reached their lifetime are left out, and those
    /// beyond the emitter's capacity. Later steps age, accelerate and move them like spawned
    /// particles, and the spawner keeps spawning beside them; a consumer that feeds every frame
    /// usually sets the emitter inactive and calls this after its step.
    pub fn set_particles(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        id: EmitterId,
        particles: &[ParticleState],
    ) {
        let e = self
            .emitters
            .items
            .get_mut(id.0 as usize)
            .and_then(Option::as_mut)
            .expect("emitter is removed");
        self.particles.set_particles(encoder, id.0, e, particles);
    }

    /// Particles the emitter's spawner and triggered bursts asked for in the last step (hanabi
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

    /// A particle texture's bind group (texture and sampler); white for none.
    pub(crate) fn particle_texture(&self, id: Option<TextureId>) -> &wgpu::BindGroup {
        let t = match id {
            Some(id) => self
                .textures
                .get(id.0)
                .expect("emitter uses a removed texture"),
            None => &self.white,
        };
        &t.particle
    }

    pub(crate) fn mesh(&self, id: MeshId) -> &GpuMesh {
        self.meshes.get(id.0).expect("instance uses a removed mesh")
    }

    pub(crate) fn material(&self, id: MaterialId) -> &GpuMaterial {
        self.materials
            .get(id.0)
            .expect("instance uses a removed material")
    }

    /// Builds a material, counting it and stamping it with a new content tick.
    fn gpu_material(&mut self, m: &StandardMaterial) -> GpuMaterial {
        self.work.material_builds += 1;
        self.work.upload_bytes += size_of::<MaterialUniform>() as u64;
        let changed = self.touch_content();
        let (uniform, textures) = self.material_uniform(m);
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }];
        for (i, t) in textures.iter().enumerate() {
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
            textures: textures.map(|t| t.serial),
            uniform,
            changed,
        }
    }

    /// A material's uniform and the textures its bind group holds, in binding order.
    fn material_uniform(&self, m: &StandardMaterial) -> (MaterialUniform, [&GpuTexture; 5]) {
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
        let textures = [
            slot(m.base_color_texture, flags::BASE_COLOR_TEXTURE),
            slot(m.emissive_texture, flags::EMISSIVE_TEXTURE),
            slot(
                m.metallic_roughness_texture,
                flags::METALLIC_ROUGHNESS_TEXTURE,
            ),
            slot(m.occlusion_texture, flags::OCCLUSION_TEXTURE),
            slot(m.normal_map_texture, flags::NORMAL_MAP),
        ];
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
                m.blend_shadow_alpha_cutoff
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
        (uniform, textures)
    }
}

fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    particle_layout: &wgpu::BindGroupLayout,
    t: &TextureData,
) -> GpuTexture {
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
    let view = texture.create_view(&Default::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("material texture"),
        address_mode_u: s.address_mode_u,
        address_mode_v: s.address_mode_v,
        mag_filter: s.mag_filter,
        min_filter: s.min_filter,
        mipmap_filter: s.mipmap_filter,
        ..Default::default()
    });
    let particle = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("particle texture"),
        layout: particle_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    GpuTexture {
        serial: 0,
        view,
        sampler,
        particle,
    }
}

/// A mesh's interleaved streams besides positions and normals (which upload as they are).
struct Streams {
    surface: Vec<SurfaceVertex>,
    skin: Option<Vec<SkinVertex>>,
}

impl Streams {
    /// Bytes a full upload of `mesh` with these streams writes.
    fn bytes(&self, mesh: &MeshData) -> u64 {
        (size_of_val(mesh.positions.as_slice())
            + size_of_val(mesh.normals.as_slice())
            + size_of_val(self.surface.as_slice())
            + self.skin.as_deref().map_or(0, size_of_val)
            + size_of_val(mesh.indices.as_slice())) as u64
    }

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
