//! The GPU half of the particle simulation (`shaders/particle_sim.wgsl`): one pool of particle
//! slots shared by every emitter of a [`World`](crate::world::World), and the compute passes of a
//! step.
//!
//! Each emitter owns a range of `capacity` slots in the pool, allocated on its first step. The
//! per-emitter counters (alive count, draw arguments) are indexed by its id. A step uploads every
//! emitter's spawn count, seed and modifiers by copies from a staging buffer recorded into the
//! encoder, so several steps can be recorded into one encoder.

use crate::{
    particles::{
        ColorBlend, EmitterState, InitModifier, MAX_GRADIENT_KEYS, MAX_INIT_MODIFIERS,
        ParticleState, ShapeDimension, Value, render_attributes,
    },
    renderer::FrameTimestamps,
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuModifier {
    kind: u32,
    mode: u32,
    _pad: [u32; 2],
    a: Vec4,
    b: Vec4,
}

/// `EmitterStep` in `particle_sim.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuEmitterStep {
    world_from_emitter: Mat4,
    accel: Vec4,
    color_t: [f32; MAX_GRADIENT_KEYS],
    color_v: [Vec4; MAX_GRADIENT_KEYS],
    size_t: [f32; MAX_GRADIENT_KEYS],
    size_v: [Vec2; MAX_GRADIENT_KEYS],
    id: u32,
    base: u32,
    capacity: u32,
    seed: u32,
    spawn: u32,
    spawn_first: u32,
    n_mods: u32,
    color_blend: u32,
    n_color: u32,
    n_size: u32,
    /// Flipbook frames over life (0: no flipbook, the sprite index stays 0).
    sprite_frames: u32,
    /// Spawns of earlier entries of the same emitter in this step (its triggered bursts, each an
    /// entry right before its spawner's): this entry's spawns take the dead-stack rows after
    /// theirs. Only an emitter's last entry ages and moves its particles.
    row_offset: u32,
    mods: [GpuModifier; MAX_INIT_MODIFIERS],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StepHeader {
    count: u32,
    total_spawn: u32,
    dt: f32,
    _pad: u32,
}

/// `Particle` in `particle_sim.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuParticle {
    position: Vec3,
    age: f32,
    velocity: Vec3,
    lifetime: f32,
    color: Vec4,
    size: f32,
    _pad: [f32; 3],
}

/// `Quad` in `particle_sim.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuQuad {
    position: Vec3,
    sprite: u32,
    velocity: Vec3,
    _pad0: u32,
    size: Vec2,
    _pad1: Vec2,
    color: Vec4,
}

const PARTICLE_BYTES: u64 = std::mem::size_of::<GpuParticle>() as u64;
/// `Quad`: what the render pass reads per instance (`particle_pass`).
pub(crate) const QUAD_BYTES: u64 = std::mem::size_of::<GpuQuad>() as u64;
/// `Counters`: draw arguments (`wgpu::util::DrawIndirectArgs`), alive count, padding.
pub(crate) const COUNTER_BYTES: u64 = 32;

mod kind {
    pub const POSITION_SPHERE: u32 = 0;
    pub const POSITION: u32 = 1;
    pub const VELOCITY_SPHERE: u32 = 2;
    pub const VELOCITY: u32 = 3;
    pub const LIFETIME: u32 = 4;
    pub const AGE: u32 = 5;
    pub const SIZE: u32 = 6;
    pub const COLOR: u32 = 7;
    pub const VELOCITY_CONE: u32 = 8;
}

/// `(mode, a, b)` of a vector: constant in `a`, or uniform between `a` and `b`.
fn value<T: Copy>(v: Value<T>, f: impl Fn(T) -> Vec4) -> (u32, Vec4, Vec4) {
    match v {
        Value::Const(c) => (0, f(c), Vec4::ZERO),
        Value::Uniform { min, max } => (1, f(min), f(max)),
    }
}

/// `(mode, a, b)` of a scalar: constant in `a.x`, or uniform between `a.x` and `a.y` (`scalar`
/// in the shader).
fn scalar(v: Value<f32>) -> (u32, Vec4, Vec4) {
    match v {
        Value::Const(c) => (0, Vec4::new(c, 0.0, 0.0, 0.0), Vec4::ZERO),
        Value::Uniform { min, max } => (1, Vec4::new(min, max, 0.0, 0.0), Vec4::ZERO),
    }
}

fn modifier(m: &InitModifier) -> GpuModifier {
    let (kind, (mode, a, b)) = match *m {
        InitModifier::PositionSphere {
            center,
            radius,
            dimension,
        } => (
            kind::POSITION_SPHERE,
            (
                match dimension {
                    ShapeDimension::Volume => 0,
                    ShapeDimension::Surface => 1,
                },
                center.extend(radius),
                Vec4::ZERO,
            ),
        ),
        InitModifier::Position(v) => (kind::POSITION, value(v, |v: Vec3| v.extend(0.0))),
        InitModifier::VelocitySphere { center, speed } => {
            // The speed goes in `b` (`b.x`, or `b.x..b.y`).
            let (mode, speed, _) = scalar(speed);
            (kind::VELOCITY_SPHERE, (mode, center.extend(0.0), speed))
        }
        InitModifier::Velocity(v) => (kind::VELOCITY, value(v, |v: Vec3| v.extend(0.0))),
        InitModifier::VelocityCone {
            direction,
            spread,
            speed,
        } => {
            // The unit axis and the cosine of the half-angle in `a`, the speed in `b`.
            let (mode, speed, _) = scalar(speed);
            let axis = direction.normalize().extend(spread.cos());
            (kind::VELOCITY_CONE, (mode, axis, speed))
        }
        InitModifier::Lifetime(v) => (kind::LIFETIME, scalar(v)),
        InitModifier::Age(v) => (kind::AGE, scalar(v)),
        InitModifier::Size(v) => (kind::SIZE, scalar(v)),
        InitModifier::Color(v) => (kind::COLOR, value(v, |v| v)),
    };
    GpuModifier {
        kind,
        mode,
        _pad: [0; 2],
        a,
        b,
    }
}

fn emitter_step(
    id: u32,
    e: &EmitterState,
    spawn: u32,
    spawn_first: u32,
    seed: u32,
) -> GpuEmitterStep {
    let d = &e.desc;
    assert!(
        d.init.len() <= MAX_INIT_MODIFIERS,
        "an emitter takes at most {MAX_INIT_MODIFIERS} init modifiers"
    );
    let mut s = GpuEmitterStep::zeroed();
    s.world_from_emitter = e.transform;
    s.accel = d.accel.extend(0.0);
    s.id = id;
    s.base = e.base.expect("allocated before the step");
    s.capacity = d.capacity;
    s.seed = seed;
    s.spawn = spawn;
    s.spawn_first = spawn_first;
    s.n_mods = d.init.len() as u32;
    for (g, m) in s.mods.iter_mut().zip(&d.init) {
        *g = modifier(m);
    }
    if let Some(g) = &d.size_over_life {
        assert!(
            g.keys.len() <= MAX_GRADIENT_KEYS,
            "at most {MAX_GRADIENT_KEYS} size keys"
        );
        s.n_size = g.keys.len() as u32;
        for (i, &(t, v)) in g.keys.iter().enumerate() {
            s.size_t[i] = t;
            s.size_v[i] = v;
        }
    }
    s.sprite_frames = d.texture.and_then(|t| t.flipbook).map_or(0, |f| f.frames);
    s.color_blend = 3;
    if let Some(c) = &d.color_over_life
        && !c.gradient.keys.is_empty()
    {
        assert!(
            c.gradient.keys.len() <= MAX_GRADIENT_KEYS,
            "at most {MAX_GRADIENT_KEYS} colour keys"
        );
        s.n_color = c.gradient.keys.len() as u32;
        for (i, &(t, v)) in c.gradient.keys.iter().enumerate() {
            s.color_t[i] = t;
            s.color_v[i] = v;
        }
        s.color_blend = match c.blend {
            ColorBlend::Overwrite => 0,
            ColorBlend::Multiply => 1,
            ColorBlend::Add => 2,
        };
    }
    s
}

/// The simulation's compute pipelines, shared by every world on the renderer's device.
#[derive(Clone)]
pub(crate) struct SimPipelines {
    layout: wgpu::BindGroupLayout,
    /// `update_args` (prepare writes it) and `update_read` (update reads it and dispatches on
    /// it, which a read-write binding would conflict with).
    update_write_layout: wgpu::BindGroupLayout,
    update_read_layout: wgpu::BindGroupLayout,
    init: wgpu::ComputePipeline,
    prepare: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
}

impl SimPipelines {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/particle_sim.wgsl"));
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
            label: Some("particle sim"),
            entries: &[
                storage(0, true),
                storage(1, false),
                storage(2, false),
                storage(3, false),
                storage(4, false),
                storage(5, false),
                storage(6, false),
            ],
        });
        let update_layout = |read_only| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("particle update"),
                entries: &[storage(0, read_only)],
            })
        };
        let (update_write_layout, update_read_layout) = (update_layout(false), update_layout(true));
        let pipeline = |entry: &str, groups: &[Option<&wgpu::BindGroupLayout>]| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry),
                bind_group_layouts: groups,
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            init: pipeline("init", &[Some(&layout)]),
            prepare: pipeline("prepare", &[Some(&layout), Some(&update_write_layout)]),
            update: pipeline("update", &[Some(&layout), None, Some(&update_read_layout)]),
            layout,
            update_write_layout,
            update_read_layout,
        }
    }
}

/// Buffers sized for `slots` particle slots, `ids` emitter ids and `per_step` emitters a step.
struct Buffers {
    slots: u32,
    ids: u32,
    per_step: u32,
    particles: wgpu::Buffer,
    dead: wgpu::Buffer,
    quads: wgpu::Buffer,
    lists: [wgpu::Buffer; 2],
    counters: wgpu::Buffer,
    update: wgpu::Buffer,
    step: wgpu::Buffer,
    /// `groups[i]` reads alive list `i` and writes list `1 - i`.
    groups: [wgpu::BindGroup; 2],
    /// `update` for prepare (read-write, group 1) and for update (read-only, group 2).
    update_groups: [wgpu::BindGroup; 2],
}

impl Buffers {
    fn new(
        device: &wgpu::Device,
        pipelines: &SimPipelines,
        slots: u32,
        ids: u32,
        per_step: u32,
    ) -> Self {
        use wgpu::BufferUsages as U;
        let buffer = |label, size: u64, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: usage | U::STORAGE | U::COPY_SRC | U::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let n = u64::from(slots);
        let particles = buffer("particles", n * PARTICLE_BYTES, U::empty());
        let dead = buffer("particle dead slots", n * 4, U::empty());
        let quads = buffer("particle quads", n * QUAD_BYTES, U::VERTEX);
        let lists = [
            buffer("particle alive list", n * 4, U::empty()),
            buffer("particle alive list", n * 4, U::empty()),
        ];
        let counters = buffer(
            "particle counters",
            u64::from(ids) * COUNTER_BYTES,
            U::INDIRECT,
        );
        let update = buffer("particle update", 16 + u64::from(per_step) * 4, U::INDIRECT);
        let step = buffer(
            "particle step",
            (std::mem::size_of::<StepHeader>()
                + per_step as usize * std::mem::size_of::<GpuEmitterStep>()) as u64,
            U::empty(),
        );
        let group = |current: &wgpu::Buffer, next: &wgpu::Buffer| {
            let all = [&step, &particles, &dead, &quads, &counters, current, next];
            let entries: Vec<_> = all
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle sim"),
                layout: &pipelines.layout,
                entries: &entries,
            })
        };
        let groups = [group(&lists[0], &lists[1]), group(&lists[1], &lists[0])];
        let update_group = |layout| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle update"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: update.as_entire_binding(),
                }],
            })
        };
        let update_groups = [
            update_group(&pipelines.update_write_layout),
            update_group(&pipelines.update_read_layout),
        ];
        Self {
            update_groups,
            slots,
            ids,
            per_step,
            particles,
            dead,
            quads,
            lists,
            counters,
            update,
            step,
            groups,
        }
    }
}

/// Uploads into the pool's buffers, recorded into the encoder: [`Uploads::write`] queues bytes for
/// a target range, and [`Uploads::flush`] puts them in one staging buffer with
/// `Queue::write_buffer` and records copies into the targets, so they land in order with the
/// encoder's other commands. A staging buffer is reused once the work of the encoder that copied
/// from it is done. Unlike `wgpu::util::StagingBelt` nothing is mapped: on the WebGPU backend the
/// belt creates its chunks `mappedAtCreation` and old ones are only freed by the JS garbage
/// collector, which ran SwiftShader out of mappable memory after about 200 frames (#108).
struct Uploads {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Staging buffers, each with whether the GPU is done with it.
    staging: Vec<(wgpu::Buffer, Arc<AtomicBool>)>,
    bytes: Vec<u8>,
    /// `(target, target offset, staging offset, size)`.
    copies: Vec<(wgpu::Buffer, u64, u64, u64)>,
}

impl Uploads {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
            staging: Vec::new(),
            bytes: Vec::new(),
            copies: Vec::new(),
        }
    }

    /// Queues writing `bytes` (a multiple of 4 long) at `offset` of `target`.
    fn write(&mut self, target: &wgpu::Buffer, offset: u64, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let at = self.bytes.len() as u64;
        self.bytes.extend_from_slice(bytes);
        self.copies
            .push((target.clone(), offset, at, bytes.len() as u64));
    }

    /// Records the queued writes into `encoder`.
    fn flush(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.copies.is_empty() {
            return;
        }
        let size = self.bytes.len() as u64;
        let free = self
            .staging
            .iter()
            .position(|(_, free)| free.load(Ordering::Acquire));
        let fits = self
            .staging
            .iter()
            .position(|(b, free)| free.load(Ordering::Acquire) && b.size() >= size);
        let i = match (fits, free) {
            (Some(i), _) => i,
            (None, free) => {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("particle uploads"),
                    size: size.next_power_of_two().max(1 << 16),
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let entry = (buffer, Arc::new(AtomicBool::new(true)));
                // A free buffer too small for this upload is replaced, not kept beside it.
                match free {
                    Some(i) => {
                        self.staging[i] = entry;
                        i
                    }
                    None => {
                        self.staging.push(entry);
                        self.staging.len() - 1
                    }
                }
            }
        };
        let (buffer, free) = &self.staging[i];
        free.store(false, Ordering::Release);
        self.queue.write_buffer(buffer, 0, &self.bytes);
        for (target, offset, at, len) in self.copies.drain(..) {
            encoder.copy_buffer_to_buffer(buffer, at, &target, offset, len);
        }
        self.bytes.clear();
        let free = free.clone();
        encoder.on_submitted_work_done(move || free.store(true, Ordering::Release));
    }
}

/// A world's particle slots and per-emitter counters on the GPU.
pub(crate) struct ParticlePool {
    device: wgpu::Device,
    pipelines: SimPipelines,
    buffers: Buffers,
    /// Which alive list the next step reads.
    current: usize,
    uploads: Uploads,
    /// Allocated slot ranges `(base, len)`, sorted by base.
    ranges: Vec<(u32, u32)>,
}

impl ParticlePool {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, pipelines: &SimPipelines) -> Self {
        Self {
            buffers: Buffers::new(device, pipelines, 0, 0, 0),
            device: device.clone(),
            pipelines: pipelines.clone(),
            current: 0,
            uploads: Uploads::new(device, queue),
            ranges: Vec::new(),
        }
    }

    /// The quads the last step wrote (`QUAD_BYTES` each, an emitter's at its base slot) and the
    /// draw arguments by emitter id (`COUNTER_BYTES` each).
    pub fn draw_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer) {
        (&self.buffers.quads, &self.buffers.counters)
    }

    /// Returns an emitter's slot range to the pool.
    pub fn free(&mut self, base: u32) {
        self.ranges.retain(|r| r.0 != base);
    }

    /// The first free range of `len` slots (first fit, else after the last range).
    fn allocate(&mut self, len: u32) -> u32 {
        let mut start = 0;
        let mut at = self.ranges.len();
        for (i, &(base, n)) in self.ranges.iter().enumerate() {
            if base - start >= len {
                at = i;
                break;
            }
            start = base + n;
        }
        self.ranges.insert(at, (start, len));
        start
    }

    /// Grows the buffers to at least the given sizes, copying what the old ones hold.
    fn reserve(&mut self, encoder: &mut wgpu::CommandEncoder, slots: u32, ids: u32, per_step: u32) {
        let old = &self.buffers;
        if slots <= old.slots && ids <= old.ids && per_step <= old.per_step {
            return;
        }
        let grow = |need: u32, have: u32| {
            if need <= have {
                have
            } else {
                need.max(have * 2)
            }
        };
        let new = Buffers::new(
            &self.device,
            &self.pipelines,
            grow(slots, old.slots),
            grow(ids, old.ids),
            grow(per_step, old.per_step),
        );
        let n = u64::from(old.slots);
        let copy = |encoder: &mut wgpu::CommandEncoder,
                    from: &wgpu::Buffer,
                    to: &wgpu::Buffer,
                    size: u64| {
            if size > 0 {
                encoder.copy_buffer_to_buffer(from, 0, to, 0, size);
            }
        };
        copy(encoder, &old.particles, &new.particles, n * PARTICLE_BYTES);
        copy(encoder, &old.dead, &new.dead, n * 4);
        copy(encoder, &old.quads, &new.quads, n * QUAD_BYTES);
        copy(encoder, &old.lists[0], &new.lists[0], n * 4);
        copy(encoder, &old.lists[1], &new.lists[1], n * 4);
        copy(
            encoder,
            &old.counters,
            &new.counters,
            u64::from(old.ids) * COUNTER_BYTES,
        );
        self.buffers = new;
    }

    /// Grows the buffers for every allocated range, `ids` emitter ids and `per_step` step
    /// entries.
    fn reserve_for(&mut self, encoder: &mut wgpu::CommandEncoder, ids: u32, per_step: u32) {
        let slots = self.ranges.last().map_or(0, |r| r.0 + r.1);
        self.reserve(encoder, slots, ids, per_step);
    }

    /// Empties emitter `id`'s range: no particles, the dead stack in slot order, nothing to draw.
    fn reset(&mut self, encoder: &mut wgpu::CommandEncoder, id: u32, base: u32, capacity: u32) {
        let b = &self.buffers;
        encoder.clear_buffer(
            &b.counters,
            u64::from(id) * COUNTER_BYTES,
            Some(COUNTER_BYTES),
        );
        let identity: Vec<u32> = (0..capacity).collect();
        self.uploads.write(
            &b.dead,
            u64::from(base) * 4,
            bytemuck::cast_slice(&identity),
        );
    }

    /// Records replacing emitter `id`'s particles with `states` (those still alive, at most its
    /// capacity), in slots `0..n` of its range: their attributes for later steps, the alive list
    /// the next step reads, the counters and the quads the render pass draws.
    pub fn set_particles(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        id: u32,
        e: &mut EmitterState,
        states: &[ParticleState],
    ) {
        let capacity = e.desc.capacity;
        let base = match e.base {
            Some(base) => base,
            None => *e.base.insert(self.allocate(capacity)),
        };
        self.reserve_for(encoder, id + 1, 0);
        self.reset(encoder, id, base, capacity);
        let states: Vec<&ParticleState> = states
            .iter()
            .filter(|p| p.age < p.lifetime)
            .take(capacity as usize)
            .collect();
        let n = states.len() as u32;
        let particles: Vec<GpuParticle> = states
            .iter()
            .map(|p| GpuParticle {
                position: p.position,
                age: p.age,
                velocity: p.velocity,
                lifetime: p.lifetime,
                color: p.color,
                size: p.size,
                _pad: [0.0; 3],
            })
            .collect();
        let quads: Vec<GpuQuad> = states
            .iter()
            .map(|p| {
                let (size, color, sprite) = render_attributes(&e.desc, p);
                GpuQuad {
                    position: p.position,
                    sprite,
                    velocity: p.velocity,
                    _pad0: 0,
                    size,
                    _pad1: Vec2::ZERO,
                    color,
                }
            })
            .collect();
        let slots: Vec<u32> = (0..n).collect();
        let counters: [u32; 8] = [6, n, 0, 0, n, 0, 0, 0];
        let b = &self.buffers;
        let base = u64::from(base);
        let mut write = |buffer: &wgpu::Buffer, offset: u64, bytes: &[u8]| {
            self.uploads.write(buffer, offset, bytes);
        };
        write(
            &b.particles,
            base * PARTICLE_BYTES,
            bytemuck::cast_slice(&particles),
        );
        write(&b.quads, base * QUAD_BYTES, bytemuck::cast_slice(&quads));
        write(
            &b.lists[self.current],
            base * 4,
            bytemuck::cast_slice(&slots),
        );
        write(
            &b.counters,
            u64::from(id) * COUNTER_BYTES,
            bytemuck::cast_slice(&counters),
        );
        self.uploads.flush(encoder);
    }

    /// Records one step of `dt` seconds for `emitters` (`(id, state)`): allocates and resets the
    /// ranges of new emitters, ticks the spawners and the seed chains, uploads the step and
    /// runs init, prepare and update in one compute pass (timed by two queries at
    /// `timestamps.first`).
    pub fn step<'a>(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        emitters: impl Iterator<Item = (u32, &'a mut EmitterState)>,
        dt: f32,
        timestamps: Option<FrameTimestamps<'_>>,
    ) {
        let mut emitters: Vec<(u32, &mut EmitterState)> = emitters.collect();
        let mut fresh = Vec::new();
        for (id, e) in &mut emitters {
            if e.base.is_none() {
                e.base = Some(self.allocate(e.desc.capacity));
                fresh.push((*id, e.base.unwrap(), e.desc.capacity));
            }
        }
        let ids = emitters.iter().map(|(id, _)| id + 1).max().unwrap_or(0);
        let entries = emitters
            .iter()
            .map(|(_, e)| 1 + e.bursts.len())
            .sum::<usize>();
        self.reserve_for(encoder, ids, entries as u32);
        for &(id, base, capacity) in &fresh {
            self.reset(encoder, id, base, capacity);
        }

        // Per emitter, an entry per triggered burst (spawning at its transform), then the
        // spawner's entry, which also ages and moves every particle of the emitter.
        let mut steps = Vec::with_capacity(entries);
        let mut total_spawn = 0;
        for (id, e) in &mut emitters {
            let (spawn, seed) = e.tick(dt);
            let mut row_offset = 0;
            for (count, transform) in std::mem::take(&mut e.bursts) {
                let mut s = emitter_step(*id, e, count, total_spawn, seed);
                s.world_from_emitter = transform;
                s.row_offset = row_offset;
                steps.push(s);
                total_spawn += count;
                row_offset += count;
            }
            let mut s = emitter_step(*id, e, spawn, total_spawn, seed);
            s.row_offset = row_offset;
            steps.push(s);
            total_spawn += spawn;
        }
        let b = &self.buffers;
        let header = StepHeader {
            count: steps.len() as u32,
            total_spawn,
            dt,
            _pad: 0,
        };
        let mut bytes = bytemuck::bytes_of(&header).to_vec();
        bytes.extend_from_slice(bytemuck::cast_slice(&steps));
        self.uploads.write(&b.step, 0, &bytes);
        self.uploads.flush(encoder);

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particle simulation"),
            timestamp_writes: timestamps
                .as_ref()
                .map(|t| wgpu::ComputePassTimestampWrites {
                    query_set: t.query_set,
                    beginning_of_pass_write_index: Some(t.first),
                    end_of_pass_write_index: Some(t.first + 1),
                }),
        });
        if !steps.is_empty() {
            pass.set_bind_group(0, &b.groups[self.current], &[]);
            if total_spawn > 0 {
                pass.set_pipeline(&self.pipelines.init);
                pass.dispatch_workgroups(total_spawn.div_ceil(64), 1, 1);
            }
            pass.set_pipeline(&self.pipelines.prepare);
            pass.set_bind_group(1, &b.update_groups[0], &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_bind_group(1, None, &[]);
            pass.set_bind_group(2, &b.update_groups[1], &[]);
            pass.set_pipeline(&self.pipelines.update);
            pass.dispatch_workgroups_indirect(&b.update, 0);
            self.current = 1 - self.current;
        }
    }

    /// Every emitter id's alive count after the last submitted step (blocks on a readback).
    pub fn read_alive(&self, queue: &wgpu::Queue) -> Vec<u32> {
        let size = u64::from(self.buffers.ids) * COUNTER_BYTES;
        if size == 0 {
            return Vec::new();
        }
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle alive readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffers.counters, 0, &readback, 0, size);
        queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .expect("map_async callback dropped")
            .expect("particle alive readback");
        let words: Vec<u32> =
            bytemuck::cast_slice(&slice.get_mapped_range().expect("mapped")).to_vec();
        // `Counters::alive` is the fifth word of each.
        let counts = words.as_chunks::<8>().0.iter().map(|c| c[4]).collect();
        readback.unmap();
        counts
    }
}
