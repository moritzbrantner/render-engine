//! Particle emitters, simulated the way `bevy_hanabi` simulates them (technical decisions D9).
//!
//! An [`Emitter`] describes an effect: a [`Spawner`], the [`InitModifier`]s that set each new
//! particle's attributes (applied in order, drawing from the particle's random stream in that
//! order), a constant acceleration, the render modifiers (size and colour over the particle's
//! life, a round mask) and how it blends. The consumer adds emitters to the
//! [`World`](crate::world::World), advances them with
//! [`World::step_particles`](crate::world::World::step_particles) at its simulation rate, and the
//! renderer draws every live particle in the main pass as a camera-facing quad.
//!
//! The simulation replays hanabi's: the spawner's per-frame count (`EffectSpawner::tick`), the
//! effect's per-frame seed chain (`StdRng::seed_from_u64(previous).random()`, starting from
//! [`Emitter::seed`] on the first step after it is added), the per-particle stream
//! (`pcg_hash(slot ^ seed)`, then `frand`/`frand3`), the slot allocator (a dead-slot stack in
//! slot order) and the update pass (age, reap, `v += a·dt`, `p += v·dt`). A particle spawned in
//! a step is aged and moved in that step too.

use glam::{Mat4, Vec3, Vec4};
use rand::{RngExt as _, SeedableRng as _};

/// How many particles an emitter spawns and when (hanabi `SpawnerSettings`, constant values).
///
/// A cycle lasts `period` seconds; `count` particles are spawned spread evenly over its first
/// `spawn_duration` seconds (all at once when that is zero). `cycles` cycles run, 0 = forever.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spawner {
    pub count: f32,
    pub spawn_duration: f32,
    pub period: f32,
    pub cycles: u32,
}

impl Spawner {
    /// `count` particles once, on the first step (hanabi `SpawnerSettings::once`).
    pub fn once(count: f32) -> Self {
        Self {
            count,
            spawn_duration: 0.0,
            period: 0.0,
            cycles: 1,
        }
    }

    /// A steady `per_second` rate; fractions carry over between steps (hanabi
    /// `SpawnerSettings::rate`).
    pub fn rate(per_second: f32) -> Self {
        Self {
            count: per_second,
            spawn_duration: 1.0,
            period: 1.0,
            cycles: 0,
        }
    }

    /// `count` particles at once every `period` seconds, forever (hanabi
    /// `SpawnerSettings::burst`).
    pub fn burst(count: f32, period: f32) -> Self {
        Self {
            count,
            spawn_duration: 0.0,
            period,
            cycles: 0,
        }
    }
}

/// A scalar or vector attribute value: constant, or uniform between `min` and `max` per particle
/// (`min + r · (max − min)`, `r` from the particle's stream: one `frand()` for a scalar, one
/// `frand3()` for a vector, like hanabi's `rand` expressions).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value<T> {
    Const(T),
    Uniform { min: T, max: T },
}

/// Whether a shape spawns in its volume or on its surface (hanabi `ShapeDimension`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeDimension {
    Volume,
    Surface,
}

/// Sets a new particle's attributes, in the order listed. Positions and velocities are in the
/// emitter's space; [`World::set_emitter_transform`](crate::world::World::set_emitter_transform)
/// places new particles in the world (the simulation runs in world space).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InitModifier {
    /// Uniform in a ball or on a sphere (hanabi `SetPositionSphereModifier`). Draws `frand()`
    /// for the radius (volume only), then the azimuth and the height.
    PositionSphere {
        center: Vec3,
        radius: f32,
        dimension: ShapeDimension,
    },
    /// Sets the position (hanabi `SetAttributeModifier(POSITION)`).
    Position(Value<Vec3>),
    /// Radially out of `center` through the particle's position at `speed` (hanabi
    /// `SetVelocitySphereModifier`). Apply after the position.
    VelocitySphere { center: Vec3, speed: Value<f32> },
    /// Sets the velocity (hanabi `SetAttributeModifier(VELOCITY)`).
    Velocity(Value<Vec3>),
    /// Seconds (hanabi `SetAttributeModifier(LIFETIME)`); without one a particle lives 1 s.
    Lifetime(Value<f32>),
    /// Starting age, seconds.
    Age(Value<f32>),
    /// Quad size in metres; [`Emitter::size_over_life`] replaces it.
    Size(Value<f32>),
    /// Linear HDR colour and alpha; [`ColorOverLife`] overwrites or multiplies it.
    Color(Value<Vec4>),
}

/// Key-framed value over a particle's normalised age (`age / lifetime`), linearly interpolated
/// between keys and clamped at both ends (hanabi `Gradient`). Keys are sorted by time.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Gradient<T> {
    pub keys: Vec<(f32, T)>,
}

impl<T: Copy + std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T>> Gradient<T> {
    pub fn new(keys: Vec<(f32, T)>) -> Self {
        Self { keys }
    }

    /// The value at normalised age `t`, as hanabi's generated gradient function computes it.
    pub fn sample(&self, t: f32) -> Option<T> {
        let (&(t0, v0), rest) = self.keys.split_first()?;
        if t <= t0 {
            return Some(v0);
        }
        let (mut ta, mut va) = (t0, v0);
        for &(tb, vb) in rest {
            if t <= tb {
                // WGSL `mix(a, b, s)` = a·(1 − s) + b·s.
                let s = (t - ta) / (tb - ta);
                return Some(va * (1.0 - s) + vb * s);
            }
            (ta, va) = (tb, vb);
        }
        Some(va)
    }
}

/// How [`ColorOverLife`] combines with the particle's colour attribute (hanabi
/// `ColorBlendMode`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorBlend {
    Overwrite,
    Multiply,
    Add,
}

/// Colour over the particle's life (hanabi `ColorOverLifetimeModifier`, all channels).
#[derive(Clone, Debug, PartialEq)]
pub struct ColorOverLife {
    pub gradient: Gradient<Vec4>,
    pub blend: ColorBlend,
}

/// How a particle's colour reaches the HDR target. Neither writes depth; both test it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleBlend {
    /// `dst += rgb · a` (hanabi `AlphaMode::Add`); order-independent.
    Add,
    /// `dst = rgb · a + dst · (1 − a)` (hanabi `AlphaMode::Blend`); the renderer draws these
    /// particles back to front across all blended emitters.
    Alpha,
}

/// An effect: what spawns, how each particle starts, moves and looks.
#[derive(Clone, Debug, PartialEq)]
pub struct Emitter {
    /// Most particles alive at once; spawns beyond it are dropped.
    pub capacity: u32,
    pub spawner: Spawner,
    /// Effect PRNG seed (hanabi `EffectAsset::prng_seed`).
    pub seed: u32,
    pub init: Vec<InitModifier>,
    /// Constant acceleration, m/s² (hanabi `AccelModifier`), e.g. gravity.
    pub accel: Vec3,
    /// Quad size over life, metres; replaces the size attribute.
    pub size_over_life: Option<Gradient<f32>>,
    pub color_over_life: Option<ColorOverLife>,
    /// Round mask (hanabi `RoundModifier`): 0 square, 1 a disc; the quad's corners are cut where
    /// `|2u − 1|^(2/r) + |2v − 1|^(2/r) > 1`.
    pub roundness: f32,
    pub blend: ParticleBlend,
    /// Whether the spawner runs; an inactive emitter keeps simulating its particles.
    pub active: bool,
}

/// An emitter's population after the last step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EmitterStats {
    /// Particles the spawner asked for in the last step (hanabi `EffectSpawner::spawn_count`).
    pub spawn_count: u32,
    /// Particles spawned in the last step (the spawn count capped by free capacity).
    pub spawned: u32,
    pub alive: u32,
}

/// hanabi's `pcg_hash` (`vfx_common.wesl`).
pub fn pcg_hash(input: u32) -> u32 {
    let state = input.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// hanabi's `to_float01`: 23 random mantissa bits as a float in [0, 1).
fn to_float01(u: u32) -> f32 {
    f32::from_bits((u & 0x007f_ffff) | 0x3f80_0000) - 1.0
}

/// The effect seed of the next step after `seed` (hanabi `compile_effects`).
pub fn next_seed(seed: u32) -> u32 {
    rand::rngs::StdRng::seed_from_u64(seed as u64).random()
}

/// One particle's random stream (hanabi's private `seed`).
struct Stream(u32);

impl Stream {
    fn frand(&mut self) -> f32 {
        self.0 = pcg_hash(self.0);
        to_float01(pcg_hash(self.0))
    }

    fn frand3(&mut self) -> Vec3 {
        let mut next = || {
            self.0 = pcg_hash(self.0);
            to_float01(self.0)
        };
        let x = next();
        let y = next();
        let z = next();
        Vec3::new(x, y, z)
    }

    fn frand4(&mut self) -> Vec4 {
        let r0 = pcg_hash(self.0);
        let r1 = pcg_hash(r0);
        let r2 = pcg_hash(r1);
        self.0 = r2;
        Vec4::new(
            to_float01(r0),
            to_float01(((r0 & 0xff00_0000) >> 8) | (r1 & 0x0000_ffff)),
            to_float01(((r1 & 0xffff_0000) >> 8) | (r2 & 0x0000_00ff)),
            to_float01(r2 >> 8),
        )
    }

    fn scalar(&mut self, v: Value<f32>) -> f32 {
        match v {
            Value::Const(c) => c,
            Value::Uniform { min, max } => min + self.frand() * (max - min),
        }
    }

    fn vec3(&mut self, v: Value<Vec3>) -> Vec3 {
        match v {
            Value::Const(c) => c,
            Value::Uniform { min, max } => self.frand3() * (max - min) + min,
        }
    }

    fn vec4(&mut self, v: Value<Vec4>) -> Vec4 {
        match v {
            Value::Const(c) => c,
            Value::Uniform { min, max } => self.frand4() * (max - min) + min,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Particle {
    pub position: Vec3,
    pub velocity: Vec3,
    pub age: f32,
    pub lifetime: f32,
    pub size: f32,
    pub color: Vec4,
}

/// hanabi's `EffectSpawner` cycle state.
#[derive(Clone, Copy, Debug, Default)]
struct SpawnerState {
    cycle_time: f32,
    completed_cycles: u32,
    sampled_period: f32,
    sampled_spawn_duration: f32,
    sampled_count: f32,
    remainder: f32,
}

impl SpawnerState {
    /// hanabi `EffectSpawner::tick`, for constant settings.
    fn tick(&mut self, s: &Spawner, mut dt: f32) -> u32 {
        let forever = s.cycles == 0;
        if !forever && self.completed_cycles >= s.cycles {
            return 0;
        }
        loop {
            if self.sampled_period == 0.0 {
                self.sampled_period = if s.cycles == 1 {
                    s.spawn_duration.max(1e-12)
                } else {
                    s.period
                };
                self.sampled_spawn_duration = s.spawn_duration;
                self.sampled_count = s.count.max(0.0);
            }
            let new_time = self.cycle_time + dt;
            if self.cycle_time <= self.sampled_spawn_duration {
                self.remainder += if self.sampled_spawn_duration < 1e-5f32.max(dt / 100.0) {
                    self.sampled_count
                } else {
                    let ratio = ((new_time.min(self.sampled_spawn_duration) - self.cycle_time)
                        / self.sampled_spawn_duration)
                        .clamp(0.0, 1.0);
                    self.sampled_count * ratio
                };
            }
            self.cycle_time = new_time;
            if self.cycle_time >= self.sampled_period {
                dt = self.cycle_time - self.sampled_period;
                self.cycle_time = 0.0;
                self.completed_cycles += 1;
                self.sampled_period = 0.0;
                if !forever && self.completed_cycles >= s.cycles {
                    break;
                }
            } else {
                break;
            }
        }
        let count = self.remainder.floor();
        self.remainder -= count;
        count as u32
    }
}

/// An emitter's live state: hanabi's particle slots, alive list and dead-slot stack.
pub(crate) struct EmitterState {
    pub desc: Emitter,
    pub transform: Mat4,
    spawner: SpawnerState,
    /// The effect seed the next step uses.
    seed: u32,
    /// One per slot.
    pub particles: Vec<Particle>,
    /// Slots of the alive particles, in hanabi's update order.
    pub alive: Vec<u32>,
    /// `dead[i]` is the slot the `i`-th next spawn takes once `i` particles are alive
    /// (hanabi's `dead_index` rows, starting as `0, 1, 2, …`).
    dead: Vec<u32>,
    pub stats: EmitterStats,
}

impl EmitterState {
    pub fn new(desc: &Emitter) -> Self {
        let n = desc.capacity as usize;
        Self {
            seed: desc.seed,
            desc: desc.clone(),
            transform: Mat4::IDENTITY,
            spawner: SpawnerState::default(),
            particles: vec![Particle::default(); n],
            alive: Vec::with_capacity(n),
            dead: (0..desc.capacity).collect(),
            stats: EmitterStats::default(),
        }
    }

    /// Replaces the description, keeping the particles, the seed chain and the spawner's cycle
    /// (a different capacity starts the population over).
    pub fn set_desc(&mut self, desc: &Emitter) {
        if desc.capacity != self.desc.capacity {
            let (seed, transform) = (self.seed, self.transform);
            *self = Self::new(desc);
            self.seed = seed;
            self.transform = transform;
        } else {
            self.desc = desc.clone();
        }
    }

    /// One simulation step of `dt` seconds: spawn and initialise (hanabi's init pass), then age,
    /// reap and integrate every alive particle (its update pass).
    pub fn step(&mut self, dt: f32) {
        let spawn_count = if self.desc.active {
            self.spawner.tick(&self.desc.spawner, dt)
        } else {
            0
        };
        let alive_before = self.alive.len() as u32;
        let spawned = spawn_count.min(self.desc.capacity - alive_before);
        for i in 0..spawned {
            let slot = self.dead[(alive_before + i) as usize];
            self.particles[slot as usize] = self.init(pcg_hash(slot ^ self.seed));
            self.alive.push(slot);
        }

        let mut next_alive = Vec::with_capacity(self.alive.len());
        for &slot in &self.alive {
            let p = &mut self.particles[slot as usize];
            p.age += dt;
            p.velocity += self.desc.accel * dt;
            p.position += p.velocity * dt;
            if p.age < p.lifetime {
                next_alive.push(slot);
            }
        }
        // Dead slots go back on the stack at the alive count they leave behind, in update order.
        let mut alive_count = self.alive.len();
        for &slot in &self.alive {
            if self.particles[slot as usize].age >= self.particles[slot as usize].lifetime {
                alive_count -= 1;
                self.dead[alive_count] = slot;
            }
        }
        self.alive = next_alive;
        self.stats = EmitterStats {
            spawn_count,
            spawned,
            alive: self.alive.len() as u32,
        };
        self.seed = next_seed(self.seed);
    }

    fn init(&self, seed: u32) -> Particle {
        let mut rng = Stream(seed);
        let mut p = Particle {
            lifetime: 1.0,
            size: 1.0,
            color: Vec4::ONE,
            ..Default::default()
        };
        for m in &self.desc.init {
            match *m {
                InitModifier::PositionSphere {
                    center,
                    radius,
                    dimension,
                } => {
                    let r = match dimension {
                        ShapeDimension::Volume => rng.frand().powf(1.0 / 3.0) * radius,
                        ShapeDimension::Surface => radius,
                    };
                    let theta = rng.frand() * std::f32::consts::TAU;
                    let z = rng.frand() * 2.0 - 1.0;
                    let sinphi = z.acos().sin();
                    let dir = Vec3::new(sinphi * theta.cos(), sinphi * theta.sin(), z);
                    p.position = center + r * dir;
                }
                InitModifier::Position(v) => p.position = rng.vec3(v),
                InitModifier::VelocitySphere { center, speed } => {
                    p.velocity = (p.position - center).normalize() * rng.scalar(speed);
                }
                InitModifier::Velocity(v) => p.velocity = rng.vec3(v),
                InitModifier::Lifetime(v) => p.lifetime = rng.scalar(v),
                InitModifier::Age(v) => p.age = rng.scalar(v),
                InitModifier::Size(v) => p.size = rng.scalar(v),
                InitModifier::Color(v) => p.color = rng.vec4(v),
            }
        }
        p.position = self.transform.transform_point3(p.position);
        p.velocity = self.transform.transform_vector3(p.velocity);
        p
    }

    /// A particle's drawn size and colour (hanabi's render modifiers).
    pub fn appearance(&self, p: &Particle) -> (f32, Vec4) {
        let t = p.age / p.lifetime;
        let size = self
            .desc
            .size_over_life
            .as_ref()
            .and_then(|g| g.sample(t))
            .unwrap_or(p.size);
        let color = match &self.desc.color_over_life {
            Some(c) => match (c.gradient.sample(t), c.blend) {
                (Some(g), ColorBlend::Overwrite) => g,
                (Some(g), ColorBlend::Multiply) => p.color * g,
                (Some(g), ColorBlend::Add) => p.color + g,
                (None, _) => p.color,
            },
            None => p.color,
        };
        (size, color)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 64.0;

    fn emitter(spawner: Spawner, lifetime: f32, capacity: u32) -> Emitter {
        Emitter {
            capacity,
            spawner,
            seed: 7,
            init: vec![InitModifier::Lifetime(Value::Const(lifetime))],
            accel: Vec3::ZERO,
            size_over_life: None,
            color_over_life: None,
            roundness: 0.0,
            blend: ParticleBlend::Add,
            active: true,
        }
    }

    #[test]
    fn pcg_hash_matches_hanabi() {
        // Values of hanabi's WGSL `pcg_hash`, computed independently from its definition.
        assert_eq!(pcg_hash(0), 129_708_002);
        assert_eq!(pcg_hash(1), 2_831_084_092);
        assert_eq!(to_float01(0x3f80_0000), 0.0);
        assert!(to_float01(u32::MAX) < 1.0);
    }

    #[test]
    fn rate_spawner_carries_fractions_and_burst_spawns_once() {
        let mut s = SpawnerState::default();
        let rate = Spawner::rate(96.0);
        let counts: Vec<u32> = (0..6).map(|_| s.tick(&rate, DT)).collect();
        assert_eq!(counts, [1, 2, 1, 2, 1, 2]);
        // 96 per second over 64 steps of 1/64 s: one full cycle spawns exactly 96.
        let mut s = SpawnerState::default();
        assert_eq!((0..64).map(|_| s.tick(&rate, DT)).sum::<u32>(), 96);
        let mut s = SpawnerState::default();
        let once = Spawner::once(96.0);
        let counts: Vec<u32> = (0..3).map(|_| s.tick(&once, DT)).collect();
        assert_eq!(counts, [96, 0, 0]);
    }

    #[test]
    fn particles_die_at_their_lifetime_and_slots_are_reused() {
        let mut e = EmitterState::new(&emitter(Spawner::rate(64.0), 3.0 * DT, 100));
        let alive: Vec<u32> = (0..5)
            .map(|_| {
                e.step(DT);
                e.stats.alive
            })
            .collect();
        // Spawned and aged in the same step: ages DT, 2 DT, then reaped at 3 DT.
        assert_eq!(alive, [1, 2, 2, 2, 2]);
        // Slot 0 died in step 3 and was reused in step 4; slot 1 in step 5.
        assert_eq!(e.alive, [0, 1]);

        let mut e = EmitterState::new(&emitter(Spawner::once(10.0), 2.0 * DT, 8));
        e.step(DT);
        assert_eq!(
            (e.stats.spawn_count, e.stats.spawned, e.stats.alive),
            (10, 8, 8)
        );
        e.step(DT);
        assert_eq!(e.stats.alive, 0);
    }

    #[test]
    fn gradients_interpolate_and_clamp() {
        let g = Gradient::new(vec![(0.0, 0.1f32), (1.0, 0.02)]);
        assert_eq!(g.sample(-1.0), Some(0.1));
        assert!((g.sample(0.5).unwrap() - 0.06).abs() < 1e-6);
        assert_eq!(g.sample(2.0), Some(0.02));
        assert_eq!(Gradient::<f32>::default().sample(0.5), None);
    }

    #[test]
    fn spawns_stay_in_their_sphere() {
        let mut desc = emitter(Spawner::once(64.0), 1.0, 64);
        desc.init.insert(
            0,
            InitModifier::PositionSphere {
                center: Vec3::new(1.0, 2.0, 3.0),
                radius: 0.5,
                dimension: ShapeDimension::Volume,
            },
        );
        let mut e = EmitterState::new(&desc);
        e.step(DT);
        for &slot in &e.alive {
            let d = e.particles[slot as usize]
                .position
                .distance(Vec3::new(1.0, 2.0, 3.0));
            assert!(d <= 0.5 + 1e-5);
        }
    }
}
