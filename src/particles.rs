//! Particle emitters, simulated the way `bevy_hanabi` simulates them (technical decisions D9).
//!
//! An [`Emitter`] describes an effect: a [`Spawner`], the [`InitModifier`]s that set each new
//! particle's attributes (applied in order, drawing from the particle's random stream in that
//! order), a constant acceleration, the render modifiers (size and colour over the particle's
//! life, a round mask, a texture with an optional flipbook, the quad's orientation) and how it
//! blends. The consumer adds emitters to the [`World`](crate::world::World), advances them with
//! [`World::step_particles`](crate::world::World::step_particles) at its simulation rate (it
//! records the step's compute passes), and the renderer draws every live particle in the main
//! pass as a quad facing the camera.
//!
//! Particles are unlit and unfogged, and their edges are hard against the scene's depth: that is
//! what bevy_hanabi renders (technical decisions D9, "Lighting, fog and soft particles").
//!
//! The simulation replays hanabi's. On the CPU: the spawner's per-frame count
//! (`EffectSpawner::tick`) and the effect's per-frame seed chain
//! (`StdRng::seed_from_u64(previous).random()`, starting from [`Emitter::seed`] on the first step
//! after it is added). On the GPU (`particle_sim.wgsl`): the per-particle stream
//! (`pcg_hash(slot ^ seed)`, then `frand`/`frand3`/`frand4`), the slot allocator (a dead-slot
//! stack in slot order) and the update pass (age, reap, `v += a·dt`, `p += v·dt`). A particle
//! spawned in a step is aged and moved in that step too.
//!
//! An emitter takes at most [`MAX_INIT_MODIFIERS`] init modifiers and gradients of at most
//! [`MAX_GRADIENT_KEYS`] keys.

use crate::world::TextureId;
use glam::{Mat4, UVec2, Vec2, Vec3, Vec4};
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

/// How a particle's colour reaches the HDR target. Neither writes depth; both test it. Like
/// Bevy's transparent phase, the renderer draws all emitters back to front by their transform's
/// position, whatever their blending; within an emitter, particles are drawn in alive-list
/// order, as hanabi draws them, which the GPU's atomics decide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleBlend {
    /// `dst += rgb · a` (hanabi `AlphaMode::Add`); order-independent among additive emitters.
    Add,
    /// `dst = rgb · a + dst · (1 − a)` (hanabi `AlphaMode::Blend`).
    Alpha,
}

/// How a particle's quad is oriented (hanabi `OrientMode`, without a rotation). The quad's x
/// axis carries the size's x, its y axis the size's y.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Orient {
    /// Parallel to the camera's image plane: x along the camera's right, y along its up
    /// (`ParallelCameraDepthPlane`).
    #[default]
    CameraPlane,
    /// x along the velocity, y across it and the view ray, so the quad turns about its velocity
    /// to face the camera (`AlongVelocity`): a spark's streak.
    AlongVelocity,
}

/// How a particle texture's sample combines with the particle's colour (hanabi
/// `ImageSampleMapping`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextureMapping {
    /// `colour · sample`.
    #[default]
    Modulate,
    /// `colour.rgb · sample.rgb`, alpha kept.
    ModulateRgb,
    /// `colour.a · sample.r`, colour kept.
    ModulateOpacityFromR,
}

/// A texture over each particle's quad (hanabi `ParticleTextureModifier`), applied before the
/// round mask. The texture comes from the same [`World`](crate::world::World) and is sampled
/// with its own sampler and implicit derivatives, as hanabi samples it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParticleTexture {
    pub texture: TextureId,
    pub mapping: TextureMapping,
    pub flipbook: Option<Flipbook>,
}

/// A sprite sheet of `grid.x × grid.y` cells, read row by row from the top left (hanabi
/// `FlipbookModifier`). The sprite index runs once over the particle's life: `i32(age /
/// lifetime · frames)` after each update step, as hanabi's `SetAttributeModifier(SPRITE_INDEX)`
/// with that expression computes it in its update pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flipbook {
    pub grid: UVec2,
    pub frames: u32,
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
    /// Quad size over life (x, y), metres; replaces the size attribute.
    pub size_over_life: Option<Gradient<Vec2>>,
    pub color_over_life: Option<ColorOverLife>,
    pub orient: Orient,
    pub texture: Option<ParticleTexture>,
    /// Round mask (hanabi `RoundModifier`): 0 square, 1 a disc; the quad's corners are cut where
    /// `|2u − 1|^(2/r) + |2v − 1|^(2/r) > 1`.
    pub roundness: f32,
    pub blend: ParticleBlend,
    /// Whether the spawner runs; an inactive emitter keeps simulating its particles.
    pub active: bool,
}

/// Most init modifiers per emitter.
pub const MAX_INIT_MODIFIERS: usize = 8;
/// Most keys per size or colour gradient.
pub const MAX_GRADIENT_KEYS: usize = 8;

/// hanabi's `pcg_hash` (`vfx_common.wesl`).
pub fn pcg_hash(input: u32) -> u32 {
    let state = input.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// hanabi's `to_float01`: 23 random mantissa bits as a float in [0, 1) (the GPU side is in
/// `particle_sim.wgsl`).
#[cfg(test)]
fn to_float01(u: u32) -> f32 {
    f32::from_bits((u & 0x007f_ffff) | 0x3f80_0000) - 1.0
}

/// The effect seed of the next step after `seed` (hanabi `compile_effects`).
pub fn next_seed(seed: u32) -> u32 {
    rand::rngs::StdRng::seed_from_u64(seed as u64).random()
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

/// An emitter's CPU side: hanabi's `EffectSpawner` and the effect's seed chain. Its particles
/// live in the world's GPU pool (`particle_sim`).
pub(crate) struct EmitterState {
    pub desc: Emitter,
    pub transform: Mat4,
    spawner: SpawnerState,
    /// The effect seed the next step uses.
    seed: u32,
    /// Particles the spawner asked for in the last step.
    pub spawn_count: u32,
    /// First slot of the emitter's range in the pool, once a step has allocated it.
    pub base: Option<u32>,
}

impl EmitterState {
    pub fn new(desc: &Emitter) -> Self {
        Self {
            seed: desc.seed,
            desc: desc.clone(),
            transform: Mat4::IDENTITY,
            spawner: SpawnerState::default(),
            spawn_count: 0,
            base: None,
        }
    }

    /// The CPU half of a step of `dt` seconds: the spawn count and the seed the GPU step uses,
    /// after which the seed chain moves on (active or not, as hanabi's).
    pub fn tick(&mut self, dt: f32) -> (u32, u32) {
        self.spawn_count = if self.desc.active {
            self.spawner.tick(&self.desc.spawner, dt)
        } else {
            0
        };
        let seed = self.seed;
        self.seed = next_seed(seed);
        (self.spawn_count, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 64.0;

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
    fn gradients_interpolate_and_clamp() {
        let g = Gradient::new(vec![(0.0, 0.1f32), (1.0, 0.02)]);
        assert_eq!(g.sample(-1.0), Some(0.1));
        assert!((g.sample(0.5).unwrap() - 0.06).abs() < 1e-6);
        assert_eq!(g.sample(2.0), Some(0.02));
        assert_eq!(Gradient::<f32>::default().sample(0.5), None);
    }
}
