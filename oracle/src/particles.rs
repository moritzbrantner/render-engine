//! Particle parity scenarios (technical decisions D9) and how they are judged.
//!
//! A particle scenario is a camera over a plain background plus emitters, simulated at a fixed
//! timestep (`DT`, exactly 1/64 s so ages and times add up exactly in `f32`) from a fixed seed.
//! It is captured at fixed simulation times. Each capture has a [`Tier`]:
//!
//! - [`Tier::Exact`]: both sides replay hanabi's per-particle random stream, so the image must
//!   match under the lighting FLIP gate (D3). Only valid while no particle slot has been recycled.
//! - [`Tier::Statistical`]: compared by [`ImageStats`] under the scenario's [`StatThresholds`].
//!
//! On every frame both sides must report the same alive-particle count (the CPU spawner is
//! deterministic and the lifetimes are fixed, see [`Population`]).
//!
//! The engine side lands with #13 (emitters); until then every particle scenario reports the
//! engine as a known gap.

use render_engine::glam::{Vec3, Vec4};
use render_engine::scene::{Camera, ShadowFilter, Tonemapping};

/// Fixed simulation timestep: exactly representable, so `k * DT` ages are exact in `f32`.
pub const DT: f32 = 1.0 / 64.0;

/// The statistics gate of one scenario: the largest [`StatDistance`] it accepts. Set from
/// `oracle --calibrate` to about 1.5× the largest distance between the scenario and the same
/// scenario under eight other seeds (seed noise), and checked to reject gravity ×1.25.
#[derive(Clone, Copy, Debug)]
pub struct StatThresholds {
    /// Relative difference of alpha coverage (share of pixels a particle touches).
    pub coverage: f32,
    /// Total-variation distance between the spatial histograms.
    pub spatial: f32,
    /// Total-variation distance between the colour histograms.
    pub colour: f32,
}

/// Spatial histogram grid (cells across and down the image).
const GRID: (u32, u32) = (16, 9);
/// A channel differing from the empty plate by more than this (8-bit steps) counts as covered.
const COVERAGE_STEP: u8 = 2;

pub struct ParticleScenario {
    pub name: &'static str,
    pub capability: &'static str,
    pub width: u32,
    pub height: u32,
    pub camera: Camera,
    /// Linear clear colour (the only background).
    pub clear_color: Vec3,
    /// Effect-level PRNG seed (hanabi `EffectAsset::prng_seed`).
    pub seed: u32,
    pub emitters: Vec<Emitter>,
    pub captures: Vec<Capture>,
    /// Gate of the statistical captures.
    pub stat_max: StatThresholds,
}

#[derive(Clone, Copy, Debug)]
pub struct Capture {
    /// Simulation frames since the emitters started (frame 1 is the first spawn).
    pub frame: u32,
    pub tier: Tier,
}

impl Capture {
    pub fn time(self) -> f32 {
        self.frame as f32 * DT
    }

    /// File stem suffix, e.g. `t0500` for 0.5 s.
    pub fn label(self) -> String {
        format!("t{:04}", (self.time() * 1000.0).round() as u32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Identical random stream on both sides: FLIP gate.
    Exact,
    /// Particle slots have been recycled: statistics gate.
    Statistical,
}

impl Tier {
    pub fn id(self) -> &'static str {
        match self {
            Tier::Exact => "exact",
            Tier::Statistical => "statistical",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Spawn {
    /// `count` particles on the first frame (hanabi `SpawnerSettings::once`).
    Burst { count: u32 },
    /// A steady rate (hanabi `SpawnerSettings::rate`); fractions carry over between frames.
    Rate { per_second: f32 },
}

#[derive(Clone, Copy, Debug)]
pub enum Velocity {
    /// Radially out of the spawn sphere's centre at `speed` (hanabi `SetVelocitySphereModifier`).
    Radial { speed: f32 },
    /// `base + (2 r - 1) * spread` per axis, `r` uniform in [0, 1)³ from the particle's stream.
    Jitter { base: Vec3, spread: Vec3 },
}

#[derive(Clone, Debug)]
pub struct Emitter {
    pub name: &'static str,
    pub capacity: u32,
    pub spawn: Spawn,
    /// Spawn position: uniform in a ball (hanabi `SetPositionSphereModifier`, volume).
    pub position: Vec3,
    pub radius: f32,
    pub velocity: Velocity,
    /// Seconds; fixed per emitter, a multiple of `DT` keeps the death frame exact.
    pub lifetime: f32,
    /// Constant acceleration, m/s².
    pub gravity: Vec3,
    /// World-space quad size over normalised age (keys `(age / lifetime, size)`).
    pub size: Vec<(f32, f32)>,
    /// Linear HDR colour and alpha over normalised age.
    pub color: Vec<(f32, Vec4)>,
}

pub fn all() -> Vec<ParticleScenario> {
    vec![fountain()]
}

/// `particles_fountain`: a one-off burst of sparks plus a continuous fountain, both under
/// gravity, shrinking and fading over their life, additively blended over a dark background.
/// Captured at 0.5 s (no particle has died yet: exact tier) and 2.0 s (the burst has died, the
/// fountain recycles slots: statistical tier).
fn fountain() -> ParticleScenario {
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    ParticleScenario {
        name: "particles_fountain",
        capability: "burst+rate+gravity+size/colour over life",
        width: 1280,
        height: 720,
        camera: Camera {
            eye: Vec3::new(0.0, 2.0, 9.0),
            target: Vec3::new(0.0, 1.8, 0.0),
            fov_y: 45f32.to_radians(),
            near: 0.1,
            ev100: 0.0,
            shadow_filter: ShadowFilter::Hardware2x2,
            tonemapping: Tonemapping::TonyMcMapface,
        },
        clear_color: Vec3::new(0.012, 0.014, 0.022),
        seed: 0x5eed_0001,
        emitters: vec![
            Emitter {
                name: "burst",
                capacity: 128,
                spawn: Spawn::Burst { count: 96 },
                position: Vec3::new(1.6, 2.8, 0.0),
                radius: 0.1,
                velocity: Velocity::Radial { speed: 3.0 },
                lifetime: 80.0 * DT,
                gravity,
                size: vec![(0.0, 0.10), (1.0, 0.02)],
                color: vec![
                    (0.0, Vec4::new(6.0, 3.0, 0.8, 1.0)),
                    (0.5, Vec4::new(3.0, 0.6, 0.1, 0.8)),
                    (1.0, Vec4::new(1.0, 0.1, 0.0, 0.0)),
                ],
            },
            Emitter {
                name: "fountain",
                capacity: 256,
                spawn: Spawn::Rate { per_second: 96.0 },
                position: Vec3::new(-1.2, 0.2, 0.0),
                radius: 0.05,
                velocity: Velocity::Jitter {
                    base: Vec3::new(0.4, 6.0, 0.0),
                    spread: Vec3::new(0.8, 0.4, 0.8),
                },
                lifetime: 96.0 * DT,
                gravity,
                size: vec![(0.0, 0.12), (1.0, 0.04)],
                color: vec![
                    (0.0, Vec4::new(1.5, 2.5, 4.0, 1.0)),
                    (0.6, Vec4::new(0.3, 1.2, 2.0, 0.7)),
                    (1.0, Vec4::new(0.1, 0.3, 0.8, 0.0)),
                ],
            },
        ],
        captures: vec![
            Capture {
                frame: 32,
                tier: Tier::Exact,
            },
            Capture {
                frame: 128,
                tier: Tier::Statistical,
            },
        ],
        // Seed noise at 2.0 s over eight seeds (RTX 3060 Ti): coverage ≤ 0.107, spatial ≤ 0.104,
        // colour ≤ 0.083. Gravity ×1.25 scores spatial 0.203.
        stat_max: StatThresholds {
            coverage: 0.16,
            spatial: 0.15,
            colour: 0.12,
        },
    }
}

/// One frame of an emitter's population.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Population {
    pub alive: u32,
    /// Mean and maximum age of the alive particles, seconds.
    pub mean_age: f32,
    pub max_age: f32,
}

/// Alive-particle count and ages after each frame, from the particles spawned per frame
/// (`spawns[k]` on frame `k + 1`). hanabi spawns (init pass) before it ages and reaps (update
/// pass) in the same frame, so a particle spawned on frame `s` has age `(k - s + 1) * DT` after
/// frame `k` and is alive while that is below its lifetime. The free slots left after the
/// previous frame cap the spawns.
pub fn population(spawns: &[u32], lifetime: f32, capacity: u32) -> Vec<Population> {
    let mut cohorts: Vec<(usize, u32)> = Vec::new();
    let mut out = Vec::with_capacity(spawns.len());
    for (k, &n) in spawns.iter().enumerate() {
        let alive: u32 = cohorts.iter().map(|c| c.1).sum();
        let n = n.min(capacity - alive);
        if n > 0 {
            cohorts.push((k, n));
        }
        let age = |s: usize| (k - s + 1) as f32 * DT;
        cohorts.retain(|&(s, _)| age(s) < lifetime);
        let alive: u32 = cohorts.iter().map(|c| c.1).sum();
        let (sum, max) = cohorts.iter().fold((0.0, 0.0f32), |(sum, max), &(s, n)| {
            (sum + age(s) * n as f32, max.max(age(s)))
        });
        out.push(Population {
            alive,
            mean_age: if alive > 0 { sum / alive as f32 } else { 0.0 },
            max_age: max,
        });
    }
    out
}

/// Image statistics of one capture against its empty plate (the same view before any spawn).
#[derive(Clone, Debug)]
pub struct ImageStats {
    /// Share of pixels a particle changed (alpha coverage).
    pub coverage: f32,
    /// Where the particles are: per grid cell, the summed largest channel change, normalised.
    pub spatial: Vec<f32>,
    /// What colour they are: covered pixels over a 4×4×4 RGB grid, normalised.
    pub colour: Vec<f32>,
}

impl ImageStats {
    pub fn new(width: u32, height: u32, rgba: &[u8], plate: &[u8]) -> ImageStats {
        let mut spatial = vec![0.0f32; (GRID.0 * GRID.1) as usize];
        let mut colour = vec![0.0f32; 64];
        let mut covered = 0u32;
        for (i, (p, b)) in rgba
            .as_chunks::<4>()
            .0
            .iter()
            .zip(plate.as_chunks::<4>().0)
            .enumerate()
        {
            let change = (0..3).map(|c| p[c].abs_diff(b[c])).max().unwrap_or(0);
            if change <= COVERAGE_STEP {
                continue;
            }
            covered += 1;
            let (x, y) = (i as u32 % width, i as u32 / width);
            let cell = (y * GRID.1 / height) * GRID.0 + x * GRID.0 / width;
            spatial[cell as usize] += change as f32;
            let bin = (p[0] as usize >> 6) * 16 + (p[1] as usize >> 6) * 4 + (p[2] as usize >> 6);
            colour[bin] += 1.0;
        }
        let normalise = |v: &mut Vec<f32>| {
            let sum: f32 = v.iter().sum();
            if sum > 0.0 {
                v.iter_mut().for_each(|x| *x /= sum);
            }
        };
        normalise(&mut spatial);
        normalise(&mut colour);
        ImageStats {
            coverage: covered as f32 / (width * height) as f32,
            spatial,
            colour,
        }
    }
}

/// How far apart two captures' statistics are.
#[derive(Clone, Copy, Debug)]
pub struct StatDistance {
    /// `|a - b| / max(a, b)` of the coverages.
    pub coverage: f32,
    /// Total-variation distances (half the L1 distance; 0 = same, 1 = disjoint).
    pub spatial: f32,
    pub colour: f32,
}

impl StatDistance {
    pub fn new(a: &ImageStats, b: &ImageStats) -> StatDistance {
        let tv =
            |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| (a - b).abs()).sum::<f32>() / 2.0;
        let max = a.coverage.max(b.coverage);
        StatDistance {
            coverage: if max > 0.0 {
                (a.coverage - b.coverage).abs() / max
            } else {
                0.0
            },
            spatial: tv(&a.spatial, &b.spatial),
            colour: tv(&a.colour, &b.colour),
        }
    }

    pub fn passes(self, max: StatThresholds) -> bool {
        self.coverage <= max.coverage && self.spatial <= max.spatial && self.colour <= max.colour
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn population_ages_and_reaps_like_hanabi() {
        // One particle a frame, living 3 frames: ages DT, 2 DT, then reaped at 3 DT.
        let p = population(&[1, 1, 1, 1, 1], 3.0 * DT, 100);
        let alive: Vec<u32> = p.iter().map(|p| p.alive).collect();
        assert_eq!(alive, [1, 2, 2, 2, 2]);
        assert_eq!(p[4].max_age, 2.0 * DT);
        assert_eq!(p[4].mean_age, 1.5 * DT);
        // A burst is capped by capacity and dies all at once.
        let p = population(&[10, 0, 0], 2.0 * DT, 8);
        assert_eq!(p.iter().map(|p| p.alive).collect::<Vec<_>>(), [8, 0, 0]);
    }

    #[test]
    fn statistics_separate_moved_particles() {
        let (w, h) = (64u32, 36u32);
        let plate = vec![10u8; (w * h * 4) as usize];
        let blob = |x0: u32| {
            let mut img = plate.clone();
            for y in 10..20 {
                for x in x0..x0 + 10 {
                    let i = ((y * w + x) * 4) as usize;
                    img[i..i + 3].copy_from_slice(&[250, 120, 20]);
                }
            }
            img
        };
        let max = all()[0].stat_max;
        let a = ImageStats::new(w, h, &blob(4), &plate);
        let same = StatDistance::new(&a, &ImageStats::new(w, h, &blob(4), &plate));
        assert!(same.passes(max) && same.spatial == 0.0 && same.colour == 0.0);
        let moved = StatDistance::new(&a, &ImageStats::new(w, h, &blob(44), &plate));
        assert_eq!(moved.coverage, 0.0);
        assert!(moved.spatial > max.spatial && !moved.passes(max));
    }
}
