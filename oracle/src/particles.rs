//! Particle parity scenarios (technical decisions D9) and how they are judged.
//!
//! A particle scenario is a scene (a camera over a plain background, or a lit set of objects)
//! plus emitters, simulated at a fixed timestep (`DT`, exactly 1/64 s so ages and times add up
//! exactly in `f32`) from a fixed seed. It is captured at fixed simulation times. Each capture
//! has a [`Tier`]:
//!
//! - [`Tier::Exact`]: both sides replay hanabi's per-particle random stream, so the image must
//!   match under the lighting FLIP gate (D3). Only valid while no particle slot has been recycled.
//! - [`Tier::Statistical`]: compared by [`ImageStats`] under the scenario's [`StatThresholds`].
//!
//! On every frame both sides must report the same spawn and alive-particle counts (the CPU
//! spawner is deterministic and the lifetimes are fixed, see [`Population`]).
//!
//! The engine side is `render_engine::particles` ([`Emitter::engine`]).

use crate::content::{Assets, Material, MeshInstance};
use render_engine::glam::{UVec2, Vec2, Vec3, Vec4};
use render_engine::material::{Sampler, TextureData};
use render_engine::mesh::Shape;
use render_engine::particles::{
    self, ColorBlend, ColorOverLife, Flipbook, Gradient, InitModifier, Orient, ParticleBlend,
    ParticleTexture, ShapeDimension, Spawner, TextureMapping, Value,
};
use render_engine::scene::{
    Ambient, Camera, PointLight, Scene, ShadowFilter, ShadowMapSizes, Sun, SunShadows, Tonemapping,
};
use render_engine::{wgpu, world::TextureId};

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
    /// Camera, clear colour and lights around the emitters.
    pub scene: Scene,
    /// Opaque geometry the particles meet (depth tested against it).
    pub objects: Vec<MeshInstance>,
    /// Mesh and texture data of the objects and the emitters' textures.
    pub assets: Assets,
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
    /// World-space quad size (x, y) over normalised age (keys `(age / lifetime, size)`).
    pub size: Vec<(f32, Vec2)>,
    /// Linear HDR colour and alpha over normalised age.
    pub color: Vec<(f32, Vec4)>,
    pub orient: Orient,
    pub texture: Option<EmitterTexture>,
    /// hanabi's `RoundModifier` with roundness 1 (a disc, an ellipse when stretched).
    pub round: bool,
    pub blend: ParticleBlend,
    /// The emitter's transform (a translation): where its spawn positions and velocities are
    /// placed, and, for alpha-blended emitters, the position they are sorted by.
    pub translation: Vec3,
}

/// An emitter's texture: [`Assets::textures`] index, mapping and flipbook.
#[derive(Clone, Copy, Debug)]
pub struct EmitterTexture {
    pub texture: usize,
    pub mapping: TextureMapping,
    pub flipbook: Option<Flipbook>,
}

/// Square quad sizes.
fn square(keys: &[(f32, f32)]) -> Vec<(f32, Vec2)> {
    keys.iter().map(|&(t, s)| (t, Vec2::splat(s))).collect()
}

impl Emitter {
    /// The same emitter on render-engine, its texture index mapped to the uploaded `textures`:
    /// hanabi's modifiers in the order `bevy_side::effect_asset` adds them (position, velocity,
    /// age, lifetime), so both draw the particle's random stream in the same order.
    pub fn engine(&self, seed: u32, textures: &[TextureId]) -> particles::Emitter {
        let velocity = match self.velocity {
            Velocity::Radial { speed } => InitModifier::VelocitySphere {
                center: self.position,
                speed: Value::Const(speed),
            },
            Velocity::Jitter { base, spread } => InitModifier::Velocity(Value::Uniform {
                min: base - spread,
                max: base + spread,
            }),
        };
        particles::Emitter {
            capacity: self.capacity,
            spawner: match self.spawn {
                Spawn::Burst { count } => Spawner::once(count as f32),
                Spawn::Rate { per_second } => Spawner::rate(per_second),
            },
            seed,
            init: vec![
                InitModifier::PositionSphere {
                    center: self.position,
                    radius: self.radius,
                    dimension: ShapeDimension::Volume,
                },
                velocity,
                InitModifier::Age(Value::Const(0.0)),
                InitModifier::Lifetime(Value::Const(self.lifetime)),
            ],
            accel: self.gravity,
            size_over_life: Some(Gradient::new(self.size.clone())),
            color_over_life: Some(ColorOverLife {
                gradient: Gradient::new(self.color.clone()),
                blend: ColorBlend::Overwrite,
            }),
            orient: self.orient,
            texture: self.texture.map(|t| ParticleTexture {
                texture: textures[t.texture],
                mapping: t.mapping,
                flipbook: t.flipbook,
            }),
            roundness: if self.round { 1.0 } else { 0.0 },
            blend: self.blend,
            active: false,
        }
    }
}

/// A camera over a clear colour, nothing lit.
fn unlit(camera: Camera, clear_color: Vec3) -> Scene {
    Scene {
        camera,
        clear_color,
        sun: None,
        moon: None,
        sky: None,
        ambient: Ambient {
            color: Vec3::ONE,
            brightness: 0.0,
        },
        point_lights: vec![],
        spot_lights: vec![],
        shadow_maps: ShadowMapSizes::default(),
        environment: None,
        fog: None,
    }
}

pub fn all() -> Vec<ParticleScenario> {
    vec![
        fountain(),
        hundred_k(),
        alpha(),
        particles_textured(),
        smoke_and_sparks(),
    ]
}

/// A 2 × 2 flipbook of soft smoke puffs, 128 × 128 texels each: white, with alpha falling off
/// from the centre, broken up by lumps that move from sprite to sprite. Linear, one level,
/// bilinear and clamped. Generated here (no external asset); both sides upload the same texels.
fn smoke_flipbook() -> TextureData {
    const CELL: u32 = 128;
    let size = 2 * CELL;
    let mut texels = vec![0u8; (size * size * 4) as usize];
    // A few lumps per sprite, (x, y, radius) in cell units around the centre.
    let lumps = |sprite: u32| {
        let a = sprite as f32 * 1.7;
        [
            (0.18 * a.cos(), 0.18 * a.sin(), 0.22),
            (-0.20 * (a + 2.1).cos(), 0.15 * (a + 2.1).sin(), 0.18),
            (0.10 * (a + 4.0).sin(), -0.22 * (a + 4.0).cos(), 0.16),
        ]
    };
    for y in 0..size {
        for x in 0..size {
            let sprite = (y / CELL) * 2 + x / CELL;
            let u = ((x % CELL) as f32 + 0.5) / CELL as f32 - 0.5;
            let v = ((y % CELL) as f32 + 0.5) / CELL as f32 - 0.5;
            let r = (u * u + v * v).sqrt() * 2.0;
            let base = (1.0 - r).clamp(0.0, 1.0).powf(1.5);
            let lump: f32 = lumps(sprite)
                .iter()
                .map(|&(cx, cy, cr)| {
                    let d = ((u - cx).powi(2) + (v - cy).powi(2)).sqrt() / cr;
                    (1.0 - d).clamp(0.0, 1.0)
                })
                .sum();
            let alpha = (base * (0.45 + 0.55 * lump.min(1.0))).clamp(0.0, 1.0);
            let i = ((y * size + x) * 4) as usize;
            texels[i..i + 4].copy_from_slice(&[255, 255, 255, (alpha * 255.0).round() as u8]);
        }
    }
    TextureData {
        width: size,
        height: size,
        format: wgpu::TextureFormat::Rgba8Unorm,
        levels: vec![texels],
        sampler: Sampler {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        },
    }
}

/// The lit set the particle scenarios with geometry share: a ground and two boxes under a low
/// sun with shadows, ambient light and a warm point light near the ground, as in a camp or a
/// forge. The particles stay unlit (hanabi's are).
fn lit_set(camera: Camera) -> (Scene, Vec<MeshInstance>) {
    let dielectric = |base_color, perceptual_roughness| Material {
        base_color,
        perceptual_roughness,
        ..Material::default()
    };
    let objects = vec![
        MeshInstance::new(
            Shape::Plane(Vec2::splat(16.0)),
            Vec3::ZERO,
            dielectric(Vec3::new(0.20, 0.19, 0.17), 0.9),
        ),
        MeshInstance::new(
            Shape::Cuboid(Vec3::splat(1.2)),
            Vec3::new(-1.5, 0.6, -0.4),
            dielectric(Vec3::new(0.45, 0.20, 0.10), 0.7),
        ),
        MeshInstance::new(
            Shape::Cuboid(Vec3::new(0.8, 1.6, 0.8)),
            Vec3::new(1.4, 0.8, -0.2),
            dielectric(Vec3::new(0.30, 0.32, 0.36), 0.5),
        ),
    ];
    let scene = Scene {
        camera,
        clear_color: Vec3::new(0.10, 0.14, 0.22),
        sun: Some(Sun {
            direction: Vec3::new(0.5, -0.7, -0.6).normalize(),
            color: Vec3::new(1.0, 0.85, 0.7),
            illuminance: 2_000.0,
            shadows: Some(SunShadows::single_cascade(20.0)),
        }),
        moon: None,
        sky: None,
        ambient: Ambient {
            color: Vec3::new(0.6, 0.7, 1.0),
            brightness: 200.0,
        },
        point_lights: vec![PointLight {
            position: Vec3::new(0.0, 0.5, 1.2),
            color: Vec3::new(1.0, 0.55, 0.25),
            intensity: 40_000.0,
            range: 20.0,
            radius: 0.0,
            shadows: None,
        }],
        spot_lights: vec![],
        shadow_maps: ShadowMapSizes::default(),
        environment: None,
        fog: None,
    };
    (scene, objects)
}

fn set_camera() -> Camera {
    Camera {
        eye: Vec3::new(0.0, 2.0, 6.5),
        target: Vec3::new(0.0, 0.9, 0.0),
        fov_y: 45f32.to_radians(),
        near: 0.1,
        ev100: 8.0,
        shadow_filter: ShadowFilter::Hardware2x2,
        tonemapping: Tonemapping::TonyMcMapface,
    }
}

/// Sparks thrown up out of a point, falling under gravity onto the ground and the boxes and
/// streaking along their velocity (`Orient::AlongVelocity`, an ellipse stretched along it).
fn sparks(translation: Vec3) -> Emitter {
    Emitter {
        name: "sparks",
        capacity: 512,
        spawn: Spawn::Rate { per_second: 384.0 },
        position: Vec3::ZERO,
        radius: 0.05,
        velocity: Velocity::Jitter {
            base: Vec3::new(0.0, 4.2, 0.0),
            spread: Vec3::new(1.6, 1.0, 1.6),
        },
        lifetime: 48.0 * DT,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        size: vec![(0.0, Vec2::new(0.22, 0.035)), (1.0, Vec2::new(0.10, 0.02))],
        color: vec![
            (0.0, Vec4::new(8.0, 4.0, 1.0, 1.0)),
            (0.6, Vec4::new(4.0, 1.0, 0.2, 0.8)),
            (1.0, Vec4::new(1.0, 0.1, 0.0, 0.0)),
        ],
        orient: Orient::AlongVelocity,
        texture: None,
        round: true,
        blend: ParticleBlend::Add,
        translation,
    }
}

/// `particles_textured`: additive embers rising out of the ground beside a box, textured with
/// the smoke flipbook (each particle runs once through its four sprites), and sparks streaking
/// along their velocity, in the lit set. The quads cut into the ground and the boxes where they
/// meet them (depth tested). Captured at 0.625 s, before the first spark dies (exact tier),
/// and at 2.5 s (statistical tier).
fn particles_textured() -> ParticleScenario {
    let (scene, objects) = lit_set(set_camera());
    let mut assets = Assets::default();
    assets.textures.push(smoke_flipbook());
    ParticleScenario {
        name: "particles_textured",
        capability: "flipbook texture + velocity-aligned sparks over lit geometry",
        width: 1280,
        height: 720,
        scene,
        objects,
        assets,
        seed: 0x5eed_0300,
        emitters: vec![
            Emitter {
                name: "embers",
                capacity: 512,
                spawn: Spawn::Rate { per_second: 160.0 },
                position: Vec3::ZERO,
                radius: 0.25,
                velocity: Velocity::Jitter {
                    base: Vec3::new(0.1, 0.9, 0.0),
                    spread: Vec3::new(0.3, 0.3, 0.3),
                },
                lifetime: 96.0 * DT,
                gravity: Vec3::new(0.0, 0.4, 0.0),
                size: square(&[(0.0, 0.3), (1.0, 0.7)]),
                color: vec![
                    (0.0, Vec4::new(2.5, 1.0, 0.3, 0.9)),
                    (0.5, Vec4::new(1.2, 0.35, 0.08, 0.6)),
                    (1.0, Vec4::new(0.4, 0.05, 0.0, 0.0)),
                ],
                orient: Orient::CameraPlane,
                texture: Some(EmitterTexture {
                    texture: 0,
                    mapping: TextureMapping::Modulate,
                    flipbook: Some(Flipbook {
                        grid: UVec2::new(2, 2),
                        frames: 4,
                    }),
                }),
                round: false,
                blend: ParticleBlend::Add,
                translation: Vec3::new(-0.7, 0.05, 0.3),
            },
            sparks(Vec3::new(0.6, 0.3, 0.5)),
        ],
        captures: vec![
            Capture {
                frame: 40,
                tier: Tier::Exact,
            },
            Capture {
                frame: 160,
                tier: Tier::Statistical,
            },
        ],
        // Seed noise at 2.5 s over eight seeds (RTX 3060 Ti): coverage ≤ 0.061, spatial ≤ 0.068,
        // colour ≤ 0.074. Gravity ×1.25 scores 0.028 / 0.064 / 0.022 there, inside seed noise
        // (the sparks' arcs shift by a few pixels); the exact capture rejects it (FLIP mean
        // 0.0155, p99 0.53), and every other seed scores mean ≥ 0.019 and p99 ≥ 0.57 there.
        stat_max: StatThresholds {
            coverage: 0.092,
            spatial: 0.10,
            colour: 0.11,
        },
    }
}

/// `smoke_and_sparks` (#15): an alpha-blended smoke column (the flipbook, modulated) rising out
/// of the ground past a box's edge in a light wind, and additive sparks in front of it, in the
/// lit set. The smoke cuts into the ground and the box where it meets them; Bevy's transparent
/// phase draws the two effects back to front. Alpha-blended, so both captures (1.0 s and 3.0 s)
/// are statistical (D9).
fn smoke_and_sparks() -> ParticleScenario {
    let (scene, objects) = lit_set(set_camera());
    let mut assets = Assets::default();
    assets.textures.push(smoke_flipbook());
    ParticleScenario {
        name: "smoke_and_sparks",
        capability: "alpha smoke column + additive sparks over lit geometry",
        width: 1280,
        height: 720,
        scene,
        objects,
        assets,
        seed: 0x5eed_0400,
        emitters: vec![
            Emitter {
                name: "smoke",
                capacity: 512,
                spawn: Spawn::Rate { per_second: 128.0 },
                position: Vec3::ZERO,
                radius: 0.35,
                velocity: Velocity::Jitter {
                    base: Vec3::new(0.15, 0.45, 0.0),
                    spread: Vec3::new(0.2, 0.15, 0.2),
                },
                lifetime: 192.0 * DT,
                gravity: Vec3::new(0.12, 0.25, 0.0),
                size: square(&[(0.0, 0.5), (1.0, 1.8)]),
                color: vec![
                    (0.0, Vec4::new(0.30, 0.28, 0.27, 0.45)),
                    (0.6, Vec4::new(0.38, 0.37, 0.36, 0.3)),
                    (1.0, Vec4::new(0.45, 0.45, 0.45, 0.0)),
                ],
                orient: Orient::CameraPlane,
                texture: Some(EmitterTexture {
                    texture: 0,
                    mapping: TextureMapping::Modulate,
                    flipbook: Some(Flipbook {
                        grid: UVec2::new(2, 2),
                        frames: 4,
                    }),
                }),
                round: false,
                blend: ParticleBlend::Alpha,
                translation: Vec3::new(-0.75, 0.1, -0.2),
            },
            sparks(Vec3::new(0.3, 0.3, 0.9)),
        ],
        captures: vec![
            Capture {
                frame: 64,
                tier: Tier::Statistical,
            },
            Capture {
                frame: 192,
                tier: Tier::Statistical,
            },
        ],
        // Seed noise over eight seeds and both captures (RTX 3060 Ti): coverage ≤ 0.061, spatial
        // ≤ 0.090, colour ≤ 0.061; a Bevy rerun alone scores up to 0.0008 (the atomics reorder
        // the smoke). Gravity and buoyancy ×1.25 scores spatial 0.160 at 1.0 s, which fails.
        stat_max: StatThresholds {
            coverage: 0.09,
            spatial: 0.135,
            colour: 0.09,
        },
    }
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
        scene: unlit(
            Camera {
                eye: Vec3::new(0.0, 2.0, 9.0),
                target: Vec3::new(0.0, 1.8, 0.0),
                fov_y: 45f32.to_radians(),
                near: 0.1,
                ev100: 0.0,
                shadow_filter: ShadowFilter::Hardware2x2,
                tonemapping: Tonemapping::TonyMcMapface,
            },
            Vec3::new(0.012, 0.014, 0.022),
        ),
        objects: vec![],
        assets: Assets::default(),
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
                size: square(&[(0.0, 0.10), (1.0, 0.02)]),
                color: vec![
                    (0.0, Vec4::new(6.0, 3.0, 0.8, 1.0)),
                    (0.5, Vec4::new(3.0, 0.6, 0.1, 0.8)),
                    (1.0, Vec4::new(1.0, 0.1, 0.0, 0.0)),
                ],
                orient: Orient::CameraPlane,
                texture: None,
                round: true,
                blend: ParticleBlend::Add,
                translation: Vec3::ZERO,
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
                size: square(&[(0.0, 0.12), (1.0, 0.04)]),
                color: vec![
                    (0.0, Vec4::new(1.5, 2.5, 4.0, 1.0)),
                    (0.6, Vec4::new(0.3, 1.2, 2.0, 0.7)),
                    (1.0, Vec4::new(0.1, 0.3, 0.8, 0.0)),
                ],
                orient: Orient::CameraPlane,
                texture: None,
                round: true,
                blend: ParticleBlend::Add,
                translation: Vec3::ZERO,
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

/// `particles_100k`: a fountain sustaining about 100k particles (51 200 a second, living 2 s)
/// plus a ring of 50 small effects (64 a second each, living 1 s), all additive. Captured at
/// 0.75 s (38 400 + 50 × 48 alive, before the first small effect's particles die: exact tier)
/// and 3.0 s (101 600 + 50 × 63 alive: statistical tier).
fn hundred_k() -> ParticleScenario {
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut emitters = vec![Emitter {
        name: "fountain",
        capacity: 102_400,
        spawn: Spawn::Rate {
            per_second: 51_200.0,
        },
        position: Vec3::new(0.0, -1.0, 0.0),
        radius: 0.1,
        velocity: Velocity::Jitter {
            base: Vec3::new(0.0, 7.0, 0.0),
            spread: Vec3::new(1.6, 0.8, 1.6),
        },
        lifetime: 128.0 * DT,
        gravity,
        size: square(&[(0.0, 0.035), (1.0, 0.015)]),
        color: vec![
            (0.0, Vec4::new(1.2, 0.8, 0.3, 0.35)),
            (0.5, Vec4::new(0.9, 0.35, 0.1, 0.25)),
            (1.0, Vec4::new(0.4, 0.08, 0.02, 0.0)),
        ],
        orient: Orient::CameraPlane,
        texture: None,
        round: true,
        blend: ParticleBlend::Add,
        translation: Vec3::ZERO,
    }];
    for i in 0..50 {
        let angle = i as f32 / 50.0 * std::f32::consts::TAU;
        // A hue around the ring.
        let hue = |offset: f32| 0.5 + 0.5 * (angle + offset).cos();
        let tint = Vec3::new(hue(0.0), hue(2.1), hue(4.2)) * 3.0;
        emitters.push(Emitter {
            name: Box::leak(format!("ring{i:02}").into_boxed_str()),
            capacity: 128,
            spawn: Spawn::Rate { per_second: 64.0 },
            position: Vec3::new(5.0 * angle.cos(), -2.5, 5.0 * angle.sin() - 1.0),
            radius: 0.08,
            velocity: Velocity::Jitter {
                base: Vec3::new(0.0, 1.6, 0.0),
                spread: Vec3::new(0.3, 0.3, 0.3),
            },
            lifetime: 64.0 * DT,
            gravity: Vec3::new(0.0, -1.0, 0.0),
            size: square(&[(0.0, 0.08), (1.0, 0.03)]),
            color: vec![(0.0, tint.extend(1.0)), (1.0, (tint * 0.3).extend(0.0))],
            orient: Orient::CameraPlane,
            texture: None,
            round: true,
            blend: ParticleBlend::Add,
            translation: Vec3::ZERO,
        });
    }
    ParticleScenario {
        name: "particles_100k",
        capability: "100k fountain + 50 effects, GPU simulation",
        width: 1280,
        height: 720,
        scene: unlit(
            Camera {
                eye: Vec3::new(0.0, 1.5, 13.0),
                target: Vec3::new(0.0, -0.5, 0.0),
                fov_y: 45f32.to_radians(),
                near: 0.1,
                ev100: 0.0,
                shadow_filter: ShadowFilter::Hardware2x2,
                tonemapping: Tonemapping::TonyMcMapface,
            },
            Vec3::new(0.010, 0.012, 0.020),
        ),
        objects: vec![],
        assets: Assets::default(),
        seed: 0x5eed_0100,
        emitters,
        captures: vec![
            Capture {
                frame: 48,
                tier: Tier::Exact,
            },
            Capture {
                frame: 192,
                tier: Tier::Statistical,
            },
        ],
        // Seed noise at 3.0 s over eight seeds (RTX 3060 Ti): coverage ≤ 0.0041, spatial ≤ 0.0146,
        // colour ≤ 0.0119. Gravity ×1.25 scores 0.072 / 0.173 / 0.134, which fail on all three.
        // In the exact tier every other seed scores FLIP mean ≥ 0.045 and p99 ≥ 0.85.
        stat_max: StatThresholds {
            coverage: 0.006,
            spatial: 0.022,
            colour: 0.018,
        },
    }
}

/// `particles_alpha`: two alpha-blended smoke columns, one behind the other, so both the
/// back-to-front order of the emitters and the blending itself show. Captured at 1.0 s and
/// 3.0 s, both in the statistical tier: hanabi draws an emitter's particles in its alive-list
/// order, which its GPU atomics decide, and over a blended background that order shows. Before
/// any particle dies the two sides draw the same particles, in a different order inside each
/// emitter: FLIP mean 0.0014, p99 0.055 at 1.0 s (RTX 3060 Ti).
fn alpha() -> ParticleScenario {
    let smoke = |name, translation, color: Vec4| Emitter {
        name,
        capacity: 512,
        spawn: Spawn::Rate { per_second: 256.0 },
        position: Vec3::ZERO,
        radius: 0.3,
        velocity: Velocity::Jitter {
            base: Vec3::new(0.0, 0.2, 0.0),
            spread: Vec3::new(0.3, 0.2, 0.3),
        },
        lifetime: 128.0 * DT,
        gravity: Vec3::new(0.0, 1.2, 0.0),
        size: square(&[(0.0, 0.15), (1.0, 0.35)]),
        color: vec![(0.0, color), (1.0, (color.truncate() * 0.4).extend(0.0))],
        orient: Orient::CameraPlane,
        texture: None,
        round: true,
        blend: ParticleBlend::Alpha,
        translation,
    };
    ParticleScenario {
        name: "particles_alpha",
        capability: "alpha blending, emitters sorted back to front",
        width: 1280,
        height: 720,
        scene: unlit(
            Camera {
                eye: Vec3::new(0.0, 1.4, 6.5),
                target: Vec3::new(0.0, 1.3, 0.0),
                fov_y: 45f32.to_radians(),
                near: 0.1,
                ev100: 0.0,
                shadow_filter: ShadowFilter::Hardware2x2,
                tonemapping: Tonemapping::TonyMcMapface,
            },
            Vec3::new(0.30, 0.32, 0.36),
        ),
        objects: vec![],
        assets: Assets::default(),
        seed: 0x5eed_0200,
        emitters: vec![
            smoke(
                "back",
                Vec3::new(-0.35, 0.0, -1.5),
                Vec4::new(1.2, 0.5, 0.2, 0.4),
            ),
            smoke(
                "front",
                Vec3::new(0.35, 0.0, 1.0),
                Vec4::new(0.15, 0.35, 0.9, 0.4),
            ),
        ],
        captures: vec![
            Capture {
                frame: 64,
                tier: Tier::Statistical,
            },
            Capture {
                frame: 192,
                tier: Tier::Statistical,
            },
        ],
        // Seed noise over eight seeds and both captures (RTX 3060 Ti): coverage ≤ 0.042, spatial
        // ≤ 0.046, colour ≤ 0.064; a Bevy rerun alone scores colour 0.007 (the atomics reorder
        // each emitter). Buoyancy ×1.25 scores coverage 0.087 at 1.0 s, 0.139 / spatial 0.087 at
        // 3.0 s, which fail.
        stat_max: StatThresholds {
            coverage: 0.065,
            spatial: 0.07,
            colour: 0.095,
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
