//! One frame of moenarch-games/shooter (`shooter/render-frame/v1` JSON from
//! `shooter-replay promote <replay> --target render --to T -o frame.json`) built through the
//! public API the way the shooter's client builds it: the greybox arena, the other fighters,
//! rockets in flight, and the weapon effects of the last 6 ticks as emissive geometry with their
//! point lights. The engine reads no files (D7); parsing lives here, beside the example and the
//! test that share it.
//!
//! Units: the shooter's centimetres become metres (× 0.01), y up. Light intensities are the
//! shooter client's own (`crates/shooter-client/src/effects.rs`), already in lumens: the
//! engine's `PointLight::intensity` is luminous power, converted to candela as Bevy does
//! (`lm / 4π`). A flash fades as `(1 - age / life)²` over its life, as in the client; the frame
//! gives each effect's age in ticks at 60 Hz. Emissive radiances are in cd/m² (nits), the same
//! values the client uses where it draws the thing, otherwise documented defaults below.

use glam::{Mat3, Mat4, Quat, Vec3, Vec4};
use render_engine::{
    material::{AlphaMode, StandardMaterial},
    mesh::{MeshData, Shape},
    scene::*,
    world::{Instance, MaterialId, MeshId, World},
};
use serde_json::Value;

/// The shooter's simulation rate (`sim_foundation::TICK_HZ`).
pub const TICK_HZ: f32 = 60.0;
const METRES_PER_UNIT: f32 = 0.01;

/// The fixtures checked in with the engine, by name.
pub const FIXTURES: [(&str, &str); 2] = [
    (
        "tick1422",
        include_str!("../../tests/fixtures/shooter/tick1422.json"),
    ),
    (
        "tick1782",
        include_str!("../../tests/fixtures/shooter/tick1782.json"),
    ),
];

/// A frame built into a [`World`]: everything retained, so a frame passes no instance list.
pub struct Frame {
    pub scene: Scene,
    /// How many instances the frame added.
    pub instances: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Weapon {
    EnergyRifle,
    RailGun,
    RocketLauncher,
    Grapple,
    EnergyBlade,
}

impl Weapon {
    fn parse(name: &str) -> Self {
        match name {
            "EnergyRifle" => Self::EnergyRifle,
            "RailGun" => Self::RailGun,
            "RocketLauncher" => Self::RocketLauncher,
            "Grapple" => Self::Grapple,
            "EnergyBlade" => Self::EnergyBlade,
            other => panic!("unknown weapon {other}"),
        }
    }

    /// The client's energy colour (`effects::tint`).
    fn tint(self) -> Vec3 {
        match self {
            Self::EnergyRifle => Vec3::new(0.25, 0.8, 1.0),
            Self::RailGun => Vec3::new(0.75, 0.35, 1.0),
            Self::RocketLauncher => Vec3::new(1.0, 0.55, 0.15),
            Self::Grapple => Vec3::new(0.45, 1.0, 0.55),
            Self::EnergyBlade => Vec3::new(1.0, 0.25, 0.6),
        }
    }

    /// How long a beam's trail motes live (seconds; the client's `TRAIL`).
    fn trail_life(self) -> f32 {
        match self {
            Self::EnergyRifle => 0.12,
            Self::RailGun => 0.7,
            Self::Grapple => 0.25,
            Self::RocketLauncher | Self::EnergyBlade => 0.1,
        }
    }
}

fn vec3(value: &Value) -> Vec3 {
    let n = |i: usize| value[i].as_f64().expect("a number") as f32;
    Vec3::new(n(0), n(1), n(2))
}

fn metres(value: &Value) -> Vec3 {
    vec3(value) * METRES_PER_UNIT
}

/// Where a `sim_foundation::trig` yaw and pitch (65 536 steps a turn, yaw 0 faces +Z and turns
/// toward +X, pitch up) look.
fn looking(yaw: f32, pitch: f32) -> Vec3 {
    let turn = std::f32::consts::TAU / 65_536.0;
    let (sy, cy) = (yaw * turn).sin_cos();
    let (sp, cp) = (pitch * turn).sin_cos();
    Vec3::new(sy * cp, sp, cy * cp)
}

/// The client's held gun: its transform and barrel end for an eye and view direction.
fn gun_pose(eye: Vec3, forward: Vec3) -> (Mat4, Vec3) {
    let forward = forward.normalize_or(Vec3::Z);
    let right = forward.cross(Vec3::Y).normalize_or(Vec3::NEG_X);
    let up = right.cross(forward);
    let center = eye + forward * 0.42 + right * 0.2 - up * 0.2;
    let rotation = Quat::from_mat3(&Mat3::from_cols(up.cross(forward), up, forward));
    (
        Mat4::from_rotation_translation(rotation, center),
        center + forward * 0.15,
    )
}

/// Appends a flat triangle wound counter-clockwise when seen from `normal`'s side.
fn triangle(mesh: &mut MeshData, corners: [Vec3; 3], normal: Vec3) {
    let [a, mut b, mut c] = corners;
    if (b - a).cross(c - a).dot(normal) < 0.0 {
        std::mem::swap(&mut b, &mut c);
    }
    let base = mesh.positions.len() as u32;
    mesh.positions.extend([a, b, c]);
    mesh.normals.extend([normal; 3]);
    mesh.indices.extend([base, base + 1, base + 2]);
}

/// A ramp wedge in its box of half extents `half` (metres), rising toward +Z.
fn wedge(half: Vec3) -> MeshData {
    let mut mesh = MeshData::default();
    let (x, y, z) = (half.x, half.y, half.z);
    let low = |s: f32| Vec3::new(s * x, -y, -z);
    let foot = |s: f32| Vec3::new(s * x, -y, z);
    let top = |s: f32| Vec3::new(s * x, y, z);
    let quad = |m: &mut MeshData, [a, b, c, d]: [Vec3; 4], n: Vec3| {
        triangle(m, [a, b, c], n);
        triangle(m, [a, c, d], n);
    };
    quad(
        &mut mesh,
        [low(-1.0), low(1.0), foot(1.0), foot(-1.0)],
        Vec3::NEG_Y,
    );
    quad(
        &mut mesh,
        [foot(-1.0), foot(1.0), top(1.0), top(-1.0)],
        Vec3::Z,
    );
    quad(
        &mut mesh,
        [low(-1.0), low(1.0), top(1.0), top(-1.0)],
        Vec3::new(0.0, z, -y).normalize(),
    );
    for s in [-1.0, 1.0] {
        triangle(&mut mesh, [low(s), foot(s), top(s)], Vec3::X * s);
    }
    mesh
}

fn matte(rgb: Vec3, roughness: f32) -> StandardMaterial {
    StandardMaterial {
        base_color: rgb.extend(1.0),
        perceptual_roughness: roughness,
        ..StandardMaterial::default()
    }
}

fn glowing(rgb: Vec3, nits: f32) -> StandardMaterial {
    StandardMaterial {
        emissive: rgb * nits,
        ..matte(rgb, 0.4)
    }
}

/// A weapon effect's glow. The client draws these as additive particles, which cast no shadows,
/// so the stand-in geometry is blended (alpha just under 1, below its shadow cutoff).
fn effect_glow(rgb: Vec3, nits: f32) -> StandardMaterial {
    StandardMaterial {
        base_color: rgb.extend(0.99),
        alpha_mode: AlphaMode::Blend,
        blend_shadow_alpha_cutoff: 1.0,
        ..glowing(rgb, nits)
    }
}

/// Builds the frame's geometry into `world` and returns its scene at `aspect` (width / height).
pub fn build(world: &mut World, json: &str, aspect: f32) -> Frame {
    let frame: Value = serde_json::from_str(json).expect("shooter frame JSON");
    assert_eq!(frame["schema"], "shooter/render-frame/v1", "frame schema");
    let mut count = 0;
    let mut add = |world: &mut World, mesh: MeshId, material: MaterialId, at: Mat4| {
        world.add_instance(Instance {
            mesh,
            material,
            world_from_local: at,
            skin: None,
        });
        count += 1;
    };
    let cube = world.add_mesh(&MeshData::shape(Shape::Cuboid(Vec3::ONE)));
    let sphere = world.add_mesh(&MeshData::shape(Shape::Sphere {
        radius: 1.0,
        sectors: 16,
        stacks: 10,
    }));
    let boxed = |center: Vec3, size: Vec3| {
        Mat4::from_scale_rotation_translation(size, Quat::IDENTITY, center)
    };

    // The arena, in the client's greybox materials.
    let arena = &frame["arena"];
    let floor = world.add_material(&matte(Vec3::new(0.20, 0.22, 0.26), 0.8));
    let wall = world.add_material(&matte(Vec3::new(0.55, 0.57, 0.62), 0.7));
    let accent = world.add_material(&matte(Vec3::new(0.85, 0.42, 0.12), 0.5));
    let crate_material = world.add_material(&matte(Vec3::new(0.10, 0.55, 0.60), 0.6));
    let half_size = arena["half_size"].as_f64().expect("half_size") as f32 * METRES_PER_UNIT;
    let floor_mesh = world.add_mesh(&MeshData::shape(Shape::Plane(glam::Vec2::splat(
        2.0 * half_size,
    ))));
    add(world, floor_mesh, floor, Mat4::IDENTITY);
    for piece in arena["pieces"].as_array().expect("pieces") {
        let (center, half) = (metres(&piece["center"]), metres(&piece["half"]));
        match piece["kind"].as_str() {
            Some("block") => {
                let material = if half.y >= 1.9 {
                    wall
                } else if half.x.max(half.z) <= 0.5 {
                    crate_material
                } else {
                    accent
                };
                add(world, cube, material, boxed(center, half * 2.0));
            }
            Some("ramp") => {
                // The wedge rises toward +Z; turn it, swapping its run onto the right axis.
                let swapped = Vec3::new(half.z, half.y, half.x);
                let (angle, local) = match piece["rises"].as_str() {
                    Some("PosZ") => (0.0, half),
                    Some("NegZ") => (std::f32::consts::PI, half),
                    Some("PosX") => (std::f32::consts::FRAC_PI_2, swapped),
                    Some("NegX") => (-std::f32::consts::FRAC_PI_2, swapped),
                    other => panic!("unknown ramp rise {other:?}"),
                };
                let mesh = world.add_mesh(&wedge(local));
                let at = Mat4::from_rotation_translation(Quat::from_rotation_y(angle), center);
                add(world, mesh, accent, at);
            }
            other => panic!("unknown arena piece {other:?}"),
        }
    }
    let pad = world.add_material(&StandardMaterial {
        base_color: Vec4::new(0.1, 0.8, 0.95, 1.0),
        emissive: Vec3::new(400.0, 3_000.0, 3_600.0),
        perceptual_roughness: 0.3,
        ..StandardMaterial::default()
    });
    for jump_pad in arena["jump_pads"].as_array().expect("jump_pads") {
        let half = &jump_pad["half"];
        let size = Vec3::new(
            half[0].as_f64().expect("pad half") as f32,
            1.0,
            half[1].as_f64().expect("pad half") as f32,
        ) * 2.0
            * METRES_PER_UNIT;
        let center = metres(&jump_pad["center"]) + Vec3::Y * (size.y / 2.0);
        add(world, cube, pad, boxed(center, size));
    }

    // The camera at the viewer's eye. The frame's 90° field of view is horizontal.
    let camera = &frame["camera"];
    let viewer = camera["slot"].as_u64();
    let eye = metres(&camera["eye"]);
    let forward = looking(
        camera["yaw"].as_f64().unwrap_or(0.0) as f32,
        camera["pitch"].as_f64().unwrap_or(0.0) as f32,
    );
    let horizontal = (camera["fov_degrees"].as_f64().unwrap_or(90.0) as f32).to_radians();
    let fov_y = 2.0 * ((horizontal / 2.0).tan() / aspect).atan();

    // The other fighters: a 0.6 × 1.8 × 0.6 m body coloured by health, and the held gun.
    let half_extents = metres(&frame["character_half_extents"]);
    let eye_height = eye.y - metres(&frame["fighters"][0]["center"]).y + half_extents.y;
    let gun_mesh = world.add_mesh(&MeshData::shape(Shape::Cuboid(Vec3::new(
        0.045, 0.055, 0.3,
    ))));
    for fighter in frame["fighters"].as_array().expect("fighters") {
        if fighter["slot"].as_u64() == viewer {
            continue;
        }
        let center = metres(&fighter["center"]);
        let health = fighter["health"].as_f64().unwrap_or(200.0) as f32 / 200.0;
        let color =
            Vec3::new(0.75, 0.12, 0.1).lerp(Vec3::new(0.15, 0.55, 0.5), health.clamp(0.0, 1.0));
        let body = world.add_material(&matte(color, 0.6));
        add(world, cube, body, boxed(center, half_extents * 2.0));
        let held = Weapon::parse(fighter["held"].as_str().unwrap_or("EnergyRifle"));
        let looks = looking(
            fighter["yaw"].as_f64().unwrap_or(0.0) as f32,
            fighter["pitch"].as_f64().unwrap_or(0.0) as f32,
        );
        let fighter_eye = center - Vec3::Y * half_extents.y + Vec3::Y * eye_height;
        let gun = world.add_material(&glowing(held.tint(), 1_200.0));
        add(world, gun_mesh, gun, gun_pose(fighter_eye, looks).0);
    }

    // Lights and emissive effects. A flash fades as (1 - age / life)² over its life.
    let mut lights = Vec::new();
    let mut flash = |position: Vec3, color: Vec3, lumens: f32, range: f32, life: f32, age: f32| {
        if age < life {
            let left = 1.0 - age / life;
            lights.push(PointLight {
                position,
                color,
                intensity: lumens * left * left,
                range,
                radius: 0.05,
                shadows: None,
            });
        }
    };
    let rocket = world.add_material(&glowing(Vec3::new(1.0, 0.6, 0.2), 60_000.0));
    for r in frame["rockets"].as_array().expect("rockets") {
        let at = metres(&r["at"]);
        add(world, sphere, rocket, boxed(at, Vec3::splat(0.09)));
        flash(at, Weapon::RocketLauncher.tint(), 120_000.0, 5.0, 1.0, 0.0);
    }
    for effect in frame["effects"].as_array().expect("effects") {
        let age = effect["age_ticks"].as_f64().unwrap_or(0.0) as f32 / TICK_HZ;
        match effect["kind"].as_str() {
            Some("fired") => {
                let weapon = Weapon::parse(effect["weapon"].as_str().unwrap_or_default());
                if weapon == Weapon::EnergyBlade {
                    continue;
                }
                let muzzle = gun_pose(metres(&effect["from"]), vec3(&effect["direction"])).1;
                let lumens = if weapon == Weapon::RailGun {
                    400_000.0
                } else {
                    150_000.0
                };
                flash(muzzle, weapon.tint(), lumens, 6.0, 0.07, age);
                // Default: the muzzle's spark burst as a 4 cm glowing ball while the flash lives.
                if age < 0.07 {
                    let glow = world.add_material(&effect_glow(weapon.tint(), 20_000.0));
                    add(world, sphere, glow, boxed(muzzle, Vec3::splat(0.04)));
                }
            }
            Some("beam") => {
                let weapon = Weapon::parse(effect["weapon"].as_str().unwrap_or_default());
                let (from, end) = (metres(&effect["from"]), metres(&effect["to"]));
                // The frame has no surface normal: the struck side faces back along the beam.
                let back = (from - end).normalize_or(Vec3::Y);
                if weapon == Weapon::EnergyBlade {
                    flash(end + back * 0.15, weapon.tint(), 200_000.0, 5.0, 0.15, age);
                    continue;
                }
                if effect["struck"].as_bool().unwrap_or(false) {
                    flash(end + back * 0.1, weapon.tint(), 80_000.0, 4.0, 0.1, age);
                }
                // Default: the trail of motes as one 3 cm glowing rod from the barrel, fading
                // over the trail's life (4 000 nits fresh, the motes' brightness × 1 000).
                let life = weapon.trail_life();
                if age < life {
                    let muzzle = gun_pose(from, end - from).1;
                    let span = end - muzzle;
                    let rotation = Quat::from_rotation_arc(Vec3::Z, span.normalize_or(Vec3::Z));
                    let at = Mat4::from_scale_rotation_translation(
                        Vec3::new(0.03, 0.03, span.length().max(0.001)),
                        rotation,
                        muzzle + span / 2.0,
                    );
                    let left = 1.0 - age / life;
                    let glow = world.add_material(&effect_glow(weapon.tint(), 4_000.0 * left));
                    add(world, cube, glow, at);
                }
            }
            Some("explosion") => {
                let at = metres(&effect["at"]);
                let color = Vec3::new(1.0, 0.55, 0.2);
                flash(at, color, 3_000_000.0, 14.0, 0.4, age);
                // Default: the fireball as a glowing ball, 0.7 m shrinking to 0.25 m over its
                // 0.45 s mean particle life, the client's fireball colour × 10 000 nits.
                let t = (age / 0.45).min(1.0);
                let glow = world.add_material(&effect_glow(
                    Vec3::new(1.0, 0.5, 0.15),
                    10_000.0 * (1.0 - t),
                ));
                add(
                    world,
                    sphere,
                    glow,
                    boxed(at, Vec3::splat(0.7 + (0.25 - 0.7) * t)),
                );
            }
            // Hits are damage numbers in the client; the beam's struck flash lights them.
            _ => {}
        }
    }

    let scene = Scene {
        camera: Camera {
            eye,
            target: eye + forward,
            fov_y,
            near: 0.05,
            ev100: 13.0,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::TonyMcMapface,
        },
        clear_color: Vec3::new(0.35, 0.45, 0.6),
        sun: Some(Sun {
            direction: Vec3::new(-0.35, -1.0, -0.55).normalize(),
            color: Vec3::new(1.0, 0.96, 0.9),
            illuminance: 20_000.0,
            shadows: Some(SunShadows {
                max_distance: 60.0,
                ..SunShadows::default()
            }),
        }),
        moon: None,
        sky: None,
        point_lights: lights,
        spot_lights: Vec::new(),
        shadow_maps: ShadowMapSizes::default(),
        ambient: Ambient {
            color: Vec3::new(0.85, 0.9, 1.0),
            brightness: 2_500.0,
        },
        environment: None,
        fog: None,
        overlay: Vec::new(),
    };
    Frame {
        scene,
        instances: count,
    }
}
