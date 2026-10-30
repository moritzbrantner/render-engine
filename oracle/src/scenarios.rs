//! Parity scenarios. Each one is a `render_engine::scene::Scene`; the Bevy side translates it
//! field by field (see `bevy_side.rs`).

use render_engine::glam::{Quat, Vec2, Vec3};
use render_engine::scene::{
    Ambient, Camera, Material, MeshInstance, Scene, Shape, Sun, SunShadows,
};

pub struct Scenario {
    pub name: &'static str,
    /// Capability the scenario covers (an issue/doc label, printed by `--list`).
    pub capability: &'static str,
    pub width: u32,
    pub height: u32,
    pub scene: Scene,
}

pub fn all() -> Vec<Scenario> {
    vec![sun_shadows()]
}

fn rgb(r: f32, g: f32, b: f32) -> Vec3 {
    Vec3::new(r, g, b)
}

/// Ground plane, PBR boxes (dielectric, metallic, rotated), a shadow-casting sun and ambient.
fn sun_shadows() -> Scenario {
    let dielectric = |base_color, perceptual_roughness| Material {
        base_color,
        perceptual_roughness,
        ..Material::default()
    };
    let meshes = vec![
        MeshInstance::new(
            Shape::Plane(Vec2::splat(24.0)),
            Vec3::ZERO,
            dielectric(rgb(0.22, 0.21, 0.19), 0.9),
        ),
        MeshInstance::new(
            Shape::Cuboid(Vec3::ONE),
            Vec3::new(-2.2, 0.5, 0.6),
            dielectric(rgb(0.70, 0.08, 0.06), 0.45),
        ),
        MeshInstance::new(
            Shape::Cuboid(Vec3::splat(1.4)),
            Vec3::new(1.4, 0.7, -0.4),
            Material {
                base_color: rgb(1.0, 0.71, 0.29),
                metallic: 1.0,
                perceptual_roughness: 0.3,
                ..Material::default()
            },
        ),
        MeshInstance::new(
            Shape::Cuboid(Vec3::new(0.8, 2.6, 0.8)),
            Vec3::new(-0.4, 1.3, -2.6),
            dielectric(rgb(0.10, 0.25, 0.65), 0.6),
        )
        .rotated(Quat::from_rotation_y(30f32.to_radians())),
        MeshInstance::new(
            Shape::Cuboid(Vec3::new(2.0, 0.3, 0.6)),
            Vec3::new(0.2, 0.15, 2.0),
            dielectric(rgb(0.85, 0.85, 0.82), 0.2),
        )
        .rotated(Quat::from_rotation_y(-20f32.to_radians())),
    ];
    Scenario {
        name: "sun_shadows",
        capability: "sun+shadow+ambient",
        width: 1280,
        height: 720,
        scene: Scene {
            camera: Camera {
                eye: Vec3::new(6.5, 5.0, 8.0),
                target: Vec3::new(0.0, 0.6, -0.5),
                fov_y: 45f32.to_radians(),
                near: 0.1,
                ev100: 10.5,
            },
            clear_color: rgb(0.30, 0.48, 0.75),
            sun: Some(Sun {
                direction: Vec3::new(0.75, -1.0, 0.25).normalize(),
                color: rgb(1.0, 0.95, 0.88),
                illuminance: 10_000.0,
                shadows: Some(SunShadows::default()),
            }),
            ambient: Ambient {
                color: rgb(0.75, 0.85, 1.0),
                brightness: 1500.0,
            },
            meshes,
        },
    }
}
