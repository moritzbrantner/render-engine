//! A synthetic scene of the scale of mmorpg's Greyhaven views on render-engine (#172): about
//! 22 000 retained scenery instances over 400 × 400 m, a few moving units, a sun with four shadow
//! cascades over 160 m, the sky with its environment light and 13 unshadowed point lights, so
//! five views are culled a frame. `KINDS` mesh/material pairs set how many draws that is: an
//! instanced draw per pair and view that sees it.
//!
//! mmorpg's `outpost_frame_time` (2026-11-28, its `starter`, `hub`, `woods`, `vale` and `lake`
//! views) prepares 19 900 to 22 100 instances in 5 views into 170 to 970 main and 2 080 to 2 960
//! shadow draws. Shared by the native example `scale_frame` and the browser's `scale_bench`.

use glam::{Mat4, Quat, Vec3};
use render_engine::{
    material::StandardMaterial,
    mesh::{MeshData, Shape},
    scene::*,
    world::{Instance, MaterialId, MeshId, World},
};

pub const SCENERY: u32 = 22_000;
pub const KINDS: u32 = 900;
pub const UNITS: u32 = 16;

pub struct Built {
    pub scene: Scene,
    kinds: Vec<(MeshId, MaterialId)>,
}

/// Adds the scenery to `world` as retained instances and returns the scene.
pub fn build(world: &mut World) -> Built {
    let kinds: Vec<_> = (0..KINDS)
        .map(|k| {
            let shape = if k % 2 == 0 {
                Shape::Cuboid(Vec3::new(1.0, 2.0 + (k % 5) as f32, 1.0))
            } else {
                Shape::Sphere {
                    radius: 0.5 + (k % 3) as f32 * 0.25,
                    sectors: 12,
                    stacks: 8,
                }
            };
            let mesh = world.add_mesh(&MeshData::shape(shape));
            let material = world.add_material(&StandardMaterial {
                perceptual_roughness: 0.3 + (k % 7) as f32 * 0.1,
                ..StandardMaterial::default()
            });
            (mesh, material)
        })
        .collect();
    // A jittered grid, hashed so neighbours are different kinds.
    let side = (SCENERY as f32).sqrt().ceil() as u32;
    for i in 0..SCENERY {
        let h = i.wrapping_mul(2_654_435_761);
        let (x, z) = (i % side, i / side);
        let pos = Vec3::new(
            (x as f32 + (h % 97) as f32 / 97.0) / side as f32 * 400.0 - 200.0,
            0.0,
            (z as f32 + (h % 89) as f32 / 89.0) / side as f32 * 400.0 - 200.0,
        );
        let (mesh, material) = kinds[(h % KINDS) as usize];
        world.add_instance(Instance {
            mesh,
            material,
            world_from_local: Mat4::from_scale_rotation_translation(
                Vec3::splat(0.8 + (h % 5) as f32 * 0.1),
                Quat::from_rotation_y((h % 360) as f32),
                pos,
            ),
            skin: None,
        });
    }
    let to_sun = Vec3::new(-0.3, 1.0, 0.4).normalize();
    let scene = Scene {
        camera: Camera {
            eye: Vec3::new(5.0, 8.5, 31.0),
            target: Vec3::new(-14.0, 3.0, 4.0),
            fov_y: 50f32.to_radians(),
            near: 0.2,
            ev100: 14.0,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::TonyMcMapface,
        },
        clear_color: Vec3::ZERO,
        sun: Some(Sun {
            direction: -to_sun,
            color: Vec3::ONE,
            illuminance: 130_000.0,
            shadows: Some(SunShadows {
                max_distance: 160.0,
                ..SunShadows::default()
            }),
        }),
        moon: None,
        sky: Some(Sky::default()),
        ambient: Ambient {
            color: Vec3::ONE,
            brightness: 0.0,
        },
        point_lights: (0..13)
            .map(|i| PointLight {
                position: Vec3::new(i as f32 * 6.0 - 36.0, 3.0, 0.0),
                range: 12.0,
                ..PointLight::default()
            })
            .collect(),
        spot_lights: Vec::new(),
        shadow_maps: ShadowMapSizes::default(),
        environment: None,
        fog: None,
        overlay: Vec::new(),
    };
    Built { scene, kinds }
}

impl Built {
    /// The moving units of frame `frame`, listed each frame.
    pub fn units(&self, frame: usize) -> Vec<Instance> {
        (0..UNITS)
            .map(|u| {
                let a = u as f32 + frame as f32 * 0.01;
                let (mesh, material) = self.kinds[u as usize];
                Instance {
                    mesh,
                    material,
                    world_from_local: Mat4::from_translation(Vec3::new(
                        a.cos() * 10.0,
                        0.0,
                        a.sin() * 10.0,
                    )),
                    skin: None,
                }
            })
            .collect()
    }
}
