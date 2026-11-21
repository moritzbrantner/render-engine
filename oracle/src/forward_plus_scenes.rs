//! The Forward+ scenes of issue #72 (`docs/forward-plus-experiment.md`): an outdoor yard and a
//! dungeon with N local lights. Shared by the bench (`forward_plus_bench`), the `many_lights` gate
//! (`scenarios`) and the browser smoke (`webgpu-smoke`, compiled by path), so this module needs
//! nothing but the scenario content.

use crate::{
    content::{self, Geometry, Material, MeshInstance},
    scenarios::{Feature, Scenario},
};
use render_engine::{
    glam::{Quat, Vec2, Vec3},
    material::AlphaMode,
    mesh::Shape,
    scene::{
        Ambient, Camera, LightShadows, PointLight, Scene, ShadowFilter, ShadowMapSizes, SpotLight,
        Sun, SunShadows, Tonemapping,
    },
};

/// A deterministic value in `0..1` for `(index, salt)`: a cheap hash, so the scenes need no
/// random state and Bevy's child process rebuilds the same content.
pub(crate) fn rand01(index: usize, salt: u32) -> f32 {
    let mut x = (index as u32)
        .wrapping_mul(2_654_435_761)
        .wrapping_add(salt.wrapping_mul(40_503));
    x ^= x >> 15;
    x = x.wrapping_mul(2_246_822_519);
    x ^= x >> 13;
    x = x.wrapping_mul(3_266_489_917);
    x ^= x >> 16;
    x as f32 / u32::MAX as f32
}

fn hue(h: f32) -> Vec3 {
    let c = |n: f32| {
        let k = (n + h * 6.0) % 6.0;
        1.0 - (k.min(4.0 - k).clamp(0.0, 1.0))
    };
    Vec3::new(c(5.0), c(3.0), c(1.0))
}

fn material(base_color: Vec3, perceptual_roughness: f32) -> Material {
    Material {
        base_color,
        perceptual_roughness,
        ..Material::default()
    }
}

/// An outdoor town-edge scene in the Greyhaven style: sun with cascaded shadows, masked foliage
/// (the CC0 `Bush_Common` prop), a blended water plane and `lights` local lights (point lights,
/// every fourth a spot light) strung along the path.
fn outdoor(name: &'static str, lights: usize) -> Scenario {
    let mut assets = content::Assets::default();
    let bush = assets.load_gltf("props/Bush_Common.gltf");
    let mut objects = vec![MeshInstance::new(
        Shape::Plane(Vec2::splat(200.0)),
        Vec3::ZERO,
        material(Vec3::new(0.18, 0.28, 0.12), 0.9),
    )];
    // Foliage in two bands either side of the path, on a jittered grid.
    for i in 0..160 {
        let side = if i % 2 == 0 { -1.0 } else { 1.0 };
        let x = side * (6.0 + 22.0 * rand01(i, 1));
        let z = 22.0 - 2.5 * (i / 2) as f32 + 3.0 * rand01(i, 2);
        let yaw = 360.0 * rand01(i, 3);
        let scale = 0.8 + 0.8 * rand01(i, 4);
        for p in &bush {
            objects.push(MeshInstance {
                geometry: Geometry::Mesh(p.mesh),
                translation: Vec3::new(x, 0.0, z),
                rotation: Quat::from_rotation_y(yaw.to_radians()),
                scale: Vec3::splat(scale),
                material: p.material,
                skin: None,
            });
        }
    }
    // A few huts.
    for i in 0..8 {
        let x = if i % 2 == 0 { -16.0 } else { 16.0 };
        let z = 14.0 - 12.0 * (i / 2) as f32;
        objects.push(MeshInstance::new(
            Shape::Cuboid(Vec3::new(5.0, 3.5, 6.0)),
            Vec3::new(x, 1.75, z),
            material(Vec3::new(0.46, 0.40, 0.32), 0.85),
        ));
    }
    // The water: blended, shading the local lights like any other fragment.
    objects.push(MeshInstance::new(
        Shape::Plane(Vec2::new(16.0, 90.0)),
        Vec3::new(0.0, 0.05, -20.0),
        Material {
            base_color: Vec3::new(0.08, 0.2, 0.28),
            alpha: 0.6,
            perceptual_roughness: 0.1,
            alpha_mode: AlphaMode::Blend,
            ..Material::default()
        },
    ));
    let (mut points, mut spots) = (Vec::new(), Vec::new());
    for i in 0..lights {
        let side = if i % 2 == 0 { -1.0 } else { 1.0 };
        let position = Vec3::new(
            side * (3.0 + 3.0 * rand01(i, 5)),
            1.5 + 2.5 * rand01(i, 6),
            26.0 - 80.0 * i as f32 / lights.max(1) as f32,
        );
        let color = hue(0.05 + 0.08 * rand01(i, 7));
        if i % 4 == 3 {
            spots.push(SpotLight {
                position,
                direction: Vec3::new(-side * 0.3, -1.0, 0.2 * (rand01(i, 8) - 0.5)),
                color,
                intensity: 30_000.0,
                range: 16.0,
                radius: 0.0,
                inner_angle: 15f32.to_radians(),
                outer_angle: 35f32.to_radians(),
                shadows: None,
            });
        } else {
            points.push(PointLight {
                position,
                color,
                intensity: 25_000.0,
                range: 12.0,
                radius: 0.05,
                shadows: None,
            });
        }
    }
    let scene = Scene {
        camera: Camera {
            eye: Vec3::new(1.5, 3.0, 34.0),
            target: Vec3::new(0.0, 1.5, 0.0),
            fov_y: 50f32.to_radians(),
            near: 0.1,
            ev100: 10.5,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::TonyMcMapface,
        },
        clear_color: Vec3::new(0.55, 0.65, 0.78),
        sun: Some(Sun {
            direction: Vec3::new(0.4, -1.0, -0.5).normalize(),
            color: Vec3::new(1.0, 0.92, 0.82),
            illuminance: 6_000.0,
            shadows: Some(SunShadows::default()),
        }),
        moon: None,
        sky: None,
        ambient: Ambient {
            color: Vec3::new(0.6, 0.7, 0.9),
            brightness: 400.0,
        },
        point_lights: points,
        spot_lights: spots,
        shadow_maps: ShadowMapSizes::default(),
        environment: None,
        fog: None,
        overlay: Vec::new(),
    };
    Scenario {
        name,
        capability: "Forward+ evaluation: outdoor",
        feature: Feature::PointSpot,
        width: 1280,
        height: 720,
        scene,
        objects,
        assets,
    }
}

/// A dense underground hall lit only by `lights` overlapping point and spot lights, some casting
/// shadows. Intensity falls with the count so the frame stays comparably exposed.
pub(crate) fn dungeon(
    name: &'static str,
    lights: usize,
    casters: usize,
    width: u32,
    height: u32,
) -> Scenario {
    let stone = material(Vec3::new(0.50, 0.47, 0.42), 0.8);
    let mut objects = vec![
        MeshInstance::new(
            Shape::Plane(Vec2::splat(80.0)),
            Vec3::ZERO,
            material(Vec3::new(0.42, 0.40, 0.36), 0.9),
        ),
        MeshInstance::new(
            Shape::Cuboid(Vec3::new(80.0, 0.5, 80.0)),
            Vec3::new(0.0, 7.25, 0.0),
            stone,
        ),
    ];
    for (x, z, sx, sz) in [
        (0.0, -40.0, 80.0, 1.0),
        (0.0, 40.0, 80.0, 1.0),
        (-40.0, 0.0, 1.0, 80.0),
        (40.0, 0.0, 1.0, 80.0),
    ] {
        objects.push(MeshInstance::new(
            Shape::Cuboid(Vec3::new(sx, 7.0, sz)),
            Vec3::new(x, 3.5, z),
            stone,
        ));
    }
    // Pillars on a grid, then scattered crates and spheres.
    for ix in -5..=5 {
        for iz in -5..=5 {
            objects.push(MeshInstance::new(
                Shape::Cuboid(Vec3::new(1.2, 7.0, 1.2)),
                Vec3::new(ix as f32 * 7.0, 3.5, iz as f32 * 7.0),
                stone,
            ));
        }
    }
    for i in 0..400 {
        let x = -36.0 + 72.0 * rand01(i, 21);
        let z = -36.0 + 72.0 * rand01(i, 22);
        let color = Vec3::new(
            0.2 + 0.5 * rand01(i, 23),
            0.2 + 0.4 * rand01(i, 24),
            0.2 + 0.3 * rand01(i, 25),
        );
        let mat = material(color, 0.3 + 0.6 * rand01(i, 26));
        objects.push(if i % 2 == 0 {
            MeshInstance::new(
                Shape::Cuboid(Vec3::splat(0.5 + 0.5 * rand01(i, 27))),
                Vec3::new(x, 0.45, z),
                mat,
            )
        } else {
            MeshInstance::new(
                Shape::Sphere {
                    radius: 0.35,
                    sectors: 16,
                    stacks: 8,
                },
                Vec3::new(x, 0.35, z),
                mat,
            )
        });
    }
    let scale = (256.0 / lights.max(1) as f32).clamp(0.25, 4.0);
    let (mut points, mut spots) = (Vec::new(), Vec::new());
    for i in 0..lights {
        let position = Vec3::new(
            -36.0 + 72.0 * rand01(i, 31),
            1.0 + 5.0 * rand01(i, 32),
            -36.0 + 72.0 * rand01(i, 33),
        );
        let color = hue(rand01(i, 34)).lerp(Vec3::ONE, 0.25);
        let range = 8.0 + 6.0 * rand01(i, 35);
        if i % 4 == 3 {
            spots.push(SpotLight {
                position,
                direction: Vec3::new(rand01(i, 36) - 0.5, -1.0, rand01(i, 37) - 0.5),
                color,
                intensity: 6_000.0 * scale,
                range,
                radius: 0.0,
                inner_angle: 20f32.to_radians(),
                outer_angle: 40f32.to_radians(),
                shadows: (spots.len() < casters / 2).then_some(LightShadows::SPOT),
            });
        } else {
            points.push(PointLight {
                position,
                color,
                intensity: 3_000.0 * scale,
                range,
                radius: 0.05,
                shadows: (points.len() < casters - casters / 2).then_some(LightShadows::POINT),
            });
        }
    }
    let scene = Scene {
        camera: Camera {
            // Between two pillar columns (the pillars stand on multiples of 7 m).
            eye: Vec3::new(3.5, 2.2, 38.0),
            target: Vec3::new(3.5, 2.0, 0.0),
            fov_y: 50f32.to_radians(),
            near: 0.1,
            ev100: 5.0,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::TonyMcMapface,
        },
        clear_color: Vec3::splat(0.01),
        sun: None,
        moon: None,
        sky: None,
        ambient: Ambient {
            color: Vec3::new(0.5, 0.45, 0.7),
            brightness: 1.5,
        },
        point_lights: points,
        spot_lights: spots,
        shadow_maps: ShadowMapSizes::default(),
        environment: None,
        fog: None,
        overlay: Vec::new(),
    };
    Scenario {
        name,
        capability: "Forward+ evaluation: dense lights",
        feature: Feature::PointSpot,
        width,
        height,
        scene,
        objects,
        assets: Default::default(),
    }
}

/// The workload matrix: both scene types with low and high light counts, the light-count ladder
/// for the crossover, shadow casters, and a 1080p row.
pub(crate) fn scenes() -> Vec<Scenario> {
    vec![
        // The pure cost of the path: no local lights to assign or loop over.
        outdoor("outdoor_0", 0),
        outdoor("outdoor_low", 4),
        outdoor("outdoor_high", 64),
        dungeon("dungeon_4", 4, 0, 1280, 720),
        dungeon("dungeon_16", 16, 0, 1280, 720),
        dungeon("dungeon_64", 64, 0, 1280, 720),
        dungeon("dungeon_256", 256, 0, 1280, 720),
        dungeon("dungeon_1024", 1024, 0, 1280, 720),
        dungeon("dungeon_64_shadowed", 64, 4, 1280, 720),
        dungeon("dungeon_256_1080p", 256, 0, 1920, 1080),
    ]
}

/// The scenes compared directly with Bevy (found by name in the Bevy child process).
pub fn bevy_scenes() -> Vec<Scenario> {
    vec![
        outdoor("outdoor_low", 4),
        outdoor("outdoor_high", 64),
        dungeon("dungeon_16", 16, 0, 1280, 720),
        dungeon("dungeon_256", 256, 0, 1280, 720),
    ]
}
