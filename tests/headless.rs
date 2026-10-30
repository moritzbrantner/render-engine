//! Renders a tiny scene on whatever adapter is available (CI: lavapipe via
//! `WGPU_FORCE_FALLBACK_ADAPTER=1`) and checks the frame's basic shape.

use glam::{Vec2, Vec3};
use render_engine::{headless::Headless, scene::*};

fn scene(shadows: bool) -> Scene {
    Scene {
        camera: Camera {
            eye: Vec3::new(0.0, 4.0, 6.0),
            target: Vec3::ZERO,
            fov_y: 45f32.to_radians(),
            near: 0.1,
            ev100: 10.5,
        },
        clear_color: Vec3::new(0.0, 0.0, 1.0),
        sun: Some(Sun {
            direction: Vec3::new(0.0, -1.0, 0.0),
            color: Vec3::ONE,
            illuminance: 10_000.0,
            shadows: shadows.then(SunShadows::default),
        }),
        ambient: Ambient {
            color: Vec3::ONE,
            brightness: 0.0,
        },
        meshes: vec![
            MeshInstance::new(
                Shape::Plane(Vec2::splat(20.0)),
                Vec3::ZERO,
                Material::default(),
            ),
            MeshInstance::new(
                Shape::Cuboid(Vec3::new(2.0, 0.2, 2.0)),
                Vec3::new(0.0, 2.0, 0.0),
                Material::default(),
            ),
        ],
    }
}

fn pixel(frame: &render_engine::headless::Frame, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * frame.width + x) * 4) as usize;
    frame.rgba[i..i + 4].try_into().unwrap()
}

#[test]
fn sky_ground_and_sun_shadow() {
    let gpu = Headless::new().expect("a wgpu adapter (install mesa-vulkan-drivers for lavapipe)");
    let (w, h) = (160, 90);
    let lit = gpu.render(&gpu.prepare(&scene(false), w, h)).unwrap();
    let shadowed = gpu.render(&gpu.prepare(&scene(true), w, h)).unwrap();
    assert_eq!(lit.rgba.len(), (w * h * 4) as usize);

    // Top row is sky: the clear colour, untouched by tone mapping.
    assert_eq!(pixel(&lit, 2, 0), [0, 0, 255, 255]);
    // Ground under the floating slab: lit without shadows, dark with them (no ambient).
    let (x, y) = (w / 2, h / 2 + 3);
    let bright = pixel(&lit, x, y)[0];
    let dark = pixel(&shadowed, x, y)[0];
    assert!(bright > 100, "lit ground {bright}");
    assert!(dark < 10, "shadowed ground {dark}");
}
