//! Renders a tiny scene on whatever adapter is available (CI: lavapipe via
//! `WGPU_FORCE_FALLBACK_ADAPTER=1`) and checks the frame's basic shape.

use glam::{Vec2, Vec3};
use render_engine::{headless::Headless, scene::*};

fn scene(shadows: bool, tonemapping: Tonemapping) -> Scene {
    Scene {
        camera: Camera {
            eye: Vec3::new(0.0, 4.0, 6.0),
            target: Vec3::ZERO,
            fov_y: 45f32.to_radians(),
            near: 0.1,
            ev100: 10.5,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping,
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
        environment: None,
        fog: None,
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
    for tonemapping in [
        Tonemapping::AcesFitted,
        Tonemapping::TonyMcMapface,
        Tonemapping::AgX,
    ] {
        let lit = gpu
            .render(&gpu.prepare(&scene(false, tonemapping), w, h))
            .unwrap();
        let shadowed = gpu
            .render(&gpu.prepare(&scene(true, tonemapping), w, h))
            .unwrap();
        assert_eq!(lit.rgba.len(), (w * h * 4) as usize);

        // Top row is sky: the blue clear colour, tone mapped with the HDR frame.
        let [r, g, b, a] = pixel(&lit, 2, 0);
        assert!(
            b > 150 && r < b / 2 && g < b && a == 255,
            "{tonemapping:?} sky {r} {g} {b} {a}"
        );
        // Ground under the floating slab: lit without shadows, dark with them (no ambient).
        let (x, y) = (w / 2, h / 2 + 3);
        let bright = pixel(&lit, x, y)[0];
        let dark = pixel(&shadowed, x, y)[0];
        assert!(bright > 100, "{tonemapping:?} lit ground {bright}");
        assert!(dark < 10, "{tonemapping:?} shadowed ground {dark}");
    }
}

#[test]
fn environment_map_lights_a_sphere_without_sun_or_ambient() {
    let cube = |bytes: &[u8]| std::sync::Arc::new(Cubemap::from_ktx2(bytes).unwrap());
    let diffuse = cube(include_bytes!("../assets/ibl/tiergarten_diffuse.ktx2"));
    let specular = cube(include_bytes!("../assets/ibl/tiergarten_specular.ktx2"));
    assert_eq!((diffuse.size, diffuse.levels.len()), (32, 1));
    assert_eq!((specular.size, specular.levels.len()), (256, 9));

    let mut s = scene(false, Tonemapping::TonyMcMapface);
    s.sun = None;
    s.meshes = vec![MeshInstance::new(
        Shape::Sphere {
            radius: 1.0,
            sectors: 32,
            stacks: 16,
        },
        Vec3::ZERO,
        Material::default(),
    )];
    let gpu = Headless::new().expect("a wgpu adapter");
    let (w, h) = (64, 64);
    let dark = gpu.render(&gpu.prepare(&s, w, h)).unwrap();
    s.environment = Some(EnvironmentMap {
        diffuse,
        specular,
        intensity: 2000.0,
    });
    let lit = gpu.render(&gpu.prepare(&s, w, h)).unwrap();
    let (unlit, bright) = (pixel(&dark, w / 2, h / 2)[1], pixel(&lit, w / 2, h / 2)[1]);
    assert!(unlit < 5 && bright > 60, "sphere centre {unlit} → {bright}");
}
