//! Renders tiny scenes on whatever adapter is available (CI: lavapipe via
//! `WGPU_FORCE_FALLBACK_ADAPTER=1`) through the consumer API and checks the frames' basic shape.

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use render_engine::{
    headless::{Frame, Headless},
    material::*,
    mesh::*,
    renderer::Viewport,
    scene::*,
    world::*,
};

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
        point_lights: Vec::new(),
        spot_lights: Vec::new(),
        shadow_maps: ShadowMapSizes::default(),
        environment: None,
        fog: None,
    }
}

fn pixel(frame: &Frame, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * frame.width + x) * 4) as usize;
    frame.rgba[i..i + 4].try_into().unwrap()
}

fn at(mesh: MeshId, material: MaterialId, translation: Vec3) -> Instance {
    Instance {
        mesh,
        material,
        world_from_local: Mat4::from_translation(translation),
    }
}

#[test]
fn sky_ground_and_sun_shadow() {
    let gpu = Headless::new().expect("a wgpu adapter (install mesa-vulkan-drivers for lavapipe)");
    let mut world = World::new(gpu.renderer());
    let ground = world.add_mesh(&MeshData::shape(Shape::Plane(Vec2::splat(20.0))));
    let slab = world.add_mesh(&MeshData::shape(Shape::Cuboid(Vec3::new(2.0, 0.2, 2.0))));
    let white = world.add_material(&StandardMaterial::default());
    let instances = [
        at(ground, white, Vec3::ZERO),
        at(slab, white, Vec3::new(0.0, 2.0, 0.0)),
    ];
    let (w, h) = (160, 90);
    let mut viewport = Viewport::new(gpu.renderer(), w, h);
    for tonemapping in [
        Tonemapping::AcesFitted,
        Tonemapping::TonyMcMapface,
        Tonemapping::AgX,
    ] {
        let lit = gpu
            .render(
                &mut viewport,
                &world,
                &scene(false, tonemapping),
                &instances,
            )
            .unwrap();
        let shadowed = gpu
            .render(&mut viewport, &world, &scene(true, tonemapping), &instances)
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
    let gpu = Headless::new().expect("a wgpu adapter");
    let mut world = World::new(gpu.renderer());
    let sphere = world.add_mesh(&MeshData::shape(Shape::Sphere {
        radius: 1.0,
        sectors: 32,
        stacks: 16,
    }));
    let white = world.add_material(&StandardMaterial::default());
    let instances = [at(sphere, white, Vec3::ZERO)];
    let (w, h) = (64, 64);
    let mut viewport = Viewport::new(gpu.renderer(), w, h);
    let dark = gpu.render(&mut viewport, &world, &s, &instances).unwrap();
    s.environment = Some(EnvironmentMap {
        diffuse,
        specular,
        intensity: 2000.0,
    });
    let lit = gpu.render(&mut viewport, &world, &s, &instances).unwrap();
    let (unlit, bright) = (pixel(&dark, w / 2, h / 2)[1], pixel(&lit, w / 2, h / 2)[1]);
    assert!(unlit < 5 && bright > 60, "sphere centre {unlit} → {bright}");
}

/// A quad textured half transparent (alpha-masked away) and half red, seen from above, then
/// moved and recoloured without uploading the mesh or texture again.
#[test]
fn masked_texture_cuts_out_and_instances_move_per_frame() {
    let gpu = Headless::new().expect("a wgpu adapter");
    let mut world = World::new(gpu.renderer());
    let mut quad = MeshData::shape(Shape::Plane(Vec2::splat(2.0)));
    quad.uvs = quad
        .positions
        .iter()
        .map(|p| Vec2::new(p.x, p.z) * 0.5 + 0.5)
        .collect();
    quad.generate_tangents().unwrap();
    let quad_data = quad;
    let quad = world.add_mesh(&quad_data);
    // Left texel transparent, right texel opaque white; nearest filtering keeps the edge sharp.
    let texture = world.add_texture(&TextureData {
        width: 2,
        height: 1,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        levels: vec![vec![255, 255, 255, 0, 255, 255, 255, 255]],
        sampler: Sampler {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Sampler::default()
        },
    });
    let mut red = StandardMaterial {
        base_color: Vec4::new(1.0, 0.0, 0.0, 1.0),
        base_color_texture: Some(texture),
        alpha_mode: AlphaMode::Mask(0.5),
        double_sided: true,
        ..StandardMaterial::default()
    };
    let material = world.add_material(&red);

    let mut s = scene(false, Tonemapping::AcesFitted);
    s.camera.eye = Vec3::new(0.0, 5.0, 0.001);
    s.camera.fov_y = 30f32.to_radians();
    s.camera.ev100 = 12.5;
    let (w, h) = (64, 64);
    let mut viewport = Viewport::new(gpu.renderer(), w, h);
    let mut instances = [at(quad, material, Vec3::ZERO)];
    let frame = gpu.render(&mut viewport, &world, &s, &instances).unwrap();
    let (left, right) = (
        pixel(&frame, w / 2 - 8, h / 2),
        pixel(&frame, w / 2 + 8, h / 2),
    );
    assert!(
        left[2] > 150 && left[0] < 100,
        "cut-out half shows the sky: {left:?}"
    );
    assert!(
        right[0] > right[1] + 60 && right[0] > right[2] + 60,
        "opaque half is red: {right:?}"
    );

    // Next frame: the quad flips over (mirrored about x) and turns green.
    instances[0].world_from_local = Mat4::from_scale_rotation_translation(
        Vec3::new(-1.0, 1.0, 1.0),
        Quat::IDENTITY,
        Vec3::ZERO,
    );
    red.base_color = Vec4::new(0.0, 1.0, 0.0, 1.0);
    world.set_material(material, &red);
    // Same data through the in-place path CPU-skinned meshes take every frame.
    world.set_mesh(quad, &quad_data);
    let frame = gpu.render(&mut viewport, &world, &s, &instances).unwrap();
    let (left, right) = (
        pixel(&frame, w / 2 - 8, h / 2),
        pixel(&frame, w / 2 + 8, h / 2),
    );
    assert!(
        left[1] > left[0] + 60 && left[1] > left[2] + 60,
        "mirrored opaque half is green: {left:?}"
    );
    assert!(
        right[2] > 150 && right[1] < 100,
        "mirrored cut-out half: {right:?}"
    );
}
