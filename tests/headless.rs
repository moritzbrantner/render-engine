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
        moon: None,
        sky: None,
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
        skin: None,
    }
}

/// A box seen from straight above, skinned to one joint on the GPU and posed on the CPU: each
/// frame it lands where its new pose puts it (outside its bind-pose bounds, so culling has to
/// follow the joints), with only the palette or the positions and normals uploaded.
#[test]
fn skinned_and_cpu_posed_meshes_follow_their_poses() {
    let gpu = Headless::new().expect("a wgpu adapter");
    let mut world = World::new(gpu.renderer());
    let mut data = MeshData::shape(Shape::Cuboid(Vec3::splat(0.6)));
    let plain = world.add_mesh(&data);
    data.joints = vec![[1, 0, 0, 0]; data.positions.len()];
    data.weights = vec![Vec4::X; data.positions.len()];
    let skinned = world.add_mesh(&data);
    let white = world.add_material(&StandardMaterial::default());
    let shift = |x: f32| Mat4::from_translation(Vec3::new(x, 0.0, 0.0));
    let skin = world.add_skin(&[Mat4::IDENTITY, shift(1.2)]);

    let mut s = scene(false, Tonemapping::AcesFitted);
    s.camera.eye = Vec3::new(0.0, 5.0, 0.001);
    s.camera.fov_y = 40f32.to_radians();
    let (w, h) = (64, 64);
    let mut viewport = Viewport::new(gpu.renderer(), w, h);
    let lit = |p: [u8; 4]| p[0] > 100 && p[1] > 100;
    // 1.2 m to either side is about 22 pixels from the centre.
    let (left, right) = ((w / 2 - 22, h / 2), (w / 2 + 22, h / 2));

    let instances = [Instance {
        skin: Some(skin),
        ..at(skinned, white, Vec3::ZERO)
    }];
    let frame = gpu.render(&mut viewport, &world, &s, &instances).unwrap();
    assert!(
        lit(pixel(&frame, right.0, right.1)),
        "skinned box moved right"
    );
    assert!(!lit(pixel(&frame, w / 2, h / 2)), "bind pose is empty");

    world.set_skin(skin, &[Mat4::IDENTITY, shift(-1.2)]);
    let frame = gpu.render(&mut viewport, &world, &s, &instances).unwrap();
    assert!(lit(pixel(&frame, left.0, left.1)), "skinned box moved left");
    assert!(!lit(pixel(&frame, right.0, right.1)));

    let moved: Vec<Vec3> = data.positions.iter().map(|p| *p + Vec3::X * 1.2).collect();
    world.set_mesh_positions_normals(plain, &moved, &data.normals);
    let instances = [at(plain, white, Vec3::ZERO)];
    let frame = gpu.render(&mut viewport, &world, &s, &instances).unwrap();
    assert!(
        lit(pixel(&frame, right.0, right.1)),
        "CPU-posed box moved right"
    );
    assert!(!lit(pixel(&frame, w / 2, h / 2)));
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

#[test]
fn sky_follows_the_time_of_day() {
    let gpu = Headless::new().expect("a wgpu adapter");
    let mut world = World::new(gpu.renderer());
    let ground = world.add_mesh(&MeshData::shape(Shape::Plane(Vec2::splat(200.0))));
    let grey = world.add_material(&StandardMaterial::default());
    let instances = [at(ground, grey, Vec3::ZERO)];
    let cycle = DayCycle::default();
    // Looking west, level, from 2 m: the upper half is sky, the lower half ground.
    let at_hour = |hours: f32, ev100: f32| {
        let mut s = scene(true, Tonemapping::TonyMcMapface);
        s.camera.eye = Vec3::new(0.0, 2.0, 0.0);
        s.camera.target = Vec3::new(-10.0, 2.0, 0.0);
        s.camera.ev100 = ev100;
        s.clear_color = Vec3::ZERO;
        s.sun = Some(Sun {
            direction: -cycle.to_sun(hours),
            color: Vec3::ONE,
            illuminance: RAW_SUNLIGHT,
            shadows: Some(SunShadows::default()),
        });
        s.sky = Some(Sky::default());
        s
    };
    let (w, h) = (160, 90);
    let mut viewport = Viewport::new(gpu.renderer(), w, h);
    let noon = gpu
        .render(&mut viewport, &world, &at_hour(12.0, 14.5), &instances)
        .unwrap();
    let sunset = gpu
        .render(
            &mut viewport,
            &world,
            &at_hour(cycle.afternoon_hour(1f32.to_radians()), 11.0),
            &instances,
        )
        .unwrap();
    let top = pixel(&noon, w / 2, 2);
    assert!(top[2] > top[0] + 40, "noon sky is blue: {top:?}");
    let horizon = pixel(&sunset, w / 2, h / 2 - 3);
    assert!(
        horizon[0] > horizon[2] + 40,
        "sunset horizon towards the sun is warm: {horizon:?}"
    );
    let lit_ground = pixel(&noon, w / 2, h - 2);
    assert!(
        lit_ground[0] > 30,
        "the sun and sky light the ground: {lit_ground:?}"
    );

    // Without a sky the same viewport falls back to the clear colour.
    let mut plain = at_hour(12.0, 14.5);
    plain.sky = None;
    plain.clear_color = Vec3::new(0.0, 0.0, 1.0);
    let plain = gpu
        .render(&mut viewport, &world, &plain, &instances)
        .unwrap();
    let top = pixel(&plain, w / 2, 0);
    assert!(top[2] > 150 && top[0] < 50, "clear colour again: {top:?}");
}

/// The sky's environment cubes are filtered at the first frame and again only when the sky
/// changes (render-engine#54): the filter is its own timestamped compute pass.
#[test]
fn sky_environment_is_filtered_only_when_the_sky_changes() {
    let gpu = Headless::new().expect("a wgpu adapter");
    let mut world = World::new(gpu.renderer());
    let ground = world.add_mesh(&MeshData::shape(Shape::Plane(Vec2::splat(200.0))));
    let grey = world.add_material(&StandardMaterial::default());
    let instances = [at(ground, grey, Vec3::ZERO)];
    let cycle = DayCycle::default();
    let at_hour = |hours: f32| {
        let mut s = scene(false, Tonemapping::TonyMcMapface);
        s.sun = Some(Sun {
            direction: -cycle.to_sun(hours),
            color: Vec3::ONE,
            illuminance: RAW_SUNLIGHT,
            shadows: None,
        });
        s.sky = Some(Sky::default());
        s
    };
    let mut viewport = Viewport::new(gpu.renderer(), 32, 32);
    let mut filters = |s: &Scene| {
        gpu.render(&mut viewport, &world, s, &instances).unwrap();
        viewport.timestamp_queries()
    };
    let first = filters(&at_hour(10.0));
    let still = filters(&at_hour(10.0));
    assert_eq!(
        first,
        still + 2,
        "a static sky skips the environment filter"
    );
    for _ in 0..3 {
        assert_eq!(filters(&at_hour(10.0)), still);
    }
    // An hour moves the sun by 15°.
    assert_eq!(filters(&at_hour(11.0)), first, "a moving sun re-filters");
    assert_eq!(filters(&at_hour(11.0)), still);
    let mut albedo = at_hour(11.0);
    albedo.sky.as_mut().unwrap().ground_albedo = Vec3::splat(0.1);
    assert_eq!(filters(&albedo), first, "a new ground albedo re-filters");
}

/// GPU particles: alive counts follow the spawns and fixed lifetimes (a particle spawned in a
/// step is aged in it too), capacity caps a burst, slots are reused, and an emitter in front of
/// the camera lights the frame.
#[test]
fn gpu_particles_spawn_age_die_and_draw() {
    use render_engine::particles::*;
    const DT: f32 = 1.0 / 64.0;
    let emitter = |spawner, lifetime: f32, capacity| Emitter {
        capacity,
        spawner,
        seed: 7,
        init: vec![
            InitModifier::PositionSphere {
                center: Vec3::ZERO,
                radius: 0.5,
                dimension: ShapeDimension::Volume,
            },
            InitModifier::Lifetime(Value::Const(lifetime)),
            InitModifier::Size(Value::Const(0.2)),
            InitModifier::Color(Value::Const(Vec4::new(4.0, 0.0, 0.0, 1.0))),
        ],
        accel: Vec3::ZERO,
        size_over_life: None,
        color_over_life: None,
        orient: Orient::CameraPlane,
        texture: None,
        roundness: 0.0,
        depth_fade: 0.0,
        blend: ParticleBlend::Add,
        lighting: None,
        fog: false,
        active: true,
    };
    let gpu = Headless::new().expect("a wgpu adapter");
    let mut world = World::new(gpu.renderer());
    let rate = world.add_emitter(&emitter(Spawner::rate(64.0), 3.0 * DT, 100));
    let burst = world.add_emitter(&emitter(Spawner::once(10.0), 2.0 * DT, 8));
    let mut s = scene(false, Tonemapping::TonyMcMapface);
    s.sun = None;
    s.clear_color = Vec3::ZERO;
    let (w, h) = (64, 64);
    let mut viewport = Viewport::new(gpu.renderer(), w, h);
    let mut alive = Vec::new();
    for step in 0..5 {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        world.step_particles(&mut encoder, DT, None);
        gpu.queue.submit([encoder.finish()]);
        if step == 0 {
            assert_eq!(world.spawn_count(burst), 10);
            let frame = gpu.render(&mut viewport, &world, &s, &[]).unwrap();
            let lit = frame.rgba.chunks(4).filter(|p| p[0] > 50).count();
            assert!(lit > 20, "particles light {lit} pixels");
        }
        let counts = world.read_alive_particles();
        let of = |id| counts.iter().find(|c| c.0 == id).unwrap().1;
        alive.push((of(rate), of(burst)));
    }
    // Ages DT, 2 DT, then reaped at 3 DT; the burst is capped at 8 and dies after two steps.
    assert_eq!(alive, [(1, 8), (2, 0), (2, 0), (2, 0), (2, 0)]);
}

/// Water can receive light without painting an opaque silhouette below it. The
/// shadow threshold must not discard the visible blended surface in the main pass.
#[test]
fn blended_shadow_cutoff_preserves_visible_water() {
    let gpu = Headless::new().expect("a wgpu adapter");
    let mut world = World::new(gpu.renderer());
    let ground = world.add_mesh(&MeshData::shape(Shape::Plane(Vec2::splat(20.))));
    let slab = world.add_mesh(&MeshData::shape(Shape::Cuboid(Vec3::new(2., 0.2, 2.))));
    let white = world.add_material(&StandardMaterial::default());
    let mut water = StandardMaterial {
        base_color: Vec4::new(0.04, 0.15, 0.6, 0.65),
        alpha_mode: AlphaMode::Blend,
        perceptual_roughness: 0.3,
        ..StandardMaterial::default()
    };
    assert_eq!(water.blend_shadow_alpha_cutoff, 0.05);
    let opaque_shadow = world.add_material(&water);
    water.blend_shadow_alpha_cutoff = 1.;
    let clear_shadow = world.add_material(&water);
    let ground = at(ground, white, Vec3::ZERO);
    let position = Vec3::new(0., 2., 0.);
    let mut scene = scene(true, Tonemapping::AcesFitted);
    scene.sun.as_mut().unwrap().direction = Vec3::new(0.4, -1., 0.25).normalize();
    let (w, h) = (160, 90);
    let mut viewport = Viewport::new(gpu.renderer(), w, h);
    let shadowed = gpu
        .render(
            &mut viewport,
            &world,
            &scene,
            &[ground, at(slab, opaque_shadow, position)],
        )
        .unwrap();
    let clear = gpu
        .render(
            &mut viewport,
            &world,
            &scene,
            &[ground, at(slab, clear_shadow, position)],
        )
        .unwrap();
    let plate = gpu
        .render(&mut viewport, &world, &scene, &[ground])
        .unwrap();
    let cleared = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            let plate = pixel(&plate, x, y);
            let clear = pixel(&clear, x, y);
            let dark = pixel(&shadowed, x, y);
            plate[0] > 100
                && (i16::from(clear[0]) - i16::from(plate[0])).abs() < 4
                && i16::from(dark[0]) + 40 < i16::from(clear[0])
        })
        .count();
    assert!(
        cleared > 20,
        "the visible ground must lose the water shadow: {cleared} pixels"
    );
    let visible = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            let p = pixel(&plate, x, y);
            let c = pixel(&clear, x, y);
            p[0] > 100
                && i16::from(c[0]) + 20 < i16::from(p[0])
                && i16::from(c[2]) > i16::from(c[0]) + 10
        })
        .count();
    assert!(
        visible > 20,
        "the blue blended water must remain visible: {visible} pixels"
    );
}

/// Soft particles (`Emitter::depth_fade`): a quad 0.1 m in front of a wall fades to a tenth with
/// a 1 m fade, one 3 m in front of it and one in front of nothing (depth 0) draw as they do with
/// hard edges, and the hard path is unchanged by a soft emitter elsewhere in the frame.
#[test]
fn soft_particles_fade_in_front_of_geometry_only() {
    use render_engine::particles::*;
    let gpu = Headless::new().expect("a wgpu adapter");
    let mut scene = scene(false, Tonemapping::AcesFitted);
    scene.sun = None;
    scene.clear_color = Vec3::ZERO;
    scene.camera.eye = Vec3::new(0.0, 0.0, 10.0);
    scene.camera.target = Vec3::ZERO;
    let quads = [
        Vec3::new(-2.0, 0.0, 0.1),
        Vec3::new(2.0, 0.0, 3.0),
        Vec3::new(0.0, 3.0, 0.0),
    ];
    let emitter = |at: Vec3, depth_fade| Emitter {
        capacity: 1,
        spawner: Spawner::once(1.0),
        seed: 3,
        init: vec![
            InitModifier::Position(Value::Const(at)),
            InitModifier::Lifetime(Value::Const(100.0)),
            InitModifier::Size(Value::Const(0.6)),
            InitModifier::Color(Value::Const(Vec4::new(0.5, 0.0, 0.0, 1.0))),
        ],
        accel: Vec3::ZERO,
        size_over_life: None,
        color_over_life: None,
        orient: Orient::CameraPlane,
        texture: None,
        roundness: 0.0,
        depth_fade,
        blend: ParticleBlend::Add,
        lighting: None,
        fog: false,
        active: true,
    };
    let (w, h) = (160, 90);
    let render = |fade: [f32; 3]| {
        let mut world = World::new(gpu.renderer());
        let wall = world.add_mesh(&MeshData::shape(Shape::Cuboid(Vec3::new(6.0, 3.0, 0.2))));
        let black = world.add_material(&StandardMaterial {
            base_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            unlit: true,
            ..StandardMaterial::default()
        });
        for (q, f) in quads.iter().zip(fade) {
            world.add_emitter(&emitter(*q, f));
        }
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        world.step_particles(&mut encoder, 1.0 / 64.0, None);
        gpu.queue.submit([encoder.finish()]);
        let mut viewport = Viewport::new(gpu.renderer(), w, h);
        let frame = gpu
            .render(
                &mut viewport,
                &world,
                &scene,
                &[at(wall, black, Vec3::new(0.0, 0.0, -0.1))],
            )
            .unwrap();
        (frame, viewport.pass_labels().contains(&"particles"))
    };
    // The camera looks down -z from z = 10, so a point projects by its distance alone.
    let f = 1.0 / (scene.camera.fov_y / 2.0).tan();
    let centre = |p: Vec3| {
        let d = scene.camera.eye.z - p.z;
        let ndc = Vec2::new(f * h as f32 / w as f32 * p.x / d, f * p.y / d);
        (
            ((ndc.x * 0.5 + 0.5) * w as f32) as u32,
            ((0.5 - ndc.y * 0.5) * h as f32) as u32,
        )
    };
    let red = |frame: &Frame, p: Vec3| {
        let (x, y) = centre(p);
        pixel(frame, x, y)[0]
    };
    let (hard, hard_pass) = render([0.0; 3]);
    let (soft, soft_pass) = render([1.0; 3]);
    let (mixed, _) = render([1.0, 0.0, 0.0]);
    assert!(
        !hard_pass && soft_pass,
        "only a soft emitter adds the particle pass"
    );
    for q in quads {
        assert!(
            red(&hard, q) > 100,
            "hard quad at {q} is drawn: {}",
            red(&hard, q)
        );
    }
    // 0.1 m in front of the wall with a 1 m fade: a tenth of the colour before tone mapping.
    assert!(
        red(&soft, quads[0]) < red(&hard, quads[0]) / 2,
        "near the wall the soft quad fades: {} vs {}",
        red(&soft, quads[0]),
        red(&hard, quads[0])
    );
    assert!(red(&soft, quads[0]) > 0, "it fades, not vanishes");
    // Far from the wall, and in front of the cleared background, nothing changes.
    for q in &quads[1..] {
        assert_eq!(red(&soft, *q), red(&hard, *q), "quad at {q}");
    }
    // Hard emitters drawn in the particle pass look as they do in the main pass.
    assert_eq!(red(&mixed, quads[1]), red(&hard, quads[1]));
    assert_eq!(red(&mixed, quads[2]), red(&hard, quads[2]));
}

/// Lit and fogged particles (`Emitter::lighting`, `Emitter::fog`): a lit quad goes dark without
/// light and follows the exposure, a fogged one takes the fog's colour, and an unlit, unfogged one
/// (hanabi's) ignores all three.
#[test]
fn lit_and_fogged_particles_follow_the_scene() {
    use render_engine::particles::*;
    let gpu = Headless::new().expect("a wgpu adapter");
    let mut base = scene(false, Tonemapping::AcesFitted);
    base.clear_color = Vec3::ZERO;
    base.camera.eye = Vec3::new(0.0, 0.0, 10.0);
    base.camera.target = Vec3::ZERO;
    // From behind the camera onto the quad's front.
    base.sun.as_mut().unwrap().direction = Vec3::new(0.0, -0.2, -1.0).normalize();
    let (w, h) = (64, 36);
    let centre = |scene: &Scene, lighting: Option<ParticleLighting>, fog: bool, depth_fade: f32| {
        let mut world = World::new(gpu.renderer());
        world.add_emitter(&Emitter {
            capacity: 1,
            spawner: Spawner::once(1.0),
            seed: 5,
            init: vec![
                InitModifier::Position(Value::Const(Vec3::ZERO)),
                InitModifier::Lifetime(Value::Const(100.0)),
                InitModifier::Size(Value::Const(3.0)),
                InitModifier::Color(Value::Const(Vec4::new(0.6, 0.6, 0.6, 1.0))),
            ],
            accel: Vec3::ZERO,
            size_over_life: None,
            color_over_life: None,
            orient: Orient::CameraPlane,
            texture: None,
            roundness: 0.0,
            depth_fade,
            lighting,
            fog,
            blend: ParticleBlend::Alpha,
            active: true,
        });
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        world.step_particles(&mut encoder, 1.0 / 64.0, None);
        gpu.queue.submit([encoder.finish()]);
        let mut viewport = Viewport::new(gpu.renderer(), w, h);
        let frame = gpu.render(&mut viewport, &world, scene, &[]).unwrap();
        pixel(&frame, w / 2, h / 2)
    };
    let lit = Some(ParticleLighting::diffuse());
    let mut dark = base.clone();
    dark.sun = None;
    let mut brighter = base.clone();
    brighter.camera.ev100 -= 1.0;
    let mut foggy = base.clone();
    foggy.fog = Some(Fog {
        color: Vec4::new(0.0, 1.0, 0.0, 1.0),
        falloff: FogFalloff::Exponential { density: 1.0 },
        ..Fog::default()
    });

    let unlit = centre(&base, None, false, 0.0);
    for s in [&dark, &brighter, &foggy] {
        assert_eq!(
            centre(s, None, false, 0.0),
            unlit,
            "hanabi's particles ignore the scene"
        );
    }
    let sunlit = centre(&base, lit, false, 0.0);
    assert!(sunlit[0] > 20, "the sun lights the quad: {sunlit:?}");
    assert_eq!(centre(&dark, lit, false, 0.0)[0], 0, "no light, no colour");
    // Soft and lit: nothing behind the quad, so the fade leaves it as it is.
    assert_eq!(centre(&base, lit, false, 1.0), sunlit, "soft, lit quad");
    let exposed = centre(&brighter, lit, false, 0.0);
    assert!(
        exposed[0] > sunlit[0],
        "one stop more exposure brightens it: {exposed:?} vs {sunlit:?}"
    );
    let fogged = centre(&foggy, None, true, 0.0);
    assert!(
        u32::from(fogged[1]) > 2 * u32::from(fogged[0])
            && u32::from(unlit[1]) < 2 * u32::from(unlit[0]),
        "10 m into dense fog the quad takes the fog's colour: {fogged:?}"
    );
}
