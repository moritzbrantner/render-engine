//! The opt-in clustered light assignment (`LightAssignment::Clustered`, #72) against the
//! per-fragment reference on one device: byte-identical images through shadowed point and spot
//! lights, masked and blended materials, lights behind the camera or removed, camera cuts,
//! near-plane crossings, forced cluster overflow and degenerate grids. The bench
//! (`oracle --forward-plus-bench`) runs the same comparison at full scale on every oracle
//! scenario.

use glam::{Mat4, Vec2, Vec3, Vec4};
use render_engine::{
    headless::Headless,
    material::{AlphaMode, StandardMaterial},
    mesh::{MeshData, Shape},
    renderer::{ClusterConfig, LightAssignment, Viewport},
    scene::*,
    world::{Instance, World},
};

const SIZE: u32 = 128;

fn scene() -> Scene {
    Scene {
        camera: Camera {
            eye: Vec3::new(0.0, 6.0, 22.0),
            target: Vec3::new(0.0, 1.0, 0.0),
            fov_y: 45f32.to_radians(),
            near: 0.1,
            ev100: 9.7,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::TonyMcMapface,
        },
        clear_color: Vec3::splat(0.1),
        sun: Some(Sun {
            direction: Vec3::new(-0.6, -1.0, -0.4),
            color: Vec3::ONE,
            illuminance: 5_000.0,
            shadows: Some(SunShadows::default()),
        }),
        moon: None,
        sky: None,
        ambient: Ambient {
            color: Vec3::ONE,
            brightness: 200.0,
        },
        point_lights: (0..24)
            .map(|i| PointLight {
                position: Vec3::new(
                    -11.0 + i as f32,
                    1.0 + (i % 3) as f32,
                    10.0 - i as f32 * 1.5,
                ),
                range: 6.0 + (i % 4) as f32,
                intensity: 60_000.0,
                radius: 0.1,
                shadows: (i == 3).then_some(LightShadows::POINT),
                ..PointLight::default()
            })
            .collect(),
        spot_lights: (0..8)
            .map(|i| SpotLight {
                position: Vec3::new(-8.0 + 2.5 * i as f32, 6.0, -2.0),
                direction: Vec3::new(0.1 * i as f32 - 0.3, -1.0, 0.2),
                range: 14.0,
                intensity: 90_000.0,
                outer_angle: 35f32.to_radians(),
                inner_angle: 20f32.to_radians(),
                shadows: (i == 2).then_some(LightShadows::SPOT),
                ..SpotLight::default()
            })
            .collect(),
        shadow_maps: ShadowMapSizes {
            directional: 512,
            point: 256,
        },
        environment: None,
        fog: None,
    }
}

struct Fixture {
    gpu: Headless,
    world: World,
    instances: Vec<Instance>,
}

fn fixture() -> Fixture {
    let gpu = Headless::new().expect("a wgpu adapter");
    let mut world = World::new(gpu.renderer());
    let ground = world.add_mesh(&MeshData::shape(Shape::Plane(Vec2::splat(120.0))));
    let cube = world.add_mesh(&MeshData::shape(Shape::Cuboid(Vec3::ONE)));
    let grey = world.add_material(&StandardMaterial::default());
    let mask = world.add_material(&StandardMaterial {
        base_color: Vec4::new(0.8, 0.5, 0.1, 0.9),
        alpha_mode: AlphaMode::Mask(0.5),
        ..StandardMaterial::default()
    });
    let blend = world.add_material(&StandardMaterial {
        base_color: Vec4::new(0.1, 0.4, 0.9, 0.5),
        alpha_mode: AlphaMode::Blend,
        ..StandardMaterial::default()
    });
    let mut instances = vec![Instance {
        mesh: ground,
        material: grey,
        world_from_local: Mat4::IDENTITY,
        skin: None,
    }];
    for z in -4..4 {
        for x in -5..5 {
            instances.push(Instance {
                mesh: cube,
                material: if (x + z) % 3 == 0 { mask } else { grey },
                world_from_local: Mat4::from_translation(Vec3::new(
                    x as f32 * 2.6,
                    0.5,
                    z as f32 * 2.6,
                )),
                skin: None,
            });
        }
    }
    instances.push(Instance {
        mesh: cube,
        material: blend,
        world_from_local: Mat4::from_translation(Vec3::new(1.0, 3.0, 13.0)),
        skin: None,
    });
    Fixture {
        gpu,
        world,
        instances,
    }
}

impl Fixture {
    fn viewport(&self, assignment: LightAssignment) -> Viewport {
        let mut v = Viewport::new(self.gpu.renderer(), SIZE, SIZE);
        v.set_light_assignment(self.gpu.renderer(), assignment)
            .expect("supported");
        v
    }

    fn compare(&self, label: &str, reference: &mut Viewport, test: &mut Viewport, scene: &Scene) {
        let a = self
            .gpu
            .render(reference, &self.world, scene, &self.instances)
            .expect("reference frame");
        let b = self
            .gpu
            .render(test, &self.world, scene, &self.instances)
            .expect("clustered frame");
        assert!(a.rgba == b.rgba, "{label}: images differ between paths");
    }
}

#[test]
fn clustered_matches_the_reference_through_cuts_and_light_changes() {
    let f = fixture();
    if !f.gpu.renderer().supports_clustered_lighting() {
        eprintln!("skipped: {} lacks the bind groups", f.gpu.adapter.name);
        return;
    }
    let config = ClusterConfig::default();
    let mut reference = f.viewport(LightAssignment::PerFragment);
    let mut test = f.viewport(LightAssignment::Clustered(config));
    let mut s = scene();
    f.compare("initial", &mut reference, &mut test, &s);

    // A hard cut crossing the near plane, close to a light.
    s.camera.eye = Vec3::new(-4.0, 1.0, 4.0);
    s.camera.target = Vec3::new(-2.0, 1.0, -6.0);
    f.compare("cut near plane", &mut reference, &mut test, &s);

    // Lights moved behind the camera, then removed; shadow map indices change with them.
    let mut moved = scene();
    for l in moved.point_lights.iter_mut().take(6) {
        l.position.z += 30.0;
    }
    f.compare("lights moved", &mut reference, &mut test, &moved);
    let mut fewer = scene();
    fewer.point_lights.truncate(5);
    fewer.spot_lights.truncate(1);
    f.compare("lights removed", &mut reference, &mut test, &fewer);
    let mut none = scene();
    none.point_lights.clear();
    none.spot_lights.clear();
    f.compare("no local lights", &mut reference, &mut test, &none);
    f.compare("reset", &mut reference, &mut test, &s);
}

#[test]
fn overflow_and_grids_keep_the_output_and_are_counted() {
    let f = fixture();
    if !f.gpu.renderer().supports_clustered_lighting() {
        return;
    }
    let mut reference = f.viewport(LightAssignment::PerFragment);
    let s = scene();

    // One light per cluster: most clusters overflow to the whole list, the same pixels.
    let mut capped = f.viewport(LightAssignment::Clustered(ClusterConfig {
        max_lights_per_cluster: 1,
        ..ClusterConfig::default()
    }));
    f.compare("capped", &mut reference, &mut capped, &s);
    let overflows = capped.work().cluster_overflows;
    assert!(overflows > 0, "the 1-light limit overflows some clusters");
    // Deterministic: the same frame overflows the same clusters.
    f.compare("capped again", &mut reference, &mut capped, &s);
    assert_eq!(capped.work().cluster_overflows, overflows);

    for (x, y, z) in [(1, 1, 1), (8, 4, 2), (32, 18, 48)] {
        let mut grid = f.viewport(LightAssignment::Clustered(ClusterConfig {
            x,
            y,
            z,
            ..ClusterConfig::default()
        }));
        f.compare(&format!("grid {x}x{y}x{z}"), &mut reference, &mut grid, &s);
    }
}

#[test]
fn the_work_counters_stay_zero_on_the_reference_path() {
    let f = fixture();
    let renderer = f.gpu.renderer();
    let mut reference = f.viewport(LightAssignment::PerFragment);
    let s = scene();
    renderer.prepare(&mut reference, &f.world, &s, &f.instances);
    let work = reference.work();
    assert_eq!(work.light_cluster_tests, 0);
    assert_eq!(work.light_cluster_entries, 0);
    assert_eq!(work.cluster_overflows, 0);
    assert_eq!(reference.memory().clustered_lighting_bytes, 0);
    assert_eq!(reference.light_assignment(), LightAssignment::PerFragment);
    if !renderer.supports_clustered_lighting() {
        return;
    }
    let mut clustered = f.viewport(LightAssignment::Clustered(ClusterConfig::default()));
    renderer.prepare(&mut clustered, &f.world, &s, &f.instances);
    let work = clustered.work();
    assert!(work.light_cluster_tests > 0);
    assert!(work.light_cluster_entries > 0);
    // Light-list memory is bounded by the configuration.
    assert!(clustered.memory().clustered_lighting_bytes > 0);
    renderer.prepare(&mut clustered, &f.world, &s, &f.instances);
    assert_eq!(
        clustered.work().allocations,
        0,
        "a steady frame allocates nothing"
    );
}
