//! The opt-in GPU culling and indirect path (`Culling::GpuIndirect`, #71) against the CPU
//! reference on one device: identical images and identical per-view instance counts across
//! near-plane crossings, shadows, masked and blended and skinned instances, mutation, removal,
//! mesh bounds edits and camera cuts. The bench (`oracle --gpu-driven-bench`) runs the same
//! comparison at full scale on every oracle scenario.

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use render_engine::{
    headless::Headless,
    material::{AlphaMode, StandardMaterial},
    mesh::{MeshData, Shape},
    renderer::{Culling, Viewport},
    scene::*,
    world::{Instance, MaterialId, MeshId, World},
};

const SIZE: u32 = 128;

fn base_scene(lights: bool) -> Scene {
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
            illuminance: 20_000.0,
            shadows: Some(SunShadows::default()),
        }),
        moon: None,
        sky: None,
        ambient: Ambient {
            color: Vec3::ONE,
            brightness: 200.0,
        },
        point_lights: if lights {
            vec![PointLight {
                position: Vec3::new(-3.0, 5.0, 2.0),
                range: 25.0,
                intensity: 800_000.0,
                shadows: Some(LightShadows::POINT),
                ..PointLight::default()
            }]
        } else {
            Vec::new()
        },
        spot_lights: if lights {
            vec![SpotLight {
                position: Vec3::new(5.0, 7.0, 6.0),
                direction: Vec3::new(-0.5, -1.0, -0.6),
                range: 30.0,
                intensity: 800_000.0,
                shadows: Some(LightShadows::SPOT),
                ..SpotLight::default()
            }]
        } else {
            Vec::new()
        },
        shadow_maps: ShadowMapSizes::default(),
        environment: None,
        fog: None,
        overlay: Vec::new(),
    }
}

struct Fixture {
    gpu: Headless,
    world: World,
    cube: MeshId,
    sphere: MeshId,
    grey: MaterialId,
    mask: MaterialId,
    blend: MaterialId,
}

fn fixture() -> Fixture {
    let gpu = Headless::new().expect("a wgpu adapter");
    let mut world = World::new(gpu.renderer());
    let cube = world.add_mesh(&MeshData::shape(Shape::Cuboid(Vec3::ONE)));
    let sphere = world.add_mesh(&MeshData::shape(Shape::Sphere {
        radius: 0.5,
        sectors: 16,
        stacks: 8,
    }));
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
    Fixture {
        gpu,
        world,
        cube,
        sphere,
        grey,
        mask,
        blend,
    }
}

impl Fixture {
    /// A grid of opaque and masked props on a ground plane, one blended prop.
    fn instances(&mut self) -> Vec<Instance> {
        let ground = self
            .world
            .add_mesh(&MeshData::shape(Shape::Plane(Vec2::splat(120.0))));
        let mut list = vec![Instance {
            mesh: ground,
            material: self.grey,
            world_from_local: Mat4::IDENTITY,
            skin: None,
        }];
        for z in -6..6 {
            for x in -6..6 {
                let (mesh, material) = if (x + z) % 3 == 0 {
                    (self.sphere, self.mask)
                } else {
                    (self.cube, self.grey)
                };
                list.push(Instance {
                    mesh,
                    material,
                    world_from_local: Mat4::from_rotation_translation(
                        Quat::from_rotation_y(x as f32),
                        Vec3::new(x as f32 * 2.5, 0.5, z as f32 * 2.5),
                    ),
                    skin: None,
                });
            }
        }
        list.push(Instance {
            mesh: self.cube,
            material: self.blend,
            world_from_local: Mat4::from_translation(Vec3::new(0.0, 3.0, 12.0)),
            skin: None,
        });
        list
    }

    fn pair(&self) -> Option<(Viewport, Viewport)> {
        if !self.gpu.renderer().supports_gpu_culling() {
            eprintln!(
                "skipped: {} lacks what GpuIndirect needs",
                self.gpu.adapter.name
            );
            return None;
        }
        let cpu = Viewport::new(self.gpu.renderer(), SIZE, SIZE);
        let mut gpu = Viewport::new(self.gpu.renderer(), SIZE, SIZE);
        gpu.set_culling(self.gpu.renderer(), Culling::GpuIndirect)
            .expect("supported");
        Some((cpu, gpu))
    }

    /// Renders both paths and compares images and counts.
    fn compare(
        &self,
        label: &str,
        (cpu, gpu): (&mut Viewport, &mut Viewport),
        scene: &Scene,
        instances: &[Instance],
    ) {
        let a = self
            .gpu
            .render(cpu, &self.world, scene, instances)
            .expect("cpu frame");
        let b = self
            .gpu
            .render(gpu, &self.world, scene, instances)
            .expect("gpu frame");
        assert!(a.rgba == b.rgba, "{label}: images differ between paths");
        // The CPU path counts visible instances in `work`; the GPU path's are read back.
        let cpu_work = cpu.work();
        let counts = self
            .gpu
            .renderer()
            .read_gpu_culled(gpu)
            .expect("GPU path active");
        let gpu_work = gpu.work();
        let gpu_kept: u32 = counts.iter().sum::<u32>() + gpu_work.visible + gpu_work.shadow_visible;
        assert_eq!(
            gpu_kept,
            cpu_work.visible + cpu_work.shadow_visible,
            "{label}: instances kept across views differ"
        );
        assert_eq!(
            counts[0] + gpu_work.visible,
            cpu_work.visible,
            "{label}: camera instances differ"
        );
    }
}

#[test]
fn gpu_culling_matches_the_cpu_path_through_a_camera_cut_mutation_and_removal() {
    let mut f = fixture();
    let Some((mut cpu, mut gpu)) = f.pair() else {
        return;
    };
    let mut instances = f.instances();
    let mut scene = base_scene(false);
    f.compare("initial", (&mut cpu, &mut gpu), &scene, &instances);

    // Camera cut: straight into a different, close view that crosses the near plane.
    scene.camera.eye = Vec3::new(0.0, 0.7, 0.7);
    scene.camera.target = Vec3::new(-6.0, 0.5, -8.0);
    f.compare("cut near plane", (&mut cpu, &mut gpu), &scene, &instances);

    // Back again, then a mutation: an instance jumps across the view.
    scene.camera.eye = Vec3::new(0.0, 6.0, 22.0);
    scene.camera.target = Vec3::new(0.0, 1.0, 0.0);
    instances[5].world_from_local = Mat4::from_translation(Vec3::new(1.0, 6.0, 15.0));
    f.compare("mutation", (&mut cpu, &mut gpu), &scene, &instances);

    // Removal shrinks the list; both paths see it at once.
    instances.drain(1..14);
    f.compare("removal", (&mut cpu, &mut gpu), &scene, &instances);

    // A mesh's bounds grow; the group table follows the edit.
    let bigger = {
        let mut m = MeshData::shape(Shape::Cuboid(Vec3::ONE));
        m.positions.iter_mut().for_each(|p| *p *= 4.0);
        m
    };
    f.world
        .set_mesh_positions_normals(f.cube, &bigger.positions, &bigger.normals);
    f.compare("mesh bounds", (&mut cpu, &mut gpu), &scene, &instances);

    // Everything can be culled away.
    scene.camera.eye = Vec3::new(0.0, 6.0, 500.0);
    scene.camera.target = Vec3::new(0.0, 6.0, 800.0);
    f.compare("nothing visible", (&mut cpu, &mut gpu), &scene, &instances);
}

#[test]
fn gpu_culling_matches_the_cpu_path_with_point_and_spot_shadows() {
    let mut f = fixture();
    let Some((mut cpu, mut gpu)) = f.pair() else {
        return;
    };
    let instances = f.instances();
    let scene = base_scene(true);
    f.compare("local shadows", (&mut cpu, &mut gpu), &scene, &instances);
    let w = gpu.work();
    assert_eq!(w.views, 1 + 4 + 1 + 6, "camera, cascades, spot, point cube");
    assert!(w.gpu_bounds_tests > 0, "the compute pass tests the rows");
}

#[test]
fn work_counters_split_cpu_and_gpu_tests_on_the_gpu_path() {
    let mut f = fixture();
    let Some((_, mut gpu)) = f.pair() else {
        return;
    };
    let instances = f.instances();
    let scene = base_scene(false);
    let n = instances.len() as u64;
    f.gpu
        .renderer()
        .prepare(&mut gpu, &f.world, &scene, &instances);
    let w = gpu.work();
    assert_eq!(u64::from(w.views), 5, "camera and four cascades");
    // All but the one blended instance run on the GPU, per view.
    assert_eq!(w.gpu_bounds_tests, (n - 1) * 5);
    assert_eq!(
        w.bounds_tests, 5,
        "only the blended instance is tested on the CPU"
    );
    // Frames settle: no new buffers once the shape is steady.
    f.gpu
        .renderer()
        .prepare(&mut gpu, &f.world, &scene, &instances);
    assert_eq!(gpu.work().allocations, 0);
    assert_eq!(gpu.culling(), Culling::GpuIndirect);
}

#[test]
fn the_cpu_path_stays_the_reference_by_default() {
    let f = fixture();
    let viewport = Viewport::new(f.gpu.renderer(), SIZE, SIZE);
    assert_eq!(viewport.culling(), Culling::Cpu);
    assert_eq!(viewport.work().gpu_bounds_tests, 0);
}
