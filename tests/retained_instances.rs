//! Retained instances (`World::add_instance`, #95) against the listed reference path on one
//! device: two worlds with the same content, one drawing its instances from the frame's list and
//! one retaining them, give identical images and per-view counts through edits, removal, mesh
//! and material changes, a mix of retained and listed instances, camera cuts and (where the
//! device has it) GPU culling.

use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use render_engine::{
    headless::Headless,
    material::{AlphaMode, StandardMaterial},
    mesh::{MeshData, Shape},
    renderer::{Culling, PrepareWork, Viewport},
    scene::*,
    world::{Instance, InstanceId, MaterialId, MeshId, SkinId, World},
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
            illuminance: 20_000.0,
            shadows: Some(SunShadows::default()),
        }),
        moon: None,
        sky: None,
        ambient: Ambient {
            color: Vec3::ONE,
            brightness: 200.0,
        },
        point_lights: vec![PointLight {
            position: Vec3::new(-3.0, 5.0, 2.0),
            range: 25.0,
            intensity: 800_000.0,
            shadows: Some(LightShadows::POINT),
            ..PointLight::default()
        }],
        spot_lights: Vec::new(),
        shadow_maps: ShadowMapSizes::default(),
        environment: None,
        fog: None,
        overlay: Vec::new(),
    }
}

/// The handles of [`build`], the same in every world it is built in.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Content {
    ground: MeshId,
    cube: MeshId,
    sphere: MeshId,
    figure: MeshId,
    skin: SkinId,
    grey: MaterialId,
    mask: MaterialId,
    blend: MaterialId,
}

fn build(world: &mut World) -> Content {
    let mut figure = MeshData::shape(Shape::Cuboid(Vec3::new(0.6, 1.8, 0.4)));
    figure.joints = figure
        .positions
        .iter()
        .map(|p| [u16::from(p.y > 0.0), 0, 0, 0])
        .collect();
    figure.weights = vec![Vec4::new(1.0, 0.0, 0.0, 0.0); figure.positions.len()];
    Content {
        ground: world.add_mesh(&MeshData::shape(Shape::Plane(Vec2::splat(120.0)))),
        cube: world.add_mesh(&MeshData::shape(Shape::Cuboid(Vec3::ONE))),
        sphere: world.add_mesh(&MeshData::shape(Shape::Sphere {
            radius: 0.5,
            sectors: 16,
            stacks: 8,
        })),
        figure: world.add_mesh(&figure),
        skin: world.add_skin(&[Mat4::IDENTITY, Mat4::from_rotation_z(0.4)]),
        grey: world.add_material(&StandardMaterial::default()),
        mask: world.add_material(&StandardMaterial {
            base_color: Vec4::new(0.8, 0.5, 0.1, 0.9),
            alpha_mode: AlphaMode::Mask(0.5),
            ..StandardMaterial::default()
        }),
        blend: world.add_material(&StandardMaterial {
            base_color: Vec4::new(0.1, 0.4, 0.9, 0.5),
            alpha_mode: AlphaMode::Blend,
            ..StandardMaterial::default()
        }),
    }
}

/// A ground plane, a grid of opaque and masked props (one mirrored), two blended props and a
/// skinned figure.
fn instances(c: &Content) -> Vec<Instance> {
    let at = |mesh, material, m| Instance {
        mesh,
        material,
        world_from_local: m,
        skin: None,
    };
    let mut list = vec![at(c.ground, c.grey, Mat4::IDENTITY)];
    for z in -6..6 {
        for x in -6..6 {
            let (mesh, material) = if (x + z) % 3 == 0 {
                (c.sphere, c.mask)
            } else {
                (c.cube, c.grey)
            };
            list.push(at(
                mesh,
                material,
                Mat4::from_rotation_translation(
                    Quat::from_rotation_y(x as f32),
                    Vec3::new(x as f32 * 2.5, 0.5, z as f32 * 2.5),
                ),
            ));
        }
    }
    list[7].world_from_local *= Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0));
    list.push(at(
        c.cube,
        c.blend,
        Mat4::from_translation(Vec3::new(0.0, 3.0, 12.0)),
    ));
    list.push(at(
        c.sphere,
        c.blend,
        Mat4::from_translation(Vec3::new(1.0, 3.0, 14.0)),
    ));
    list.push(Instance {
        skin: Some(c.skin),
        ..at(
            c.figure,
            c.grey,
            Mat4::from_translation(Vec3::new(-2.0, 0.9, 8.0)),
        )
    });
    list
}

/// What both paths must agree on.
fn counts(w: PrepareWork) -> [u64; 6] {
    [
        w.instances.into(),
        w.bounds_tests,
        w.visible.into(),
        w.shadow_visible.into(),
        w.draws.into(),
        w.shadow_draws.into(),
    ]
}

struct Pair {
    gpu: Headless,
    /// Draws its instances from the frame's list.
    listed: World,
    /// Retains them.
    retained: World,
    content: Content,
    reference: Viewport,
    viewport: Viewport,
    /// `retained`'s instances on the GPU culling path, where the device has it.
    gpu_culled: Option<Viewport>,
}

impl Pair {
    fn new() -> Self {
        let gpu = Headless::new().expect("a wgpu adapter");
        let mut listed = World::new(gpu.renderer());
        let mut retained = World::new(gpu.renderer());
        let content = build(&mut listed);
        assert_eq!(build(&mut retained), content);
        let renderer = gpu.renderer();
        let gpu_culled = renderer.supports_gpu_culling().then(|| {
            let mut v = Viewport::new(renderer, SIZE, SIZE);
            v.set_culling(renderer, Culling::GpuIndirect)
                .expect("supported");
            v
        });
        Self {
            reference: Viewport::new(renderer, SIZE, SIZE),
            viewport: Viewport::new(renderer, SIZE, SIZE),
            gpu_culled,
            gpu,
            listed,
            retained,
            content,
        }
    }

    /// Renders `all` listed in the reference world, and the retained world with `listed`.
    fn compare(&mut self, label: &str, scene: &Scene, all: &[Instance], listed: &[Instance]) {
        let a = self
            .gpu
            .render(&mut self.reference, &self.listed, scene, all)
            .expect("listed frame");
        let b = self
            .gpu
            .render(&mut self.viewport, &self.retained, scene, listed)
            .expect("retained frame");
        assert!(a.rgba == b.rgba, "{label}: images differ");
        assert_eq!(
            counts(self.reference.work()),
            counts(self.viewport.work()),
            "{label}: work differs"
        );
        if let Some(v) = &mut self.gpu_culled {
            let c = self
                .gpu
                .render(v, &self.retained, scene, listed)
                .expect("gpu-culled frame");
            assert!(a.rgba == c.rgba, "{label}: images differ with GPU culling");
        }
    }
}

#[test]
fn retained_instances_match_the_listed_path_through_edits() {
    let mut p = Pair::new();
    let mut scene = scene();
    let mut all = instances(&p.content);
    let mut ids: Vec<InstanceId> = all.iter().map(|i| p.retained.add_instance(*i)).collect();
    p.compare("all retained", &scene, &all, &[]);
    p.compare("unchanged", &scene, &all, &[]);
    assert_eq!(
        p.viewport.work().instance_writes,
        1,
        "the skinned figure alone"
    );

    // Moved into view, made blended, removed, added.
    all[3].world_from_local = Mat4::from_translation(Vec3::new(0.5, 1.0, 16.0));
    p.retained.set_instance(ids[3], all[3]);
    all[10].material = p.content.blend;
    p.retained.set_instance(ids[10], all[10]);
    all.remove(20);
    p.retained.remove_instance(ids.remove(20));
    p.compare("edited", &scene, &all, &[]);
    assert_eq!(p.viewport.work().instance_writes, 1 + 3);
    let extra = Instance {
        world_from_local: Mat4::from_translation(Vec3::new(-1.0, 2.0, 15.0)),
        ..all[4]
    };
    // The new instance takes the freed slot, which sorts it where the removed one stood.
    all.insert(20, extra);
    ids.insert(20, p.retained.add_instance(extra));
    p.compare("added", &scene, &all, &[]);

    // Mesh bounds and a material's phase change under the instances that use them.
    let mut big = MeshData::shape(Shape::Cuboid(Vec3::ONE));
    big.positions.iter_mut().for_each(|v| *v *= 1.5);
    for world in [&mut p.listed, &mut p.retained] {
        world.set_mesh_positions_normals(p.content.cube, &big.positions, &big.normals);
        world.set_material(
            p.content.mask,
            &StandardMaterial {
                base_color: Vec4::new(0.8, 0.5, 0.1, 0.6),
                alpha_mode: AlphaMode::Blend,
                ..StandardMaterial::default()
            },
        );
    }
    p.compare("mesh and material edits", &scene, &all, &[]);

    // A pose moves the skinned figure's bounds and vertices.
    for world in [&mut p.listed, &mut p.retained] {
        world.set_skin(
            p.content.skin,
            &[Mat4::from_rotation_y(0.5), Mat4::from_rotation_z(-0.6)],
        );
    }
    p.compare("posed", &scene, &all, &[]);

    // Camera cut.
    scene.camera.eye = Vec3::new(10.0, 4.0, -18.0);
    p.compare("camera cut", &scene, &all, &[]);
}

#[test]
fn retained_and_listed_instances_draw_together() {
    let mut p = Pair::new();
    let scene = scene();
    let all = instances(&p.content);
    // Retained instances come first in the rows, so tie-breaks match a list in that order.
    let (kept, listed) = all.split_at(all.len() / 2);
    for i in kept {
        p.retained.add_instance(*i);
    }
    p.compare("mixed", &scene, &all, listed);
    p.compare("mixed again", &scene, &all, listed);
    assert_eq!(
        p.viewport.work().instance_writes,
        listed.len() as u32,
        "only the listed instances are written"
    );
}

#[test]
fn a_viewport_follows_the_world_it_is_given() {
    let mut p = Pair::new();
    let scene = scene();
    let all = instances(&p.content);
    for i in &all {
        p.retained.add_instance(*i);
    }
    p.compare("retained world", &scene, &all, &[]);
    // The same viewport on the listed world drops the retained rows.
    let renderer = p.gpu.renderer();
    renderer.prepare(&mut p.viewport, &p.listed, &scene, &all);
    assert_eq!(p.viewport.work().instances, all.len() as u32);
    // And back: every retained row is written again.
    renderer.prepare(&mut p.viewport, &p.retained, &scene, &[]);
    assert_eq!(p.viewport.work().instance_writes, all.len() as u32);
    p.compare("back", &scene, &all, &[]);
}
