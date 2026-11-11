//! Work contracts (BENCH-018; `docs/consumer-integration.md`, "Work contracts"): what
//! `Renderer::prepare` and the `World`'s edits do, counted by `Viewport::work` and
//! `World::work`, over a small matrix of instance counts and sparse or dense visibility.
//! Exact counts, not timings: a change that adds hidden, repeated or unrelated work fails here.

use glam::{Mat4, Vec3};
use render_engine::{
    headless::Headless,
    material::StandardMaterial,
    mesh::{MeshData, Shape},
    renderer::{PrepareWork, Viewport},
    scene::*,
    world::{Instance, MaterialId, MeshId, World},
};

/// Instance counts: small, medium and the size of the oracle's `many_instances` (about
/// 22 400 instances) as the large control.
const SIZES: [u32; 3] = [64, 1024, 22_400];
/// Instances the camera sees in a sparse scene; the rest stand behind it.
const SPARSE: u32 = 8;

#[derive(Clone, Copy)]
enum Lights {
    None,
    SunShadows,
    PointShadows,
}

/// A camera at the origin looking down −Z with a 90° square view.
fn scene(lights: Lights) -> Scene {
    Scene {
        camera: Camera {
            eye: Vec3::ZERO,
            target: Vec3::NEG_Z,
            fov_y: 90f32.to_radians(),
            near: 0.1,
            ev100: 10.0,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::default(),
        },
        clear_color: Vec3::ZERO,
        sun: matches!(lights, Lights::SunShadows).then(|| Sun {
            direction: Vec3::new(0.2, -1.0, -0.3),
            color: Vec3::ONE,
            illuminance: 10_000.0,
            shadows: Some(SunShadows::default()),
        }),
        moon: None,
        sky: None,
        ambient: Ambient {
            color: Vec3::ONE,
            brightness: 100.0,
        },
        point_lights: match lights {
            Lights::PointShadows => vec![PointLight {
                position: Vec3::new(0.0, 5.0, -10.0),
                range: 30.0,
                shadows: Some(LightShadows::POINT),
                ..PointLight::default()
            }],
            _ => Vec::new(),
        },
        spot_lights: Vec::new(),
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
    materials: [MaterialId; 2],
}

impl Fixture {
    fn new() -> Self {
        let gpu = Headless::new().expect("a wgpu adapter");
        let mut world = World::new(gpu.renderer());
        let cube = world.add_mesh(&MeshData::shape(Shape::Cuboid(Vec3::ONE)));
        let materials = [0.2, 0.8].map(|g| {
            world.add_material(&StandardMaterial {
                base_color: glam::Vec4::new(g, g, g, 1.0),
                ..StandardMaterial::default()
            })
        });
        Self {
            gpu,
            world,
            cube,
            materials,
        }
    }

    fn at(&self, i: u32, translation: Vec3) -> Instance {
        Instance {
            mesh: self.cube,
            material: self.materials[i as usize % 2],
            world_from_local: Mat4::from_translation(translation),
            skin: None,
        }
    }

    /// `n` unit cubes on a 10 × 10 m grid, the first `visible` 10 m in front of the camera,
    /// the rest 10 m behind it.
    fn instances(&self, n: u32, visible: u32) -> Vec<Instance> {
        (0..n)
            .map(|i| {
                let (x, y) = ((i % 150) as f32, (i / 150) as f32);
                let z = if i < visible { -10.0 } else { 10.0 };
                self.at(i, Vec3::new(x / 15.0 - 5.0, y / 15.0 - 5.0, z))
            })
            .collect()
    }

    fn prepare(
        &self,
        viewport: &mut Viewport,
        scene: &Scene,
        instances: &[Instance],
    ) -> PrepareWork {
        self.gpu
            .renderer()
            .prepare(viewport, &self.world, scene, instances);
        viewport.work()
    }

    fn viewport(&self) -> Viewport {
        Viewport::new(self.gpu.renderer(), 64, 64)
    }
}

/// Culling tests every instance once per view, however few the views keep (no hierarchy:
/// `O(N · V)`), while what the camera keeps alone drives its draws and its share of the
/// visible-index upload. Each light dimension adds its own exact term.
#[test]
fn culling_is_linear_in_instances_and_views_and_uploads_follow_visibility() {
    let f = Fixture::new();
    let mut viewport = f.viewport();
    let none = scene(Lights::None);
    // Upload bytes per frame, by size: (sparse, dense).
    let mut bytes = Vec::new();
    for n in SIZES {
        let mut per_size = [0; 2];
        for (j, k) in [SPARSE, n].into_iter().enumerate() {
            let w = f.prepare(&mut viewport, &none, &f.instances(n, k));
            let at = format!("N = {n}, k = {k}");
            assert_eq!(w.instances, n, "{at}");
            assert_eq!(w.views, 1, "{at}");
            assert_eq!(w.bounds_tests, u64::from(n), "{at}");
            assert_eq!(w.visible, k, "{at}");
            // One draw per material, however many instances share it.
            assert_eq!(w.draws, 2, "{at}");
            assert_eq!((w.shadow_visible, w.shadow_draws), (0, 0), "{at}");
            per_size[j] = w.upload_bytes;
        }
        // Dense uploads one more visible index (4 bytes) per extra visible instance.
        assert_eq!(
            per_size[1] - per_size[0],
            4 * u64::from(n - SPARSE),
            "N = {n}"
        );
        bytes.push(per_size[0]);
    }
    // Sparse frames grow by the same bytes per instance (its instance data) at every size:
    // hidden instances cost their instance data and nothing that grows with them.
    let per_instance = (bytes[1] - bytes[0]) / u64::from(SIZES[1] - SIZES[0]);
    assert!(per_instance > 0);
    for (n, b) in SIZES.iter().zip(&bytes) {
        assert_eq!(
            b - bytes[0],
            per_instance * u64::from(n - SIZES[0]),
            "N = {n}"
        );
    }

    // A shadowed sun adds one view per cascade, each testing every instance.
    let cascades = SunShadows::default().num_cascades as u32;
    for n in [SIZES[0], SIZES[1]] {
        let w = f.prepare(
            &mut viewport,
            &scene(Lights::SunShadows),
            &f.instances(n, SPARSE),
        );
        assert_eq!(w.views, 1 + cascades, "N = {n}");
        assert_eq!(w.bounds_tests, u64::from(n * (1 + cascades)), "N = {n}");
        assert_eq!(w.visible, SPARSE, "N = {n}");
    }
    // A shadowed point light adds six cube faces plus one range test per instance.
    for n in [SIZES[0], SIZES[1]] {
        let w = f.prepare(
            &mut viewport,
            &scene(Lights::PointShadows),
            &f.instances(n, SPARSE),
        );
        assert_eq!(w.views, 7, "N = {n}");
        assert_eq!(w.bounds_tests, u64::from(n * 8), "N = {n}");
        assert_eq!(w.visible, SPARSE, "N = {n}");
    }
}

/// Preparing the same frame again repeats the same per-frame work, allocates nothing and
/// leaves the world's content alone.
#[test]
fn an_unchanged_frame_repeats_its_work_without_allocating() {
    let f = Fixture::new();
    let mut viewport = f.viewport();
    for lights in [Lights::None, Lights::SunShadows, Lights::PointShadows] {
        let scene = scene(lights);
        for n in [SIZES[0], SIZES[1]] {
            let instances = f.instances(n, SPARSE);
            f.prepare(&mut viewport, &scene, &instances);
            let world_before = f.world.work();
            let first = f.prepare(&mut viewport, &scene, &instances);
            let second = f.prepare(&mut viewport, &scene, &instances);
            assert_eq!(first.allocations, 0, "N = {n}");
            assert_eq!(second, first, "N = {n}");
            assert_eq!(f.world.work(), world_before, "prepare uploads no content");
        }
    }
}

/// One edit costs that edit alone, whatever else the world holds; a moved instance changes
/// nothing about the next `prepare`'s work, which is a full pass either way.
#[test]
fn one_mesh_material_or_transform_edit_costs_that_edit_alone() {
    let mut deltas = Vec::new();
    for meshes in [4, 256] {
        let mut f = Fixture::new();
        let shape = MeshData::shape(Shape::Cuboid(Vec3::ONE));
        let ids: Vec<MeshId> = (0..meshes).map(|_| f.world.add_mesh(&shape)).collect();
        let materials: Vec<MaterialId> = (0..meshes)
            .map(|_| f.world.add_material(&StandardMaterial::default()))
            .collect();

        let before = f.world.work();
        f.world.set_mesh(ids[1], &shape);
        let mesh = f.world.work() - before;
        assert_eq!((mesh.mesh_rewrites, mesh.mesh_builds), (1, 0));

        let before = f.world.work();
        let mut bigger = shape.clone();
        bigger.positions.iter_mut().for_each(|p| *p *= 2.0);
        f.world
            .set_mesh_positions_normals(ids[2], &bigger.positions, &bigger.normals);
        let pose = f.world.work() - before;
        assert_eq!(pose.mesh_rewrites, 1);
        assert!(pose.upload_bytes < mesh.upload_bytes);

        let before = f.world.work();
        f.world.set_material(
            materials[3],
            &StandardMaterial {
                perceptual_roughness: 0.2,
                ..StandardMaterial::default()
            },
        );
        let material = f.world.work() - before;
        assert_eq!(material.material_builds, 1);
        deltas.push((mesh, pose, material));

        // A moved instance: the same prepare work as the unchanged frame.
        let mut viewport = f.viewport();
        let scene = scene(Lights::SunShadows);
        let mut instances = f.instances(1024, SPARSE);
        f.prepare(&mut viewport, &scene, &instances);
        let unchanged = f.prepare(&mut viewport, &scene, &instances);
        instances[3].world_from_local *= Mat4::from_translation(Vec3::new(0.1, 0.0, 0.0));
        assert_eq!(f.prepare(&mut viewport, &scene, &instances), unchanged);
    }
    assert_eq!(
        deltas[0], deltas[1],
        "edits cost the same in a larger world"
    );
}

/// Nothing culled is kept from one frame to the next: moved, removed and resized instances and
/// a camera cut show in the very next `prepare`.
#[test]
fn mutation_removal_bounds_and_camera_cuts_take_effect_next_frame() {
    let mut f = Fixture::new();
    let mut viewport = f.viewport();
    let mut scene = scene(Lights::None);
    let n = 256;
    let mut instances = f.instances(n, SPARSE);
    assert_eq!(f.prepare(&mut viewport, &scene, &instances).visible, SPARSE);

    // A hidden instance moves into view.
    instances[SPARSE as usize].world_from_local = Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0));
    assert_eq!(
        f.prepare(&mut viewport, &scene, &instances).visible,
        SPARSE + 1
    );

    // A visible instance is removed.
    instances.remove(0);
    let w = f.prepare(&mut viewport, &scene, &instances);
    assert_eq!((w.visible, w.bounds_tests), (SPARSE, u64::from(n - 1)));

    // Bounds follow a mesh edit: a small mesh just outside the view's edge grows into it.
    let small = MeshData::shape(Shape::Cuboid(Vec3::ONE));
    let edge = f.world.add_mesh(&small);
    instances.push(Instance {
        mesh: edge,
        ..f.at(0, Vec3::new(12.0, 0.0, -10.0))
    });
    assert_eq!(f.prepare(&mut viewport, &scene, &instances).visible, SPARSE);
    let mut grown = small.clone();
    grown.positions.iter_mut().for_each(|p| *p *= 6.0);
    f.world
        .set_mesh_positions_normals(edge, &grown.positions, &grown.normals);
    assert_eq!(
        f.prepare(&mut viewport, &scene, &instances).visible,
        SPARSE + 1
    );

    // A camera cut: turned around, it sees the instances that stood behind it.
    scene.camera.target = Vec3::Z;
    let w = f.prepare(&mut viewport, &scene, &instances);
    assert_eq!(w.visible, n - SPARSE - 1);
    assert_eq!(w.bounds_tests, u64::from(n));
}

/// Retained instances (`World::add_instance`) are culled like listed ones every frame, but
/// written once: an unchanged frame writes none, and an edit writes that instance alone,
/// whatever else the world retains.
#[test]
fn retained_instances_are_written_once_and_again_only_when_edited() {
    let scene = scene(Lights::SunShadows);
    let stride = 128;
    let mut edits = Vec::new();
    for n in [SIZES[0], SIZES[1]] {
        let mut f = Fixture::new();
        let all = f.instances(n, SPARSE);
        let mut listed_viewport = f.viewport();
        f.prepare(&mut listed_viewport, &scene, &all);
        let listed = f.prepare(&mut listed_viewport, &scene, &all);
        let ids: Vec<_> = all.iter().map(|i| f.world.add_instance(*i)).collect();
        assert_eq!(f.world.work().instance_edits, u64::from(n));

        let mut viewport = f.viewport();
        let first = f.prepare(&mut viewport, &scene, &[]);
        assert_eq!(
            first.instance_writes, n,
            "N = {n}: the first frame writes them all"
        );
        let steady = f.prepare(&mut viewport, &scene, &[]);
        assert_eq!(steady.instance_writes, 0, "N = {n}");
        assert_eq!(steady.allocations, 0, "N = {n}");
        // The same culling and draws as listing them, without the instance upload.
        assert_eq!(
            PrepareWork {
                instance_writes: n,
                upload_bytes: steady.upload_bytes + stride * u64::from(n),
                ..steady
            },
            listed,
            "N = {n}"
        );

        // Setting an unchanged value is no edit.
        let before = f.world.work();
        f.world.set_instance(ids[1], all[1]);
        assert_eq!(f.world.work(), before);
        assert_eq!(f.prepare(&mut viewport, &scene, &[]), steady);

        // A moved instance: one row written.
        let mut moved = all[3];
        moved.world_from_local *= Mat4::from_translation(Vec3::new(0.1, 0.0, 0.0));
        f.world.set_instance(ids[3], moved);
        let w = f.prepare(&mut viewport, &scene, &[]);
        assert_eq!(w.instance_writes, 1, "N = {n}");
        assert_eq!(w.upload_bytes, steady.upload_bytes + stride, "N = {n}");

        // A removed instance: its row is cleared, and it is no longer drawn or tested.
        f.world.remove_instance(ids[0]);
        let w = f.prepare(&mut viewport, &scene, &[]);
        assert_eq!((w.instances, w.instance_writes), (n - 1, 1), "N = {n}");
        assert_eq!(w.visible, SPARSE - 1, "N = {n}");
        edits.push(f.world.work() - before);

        // A second viewport writes every row once.
        let mut other = f.viewport();
        assert_eq!(f.prepare(&mut other, &scene, &[]).instance_writes, n);
    }
    assert_eq!(edits[0], edits[1], "edits cost the same in a larger world");
}
