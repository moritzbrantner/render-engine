//! Microbenchmark for per-frame pose uploads (render-engine#34): the mmorpg Outpost case of 89
//! moving figures with about 110 k vertices in all, posed three ways each frame:
//!
//! - `set_mesh`: CPU skinning, the whole mesh re-interleaved and rewritten with its indices;
//! - `set_mesh_positions_normals`: CPU skinning, only the two pose streams rewritten;
//! - `set_skin`: GPU skinning, a 65-joint palette per figure, gathered by `Renderer::prepare`.
//!
//! Each frame's time covers the `World` calls (plus `prepare` for the palettes), the queue
//! submit and the wait for the GPU to take the writes.
//!
//! `cargo run --release --example mesh_upload`

use glam::{Mat4, Vec2, Vec3, Vec4};
use render_engine::{
    headless::Headless,
    material::StandardMaterial,
    mesh::{MeshData, Shape},
    renderer::Viewport,
    scene::*,
    world::{Instance, World},
};
use std::time::Instant;

const FIGURES: usize = 89;
const JOINTS: usize = 65;
const FRAMES: usize = 200;

fn main() {
    let gpu = Headless::new().expect("a wgpu adapter");
    let renderer = gpu.renderer();
    let (device, queue) = (renderer.device(), renderer.queue());
    let mut world = World::new(renderer);
    let material = world.add_material(&StandardMaterial::default());

    // About 1240 vertices per figure, with every attribute a glTF character has.
    let mut figure = MeshData::shape(Shape::Sphere {
        radius: 0.5,
        sectors: 39,
        stacks: 30,
    });
    let n = figure.positions.len();
    figure.uvs = figure
        .positions
        .iter()
        .map(|p| Vec2::new(p.x, p.y))
        .collect();
    figure.tangents = vec![Vec4::new(1.0, 0.0, 0.0, 1.0); n];
    figure.colors = vec![Vec4::ONE; n];
    let mut skinned = figure.clone();
    skinned.joints = (0..n)
        .map(|i| {
            let j = (i % JOINTS) as u16;
            [j, (j + 1) % JOINTS as u16, 0, 0]
        })
        .collect();
    skinned.weights = vec![Vec4::new(0.7, 0.3, 0.0, 0.0); n];
    let cpu: Vec<_> = (0..FIGURES).map(|_| world.add_mesh(&figure)).collect();
    let gpu_mesh = world.add_mesh(&skinned);
    let skins: Vec<_> = (0..FIGURES)
        .map(|_| world.add_skin(&[Mat4::IDENTITY; JOINTS]))
        .collect();
    println!(
        "{FIGURES} figures × {n} vertices = {} vertices, {} indices each",
        FIGURES * n,
        figure.indices.len()
    );

    let wait = || {
        queue.submit([]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
    };
    let time = |label: &str, frame: &mut dyn FnMut(usize)| {
        for f in 0..10 {
            frame(f);
            wait();
        }
        let start = Instant::now();
        for f in 0..FRAMES {
            frame(f);
            wait();
        }
        let ms = start.elapsed().as_secs_f64() * 1000.0 / FRAMES as f64;
        println!(
            "{label:<28} {ms:7.3} ms/frame  {:6.2} µs/mesh",
            ms * 1000.0 / FIGURES as f64
        );
    };

    let pose = |f: usize, mesh: &mut MeshData| {
        let s = 1.0 + 0.01 * (f % 7) as f32;
        for (p, src) in mesh.positions.iter_mut().zip(&figure.positions) {
            *p = *src * s;
        }
    };
    let mut posed = figure.clone();
    time("set_mesh", &mut |f| {
        pose(f, &mut posed);
        for &id in &cpu {
            world.set_mesh(id, &posed);
        }
    });
    time("set_mesh_positions_normals", &mut |f| {
        pose(f, &mut posed);
        for &id in &cpu {
            world.set_mesh_positions_normals(id, &posed.positions, &posed.normals);
        }
    });

    let scene = Scene {
        camera: Camera {
            eye: Vec3::new(0.0, 4.0, 6.0),
            target: Vec3::ZERO,
            fov_y: 45f32.to_radians(),
            near: 0.1,
            ev100: 10.5,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::AcesFitted,
        },
        clear_color: Vec3::ZERO,
        sun: None,
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
    };
    let instances: Vec<Instance> = skins
        .iter()
        .enumerate()
        .map(|(i, &skin)| Instance {
            mesh: gpu_mesh,
            material,
            world_from_local: Mat4::from_translation(Vec3::new(i as f32, 0.0, 0.0)),
            skin: Some(skin),
        })
        .collect();
    let mut viewport = Viewport::new(renderer, 64, 64);
    let mut palette = vec![Mat4::IDENTITY; JOINTS];
    time("set_skin + prepare", &mut |f| {
        let a = 0.01 * (f % 7) as f32;
        for (j, m) in palette.iter_mut().enumerate() {
            *m = Mat4::from_rotation_y(a * j as f32);
        }
        for &skin in &skins {
            world.set_skin(skin, &palette);
        }
        renderer.prepare(&mut viewport, &world, &scene, &instances);
    });
    let unskinned: Vec<Instance> = cpu
        .iter()
        .enumerate()
        .map(|(i, &mesh)| Instance {
            mesh,
            material,
            world_from_local: Mat4::from_translation(Vec3::new(i as f32, 0.0, 0.0)),
            skin: None,
        })
        .collect();
    time("prepare alone (unskinned)", &mut |_| {
        renderer.prepare(&mut viewport, &world, &scene, &unskinned);
    });
}
