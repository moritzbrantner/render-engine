//! Microbenchmark for `Renderer::prepare` on a Greyhaven-sized scene (render-engine#95): about
//! 26 000 static scenery instances of 300 mesh/material pairs over 400 × 400 m, a few moving
//! units, a sun with four shadow cascades and 13 unshadowed point lights, so five views are
//! culled. Each frame moves the units and prepares the viewport, then submits and waits.
//!
//! - `listed`: every instance in the per-frame list (the reference path);
//! - `retained`: the scenery added once with `World::add_instance`, the units listed.
//!
//! Both print the median `prepare` time and the frame's `PrepareWork`.
//!
//! `cargo run --release --example retained_instances`

use glam::{Mat4, Quat, Vec3};
use render_engine::{
    headless::Headless,
    material::StandardMaterial,
    mesh::{MeshData, Shape},
    renderer::Viewport,
    scene::*,
    world::{Instance, World},
};
use std::time::Instant;

const SCENERY: u32 = 26_000;
const KINDS: u32 = 300;
const UNITS: u32 = 16;
const FRAMES: usize = 300;

fn main() {
    let gpu = Headless::new().expect("a wgpu adapter");
    let renderer = gpu.renderer();
    let (device, queue) = (renderer.device(), renderer.queue());
    let mut world = World::new(renderer);
    let kinds: Vec<_> = (0..KINDS)
        .map(|k| {
            let shape = if k % 2 == 0 {
                Shape::Cuboid(Vec3::new(1.0, 2.0 + (k % 5) as f32, 1.0))
            } else {
                Shape::Sphere {
                    radius: 0.5 + (k % 3) as f32 * 0.25,
                    sectors: 12,
                    stacks: 8,
                }
            };
            let mesh = world.add_mesh(&MeshData::shape(shape));
            let material = world.add_material(&StandardMaterial {
                perceptual_roughness: 0.3 + (k % 7) as f32 * 0.1,
                ..StandardMaterial::default()
            });
            (mesh, material)
        })
        .collect();

    // A jittered grid, hashed so neighbours are different kinds.
    let side = (SCENERY as f32).sqrt().ceil() as u32;
    let scenery: Vec<Instance> = (0..SCENERY)
        .map(|i| {
            let h = i.wrapping_mul(2_654_435_761);
            let (x, z) = (i % side, i / side);
            let pos = Vec3::new(
                (x as f32 + (h % 97) as f32 / 97.0) / side as f32 * 400.0 - 200.0,
                0.0,
                (z as f32 + (h % 89) as f32 / 89.0) / side as f32 * 400.0 - 200.0,
            );
            let (mesh, material) = kinds[(h % KINDS) as usize];
            Instance {
                mesh,
                material,
                world_from_local: Mat4::from_scale_rotation_translation(
                    Vec3::splat(0.8 + (h % 5) as f32 * 0.1),
                    Quat::from_rotation_y((h % 360) as f32),
                    pos,
                ),
                skin: None,
            }
        })
        .collect();
    let units = |f: usize| -> Vec<Instance> {
        (0..UNITS)
            .map(|u| {
                let a = u as f32 + f as f32 * 0.01;
                let (mesh, material) = kinds[u as usize];
                Instance {
                    mesh,
                    material,
                    world_from_local: Mat4::from_translation(Vec3::new(
                        a.cos() * 10.0,
                        0.0,
                        a.sin() * 10.0,
                    )),
                    skin: None,
                }
            })
            .collect()
    };

    let scene = Scene {
        camera: Camera {
            eye: Vec3::new(5.0, 8.5, 31.0),
            target: Vec3::new(-14.0, 3.0, 4.0),
            fov_y: 50f32.to_radians(),
            near: 0.2,
            ev100: 12.0,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::TonyMcMapface,
        },
        clear_color: Vec3::ZERO,
        sun: Some(Sun {
            direction: Vec3::new(0.3, -1.0, -0.4).normalize(),
            color: Vec3::ONE,
            illuminance: 20_000.0,
            shadows: Some(SunShadows {
                max_distance: 160.0,
                ..SunShadows::default()
            }),
        }),
        moon: None,
        sky: None,
        ambient: Ambient {
            color: Vec3::ONE,
            brightness: 100.0,
        },
        point_lights: (0..13)
            .map(|i| PointLight {
                position: Vec3::new(i as f32 * 6.0 - 36.0, 3.0, 0.0),
                range: 12.0,
                ..PointLight::default()
            })
            .collect(),
        spot_lights: Vec::new(),
        shadow_maps: ShadowMapSizes::default(),
        environment: None,
        fog: None,
        overlay: Vec::new(),
    };

    let wait = || {
        queue.submit([]);
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
    };
    let time = |label: &str, world: &World, listed: &dyn Fn(usize) -> Vec<Instance>| {
        let mut viewport = Viewport::new(renderer, 1280, 720);
        let mut samples = Vec::with_capacity(FRAMES);
        for f in 0..FRAMES + 20 {
            let instances = listed(f);
            let start = Instant::now();
            renderer.prepare(&mut viewport, world, &scene, &instances);
            if f >= 20 {
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            wait();
        }
        samples.sort_by(f64::total_cmp);
        let w = viewport.work();
        println!(
            "{label:<10} prepare median {:6.3} ms  p95 {:6.3} ms | {} instances ({} written), {} views, \
             {} bounds tests, visible {} / {}, draws {} / {}, upload {:.2} MB, {} allocations",
            samples[FRAMES / 2],
            samples[FRAMES * 95 / 100],
            w.instances,
            w.instance_writes,
            w.views,
            w.bounds_tests,
            w.visible,
            w.shadow_visible,
            w.draws,
            w.shadow_draws,
            w.upload_bytes as f64 / 1e6,
            w.allocations,
        );
    };

    time("listed", &world, &|f| {
        let mut all = scenery.clone();
        all.extend(units(f));
        all
    });
    for instance in &scenery {
        world.add_instance(*instance);
    }
    time("retained", &world, &units);
}
