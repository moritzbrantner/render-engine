//! Microbenchmark for the per-emitter CPU cost of particles (render-engine#85): `n` small burst
//! emitters (10 particles every 0.6 s, the asset-tooling combat spark), the case of hit effects
//! in a crowd fight. Each frame steps them, prepares and encodes a 640×360 frame with no meshes,
//! and submits; the wait for the GPU is outside the timed spans.
//!
//! `cargo run --release --example emitter_cost`

use glam::{Mat4, Vec2, Vec3, Vec4};
use render_engine::{
    headless::{COLOR_FORMAT, Headless},
    material::{Sampler, TextureData},
    particles::*,
    renderer::Viewport,
    scene::*,
    world::World,
};
use std::time::Instant;

const DT: f32 = 1.0 / 60.0;
const FRAMES: usize = 120;

fn spark(texture: render_engine::world::TextureId, seed: u32) -> Emitter {
    Emitter {
        capacity: 20,
        spawner: Spawner::burst(10.0, 0.6),
        seed,
        init: vec![
            InitModifier::PositionSphere {
                center: Vec3::ZERO,
                radius: 0.04,
                dimension: ShapeDimension::Volume,
            },
            InitModifier::Velocity(Value::Uniform {
                min: Vec3::new(-2.7, 0.7, -2.7),
                max: Vec3::new(2.7, 3.0, 2.7),
            }),
            InitModifier::Lifetime(Value::Const(0.28)),
            InitModifier::Color(Value::Const(Vec4::ONE)),
        ],
        accel: Vec3::new(0.0, -6.0, 0.0),
        size_over_life: Some(Gradient::new(vec![
            (0.0, Vec2::splat(0.16)),
            (1.0, Vec2::splat(0.064)),
        ])),
        color_over_life: None,
        orient: Orient::CameraPlane,
        texture: Some(ParticleTexture {
            texture,
            mapping: TextureMapping::Modulate,
            flipbook: Some(Flipbook {
                grid: glam::UVec2::new(4, 4),
                frames: 16,
            }),
        }),
        roundness: 0.0,
        depth_fade: 0.0,
        blend: ParticleBlend::Add,
        lighting: None,
        fog: false,
        active: true,
    }
}

fn scene() -> Scene {
    Scene {
        camera: Camera {
            eye: Vec3::new(0.0, 6.0, 12.0),
            target: Vec3::new(0.0, 0.0, -6.0),
            fov_y: 50f32.to_radians(),
            near: 0.1,
            ev100: 12.0,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::TonyMcMapface,
        },
        clear_color: Vec3::ZERO,
        sun: None,
        moon: None,
        sky: None,
        point_lights: vec![],
        spot_lights: vec![],
        shadow_maps: ShadowMapSizes::default(),
        ambient: Ambient {
            color: Vec3::ONE,
            brightness: 0.0,
        },
        environment: None,
        fog: None,
        overlay: Vec::new(),
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let gpu = Headless::new().expect("a wgpu adapter");
    println!("adapter: {}", gpu.adapter.name);
    let r = gpu.renderer();
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 640,
            height: 360,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let scene = scene();
    println!(
        "{:>9} {:>9} {:>9} {:>9} {:>9} {:>11}",
        "emitters", "step ms", "prepare", "encode", "submit", "µs/emitter"
    );
    for n in [1u32, 10, 100, 500, 1000, 2000] {
        let mut world = World::new(r);
        // Four textures, as a game's handful of hit effects would share.
        let textures: Vec<_> = (0..4)
            .map(|i| {
                world.add_texture(&TextureData {
                    width: 4,
                    height: 4,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    levels: vec![vec![255 - i * 40; 64]],
                    sampler: Sampler::default(),
                })
            })
            .collect();
        let side = (n as f32).sqrt().ceil() as u32;
        for i in 0..n {
            let id = world.add_emitter(&spark(textures[(i % 4) as usize], i));
            let at = Vec3::new(
                (i % side) as f32 * 0.6 - side as f32 * 0.3,
                0.3,
                -((i / side) as f32) * 0.6,
            );
            world.set_emitter_transform(id, Mat4::from_translation(at));
        }
        let mut viewport = Viewport::new(r, 640, 360);
        let mut times = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for frame in 0..FRAMES + 20 {
            let t0 = Instant::now();
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            world.step_particles(&mut encoder, DT, None);
            let t1 = Instant::now();
            r.prepare(&mut viewport, &world, &scene, &[]);
            let t2 = Instant::now();
            r.encode(&mut encoder, &viewport, &world, &view, None);
            let t3 = Instant::now();
            gpu.queue.submit([encoder.finish()]);
            let t4 = Instant::now();
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            if frame >= 20 {
                for (k, (a, b)) in [(t0, t1), (t1, t2), (t2, t3), (t3, t4)].iter().enumerate() {
                    times[k].push((*b - *a).as_secs_f64() * 1e3);
                }
            }
        }
        let [s, p, e, u] = times.map(median);
        println!(
            "{n:>9} {s:>9.3} {p:>9.3} {e:>9.3} {u:>9.3} {:>11.2}",
            (s + p + e + u) * 1e3 / f64::from(n)
        );
    }
}
