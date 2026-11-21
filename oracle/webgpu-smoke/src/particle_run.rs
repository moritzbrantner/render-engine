//! A D9 particle scenario on render-engine, as the oracle runs it (`run_engine_particles` in
//! `oracle/src/main.rs`): the same spawn alignment with Bevy, triggered bursts, consumer-fed
//! snapshots replayed from Bevy's spawn counts, and every frame's spawn and alive counts. Shared
//! by the browser (`lib.rs`, wasm32) and the native baseline (`bin/compare.rs`), so both run the
//! same frames; only the capture differs. The alive counts come from the async
//! `World::read_alive_particles` (#129).

use crate::{content, particles};
use render_engine::{
    glam::Mat4,
    renderer::{Renderer, Viewport},
    scene::Scene,
    wgpu,
    world::{Instance, World},
};

/// What Bevy's run of the scenario fixed (`<name>.bevy.json` from the oracle): the update in
/// which its spawners first spawn, and its spawn count per emitter and frame.
pub struct BevyRun {
    pub first_spawn_update: u32,
    pub spawns: Vec<Vec<u32>>,
}

/// One run: the plate (no particles), each capture's frame and pixels, and per emitter and frame
/// the spawn and alive counts.
pub struct Run {
    pub plate: Vec<u8>,
    pub captures: Vec<(u32, Vec<u8>)>,
    pub spawn_counts: Vec<Vec<u32>>,
    pub alive: Vec<Vec<u32>>,
}

/// Runs `s` with `depth_fade` metres of soft fade on every emitter when given (soft particles
/// on any scenario, against the same run natively). `capture` prepares and renders a frame to
/// RGBA8.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    renderer: &Renderer,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    s: &particles::ParticleScenario,
    bevy: &BevyRun,
    depth_fade: Option<f32>,
    mut capture: impl AsyncFnMut(&mut Viewport, &World, &Scene, &[Instance]) -> Result<Vec<u8>, String>,
) -> Result<Run, String> {
    use particles::Driver;
    let (mut world, instances, textures) = content::upload(renderer, &s.assets, &s.objects);
    let emitters: Vec<_> = s
        .emitters
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let mut desc = e.engine(s.seed.wrapping_add(i as u32), &textures);
            if let Some(d) = depth_fade {
                desc.depth_fade = d;
            }
            let id = world.add_emitter(&desc);
            world.set_emitter_transform(id, Mat4::from_translation(e.translation));
            (id, desc)
        })
        .collect();
    let mut viewport = Viewport::new(renderer, s.width, s.height);
    let plate = capture(&mut viewport, &world, &s.scene, &instances).await?;
    let step = |world: &mut World| {
        let mut encoder = device.create_command_encoder(&Default::default());
        world.step_particles(&mut encoder, particles::DT, None);
        queue.submit([encoder.finish()]);
    };
    for _ in 1..bevy.first_spawn_update {
        step(&mut world);
    }
    let mut first_seeds: Vec<u32> = (0..emitters.len())
        .map(|i| s.seed.wrapping_add(i as u32))
        .collect();
    for _ in 1..bevy.first_spawn_update {
        first_seeds = first_seeds
            .into_iter()
            .map(render_engine::particles::next_seed)
            .collect();
    }
    for ((id, desc), e) in emitters.iter().zip(&s.emitters) {
        world.set_emitter(
            *id,
            &render_engine::particles::Emitter {
                active: e.driver == Driver::Spawner,
                ..desc.clone()
            },
        );
    }
    let last = s.captures.iter().map(|c| c.frame).max().unwrap_or(0);
    let mut out = Run {
        plate,
        captures: Vec::new(),
        spawn_counts: vec![Vec::new(); emitters.len()],
        alive: vec![Vec::new(); emitters.len()],
    };
    for frame in 1..=last {
        for ((id, _), e) in emitters.iter().zip(&s.emitters) {
            if let (Driver::Trigger, particles::Spawn::Burst { count }, 1) =
                (e.driver, e.spawn, frame)
            {
                world.burst(*id, count, Mat4::from_translation(e.translation));
            }
        }
        step(&mut world);
        let mut encoder = device.create_command_encoder(&Default::default());
        for (i, ((id, _), e)) in emitters.iter().zip(&s.emitters).enumerate() {
            if e.driver == Driver::Snapshot {
                let states = e.replay(first_seeds[i], &bevy.spawns[i], frame);
                world.set_particles(&mut encoder, *id, &states);
            }
        }
        queue.submit([encoder.finish()]);
        let alive = world.read_alive_particles().await;
        for (i, ((id, _), e)) in emitters.iter().zip(&s.emitters).enumerate() {
            out.spawn_counts[i].push(if e.driver == Driver::Snapshot {
                bevy.spawns[i][frame as usize - 1]
            } else {
                world.spawn_count(*id)
            });
            out.alive[i].push(alive.iter().find(|a| a.0 == *id).map_or(0, |a| a.1));
        }
        if s.captures.iter().any(|c| c.frame == frame) {
            let rgba = capture(&mut viewport, &world, &s.scene, &instances).await?;
            out.captures.push((frame, rgba));
        }
    }
    Ok(out)
}

/// Whether the run's spawn counts are Bevy's and its alive counts the ones those spawns and the
/// lifetimes imply (the oracle's population gate).
pub fn population_ok(s: &particles::ParticleScenario, bevy: &BevyRun, run: &Run) -> bool {
    s.emitters.iter().enumerate().all(|(i, e)| {
        let expected: Vec<u32> = particles::population(&bevy.spawns[i], e.lifetime, e.capacity)
            .iter()
            .map(|p| p.alive)
            .collect();
        run.spawn_counts[i] == bevy.spawns[i] && run.alive[i] == expected
    })
}
