//! Renders frames of moenarch-games/shooter (#105) headless to PNG, and with `--time` uses them
//! as a fixed workload for the light assignment: GPU time per frame with
//! `LightAssignment::PerFragment` against `LightAssignment::Clustered` (evidence, not a gate).
//!
//! The fixtures in `tests/fixtures/shooter/` are real frames of a deterministic scripted bot
//! match (nothing from real players), recorded and promoted with the shooter's own tool at
//! moenarch-games/shooter@3bc4e09:
//!
//! ```text
//! cargo run --release -p shooter-server --bin shooter-replay -- record -o dm.shrp
//! cargo run --release -p shooter-server --bin shooter-replay -- promote dm.shrp --target render --to 1422 -o tick1422.json
//! cargo run --release -p shooter-server --bin shooter-replay -- promote dm.shrp --target render --to 1782 -o tick1782.json
//! ```
//!
//! `fixture.rs` documents how the JSON maps onto the engine (units, light intensities).
//!
//! ```sh
//! cargo run --release --example shooter_frame                      # every fixture → target/shooter_frame/<name>.png
//! cargo run --release --example shooter_frame -- frame.json        # any promoted frame
//! cargo run --release --example shooter_frame -- --time            # and the light-assignment timing table
//! ```

mod fixture;

use render_engine::{
    headless::Headless,
    renderer::{ClusterConfig, LightAssignment, Viewport},
    world::World,
};
use std::path::{Path, PathBuf};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const TIMED_FRAMES: usize = 600;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let time = args.iter().any(|a| a == "--time");
    let frames: Vec<(String, String)> = match args.iter().find(|a| !a.starts_with("--")) {
        Some(path) => vec![(
            Path::new(path)
                .file_stem()
                .map_or("frame".into(), |s| s.to_string_lossy().into_owned()),
            std::fs::read_to_string(path).expect("readable frame JSON"),
        )],
        None => fixture::FIXTURES
            .iter()
            .map(|(name, json)| (name.to_string(), json.to_string()))
            .collect(),
    };
    let out = PathBuf::from("target/shooter_frame");
    std::fs::create_dir_all(&out).expect("output directory");

    let gpu = Headless::new().expect("a wgpu adapter");
    println!("adapter: {} ({:?})", gpu.adapter.name, gpu.adapter.backend);
    if time {
        println!(
            "\n| frame | instances | point lights | per-fragment ms | clustered ms |\n| --- | ---: | ---: | ---: | ---: |"
        );
    }
    for (name, json) in frames {
        let mut world = World::new(gpu.renderer());
        let frame = fixture::build(&mut world, &json, WIDTH as f32 / HEIGHT as f32);
        let mut viewport = Viewport::new(gpu.renderer(), WIDTH, HEIGHT);
        let image = gpu
            .render(&mut viewport, &world, &frame.scene, &[])
            .expect("frame");
        let path = out.join(format!("{name}.png"));
        image::save_buffer(
            &path,
            &image.rgba,
            image.width,
            image.height,
            image::ColorType::Rgba8,
        )
        .expect("PNG written");
        if !time {
            println!(
                "{name}: {} instances, {} point lights → {}",
                frame.instances,
                frame.scene.point_lights.len(),
                path.display()
            );
            continue;
        }
        let mut ms = |assignment| {
            let mut v = Viewport::new(gpu.renderer(), WIDTH, HEIGHT);
            v.set_light_assignment(gpu.renderer(), assignment)
                .expect("supported light assignment");
            let times = gpu
                .gpu_frame_times(&mut v, &mut world, &frame.scene, &[], TIMED_FRAMES, None)
                .expect("GPU timestamps");
            median(times.total)
        };
        let per_fragment = ms(LightAssignment::PerFragment);
        let clustered = ms(LightAssignment::Clustered(ClusterConfig::default()));
        println!(
            "| {name} | {} | {} | {per_fragment:.3} | {clustered:.3} |",
            frame.instances,
            frame.scene.point_lights.len()
        );
    }
}
