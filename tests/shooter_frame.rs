//! Shooter frames as scene fixtures (#105): each checked-in frame of moenarch-games/shooter,
//! built by `examples/shooter_frame`, renders on any adapter (lavapipe in CI) to a frame that is
//! not just the clear colour, and its weapon lights brighten it.

#[path = "../examples/shooter_frame/fixture.rs"]
mod fixture;

use render_engine::{headless::Headless, renderer::Viewport, world::World};

const WIDTH: u32 = 192;
const HEIGHT: u32 = 108;

fn brightness(rgba: &[u8]) -> u64 {
    rgba.as_chunks::<4>()
        .0
        .iter()
        .map(|p| u64::from(p[0]) + u64::from(p[1]) + u64::from(p[2]))
        .sum()
}

#[test]
fn shooter_frames_render_and_their_lights_contribute() {
    let gpu = Headless::new().expect("a wgpu adapter");
    for (name, json) in fixture::FIXTURES {
        let mut world = World::new(gpu.renderer());
        let frame = fixture::build(&mut world, json, WIDTH as f32 / HEIGHT as f32);
        assert!(frame.instances > 0, "{name}: the frame has geometry");
        assert!(
            !frame.scene.point_lights.is_empty(),
            "{name}: the frame has weapon lights"
        );
        let mut viewport = Viewport::new(gpu.renderer(), WIDTH, HEIGHT);
        let lit = gpu
            .render(&mut viewport, &world, &frame.scene, &[])
            .expect("frame");
        let first = &lit.rgba[..4];
        assert!(
            lit.rgba.as_chunks::<4>().0.iter().any(|p| p[..] != *first),
            "{name}: the frame is a single colour"
        );

        let mut dark = frame.scene.clone();
        dark.point_lights.clear();
        let unlit = gpu
            .render(&mut viewport, &world, &dark, &[])
            .expect("frame without point lights");
        let (with, without) = (brightness(&lit.rgba), brightness(&unlit.rgba));
        assert!(
            with > without,
            "{name}: point lights add no light ({with} vs {without})"
        );
    }
}
