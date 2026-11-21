//! Headless contexts created on several threads while another context labels objects (#122).
//!
//! With one wgpu instance per context, the Vulkan loader unloaded the drivers without a
//! physical device during a new instance's enumeration while `vkSetDebugUtilsObjectNameEXT` on
//! another thread walked those drivers: the test binary died with SIGSEGV. Before the shared
//! instance in `headless.rs` this test crashed in 7 of 8 runs on lavapipe.

use render_engine::headless::Headless;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[test]
fn contexts_are_created_while_another_context_names_objects() {
    let gpu = Arc::new(Headless::new().expect("no adapter"));
    let stop = Arc::new(AtomicBool::new(false));
    // Every labelled object goes through `vkSetDebugUtilsObjectNameEXT` in debug builds.
    let namers: Vec<_> = (0..4)
        .map(|_| {
            let (gpu, stop) = (gpu.clone(), stop.clone());
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let _ = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("named"),
                        size: 256,
                        usage: wgpu::BufferUsages::UNIFORM,
                        mapped_at_creation: false,
                    });
                }
            })
        })
        .collect();
    let creators: Vec<_> = (0..6)
        .map(|_| {
            std::thread::spawn(|| {
                for _ in 0..2 {
                    Headless::new().expect("no adapter");
                }
            })
        })
        .collect();
    for creator in creators {
        creator.join().unwrap();
    }
    stop.store(true, Ordering::Relaxed);
    for namer in namers {
        namer.join().unwrap();
    }
}
