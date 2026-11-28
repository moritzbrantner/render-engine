//! Frame cost of the Greyhaven-sized scene in the browser (#172): `examples/scale_scene/scene.rs`
//! through `render_engine_web`'s device and `timing::frame_times`, the protocol of the native
//! `scale_frame` example. `web/scale/` is its page, `scripts/web-scale.sh` runs it.
#![cfg(target_arch = "wasm32")]

#[path = "../../../examples/scale_scene/scene.rs"]
mod scene;

use render_engine::{renderer::Viewport, world::World};
use render_engine_web::{Gpu, Options, now, timing};
use wasm_bindgen::prelude::*;

/// Times `frames` frames at `width × height` and resolves to one JSON object: the medians, the
/// frame's `PrepareWork`, the adapter, and how long building the world took.
#[wasm_bindgen]
pub async fn run(width: u32, height: u32, frames: u32) -> Result<String, JsValue> {
    std::panic::set_hook(Box::new(|info| {
        web_sys::console::error_1(&JsValue::from_str(&info.to_string()))
    }));
    let gpu = Gpu::new(&Options::default()).await?;
    let start = now();
    let mut world = World::new(&gpu.renderer);
    let built = scene::build(&mut world);
    let world_ms = now() - start;
    let mut viewport = Viewport::new(&gpu.renderer, width, height);
    let units = built.units(0);
    let t = timing::frame_times(&gpu, &mut viewport, &world, &built.scene, &units, frames).await?;
    let w = viewport.work();
    let info = gpu.adapter.get_info();
    let (errors, first_error) = gpu.errors();
    Ok(format!(
        "{{\"adapter\":{:?},\"backend\":\"{:?}\",\"frames\":{},\"gpuMs\":{},\"prepareMs\":{:.3},\"encodeMs\":{:.3},\"submitMs\":{:.3},\"wallMs\":{:.3},\"worldMs\":{:.1},\"instances\":{},\"views\":{},\"visible\":{},\"shadowVisible\":{},\"draws\":{},\"shadowDraws\":{},\"triangles\":{},\"shadowTriangles\":{},\"errors\":{},\"firstError\":{:?}}}",
        format!("{} {}", info.name, info.driver).trim(),
        info.backend,
        t.frames,
        t.gpu_ms.map_or("null".into(), |g| format!("{g:.3}")),
        t.prepare_ms,
        t.encode_ms,
        t.submit_ms,
        t.wall_ms,
        world_ms,
        w.instances,
        w.views,
        w.visible,
        w.shadow_visible,
        w.draws,
        w.shadow_draws,
        w.cpu_triangles,
        w.cpu_shadow_triangles,
        errors,
        first_error.unwrap_or_default(),
    ))
}
