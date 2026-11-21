//! The browser showcase (#128): frames of moenarch-games/shooter (`tests/fixtures/shooter/`),
//! built through the public API by `examples/shooter_frame/fixture.rs` (the greybox arena, the
//! fighters, weapon effects and their point lights), rendered live on WebGPU under an orbiting
//! camera with `render-engine-web`. The page (`web/showcase/`) fetches the frame JSON and passes
//! it in, so the wasm carries no fixtures; any promoted `shooter/render-frame/v1` frame works.
//!
//! Built by `scripts/build-web.sh` (cdylib for `wasm32-unknown-unknown`, wasm-bindgen,
//! wasm-opt). Besides the live loop it has what the evidence harness needs
//! (`scripts/web-showcase.sh`): a fixed view rendered on demand, the frame-time protocol and a
//! device-loss trigger.
#![cfg(target_arch = "wasm32")]

#[allow(dead_code)]
#[path = "../../../examples/shooter_frame/fixture.rs"]
mod fixture;
mod view;

use render_engine::{renderer::Viewport, scene::Scene, world::World};
use render_engine_web::{App, DeviceLost, FrameInfo, Gpu, Options, Running, now, wgpu};
use serde_json::{Value, json};
use std::collections::VecDeque;
use view::View;
use wasm_bindgen::prelude::*;

/// Frames the rolling CPU-time and frame-interval averages span.
const ROLLING: usize = 120;

struct Showcase {
    frames: Vec<(String, String)>,
    selected: usize,
    /// A fixed view, or `None` to orbit.
    fixed: Option<View>,
    orbit_degrees: f32,
    on_event: js_sys::Function,
    /// Per frame: interval, prepare, encode, submit (ms).
    rolling: VecDeque<[f64; 4]>,
    first_frame_at: Option<f64>,
    world_ms: f64,
    losses: u32,
}

struct State {
    world: World,
    viewport: Viewport,
    scene: Scene,
    built: usize,
    instances: usize,
}

impl Showcase {
    fn emit(&self, kind: &str, detail: Value) {
        let _ = self.on_event.call2(
            &JsValue::NULL,
            &JsValue::from_str(kind),
            &JsValue::from_str(&detail.to_string()),
        );
    }

    fn build(&mut self, gpu: &Gpu, size: (u32, u32)) -> State {
        let start = now();
        let mut world = World::new(&gpu.renderer);
        let frame = fixture::build(
            &mut world,
            &self.frames[self.selected].1,
            view::BUILD_ASPECT,
        );
        self.world_ms = now() - start;
        State {
            world,
            viewport: Viewport::new(&gpu.renderer, size.0, size.1),
            scene: frame.scene,
            built: self.selected,
            instances: frame.instances,
        }
    }

    fn view(&self) -> View {
        self.fixed.unwrap_or(View::Orbit(self.orbit_degrees))
    }
}

impl App for Showcase {
    type State = State;

    fn init(&mut self, gpu: &Gpu, size: (u32, u32)) -> State {
        self.build(gpu, size)
    }

    fn frame(
        &mut self,
        state: &mut State,
        gpu: &Gpu,
        target: &wgpu::TextureView,
        info: &FrameInfo,
    ) {
        if state.built != self.selected {
            *state = self.build(gpu, (info.width, info.height));
        } else if info.resized {
            state
                .viewport
                .resize(&gpu.renderer, info.width, info.height);
        }
        if self.fixed.is_none() {
            let degrees_per_ms = 360.0 / (view::ORBIT_PERIOD_S * 1e3);
            self.orbit_degrees =
                ((f64::from(self.orbit_degrees) + info.dt_ms * degrees_per_ms) % 360.0) as f32;
        }
        let recorded = state.scene.camera;
        state.scene.camera = self
            .view()
            .camera(&recorded, info.width as f32 / info.height as f32);
        let t0 = now();
        gpu.renderer
            .prepare(&mut state.viewport, &state.world, &state.scene, &[]);
        let t1 = now();
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        gpu.renderer
            .encode(&mut encoder, &state.viewport, &state.world, target, None);
        let buffer = encoder.finish();
        let t2 = now();
        gpu.queue.submit([buffer]);
        let t3 = now();
        state.scene.camera = recorded;
        if self.rolling.len() == ROLLING {
            self.rolling.pop_front();
        }
        self.rolling
            .push_back([info.dt_ms, t1 - t0, t2 - t1, t3 - t2]);
        if self.first_frame_at.is_none() {
            self.first_frame_at = Some(now());
        }
    }

    fn device_lost(&mut self, lost: &DeviceLost) {
        self.losses += 1;
        self.emit(
            "lost",
            json!({"reason": format!("{:?}", lost.reason), "message": lost.message}),
        );
    }

    fn ready(&mut self, gpu: &Gpu, generation: u32) {
        let info = gpu.adapter.get_info();
        self.emit(
            "ready",
            json!({
                "generation": generation,
                "adapter": adapter_label(&info),
                "backend": format!("{:?}", info.backend),
            }),
        );
    }

    fn failed(&mut self, error: &render_engine_web::Error) {
        self.emit("failed", json!({"error": error.to_string()}));
    }
}

/// The showcase as the page sees it.
#[wasm_bindgen]
pub struct ShooterShowcase {
    running: Running<Showcase>,
}

#[wasm_bindgen]
impl ShooterShowcase {
    /// Starts on `canvas` with the frames `names[i]` → `frames[i]` (shooter frame JSON).
    /// `on_event(kind, json)` hears `ready`, `lost` and `failed`. Rejects if WebGPU cannot start.
    pub async fn start(
        canvas: web_sys::HtmlCanvasElement,
        names: Vec<String>,
        frames: Vec<String>,
        on_event: js_sys::Function,
    ) -> Result<ShooterShowcase, JsValue> {
        std::panic::set_hook(Box::new(|info| {
            web_sys::console::error_1(&JsValue::from_str(&info.to_string()))
        }));
        if names.is_empty() || names.len() != frames.len() {
            return Err(JsValue::from_str("one name per frame, at least one frame"));
        }
        let app = Showcase {
            frames: names.into_iter().zip(frames).collect(),
            selected: 0,
            fixed: None,
            orbit_degrees: 0.0,
            on_event,
            rolling: VecDeque::with_capacity(ROLLING),
            first_frame_at: None,
            world_ms: 0.0,
            losses: 0,
        };
        let running = render_engine_web::start(canvas, Options::default(), app).await?;
        Ok(ShooterShowcase { running })
    }

    /// Shows frame `index` (rebuilt on the next frame).
    pub fn select(&self, index: usize) {
        self.running.with(|app, _| {
            if index < app.frames.len() {
                app.selected = index;
            }
        });
    }

    /// `"orbit"` to orbit, `"eye"` for the recorded viewer's eye, or an orbit angle in degrees
    /// to hold.
    pub fn set_view(&self, view: &str) -> Result<(), JsValue> {
        let fixed = match view {
            "orbit" => None,
            other => Some(View::parse(other).ok_or_else(|| format!("unknown view {other}"))?),
        };
        self.running.with(|app, _| app.fixed = fixed);
        Ok(())
    }

    /// Draws one frame now, for the page to read the canvas back in the same task.
    pub fn render_now(&self) -> bool {
        self.running.render_now()
    }

    /// Which device the loop is on: 1, plus one per loss.
    pub fn generation(&self) -> u32 {
        self.running.generation()
    }

    /// Destroys the device, as a GPU reset would lose it: the loop re-initialises.
    pub fn lose_device(&self) {
        self.running.with(|_, live| {
            if let Some((gpu, _)) = live {
                gpu.destroy();
            }
        });
    }

    /// Rolling averages, startup and device state, as JSON.
    pub fn stats(&self) -> String {
        let generation = self.running.generation();
        self.running.with(|app, live| {
            let n = app.rolling.len().max(1) as f64;
            let mean = |i: usize| app.rolling.iter().map(|r| r[i]).sum::<f64>() / n;
            let mut stats = json!({
                "frame": app.frames[app.selected].0,
                "view": app.view().tag(),
                "frameIntervalMs": mean(0),
                "prepareMs": mean(1),
                "encodeMs": mean(2),
                "submitMs": mean(3),
                "firstFrameAt": app.first_frame_at,
                "worldMs": app.world_ms,
                "generation": generation,
                "losses": app.losses,
            });
            if let Some((gpu, state)) = live {
                let info = gpu.adapter.get_info();
                let (errors, last_error) = gpu.errors();
                let (width, height) = state.viewport.size();
                let extra = json!({
                    "adapter": adapter_label(&info),
                    "features": format!("{:?}", gpu.device.features()),
                    "width": width,
                    "height": height,
                    "instances": state.instances,
                    "pointLights": state.scene.point_lights.len(),
                    "lightAssignment": format!("{:?}", state.viewport.light_assignment()),
                    "adapterMs": gpu.startup.adapter_ms,
                    "deviceMs": gpu.startup.device_ms,
                    "rendererMs": gpu.startup.renderer_ms,
                    "errors": errors,
                    "lastError": last_error,
                });
                if let (Value::Object(s), Value::Object(e)) = (&mut stats, extra) {
                    s.extend(e);
                }
            }
            stats.to_string()
        })
    }

    /// `render_engine_web::timing::frame_times` of the current frame and view at `width` ×
    /// `height` (offscreen), with the loop paused. JSON.
    pub async fn time(&self, width: u32, height: u32, frames: u32) -> Result<String, JsValue> {
        let view = self.running.with(|app, _| app.view());
        let result = self
            .running
            .suspend(async |gpu: &Gpu, state: &mut State| {
                let mut viewport = Viewport::new(&gpu.renderer, width, height);
                let mut scene = state.scene.clone();
                scene.camera = view.camera(&scene.camera, width as f32 / height as f32);
                render_engine_web::timing::frame_times(
                    gpu,
                    &mut viewport,
                    &state.world,
                    &scene,
                    &[],
                    frames,
                )
                .await
            })
            .await
            .ok_or_else(|| JsValue::from_str("no device"))??;
        Ok(json!({
            "view": view.tag(),
            "width": width,
            "height": height,
            "frames": result.frames,
            "gpuMs": result.gpu_ms,
            "prepareMs": result.prepare_ms,
            "encodeMs": result.encode_ms,
            "submitMs": result.submit_ms,
        })
        .to_string())
    }
}

/// Browsers expose no adapter name to wgpu (only vendor and architecture, which wgpu does not
/// map), so the label falls back to the backend.
fn adapter_label(info: &wgpu::AdapterInfo) -> String {
    if info.name.is_empty() {
        format!("{:?} adapter", info.backend)
    } else {
        info.name.clone()
    }
}
