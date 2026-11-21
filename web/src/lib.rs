//! render-engine in the browser on WebGPU (#128): the pieces a page needs around the engine's
//! own [`Renderer`], and nothing about scenes. A consumer builds its `World` and `Scene` exactly
//! as natively (D7); this crate gives it a device, a canvas to draw into and a frame loop.
//!
//! - [`Gpu::new`]: async adapter and device creation on the browser's WebGPU, with the optional
//!   features the renderer uses where the adapter has them, and the [`Renderer`].
//! - [`Canvas`]: the `<canvas>` surface. Browsers offer `rgba8unorm` or `bgra8unorm`, not an
//!   sRGB format, so the surface is configured as the browser's preferred canvas format with an
//!   sRGB view, which the renderer writes ([`Gpu::color_format`]). The assessment found the sRGB
//!   view (#75); the preferred format matters because Playwright's Firefox 159 crashes on an
//!   `rgba8unorm` canvas on Linux, where it prefers `bgra8unorm` (#128). Its
//!   size follows the element's CSS box times `devicePixelRatio`, capped by
//!   [`Options::max_pixel_ratio`] and the device's texture limit.
//! - [`start`]: a `requestAnimationFrame` loop around an [`App`]. When the device is lost it
//!   reports the loss ([`App::device_lost`]), creates a new device, surface and renderer and
//!   calls [`App::init`] again, so the app rebuilds its GPU resources; nothing else restarts.
//! - [`timing::frame_times`]: the async counterpart of `Headless::gpu_frame_times` (GPU
//!   timestamps plus `prepare`, `encode` and submit on the CPU), for measuring a page.
//!
//! The library is empty on targets other than `wasm32`.
#![cfg(target_arch = "wasm32")]

pub mod timing;

pub use render_engine;
use render_engine::renderer::Renderer;
pub use render_engine::wgpu;
use std::{
    cell::{Cell, RefCell},
    fmt,
    rc::Rc,
    sync::{Arc, Mutex},
};
use wasm_bindgen::{JsCast, prelude::*};

/// `navigator.gpu.getPreferredCanvasFormat()`: `bgra8unorm` or `rgba8unorm`, by the spec.
pub fn preferred_canvas_format() -> wgpu::TextureFormat {
    let preferred = (|| {
        let gpu = js_sys::Reflect::get(&web_sys::window()?.navigator(), &"gpu".into()).ok()?;
        let get = js_sys::Reflect::get(&gpu, &"getPreferredCanvasFormat".into()).ok()?;
        get.dyn_ref::<js_sys::Function>()?
            .call0(&gpu)
            .ok()?
            .as_string()
    })();
    match preferred.as_deref() {
        Some("bgra8unorm") => wgpu::TextureFormat::Bgra8Unorm,
        _ => wgpu::TextureFormat::Rgba8Unorm,
    }
}

/// Optional device features the renderer uses where the adapter has them (as `Headless::new`
/// asks for them natively): timestamps, unclipped shadow depth, the sky's dual-source blend,
/// filterable float LUTs and indirect first instance.
pub const RENDERER_FEATURES: wgpu::Features = wgpu::Features::TIMESTAMP_QUERY
    .union(wgpu::Features::DEPTH_CLIP_CONTROL)
    .union(wgpu::Features::DUAL_SOURCE_BLENDING)
    .union(wgpu::Features::FLOAT32_FILTERABLE)
    .union(wgpu::Features::INDIRECT_FIRST_INSTANCE);

/// Why the browser build could not start.
#[derive(Clone, Debug)]
pub enum Error {
    /// No WebGPU adapter (no `navigator.gpu`, WebGPU disabled, or the GPU is blocklisted).
    NoAdapter(String),
    /// The adapter refused the device.
    Device(String),
    /// The canvas could not be configured for WebGPU.
    Surface(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter(e) => write!(f, "no WebGPU adapter: {e}"),
            Self::Device(e) => write!(f, "WebGPU device request failed: {e}"),
            Self::Surface(e) => write!(f, "canvas surface: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<Error> for JsValue {
    fn from(e: Error) -> Self {
        js_sys::Error::new(&e.to_string()).into()
    }
}

/// How [`Gpu::new`] and [`Canvas`] set up.
#[derive(Clone, Debug)]
pub struct Options {
    pub power_preference: wgpu::PowerPreference,
    /// Requested where the adapter has them; the rest are left out.
    pub optional_features: wgpu::Features,
    pub limits: wgpu::Limits,
    /// The canvas renders at most this many device pixels per CSS pixel: a 3× phone display
    /// costs 9× the fragments of 1×.
    pub max_pixel_ratio: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            power_preference: wgpu::PowerPreference::HighPerformance,
            optional_features: RENDERER_FEATURES,
            limits: wgpu::Limits::default(),
            max_pixel_ratio: 2.0,
        }
    }
}

/// The device was lost: a GPU process crash or reset, a driver update, or `destroy()`.
#[derive(Clone, Debug)]
pub struct DeviceLost {
    pub reason: wgpu::DeviceLostReason,
    pub message: String,
}

/// Milliseconds spent in each step of [`Gpu::new`].
#[derive(Clone, Copy, Debug, Default)]
pub struct StartupTimes {
    pub adapter_ms: f64,
    pub device_ms: f64,
    pub renderer_ms: f64,
}

/// A WebGPU device and the engine's renderer on it.
pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub renderer: Renderer,
    /// The format the renderer writes: the sRGB form of [`preferred_canvas_format`], so a
    /// [`Canvas`] can be its target.
    pub color_format: wgpu::TextureFormat,
    pub startup: StartupTimes,
    lost: Arc<Mutex<Option<DeviceLost>>>,
    errors: Arc<Mutex<Errors>>,
}

#[derive(Default)]
struct Errors {
    count: u32,
    last: Option<String>,
}

/// `performance.now()`, milliseconds since the page's time origin.
pub fn now() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map_or(0.0, |p| p.now())
}

impl Gpu {
    /// Requests a WebGPU adapter and device and creates the renderer for the sRGB view of the
    /// browser's preferred canvas format ([`Gpu::color_format`]).
    ///
    /// Validation errors do not panic (wgpu's default): they go to the console and are counted
    /// ([`Gpu::errors`]), so one bad frame does not abort the module.
    pub async fn new(options: &Options) -> Result<Self, Error> {
        let start = now();
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::BROWSER_WEBGPU;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: options.power_preference,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| Error::NoAdapter(e.to_string()))?;
        let adapter_ms = now() - start;
        let start = now();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("render-engine-web"),
                required_features: adapter.features() & options.optional_features,
                required_limits: options.limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Device(e.to_string()))?;
        let device_ms = now() - start;

        let lost = Arc::new(Mutex::new(None));
        let sink = lost.clone();
        device.set_device_lost_callback(move |reason, message| {
            web_sys::console::warn_1(&format!("WebGPU device lost ({reason:?}): {message}").into());
            sink.lock()
                .expect("single-threaded")
                .get_or_insert(DeviceLost { reason, message });
        });
        let errors = Arc::new(Mutex::new(Errors::default()));
        let sink = errors.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let text = e.to_string();
            web_sys::console::error_1(&text.clone().into());
            let mut errors = sink.lock().expect("single-threaded");
            errors.count += 1;
            errors.last = Some(text);
        }));

        let start = now();
        let color_format = preferred_canvas_format().add_srgb_suffix();
        let renderer = Renderer::new(&device, &queue, color_format);
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            renderer,
            color_format,
            startup: StartupTimes {
                adapter_ms,
                device_ms,
                renderer_ms: now() - start,
            },
            lost,
            errors,
        })
    }

    /// Destroys the device as a loss would. The loss is recorded first, so the loop stops drawing
    /// to the canvas at once instead of when the browser's `lost` promise resolves.
    pub fn destroy(&self) {
        self.lost
            .lock()
            .expect("single-threaded")
            .get_or_insert(DeviceLost {
                reason: wgpu::DeviceLostReason::Destroyed,
                message: "destroyed by the page".into(),
            });
        self.device.destroy();
    }

    /// Set once the device is lost; a lost device's work is silently dropped.
    pub fn lost(&self) -> Option<DeviceLost> {
        self.lost.lock().expect("single-threaded").clone()
    }

    /// Uncaptured validation and out-of-memory errors so far, and the last one's text.
    pub fn errors(&self) -> (u32, Option<String>) {
        let errors = self.errors.lock().expect("single-threaded");
        (errors.count, errors.last.clone())
    }
}

/// A `<canvas>` configured as the renderer's colour target.
pub struct Canvas {
    element: web_sys::HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    max_pixel_ratio: f64,
}

impl Canvas {
    /// Configures `element` for `gpu`'s device at its current CSS size × pixel ratio.
    pub fn new(
        gpu: &Gpu,
        element: web_sys::HtmlCanvasElement,
        max_pixel_ratio: f64,
    ) -> Result<Self, Error> {
        let surface = gpu
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(element.clone()))
            .map_err(|e| Error::Surface(e.to_string()))?;
        let formats = surface.get_capabilities(&gpu.adapter).formats;
        let format = gpu.color_format.remove_srgb_suffix();
        if !formats.contains(&format) {
            return Err(Error::Surface(format!(
                "the canvas offers {formats:?}, not {format:?}"
            )));
        }
        let mut canvas = Self {
            element,
            surface,
            config: wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: 0,
                height: 0,
                present_mode: wgpu::PresentMode::Fifo,
                desired_maximum_frame_latency: 2,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                view_formats: vec![gpu.color_format],
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
            max_pixel_ratio,
        };
        canvas.resize(gpu);
        Ok(canvas)
    }

    pub fn element(&self) -> &web_sys::HtmlCanvasElement {
        &self.element
    }

    /// The pixel ratio the canvas renders at: `devicePixelRatio`, capped.
    pub fn pixel_ratio(&self) -> f64 {
        let dpr = web_sys::window().map_or(1.0, |w| w.device_pixel_ratio());
        dpr.min(self.max_pixel_ratio).max(f64::MIN_POSITIVE)
    }

    /// The size in device pixels the element's CSS box asks for, at least 1 × 1 and at most
    /// the device's 2D texture limit.
    pub fn wanted_size(&self, gpu: &Gpu) -> (u32, u32) {
        let ratio = self.pixel_ratio();
        let max = gpu.device.limits().max_texture_dimension_2d;
        let px = |css: i32| ((f64::from(css) * ratio).round() as u32).clamp(1, max);
        (
            px(self.element.client_width()),
            px(self.element.client_height()),
        )
    }

    /// The configured size in device pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Follows the element's CSS size and the pixel ratio: reconfigures the surface (and the
    /// canvas's `width`/`height`) when they changed. True if it did.
    pub fn resize(&mut self, gpu: &Gpu) -> bool {
        let (width, height) = self.wanted_size(gpu);
        if (width, height) == self.size() {
            return false;
        }
        self.element.set_width(width);
        self.element.set_height(height);
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&gpu.device, &self.config);
        true
    }

    /// The next canvas texture and its sRGB view, or `None` for a frame to skip (the surface is
    /// reconfigured where that helps).
    pub fn frame(&mut self, gpu: &Gpu) -> Option<SurfaceFrame> {
        let texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&gpu.device, &self.config);
                return None;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return None,
        };
        let view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(gpu.color_format),
            ..Default::default()
        });
        Some(SurfaceFrame { texture, view })
    }
}

/// One canvas texture: render into `view`, then [`SurfaceFrame::present`].
pub struct SurfaceFrame {
    texture: wgpu::SurfaceTexture,
    pub view: wgpu::TextureView,
}

impl SurfaceFrame {
    pub fn present(self, gpu: &Gpu) {
        gpu.queue.present(self.texture);
    }
}

/// What a frame of [`start`]'s loop is.
#[derive(Clone, Copy, Debug)]
pub struct FrameInfo {
    /// `requestAnimationFrame`'s timestamp (or `performance.now()` for [`Running::render_now`]).
    pub time_ms: f64,
    /// Since the previous frame; 0 on the first frame of a device.
    pub dt_ms: f64,
    /// The colour target's size in device pixels.
    pub width: u32,
    pub height: u32,
    /// The canvas changed size (or this is the first frame of a device).
    pub resized: bool,
    /// Which device this is: 1 for the first, one more after each loss.
    pub generation: u32,
}

/// A page's renderer state, driven by [`start`].
///
/// The app itself lives for the whole loop; its [`App::State`] holds what lives on one device
/// (worlds, viewports, other GPU resources) and is rebuilt by [`App::init`] after a loss.
pub trait App: 'static {
    type State;

    /// Creates the device's resources. Called once at start and again after every device loss,
    /// with the new device.
    fn init(&mut self, gpu: &Gpu, size: (u32, u32)) -> Self::State;

    /// Draws one frame into `target` (prepare, encode, submit); the loop presents it.
    fn frame(
        &mut self,
        state: &mut Self::State,
        gpu: &Gpu,
        target: &wgpu::TextureView,
        info: &FrameInfo,
    );

    /// The device was lost. The loop drops the state and starts creating a new device.
    fn device_lost(&mut self, _lost: &DeviceLost) {}

    /// A device is ready and [`App::init`] ran (`generation` as in [`FrameInfo`]).
    fn ready(&mut self, _gpu: &Gpu, _generation: u32) {}

    /// Re-initialisation after a loss failed for good; the loop stops.
    fn failed(&mut self, _error: &Error) {}
}

struct Live<S> {
    gpu: Gpu,
    canvas: Canvas,
    state: S,
    last_time: Option<f64>,
    fresh: bool,
}

struct Inner<A: App> {
    app: RefCell<A>,
    live: RefCell<Option<Live<A::State>>>,
    element: web_sys::HtmlCanvasElement,
    options: Options,
    generation: Cell<u32>,
    stopped: Cell<bool>,
    paused: Cell<bool>,
    /// The pending `requestAnimationFrame` request.
    pending: Cell<Option<i32>>,
    raf: RefCell<Option<AnimationFrame>>,
    /// The loss check on a timer, and its interval id.
    watchdog: RefCell<Option<Watchdog>>,
}

/// The `requestAnimationFrame` callback.
type AnimationFrame = Closure<dyn FnMut(f64)>;

/// The timer callback that checks for a loss, and its interval id.
type Watchdog = (Closure<dyn FnMut()>, i32);

/// The loop [`start`] runs. Dropping it does not stop the loop; [`Running::stop`] does.
pub struct Running<A: App> {
    inner: Rc<Inner<A>>,
}

impl<A: App> Clone for Running<A> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Times re-initialisation is tried after a loss, waiting 0.5 s × the attempt in between.
const REINIT_ATTEMPTS: u32 = 5;

/// How often the loss is checked besides every animation frame: Firefox 159 stops animation
/// frames for a page whose WebGPU canvas lost its device, so a frame-driven check never runs.
const WATCHDOG_MS: i32 = 250;

/// Creates the device on `element`, calls [`App::init`], and runs [`App::frame`] on every
/// animation frame until [`Running::stop`]. Fails if the first device or surface cannot be
/// created, so a page can fall back (`selectBackend`).
pub async fn start<A: App>(
    element: web_sys::HtmlCanvasElement,
    options: Options,
    app: A,
) -> Result<Running<A>, Error> {
    let inner = Rc::new(Inner {
        app: RefCell::new(app),
        live: RefCell::new(None),
        element,
        options,
        generation: Cell::new(0),
        stopped: Cell::new(false),
        paused: Cell::new(false),
        pending: Cell::new(None),
        raf: RefCell::new(None),
        watchdog: RefCell::new(None),
    });
    let live = create(&inner).await?;
    *inner.live.borrow_mut() = Some(live);
    let weak = Rc::downgrade(&inner);
    *inner.raf.borrow_mut() = Some(Closure::new(move |time: f64| {
        if let Some(inner) = weak.upgrade() {
            inner.pending.set(None);
            tick(&inner, time);
        }
    }));
    let weak = Rc::downgrade(&inner);
    let check: Closure<dyn FnMut()> = Closure::new(move || {
        if let Some(inner) = weak.upgrade()
            && !inner.stopped.get()
            && !inner.paused.get()
        {
            handle_loss(&inner);
        }
    });
    if let Some(window) = web_sys::window()
        && let Ok(id) = window.set_interval_with_callback_and_timeout_and_arguments_0(
            check.as_ref().unchecked_ref(),
            WATCHDOG_MS,
        )
    {
        *inner.watchdog.borrow_mut() = Some((check, id));
    }
    schedule(&inner);
    Ok(Running { inner })
}

async fn create<A: App>(inner: &Inner<A>) -> Result<Live<A::State>, Error> {
    let gpu = Gpu::new(&inner.options).await?;
    let canvas = Canvas::new(&gpu, inner.element.clone(), inner.options.max_pixel_ratio)?;
    let generation = inner.generation.get() + 1;
    inner.generation.set(generation);
    let mut app = inner.app.borrow_mut();
    let state = app.init(&gpu, canvas.size());
    app.ready(&gpu, generation);
    Ok(Live {
        gpu,
        canvas,
        state,
        last_time: None,
        fresh: true,
    })
}

/// Requests the next animation frame, unless one is pending.
fn schedule<A: App>(inner: &Inner<A>) {
    if inner.stopped.get() || inner.paused.get() || inner.pending.get().is_some() {
        return;
    }
    if let (Some(window), Some(callback)) = (web_sys::window(), inner.raf.borrow().as_ref())
        && let Ok(id) = window.request_animation_frame(callback.as_ref().unchecked_ref())
    {
        inner.pending.set(Some(id));
    }
}

/// Drops a pending animation frame request (one the browser may never serve).
fn cancel<A: App>(inner: &Inner<A>) {
    if let (Some(window), Some(id)) = (web_sys::window(), inner.pending.take()) {
        let _ = window.cancel_animation_frame(id);
    }
}

fn tick<A: App>(inner: &Rc<Inner<A>>, time: f64) {
    if inner.stopped.get() || inner.paused.get() || handle_loss(inner) {
        return;
    }
    draw(inner, time);
    schedule(inner);
}

/// If the live device is lost: drops everything on it, tells the app and starts making a new
/// one. True if it was lost.
fn handle_loss<A: App>(inner: &Rc<Inner<A>>) -> bool {
    let lost = inner
        .live
        .borrow()
        .as_ref()
        .and_then(|live| live.gpu.lost());
    let Some(lost) = lost else {
        return false;
    };
    inner.live.borrow_mut().take();
    cancel(inner);
    inner.app.borrow_mut().device_lost(&lost);
    let inner = inner.clone();
    wasm_bindgen_futures::spawn_local(async move { reinitialise(inner).await });
    true
}

async fn reinitialise<A: App>(inner: Rc<Inner<A>>) {
    let mut error = None;
    for attempt in 1..=REINIT_ATTEMPTS {
        if inner.stopped.get() {
            return;
        }
        match create(&inner).await {
            Ok(live) => {
                *inner.live.borrow_mut() = Some(live);
                cancel(&inner);
                schedule(&inner);
                return;
            }
            Err(e) => {
                web_sys::console::warn_1(&format!("re-initialisation {attempt}: {e}").into());
                error = Some(e);
                sleep(500.0 * f64::from(attempt)).await;
            }
        }
    }
    inner.stopped.set(true);
    if let Some(e) = error {
        inner.app.borrow_mut().failed(&e);
    }
}

async fn sleep(ms: f64) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        if let Some(w) = web_sys::window() {
            let _ = w.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms as i32);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

/// One frame on the live device, if there is one. False if nothing was presented.
fn draw<A: App>(inner: &Inner<A>, time: f64) -> bool {
    let mut live = inner.live.borrow_mut();
    let Some(live) = live.as_mut() else {
        return false;
    };
    let resized = live.canvas.resize(&live.gpu) || live.fresh;
    let Some(frame) = live.canvas.frame(&live.gpu) else {
        return false;
    };
    let (width, height) = live.canvas.size();
    let info = FrameInfo {
        time_ms: time,
        dt_ms: live.last_time.map_or(0.0, |last| (time - last).max(0.0)),
        width,
        height,
        resized,
        generation: inner.generation.get(),
    };
    live.fresh = false;
    live.last_time = Some(time);
    inner
        .app
        .borrow_mut()
        .frame(&mut live.state, &live.gpu, &frame.view, &info);
    frame.present(&live.gpu);
    true
}

impl<A: App> Running<A> {
    /// The app, and the device and its state while one is live (not between a loss and the
    /// next device, nor inside [`Running::suspend`]).
    pub fn with<R>(&self, f: impl FnOnce(&mut A, Option<(&Gpu, &mut A::State)>) -> R) -> R {
        let mut app = self.inner.app.borrow_mut();
        let mut live = self.inner.live.borrow_mut();
        f(&mut app, live.as_mut().map(|l| (&l.gpu, &mut l.state)))
    }

    /// Draws and presents one frame now, outside the animation frame: a page can read the
    /// canvas back in the same task (WebGPU canvas contents are only kept until it ends).
    pub fn render_now(&self) -> bool {
        draw(&self.inner, now())
    }

    /// Runs `f` with the device and state while the loop is paused, then resumes it: for work
    /// that awaits the GPU (timing runs, readbacks) without a frame in between.
    pub async fn suspend<R>(&self, f: impl AsyncFnOnce(&Gpu, &mut A::State) -> R) -> Option<R> {
        self.inner.paused.set(true);
        let taken = self.inner.live.borrow_mut().take();
        let result = match taken {
            Some(mut live) => {
                let r = f(&live.gpu, &mut live.state).await;
                live.last_time = None;
                *self.inner.live.borrow_mut() = Some(live);
                Some(r)
            }
            None => None,
        };
        self.inner.paused.set(false);
        schedule(&self.inner);
        result
    }

    /// Which device the loop is on: 1 for the first, one more after each loss.
    pub fn generation(&self) -> u32 {
        self.inner.generation.get()
    }

    /// Stops the loop and drops the device's state.
    pub fn stop(&self) {
        self.inner.stopped.set(true);
        cancel(&self.inner);
        if let (Some(window), Some((_, id))) =
            (web_sys::window(), self.inner.watchdog.borrow_mut().take())
        {
            window.clear_interval_with_handle(id);
        }
        self.inner.live.borrow_mut().take();
        self.inner.raf.borrow_mut().take();
    }
}
