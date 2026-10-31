//! render-engine: lighting (and later particles) for the owner's games, built on wgpu and
//! measured against Bevy. See `docs/technical-decisions.md`.
//!
//! ```no_run
//! use render_engine::{headless::Headless, scene::*};
//! use glam::{Vec2, Vec3};
//!
//! let scene = Scene {
//!     camera: Camera { eye: Vec3::new(6.0, 5.0, 8.0), target: Vec3::ZERO, fov_y: 0.785, near: 0.1, ev100: 9.7, shadow_filter: ShadowFilter::Gaussian },
//!     clear_color: Vec3::new(0.3, 0.5, 0.8),
//!     sun: Some(Sun { direction: Vec3::new(-0.4, -1.0, -0.3), color: Vec3::ONE, illuminance: 10_000.0, shadows: Some(SunShadows::default()) }),
//!     ambient: Ambient { color: Vec3::ONE, brightness: 500.0 },
//!     meshes: vec![MeshInstance::new(Shape::Plane(Vec2::splat(20.0)), Vec3::ZERO, Material::default())],
//! };
//! let gpu = Headless::new()?;
//! let frame = gpu.render(&gpu.prepare(&scene, 1280, 720))?;
//! assert_eq!(frame.rgba.len(), 1280 * 720 * 4);
//! # Ok::<(), render_engine::headless::Error>(())
//! ```

pub use glam;
pub use wgpu;

pub mod headless;
pub mod mesh;
pub mod renderer;
pub mod scene;
