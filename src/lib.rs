//! render-engine: lighting and particles for the owner's games, built on wgpu and
//! measured against Bevy. See `docs/technical-decisions.md` and, for consumers,
//! `docs/consumer-integration.md`.
//!
//! The consumer owns the device, the window and the frame. It uploads meshes, textures and
//! materials once into a [`world::World`], then each frame describes the camera and lights as a
//! [`scene::Scene`], lists [`world::Instance`]s and has the [`renderer::Renderer`] encode into its
//! own encoder and target. [`headless::Headless`] does the same offscreen.
//!
//! ```no_run
//! use render_engine::{headless::Headless, material::*, mesh::*, renderer::Viewport, scene::*, world::*};
//! use glam::{Mat4, Vec2, Vec3};
//!
//! let gpu = Headless::new()?;
//! let mut world = World::new(gpu.renderer());
//! let ground = world.add_mesh(&MeshData::shape(Shape::Plane(Vec2::splat(20.0))));
//! let grey = world.add_material(&StandardMaterial::default());
//! let scene = Scene {
//!     camera: Camera { eye: Vec3::new(6.0, 5.0, 8.0), target: Vec3::ZERO, fov_y: 0.785, near: 0.1, ev100: 9.7, shadow_filter: ShadowFilter::Gaussian, tonemapping: Tonemapping::TonyMcMapface },
//!     clear_color: Vec3::ZERO,
//!     // The game's clock picks the hour; the sky filters raw sunlight through the atmosphere.
//!     sun: Some(Sun { direction: -DayCycle::default().to_sun(16.0), color: Vec3::ONE, illuminance: RAW_SUNLIGHT, shadows: Some(SunShadows::default()) }),
//!     moon: None,
//!     sky: Some(Sky::default()),
//!     ambient: Ambient { color: Vec3::ONE, brightness: 0.0 },
//!     point_lights: vec![],
//!     spot_lights: vec![],
//!     shadow_maps: ShadowMapSizes::default(),
//!     environment: None,
//!     fog: None,
//! };
//! let instances = [Instance {
//!     mesh: ground,
//!     material: grey,
//!     world_from_local: Mat4::IDENTITY,
//!     skin: None,
//! }];
//! let mut viewport = Viewport::new(gpu.renderer(), 1280, 720);
//! let frame = gpu.render(&mut viewport, &world, &scene, &instances)?;
//! assert_eq!(frame.rgba.len(), 1280 * 720 * 4);
//! # Ok::<(), render_engine::headless::Error>(())
//! ```

pub use glam;
pub use wgpu;

mod cull;
pub mod headless;
pub mod ktx;
pub mod material;
pub mod mesh;
mod particle_pass;
mod particle_sim;
pub mod particles;
pub mod renderer;
pub mod scene;
mod sky;
pub mod world;
