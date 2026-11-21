//! The showcase's cameras, shared with the native comparison (`oracle/webgpu-smoke`'s
//! `compare showcase`), so both sides render the same view of a shooter frame.

use glam::Vec3;
use render_engine::scene::Camera;

/// The aspect the frame is built at (`fixture::build`); its 90° field of view is horizontal.
pub const BUILD_ASPECT: f32 = 16.0 / 9.0;

/// The orbit: around the arena's centre (the arena is 40 × 40 m), once every `ORBIT_PERIOD_S`.
const ORBIT_CENTRE: Vec3 = Vec3::new(0.0, 1.5, 0.0);
const ORBIT_RADIUS: f32 = 26.0;
const ORBIT_HEIGHT: f32 = 15.0;
const ORBIT_FOV_Y_DEGREES: f32 = 55.0;
pub const ORBIT_PERIOD_S: f64 = 40.0;

/// Which camera a frame uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum View {
    /// The recorded viewer's eye, as `examples/shooter_frame` renders it.
    Eye,
    /// On the orbit, at this angle in degrees (0 looks from +Z toward the centre).
    Orbit(f32),
}

impl View {
    /// `eye` or the orbit angle in degrees.
    pub fn parse(text: &str) -> Option<Self> {
        if text == "eye" {
            Some(Self::Eye)
        } else {
            text.parse().ok().map(Self::Orbit)
        }
    }

    pub fn tag(self) -> String {
        match self {
            Self::Eye => "eye".into(),
            Self::Orbit(degrees) => format!("orbit{degrees:03.0}"),
        }
    }

    /// The camera for a frame whose own camera is `recorded` (built at [`BUILD_ASPECT`]), on a
    /// target of `aspect` (width / height).
    pub fn camera(self, recorded: &Camera, aspect: f32) -> Camera {
        match self {
            Self::Eye => {
                // Keep the horizontal field of view the frame records.
                let horizontal = 2.0 * ((recorded.fov_y / 2.0).tan() * BUILD_ASPECT).atan();
                Camera {
                    fov_y: 2.0 * ((horizontal / 2.0).tan() / aspect).atan(),
                    ..*recorded
                }
            }
            Self::Orbit(degrees) => {
                let (s, c) = degrees.to_radians().sin_cos();
                Camera {
                    eye: ORBIT_CENTRE + Vec3::new(s * ORBIT_RADIUS, ORBIT_HEIGHT, c * ORBIT_RADIUS),
                    target: ORBIT_CENTRE,
                    fov_y: ORBIT_FOV_Y_DEGREES.to_radians(),
                    ..*recorded
                }
            }
        }
    }
}
