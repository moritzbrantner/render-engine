//! Engine-neutral scene description. The oracle builds the same `Scene` on Bevy and on this
//! renderer, so every field here has a direct Bevy counterpart (listed per field).
//!
//! Units follow Bevy's physical conventions: lengths in metres, directional illuminance in lux,
//! ambient brightness in cd/m², exposure as EV100. Colours are linear RGB.

use glam::{Quat, Vec2, Vec3};

/// Everything one frame needs.
#[derive(Clone, Debug)]
pub struct Scene {
    pub camera: Camera,
    /// Linear RGB written where no geometry is drawn (Bevy `ClearColorConfig::Custom`). Not
    /// tone mapped, as in Bevy.
    pub clear_color: Vec3,
    pub sun: Option<Sun>,
    pub ambient: Ambient,
    pub meshes: Vec<MeshInstance>,
}

/// A perspective camera with an infinite reverse-Z projection (Bevy `PerspectiveProjection`).
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    pub near: f32,
    /// Bevy `Exposure::ev100`; scene radiance is multiplied by `2^-ev100 / 1.2`.
    pub ev100: f32,
}

impl Camera {
    /// Bevy `Exposure::exposure()`.
    pub fn exposure(&self) -> f32 {
        (-self.ev100).exp2() / 1.2
    }
}

/// A directional light (Bevy `DirectionalLight`).
#[derive(Clone, Copy, Debug)]
pub struct Sun {
    /// Direction the light travels (Bevy: the light transform's forward).
    pub direction: Vec3,
    pub color: Vec3,
    /// Lux.
    pub illuminance: f32,
    pub shadows: Option<SunShadows>,
}

/// A single-cascade directional shadow map (Bevy `CascadeShadowConfig` with one cascade,
/// `DirectionalLightShadowMap`, `ShadowFilteringMethod::Hardware2x2`).
#[derive(Clone, Copy, Debug)]
pub struct SunShadows {
    /// Texels per side.
    pub map_size: u32,
    /// View distance where shadows start (Bevy `minimum_distance`).
    pub min_distance: f32,
    /// View distance where shadows end (Bevy `maximum_distance`).
    pub max_distance: f32,
    /// World-space offset towards the light (Bevy `shadow_depth_bias`).
    pub depth_bias: f32,
    /// Offset along the surface normal in shadow texels (Bevy `shadow_normal_bias`).
    pub normal_bias: f32,
}

impl Default for SunShadows {
    fn default() -> Self {
        Self {
            map_size: 2048,
            min_distance: 0.1,
            max_distance: 30.0,
            depth_bias: 0.02,
            normal_bias: 1.8,
        }
    }
}

/// Uniform ambient light (Bevy `GlobalAmbientLight`).
#[derive(Clone, Copy, Debug)]
pub struct Ambient {
    pub color: Vec3,
    /// cd/m².
    pub brightness: f32,
}

/// A metallic-roughness material (Bevy `StandardMaterial` without textures).
#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// Linear RGB.
    pub base_color: Vec3,
    pub metallic: f32,
    pub perceptual_roughness: f32,
    /// Dielectric specular reflectance, 0.5 = 4% F0.
    pub reflectance: f32,
}

impl Default for Material {
    /// Bevy's `StandardMaterial` defaults.
    fn default() -> Self {
        Self {
            base_color: Vec3::ONE,
            metallic: 0.0,
            perceptual_roughness: 0.5,
            reflectance: 0.5,
        }
    }
}

/// Built-in shapes, sized like their Bevy primitives.
#[derive(Clone, Copy, Debug)]
pub enum Shape {
    /// Bevy `Cuboid::new(x, y, z)`: full side lengths.
    Cuboid(Vec3),
    /// Bevy `Plane3d::default().mesh().size(x, z)`: an XZ plane facing +Y.
    Plane(Vec2),
}

#[derive(Clone, Copy, Debug)]
pub struct MeshInstance {
    pub shape: Shape,
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub material: Material,
}

impl MeshInstance {
    pub fn new(shape: Shape, translation: Vec3, material: Material) -> Self {
        Self {
            shape,
            translation,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            material,
        }
    }

    pub fn rotated(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }
}
