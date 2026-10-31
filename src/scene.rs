//! Engine-neutral scene description. The oracle builds the same `Scene` on Bevy and on this
//! renderer, so every field here has a direct Bevy counterpart (listed per field).
//!
//! Units follow Bevy's physical conventions: lengths in metres, directional illuminance in lux,
//! ambient and environment-map brightness in cd/m², exposure as EV100. Colours are linear RGB.

use glam::{Quat, Vec2, Vec3};
use std::sync::Arc;

/// Everything one frame needs.
#[derive(Clone, Debug)]
pub struct Scene {
    pub camera: Camera,
    /// Linear RGB written into the HDR frame where no geometry is drawn (Bevy
    /// `ClearColorConfig::Custom`). Tone mapped with the frame but not exposed, as in Bevy.
    pub clear_color: Vec3,
    pub sun: Option<Sun>,
    pub ambient: Ambient,
    /// Image-based ambient light, added to `ambient` (Bevy `EnvironmentMapLight` on the camera).
    pub environment: Option<EnvironmentMap>,
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
    /// Bevy `ShadowFilteringMethod` on the camera.
    pub shadow_filter: ShadowFilter,
    /// Bevy `Tonemapping` on an HDR camera (`Hdr`).
    pub tonemapping: Tonemapping,
}

/// The display transform applied to the HDR frame (Bevy `Tonemapping`). The default is Bevy's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tonemapping {
    /// Stephen Hill's ACES fit.
    AcesFitted,
    /// Tomasz Stachowiak's Tony McMapface (48³ LUT).
    #[default]
    TonyMcMapface,
    /// Troy Sobotka's AgX with the default-contrast look (32³ LUT).
    AgX,
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

/// Most cascades a sun can have (Bevy `MAX_CASCADES_PER_LIGHT`).
pub const MAX_CASCADES: usize = 4;

/// Cascaded directional shadow maps (Bevy `CascadeShadowConfigBuilder`,
/// `DirectionalLightShadowMap`, the light's shadow biases). The defaults are Bevy's.
#[derive(Clone, Copy, Debug)]
pub struct SunShadows {
    /// Texels per side of every cascade's map (Bevy `DirectionalLightShadowMap::size`).
    pub map_size: u32,
    /// 1 to [`MAX_CASCADES`] (Bevy `num_cascades`).
    pub num_cascades: usize,
    /// View distance where shadows start (Bevy `minimum_distance`).
    pub min_distance: f32,
    /// Far bound of the first cascade; the others are spaced exponentially up to
    /// `max_distance`. Ignored with one cascade (Bevy `first_cascade_far_bound`).
    pub first_cascade_far_bound: f32,
    /// View distance where shadows end (Bevy `maximum_distance`).
    pub max_distance: f32,
    /// Share of each cascade's far bound blended into the next cascade, in `[0, 1)` (Bevy
    /// `overlap_proportion`).
    pub overlap_proportion: f32,
    /// World-space offset towards the light (Bevy `shadow_depth_bias`).
    pub depth_bias: f32,
    /// Offset along the surface normal in shadow texels (Bevy `shadow_normal_bias`).
    pub normal_bias: f32,
}

impl Default for SunShadows {
    fn default() -> Self {
        Self {
            map_size: 2048,
            num_cascades: 4,
            min_distance: 0.1,
            first_cascade_far_bound: 10.0,
            max_distance: 150.0,
            overlap_proportion: 0.2,
            depth_bias: 0.02,
            normal_bias: 1.8,
        }
    }
}

impl SunShadows {
    /// One cascade covering `min_distance..max_distance`.
    pub fn single_cascade(max_distance: f32) -> Self {
        Self {
            num_cascades: 1,
            max_distance,
            ..Self::default()
        }
    }

    /// Far bound of each cascade (Bevy `calculate_cascade_bounds`).
    pub fn far_bounds(&self) -> Vec<f32> {
        assert!(
            (1..=MAX_CASCADES).contains(&self.num_cascades),
            "num_cascades must be 1..={MAX_CASCADES}, was {}",
            self.num_cascades
        );
        if self.num_cascades == 1 {
            return vec![self.max_distance];
        }
        let base = (self.max_distance / self.first_cascade_far_bound)
            .powf(1.0 / (self.num_cascades - 1) as f32);
        (0..self.num_cascades)
            .map(|i| self.first_cascade_far_bound * base.powf(i as f32))
            .collect()
    }

    /// `(near, far)` view distances of each cascade: the first starts at `min_distance`, each
    /// later one at `1 - overlap_proportion` of the previous far bound (Bevy
    /// `build_directional_light_cascades`).
    pub fn cascade_ranges(&self) -> Vec<(f32, f32)> {
        let far = self.far_bounds();
        let near = std::iter::once(self.min_distance)
            .chain(far.iter().map(|f| (1.0 - self.overlap_proportion) * f));
        near.zip(far.iter().copied()).collect()
    }
}

/// How shadow maps are filtered (Bevy `ShadowFilteringMethod`, a camera setting). The default is
/// Bevy's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShadowFilter {
    /// One hardware 2×2 PCF comparison.
    Hardware2x2,
    /// Castaño's 13-tap fixed Gaussian (9 hardware PCF samples).
    #[default]
    Gaussian,
}

/// Uniform ambient light (Bevy `GlobalAmbientLight`).
#[derive(Clone, Copy, Debug)]
pub struct Ambient {
    pub color: Vec3,
    /// cd/m².
    pub brightness: f32,
}

/// Image-based light from prefiltered cube maps (Bevy `EnvironmentMapLight`), split-sum style:
/// Lambertian irradiance for diffuse, GGX-prefiltered radiance for specular.
#[derive(Clone, Debug)]
pub struct EnvironmentMap {
    /// Irradiance / π per direction: the radiance a white Lambertian surface facing that way
    /// reflects.
    pub diffuse: Arc<Cubemap>,
    /// Prefiltered radiance with a mip chain; level `m` of `n` is filtered for perceptual
    /// roughness `m / (n - 1)`.
    pub specular: Arc<Cubemap>,
    /// Scales both maps to cd/m² (Bevy `intensity`).
    pub intensity: f32,
}

/// Cube-map texels. Faces follow the wgpu/Vulkan order (+X, -X, +Y, -Y, +Z, -Z) and are looked up
/// with the world direction's z negated, like Bevy's (left-handed) cube maps.
#[derive(Clone)]
pub struct Cubemap {
    /// Texels per face side at level 0.
    pub size: u32,
    pub format: wgpu::TextureFormat,
    /// One entry per mip level, each holding its six faces in order.
    pub levels: Vec<Vec<u8>>,
}

impl Cubemap {
    /// Decodes a cube-map KTX2 file (`E5B9G9R9_UFLOAT_PACK32` or `R16G16B16A16_SFLOAT`,
    /// optionally zstd-supercompressed), as written by the glTF IBL Sampler or the oracle's
    /// `bake_ibl`.
    pub fn from_ktx2(bytes: &[u8]) -> Result<Self, crate::ktx::Error> {
        let t = crate::ktx::decode(bytes)?;
        if t.faces != 6 || t.width != t.height || t.depth != 0 {
            return Err(crate::ktx::Error::new(format!(
                "expected a square cube map, got {}×{} with {} faces",
                t.width, t.height, t.faces
            )));
        }
        Ok(Self {
            size: t.width,
            format: t.format,
            levels: t.levels,
        })
    }
}

impl std::fmt::Debug for Cubemap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Cubemap")
            .field("size", &self.size)
            .field("format", &self.format)
            .field("mip_levels", &self.levels.len())
            .finish()
    }
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
    /// Bevy `Sphere::new(radius).mesh().uv(sectors, stacks)`: a UV sphere with its poles on ±Z
    /// and smooth normals.
    Sphere {
        radius: f32,
        sectors: u32,
        stacks: u32,
    },
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
