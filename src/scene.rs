//! Engine-neutral description of a frame's view and lighting. The oracle builds the same `Scene`
//! on Bevy and on this renderer, so every field here has a direct Bevy counterpart (listed per
//! field). The geometry comes separately, as [`Instance`](crate::world::Instance)s of a
//! [`World`](crate::world::World).
//!
//! Units follow Bevy's physical conventions: lengths in metres, directional illuminance in lux,
//! point and spot light power in lumens, ambient and environment-map brightness in cd/m², exposure as EV100. Colours are linear RGB.

use glam::{Vec3, Vec4};
use std::sync::Arc;

/// The camera and lighting of one frame: plain values, rebuilt or edited every frame at no
/// cost. Environment maps are uploaded once per distinct `Arc`.
#[derive(Clone, Debug)]
pub struct Scene {
    pub camera: Camera,
    /// Linear RGB written into the HDR frame where no geometry is drawn (Bevy
    /// `ClearColorConfig::Custom`). Tone mapped with the frame but not exposed, as in Bevy.
    pub clear_color: Vec3,
    pub sun: Option<Sun>,
    /// A second, unshadowed directional light (Bevy: another `DirectionalLight` without shadow
    /// maps), for moonlight. It lights the scene and, with a [`Sky`], scatters in the sky and
    /// shows as a disc.
    pub moon: Option<Moon>,
    /// A physically based sky over everything not drawn, with aerial perspective over what is
    /// (Bevy `Atmosphere` + `AtmosphereSettings` on the camera). The sun and moon colours are
    /// filtered by the atmosphere's transmittance, so give them their light above the
    /// atmosphere (Bevy `lux::RAW_SUNLIGHT` for the sun).
    pub sky: Option<Sky>,
    /// Point and spot lights are shaded in list order, shadow-casting lights first (Bevy sorts
    /// its clustered lights the same way). Every shadow-casting light renders its own maps.
    pub point_lights: Vec<PointLight>,
    pub spot_lights: Vec<SpotLight>,
    pub shadow_maps: ShadowMapSizes,
    pub ambient: Ambient,
    /// Image-based ambient light, added to `ambient` (Bevy `EnvironmentMapLight` on the camera).
    pub environment: Option<EnvironmentMap>,
    /// Distance fog over lit geometry (Bevy `DistanceFog` on the camera). The clear colour is not
    /// fogged, as in Bevy; set it to the fog colour for a fogged horizon.
    pub fog: Option<Fog>,
}

/// Distance fog (Bevy `DistanceFog`), blended into the exposed HDR colour of every lit fragment
/// by its distance to the camera.
#[derive(Clone, Copy, Debug)]
pub struct Fog {
    /// Linear RGB fog colour, not exposed; alpha scales how much fog is blended in (Bevy `color`).
    pub color: Vec4,
    /// Sun inscattering: `rgb × a` times the exposed sun colour is added to the fog colour where
    /// the view looks towards the sun, attenuated by the sun's shadow at the fragment. `a = 0`
    /// turns it off (Bevy `directional_light_color`).
    pub directional_light_color: Vec4,
    /// Sharpness of the sun glow: `max(dot(view, to_sun), 0)^exponent` (Bevy
    /// `directional_light_exponent`).
    pub directional_light_exponent: f32,
    pub falloff: FogFalloff,
}

impl Default for Fog {
    /// Bevy's `DistanceFog` defaults: white, linear from 0 to 100 m, no sun glow.
    fn default() -> Self {
        Self {
            color: Vec4::ONE,
            directional_light_color: Vec4::ZERO,
            directional_light_exponent: 8.0,
            falloff: FogFalloff::Linear {
                start: 0.0,
                end: 100.0,
            },
        }
    }
}

/// How fog builds up with distance `d` (Bevy `FogFalloff`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FogFalloff {
    /// `clamp((d - start) / (end - start), 0, 1)`, in metres.
    Linear { start: f32, end: f32 },
    /// `1 - e^(-d·density)`.
    Exponential { density: f32 },
    /// `1 - e^(-(d·density)²)`.
    ExponentialSquared { density: f32 },
    /// Per channel: the scene colour is attenuated by `1 - e^(-d·extinction)` and the fog colour
    /// added by `1 - e^(-d·inscattering)`.
    Atmospheric {
        extinction: Vec3,
        inscattering: Vec3,
    },
}

impl FogFalloff {
    /// Bevy `REVISED_KOSCHMIEDER_CONTRAST_THRESHOLD`: the contrast at which an object counts as
    /// no longer visible.
    pub const REVISED_KOSCHMIEDER_CONTRAST_THRESHOLD: f32 = 0.05;

    /// Extinction coefficient for which contrast falls to `contrast_threshold` at `visibility`
    /// metres (Bevy `koschmieder`).
    pub fn koschmieder(visibility: f32, contrast_threshold: f32) -> f32 {
        -contrast_threshold.ln() / visibility
    }

    /// Exponential fog that hides objects beyond `visibility` metres (Bevy `from_visibility`).
    pub fn from_visibility(visibility: f32) -> Self {
        Self::Exponential {
            density: Self::koschmieder(visibility, Self::REVISED_KOSCHMIEDER_CONTRAST_THRESHOLD),
        }
    }

    /// Exponential-squared fog that hides objects beyond `visibility` metres (Bevy
    /// `from_visibility_squared`).
    pub fn from_visibility_squared(visibility: f32) -> Self {
        let k = Self::koschmieder(visibility, Self::REVISED_KOSCHMIEDER_CONTRAST_THRESHOLD);
        Self::ExponentialSquared {
            density: (k / visibility).sqrt(),
        }
    }

    /// Atmospheric fog from an extinction and an inscattering colour (linear RGBA; alpha scales
    /// the effect) that hides objects beyond `visibility` metres (Bevy `from_visibility_colors`).
    pub fn from_visibility_colors(visibility: f32, extinction: Vec4, inscattering: Vec4) -> Self {
        use std::f32::consts::E;
        let k = Self::koschmieder(visibility, Self::REVISED_KOSCHMIEDER_CONTRAST_THRESHOLD);
        let pow = |v: Vec3| Vec3::new(v.x.powf(E), v.y.powf(E), v.z.powf(E));
        Self::Atmospheric {
            // Inverted so the colour means what it looks like (blue extinction gives blue, not
            // yellow, results), as in Bevy.
            extinction: pow(Vec3::ONE - extinction.truncate()) * k * extinction.w.powf(E),
            inscattering: pow(inscattering.truncate()) * k * inscattering.w.powf(E),
        }
    }
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

/// An unshadowed directional light: the moon (Bevy `DirectionalLight` with
/// `shadow_maps_enabled: false`).
#[derive(Clone, Copy, Debug)]
pub struct Moon {
    /// Direction the light travels.
    pub direction: Vec3,
    pub color: Vec3,
    /// Lux.
    pub illuminance: f32,
}

/// Bevy `lux::RAW_SUNLIGHT`: sunlight above the atmosphere, the sun's illuminance under a
/// [`Sky`].
pub const RAW_SUNLIGHT: f32 = 130_000.0;

/// Hillaire's physically based atmosphere, as Bevy 0.20 renders it: transmittance,
/// multiscattering, sky-view and aerial-view LUTs, recomputed every frame from the sun and moon
/// directions. The planet and medium are Bevy's Earth (`Atmosphere::earth` with
/// `ScatteringMedium::earth(256, 256)`, the planet centre 6 360 km below the world origin),
/// with Bevy's default `AtmosphereSettings`.
///
/// Needs `wgpu::Features::FLOAT32_FILTERABLE` on the device (the medium LUTs are `Rgba32Float`,
/// as in Bevy).
#[derive(Clone, Copy, Debug)]
pub struct Sky {
    /// Average albedo of the planet's surface, for light reflected into the sky (Bevy
    /// `Atmosphere::ground_albedo`).
    pub ground_albedo: Vec3,
    /// The sun's visible disc (Bevy `SunDisk` on the sun).
    pub sun_disk: SunDisk,
    /// The moon's visible disc (Bevy `SunDisk` on the moon light).
    pub moon_disk: SunDisk,
    /// Ambient and reflected light from the sky: each frame the sky is rendered into a 128²
    /// cube map and filtered into diffuse and specular maps on the GPU (Bevy
    /// `AtmosphereEnvironmentMapLight` with this `intensity`, which drives a
    /// `GeneratedEnvironmentMapLight`). `None` leaves ambient light to `Scene::ambient` and
    /// `Scene::environment`; with `Some`, `Scene::environment` must be `None`.
    pub environment_intensity: Option<f32>,
}

impl Default for Sky {
    /// Bevy's Earth defaults, its sun disc, a moon disc and the sky's environment light at
    /// intensity 1.
    fn default() -> Self {
        Self {
            ground_albedo: Vec3::splat(0.3),
            sun_disk: SunDisk::EARTH,
            moon_disk: SunDisk::MOON,
            environment_intensity: Some(1.0),
        }
    }
}

/// The visible disc of a directional light in the sky (Bevy `SunDisk`). It changes how the disc
/// looks, not the light's illuminance; the disc's radiance is the illuminance over its solid
/// angle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SunDisk {
    /// Angular diameter in radians.
    pub angular_size: f32,
    /// Brightness multiplier; 0 hides the disc.
    pub intensity: f32,
}

impl SunDisk {
    /// Bevy `SunDisk::EARTH`: the sun seen from Earth (about 32′).
    pub const EARTH: Self = Self {
        angular_size: 0.00930842,
        intensity: 1.0,
    };
    /// The moon seen from Earth (about 31′).
    pub const MOON: Self = Self {
        angular_size: 0.00904,
        intensity: 1.0,
    };
    /// No disc (Bevy `SunDisk::OFF`).
    pub const OFF: Self = Self {
        angular_size: 0.0,
        intensity: 0.0,
    };
}

/// Sun and moon directions for a time of day, for games that move them along a fixed path. The
/// game owns the clock; this only maps an hour to directions. The sun rises in the east (+X) at
/// 06:00, culminates towards the south (+Z) at `noon_elevation` at 12:00 and sets in the west
/// at 18:00; the moon follows the same path `moon_lag_hours` behind it.
#[derive(Clone, Copy, Debug)]
pub struct DayCycle {
    /// Radians above the horizon at noon.
    pub noon_elevation: f32,
    /// Hours the moon trails the sun (12 puts it opposite).
    pub moon_lag_hours: f32,
}

impl Default for DayCycle {
    fn default() -> Self {
        Self {
            noon_elevation: 60f32.to_radians(),
            moon_lag_hours: 12.0,
        }
    }
}

impl DayCycle {
    /// Unit direction towards the sun at `hours` (any value; the path repeats every 24 h).
    pub fn to_sun(&self, hours: f32) -> Vec3 {
        let angle = (hours - 6.0) / 12.0 * std::f32::consts::PI;
        let noon = Vec3::new(0.0, self.noon_elevation.sin(), self.noon_elevation.cos());
        Vec3::X * angle.cos() + noon * angle.sin()
    }

    /// Unit direction towards the moon at `hours`.
    pub fn to_moon(&self, hours: f32) -> Vec3 {
        self.to_sun(hours - self.moon_lag_hours)
    }

    /// The hour between noon and midnight at which the sun stands `elevation` radians above
    /// (or, negative, below) the horizon; clamped to noon and midnight.
    pub fn afternoon_hour(&self, elevation: f32) -> f32 {
        let s = (elevation.sin() / self.noon_elevation.sin()).clamp(-1.0, 1.0);
        6.0 + 12.0 * (std::f32::consts::PI - s.asin()) / std::f32::consts::PI
    }
}

/// Shadow-map resolutions shared by every light of a kind (Bevy's `DirectionalLightShadowMap`
/// and `PointLightShadowMap` resources). The defaults are Bevy's.
#[derive(Clone, Copy, Debug)]
pub struct ShadowMapSizes {
    /// Texels per side of each sun cascade and each spot-light map (Bevy
    /// `DirectionalLightShadowMap::size`; Bevy keeps spot maps beside the cascades).
    pub directional: u32,
    /// Texels per side of each point-light cube face (Bevy `PointLightShadowMap::size`).
    pub point: u32,
}

impl Default for ShadowMapSizes {
    fn default() -> Self {
        Self {
            directional: 2048,
            point: 1024,
        }
    }
}

/// An omnidirectional light (Bevy `PointLight`).
#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    /// Luminous power in lumens.
    pub intensity: f32,
    /// Distance where the light has faded out completely (Bevy's smooth window over `1/d²`).
    pub range: f32,
    /// Radius of the light's sphere: widens specular highlights and, with soft shadows, the
    /// penumbra.
    pub radius: f32,
    /// Cube-map shadows; `None` casts none.
    pub shadows: Option<LightShadows>,
}

impl Default for PointLight {
    /// Bevy's `PointLight` defaults (a 1 000 000 lm cinema light, 20 m range).
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            color: Vec3::ONE,
            intensity: 1_000_000.0,
            range: 20.0,
            radius: 0.0,
            shadows: None,
        }
    }
}

/// A cone light (Bevy `SpotLight`): a point light that fades from `inner_angle` to `outer_angle`
/// off its axis.
#[derive(Clone, Copy, Debug)]
pub struct SpotLight {
    pub position: Vec3,
    /// Direction the light points (Bevy: the light transform's forward).
    pub direction: Vec3,
    pub color: Vec3,
    /// Luminous power in lumens as if spread over a full sphere, like a point light's (Bevy
    /// divides by 4π for spot lights too).
    pub intensity: f32,
    pub range: f32,
    pub radius: f32,
    /// Radians off the axis where the falloff starts.
    pub inner_angle: f32,
    /// Radians off the axis where the light ends; below π/2.
    pub outer_angle: f32,
    /// A perspective shadow map; `None` casts none.
    pub shadows: Option<LightShadows>,
}

impl Default for SpotLight {
    /// Bevy's `SpotLight` defaults, pointing down.
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            direction: Vec3::NEG_Y,
            color: Vec3::ONE,
            intensity: 1_000_000.0,
            range: 20.0,
            radius: 0.0,
            inner_angle: 0.0,
            outer_angle: std::f32::consts::FRAC_PI_4,
            shadows: None,
        }
    }
}

/// Shadow settings of a point or spot light (Bevy `shadow_depth_bias`, `shadow_normal_bias`,
/// `shadow_map_near_z`, `soft_shadows_enabled`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightShadows {
    /// World-space offset towards the light.
    pub depth_bias: f32,
    /// Offset along the surface normal in shadow texels.
    pub normal_bias: f32,
    /// Near plane of the shadow projection; casters closer to the light cast nothing.
    pub near_z: f32,
    /// Percentage-closer soft shadows with a penumbra from the light's `radius` (Bevy's
    /// `experimental_pbr_pcss`). Otherwise the camera's [`ShadowFilter`] applies.
    pub soft: bool,
}

impl LightShadows {
    /// Bevy's `PointLight` shadow defaults.
    pub const POINT: Self = Self {
        depth_bias: 0.08,
        normal_bias: 0.6,
        near_z: 0.1,
        soft: false,
    };
    /// Bevy's `SpotLight` shadow defaults.
    pub const SPOT: Self = Self {
        depth_bias: 0.02,
        normal_bias: 1.8,
        near_z: 0.1,
        soft: false,
    };

    /// The same settings with soft shadows.
    pub fn soft(self) -> Self {
        Self { soft: true, ..self }
    }
}

/// Most cascades a sun can have (Bevy `MAX_CASCADES_PER_LIGHT`).
pub const MAX_CASCADES: usize = 4;

/// Cascaded directional shadow maps (Bevy `CascadeShadowConfigBuilder`, the light's shadow
/// biases; the map size is [`ShadowMapSizes::directional`]). The defaults are Bevy's.
#[derive(Clone, Copy, Debug)]
pub struct SunShadows {
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
