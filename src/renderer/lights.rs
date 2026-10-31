//! Point and spot lights as the shader reads them, with their shadow views.

use super::{
    gpu::GpuLight,
    projection::{looking_to, perspective_infinite_reverse},
};
use crate::scene::{LightShadows, Scene};
use bytemuck::Zeroable;
use glam::{Mat3, Mat4, Vec3, Vec4};

/// `GpuLight::flags`: the light samples its shadow map.
const LIGHT_SHADOWS: u32 = 1;
/// `GpuLight::flags`: the spot light points downwards (Bevy `SPOT_LIGHT_Y_NEGATIVE`).
const LIGHT_SPOT_Y_NEGATIVE: u32 = 2;
/// Bevy's point-light cube faces (`CUBE_MAP_FACES`): look target and up per cube layer.
const CUBE_FACES: [(Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::Y),
    (Vec3::NEG_X, Vec3::Y),
    (Vec3::Y, Vec3::Z),
    (Vec3::NEG_Y, Vec3::NEG_Z),
    (Vec3::NEG_Z, Vec3::Y),
    (Vec3::Z, Vec3::Y),
];

/// The scene's point and spot lights as the shader reads them, with their shadow views.
pub(super) struct LocalLights {
    /// Point lights, then spot lights, each with the shadow casters first (Bevy's clustered
    /// light order). Never empty: a zeroed light stands in for none.
    pub(super) gpu: Vec<GpuLight>,
    pub(super) point_count: u32,
    pub(super) spot_count: u32,
    /// `clip_from_world` per spot shadow map, in layer order after the sun cascades.
    pub(super) spot_views: Vec<Mat4>,
    /// `clip_from_world` per point-light cube face, six per shadow-casting point light.
    pub(super) point_views: Vec<Mat4>,
}

impl LocalLights {
    /// Bevy's `extract_lights` and `prepare_lights` for point and spot lights. `cascades` is the
    /// number of sun cascade layers ahead of the spot maps.
    pub(super) fn new(scene: &Scene, cascades: usize) -> Self {
        use std::f32::consts::{FRAC_PI_2, PI, SQRT_2};
        let sizes = scene.shadow_maps;
        let base = |position: Vec3, color: Vec3, intensity: f32, range: f32, radius: f32| {
            GpuLight {
                // Luminous power to luminous intensity (lm/sr), for spot lights too.
                color_inverse_square_range: (color * (intensity / (4.0 * PI)))
                    .extend(1.0 / (range * range)),
                position_radius: position.extend(radius),
                ..GpuLight::zeroed()
            }
        };
        let shadowed = |light: &mut GpuLight, s: LightShadows, texel_size: f32, radius: f32| {
            light.flags |= LIGHT_SHADOWS;
            light.shadow_depth_bias = s.depth_bias;
            // √2 for the worst-case diagonal texel offset, as in Bevy.
            light.shadow_normal_bias = s.normal_bias * texel_size * SQRT_2;
            light.shadow_map_near_z = s.near_z;
            light.soft_shadow_size = if s.soft { radius } else { 0.0 };
        };

        let mut gpu = Vec::new();
        let mut point_views = Vec::new();
        let mut spot_views = Vec::new();
        let mut points: Vec<_> = scene.point_lights.iter().collect();
        points.sort_by_key(|l| l.shadows.is_none());
        for l in points {
            let mut light = base(l.position, l.color, l.intensity, l.range, l.radius);
            let near_z = l.shadows.map_or(LightShadows::POINT.near_z, |s| s.near_z);
            let projection = perspective_infinite_reverse(FRAC_PI_2, 1.0, near_z);
            light.light_custom_data = Vec4::new(
                projection.z_axis.z,
                projection.z_axis.w,
                projection.w_axis.z,
                projection.w_axis.w,
            );
            if let Some(s) = l.shadows {
                shadowed(&mut light, s, 2.0 / sizes.point as f32, l.radius);
                light.shadow_index = (point_views.len() / 6) as u32;
                // Axis-aligned faces: the light's translation only.
                for (target, up) in CUBE_FACES {
                    let world_from_view = Mat4::from_translation(l.position)
                        * Mat4::from_mat3(looking_to(target, up));
                    point_views.push(projection * world_from_view.inverse());
                }
            }
            gpu.push(light);
        }
        let point_count = gpu.len() as u32;

        let mut spots: Vec<_> = scene.spot_lights.iter().collect();
        spots.sort_by_key(|l| l.shadows.is_none());
        for l in spots {
            let mut light = base(l.position, l.color, l.intensity, l.range, l.radius);
            let outer = l.outer_angle.min(FRAC_PI_2 - 1e-4);
            let inner = l.inner_angle.min(outer);
            let direction = l.direction.normalize();
            if direction.y.is_sign_negative() {
                light.flags |= LIGHT_SPOT_Y_NEGATIVE;
            }
            let cos_outer = outer.cos();
            let spot_scale = 1.0 / (inner.cos() - cos_outer).max(1e-4);
            light.light_custom_data = Vec4::new(
                direction.x,
                direction.z,
                spot_scale,
                -cos_outer * spot_scale,
            );
            light.spot_light_tan_angle = outer.tan();
            if let Some(s) = l.shadows {
                let texel_size = 2.0 * outer.tan() / sizes.directional as f32;
                shadowed(&mut light, s, texel_size, l.radius);
                light.shadow_index = (cascades + spot_views.len()) as u32;
                // Bevy `spot_light_world_from_view` / `spot_light_clip_from_view`.
                let world_from_view = Mat4::from_translation(l.position)
                    * Mat4::from_mat3(orthonormalize(-direction));
                let clip_from_view = perspective_infinite_reverse(2.0 * outer, 1.0, s.near_z);
                spot_views.push(clip_from_view * world_from_view.inverse());
            }
            gpu.push(light);
        }
        let spot_count = gpu.len() as u32 - point_count;
        if gpu.is_empty() {
            gpu.push(GpuLight::zeroed());
        }
        Self {
            gpu,
            point_count,
            spot_count,
            spot_views,
            point_views,
        }
    }
}

/// Bevy's `orthonormalize`: a right-handed basis whose z axis is `z` (the shaders rebuild the
/// same basis from the spot direction).
fn orthonormalize(z: Vec3) -> Mat3 {
    let sign = 1f32.copysign(z.z);
    let a = -1.0 / (sign + z.z);
    let b = z.x * z.y * a;
    let x = Vec3::new(1.0 + sign * z.x * z.x * a, sign * b, -sign * z.x);
    let y = Vec3::new(b, sign + z.y * z.y * a, -z.y);
    Mat3::from_cols(x, y, z)
}
