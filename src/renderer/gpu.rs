//! The uniform and storage layouts the shaders read (`shaders/lit.wgsl`).

use crate::scene::MAX_CASCADES;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, UVec4, Vec4};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct FrameUniform {
    pub(super) clip_from_world: Mat4,
    pub(super) view_from_world: Mat4,
    pub(super) cascade_clip_from_world: [Mat4; MAX_CASCADES],
    /// Per cascade: x far bound (m), y texel size (m).
    pub(super) cascade_params: [Vec4; MAX_CASCADES],
    pub(super) eye: Vec4,
    pub(super) direction_to_light: Vec4,
    pub(super) sun_color: Vec4,
    pub(super) ambient_color: Vec4,
    /// x depth bias (m), y normal bias (texels), z overlap proportion, w cascade count.
    pub(super) shadow: Vec4,
    /// x exposure, y 1 when the sun casts shadows, z 1 when there is a sun, w 1 for Gaussian
    /// filtering.
    pub(super) flags: Vec4,
    /// x environment-map intensity (cd/m²), y smallest specular mip level, z 1 when there is an
    /// environment map.
    pub(super) environment: Vec4,
    /// Distance fog (Bevy `GpuFog`): linear RGBA colour.
    pub(super) fog_color: Vec4,
    /// Sun inscattering colour, alpha = strength (0 = off).
    pub(super) fog_directional_light_color: Vec4,
    /// xyz: Bevy `be` (linear: start, end; exponential: density; atmospheric: extinction),
    /// w: directional-light exponent.
    pub(super) fog_be: Vec4,
    /// xyz: Bevy `bi` (atmospheric inscattering), w: mode (0 off, 1 linear, 2 exponential,
    /// 3 exponential squared, 4 atmospheric).
    pub(super) fog_bi: Vec4,
    /// x point lights, y spot lights (the spot lights follow the point lights in the list).
    pub(super) light_counts: UVec4,
    /// Unit direction towards the moon, w 1 when there is a moon.
    pub(super) moon_direction_to_light: Vec4,
    pub(super) moon_color: Vec4,
    pub(super) world_to_atmosphere: Mat4,
    /// x planet radius, y atmosphere radius, z 1 with a sky, w 1 when the moon is shaded first.
    pub(super) atmosphere: Vec4,
    /// x sun disc, y moon disc angular size.
    pub(super) disks: Vec4,
}

/// One point or spot light, laid out like Bevy's `GpuClusteredLight`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct GpuLight {
    /// Point: the cube-face projection's lower-right 2×2 terms. Spot: direction x and z, spot
    /// scale, spot offset.
    pub(super) light_custom_data: Vec4,
    /// rgb colour × intensity / 4π, w 1 / range².
    pub(super) color_inverse_square_range: Vec4,
    pub(super) position_radius: Vec4,
    pub(super) flags: u32,
    pub(super) shadow_depth_bias: f32,
    /// Premultiplied by the shadow texel size at 1 m and √2, like Bevy.
    pub(super) shadow_normal_bias: f32,
    pub(super) shadow_map_near_z: f32,
    pub(super) spot_light_tan_angle: f32,
    /// The light radius with soft shadows, else 0.
    pub(super) soft_shadow_size: f32,
    /// Point: cube index in the cube array. Spot: layer in the directional shadow array.
    pub(super) shadow_index: u32,
    pub(super) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct InstanceData {
    pub(super) world_from_local: Mat4,
    // mat3x3<f32> in WGSL: three columns padded to 16 bytes.
    pub(super) normal_from_local: [Vec4; 3],
    /// x 1 when the mesh has tangents, y 1 when the model matrix mirrors.
    pub(super) flags: UVec4,
}
