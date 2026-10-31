// Shared code of the sky passes: Hillaire's LUT-based atmosphere as Bevy 0.20 implements it
// (`bevy_pbr/src/atmosphere`: `types.wesl`, `bindings.wesl`, `bruneton_functions.wesl`,
// `functions.wesl`, plus the `bevy_render::maths` helpers they use). Bevy's code is MIT or
// Apache-2.0 (`src/luts/LICENSE-*.bevy`); the Bruneton functions keep their own notice below.
// Each pass prepends this file to its own source, so binding 13 is free for the pass's output.
//
// Conventions (Bevy's): r is the distance from the planet centre, mu the cosine of a ray's
// zenith angle, "atmosphere space" puts the camera on the planet with local up as +y.

const PI: f32 = 3.141592653589793;
const PI_2: f32 = 6.283185307179586;
const HALF_PI: f32 = 1.57079632679;

struct Atmosphere {
    ground_albedo: vec3<f32>,
    inner_radius: f32,
    outer_radius: f32,
    world_to_atmosphere: mat4x4<f32>,
}

// Bevy `GpuAtmosphereSettings` (fields reordered for a padding-free layout).
struct AtmosphereSettings {
    transmittance_lut_size: vec2<u32>,
    multiscattering_lut_size: vec2<u32>,
    sky_view_lut_size: vec2<u32>,
    transmittance_lut_samples: u32,
    multiscattering_lut_dirs: u32,
    aerial_view_lut_size: vec3<u32>,
    multiscattering_lut_samples: u32,
    sky_view_lut_samples: u32,
    aerial_view_lut_samples: u32,
    aerial_view_lut_max_distance: f32,
    sky_max_samples: u32,
    rendering_method: u32,
}

struct AtmosphereTransforms {
    world_from_atmosphere: mat4x4<f32>,
    atmosphere_from_world: mat4x4<f32>,
}

// The fields of Bevy's `View` these passes read.
struct View {
    world_from_view: mat4x4<f32>,
    view_from_clip: mat4x4<f32>,
    world_position: vec3<f32>,
    exposure: f32,
}

// The fields of Bevy's `DirectionalLight` these passes read.
struct DirectionalLight {
    // rgb: colour × illuminance (lux).
    color: vec4<f32>,
    direction_to_light: vec3<f32>,
    sun_disk_angular_size: f32,
    sun_disk_intensity: f32,
}

struct Lights {
    directional_lights: array<DirectionalLight, 2>,
    n_directional_lights: u32,
}

@group(0) @binding(0) var<uniform> atmosphere: Atmosphere;
@group(0) @binding(1) var<uniform> settings: AtmosphereSettings;
@group(0) @binding(2) var<uniform> atmosphere_transforms: AtmosphereTransforms;
@group(0) @binding(3) var<uniform> view: View;
@group(0) @binding(4) var<uniform> lights: Lights;

@group(0) @binding(5) var medium_density_lut: texture_2d<f32>;
@group(0) @binding(6) var medium_scattering_lut: texture_2d<f32>;
@group(0) @binding(7) var medium_sampler: sampler;

@group(0) @binding(8) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(9) var multiscattering_lut: texture_2d<f32>;
@group(0) @binding(10) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(11) var aerial_view_lut: texture_3d<f32>;
@group(0) @binding(12) var atmosphere_lut_sampler: sampler;

// ---------------------------------------------------------------- bevy_render::maths

fn ray_sphere_intersect(r: f32, mu: f32, sphere_radius: f32) -> vec2<f32> {
    let discriminant = r * r * (mu * mu - 1.0) + sphere_radius * sphere_radius;
    if discriminant < 0.0 {
        return vec2(-1.0);
    }
    let q = -r * mu;
    let sqrt_discriminant = sqrt(discriminant);
    return vec2(q - sqrt_discriminant, q + sqrt_discriminant);
}

fn fast_acos(in_x: f32) -> f32 {
    let x = abs(in_x);
    var res = -0.156583 * x + HALF_PI;
    res *= sqrt(1.0 - x);
    return select(PI - res, res, in_x >= 0.0);
}

fn fast_acos_4(x: f32) -> f32 {
    let x1 = abs(x);
    let x2 = x1 * x1;
    let x3 = x2 * x1;
    var s: f32;
    s = -0.2121144 * x1 + 1.5707288;
    s = 0.0742610 * x2 + s;
    s = -0.0187293 * x3 + s;
    s = sqrt(1.0 - x1) * s;
    return select(PI - s, s, x >= 0.0);
}

fn fast_atan2(y: f32, x: f32) -> f32 {
    var t0 = max(abs(x), abs(y));
    var t1 = min(abs(x), abs(y));
    var t3 = t1 / t0;
    var t4 = t3 * t3;

    t0 = 0.0872929;
    t0 = t0 * t4 - 0.301895;
    t0 = t0 * t4 + 1.0;
    t3 = t0 * t3;

    t3 = select(t3, (0.5 * PI) - t3, abs(y) > abs(x));
    t3 = select(t3, PI - t3, x < 0);
    t3 = select(-t3, t3, y > 0);
    return t3;
}

// ---------------------------------------------------------------- bruneton_functions
//
// Copyright (c) 2017 Eric Bruneton
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holders nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.
//
// Precomputed Atmospheric Scattering
// Copyright (c) 2008 INRIA
// All rights reserved.
//
// (The same three-clause licence as above applies.)

/// Mapping from view height (r) and zenith cos angle (mu) to UV coordinates in the transmittance
/// LUT (Bruneton and Neyret 2008, section 4).
fn transmittance_lut_r_mu_to_uv(atm: Atmosphere, r: f32, mu: f32) -> vec2<f32> {
    let H = sqrt(atm.outer_radius * atm.outer_radius - atm.inner_radius * atm.inner_radius);
    let rho = sqrt(max(r * r - atm.inner_radius * atm.inner_radius, 0.0));
    let d = distance_to_top_atmosphere_boundary(atm, r, mu);
    let d_min = atm.outer_radius - r;
    let d_max = rho + H;
    let u = (d - d_min) / (d_max - d_min);
    let v = rho / H;
    return vec2<f32>(u, v);
}

/// Inverse of the mapping above.
fn transmittance_lut_uv_to_r_mu(uv: vec2<f32>) -> vec2<f32> {
    let H = sqrt(atmosphere.outer_radius * atmosphere.outer_radius - atmosphere.inner_radius * atmosphere.inner_radius);
    let rho = H * uv.y;
    let r = sqrt(rho * rho + atmosphere.inner_radius * atmosphere.inner_radius);
    let d_min = atmosphere.outer_radius - r;
    let d_max = rho + H;
    let d = d_min + uv.x * (d_max - d_min);

    var mu: f32;
    if d == 0.0 {
        mu = 1.0;
    } else {
        mu = (H * H - rho * rho - d * d) / (2.0 * r * d);
    }
    mu = clamp(mu, -1.0, 1.0);
    return vec2<f32>(r, mu);
}

fn distance_to_top_atmosphere_boundary(atm: Atmosphere, r: f32, mu: f32) -> f32 {
    let positive_discriminant = max(r * r * (mu * mu - 1.0) + atm.outer_radius * atm.outer_radius, 0.0);
    return max(-r * mu + sqrt(positive_discriminant), 0.0);
}

fn distance_to_bottom_atmosphere_boundary(r: f32, mu: f32) -> f32 {
    let positive_discriminant = max(r * r * (mu * mu - 1.0) + atmosphere.inner_radius * atmosphere.inner_radius, 0.0);
    return max(-r * mu - sqrt(positive_discriminant), 0.0);
}

fn ray_intersects_ground(r: f32, mu: f32) -> bool {
    return mu < 0.0 && r * r * (mu * mu - 1.0) + atmosphere.inner_radius * atmosphere.inner_radius >= 0.0;
}

fn sample_transmittance_lut_segment(r: f32, mu: f32, t: f32) -> vec3<f32> {
    let r_t = get_local_r(r, mu, t);
    let mu_t = clamp((r * mu + t) / r_t, -1.0, 1.0);

    if ray_intersects_ground(r, mu) {
        return min(
            sample_transmittance_lut(r_t, -mu_t) / sample_transmittance_lut(r, -mu),
            vec3(1.0)
        );
    } else {
        return min(
            sample_transmittance_lut(r, mu) / sample_transmittance_lut(r_t, mu_t), vec3(1.0)
        );
    }
}

// ---------------------------------------------------------------- functions

const FRAC_PI: f32 = 0.3183098862;
const FRAC_2_PI: f32 = 0.15915494309;
const FRAC_3_16_PI: f32 = 0.0596831036594607509;
const FRAC_4_PI: f32 = 0.07957747154594767;
const ROOT_2: f32 = 1.41421356;
/// 1 metre.
const EPSILON: f32 = 1.0;
const MIN_EXTINCTION: vec3<f32> = vec3(1e-12);
/// Where in each raymarch segment the single sample sits (0 start, 1 end).
const MIDPOINT_RATIO: f32 = 0.3;

fn unit_to_sub_uvs(val: vec2<f32>, resolution: vec2<f32>) -> vec2<f32> {
    return (val + 0.5f / resolution) * (resolution / (resolution + 1.0f));
}

fn sub_uvs_to_unit(val: vec2<f32>, resolution: vec2<f32>) -> vec2<f32> {
    return (val - 0.5f / resolution) * (resolution / (resolution - 1.0f));
}

fn multiscattering_lut_r_mu_to_uv(r: f32, mu: f32) -> vec2<f32> {
    let u = 0.5 + 0.5 * mu;
    let v = saturate((r - atmosphere.inner_radius) / (atmosphere.outer_radius - atmosphere.inner_radius));
    return unit_to_sub_uvs(vec2(u, v), vec2<f32>(settings.multiscattering_lut_size));
}

fn multiscattering_lut_uv_to_r_mu(uv: vec2<f32>) -> vec2<f32> {
    let adj_uv = sub_uvs_to_unit(uv, vec2<f32>(settings.multiscattering_lut_size));
    let r = mix(atmosphere.inner_radius, atmosphere.outer_radius, adj_uv.y);
    let mu = adj_uv.x * 2 - 1;
    return vec2(r, mu);
}

fn sky_view_lut_r_mu_azimuth_to_uv(r: f32, mu: f32, azimuth: f32) -> vec2<f32> {
    let u = (azimuth * FRAC_2_PI) + 0.5;

    let v_horizon = sqrt(r * r - atmosphere.inner_radius * atmosphere.inner_radius);
    let cos_beta = v_horizon / r;
    let beta = fast_acos_4(cos_beta);
    let horizon_zenith = PI - beta;
    let view_zenith = fast_acos_4(mu);

    // Non-linear latitude: more texels near the horizon.
    let l = view_zenith - horizon_zenith;
    let abs_l = abs(l);
    let v = 0.5 + 0.5 * sign(l) * sqrt(abs_l / HALF_PI);

    return unit_to_sub_uvs(vec2(u, v), vec2<f32>(settings.sky_view_lut_size));
}

fn sky_view_lut_uv_to_zenith_azimuth(r: f32, uv: vec2<f32>) -> vec2<f32> {
    let adj_uv = sub_uvs_to_unit(vec2(uv.x, 1.0 - uv.y), vec2<f32>(settings.sky_view_lut_size));
    let azimuth = (adj_uv.x - 0.5) * PI_2;

    let v_horizon = sqrt(r * r - atmosphere.inner_radius * atmosphere.inner_radius);
    let cos_beta = v_horizon / r;
    let beta = fast_acos_4(cos_beta);
    let horizon_zenith = PI - beta;

    let t = abs(2.0 * (adj_uv.y - 0.5));
    let l = sign(adj_uv.y - 0.5) * HALF_PI * t * t;

    return vec2(horizon_zenith - l, azimuth);
}

fn sample_transmittance_lut(r: f32, mu: f32) -> vec3<f32> {
    let uv = transmittance_lut_r_mu_to_uv(atmosphere, r, mu);
    return textureSampleLevel(transmittance_lut, atmosphere_lut_sampler, uv, 0.0).rgb;
}

fn sample_multiscattering_lut(r: f32, mu: f32) -> vec3<f32> {
    let uv = multiscattering_lut_r_mu_to_uv(r, mu);
    return textureSampleLevel(multiscattering_lut, atmosphere_lut_sampler, uv, 0.0).rgb;
}

fn sample_sky_view_lut(r: f32, ray_dir_as: vec3<f32>) -> vec3<f32> {
    let mu = ray_dir_as.y;
    let azimuth = fast_atan2(ray_dir_as.x, -ray_dir_as.z);
    let uv = sky_view_lut_r_mu_azimuth_to_uv(r, mu, azimuth);
    return textureSampleLevel(sky_view_lut, atmosphere_lut_sampler, uv, 0.0).rgb;
}

fn ndc_to_camera_dist(ndc: vec3<f32>) -> f32 {
    let view_pos = view.view_from_clip * vec4(ndc, 1.0);
    let p_view = view_pos.xyz / view_pos.w;
    let p_world = (view.world_from_view * vec4(p_view, 1.0)).xyz;
    let p_atmo = (atmosphere.world_to_atmosphere * vec4(p_world, 1.0)).xyz;
    let cam_atmo = (atmosphere.world_to_atmosphere * vec4(view.world_position, 1.0)).xyz;
    return length(p_atmo - cam_atmo);
}

/// RGB: inscattered light along the camera ray up to distance `t`.
fn sample_aerial_view_lut(uv: vec2<f32>, t: f32) -> vec3<f32> {
    let t_max = settings.aerial_view_lut_max_distance;
    let num_slices = f32(settings.aerial_view_lut_size.z);
    // Each slice holds the integral over the whole slice: offset by half a slice.
    let uvw = vec3(uv, saturate(t / t_max - 0.5 / num_slices));
    let sample = textureSampleLevel(aerial_view_lut, atmosphere_lut_sampler, uvw, 0.0);
    // Fade linearly to zero at the camera across the first slice.
    let t_slice = t_max / num_slices;
    let fade = saturate(t / t_slice);
    return exp(sample.rgb) * fade;
}

const ABSORPTION_DENSITY: f32 = 0.0;
const SCATTERING_DENSITY: f32 = 1.0;

/// The medium's absorption (`component = 0`) or scattering (`component = 1`) density at radius r.
fn sample_density_lut(r: f32, component: f32) -> vec3<f32> {
    let normalized_altitude = (r - atmosphere.inner_radius) / (atmosphere.outer_radius - atmosphere.inner_radius);
    let uv = vec2(1.0 - normalized_altitude, component);
    return textureSampleLevel(medium_density_lut, medium_sampler, uv, 0.0).xyz;
}

const PHASE_MAPPING_N: f32 = 0.5;

/// Scattering density times the phase function for `neg_LdotV`, through Bevy's non-linear
/// phase mapping.
fn sample_scattering_lut(r: f32, neg_LdotV: f32) -> vec3<f32> {
    let normalized_altitude = (r - atmosphere.inner_radius) / (atmosphere.outer_radius - atmosphere.inner_radius);
    let phase_uv = 0.5 + 0.5 * sign(neg_LdotV) * (1.0 - pow(1.0 - abs(neg_LdotV), PHASE_MAPPING_N));
    let uv = vec2(1.0 - normalized_altitude, phase_uv);
    return textureSampleLevel(medium_scattering_lut, medium_sampler, uv, 0.0).xyz;
}

/// L_scat (Hillaire, equation 3): single scattering towards the view at one point, plus the
/// multiscattering term.
fn sample_local_inscattering(local_scattering: vec3<f32>, ray_dir: vec3<f32>, world_pos: vec3<f32>) -> vec3<f32> {
    let local_r = length(world_pos);
    let local_up = normalize(world_pos);
    var inscattering = vec3(0.0);
    for (var light_i: u32 = 0u; light_i < lights.n_directional_lights; light_i++) {
        let light = &lights.directional_lights[light_i];

        let mu_light = dot((*light).direction_to_light, local_up);
        let neg_LdotV = dot((*light).direction_to_light, ray_dir);

        let transmittance_to_light = sample_transmittance_lut(local_r, mu_light);
        let shadow_factor = transmittance_to_light * f32(!ray_intersects_ground(local_r, mu_light));
        let scattering_coeff = sample_scattering_lut(local_r, neg_LdotV);

        let scattering_factor = shadow_factor * scattering_coeff;

        let psi_ms = sample_multiscattering_lut(local_r, mu_light);
        let multiscattering_factor = psi_ms * local_scattering;

        inscattering += (*light).color.rgb * (scattering_factor + multiscattering_factor);
    }
    return inscattering;
}

fn sample_sun_radiance(ray_dir_ws: vec3<f32>) -> vec3<f32> {
    let view_pos = get_view_position();
    let r = length(view_pos);
    let up = normalize(view_pos);
    let mu_view = dot(ray_dir_ws, up);
    let shadow_factor = f32(!ray_intersects_ground(r, mu_view));
    var sun_radiance = vec3(0.0);
    for (var light_i: u32 = 0u; light_i < lights.n_directional_lights; light_i++) {
        let light = &lights.directional_lights[light_i];
        let neg_LdotV = dot((*light).direction_to_light, ray_dir_ws);
        let angle_to_sun = fast_acos(clamp(neg_LdotV, -1.0, 1.0));
        let w = max(0.5 * fwidth(angle_to_sun), 1e-6);
        let sun_angular_size = (*light).sun_disk_angular_size;
        let sun_intensity = (*light).sun_disk_intensity;
        if sun_angular_size > 0.0 && sun_intensity > 0.0 {
            let factor = 1 - smoothstep(sun_angular_size * 0.5 - w, sun_angular_size * 0.5 + w, angle_to_sun);
            let sun_solid_angle = (sun_angular_size * sun_angular_size) * 0.25 * PI;
            sun_radiance += ((*light).color.rgb / sun_solid_angle) * sun_intensity * factor * shadow_factor;
        }
    }
    return sun_radiance;
}

fn clamp_to_surface(atmosphere: Atmosphere, position: vec3<f32>) -> vec3<f32> {
    let min_radius = atmosphere.inner_radius + EPSILON;
    let r = length(position);
    if r < min_radius {
        let up = normalize(position);
        return up * min_radius;
    }
    return position;
}

fn max_atmosphere_distance(r: f32, mu: f32) -> f32 {
    let t_top = distance_to_top_atmosphere_boundary(atmosphere, r, mu);
    let t_bottom = distance_to_bottom_atmosphere_boundary(r, mu);
    let hits = ray_intersects_ground(r, mu);
    return mix(t_top, t_bottom, f32(hits));
}

/// The observer's position in the atmosphere.
fn get_view_position() -> vec3<f32> {
    let atmo_pos = (atmosphere.world_to_atmosphere * vec4(view.world_position, 1.0)).xyz;
    return clamp_to_surface(atmosphere, atmo_pos);
}

/// Up at distance t along the ray, with up at the view position taken as +y.
fn get_local_up(r: f32, t: f32, ray_dir: vec3<f32>) -> vec3<f32> {
    return normalize(vec3(0.0, r, 0.0) + t * ray_dir);
}

/// The radius at distance t along a ray starting at radius r with zenith cosine mu.
fn get_local_r(r: f32, mu: f32, t: f32) -> f32 {
    return sqrt(t * t + 2.0 * r * mu * t + r * r);
}

fn uv_to_ndc(uv: vec2<f32>) -> vec2<f32> {
    return uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
}

fn ndc_to_uv(ndc: vec2<f32>) -> vec2<f32> {
    return ndc * vec2(0.5, -0.5) + vec2(0.5);
}

fn direction_world_to_atmosphere(dir_ws: vec3<f32>) -> vec3<f32> {
    let dir_as = atmosphere_transforms.atmosphere_from_world * vec4(dir_ws, 0.0);
    return dir_as.xyz;
}

fn direction_atmosphere_to_world(dir_as: vec3<f32>) -> vec3<f32> {
    let dir_ws = atmosphere_transforms.world_from_atmosphere * vec4(dir_as, 0.0);
    return dir_ws.xyz;
}

/// The normalised world-space ray through `uv` (through the near plane, like Bevy).
fn uv_to_ray_direction(uv: vec2<f32>) -> vec3<f32> {
    let view_position_homogeneous = view.view_from_clip * vec4(
        uv_to_ndc(uv),
        1.0,
        1.0,
    );
    let view_ray_direction = view_position_homogeneous.xyz / view_position_homogeneous.w;
    let ray_direction = (view.world_from_view * vec4(view_ray_direction, 0.0)).xyz;
    return normalize(ray_direction);
}

fn zenith_azimuth_to_ray_dir(zenith: f32, azimuth: f32) -> vec3<f32> {
    let sin_zenith = sin(zenith);
    let mu = cos(zenith);
    let sin_azimuth = sin(azimuth);
    let cos_azimuth = cos(azimuth);
    return vec3(sin_azimuth * sin_zenith, mu, -cos_azimuth * sin_zenith);
}

struct RaymarchSegment {
    start: f32,
    end: f32,
}

fn get_raymarch_segment(r: f32, mu: f32) -> RaymarchSegment {
    let atmosphere_intersections = ray_sphere_intersect(r, mu, atmosphere.outer_radius);
    let ground_intersections = ray_sphere_intersect(r, mu, atmosphere.inner_radius);

    var segment: RaymarchSegment;

    if r < atmosphere.inner_radius {
        segment.start = ground_intersections.y;
        segment.end = atmosphere_intersections.y;
    } else if r < atmosphere.outer_radius {
        segment.start = 0.0;
        segment.end = select(atmosphere_intersections.y, ground_intersections.x, ray_intersects_ground(r, mu));
    } else {
        if atmosphere_intersections.x < 0.0 {
            return segment;
        }
        segment.start = atmosphere_intersections.x;
        segment.end = select(atmosphere_intersections.y, ground_intersections.x, ray_intersects_ground(r, mu));
    }

    return segment;
}

struct RaymarchResult {
    inscattering: vec3<f32>,
    transmittance: vec3<f32>,
}

fn raymarch_atmosphere(
    pos: vec3<f32>,
    ray_dir: vec3<f32>,
    t_max: f32,
    max_samples: u32,
    uv: vec2<f32>,
    ground: bool
) -> RaymarchResult {
    let r = length(pos);
    let up = normalize(pos);
    let mu = dot(ray_dir, up);

    // Fewer samples close to the scene.
    let sample_count = mix(1.0, f32(max_samples), saturate(t_max * 0.01));

    let segment = get_raymarch_segment(r, mu);
    let t_start = segment.start;
    var t_end = segment.end;

    t_end = min(t_end, t_max);
    let t_total = t_end - t_start;

    var result: RaymarchResult;
    result.inscattering = vec3(0.0);
    result.transmittance = vec3(1.0);

    if t_total <= 0.0 {
        return result;
    }

    var prev_t = t_start;
    var optical_depth = vec3(0.0);
    for (var s = 0.0; s < sample_count; s += 1.0) {
        let t_i = t_start + t_total * (s + MIDPOINT_RATIO) / sample_count;
        let dt_i = (t_i - prev_t);
        prev_t = t_i;

        let sample_pos = pos + ray_dir * t_i;
        let local_r = length(sample_pos);
        let local_up = normalize(sample_pos);

        let absorption = sample_density_lut(local_r, ABSORPTION_DENSITY);
        let scattering = sample_density_lut(local_r, SCATTERING_DENSITY);
        let extinction = absorption + scattering;

        let sample_optical_depth = extinction * dt_i;
        optical_depth += sample_optical_depth;
        let sample_transmittance = exp(-sample_optical_depth);

        let inscattering = sample_local_inscattering(
            scattering,
            ray_dir,
            sample_pos
        );

        let s_int = (inscattering - inscattering * sample_transmittance) / max(extinction, MIN_EXTINCTION);
        result.inscattering += result.transmittance * s_int;

        result.transmittance *= sample_transmittance;
        if all(result.transmittance < vec3(0.001)) {
            break;
        }
    }

    // Light reflected by the planet's ground.
    if ground && ray_intersects_ground(r, mu) {
        for (var light_i: u32 = 0u; light_i < lights.n_directional_lights; light_i++) {
            let light = &lights.directional_lights[light_i];
            let light_dir = (*light).direction_to_light;
            let light_color = (*light).color.rgb;
            let transmittance_to_ground = exp(-optical_depth);
            let sphere_point = pos + ray_dir * t_end;
            let sphere_normal = normalize(sphere_point);
            let mu_light = dot(light_dir, sphere_normal);
            let transmittance_to_light = sample_transmittance_lut(0.0, mu_light);
            let light_luminance = transmittance_to_light * max(mu_light, 0.0) * light_color;
            let ground_luminance = transmittance_to_ground * atmosphere.ground_albedo / PI;
            result.inscattering += ground_luminance * light_luminance;
        }
    }

    return result;
}
