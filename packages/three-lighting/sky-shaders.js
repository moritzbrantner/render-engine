// GLSL ES 3.0 ports of render-engine's sky passes (`src/shaders/sky/`): Bevy 0.20's atmosphere
// (Hillaire's LUT method, `bevy_pbr/src/atmosphere`) and the runtime environment-map filter of
// its `GeneratedEnvironmentMapLight`. Bevy's code is MIT or Apache-2.0 (`src/luts/LICENSE-*.bevy`);
// the Bruneton functions keep their own notice below.
//
// WebGL 2 has no compute shaders, so every LUT texel is a fragment of a fullscreen pass that
// writes it at gl_FragCoord (row 0 at t = 0, as in the wgpu storage textures). Where a compute
// pass shares work across invocations it is split into passes that give the same sums: the
// multiscattering LUT reduces its 64 directions per texel in the workgroup's tree order inside
// one fragment, and the aerial-view LUT marches slice by slice, carrying its running sums
// between passes in Rgba32Float targets.

// ---------------------------------------------------------------- common.wgsl

const common = /* glsl */ `
const float PI = 3.141592653589793;
const float PI_2 = 6.283185307179586;
const float HALF_PI = 1.57079632679;

// Bevy Atmosphere::earth and AtmosphereSettings::default().
const float INNER_RADIUS = 6360000.0;
const float OUTER_RADIUS = 6460000.0;
const vec2 MULTISCATTERING_LUT_SIZE = vec2(32.0, 32.0);
const vec2 SKY_VIEW_LUT_SIZE = vec2(400.0, 200.0);
const vec3 AERIAL_VIEW_LUT_SIZE = vec3(32.0, 32.0, 32.0);
const float AERIAL_VIEW_LUT_MAX_DISTANCE = 3.2e4;

uniform vec3 groundAlbedo;
uniform mat4 worldToAtmosphere;
uniform mat4 worldFromAtmosphere;
uniform mat4 atmosphereFromWorld;
// The fields of Bevy's View these passes read.
uniform mat4 worldFromView;
uniform mat4 viewFromClip;
uniform vec3 viewPosition;
uniform float viewExposure;
// Bevy's directional lights: rgb colour × illuminance, direction to the light, disc size and
// intensity; unshadowed lights first.
uniform vec4 lightColor[2];
uniform vec4 lightDirection[2];
uniform vec4 lightDisk[2];
uniform int lightCount;

uniform sampler2D mediumDensityLut;
uniform sampler2D mediumScatteringLut;
uniform sampler2D transmittanceLut;
uniform sampler2D multiscatteringLut;
uniform sampler2D skyViewLut;
uniform highp sampler3D aerialViewLut;

// ---------------------------------------------------------------- bevy_render::maths

vec2 ray_sphere_intersect(float r, float mu, float sphere_radius) {
    float discriminant = r * r * (mu * mu - 1.0) + sphere_radius * sphere_radius;
    if (discriminant < 0.0) {
        return vec2(-1.0);
    }
    float q = -r * mu;
    float sqrt_discriminant = sqrt(discriminant);
    return vec2(q - sqrt_discriminant, q + sqrt_discriminant);
}

float fast_acos(float in_x) {
    float x = abs(in_x);
    float res = -0.156583 * x + HALF_PI;
    res *= sqrt(1.0 - x);
    return in_x >= 0.0 ? res : PI - res;
}

float fast_acos_4(float x) {
    float x1 = abs(x);
    float x2 = x1 * x1;
    float x3 = x2 * x1;
    float s;
    s = -0.2121144 * x1 + 1.5707288;
    s = 0.0742610 * x2 + s;
    s = -0.0187293 * x3 + s;
    s = sqrt(1.0 - x1) * s;
    return x >= 0.0 ? s : PI - s;
}

float fast_atan2(float y, float x) {
    float t0 = max(abs(x), abs(y));
    float t1 = min(abs(x), abs(y));
    float t3 = t1 / t0;
    float t4 = t3 * t3;

    t0 = 0.0872929;
    t0 = t0 * t4 - 0.301895;
    t0 = t0 * t4 + 1.0;
    t3 = t0 * t3;

    t3 = abs(y) > abs(x) ? (0.5 * PI) - t3 : t3;
    t3 = x < 0.0 ? PI - t3 : t3;
    t3 = y > 0.0 ? t3 : -t3;
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

float distance_to_top_atmosphere_boundary(float r, float mu) {
    float positive_discriminant = max(r * r * (mu * mu - 1.0) + OUTER_RADIUS * OUTER_RADIUS, 0.0);
    return max(-r * mu + sqrt(positive_discriminant), 0.0);
}

float distance_to_bottom_atmosphere_boundary(float r, float mu) {
    float positive_discriminant = max(r * r * (mu * mu - 1.0) + INNER_RADIUS * INNER_RADIUS, 0.0);
    return max(-r * mu - sqrt(positive_discriminant), 0.0);
}

// Mapping from view height (r) and zenith cos angle (mu) to UV coordinates in the transmittance
// LUT (Bruneton and Neyret 2008, section 4).
vec2 transmittance_lut_r_mu_to_uv(float r, float mu) {
    float H = sqrt(OUTER_RADIUS * OUTER_RADIUS - INNER_RADIUS * INNER_RADIUS);
    float rho = sqrt(max(r * r - INNER_RADIUS * INNER_RADIUS, 0.0));
    float d = distance_to_top_atmosphere_boundary(r, mu);
    float d_min = OUTER_RADIUS - r;
    float d_max = rho + H;
    float u = (d - d_min) / (d_max - d_min);
    float v = rho / H;
    return vec2(u, v);
}

// Inverse of the mapping above.
vec2 transmittance_lut_uv_to_r_mu(vec2 uv) {
    float H = sqrt(OUTER_RADIUS * OUTER_RADIUS - INNER_RADIUS * INNER_RADIUS);
    float rho = H * uv.y;
    float r = sqrt(rho * rho + INNER_RADIUS * INNER_RADIUS);
    float d_min = OUTER_RADIUS - r;
    float d_max = rho + H;
    float d = d_min + uv.x * (d_max - d_min);

    float mu;
    if (d == 0.0) {
        mu = 1.0;
    } else {
        mu = (H * H - rho * rho - d * d) / (2.0 * r * d);
    }
    mu = clamp(mu, -1.0, 1.0);
    return vec2(r, mu);
}

bool ray_intersects_ground(float r, float mu) {
    return mu < 0.0 && r * r * (mu * mu - 1.0) + INNER_RADIUS * INNER_RADIUS >= 0.0;
}

// ---------------------------------------------------------------- functions

const float FRAC_PI = 0.3183098862;
const float FRAC_2_PI = 0.15915494309;
const float FRAC_3_16_PI = 0.0596831036594607509;
const float FRAC_4_PI = 0.07957747154594767;
// 1 metre.
const float EPSILON = 1.0;
const vec3 MIN_EXTINCTION = vec3(1e-12);
// Where in each raymarch segment the single sample sits (0 start, 1 end).
const float MIDPOINT_RATIO = 0.3;

vec2 unit_to_sub_uvs(vec2 val, vec2 resolution) {
    return (val + 0.5 / resolution) * (resolution / (resolution + 1.0));
}

vec2 sub_uvs_to_unit(vec2 val, vec2 resolution) {
    return (val - 0.5 / resolution) * (resolution / (resolution - 1.0));
}

vec2 multiscattering_lut_r_mu_to_uv(float r, float mu) {
    float u = 0.5 + 0.5 * mu;
    float v = clamp((r - INNER_RADIUS) / (OUTER_RADIUS - INNER_RADIUS), 0.0, 1.0);
    return unit_to_sub_uvs(vec2(u, v), MULTISCATTERING_LUT_SIZE);
}

vec2 multiscattering_lut_uv_to_r_mu(vec2 uv) {
    vec2 adj_uv = sub_uvs_to_unit(uv, MULTISCATTERING_LUT_SIZE);
    float r = mix(INNER_RADIUS, OUTER_RADIUS, adj_uv.y);
    float mu = adj_uv.x * 2.0 - 1.0;
    return vec2(r, mu);
}

vec2 sky_view_lut_r_mu_azimuth_to_uv(float r, float mu, float azimuth) {
    float u = (azimuth * FRAC_2_PI) + 0.5;

    float v_horizon = sqrt(r * r - INNER_RADIUS * INNER_RADIUS);
    float cos_beta = v_horizon / r;
    float beta = fast_acos_4(cos_beta);
    float horizon_zenith = PI - beta;
    float view_zenith = fast_acos_4(mu);

    // Non-linear latitude: more texels near the horizon.
    float l = view_zenith - horizon_zenith;
    float abs_l = abs(l);
    float v = 0.5 + 0.5 * sign(l) * sqrt(abs_l / HALF_PI);

    return unit_to_sub_uvs(vec2(u, v), SKY_VIEW_LUT_SIZE);
}

vec2 sky_view_lut_uv_to_zenith_azimuth(float r, vec2 uv) {
    vec2 adj_uv = sub_uvs_to_unit(vec2(uv.x, 1.0 - uv.y), SKY_VIEW_LUT_SIZE);
    float azimuth = (adj_uv.x - 0.5) * PI_2;

    float v_horizon = sqrt(r * r - INNER_RADIUS * INNER_RADIUS);
    float cos_beta = v_horizon / r;
    float beta = fast_acos_4(cos_beta);
    float horizon_zenith = PI - beta;

    float t = abs(2.0 * (adj_uv.y - 0.5));
    float l = sign(adj_uv.y - 0.5) * HALF_PI * t * t;

    return vec2(horizon_zenith - l, azimuth);
}

vec3 sample_transmittance_lut(float r, float mu) {
    vec2 uv = transmittance_lut_r_mu_to_uv(r, mu);
    return textureLod(transmittanceLut, uv, 0.0).rgb;
}

vec3 sample_multiscattering_lut(float r, float mu) {
    vec2 uv = multiscattering_lut_r_mu_to_uv(r, mu);
    return textureLod(multiscatteringLut, uv, 0.0).rgb;
}

vec3 sample_sky_view_lut(float r, vec3 ray_dir_as) {
    float mu = ray_dir_as.y;
    float azimuth = fast_atan2(ray_dir_as.x, -ray_dir_as.z);
    vec2 uv = sky_view_lut_r_mu_azimuth_to_uv(r, mu, azimuth);
    return textureLod(skyViewLut, uv, 0.0).rgb;
}

// The radius at distance t along a ray starting at radius r with zenith cosine mu.
float get_local_r(float r, float mu, float t) {
    return sqrt(t * t + 2.0 * r * mu * t + r * r);
}

vec3 sample_transmittance_lut_segment(float r, float mu, float t) {
    float r_t = get_local_r(r, mu, t);
    float mu_t = clamp((r * mu + t) / r_t, -1.0, 1.0);

    if (ray_intersects_ground(r, mu)) {
        return min(sample_transmittance_lut(r_t, -mu_t) / sample_transmittance_lut(r, -mu), vec3(1.0));
    } else {
        return min(sample_transmittance_lut(r, mu) / sample_transmittance_lut(r_t, mu_t), vec3(1.0));
    }
}

// RGB: inscattered light along the camera ray up to distance t.
vec3 sample_aerial_view_lut(vec2 uv, float t) {
    float t_max = AERIAL_VIEW_LUT_MAX_DISTANCE;
    float num_slices = AERIAL_VIEW_LUT_SIZE.z;
    // Each slice holds the integral over the whole slice: offset by half a slice.
    vec3 uvw = vec3(uv, clamp(t / t_max - 0.5 / num_slices, 0.0, 1.0));
    vec4 s = textureLod(aerialViewLut, uvw, 0.0);
    // Fade linearly to zero at the camera across the first slice.
    float t_slice = t_max / num_slices;
    float fade = clamp(t / t_slice, 0.0, 1.0);
    return exp(s.rgb) * fade;
}

const float ABSORPTION_DENSITY = 0.0;
const float SCATTERING_DENSITY = 1.0;

// The medium's absorption (component = 0) or scattering (component = 1) density at radius r.
vec3 sample_density_lut(float r, float component) {
    float normalized_altitude = (r - INNER_RADIUS) / (OUTER_RADIUS - INNER_RADIUS);
    vec2 uv = vec2(1.0 - normalized_altitude, component);
    return textureLod(mediumDensityLut, uv, 0.0).xyz;
}

const float PHASE_MAPPING_N = 0.5;

// Scattering density times the phase function for neg_LdotV, through Bevy's non-linear phase
// mapping.
vec3 sample_scattering_lut(float r, float neg_LdotV) {
    float normalized_altitude = (r - INNER_RADIUS) / (OUTER_RADIUS - INNER_RADIUS);
    float phase_uv = 0.5 + 0.5 * sign(neg_LdotV) * (1.0 - pow(1.0 - abs(neg_LdotV), PHASE_MAPPING_N));
    vec2 uv = vec2(1.0 - normalized_altitude, phase_uv);
    return textureLod(mediumScatteringLut, uv, 0.0).xyz;
}

// L_scat (Hillaire, equation 3): single scattering towards the view at one point, plus the
// multiscattering term.
vec3 sample_local_inscattering(vec3 local_scattering, vec3 ray_dir, vec3 world_pos) {
    float local_r = length(world_pos);
    vec3 local_up = normalize(world_pos);
    vec3 inscattering = vec3(0.0);
    for (int light_i = 0; light_i < lightCount; light_i++) {
        vec3 direction_to_light = lightDirection[light_i].xyz;

        float mu_light = dot(direction_to_light, local_up);
        float neg_LdotV = dot(direction_to_light, ray_dir);

        vec3 transmittance_to_light = sample_transmittance_lut(local_r, mu_light);
        vec3 shadow_factor = transmittance_to_light * float(!ray_intersects_ground(local_r, mu_light));
        vec3 scattering_coeff = sample_scattering_lut(local_r, neg_LdotV);

        vec3 scattering_factor = shadow_factor * scattering_coeff;

        vec3 psi_ms = sample_multiscattering_lut(local_r, mu_light);
        vec3 multiscattering_factor = psi_ms * local_scattering;

        inscattering += lightColor[light_i].rgb * (scattering_factor + multiscattering_factor);
    }
    return inscattering;
}

vec3 clamp_to_surface(vec3 position) {
    float min_radius = INNER_RADIUS + EPSILON;
    float r = length(position);
    if (r < min_radius) {
        vec3 up = normalize(position);
        return up * min_radius;
    }
    return position;
}

float max_atmosphere_distance(float r, float mu) {
    float t_top = distance_to_top_atmosphere_boundary(r, mu);
    float t_bottom = distance_to_bottom_atmosphere_boundary(r, mu);
    bool hits = ray_intersects_ground(r, mu);
    return mix(t_top, t_bottom, float(hits));
}

// The observer's position in the atmosphere.
vec3 get_view_position() {
    vec3 atmo_pos = (worldToAtmosphere * vec4(viewPosition, 1.0)).xyz;
    return clamp_to_surface(atmo_pos);
}

// Up at distance t along the ray, with up at the view position taken as +y.
vec3 get_local_up(float r, float t, vec3 ray_dir) {
    return normalize(vec3(0.0, r, 0.0) + t * ray_dir);
}

vec2 uv_to_ndc(vec2 uv) {
    return uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
}

vec3 direction_world_to_atmosphere(vec3 dir_ws) {
    return (atmosphereFromWorld * vec4(dir_ws, 0.0)).xyz;
}

vec3 direction_atmosphere_to_world(vec3 dir_as) {
    return (worldFromAtmosphere * vec4(dir_as, 0.0)).xyz;
}

// The normalised world-space ray through uv (through the near plane, like Bevy).
vec3 uv_to_ray_direction(vec2 uv) {
    vec4 view_position_homogeneous = viewFromClip * vec4(uv_to_ndc(uv), 1.0, 1.0);
    vec3 view_ray_direction = view_position_homogeneous.xyz / view_position_homogeneous.w;
    vec3 ray_direction = (worldFromView * vec4(view_ray_direction, 0.0)).xyz;
    return normalize(ray_direction);
}

vec3 zenith_azimuth_to_ray_dir(float zenith, float azimuth) {
    float sin_zenith = sin(zenith);
    float mu = cos(zenith);
    float sin_azimuth = sin(azimuth);
    float cos_azimuth = cos(azimuth);
    return vec3(sin_azimuth * sin_zenith, mu, -cos_azimuth * sin_zenith);
}

struct RaymarchSegment {
    float start;
    float end;
};

RaymarchSegment get_raymarch_segment(float r, float mu) {
    vec2 atmosphere_intersections = ray_sphere_intersect(r, mu, OUTER_RADIUS);
    vec2 ground_intersections = ray_sphere_intersect(r, mu, INNER_RADIUS);

    RaymarchSegment segment = RaymarchSegment(0.0, 0.0);

    if (r < INNER_RADIUS) {
        segment.start = ground_intersections.y;
        segment.end = atmosphere_intersections.y;
    } else if (r < OUTER_RADIUS) {
        segment.start = 0.0;
        segment.end = ray_intersects_ground(r, mu) ? ground_intersections.x : atmosphere_intersections.y;
    } else {
        if (atmosphere_intersections.x < 0.0) {
            return segment;
        }
        segment.start = atmosphere_intersections.x;
        segment.end = ray_intersects_ground(r, mu) ? ground_intersections.x : atmosphere_intersections.y;
    }

    return segment;
}

struct RaymarchResult {
    vec3 inscattering;
    vec3 transmittance;
};

RaymarchResult raymarch_atmosphere(vec3 pos, vec3 ray_dir, float t_max, float max_samples, bool ground) {
    float r = length(pos);
    vec3 up = normalize(pos);
    float mu = dot(ray_dir, up);

    // Fewer samples close to the scene.
    float sample_count = mix(1.0, max_samples, clamp(t_max * 0.01, 0.0, 1.0));

    RaymarchSegment segment = get_raymarch_segment(r, mu);
    float t_start = segment.start;
    float t_end = segment.end;

    t_end = min(t_end, t_max);
    float t_total = t_end - t_start;

    RaymarchResult result = RaymarchResult(vec3(0.0), vec3(1.0));

    if (t_total <= 0.0) {
        return result;
    }

    float prev_t = t_start;
    vec3 optical_depth = vec3(0.0);
    for (float s = 0.0; s < sample_count; s += 1.0) {
        float t_i = t_start + t_total * (s + MIDPOINT_RATIO) / sample_count;
        float dt_i = (t_i - prev_t);
        prev_t = t_i;

        vec3 sample_pos = pos + ray_dir * t_i;
        float local_r = length(sample_pos);

        vec3 absorption = sample_density_lut(local_r, ABSORPTION_DENSITY);
        vec3 scattering = sample_density_lut(local_r, SCATTERING_DENSITY);
        vec3 extinction = absorption + scattering;

        vec3 sample_optical_depth = extinction * dt_i;
        optical_depth += sample_optical_depth;
        vec3 sample_transmittance = exp(-sample_optical_depth);

        vec3 inscattering = sample_local_inscattering(scattering, ray_dir, sample_pos);

        vec3 s_int = (inscattering - inscattering * sample_transmittance) / max(extinction, MIN_EXTINCTION);
        result.inscattering += result.transmittance * s_int;

        result.transmittance *= sample_transmittance;
        if (all(lessThan(result.transmittance, vec3(0.001)))) {
            break;
        }
    }

    // Light reflected by the planet's ground.
    if (ground && ray_intersects_ground(r, mu)) {
        for (int light_i = 0; light_i < lightCount; light_i++) {
            vec3 light_dir = lightDirection[light_i].xyz;
            vec3 light_color = lightColor[light_i].rgb;
            vec3 transmittance_to_ground = exp(-optical_depth);
            vec3 sphere_point = pos + ray_dir * t_end;
            vec3 sphere_normal = normalize(sphere_point);
            float mu_light = dot(light_dir, sphere_normal);
            vec3 transmittance_to_light = sample_transmittance_lut(0.0, mu_light);
            vec3 light_luminance = transmittance_to_light * max(mu_light, 0.0) * light_color;
            vec3 ground_luminance = transmittance_to_ground * groundAlbedo / PI;
            result.inscattering += ground_luminance * light_luminance;
        }
    }

    return result;
}
`;

const out = /* glsl */ `
layout(location = 0) out highp vec4 fragColor;
`;

/** Every LUT pass draws a fullscreen triangle (three's FullScreenQuad). */
export const skyVertex = /* glsl */ `
void main() {
    gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

// ---------------------------------------------------------------- transmittance.wgsl

export const transmittanceFragment = /* glsl */ `
${common}
${out}
const vec2 TRANSMITTANCE_LUT_SIZE = vec2(256.0, 128.0);
const int TRANSMITTANCE_LUT_SAMPLES = 40;

vec3 ray_optical_depth(float r, float mu, int sample_count) {
    float t_max = max_atmosphere_distance(r, mu);
    vec3 optical_depth = vec3(0.0);
    float prev_t = 0.0;

    for (int i = 0; i < sample_count; i++) {
        float t_i = t_max * (float(i) + MIDPOINT_RATIO) / float(sample_count);
        float dt = t_i - prev_t;
        prev_t = t_i;

        float r_i = get_local_r(r, mu, t_i);

        vec3 absorption = sample_density_lut(r_i, ABSORPTION_DENSITY);
        vec3 scattering = sample_density_lut(r_i, SCATTERING_DENSITY);
        vec3 extinction = absorption + scattering;
        optical_depth += extinction * dt;
    }

    return optical_depth;
}

void main() {
    vec2 uv = (floor(gl_FragCoord.xy) + 0.5) / TRANSMITTANCE_LUT_SIZE;
    vec2 r_mu = transmittance_lut_uv_to_r_mu(uv);
    vec3 optical_depth = ray_optical_depth(r_mu.x, r_mu.y, TRANSMITTANCE_LUT_SAMPLES);
    fragColor = vec4(exp(-optical_depth), 1.0);
}
`;

// ---------------------------------------------------------------- multiscattering.wgsl

export const multiscatteringFragment = /* glsl */ `
${common}
${out}
const int MULTISCATTERING_LUT_DIRS = 64;
const int MULTISCATTERING_LUT_SAMPLES = 20;
const vec2 PHI_2 = vec2(1.3247179572447460259609088, 1.7548776662466927600495087);

vec2 s2_sequence(int n) {
    return fract(0.5 + float(n) * PHI_2);
}

// Lambert equal-area projection.
vec3 uv_to_sphere(vec2 uv) {
    float phi = PI_2 * uv.y;
    float sin_lambda = 2.0 * uv.x - 1.0;
    float cos_lambda = sqrt(1.0 - sin_lambda * sin_lambda);
    return vec3(cos_lambda * cos(phi), cos_lambda * sin(phi), sin_lambda);
}

void sample_multiscattering_dir(float r, vec3 ray_dir, vec3 light_dir, out vec3 l_2, out vec3 f_ms) {
    float mu_view = ray_dir.y;
    float t_max = max_atmosphere_distance(r, mu_view);

    float dt = t_max / float(MULTISCATTERING_LUT_SAMPLES);
    vec3 optical_depth = vec3(0.0);

    l_2 = vec3(0.0);
    f_ms = vec3(0.0);
    vec3 throughput = vec3(1.0);
    for (int i = 0; i < MULTISCATTERING_LUT_SAMPLES; i++) {
        float t_i = dt * (float(i) + 0.5);
        float local_r = get_local_r(r, mu_view, t_i);
        vec3 local_up = get_local_up(r, t_i, ray_dir);

        vec3 absorption = sample_density_lut(local_r, ABSORPTION_DENSITY);
        vec3 scattering = sample_density_lut(local_r, SCATTERING_DENSITY);
        vec3 extinction = absorption + scattering;

        vec3 sample_optical_depth = extinction * dt;
        vec3 sample_transmittance = exp(-sample_optical_depth);
        optical_depth += sample_optical_depth;

        vec3 ms = scattering;
        vec3 ms_int = (ms - ms * sample_transmittance) / max(extinction, MIN_EXTINCTION);
        f_ms += throughput * ms_int;

        float mu_light = dot(light_dir, local_up);
        vec3 transmittance_to_light = sample_transmittance_lut(local_r, mu_light);
        vec3 shadow_factor = transmittance_to_light * float(!ray_intersects_ground(local_r, mu_light));

        vec3 s = scattering * shadow_factor * FRAC_4_PI;
        vec3 s_int = (s - s * sample_transmittance) / max(extinction, MIN_EXTINCTION);
        l_2 += throughput * s_int;

        throughput *= sample_transmittance;
        if (all(lessThan(throughput, vec3(0.001)))) {
            break;
        }
    }

    // Light reflected by the planet's ground.
    if (ray_intersects_ground(r, mu_view)) {
        vec3 transmittance_to_ground = exp(-optical_depth);
        vec3 local_up = get_local_up(r, t_max, ray_dir);
        float mu_light = dot(light_dir, local_up);
        vec3 transmittance_to_light = sample_transmittance_lut(0.0, mu_light);
        vec3 ground_luminance = transmittance_to_light * transmittance_to_ground * max(mu_light, 0.0) * groundAlbedo;
        l_2 += ground_luminance;
    }
}

void main() {
    vec2 uv = (floor(gl_FragCoord.xy) + 0.5) / MULTISCATTERING_LUT_SIZE;

    vec2 r_mu = multiscattering_lut_uv_to_r_mu(uv);
    vec3 light_dir = normalize(vec3(0.0, r_mu.y, -1.0));

    // The compute pass's 64 invocations per texel, then its workgroup reduction in tree order.
    float sample_weight = 4.0 * PI / 64.0;
    vec3 multi_scat[MULTISCATTERING_LUT_DIRS];
    vec3 l[MULTISCATTERING_LUT_DIRS];
    for (int z = 0; z < MULTISCATTERING_LUT_DIRS; z++) {
        vec3 ray_dir = uv_to_sphere(s2_sequence(z));
        vec3 l_2, f_ms;
        sample_multiscattering_dir(r_mu.x, ray_dir, light_dir, l_2, f_ms);
        multi_scat[z] = f_ms * sample_weight;
        l[z] = l_2 * sample_weight;
    }
    for (int stride = 32; stride > 0; stride >>= 1) {
        for (int z = 0; z < stride; z++) {
            multi_scat[z] += multi_scat[z + stride];
            l[z] += l[z + stride];
        }
    }

    // Isotropic phase function.
    vec3 f_ms = multi_scat[0] * FRAC_4_PI;
    vec3 l_2 = l[0] * FRAC_4_PI;

    // Equation 10: geometric series over all scattering orders.
    vec3 psi_ms = l_2 / (1.0 - f_ms);
    fragColor = vec4(psi_ms, 1.0);
}
`;

// ---------------------------------------------------------------- sky_view.wgsl

export const skyViewFragment = /* glsl */ `
${common}
${out}
const float SKY_VIEW_LUT_SAMPLES = 16.0;

void main() {
    vec2 uv = floor(gl_FragCoord.xy) / SKY_VIEW_LUT_SIZE;

    vec3 cam_pos = get_view_position();
    float r = length(cam_pos);
    vec2 zenith_azimuth = sky_view_lut_uv_to_zenith_azimuth(r, uv);

    vec3 ray_dir_as = zenith_azimuth_to_ray_dir(zenith_azimuth.x, zenith_azimuth.y);
    vec3 ray_dir_ws = direction_atmosphere_to_world(ray_dir_as);

    vec3 world_pos = cam_pos;
    vec3 up = normalize(world_pos);
    float mu = dot(ray_dir_ws, up);
    float t_max = max_atmosphere_distance(r, mu);

    RaymarchResult result = raymarch_atmosphere(world_pos, ray_dir_ws, t_max, SKY_VIEW_LUT_SAMPLES, true);

    fragColor = vec4(result.inscattering, 1.0);
}
`;

// ---------------------------------------------------------------- aerial_view.wgsl

// One depth slice of the aerial-view march: continues the running sums of the previous slice
// (prev_t in totalIn.a) and writes them for the next one.
export const aerialMarchFragment = /* glsl */ `
${common}
const int AERIAL_VIEW_LUT_SAMPLES = 10;
uniform int slice;
uniform sampler2D totalIn;
uniform sampler2D throughputIn;
layout(location = 0) out highp vec4 totalOut;
layout(location = 1) out highp vec4 throughputOut;

void main() {
    ivec2 idx = ivec2(gl_FragCoord.xy);
    vec2 uv = (vec2(idx) + 0.5) / AERIAL_VIEW_LUT_SIZE.xy;
    vec3 ray_dir = uv_to_ray_direction(uv);
    vec3 world_pos = get_view_position();

    float t_max = AERIAL_VIEW_LUT_MAX_DISTANCE;

    float prev_t = 0.0;
    vec3 total_inscattering = vec3(0.0);
    vec3 throughput = vec3(1.0);
    if (slice > 0) {
        vec4 total = texelFetch(totalIn, idx, 0);
        total_inscattering = total.rgb;
        prev_t = total.a;
        throughput = texelFetch(throughputIn, idx, 0).rgb;
    }

    for (int step_i = 0; step_i < AERIAL_VIEW_LUT_SAMPLES; step_i++) {
        float t_i = t_max * (float(slice) + ((float(step_i) + MIDPOINT_RATIO) / float(AERIAL_VIEW_LUT_SAMPLES))) / AERIAL_VIEW_LUT_SIZE.z;
        float dt = (t_i - prev_t);
        prev_t = t_i;

        vec3 sample_pos = world_pos + ray_dir * t_i;
        float local_r = length(sample_pos);

        vec3 absorption = sample_density_lut(local_r, ABSORPTION_DENSITY);
        vec3 scattering = sample_density_lut(local_r, SCATTERING_DENSITY);
        vec3 extinction = absorption + scattering;

        vec3 sample_optical_depth = extinction * dt;
        vec3 sample_transmittance = exp(-sample_optical_depth);

        vec3 inscattering = sample_local_inscattering(scattering, ray_dir, sample_pos);

        vec3 s_int = (inscattering - inscattering * sample_transmittance) / max(extinction, MIN_EXTINCTION);
        total_inscattering += throughput * s_int;

        throughput *= sample_transmittance;
        if (all(lessThan(throughput, vec3(0.001)))) {
            break;
        }
    }
    totalOut = vec4(total_inscattering, prev_t);
    throughputOut = vec4(throughput, 0.0);
}
`;

// Stores one slice of the march in log space, so slices interpolate exponential values linearly.
export const aerialStoreFragment = /* glsl */ `
uniform sampler2D totalIn;
${out}
void main() {
    vec3 total_inscattering = texelFetch(totalIn, ivec2(gl_FragCoord.xy), 0).rgb;
    fragColor = vec4(log(max(total_inscattering, vec3(1e-6))), 0.0);
}
`;

// ---------------------------------------------------------------- environment.wgsl

const sampleCubeDir = /* glsl */ `
// Bevy render::utils::sample_cube_dir.
vec3 sample_cube_dir(vec2 uv, int face) {
    vec2 uvc = 2.0 * uv - 1.0;
    vec3 dir = vec3(0.0);
    if (face == 0) dir = vec3(1.0, -uvc.y, -uvc.x);
    else if (face == 1) dir = vec3(-1.0, -uvc.y, uvc.x);
    else if (face == 2) dir = vec3(uvc.x, 1.0, uvc.y);
    else if (face == 3) dir = vec3(uvc.x, -1.0, -uvc.y);
    else if (face == 4) dir = vec3(uvc.x, -uvc.y, 1.0);
    else if (face == 5) dir = vec3(-uvc.x, -uvc.y, -1.0);
    return normalize(dir);
}
`;

// The sky-view LUT rendered into one face of a cube (a layer of the filter's source array), the
// source of the sky's environment-map light.
export const environmentFragment = /* glsl */ `
${common}
${sampleCubeDir}
${out}
uniform int face;
uniform float size;

void main() {
    vec2 uv = (floor(gl_FragCoord.xy) + 0.5) / size;

    vec3 ray_dir_ws = sample_cube_dir(uv, face);

    // Cube maps are left-handed.
    ray_dir_ws.z = -ray_dir_ws.z;

    vec3 world_pos = get_view_position();
    float r = length(world_pos);

    vec3 ray_dir_as = direction_world_to_atmosphere(ray_dir_ws);
    vec3 inscattering = sample_sky_view_lut(r, ray_dir_as);
    fragColor = vec4(inscattering, 1.0);
}
`;

// ---------------------------------------------------------------- environment_filter.wgsl

const filterCommon = /* glsl */ `
const float PI = 3.141592653589793;
const float PI_2 = 6.283185307179586;
// The source cube (Bevy's intermediate 2D array, one layer per face) with its mip chain.
uniform highp sampler2DArray inputTexture;
uniform int face;
uniform float size;
${out}
`;

// Bevy's single-pass downsampler, one pass per level. The first level is point sampled: Bevy
// samples it bilinearly at the centre of each block's first texel. GL cannot sample a level of
// the texture it renders into, so each level is written to a scratch array and copied back.
export const downsampleFragment = /* glsl */ `
${filterCommon}
uniform int mipLevel;

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy) * 2;
    int level = mipLevel - 1;
    if (mipLevel < 2) {
        fragColor = texelFetch(inputTexture, ivec3(p, face), level);
        return;
    }
    vec4 v0 = texelFetch(inputTexture, ivec3(p, face), level);
    vec4 v1 = texelFetch(inputTexture, ivec3(p + ivec2(1, 0), face), level);
    vec4 v2 = texelFetch(inputTexture, ivec3(p + ivec2(0, 1), face), level);
    vec4 v3 = texelFetch(inputTexture, ivec3(p + ivec2(1, 1), face), level);
    fragColor = (v0 + v1 + v2 + v3) * 0.25;
}
`;

// One face of the radiance or irradiance atlas, into its cube face.
export const copyFaceFragment = /* glsl */ `
uniform sampler2D atlas;
uniform int face;
uniform float size;
${out}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    fragColor = texelFetch(atlas, ivec2(face * int(size) + p.x, p.y), 0);
}
`;

export const copyLevelFragment = /* glsl */ `
${filterCommon}
uniform int mipLevel;

void main() {
    fragColor = texelFetch(inputTexture, ivec3(ivec2(gl_FragCoord.xy), face), mipLevel);
}
`;

const filterFunctions = /* glsl */ `
// bevy_render::maths, utils

float copysign(float a, float b) {
    return uintBitsToFloat((floatBitsToUint(a) & 0x7FFFFFFFu) | (floatBitsToUint(b) & 0x80000000u));
}

mat3 orthonormalize(vec3 z_basis) {
    float sgn = copysign(1.0, z_basis.z);
    float a = -1.0 / (sgn + z_basis.z);
    float b = z_basis.x * z_basis.y * a;
    vec3 x_basis = vec3(1.0 + sgn * z_basis.x * z_basis.x * a, sgn * b, -sgn * z_basis.x);
    vec3 y_basis = vec3(b, sgn + z_basis.y * z_basis.y * a, -z_basis.y);
    return mat3(x_basis, y_basis, z_basis);
}

float rand_f(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) * uintBitsToFloat(0x2f800004u);
}

vec2 rand_vec2f(inout uint state) {
    float x = rand_f(state);
    float y = rand_f(state);
    return vec2(x, y);
}

// WGSL reverseBits (GLSL ES 3.0 has no bitfieldReverse).
uint reverse_bits(uint b) {
    b = (b << 16u) | (b >> 16u);
    b = ((b & 0x00ff00ffu) << 8u) | ((b & 0xff00ff00u) >> 8u);
    b = ((b & 0x0f0f0f0fu) << 4u) | ((b & 0xf0f0f0f0u) >> 4u);
    b = ((b & 0x33333333u) << 2u) | ((b & 0xccccccccu) >> 2u);
    b = ((b & 0x55555555u) << 1u) | ((b & 0xaaaaaaaau) >> 1u);
    return b;
}

vec2 hammersley_2d(uint i, uint n) {
    float inv_n = 1.0 / float(n);
    float vdc = float(reverse_bits(i)) * 2.3283064365386963e-10;
    return vec2(float(i) * inv_n, vdc);
}

vec3 sample_cosine_hemisphere(vec3 normal, inout uint rng) {
    float cos_theta = 1.0 - 2.0 * rand_f(rng);
    float phi = PI_2 * rand_f(rng);
    float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    vec3 direction = normal + vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    float len_sq = dot(direction, direction);
    if (len_sq < 1e-8) return normal;
    return direction * inversesqrt(len_sq);
}

${sampleCubeDir}

// The face (z) and its uv (xy) a direction falls on.
vec3 dir_to_cube_uv(vec3 dir) {
    vec3 abs_dir = abs(dir);
    float f = 0.0;
    vec2 uv = vec2(0.0);

    if (abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z) {
        if (dir.x > 0.0) {
            f = 0.0;
            uv = vec2(-dir.z, -dir.y) / dir.x;
        } else {
            f = 1.0;
            uv = vec2(dir.z, -dir.y) / abs_dir.x;
        }
    } else if (abs_dir.y >= abs_dir.x && abs_dir.y >= abs_dir.z) {
        if (dir.y > 0.0) {
            f = 2.0;
            uv = vec2(dir.x, dir.z) / dir.y;
        } else {
            f = 3.0;
            uv = vec2(dir.x, -dir.z) / abs_dir.y;
        }
    } else {
        if (dir.z > 0.0) {
            f = 4.0;
            uv = vec2(dir.x, -dir.y) / dir.z;
        } else {
            f = 5.0;
            uv = vec2(-dir.x, -dir.y) / abs_dir.z;
        }
    }

    return vec3(uv * 0.5 + 0.5, f);
}

// pbr_lighting

float D_GGX(float roughness, float NdotH) {
    float oneMinusNdotHSquared = 1.0 - NdotH * NdotH;
    float a = NdotH * roughness;
    float k = roughness / (oneMinusNdotHSquared + a * a);
    float d = k * k * (1.0 / PI);
    return d;
}

float ggx_vndf_pdf(vec3 i, float NdotH, float roughness) {
    float ndf = D_GGX(roughness, NdotH);

    vec2 ai = roughness * i.xy;
    float len2 = dot(ai, ai);
    float t = sqrt(len2 + i.z * i.z);
    if (i.z >= 0.0) {
        float a = roughness;
        float s = 1.0 + length(i.xy);
        float a2 = a * a;
        float s2 = s * s;
        float k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
        return ndf / (2.0 * (k * i.z + t));
    }

    return ndf * (t - i.z) / (2.0 * len2);
}

vec3 sample_visible_ggx(vec2 xi, float roughness, vec3 normal, vec3 view) {
    vec3 n = normal;
    float alpha = roughness;

    float wi_n = dot(view, n);
    vec3 wi_z = -n * wi_n;
    vec3 wi_xy = view + wi_z;

    vec3 wi_std = -normalize(alpha * wi_xy + wi_z);

    float wi_std_z = dot(wi_std, n);

    float s = 1.0 + length(wi_xy);
    float a = clamp(alpha, 0.0, 1.0);
    float a2 = a * a;
    float s2 = s * s;
    float k = (1.0 - a2) * s2 / (s2 + a2 * wi_n * wi_n);
    float b = wi_n > 0.0 ? k * wi_std_z : wi_std_z;

    float z = 1.0 - xi.y * (1.0 + b);
    float sin_theta = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * PI * xi.x - PI;
    float x = sin_theta * cos(phi);
    float y = sin_theta * sin(phi);
    vec3 c_std = vec3(x, y, z);

    mat3 TBN = orthonormalize(n);
    vec3 c = TBN * c_std;

    vec3 wm_std = c + wi_std;
    vec3 wm_std_z = n * dot(n, wm_std);
    vec3 wm_std_xy = wm_std_z - wm_std;

    vec3 H = normalize(alpha * wm_std_xy + wm_std_z);

    return reflect(-view, H);
}

float G_Smith(float NdotV, float NdotL, float roughness) {
    float k = roughness / 2.0;
    float GGXL = NdotL / (NdotL * (1.0 - k) + k);
    float GGXV = NdotV / (NdotV * (1.0 - k) + k);
    return GGXL * GGXV;
}

float perceptualRoughnessToRoughness(float perceptualRoughness) {
    float clampedPerceptualRoughness = clamp(perceptualRoughness, 0.089, 1.0);
    return clampedPerceptualRoughness * clampedPerceptualRoughness;
}

// environment_filter

vec4 sample_environment(vec3 dir, float level) {
    vec3 cube_uv = dir_to_cube_uv(dir);
    return textureLod(inputTexture, cube_uv, level);
}

vec4 sample_noise(uvec2 pixel_coords) {
    uint rng_state = (pixel_coords.x * 3966231743u) ^ (pixel_coords.y * 3928936651u);
    vec2 rnd = rand_vec2f(rng_state);
    return vec4(rnd, 0.0, 0.0);
}

// Filtered importance sampling: the source mip whose texels match a sample's solid angle.
float calculate_environment_map_lod(float pdf, float width, float samples) {
    float omega_s = 1.0 / (samples * pdf);
    float omega_p = 4.0 * PI / (6.0 * width * width);
    return 0.5 * log2(omega_s / omega_p);
}
`;

// The radiance and irradiance passes write all six faces of a level side by side into an atlas
// (face = column block of \`size\` texels), one draw per level as Bevy has one dispatch, then
// copyFaceFragment copies each face into the cube. Per-face draws of these long loops leave the
// GPU mostly idle.

// One mip of the GGX-prefiltered specular cube.
export const radianceFragment = /* glsl */ `
${filterCommon}
${filterFunctions}
uniform int sampleCount;
uniform float roughnessLevel;

void main() {
    float source_width = float(textureSize(inputTexture, 0).x);
    float invSize = 1.0 / size;

    ivec2 p = ivec2(gl_FragCoord.xy);
    int cube_face = p.x / int(size);
    uvec2 coords = uvec2(p.x - cube_face * int(size), p.y);

    vec2 uv = (vec2(coords) + 0.5) * invSize;
    vec3 normal = sample_cube_dir(uv, cube_face);

    // View direction = normal for the radiance map.
    vec3 view = normal;

    float perceptual_roughness = roughnessLevel;
    float roughness = perceptualRoughnessToRoughness(perceptual_roughness);

    vec4 vector_noise = sample_noise(coords);

    vec3 radiance = vec3(0.0);
    float total_weight = 0.0;

    // Mirror reflection.
    if (roughness < 0.01) {
        radiance = sample_environment(normal, 0.0).rgb;
        fragColor = vec4(radiance, 1.0);
        return;
    }

    uint sample_count = uint(sampleCount);

    for (uint i = 0u; i < sample_count; i++) {
        vec2 xi = hammersley_2d(i, sample_count);
        xi = fract(xi + vector_noise.rg);

        vec3 light_dir = sample_visible_ggx(xi, roughness, normal, view);

        float NdotL = dot(normal, light_dir);

        if (NdotL > 0.0) {
            vec3 half_vector = normalize(view + light_dir);
            float NdotH = dot(normal, half_vector);
            float NdotV = dot(normal, view);

            float G = G_Smith(NdotV, NdotL, roughness);

            float pdf = ggx_vndf_pdf(view, NdotH, roughness);

            float lod = calculate_environment_map_lod(pdf, source_width, float(sample_count));

            float source_mip = max(0.0, lod);

            vec3 sample_color = sample_environment(light_dir, source_mip).rgb;

            radiance += sample_color * NdotL * G;
            total_weight += NdotL * G;
        }
    }

    if (total_weight > 0.0) {
        radiance = radiance / total_weight;
    }

    fragColor = vec4(radiance, 1.0);
}
`;

// The 32² diffuse cube.
export const irradianceFragment = /* glsl */ `
${filterCommon}
${filterFunctions}
uniform int sampleCount;

void main() {
    float invSize = 1.0 / size;

    ivec2 p = ivec2(gl_FragCoord.xy);
    int cube_face = p.x / int(size);
    uvec2 coords = uvec2(p.x - cube_face * int(size), p.y);

    vec2 uv = (vec2(coords) + 0.5) * invSize;
    vec3 normal = sample_cube_dir(uv, cube_face);

    vec3 irradiance = vec3(0.0);

    uint sample_count = uint(sampleCount);
    for (uint i = 0u; i < sample_count; i++) {
        uint rng = (coords.x * 2131358057u) ^ (coords.y * 3416869721u) ^ (uint(cube_face) * 1199786941u) ^ (i * 566200673u);

        vec3 sample_dir = sample_cosine_hemisphere(normal, rng);

        vec3 sample_color = sample_environment(sample_dir, 0.0).rgb;

        irradiance += sample_color;
    }

    irradiance = irradiance / float(sample_count);

    fragColor = vec4(irradiance, 1.0);
}
`;

// ---------------------------------------------------------------- render_sky.wgsl

// After the opaque draws: the sky where nothing was drawn and the aerial perspective over
// geometry, `inscattering + dst × transmittance`. Bevy blends with the second source
// (per-channel transmittance) where the device has dual-source blending; WebGL 2 has none, so
// the pass runs once per colour channel with only that channel written and its transmittance in
// alpha (blend ONE, SRC_ALPHA). The frame is stored top row first (the main pass renders upside
// down), so gl_FragCoord is wgpu's pixel position and its uv Bevy's.
export const renderSkyFragment = /* glsl */ `
${common}
${out}
// The colour channel this pass writes.
uniform int channel;
// The opaque draws' depth (three's projection, the MSAA depth resolved).
uniform highp sampler2D sceneDepth;
// three's clip-to-view for that depth.
uniform mat4 cameraViewFromClip;
uniform vec2 viewportSize;

// Bevy ndc_to_camera_dist, from three's depth.
float ndc_to_camera_dist(vec2 ndc, float depth) {
    vec4 view_pos = cameraViewFromClip * vec4(ndc, depth * 2.0 - 1.0, 1.0);
    vec3 p_view = view_pos.xyz / view_pos.w;
    vec3 p_world = (worldFromView * vec4(p_view, 1.0)).xyz;
    vec3 p_atmo = (worldToAtmosphere * vec4(p_world, 1.0)).xyz;
    vec3 cam_atmo = (worldToAtmosphere * vec4(viewPosition, 1.0)).xyz;
    return length(p_atmo - cam_atmo);
}

vec3 sample_sun_radiance(vec3 ray_dir_ws) {
    vec3 view_pos = get_view_position();
    float r = length(view_pos);
    vec3 up = normalize(view_pos);
    float mu_view = dot(ray_dir_ws, up);
    float shadow_factor = float(!ray_intersects_ground(r, mu_view));
    vec3 sun_radiance = vec3(0.0);
    for (int light_i = 0; light_i < lightCount; light_i++) {
        float neg_LdotV = dot(lightDirection[light_i].xyz, ray_dir_ws);
        float angle_to_sun = fast_acos(clamp(neg_LdotV, -1.0, 1.0));
        float w = max(0.5 * fwidth(angle_to_sun), 1e-6);
        float sun_angular_size = lightDisk[light_i].x;
        float sun_intensity = lightDisk[light_i].y;
        if (sun_angular_size > 0.0 && sun_intensity > 0.0) {
            float factor = 1.0 - smoothstep(sun_angular_size * 0.5 - w, sun_angular_size * 0.5 + w, angle_to_sun);
            float sun_solid_angle = (sun_angular_size * sun_angular_size) * 0.25 * PI;
            sun_radiance += (lightColor[light_i].rgb / sun_solid_angle) * sun_intensity * factor * shadow_factor;
        }
    }
    return sun_radiance;
}

void main() {
    vec2 uv = gl_FragCoord.xy / viewportSize;
    float depth = texelFetch(sceneDepth, ivec2(gl_FragCoord.xy), 0).r;

    vec3 ray_dir_ws = uv_to_ray_direction(uv);
    vec3 world_pos = get_view_position();
    float r = length(world_pos);
    vec3 up = normalize(world_pos);
    float mu = dot(ray_dir_ws, up);

    vec3 transmittance;
    vec3 inscattering;

    vec3 sun_radiance = sample_sun_radiance(ray_dir_ws);

    // Nothing drawn: three's cleared depth (Bevy's reverse-Z 0).
    if (depth >= 1.0) {
        vec3 ray_dir_as = direction_world_to_atmosphere(ray_dir_ws);
        transmittance = sample_transmittance_lut(r, mu);
        inscattering = sample_sky_view_lut(r, ray_dir_as);
        inscattering += sun_radiance * transmittance;
    } else {
        float t = ndc_to_camera_dist(uv_to_ndc(uv), depth);
        inscattering = sample_aerial_view_lut(uv, t);
        transmittance = sample_transmittance_lut_segment(r, mu, t);
    }

    // Exposure compensation.
    inscattering *= viewExposure;
    fragColor = vec4(inscattering, transmittance[channel]);
}
`;
