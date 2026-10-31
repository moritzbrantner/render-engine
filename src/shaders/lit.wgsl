// Forward PBR pass and sun shadow-map pass.
//
// The BRDF follows Bevy 0.20 `bevy_pbr` (itself Filament): Burley diffuse, GGX distribution,
// height-correlated Smith visibility, Schlick Fresnel with the f90 specular-occlusion trick, the
// multiscatter energy compensation and the polynomial F_AB environment approximation for ambient
// light. Output is ACES-fitted tone mapped linear colour; the sRGB target encodes it.

const PI: f32 = 3.141592653589793;

const MAX_CASCADES: u32 = 4u;

struct Frame {
    clip_from_world: mat4x4<f32>,
    view_from_world: mat4x4<f32>,
    cascade_clip_from_world: array<mat4x4<f32>, MAX_CASCADES>,
    // x: far bound (m), y: texel size (m).
    cascade_params: array<vec4<f32>, MAX_CASCADES>,
    // xyz: camera position.
    eye: vec4<f32>,
    // xyz: unit direction towards the sun.
    direction_to_light: vec4<f32>,
    // rgb: sun colour × illuminance (lux).
    sun_color: vec4<f32>,
    // rgb: ambient colour × brightness (cd/m²).
    ambient_color: vec4<f32>,
    // x: depth bias (m), y: normal bias (texels), z: cascade overlap proportion, w: cascade count.
    shadow: vec4<f32>,
    // x: exposure, y: 1 when the sun casts shadows, z: 1 when there is a sun, w: 1 for the
    // Gaussian shadow filter (else hardware 2×2).
    flags: vec4<f32>,
}

struct Instance {
    world_from_local: mat4x4<f32>,
    normal_from_local: mat3x3<f32>,
    base_color: vec4<f32>,
    // x: metallic, y: perceptual roughness, z: reflectance.
    pbr: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
// Main pass only.
@group(1) @binding(0) var shadow_maps: texture_depth_2d_array;
@group(1) @binding(1) var shadow_sampler: sampler_comparison;
// Shadow pass only: the cascade being rendered (dynamic offset).
@group(1) @binding(2) var<uniform> cascade_clip_from_world: mat4x4<f32>;

struct VertexIn {
    @builtin(instance_index) instance: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

// ---------------------------------------------------------------- shadow pass

@vertex
fn shadow_vs(in: VertexIn) -> @builtin(position) vec4<f32> {
    let world = instances[in.instance].world_from_local * vec4(in.position, 1.0);
    var clip = cascade_clip_from_world * world;
    // Shadow pancaking: casters between the sun and the cascade's near plane clamp onto it
    // (reverse Z, orthographic, so w == 1).
    clip.z = min(clip.z, 1.0);
    return clip;
}

// ---------------------------------------------------------------- main pass

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) instance: u32,
}

@vertex
fn lit_vs(in: VertexIn) -> VertexOut {
    let inst = instances[in.instance];
    let world = inst.world_from_local * vec4(in.position, 1.0);
    var out: VertexOut;
    out.clip = frame.clip_from_world * world;
    out.world = world.xyz;
    out.normal = inst.normal_from_local * in.normal;
    out.instance = in.instance;
    return out;
}

fn perceptual_roughness_to_roughness(p: f32) -> f32 {
    let c = clamp(p, 0.089, 1.0);
    return c * c;
}

fn D_GGX(roughness: f32, NdotH: f32) -> f32 {
    let one_minus = 1.0 - NdotH * NdotH;
    let a = NdotH * roughness;
    let k = roughness / (one_minus + a * a);
    return k * k * (1.0 / PI);
}

fn V_SmithGGXCorrelated(roughness: f32, NdotV: f32, NdotL: f32) -> f32 {
    let a2 = roughness * roughness;
    let lambda_v = NdotL * sqrt((NdotV - a2 * NdotV) * NdotV + a2);
    let lambda_l = NdotV * sqrt((NdotL - a2 * NdotL) * NdotL + a2);
    return 0.5 / (lambda_v + lambda_l);
}

fn F_Schlick(f0: f32, f90: f32, VdotH: f32) -> f32 {
    return f0 + (f90 - f0) * pow(1.0 - VdotH, 5.0);
}

fn F_Schlick_vec(f0: vec3<f32>, f90: f32, VdotH: f32) -> vec3<f32> {
    return f0 + (f90 - f0) * pow(1.0 - VdotH, 5.0);
}

fn fresnel(f0: vec3<f32>, LdotH: f32) -> vec3<f32> {
    let f90 = saturate(dot(f0, vec3(50.0 * 0.33)));
    return F_Schlick_vec(f0, f90, LdotH);
}

// Polynomial approximation of the split-sum DFG terms (Karis, mobile).
fn F_AB(perceptual_roughness: f32, NdotV: f32) -> vec2<f32> {
    let c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    let r = perceptual_roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    return max(vec2(-1.04, 1.04) * a004 + r.zw, vec2(0.00005));
}

fn env_brdf_approx(f0: vec3<f32>, f_ab: vec2<f32>) -> vec3<f32> {
    return f0 * f_ab.x + f_ab.y;
}

fn RRTAndODTFit(v: vec3<f32>) -> vec3<f32> {
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

fn aces_fitted(color: vec3<f32>) -> vec3<f32> {
    let rgb_to_rrt = mat3x3<f32>(
        vec3(0.59719, 0.35458, 0.04823),
        vec3(0.07600, 0.90834, 0.01566),
        vec3(0.02840, 0.13383, 0.83777),
    );
    let odt_to_rgb = mat3x3<f32>(
        vec3(1.60475, -0.53108, -0.07367),
        vec3(-0.10208, 1.10813, -0.00605),
        vec3(-0.00327, -0.07276, 1.07602),
    );
    var c = color * rgb_to_rrt;
    c = RRTAndODTFit(c);
    c = c * odt_to_rgb;
    return saturate(c);
}

fn sample_hardware(uv: vec2<f32>, depth: f32, layer: u32) -> f32 {
    return textureSampleCompareLevel(shadow_maps, shadow_sampler, uv, layer, depth);
}

// Bevy `sample_shadow_map_castano_thirteen`: a 13-tap Gaussian from 9 hardware PCF taps.
// https://web.archive.org/web/20230210095515/http://the-witness.net/news/2013/09/shadow-mapping-summary-part-1
fn sample_gaussian(light_local: vec2<f32>, depth: f32, layer: u32) -> f32 {
    let size = vec2<f32>(textureDimensions(shadow_maps));
    let inv = 1.0 / size;
    let uv = light_local * size;
    var base_uv = floor(uv + 0.5);
    let s = uv.x + 0.5 - base_uv.x;
    let t = uv.y + 0.5 - base_uv.y;
    base_uv = (base_uv - 0.5) * inv;

    let uw = vec3(4.0 - 3.0 * s, 7.0, 1.0 + 3.0 * s);
    let u = vec3((3.0 - 2.0 * s) / uw.x - 2.0, (3.0 + s) / uw.y, s / uw.z + 2.0);
    let vw = vec3(4.0 - 3.0 * t, 7.0, 1.0 + 3.0 * t);
    let v = vec3((3.0 - 2.0 * t) / vw.x - 2.0, (3.0 + t) / vw.y, t / vw.z + 2.0);

    var sum = 0.0;
    for (var j = 0u; j < 3u; j++) {
        for (var i = 0u; i < 3u; i++) {
            sum += uw[i] * vw[j] * sample_hardware(base_uv + vec2(u[i], v[j]) * inv, depth, layer);
        }
    }
    return sum * (1.0 / 144.0);
}

// Bevy `sample_directional_cascade`.
fn sample_cascade(cascade: u32, world: vec3<f32>, normal: vec3<f32>) -> f32 {
    let texel_size = frame.cascade_params[cascade].y;
    let offset = world
        + frame.shadow.y * texel_size * normal
        + frame.shadow.x * frame.direction_to_light.xyz;
    let clip = frame.cascade_clip_from_world[cascade] * vec4(offset, 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    if (any(ndc.xy < vec2(-1.0)) || ndc.z < 0.0 || any(ndc > vec3(1.0))) {
        return 1.0;
    }
    let uv = ndc.xy * vec2(0.5, -0.5) + vec2(0.5);
    if (frame.flags.w > 0.5) {
        return sample_gaussian(uv, ndc.z, cascade);
    }
    return sample_hardware(uv, ndc.z, cascade);
}

// Bevy `fetch_directional_shadow`: pick the cascade by view depth, blend into the next one
// across the overlap.
fn sun_shadow(world: vec3<f32>, normal: vec3<f32>) -> f32 {
    let depth = -(frame.view_from_world * vec4(world, 1.0)).z;
    let count = u32(frame.shadow.w);
    var cascade = count;
    for (var i = 0u; i < count; i++) {
        if (depth < frame.cascade_params[i].x) {
            cascade = i;
            break;
        }
    }
    if (cascade >= count) {
        return 1.0;
    }
    var shadow = sample_cascade(cascade, world, normal);
    let next = cascade + 1u;
    if (next < count) {
        let this_far = frame.cascade_params[cascade].x;
        let next_near = (1.0 - frame.shadow.z) * this_far;
        if (depth >= next_near) {
            let next_shadow = sample_cascade(next, world, normal);
            shadow = mix(shadow, next_shadow, (depth - next_near) / (this_far - next_near));
        }
    }
    return shadow;
}

@fragment
fn lit_fs(in: VertexOut) -> @location(0) vec4<f32> {
    let inst = instances[in.instance];
    let base = inst.base_color.rgb;
    let metallic = inst.pbr.x;
    let perceptual_roughness = inst.pbr.y;
    let reflectance = inst.pbr.z;
    let roughness = perceptual_roughness_to_roughness(perceptual_roughness);

    let N = normalize(in.normal);
    let V = normalize(frame.eye.xyz - in.world);
    let NdotV = max(dot(N, V), 0.0001);

    let diffuse_color = base * (1.0 - metallic);
    let f0_dielectric = vec3(0.16 * reflectance * reflectance);
    let F0 = mix(f0_dielectric, base, metallic);
    let f_ab = F_AB(perceptual_roughness, NdotV);

    var direct = vec3(0.0);
    if (frame.flags.z > 0.5) {
        let L = frame.direction_to_light.xyz;
        let H = normalize(L + V);
        let NdotL = saturate(dot(N, L));
        let NdotH = saturate(dot(N, H));
        let LdotH = saturate(dot(L, H));

        let f90 = 0.5 + 2.0 * roughness * LdotH * LdotH;
        let fd = F_Schlick(1.0, f90, NdotL) * F_Schlick(1.0, f90, NdotV) * (1.0 / PI);
        let diffuse = diffuse_color * fd;

        let D = D_GGX(roughness, NdotH);
        let Vis = V_SmithGGXCorrelated(roughness, NdotV, NdotL);
        let F = fresnel(F0, LdotH);
        var specular = (D * Vis) * F;
        specular *= 1.0 + F0 * (1.0 / (f_ab.x + f_ab.y) - 1.0);

        var shadow = 1.0;
        if (frame.flags.y > 0.5) {
            shadow = sun_shadow(in.world, N);
        }
        direct = (diffuse + specular) * NdotL * frame.sun_color.rgb * shadow;
    }

    let diffuse_ambient = env_brdf_approx(diffuse_color, F_AB(1.0, NdotV));
    let specular_ambient = env_brdf_approx(F0, f_ab);
    let specular_occlusion = saturate(dot(F0, vec3(50.0 * 0.33)));
    let indirect = (diffuse_ambient + specular_ambient * specular_occlusion) * frame.ambient_color.rgb;

    let color = frame.flags.x * (direct + indirect);
    return vec4(aces_fitted(color), 1.0);
}
