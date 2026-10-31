// Forward PBR pass and sun shadow-map pass.
//
// The BRDF follows Bevy 0.20 `bevy_pbr` (itself Filament): Burley diffuse, GGX distribution,
// height-correlated Smith visibility, Schlick Fresnel with the f90 specular-occlusion trick, the
// multiscatter energy compensation, the polynomial F_AB environment approximation for ambient
// light and Bevy's split-sum environment-map light (prefiltered diffuse + specular cube maps with
// multiscattering). Output is exposed linear HDR colour; `tonemap.wgsl` maps it for display.

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
    // x: environment-map intensity (cd/m²), y: smallest specular mip level, z: 1 when there is an
    // environment map.
    environment: vec4<f32>,
    // Distance fog, laid out like Bevy's `Fog`: colour, sun inscattering colour (a = strength),
    // be.xyz + directional-light exponent, bi.xyz + mode (0 off, 1 linear, 2 exponential,
    // 3 exponential squared, 4 atmospheric).
    fog_color: vec4<f32>,
    fog_directional_light_color: vec4<f32>,
    fog_be: vec4<f32>,
    fog_bi: vec4<f32>,
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
@group(1) @binding(3) var diffuse_environment_map: texture_cube<f32>;
@group(1) @binding(4) var specular_environment_map: texture_cube<f32>;
@group(1) @binding(5) var environment_map_sampler: sampler;
// Shadow pass only: the cascade being rendered (dynamic offset).
@group(1) @binding(2) var<uniform> cascade_clip_from_world: mat4x4<f32>;

struct VertexIn {
    @builtin(instance_index) instance: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
}

// ---------------------------------------------------------------- shadow pass

// Shadow pancaking, as in Bevy: casters between the sun and the cascade's near plane clamp onto
// it per fragment (reverse Z, orthographic, so w == 1). With `DEPTH_CLIP_CONTROL` the pipeline
// leaves depth unclipped and the rasterizer clamps; otherwise `shadow_emulated_*` clamp in the
// fragment shader. Clamping per vertex instead would bend the depth of large triangles that
// reach past the near plane (a ground plane stretching towards a low sun).
@vertex
fn shadow_vs(in: VertexIn) -> @builtin(position) vec4<f32> {
    let world = instances[in.instance].world_from_local * vec4(in.position, 1.0);
    return cascade_clip_from_world * world;
}

struct ShadowEmulatedOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) unclipped_depth: f32,
}

@vertex
fn shadow_emulated_vs(in: VertexIn) -> ShadowEmulatedOut {
    let world = instances[in.instance].world_from_local * vec4(in.position, 1.0);
    var out: ShadowEmulatedOut;
    out.clip = cascade_clip_from_world * world;
    out.unclipped_depth = out.clip.z;
    out.clip.z = min(out.clip.z, 1.0);
    return out;
}

@fragment
fn shadow_emulated_fs(in: ShadowEmulatedOut) -> @builtin(frag_depth) f32 {
    return min(in.unclipped_depth, 1.0);
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

// Bevy `radiance_sample_direction` ("Moving Frostbite to PBR 3.0", listing 22).
fn radiance_sample_direction(N: vec3<f32>, R: vec3<f32>, roughness: f32) -> vec3<f32> {
    let smoothness = saturate(1.0 - roughness);
    let lerp_factor = smoothness * (sqrt(smoothness) + roughness);
    return mix(N, R, lerp_factor);
}

struct Multiscatter {
    FssEss: vec3<f32>,
    FmsEms: vec3<f32>,
    Edss: vec3<f32>,
}

// Bevy `compute_multiscatter` (Fdez-Agüera 2019).
fn compute_multiscatter(F0: vec3<f32>, f_ab: vec2<f32>, Ems: f32, specular_occlusion: f32) -> Multiscatter {
    let FssEss = (F0 * f_ab.x + f_ab.y) * specular_occlusion;
    let Favg = F0 + (1.0 - F0) / 21.0;
    let FmsEms = FssEss * Favg / (1.0 - Ems * Favg) * Ems;
    return Multiscatter(FssEss, FmsEms, 1.0 - (FssEss + FmsEms));
}

struct EnvironmentLight {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

// Bevy `environment_map_light` for a view environment map (no light probes, no rotation).
fn environment_map_light(
    N: vec3<f32>,
    V: vec3<f32>,
    perceptual_roughness: f32,
    roughness: f32,
    diffuse_color: vec3<f32>,
    metallic: f32,
    F0_dielectric: vec3<f32>,
    F0_metallic: vec3<f32>,
    f_ab: vec2<f32>,
) -> EnvironmentLight {
    let intensity = frame.environment.x;
    let R = reflect(-V, N);
    let radiance_level = perceptual_roughness * frame.environment.y;
    // Cube maps are left-handed, so z is negated.
    var irradiance_dir = N;
    irradiance_dir.z = -irradiance_dir.z;
    let irradiance = textureSampleLevel(
        diffuse_environment_map, environment_map_sampler, irradiance_dir, 0.0).rgb * intensity;
    var radiance_dir = radiance_sample_direction(N, R, roughness);
    radiance_dir.z = -radiance_dir.z;
    let radiance = textureSampleLevel(
        specular_environment_map, environment_map_sampler, radiance_dir, radiance_level).rgb * intensity;

    var out: EnvironmentLight;
    if (all(irradiance == vec3(0.0)) && all(radiance == vec3(0.0))) {
        out.diffuse = vec3(0.0);
        out.specular = vec3(0.0);
        return out;
    }
    let F0_surface = mix(F0_dielectric, F0_metallic, metallic);
    let specular_occlusion = saturate(dot(F0_surface, vec3(50.0 * 0.33)));
    // Dielectric and metallic separately, then mixed: multiscattering is nonlinear in F0.
    let Ems = 1.0 - (f_ab.x + f_ab.y);
    let ms_dielectric = compute_multiscatter(F0_dielectric, f_ab, Ems, specular_occlusion);
    let ms_metallic = compute_multiscatter(F0_metallic, f_ab, Ems, specular_occlusion);
    let FssEss = mix(ms_dielectric.FssEss, ms_metallic.FssEss, metallic);
    let FmsEms = mix(ms_dielectric.FmsEms, ms_metallic.FmsEms, metallic);
    let kD = diffuse_color * ms_dielectric.Edss;
    out.diffuse = (FmsEms + kD) * irradiance;
    out.specular = FssEss * radiance;
    return out;
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
    // Bevy's order: environment diffuse, environment specular, then the uniform ambient light.
    var indirect = vec3(0.0);
    if (frame.environment.z > 0.5) {
        let env = environment_map_light(
            N, V, perceptual_roughness, roughness, diffuse_color, metallic, f0_dielectric, base, f_ab);
        indirect += env.diffuse;
        indirect += env.specular;
    }
    indirect += (diffuse_ambient + specular_ambient * specular_occlusion) * frame.ambient_color.rgb;

    let color = frame.flags.x * (direct + indirect);
    if (frame.fog_bi.w > 0.5) {
        return vec4(apply_fog(color, in.world), 1.0);
    }
    return vec4(color, 1.0);
}

// ---------------------------------------------------------------- distance fog

// Bevy `scattering_adjusted_fog_color`.
fn scattering_adjusted_fog_color(scattering: vec3<f32>) -> vec4<f32> {
    let glow = frame.fog_directional_light_color;
    if (glow.a > 0.0) {
        return vec4(frame.fog_color.rgb + scattering * glow.rgb * glow.a, frame.fog_color.a);
    }
    return frame.fog_color;
}

// Bevy `apply_fog` (one directional light) and the falloffs of `fog.wesl`, on the exposed colour.
fn apply_fog(input: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let view_to_world = world - frame.eye.xyz;
    let distance = length(view_to_world);

    var scattering = vec3(0.0);
    if (frame.fog_directional_light_color.a > 0.0 && frame.flags.z > 0.5) {
        let contribution = pow(
            max(dot(view_to_world / distance, frame.direction_to_light.xyz), 0.0),
            frame.fog_be.w,
        ) * frame.sun_color.rgb * frame.flags.x;
        var shadow = 1.0;
        if (frame.flags.y > 0.5) {
            // Bevy approximates the surface normal by the direction to the camera here.
            shadow = sun_shadow(world, normalize(-view_to_world));
        }
        scattering += contribution * shadow;
    }

    var fog_color = scattering_adjusted_fog_color(scattering);
    let mode = u32(frame.fog_bi.w + 0.5);
    if (mode == 1u) {
        let start = frame.fog_be.x;
        let end = frame.fog_be.y;
        fog_color.a *= 1.0 - clamp((end - distance) / (end - start), 0.0, 1.0);
        return mix(input, fog_color.rgb, fog_color.a);
    } else if (mode == 2u) {
        fog_color.a *= 1.0 - 1.0 / exp(distance * frame.fog_be.x);
        return mix(input, fog_color.rgb, fog_color.a);
    } else if (mode == 3u) {
        let distance_times_density = distance * frame.fog_be.x;
        fog_color.a *= 1.0 - 1.0 / exp(distance_times_density * distance_times_density);
        return mix(input, fog_color.rgb, fog_color.a);
    }
    let extinction_factor = 1.0 - 1.0 / exp(distance * frame.fog_be.xyz);
    let inscattering_factor = 1.0 - 1.0 / exp(distance * frame.fog_bi.xyz);
    return input * (1.0 - extinction_factor * fog_color.a)
        + fog_color.rgb * inscattering_factor * fog_color.a;
}
