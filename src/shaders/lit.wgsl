// Forward PBR pass and the shadow-map passes (sun cascades, spot maps, point-light cube faces).
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
    // x: point lights, y: spot lights (after the point lights in `lights`).
    light_counts: vec4<u32>,
    // xyz: unit direction towards the moon, w: 1 when there is a moon.
    moon_direction_to_light: vec4<f32>,
    // rgb: moon colour × illuminance (lux).
    moon_color: vec4<f32>,
    // The sky's atmosphere (Bevy `Atmosphere`): world to atmosphere space.
    world_to_atmosphere: mat4x4<f32>,
    // x: planet radius, y: atmosphere radius (m), z: 1 with a sky, w: 1 when the moon is
    // shaded before the sun (Bevy's directional-light order).
    atmosphere: vec4<f32>,
    // x: sun disc, y: moon disc angular size (radians).
    disks: vec4<f32>,
}

// A point or spot light, laid out like Bevy's `ClusteredLight`.
struct Light {
    // Point: the cube-face projection's lower-right 2×2 terms. Spot: direction x and z, spot
    // scale, spot offset.
    light_custom_data: vec4<f32>,
    // rgb: colour × intensity / 4π, w: 1 / range².
    color_inverse_square_range: vec4<f32>,
    position_radius: vec4<f32>,
    flags: u32,
    shadow_depth_bias: f32,
    shadow_normal_bias: f32,
    shadow_map_near_z: f32,
    spot_light_tan_angle: f32,
    soft_shadow_size: f32,
    // Point: cube index. Spot: layer of `shadow_maps`.
    shadow_index: u32,
    _pad: u32,
}

const LIGHT_SHADOWS: u32 = 1u;
const LIGHT_SPOT_Y_NEGATIVE: u32 = 2u;

struct Instance {
    world_from_local: mat4x4<f32>,
    normal_from_local: mat3x3<f32>,
    // x: 1 when the mesh has tangents, y: 1 when the model matrix mirrors (negative determinant),
    // z: the skin's first joint in `joint_matrices`.
    flags: vec4<u32>,
}

// Bevy `StandardMaterial` (the glTF core subset), see `world.rs`.
struct Material {
    base_color: vec4<f32>,
    // rgb: emissive radiance (cd/m²), a: how much exposure applies (1).
    emissive: vec4<f32>,
    metallic: f32,
    perceptual_roughness: f32,
    reflectance: f32,
    alpha_cutoff: f32,
    flags: u32,
}

const MATERIAL_BASE_COLOR_TEXTURE: u32 = 1u;
const MATERIAL_EMISSIVE_TEXTURE: u32 = 2u;
const MATERIAL_METALLIC_ROUGHNESS_TEXTURE: u32 = 4u;
const MATERIAL_OCCLUSION_TEXTURE: u32 = 8u;
const MATERIAL_NORMAL_MAP: u32 = 16u;
const MATERIAL_DOUBLE_SIDED: u32 = 32u;
const MATERIAL_UNLIT: u32 = 64u;
const MATERIAL_ALPHA_BLEND: u32 = 256u;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
// Per view, the indices of the instances that survived frustum culling; draws index into it.
@group(0) @binding(2) var<storage, read> visible: array<u32>;
// The joint palettes of the frame's skins; a skinned instance's run starts at `flags.z`.
@group(0) @binding(3) var<storage, read> joint_matrices: array<mat4x4<f32>>;
// Main pass only.
@group(1) @binding(0) var shadow_maps: texture_depth_2d_array;
@group(1) @binding(1) var shadow_sampler: sampler_comparison;
@group(1) @binding(3) var diffuse_environment_map: texture_cube<f32>;
@group(1) @binding(4) var specular_environment_map: texture_cube<f32>;
@group(1) @binding(5) var environment_map_sampler: sampler;
@group(1) @binding(6) var<storage, read> lights: array<Light>;
@group(1) @binding(7) var point_shadow_maps: texture_depth_cube_array;
// Non-comparison sampler for the soft-shadow blocker search.
@group(1) @binding(8) var shadow_linear_sampler: sampler;
// The sky's transmittance LUT (a 1² placeholder without a sky).
@group(1) @binding(9) var atmosphere_transmittance_texture: texture_2d<f32>;
@group(1) @binding(10) var atmosphere_transmittance_sampler: sampler;
// Shadow pass only: the cascade being rendered (dynamic offset).
@group(1) @binding(2) var<uniform> cascade_clip_from_world: mat4x4<f32>;
// Both passes: the draw's material. Empty texture slots hold a white texel and are not sampled.
@group(2) @binding(0) var<uniform> material: Material;
@group(2) @binding(1) var base_color_texture: texture_2d<f32>;
@group(2) @binding(2) var base_color_sampler: sampler;
@group(2) @binding(3) var emissive_texture: texture_2d<f32>;
@group(2) @binding(4) var emissive_sampler: sampler;
@group(2) @binding(5) var metallic_roughness_texture: texture_2d<f32>;
@group(2) @binding(6) var metallic_roughness_sampler: sampler;
@group(2) @binding(7) var occlusion_texture: texture_2d<f32>;
@group(2) @binding(8) var occlusion_sampler: sampler;
@group(2) @binding(9) var normal_map_texture: texture_2d<f32>;
@group(2) @binding(10) var normal_map_sampler: sampler;

struct VertexIn {
    @builtin(instance_index) instance: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) color: vec4<f32>,
}

// A skinned mesh's vertex: `VertexIn` plus `JOINTS_0` and `WEIGHTS_0`.
struct SkinnedVertexIn {
    @builtin(instance_index) instance: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
    @location(4) color: vec4<f32>,
    @location(5) joint_indices: vec4<u32>,
    @location(6) joint_weights: vec4<f32>,
}

fn unskinned(in: SkinnedVertexIn) -> VertexIn {
    return VertexIn(in.instance, in.position, in.normal, in.uv, in.tangent, in.color);
}

// Bevy `skin_model`. The palette is in the mesh's local space, so the instance's model matrix
// then places the skinned mesh (Bevy's joints are in world space and replace it).
fn skinned_world_from_local(index: u32, joints: vec4<u32>, weights: vec4<f32>) -> mat4x4<f32> {
    let base = instances[index].flags.z;
    let skin = weights.x * joint_matrices[base + joints.x]
        + weights.y * joint_matrices[base + joints.y]
        + weights.z * joint_matrices[base + joints.z]
        + weights.w * joint_matrices[base + joints.w];
    return instances[index].world_from_local * skin;
}

// Bevy `inverse_transpose_3x3m`.
fn inverse_transpose_3x3(m: mat3x3<f32>) -> mat3x3<f32> {
    let x = cross(m[1], m[2]);
    let y = cross(m[2], m[0]);
    let z = cross(m[0], m[1]);
    let det = dot(m[2], z);
    return mat3x3<f32>(x / det, y / det, z / det);
}

fn linear_part(m: mat4x4<f32>) -> mat3x3<f32> {
    return mat3x3(m[0].xyz, m[1].xyz, m[2].xyz);
}

// The material's base colour times its texture (Bevy samples it the same way in the main pass
// and, for alpha-masked casters, the shadow pass).
fn textured_base_color(uv: vec2<f32>) -> vec4<f32> {
    var color = material.base_color;
    if ((material.flags & MATERIAL_BASE_COLOR_TEXTURE) != 0u) {
        color *= textureSample(base_color_texture, base_color_sampler, uv);
    }
    return color;
}

// ---------------------------------------------------------------- shadow pass

// Shadow pancaking, as in Bevy: casters between the sun and the cascade's near plane clamp onto
// it per fragment (reverse Z, orthographic, so w == 1). With `DEPTH_CLIP_CONTROL` the pipeline
// leaves depth unclipped and the rasterizer clamps; otherwise `shadow_emulated_*` clamp in the
// fragment shader. Clamping per vertex instead would bend the depth of large triangles that
// reach past the near plane (a ground plane stretching towards a low sun).
//
// Alpha-masked and blended casters cut their shadows out by the base colour's alpha
// (`shadow_masked_fs`, Bevy `prepass_alpha_discard`).
struct ShadowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) unclipped_depth: f32,
}

fn shadow_vertex(world_from_local: mat4x4<f32>, position: vec3<f32>, uv: vec2<f32>) -> ShadowOut {
    let world = world_from_local * vec4(position, 1.0);
    var out: ShadowOut;
    out.clip = cascade_clip_from_world * world;
    out.uv = uv;
    out.unclipped_depth = out.clip.z;
    return out;
}

fn shadow_unskinned(in: VertexIn) -> ShadowOut {
    return shadow_vertex(instances[visible[in.instance]].world_from_local, in.position, in.uv);
}

fn shadow_skinned(in: SkinnedVertexIn) -> ShadowOut {
    let model = skinned_world_from_local(visible[in.instance], in.joint_indices, in.joint_weights);
    return shadow_vertex(model, in.position, in.uv);
}

@vertex
fn shadow_vs(in: VertexIn) -> ShadowOut {
    return shadow_unskinned(in);
}

@vertex
fn shadow_emulated_vs(in: VertexIn) -> ShadowOut {
    var out = shadow_unskinned(in);
    out.clip.z = min(out.clip.z, 1.0);
    return out;
}

@vertex
fn shadow_skinned_vs(in: SkinnedVertexIn) -> ShadowOut {
    return shadow_skinned(in);
}

@vertex
fn shadow_emulated_skinned_vs(in: SkinnedVertexIn) -> ShadowOut {
    var out = shadow_skinned(in);
    out.clip.z = min(out.clip.z, 1.0);
    return out;
}

fn shadow_alpha_discard(uv: vec2<f32>) {
    if (textured_base_color(uv).a < material.alpha_cutoff) {
        discard;
    }
}

@fragment
fn shadow_masked_fs(in: ShadowOut) {
    shadow_alpha_discard(in.uv);
}

@fragment
fn shadow_emulated_fs(in: ShadowOut) -> @builtin(frag_depth) f32 {
    return min(in.unclipped_depth, 1.0);
}

@fragment
fn shadow_emulated_masked_fs(in: ShadowOut) -> @builtin(frag_depth) f32 {
    shadow_alpha_discard(in.uv);
    return min(in.unclipped_depth, 1.0);
}

// ---------------------------------------------------------------- main pass

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) instance: u32,
    @location(3) uv: vec2<f32>,
    @location(4) tangent: vec4<f32>,
    @location(5) color: vec4<f32>,
}

// Bevy `mesh_normal_local_to_world` and `mesh_tangent_local_to_world`: normalised per vertex
// (MikkTSpace), not again per fragment.
fn lit_vertex(
    in: VertexIn,
    index: u32,
    world_from_local: mat4x4<f32>,
    world_normal: vec3<f32>,
) -> VertexOut {
    let world = world_from_local * vec4(in.position, 1.0);
    var out: VertexOut;
    out.clip = frame.clip_from_world * world;
    out.world = world.xyz;
    out.normal = world_normal;
    out.tangent = in.tangent;
    if (any(in.tangent != vec4(0.0))) {
        let sign = select(1.0, -1.0, instances[index].flags.y != 0u);
        out.tangent = vec4(
            normalize(linear_part(world_from_local) * in.tangent.xyz),
            in.tangent.w * sign,
        );
    }
    out.instance = index;
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@vertex
fn lit_vs(in: VertexIn) -> VertexOut {
    let index = visible[in.instance];
    let inst = instances[index];
    var normal = in.normal;
    if (any(in.normal != vec3(0.0))) {
        normal = normalize(inst.normal_from_local * in.normal);
    }
    return lit_vertex(in, index, inst.world_from_local, normal);
}

// Bevy's skinned path: the skinned model matrix replaces the instance's for positions and
// tangents, and normals go through its inverse transpose (`skin_normals`).
@vertex
fn lit_skinned_vs(in: SkinnedVertexIn) -> VertexOut {
    let index = visible[in.instance];
    let model = skinned_world_from_local(index, in.joint_indices, in.joint_weights);
    let normal = normalize(inverse_transpose_3x3(linear_part(model)) * in.normal);
    return lit_vertex(unskinned(in), index, model, normal);
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

// ---------------------------------------------------------------- point and spot lights

// Bevy's `shadow_sampling.wesl` constants ("determined by trial and error").
const SPOT_SHADOW_TEXEL_SIZE: f32 = 0.0134277345;
const POINT_SHADOW_SCALE: f32 = 0.003;
const POINT_SHADOW_TEMPORAL_OFFSET_SCALE: f32 = 0.5;

// The D3D 8× MSAA sample positions and a unit Gaussian's weights at them.
const D3D_SAMPLE_POINT_POSITIONS: array<vec2<f32>, 8> = array(
    vec2( 0.125, -0.375),
    vec2(-0.125,  0.375),
    vec2( 0.625,  0.125),
    vec2(-0.375, -0.625),
    vec2(-0.625,  0.625),
    vec2(-0.875, -0.125),
    vec2( 0.375,  0.875),
    vec2( 0.875, -0.875),
);
const D3D_SAMPLE_POINT_COEFFS: array<f32, 8> = array(
    0.157112, 0.157112, 0.138651, 0.130251, 0.114946, 0.114946, 0.107982, 0.079001,
);
// Jimenez 2014 spiral (Bevy `utils::SPIRAL_OFFSET_*`).
const SPIRAL_OFFSETS: array<vec2<f32>, 8> = array(
    vec2(-0.7071,  0.7071),
    vec2(-0.0000, -0.8750),
    vec2( 0.5303,  0.5303),
    vec2(-0.6250, -0.0000),
    vec2( 0.3536, -0.3536),
    vec2(-0.0000,  0.3750),
    vec2(-0.1768, -0.1768),
    vec2( 0.1250,  0.0000),
);

// Bevy `maths::orthonormalize`.
fn orthonormalize(z_basis: vec3<f32>) -> mat3x3<f32> {
    let sign = select(-1.0, 1.0, z_basis.z >= 0.0);
    let a = -1.0 / (sign + z_basis.z);
    let b = z_basis.x * z_basis.y * a;
    let x_basis = vec3(1.0 + sign * z_basis.x * z_basis.x * a, sign * b, -sign * z_basis.x);
    let y_basis = vec3(b, sign + z_basis.y * z_basis.y * a, -z_basis.y);
    return mat3x3(x_basis, y_basis, z_basis);
}

// Bevy `interleaved_gradient_noise`.
fn interleaved_gradient_noise(pixel_coordinates: vec2<f32>, frame_index: u32) -> f32 {
    let xy = pixel_coordinates + 5.588238 * f32(frame_index % 64u);
    return fract(52.9829189 * fract(0.06711056 * xy.x + 0.00583715 * xy.y));
}

// Bevy `random_rotation_matrix`, non-temporal (frame 1).
fn random_rotation_matrix(scale: vec2<f32>) -> mat2x2<f32> {
    let random_angle = 2.0 * PI * interleaved_gradient_noise(scale, 1u);
    let m = vec2(sin(random_angle), cos(random_angle));
    return mat2x2(m.y, -m.x, m.x, m.y);
}

// Bevy `getDistanceAttenuation`: inverse square with a smooth window at the light's range.
fn get_distance_attenuation(distance_square: f32, inverse_range_squared: f32) -> f32 {
    let factor = distance_square * inverse_range_squared;
    let smooth_factor = saturate(1.0 - factor * factor);
    return smooth_factor * smooth_factor * 1.0 / max(distance_square, 0.0001);
}

fn specular_fix_remap(a: f32) -> f32 {
    let inv_a_sq = (1.0 - a) * (1.0 - a);
    return 1.0 - inv_a_sq * inv_a_sq;
}

struct Surface {
    P: vec3<f32>,
    N: vec3<f32>,
    V: vec3<f32>,
    R: vec3<f32>,
    NdotV: f32,
    roughness: f32,
    diffuse_color: vec3<f32>,
    F0: vec3<f32>,
    f_ab: vec2<f32>,
}

// Bevy `point_light` (no clearcoat, anisotropy or light textures): Karis's representative-point
// sphere light for specular, Burley diffuse towards the centre.
fn point_light(light: Light, s: Surface) -> vec3<f32> {
    let light_to_frag = light.position_radius.xyz - s.P;
    let L = normalize(light_to_frag);
    let distance_square = dot(light_to_frag, light_to_frag);
    let distance = sqrt(distance_square);
    let range_attenuation =
        get_distance_attenuation(distance_square, light.color_inverse_square_range.w);
    let radius = light.position_radius.w;
    let a = s.roughness;

    // `compute_specular_layer_values_for_point_light`.
    let LtFdotR = max(0.0001, dot(light_to_frag, s.R));
    let center_to_ray = LtFdotR * s.R - light_to_frag;
    let closest_point = light_to_frag
        + center_to_ray * saturate(radius * inverseSqrt(dot(center_to_ray, center_to_ray)));
    let L_spec = closest_point * inverseSqrt(dot(closest_point, closest_point));
    let a_prime = saturate(a + radius / (2.0 * distance));

    let H_spec = normalize(L_spec + s.V);
    let NdotL_spec = saturate(dot(s.N, L_spec));
    let NdotH_spec = saturate(dot(s.N, H_spec));
    let LdotH_spec = saturate(dot(L_spec, H_spec));
    let normalization_factor = a / a_prime;
    let specular_intensity = normalization_factor * normalization_factor;
    let brdf_roughness = mix(a, a_prime, specular_fix_remap(a));

    let D = D_GGX(brdf_roughness, NdotH_spec);
    let Vis = V_SmithGGXCorrelated(brdf_roughness, s.NdotV, NdotL_spec);
    let F = fresnel(s.F0, LdotH_spec);
    var specular_light = (specular_intensity * D * Vis) * F;
    specular_light *= 1.0 + s.F0 * (1.0 / (s.f_ab.x + s.f_ab.y) - 1.0);
    // Sphere-light visibility (solid-angle attenuation).
    if (radius > 0.0) {
        let solid_angle = radius * radius / (distance * distance);
        specular_light *= saturate(NdotL_spec / max(NdotL_spec + solid_angle, 1e-4));
    }

    let H = normalize(L + s.V);
    let NdotL = saturate(dot(s.N, L));
    let LdotH = saturate(dot(L, H));
    let f90 = 0.5 + 2.0 * s.roughness * LdotH * LdotH;
    let fd = F_Schlick(1.0, f90, NdotL) * F_Schlick(1.0, f90, s.NdotV) * (1.0 / PI);
    let diffuse = s.diffuse_color * fd;

    let color_times_NdotL = diffuse * NdotL + specular_light * NdotL_spec;
    return color_times_NdotL * light.color_inverse_square_range.rgb * range_attenuation;
}

// The spot axis, rebuilt from x, z and the sign of y like Bevy.
fn spot_direction(light: Light) -> vec3<f32> {
    var spot_dir = vec3(light.light_custom_data.x, 0.0, light.light_custom_data.y);
    spot_dir.y = sqrt(max(0.0, 1.0 - spot_dir.x * spot_dir.x - spot_dir.z * spot_dir.z));
    if ((light.flags & LIGHT_SPOT_Y_NEGATIVE) != 0u) {
        spot_dir.y = -spot_dir.y;
    }
    return spot_dir;
}

// Bevy `spot_light`: the point light times Filament's smooth cone falloff.
fn spot_light(light: Light, s: Surface) -> vec3<f32> {
    let point = point_light(light, s);
    let light_to_frag = light.position_radius.xyz - s.P;
    let cd = dot(-spot_direction(light), normalize(light_to_frag));
    let attenuation = saturate(cd * light.light_custom_data.z + light.light_custom_data.w);
    return point * (attenuation * attenuation);
}

fn sample_cube_hardware(light_local: vec3<f32>, depth: f32, cube: u32) -> f32 {
    return textureSampleCompareLevel(point_shadow_maps, shadow_sampler, light_local, cube, depth);
}

// Bevy `sample_shadow_cubemap_gaussian`: a Gaussian over the D3D pattern on the plane facing the
// light.
fn sample_cube_gaussian(light_local: vec3<f32>, depth: f32, scale: f32, distance_to_light: f32, cube: u32) -> f32 {
    let basis = orthonormalize(normalize(light_local)) * scale * distance_to_light;
    var sum = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let p = D3D_SAMPLE_POINT_POSITIONS[i];
        sum += sample_cube_hardware(light_local + p.x * basis[0] + p.y * basis[1], depth, cube)
            * D3D_SAMPLE_POINT_COEFFS[i];
    }
    return sum;
}

// Bevy `sample_shadow_cubemap_jittered` (non-temporal): the spiral, rotated per pixel.
fn sample_cube_jittered(light_local: vec3<f32>, depth: f32, frag_coord_xy: vec2<f32>, scale: f32, distance_to_light: f32, cube: u32) -> f32 {
    let rotation = random_rotation_matrix(frag_coord_xy);
    let basis = orthonormalize(normalize(light_local)) * scale * distance_to_light;
    var sum = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let p = rotation * SPIRAL_OFFSETS[i] * POINT_SHADOW_TEMPORAL_OFFSET_SCALE;
        sum += sample_cube_hardware(light_local + p.x * basis[0] + p.y * basis[1], depth, cube)
            * 0.125;
    }
    return sum;
}

// Bevy `search_for_blockers_in_shadow_cubemap`: mean depth of the occluding texels.
fn search_cube_blockers(light_local: vec3<f32>, depth: f32, scale: f32, distance_to_light: f32, cube: u32) -> f32 {
    let basis = orthonormalize(normalize(light_local)) * scale * distance_to_light;
    var sum = vec2(0.0);
    for (var i = 0u; i < 8u; i++) {
        let p = D3D_SAMPLE_POINT_POSITIONS[i];
        let sampled_depth = textureSampleLevel(
            point_shadow_maps, shadow_linear_sampler, light_local + p.x * basis[0] + p.y * basis[1], cube, 0);
        sum += select(vec2(0.0), vec2(sampled_depth, 1.0), sampled_depth >= depth);
    }
    if (sum.y == 0.0) {
        return 0.0;
    }
    return sum.x / sum.y;
}

// Bevy `fetch_point_shadow`.
fn fetch_point_shadow(light: Light, P: vec3<f32>, N: vec3<f32>, frag_coord_xy: vec2<f32>) -> f32 {
    let surface_to_light = light.position_radius.xyz - P;
    let surface_to_light_abs = abs(surface_to_light);
    let distance_to_light = max(surface_to_light_abs.x, max(surface_to_light_abs.y, surface_to_light_abs.z));

    // The normal bias is per texel at 1 m; texels grow with the distance to the light.
    let normal_offset = light.shadow_normal_bias * distance_to_light * N;
    let depth_offset = light.shadow_depth_bias * normalize(surface_to_light);
    let offset_position = P + normal_offset + depth_offset;

    let frag_ls = offset_position - light.position_radius.xyz;
    let abs_position_ls = abs(frag_ls);
    let major_axis_magnitude = max(abs_position_ls.x, max(abs_position_ls.y, abs_position_ls.z));
    let zw = -major_axis_magnitude * light.light_custom_data.xy + light.light_custom_data.zw;
    let depth = zw.x / zw.y;

    // Cube maps are left-handed: z is flipped for the lookup.
    let light_local = frag_ls * vec3(1.0, 1.0, -1.0);
    let cube = light.shadow_index;
    if (light.soft_shadow_size > 0.0) {
        let light_size = light.soft_shadow_size;
        let z_blocker = search_cube_blockers(light_local, depth, light_size, distance_to_light, cube);
        let blur_size = max((z_blocker - depth) * light_size / depth, 0.5);
        return sample_cube_jittered(
            light_local, depth, frag_coord_xy, POINT_SHADOW_SCALE * blur_size, distance_to_light, cube);
    }
    if (frame.flags.w > 0.5) {
        return sample_cube_gaussian(light_local, depth, POINT_SHADOW_SCALE, distance_to_light, cube);
    }
    return sample_cube_hardware(light_local, depth, cube);
}

// Bevy `sample_shadow_map_jimenez_fourteen` (non-temporal).
fn sample_jimenez(light_local: vec2<f32>, depth: f32, layer: u32, frag_coord_xy: vec2<f32>, texel_size: f32, blur_size: f32) -> f32 {
    let rotation = random_rotation_matrix(frag_coord_xy);
    let size = vec2<f32>(textureDimensions(shadow_maps));
    // Bevy's fudge factor (`map(0.00390625, 0.022949219, 0.015, 0.035, texel_size)`).
    let f = 0.015 + (texel_size - 0.00390625) * (0.035 - 0.015) / (0.022949219 - 0.00390625);
    let uv_offset_scale = f * blur_size / (texel_size * size);
    var sum = 0.0;
    for (var i = 0u; i < 8u; i++) {
        sum += sample_hardware(light_local + (rotation * SPIRAL_OFFSETS[i]) * uv_offset_scale, depth, layer);
    }
    return sum / 8.0;
}

// Bevy `search_for_blockers_in_shadow_map`.
fn search_blockers(light_local: vec2<f32>, depth: f32, layer: u32, texel_size: f32, search_size: f32) -> f32 {
    let size = vec2<f32>(textureDimensions(shadow_maps));
    let uv_offset_scale = search_size / (texel_size * size);
    var sum = vec2(0.0);
    for (var i = 0u; i < 8u; i++) {
        let sampled_depth = textureSampleLevel(
            shadow_maps, shadow_linear_sampler, light_local + D3D_SAMPLE_POINT_POSITIONS[i] * uv_offset_scale, layer, 0);
        sum += select(vec2(0.0), vec2(sampled_depth, 1.0), sampled_depth >= depth);
    }
    if (sum.y == 0.0) {
        return 0.0;
    }
    return sum.x / sum.y;
}

// Bevy `fetch_spot_shadow`.
fn fetch_spot_shadow(light: Light, P: vec3<f32>, N: vec3<f32>, frag_coord_xy: vec2<f32>) -> f32 {
    let surface_to_light = light.position_radius.xyz - P;
    // The view's z axis is the reverse of the spot direction.
    let fwd = -spot_direction(light);
    let distance_to_light = dot(fwd, surface_to_light);
    let offset_position = -surface_to_light
        + (light.shadow_depth_bias * normalize(surface_to_light))
        + (N * light.shadow_normal_bias) * distance_to_light;
    // A rotation: multiplying from the left by the vector applies its transpose (inverse).
    let projected_position = offset_position * orthonormalize(fwd);
    let f_div_minus_z = 1.0 / (light.spot_light_tan_angle * -projected_position.z);
    let shadow_uv = projected_position.xy * f_div_minus_z * vec2(0.5, -0.5) + vec2(0.5, 0.5);
    let depth = light.shadow_map_near_z / -projected_position.z;

    let layer = light.shadow_index;
    if (light.soft_shadow_size > 0.0) {
        let light_size = light.soft_shadow_size;
        let z_blocker = search_blockers(shadow_uv, depth, layer, SPOT_SHADOW_TEXEL_SIZE, light_size);
        let blur_size = max((z_blocker - depth) * light_size / depth, 0.5);
        return sample_jimenez(shadow_uv, depth, layer, frag_coord_xy, SPOT_SHADOW_TEXEL_SIZE, blur_size);
    }
    if (frame.flags.w > 0.5) {
        return sample_gaussian(shadow_uv, depth, layer);
    }
    return sample_hardware(shadow_uv, depth, layer);
}

// ---------------------------------------------------------------- image-based light

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

// Every material input of a fragment (Bevy `pbr_input_from_standard_material`). All texture
// sampling happens here, before any alpha discard, in uniform control flow.
struct MaterialSample {
    base_color: vec4<f32>,
    emissive: vec3<f32>,
    metallic: f32,
    perceptual_roughness: f32,
    diffuse_occlusion: f32,
    // Tangent-space normal in [-1, 1], when the material has a normal map.
    Nt: vec3<f32>,
}

fn sample_material(in: VertexOut) -> MaterialSample {
    var m: MaterialSample;
    // Bevy: vertex colour, times the material colour, times the texture.
    m.base_color = in.color * material.base_color;
    if ((material.flags & MATERIAL_BASE_COLOR_TEXTURE) != 0u) {
        m.base_color *= textureSample(base_color_texture, base_color_sampler, in.uv);
    }
    m.emissive = material.emissive.rgb;
    if ((material.flags & MATERIAL_EMISSIVE_TEXTURE) != 0u) {
        m.emissive *= textureSample(emissive_texture, emissive_sampler, in.uv).rgb;
    }
    m.metallic = material.metallic;
    m.perceptual_roughness = material.perceptual_roughness;
    if ((material.flags & MATERIAL_METALLIC_ROUGHNESS_TEXTURE) != 0u) {
        let mr = textureSample(metallic_roughness_texture, metallic_roughness_sampler, in.uv);
        m.perceptual_roughness *= mr.g;
        m.metallic *= mr.b;
    }
    m.diffuse_occlusion = 1.0;
    if ((material.flags & MATERIAL_OCCLUSION_TEXTURE) != 0u) {
        m.diffuse_occlusion = textureSample(occlusion_texture, occlusion_sampler, in.uv).r;
    }
    if ((material.flags & MATERIAL_NORMAL_MAP) != 0u) {
        m.Nt = textureSample(normal_map_texture, normal_map_sampler, in.uv).rgb * 2.0 - 1.0;
    }
    return m;
}

// Opaque and blended materials.
@fragment
fn lit_fs(in: VertexOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    var m = sample_material(in);
    if ((material.flags & MATERIAL_ALPHA_BLEND) == 0u) {
        m.base_color.a = 1.0;
    }
    return shade(in, is_front, m);
}

// Alpha-masked materials (Bevy `alpha_discard`).
@fragment
fn lit_masked_fs(in: VertexOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    var m = sample_material(in);
    if (m.base_color.a < material.alpha_cutoff) {
        discard;
    }
    m.base_color.a = 1.0;
    return shade(in, is_front, m);
}

// Bevy `pbr_fragment` + `apply_pbr_lighting` + `main_pass_post_lighting_processing`. Front faces
// come from the pipeline, which winds mirrored instances clockwise (Bevy's
// `winding_corrected_front_facing`).
fn shade(in: VertexOut, is_front: bool, m: MaterialSample) -> vec4<f32> {
    let double_sided = (material.flags & MATERIAL_DOUBLE_SIDED) != 0u;
    let has_tangents = instances[in.instance].flags.x != 0u;
    let normal_mapped = has_tangents && (material.flags & MATERIAL_NORMAL_MAP) != 0u;

    // Bevy `prepare_world_normal`: back faces of double-sided materials flip the normal, unless
    // the mesh has tangents or the material a normal map (then the normal map flips it).
    var world_normal = in.normal;
    if (double_sided && !is_front && !has_tangents && (material.flags & MATERIAL_NORMAL_MAP) == 0u) {
        world_normal = -world_normal;
    }

    var color: vec4<f32>;
    if ((material.flags & MATERIAL_UNLIT) != 0u) {
        color = m.base_color;
    } else {
        var N = normalize(world_normal);
        if (normal_mapped) {
            // Bevy `calculate_tbn_mikktspace` + `apply_normal_mapping`: no renormalisation or
            // Gram-Schmidt before the map is applied.
            let T = in.tangent.xyz;
            let B = in.tangent.w * cross(world_normal, T);
            var Nt = m.Nt;
            if (double_sided && !is_front) {
                Nt = -Nt;
            }
            N = normalize(Nt.x * T + Nt.y * B + Nt.z * world_normal);
        }
        color = vec4(lit_radiance(in, world_normal, N, m), m.base_color.a);
    }
    if (frame.fog_bi.w > 0.5) {
        return vec4(apply_fog(color.rgb, in.world), color.a);
    }
    return color;
}

// Exposed radiance of a lit fragment with shading normal `N`; shadows are looked up with the
// interpolated vertex normal, as in Bevy.
fn lit_radiance(in: VertexOut, world_normal: vec3<f32>, N: vec3<f32>, m: MaterialSample) -> vec3<f32> {
    let base = m.base_color.rgb;
    let metallic = m.metallic;
    let perceptual_roughness = m.perceptual_roughness;
    let reflectance = material.reflectance;
    let roughness = perceptual_roughness_to_roughness(perceptual_roughness);

    let V = normalize(frame.eye.xyz - in.world);
    let NdotV = max(dot(N, V), 0.0001);

    let diffuse_color = base * (1.0 - metallic);
    let f0_dielectric = vec3(0.16 * reflectance * reflectance);
    let F0 = mix(f0_dielectric, base, metallic);
    let f_ab = F_AB(perceptual_roughness, NdotV);

    // Bevy's order: point lights, spot lights, then the sun.
    var direct = vec3(0.0);
    var surface: Surface;
    surface.P = in.world;
    surface.N = N;
    surface.V = V;
    surface.R = reflect(-V, N);
    surface.NdotV = NdotV;
    surface.roughness = roughness;
    surface.diffuse_color = diffuse_color;
    surface.F0 = F0;
    surface.f_ab = f_ab;
    let point_count = frame.light_counts.x;
    for (var i = 0u; i < point_count; i++) {
        let light = lights[i];
        var shadow = 1.0;
        if ((light.flags & LIGHT_SHADOWS) != 0u) {
            shadow = fetch_point_shadow(light, in.world, world_normal, in.clip.xy);
        }
        direct += point_light(light, surface) * shadow;
    }
    for (var i = point_count; i < point_count + frame.light_counts.y; i++) {
        let light = lights[i];
        var shadow = 1.0;
        if ((light.flags & LIGHT_SHADOWS) != 0u) {
            shadow = fetch_spot_shadow(light, in.world, world_normal, in.clip.xy);
        }
        direct += spot_light(light, surface) * shadow;
    }
    // Directional lights in Bevy's order: unshadowed first, then by spawn order (sun, moon).
    let has_moon = frame.moon_direction_to_light.w > 0.5;
    let moon_first = frame.atmosphere.w > 0.5;
    if (has_moon && moon_first) {
        direct += directional_light(
            frame.moon_direction_to_light.xyz, frame.moon_color.rgb, frame.disks.y, surface);
    }
    if (frame.flags.z > 0.5) {
        var shadow = 1.0;
        if (frame.flags.y > 0.5) {
            shadow = sun_shadow(in.world, world_normal);
        }
        direct += directional_light(
            frame.direction_to_light.xyz, frame.sun_color.rgb, frame.disks.x, surface) * shadow;
    }
    if (has_moon && !moon_first) {
        direct += directional_light(
            frame.moon_direction_to_light.xyz, frame.moon_color.rgb, frame.disks.y, surface);
    }

    let diffuse_ambient = env_brdf_approx(diffuse_color, F_AB(1.0, NdotV));
    let specular_ambient = env_brdf_approx(F0, f_ab);
    let specular_occlusion = saturate(dot(F0, vec3(50.0 * 0.33)));
    // Bevy's order: environment diffuse, environment specular, then the uniform ambient light.
    // The occlusion texture darkens the diffuse environment light and all of the ambient light.
    var indirect = vec3(0.0);
    if (frame.environment.z > 0.5) {
        let env = environment_map_light(
            N, V, perceptual_roughness, roughness, diffuse_color, metallic, f0_dielectric, base, f_ab);
        indirect += env.diffuse * m.diffuse_occlusion;
        indirect += env.specular;
    }
    indirect += (diffuse_ambient + specular_ambient * specular_occlusion) * frame.ambient_color.rgb
        * m.diffuse_occlusion;

    // Emissive light is scaled by alpha and exposed like the rest (Bevy, `emissive.a` = 1).
    let emissive = m.emissive * m.base_color.a * mix(1.0, frame.flags.x, material.emissive.a);
    return frame.flags.x * (direct + indirect) + emissive;
}

// ---------------------------------------------------------------- directional lights

// Bevy `directional_light` (without the shadow): Burley diffuse and GGX specular with the
// multiscatter compensation, filtered by the sky's transmittance and the share of the disc
// above the horizon when there is a sky.
fn directional_light(L: vec3<f32>, light_color: vec3<f32>, disk_size: f32, s: Surface) -> vec3<f32> {
    let H = normalize(L + s.V);
    let NdotL = saturate(dot(s.N, L));
    let NdotH = saturate(dot(s.N, H));
    let LdotH = saturate(dot(L, H));

    let f90 = 0.5 + 2.0 * s.roughness * LdotH * LdotH;
    let fd = F_Schlick(1.0, f90, NdotL) * F_Schlick(1.0, f90, s.NdotV) * (1.0 / PI);
    let diffuse = s.diffuse_color * fd;

    let D = D_GGX(s.roughness, NdotH);
    let Vis = V_SmithGGXCorrelated(s.roughness, s.NdotV, NdotL);
    let F = fresnel(s.F0, LdotH);
    var specular = (D * Vis) * F;
    specular *= 1.0 + s.F0 * (1.0 / (s.f_ab.x + s.f_ab.y) - 1.0);

    var color = (diffuse + specular) * NdotL;
    color *= light_color;
    if (frame.atmosphere.z > 0.5) {
        let P_as = (frame.world_to_atmosphere * vec4(s.P, 1.0)).xyz;
        let P_clamped = atmosphere_clamp_to_surface(P_as);
        let r = length(P_clamped);
        let local_up = normalize(P_clamped);
        let mu_light = dot(L, local_up);
        let transmittance = sample_atmosphere_transmittance(r, mu_light);
        let sun_visibility = calculate_visible_sun_ratio(r, mu_light, disk_size);
        color *= transmittance * sun_visibility;
    }
    return color;
}

// Bevy atmosphere `clamp_to_surface`: at least 1 m above the planet.
fn atmosphere_clamp_to_surface(position: vec3<f32>) -> vec3<f32> {
    let min_radius = frame.atmosphere.x + 1.0;
    let r = length(position);
    if r < min_radius {
        let up = normalize(position);
        return up * min_radius;
    }
    return position;
}

// Bevy `fast_acos_4`.
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

// Bevy `calculate_visible_sun_ratio`: the share of the light's disc above the horizon.
fn calculate_visible_sun_ratio(r: f32, mu: f32, sun_angular_size: f32) -> f32 {
    let inner_radius = frame.atmosphere.x;
    let horizon_cos = -sqrt(1.0 - (inner_radius * inner_radius) / (r * r));
    let horizon_angle = fast_acos_4(horizon_cos);
    let sun_zenith_angle = fast_acos_4(mu);
    if sun_zenith_angle + sun_angular_size * 0.5 <= horizon_angle {
        return 1.0;
    }
    if sun_zenith_angle - sun_angular_size * 0.5 >= horizon_angle {
        return 0.0;
    }
    let d = (horizon_angle - sun_zenith_angle) / (sun_angular_size * 0.5);
    let visible_ratio = 0.5 + d * 0.5;
    return clamp(visible_ratio, 0.0, 1.0);
}

// Bevy `sample_transmittance_lut` with `transmittance_lut_r_mu_to_uv` (Bruneton).
fn sample_atmosphere_transmittance(r: f32, mu: f32) -> vec3<f32> {
    let inner = frame.atmosphere.x;
    let outer = frame.atmosphere.y;
    let H = sqrt(outer * outer - inner * inner);
    let rho = sqrt(max(r * r - inner * inner, 0.0));
    let positive_discriminant = max(r * r * (mu * mu - 1.0) + outer * outer, 0.0);
    let d = max(-r * mu + sqrt(positive_discriminant), 0.0);
    let d_min = outer - r;
    let d_max = rho + H;
    let uv = vec2((d - d_min) / (d_max - d_min), rho / H);
    return textureSampleLevel(
        atmosphere_transmittance_texture, atmosphere_transmittance_sampler, uv, 0.0).rgb;
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
