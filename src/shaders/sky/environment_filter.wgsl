// Runtime environment-map filtering, ported from Bevy 0.20 `GeneratedEnvironmentMapLight`
// (`bevy_pbr/src/light_probe/environment_filter.wesl` and the GGX helpers of
// `pbr_lighting.wesl` / `utils.wesl`), MIT or Apache-2.0. The oracle's Bevy build has no
// `bluenoise_texture` feature, so the radiance filter uses Bevy's hash-based noise.
//
// `downsample` builds the source cube's mip chain like Bevy's single-pass downsampling (a 2×2
// box filter per level after the first), one dispatch per level, `generate_radiance_map` the GGX-prefiltered specular cube, one
// dispatch per mip, and `generate_irradiance_map` the 32² diffuse cube.

const PI: f32 = 3.141592653589793;
const PI_2: f32 = 6.283185307179586;

struct FilteringConstants {
    mip_level: f32,
    sample_count: u32,
    roughness: f32,
    noise_size_bits: vec2u,
}

@group(0) @binding(0) var input_texture: texture_2d_array<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var output_texture: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(3) var<uniform> constants: FilteringConstants;

// ---------------------------------------------------------------- downsampling

@compute
@workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) global_id: vec3u) {
    let size = textureDimensions(output_texture).xy;
    if (any(global_id.xy >= size)) {
        return;
    }
    let p = global_id.xy * 2u;
    let face = global_id.z;
    // Bevy's SPD port samples the first level bilinearly at the centre of the block's first
    // texel (`(uv + 0.5) * inverse_input_size`), which returns that texel alone.
    if (constants.mip_level < 1.5) {
        textureStore(output_texture, global_id.xy, face, textureLoad(input_texture, p, face, 0));
        return;
    }
    let v0 = textureLoad(input_texture, p, face, 0);
    let v1 = textureLoad(input_texture, p + vec2u(1u, 0u), face, 0);
    let v2 = textureLoad(input_texture, p + vec2u(0u, 1u), face, 0);
    let v3 = textureLoad(input_texture, p + vec2u(1u, 1u), face, 0);
    textureStore(output_texture, global_id.xy, face, (v0 + v1 + v2 + v3) * 0.25);
}

// ---------------------------------------------------------------- bevy_render::maths, utils

fn copysign(a: f32, b: f32) -> f32 {
    return bitcast<f32>((bitcast<u32>(a) & 0x7FFFFFFF) | (bitcast<u32>(b) & 0x80000000));
}

fn orthonormalize(z_basis: vec3<f32>) -> mat3x3<f32> {
    let sign = copysign(1.0, z_basis.z);
    let a = -1.0 / (sign + z_basis.z);
    let b = z_basis.x * z_basis.y * a;
    let x_basis = vec3(1.0 + sign * z_basis.x * z_basis.x * a, sign * b, -sign * z_basis.x);
    let y_basis = vec3(b, sign + z_basis.y * z_basis.y * a, -z_basis.y);
    return mat3x3(x_basis, y_basis, z_basis);
}

fn rand_f(state: ptr<function, u32>) -> f32 {
    *state = *state * 747796405u + 2891336453u;
    let word = ((*state >> ((*state >> 28u) + 4u)) ^ *state) * 277803737u;
    return f32((word >> 22u) ^ word) * bitcast<f32>(0x2f800004u);
}

fn rand_vec2f(state: ptr<function, u32>) -> vec2<f32> {
    return vec2(rand_f(state), rand_f(state));
}

fn hammersley_2d(i: u32, n: u32) -> vec2f {
    let inv_n = 1.0 / f32(n);
    let vdc = f32(reverseBits(i)) * 2.3283064365386963e-10;
    return vec2f(f32(i) * inv_n, vdc);
}

fn sample_cosine_hemisphere(normal: vec3<f32>, rng: ptr<function, u32>) -> vec3<f32> {
    let cos_theta = 1.0 - 2.0 * rand_f(rng);
    let phi = PI_2 * rand_f(rng);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let direction = normal + vec3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    let len_sq = dot(direction, direction);
    if len_sq < 1e-8 { return normal; }
    return direction * inverseSqrt(len_sq);
}

fn sample_cube_dir(uv: vec2f, face: u32) -> vec3f {
    let uvc = 2.0 * uv - 1.0;
    var dir: vec3f;
    switch(face) {
        case 0u: { dir = vec3f( 1.0,  -uvc.y, -uvc.x); }
        case 1u: { dir = vec3f(-1.0,  -uvc.y,  uvc.x); }
        case 2u: { dir = vec3f( uvc.x,  1.0,   uvc.y); }
        case 3u: { dir = vec3f( uvc.x, -1.0,  -uvc.y); }
        case 4u: { dir = vec3f( uvc.x, -uvc.y,  1.0);  }
        case 5u: { dir = vec3f(-uvc.x, -uvc.y, -1.0);  }
        default: { dir = vec3f(0.0); }
    }
    return normalize(dir);
}

struct CubeUV {
    uv: vec2f,
    face: u32,
}

fn dir_to_cube_uv(dir: vec3f) -> CubeUV {
    let abs_dir = abs(dir);
    var face: u32 = 0u;
    var uv: vec2f = vec2f(0.0);

    if (abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z) {
        if (dir.x > 0.0) {
            face = 0u;
            uv = vec2f(-dir.z, -dir.y) / dir.x;
        } else {
            face = 1u;
            uv = vec2f(dir.z, -dir.y) / abs_dir.x;
        }
    } else if (abs_dir.y >= abs_dir.x && abs_dir.y >= abs_dir.z) {
        if (dir.y > 0.0) {
            face = 2u;
            uv = vec2f(dir.x, dir.z) / dir.y;
        } else {
            face = 3u;
            uv = vec2f(dir.x, -dir.z) / abs_dir.y;
        }
    } else {
        if (dir.z > 0.0) {
            face = 4u;
            uv = vec2f(dir.x, -dir.y) / dir.z;
        } else {
            face = 5u;
            uv = vec2f(-dir.x, -dir.y) / abs_dir.z;
        }
    }

    return CubeUV(uv * 0.5 + 0.5, face);
}

// ---------------------------------------------------------------- pbr_lighting

fn D_GGX(roughness: f32, NdotH: f32) -> f32 {
    let oneMinusNdotHSquared = 1.0 - NdotH * NdotH;
    let a = NdotH * roughness;
    let k = roughness / (oneMinusNdotHSquared + a * a);
    let d = k * k * (1.0 / PI);
    return d;
}

fn ggx_vndf_pdf(i: vec3<f32>, NdotH: f32, roughness: f32) -> f32 {
    let ndf = D_GGX(roughness, NdotH);

    let ai = roughness * i.xy;
    let len2 = dot(ai, ai);
    let t = sqrt(len2 + i.z * i.z);
    if i.z >= 0.0 {
        let a = roughness;
        let s = 1.0 + length(i.xy);
        let a2 = a * a;
        let s2 = s * s;
        let k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
        return ndf / (2.0 * (k * i.z + t));
    }

    return ndf * (t - i.z) / (2.0 * len2);
}

fn sample_visible_ggx(
    xi: vec2<f32>,
    roughness: f32,
    normal: vec3<f32>,
    view: vec3<f32>,
) -> vec3<f32> {
    let n = normal;
    let alpha = roughness;

    let wi_n = dot(view, n);
    let wi_z = -n * wi_n;
    let wi_xy = view + wi_z;

    let wi_std = -normalize(alpha * wi_xy + wi_z);

    let wi_std_z = dot(wi_std, n);

    let s = 1.0 + length(wi_xy);
    let a = clamp(alpha, 0.0, 1.0);
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / (s2 + a2 * wi_n * wi_n);
    let b = select(wi_std_z, k * wi_std_z, wi_n > 0.0);

    let z = 1.0 - xi.y * (1.0 + b);
    let sin_theta = sqrt(max(0.0, 1.0 - z * z));
    let phi = 2.0 * PI * xi.x - PI;
    let x = sin_theta * cos(phi);
    let y = sin_theta * sin(phi);
    let c_std = vec3f(x, y, z);

    let TBN = orthonormalize(n);
    let c = TBN * c_std;

    let wm_std = c + wi_std;
    let wm_std_z = n * dot(n, wm_std);
    let wm_std_xy = wm_std_z - wm_std;

    let H = normalize(alpha * wm_std_xy + wm_std_z);

    return reflect(-view, H);
}

fn G_Smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let k = roughness / 2.0;
    let GGXL = NdotL / (NdotL * (1.0 - k) + k);
    let GGXV = NdotV / (NdotV * (1.0 - k) + k);
    return GGXL * GGXV;
}

fn perceptualRoughnessToRoughness(perceptualRoughness: f32) -> f32 {
    let clampedPerceptualRoughness = clamp(perceptualRoughness, 0.089, 1.0);
    return clampedPerceptualRoughness * clampedPerceptualRoughness;
}

// ---------------------------------------------------------------- environment_filter

fn sample_environment(dir: vec3f, level: f32) -> vec4f {
    let cube_uv = dir_to_cube_uv(dir);
    return textureSampleLevel(input_texture, input_sampler, cube_uv.uv, cube_uv.face, level);
}

fn sample_noise(pixel_coords: vec2u) -> vec4f {
    var rng_state: u32 = (pixel_coords.x * 3966231743u) ^ (pixel_coords.y * 3928936651u);
    let rnd = rand_vec2f(&rng_state);
    return vec4f(rnd, 0.0, 0.0);
}

/// Filtered importance sampling: the source mip whose texels match a sample's solid angle.
fn calculate_environment_map_lod(pdf: f32, width: f32, samples: f32) -> f32 {
    let omega_s = 1.0 / (samples * pdf);
    let omega_p = 4.0 * PI / (6.0 * width * width);
    return 0.5 * log2(omega_s / omega_p);
}

@compute
@workgroup_size(8, 8, 1)
fn generate_radiance_map(@builtin(global_invocation_id) global_id: vec3u) {
    let size = textureDimensions(output_texture).xy;
    let source_width = f32(textureDimensions(input_texture, 0).x);
    let invSize = 1.0 / vec2f(size);

    let coords = vec2u(global_id.xy);
    let face = global_id.z;

    if (any(coords >= size)) {
        return;
    }

    let uv = (vec2f(coords) + 0.5) * invSize;
    let normal = sample_cube_dir(uv, face);

    // View direction = normal for the radiance map.
    let view = normal;

    let perceptual_roughness = constants.roughness;
    let roughness = perceptualRoughnessToRoughness(perceptual_roughness);

    let vector_noise = sample_noise(coords);

    var radiance = vec3f(0.0);
    var total_weight = 0.0;

    // Mirror reflection.
    if (roughness < 0.01) {
        radiance = sample_environment(normal, 0.0).rgb;
        textureStore(output_texture, coords, face, vec4f(radiance, 1.0));
        return;
    }

    let sample_count = constants.sample_count;

    for (var i = 0u; i < sample_count; i++) {
        var xi = hammersley_2d(i, sample_count);
        xi = fract(xi + vector_noise.rg);

        let light_dir = sample_visible_ggx(xi, roughness, normal, view);

        let NdotL = dot(normal, light_dir);

        if (NdotL > 0.0) {
            let half_vector = normalize(view + light_dir);
            let NdotH = dot(normal, half_vector);
            let NdotV = dot(normal, view);

            let G = G_Smith(NdotV, NdotL, roughness);

            let pdf = ggx_vndf_pdf(view, NdotH, roughness);

            let lod = calculate_environment_map_lod(pdf, source_width, f32(sample_count));

            let source_mip = max(0.0, lod);

            var sample_color = sample_environment(light_dir, source_mip).rgb;

            radiance += sample_color * NdotL * G;
            total_weight += NdotL * G;
        }
    }

    if (total_weight > 0.0) {
        radiance = radiance / total_weight;
    }

    textureStore(output_texture, coords, face, vec4f(radiance, 1.0));
}

@compute
@workgroup_size(8, 8, 1)
fn generate_irradiance_map(@builtin(global_invocation_id) global_id: vec3u) {
    let size = textureDimensions(output_texture).xy;
    let invSize = 1.0 / vec2f(size);

    let coords = vec2u(global_id.xy);
    let face = global_id.z;

    if (any(coords >= size)) {
        return;
    }

    let uv = (vec2f(coords) + 0.5) * invSize;
    let normal = sample_cube_dir(uv, face);

    var irradiance = vec3f(0.0);

    for (var i = 0u; i < constants.sample_count; i++) {
        var rng: u32 = (coords.x * 2131358057u) ^ (coords.y * 3416869721u) ^ (face * 1199786941u) ^ (i * 566200673u);

        var sample_dir = sample_cosine_hemisphere(normal, &rng);

        var sample_color = sample_environment(sample_dir, 0.0).rgb;

        irradiance += sample_color;
    }

    irradiance = irradiance / f32(constants.sample_count);

    textureStore(output_texture, coords, face, vec4f(irradiance, 1.0));
}
