// Tone-mapping pass: reads the resolved HDR frame and writes display colour into an Rgba16Float
// target, which the blit pass copies into the (sRGB) output.
//
// Ported from Bevy 0.20 `bevy_core_pipeline` (`tonemapping.wesl`, `tonemapping_frag.wesl`), MIT or
// Apache-2.0, see `src/luts/README.md`. Bevy's default colour grading is the identity and
// `DebandDither` stays off, so only the operators are ported. The LUTs are Bevy's own files.

// 0: ACES fitted, 1: Tony McMapface, 2: AgX (`scene::Tonemapping`).
override METHOD: u32 = 1u;

@group(0) @binding(0) var hdr_texture: texture_2d<f32>;
@group(0) @binding(1) var lut_texture: texture_3d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;

@vertex
fn tonemap_vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    // One triangle covering the screen.
    let uv = vec2(f32(index >> 1u), f32(index & 1u)) * 2.0;
    return vec4(uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.0, 1.0);
}

fn sample_lut(p: vec3<f32>) -> vec3<f32> {
    return textureSampleLevel(lut_texture, lut_sampler, p, 0.0).rgb;
}

// Rec. 709 luminance.
fn luminance(v: vec3<f32>) -> f32 {
    return dot(v, vec3(0.2126, 0.7152, 0.0722));
}

// --- Tony McMapface (Tomasz Stachowiak, https://github.com/h3r2tic/tony-mc-mapface) ---

const TONY_MC_MAPFACE_LUT_DIMS: f32 = 48.0;

fn tony_mc_mapface(stimulus: vec3<f32>) -> vec3<f32> {
    let uv = (stimulus / (stimulus + 1.0)) * ((TONY_MC_MAPFACE_LUT_DIMS - 1.0) / TONY_MC_MAPFACE_LUT_DIMS)
        + 0.5 / TONY_MC_MAPFACE_LUT_DIMS;
    return sample_lut(saturate(uv));
}

// --- ACES fitted (Stephen Hill, https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl) ---

fn rrt_and_odt_fit(v: vec3<f32>) -> vec3<f32> {
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

fn aces_fitted(color: vec3<f32>) -> vec3<f32> {
    // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
    let rgb_to_rrt = mat3x3<f32>(
        vec3(0.59719, 0.35458, 0.04823),
        vec3(0.07600, 0.90834, 0.01566),
        vec3(0.02840, 0.13383, 0.83777),
    );
    // ODT_SAT => XYZ => D60_2_D65 => sRGB
    let odt_to_rgb = mat3x3<f32>(
        vec3(1.60475, -0.53108, -0.07367),
        vec3(-0.10208, 1.10813, -0.00605),
        vec3(-0.00327, -0.07276, 1.07602),
    );
    var c = color * rgb_to_rrt;
    c = rrt_and_odt_fit(c);
    c = c * odt_to_rgb;
    return saturate(c);
}

// --- AgX (Troy Sobotka, https://github.com/sobotka/AgX, LUT from https://github.com/MrLixm/AgXc) ---

fn open_domain_to_normalized_log2(color: vec3<f32>, minimum_ev: f32, maximum_ev: f32) -> vec3<f32> {
    let in_midgray = 0.18;
    var c = max(vec3(0.0), color);
    c = select(c, 0.00001525878 + c, c < vec3<f32>(0.00003051757));
    c = clamp(log2(c / in_midgray), vec3(minimum_ev), vec3(maximum_ev));
    return (c - minimum_ev) / (maximum_ev - minimum_ev);
}

fn agx_log(image: vec3<f32>) -> vec3<f32> {
    var c = max(vec3(0.0), image);
    let r = dot(c, vec3(0.84247906, 0.0784336, 0.07922375));
    let g = dot(c, vec3(0.04232824, 0.87846864, 0.07916613));
    let b = dot(c, vec3(0.04237565, 0.0784336, 0.87914297));
    c = open_domain_to_normalized_log2(vec3(r, g, b), -10.0, 6.5);
    return clamp(c, vec3(0.0), vec3(1.0));
}

fn apply_lut_3d(image: vec3<f32>, block_size: f32) -> vec3<f32> {
    return sample_lut(image * ((block_size - 1.0) / block_size) + 0.5 / block_size);
}

fn agx(color: vec3<f32>) -> vec3<f32> {
    return apply_lut_3d(agx_log(color), 32.0);
}

@fragment
fn tonemap_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let hdr = textureLoad(hdr_texture, vec2<i32>(position.xy), 0);
    let color = max(hdr.rgb, vec3(0.0));
    var out: vec3<f32>;
    switch METHOD {
        case 0u: { out = aces_fitted(color); }
        case 2u: { out = agx(color); }
        default: { out = tony_mc_mapface(color); }
    }
    // Bevy's post-tonemapping saturation at its default of 1.
    let luma = luminance(out);
    out = mix(vec3(luma), out, vec3(1.0));
    return vec4(out, hdr.a);
}

// --- Blit (Bevy's upscaling node, `blit.wesl`) ---
//
// Bevy stores the tone-mapped frame in its Rgba16Float main texture and copies it into the
// output with a non-filtering sampler. Tone mapping into a real Rgba16Float target and copying
// it here leaves that rounding to the device's own f16 store, as on Bevy: lavapipe's store
// does not round to nearest like a shader-side `pack2x16float` (D4).

@group(0) @binding(0) var post_texture: texture_2d<f32>;

@fragment
fn blit_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(post_texture, vec2<i32>(position.xy), 0);
}
