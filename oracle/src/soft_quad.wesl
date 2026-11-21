// Soft particles on the Bevy side of `particles_soft` (bevy_side::SoftFade): StandardMaterial's
// forward fragment stage (bevy_pbr pbr.wesl, without its OIT, decal and dither branches), with
// render-engine's depth fade (particles.wgsl depth_fade) on the base colour's alpha before
// lighting: alpha fades to zero over the last `depth_fade` metres of view distance in front of
// the opaque scene, read from sample 0 of the depth prepass. Emissive light is zero on these
// quads, so the faded alpha scales the lit colour as render-engine's fade after shading does.
import bevy_pbr::{
    render::{
        forward_io::{VertexOutput, FragmentOutput},
        pbr_fragment::pbr_input_from_standard_material,
        pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
        pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
        view_transformations::depth_ndc_to_view_z,
    },
    prepass::utils::prepass_depth,
};

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<uniform> depth_fade: f32;

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(
        pbr_input.material.flags,
        pbr_input.material.alpha_cutoff,
        pbr_input.material.base_color,
    );

    // Bevy's infinite reverse-Z: view z = -near / depth; depth 0 (nothing drawn) is infinitely far.
    let scene = -depth_ndc_to_view_z(max(prepass_depth(in.position, 0u), 1e-30));
    let distance = -depth_ndc_to_view_z(in.position.z);
    pbr_input.material.base_color.a *= clamp((scene - distance) / depth_fade, 0.0, 1.0);

    var out: FragmentOutput;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        out.color = apply_pbr_lighting(pbr_input);
    } else {
        out.color = pbr_input.material.base_color;
    }
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
    return out;
}
