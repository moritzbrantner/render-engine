// Lit and fogged particles (opt-in, not in bevy_hanabi), appended to `lit.wgsl` and
// `particles.wgsl` (its groups moved to 2 and 3; `particle_pass.rs`). The lit pass's frame and
// lighting groups are 0 and 1. A lit particle is shaded as a Bevy `StandardMaterial` on the
// camera-facing quad: base colour = the particle's colour after its texture, normal = the quad's,
// towards the camera; then distance fog applies as Bevy's `main_pass_post_lighting_processing`
// applies it, to lit and unlit particles alike. Technical decisions D9.

fn shaded_color(in: VertexOutput) -> vec4<f32> {
    var color = particle_color(in);
    if ((emitter.flags & EMITTER_LIT) != 0u) {
        var v: VertexOut;
        v.clip = in.position;
        v.world = in.world;
        v.normal = in.normal;
        v.instance = 0u;
        v.uv = in.uv;
        v.tangent = vec4(0.0);
        v.color = vec4(1.0);
        var m: MaterialSample;
        m.base_color = color;
        m.emissive = vec3(0.0);
        m.perceptual_roughness = emitter.surface.x;
        m.metallic = emitter.surface.y;
        m.reflectance = emitter.surface.z;
        m.diffuse_occlusion = 1.0;
        m.Nt = vec3(0.0);
        color = vec4(lit_radiance(v, in.normal, normalize(in.normal), m), color.a);
    }
    if ((emitter.flags & EMITTER_FOG) != 0u && frame.fog_bi.w > 0.5) {
        color = vec4(apply_fog(color.rgb, in.world), color.a);
    }
    return color;
}

@fragment
fn fragment_shaded(in: VertexOutput) -> @location(0) vec4<f32> {
    return shaded_color(in);
}

@fragment
fn fragment_shaded_soft(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = shaded_color(in);
    color.a = color.a * depth_fade(in);
    return color;
}
