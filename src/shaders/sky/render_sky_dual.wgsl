
// Dual-source output: the blend multiplies the frame by the per-channel transmittance.
struct RenderSkyOutput {
    @location(0) @blend_src(0) inscattering: vec4<f32>,
    @location(0) @blend_src(1) transmittance: vec4<f32>,
}

@fragment
fn main(in: FullscreenVertexOutput) -> RenderSkyOutput {
    let sky = render_sky(in);
    return RenderSkyOutput(vec4(sky.inscattering, 0.0), vec4(sky.transmittance, 1.0));
}
