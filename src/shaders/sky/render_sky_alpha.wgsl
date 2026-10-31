
// Single-source output: the blend multiplies the frame by the mean transmittance in alpha.
@fragment
fn main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let sky = render_sky(in);
    let mean_transmittance = (sky.transmittance.r + sky.transmittance.g + sky.transmittance.b) / 3.0;
    return vec4(sky.inscattering, mean_transmittance);
}
