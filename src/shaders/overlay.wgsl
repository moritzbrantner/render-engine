// Screen-space overlay (`Scene::overlay`): solid quads drawn over the tone-mapped Rgba16Float
// frame before the blit, alpha blended source-over like Bevy UI's nodes over its main texture.
// Vertices arrive in clip space with a linear RGB colour and straight alpha; no depth, exposure
// or tone mapping applies.

struct Vertex {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn overlay_vs(v: Vertex) -> Varyings {
    return Varyings(vec4(v.position, 0.0, 1.0), v.color);
}

@fragment
fn overlay_fs(in: Varyings) -> @location(0) vec4<f32> {
    return in.color;
}
