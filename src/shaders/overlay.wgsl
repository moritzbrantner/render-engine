// Screen-space overlay (`Scene::overlay`): quads drawn over the tone-mapped Rgba16Float frame
// before the blit, alpha blended source-over like Bevy UI's nodes over its main texture.
// Vertices arrive in clip space with a linear RGB colour and straight alpha; no depth, exposure
// or tone mapping applies. A quad's colour tints its image (white for none, Bevy `ImageNode`),
// and its rounded corners are cut by its signed distance, anti-aliased over a pixel as in Bevy
// UI (`ui.wgsl`: `sd_rounded_box`, `antialias`).

@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;

struct Vertex {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    // Pixels from the quad's centre.
    @location(2) local: vec2<f32>,
    // Half width, half height, corner radius (pixels).
    @location(3) shape: vec4<f32>,
    @location(4) uv: vec2<f32>,
}

struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local: vec2<f32>,
    @location(2) shape: vec4<f32>,
    @location(3) uv: vec2<f32>,
}

@vertex
fn overlay_vs(v: Vertex) -> Varyings {
    return Varyings(vec4(v.position, 0.0, 1.0), v.color, v.local, v.shape, v.uv);
}

// Signed distance from `point` to a box of half size `half` with corners rounded to `radius`.
fn rounded_box(point: vec2<f32>, half: vec2<f32>, radius: f32) -> f32 {
    let q = abs(point) - half + radius;
    return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

@fragment
fn overlay_fs(in: Varyings) -> @location(0) vec4<f32> {
    let texel = textureSample(image, image_sampler, in.uv);
    let coverage = saturate(0.5 - rounded_box(in.local, in.shape.xy, in.shape.z));
    let color = in.color * texel;
    return vec4(color.rgb, color.a * coverage);
}
