// Particles as camera-facing quads (bevy_hanabi's render pass with `OrientMode::
// ParallelCameraDepthPlane` and its default unit quad), one instance per particle. Size and colour
// over life are evaluated on the CPU (`particles.rs`); the round mask is hanabi's `RoundModifier`.

struct View {
    clip_from_world: mat4x4<f32>,
    // The camera's world-space right and up axes (columns 0 and 1 of world_from_view).
    right: vec4<f32>,
    up: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: View;

struct Instance {
    @location(0) position_size: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) roundness: f32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) roundness: f32,
}

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32, instance: Instance) -> VertexOutput {
    // Two triangles over hanabi's `Plane3d::new(Vec3::Z, Vec2::splat(0.5))`.
    var corners = array<vec2<f32>, 6>(
        vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0.5, 0.5),
        vec2(-0.5, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5),
    );
    let corner = corners[vertex_index];
    let vpos = corner * instance.position_size.w;
    let world = instance.position_size.xyz + view.right.xyz * vpos.x + view.up.xyz * vpos.y;
    var out: VertexOutput;
    out.position = view.clip_from_world * vec4<f32>(world, 1.0);
    out.color = instance.color;
    out.uv = vec2(corner.x + 0.5, 0.5 - corner.y);
    out.roundness = instance.roundness;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.roundness > 0.0) {
        let n = 2.0 / in.roundness;
        if (pow(abs(1.0 - 2.0 * in.uv.x), n) + pow(abs(1.0 - 2.0 * in.uv.y), n) > 1.0) {
            discard;
        }
    }
    return in.color;
}
