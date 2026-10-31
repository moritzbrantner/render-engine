// Particle quads: bevy_hanabi's render pass (`vfx_render.wesl`) over its default unit quad, one
// instance per alive particle. The simulation's update pass evaluates size and colour over life
// and the flipbook's sprite index (`particle_sim.wgsl`). The vertex stage orients the quad like
// hanabi's `OrientModifier` and offsets its uvs to the sprite like `FlipbookModifier`; the
// fragment stage applies `ParticleTextureModifier`, then `RoundModifier`'s mask.

struct View {
    clip_from_world: mat4x4<f32>,
    // The camera's world-space right and up axes (columns 0 and 1 of world_from_view) and its
    // position.
    right: vec4<f32>,
    up: vec4<f32>,
    eye: vec4<f32>,
}

const ORIENT_CAMERA_PLANE: u32 = 0u;
const ORIENT_ALONG_VELOCITY: u32 = 1u;

const MAPPING_NONE: u32 = 0u;
const MAPPING_MODULATE: u32 = 1u;
const MAPPING_MODULATE_RGB: u32 = 2u;
const MAPPING_MODULATE_OPACITY_FROM_R: u32 = 3u;

struct Emitter {
    roundness: f32,
    orient: u32,
    mapping: u32,
    // Flipbook sprites per row (0: no flipbook) and 1 / grid size.
    flipbook_columns: u32,
    flipbook_scale: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<uniform> emitter: Emitter;
@group(1) @binding(0) var particle_texture: texture_2d<f32>;
@group(1) @binding(1) var particle_sampler: sampler;

struct Instance {
    @location(0) position: vec3<f32>,
    @location(1) sprite: u32,
    @location(2) velocity: vec3<f32>,
    @location(3) size: vec2<f32>,
    @location(4) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32, instance: Instance) -> VertexOutput {
    // Two triangles over hanabi's `Plane3d::new(Vec3::Z, Vec2::splat(0.5))`.
    var corners = array<vec2<f32>, 6>(
        vec2(-0.5, -0.5), vec2(0.5, -0.5), vec2(0.5, 0.5),
        vec2(-0.5, -0.5), vec2(0.5, 0.5), vec2(-0.5, 0.5),
    );
    let corner = corners[vertex_index];
    var axis_x = view.right.xyz;
    var axis_y = view.up.xyz;
    if (emitter.orient == ORIENT_ALONG_VELOCITY) {
        let dir = normalize(instance.position - view.eye.xyz);
        axis_x = normalize(instance.velocity);
        axis_y = cross(dir, axis_x);
    }
    let vpos = corner * instance.size;
    let world = instance.position + axis_x * vpos.x + axis_y * vpos.y;
    var out: VertexOutput;
    out.position = view.clip_from_world * vec4<f32>(world, 1.0);
    out.color = instance.color;
    var uv = vec2(corner.x + 0.5, 0.5 - corner.y);
    if (emitter.flipbook_columns > 0u) {
        let ij = vec2<f32>(
            f32(instance.sprite % emitter.flipbook_columns),
            f32(instance.sprite / emitter.flipbook_columns),
        );
        uv = (ij + uv) * emitter.flipbook_scale;
    }
    out.uv = uv;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = in.color;
    // Sampled in uniform control flow: every fragment of the draw takes the same branch.
    let tex = textureSample(particle_texture, particle_sampler, in.uv);
    switch emitter.mapping {
        case MAPPING_MODULATE: { color = color * tex; }
        case MAPPING_MODULATE_RGB: { color = vec4<f32>(color.rgb * tex.rgb, color.a); }
        case MAPPING_MODULATE_OPACITY_FROM_R: { color.a = color.a * tex.r; }
        default: {}
    }
    if (emitter.roundness > 0.0) {
        let n = 2.0 / emitter.roundness;
        if (pow(abs(1.0 - 2.0 * in.uv.x), n) + pow(abs(1.0 - 2.0 * in.uv.y), n) > 1.0) {
            discard;
        }
    }
    return color;
}
