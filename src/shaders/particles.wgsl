// Particle quads: bevy_hanabi's render pass (`vfx_render.wesl`) over its default unit quad, one
// instance per alive particle. The simulation's update pass evaluates size and colour over life
// and the flipbook's sprite index (`particle_sim.wgsl`). The vertex stage orients the quad like
// hanabi's `OrientModifier` and offsets its uvs to the sprite like `FlipbookModifier`; the
// fragment stage applies `ParticleTextureModifier`, then `RoundModifier`'s mask.
//
// Lit and fogged emitters compile this file after `lit.wgsl` with its groups moved from 0 and 1
// to 2 and 3, followed by `particles_shaded.wgsl` (`particle_pass.rs`).

struct ParticleView {
    clip_from_world: mat4x4<f32>,
    // The camera's world-space right and up axes (columns 0 and 1 of world_from_view) and its
    // position.
    right: vec4<f32>,
    up: vec4<f32>,
    eye: vec4<f32>,
    // x: the camera's near plane. Reverse-Z infinite depth d is near / view distance.
    params: vec4<f32>,
}

const ORIENT_CAMERA_PLANE: u32 = 0u;
const ORIENT_ALONG_VELOCITY: u32 = 1u;

const MAPPING_NONE: u32 = 0u;
const MAPPING_MODULATE: u32 = 1u;
const MAPPING_MODULATE_RGB: u32 = 2u;
const MAPPING_MODULATE_OPACITY_FROM_R: u32 = 3u;

const EMITTER_LIT: u32 = 1u;
const EMITTER_FOG: u32 = 2u;

struct Emitter {
    roundness: f32,
    orient: u32,
    mapping: u32,
    // Flipbook sprites per row (0: no flipbook) and 1 / grid size.
    flipbook_columns: u32,
    flipbook_scale: vec2<f32>,
    // Soft particles: metres over which the particle fades out in front of the scene (0: off).
    depth_fade: f32,
    // EMITTER_LIT, EMITTER_FOG.
    flags: u32,
    // Lit particles: perceptual roughness, metallic, reflectance.
    surface: vec4<f32>,
}

@group(0) @binding(0) var<uniform> view: ParticleView;
@group(0) @binding(1) var<uniform> emitter: Emitter;
// The opaque scene's depth, for soft emitters (the particle pass binds the target read-only).
@group(0) @binding(2) var scene_depth: texture_depth_multisampled_2d;
@group(1) @binding(0) var particle_texture: texture_2d<f32>;
@group(1) @binding(1) var particle_sampler: sampler;

struct ParticleInstance {
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
    @location(2) world: vec3<f32>,
    // The quad's unit normal, towards the camera (lit particles).
    @location(3) normal: vec3<f32>,
}

@vertex
fn vertex(@builtin(vertex_index) vertex_index: u32, instance: ParticleInstance) -> VertexOutput {
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
    out.world = world;
    // x × y is the camera's back axis for camera-plane quads; a velocity-aligned quad's turns
    // to the camera too.
    let normal = normalize(cross(axis_x, axis_y));
    out.normal = select(normal, -normal, dot(normal, view.eye.xyz - instance.position) < 0.0);
    return out;
}

fn particle_color(in: VertexOutput) -> vec4<f32> {
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

// Soft particles (not in bevy_hanabi): alpha fades to zero over the last `depth_fade` metres of
// view distance in front of the opaque scene, read from sample 0 of the multisampled depth.
// Additive and alpha blending both scale the colour by alpha, so one fade serves both.
fn depth_fade(in: VertexOutput) -> f32 {
    let near = view.params.x;
    let scene = textureLoad(scene_depth, vec2<i32>(in.position.xy), 0);
    // Depth 0 (nothing drawn: sky) is infinitely far.
    let scene_distance = near / max(scene, 1e-30);
    let distance = near / in.position.z;
    return clamp((scene_distance - distance) / emitter.depth_fade, 0.0, 1.0);
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return particle_color(in);
}

@fragment
fn fragment_soft(in: VertexOutput) -> @location(0) vec4<f32> {
    var color = particle_color(in);
    color.a = color.a * depth_fade(in);
    return color;
}
