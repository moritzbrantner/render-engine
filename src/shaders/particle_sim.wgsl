// GPU particle simulation, replaying bevy_hanabi's init and update passes (technical decisions
// D9). All emitters of a world share one pool of particle slots; each emitter owns the range
// [base, base + capacity). One step is three dispatches over every emitter at once:
//
// - `init`: one thread per particle the CPU spawners asked for this step. Spawn `i` of an
//   emitter with `alive` particles takes the slot in dead-stack row `alive + i` (hanabi's
//   `dead_index`, starting as 0, 1, 2, …), seeds its stream with `pcg_hash(slot ^ seed)` and runs
//   the init modifiers in order. It appends the slot to the emitter's current alive list.
// - `prepare`: one thread. Caps each emitter's alive count at its capacity, lays the alive lists
//   end to end for `update` (the prefix sums its threads search) and writes `update`'s indirect
//   dispatch and every emitter's draw arguments.
// - `update`: one thread per alive particle. Ages, integrates and reaps it; a dead slot goes back
//   on the dead stack at the decremented alive count, a survivor is appended to the next alive
//   list (through the draw arguments' instance count) along with its quad for the render pass.
//
// The alive lists ping-pong between two buffers from step to step.

struct Modifier {
    kind: u32,
    mode: u32,
    _pad0: u32,
    _pad1: u32,
    a: vec4<f32>,
    b: vec4<f32>,
}

// `particle_sim::GpuEmitterStep`.
struct EmitterStep {
    world_from_emitter: mat4x4<f32>,
    accel: vec4<f32>,
    color_t: array<vec4<f32>, 2>,
    color_v: array<vec4<f32>, 8>,
    size_t: array<vec4<f32>, 2>,
    // Two (x, y) keys per element: key i in `.xy` (even i) or `.zw` (odd i) of element i / 2.
    size_v: array<vec4<f32>, 4>,
    id: u32,
    base: u32,
    capacity: u32,
    seed: u32,
    spawn: u32,
    spawn_first: u32,
    n_mods: u32,
    color_blend: u32,
    n_color: u32,
    n_size: u32,
    sprite_frames: u32,
    _pad0: u32,
    mods: array<Modifier, 8>,
}

struct Step {
    count: u32,
    total_spawn: u32,
    dt: f32,
    _pad: u32,
    emitters: array<EmitterStep>,
}

struct Particle {
    position: vec3<f32>,
    age: f32,
    velocity: vec3<f32>,
    lifetime: f32,
    color: vec4<f32>,
    size: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// One quad of the render pass (`particle_pass`): position, flipbook sprite, velocity (for
// `Orient::AlongVelocity`), size, linear colour and alpha.
struct Quad {
    position: vec3<f32>,
    sprite: u32,
    velocity: vec3<f32>,
    _pad0: u32,
    size: vec2<f32>,
    _pad1: vec2<f32>,
    color: vec4<f32>,
}

// Per emitter id: its draw arguments (`draw_indirect`, the instance count is the length of the
// next alive list), then its alive count.
struct Counters {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
    alive: atomic<u32>,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct Update {
    dispatch: array<u32, 3>,
    total: u32,
    // Per emitter of the step (in step order): the first update thread of its alive list.
    first: array<u32>,
}

@group(0) @binding(0) var<storage, read> step: Step;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> dead: array<u32>;
@group(0) @binding(3) var<storage, read_write> quads: array<Quad>;
// Indexed by emitter id.
@group(0) @binding(4) var<storage, read_write> counters: array<Counters>;
// The update pass reads what prepare wrote, and dispatches on it (so it cannot be read-write).
@group(1) @binding(0) var<storage, read_write> update_args: Update;
@group(2) @binding(0) var<storage, read> update_read: Update;
// This step's alive lists: `current` is read by update, `next` written by it.
@group(0) @binding(5) var<storage, read_write> current: array<u32>;
@group(0) @binding(6) var<storage, read_write> next: array<u32>;

const KIND_POSITION_SPHERE: u32 = 0u;
const KIND_POSITION: u32 = 1u;
const KIND_VELOCITY_SPHERE: u32 = 2u;
const KIND_VELOCITY: u32 = 3u;
const KIND_LIFETIME: u32 = 4u;
const KIND_AGE: u32 = 5u;
const KIND_SIZE: u32 = 6u;
const KIND_COLOR: u32 = 7u;

const MODE_CONST: u32 = 0u;
// PositionSphere: volume = const mode, surface = this.
const MODE_UNIFORM: u32 = 1u;

const BLEND_OVERWRITE: u32 = 0u;
const BLEND_MULTIPLY: u32 = 1u;
const BLEND_ADD: u32 = 2u;
const BLEND_NONE: u32 = 3u;

const TAU: f32 = 6.283185307179586;

// hanabi's `vfx_common.wesl`.
var<private> seed: u32;

fn pcg_hash(input: u32) -> u32 {
    var state: u32 = input * 747796405u + 2891336453u;
    var word: u32 = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn to_float01(u: u32) -> f32 {
    return bitcast<f32>((u & 0x007fffffu) | 0x3f800000u) - 1.;
}

fn frand() -> f32 {
    seed = pcg_hash(seed);
    return to_float01(pcg_hash(seed));
}

fn frand3() -> vec3<f32> {
    seed = pcg_hash(seed);
    let x = to_float01(seed);
    seed = pcg_hash(seed);
    let y = to_float01(seed);
    seed = pcg_hash(seed);
    let z = to_float01(seed);
    return vec3<f32>(x, y, z);
}

fn frand4() -> vec4<f32> {
    let r0 = pcg_hash(seed);
    let r1 = pcg_hash(r0);
    let r2 = pcg_hash(r1);
    seed = r2;
    let x = to_float01(r0);
    let y = to_float01(((r0 & 0xff000000u) >> 8u) | (r1 & 0x0000ffffu));
    let z = to_float01(((r1 & 0xffff0000u) >> 8u) | (r2 & 0x000000ffu));
    let w = to_float01(r2 >> 8u);
    return vec4<f32>(x, y, z, w);
}

fn scalar(m: Modifier) -> f32 {
    if (m.mode == MODE_CONST) {
        return m.a.x;
    }
    return m.a.x + frand() * (m.a.y - m.a.x);
}

fn vector3(m: Modifier) -> vec3<f32> {
    if (m.mode == MODE_CONST) {
        return m.a.xyz;
    }
    return frand3() * (m.b.xyz - m.a.xyz) + m.a.xyz;
}

fn vector4(m: Modifier) -> vec4<f32> {
    if (m.mode == MODE_CONST) {
        return m.a;
    }
    return frand4() * (m.b - m.a) + m.a;
}

// The last emitter of the step whose first thread is at or before `thread`
// (`first(k)` is non-decreasing in k; emitters with no threads share their successor's first).
fn find_spawn_emitter(thread: u32) -> u32 {
    var lo = 0u;
    var hi = step.count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (step.emitters[mid].spawn_first <= thread) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo - 1u;
}

fn find_update_emitter(thread: u32) -> u32 {
    var lo = 0u;
    var hi = step.count;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (update_read.first[mid] <= thread) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo - 1u;
}

@compute @workgroup_size(64)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread = gid.x;
    if (thread >= step.total_spawn) {
        return;
    }
    let k = find_spawn_emitter(thread);
    let e = &step.emitters[k];
    let row = atomicLoad(&counters[(*e).id].alive) + (thread - (*e).spawn_first);
    if (row >= (*e).capacity) {
        return;
    }
    let base = (*e).base;
    let slot = dead[base + row];
    seed = pcg_hash(slot ^ (*e).seed);

    var p: Particle;
    p.lifetime = 1.0;
    p.size = 1.0;
    p.color = vec4<f32>(1.0);
    for (var i = 0u; i < (*e).n_mods; i++) {
        let m = (*e).mods[i];
        switch m.kind {
            case KIND_POSITION_SPHERE: {
                // hanabi `SetPositionSphereModifier`.
                var r = m.a.w;
                if (m.mode == MODE_CONST) {
                    r = pow(frand(), 1. / 3.) * m.a.w;
                }
                let theta = frand() * TAU;
                let z = frand() * 2. - 1.;
                let phi = acos(z);
                let sinphi = sin(phi);
                let dir = vec3<f32>(sinphi * cos(theta), sinphi * sin(theta), z);
                p.position = m.a.xyz + r * dir;
            }
            case KIND_POSITION: {
                p.position = vector3(m);
            }
            case KIND_VELOCITY_SPHERE: {
                // hanabi `SetVelocitySphereModifier`; the speed is in `b`.
                var speed = m.b.x;
                if (m.mode == MODE_UNIFORM) {
                    speed = m.b.x + frand() * (m.b.y - m.b.x);
                }
                p.velocity = normalize(p.position - m.a.xyz) * speed;
            }
            case KIND_VELOCITY: {
                p.velocity = vector3(m);
            }
            case KIND_LIFETIME: {
                p.lifetime = scalar(m);
            }
            case KIND_AGE: {
                p.age = scalar(m);
            }
            case KIND_SIZE: {
                p.size = scalar(m);
            }
            case KIND_COLOR: {
                p.color = vector4(m);
            }
            default: {}
        }
    }
    p.position = ((*e).world_from_emitter * vec4<f32>(p.position, 1.0)).xyz;
    p.velocity = ((*e).world_from_emitter * vec4<f32>(p.velocity, 0.0)).xyz;
    particles[base + slot] = p;
    current[base + row] = slot;
}

@compute @workgroup_size(1)
fn prepare() {
    var total = 0u;
    for (var k = 0u; k < step.count; k++) {
        let e = &step.emitters[k];
        let n = min(atomicLoad(&counters[(*e).id].alive) + (*e).spawn, (*e).capacity);
        atomicStore(&counters[(*e).id].alive, n);
        update_args.first[k] = total;
        total += n;
        counters[(*e).id].vertex_count = 6u;
        atomicStore(&counters[(*e).id].instance_count, 0u);
        counters[(*e).id].first_vertex = 0u;
        counters[(*e).id].first_instance = 0u;
    }
    update_args.total = total;
    update_args.dispatch = array<u32, 3>((total + 63u) / 64u, 1u, 1u);
}

// hanabi's generated gradient function: clamped at both ends, `mix` between keys.
fn sample_color(k: u32, t: f32) -> vec4<f32> {
    let e = &step.emitters[k];
    let n = (*e).n_color;
    if (n == 1u || t <= (*e).color_t[0].x) {
        return (*e).color_v[0];
    }
    for (var i = 1u; i < n; i++) {
        let ta = (*e).color_t[(i - 1u) / 4u][(i - 1u) % 4u];
        let tb = (*e).color_t[i / 4u][i % 4u];
        if (t <= tb) {
            return mix((*e).color_v[i - 1u], (*e).color_v[i], (t - ta) / (tb - ta));
        }
    }
    return (*e).color_v[n - 1u];
}

fn size_key(k: u32, i: u32) -> vec2<f32> {
    let v = step.emitters[k].size_v[i / 2u];
    if (i % 2u == 0u) {
        return v.xy;
    }
    return v.zw;
}

fn sample_size(k: u32, t: f32) -> vec2<f32> {
    let e = &step.emitters[k];
    let n = (*e).n_size;
    if (n == 1u || t <= (*e).size_t[0].x) {
        return size_key(k, 0u);
    }
    for (var i = 1u; i < n; i++) {
        let ta = (*e).size_t[(i - 1u) / 4u][(i - 1u) % 4u];
        let tb = (*e).size_t[i / 4u][i % 4u];
        if (t <= tb) {
            return mix(size_key(k, i - 1u), size_key(k, i), (t - ta) / (tb - ta));
        }
    }
    return size_key(k, n - 1u);
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread = gid.x;
    if (thread >= update_read.total) {
        return;
    }
    let k = find_update_emitter(thread);
    let e = &step.emitters[k];
    let base = (*e).base;
    let slot = current[base + thread - update_read.first[k]];
    var p = particles[base + slot];
    let dt = step.dt;
    p.age += dt;
    p.velocity += (*e).accel.xyz * dt;
    p.position += p.velocity * dt;
    particles[base + slot] = p;
    if (p.age >= p.lifetime) {
        let row = atomicSub(&counters[(*e).id].alive, 1u) - 1u;
        dead[base + row] = slot;
        return;
    }
    let j = atomicAdd(&counters[(*e).id].instance_count, 1u);
    next[base + j] = slot;

    // hanabi's render modifiers, evaluated once per step instead of per vertex.
    let t = p.age / p.lifetime;
    var size = vec2<f32>(p.size);
    if ((*e).n_size > 0u) {
        size = sample_size(k, t);
    }
    var color = p.color;
    if ((*e).color_blend != BLEND_NONE) {
        let g = sample_color(k, t);
        switch (*e).color_blend {
            case BLEND_OVERWRITE: { color = g; }
            case BLEND_MULTIPLY: { color = p.color * g; }
            case BLEND_ADD: { color = p.color + g; }
            default: {}
        }
    }
    // The flipbook's `SetAttributeModifier(SPRITE_INDEX, i32(age / lifetime * frames))`.
    var sprite = 0u;
    if ((*e).sprite_frames > 0u) {
        sprite = u32(i32(p.age / p.lifetime * f32((*e).sprite_frames)));
    }
    quads[base + j] = Quad(p.position, sprite, p.velocity, 0u, size, vec2<f32>(0.0), color);
}
