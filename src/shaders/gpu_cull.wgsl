// Experimental opt-in GPU frustum culling and compaction (`renderer/gpu_culling.rs`,
// `docs/gpu-driven-experiment.md`). One compute pass tests every instance against every view,
// appending the survivors to the view's run of the visible-index buffer and bumping the run's
// indirect-draw instance count. The tests are the CPU path's (`src/cull.rs`).

struct Instance {
    world_from_local: mat4x4<f32>,
    normal_from_local: mat3x3<f32>,
    // x tangents, y mirrored, z first joint, w the instance's group row + 1 (0: not culled here).
    flags: vec4<u32>,
}

// A mesh/material/face run the pass culls.
struct Group {
    center: vec4<f32>,
    half_extents: vec4<f32>,
}

struct View {
    planes: array<vec4<f32>, 6>,
    // Position and range of a local light's sphere (Bevy's first test).
    sphere: vec4<f32>,
    plane_count: u32,
    has_sphere: u32,
}

struct Params {
    group_count: u32,
    instance_count: u32,
}

struct DrawArgs {
    index_count: u32,
    instance_count: atomic<u32>,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> instances: array<Instance>;
@group(0) @binding(2) var<storage, read> groups: array<Group>;
@group(0) @binding(3) var<storage, read> views: array<View>;
@group(0) @binding(4) var<storage, read_write> visible: array<u32>;
@group(0) @binding(5) var<storage, read_write> draw_args: array<DrawArgs>;

fn relative_radius(axes: mat3x3<f32>, direction: vec3<f32>) -> f32 {
    return abs(dot(direction, axes[0])) + abs(dot(direction, axes[1]))
        + abs(dot(direction, axes[2]));
}

// One instance against one view: x is the instance, y the view (a 2D dispatch).
@compute @workgroup_size(64)
fn cull(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.instance_count {
        return;
    }
    let instance_index = id.x;
    let view_index = id.y;
    let row_plus_one = instances[instance_index].flags.w;
    if row_plus_one == 0u {
        return;
    }
    let row = row_plus_one - 1u;

    // `Obb::new`: the mesh box under the model matrix.
    let m = instances[instance_index].world_from_local;
    let group = groups[row];
    let center = (m * vec4(group.center.xyz, 1.0)).xyz;
    let axes = mat3x3<f32>(
        m[0].xyz * group.half_extents.x,
        m[1].xyz * group.half_extents.y,
        m[2].xyz * group.half_extents.z,
    );

    let view = views[view_index];
    if view.has_sphere != 0u {
        // `Obb::intersects_sphere`, as the CPU path tests a light's casters.
        let v = center - view.sphere.xyz;
        let d = length(v);
        let reaches = d <= view.sphere.w || d <= view.sphere.w + relative_radius(axes, v / d);
        if !reaches {
            return;
        }
    }
    for (var p = 0u; p < view.plane_count; p += 1u) {
        let plane = view.planes[p];
        let n = plane.xyz;
        let distance = dot(n, center) + plane.w;
        // A centre inside the half space passes without the box's extent, as on the CPU.
        if distance <= 0.0 && distance + relative_radius(axes, n) <= 0.0 {
            return;
        }
    }

    let slot = view_index * params.group_count + row;
    let index = atomicAdd(&draw_args[slot].instance_count, 1u);
    visible[draw_args[slot].first_instance + index] = instance_index;
}
