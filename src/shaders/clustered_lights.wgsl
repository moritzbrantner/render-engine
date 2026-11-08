// Appended to `lit.wgsl` for the opt-in clustered light assignment (`LightAssignment::Clustered`,
// `renderer/clustered.rs`, `docs/forward-plus-experiment.md`). The light loop's text in `lit.wgsl`
// is replaced by a call to `clustered_local_lights`. The default path compiles `lit.wgsl` as is.

struct ClusterParams {
    // x, y, z: tiles and depth slices; w: unused.
    grid: vec4<u32>,
    // x, y: tiles per pixel; z: far edge of depth slice 0 (m); w: slices per ln(depth / z).
    factors: vec4<f32>,
}

// `(offset, points, spots)` per cluster, indexed by `cluster_index`.
@group(3) @binding(0) var<uniform> cluster: ClusterParams;
@group(3) @binding(1) var<storage, read> clusters: array<u32>;
@group(3) @binding(2) var<storage, read> cluster_light_indices: array<u32>;

fn cluster_index(frag_xy: vec2<f32>, world: vec3<f32>) -> u32 {
    let depth = -(frame.view_from_world * vec4(world, 1.0)).z;
    let tile = min(vec2<u32>(frag_xy * cluster.factors.xy), cluster.grid.xy - 1u);
    var slice = 0u;
    if (depth > cluster.factors.z) {
        slice = min(1u + u32(cluster.factors.w * log(depth / cluster.factors.z)), cluster.grid.z - 1u);
    }
    return (slice * cluster.grid.y + tile.y) * cluster.grid.x + tile.x;
}

// The same terms as the per-fragment loop, in the same order, over this cluster's lights. A light
// missing from the list contributes exactly zero to the sum (it cannot reach any fragment in the
// cluster), so the result is identical.
fn clustered_local_lights(in: VertexOut, world_normal: vec3<f32>, surface: Surface) -> vec3<f32> {
    let c = cluster_index(in.clip.xy, in.world) * 3u;
    let offset = clusters[c];
    let point_end = offset + clusters[c + 1u];
    let spot_end = point_end + clusters[c + 2u];
    // cluster-diagnostic-hook
    var sum = vec3(0.0);
    for (var i = offset; i < point_end; i++) {
        let light = lights[cluster_light_indices[i]];
        var shadow = 1.0;
        if ((light.flags & LIGHT_SHADOWS) != 0u) {
            shadow = fetch_point_shadow(light, in.world, world_normal, in.clip.xy);
        }
        sum += point_light(light, surface) * shadow;
    }
    for (var i = point_end; i < spot_end; i++) {
        let light = lights[cluster_light_indices[i]];
        var shadow = 1.0;
        if ((light.flags & LIGHT_SHADOWS) != 0u) {
            shadow = fetch_spot_shadow(light, in.world, world_normal, in.clip.xy);
        }
        sum += spot_light(light, surface) * shadow;
    }
    return sum;
}
