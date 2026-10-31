// Bevy `atmosphere/environment.wesl`: the sky-view LUT rendered into a cube map, the source of
// the sky's environment-map light.

@group(0) @binding(13) var output: texture_storage_2d_array<rgba16float, write>;

// Bevy `render::utils::sample_cube_dir`.
fn sample_cube_dir(uv: vec2f, face: u32) -> vec3f {
    let uvc = 2.0 * uv - 1.0;
    var dir: vec3f;
    switch(face) {
        case 0u: { dir = vec3f( 1.0,  -uvc.y, -uvc.x); }
        case 1u: { dir = vec3f(-1.0,  -uvc.y,  uvc.x); }
        case 2u: { dir = vec3f( uvc.x,  1.0,   uvc.y); }
        case 3u: { dir = vec3f( uvc.x, -1.0,  -uvc.y); }
        case 4u: { dir = vec3f( uvc.x, -uvc.y,  1.0);  }
        case 5u: { dir = vec3f(-uvc.x, -uvc.y, -1.0);  }
        default: { dir = vec3f(0.0); }
    }
    return normalize(dir);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(output);
    let slice_index = global_id.z;

    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y || slice_index >= 6u) {
        return;
    }

    let uv = vec2<f32>(
        (f32(global_id.x) + 0.5) / f32(dimensions.x),
        (f32(global_id.y) + 0.5) / f32(dimensions.y)
    );

    var ray_dir_ws = sample_cube_dir(uv, slice_index);

    // Cube maps are left-handed.
    ray_dir_ws.z = -ray_dir_ws.z;

    let world_pos = get_view_position();
    let r = length(world_pos);

    let ray_dir_as = direction_world_to_atmosphere(ray_dir_ws.xyz);
    let inscattering = sample_sky_view_lut(r, ray_dir_as);
    let color = vec4<f32>(inscattering, 1.0);

    textureStore(output, vec2<i32>(global_id.xy), i32(slice_index), color);
}
