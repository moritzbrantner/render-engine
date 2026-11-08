//! One-shot texture uploads: tone-mapping LUTs and environment cube maps.

use crate::scene::Cubemap;
use wgpu::util::DeviceExt;

/// Uploads one of Bevy's tone-mapping LUTs: a zstd-supercompressed single-level 3D KTX2 file.
pub(super) fn load_lut(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    ktx2: &[u8],
) -> wgpu::TextureView {
    let lut = crate::ktx::decode(ktx2).expect("embedded LUT is a supported KTX2 file");
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("tonemapping lut"),
            size: wgpu::Extent3d {
                width: lut.width,
                height: lut.height,
                depth_or_array_layers: lut.depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: lut.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &lut.levels[0],
    );
    texture.create_view(&Default::default())
}

/// Uploads a cube map with all its mip levels.
pub(super) fn upload_cube(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cube: &Cubemap,
) -> wgpu::TextureView {
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("environment map"),
            size: wgpu::Extent3d {
                width: cube.size,
                height: cube.size,
                depth_or_array_layers: 6,
            },
            mip_level_count: cube.levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: cube.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::MipMajor,
        &cube.levels.concat(),
    );
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    })
}
