//! GPU data uploads whose destination must not be mapped at creation. Browser WebGPU
//! implementations can reject mapped destination buffers even for small resources.

/// Creates a destination buffer and transfers its initial contents through the queue.
/// The streams and uniforms using this helper are 4-byte aligned for `write_buffer`.
pub(crate) fn buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: contents.len() as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !contents.is_empty() {
        queue.write_buffer(&buffer, 0, contents);
    }
    buffer
}
