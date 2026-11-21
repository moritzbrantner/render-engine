//! GPU data uploads whose destination must not be mapped at creation. Browser WebGPU
//! implementations can reject mapped destination buffers even for small resources: SwiftShader
//! fails `mappedAtCreation` after many small allocations (#116), so every buffer the engine
//! fills from consumer or load-time data goes through [`buffer`].

/// Creates an unmapped destination buffer and transfers its initial contents through the queue.
/// The size is padded to `COPY_BUFFER_ALIGNMENT` (the padding is zero), so any byte slice works
/// with `write_buffer`.
pub(crate) fn buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let size = padded(contents.len() as u64);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if size == contents.len() as u64 {
        if !contents.is_empty() {
            queue.write_buffer(&buffer, 0, contents);
        }
    } else {
        let mut bytes = contents.to_vec();
        bytes.resize(size as usize, 0);
        queue.write_buffer(&buffer, 0, &bytes);
    }
    buffer
}

fn padded(len: u64) -> u64 {
    len.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
}

#[cfg(test)]
mod tests {
    #[test]
    fn sizes_pad_to_the_copy_alignment() {
        assert_eq!(super::padded(0), 0);
        assert_eq!(super::padded(1), 4);
        assert_eq!(super::padded(4), 4);
        assert_eq!(super::padded(5952), 5952);
        assert_eq!(super::padded(5953), 5956);
    }
}
