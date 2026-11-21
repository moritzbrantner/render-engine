//! Buffer readbacks for the diagnostics (`World::read_alive_particles`,
//! `Renderer::read_gpu_culled`, `Renderer::read_cluster_diagnostic`), as futures so they work on
//! the browser's WebGPU, where a mapping completes only after the calling task yields (#129).
//! Natively the future polls the device until the mapping completes, so `pollster::block_on`
//! (or any executor) finishes it at once; in the browser the browser completes it.

use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

/// Copies `size` bytes of `source` from its start into a mappable buffer, submits the copy and
/// resolves to the bytes once the GPU has written them; `None` if the mapping failed (a lost
/// device).
pub(crate) async fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
    size: u64,
    label: &'static str,
) -> Option<Vec<u8>> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    let mapped = Mapped::default();
    let done = mapped.clone();
    slice.map_async(wgpu::MapMode::Read, move |r| done.complete(r.is_ok()));
    #[cfg(not(target_arch = "wasm32"))]
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    if !mapped.await {
        return None;
    }
    let bytes = slice.get_mapped_range().expect("mapped").to_vec();
    staging.unmap();
    Some(bytes)
}

/// Resolves to whether a `map_async` succeeded, once its callback has run.
#[derive(Clone, Default)]
struct Mapped(Arc<Mutex<(Option<bool>, Option<Waker>)>>);

impl Mapped {
    fn complete(&self, ok: bool) {
        let mut state = self.0.lock().expect("readback state");
        state.0 = Some(ok);
        if let Some(waker) = state.1.take() {
            waker.wake();
        }
    }
}

impl Future for Mapped {
    type Output = bool;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<bool> {
        let mut state = self.0.lock().expect("readback state");
        match state.0 {
            Some(ok) => Poll::Ready(ok),
            None => {
                state.1 = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}
