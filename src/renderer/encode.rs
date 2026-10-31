//! Records a prepared frame's passes into the consumer's encoder.

use super::{
    Renderer,
    pipelines::{shadow_pipeline_index, tonemap_index},
    viewport::{Draw, Phase, SHADOW_VIEW_STRIDE, Targets, Viewport},
};
use crate::world::World;
use std::ops::Range;

/// Where a frame writes GPU timestamps: index 0 at the start of the first pass, index 1 at the
/// end of the last.
pub struct FrameTimestamps<'a> {
    pub query_set: &'a wgpu::QuerySet,
}

impl Renderer {
    /// Records the frame last prepared into `viewport`, ending in `target`: a view of the colour
    /// format the renderer was created with, `viewport`-sized, fully overwritten. Draw the UI
    /// into `target` afterwards (`LoadOp::Load`).
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
        world: &World,
        target: &wgpu::TextureView,
        timestamps: Option<FrameTimestamps<'_>>,
    ) {
        let frame_group = viewport
            .frame_group
            .as_ref()
            .expect("Renderer::prepare before encode");
        let draw = |pass: &mut wgpu::RenderPass<'_>,
                    pipelines: &[wgpu::RenderPipeline],
                    phases: Range<Phase>,
                    index: &dyn Fn(&Draw) -> usize| {
            pass.set_bind_group(0, frame_group, &[]);
            let (mut mesh_bound, mut material_bound) = (None, None);
            for d in viewport.draws.iter().filter(|d| phases.contains(&d.phase)) {
                pass.set_pipeline(&pipelines[index(d)]);
                if material_bound != Some(d.material) {
                    pass.set_bind_group(2, &world.material(d.material).group, &[]);
                    material_bound = Some(d.material);
                }
                let mesh = world.mesh(d.mesh);
                if mesh_bound != Some(d.mesh) {
                    pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                    pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                    mesh_bound = Some(d.mesh);
                }
                pass.draw_indexed(0..mesh.index_count, 0, d.instances.clone());
            }
        };
        let query_set = timestamps.map(|t| t.query_set);
        // The frame's start timestamp goes to whichever pass comes first.
        let mut start = query_set;
        if let Some(sky) = &viewport.sky {
            sky.encode_luts(encoder, &self.sky, start.take());
        }
        let mut render_start = || {
            start
                .take()
                .map(|query_set| wgpu::RenderPassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: None,
                })
        };

        for shadow in &viewport.shadow_passes {
            let maps = if shadow.point {
                &viewport.point_shadows
            } else {
                &viewport.directional_shadows
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &maps.layer_views[shadow.layer as usize],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: render_start(),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(
                1,
                viewport.shadow_view_group.as_ref(),
                &[shadow.view * SHADOW_VIEW_STRIDE as u32],
            );
            draw(
                &mut pass,
                &self.shadow_pipelines,
                Phase::Opaque..Phase::End,
                &|d: &Draw| shadow_pipeline_index(shadow.local, d.face, d.phase != Phase::Opaque),
            );
        }

        // The main pass. With a sky it is split like Bevy's: opaque and alpha-masked draws, the
        // sky over them (reading their depth), then the blended draws.
        let targets = &viewport.targets;
        let part = |label, clear, depth, resolve| MainPass {
            label,
            clear,
            depth,
            resolve,
        };
        let clear_color = viewport.clear_color;
        let lit_index = |d: &Draw| d.face * 3 + d.phase as usize;
        match &viewport.sky {
            None => {
                let mut pass = begin_main_pass(
                    encoder,
                    targets,
                    clear_color,
                    part("lit", true, true, true),
                    render_start(),
                );
                pass.set_bind_group(1, viewport.lighting_group.as_ref(), &[]);
                draw(
                    &mut pass,
                    &self.lit_pipelines,
                    Phase::Opaque..Phase::End,
                    &lit_index,
                );
            }
            Some(sky) => {
                let mut pass = begin_main_pass(
                    encoder,
                    targets,
                    clear_color,
                    part("lit opaque", true, true, false),
                    render_start(),
                );
                pass.set_bind_group(1, viewport.lighting_group.as_ref(), &[]);
                draw(
                    &mut pass,
                    &self.lit_pipelines,
                    Phase::Opaque..Phase::Blend,
                    &lit_index,
                );
                drop(pass);
                let mut pass = begin_main_pass(
                    encoder,
                    targets,
                    clear_color,
                    part("render sky", false, false, false),
                    None,
                );
                sky.draw(&mut pass, &self.sky);
                drop(pass);
                let mut pass = begin_main_pass(
                    encoder,
                    targets,
                    clear_color,
                    part("lit transparent", false, true, true),
                    None,
                );
                pass.set_bind_group(1, viewport.lighting_group.as_ref(), &[]);
                draw(
                    &mut pass,
                    &self.lit_pipelines,
                    Phase::Blend..Phase::End,
                    &lit_index,
                );
            }
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tonemap"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.post_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.tonemap_pipelines[tonemap_index(viewport.tonemapping)]);
        pass.set_bind_group(0, viewport.tonemap.as_ref().map(|(_, g)| g), &[]);
        pass.draw(0..3, 0..1);
        drop(pass);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            // The frame ends at the blit.
            timestamp_writes: query_set.map(|query_set| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: Some(1),
            }),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, &targets.blit_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// How one part of the main pass treats the MSAA colour and depth targets.
struct MainPass {
    label: &'static str,
    /// Clear colour and depth (else load them).
    clear: bool,
    /// Attach the depth target.
    depth: bool,
    /// Resolve into the HDR frame and discard the MSAA targets (else store them for the next
    /// part).
    resolve: bool,
}

fn begin_main_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    targets: &'a Targets,
    clear_color: wgpu::Color,
    p: MainPass,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'a>>,
) -> wgpu::RenderPass<'a> {
    let store = if p.resolve {
        wgpu::StoreOp::Discard
    } else {
        wgpu::StoreOp::Store
    };
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(p.label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &targets.msaa_view,
            depth_slice: None,
            resolve_target: p.resolve.then_some(&targets.hdr_view),
            ops: wgpu::Operations {
                load: if p.clear {
                    wgpu::LoadOp::Clear(clear_color)
                } else {
                    wgpu::LoadOp::Load
                },
                store,
            },
        })],
        depth_stencil_attachment: p.depth.then_some(wgpu::RenderPassDepthStencilAttachment {
            view: &targets.depth_view,
            depth_ops: Some(wgpu::Operations {
                load: if p.clear {
                    wgpu::LoadOp::Clear(0.0)
                } else {
                    wgpu::LoadOp::Load
                },
                store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}
