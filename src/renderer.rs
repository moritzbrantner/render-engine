//! The wgpu renderer: one shadow pass per sun cascade and one forward PBR pass with 4× MSAA.

use crate::{
    mesh::{Geometry, Vertex},
    scene::{MAX_CASCADES, Scene, ShadowFilter},
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

/// MSAA sample count of the main pass (Bevy's default `Msaa::Sample4`).
pub const SAMPLE_COUNT: u32 = 4;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Stride of the per-cascade shadow-view uniforms (dynamic-offset alignment).
const CASCADE_STRIDE: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    clip_from_world: Mat4,
    view_from_world: Mat4,
    cascade_clip_from_world: [Mat4; MAX_CASCADES],
    /// Per cascade: x far bound (m), y texel size (m).
    cascade_params: [Vec4; MAX_CASCADES],
    eye: Vec4,
    direction_to_light: Vec4,
    sun_color: Vec4,
    ambient_color: Vec4,
    /// x depth bias (m), y normal bias (texels), z overlap proportion, w cascade count.
    shadow: Vec4,
    /// x exposure, y 1 when the sun casts shadows, z 1 when there is a sun, w 1 for Gaussian
    /// filtering.
    flags: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceData {
    world_from_local: Mat4,
    // mat3x3<f32> in WGSL: three columns padded to 16 bytes.
    normal_from_local: [Vec4; 3],
    base_color: Vec4,
    pbr: Vec4,
}

/// Fixed GPU state: pipelines, layouts and samplers.
pub struct Renderer {
    color_format: wgpu::TextureFormat,
    frame_layout: wgpu::BindGroupLayout,
    cascade_layout: wgpu::BindGroupLayout,
    shadow_layout: wgpu::BindGroupLayout,
    shadow_pipeline: wgpu::RenderPipeline,
    lit_pipeline: wgpu::RenderPipeline,
    shadow_sampler: wgpu::Sampler,
}

/// A scene uploaded for one target size; render it any number of times.
pub struct PreparedScene {
    width: u32,
    height: u32,
    clear_color: wgpu::Color,
    frame_group: wgpu::BindGroup,
    shadow_group: wgpu::BindGroup,
    cascade_group: wgpu::BindGroup,
    /// One depth view per cascade (render attachments); empty without sun shadows.
    cascade_views: Vec<wgpu::TextureView>,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    /// (first index, index count, base vertex) per instance, in instance order.
    draws: Vec<(u32, u32, i32)>,
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

/// Where a frame writes GPU timestamps: index 0 at the start of the first pass, index 1 at the
/// end of the last.
pub struct FrameTimestamps<'a> {
    pub query_set: &'a wgpu::QuerySet,
}

impl Renderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/lit.wgsl"));
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sun shadow"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });
        // The shadow pass picks its cascade's matrix through a dynamic offset.
        let cascade_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow cascade"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            }],
        });
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        };
        let primitive = wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        };
        // Reverse Z everywhere: nearer is larger.
        let depth = |format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sun shadow"),
                bind_group_layouts: &[Some(&frame_layout), Some(&cascade_layout)],
                immediate_size: 0,
            });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sun shadow"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("shadow_vs"),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout.clone())],
            },
            primitive,
            depth_stencil: Some(depth(DEPTH_FORMAT)),
            multisample: wgpu::MultisampleState::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });

        let lit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lit"),
            bind_group_layouts: &[Some(&frame_layout), Some(&shadow_layout)],
            immediate_size: 0,
        });
        let lit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lit"),
            layout: Some(&lit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("lit_vs"),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout)],
            },
            primitive,
            depth_stencil: Some(depth(DEPTH_FORMAT)),
            multisample: wgpu::MultisampleState {
                count: SAMPLE_COUNT,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("lit_fs"),
                compilation_options: Default::default(),
                targets: &[Some(color_format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });

        // Hardware 2×2 PCF; the Gaussian filter combines 9 of these taps.
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sun shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::GreaterEqual),
            ..Default::default()
        });

        Self {
            color_format,
            frame_layout,
            cascade_layout,
            shadow_layout,
            shadow_pipeline,
            lit_pipeline,
            shadow_sampler,
        }
    }

    /// Uploads `scene` for a `width`×`height` target.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        scene: &Scene,
        width: u32,
        height: u32,
    ) -> PreparedScene {
        let aspect = width as f32 / height as f32;
        let cam = &scene.camera;
        let view_from_world = look_at(cam.eye, cam.target);
        let clip_from_view = perspective_infinite_reverse(cam.fov_y, aspect, cam.near);

        let shadows = scene.sun.and_then(|s| s.shadows.map(|cfg| (s, cfg)));
        let world_from_view = view_from_world.inverse();
        let cascades: Vec<(f32, Cascade)> = shadows.map_or_else(Vec::new, |(sun, cfg)| {
            cfg.cascade_ranges()
                .into_iter()
                .map(|(near, far)| {
                    let c = Cascade::fit(
                        near,
                        far,
                        cfg.map_size,
                        sun.direction,
                        cam.fov_y,
                        aspect,
                        world_from_view,
                    );
                    (far, c)
                })
                .collect()
        });
        let mut cascade_clip_from_world = [Mat4::IDENTITY; MAX_CASCADES];
        let mut cascade_params = [Vec4::ZERO; MAX_CASCADES];
        for (i, (far, c)) in cascades.iter().enumerate() {
            cascade_clip_from_world[i] = c.clip_from_world;
            cascade_params[i] = Vec4::new(*far, c.texel_size, 0.0, 0.0);
        }

        let sun = scene.sun;
        let frame = FrameUniform {
            clip_from_world: clip_from_view * view_from_world,
            view_from_world,
            cascade_clip_from_world,
            cascade_params,
            eye: cam.eye.extend(1.0),
            direction_to_light: sun.map_or(Vec4::ZERO, |s| (-s.direction.normalize()).extend(0.0)),
            sun_color: sun.map_or(Vec4::ZERO, |s| (s.color * s.illuminance).extend(1.0)),
            ambient_color: (scene.ambient.color * scene.ambient.brightness).extend(1.0),
            shadow: shadows.map_or(Vec4::ZERO, |(_, cfg)| {
                Vec4::new(
                    cfg.depth_bias,
                    // Bevy scales the normal bias by √2 for the worst-case diagonal texel offset.
                    cfg.normal_bias * std::f32::consts::SQRT_2,
                    cfg.overlap_proportion,
                    cascades.len() as f32,
                )
            }),
            flags: Vec4::new(
                cam.exposure(),
                f32::from(u8::from(shadows.is_some())),
                f32::from(u8::from(sun.is_some())),
                f32::from(u8::from(cam.shadow_filter == ShadowFilter::Gaussian)),
            ),
        };

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut draws = Vec::new();
        let mut instances = Vec::new();
        for mesh in &scene.meshes {
            let g = Geometry::of(mesh.shape);
            draws.push((
                indices.len() as u32,
                g.indices.len() as u32,
                vertices.len() as i32,
            ));
            vertices.extend(g.vertices);
            indices.extend(g.indices);
            let world_from_local =
                Mat4::from_scale_rotation_translation(mesh.scale, mesh.rotation, mesh.translation);
            let normal = Mat3::from_mat4(world_from_local).inverse().transpose();
            let m = mesh.material;
            instances.push(InstanceData {
                world_from_local,
                normal_from_local: [
                    normal.x_axis.extend(0.0),
                    normal.y_axis.extend(0.0),
                    normal.z_axis.extend(0.0),
                ],
                base_color: m.base_color.extend(1.0),
                pbr: Vec4::new(m.metallic, m.perceptual_roughness, m.reflectance, 0.0),
            });
        }
        if instances.is_empty() {
            instances.push(InstanceData::zeroed());
        }

        let frame_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("frame"),
            contents: bytemuck::bytes_of(&frame),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let frame_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame"),
            layout: &self.frame_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instance_buffer.as_entire_binding(),
                },
            ],
        });

        let shadow_size = shadows.map_or(1, |(_, cfg)| cfg.map_size);
        let layers = cascades.len().max(1) as u32;
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("sun shadow maps"),
            size: wgpu::Extent3d {
                width: shadow_size,
                height: shadow_size,
                depth_or_array_layers: layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let cascade_views = (0..cascades.len() as u32)
            .map(|layer| {
                shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let mut cascade_bytes = vec![0u8; CASCADE_STRIDE as usize * MAX_CASCADES];
        for (i, m) in cascade_clip_from_world.iter().enumerate() {
            let at = i * CASCADE_STRIDE as usize;
            cascade_bytes[at..at + 64].copy_from_slice(bytemuck::bytes_of(m));
        }
        let cascade_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shadow cascades"),
            contents: &cascade_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let cascade_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow cascade"),
            layout: &self.cascade_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &cascade_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(64),
                }),
            }],
        });
        let shadow_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sun shadow"),
            layout: &self.shadow_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                },
            ],
        });

        let target = |format, label| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: SAMPLE_COUNT,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };

        let c = scene.clear_color;
        PreparedScene {
            width,
            height,
            clear_color: wgpu::Color {
                r: c.x.into(),
                g: c.y.into(),
                b: c.z.into(),
                a: 1.0,
            },
            frame_group,
            shadow_group,
            cascade_group,
            cascade_views,
            vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("indices"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            draws,
            msaa_view: target(self.color_format, "msaa colour"),
            depth_view: target(DEPTH_FORMAT, "depth"),
        }
    }

    /// Records one frame that resolves into `target` (a `width`×`height` view of the colour
    /// format the renderer was created with).
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scene: &PreparedScene,
        target: &wgpu::TextureView,
        timestamps: Option<FrameTimestamps<'_>>,
    ) {
        let draw = |pass: &mut wgpu::RenderPass<'_>| {
            pass.set_bind_group(0, &scene.frame_group, &[]);
            pass.set_vertex_buffer(0, scene.vertices.slice(..));
            pass.set_index_buffer(scene.indices.slice(..), wgpu::IndexFormat::Uint32);
            for (instance, &(first, count, base)) in scene.draws.iter().enumerate() {
                let instance = instance as u32;
                pass.draw_indexed(first..first + count, base, instance..instance + 1);
            }
        };
        let query_set = timestamps.map(|t| t.query_set);

        for (i, shadow_view) in scene.cascade_views.iter().enumerate() {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sun shadow cascade"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                // The frame starts at the first cascade's pass.
                timestamp_writes: query_set.filter(|_| i == 0).map(|query_set| {
                    wgpu::RenderPassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: None,
                    }
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(1, &scene.cascade_group, &[i as u32 * CASCADE_STRIDE as u32]);
            draw(&mut pass);
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &scene.msaa_view,
                depth_slice: None,
                resolve_target: Some(target),
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(scene.clear_color),
                    store: wgpu::StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &scene.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: query_set.map(|query_set| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: scene.cascade_views.is_empty().then_some(0),
                end_of_pass_write_index: Some(1),
            }),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.lit_pipeline);
        pass.set_bind_group(1, &scene.shadow_group, &[]);
        draw(&mut pass);
    }
}

impl PreparedScene {
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Right-handed view matrix with +Y up (the inverse of Bevy `Transform::looking_at(target, Y)`).
pub fn look_at(eye: Vec3, target: Vec3) -> Mat4 {
    let back = (eye - target).normalize();
    let right = Vec3::Y.cross(back).normalize();
    let up = back.cross(right);
    Mat4::from_cols(
        right.extend(0.0),
        up.extend(0.0),
        back.extend(0.0),
        eye.extend(1.0),
    )
    .inverse()
}

/// Bevy's right-handed infinite reverse-Z perspective projection.
fn perspective_infinite_reverse(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let h = 1.0 / (0.5 * fov_y).tan();
    Mat4::from_cols(
        Vec4::new(h / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, h, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, -1.0),
        Vec4::new(0.0, 0.0, near, 0.0),
    )
}

/// One sun shadow cascade, fitted to the camera frustum slice `near..far` (view distances)
/// exactly like Bevy's `calculate_cascade` (diameter rounded up to whole metres, centre snapped
/// to texels).
#[derive(Clone, Copy, Debug)]
pub struct Cascade {
    pub clip_from_world: Mat4,
    pub texel_size: f32,
}

impl Cascade {
    pub fn fit(
        near: f32,
        far: f32,
        map_size: u32,
        sun_direction: Vec3,
        fov_y: f32,
        aspect: f32,
        world_from_view: Mat4,
    ) -> Self {
        // Light orientation of Bevy `Transform::looking_to(direction, Vec3::Y)`.
        let back = -sun_direction.normalize();
        let right = Vec3::Y.cross(back).normalize();
        let up = back.cross(right);
        let world_from_light = Mat4::from_mat3(Mat3::from_cols(right, up, back));
        let light_from_view = world_from_light.transpose() * world_from_view;

        let t = (fov_y * 0.5).tan();
        let corners = [-near, -far].map(|z| {
            let a = z.abs() * t;
            [
                Vec3::new(a * aspect, -a, z),
                Vec3::new(a * aspect, a, z),
                Vec3::new(-a * aspect, a, z),
                Vec3::new(-a * aspect, -a, z),
            ]
        });
        let corners: [Vec3; 8] = [corners[0], corners[1]].concat().try_into().unwrap();
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for c in corners {
            let p = light_from_view.transform_point3(c);
            min = min.min(p);
            max = max.max(p);
        }
        let body = (corners[0] - corners[6]).length_squared();
        let far = (corners[4] - corners[6]).length_squared();
        let diameter = body.max(far).sqrt().ceil();
        let texel_size = diameter / map_size as f32;
        let centre = Vec3::new(
            (0.5 * (min.x + max.x) / texel_size).floor() * texel_size,
            (0.5 * (min.y + max.y) / texel_size).floor() * texel_size,
            max.z,
        );
        let t = world_from_light.transpose();
        let cascade_from_world =
            Mat4::from_cols(t.x_axis, t.y_axis, t.z_axis, (-centre).extend(1.0));
        let r = (max.z - min.z).recip();
        let clip_from_cascade = Mat4::from_cols(
            Vec4::new(2.0 / diameter, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 / diameter, 0.0, 0.0),
            Vec4::new(0.0, 0.0, r, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 1.0),
        );
        Self {
            clip_from_world: clip_from_cascade * cascade_from_world,
            texel_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascade_covers_the_view_slice() {
        let eye = Vec3::new(6.0, 5.0, 8.0);
        let world_from_view = look_at(eye, Vec3::ZERO).inverse();
        let c = Cascade::fit(
            0.1,
            30.0,
            2048,
            Vec3::new(-0.4, -1.0, -0.3),
            45f32.to_radians(),
            16.0 / 9.0,
            world_from_view,
        );
        // A point the camera looks at, inside the shadow range, lands inside the cascade.
        let p = c.clip_from_world.project_point3(Vec3::ZERO);
        assert!(
            p.x.abs() < 1.0 && p.y.abs() < 1.0 && (0.0..=1.0).contains(&p.z),
            "{p}"
        );
        assert!(c.texel_size > 0.0 && c.texel_size < 0.05);
    }

    #[test]
    fn default_cascades_follow_bevy_spacing() {
        let ranges = crate::scene::SunShadows::default().cascade_ranges();
        let expected = [(0.1, 10.0), (8.0, 24.662), (19.73, 60.822), (48.66, 150.0)];
        assert_eq!(ranges.len(), expected.len());
        for ((near, far), (en, ef)) in ranges.into_iter().zip(expected) {
            assert!(
                (near - en).abs() < 0.01 && (far - ef).abs() < 0.01,
                "{near}..{far}"
            );
        }
    }
}
