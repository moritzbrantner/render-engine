//! The wgpu renderer: one shadow pass per sun cascade, one forward PBR pass with 4× MSAA into an
//! HDR target, and a tone-mapping pass into the output.

use crate::{
    mesh::{Geometry, Vertex},
    scene::{Cubemap, FogFalloff, MAX_CASCADES, Scene, ShadowFilter, Tonemapping},
};
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;

/// MSAA sample count of the main pass (Bevy's default `Msaa::Sample4`).
pub const SAMPLE_COUNT: u32 = 4;
/// Format of the main pass (Bevy's `ViewTarget::TEXTURE_FORMAT_HDR`).
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const TONY_MC_MAPFACE_LUT: &[u8] = include_bytes!("luts/tony_mc_mapface.ktx2");
const AGX_LUT: &[u8] = include_bytes!("luts/AgX-default_contrast.ktx2");
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
    /// x environment-map intensity (cd/m²), y smallest specular mip level, z 1 when there is an
    /// environment map.
    environment: Vec4,
    /// Distance fog (Bevy `GpuFog`): linear RGBA colour.
    fog_color: Vec4,
    /// Sun inscattering colour, alpha = strength (0 = off).
    fog_directional_light_color: Vec4,
    /// xyz: Bevy `be` (linear: start, end; exponential: density; atmospheric: extinction),
    /// w: directional-light exponent.
    fog_be: Vec4,
    /// xyz: Bevy `bi` (atmospheric inscattering), w: mode (0 off, 1 linear, 2 exponential,
    /// 3 exponential squared, 4 atmospheric).
    fog_bi: Vec4,
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
    frame_layout: wgpu::BindGroupLayout,
    cascade_layout: wgpu::BindGroupLayout,
    lighting_layout: wgpu::BindGroupLayout,
    tonemap_layout: wgpu::BindGroupLayout,
    shadow_pipeline: wgpu::RenderPipeline,
    lit_pipeline: wgpu::RenderPipeline,
    /// Indexed by [`tonemap_index`].
    tonemap_pipelines: [wgpu::RenderPipeline; 3],
    shadow_sampler: wgpu::Sampler,
    /// Bevy's default image sampler (trilinear, clamped), which Bevy uses for environment maps.
    environment_sampler: wgpu::Sampler,
    /// A black 1² cube bound when the scene has no environment map.
    no_environment: wgpu::TextureView,
    tony_mc_mapface_lut: wgpu::TextureView,
    agx_lut: wgpu::TextureView,
    lut_sampler: wgpu::Sampler,
}

/// The tone-mapping shader's `METHOD` constant.
fn tonemap_index(t: Tonemapping) -> usize {
    match t {
        Tonemapping::AcesFitted => 0,
        Tonemapping::TonyMcMapface => 1,
        Tonemapping::AgX => 2,
    }
}

/// A scene uploaded for one target size; render it any number of times.
pub struct PreparedScene {
    width: u32,
    height: u32,
    clear_color: wgpu::Color,
    frame_group: wgpu::BindGroup,
    lighting_group: wgpu::BindGroup,
    cascade_group: wgpu::BindGroup,
    /// One depth view per cascade (render attachments); empty without sun shadows.
    cascade_views: Vec<wgpu::TextureView>,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    /// (first index, index count, base vertex) per instance, in instance order.
    draws: Vec<(u32, u32, i32)>,
    msaa_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    /// The resolved HDR frame, read by the tone-mapping pass.
    hdr_view: wgpu::TextureView,
    tonemap_group: wgpu::BindGroup,
    tonemapping: Tonemapping,
}

/// Where a frame writes GPU timestamps: index 0 at the start of the first pass, index 1 at the
/// end of the last.
pub struct FrameTimestamps<'a> {
    pub query_set: &'a wgpu::QuerySet,
}

impl Renderer {
    /// Creates the pipelines for frames written to `color_format` (tone mapped; an sRGB format
    /// encodes the output) and uploads the tone-mapping LUTs.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
    ) -> Self {
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
        let cube = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::Cube,
                multisampled: false,
            },
            count: None,
        };
        // Main pass: sun shadow maps and the environment map.
        let lighting_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lighting"),
            entries: &[
                cube(3),
                cube(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
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
        // Unclipped depth like Bevy's directional shadow views: natively where the device has
        // `DEPTH_CLIP_CONTROL`, else clamped in a fragment shader (see `lit.wgsl`).
        let unclipped_depth = device
            .features()
            .contains(wgpu::Features::DEPTH_CLIP_CONTROL);
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sun shadow"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(if unclipped_depth {
                    "shadow_vs"
                } else {
                    "shadow_emulated_vs"
                }),
                compilation_options: Default::default(),
                buffers: &[Some(vertex_layout.clone())],
            },
            primitive: wgpu::PrimitiveState {
                unclipped_depth,
                ..primitive
            },
            depth_stencil: Some(depth(DEPTH_FORMAT)),
            multisample: wgpu::MultisampleState::default(),
            fragment: (!unclipped_depth).then(|| wgpu::FragmentState {
                module: &shader,
                entry_point: Some("shadow_emulated_fs"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            multiview_mask: None,
            cache: None,
        });

        let lit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lit"),
            bind_group_layouts: &[Some(&frame_layout), Some(&lighting_layout)],
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
                targets: &[Some(HDR_FORMAT.into())],
            }),
            multiview_mask: None,
            cache: None,
        });

        let tonemap_shader =
            device.create_shader_module(wgpu::include_wgsl!("shaders/tonemap.wgsl"));
        let tonemap_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tonemap"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let tonemap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tonemap"),
                bind_group_layouts: &[Some(&tonemap_layout)],
                immediate_size: 0,
            });
        let tonemap_pipelines = [0.0, 1.0, 2.0].map(|method| {
            let constants = [("METHOD", method)];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("tonemap"),
                layout: Some(&tonemap_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &tonemap_shader,
                    entry_point: Some("tonemap_vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &tonemap_shader,
                    entry_point: Some("tonemap_fs"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &[Some(color_format.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        // Bevy's tone-mapping LUT sampler: trilinear, clamped.
        let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tonemapping lut"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // Hardware 2×2 PCF; the Gaussian filter combines 9 of these taps.
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sun shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::GreaterEqual),
            ..Default::default()
        });

        // Bevy `ImageSamplerDescriptor::linear()`.
        let environment_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("environment map"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let no_environment = upload_cube(
            device,
            queue,
            &Cubemap {
                size: 1,
                format: wgpu::TextureFormat::Rgba16Float,
                levels: vec![vec![0; 6 * 8]],
            },
        );

        Self {
            frame_layout,
            cascade_layout,
            lighting_layout,
            tonemap_layout,
            shadow_pipeline,
            lit_pipeline,
            tonemap_pipelines,
            shadow_sampler,
            environment_sampler,
            no_environment,
            tony_mc_mapface_lut: load_lut(device, queue, TONY_MC_MAPFACE_LUT),
            agx_lut: load_lut(device, queue, AGX_LUT),
            lut_sampler,
        }
    }

    /// Uploads `scene` for a `width`×`height` target.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
            environment: scene.environment.as_ref().map_or(Vec4::ZERO, |e| {
                Vec4::new(e.intensity, (e.specular.levels.len() - 1) as f32, 1.0, 0.0)
            }),
            fog_color: scene.fog.map_or(Vec4::ZERO, |f| f.color),
            fog_directional_light_color: scene
                .fog
                .map_or(Vec4::ZERO, |f| f.directional_light_color),
            fog_be: scene.fog.map_or(Vec4::ZERO, |f| {
                let be = match f.falloff {
                    FogFalloff::Linear { start, end } => Vec3::new(start, end, 0.0),
                    FogFalloff::Exponential { density }
                    | FogFalloff::ExponentialSquared { density } => Vec3::new(density, 0.0, 0.0),
                    FogFalloff::Atmospheric { extinction, .. } => extinction,
                };
                be.extend(f.directional_light_exponent)
            }),
            fog_bi: scene.fog.map_or(Vec4::ZERO, |f| match f.falloff {
                FogFalloff::Linear { .. } => Vec4::new(0.0, 0.0, 0.0, 1.0),
                FogFalloff::Exponential { .. } => Vec4::new(0.0, 0.0, 0.0, 2.0),
                FogFalloff::ExponentialSquared { .. } => Vec4::new(0.0, 0.0, 0.0, 3.0),
                FogFalloff::Atmospheric { inscattering, .. } => inscattering.extend(4.0),
            }),
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
        let environment = scene.environment.as_ref().map(|e| {
            (
                upload_cube(device, queue, &e.diffuse),
                upload_cube(device, queue, &e.specular),
            )
        });
        let (diffuse_view, specular_view) = environment
            .as_ref()
            .map_or((&self.no_environment, &self.no_environment), |(d, s)| {
                (d, s)
            });
        let lighting_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting"),
            layout: &self.lighting_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(diffuse_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(specular_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.environment_sampler),
                },
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
        let hdr_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("hdr colour"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let tonemapping = cam.tonemapping;
        // Like Bevy, operators without a LUT still bind one (the small AgX LUT).
        let lut = match tonemapping {
            Tonemapping::TonyMcMapface => &self.tony_mc_mapface_lut,
            Tonemapping::AcesFitted | Tonemapping::AgX => &self.agx_lut,
        };
        let tonemap_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap"),
            layout: &self.tonemap_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(lut),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.lut_sampler),
                },
            ],
        });

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
            lighting_group,
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
            msaa_view: target(HDR_FORMAT, "msaa hdr colour"),
            depth_view: target(DEPTH_FORMAT, "depth"),
            hdr_view,
            tonemap_group,
            tonemapping,
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
                resolve_target: Some(&scene.hdr_view),
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
            timestamp_writes: query_set.filter(|_| scene.cascade_views.is_empty()).map(
                |query_set| wgpu::RenderPassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: None,
                },
            ),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.lit_pipeline);
        pass.set_bind_group(1, &scene.lighting_group, &[]);
        draw(&mut pass);
        drop(pass);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tonemap"),
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
            // The frame ends at the tone-mapping pass.
            timestamp_writes: query_set.map(|query_set| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: None,
                end_of_pass_write_index: Some(1),
            }),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.tonemap_pipelines[tonemap_index(scene.tonemapping)]);
        pass.set_bind_group(0, &scene.tonemap_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

impl PreparedScene {
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Uploads one of Bevy's tone-mapping LUTs: a zstd-supercompressed single-level 3D KTX2 file.
fn load_lut(device: &wgpu::Device, queue: &wgpu::Queue, ktx2: &[u8]) -> wgpu::TextureView {
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
fn upload_cube(device: &wgpu::Device, queue: &wgpu::Queue, cube: &Cubemap) -> wgpu::TextureView {
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
