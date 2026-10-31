//! Creates the renderer's fixed GPU state: bind group layouts, pipelines and samplers.

use super::{
    DEPTH_FORMAT, HDR_FORMAT, Renderer, SAMPLE_COUNT,
    textures::{load_lut, upload_cube},
    viewport::Phase,
};
use crate::{
    scene::{Cubemap, Tonemapping},
    sky::SkyPipelines,
    world::Vertex,
};
use wgpu::util::DeviceExt;

const TONY_MC_MAPFACE_LUT: &[u8] = include_bytes!("../luts/tony_mc_mapface.ktx2");
const AGX_LUT: &[u8] = include_bytes!("../luts/AgX-default_contrast.ktx2");

/// Which faces a pipeline rasterises: bit 0 winds mirrored instances clockwise, bit 1 draws back
/// faces (double-sided materials).
pub(super) const FACE_VARIANTS: usize = 4;

pub(super) fn face_variant(mirrored: bool, double_sided: bool) -> usize {
    usize::from(mirrored) | usize::from(double_sided) << 1
}

/// Index into `Renderer::shadow_pipelines`, in the order `Renderer::new` creates them.
pub(super) fn shadow_pipeline_index(local: bool, face: usize, masked: bool) -> usize {
    (usize::from(local) * FACE_VARIANTS + face) * 2 + usize::from(masked)
}

fn primitive(face: usize) -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        front_face: if face & 1 != 0 {
            wgpu::FrontFace::Cw
        } else {
            wgpu::FrontFace::Ccw
        },
        cull_mode: (face & 2 == 0).then_some(wgpu::Face::Back),
        ..Default::default()
    }
}

/// The tone-mapping shader's `METHOD` constant.
pub(super) fn tonemap_index(t: Tonemapping) -> usize {
    match t {
        Tonemapping::AcesFitted => 0,
        Tonemapping::TonyMcMapface => 1,
        Tonemapping::AgX => 2,
    }
}

impl Renderer {
    /// Creates the pipelines on the consumer's device for frames written to `color_format` (tone
    /// mapped; an sRGB format encodes the output) and uploads the tone-mapping LUTs. Unclipped
    /// sun shadow depth uses `DEPTH_CLIP_CONTROL` where the device has it enabled; the sky
    /// blends with `DUAL_SOURCE_BLENDING` where enabled (as Bevy does) and needs
    /// `FLOAT32_FILTERABLE`.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/lit.wgsl"));
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
        // Main pass: lights, shadow maps and the environment map.
        let lighting_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lighting"),
            entries: &[
                cube(3),
                cube(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::CubeArray,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // A material: its uniform, then five texture/sampler pairs (base colour, emissive,
        // metallic-roughness, occlusion, normal map).
        let mut material_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for i in 0..5 {
            material_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 1 + 2 * i,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
            material_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2 + 2 * i,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &material_entries,
        });
        // A shadow pass picks its view's matrix through a dynamic offset.
        let shadow_view_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow view"),
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
            attributes: &Vertex::ATTRIBUTES,
        };
        // Reverse Z everywhere: nearer is larger.
        let depth = |write: bool| wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(write),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        };

        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("shadow"),
                bind_group_layouts: &[
                    Some(&frame_layout),
                    Some(&shadow_view_layout),
                    Some(&material_layout),
                ],
                immediate_size: 0,
            });
        // Unclipped depth like Bevy's directional shadow views: natively where the device has
        // `DEPTH_CLIP_CONTROL`, else clamped in a fragment shader (see `lit.wgsl`).
        let unclipped_depth = device
            .features()
            .contains(wgpu::Features::DEPTH_CLIP_CONTROL);
        let mut shadow_pipelines = Vec::new();
        for local in [false, true] {
            for face in 0..FACE_VARIANTS {
                for masked in [false, true] {
                    let emulated = !local && !unclipped_depth;
                    let fragment = match (emulated, masked) {
                        (false, false) => None,
                        (false, true) => Some("shadow_masked_fs"),
                        (true, false) => Some("shadow_emulated_fs"),
                        (true, true) => Some("shadow_emulated_masked_fs"),
                    };
                    shadow_pipelines.push(device.create_render_pipeline(
                        &wgpu::RenderPipelineDescriptor {
                            label: Some(if local { "local shadow" } else { "sun shadow" }),
                            layout: Some(&shadow_pipeline_layout),
                            vertex: wgpu::VertexState {
                                module: &shader,
                                entry_point: Some(if emulated {
                                    "shadow_emulated_vs"
                                } else {
                                    "shadow_vs"
                                }),
                                compilation_options: Default::default(),
                                buffers: &[Some(vertex_layout.clone())],
                            },
                            primitive: wgpu::PrimitiveState {
                                unclipped_depth: !local && unclipped_depth,
                                ..primitive(face)
                            },
                            depth_stencil: Some(depth(true)),
                            multisample: wgpu::MultisampleState::default(),
                            fragment: fragment.map(|entry_point| wgpu::FragmentState {
                                module: &shader,
                                entry_point: Some(entry_point),
                                compilation_options: Default::default(),
                                targets: &[],
                            }),
                            multiview_mask: None,
                            cache: None,
                        },
                    ));
                }
            }
        }

        let lit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lit"),
            bind_group_layouts: &[
                Some(&frame_layout),
                Some(&lighting_layout),
                Some(&material_layout),
            ],
            immediate_size: 0,
        });
        let mut lit_pipelines = Vec::new();
        for face in 0..FACE_VARIANTS {
            for phase in [Phase::Opaque, Phase::Mask, Phase::Blend] {
                let blend = phase == Phase::Blend;
                lit_pipelines.push(device.create_render_pipeline(
                    &wgpu::RenderPipelineDescriptor {
                        label: Some("lit"),
                        layout: Some(&lit_pipeline_layout),
                        vertex: wgpu::VertexState {
                            module: &shader,
                            entry_point: Some("lit_vs"),
                            compilation_options: Default::default(),
                            buffers: &[Some(vertex_layout.clone())],
                        },
                        primitive: primitive(face),
                        // Blended fragments test depth but do not write it (Bevy).
                        depth_stencil: Some(depth(!blend)),
                        multisample: wgpu::MultisampleState {
                            count: SAMPLE_COUNT,
                            ..Default::default()
                        },
                        fragment: Some(wgpu::FragmentState {
                            module: &shader,
                            entry_point: Some(if phase == Phase::Mask {
                                "lit_masked_fs"
                            } else {
                                "lit_fs"
                            }),
                            compilation_options: Default::default(),
                            targets: &[Some(wgpu::ColorTargetState {
                                format: HDR_FORMAT,
                                blend: blend.then_some(wgpu::BlendState::ALPHA_BLENDING),
                                write_mask: wgpu::ColorWrites::ALL,
                            })],
                        }),
                        multiview_mask: None,
                        cache: None,
                    },
                ));
            }
        }

        let tonemap_shader =
            device.create_shader_module(wgpu::include_wgsl!("../shaders/tonemap.wgsl"));
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
                    targets: &[Some(HDR_FORMAT.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        });
        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("blit"),
                    bind_group_layouts: &[Some(&blit_layout)],
                    immediate_size: 0,
                }),
            ),
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
                entry_point: Some("blit_fs"),
                compilation_options: Default::default(),
                targets: &[Some(color_format.into())],
            }),
            multiview_mask: None,
            cache: None,
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
        let shadow_linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
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
            device: device.clone(),
            queue: queue.clone(),
            frame_layout,
            shadow_view_layout,
            lighting_layout,
            material_layout,
            tonemap_layout,
            blit_layout,
            blit_pipeline,
            shadow_pipelines,
            lit_pipelines,
            tonemap_pipelines,
            shadow_sampler,
            shadow_linear_sampler,
            environment_sampler,
            no_environment,
            tony_mc_mapface_lut: load_lut(device, queue, TONY_MC_MAPFACE_LUT),
            agx_lut: load_lut(device, queue, AGX_LUT),
            lut_sampler,
            sky: SkyPipelines::new(device, queue),
            no_transmittance: device
                .create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("no transmittance"),
                        size: wgpu::Extent3d::default(),
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    &[0; 8],
                )
                .create_view(&Default::default()),
        }
    }
}
