//! The sky passes: Bevy 0.20's atmosphere (`bevy_pbr::atmosphere`, Hillaire's LUT method) and
//! the runtime environment-map filter its `AtmosphereEnvironmentMapLight` drives
//! (`GeneratedEnvironmentMapLight`).
//!
//! Each frame, before the shadow and main passes: the transmittance, multiscattering, sky-view
//! and aerial-view LUTs (one compute pass), then, with sky ambient light, the sky rendered into
//! a 128² cube, its mip chain, the GGX-prefiltered specular cube (one dispatch per mip) and the
//! 32² irradiance cube. After the opaque draws a fullscreen pass blends the sky and the aerial
//! perspective into the HDR frame; the main pass reads the transmittance LUT to filter the
//! sun and moon light.

use crate::scene::{Camera, Scene, Sky};
use bytemuck::{Pod, Zeroable};
use glam::{Affine3A, Mat4, Quat, UVec2, Vec3, Vec3A, Vec4};

/// Bevy `Atmosphere::earth`.
pub(crate) const INNER_RADIUS: f32 = 6_360_000.0;
pub(crate) const OUTER_RADIUS: f32 = 6_460_000.0;
/// Bevy `AtmosphereSettings::default()`.
const TRANSMITTANCE_SIZE: UVec2 = UVec2::new(256, 128);
const MULTISCATTERING_SIZE: UVec2 = UVec2::new(32, 32);
const SKY_VIEW_SIZE: UVec2 = UVec2::new(400, 200);
const AERIAL_VIEW_SIZE: [u32; 3] = [32, 32, 32];
/// Bevy `AtmosphereEnvironmentMapLight::default().size`, and the diffuse cube of
/// `GeneratedEnvironmentMapLight`.
const ENVIRONMENT_SIZE: u32 = 128;
const ENVIRONMENT_MIPS: u32 = ENVIRONMENT_SIZE.ilog2() + 1;
const IRRADIANCE_SIZE: u32 = 32;
/// `ScatteringMedium::earth(256, 256)`.
const FALLOFF_RESOLUTION: u32 = 256;
const PHASE_RESOLUTION: u32 = 256;
/// Byte offsets of the uniforms in the sky buffer (the largest allowed uniform offset
/// alignment).
const UNIFORM_STRIDE: u64 = 256;
const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuAtmosphere {
    ground_albedo: Vec3,
    inner_radius: f32,
    outer_radius: f32,
    _pad: [f32; 3],
    world_to_atmosphere: Mat4,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSettings {
    transmittance_lut_size: UVec2,
    multiscattering_lut_size: UVec2,
    sky_view_lut_size: UVec2,
    transmittance_lut_samples: u32,
    multiscattering_lut_dirs: u32,
    aerial_view_lut_size: [u32; 3],
    multiscattering_lut_samples: u32,
    sky_view_lut_samples: u32,
    aerial_view_lut_samples: u32,
    aerial_view_lut_max_distance: f32,
    sky_max_samples: u32,
    rendering_method: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTransforms {
    world_from_atmosphere: Mat4,
    atmosphere_from_world: Mat4,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuView {
    world_from_view: Mat4,
    view_from_clip: Mat4,
    world_position: Vec3,
    exposure: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuDirectionalLight {
    color: Vec4,
    direction_to_light: Vec3,
    sun_disk_angular_size: f32,
    sun_disk_intensity: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLights {
    directional_lights: [GpuDirectionalLight; 2],
    n_directional_lights: u32,
    _pad: [u32; 3],
}

/// Bevy `FilteringConstants`, one per dispatch of the environment filter.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FilteringConstants {
    mip_level: f32,
    sample_count: u32,
    roughness: f32,
    _pad: u32,
    noise_size_bits: UVec2,
    _pad2: UVec2,
}

/// A directional light as the sky passes and the main pass see it, in Bevy's light order.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SkyLight {
    pub direction_to_light: Vec3,
    /// Colour × illuminance.
    pub color: Vec3,
    pub disk_size: f32,
    pub disk_intensity: f32,
}

/// The scene's directional lights in Bevy's order (`prepare_lights` sorts by
/// `shadow_maps_enabled`, then by entity, and the oracle spawns the sun before the moon):
/// unshadowed lights first. Returns the lights and whether the moon comes first.
pub(crate) fn directional_lights(scene: &Scene) -> (Vec<SkyLight>, bool) {
    let disks = scene.sky.map(|s| (s.sun_disk, s.moon_disk));
    let sun = scene.sun.map(|s| SkyLight {
        direction_to_light: -s.direction.normalize(),
        color: s.color * s.illuminance,
        disk_size: disks.map_or(0.0, |d| d.0.angular_size),
        disk_intensity: disks.map_or(0.0, |d| d.0.intensity),
    });
    let moon = scene.moon.map(|m| SkyLight {
        direction_to_light: -m.direction.normalize(),
        color: m.color * m.illuminance,
        disk_size: disks.map_or(0.0, |d| d.1.angular_size),
        disk_intensity: disks.map_or(0.0, |d| d.1.intensity),
    });
    let moon_first = scene.sun.is_some_and(|s| s.shadows.is_some());
    let ordered = if moon_first { [moon, sun] } else { [sun, moon] };
    (ordered.into_iter().flatten().collect(), moon_first)
}

/// Bevy `extract_atmosphere`: the planet centre `INNER_RADIUS` below the origin, unscaled.
pub(crate) fn world_to_atmosphere() -> Mat4 {
    Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, -Vec3::Y * INNER_RADIUS)
        .inverse()
}

/// Bevy's `ScatteringMedium::earth(256, 256)` LUTs (`GpuScatteringMedium::prepare_asset`):
/// absorption and scattering density per falloff parameter, and scattering × phase per falloff
/// and (non-linearly mapped) `neg_LdotV`.
fn earth_medium() -> (Vec<Vec4>, Vec<Vec4>) {
    enum Falloff {
        Exponential(f32),
        Tent { center: f32, width: f32 },
    }
    enum Phase {
        Rayleigh,
        Mie(f32),
        Isotropic,
    }
    struct Term {
        absorption: Vec3,
        scattering: Vec3,
        falloff: Falloff,
        phase: Phase,
    }
    let terms = [
        Term {
            absorption: Vec3::ZERO,
            scattering: Vec3::new(5.802e-6, 13.558e-6, 33.100e-6),
            falloff: Falloff::Exponential(8.0 / 60.0),
            phase: Phase::Rayleigh,
        },
        Term {
            absorption: Vec3::splat(3.996e-6),
            scattering: Vec3::splat(0.444e-6),
            falloff: Falloff::Exponential(1.2 / 60.0),
            phase: Phase::Mie(0.8),
        },
        Term {
            absorption: Vec3::new(0.650e-6, 1.881e-6, 0.085e-6),
            scattering: Vec3::ZERO,
            falloff: Falloff::Tent {
                center: 0.75,
                width: 0.3,
            },
            phase: Phase::Isotropic,
        },
    ];
    let falloff = |f: &Falloff, p: f32| match *f {
        Falloff::Exponential(scale) => {
            let s = -1.0 / scale;
            let exp_p_s = ((1.0 - p) * s).exp();
            let exp_s = s.exp();
            (exp_p_s - exp_s) / (1.0 - exp_s)
        }
        Falloff::Tent { center, width } => (1.0 - (p - center).abs() / (0.5 * width)).max(0.0),
    };
    let phase = |p: &Phase, neg_l_dot_v: f32| {
        const FRAC_4_PI: f32 = 0.25 / std::f32::consts::PI;
        const FRAC_3_16_PI: f32 = 0.1875 / std::f32::consts::PI;
        match *p {
            Phase::Isotropic => FRAC_4_PI,
            Phase::Rayleigh => FRAC_3_16_PI * (1.0 + neg_l_dot_v * neg_l_dot_v),
            Phase::Mie(g) => {
                let denom = 1.0 + g * g - 2.0 * g * neg_l_dot_v;
                FRAC_4_PI * (1.0 - g * g) / (denom * denom.sqrt())
            }
        }
    };
    let n = FALLOFF_RESOLUTION;
    let p_at = |i: u32| (i as f32 + 0.5) / n as f32;
    let mut density: Vec<Vec4> = (0..n)
        .map(|i| {
            terms
                .iter()
                .map(|t| t.absorption.extend(0.0) * falloff(&t.falloff, p_at(i)))
                .sum()
        })
        .collect();
    density.extend((0..n).map(|i| {
        terms
            .iter()
            .map(|t| t.scattering.extend(0.0) * falloff(&t.falloff, p_at(i)))
            .sum::<Vec4>()
    }));
    let inv_n = 1.0 / 0.5f32;
    let scattering = (0..FALLOFF_RESOLUTION * PHASE_RESOLUTION)
        .map(|raw_i| {
            // Bevy indexes the falloff with `raw_i % phase_resolution`; both are 256 here.
            let i = raw_i % PHASE_RESOLUTION;
            let j = raw_i / PHASE_RESOLUTION;
            let f = (i as f32 + 0.5) / FALLOFF_RESOLUTION as f32;
            let ph = (j as f32 + 0.5) / PHASE_RESOLUTION as f32;
            let s = 2.0 * ph - 1.0;
            let neg_l_dot_v = s.signum() * (1.0 - (1.0 - s.abs()).powf(inv_n));
            terms
                .iter()
                .map(|t| {
                    (t.scattering * phase(&t.phase, neg_l_dot_v) * falloff(&t.falloff, f))
                        .extend(0.0)
                })
                .sum::<Vec4>()
        })
        .collect();
    (density, scattering)
}

#[derive(Clone, Copy)]
enum Entry {
    Uniform,
    Texture2d,
    Texture3d,
    Texture2dArray,
    Sampler,
    Storage2d,
    Storage3d,
    Storage2dArray,
    DepthMs,
}

fn layout(
    device: &wgpu::Device,
    label: &str,
    visibility: wgpu::ShaderStages,
    entries: &[(u32, Entry)],
) -> wgpu::BindGroupLayout {
    let float = wgpu::TextureSampleType::Float { filterable: true };
    let texture = |view_dimension, sample_type, multisampled| wgpu::BindingType::Texture {
        sample_type,
        view_dimension,
        multisampled,
    };
    let storage = |view_dimension| wgpu::BindingType::StorageTexture {
        access: wgpu::StorageTextureAccess::WriteOnly,
        format: LUT_FORMAT,
        view_dimension,
    };
    let entries: Vec<_> = entries
        .iter()
        .map(|&(binding, e)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: match e {
                Entry::Uniform => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                Entry::Texture2d => texture(wgpu::TextureViewDimension::D2, float, false),
                Entry::Texture3d => texture(wgpu::TextureViewDimension::D3, float, false),
                Entry::Texture2dArray => texture(wgpu::TextureViewDimension::D2Array, float, false),
                Entry::DepthMs => texture(
                    wgpu::TextureViewDimension::D2,
                    wgpu::TextureSampleType::Depth,
                    true,
                ),
                Entry::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                Entry::Storage2d => storage(wgpu::TextureViewDimension::D2),
                Entry::Storage3d => storage(wgpu::TextureViewDimension::D3),
                Entry::Storage2dArray => storage(wgpu::TextureViewDimension::D2Array),
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn tex(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn sampler(binding: u32, sampler: &wgpu::Sampler) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Sampler(sampler),
    }
}

const COMMON: &str = include_str!("shaders/sky/common.wgsl");

fn module(device: &wgpu::Device, label: &str, source: String) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

fn compute(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    module: &wgpu::ShaderModule,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            }),
        ),
        module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// The sky's pipelines, layouts, samplers and medium LUTs on the consumer's device.
pub(crate) struct SkyPipelines {
    transmittance_layout: wgpu::BindGroupLayout,
    multiscattering_layout: wgpu::BindGroupLayout,
    sky_view_layout: wgpu::BindGroupLayout,
    aerial_view_layout: wgpu::BindGroupLayout,
    environment_layout: wgpu::BindGroupLayout,
    render_layout: wgpu::BindGroupLayout,
    filter_layout: wgpu::BindGroupLayout,
    transmittance: wgpu::ComputePipeline,
    multiscattering: wgpu::ComputePipeline,
    sky_view: wgpu::ComputePipeline,
    aerial_view: wgpu::ComputePipeline,
    environment: wgpu::ComputePipeline,
    downsample: wgpu::ComputePipeline,
    radiance: wgpu::ComputePipeline,
    irradiance: wgpu::ComputePipeline,
    render: wgpu::RenderPipeline,
    /// Bevy `AtmosphereSampler`: bilinear, nearest mip, clamped. The main pass samples the
    /// transmittance LUT with it too.
    pub lut_sampler: wgpu::Sampler,
    /// Bevy `ScatteringMediumSampler`.
    medium_sampler: wgpu::Sampler,
    /// Bevy's `generator_linear_sampler`: trilinear, clamped.
    filter_sampler: wgpu::Sampler,
    /// `Rgba32Float` density and scattering LUTs; `None` without `FLOAT32_FILTERABLE`.
    medium: Option<(wgpu::TextureView, wgpu::TextureView)>,
}

impl SkyPipelines {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        use Entry::*;
        let compute_stage = wgpu::ShaderStages::COMPUTE;
        let medium_entries = [(5, Texture2d), (6, Texture2d), (7, Sampler)];
        let with = |base: &[(u32, Entry)], extra: &[(u32, Entry)]| -> Vec<(u32, Entry)> {
            base.iter().chain(extra).copied().collect()
        };
        // Bevy's `AtmosphereBindGroupLayouts`, `RenderSkyBindGroupLayouts` and the probe layout.
        let transmittance_layout = layout(
            device,
            "transmittance lut",
            compute_stage,
            &with(
                &[(0, Uniform), (1, Uniform)],
                &with(&medium_entries, &[(13, Storage2d)]),
            ),
        );
        let multiscattering_layout = layout(
            device,
            "multiscattering lut",
            compute_stage,
            &with(
                &[(0, Uniform), (1, Uniform)],
                &with(
                    &medium_entries,
                    &[(8, Texture2d), (12, Sampler), (13, Storage2d)],
                ),
            ),
        );
        let all_uniforms = [
            (0, Uniform),
            (1, Uniform),
            (2, Uniform),
            (3, Uniform),
            (4, Uniform),
        ];
        let sky_view_layout = layout(
            device,
            "sky view lut",
            compute_stage,
            &with(
                &all_uniforms,
                &with(
                    &medium_entries,
                    &[
                        (8, Texture2d),
                        (9, Texture2d),
                        (12, Sampler),
                        (13, Storage2d),
                    ],
                ),
            ),
        );
        let aerial_view_layout = layout(
            device,
            "aerial view lut",
            compute_stage,
            &with(
                &[(0, Uniform), (1, Uniform), (3, Uniform), (4, Uniform)],
                &with(
                    &medium_entries,
                    &[
                        (8, Texture2d),
                        (9, Texture2d),
                        (12, Sampler),
                        (13, Storage3d),
                    ],
                ),
            ),
        );
        let all_luts = [
            (8, Texture2d),
            (9, Texture2d),
            (10, Texture2d),
            (11, Texture3d),
            (12, Sampler),
        ];
        let environment_layout = layout(
            device,
            "sky environment",
            compute_stage,
            &with(&all_uniforms, &with(&all_luts, &[(13, Storage2dArray)])),
        );
        let render_layout = layout(
            device,
            "render sky",
            wgpu::ShaderStages::FRAGMENT,
            &with(
                &all_uniforms,
                &with(&medium_entries, &with(&all_luts, &[(13, DepthMs)])),
            ),
        );
        let filter_layout = layout(
            device,
            "environment filter",
            compute_stage,
            &[
                (0, Texture2dArray),
                (1, Sampler),
                (2, Storage2dArray),
                (3, Uniform),
            ],
        );

        let pass = |name: &str, source: &str| module(device, name, format!("{COMMON}\n{source}"));
        let transmittance = compute(
            device,
            "transmittance lut",
            &transmittance_layout,
            &pass(
                "transmittance lut",
                include_str!("shaders/sky/transmittance.wgsl"),
            ),
            "main",
        );
        let multiscattering = compute(
            device,
            "multiscattering lut",
            &multiscattering_layout,
            &pass(
                "multiscattering lut",
                include_str!("shaders/sky/multiscattering.wgsl"),
            ),
            "main",
        );
        let sky_view = compute(
            device,
            "sky view lut",
            &sky_view_layout,
            &pass("sky view lut", include_str!("shaders/sky/sky_view.wgsl")),
            "main",
        );
        let aerial_view = compute(
            device,
            "aerial view lut",
            &aerial_view_layout,
            &pass(
                "aerial view lut",
                include_str!("shaders/sky/aerial_view.wgsl"),
            ),
            "main",
        );
        let environment = compute(
            device,
            "sky environment",
            &environment_layout,
            &pass(
                "sky environment",
                include_str!("shaders/sky/environment.wgsl"),
            ),
            "main",
        );
        let filter = module(
            device,
            "environment filter",
            include_str!("shaders/sky/environment_filter.wgsl").into(),
        );
        let downsample = compute(
            device,
            "environment downsample",
            &filter_layout,
            &filter,
            "downsample",
        );
        let radiance = compute(
            device,
            "environment radiance",
            &filter_layout,
            &filter,
            "generate_radiance_map",
        );
        let irradiance = compute(
            device,
            "environment irradiance",
            &filter_layout,
            &filter,
            "generate_irradiance_map",
        );

        // Bevy blends with the second source (per-channel transmittance) where the device
        // has dual-source blending, else with the mean transmittance in alpha.
        let dual = device
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING);
        let render_source = format!(
            "{}{COMMON}\n{}\n{}",
            if dual {
                "enable dual_source_blending;\n"
            } else {
                ""
            },
            include_str!("shaders/sky/render_sky.wgsl"),
            if dual {
                include_str!("shaders/sky/render_sky_dual.wgsl")
            } else {
                include_str!("shaders/sky/render_sky_alpha.wgsl")
            },
        );
        let render_module = module(device, "render sky", render_source);
        let render = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render sky"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("render sky"),
                    bind_group_layouts: &[Some(&render_layout)],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &render_module,
                entry_point: Some("fullscreen_vertex_shader"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: crate::renderer::SAMPLE_COUNT,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: crate::renderer::HDR_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: if dual {
                                wgpu::BlendFactor::Src1
                            } else {
                                wgpu::BlendFactor::SrcAlpha
                            },
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atmosphere lut"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let medium_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scattering medium"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let filter_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("environment filter"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let medium = device
            .features()
            .contains(wgpu::Features::FLOAT32_FILTERABLE)
            .then(|| {
                let (density, scattering) = earth_medium();
                let upload = |label, height, texels: &[Vec4]| {
                    use wgpu::util::DeviceExt;
                    device
                        .create_texture_with_data(
                            queue,
                            &wgpu::TextureDescriptor {
                                label: Some(label),
                                size: wgpu::Extent3d {
                                    width: FALLOFF_RESOLUTION,
                                    height,
                                    depth_or_array_layers: 1,
                                },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format: wgpu::TextureFormat::Rgba32Float,
                                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                                view_formats: &[],
                            },
                            wgpu::util::TextureDataOrder::LayerMajor,
                            bytemuck::cast_slice(texels),
                        )
                        .create_view(&Default::default())
                };
                (
                    upload("medium density lut", 2, &density),
                    upload("medium scattering lut", PHASE_RESOLUTION, &scattering),
                )
            });

        Self {
            transmittance_layout,
            multiscattering_layout,
            sky_view_layout,
            aerial_view_layout,
            environment_layout,
            render_layout,
            filter_layout,
            transmittance,
            multiscattering,
            sky_view,
            aerial_view,
            environment,
            downsample,
            radiance,
            irradiance,
            render,
            lut_sampler,
            medium_sampler,
            filter_sampler,
            medium,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Filter {
    Downsample,
    Radiance,
    Irradiance,
}

/// One filter dispatch: its bind group, pipeline and workgroup count per face side.
struct FilterDispatch {
    group: wgpu::BindGroup,
    kind: Filter,
    workgroups: u32,
}

/// A viewport's sky state: LUTs, uniforms, the environment cubes and their bind groups.
pub(crate) struct SkyTargets {
    uniforms: wgpu::Buffer,
    pub transmittance_view: wgpu::TextureView,
    transmittance_group: wgpu::BindGroup,
    multiscattering_group: wgpu::BindGroup,
    sky_view_group: wgpu::BindGroup,
    aerial_view_group: wgpu::BindGroup,
    environment_group: wgpu::BindGroup,
    /// Downsampling (one per mip from 1), then radiance (one per mip), then irradiance.
    filter: Vec<FilterDispatch>,
    /// Bound to the viewport's depth target; rebuilt when it changes.
    render_group: Option<wgpu::BindGroup>,
    luts: [wgpu::TextureView; 4],
    /// The filtered sky cubes the main pass samples (Rgba16Float, like Bevy's).
    pub diffuse_view: wgpu::TextureView,
    pub specular_view: wgpu::TextureView,
    /// Whether this frame renders the sky's environment light.
    environment: bool,
}

/// Mip levels of the specular cube the main pass samples.
pub(crate) const SPECULAR_MIPS: u32 = ENVIRONMENT_MIPS;

impl SkyTargets {
    pub fn new(device: &wgpu::Device, p: &SkyPipelines) -> Self {
        let (density, scattering) = p.medium.as_ref().expect(
            "Scene::sky needs wgpu::Features::FLOAT32_FILTERABLE on the device (Rgba32Float \
             medium LUTs, as in Bevy)",
        );
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky uniforms"),
            size: 5 * UNIFORM_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let texture = |label, size: wgpu::Extent3d, dimension, mips, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: mips,
                sample_count: 1,
                dimension,
                format: LUT_FORMAT,
                usage,
                view_formats: &[],
            })
        };
        let lut_usage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let extent = |w, h, d| wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: d,
        };
        let d2 = wgpu::TextureDimension::D2;
        let luts = [
            texture(
                "transmittance lut",
                extent(TRANSMITTANCE_SIZE.x, TRANSMITTANCE_SIZE.y, 1),
                d2,
                1,
                lut_usage,
            ),
            texture(
                "multiscattering lut",
                extent(MULTISCATTERING_SIZE.x, MULTISCATTERING_SIZE.y, 1),
                d2,
                1,
                lut_usage,
            ),
            texture(
                "sky view lut",
                extent(SKY_VIEW_SIZE.x, SKY_VIEW_SIZE.y, 1),
                d2,
                1,
                lut_usage,
            ),
            texture(
                "aerial view lut",
                extent(
                    AERIAL_VIEW_SIZE[0],
                    AERIAL_VIEW_SIZE[1],
                    AERIAL_VIEW_SIZE[2],
                ),
                wgpu::TextureDimension::D3,
                1,
                lut_usage,
            ),
        ]
        .map(|t| t.create_view(&Default::default()));

        let uniform = |binding: u32, index: u64, size: usize| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &uniforms,
                offset: index * UNIFORM_STRIDE,
                size: wgpu::BufferSize::new(size as u64),
            }),
        };
        let atmosphere = || uniform(0, 0, size_of::<GpuAtmosphere>());
        let settings = || uniform(1, 1, size_of::<GpuSettings>());
        let transforms = || uniform(2, 2, size_of::<GpuTransforms>());
        let view = || uniform(3, 3, size_of::<GpuView>());
        let lights = || uniform(4, 4, size_of::<GpuLights>());
        let medium = || {
            [
                tex(5, density),
                tex(6, scattering),
                sampler(7, &p.medium_sampler),
            ]
        };
        let group = |label, layout: &wgpu::BindGroupLayout, entries: Vec<wgpu::BindGroupEntry>| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &entries,
            })
        };
        let lut_sampler = || sampler(12, &p.lut_sampler);

        let transmittance_group = group(
            "transmittance lut",
            &p.transmittance_layout,
            [atmosphere(), settings()]
                .into_iter()
                .chain(medium())
                .chain([tex(13, &luts[0])])
                .collect(),
        );
        let multiscattering_group = group(
            "multiscattering lut",
            &p.multiscattering_layout,
            [atmosphere(), settings()]
                .into_iter()
                .chain(medium())
                .chain([tex(8, &luts[0]), lut_sampler(), tex(13, &luts[1])])
                .collect(),
        );
        let sky_view_group = group(
            "sky view lut",
            &p.sky_view_layout,
            [atmosphere(), settings(), transforms(), view(), lights()]
                .into_iter()
                .chain(medium())
                .chain([
                    tex(8, &luts[0]),
                    tex(9, &luts[1]),
                    lut_sampler(),
                    tex(13, &luts[2]),
                ])
                .collect(),
        );
        let aerial_view_group = group(
            "aerial view lut",
            &p.aerial_view_layout,
            [atmosphere(), settings(), view(), lights()]
                .into_iter()
                .chain(medium())
                .chain([
                    tex(8, &luts[0]),
                    tex(9, &luts[1]),
                    lut_sampler(),
                    tex(13, &luts[3]),
                ])
                .collect(),
        );

        // The sky cube with its mip chain (Bevy's intermediate texture; the sky is rendered
        // straight into its first level instead of a separate cube Bevy copies from), the
        // specular cube and the diffuse cube.
        let filter_usage =
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let source = texture(
            "sky environment",
            extent(ENVIRONMENT_SIZE, ENVIRONMENT_SIZE, 6),
            d2,
            ENVIRONMENT_MIPS,
            filter_usage,
        );
        let specular = texture(
            "sky specular",
            extent(ENVIRONMENT_SIZE, ENVIRONMENT_SIZE, 6),
            d2,
            ENVIRONMENT_MIPS,
            filter_usage,
        );
        let diffuse = texture(
            "sky diffuse",
            extent(IRRADIANCE_SIZE, IRRADIANCE_SIZE, 6),
            d2,
            1,
            filter_usage,
        );
        let array_view = |t: &wgpu::Texture, mip: Option<u32>| {
            t.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_mip_level: mip.unwrap_or(0),
                mip_level_count: mip.map(|_| 1),
                ..Default::default()
            })
        };
        let cube_view = |t: &wgpu::Texture| {
            t.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            })
        };
        let environment_group = group(
            "sky environment",
            &p.environment_layout,
            [atmosphere(), settings(), transforms(), view(), lights()]
                .into_iter()
                .chain([
                    tex(8, &luts[0]),
                    tex(9, &luts[1]),
                    tex(10, &luts[2]),
                    tex(11, &luts[3]),
                    lut_sampler(),
                ])
                .chain([wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(&array_view(&source, Some(0))),
                }])
                .collect(),
        );

        // Bevy `prepare_generated_environment_map_bind_groups`: radiance roughness `mip / 7`
        // with `32 · 2^⌊4·roughness⌋` samples, irradiance 1024 samples.
        let mut constants = Vec::new();
        let mut dispatch = Vec::new();
        for mip in 1..ENVIRONMENT_MIPS {
            dispatch.push((
                array_view(&source, Some(mip - 1)),
                array_view(&source, Some(mip)),
                Filter::Downsample,
                (ENVIRONMENT_SIZE >> mip).div_ceil(8),
            ));
            // The downsample pass reads only the level it writes.
            constants.push(FilteringConstants {
                mip_level: mip as f32,
                ..FilteringConstants::zeroed()
            });
        }
        let all_mips = array_view(&source, None);
        for mip in 0..ENVIRONMENT_MIPS {
            let roughness = mip as f32 / (ENVIRONMENT_MIPS - 1) as f32;
            constants.push(FilteringConstants {
                mip_level: mip as f32,
                sample_count: 32 * 2u32.pow((roughness * 4.0) as u32),
                roughness,
                ..FilteringConstants::zeroed()
            });
            dispatch.push((
                all_mips.clone(),
                array_view(&specular, Some(mip)),
                Filter::Radiance,
                (ENVIRONMENT_SIZE >> mip).div_ceil(8),
            ));
        }
        constants.push(FilteringConstants {
            sample_count: 1024,
            roughness: 1.0,
            ..FilteringConstants::zeroed()
        });
        dispatch.push((
            all_mips.clone(),
            array_view(&diffuse, None),
            Filter::Irradiance,
            IRRADIANCE_SIZE.div_ceil(8),
        ));
        let mut constant_bytes = vec![0u8; constants.len() * UNIFORM_STRIDE as usize];
        for (i, c) in constants.iter().enumerate() {
            let at = i * UNIFORM_STRIDE as usize;
            constant_bytes[at..at + size_of::<FilteringConstants>()]
                .copy_from_slice(bytemuck::bytes_of(c));
        }
        let constants_buffer = {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("environment filter constants"),
                contents: &constant_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let filter = dispatch
            .into_iter()
            .enumerate()
            .map(|(i, (input, output, kind, workgroups))| FilterDispatch {
                group: group(
                    "environment filter",
                    &p.filter_layout,
                    vec![
                        tex(0, &input),
                        sampler(1, &p.filter_sampler),
                        tex(2, &output),
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &constants_buffer,
                                offset: i as u64 * UNIFORM_STRIDE,
                                size: wgpu::BufferSize::new(size_of::<FilteringConstants>() as u64),
                            }),
                        },
                    ],
                ),
                kind,
                workgroups,
            })
            .collect();

        Self {
            uniforms,
            transmittance_view: luts[0].clone(),
            transmittance_group,
            multiscattering_group,
            sky_view_group,
            aerial_view_group,
            environment_group,
            filter,
            render_group: None,
            luts,
            diffuse_view: cube_view(&diffuse),
            specular_view: cube_view(&specular),
            environment: false,
        }
    }

    /// Drops the bind group that reads the old depth target (after a resize).
    pub fn depth_changed(&mut self) {
        self.render_group = None;
    }

    /// Writes this frame's uniforms and (re)binds the depth target.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        p: &SkyPipelines,
        sky: &Sky,
        lights: &[SkyLight],
        camera: &Camera,
        world_from_view: Mat4,
        clip_from_view: Mat4,
        depth_view: &wgpu::TextureView,
    ) {
        self.environment = sky.environment_intensity.is_some();
        let world_to_atmosphere = world_to_atmosphere();
        let atmosphere = GpuAtmosphere {
            ground_albedo: sky.ground_albedo,
            inner_radius: INNER_RADIUS,
            outer_radius: OUTER_RADIUS,
            _pad: [0.0; 3],
            world_to_atmosphere,
        };
        let settings = GpuSettings {
            transmittance_lut_size: TRANSMITTANCE_SIZE,
            multiscattering_lut_size: MULTISCATTERING_SIZE,
            sky_view_lut_size: SKY_VIEW_SIZE,
            transmittance_lut_samples: 40,
            multiscattering_lut_dirs: 64,
            aerial_view_lut_size: AERIAL_VIEW_SIZE,
            multiscattering_lut_samples: 20,
            sky_view_lut_samples: 16,
            aerial_view_lut_samples: 10,
            aerial_view_lut_max_distance: 3.2e4,
            sky_max_samples: 16,
            rendering_method: 0,
            _pad: [0; 3],
        };
        // Bevy `prepare_atmosphere_transforms`.
        let cam_world = world_from_view.w_axis.truncate();
        let cam_pos = Vec3A::from(world_to_atmosphere.transform_point3(cam_world));
        let atmo_y = cam_pos.try_normalize().unwrap_or(Vec3A::Y);
        let world_ref = Vec3A::NEG_Z;
        let ref_horizontal = world_ref - atmo_y * atmo_y.dot(world_ref);
        let atmo_z = ref_horizontal.try_normalize().unwrap_or_else(|| {
            let fallback_ref = Vec3A::NEG_Y;
            (fallback_ref - atmo_y * atmo_y.dot(fallback_ref)).normalize()
        });
        let atmo_x = atmo_y.cross(atmo_z).normalize();
        let world_from_atmosphere = Mat4::from(Affine3A::from_cols(
            atmo_x,
            atmo_y,
            atmo_z,
            Vec3A::from(cam_world),
        ));
        let transforms = GpuTransforms {
            world_from_atmosphere,
            atmosphere_from_world: world_from_atmosphere.transpose(),
        };
        let view = GpuView {
            world_from_view,
            view_from_clip: clip_from_view.inverse(),
            world_position: camera.eye,
            exposure: camera.exposure(),
        };
        let mut gpu_lights = GpuLights::zeroed();
        for (slot, l) in gpu_lights.directional_lights.iter_mut().zip(lights) {
            *slot = GpuDirectionalLight {
                color: l.color.extend(1.0),
                direction_to_light: l.direction_to_light,
                sun_disk_angular_size: l.disk_size,
                sun_disk_intensity: l.disk_intensity,
                _pad: [0.0; 3],
            };
        }
        gpu_lights.n_directional_lights = lights.len().min(2) as u32;

        let mut bytes = vec![0u8; 5 * UNIFORM_STRIDE as usize];
        let parts: [&[u8]; 5] = [
            bytemuck::bytes_of(&atmosphere),
            bytemuck::bytes_of(&settings),
            bytemuck::bytes_of(&transforms),
            bytemuck::bytes_of(&view),
            bytemuck::bytes_of(&gpu_lights),
        ];
        for (i, part) in parts.iter().enumerate() {
            let at = i * UNIFORM_STRIDE as usize;
            bytes[at..at + part.len()].copy_from_slice(part);
        }
        queue.write_buffer(&self.uniforms, 0, &bytes);

        if self.render_group.is_none() {
            let (density, scattering) = p.medium.as_ref().expect("checked in SkyTargets::new");
            let uniform = |binding: u32, size: usize| wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.uniforms,
                    offset: u64::from(binding) * UNIFORM_STRIDE,
                    size: wgpu::BufferSize::new(size as u64),
                }),
            };
            self.render_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("render sky"),
                layout: &p.render_layout,
                entries: &[
                    uniform(0, size_of::<GpuAtmosphere>()),
                    uniform(1, size_of::<GpuSettings>()),
                    uniform(2, size_of::<GpuTransforms>()),
                    uniform(3, size_of::<GpuView>()),
                    uniform(4, size_of::<GpuLights>()),
                    tex(5, density),
                    tex(6, scattering),
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(&p.medium_sampler),
                    },
                    tex(8, &self.luts[0]),
                    tex(9, &self.luts[1]),
                    tex(10, &self.luts[2]),
                    tex(11, &self.luts[3]),
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: wgpu::BindingResource::Sampler(&p.lut_sampler),
                    },
                    tex(13, depth_view),
                ],
            }));
        }
    }

    /// Records the LUT passes and, with sky ambient light, the environment cubes. The first
    /// pass starts the frame's timestamps when `start` is given.
    pub fn encode_luts(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        p: &SkyPipelines,
        start: Option<&wgpu::QuerySet>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("atmosphere luts"),
            timestamp_writes: start.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: None,
            }),
        });
        let groups_2d = |size: UVec2| (size.x.div_ceil(16), size.y.div_ceil(16));
        pass.set_pipeline(&p.transmittance);
        pass.set_bind_group(0, &self.transmittance_group, &[]);
        let (x, y) = groups_2d(TRANSMITTANCE_SIZE);
        pass.dispatch_workgroups(x, y, 1);
        pass.set_pipeline(&p.multiscattering);
        pass.set_bind_group(0, &self.multiscattering_group, &[]);
        pass.dispatch_workgroups(MULTISCATTERING_SIZE.x, MULTISCATTERING_SIZE.y, 1);
        pass.set_pipeline(&p.sky_view);
        pass.set_bind_group(0, &self.sky_view_group, &[]);
        let (x, y) = groups_2d(SKY_VIEW_SIZE);
        pass.dispatch_workgroups(x, y, 1);
        pass.set_pipeline(&p.aerial_view);
        pass.set_bind_group(0, &self.aerial_view_group, &[]);
        let (x, y) = groups_2d(UVec2::new(AERIAL_VIEW_SIZE[0], AERIAL_VIEW_SIZE[1]));
        pass.dispatch_workgroups(x, y, 1);
        drop(pass);

        if !self.environment {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sky environment"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&p.environment);
        pass.set_bind_group(0, &self.environment_group, &[]);
        pass.dispatch_workgroups(ENVIRONMENT_SIZE / 8, ENVIRONMENT_SIZE / 8, 6);
        for d in &self.filter {
            pass.set_pipeline(match d.kind {
                Filter::Downsample => &p.downsample,
                Filter::Radiance => &p.radiance,
                Filter::Irradiance => &p.irradiance,
            });
            pass.set_bind_group(0, &d.group, &[]);
            pass.dispatch_workgroups(d.workgroups, d.workgroups, 6);
        }
    }

    /// Blends the sky and aerial perspective into the HDR frame; `pass` targets the MSAA colour
    /// with the depth written by the opaque draws.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, p: &SkyPipelines) {
        pass.set_pipeline(&p.render);
        pass.set_bind_group(0, self.render_group.as_ref(), &[]);
        pass.draw(0..3, 0..1);
    }
}

// The uniform layouts the WGSL structs expect.
const _: () = assert!(size_of::<GpuAtmosphere>() == 96);
const _: () = assert!(size_of::<GpuSettings>() == 80);
const _: () = assert!(size_of::<GpuView>() == 144);
const _: () = assert!(size_of::<GpuDirectionalLight>() == 48);
const _: () = assert!(size_of::<GpuLights>() == 112);
const _: () = assert!(size_of::<FilteringConstants>() == 32);
