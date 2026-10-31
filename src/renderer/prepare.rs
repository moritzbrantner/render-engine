//! Writes one frame's camera, lights and instances into a viewport.

use super::{
    Renderer,
    gpu::{FrameUniform, InstanceData},
    lights::LocalLights,
    pipelines::face_variant,
    projection::{Cascade, look_at, looking_to, perspective_infinite_reverse},
    textures::upload_cube,
    viewport::{
        Draw, Phase, SHADOW_VIEW_STRIDE, ShadowArray, ShadowPass, UploadedEnvironment, Viewport,
    },
};
use crate::{
    material::AlphaMode,
    scene::{FogFalloff, MAX_CASCADES, Scene, ShadowFilter, Tonemapping},
    sky::SkyTargets,
    world::{Instance, MaterialId, MeshId, World},
};
use bytemuck::Zeroable;
use glam::{Mat3, Mat4, UVec4, Vec3, Vec4};
use std::sync::Arc;

impl Renderer {
    /// Writes one frame's camera, lights and instances into `viewport`. Only buffers whose
    /// contents changed size are reallocated; meshes, textures and materials stay in `world`.
    pub fn prepare(
        &self,
        viewport: &mut Viewport,
        world: &World,
        scene: &Scene,
        instances: &[Instance],
    ) {
        let (device, queue) = (&self.device, &self.queue);
        let aspect = viewport.width as f32 / viewport.height as f32;
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
                        scene.shadow_maps.directional,
                        sun.direction,
                        cam.fov_y,
                        aspect,
                        world_from_view,
                    );
                    (far, c)
                })
                .collect()
        });
        let lights = LocalLights::new(scene, cascades.len());

        assert!(
            !(scene.environment.is_some()
                && scene.sky.is_some_and(|s| s.environment_intensity.is_some())),
            "a sky with environment light replaces Scene::environment; set one of them"
        );
        let (sky_lights, moon_first) = crate::sky::directional_lights(scene);
        let frame = frame_uniform(
            scene,
            clip_from_view * view_from_world,
            view_from_world,
            &cascades,
            &lights,
            moon_first,
        );
        queue.write_buffer(&viewport.frame_buffer, 0, bytemuck::bytes_of(&frame));

        write_instances(viewport, device, queue, world, view_from_world, instances);

        let mut lighting_dirty = viewport.lighting_group.is_none();
        lighting_dirty |=
            viewport
                .light_buffer
                .write(device, queue, bytemuck::cast_slice(&lights.gpu));

        // Like Bevy, spot maps share the sun's array (and its size), after the cascades.
        let directional_layers = (cascades.len() + lights.spot_views.len()) as u32;
        let directional = if directional_layers > 0 {
            (scene.shadow_maps.directional, directional_layers)
        } else {
            (1, 1)
        };
        let shadow_arrays = &mut viewport.directional_shadows;
        if (shadow_arrays.size, shadow_arrays.layers) != directional {
            *shadow_arrays = ShadowArray::new(
                device,
                "directional shadow maps",
                directional.0,
                directional.1,
                false,
            );
            lighting_dirty = true;
        }
        let cube_count = (lights.point_views.len() / 6) as u32;
        let point = if cube_count > 0 {
            (scene.shadow_maps.point, 6 * cube_count)
        } else {
            (1, 6)
        };
        let point_arrays = &mut viewport.point_shadows;
        if (point_arrays.size, point_arrays.layers) != point {
            *point_arrays = ShadowArray::new(device, "point shadow maps", point.0, point.1, true);
            lighting_dirty = true;
        }

        write_shadow_views(viewport, device, queue, &cascades, &lights);

        // Environment maps are uploaded once per distinct pair of cube maps.
        let cached = match (&scene.environment, &viewport.environment) {
            (Some(e), Some(u)) => {
                Arc::ptr_eq(&e.diffuse, &u.diffuse) && Arc::ptr_eq(&e.specular, &u.specular)
            }
            (None, None) => true,
            _ => false,
        };
        if !cached {
            viewport.environment = scene.environment.as_ref().map(|e| UploadedEnvironment {
                diffuse: e.diffuse.clone(),
                specular: e.specular.clone(),
                diffuse_view: upload_cube(device, queue, &e.diffuse),
                specular_view: upload_cube(device, queue, &e.specular),
            });
            lighting_dirty = true;
        }

        match (&scene.sky, viewport.sky.is_some()) {
            (Some(_), false) => {
                viewport.sky = Some(SkyTargets::new(device, &self.sky));
                lighting_dirty = true;
            }
            (None, true) => {
                viewport.sky = None;
                lighting_dirty = true;
            }
            _ => {}
        }
        if let (Some(sky), Some(targets)) = (&scene.sky, &mut viewport.sky) {
            targets.prepare(
                device,
                queue,
                &self.sky,
                sky,
                &sky_lights,
                cam,
                // The camera transform itself, as Bevy's view uniform has it.
                Mat4::from_translation(cam.eye)
                    * Mat4::from_mat3(looking_to(cam.target - cam.eye, Vec3::Y)),
                clip_from_view,
                &viewport.targets.depth_view,
            );
        }

        if lighting_dirty {
            viewport.lighting_group = Some(self.lighting_group(viewport));
        }
        if viewport.frame_group.is_none() {
            viewport.frame_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("frame"),
                layout: &self.frame_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: viewport.frame_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: viewport.instance_buffer.buffer.as_entire_binding(),
                    },
                ],
            }));
        }
        if viewport.shadow_view_group.is_none() {
            viewport.shadow_view_group =
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shadow view"),
                    layout: &self.shadow_view_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &viewport.shadow_view_buffer.buffer,
                            offset: 0,
                            size: wgpu::BufferSize::new(64),
                        }),
                    }],
                }));
        }
        if viewport.tonemap.as_ref().map(|(t, _)| *t) != Some(cam.tonemapping) {
            viewport.tonemap = Some((
                cam.tonemapping,
                self.tonemap_group(viewport, cam.tonemapping),
            ));
        }
        viewport.tonemapping = cam.tonemapping;
        let c = scene.clear_color;
        viewport.clear_color = wgpu::Color {
            r: c.x.into(),
            g: c.y.into(),
            b: c.z.into(),
            a: 1.0,
        };
    }

    fn lighting_group(&self, viewport: &Viewport) -> wgpu::BindGroup {
        let (diffuse_view, specular_view) = match (&viewport.environment, &viewport.sky) {
            (Some(e), _) => (&e.diffuse_view, &e.specular_view),
            (None, Some(sky)) => (&sky.diffuse_view, &sky.specular_view),
            (None, None) => (&self.no_environment, &self.no_environment),
        };
        let transmittance = viewport
            .sky
            .as_ref()
            .map_or(&self.no_transmittance, |s| &s.transmittance_view);
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                    resource: wgpu::BindingResource::TextureView(
                        &viewport.directional_shadows.view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: viewport.light_buffer.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&viewport.point_shadows.view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(transmittance),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(&self.sky.lut_sampler),
                },
            ],
        })
    }

    fn tonemap_group(&self, viewport: &Viewport, tonemapping: Tonemapping) -> wgpu::BindGroup {
        // Like Bevy, operators without a LUT still bind one (the small AgX LUT).
        let lut = match tonemapping {
            Tonemapping::TonyMcMapface => &self.tony_mc_mapface_lut,
            Tonemapping::AcesFitted | Tonemapping::AgX => &self.agx_lut,
        };
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tonemap"),
            layout: &self.tonemap_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&viewport.targets.hdr_view),
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
        })
    }
}

/// The frame's uniform block.
fn frame_uniform(
    scene: &Scene,
    clip_from_world: Mat4,
    view_from_world: Mat4,
    cascades: &[(f32, Cascade)],
    lights: &LocalLights,
    moon_first: bool,
) -> FrameUniform {
    let cam = &scene.camera;
    let shadows = scene.sun.and_then(|s| s.shadows);
    let sky_environment = scene.sky.and_then(|s| s.environment_intensity);
    let mut cascade_clip_from_world = [Mat4::IDENTITY; MAX_CASCADES];
    let mut cascade_params = [Vec4::ZERO; MAX_CASCADES];
    for (i, (far, c)) in cascades.iter().enumerate() {
        cascade_clip_from_world[i] = c.clip_from_world;
        cascade_params[i] = Vec4::new(*far, c.texel_size, 0.0, 0.0);
    }

    let sun = scene.sun;
    FrameUniform {
        clip_from_world,
        view_from_world,
        cascade_clip_from_world,
        cascade_params,
        eye: cam.eye.extend(1.0),
        direction_to_light: sun.map_or(Vec4::ZERO, |s| (-s.direction.normalize()).extend(0.0)),
        sun_color: sun.map_or(Vec4::ZERO, |s| (s.color * s.illuminance).extend(1.0)),
        ambient_color: (scene.ambient.color * scene.ambient.brightness).extend(1.0),
        shadow: shadows.map_or(Vec4::ZERO, |cfg| {
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
        environment: match (&scene.environment, sky_environment) {
            (Some(e), _) => Vec4::new(e.intensity, (e.specular.levels.len() - 1) as f32, 1.0, 0.0),
            (None, Some(intensity)) => {
                Vec4::new(intensity, (crate::sky::SPECULAR_MIPS - 1) as f32, 1.0, 0.0)
            }
            (None, None) => Vec4::ZERO,
        },
        fog_color: scene.fog.map_or(Vec4::ZERO, |f| f.color),
        fog_directional_light_color: scene.fog.map_or(Vec4::ZERO, |f| f.directional_light_color),
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
        light_counts: UVec4::new(lights.point_count, lights.spot_count, 0, 0),
        moon_direction_to_light: scene
            .moon
            .map_or(Vec4::ZERO, |m| (-m.direction.normalize()).extend(1.0)),
        moon_color: scene
            .moon
            .map_or(Vec4::ZERO, |m| (m.color * m.illuminance).extend(1.0)),
        world_to_atmosphere: crate::sky::world_to_atmosphere(),
        atmosphere: Vec4::new(
            crate::sky::INNER_RADIUS,
            crate::sky::OUTER_RADIUS,
            f32::from(u8::from(scene.sky.is_some())),
            f32::from(u8::from(moon_first)),
        ),
        disks: scene.sky.map_or(Vec4::ZERO, |s| {
            Vec4::new(s.sun_disk.angular_size, s.moon_disk.angular_size, 0.0, 0.0)
        }),
    }
}

/// Writes the instance buffer and the draw list.
fn write_instances(
    viewport: &mut Viewport,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    world: &World,
    view_from_world: Mat4,
    instances: &[Instance],
) {
    // Instances, grouped into draws: opaque, then alpha-masked (each by pipeline, material
    // and mesh), then blended from back to front by origin (Bevy's transparent sort).
    struct Sorted {
        phase: Phase,
        face: usize,
        material: MaterialId,
        mesh: MeshId,
        view_z: f32,
        data: InstanceData,
    }
    let mut sorted: Vec<Sorted> = instances
        .iter()
        .map(|inst| {
            let material = &world.material(inst.material).material;
            let mesh = world.mesh(inst.mesh);
            let m = inst.world_from_local;
            let linear = Mat3::from_mat4(m);
            let mirrored = linear.determinant() < 0.0;
            let normal = linear.inverse().transpose();
            Sorted {
                phase: match material.alpha_mode {
                    AlphaMode::Opaque => Phase::Opaque,
                    AlphaMode::Mask(_) => Phase::Mask,
                    AlphaMode::Blend => Phase::Blend,
                },
                face: face_variant(mirrored, material.double_sided),
                material: inst.material,
                mesh: inst.mesh,
                view_z: (view_from_world * m.w_axis).z,
                data: InstanceData {
                    world_from_local: m,
                    normal_from_local: [
                        normal.x_axis.extend(0.0),
                        normal.y_axis.extend(0.0),
                        normal.z_axis.extend(0.0),
                    ],
                    flags: UVec4::new(u32::from(mesh.has_tangents), u32::from(mirrored), 0, 0),
                },
            }
        })
        .collect();
    sorted.sort_by(|a, b| {
        a.phase.cmp(&b.phase).then_with(|| {
            if a.phase == Phase::Blend {
                a.view_z.total_cmp(&b.view_z)
            } else {
                (a.face, a.material, a.mesh).cmp(&(b.face, b.material, b.mesh))
            }
        })
    });
    viewport.draws.clear();
    for (i, s) in sorted.iter().enumerate() {
        let i = i as u32;
        match viewport.draws.last_mut() {
            Some(d)
                if (d.phase, d.face, d.material, d.mesh)
                    == (s.phase, s.face, s.material, s.mesh) =>
            {
                d.instances.end = i + 1;
            }
            _ => viewport.draws.push(Draw {
                face: s.face,
                phase: s.phase,
                mesh: s.mesh,
                material: s.material,
                instances: i..i + 1,
            }),
        }
    }
    let mut instance_data: Vec<InstanceData> = sorted.iter().map(|s| s.data).collect();
    if instance_data.is_empty() {
        instance_data.push(InstanceData::zeroed());
    }
    if viewport
        .instance_buffer
        .write(device, queue, bytemuck::cast_slice(&instance_data))
    {
        viewport.frame_group = None;
    }
}

/// Writes the shadow-view buffer and the list of shadow passes.
fn write_shadow_views(
    viewport: &mut Viewport,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cascades: &[(f32, Cascade)],
    lights: &LocalLights,
) {
    // Shadow views: sun cascades, then spot maps, then point-light faces.
    viewport.shadow_passes.clear();
    let mut view_matrices = Vec::new();
    for (i, (_, c)) in cascades.iter().enumerate() {
        viewport.shadow_passes.push(ShadowPass {
            local: false,
            layer: i as u32,
            point: false,
            view: view_matrices.len() as u32,
        });
        view_matrices.push(c.clip_from_world);
    }
    for (i, m) in lights.spot_views.iter().enumerate() {
        viewport.shadow_passes.push(ShadowPass {
            local: true,
            layer: (cascades.len() + i) as u32,
            point: false,
            view: view_matrices.len() as u32,
        });
        view_matrices.push(*m);
    }
    for (i, m) in lights.point_views.iter().enumerate() {
        viewport.shadow_passes.push(ShadowPass {
            local: true,
            layer: i as u32,
            point: true,
            view: view_matrices.len() as u32,
        });
        view_matrices.push(*m);
    }
    let mut view_bytes = vec![0u8; SHADOW_VIEW_STRIDE as usize * view_matrices.len().max(1)];
    for (i, m) in view_matrices.iter().enumerate() {
        let at = i * SHADOW_VIEW_STRIDE as usize;
        view_bytes[at..at + 64].copy_from_slice(bytemuck::bytes_of(m));
    }
    if viewport
        .shadow_view_buffer
        .write(device, queue, &view_bytes)
    {
        viewport.shadow_view_group = None;
    }
}
