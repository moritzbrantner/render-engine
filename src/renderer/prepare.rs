//! Writes one frame's camera, lights and instances into a viewport.

use super::{
    Growable, Renderer,
    gpu::{FrameUniform, InstanceData},
    gpu_culling::{GpuCulling, GroupRow},
    lights::LocalLights,
    pipelines::face_variant,
    projection::{Cascade, look_at, looking_to, perspective_infinite_reverse},
    textures::upload_cube,
    viewport::{
        Draw, DrawKey, Phase, PrepareWork, SHADOW_VIEW_STRIDE, ShadowArray, ShadowPass,
        UploadedEnvironment, Viewport,
    },
};
use crate::{
    cull::{Frustum, Obb},
    material::AlphaMode,
    scene::{FogFalloff, MAX_CASCADES, Scene, ShadowFilter, Tonemapping},
    sky::SkyTargets,
    world::{Instance, SkinId, World},
};
use bytemuck::Zeroable;
use glam::{Mat3, Mat4, UVec4, Vec3, Vec4};
use std::{collections::HashMap, sync::Arc};

impl Renderer {
    /// Writes one frame's camera, lights and instances into `viewport`. The frame draws the
    /// world's retained instances ([`World::add_instance`]) and `instances`, listed for this
    /// frame alone; a retained instance is written to the viewport only when it was added or
    /// changed since the viewport's last `prepare`, listed ones every frame. Only buffers whose
    /// contents changed size are reallocated; meshes, textures and materials stay in `world`.
    pub fn prepare(
        &self,
        viewport: &mut Viewport,
        world: &World,
        scene: &Scene,
        instances: &[Instance],
    ) {
        let (device, queue) = (&self.device, &self.queue);
        viewport.work = PrepareWork::default();
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
        viewport.work.upload_bytes += std::mem::size_of::<FrameUniform>() as u64;

        let grouped = write_instances(viewport, device, queue, world, view_from_world, instances);

        let mut lighting_dirty = viewport.lighting_group.is_none();
        lighting_dirty |= write_growable(
            &mut viewport.light_buffer,
            &mut viewport.work,
            device,
            queue,
            bytemuck::cast_slice(&lights.gpu),
        );

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
            viewport.work.allocations += 1;
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
            viewport.work.allocations += 1;
            lighting_dirty = true;
        }

        write_shadow_views(viewport, device, queue, &cascades, &lights);
        cull_views(
            viewport,
            device,
            queue,
            clip_from_view * view_from_world,
            &grouped,
        );
        // A replaced instance or visible buffer clears `frame_group`; the GPU pass binds both.
        if viewport.frame_group.is_none()
            && let Some(gpu) = &mut viewport.gpu_culling
        {
            gpu.invalidate_bind_group();
        }

        // Environment maps are uploaded once per distinct pair of cube maps.
        let cached = match (&scene.environment, &viewport.environment) {
            (Some(e), Some(u)) => {
                Arc::ptr_eq(&e.diffuse, &u.diffuse) && Arc::ptr_eq(&e.specular, &u.specular)
            }
            (None, None) => true,
            _ => false,
        };
        if !cached {
            if let Some(e) = &scene.environment {
                viewport.work.allocations += 2;
                viewport.work.upload_bytes += [&e.diffuse, &e.specular]
                    .iter()
                    .flat_map(|c| &c.levels)
                    .map(|l| l.len() as u64)
                    .sum::<u64>();
            }
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
                viewport.work.allocations += 1;
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
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: viewport.visible_buffer.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: viewport.joint_buffer.buffer.as_entire_binding(),
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
        if viewport.gpu_culling.is_some() {
            GpuCulling::rebuild_bind_group(viewport, device);
        }
        if let Some(clustered) = &mut viewport.clustered {
            clustered.prepare(
                device,
                queue,
                &lights,
                view_from_world,
                (viewport.width, viewport.height),
                cam.fov_y,
                &mut viewport.work,
            );
        }
        viewport.particles.prepare(
            device,
            queue,
            &self.particles,
            world,
            clip_from_view * view_from_world,
            world_from_view,
            cam.near,
            &viewport.targets.depth_view,
        );
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

/// The frame's instances in draw order, grouped for culling.
struct Grouped {
    /// Consecutive sorted instances sharing a pipeline, material and mesh: ranges over the
    /// draw order.
    groups: Vec<Draw>,
    /// Per position in the draw order: the instance's row in the instance buffer.
    rows: Vec<u32>,
    /// Per position in the draw order: the instance's box, compact for the per-view culling
    /// loops.
    obbs: Vec<Obb>,
    /// With `Culling::GpuIndirect`, the groups the compute pass culls; each group's `gpu` is
    /// its row here.
    gpu_rows: Vec<GroupRow>,
}

/// An instance's key (`index` set, `view_z` not), box and GPU row (`flags.w` zero). A skinned
/// instance is posed with its skin at `joint_offset` in the joint buffer; without one its box
/// and row are left empty (the retained copy poses skinned instances each frame).
pub(super) fn derive_instance(
    world: &World,
    instance: &Instance,
    index: u32,
    joint_offset: Option<u32>,
) -> (DrawKey, Obb, InstanceData) {
    let material = &world.material(instance.material).material;
    let mesh = world.mesh(instance.mesh);
    let m = instance.world_from_local;
    let skinned = mesh.skin.is_some();
    match (skinned, instance.skin) {
        (true, None) => panic!("an instance of a skinned mesh needs a skin"),
        (false, Some(_)) => panic!("a skin on an instance of an unskinned mesh"),
        _ => {}
    }
    let key = DrawKey {
        phase: match material.alpha_mode {
            AlphaMode::Opaque => Phase::Opaque,
            AlphaMode::Mask(_) => Phase::Mask,
            AlphaMode::Blend => Phase::Blend,
        },
        face: face_variant(
            Mat3::from_mat4(m).determinant() < 0.0,
            material.double_sided,
        ),
        material: instance.material,
        mesh: instance.mesh,
        view_z: 0.0,
        skinned,
        index,
    };
    let local_aabb = match (instance.skin, joint_offset) {
        (None, _) => mesh.aabb,
        (Some(skin), Some(_)) => {
            let palette = world.skin(skin);
            mesh.joint_bounds
                .iter()
                .filter_map(|(j, b)| Some(b.transformed(*palette.get(*j as usize)?)))
                .reduce(|a, b| a.union(&b))
                .unwrap_or(mesh.aabb)
        }
        (Some(_), None) => return (key, Obb::default(), InstanceData::zeroed()),
    };
    let normal = Mat3::from_mat4(m).inverse().transpose();
    let data = InstanceData {
        world_from_local: m,
        normal_from_local: [
            normal.x_axis.extend(0.0),
            normal.y_axis.extend(0.0),
            normal.z_axis.extend(0.0),
        ],
        flags: UVec4::new(
            u32::from(mesh.has_tangents),
            (key.face & 1) as u32,
            joint_offset.unwrap_or(0),
            0,
        ),
    };
    (key, Obb::new(local_aabb, m), data)
}

/// Writes the instance and joint buffers and groups the instances into draws: the world's
/// retained instances (rows `0..R`, written where they changed) and the frame's listed ones
/// (rows `R..`, written every frame).
fn write_instances(
    viewport: &mut Viewport,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    world: &World,
    view_from_world: Mat4,
    instances: &[Instance],
) -> Grouped {
    let retained = &mut viewport.retained;
    retained.sync(world);
    let first_listed = retained.rows();

    // Each skin's palette once, in order of first use (Bevy `prepare_skins`): retained
    // instances by slot, then the listed ones.
    let mut joints: Vec<Mat4> = Vec::new();
    let mut skin_offsets: HashMap<SkinId, u32> = HashMap::new();
    let mut joint_offset = |skin: Option<SkinId>| {
        skin.map(|skin| {
            *skin_offsets.entry(skin).or_insert_with(|| {
                let palette = world.skin(skin);
                joints.extend_from_slice(palette);
                (joints.len() - palette.len()) as u32
            })
        })
    };
    for &slot in &retained.skinned {
        let instance = world.instance_at(slot).expect("a skinned slot is live");
        let (_, obb, data) = derive_instance(world, instance, slot, joint_offset(instance.skin));
        retained.obbs[slot as usize] = obb;
        retained.data[slot as usize] = data;
        retained.dirty.push(slot);
    }

    // Draw order: opaque, then alpha-masked (each by pipeline, material and mesh), then
    // blended from back to front by origin (Bevy's transparent sort); ties keep the rows'
    // order. The retained opaque and masked keys are kept sorted; this frame's keys (the
    // listed instances and the retained blended ones) are sorted and merged into them.
    let frame_keys = &mut viewport.frame_keys;
    frame_keys.clear();
    for &slot in &retained.blended {
        let mut key = retained.key(slot);
        key.view_z = (view_from_world * retained.data[slot as usize].world_from_local.w_axis).z;
        frame_keys.push(key);
    }
    let listed_data = &mut viewport.instance_data;
    listed_data.clear();
    let mut listed_obbs: Vec<Obb> = Vec::with_capacity(instances.len());
    for (i, instance) in instances.iter().enumerate() {
        let (mut key, obb, data) = derive_instance(
            world,
            instance,
            first_listed + i as u32,
            joint_offset(instance.skin),
        );
        key.view_z = (view_from_world * instance.world_from_local.w_axis).z;
        frame_keys.push(key);
        listed_obbs.push(obb);
        listed_data.push(data);
    }
    frame_keys.sort_unstable_by(DrawKey::draw_order);
    let keys = &mut viewport.draw_keys;
    merge_sorted(&retained.sorted, frame_keys, keys);

    // Rows of the compute pass's group table (empty on the CPU path); the instance data
    // carries `row + 1`, zero for instances the pass skips.
    let gpu_on = viewport.gpu_culling.is_some();
    let mut gpu_rows: Vec<GroupRow> = Vec::new();
    let mut groups: Vec<Draw> = Vec::new();
    let mut rows: Vec<u32> = Vec::with_capacity(keys.len());
    let mut obbs: Vec<Obb> = Vec::with_capacity(keys.len());
    for (i, key) in keys.iter().enumerate() {
        let i = i as u32;
        match groups.last_mut() {
            Some(d)
                if (d.phase, d.face, d.material, d.mesh)
                    == (key.phase, key.face, key.material, key.mesh) =>
            {
                d.instances.end = i + 1;
            }
            _ => {
                let gpu = (gpu_on && key.phase != Phase::Blend && !key.skinned).then(|| {
                    let mesh = world.mesh(key.mesh);
                    gpu_rows.push(GroupRow::new(mesh.aabb, mesh.index_count));
                    gpu_rows.len() as u32 - 1
                });
                groups.push(Draw {
                    face: key.face,
                    phase: key.phase,
                    skinned: key.skinned,
                    mesh: key.mesh,
                    material: key.material,
                    instances: i..i + 1,
                    gpu,
                });
            }
        }
        let gpu_row = groups.last().and_then(|d| d.gpu);
        if let Some(row) = gpu_row {
            gpu_rows[row as usize].count += 1;
        }
        let row_plus_one = gpu_row.map_or(0, |row| row + 1);
        rows.push(key.index);
        if key.index < first_listed {
            obbs.push(retained.obbs[key.index as usize]);
            if gpu_on {
                retained.set_gpu_row(key.index, row_plus_one);
            }
        } else {
            let listed = (key.index - first_listed) as usize;
            obbs.push(listed_obbs[listed]);
            if gpu_on {
                listed_data[listed].flags.w = row_plus_one;
            }
        }
    }

    // The instance buffer: the retained rows that changed, then every listed row (one zeroed
    // row when there is no instance at all).
    let stride = size_of::<InstanceData>() as u64;
    if first_listed == 0 && listed_data.is_empty() {
        listed_data.push(InstanceData::zeroed());
    }
    let total = u64::from(first_listed) + listed_data.len() as u64;
    let replaced = viewport.instance_buffer.reserve(device, total * stride);
    let work = &mut viewport.work;
    let retained_written = retained.upload(&viewport.instance_buffer, replaced, queue, work);
    let listed_bytes: &[u8] = bytemuck::cast_slice(listed_data);
    if !listed_bytes.is_empty() {
        queue.write_buffer(
            &viewport.instance_buffer.buffer,
            u64::from(first_listed) * stride,
            listed_bytes,
        );
    }
    work.upload_bytes += listed_bytes.len() as u64;
    work.allocations += u32::from(replaced);
    work.instances = retained.count() + instances.len() as u32;
    work.instance_writes = retained_written + instances.len() as u32;
    viewport.instance_rows = total as u32;
    if replaced {
        viewport.frame_group = None;
    }

    if joints.is_empty() {
        joints.push(Mat4::IDENTITY);
    }
    if write_growable(
        &mut viewport.joint_buffer,
        &mut viewport.work,
        device,
        queue,
        bytemuck::cast_slice(&joints),
    ) {
        viewport.frame_group = None;
    }
    Grouped {
        groups,
        rows,
        obbs,
        gpu_rows,
    }
}

/// Merges two key lists, each in draw order, into `out` (taking `b`'s allocation when `a` is
/// empty).
fn merge_sorted(a: &[DrawKey], b: &mut Vec<DrawKey>, out: &mut Vec<DrawKey>) {
    if a.is_empty() {
        std::mem::swap(b, out);
        return;
    }
    out.clear();
    out.reserve(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i].draw_order(&b[j]).is_le() {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
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
            frustum: Frustum::cascade(c.clip_from_world),
            sphere: None,
            draws: 0..0,
        });
        view_matrices.push(c.clip_from_world);
    }
    let local_views = lights
        .spot_views
        .iter()
        .enumerate()
        .map(|(i, v)| ((cascades.len() + i) as u32, false, v))
        .chain(
            lights
                .point_views
                .iter()
                .enumerate()
                .map(|(i, v)| (i as u32, true, v)),
        );
    for (layer, point, v) in local_views {
        viewport.shadow_passes.push(ShadowPass {
            local: true,
            layer,
            point,
            view: view_matrices.len() as u32,
            frustum: v.frustum,
            sphere: Some(v.sphere),
            draws: 0..0,
        });
        view_matrices.push(v.clip_from_world);
    }
    let mut view_bytes = vec![0u8; SHADOW_VIEW_STRIDE as usize * view_matrices.len().max(1)];
    for (i, m) in view_matrices.iter().enumerate() {
        let at = i * SHADOW_VIEW_STRIDE as usize;
        view_bytes[at..at + 64].copy_from_slice(bytemuck::bytes_of(m));
    }
    if write_growable(
        &mut viewport.shadow_view_buffer,
        &mut viewport.work,
        device,
        queue,
        &view_bytes,
    ) {
        viewport.shadow_view_group = None;
    }
}

/// Writes a per-frame buffer, counting the bytes and a replacement. Returns whether the buffer
/// was replaced.
fn write_growable(
    buffer: &mut Growable,
    work: &mut PrepareWork,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    bytes: &[u8],
) -> bool {
    work.upload_bytes += bytes.len() as u64;
    let grown = buffer.write(device, queue, bytes);
    work.allocations += u32::from(grown);
    grown
}

/// Frustum culling per view (Bevy `check_visibility` and the shadow views' culling): each view
/// keeps its own draws over a run of visible instance indices. Writes the visible-index buffer.
///
/// With `Culling::GpuIndirect`, a group with a GPU row is not tested here: each view gets one
/// indirect draw per row, over a run the compute pass fills. The runs take the start of the
/// visible-index buffer; the indices of the groups still culled here follow them.
fn cull_views(
    viewport: &mut Viewport,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    clip_from_world: Mat4,
    grouped: &Grouped,
) {
    let Grouped {
        groups,
        rows,
        obbs,
        gpu_rows,
    } = grouped;
    let views = 1 + viewport.shadow_passes.len() as u32;
    let gpu_instances: u32 = gpu_rows.iter().map(|r| r.count).sum();
    let gpu_region = views * gpu_instances;
    let gpu_row_count = gpu_rows.len() as u32;
    viewport.draws.clear();
    let mut visible: Vec<u32> = Vec::new();
    let mut tests = 0u64;
    let mut cull = |draws: &mut Vec<Draw>, view: u32, sees: &dyn Fn(usize) -> bool| {
        let start = draws.len();
        for group in groups {
            if let Some(row) = group.gpu {
                draws.push(Draw {
                    instances: 0..0,
                    gpu: Some(view * gpu_row_count + row),
                    ..group.clone()
                });
                continue;
            }
            tests += u64::from(group.instances.end - group.instances.start);
            let first = visible.len() as u32;
            visible.extend(
                group
                    .instances
                    .clone()
                    .filter(|&i| sees(i as usize))
                    .map(|i| rows[i as usize]),
            );
            let end = visible.len() as u32;
            if end > first {
                draws.push(Draw {
                    instances: first + gpu_region..end + gpu_region,
                    gpu: None,
                    ..group.clone()
                });
            }
        }
        start..draws.len()
    };
    let camera = Frustum::camera(clip_from_world);
    viewport.main_draws = cull(&mut viewport.draws, 0, &|i| camera.intersects(&obbs[i]));
    // A light's sphere test runs once for all its views (six cube faces), and only over the
    // groups this loop culls.
    let mut in_range: Option<((Vec3, f32), Vec<bool>)> = None;
    let mut sphere_tests = 0u64;
    for (i, pass) in viewport.shadow_passes.iter_mut().enumerate() {
        let view = 1 + i as u32;
        let draws = match pass.sphere {
            None => cull(&mut viewport.draws, view, &|i| {
                pass.frustum.intersects(&obbs[i])
            }),
            Some(sphere) => {
                if in_range.as_ref().is_none_or(|(s, _)| *s != sphere) {
                    let mut near = vec![false; obbs.len()];
                    for group in groups.iter().filter(|g| g.gpu.is_none()) {
                        sphere_tests += u64::from(group.instances.end - group.instances.start);
                        for i in group.instances.clone() {
                            near[i as usize] =
                                obbs[i as usize].intersects_sphere(sphere.0, sphere.1);
                        }
                    }
                    in_range = Some((sphere, near));
                }
                let near = &in_range.as_ref().expect("filled above").1;
                cull(&mut viewport.draws, view, &|i| {
                    near[i] && pass.frustum.intersects(&obbs[i])
                })
            }
        };
        pass.draws = draws;
    }
    // The camera's kept indices come first among those culled here.
    let main_visible: u32 = viewport.draws[viewport.main_draws.clone()]
        .iter()
        .filter(|d| d.gpu.is_none())
        .map(|d| d.instances.end - d.instances.start)
        .sum();
    let work = &mut viewport.work;
    work.views = views;
    work.bounds_tests = tests + sphere_tests;
    work.gpu_bounds_tests = u64::from(gpu_region);
    work.visible = main_visible;
    work.shadow_visible = visible.len() as u32 - main_visible;
    work.draws = viewport.main_draws.len() as u32;
    work.shadow_draws = (viewport.draws.len() - viewport.main_draws.len()) as u32;
    if visible.is_empty() && gpu_region == 0 {
        visible.push(0);
    }
    let bytes: &[u8] = bytemuck::cast_slice(&visible);
    work.upload_bytes += bytes.len() as u64;
    let offset = u64::from(gpu_region) * 4;
    let size = offset + bytes.len() as u64;
    let grown = viewport
        .visible_buffer
        .write_at(device, queue, offset, bytes, size);
    work.allocations += u32::from(grown);
    if grown {
        viewport.frame_group = None;
    }
    if viewport.gpu_culling.is_some() {
        GpuCulling::write_tables(viewport, device, queue, gpu_rows, clip_from_world);
    }
}
