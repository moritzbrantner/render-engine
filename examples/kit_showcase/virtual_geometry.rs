//! Development-only CPU cluster-cut experiment. No production API or reference dependency.
//! Reuses the showcase's real kit materials. All instances, including off-camera casters,
//! reach the renderer, which frustum-culls separately in every camera/shadow view.
use super::*;
use render_engine::renderer::FrameTimestamps;

struct Cluster {
    mesh: MeshId,
    producer: i32,
    consumer: usize,
    triangles: u32,
}

struct Asset {
    source: Prop,
    lods: Vec<Prop>,
    clusters: Vec<Cluster>,
    groups: Vec<[f32; 5]>,
    source_triangles: u64,
    lod_triangles: Vec<u64>,
    uploaded_bytes: u64,
}

fn floats(bytes: &[u8], meta: &Json, name: &str) -> Vec<f32> {
    section(bytes, meta, name)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

fn section<'a>(bytes: &'a [u8], meta: &Json, name: &str) -> &'a [u8] {
    let s = &meta["sections"][name];
    let offset = s["offset"].as_u64().expect("section offset") as usize;
    let size = s["bytes"].as_u64().expect("section size") as usize;
    bytes
        .get(offset..offset.checked_add(size).expect("section range"))
        .expect("section inside binary")
}

fn integers(bytes: &[u8], meta: &Json, name: &str) -> Vec<i32> {
    section(bytes, meta, name)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect()
}

fn load(
    gpu: &Headless,
    world: &mut World,
    kit_path: &Path,
    hierarchy: &Path,
    name: &str,
    kind: &str,
) -> Asset {
    let mut textures = Textures::default();
    let mut kit = Kit::open(kit_path.to_owned());
    let source_path = kit.entries.get(name).expect("kit entry").clone();
    let source = load_prop(world, &mut textures, &kit, name);
    assert_eq!(
        source.parts.len(),
        1,
        "experiment requires one primitive/material per asset"
    );
    let mut lods = Vec::new();
    let mut lod_triangles = Vec::new();
    for lod in ["lod1", "lod2", "lod3", "lod4"]
        .into_iter()
        .filter(|_| matches!(kind, "all" | "discrete"))
    {
        let path = source_path.with_file_name(format!("{lod}.gltf"));
        if path.exists() {
            let json: Json = serde_json::from_slice(&read(&path)).unwrap();
            let count: u64 = json["meshes"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|m| m["primitives"].as_array().unwrap())
                .map(|pr| {
                    let accessor = pr["indices"].as_u64().unwrap() as usize;
                    json["accessors"][accessor]["count"].as_u64().unwrap() / 3
                })
                .sum();
            lod_triangles.push(count);
            kit.entries.insert(name.to_owned(), path);
            let prop = load_prop(world, &mut textures, &kit, name);
            assert_eq!(prop.parts.len(), 1);
            lods.push(prop);
        }
    }
    let meta: Json =
        serde_json::from_slice(&read(&hierarchy.join(format!("{name}.json")))).unwrap();
    assert_eq!(meta["version"], 3);
    if matches!(kind, "source" | "discrete") {
        return Asset {
            source,
            lods,
            lod_triangles,
            clusters: Vec::new(),
            groups: Vec::new(),
            source_triangles: meta["sourceTris"].as_u64().unwrap(),
            uploaded_bytes: 0,
        };
    }
    let bytes = read(&hierarchy.join(format!("{name}.bin")));
    let positions = floats(&bytes, &meta, "positions");
    let normals = section(&bytes, &meta, "normals");
    let uvs = floats(&bytes, &meta, "uvs");
    let indices = integers(&bytes, &meta, "indices");
    let cluster_u = integers(&bytes, &meta, "clusterU");
    let group_f = floats(&bytes, &meta, "groupF");
    let offsets = integers(&bytes, &meta, "memberOff");
    let members = integers(&bytes, &meta, "memberIds");
    // The reference format omits colors/tangents and quantizes normals. Restore the
    // source attributes for this geometry-selection experiment so loss of material
    // attributes cannot masquerade as a cluster-LOD quality/performance tradeoff.
    let gltf = gltf::Gltf::from_slice_without_validation(&read(&source_path)).unwrap();
    assert!(
        gltf.nodes()
            .all(|n| Mat4::from_cols_array_2d(&n.transform().matrix()) == Mat4::IDENTITY),
        "experiment attribute remap requires baked node transforms"
    );
    let buffers: Vec<_> = gltf
        .buffers()
        .map(|b| match b.source() {
            gltf::buffer::Source::Uri(uri) => read(&resolve(&source_path, uri)),
            _ => panic!("external kit buffers required"),
        })
        .collect();
    let primitive = gltf.meshes().next().unwrap().primitives().next().unwrap();
    let reader = primitive.reader(|b| Some(buffers[b.index()].as_slice()));
    let mut original = MeshData {
        positions: reader.read_positions().unwrap().map(Vec3::from).collect(),
        normals: reader
            .read_normals()
            .unwrap()
            .map(|n| Vec3::from(n).normalize_or(Vec3::Y))
            .collect(),
        uvs: reader
            .read_tex_coords(0)
            .unwrap()
            .into_f32()
            .map(Vec2::from)
            .collect(),
        indices: reader.read_indices().unwrap().into_u32().collect(),
        colors: reader
            .read_colors(0)
            .map_or_else(Vec::new, |c| c.into_rgba_f32().map(Vec4::from).collect()),
        ..Default::default()
    };
    original.generate_tangents().expect("source tangent basis");
    let key = |p: Vec3, uv: Vec2| {
        [p.x, p.y, p.z, uv.x, uv.y].map(|v| if v == 0.0 { 0 } else { v.to_bits() })
    };
    let mut attributes: HashMap<_, Vec<usize>> = HashMap::new();
    for (i, (&p, &uv)) in original.positions.iter().zip(&original.uvs).enumerate() {
        attributes.entry(key(p, uv)).or_default().push(i);
    }
    // Derive the tangent basis on the complete leaf topology, then carry it through
    // every cluster. Generating tangents per meshlet introduces shading seams.
    let mut leaf = MeshData {
        positions: positions
            .as_chunks::<3>()
            .0
            .iter()
            .map(|v| Vec3::from_array(*v))
            .collect(),
        normals: normals
            .as_chunks::<4>()
            .0
            .iter()
            .map(|n| Vec3::new(n[0] as i8 as f32, n[1] as i8 as f32, n[2] as i8 as f32) / 127.0)
            .collect(),
        uvs: uvs
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| Vec2::from_array(*v))
            .collect(),
        ..Default::default()
    };
    leaf.tangents.resize(leaf.positions.len(), Vec4::ZERO);
    for i in 0..leaf.positions.len() {
        let candidates = attributes
            .get(&key(leaf.positions[i], leaf.uvs[i]))
            .expect("cluster position/UV matches source");
        let index = *candidates
            .iter()
            .max_by(|&&a, &&b| {
                original.normals[a]
                    .dot(leaf.normals[i])
                    .total_cmp(&original.normals[b].dot(leaf.normals[i]))
            })
            .unwrap();
        leaf.normals[i] = original.normals[index];
        leaf.tangents[i] = original.tangents[index];
        leaf.colors
            .push(original.colors.get(index).copied().unwrap_or(Vec4::ONE));
    }
    let groups = group_f.as_chunks::<5>().0.to_vec();
    let mut consumers = vec![0; cluster_u.len() / 4];
    for (group, range) in offsets.windows(2).enumerate() {
        for &member in &members[range[0] as usize..range[1] as usize] {
            consumers[member as usize] = group;
        }
    }
    let before = world.work().upload_bytes;
    let clusters = cluster_u
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .map(|(c, u)| {
            let mut mesh = MeshData::default();
            let mut remap = HashMap::new();
            for &index in &indices[u[0] as usize..(u[0] + u[1]) as usize] {
                let next = remap.len() as u32;
                let mapped = *remap.entry(index).or_insert_with(|| {
                    let i = index as usize;
                    mesh.positions
                        .push(Vec3::from_slice(&positions[i * 3..i * 3 + 3]));
                    mesh.normals.push(leaf.normals[i]);
                    mesh.uvs.push(Vec2::from_slice(&uvs[i * 2..i * 2 + 2]));
                    mesh.tangents.push(leaf.tangents[i]);
                    mesh.colors.push(leaf.colors[i]);
                    next
                });
                mesh.indices.push(mapped);
            }
            Cluster {
                mesh: world.add_mesh(&mesh),
                producer: u[2],
                consumer: consumers[c],
                triangles: u[1] as u32 / 3,
            }
        })
        .collect();
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    Asset {
        source,
        lods,
        clusters,
        groups,
        source_triangles: meta["sourceTris"].as_u64().unwrap(),
        lod_triangles,
        uploaded_bytes: world.work().upload_bytes - before,
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Source,
    Lod(usize),
    Cluster(f32),
}

impl Mode {
    fn name(self) -> String {
        match self {
            Self::Source => "source".into(),
            Self::Lod(i) => format!("lod{}", i + 1),
            Self::Cluster(e) => format!("cluster-{e}px"),
        }
    }
}

fn grid_camera(frame: usize, count: usize) -> (Vec3, Vec3) {
    // A complete deterministic orbit: identical poses in every mode and round.
    let angle = frame as f32 * std::f32::consts::TAU / 240.0;
    let radius = (count as f32).sqrt() * 9.0 + 15.0;
    (
        Vec3::new(angle.sin() * radius, 18.0, angle.cos() * radius),
        Vec3::new(0.0, 3.0, 0.0),
    )
}

fn camera(frame: usize, count: usize, path: Option<&Json>) -> (Vec3, Vec3) {
    match path {
        None => grid_camera(frame, count),
        Some(path) => {
            let poses = path["frames"].as_array().expect("camera frames");
            let pose = &poses[frame % poses.len()];
            let vector = |key: &str| {
                let v = pose[key].as_array().expect("camera vector");
                Vec3::new(
                    v[0].as_f64().unwrap() as f32,
                    v[1].as_f64().unwrap() as f32,
                    v[2].as_f64().unwrap() as f32,
                )
            };
            (vector("eye"), vector("target"))
        }
    }
}

fn select(
    asset: &Asset,
    mode: Mode,
    positions: &[Mat4],
    eye: Vec3,
    instances: &mut Vec<Instance>,
    errors: &mut Vec<f32>,
) -> u64 {
    instances.clear();
    let mut triangles = 0;
    let k = HEIGHT as f32 / (2.0 * (50f32.to_radians() / 2.0).tan());
    for &position in positions {
        let transform = position;
        match mode {
            Mode::Source | Mode::Lod(_) => {
                triangles += match mode {
                    Mode::Lod(i) => asset.lod_triangles[i],
                    _ => asset.source_triangles,
                };
                let prop = match mode {
                    Mode::Lod(i) => &asset.lods[i],
                    _ => &asset.source,
                };
                for &(mesh, material) in &prop.parts {
                    instances.push(Instance {
                        mesh,
                        material,
                        world_from_local: transform,
                        skin: None,
                    });
                }
            }
            Mode::Cluster(threshold) => {
                errors.clear();
                let scale = transform
                    .x_axis
                    .truncate()
                    .length()
                    .max(transform.y_axis.truncate().length())
                    .max(transform.z_axis.truncate().length());
                errors.extend(asset.groups.iter().map(|g| {
                    let d = eye.distance(transform.transform_point3(Vec3::new(g[0], g[1], g[2])))
                        - g[3] * scale;
                    if d <= 1e-6 {
                        f32::INFINITY
                    } else {
                        g[4] * scale * k / d
                    }
                }));
                for cluster in &asset.clusters {
                    let from =
                        cluster.producer < 0 || errors[cluster.producer as usize] <= threshold;
                    let to = cluster.consumer == 0 || errors[cluster.consumer] > threshold;
                    if from && to {
                        triangles += u64::from(cluster.triangles);
                        instances.push(Instance {
                            mesh: cluster.mesh,
                            material: asset.source.parts[0].1,
                            world_from_local: transform,
                            skin: None,
                        });
                    }
                }
            }
        }
    }
    triangles
}

fn readback(gpu: &Headless, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    rx.recv().unwrap().unwrap();
    let bytes = buffer.slice(..).get_mapped_range().unwrap().to_vec();
    buffer.unmap();
    bytes
}

#[allow(clippy::too_many_arguments)]
fn measure(
    gpu: &Headless,
    world: &World,
    asset: &Asset,
    mode: Mode,
    positions: &[Mat4],
    out: &Path,
    round: usize,
    path: Option<&Json>,
) -> Json {
    let renderer = gpu.renderer();
    let mut viewport = Viewport::new(renderer, WIDTH, HEIGHT);
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("virtual geometry benchmark"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: render_engine::headless::COLOR_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let (eye, target) = camera(0, positions.len(), path);
    let mut scene = daylight(eye, target, 12.0);
    // A fixed sky-free scene isolates geometry and shadows from atmosphere LUT startup.
    scene.sky = None;
    scene.clear_color = Vec3::splat(0.15);
    scene.ambient.brightness = 100.0;
    let mut instances = Vec::new();
    let mut errors = Vec::new();
    let start = Instant::now();
    let mut warm = 0;
    while warm < 60 || start.elapsed() < render_engine::headless::WARMUP {
        (scene.camera.eye, scene.camera.target) = camera(warm % 240, positions.len(), path);
        select(
            asset,
            mode,
            positions,
            scene.camera.eye,
            &mut instances,
            &mut errors,
        );
        renderer.prepare(&mut viewport, world, &scene, &instances);
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        renderer.encode(&mut encoder, &viewport, world, &view, None);
        gpu.queue.submit([encoder.finish()]);
        warm += 1;
        if warm % 32 == 0 {
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
        }
    }
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let per_frame = viewport.timestamp_queries();
    let batch = 16;
    let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: per_frame * batch,
    });
    let size = u64::from(per_frame * batch) * 8;
    let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut cpu = Vec::new();
    let mut selection = Vec::new();
    let mut gpu_times = Vec::new();
    let mut selected_triangles = Vec::new();
    let mut draws = Vec::new();
    let mut submitted_triangles = Vec::new();
    for block in 0..15 {
        for i in 0..batch {
            let frame = block * batch + i;
            let start = Instant::now();
            (scene.camera.eye, scene.camera.target) = camera(frame as usize, positions.len(), path);
            let tris = select(
                asset,
                mode,
                positions,
                scene.camera.eye,
                &mut instances,
                &mut errors,
            );
            let selected = start.elapsed().as_secs_f64() * 1e3;
            renderer.prepare(&mut viewport, world, &scene, &instances);
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            renderer.encode(
                &mut encoder,
                &viewport,
                world,
                &view,
                Some(FrameTimestamps {
                    query_set: &queries,
                    first: i * per_frame,
                }),
            );
            if i + 1 == batch {
                encoder.resolve_query_set(&queries, 0..per_frame * batch, &resolve, 0);
                encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, size);
            }
            gpu.queue.submit([encoder.finish()]);
            cpu.push(start.elapsed().as_secs_f64() * 1e3);
            selection.push(selected);
            selected_triangles.push(tris);
            let work = viewport.work();
            submitted_triangles.push([work.cpu_triangles, work.cpu_shadow_triangles]);
            draws.push([
                work.draws,
                work.shadow_draws,
                work.visible,
                work.shadow_visible,
            ]);
        }
        let bytes = readback(gpu, &read);
        let ticks: Vec<u64> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| u64::from_le_bytes(*b))
            .collect();
        for frame in ticks.chunks_exact(per_frame as usize) {
            let ns: u64 = frame
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| p[1].wrapping_sub(p[0]))
                .sum();
            gpu_times.push(ns as f64 * f64::from(gpu.queue.get_timestamp_period()) / 1e6);
        }
    }
    if round == 0 {
        for frame in [0, 60, 120, 180] {
            (scene.camera.eye, scene.camera.target) = camera(frame, positions.len(), path);
            select(
                asset,
                mode,
                positions,
                scene.camera.eye,
                &mut instances,
                &mut errors,
            );
            let image = gpu
                .render(&mut viewport, world, &scene, &instances)
                .unwrap();
            save(
                &image,
                &out.join(format!("{}-{}-{frame}.png", positions.len(), mode.name())),
            );
        }
    }
    let memory = viewport.memory();
    serde_json::json!({ "mode": mode.name(), "instances": positions.len(), "round": round,
        "cpu_frame_ms": stats(cpu.clone()), "selection_ms": stats(selection.clone()), "gpu_pass_sum_ms": stats(gpu_times.clone()),
        "raw": { "cpu_frame_ms": cpu, "selection_ms": selection, "gpu_pass_sum_ms": gpu_times, "selected_triangles_before_frustum": selected_triangles, "submitted_triangles_main_shadow": submitted_triangles, "draws_main_shadow_visible_main_shadow": draws },
        "viewport_gpu_buffer_bytes": memory.gpu_buffer_bytes, "viewport_cpu_scratch_bytes": memory.cpu_scratch_bytes,
        "cluster_mesh_upload_bytes": asset.uploaded_bytes, "source_triangles_per_asset": asset.source_triangles,
    })
}

pub(super) fn run(args: &[String]) {
    assert!(
        matches!(args.len(), 4 | 5),
        "--virtual-geometry KIT HIERARCHY NAME OUTPUT [source|discrete|cluster]"
    );
    let kind = args.get(4).map_or("all", String::as_str);
    assert!(matches!(kind, "all" | "source" | "discrete" | "cluster"));
    let kit = Path::new(&args[0]);
    let hierarchy = Path::new(&args[1]);
    let name = &args[2];
    let out = Path::new(&args[3]);
    std::fs::create_dir_all(out).unwrap();
    let gpu = Headless::new().expect("hardware adapter with timestamps");
    assert!(
        gpu.adapter_features
            .contains(wgpu::Features::TIMESTAMP_QUERY)
    );
    println!("adapter: {:?}", gpu.adapter);
    let mut world = World::new(gpu.renderer());
    let load_start = Instant::now();
    let asset = load(&gpu, &mut world, kit, hierarchy, name, kind);
    gpu.queue.submit([]);
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let load_ms = load_start.elapsed().as_secs_f64() * 1e3;
    if kind != "all" {
        use std::io::Write;
        let mut viewport = Viewport::new(gpu.renderer(), WIDTH, HEIGHT);
        let (eye, target) = camera(0, 1, None);
        let mut scene = daylight(eye, target, 12.0);
        scene.sky = None;
        let mut instances = Vec::new();
        let mode = if kind == "cluster" {
            Mode::Cluster(2.0)
        } else {
            Mode::Source
        };
        select(
            &asset,
            mode,
            &[Mat4::IDENTITY],
            eye,
            &mut instances,
            &mut Vec::new(),
        );
        for _ in 0..16 {
            gpu.render(&mut viewport, &world, &scene, &instances)
                .unwrap();
        }
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        let memory: Vec<_> = status
            .lines()
            .filter(|l| l.starts_with("VmRSS:") || l.starts_with("VmHWM:"))
            .collect();
        let result = serde_json::json!({"mode":kind,"asset":name,"adapter":format!("{:?}",gpu.adapter),"load_ms":load_ms,"world_upload_bytes":world.work().upload_bytes,"process_memory_kib":memory,"cluster_mesh_upload_bytes":asset.uploaded_bytes,"note":"Fresh process, warm filesystem; load span excludes device setup and the following 16 rendered warm frames. RSS/HWM include allocator and driver state. Discrete retains source and LODs; cluster retains source fallback and all cluster buffers. No streaming or eviction."});
        println!("VG_LOAD {result}");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut String::new()).unwrap();
        return;
    }
    let mut modes = vec![Mode::Source];
    modes.extend((0..asset.lods.len()).map(Mode::Lod));
    modes.extend([Mode::Cluster(1.0), Mode::Cluster(2.0), Mode::Cluster(4.0)]);
    let game_scene: Option<Json> = std::env::var("VG_SCENE")
        .ok()
        .map(|p| serde_json::from_slice(&read(Path::new(&p))).unwrap());
    if let Some(scene) = &game_scene {
        assert_eq!(scene["asset"].as_str(), Some(name.as_str()));
    }
    let counts = game_scene.as_ref().map_or_else(
        || vec![1, 64, 256],
        |s| vec![s["transforms"].as_array().unwrap().len()],
    );
    let mut rows = Vec::new();
    for count in counts {
        assert!(count > 0);

        let side = (count as f32).sqrt().ceil() as usize;
        let positions: Vec<Mat4> = if let Some(scene) = &game_scene {
            scene["transforms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| {
                    let values: Vec<f32> = v
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|n| n.as_f64().unwrap() as f32)
                        .collect();
                    Mat4::from_cols_slice(&values)
                })
                .collect()
        } else {
            (0..count)
                .map(|i| {
                    Mat4::from_translation(
                        Vec3::new(
                            (i % side) as f32 - (side - 1) as f32 / 2.0,
                            0.0,
                            (i / side) as f32 - (side - 1) as f32 / 2.0,
                        ) * 16.0,
                    )
                })
                .collect()
        };
        for round in 0..2 {
            let order: Vec<_> = if round == 0 {
                modes.clone()
            } else {
                modes.iter().rev().copied().collect()
            };
            for mode in order {
                let row = measure(
                    &gpu,
                    &world,
                    &asset,
                    mode,
                    &positions,
                    out,
                    round,
                    game_scene.as_ref(),
                );
                println!(
                    "{name} {count} {} round {round}: CPU {} GPU {}",
                    mode.name(),
                    row["cpu_frame_ms"]["median"],
                    row["gpu_pass_sum_ms"]["median"]
                );
                rows.push(row);
                let report = serde_json::json!({ "asset": name, "adapter": format!("{:?}", gpu.adapter), "resolution": [WIDTH, HEIGHT], "scene_input": game_scene,
                    "scope": "Asset geometry experiment, not whole gameplay. See scene_input for real consumer placements; otherwise repeated grid. CPU selection + prepare + encode + submit; GPU pass sum, no waits in CPU span. Shadows retain off-camera instances. Fixed authored LODs are quality candidates, not assumed equivalent. All modes resident during measurement; upload counter is cluster mesh buffers only, not total residency.", "rows": rows });
                std::fs::write(
                    out.join("report.json"),
                    serde_json::to_vec_pretty(&report).unwrap(),
                )
                .unwrap();
            }
        }
    }
}
