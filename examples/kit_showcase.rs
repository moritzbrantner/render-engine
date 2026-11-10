//! Particle effects on the owner's asset-tooling kits: renders render-engine's particles with
//! asset-tooling's flipbooks (`combat-effects`, `zoo-effects`) and procedural sprites, over the
//! Greyhaven terrain and nature bundles and the crag kit, and measures what each scene costs.
//!
//! ```text
//! bash scripts/fetch-kits.sh                                   # pinned bundles into target/kits/
//! cargo run --release --example kit_showcase                   # every scene
//! cargo run --release --example kit_showcase -- campfire_day   # one scene
//! cargo run --release --example kit_showcase -- --scaling      # particle-count scaling
//! ```
//!
//! Each scene writes `target/kit-showcase/<scene>.png` and a line of `report.json`: alive
//! particles, draws, particle GPU memory, the step's upload, CPU time of a frame's engine calls
//! (`step_particles`, `prepare`, `encode`, submit) and, on an adapter with timestamp queries, the
//! GPU time of the particle simulation and of the whole frame (median, p95, max).
//!
//! The kit loading follows the mmorpg native client (`crates/mmorpg-client/src/nature.rs` and
//! `ground.rs`): glTF read with the `gltf` crate, KTX2 (Basis Universal) textures transcoded to
//! RGBA8 by `basisu`, PNGs box-filtered into a mip chain. It is consumer code: the engine reads no
//! files (technical decisions D7).

use glam::{Mat4, Quat, UVec2, Vec2, Vec3, Vec4};
use render_engine::{
    headless::{Error as HeadlessError, Headless},
    material::{AlphaMode, Sampler, StandardMaterial, TextureData},
    mesh::{MeshData, Shape},
    particles::*,
    renderer::Viewport,
    scene::*,
    world::{EmitterId, Instance, MaterialId, MeshId, TextureId, World},
};
use serde_json::Value as Json;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Instant,
};

/// `--hard`: every emitter without its depth fade, for before/after comparisons.
static HARD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const DT: f32 = 1.0 / 60.0;
/// `Particle` + `Quad` + dead slot + two alive-list entries (`particle_sim.rs`).
const BYTES_PER_SLOT: u64 = 64 + 64 + 4 + 4 + 4;
/// `StepHeader` and one `GpuEmitterStep` per emitter (`particle_sim.rs`).
const STEP_HEADER_BYTES: u64 = 16;
const STEP_EMITTER_BYTES: u64 = 768;

// ---------------------------------------------------------------------------------------------
// Kit loading (consumer side).

fn read(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (run `bash scripts/fetch-kits.sh` first)",
            path.display()
        )
    })
}

fn box_mips(rgba: Vec<u8>, width: u32, height: u32) -> Vec<Vec<u8>> {
    let mut levels = vec![rgba];
    let (mut w, mut h) = (width as usize, height as usize);
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let prev = levels.last().unwrap();
        let mut next = vec![0u8; nw * nh * 4];
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..4 {
                    let mut sum = 0u32;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let sx = (x * 2 + dx).min(w - 1);
                        let sy = (y * 2 + dy).min(h - 1);
                        sum += u32::from(prev[(sy * w + sx) * 4 + c]);
                    }
                    next[(y * nw + x) * 4 + c] = ((sum + 2) / 4) as u8;
                }
            }
        }
        levels.push(next);
        (w, h) = (nw, nh);
    }
    levels
}

fn format(srgb: bool) -> wgpu::TextureFormat {
    if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    }
}

/// A KTX2 (Basis Universal) texture with its own mip chain, or a PNG mipmapped here.
fn decode_texture(bytes: &[u8], srgb: bool, sampler: Sampler) -> TextureData {
    if bytes.starts_with(b"\xABKTX 20\xBB\r\n\x1A\n") {
        let t = basisu::Transcoder::new(bytes).expect("a Basis Universal KTX2");
        let (width, height) = t.base_dimensions();
        let levels = (0..t.level_count())
            .map(|l| {
                t.transcode(l, basisu::TargetFormat::Rgba32, basisu::DecodeFlags::NONE)
                    .expect("transcodes")
            })
            .collect();
        return TextureData {
            width,
            height,
            format: format(srgb),
            levels,
            sampler,
        };
    }
    let image = image::load_from_memory(bytes).expect("PNG").to_rgba8();
    let (width, height) = image.dimensions();
    TextureData {
        width,
        height,
        format: format(srgb),
        levels: box_mips(image.into_raw(), width, height),
        sampler,
    }
}

/// `base`'s directory joined with a relative `uri`.
fn resolve(base: &Path, uri: &str) -> PathBuf {
    let mut out = base.parent().unwrap().to_path_buf();
    for part in uri.split('/') {
        match part {
            ".." => {
                out.pop();
            }
            "." | "" => {}
            p => out.push(p),
        }
    }
    out
}

struct Kit {
    root: PathBuf,
    /// Entry key → its default (full-detail) scene.
    entries: HashMap<String, PathBuf>,
}

impl Kit {
    fn open(root: PathBuf) -> Self {
        let manifest: Json = serde_json::from_slice(&read(&root.join("manifest.json"))).unwrap();
        let mut entries = HashMap::new();
        for a in manifest["assets"].as_array().unwrap() {
            let path = a["path"].as_str().unwrap();
            if path.ends_with("default.gltf") {
                entries.insert(a["key"].as_str().unwrap().to_owned(), root.join(path));
            }
        }
        Self { root, entries }
    }
}

/// A kit entry in the world: its primitives and their bounds (entry space).
#[derive(Clone)]
struct Prop {
    parts: Vec<(MeshId, MaterialId)>,
    /// The materials of `parts`, as added.
    materials: Vec<StandardMaterial>,
    max: Vec3,
}

#[derive(Default)]
struct Textures {
    by_path: HashMap<(PathBuf, bool), TextureId>,
}

fn load_prop(world: &mut World, textures: &mut Textures, kit: &Kit, key: &str) -> Prop {
    let path = kit
        .entries
        .get(key)
        .unwrap_or_else(|| panic!("{} has no entry {key}", kit.root.display()));
    let bytes = read(path);
    // `KHR_texture_basisu` puts a texture's source in an extension, which the gltf crate does not
    // resolve: images are looked up in the JSON (like mmorpg-client's ground.rs).
    let json: Json = serde_json::from_slice(&bytes).unwrap();
    let gltf = gltf::Gltf::from_slice_without_validation(&bytes).expect("glTF");
    let buffers: Vec<Vec<u8>> = gltf
        .buffers()
        .map(|b| match b.source() {
            gltf::buffer::Source::Uri(uri) => read(&resolve(path, uri)),
            gltf::buffer::Source::Bin => panic!("kit entries use external buffers"),
        })
        .collect();
    let mut texture = |index: Option<usize>, srgb: bool, world: &mut World| -> Option<TextureId> {
        let t = &json["textures"][index?];
        let source = t["extensions"]["KHR_texture_basisu"]["source"]
            .as_u64()
            .or_else(|| t["source"].as_u64())?;
        let uri = json["images"][source as usize]["uri"].as_str()?;
        let file = resolve(path, uri);
        Some(
            *textures
                .by_path
                .entry((file.clone(), srgb))
                .or_insert_with(|| {
                    world.add_texture(&decode_texture(&read(&file), srgb, Sampler::default()))
                }),
        )
    };
    let scene = gltf
        .default_scene()
        .or_else(|| gltf.scenes().next())
        .unwrap();
    let mut stack: Vec<(gltf::Node, Mat4)> = scene.nodes().map(|n| (n, Mat4::IDENTITY)).collect();
    let mut max = Vec3::splat(f32::MIN);
    let mut parts = Vec::new();
    let mut materials = Vec::new();
    while let Some((node, parent)) = stack.pop() {
        let local = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        stack.extend(node.children().map(|c| (c, local)));
        let Some(mesh) = node.mesh() else { continue };
        let normal_matrix = local.inverse().transpose();
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|b| buffers.get(b.index()).map(Vec::as_slice));
            let Some(positions) = reader.read_positions() else {
                continue;
            };
            let mut data = MeshData {
                positions: positions
                    .map(|p| local.transform_point3(Vec3::from(p)))
                    .collect(),
                ..MeshData::default()
            };
            for p in &data.positions {
                max = max.max(*p);
            }
            data.normals = reader.read_normals().map_or_else(
                || vec![Vec3::Y; data.positions.len()],
                |n| {
                    n.map(|n| {
                        normal_matrix
                            .transform_vector3(Vec3::from(n))
                            .normalize_or(Vec3::Y)
                    })
                    .collect()
                },
            );
            if let Some(uvs) = reader.read_tex_coords(0) {
                data.uvs = uvs.into_f32().map(Vec2::from).collect();
            }
            if let Some(colors) = reader.read_colors(0) {
                data.colors = colors.into_rgba_f32().map(Vec4::from).collect();
            }
            data.indices = reader.read_indices().map_or_else(
                || (0..data.positions.len() as u32).collect(),
                |i| i.into_u32().collect(),
            );
            let m = primitive.material();
            let pbr = m.pbr_metallic_roughness();
            let normal_map = texture(
                m.normal_texture().map(|t| t.texture().index()),
                false,
                world,
            );
            if normal_map.is_some() && data.generate_tangents().is_err() {
                data.tangents.clear();
            }
            let material = StandardMaterial {
                base_color: Vec4::from(pbr.base_color_factor()),
                base_color_texture: texture(
                    pbr.base_color_texture().map(|t| t.texture().index()),
                    true,
                    world,
                ),
                metallic: pbr.metallic_factor(),
                perceptual_roughness: pbr.roughness_factor(),
                metallic_roughness_texture: texture(
                    pbr.metallic_roughness_texture()
                        .map(|t| t.texture().index()),
                    false,
                    world,
                ),
                normal_map_texture: if data.tangents.is_empty() {
                    None
                } else {
                    normal_map
                },
                occlusion_texture: texture(
                    m.occlusion_texture().map(|t| t.texture().index()),
                    false,
                    world,
                ),
                emissive: Vec3::from(m.emissive_factor()),
                emissive_texture: texture(
                    m.emissive_texture().map(|t| t.texture().index()),
                    true,
                    world,
                ),
                alpha_mode: match m.alpha_mode() {
                    gltf::material::AlphaMode::Mask => {
                        AlphaMode::Mask(m.alpha_cutoff().unwrap_or(0.5))
                    }
                    gltf::material::AlphaMode::Blend => AlphaMode::Blend,
                    gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                },
                double_sided: m.double_sided(),
                ..StandardMaterial::default()
            };
            parts.push((world.add_mesh(&data), world.add_material(&material)));
            materials.push(material);
        }
    }
    Prop {
        parts,
        materials,
        max,
    }
}

/// A Greyhaven terrain layer's material on a `size` m ground plane, its texture repeated every
/// `tile` m. The layer's base colour alpha is a height for blending layers (mmorpg-client
/// ground.rs), not coverage, so the ground is drawn opaque.
fn ground(
    world: &mut World,
    textures: &mut Textures,
    kit: &Kit,
    layer: &str,
    size: f32,
    tile: f32,
) -> Prop {
    let layer = load_prop(world, textures, kit, layer);
    let mut desc = layer.materials[0];
    desc.alpha_mode = AlphaMode::Opaque;
    let material = world.add_material(&desc);
    let mut mesh = MeshData::shape(Shape::Plane(Vec2::splat(size)));
    mesh.uvs = mesh
        .positions
        .iter()
        .map(|p| Vec2::new(p.x, p.z) / tile)
        .collect();
    mesh.generate_tangents().expect("plane tangents");
    Prop {
        parts: vec![(world.add_mesh(&mesh), material)],
        materials: vec![desc],
        max: Vec3::new(size / 2.0, 0.0, size / 2.0),
    }
}

// ---------------------------------------------------------------------------------------------
// Effects (consumer content, built from render-engine's emitter API).

/// A flipbook from an asset-tooling effect release: its texture and its descriptor entry.
struct FlipbookEffect {
    texture: TextureId,
    grid: UVec2,
    frames: u32,
    scale: f32,
    presets: Json,
}

fn load_effects(
    world: &mut World,
    root: &Path,
    descriptor: &str,
) -> HashMap<String, FlipbookEffect> {
    let manifest: Json = serde_json::from_slice(&read(&root.join("manifest.json"))).unwrap();
    let descriptor: Json = serde_json::from_slice(&read(&root.join(descriptor))).unwrap();
    let mut out = HashMap::new();
    for e in descriptor["effects"].as_array().unwrap() {
        let key = e["key"].as_str().unwrap();
        let asset = manifest["assets"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["key"] == key)
            .unwrap();
        let atlas = &e["atlas"];
        // Straight-alpha sRGB atlases, linear filtering, clamped (cells never wrap).
        let sampler = Sampler {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Sampler::default()
        };
        let data = decode_texture(
            &read(&root.join(asset["path"].as_str().unwrap())),
            true,
            sampler,
        );
        out.insert(
            key.trim_end_matches(".flipbook").to_owned(),
            FlipbookEffect {
                texture: world.add_texture(&data),
                grid: UVec2::new(
                    atlas["columns"].as_u64().unwrap() as u32,
                    atlas["rows"].as_u64().unwrap() as u32,
                ),
                frames: atlas["frameCount"].as_u64().unwrap() as u32,
                scale: e["scale"].as_f64().unwrap() as f32,
                presets: e["presets"].clone(),
            },
        );
    }
    out
}

fn f(v: &Json) -> f32 {
    v.as_f64().unwrap() as f32
}

fn v3(v: &Json) -> Vec3 {
    Vec3::new(f(&v[0]), f(&v[1]), f(&v[2]))
}

/// An asset-tooling burst preset (`count`, `direction` (+Y in both kits), `spread`, `speed`,
/// `gravity`, `lifetime`, `endScale`, `fadeOut`, `emitDuration`) as an emitter repeating every
/// `period` s. The preset's cone of half-angle `spread` around +Y is approximated by a box of
/// velocities (render-engine has hanabi's init modifiers, which have no velocity cone).
fn burst(
    fx: &FlipbookEffect,
    preset: &str,
    period: f32,
    blend: ParticleBlend,
    seed: u32,
) -> Emitter {
    let p = &fx.presets[preset];
    let (count, lifetime) = (f(&p["count"]), f(&p["lifetime"]));
    let (s0, s1) = (f(&p["speed"][0]), f(&p["speed"][1]));
    let spread = f(&p["spread"]).min(std::f32::consts::FRAC_PI_2);
    let side = s1 * spread.sin();
    let emit = f(&p["emitDuration"]);
    let fade = p["fadeOut"].as_bool().unwrap();
    Emitter {
        capacity: (count as u32) * (lifetime / period).ceil().max(1.0) as u32 * 2,
        spawner: Spawner {
            count,
            spawn_duration: emit,
            period,
            cycles: 0,
        },
        seed,
        init: vec![
            InitModifier::PositionSphere {
                center: Vec3::ZERO,
                radius: 0.04,
                dimension: ShapeDimension::Volume,
            },
            InitModifier::Velocity(Value::Uniform {
                min: Vec3::new(-side, s0 * spread.cos(), -side),
                max: Vec3::new(side, s1, side),
            }),
            InitModifier::Lifetime(Value::Const(lifetime)),
            InitModifier::Color(Value::Const(Vec4::ONE)),
        ],
        accel: v3(&p["gravity"]),
        size_over_life: Some(Gradient::new(vec![
            (0.0, Vec2::splat(fx.scale)),
            (1.0, Vec2::splat(fx.scale * f(&p["endScale"]))),
        ])),
        color_over_life: fade.then(|| ColorOverLife {
            gradient: Gradient::new(vec![(0.0, Vec4::ONE), (1.0, Vec4::new(1.0, 1.0, 1.0, 0.0))]),
            blend: ColorBlend::Overwrite,
        }),
        orient: Orient::CameraPlane,
        texture: Some(ParticleTexture {
            texture: fx.texture,
            mapping: TextureMapping::Modulate,
            flipbook: Some(Flipbook {
                grid: fx.grid,
                frames: fx.frames,
            }),
        }),
        roundness: 0.0,
        depth_fade: if blend == ParticleBlend::Alpha {
            0.15
        } else {
            0.0
        },
        blend,
        active: true,
    }
}

/// A soft round sprite: white, alpha `(1 − r²)²` with a little low-frequency noise, so puffs do
/// not look like discs. Procedural, consumer-side.
fn puff_texture(world: &mut World) -> TextureId {
    const N: u32 = 64;
    let mut rgba = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        for x in 0..N {
            let p = (Vec2::new(x as f32, y as f32) + 0.5) / N as f32 * 2.0 - 1.0;
            let r2 = p.length_squared();
            let wobble = 1.0 + 0.18 * ((p.x * 5.1).sin() * (p.y * 4.3 + 1.0).cos());
            let a = ((1.0 - r2 * wobble).max(0.0)).powi(2);
            rgba.extend_from_slice(&[255, 255, 255, (a * 255.0).round() as u8]);
        }
    }
    world.add_texture(&TextureData {
        width: N,
        height: N,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        levels: box_mips(rgba, N, N),
        sampler: Sampler {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Sampler::default()
        },
    })
}

fn tex(texture: TextureId) -> Option<ParticleTexture> {
    Some(ParticleTexture {
        texture,
        mapping: TextureMapping::Modulate,
        flipbook: None,
    })
}

fn rate(per_second: f32) -> Spawner {
    Spawner::rate(per_second)
}

fn fire(puff: TextureId, seed: u32) -> Emitter {
    Emitter {
        capacity: 128,
        spawner: rate(110.0),
        seed,
        init: vec![
            InitModifier::PositionSphere {
                center: Vec3::new(0.0, 0.1, 0.0),
                radius: 0.22,
                dimension: ShapeDimension::Volume,
            },
            InitModifier::Velocity(Value::Uniform {
                min: Vec3::new(-0.15, 0.7, -0.15),
                max: Vec3::new(0.15, 1.5, 0.15),
            }),
            InitModifier::Lifetime(Value::Uniform {
                min: 0.45,
                max: 0.8,
            }),
        ],
        accel: Vec3::new(0.0, 0.8, 0.0),
        size_over_life: Some(Gradient::new(vec![
            (0.0, Vec2::splat(0.5)),
            (1.0, Vec2::splat(0.12)),
        ])),
        color_over_life: Some(ColorOverLife {
            gradient: Gradient::new(vec![
                (0.0, Vec4::new(6.0, 2.6, 0.7, 0.9)),
                (0.45, Vec4::new(3.2, 0.9, 0.15, 0.7)),
                (1.0, Vec4::new(0.6, 0.08, 0.0, 0.0)),
            ]),
            blend: ColorBlend::Overwrite,
        }),
        orient: Orient::CameraPlane,
        texture: tex(puff),
        roundness: 0.0,
        depth_fade: 0.35,
        blend: ParticleBlend::Add,
        active: true,
    }
}

fn embers(seed: u32) -> Emitter {
    Emitter {
        capacity: 96,
        spawner: rate(30.0),
        seed,
        init: vec![
            InitModifier::PositionSphere {
                center: Vec3::new(0.0, 0.2, 0.0),
                radius: 0.2,
                dimension: ShapeDimension::Volume,
            },
            InitModifier::Velocity(Value::Uniform {
                min: Vec3::new(-0.6, 1.2, -0.6),
                max: Vec3::new(0.8, 3.0, 0.6),
            }),
            InitModifier::Lifetime(Value::Uniform { min: 1.0, max: 2.2 }),
        ],
        accel: Vec3::new(0.25, -0.6, 0.0),
        size_over_life: Some(Gradient::new(vec![
            (0.0, Vec2::new(0.06, 0.014)),
            (1.0, Vec2::new(0.03, 0.01)),
        ])),
        color_over_life: Some(ColorOverLife {
            gradient: Gradient::new(vec![
                (0.0, Vec4::new(9.0, 3.5, 0.6, 1.0)),
                (1.0, Vec4::new(2.0, 0.25, 0.0, 0.0)),
            ]),
            blend: ColorBlend::Overwrite,
        }),
        orient: Orient::AlongVelocity,
        texture: None,
        roundness: 0.0,
        depth_fade: 0.0,
        blend: ParticleBlend::Add,
        active: true,
    }
}

fn smoke(puff: TextureId, seed: u32) -> Emitter {
    Emitter {
        capacity: 256,
        spawner: rate(24.0),
        seed,
        init: vec![
            InitModifier::PositionSphere {
                center: Vec3::new(0.0, 0.9, 0.0),
                radius: 0.2,
                dimension: ShapeDimension::Volume,
            },
            InitModifier::Velocity(Value::Uniform {
                min: Vec3::new(-0.1, 0.5, -0.1),
                max: Vec3::new(0.1, 0.9, 0.1),
            }),
            InitModifier::Lifetime(Value::Uniform { min: 4.0, max: 6.0 }),
        ],
        // Buoyancy and a light breeze along +x.
        accel: Vec3::new(0.12, 0.05, 0.03),
        size_over_life: Some(Gradient::new(vec![
            (0.0, Vec2::splat(0.5)),
            (1.0, Vec2::splat(2.8)),
        ])),
        color_over_life: Some(ColorOverLife {
            gradient: Gradient::new(vec![
                (0.0, Vec4::new(0.10, 0.09, 0.085, 0.0)),
                (0.08, Vec4::new(0.12, 0.11, 0.10, 0.5)),
                (0.6, Vec4::new(0.28, 0.28, 0.28, 0.25)),
                (1.0, Vec4::new(0.35, 0.35, 0.35, 0.0)),
            ]),
            blend: ColorBlend::Overwrite,
        }),
        orient: Orient::CameraPlane,
        texture: tex(puff),
        roundness: 0.0,
        depth_fade: 0.8,
        blend: ParticleBlend::Alpha,
        active: true,
    }
}

/// Precipitation over a `half`-extent box above the emitter, falling at `velocity`.
fn precipitation(
    count: u32,
    half: Vec3,
    velocity: Vec3,
    jitter: Vec3,
    snow: bool,
    seed: u32,
) -> Emitter {
    let fall = half.y * 2.0 / -velocity.y;
    Emitter {
        capacity: count,
        spawner: rate(count as f32 / fall * 0.98),
        seed,
        init: vec![
            InitModifier::Position(Value::Uniform {
                min: Vec3::new(-half.x, 0.0, -half.z),
                max: Vec3::new(half.x, 2.0 * half.y, half.z),
            }),
            InitModifier::Velocity(Value::Uniform {
                min: velocity - jitter,
                max: velocity + jitter,
            }),
            InitModifier::Lifetime(Value::Const(fall * 0.98)),
        ],
        accel: Vec3::ZERO,
        size_over_life: Some(Gradient::new(vec![(
            0.0,
            if snow {
                Vec2::splat(0.035)
            } else {
                Vec2::new(0.45, 0.012)
            },
        )])),
        color_over_life: Some(ColorOverLife {
            gradient: Gradient::new(if snow {
                vec![
                    (0.0, Vec4::new(0.9, 0.92, 0.95, 0.0)),
                    (0.05, Vec4::new(0.9, 0.92, 0.95, 0.9)),
                    (1.0, Vec4::new(0.9, 0.92, 0.95, 0.9)),
                ]
            } else {
                vec![
                    (0.0, Vec4::new(0.55, 0.6, 0.65, 0.0)),
                    (0.05, Vec4::new(0.55, 0.6, 0.65, 0.35)),
                    (1.0, Vec4::new(0.55, 0.6, 0.65, 0.35)),
                ]
            }),
            blend: ColorBlend::Overwrite,
        }),
        orient: if snow {
            Orient::CameraPlane
        } else {
            Orient::AlongVelocity
        },
        texture: None,
        roundness: if snow { 1.0 } else { 0.0 },
        depth_fade: 0.0,
        blend: ParticleBlend::Alpha,
        active: true,
    }
}

/// Low, slow, large translucent puffs: ground mist or waterfall spray.
fn mist(puff: TextureId, count: u32, half: Vec3, color: Vec4, size: f32, seed: u32) -> Emitter {
    let lifetime = 10.0;
    Emitter {
        capacity: count,
        spawner: rate(count as f32 / lifetime),
        seed,
        init: vec![
            InitModifier::Position(Value::Uniform {
                min: -half,
                max: half,
            }),
            InitModifier::Velocity(Value::Uniform {
                min: Vec3::new(-0.08, -0.01, -0.08),
                max: Vec3::new(0.15, 0.03, 0.08),
            }),
            InitModifier::Lifetime(Value::Const(lifetime)),
            // Start part-way through life so the effect is already established.
            InitModifier::Age(Value::Uniform { min: 0.0, max: 6.0 }),
        ],
        accel: Vec3::ZERO,
        size_over_life: Some(Gradient::new(vec![
            (0.0, Vec2::splat(size * 0.7)),
            (1.0, Vec2::splat(size * 1.3)),
        ])),
        color_over_life: Some(ColorOverLife {
            gradient: Gradient::new(vec![
                (0.0, color.with_w(0.0)),
                (0.3, color),
                (0.7, color),
                (1.0, color.with_w(0.0)),
            ]),
            blend: ColorBlend::Overwrite,
        }),
        orient: Orient::CameraPlane,
        texture: tex(puff),
        roundness: 0.0,
        depth_fade: 1.5,
        blend: ParticleBlend::Alpha,
        active: true,
    }
}

/// A sheet of water falling off a `width` m ledge, pushed out along +z.
fn waterfall(puff: TextureId, width: f32, seed: u32) -> Emitter {
    Emitter {
        capacity: 6000,
        spawner: rate(3200.0),
        seed,
        init: vec![
            InitModifier::Position(Value::Uniform {
                min: Vec3::new(-width / 2.0, -0.05, -0.05),
                max: Vec3::new(width / 2.0, 0.05, 0.05),
            }),
            InitModifier::Velocity(Value::Uniform {
                min: Vec3::new(-0.1, -0.2, 0.9),
                max: Vec3::new(0.1, 0.1, 1.4),
            }),
            InitModifier::Lifetime(Value::Uniform { min: 1.1, max: 1.4 }),
        ],
        accel: Vec3::new(0.0, -9.81, 0.0),
        size_over_life: Some(Gradient::new(vec![
            (0.0, Vec2::splat(0.25)),
            (1.0, Vec2::splat(0.7)),
        ])),
        color_over_life: Some(ColorOverLife {
            gradient: Gradient::new(vec![
                (0.0, Vec4::new(0.55, 0.62, 0.66, 0.5)),
                (0.8, Vec4::new(0.7, 0.75, 0.78, 0.35)),
                (1.0, Vec4::new(0.8, 0.82, 0.85, 0.0)),
            ]),
            blend: ColorBlend::Overwrite,
        }),
        orient: Orient::CameraPlane,
        texture: tex(puff),
        roundness: 0.0,
        depth_fade: 0.6,
        blend: ParticleBlend::Alpha,
        active: true,
    }
}

/// A dense additive cloud: `count` motes leaving a sphere's surface.
fn magic_cloud(count: u32, radius: f32, seed: u32) -> Emitter {
    let lifetime = 3.0;
    Emitter {
        capacity: count,
        spawner: rate(count as f32 / lifetime * 0.98),
        seed,
        init: vec![
            InitModifier::PositionSphere {
                center: Vec3::ZERO,
                radius,
                dimension: ShapeDimension::Surface,
            },
            InitModifier::VelocitySphere {
                center: Vec3::ZERO,
                speed: Value::Uniform {
                    min: -0.25,
                    max: 0.35,
                },
            },
            InitModifier::Lifetime(Value::Uniform {
                min: lifetime * 0.6,
                max: lifetime * 0.98,
            }),
        ],
        accel: Vec3::new(0.0, 0.3, 0.0),
        size_over_life: Some(Gradient::new(vec![
            (0.0, Vec2::splat(0.03)),
            (1.0, Vec2::splat(0.01)),
        ])),
        color_over_life: Some(ColorOverLife {
            gradient: Gradient::new(vec![
                (0.0, Vec4::new(0.3, 1.2, 3.0, 0.0)),
                (0.2, Vec4::new(0.6, 1.5, 4.0, 0.6)),
                (1.0, Vec4::new(1.6, 0.3, 3.0, 0.0)),
            ]),
            blend: ColorBlend::Overwrite,
        }),
        orient: Orient::CameraPlane,
        texture: None,
        roundness: 1.0,
        depth_fade: 0.0,
        blend: ParticleBlend::Add,
        active: true,
    }
}

// ---------------------------------------------------------------------------------------------
// Scenes.

struct Assets {
    terrain: Kit,
    nature: Kit,
    crag: Kit,
    root: PathBuf,
}

struct Built {
    world: World,
    scene: Scene,
    instances: Vec<Instance>,
    /// `(emitter, capacity)`.
    emitters: Vec<(EmitterId, u32)>,
    /// Seconds simulated before the capture.
    warmup: f32,
}

struct Placer<'a> {
    world: &'a mut World,
    textures: Textures,
    props: HashMap<String, Prop>,
    instances: Vec<Instance>,
}

impl Placer<'_> {
    fn prop(&mut self, kit: &Kit, key: &str) -> Prop {
        if let Some(p) = self.props.get(key) {
            return p.clone();
        }
        let p = load_prop(self.world, &mut self.textures, kit, key);
        self.props.insert(key.to_owned(), p.clone());
        p
    }

    /// Places `key` with its base at `at`, turned `yaw` radians and scaled `scale`.
    fn place(&mut self, kit: &Kit, key: &str, at: Vec3, yaw: f32, scale: f32) -> Prop {
        let p = self.prop(kit, key);
        let m = Mat4::from_scale_rotation_translation(
            Vec3::splat(scale),
            Quat::from_rotation_y(yaw),
            at,
        );
        self.add(&p, m);
        p
    }

    fn add(&mut self, p: &Prop, m: Mat4) {
        for &(mesh, material) in &p.parts {
            self.instances.push(Instance {
                mesh,
                material,
                world_from_local: m,
                skin: None,
            });
        }
    }
}

fn daylight(eye: Vec3, target: Vec3, hour: f32) -> Scene {
    let to_sun = DayCycle::default().to_sun(hour);
    Scene {
        camera: Camera {
            eye,
            target,
            fov_y: 50f32.to_radians(),
            near: 0.1,
            ev100: 14.0,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::TonyMcMapface,
        },
        clear_color: Vec3::ZERO,
        sun: Some(Sun {
            direction: -to_sun,
            color: Vec3::ONE,
            illuminance: RAW_SUNLIGHT,
            shadows: Some(SunShadows {
                max_distance: 80.0,
                ..SunShadows::default()
            }),
        }),
        moon: None,
        sky: Some(Sky::default()),
        point_lights: vec![],
        spot_lights: vec![],
        shadow_maps: ShadowMapSizes::default(),
        ambient: Ambient {
            color: Vec3::ONE,
            brightness: 0.0,
        },
        environment: None,
        fog: None,
    }
}

fn night(eye: Vec3, target: Vec3) -> Scene {
    let mut s = daylight(eye, target, 22.5);
    s.sun = None;
    s.moon = Some(Moon {
        direction: -DayCycle::default().to_moon(22.5),
        color: Vec3::new(0.8, 0.85, 1.0),
        illuminance: 0.3,
    });
    s.camera.ev100 = 2.0;
    s.ambient = Ambient {
        color: Vec3::new(0.6, 0.7, 1.0),
        brightness: 1.5,
    };
    s
}

/// A Greyhaven glade below the crag kit's cliffs: grass ground, a gravel patch, conifers,
/// broadleaf trees, rocks and bushes.
fn glade(a: &Assets, p: &mut Placer) {
    let g = ground(
        p.world,
        &mut p.textures,
        &a.terrain,
        "greyhaven.terrain.grass",
        80.0,
        4.0,
    );
    p.add(&g, Mat4::IDENTITY);
    let gravel = ground(
        p.world,
        &mut p.textures,
        &a.terrain,
        "greyhaven.terrain.gravel",
        6.0,
        3.0,
    );
    p.add(&gravel, Mat4::from_translation(Vec3::new(0.0, 0.01, 0.0)));
    for (i, x) in [-9.0f32, -4.5, 0.0, 4.5, 9.0].iter().enumerate() {
        let key = [
            "crag.kit.face-sheer-1",
            "crag.kit.face-climb-1",
            "crag.kit.face-sheer-2",
            "crag.kit.face-climb-2",
            "crag.kit.face-sheer-3",
        ][i];
        p.place(&a.crag, key, Vec3::new(*x, 0.0, -9.0), 0.0, 1.0);
    }
    for (x, z, yaw) in [
        (-7.0, -4.0, 0.3),
        (6.5, -5.0, 1.2),
        (-11.0, 2.0, 2.0),
        (10.0, 1.0, 0.7),
    ] {
        p.place(
            &a.nature,
            "greyhaven.nature.tree.conifer",
            Vec3::new(x, 0.0, z),
            yaw,
            1.0,
        );
    }
    p.place(
        &a.nature,
        "greyhaven.nature.tree.broadleaf",
        Vec3::new(-4.0, 0.0, 6.0),
        0.5,
        1.0,
    );
    p.place(
        &a.nature,
        "greyhaven.nature.rock.1",
        Vec3::new(2.4, 0.0, 1.2),
        0.2,
        0.6,
    );
    p.place(
        &a.nature,
        "greyhaven.nature.rock.2",
        Vec3::new(-2.2, 0.0, -1.5),
        1.1,
        0.5,
    );
    p.place(
        &a.nature,
        "greyhaven.nature.bush.1",
        Vec3::new(3.5, 0.0, -2.5),
        0.0,
        1.0,
    );
    p.place(
        &a.nature,
        "greyhaven.nature.bush.2",
        Vec3::new(-3.6, 0.0, 2.6),
        0.0,
        1.0,
    );
}

/// Campfire stones: a ring of small rocks.
fn fire_ring(a: &Assets, p: &mut Placer, at: Vec3) {
    for i in 0..7 {
        let t = i as f32 / 7.0 * std::f32::consts::TAU;
        p.place(
            &a.nature,
            "greyhaven.nature.rock.3",
            at + Vec3::new(t.cos(), 0.0, t.sin()) * 0.45,
            t,
            0.12,
        );
    }
}

fn build(name: &str, a: &Assets, gpu: &Headless) -> Option<Built> {
    let mut world = World::new(gpu.renderer());
    let mut p = Placer {
        world: &mut world,
        textures: Textures::default(),
        props: HashMap::new(),
        instances: Vec::new(),
    };
    let fire_at = Vec3::new(0.0, 0.0, 0.0);
    let mut emitters: Vec<(Emitter, Mat4)> = Vec::new();
    let mut warmup = 6.0;
    let scene = match name {
        "campfire_day" | "campfire_night" => {
            glade(a, &mut p);
            fire_ring(a, &mut p, fire_at);
            let puff = puff_texture(p.world);
            let at = Mat4::from_translation(fire_at);
            emitters.push((smoke(puff, 11), at));
            emitters.push((fire(puff, 12), at));
            emitters.push((embers(13), at));
            let (eye, target) = (Vec3::new(3.2, 1.7, 4.6), Vec3::new(0.0, 1.2, 0.0));
            let mut s = if name == "campfire_day" {
                daylight(eye, target, 16.5)
            } else {
                night(eye, target)
            };
            s.point_lights.push(PointLight {
                position: fire_at + Vec3::new(0.0, 0.5, 0.0),
                color: Vec3::new(1.0, 0.55, 0.2),
                intensity: if name == "campfire_day" {
                    40_000.0
                } else {
                    3_000.0
                },
                range: 15.0,
                radius: 0.2,
                shadows: Some(LightShadows::POINT),
            });
            s
        }
        "combat_bursts" | "zoo_bursts" => {
            glade(a, &mut p);
            let (root, descriptor, list): (&str, &str, Vec<(&str, &str, ParticleBlend)>) =
                if name == "combat_bursts" {
                    (
                        "combat-effects",
                        "combat-effects.descriptor.json",
                        vec![
                            ("spark", "standard", ParticleBlend::Add),
                            ("block-spark", "heavy", ParticleBlend::Add),
                            ("impact-dust", "standard", ParticleBlend::Alpha),
                            ("trail", "standard", ParticleBlend::Alpha),
                        ],
                    )
                } else {
                    (
                        "zoo-effects",
                        "zoo-effects.descriptor.json",
                        vec![
                            ("dust", "standard", ParticleBlend::Alpha),
                            ("leaf", "standard", ParticleBlend::Alpha),
                            ("splash", "standard", ParticleBlend::Alpha),
                        ],
                    )
                };
            let fx = load_effects(p.world, &a.root.join(root), descriptor);
            let n = list.len() as f32;
            for (i, (key, preset, blend)) in list.into_iter().enumerate() {
                let x = (i as f32 - (n - 1.0) / 2.0) * 1.1;
                // Each burst repeats every 0.6 s; the capture lands mid-burst.
                emitters.push((
                    burst(&fx[key], preset, 0.6, blend, 100 + i as u32),
                    Mat4::from_translation(Vec3::new(x, 0.25, 0.0)),
                ));
            }
            warmup = 0.6 * 4.0 + 0.12;
            daylight(Vec3::new(0.0, 1.0, 2.6), Vec3::new(0.0, 0.4, 0.0), 15.0)
        }
        "rain" | "snow" => {
            glade(a, &mut p);
            let (eye, target) = (Vec3::new(6.0, 1.8, 9.0), Vec3::new(0.0, 1.5, -2.0));
            let mut s = daylight(eye, target, 11.0);
            s.sun.as_mut().unwrap().illuminance = 12_000.0;
            s.camera.ev100 = 11.5;
            s.fog = Some(Fog {
                color: Vec4::new(0.55, 0.58, 0.62, 1.0),
                falloff: FogFalloff::from_visibility(if name == "rain" { 60.0 } else { 45.0 }),
                ..Fog::default()
            });
            let half = Vec3::new(14.0, 5.0, 14.0);
            let around = Mat4::from_translation(Vec3::new(eye.x * 0.5, 0.0, eye.z * 0.5 - 4.0));
            if name == "rain" {
                emitters.push((
                    precipitation(
                        60_000,
                        half,
                        Vec3::new(0.6, -8.5, 0.3),
                        Vec3::new(0.1, 0.6, 0.1),
                        false,
                        21,
                    ),
                    around,
                ));
            } else {
                emitters.push((
                    precipitation(
                        60_000,
                        half,
                        Vec3::new(0.25, -0.9, 0.1),
                        Vec3::new(0.25, 0.25, 0.25),
                        true,
                        22,
                    ),
                    around,
                ));
            }
            let puff = puff_texture(p.world);
            emitters.push((
                mist(
                    puff,
                    220,
                    Vec3::new(12.0, 0.4, 8.0),
                    Vec4::new(0.6, 0.63, 0.67, 0.16),
                    4.0,
                    23,
                ),
                Mat4::from_translation(Vec3::new(0.0, 0.5, -3.0)),
            ));
            warmup = 12.0;
            s
        }
        "waterfall" => {
            let g = ground(
                p.world,
                &mut p.textures,
                &a.terrain,
                "greyhaven.terrain.rock",
                60.0,
                3.0,
            );
            p.add(&g, Mat4::IDENTITY);
            let cliff = p.place(
                &a.crag,
                "crag.kit.face-sheer-2",
                Vec3::new(0.0, 0.0, -2.0),
                0.0,
                1.0,
            );
            p.place(
                &a.crag,
                "crag.kit.face-sheer-1",
                Vec3::new(-4.5, 0.0, -2.0),
                0.0,
                1.0,
            );
            p.place(
                &a.crag,
                "crag.kit.face-sheer-3",
                Vec3::new(4.5, 0.0, -2.0),
                0.0,
                1.0,
            );
            p.place(
                &a.nature,
                "greyhaven.nature.rock.1",
                Vec3::new(2.0, 0.0, 1.5),
                0.4,
                0.7,
            );
            p.place(
                &a.nature,
                "greyhaven.nature.rock.2",
                Vec3::new(-2.3, 0.0, 1.8),
                2.0,
                0.6,
            );
            let top = Vec3::new(0.0, cliff.max.y - 0.2, -2.0 + cliff.max.z + 0.1);
            let puff = puff_texture(p.world);
            emitters.push((waterfall(puff, 1.6, 31), Mat4::from_translation(top)));
            let fx = load_effects(
                p.world,
                &a.root.join("zoo-effects"),
                "zoo-effects.descriptor.json",
            );
            let landing = Vec3::new(0.0, 0.05, top.z + 1.3);
            for (i, dx) in [-0.5f32, 0.0, 0.5].iter().enumerate() {
                emitters.push((
                    burst(
                        &fx["splash"],
                        "standard",
                        0.17 + 0.05 * i as f32,
                        ParticleBlend::Alpha,
                        40 + i as u32,
                    ),
                    Mat4::from_translation(landing + Vec3::new(*dx, 0.0, 0.0)),
                ));
            }
            emitters.push((
                mist(
                    puff,
                    160,
                    Vec3::new(2.0, 0.6, 1.2),
                    Vec4::new(0.8, 0.84, 0.88, 0.12),
                    2.2,
                    33,
                ),
                Mat4::from_translation(landing + Vec3::new(0.0, 0.6, 0.3)),
            ));
            daylight(
                Vec3::new(3.0, 2.0, top.z + 7.5),
                Vec3::new(0.0, top.y * 0.5, top.z),
                14.0,
            )
        }
        "magic_cloud" => {
            glade(a, &mut p);
            emitters.push((
                magic_cloud(200_000, 1.1, 51),
                Mat4::from_translation(Vec3::new(0.0, 1.5, 0.0)),
            ));
            warmup = 4.0;
            night(Vec3::new(2.8, 1.8, 4.2), Vec3::new(0.0, 1.3, 0.0))
        }
        _ => return None,
    };
    let instances = std::mem::take(&mut p.instances);
    let ids = emitters
        .into_iter()
        .map(|(mut e, m)| {
            if HARD.load(std::sync::atomic::Ordering::Relaxed) {
                e.depth_fade = 0.0;
            }
            let id = world.add_emitter(&e);
            world.set_emitter_transform(id, m);
            (id, e.capacity)
        })
        .collect();
    Some(Built {
        world,
        scene,
        instances,
        emitters: ids,
        warmup,
    })
}

const SCENES: [&str; 8] = [
    "campfire_day",
    "campfire_night",
    "combat_bursts",
    "zoo_bursts",
    "rain",
    "snow",
    "waterfall",
    "magic_cloud",
];

// ---------------------------------------------------------------------------------------------
// Measurement.

fn stats(mut v: Vec<f64>) -> Json {
    if v.is_empty() {
        return Json::Null;
    }
    v.sort_by(f64::total_cmp);
    let at = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    serde_json::json!({ "median": at(0.5), "p95": at(0.95), "max": v[v.len() - 1] })
}

fn step(gpu: &Headless, world: &mut World, seconds: f32) {
    let steps = (seconds / DT).round() as usize;
    for chunk in (0..steps).collect::<Vec<_>>().chunks(32) {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for _ in chunk {
            world.step_particles(&mut encoder, DT, None);
        }
        gpu.queue.submit([encoder.finish()]);
    }
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
}

/// CPU milliseconds of a frame's engine calls, frame by frame: step, prepare, encode, submit
/// (waiting for the GPU happens outside the timed span).
fn cpu_frames(gpu: &Headless, viewport: &mut Viewport, b: &mut Built, frames: usize) -> Vec<f64> {
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("showcase target"),
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
    let r = gpu.renderer();
    (0..frames)
        .map(|_| {
            let start = Instant::now();
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            b.world.step_particles(&mut encoder, DT, None);
            r.prepare(viewport, &b.world, &b.scene, &b.instances);
            r.encode(&mut encoder, viewport, &b.world, &view, None);
            gpu.queue.submit([encoder.finish()]);
            let ms = start.elapsed().as_secs_f64() * 1e3;
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            ms
        })
        .collect()
}

fn measure(gpu: &Headless, b: &mut Built, frames: usize) -> Json {
    let mut viewport = Viewport::new(gpu.renderer(), WIDTH, HEIGHT);
    let cpu = cpu_frames(gpu, &mut viewport, b, frames);
    let alive: u32 = b.world.read_alive_particles().iter().map(|(_, n)| n).sum();
    let capacity: u64 = b
        .emitters
        .iter()
        .map(|&(_, capacity)| u64::from(capacity))
        .sum();
    let work = viewport.work();
    let gpu_times = match gpu.gpu_frame_times(
        &mut viewport,
        &mut b.world,
        &b.scene,
        &b.instances,
        frames,
        Some(DT),
    ) {
        Ok(t) => serde_json::json!({
            "frame_ms": stats(t.total),
            "particle_sim_ms": stats(t.particles),
        }),
        Err(HeadlessError::NoTimestamps) => Json::from("no timestamp queries on this adapter"),
        Err(e) => Json::from(e.to_string()),
    };
    serde_json::json!({
        "alive": alive,
        "emitters": b.emitters.len(),
        "particle_draws": b.emitters.len(),
        "mesh_draws": work.draws,
        "shadow_draws": work.shadow_draws,
        "particle_gpu_bytes": capacity * BYTES_PER_SLOT,
        "step_upload_bytes": STEP_HEADER_BYTES + STEP_EMITTER_BYTES * b.emitters.len() as u64,
        "cpu_frame_ms": stats(cpu),
        "gpu": gpu_times,
    })
}

/// `n` timed frames on a GPU; a few on a software adapter (lavapipe takes about a second a
/// frame, and its GPU times say nothing about a GPU's).
fn frames(gpu: &Headless, n: usize) -> usize {
    if gpu.adapter.device_type == wgpu::DeviceType::Cpu {
        6
    } else {
        n
    }
}

fn save(frame: &render_engine::headless::Frame, path: &Path) {
    image::save_buffer(
        path,
        &frame.rgba,
        frame.width,
        frame.height,
        image::ColorType::Rgba8,
    )
    .expect("write png");
}

/// Rain at increasing counts in the campfire glade: CPU and GPU cost per alive particle.
fn scaling(gpu: &Headless, a: &Assets, out: &Path) -> Json {
    let mut rows = Vec::new();
    for count in [1_000u32, 10_000, 100_000, 300_000, 1_000_000] {
        let mut b = build("rain", a, gpu).unwrap();
        // Replace the scene's emitters by one rain emitter of `count`.
        for (id, _) in b.emitters.drain(..) {
            b.world.remove_emitter(id);
        }
        let half = Vec3::new(14.0, 5.0, 14.0);
        let id = b.world.add_emitter(&precipitation(
            count,
            half,
            Vec3::new(0.6, -8.5, 0.3),
            Vec3::new(0.1, 0.6, 0.1),
            false,
            7,
        ));
        b.world
            .set_emitter_transform(id, Mat4::from_translation(Vec3::new(3.0, 0.0, 0.5)));
        b.emitters.push((id, count));
        step(gpu, &mut b.world, 1.5);
        let mut row = measure(gpu, &mut b, frames(gpu, 120));
        row["count"] = count.into();
        println!("rain {count:>8}: {row}");
        if count == 100_000 {
            let mut viewport = Viewport::new(gpu.renderer(), WIDTH, HEIGHT);
            save(
                &gpu.render(&mut viewport, &b.world, &b.scene, &b.instances)
                    .unwrap(),
                &out.join("rain_100k.png"),
            );
        }
        rows.push(row);
    }
    // Many small emitters: the combat spark burst (10 sprites every 0.6 s) at `n` places, the
    // case of a crowd fight. Each emitter is a draw, a step upload and a texture bind group.
    let mut emitter_rows = Vec::new();
    for n in [1u32, 10, 100, 500, 1000] {
        let mut b = build("rain", a, gpu).unwrap();
        for (id, _) in b.emitters.drain(..) {
            b.world.remove_emitter(id);
        }
        let fx = load_effects(
            &mut b.world,
            &a.root.join("combat-effects"),
            "combat-effects.descriptor.json",
        );
        let side = (n as f32).sqrt().ceil() as u32;
        for i in 0..n {
            let e = burst(&fx["spark"], "standard", 0.6, ParticleBlend::Add, 1000 + i);
            let at = Vec3::new(
                (i % side) as f32 * 0.6 - side as f32 * 0.3,
                0.3,
                -((i / side) as f32) * 0.6,
            );
            let id = b.world.add_emitter(&e);
            b.world
                .set_emitter_transform(id, Mat4::from_translation(at));
            b.emitters.push((id, e.capacity));
        }
        step(gpu, &mut b.world, 1.0);
        let mut row = measure(gpu, &mut b, frames(gpu, 120));
        row["emitter_count"] = n.into();
        println!("bursts {n:>5}: {row}");
        emitter_rows.push(row);
    }
    serde_json::json!({ "rain_count": rows, "burst_emitters": emitter_rows })
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(at) = args.iter().position(|a| a == "--hard") {
        args.remove(at);
        HARD.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let kits = repo.join("target/kits");
    let out = repo.join(if HARD.load(std::sync::atomic::Ordering::Relaxed) {
        "target/kit-showcase/hard"
    } else {
        "target/kit-showcase"
    });
    std::fs::create_dir_all(&out).unwrap();
    let a = Assets {
        terrain: Kit::open(kits.join("greyhaven-terrain")),
        nature: Kit::open(kits.join("greyhaven-nature")),
        crag: Kit::open(kits.join("crag-kit")),
        root: kits.clone(),
    };
    let gpu = Headless::new().expect("a wgpu adapter");
    println!("adapter: {} ({:?})", gpu.adapter.name, gpu.adapter.backend);
    let mut report = serde_json::Map::new();
    report.insert("adapter".into(), gpu.adapter.name.clone().into());
    if let Some(at) = args.iter().position(|a| a == "--record") {
        // `--record <scene> [frames]`: one PNG per 1/30 s after the warmup, for a short clip
        // (`ffmpeg -framerate 30 -i <scene>/%03d.png <scene>.webp`).
        let name = args[at + 1].as_str();
        let n: usize = args.get(at + 2).map_or(60, |n| n.parse().unwrap());
        let mut b = build(name, &a, &gpu).expect("a scene name");
        step(&gpu, &mut b.world, b.warmup);
        let dir = out.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        let mut viewport = Viewport::new(gpu.renderer(), WIDTH, HEIGHT);
        for i in 0..n {
            step(&gpu, &mut b.world, 2.0 * DT);
            let frame = gpu
                .render(&mut viewport, &b.world, &b.scene, &b.instances)
                .expect("render");
            save(&frame, &dir.join(format!("{i:03}.png")));
        }
        println!("recorded {n} frames into {}", dir.display());
        return;
    }
    if args.iter().any(|a| a == "--scaling") {
        report.insert("scaling".into(), scaling(&gpu, &a, &out));
    } else {
        let only: Vec<&str> = args.iter().map(String::as_str).collect();
        for name in SCENES
            .iter()
            .filter(|n| only.is_empty() || only.contains(n))
        {
            let t0 = Instant::now();
            let mut b = build(name, &a, &gpu).unwrap();
            let load_s = t0.elapsed().as_secs_f64();
            step(&gpu, &mut b.world, b.warmup);
            let mut viewport = Viewport::new(gpu.renderer(), WIDTH, HEIGHT);
            let frame = gpu
                .render(&mut viewport, &b.world, &b.scene, &b.instances)
                .expect("render");
            save(&frame, &out.join(format!("{name}.png")));
            let mut row = measure(&gpu, &mut b, frames(&gpu, 240));
            row["load_s"] = load_s.into();
            println!("{name}: {row}");
            report.insert((*name).into(), row);
        }
    }
    std::fs::write(
        out.join(if args.iter().any(|a| a == "--scaling") {
            "scaling.json"
        } else {
            "report.json"
        }),
        serde_json::to_string_pretty(&Json::Object(report)).unwrap(),
    )
    .unwrap();
    println!("wrote {}", out.display());
}
