//! A scenario's geometry: objects (a built-in shape or a mesh, a transform and a material) over
//! shared mesh and texture data, some of it read from glTF files. Both sides upload the same
//! data: `upload` into a render-engine `World`, `bevy_side` into Bevy assets.

use render_engine::{
    glam::{Mat4, Quat, Vec2, Vec3, Vec4},
    material::{AlphaMode, Sampler, StandardMaterial, TextureData},
    mesh::{MeshData, Shape},
    renderer::Renderer,
    wgpu,
    world::{Instance, TextureId, World},
};
use std::{collections::HashMap, path::Path};

#[derive(Clone, Copy, Debug)]
pub enum Geometry {
    Shape(Shape),
    /// Index into [`Assets::meshes`].
    Mesh(usize),
}

/// A `StandardMaterial` whose textures index [`Assets::textures`]. The defaults are Bevy's.
#[derive(Clone, Copy, Debug)]
pub struct Material {
    /// Linear RGB.
    pub base_color: Vec3,
    pub alpha: f32,
    pub metallic: f32,
    pub perceptual_roughness: f32,
    /// Dielectric specular reflectance, 0.5 = 4% F0.
    pub reflectance: f32,
    pub emissive: Vec3,
    pub base_color_texture: Option<usize>,
    pub emissive_texture: Option<usize>,
    pub metallic_roughness_texture: Option<usize>,
    pub normal_map_texture: Option<usize>,
    pub occlusion_texture: Option<usize>,
    pub alpha_mode: AlphaMode,
    pub double_sided: bool,
    pub unlit: bool,
}

impl Default for Material {
    fn default() -> Self {
        let m = StandardMaterial::default();
        Self {
            base_color: m.base_color.truncate(),
            alpha: m.base_color.w,
            metallic: m.metallic,
            perceptual_roughness: m.perceptual_roughness,
            reflectance: m.reflectance,
            emissive: m.emissive,
            base_color_texture: None,
            emissive_texture: None,
            metallic_roughness_texture: None,
            normal_map_texture: None,
            occlusion_texture: None,
            alpha_mode: m.alpha_mode,
            double_sided: m.double_sided,
            unlit: m.unlit,
        }
    }
}

impl Material {
    /// The engine material, with texture indices mapped to the uploaded handles.
    fn engine(&self, textures: &[TextureId]) -> StandardMaterial {
        let t = |i: Option<usize>| i.map(|i| textures[i]);
        StandardMaterial {
            base_color: self.base_color.extend(self.alpha),
            base_color_texture: t(self.base_color_texture),
            emissive: self.emissive,
            emissive_texture: t(self.emissive_texture),
            metallic: self.metallic,
            perceptual_roughness: self.perceptual_roughness,
            metallic_roughness_texture: t(self.metallic_roughness_texture),
            reflectance: self.reflectance,
            normal_map_texture: t(self.normal_map_texture),
            occlusion_texture: t(self.occlusion_texture),
            alpha_mode: self.alpha_mode,
            double_sided: self.double_sided,
            unlit: self.unlit,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MeshInstance {
    pub geometry: Geometry,
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub material: Material,
    /// Index into [`Assets::skins`] for a skinned mesh.
    pub skin: Option<usize>,
}

impl MeshInstance {
    pub fn new(shape: Shape, translation: Vec3, material: Material) -> Self {
        Self {
            geometry: Geometry::Shape(shape),
            translation,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            material,
            skin: None,
        }
    }

    pub fn rotated(mut self, rotation: Quat) -> Self {
        self.rotation = rotation;
        self
    }

    /// Bevy `Transform::to_matrix`.
    pub fn transform(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

/// A glTF skin in one pose: per joint, its transform relative to the skinned mesh and its
/// inverse bind matrix.
pub struct Skin {
    pub model_from_joint: Vec<Mat4>,
    pub inverse_bindposes: Vec<Mat4>,
}

impl Skin {
    /// render-engine's joint palette (`World::add_skin`).
    pub fn palette(&self) -> Vec<Mat4> {
        self.model_from_joint
            .iter()
            .zip(&self.inverse_bindposes)
            .map(|(j, ib)| *j * *ib)
            .collect()
    }
}

/// Mesh and texture data shared by a scenario's objects.
#[derive(Default)]
pub struct Assets {
    pub meshes: Vec<MeshData>,
    pub textures: Vec<TextureData>,
    pub skins: Vec<Skin>,
    /// Texture index per (image file, sRGB), so props sharing a texture share the upload.
    loaded: HashMap<(std::path::PathBuf, bool), usize>,
}

/// One glTF primitive: a mesh and its material.
pub struct Primitive {
    pub mesh: usize,
    pub material: Material,
    /// The node's transform in the glTF scene.
    pub node_transform: Mat4,
}

impl Assets {
    /// Reads every mesh primitive of a glTF file's default scene, its material and textures
    /// (with a box-filtered mip chain each), the way Bevy's glTF loader maps them. Tangents are
    /// generated with MikkTSpace where a normal map needs them and the file has none (as Bevy's
    /// loader does).
    pub fn load_gltf(&mut self, path: &Path) -> Vec<Primitive> {
        let (doc, buffers, images) =
            gltf::import(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let dir = path.parent().expect("file in a directory");
        let mut out = Vec::new();
        let scene = doc.default_scene().expect("glTF default scene");
        let mut stack: Vec<(gltf::Node, Mat4)> =
            scene.nodes().map(|n| (n, Mat4::IDENTITY)).collect();
        while let Some((node, parent)) = stack.pop() {
            let transform = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
            stack.extend(node.children().map(|c| (c, transform)));
            let Some(mesh) = node.mesh() else { continue };
            for prim in mesh.primitives() {
                let r = prim.reader(|b| Some(&buffers[b.index()]));
                let mut data = MeshData {
                    positions: r
                        .read_positions()
                        .expect("POSITION")
                        .map(Vec3::from)
                        .collect(),
                    normals: r.read_normals().expect("NORMAL").map(Vec3::from).collect(),
                    uvs: r
                        .read_tex_coords(0)
                        .map(|t| t.into_f32().map(Vec2::from).collect())
                        .unwrap_or_default(),
                    tangents: r
                        .read_tangents()
                        .map(|t| t.map(Vec4::from).collect())
                        .unwrap_or_default(),
                    colors: r
                        .read_colors(0)
                        .map(|c| c.into_rgba_f32().map(Vec4::from).collect())
                        .unwrap_or_default(),
                    joints: r
                        .read_joints(0)
                        .map(|j| j.into_u16().collect())
                        .unwrap_or_default(),
                    weights: r
                        .read_weights(0)
                        .map(|w| w.into_f32().map(Vec4::from).collect())
                        .unwrap_or_default(),
                    indices: r.read_indices().expect("indexed").into_u32().collect(),
                };
                let m = prim.material();
                let pbr = m.pbr_metallic_roughness();
                let mut texture = |t: gltf::Texture, srgb: bool| {
                    let gltf::image::Source::Uri { uri, .. } = t.source().source() else {
                        panic!("{}: embedded images are not supported", path.display());
                    };
                    let key = (dir.join(uri), srgb);
                    if let Some(&i) = self.loaded.get(&key) {
                        return i;
                    }
                    let image = &images[t.source().index()];
                    self.textures
                        .push(texture_data(image, srgb, sampler(&t.sampler())));
                    let i = self.textures.len() - 1;
                    self.loaded.insert(key, i);
                    i
                };
                let [r_, g, b, a] = pbr.base_color_factor();
                let material = Material {
                    base_color: Vec3::new(r_, g, b),
                    alpha: a,
                    metallic: pbr.metallic_factor(),
                    perceptual_roughness: pbr.roughness_factor(),
                    emissive: Vec3::from(m.emissive_factor()),
                    base_color_texture: pbr
                        .base_color_texture()
                        .map(|i| texture(i.texture(), true)),
                    emissive_texture: m.emissive_texture().map(|i| texture(i.texture(), true)),
                    metallic_roughness_texture: pbr
                        .metallic_roughness_texture()
                        .map(|i| texture(i.texture(), false)),
                    normal_map_texture: m.normal_texture().map(|i| texture(i.texture(), false)),
                    occlusion_texture: m.occlusion_texture().map(|i| texture(i.texture(), false)),
                    alpha_mode: match m.alpha_mode() {
                        gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                        gltf::material::AlphaMode::Mask => {
                            AlphaMode::Mask(m.alpha_cutoff().unwrap_or(0.5))
                        }
                        gltf::material::AlphaMode::Blend => AlphaMode::Blend,
                    },
                    double_sided: m.double_sided(),
                    ..Material::default()
                };
                if material.normal_map_texture.is_some() && data.tangents.is_empty() {
                    data.generate_tangents()
                        .expect("tangents for the normal map");
                }
                self.meshes.push(data);
                out.push(Primitive {
                    mesh: self.meshes.len() - 1,
                    material,
                    node_transform: transform,
                });
            }
        }
        out
    }

    /// Poses a glTF file's first skin: each named joint turns by its rotation on top of its rest
    /// rotation, and every joint's transform follows from the node hierarchy (the skinned mesh
    /// sits at the scene root, so these are relative to it). Returns the index into
    /// [`Assets::skins`].
    pub fn load_skin(&mut self, path: &Path, pose: &[(&str, Quat)]) -> usize {
        let (doc, buffers, _) =
            gltf::import(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let skin = doc.skins().next().expect("a glTF skin");
        let mut global = vec![Mat4::IDENTITY; doc.nodes().len()];
        let scene = doc.default_scene().expect("glTF default scene");
        let mut stack: Vec<(gltf::Node, Mat4)> =
            scene.nodes().map(|n| (n, Mat4::IDENTITY)).collect();
        while let Some((node, parent)) = stack.pop() {
            let (t, r, s) = node.transform().decomposed();
            let mut rotation = Quat::from_array(r);
            if let Some((_, q)) = pose.iter().find(|(name, _)| Some(*name) == node.name()) {
                rotation *= *q;
            }
            let local = Mat4::from_scale_rotation_translation(s.into(), rotation, t.into());
            let transform = parent * local;
            global[node.index()] = transform;
            stack.extend(node.children().map(|c| (c, transform)));
        }
        for (name, _) in pose {
            assert!(
                skin.joints().any(|j| j.name() == Some(*name)),
                "{}: no joint {name}",
                path.display()
            );
        }
        let inverse_bindposes = skin
            .reader(|b| Some(&buffers[b.index()]))
            .read_inverse_bind_matrices()
            .expect("inverse bind matrices")
            .map(|m| Mat4::from_cols_array_2d(&m))
            .collect();
        self.skins.push(Skin {
            model_from_joint: skin.joints().map(|j| global[j.index()]).collect(),
            inverse_bindposes,
        });
        self.skins.len() - 1
    }
}

/// Bevy's glTF loader sampler mapping (`texture_sampler`).
fn sampler(s: &gltf::texture::Sampler) -> Sampler {
    use gltf::texture::{MagFilter, MinFilter, WrappingMode};
    let address = |w: WrappingMode| match w {
        WrappingMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        WrappingMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
        WrappingMode::Repeat => wgpu::AddressMode::Repeat,
    };
    let mut out = Sampler {
        address_mode_u: address(s.wrap_s()),
        address_mode_v: address(s.wrap_t()),
        ..Sampler::default()
    };
    if let Some(f) = s.mag_filter() {
        out.mag_filter = match f {
            MagFilter::Nearest => wgpu::FilterMode::Nearest,
            MagFilter::Linear => wgpu::FilterMode::Linear,
        };
    }
    if let Some(f) = s.min_filter() {
        use MinFilter::*;
        out.min_filter = match f {
            Nearest | NearestMipmapNearest | NearestMipmapLinear => wgpu::FilterMode::Nearest,
            Linear | LinearMipmapNearest | LinearMipmapLinear => wgpu::FilterMode::Linear,
        };
        out.mipmap_filter = match f {
            Nearest | Linear | NearestMipmapNearest | LinearMipmapNearest => {
                wgpu::MipmapFilterMode::Nearest
            }
            NearestMipmapLinear | LinearMipmapLinear => wgpu::MipmapFilterMode::Linear,
        };
    }
    out
}

/// RGBA8 texels with a mip chain down to 1×1, each level a 2×2 box filter of the one above.
fn texture_data(image: &gltf::image::Data, srgb: bool, sampler: Sampler) -> TextureData {
    use gltf::image::Format;
    let channels = match image.format {
        Format::R8 => 1,
        Format::R8G8 => 2,
        Format::R8G8B8 => 3,
        Format::R8G8B8A8 => 4,
        f => panic!("unsupported glTF image format {f:?}"),
    };
    let rgba: Vec<u8> = image
        .pixels
        .chunks(channels)
        .flat_map(|p| match channels {
            1 => [p[0], p[0], p[0], 255],
            2 => [p[0], p[1], 0, 255],
            3 => [p[0], p[1], p[2], 255],
            _ => [p[0], p[1], p[2], p[3]],
        })
        .collect();
    let (mut w, mut h) = (image.width, image.height);
    let mut levels = vec![rgba];
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let prev = levels.last().unwrap();
        let mut next = Vec::with_capacity((nw * nh * 4) as usize);
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..4 {
                    let mut sum = 0u32;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let sx = (2 * x + dx).min(w - 1);
                        let sy = (2 * y + dy).min(h - 1);
                        sum += u32::from(prev[((sy * w + sx) * 4 + c) as usize]);
                    }
                    next.push(((sum + 2) / 4) as u8);
                }
            }
        }
        levels.push(next);
        (w, h) = (nw, nh);
    }
    TextureData {
        width: image.width,
        height: image.height,
        format: if srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        },
        levels,
        sampler,
    }
}

/// Uploads a scenario's data and objects into a new `World`: each texture, mesh, shape and
/// distinct material once (the Bevy side shares its handles the same way).
pub fn upload(
    renderer: &Renderer,
    assets: &Assets,
    objects: &[MeshInstance],
) -> (World, Vec<Instance>) {
    let mut world = World::new(renderer);
    let textures: Vec<TextureId> = assets
        .textures
        .iter()
        .map(|t| world.add_texture(t))
        .collect();
    let meshes: Vec<_> = assets.meshes.iter().map(|m| world.add_mesh(m)).collect();
    let skins: Vec<_> = assets
        .skins
        .iter()
        .map(|s| world.add_skin(&s.palette()))
        .collect();
    let mut shapes = HashMap::new();
    let mut materials = HashMap::new();
    let instances = objects
        .iter()
        .map(|o| Instance {
            mesh: match o.geometry {
                Geometry::Mesh(i) => meshes[i],
                Geometry::Shape(s) => *shapes
                    .entry(format!("{s:?}"))
                    .or_insert_with(|| world.add_mesh(&MeshData::shape(s))),
            },
            material: *materials
                .entry(format!("{:?}", o.material))
                .or_insert_with(|| world.add_material(&o.material.engine(&textures))),
            world_from_local: o.transform(),
            skin: o.skin.map(|i| skins[i]),
        })
        .collect();
    (world, instances)
}
