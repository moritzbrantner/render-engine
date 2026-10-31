//! Textures and metallic-roughness materials (Bevy `Image` and `StandardMaterial`), uploaded once
//! into a [`World`](crate::world::World) and referenced by handle.

use crate::world::TextureId;
use glam::{Vec3, Vec4};

/// Texels of a 2D texture with its mip chain and sampler (Bevy `Image`).
#[derive(Clone)]
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    /// `Rgba8UnormSrgb` for colour (base colour, emissive), `Rgba8Unorm` for data (normal maps,
    /// metallic-roughness, occlusion), as glTF prescribes; any filterable format works.
    pub format: wgpu::TextureFormat,
    /// Level 0 first; each level is half the size of the previous one. One level means no
    /// mipmaps: the engine never generates them.
    pub levels: Vec<Vec<u8>>,
    pub sampler: Sampler,
}

impl std::fmt::Debug for TextureData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextureData")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("mip_levels", &self.levels.len())
            .field("sampler", &self.sampler)
            .finish()
    }
}

/// How a texture is sampled (glTF `sampler`, Bevy `ImageSamplerDescriptor`). The default is
/// glTF's: repeat, trilinear.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sampler {
    pub address_mode_u: wgpu::AddressMode,
    pub address_mode_v: wgpu::AddressMode,
    pub mag_filter: wgpu::FilterMode,
    pub min_filter: wgpu::FilterMode,
    pub mipmap_filter: wgpu::MipmapFilterMode,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
        }
    }
}

/// How a material's alpha is used (Bevy `AlphaMode`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AlphaMode {
    /// Alpha is ignored.
    #[default]
    Opaque,
    /// Fragments with alpha below the cutoff are discarded, the rest are opaque; shadows are cut
    /// out the same way (glTF `MASK` with `alphaCutoff`).
    Mask(f32),
    /// Blended over what is behind (straight alpha), drawn after the opaque geometry from back to
    /// front by instance origin, without depth writes. Casts shadows where alpha ≥ 0.05, like
    /// Bevy.
    Blend,
}

/// A metallic-roughness PBR material (Bevy `StandardMaterial`, the subset glTF core uses). Every
/// texture factor multiplies its texture; textures sample `TEXCOORD_0`. The defaults are Bevy's.
#[derive(Clone, Copy, Debug)]
pub struct StandardMaterial {
    /// Linear RGBA (glTF `baseColorFactor`).
    pub base_color: Vec4,
    /// sRGB colour texture; alpha feeds the [`AlphaMode`].
    pub base_color_texture: Option<TextureId>,
    /// Linear RGB radiance in cd/m², exposed with the frame (glTF `emissiveFactor` ×
    /// `KHR_materials_emissive_strength`).
    pub emissive: Vec3,
    pub emissive_texture: Option<TextureId>,
    pub metallic: f32,
    pub perceptual_roughness: f32,
    /// Green channel scales roughness, blue scales metallic (glTF).
    pub metallic_roughness_texture: Option<TextureId>,
    /// Dielectric specular reflectance, 0.5 = 4% F0.
    pub reflectance: f32,
    /// Tangent-space normal map (OpenGL convention, +Y up); needs mesh tangents.
    pub normal_map_texture: Option<TextureId>,
    /// Red channel darkens ambient and environment diffuse light.
    pub occlusion_texture: Option<TextureId>,
    pub alpha_mode: AlphaMode,
    /// Draws back faces too, with the normal flipped (glTF `doubleSided`; Bevy `double_sided`
    /// with `cull_mode: None`).
    pub double_sided: bool,
    /// Outputs the base colour as is, without lighting or exposure (glTF `KHR_materials_unlit`).
    pub unlit: bool,
}

impl Default for StandardMaterial {
    fn default() -> Self {
        Self {
            base_color: Vec4::ONE,
            base_color_texture: None,
            emissive: Vec3::ZERO,
            emissive_texture: None,
            metallic: 0.0,
            perceptual_roughness: 0.5,
            metallic_roughness_texture: None,
            reflectance: 0.5,
            normal_map_texture: None,
            occlusion_texture: None,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
            unlit: false,
        }
    }
}

impl StandardMaterial {
    /// An untextured opaque material with a linear RGB base colour.
    pub fn color(base_color: Vec3) -> Self {
        Self {
            base_color: base_color.extend(1.0),
            ..Self::default()
        }
    }
}
