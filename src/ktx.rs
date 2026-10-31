//! Decoding of the uncompressed (optionally zstd-supercompressed) KTX2 files the engine uses:
//! Bevy's tone-mapping LUTs and prefiltered environment cube maps.

use std::io::Read;

/// A KTX2 file's levels, decompressed. Each level holds its faces (or depth slices) in file
/// order, so the levels concatenate to wgpu's `TextureDataOrder::MipMajor`.
pub struct Texture {
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
    /// 0 for 2D textures.
    pub depth: u32,
    pub faces: u32,
    pub levels: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(String);

impl Error {
    pub(crate) fn new(what: String) -> Self {
        Self(what)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KTX2: {}", self.0)
    }
}

impl std::error::Error for Error {}

/// Decodes a single-layer KTX2 file in `R16G16B16A16_SFLOAT` or `E5B9G9R9_UFLOAT_PACK32`.
pub fn decode(bytes: &[u8]) -> Result<Texture, Error> {
    let err = Error::new;
    let reader = ::ktx2::Reader::new(bytes).map_err(|e| err(format!("{e:?}")))?;
    let header = reader.header();
    let (format, texel_bytes) = match header.format {
        Some(::ktx2::Format::R16G16B16A16_SFLOAT) => (wgpu::TextureFormat::Rgba16Float, 8),
        Some(::ktx2::Format::E5B9G9R9_UFLOAT_PACK32) => (wgpu::TextureFormat::Rgb9e5Ufloat, 4),
        other => return Err(err(format!("unsupported format {other:?}"))),
    };
    if header.layer_count > 1 {
        return Err(err("array textures are not supported".into()));
    }
    let faces = header.face_count.max(1);
    let mut levels = Vec::new();
    for (i, level) in reader.levels().enumerate() {
        let mut data = Vec::with_capacity(level.uncompressed_byte_length as usize);
        match header.supercompression_scheme {
            Some(::ktx2::SupercompressionScheme::Zstandard) => {
                ruzstd::decoding::StreamingDecoder::new(level.data)
                    .map_err(|e| err(format!("level {i}: {e}")))?
                    .read_to_end(&mut data)
                    .map_err(|e| err(format!("level {i}: {e}")))?;
            }
            None => data.extend_from_slice(level.data),
            Some(other) => return Err(err(format!("unsupported supercompression {other:?}"))),
        }
        let dim = |d: u32| (d >> i).max(1);
        let expected = dim(header.pixel_width)
            * dim(header.pixel_height)
            * dim(header.pixel_depth)
            * faces
            * texel_bytes;
        if data.len() != expected as usize {
            return Err(err(format!(
                "level {i} has {} bytes, expected {expected}",
                data.len()
            )));
        }
        levels.push(data);
    }
    Ok(Texture {
        format,
        width: header.pixel_width,
        height: header.pixel_height,
        depth: header.pixel_depth,
        faces,
        levels,
    })
}
