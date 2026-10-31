//! Prefilters an equirectangular HDRI into the diffuse and specular cube maps of a Bevy
//! `EnvironmentMapLight`, the split-sum layout the [glTF IBL Sampler] produces (Bevy's documented
//! tool):
//!
//! - `<prefix>_specular.ktx2`: 256² faces, full mip chain; mip `m` is GGX-filtered for perceptual
//!   roughness `m / (mips - 1)` (Bevy samples level `perceptual_roughness × (mips - 1)`), with
//!   Karis' N = V = R assumption and PDF-based source-mip selection;
//! - `<prefix>_diffuse.ktx2`: 32² faces, the exact Lambertian convolution `(1/π) ∫ L cos θ dω` of
//!   the 32² source mip.
//!
//! Both are `E5B9G9R9_UFLOAT_PACK32`, zstd-supercompressed, written by KTX-Software's `ktx create`
//! (≥ 4.3, on `PATH`). Cube faces follow the Vulkan/wgpu convention; Bevy and render-engine look a
//! world direction `d` up at `(d.x, d.y, -d.z)`.
//!
//! ```text
//! cargo run --release --manifest-path oracle/Cargo.toml --bin bake_ibl -- assets/ibl/tiergarten_1k.hdr assets/ibl/tiergarten
//! ```
//!
//! [glTF IBL Sampler]: https://github.com/KhronosGroup/glTF-IBL-Sampler

use render_engine::glam::{Vec2, Vec3};
use std::{f32::consts::PI, path::Path, process::Command};

const SPECULAR_SIZE: usize = 256;
const DIFFUSE_SIZE: usize = 32;
/// GGX samples per specular texel.
const SAMPLES: u32 = 1024;

/// Linear RGB faces of one cube map level, `size`² texels each, rows top to bottom.
#[derive(Clone)]
struct Level {
    size: usize,
    faces: [Vec<Vec3>; 6],
}

/// Cube-space direction through texel centre `(x, y)` of `face`.
fn texel_dir(face: usize, x: usize, y: usize, size: usize) -> Vec3 {
    let sc = 2.0 * (x as f32 + 0.5) / size as f32 - 1.0;
    let tc = 2.0 * (y as f32 + 0.5) / size as f32 - 1.0;
    match face {
        0 => Vec3::new(1.0, -tc, -sc),
        1 => Vec3::new(-1.0, -tc, sc),
        2 => Vec3::new(sc, 1.0, tc),
        3 => Vec3::new(sc, -1.0, -tc),
        4 => Vec3::new(sc, -tc, 1.0),
        _ => Vec3::new(-sc, -tc, -1.0),
    }
    .normalize()
}

/// Face and `[0, 1]²` coordinates of a cube-space direction.
fn face_uv(c: Vec3) -> (usize, Vec2) {
    let a = c.abs();
    let (face, sc, tc, ma) = if a.x >= a.y && a.x >= a.z {
        if c.x > 0.0 {
            (0, -c.z, -c.y, a.x)
        } else {
            (1, c.z, -c.y, a.x)
        }
    } else if a.y >= a.z {
        if c.y > 0.0 {
            (2, c.x, c.z, a.y)
        } else {
            (3, c.x, -c.z, a.y)
        }
    } else if c.z > 0.0 {
        (4, c.x, -c.y, a.z)
    } else {
        (5, -c.x, -c.y, a.z)
    };
    (face, Vec2::new(sc / ma + 1.0, tc / ma + 1.0) * 0.5)
}

impl Level {
    fn new(size: usize, texel: impl Fn(Vec3) -> Vec3 + Sync) -> Self {
        let faces = std::array::from_fn(|face| {
            parallel_rows(size, |y| {
                (0..size)
                    .map(|x| texel(texel_dir(face, x, y, size)))
                    .collect()
            })
        });
        Level { size, faces }
    }

    /// Bilinear lookup, clamped at face edges.
    fn sample(&self, c: Vec3) -> Vec3 {
        let (face, uv) = face_uv(c);
        let p = uv * self.size as f32 - 0.5;
        let p0 = p.floor();
        let t = p - p0;
        let max = self.size as i32 - 1;
        let at = |x: f32, y: f32| {
            let x = (x as i32).clamp(0, max) as usize;
            let y = (y as i32).clamp(0, max) as usize;
            self.faces[face][y * self.size + x]
        };
        let top = at(p0.x, p0.y).lerp(at(p0.x + 1.0, p0.y), t.x);
        let bottom = at(p0.x, p0.y + 1.0).lerp(at(p0.x + 1.0, p0.y + 1.0), t.x);
        top.lerp(bottom, t.y)
    }

    fn downsample(&self) -> Self {
        let size = self.size / 2;
        let faces = std::array::from_fn(|face| {
            let src = &self.faces[face];
            let mut out = Vec::with_capacity(size * size);
            for y in 0..size {
                for x in 0..size {
                    let at = |dx, dy| src[(2 * y + dy) * self.size + 2 * x + dx];
                    out.push((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) * 0.25);
                }
            }
            out
        });
        Level { size, faces }
    }
}

/// `rows` rows computed on all cores.
fn parallel_rows(rows: usize, row: impl Fn(usize) -> Vec<Vec3> + Sync) -> Vec<Vec3> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let chunk = rows.div_ceil(threads);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..rows)
            .step_by(chunk)
            .map(|start| {
                let row = &row;
                s.spawn(move || {
                    (start..(start + chunk).min(rows))
                        .flat_map(row)
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("bake thread"))
            .collect()
    })
}

/// A source mip chain, sampled trilinearly by fractional level.
struct Chain(Vec<Level>);

impl Chain {
    fn sample(&self, c: Vec3, lod: f32) -> Vec3 {
        let lod = lod.clamp(0.0, (self.0.len() - 1) as f32);
        let l0 = lod.floor() as usize;
        let l1 = (l0 + 1).min(self.0.len() - 1);
        self.0[l0]
            .sample(c)
            .lerp(self.0[l1].sample(c), lod - l0 as f32)
    }
}

fn hammersley(i: u32, n: u32) -> Vec2 {
    Vec2::new(
        i as f32 / n as f32,
        i.reverse_bits() as f32 / 4_294_967_296.0,
    )
}

/// Tangent frame around `n`.
fn basis(n: Vec3) -> (Vec3, Vec3) {
    let up = if n.z.abs() < 0.999 { Vec3::Z } else { Vec3::X };
    let t = up.cross(n).normalize();
    (t, n.cross(t))
}

/// GGX-prefiltered radiance around `n` (N = V = R) for `alpha` = perceptual roughness².
fn prefilter(source: &Chain, n: Vec3, alpha: f32) -> Vec3 {
    let (t, b) = basis(n);
    let a2 = alpha * alpha;
    let base = source.0[0].size as f32;
    let texel_solid_angle = 4.0 * PI / (6.0 * base * base);
    let mut sum = Vec3::ZERO;
    let mut weight = 0.0;
    for i in 0..SAMPLES {
        let xi = hammersley(i, SAMPLES);
        let phi = 2.0 * PI * xi.x;
        let cos_theta = ((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y)).sqrt();
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let h = (t * phi.cos() + b * phi.sin()) * sin_theta + n * cos_theta;
        let l = 2.0 * n.dot(h) * h - n;
        let n_dot_l = n.dot(l);
        if n_dot_l <= 0.0 {
            continue;
        }
        // pdf(L) = D(H) · NdotH / (4 VdotH) = D / 4 with N = V.
        let d = a2 / (PI * (cos_theta * cos_theta * (a2 - 1.0) + 1.0).powi(2));
        let sample_solid_angle = 4.0 / (SAMPLES as f32 * d);
        let lod = 0.5 * (sample_solid_angle / texel_solid_angle).log2() + 1.0;
        sum += source.sample(l, lod) * n_dot_l;
        weight += n_dot_l;
    }
    sum / weight
}

/// `(1/π) ∫ L(ω) max(0, n·ω) dω` over every texel of `source`.
fn irradiance(source: &Level, n: Vec3) -> Vec3 {
    let size = source.size;
    let mut sum = Vec3::ZERO;
    for (face, texels) in source.faces.iter().enumerate() {
        for y in 0..size {
            for x in 0..size {
                let sc = 2.0 * (x as f32 + 0.5) / size as f32 - 1.0;
                let tc = 2.0 * (y as f32 + 0.5) / size as f32 - 1.0;
                let w = texel_dir(face, x, y, size);
                let cos = n.dot(w);
                if cos > 0.0 {
                    // Solid angle of a texel at (sc, tc) on the unit cube.
                    let d2 = 1.0 + sc * sc + tc * tc;
                    let solid_angle = (2.0 / size as f32).powi(2) / (d2 * d2.sqrt());
                    sum += texels[y * size + x] * cos * solid_angle;
                }
            }
        }
    }
    sum / PI
}

/// OpenGL `EXT_texture_shared_exponent` packing (wgpu `Rgb9e5Ufloat`).
fn rgb9e5(c: Vec3) -> u32 {
    const N: i32 = 9;
    const B: i32 = 15;
    let max_value = (511.0 / 512.0) * 2f32.powi(31 - B);
    let c = c.clamp(Vec3::ZERO, Vec3::splat(max_value));
    let max_c = c.max_element();
    let mut exp = (max_c.log2().floor() as i32).max(-B - 1) + 1 + B;
    if max_c == 0.0 {
        exp = 0;
    }
    let mut scale = 2f32.powi(exp - B - N);
    if (max_c / scale + 0.5).floor() as i32 == 1 << N {
        exp += 1;
        scale *= 2.0;
    }
    let m = |v: f32| (v / scale + 0.5).floor() as u32;
    m(c.x) | (m(c.y) << 9) | (m(c.z) << 18) | ((exp as u32) << 27)
}

/// Writes the levels as one zstd-supercompressed cube-map KTX2 file through `ktx create --raw`.
fn write_ktx2(levels: &[Level], tmp: &Path, out: &Path) {
    let mut inputs = Vec::new();
    for (l, level) in levels.iter().enumerate() {
        for (f, face) in level.faces.iter().enumerate() {
            let path = tmp.join(format!("{}-{l}-{f}.raw", out.display()).replace('/', "_"));
            let bytes: Vec<u8> = face.iter().flat_map(|&c| rgb9e5(c).to_le_bytes()).collect();
            std::fs::write(&path, bytes).expect("write raw face");
            inputs.push(path);
        }
    }
    let size = levels[0].size.to_string();
    let status = Command::new("ktx")
        .args([
            "create",
            "--raw",
            "--cubemap",
            "--format",
            "E5B9G9R9_UFLOAT_PACK32",
        ])
        .args(["--width", &size, "--height", &size])
        .args(["--levels", &levels.len().to_string()])
        .args(["--assign-tf", "linear", "--assign-primaries", "bt709"])
        .args(["--zstd", "19"])
        .args(&inputs)
        .arg(out)
        .status()
        .expect("run `ktx create` (KTX-Software ≥ 4.3 on PATH)");
    assert!(status.success(), "ktx create failed for {}", out.display());
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [hdr, prefix] = args.as_slice() else {
        eprintln!("usage: bake_ibl <equirect.hdr> <output prefix>");
        std::process::exit(2);
    };
    let image = image::open(hdr).expect("read HDRI").into_rgb32f();
    let (w, h) = image.dimensions();
    let texel = |x: u32, y: u32| Vec3::from(image.get_pixel(x % w, y.min(h - 1)).0);
    // World direction → equirect: -Z (Bevy's forward) is the image centre, +Y the top row.
    let equirect = |d: Vec3| {
        let u = 0.5 + d.x.atan2(-d.z) / (2.0 * PI);
        let v = d.y.clamp(-1.0, 1.0).acos() / PI;
        let p = Vec2::new(u * w as f32 - 0.5, v * h as f32 - 0.5);
        let p0 = p.floor();
        let t = p - p0;
        let (x, y) = (p0.x.rem_euclid(w as f32) as u32, p0.y.max(0.0) as u32);
        let top = texel(x, y).lerp(texel(x + 1, y), t.x);
        let bottom = texel(x, y + 1).lerp(texel(x + 1, y + 1), t.x);
        top.lerp(bottom, t.y)
    };
    // Cube space → world: the shaders flip z when sampling.
    let world = |c: Vec3| Vec3::new(c.x, c.y, -c.z);

    // Source chain: 2×2 supersampled base level, box-filtered mips down to 1².
    let base = Level::new(SPECULAR_SIZE * 2, |c| equirect(world(c))).downsample();
    let mut chain = vec![base];
    while chain.last().unwrap().size > 1 {
        let next = chain.last().unwrap().downsample();
        chain.push(next);
    }
    let source = Chain(chain);

    let mips = source.0.len();
    let specular: Vec<Level> = (0..mips)
        .map(|m| {
            let size = SPECULAR_SIZE >> m;
            if m == 0 {
                return source.0[0].clone();
            }
            let perceptual = m as f32 / (mips - 1) as f32;
            Level::new(size, |n| prefilter(&source, n, perceptual * perceptual))
        })
        .collect();
    let diffuse_source = source
        .0
        .iter()
        .find(|l| l.size == DIFFUSE_SIZE)
        .expect("32² source mip");
    let diffuse = Level::new(DIFFUSE_SIZE, |n| irradiance(diffuse_source, n));

    let tmp = std::env::temp_dir().join(format!("bake_ibl-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).expect("temp dir");
    write_ktx2(
        &specular,
        &tmp,
        Path::new(&format!("{prefix}_specular.ktx2")),
    );
    write_ktx2(
        &[diffuse],
        &tmp,
        Path::new(&format!("{prefix}_diffuse.ktx2")),
    );
    std::fs::remove_dir_all(&tmp).ok();
    println!("wrote {prefix}_specular.ktx2 ({mips} mips) and {prefix}_diffuse.ktx2");
}
