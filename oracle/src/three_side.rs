//! The oracle's three.js side (technical decisions D8): each lighting scenario rendered in
//! headless Chromium through `packages/three-lighting` and compared with Bevy under the same FLIP
//! gate as render-engine.
//!
//! `export` writes a scenario as `<name>.three.json` (scene values in Bevy units, objects with
//! their transforms and materials) plus `<name>.three.bin` (vertex and index data, little
//! endian). `render` runs `oracle/three/render.mjs` once for all exported scenarios, which writes
//! `<name>.three.png` next to them.
//!
//! Scenarios that use something the port does not cover yet are not exported; the reason is
//! reported as a known gap with its follow-up issue (`GAPS`).

use crate::content::Geometry;
use crate::scenarios::Scenario;
use render_engine::mesh::MeshData;
use render_engine::scene::{FogFalloff, LightShadows, ShadowFilter, Tonemapping};
use serde_json::{Value, json};
use std::{collections::HashMap, path::Path};

/// Scenarios whose three.js side does not pass yet, with the issue that tracks each. A listed
/// scenario's failure does not fail the run; every other exported scenario must pass the gate.
pub const GAPS: &[(&str, u32)] = &[("many_instances", 42)];

/// Follow-up issue per capability the port does not cover yet.
const UNPORTED_ISSUE: &[(&str, u32)] = &[
    ("sky and moon", 39),
    ("textured materials", 40),
    ("double-sided, unlit or transparent materials", 40),
    ("skinning", 41),
];

pub fn gap_issue(name: &str) -> Option<u32> {
    GAPS.iter().find(|(n, _)| *n == name).map(|(_, i)| *i)
}

/// Why `s` cannot be exported yet, if anything is missing on the three.js side.
pub fn unsupported(s: &Scenario) -> Option<&'static str> {
    let scene = &s.scene;
    if scene.sky.is_some() || scene.moon.is_some() {
        return Some("sky and moon");
    }
    if s.objects.iter().any(|o| o.skin.is_some()) {
        return Some("skinning");
    }
    if !s.assets.textures.is_empty() {
        return Some("textured materials");
    }
    if s.objects
        .iter()
        .any(|o| o.material.double_sided || o.material.unlit || o.material.alpha < 1.0)
    {
        return Some("double-sided, unlit or transparent materials");
    }
    None
}

pub fn unported_issue(reason: &str) -> Option<u32> {
    UNPORTED_ISSUE
        .iter()
        .find(|(r, _)| *r == reason)
        .map(|(_, i)| *i)
}

fn v3(v: render_engine::glam::Vec3) -> Value {
    json!([v.x, v.y, v.z])
}

fn v4(v: render_engine::glam::Vec4) -> Value {
    json!([v.x, v.y, v.z, v.w])
}

/// Writes `<name>.three.json` and `<name>.three.bin` into `dir`.
pub fn export(s: &Scenario, dir: &Path) -> Result<(), String> {
    let mut bin: Vec<u8> = Vec::new();
    let push_f32 = |bin: &mut Vec<u8>, data: &[f32]| {
        let offset = bin.len();
        for v in data {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        json!({ "offset": offset, "count": data.len() })
    };
    let mut meshes = Vec::new();
    let mut shapes: HashMap<String, usize> = HashMap::new();
    let add_mesh = |bin: &mut Vec<u8>, m: &MeshData| -> Value {
        let positions: Vec<f32> = m.positions.iter().flat_map(|p| p.to_array()).collect();
        let normals: Vec<f32> = m.normals.iter().flat_map(|p| p.to_array()).collect();
        let positions = push_f32(bin, &positions);
        let normals = push_f32(bin, &normals);
        let offset = bin.len();
        for i in &m.indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        json!({
            "positions": positions,
            "normals": normals,
            "indices": { "offset": offset, "count": m.indices.len() },
        })
    };
    let mut objects = Vec::new();
    for o in &s.objects {
        let mesh = match o.geometry {
            Geometry::Shape(shape) => {
                let key = format!("{shape:?}");
                match shapes.get(&key) {
                    Some(&i) => i,
                    None => {
                        meshes.push(add_mesh(&mut bin, &MeshData::shape(shape)));
                        shapes.insert(key, meshes.len() - 1);
                        meshes.len() - 1
                    }
                }
            }
            Geometry::Mesh(i) => {
                let key = format!("mesh{i}");
                match shapes.get(&key) {
                    Some(&j) => j,
                    None => {
                        meshes.push(add_mesh(&mut bin, &s.assets.meshes[i]));
                        shapes.insert(key, meshes.len() - 1);
                        meshes.len() - 1
                    }
                }
            }
        };
        let m = &o.material;
        objects.push(json!({
            "mesh": mesh,
            "matrix": o.transform().to_cols_array(),
            "material": {
                "baseColor": v3(m.base_color),
                "alpha": m.alpha,
                "metallic": m.metallic,
                "perceptualRoughness": m.perceptual_roughness,
                "reflectance": m.reflectance,
                "emissive": v3(m.emissive),
            },
        }));
    }

    let scene = &s.scene;
    let cam = &scene.camera;
    let sun = scene.sun.map(|sun| {
        json!({
            "direction": v3(sun.direction),
            "color": v3(sun.color),
            "illuminance": sun.illuminance,
            "shadows": sun.shadows.map(|c| json!({
                "numCascades": c.num_cascades,
                "minDistance": c.min_distance,
                "firstCascadeFarBound": c.first_cascade_far_bound,
                "maxDistance": c.max_distance,
                "overlapProportion": c.overlap_proportion,
                "depthBias": c.depth_bias,
                "normalBias": c.normal_bias,
            })),
        })
    });
    let shadows = |s: Option<LightShadows>| {
        s.map(|s| {
            json!({
                "depthBias": s.depth_bias,
                "normalBias": s.normal_bias,
                "nearZ": s.near_z,
                "soft": s.soft,
            })
        })
    };
    let point_lights: Vec<Value> = scene
        .point_lights
        .iter()
        .map(|l| {
            json!({
                "position": v3(l.position),
                "color": v3(l.color),
                "intensity": l.intensity,
                "range": l.range,
                "radius": l.radius,
                "shadows": shadows(l.shadows),
            })
        })
        .collect();
    let spot_lights: Vec<Value> = scene
        .spot_lights
        .iter()
        .map(|l| {
            json!({
                "position": v3(l.position),
                "direction": v3(l.direction),
                "color": v3(l.color),
                "intensity": l.intensity,
                "range": l.range,
                "radius": l.radius,
                "innerAngle": l.inner_angle,
                "outerAngle": l.outer_angle,
                "shadows": shadows(l.shadows),
            })
        })
        .collect();
    // Tiergarten is the only committed environment map (assets/ibl).
    let environment = scene.environment.as_ref().map(|e| {
        json!({
            "diffuse": "assets/ibl/tiergarten_diffuse.ktx2",
            "specular": "assets/ibl/tiergarten_specular.ktx2",
            "intensity": e.intensity,
        })
    });
    let fog = scene.fog.map(|f| {
        let falloff = match f.falloff {
            FogFalloff::Linear { start, end } => {
                json!({ "type": "linear", "start": start, "end": end })
            }
            FogFalloff::Exponential { density } => {
                json!({ "type": "exponential", "density": density })
            }
            FogFalloff::ExponentialSquared { density } => {
                json!({ "type": "exponentialSquared", "density": density })
            }
            FogFalloff::Atmospheric {
                extinction,
                inscattering,
            } => json!({
                "type": "atmospheric",
                "extinction": v3(extinction),
                "inscattering": v3(inscattering),
            }),
        };
        json!({
            "color": v4(f.color),
            "directionalLightColor": v4(f.directional_light_color),
            "directionalLightExponent": f.directional_light_exponent,
            "falloff": falloff,
        })
    });
    let doc = json!({
        "name": s.name,
        "width": s.width,
        "height": s.height,
        "camera": {
            "eye": v3(cam.eye),
            "target": v3(cam.target),
            "fovY": cam.fov_y,
            "near": cam.near,
            "ev100": cam.ev100,
            "shadowFilter": match cam.shadow_filter {
                ShadowFilter::Gaussian => "gaussian",
                ShadowFilter::Hardware2x2 => "hardware2x2",
            },
            "tonemapping": match cam.tonemapping {
                Tonemapping::AcesFitted => "acesFitted",
                Tonemapping::TonyMcMapface => "tonyMcMapface",
                Tonemapping::AgX => "agx",
            },
        },
        "clearColor": v3(scene.clear_color),
        "sun": sun,
        "ambient": { "color": v3(scene.ambient.color), "brightness": scene.ambient.brightness },
        "shadowMapSize": scene.shadow_maps.directional,
        "pointShadowMapSize": scene.shadow_maps.point,
        "pointLights": point_lights,
        "spotLights": spot_lights,
        "environment": environment,
        "fog": fog,
        "meshes": meshes,
        "objects": objects,
    });
    std::fs::write(dir.join(format!("{}.three.json", s.name)), doc.to_string())
        .map_err(|e| e.to_string())?;
    std::fs::write(dir.join(format!("{}.three.bin", s.name)), bin).map_err(|e| e.to_string())
}

/// Renders the exported scenarios in headless Chromium (`oracle/three/render.mjs`, WebGL through
/// ANGLE on Vulkan). Needs `node` and the repository's npm dependencies (`bun install`). On a
/// software adapter Chromium gets lavapipe's Vulkan ICD, the device the Bevy side runs on, so
/// both sides store f16 and sRGB values the same way.
pub fn render(names: &[&str], dir: &Path, software: bool, frames: u32) -> Result<(), String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let dir = std::path::absolute(dir).map_err(|e| e.to_string())?;
    let mut command = std::process::Command::new("node");
    command
        .arg(root.join("oracle/three/render.mjs"))
        .arg(&dir)
        .args(names)
        .current_dir(&root)
        .env("THREE_FRAMES", frames.to_string());
    if software {
        command.env("THREE_VK_ICD", lavapipe_icd()?);
    }
    let status = command
        .status()
        .map_err(|e| format!("cannot run node: {e}"))?;
    if !status.success() {
        return Err(format!("three side exited with {status}"));
    }
    Ok(())
}

/// lavapipe's Vulkan ICD manifest (`lvp_icd.json`, or `lvp_icd.<arch>.json` on Ubuntu).
fn lavapipe_icd() -> Result<std::path::PathBuf, String> {
    let dir = Path::new("/usr/share/vulkan/icd.d");
    std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("lvp_icd") && n.ends_with(".json"))
        })
        .ok_or_else(|| format!("no lavapipe ICD in {}", dir.display()))
}
