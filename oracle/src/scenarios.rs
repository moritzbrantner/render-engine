//! Parity scenarios. Each one is a `render_engine::scene::Scene`; the Bevy side translates it
//! field by field (see `bevy_side.rs`).

use render_engine::glam::{Quat, Vec2, Vec3, Vec4};
use render_engine::scene::{
    Ambient, Camera, Cubemap, EnvironmentMap, Fog, FogFalloff, LightShadows, Material,
    MeshInstance, PointLight, Scene, ShadowFilter, ShadowMapSizes, Shape, SpotLight, Sun,
    SunShadows, Tonemapping,
};
use std::sync::Arc;

pub struct Scenario {
    pub name: &'static str,
    /// Capability the scenario covers (an issue/doc label, printed by `--list`).
    pub capability: &'static str,
    pub width: u32,
    pub height: u32,
    pub scene: Scene,
}

pub fn all() -> Vec<Scenario> {
    let mut all = vec![sun_shadows(), sun_cascades()];
    all.extend(tonemap_range());
    all.extend(ibl_materials());
    all.extend(fog_modes());
    all.extend(point_spot());
    all
}

/// A dusk courtyard under low ambient light: a shadow-casting lantern hanging among PBR boxes, a
/// wall torch, a short-range ground light whose range ends inside the view, a shadow-casting
/// spot light from above and an unshadowed uplight washing a wall (a spot pointing upwards).
/// Rendered with the Gaussian filter, with 2×2 PCF on smaller maps, and with soft shadows (PCSS)
/// from both shadow-casting lights grown to 0.4 m spheres.
fn point_spot() -> [Scenario; 3] {
    let dielectric = |base_color, perceptual_roughness| Material {
        base_color,
        perceptual_roughness,
        ..Material::default()
    };
    let stone = dielectric(rgb(0.55, 0.52, 0.47), 0.85);
    let meshes = vec![
        MeshInstance::new(
            Shape::Plane(Vec2::splat(14.0)),
            Vec3::ZERO,
            dielectric(rgb(0.40, 0.38, 0.34), 0.8),
        ),
        // North and west walls.
        MeshInstance::new(
            Shape::Cuboid(Vec3::new(14.0, 3.0, 0.5)),
            Vec3::new(0.0, 1.5, -6.75),
            stone,
        ),
        MeshInstance::new(
            Shape::Cuboid(Vec3::new(0.5, 3.0, 13.0)),
            Vec3::new(-6.75, 1.5, -0.25),
            stone,
        ),
        // Boxes around the lantern.
        MeshInstance::new(
            Shape::Cuboid(Vec3::ONE),
            Vec3::new(-1.8, 0.5, 0.9),
            dielectric(rgb(0.70, 0.08, 0.06), 0.45),
        ),
        MeshInstance::new(
            Shape::Cuboid(Vec3::splat(1.2)),
            Vec3::new(1.6, 0.6, -1.2),
            Material {
                base_color: rgb(1.0, 0.71, 0.29),
                metallic: 1.0,
                perceptual_roughness: 0.3,
                ..Material::default()
            },
        ),
        MeshInstance::new(
            Shape::Cuboid(Vec3::new(0.7, 2.4, 0.7)),
            Vec3::new(-0.6, 1.2, -2.6),
            dielectric(rgb(0.10, 0.25, 0.65), 0.6),
        )
        .rotated(Quat::from_rotation_y(30f32.to_radians())),
        MeshInstance::new(
            Shape::Cuboid(Vec3::new(1.8, 0.4, 0.6)),
            Vec3::new(0.6, 0.2, 2.0),
            dielectric(rgb(0.85, 0.85, 0.82), 0.2),
        )
        .rotated(Quat::from_rotation_y(-20f32.to_radians())),
        MeshInstance::new(
            Shape::Sphere {
                radius: 0.5,
                sectors: 48,
                stacks: 24,
            },
            Vec3::new(2.8, 0.5, 1.4),
            dielectric(rgb(0.20, 0.55, 0.25), 0.35),
        ),
    ];
    let scene = Scene {
        camera: Camera {
            eye: Vec3::new(7.0, 6.0, 9.0),
            target: Vec3::new(-0.5, 0.5, -1.0),
            fov_y: 45f32.to_radians(),
            near: 0.1,
            ev100: 4.0,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::TonyMcMapface,
        },
        clear_color: rgb(0.03, 0.04, 0.08),
        sun: None,
        point_lights: vec![
            // Wall torch.
            PointLight {
                position: Vec3::new(-6.2, 2.0, -2.5),
                color: rgb(1.0, 0.6, 0.3),
                intensity: 3000.0,
                range: 7.0,
                radius: 0.1,
                shadows: None,
            },
            // Hanging lantern; listed second, shaded first (shadow casters lead, as in Bevy).
            PointLight {
                position: Vec3::new(0.0, 2.2, 0.0),
                color: rgb(1.0, 0.82, 0.6),
                intensity: 8000.0,
                range: 12.0,
                radius: 0.08,
                shadows: Some(LightShadows::POINT),
            },
            // A ground light whose 2.5 m range ends in plain view.
            PointLight {
                position: Vec3::new(4.0, 0.3, 3.5),
                color: rgb(0.5, 0.7, 1.0),
                intensity: 1500.0,
                range: 2.5,
                radius: 0.0,
                shadows: None,
            },
        ],
        spot_lights: vec![
            // Uplight on the north wall: points upwards (the spot direction's y sign).
            SpotLight {
                position: Vec3::new(3.5, 0.1, -5.8),
                direction: Vec3::new(0.0, 1.0, -0.35),
                color: rgb(0.7, 0.8, 1.0),
                intensity: 20_000.0,
                range: 6.0,
                radius: 0.0,
                inner_angle: 10f32.to_radians(),
                outer_angle: 30f32.to_radians(),
                shadows: None,
            },
            SpotLight {
                position: Vec3::new(4.5, 5.0, -3.5),
                direction: Vec3::new(-0.5, 0.0, 0.2) - Vec3::new(4.5, 5.0, -3.5),
                color: rgb(1.0, 0.95, 0.85),
                intensity: 60_000.0,
                range: 15.0,
                radius: 0.1,
                inner_angle: 20f32.to_radians(),
                outer_angle: 32f32.to_radians(),
                shadows: Some(LightShadows::SPOT),
            },
        ],
        shadow_maps: ShadowMapSizes::default(),
        ambient: Ambient {
            color: rgb(0.45, 0.55, 0.85),
            brightness: 4.0,
        },
        environment: None,
        fog: None,
        meshes,
    };
    // Coarse maps, so the 2×2 filter's texel steps cover enough pixels to gate.
    let mut hardware = scene.clone();
    hardware.camera.shadow_filter = ShadowFilter::Hardware2x2;
    hardware.shadow_maps = ShadowMapSizes {
        directional: 512,
        point: 256,
    };
    // Bevy's penumbrae stay narrow even from large lights, so the camera moves in on the
    // sphere's and the gold box's shadows.
    let mut soft = scene.clone();
    soft.camera.eye = Vec3::new(5.5, 2.2, 4.5);
    soft.camera.target = Vec3::new(3.0, 0.0, 0.0);
    for l in &mut soft.point_lights {
        if l.shadows.is_some() {
            l.radius = 0.5;
            l.shadows = l.shadows.map(LightShadows::soft);
        }
    }
    for l in &mut soft.spot_lights {
        if l.shadows.is_some() {
            l.radius = 0.5;
            l.shadows = l.shadows.map(LightShadows::soft);
        }
    }
    [
        Scenario {
            name: "point_spot",
            capability: "point + spot lights, cube + spot shadows (Gaussian)",
            width: 1280,
            height: 720,
            scene,
        },
        Scenario {
            name: "point_spot_hardware",
            capability: "point + spot lights, cube + spot shadows (2×2 PCF)",
            width: 1280,
            height: 720,
            scene: hardware,
        },
        Scenario {
            name: "point_spot_soft",
            capability: "point + spot lights, soft shadows (PCSS)",
            width: 1280,
            height: 720,
            scene: soft,
        },
    ]
}

/// A long receding double row of pillars towards a low sun, under each `FogFalloff`, once
/// without and once with sun inscattering (whose glow the pillars' shadows break up).
fn fog_modes() -> Vec<Scenario> {
    let stone = Material {
        base_color: rgb(0.62, 0.58, 0.52),
        perceptual_roughness: 0.8,
        ..Material::default()
    };
    let mut meshes = vec![MeshInstance::new(
        Shape::Plane(Vec2::new(60.0, 300.0)),
        Vec3::new(0.0, 0.0, -120.0),
        Material {
            base_color: rgb(0.24, 0.26, 0.20),
            perceptual_roughness: 0.9,
            ..Material::default()
        },
    )];
    for i in 0..30 {
        for x in [-3.5, 3.5] {
            meshes.push(MeshInstance::new(
                Shape::Cuboid(Vec3::new(0.8, 5.0, 0.8)),
                Vec3::new(x, 2.5, -6.0 * i as f32),
                stone,
            ));
        }
    }
    let (elevation, azimuth) = (10f32.to_radians(), 12f32.to_radians());
    let haze = Vec4::new(0.40, 0.48, 0.58, 1.0);
    let base = Scene {
        camera: Camera {
            eye: Vec3::new(0.0, 2.5, 8.0),
            target: Vec3::new(0.0, 2.0, -30.0),
            fov_y: 45f32.to_radians(),
            near: 0.1,
            ev100: 10.5,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::TonyMcMapface,
        },
        clear_color: haze.truncate(),
        // Travelling towards the camera, so the glow sits behind the pillars.
        sun: Some(Sun {
            direction: Vec3::new(
                -elevation.cos() * azimuth.sin(),
                -elevation.sin(),
                elevation.cos() * azimuth.cos(),
            ),
            color: rgb(1.0, 0.92, 0.80),
            illuminance: 10_000.0,
            shadows: Some(SunShadows::default()),
        }),
        ambient: Ambient {
            color: rgb(0.75, 0.85, 1.0),
            brightness: 1500.0,
        },
        point_lights: Vec::new(),
        spot_lights: Vec::new(),
        shadow_maps: ShadowMapSizes::default(),
        environment: None,
        fog: None,
        meshes,
    };
    let falloffs = [
        (
            "fog_linear",
            FogFalloff::Linear {
                start: 5.0,
                end: 90.0,
            },
        ),
        ("fog_exponential", FogFalloff::from_visibility(120.0)),
        (
            "fog_exponential_squared",
            FogFalloff::from_visibility_squared(100.0),
        ),
        (
            "fog_atmospheric",
            FogFalloff::from_visibility_colors(
                150.0,
                Vec4::new(0.35, 0.5, 0.66, 1.0),
                Vec4::new(0.8, 0.844, 1.0, 1.0),
            ),
        ),
    ];
    let names_sun = [
        "fog_linear_sun",
        "fog_exponential_sun",
        "fog_exponential_squared_sun",
        "fog_atmospheric_sun",
    ];
    let mut out = Vec::new();
    for ((name, falloff), name_sun) in falloffs.into_iter().zip(names_sun) {
        let fog = Fog {
            color: haze,
            falloff,
            ..Fog::default()
        };
        let mut scene = base.clone();
        scene.fog = Some(fog);
        out.push(Scenario {
            name,
            capability: "distance fog",
            width: 1280,
            height: 720,
            scene: scene.clone(),
        });
        scene.fog = Some(Fog {
            directional_light_color: Vec4::new(1.0, 0.85, 0.6, 0.5),
            directional_light_exponent: 20.0,
            ..fog
        });
        out.push(Scenario {
            name: name_sun,
            capability: "distance fog + sun inscattering",
            width: 1280,
            height: 720,
            scene,
        });
    }
    out
}

/// Poly Haven's CC0 "Tiergarten" HDRI (overcast park), prefiltered by `bake_ibl` (see
/// `assets/ibl/README.md`).
fn tiergarten(intensity: f32) -> EnvironmentMap {
    let cube = |bytes: &[u8]| Arc::new(Cubemap::from_ktx2(bytes).expect("committed cube map"));
    EnvironmentMap {
        diffuse: cube(include_bytes!("../../assets/ibl/tiergarten_diffuse.ktx2")),
        specular: cube(include_bytes!("../../assets/ibl/tiergarten_specular.ktx2")),
        intensity,
    }
}

/// Two rows of UV spheres sweeping perceptual roughness 0 → 1, dielectric in front and metallic
/// behind, on a ground plane under the Tiergarten environment map: once with image-based light
/// only, once with a shadowing sun as well.
fn ibl_materials() -> [Scenario; 2] {
    let mut meshes = vec![MeshInstance::new(
        Shape::Plane(Vec2::splat(14.0)),
        Vec3::ZERO,
        Material {
            base_color: rgb(0.30, 0.30, 0.28),
            perceptual_roughness: 0.9,
            ..Material::default()
        },
    )];
    let roughness = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0];
    for (row, (metallic, base_color)) in [(0.0, rgb(0.75, 0.15, 0.12)), (1.0, rgb(1.0, 0.71, 0.29))]
        .into_iter()
        .enumerate()
    {
        for (i, &perceptual_roughness) in roughness.iter().enumerate() {
            meshes.push(MeshInstance::new(
                Shape::Sphere {
                    radius: 0.4,
                    sectors: 64,
                    stacks: 32,
                },
                Vec3::new(i as f32 - 3.0, 0.4, -1.1 * row as f32),
                Material {
                    base_color,
                    metallic,
                    perceptual_roughness,
                    ..Material::default()
                },
            ));
        }
    }
    let scene = Scene {
        camera: Camera {
            eye: Vec3::new(0.0, 2.2, 5.5),
            target: Vec3::new(0.0, 0.3, -0.6),
            fov_y: 45f32.to_radians(),
            near: 0.1,
            ev100: 10.8,
            shadow_filter: ShadowFilter::Gaussian,
            tonemapping: Tonemapping::TonyMcMapface,
        },
        clear_color: rgb(0.45, 0.50, 0.55),
        sun: None,
        ambient: Ambient {
            color: Vec3::ONE,
            brightness: 0.0,
        },
        point_lights: Vec::new(),
        spot_lights: Vec::new(),
        shadow_maps: ShadowMapSizes::default(),
        environment: Some(tiergarten(IBL_INTENSITY)),
        fog: None,
        meshes,
    };
    let mut sunlit = scene.clone();
    sunlit.sun = Some(Sun {
        direction: Vec3::new(0.6, -1.0, -0.5).normalize(),
        color: rgb(1.0, 0.95, 0.88),
        illuminance: 10_000.0,
        shadows: Some(SunShadows::single_cascade(20.0)),
    });
    sunlit.camera.ev100 = 11.5;
    [
        Scenario {
            name: "ibl_materials",
            capability: "environment map (IBL)",
            width: 1280,
            height: 720,
            scene,
        },
        Scenario {
            name: "ibl_materials_sun",
            capability: "IBL + sun + shadow",
            width: 1280,
            height: 720,
            scene: sunlit,
        },
    ]
}

/// cd/m² scale of the Tiergarten maps.
const IBL_INTENSITY: f32 = 2000.0;

/// The `sun_shadows` scene at a noon exposure (EV100 10.5, bright highlights on the gold box)
/// and three stops dimmer (EV100 13.5, dusk-like shadows near black), each with Tony McMapface and
/// AgX: the shoulder and the toe of both LUT operators.
fn tonemap_range() -> [Scenario; 4] {
    [
        ("tonemap_range_noon_tony", 10.5, Tonemapping::TonyMcMapface),
        ("tonemap_range_noon_agx", 10.5, Tonemapping::AgX),
        ("tonemap_range_dim_tony", 13.5, Tonemapping::TonyMcMapface),
        ("tonemap_range_dim_agx", 13.5, Tonemapping::AgX),
    ]
    .map(|(name, ev100, tonemapping)| {
        let mut s = sun_shadows();
        s.name = name;
        s.capability = "HDR + tone mapping";
        s.scene.camera.ev100 = ev100;
        s.scene.camera.tonemapping = tonemapping;
        s
    })
}

fn rgb(r: f32, g: f32, b: f32) -> Vec3 {
    Vec3::new(r, g, b)
}

/// Ground plane, PBR boxes (dielectric, metallic, rotated), a shadow-casting sun and ambient.
fn sun_shadows() -> Scenario {
    let dielectric = |base_color, perceptual_roughness| Material {
        base_color,
        perceptual_roughness,
        ..Material::default()
    };
    let meshes = vec![
        MeshInstance::new(
            Shape::Plane(Vec2::splat(24.0)),
            Vec3::ZERO,
            dielectric(rgb(0.22, 0.21, 0.19), 0.9),
        ),
        MeshInstance::new(
            Shape::Cuboid(Vec3::ONE),
            Vec3::new(-2.2, 0.5, 0.6),
            dielectric(rgb(0.70, 0.08, 0.06), 0.45),
        ),
        MeshInstance::new(
            Shape::Cuboid(Vec3::splat(1.4)),
            Vec3::new(1.4, 0.7, -0.4),
            Material {
                base_color: rgb(1.0, 0.71, 0.29),
                metallic: 1.0,
                perceptual_roughness: 0.3,
                ..Material::default()
            },
        ),
        MeshInstance::new(
            Shape::Cuboid(Vec3::new(0.8, 2.6, 0.8)),
            Vec3::new(-0.4, 1.3, -2.6),
            dielectric(rgb(0.10, 0.25, 0.65), 0.6),
        )
        .rotated(Quat::from_rotation_y(30f32.to_radians())),
        MeshInstance::new(
            Shape::Cuboid(Vec3::new(2.0, 0.3, 0.6)),
            Vec3::new(0.2, 0.15, 2.0),
            dielectric(rgb(0.85, 0.85, 0.82), 0.2),
        )
        .rotated(Quat::from_rotation_y(-20f32.to_radians())),
    ];
    Scenario {
        name: "sun_shadows",
        capability: "sun+shadow+ambient",
        width: 1280,
        height: 720,
        scene: Scene {
            camera: Camera {
                eye: Vec3::new(6.5, 5.0, 8.0),
                target: Vec3::new(0.0, 0.6, -0.5),
                fov_y: 45f32.to_radians(),
                near: 0.1,
                ev100: 10.5,
                shadow_filter: ShadowFilter::Hardware2x2,
                tonemapping: Tonemapping::AcesFitted,
            },
            clear_color: rgb(0.30, 0.48, 0.75),
            sun: Some(Sun {
                direction: Vec3::new(0.75, -1.0, 0.25).normalize(),
                color: rgb(1.0, 0.95, 0.88),
                illuminance: 10_000.0,
                shadows: Some(SunShadows::single_cascade(30.0)),
            }),
            ambient: Ambient {
                color: rgb(0.75, 0.85, 1.0),
                brightness: 1500.0,
            },
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
            shadow_maps: ShadowMapSizes::default(),
            environment: None,
            fog: None,
            meshes,
        },
    }
}

/// Bevy's default cascades (4 cascades, 10 m first far bound, 150 m, 20% overlap) and Gaussian
/// filtering over a long corridor of pillars and boxes seen from a low camera, with the sun at
/// 35° elevation so shadows cross every cascade boundary and the maximum shadow distance.
fn sun_cascades() -> Scenario {
    let stone = Material {
        base_color: rgb(0.62, 0.58, 0.52),
        perceptual_roughness: 0.8,
        ..Material::default()
    };
    let mut meshes = vec![MeshInstance::new(
        Shape::Plane(Vec2::splat(200.0)),
        Vec3::new(0.0, 0.0, -60.0),
        Material {
            base_color: rgb(0.24, 0.26, 0.20),
            perceptual_roughness: 0.9,
            ..Material::default()
        },
    )];
    for i in 0..23 {
        let z = 4.0 - 7.0 * i as f32;
        for x in [-4.5, 4.5] {
            meshes.push(MeshInstance::new(
                Shape::Cuboid(Vec3::new(0.8, 4.0, 0.8)),
                Vec3::new(x, 2.0, z),
                stone,
            ));
        }
        if i % 2 == 1 {
            let hue = [
                rgb(0.70, 0.12, 0.08),
                rgb(0.12, 0.30, 0.65),
                rgb(0.80, 0.62, 0.20),
            ][i % 3];
            meshes.push(
                MeshInstance::new(
                    Shape::Cuboid(Vec3::splat(1.5)),
                    Vec3::new(if i % 4 == 1 { -1.2 } else { 1.2 }, 0.75, z - 3.5),
                    Material {
                        base_color: hue,
                        perceptual_roughness: 0.5,
                        ..Material::default()
                    },
                )
                .rotated(Quat::from_rotation_y((15.0 * i as f32).to_radians())),
            );
        }
    }
    let (elevation, azimuth) = (35f32.to_radians(), 120f32.to_radians());
    Scenario {
        name: "sun_cascades",
        capability: "sun+cascaded shadows (Gaussian)",
        width: 1280,
        height: 720,
        scene: Scene {
            camera: Camera {
                eye: Vec3::new(0.0, 3.0, 8.0),
                target: Vec3::new(0.0, 1.0, -20.0),
                fov_y: 45f32.to_radians(),
                near: 0.1,
                ev100: 10.5,
                shadow_filter: ShadowFilter::Gaussian,
                tonemapping: Tonemapping::AcesFitted,
            },
            clear_color: rgb(0.32, 0.50, 0.78),
            sun: Some(Sun {
                direction: Vec3::new(
                    elevation.cos() * azimuth.sin(),
                    -elevation.sin(),
                    elevation.cos() * azimuth.cos(),
                ),
                color: rgb(1.0, 0.95, 0.88),
                illuminance: 10_000.0,
                shadows: Some(SunShadows::default()),
            }),
            ambient: Ambient {
                color: rgb(0.75, 0.85, 1.0),
                brightness: 1500.0,
            },
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
            shadow_maps: ShadowMapSizes::default(),
            environment: None,
            fog: None,
            meshes,
        },
    }
}
