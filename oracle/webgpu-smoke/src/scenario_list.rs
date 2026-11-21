use crate::scenarios::{self, Scenario};

/// A lighting scenario of `scenarios::all()` by name, built on demand (`scenarios::all()` would
/// build every one, glTF content included).
pub fn find(name: &str) -> Option<Scenario> {
    match name {
        "sun_shadows" => return Some(scenarios::sun_shadows()),
        "sun_cascades" => return Some(scenarios::sun_cascades()),
        "many_instances" => return Some(scenarios::many_instances()),
        "many_lights" => return Some(scenarios::many_lights()),
        "textured_props" => return Some(scenarios::textured_props()),
        "skinned_figure" => return Some(scenarios::skinned_figure()),
        "hud_overlay" => return Some(scenarios::hud_overlay()),
        _ => {}
    }
    let mut all = Vec::new();
    all.extend(scenarios::tonemap_range());
    all.extend(scenarios::ibl_materials());
    all.extend(scenarios::fog_modes());
    all.extend(scenarios::point_spot());
    all.extend(scenarios::sky_day_cycle());
    all.into_iter().find(|s| s.name == name)
}

/// The quaternius props of `textured_props` (`assets/props/`).
const PROPS: &[&str] = &[
    "props/Anvil.gltf",
    "props/Anvil.bin",
    "props/Cauldron.gltf",
    "props/Cauldron.bin",
    "props/FarmCrate_Empty.gltf",
    "props/FarmCrate_Empty.bin",
    "props/Bush_Common.gltf",
    "props/Bush_Common.bin",
    "props/textures/Leaves_TwistedTree_C.png",
    "props/textures/T_Trim_Furniture_BaseColor.png",
    "props/textures/T_Trim_Furniture_Normal.png",
    "props/textures/T_Trim_Furniture_ORM.png",
    "props/textures/T_Trim_Metal_BaseColor.png",
    "props/textures/T_Trim_Metal_Normal.png",
    "props/textures/T_Trim_Metal_ORM.png",
];

/// The rigged figure of `skinned_figure` (`assets/figure/`).
const FIGURE: &[&str] = &[
    "figure/Superhero_Male.gltf",
    "figure/Superhero_Male.bin",
    "figure/textures/T_Superhero_Male_Dark.png",
    "figure/textures/T_Superhero_Male_Normal.png",
    "figure/textures/T_Superhero_Male_Roughness.png",
];

/// The files under `assets/` a scenario reads (`content::asset`): the page fetches them and
/// hands them to the wasm before building it, since the engine and the wasm read no files.
pub fn files(name: &str) -> Vec<&'static str> {
    match name {
        "textured_props" => PROPS.to_vec(),
        // It reuses `textured_props`' scene, which loads the props.
        "skinned_figure" => [PROPS, FIGURE].concat(),
        _ => Vec::new(),
    }
}
