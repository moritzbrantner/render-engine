use crate::scenarios::{self, Scenario};

/// The scenarios that need no file system (`scenarios::all()` reads the glTF props and figure).
pub fn find(name: &str) -> Option<Scenario> {
    let mut all = vec![
        scenarios::sun_shadows(),
        scenarios::sun_cascades(),
        scenarios::many_instances(),
    ];
    all.extend(scenarios::tonemap_range());
    all.extend(scenarios::ibl_materials());
    all.extend(scenarios::fog_modes());
    all.extend(scenarios::point_spot());
    all.extend(scenarios::sky_day_cycle());
    all.into_iter().find(|s| s.name == name)
}
