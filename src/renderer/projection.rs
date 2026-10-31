//! View and projection matrices and the sun's cascade fit.

use glam::{Mat3, Mat4, Vec3, Vec4};

/// The rotation of Bevy `Transform::looking_to(direction, up)`.
pub(super) fn looking_to(direction: Vec3, up: Vec3) -> Mat3 {
    let back = -direction.normalize();
    let right = up.cross(back).normalize();
    Mat3::from_cols(right, back.cross(right), back)
}

/// Right-handed view matrix with +Y up (the inverse of Bevy `Transform::looking_at(target, Y)`).
pub fn look_at(eye: Vec3, target: Vec3) -> Mat4 {
    (Mat4::from_translation(eye) * Mat4::from_mat3(looking_to(target - eye, Vec3::Y))).inverse()
}

/// Bevy's right-handed infinite reverse-Z perspective projection.
pub(super) fn perspective_infinite_reverse(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let h = 1.0 / (0.5 * fov_y).tan();
    Mat4::from_cols(
        Vec4::new(h / aspect, 0.0, 0.0, 0.0),
        Vec4::new(0.0, h, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 0.0, -1.0),
        Vec4::new(0.0, 0.0, near, 0.0),
    )
}

/// One sun shadow cascade, fitted to the camera frustum slice `near..far` (view distances)
/// exactly like Bevy's `calculate_cascade` (diameter rounded up to whole metres, centre snapped
/// to texels).
#[derive(Clone, Copy, Debug)]
pub struct Cascade {
    pub clip_from_world: Mat4,
    pub texel_size: f32,
}

impl Cascade {
    pub fn fit(
        near: f32,
        far: f32,
        map_size: u32,
        sun_direction: Vec3,
        fov_y: f32,
        aspect: f32,
        world_from_view: Mat4,
    ) -> Self {
        // Light orientation of Bevy `Transform::looking_to(direction, Vec3::Y)`.
        let back = -sun_direction.normalize();
        let right = Vec3::Y.cross(back).normalize();
        let up = back.cross(right);
        let world_from_light = Mat4::from_mat3(Mat3::from_cols(right, up, back));
        let light_from_view = world_from_light.transpose() * world_from_view;

        let t = (fov_y * 0.5).tan();
        let corners = [-near, -far].map(|z| {
            let a = z.abs() * t;
            [
                Vec3::new(a * aspect, -a, z),
                Vec3::new(a * aspect, a, z),
                Vec3::new(-a * aspect, a, z),
                Vec3::new(-a * aspect, -a, z),
            ]
        });
        let corners: [Vec3; 8] = [corners[0], corners[1]].concat().try_into().unwrap();
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for c in corners {
            let p = light_from_view.transform_point3(c);
            min = min.min(p);
            max = max.max(p);
        }
        let body = (corners[0] - corners[6]).length_squared();
        let far = (corners[4] - corners[6]).length_squared();
        let diameter = body.max(far).sqrt().ceil();
        let texel_size = diameter / map_size as f32;
        let centre = Vec3::new(
            (0.5 * (min.x + max.x) / texel_size).floor() * texel_size,
            (0.5 * (min.y + max.y) / texel_size).floor() * texel_size,
            max.z,
        );
        let t = world_from_light.transpose();
        let cascade_from_world =
            Mat4::from_cols(t.x_axis, t.y_axis, t.z_axis, (-centre).extend(1.0));
        let r = (max.z - min.z).recip();
        let clip_from_cascade = Mat4::from_cols(
            Vec4::new(2.0 / diameter, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 / diameter, 0.0, 0.0),
            Vec4::new(0.0, 0.0, r, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 1.0),
        );
        Self {
            clip_from_world: clip_from_cascade * cascade_from_world,
            texel_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascade_covers_the_view_slice() {
        let eye = Vec3::new(6.0, 5.0, 8.0);
        let world_from_view = look_at(eye, Vec3::ZERO).inverse();
        let c = Cascade::fit(
            0.1,
            30.0,
            2048,
            Vec3::new(-0.4, -1.0, -0.3),
            45f32.to_radians(),
            16.0 / 9.0,
            world_from_view,
        );
        // A point the camera looks at, inside the shadow range, lands inside the cascade.
        let p = c.clip_from_world.project_point3(Vec3::ZERO);
        assert!(
            p.x.abs() < 1.0 && p.y.abs() < 1.0 && (0.0..=1.0).contains(&p.z),
            "{p}"
        );
        assert!(c.texel_size > 0.0 && c.texel_size < 0.05);
    }

    #[test]
    fn default_cascades_follow_bevy_spacing() {
        let ranges = crate::scene::SunShadows::default().cascade_ranges();
        let expected = [(0.1, 10.0), (8.0, 24.662), (19.73, 60.822), (48.66, 150.0)];
        assert_eq!(ranges.len(), expected.len());
        for ((near, far), (en, ef)) in ranges.into_iter().zip(expected) {
            assert!(
                (near - en).abs() < 0.01 && (far - ef).abs() < 0.01,
                "{near}..{far}"
            );
        }
    }
}
