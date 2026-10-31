//! Frustum culling as Bevy does it (`bevy_camera::primitives`): each mesh's local bounding box
//! is computed once when the [`World`](crate::world::World) uploads it, and each view (the
//! camera, every sun cascade, spot map and point-light cube face) keeps only the instances whose
//! oriented box reaches into its frustum. Skinned meshes keep a box per joint in the bind pose
//! (Bevy `SkinnedMeshBounds`); their local box is the union of those under the current palette.

use glam::{Mat4, Vec3, Vec4};

/// A mesh's local axis-aligned bounding box (Bevy `Aabb`, from the vertex positions).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Aabb {
    pub center: Vec3,
    pub half_extents: Vec3,
}

impl Aabb {
    pub fn from_points(points: &[Vec3]) -> Self {
        let Some(&first) = points.first() else {
            return Self::default();
        };
        let (min, max) = points
            .iter()
            .fold((first, first), |(min, max), &p| (min.min(p), max.max(p)));
        Self::from_min_max(min, max)
    }

    fn from_min_max(min: Vec3, max: Vec3) -> Self {
        Self {
            center: 0.5 * (min + max),
            half_extents: 0.5 * (max - min),
        }
    }

    /// The smallest box around this one under an affine `m` (Arvo; Bevy `transform_aabb`).
    pub fn transformed(&self, m: Mat4) -> Self {
        let abs = glam::Mat3::from_cols(
            m.x_axis.truncate().abs(),
            m.y_axis.truncate().abs(),
            m.z_axis.truncate().abs(),
        );
        Self {
            center: m.transform_point3(self.center),
            half_extents: abs * self.half_extents,
        }
    }

    /// The box around both.
    pub fn union(&self, other: &Self) -> Self {
        Self::from_min_max(
            (self.center - self.half_extents).min(other.center - other.half_extents),
            (self.center + self.half_extents).max(other.center + other.half_extents),
        )
    }
}

/// An instance's bounding box in world space: the mesh box under its model matrix.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Obb {
    center: Vec3,
    /// The model matrix's axes scaled by the box's half extents.
    axes: [Vec3; 3],
}

impl Obb {
    pub fn new(aabb: Aabb, world_from_local: Mat4) -> Self {
        let m = world_from_local;
        Self {
            center: m.transform_point3(aabb.center),
            axes: [
                m.x_axis.truncate() * aabb.half_extents.x,
                m.y_axis.truncate() * aabb.half_extents.y,
                m.z_axis.truncate() * aabb.half_extents.z,
            ],
        }
    }

    /// Bevy `Aabb::relative_radius`: the box's extent along `direction`.
    fn relative_radius(&self, direction: Vec3) -> f32 {
        self.axes.iter().map(|a| direction.dot(*a).abs()).sum()
    }

    /// Bevy `Sphere::intersects_obb`.
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        let v = self.center - center;
        let d = v.length();
        d <= radius || d <= radius + self.relative_radius(v / d)
    }
}

/// Up to six half spaces `n·p + d > 0` with unit normals (Bevy `ViewFrustum`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Frustum {
    planes: [Vec4; 6],
    count: usize,
}

impl Frustum {
    /// Left, right, bottom and top from `clip_from_world`, then the near plane when `near`.
    fn sides(clip_from_world: Mat4, near: bool) -> Self {
        let [r0, r1, r2, r3] = [0, 1, 2, 3].map(|i| clip_from_world.row(i));
        let mut f = Self {
            planes: [Vec4::ZERO; 6],
            count: 0,
        };
        for p in [r3 + r0, r3 - r0, r3 + r1, r3 - r1] {
            f.push(p);
        }
        if near {
            f.push(r3 + r2);
        }
        f
    }

    fn push(&mut self, plane: Vec4) {
        self.planes[self.count] = plane / plane.truncate().length();
        self.count += 1;
    }

    /// The camera: Bevy culls against the near plane but not the far one (the projection is
    /// infinite).
    pub fn camera(clip_from_world: Mat4) -> Self {
        Self::sides(clip_from_world, true)
    }

    /// A sun cascade: no near plane, since casters between the sun and the cascade still cast
    /// (Bevy's directional-light culling, with depth pancaking), but its far plane.
    pub fn cascade(clip_from_world: Mat4) -> Self {
        let mut f = Self::sides(clip_from_world, false);
        f.push(clip_from_world.row(2));
        f
    }

    /// A spot map or point-light cube face: far plane at the light's range (Bevy
    /// `from_clip_from_world_custom_far`).
    pub fn light(clip_from_world: Mat4, position: Vec3, forward: Vec3, range: f32) -> Self {
        let mut f = Self::sides(clip_from_world, true);
        let far_center = position + range * forward;
        f.push((-forward).extend(forward.dot(far_center)));
        f
    }

    /// Bevy `ViewFrustum::intersects_obb`. A centre inside a half space passes it without the
    /// box's extent (which is never negative), so most planes cost one dot product.
    pub fn intersects(&self, obb: &Obb) -> bool {
        self.planes[..self.count].iter().all(|p| {
            let n = p.truncate();
            let distance = n.dot(obb.center) + p.w;
            distance > 0.0 || distance + obb.relative_radius(n) > 0.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box_at(p: Vec3) -> Obb {
        Obb::new(
            Aabb {
                center: Vec3::ZERO,
                half_extents: Vec3::splat(0.5),
            },
            Mat4::from_translation(p),
        )
    }

    #[test]
    fn camera_keeps_what_it_sees_and_drops_what_is_behind_or_beside() {
        let view = crate::renderer::look_at(Vec3::ZERO, Vec3::NEG_Z);
        let clip = crate::renderer::perspective_infinite_reverse(1.0, 1.0, 0.1) * view;
        let f = Frustum::camera(clip);
        assert!(f.intersects(&unit_box_at(Vec3::new(0.0, 0.0, -10.0))));
        assert!(f.intersects(&unit_box_at(Vec3::new(0.0, 0.0, -10_000.0))));
        assert!(!f.intersects(&unit_box_at(Vec3::new(0.0, 0.0, 10.0))));
        assert!(!f.intersects(&unit_box_at(Vec3::new(100.0, 0.0, -10.0))));
        // Straddling the left plane.
        let edge = 10.0 * (0.5f32).tan();
        assert!(f.intersects(&unit_box_at(Vec3::new(-edge - 0.4, 0.0, -10.0))));
    }

    #[test]
    fn light_frustum_ends_at_the_range() {
        let view = crate::renderer::look_at(Vec3::ZERO, Vec3::NEG_Z);
        let clip = crate::renderer::perspective_infinite_reverse(1.0, 1.0, 0.1) * view;
        let f = Frustum::light(clip, Vec3::ZERO, Vec3::NEG_Z, 20.0);
        assert!(f.intersects(&unit_box_at(Vec3::new(0.0, 0.0, -19.0))));
        assert!(!f.intersects(&unit_box_at(Vec3::new(0.0, 0.0, -21.0))));
        assert!(unit_box_at(Vec3::new(0.0, 0.0, -20.3)).intersects_sphere(Vec3::ZERO, 20.0));
        assert!(!unit_box_at(Vec3::new(0.0, 0.0, -21.0)).intersects_sphere(Vec3::ZERO, 20.0));
    }

    #[test]
    fn aabb_spans_the_points() {
        let a = Aabb::from_points(&[Vec3::new(-1.0, 0.0, 2.0), Vec3::new(3.0, 4.0, 2.0)]);
        assert_eq!(a.center, Vec3::new(1.0, 2.0, 2.0));
        assert_eq!(a.half_extents, Vec3::new(2.0, 2.0, 0.0));
    }
}
