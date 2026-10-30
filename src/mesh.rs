//! Geometry for the built-in shapes, laid out like Bevy's primitive meshes (flat face normals).

use crate::scene::Shape;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// Triangle list with counter-clockwise front faces.
#[derive(Clone, Debug, Default)]
pub struct Geometry {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Geometry {
    pub fn of(shape: Shape) -> Self {
        let mut g = Geometry::default();
        match shape {
            Shape::Cuboid(size) => {
                let h = size * 0.5;
                for axis in 0..3 {
                    for sign in [1.0f32, -1.0] {
                        let mut n = Vec3::ZERO;
                        n[axis] = sign;
                        g.quad(n, h);
                    }
                }
            }
            Shape::Plane(size) => g.quad(Vec3::Y, Vec3::new(size.x * 0.5, 0.0, size.y * 0.5)),
        }
        g
    }

    /// One face with outward normal `n` of a box with half extents `h` (a plane when `h` is 0
    /// along `n`).
    fn quad(&mut self, n: Vec3, h: Vec3) {
        let u = if n.y.abs() > 0.5 {
            Vec3::X
        } else {
            Vec3::Y.cross(n).normalize()
        };
        let v = n.cross(u);
        let centre = n * h;
        let (eu, ev) = (u * h, v * h);
        let base = self.vertices.len() as u32;
        for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let p = centre + eu * a + ev * b;
            self.vertices.push(Vertex {
                position: p.to_array(),
                normal: n.to_array(),
            });
        }
        // v = n × u, so u × v = n and the corner order above is counter-clockwise seen from n.
        self.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    fn winding_matches_normals(g: &Geometry) {
        for tri in g.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|i| Vec3::from(g.vertices[tri[i] as usize].position));
            let n = Vec3::from(g.vertices[tri[0] as usize].normal);
            assert!((b - a).cross(c - a).dot(n) > 0.0, "clockwise face {tri:?}");
        }
    }

    #[test]
    fn cuboid_has_six_outward_faces() {
        let g = Geometry::of(Shape::Cuboid(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!((g.vertices.len(), g.indices.len()), (24, 36));
        winding_matches_normals(&g);
        for v in &g.vertices {
            let (p, n) = (Vec3::from(v.position), Vec3::from(v.normal));
            assert!(p.dot(n) > 0.0, "face {n} is not outward");
        }
        let max = g
            .vertices
            .iter()
            .fold(Vec3::ZERO, |m, v| m.max(Vec3::from(v.position)));
        assert_eq!(max, Vec3::new(0.5, 1.0, 1.5));
    }

    #[test]
    fn plane_faces_up() {
        let g = Geometry::of(Shape::Plane(Vec2::new(4.0, 2.0)));
        assert_eq!(g.indices.len(), 6);
        winding_matches_normals(&g);
        assert!(g.vertices.iter().all(|v| v.normal == [0.0, 1.0, 0.0]));
    }
}
