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
            Shape::Sphere {
                radius,
                sectors,
                stacks,
            } => g.uv_sphere(radius, sectors, stacks),
        }
        g
    }

    /// Bevy `SphereMeshBuilder::uv`, vertex for vertex and index for index.
    fn uv_sphere(&mut self, radius: f32, sectors: u32, stacks: u32) {
        use std::f32::consts::PI;
        let sector_step = 2.0 * PI / sectors as f32;
        let stack_step = PI / stacks as f32;
        for i in 0..=stacks {
            let stack_angle = PI / 2.0 - i as f32 * stack_step;
            let xy = radius * stack_angle.cos();
            let z = radius * stack_angle.sin();
            for j in 0..=sectors {
                let sector_angle = j as f32 * sector_step;
                let p = Vec3::new(xy * sector_angle.cos(), xy * sector_angle.sin(), z);
                self.vertices.push(Vertex {
                    position: p.to_array(),
                    normal: (p / radius).to_array(),
                });
            }
        }
        for i in 0..stacks {
            for j in 0..sectors {
                let k1 = i * (sectors + 1) + j;
                let k2 = k1 + sectors + 1;
                if i != 0 {
                    self.indices.extend([k1, k2, k1 + 1]);
                }
                if i != stacks - 1 {
                    self.indices.extend([k1 + 1, k2, k2 + 1]);
                }
            }
        }
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

    #[test]
    fn uv_sphere_matches_bevy_counts_and_faces_outward() {
        let (sectors, stacks) = (32, 18);
        let g = Geometry::of(Shape::Sphere {
            radius: 0.5,
            sectors,
            stacks,
        });
        assert_eq!(g.vertices.len() as u32, (sectors + 1) * (stacks + 1));
        assert_eq!(g.indices.len() as u32, 6 * sectors * (stacks - 1));
        for tri in g.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|i| Vec3::from(g.vertices[tri[i] as usize].position));
            assert!(
                (b - a).cross(c - a).dot(a + b + c) > 0.0,
                "inward face {tri:?}"
            );
        }
        for v in &g.vertices {
            let p = Vec3::from(v.position);
            assert!((p.length() - 0.5).abs() < 1e-5);
            assert!((Vec3::from(v.normal) - p * 2.0).length() < 1e-5);
        }
    }
}
