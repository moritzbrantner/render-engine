//! Consumer mesh data (indexed triangle lists with glTF's vertex attributes) and the built-in
//! shapes, laid out like Bevy's primitive meshes.

use glam::{Vec2, Vec3, Vec4};

/// An indexed triangle list with counter-clockwise front faces, uploaded once with
/// [`World::add_mesh`](crate::world::World::add_mesh). Attribute names follow glTF; the optional
/// attributes are either empty or hold one value per position.
#[derive(Clone, Debug, Default)]
pub struct MeshData {
    /// `POSITION`, in metres.
    pub positions: Vec<Vec3>,
    /// `NORMAL`; one per position.
    pub normals: Vec<Vec3>,
    /// `TEXCOORD_0`; needed by textured materials.
    pub uvs: Vec<Vec2>,
    /// `TANGENT`: xyz tangent, w the bitangent sign (MikkTSpace). Needed by normal maps; see
    /// [`MeshData::generate_tangents`].
    pub tangents: Vec<Vec4>,
    /// `COLOR_0`: linear RGBA multiplied into the material's base colour (Bevy
    /// `ATTRIBUTE_COLOR`).
    pub colors: Vec<Vec4>,
    pub indices: Vec<u32>,
}

/// Built-in shapes, sized like their Bevy primitives. They have positions and normals only.
#[derive(Clone, Copy, Debug)]
pub enum Shape {
    /// Bevy `Cuboid::new(x, y, z)`: full side lengths.
    Cuboid(Vec3),
    /// Bevy `Plane3d::default().mesh().size(x, z)`: an XZ plane facing +Y.
    Plane(Vec2),
    /// Bevy `Sphere::new(radius).mesh().uv(sectors, stacks)`: a UV sphere with its poles on ±Z
    /// and smooth normals.
    Sphere {
        radius: f32,
        sectors: u32,
        stacks: u32,
    },
}

/// Why [`MeshData::generate_tangents`] failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TangentError {
    /// The mesh has no `TEXCOORD_0` (or not one per position).
    NoUvs,
    /// MikkTSpace rejected the geometry.
    Failed,
}

impl std::fmt::Display for TangentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TangentError::NoUvs => write!(f, "mesh has no uvs to derive tangents from"),
            TangentError::Failed => write!(f, "mikktspace could not generate tangents"),
        }
    }
}

impl std::error::Error for TangentError {}

impl MeshData {
    pub fn shape(shape: Shape) -> Self {
        let mut g = MeshData::default();
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

    /// Fills [`MeshData::tangents`] with MikkTSpace tangents from the positions, normals and uvs,
    /// like Bevy's `Mesh::generate_tangents` and glTF importers. Normal maps baked by Blender and
    /// most tools expect exactly these.
    pub fn generate_tangents(&mut self) -> Result<(), TangentError> {
        if self.uvs.len() != self.positions.len() {
            return Err(TangentError::NoUvs);
        }
        self.tangents = vec![Vec4::ZERO; self.positions.len()];
        bevy_mikktspace::generate_tangents(self).map_err(|_| TangentError::Failed)
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
                self.positions.push(p);
                self.normals.push(p / radius);
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
        let base = self.positions.len() as u32;
        for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            self.positions.push(centre + eu * a + ev * b);
            self.normals.push(n);
        }
        // v = n × u, so u × v = n and the corner order above is counter-clockwise seen from n.
        self.indices
            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// The MikkTSpace port Bevy uses (a standalone crate without Bevy dependencies), so generated
/// tangents match Bevy's bit for bit.
impl bevy_mikktspace::Geometry for MeshData {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.indices[face * 3 + vert] as usize].to_array()
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.indices[face * 3 + vert] as usize].to_array()
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.uvs[self.indices[face * 3 + vert] as usize].to_array()
    }

    fn set_tangent(
        &mut self,
        tangent_space: Option<bevy_mikktspace::TangentSpace>,
        face: usize,
        vert: usize,
    ) {
        let i = self.indices[face * 3 + vert] as usize;
        self.tangents[i] = Vec4::from_array(tangent_space.unwrap_or_default().tangent_encoded());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn winding_matches_normals(g: &MeshData) {
        for tri in g.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|i| g.positions[tri[i] as usize]);
            let n = g.normals[tri[0] as usize];
            assert!((b - a).cross(c - a).dot(n) > 0.0, "clockwise face {tri:?}");
        }
    }

    #[test]
    fn cuboid_has_six_outward_faces() {
        let g = MeshData::shape(Shape::Cuboid(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!((g.positions.len(), g.indices.len()), (24, 36));
        winding_matches_normals(&g);
        for (p, n) in g.positions.iter().zip(&g.normals) {
            assert!(p.dot(*n) > 0.0, "face {n} is not outward");
        }
        let max = g.positions.iter().fold(Vec3::ZERO, |m, p| m.max(*p));
        assert_eq!(max, Vec3::new(0.5, 1.0, 1.5));
    }

    #[test]
    fn plane_faces_up() {
        let g = MeshData::shape(Shape::Plane(Vec2::new(4.0, 2.0)));
        assert_eq!(g.indices.len(), 6);
        winding_matches_normals(&g);
        assert!(g.normals.iter().all(|n| *n == Vec3::Y));
    }

    #[test]
    fn uv_sphere_matches_bevy_counts_and_faces_outward() {
        let (sectors, stacks) = (32, 18);
        let g = MeshData::shape(Shape::Sphere {
            radius: 0.5,
            sectors,
            stacks,
        });
        assert_eq!(g.positions.len() as u32, (sectors + 1) * (stacks + 1));
        assert_eq!(g.indices.len() as u32, 6 * sectors * (stacks - 1));
        for tri in g.indices.chunks(3) {
            let [a, b, c] = [0, 1, 2].map(|i| g.positions[tri[i] as usize]);
            assert!(
                (b - a).cross(c - a).dot(a + b + c) > 0.0,
                "inward face {tri:?}"
            );
        }
        for (p, n) in g.positions.iter().zip(&g.normals) {
            assert!((p.length() - 0.5).abs() < 1e-5);
            assert!((*n - *p * 2.0).length() < 1e-5);
        }
    }

    #[test]
    fn tangents_follow_the_u_direction() {
        let mut g = MeshData::shape(Shape::Plane(Vec2::splat(2.0)));
        assert_eq!(g.generate_tangents(), Err(TangentError::NoUvs));
        // u along +X, v along +Z.
        g.uvs = g
            .positions
            .iter()
            .map(|p| Vec2::new(p.x, p.z) * 0.5 + 0.5)
            .collect();
        g.generate_tangents().unwrap();
        for t in &g.tangents {
            assert!((t.truncate() - Vec3::X).length() < 1e-5, "tangent {t}");
            assert_eq!(t.w.abs(), 1.0);
        }
    }
}
