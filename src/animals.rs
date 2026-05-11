use std::f32::consts::PI;

use bevy::{
    asset::RenderAssetUsages, gltf::GltfAssetLabel, mesh::Indices, prelude::*,
    render::render_resource::PrimitiveTopology,
};
use rand::Rng;

use crate::block_types::BlockType;
use crate::chunk_manager::ChunkManager;
use crate::terrain;
use crate::GameState;
use crate::save_load::FileDialogOpen;

// ── Mesh data helpers ─────────────────────────────────────────────────

/// Intermediate mesh data before conversion to Bevy `Mesh`.
struct MeshData {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl MeshData {
    fn new() -> Self {
        Self {
            positions: Vec::new(),
            normals: Vec::new(),
            colors: Vec::new(),
            indices: Vec::new(),
        }
    }
}

/// Generate a faceted ellipsoid (flat-shaded UV sphere stretched by radii).
fn make_ellipsoid(radii: Vec3, segments: u32, color: [f32; 4]) -> MeshData {
    let mut data = MeshData::new();
    let rings = segments;
    let sectors = segments * 2;

    // Generate vertices on the sphere surface
    let mut sphere_verts: Vec<Vec3> = Vec::new();
    for r in 0..=rings {
        let phi = PI * (r as f32) / (rings as f32);
        for s in 0..=sectors {
            let theta = 2.0 * PI * (s as f32) / (sectors as f32);
            let x = phi.sin() * theta.cos() * radii.x;
            let y = phi.cos() * radii.y;
            let z = phi.sin() * theta.sin() * radii.z;
            sphere_verts.push(Vec3::new(x, y, z));
        }
    }

    // Build triangles with flat normals (duplicate verts per triangle)
    for r in 0..rings {
        for s in 0..sectors {
            let cols = sectors + 1;
            let i0 = r * cols + s;
            let i1 = r * cols + s + 1;
            let i2 = (r + 1) * cols + s;
            let i3 = (r + 1) * cols + s + 1;

            let v0 = sphere_verts[i0 as usize];
            let v1 = sphere_verts[i1 as usize];
            let v2 = sphere_verts[i2 as usize];
            let v3 = sphere_verts[i3 as usize];

            // Upper triangle
            if r > 0 {
                let normal = (v1 - v0).cross(v2 - v0).normalize_or_zero();
                let n = [normal.x, normal.y, normal.z];
                let base = data.positions.len() as u32;
                data.positions.push(v0.into());
                data.positions.push(v1.into());
                data.positions.push(v2.into());
                data.normals.push(n);
                data.normals.push(n);
                data.normals.push(n);
                data.colors.push(color);
                data.colors.push(color);
                data.colors.push(color);
                data.indices.push(base);
                data.indices.push(base + 1);
                data.indices.push(base + 2);
            }

            // Lower triangle
            if r < rings - 1 {
                let normal = (v2 - v1).cross(v3 - v1).normalize_or_zero();
                let n = [normal.x, normal.y, normal.z];
                let base = data.positions.len() as u32;
                data.positions.push(v1.into());
                data.positions.push(v3.into());
                data.positions.push(v2.into());
                data.normals.push(n);
                data.normals.push(n);
                data.normals.push(n);
                data.colors.push(color);
                data.colors.push(color);
                data.colors.push(color);
                data.indices.push(base);
                data.indices.push(base + 1);
                data.indices.push(base + 2);
            }
        }
    }

    data
}

/// Generate a tapered cylinder (for legs, tail, snout, ears).
/// Centered at origin, height along Y axis (bottom at -height/2, top at +height/2).
fn make_tapered_cylinder(
    bottom_r: f32,
    top_r: f32,
    height: f32,
    segments: u32,
    color: [f32; 4],
) -> MeshData {
    let mut data = MeshData::new();
    let half_h = height / 2.0;

    // Generate ring vertices
    let mut bottom_ring: Vec<Vec3> = Vec::new();
    let mut top_ring: Vec<Vec3> = Vec::new();
    for i in 0..=segments {
        let theta = 2.0 * PI * (i as f32) / (segments as f32);
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        bottom_ring.push(Vec3::new(cos_t * bottom_r, -half_h, sin_t * bottom_r));
        top_ring.push(Vec3::new(cos_t * top_r, half_h, sin_t * top_r));
    }

    // Side faces (quads as 2 triangles, flat-shaded)
    for i in 0..segments {
        let b0 = bottom_ring[i as usize];
        let b1 = bottom_ring[i as usize + 1];
        let t0 = top_ring[i as usize];
        let t1 = top_ring[i as usize + 1];

        // Triangle 1: b0, b1, t0
        let n1 = (b1 - b0).cross(t0 - b0).normalize_or_zero();
        let n1a = [n1.x, n1.y, n1.z];
        let base = data.positions.len() as u32;
        data.positions.push(b0.into());
        data.positions.push(b1.into());
        data.positions.push(t0.into());
        data.normals.push(n1a);
        data.normals.push(n1a);
        data.normals.push(n1a);
        data.colors.push(color);
        data.colors.push(color);
        data.colors.push(color);
        data.indices.push(base);
        data.indices.push(base + 1);
        data.indices.push(base + 2);

        // Triangle 2: b1, t1, t0
        let n2 = (t1 - b1).cross(t0 - b1).normalize_or_zero();
        let n2a = [n2.x, n2.y, n2.z];
        let base = data.positions.len() as u32;
        data.positions.push(b1.into());
        data.positions.push(t1.into());
        data.positions.push(t0.into());
        data.normals.push(n2a);
        data.normals.push(n2a);
        data.normals.push(n2a);
        data.colors.push(color);
        data.colors.push(color);
        data.colors.push(color);
        data.indices.push(base);
        data.indices.push(base + 1);
        data.indices.push(base + 2);
    }

    // Bottom cap
    let center_b = Vec3::new(0.0, -half_h, 0.0);
    let cap_n_b = [0.0, -1.0, 0.0];
    for i in 0..segments {
        let base = data.positions.len() as u32;
        data.positions.push(center_b.into());
        data.positions.push(bottom_ring[i as usize + 1].into());
        data.positions.push(bottom_ring[i as usize].into());
        data.normals.push(cap_n_b);
        data.normals.push(cap_n_b);
        data.normals.push(cap_n_b);
        data.colors.push(color);
        data.colors.push(color);
        data.colors.push(color);
        data.indices.push(base);
        data.indices.push(base + 1);
        data.indices.push(base + 2);
    }

    // Top cap
    let center_t = Vec3::new(0.0, half_h, 0.0);
    let cap_n_t = [0.0, 1.0, 0.0];
    for i in 0..segments {
        let base = data.positions.len() as u32;
        data.positions.push(center_t.into());
        data.positions.push(top_ring[i as usize].into());
        data.positions.push(top_ring[i as usize + 1].into());
        data.normals.push(cap_n_t);
        data.normals.push(cap_n_t);
        data.normals.push(cap_n_t);
        data.colors.push(color);
        data.colors.push(color);
        data.colors.push(color);
        data.indices.push(base);
        data.indices.push(base + 1);
        data.indices.push(base + 2);
    }

    data
}

/// Merge multiple MeshData parts, each with its own transform, into one.
fn merge_mesh_data(parts: &[(MeshData, Transform)]) -> MeshData {
    let mut merged = MeshData::new();
    for (part, xform) in parts {
        let base_idx = merged.positions.len() as u32;
        let rot_mat = Mat3::from_quat(xform.rotation);
        for (i, pos) in part.positions.iter().enumerate() {
            let p = Vec3::from(*pos);
            let transformed = xform.rotation * (p * xform.scale) + xform.translation;
            merged.positions.push(transformed.into());

            let n = Vec3::from(part.normals[i]);
            let transformed_n = (rot_mat * n).normalize_or_zero();
            merged.normals.push(transformed_n.into());

            merged.colors.push(part.colors[i]);
        }
        for &idx in &part.indices {
            merged.indices.push(base_idx + idx);
        }
    }
    merged
}

/// Convert MeshData into a Bevy Mesh.
fn build_final_mesh(data: &MeshData) -> Mesh {
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, data.positions.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, data.normals.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, data.colors.clone());
    mesh.insert_indices(Indices::U32(data.indices.clone()));
    mesh
}

// ── Body plan definitions ─────────────────────────────────────────────

/// Shape primitive for a body part.
enum Shape {
    Ellipsoid { radii: Vec3 },
    Cylinder { bottom_r: f32, top_r: f32, height: f32 },
}

/// Definition of a single static body part.
struct PartDef {
    shape: Shape,
    offset: Vec3,
    rotation: Quat,
    color: [f32; 4],
    segments: u32,
}

/// Leg definition with shape and attachment point.
struct LegDef {
    bottom_r: f32,
    top_r: f32,
    height: f32,
    segments: u32,
    color: [f32; 4],
    /// Attachment offset from body center (local space). X is for left leg (negated for right).
    front_attach: Vec3,
    back_attach: Vec3,
}

/// Motion parameters preserved from original.
struct MotionParams {
    min_speed: f32,
    max_speed: f32,
    _leg_freq: f32,
}

/// Complete body plan for one animal type.
struct BodyPlan {
    parts: Vec<PartDef>,
    leg: LegDef,
    leg_height: f32,
    motion: MotionParams,
}

fn squirrel_plan() -> BodyPlan {
    let brown = [0.72, 0.36, 0.10, 1.0];
    let tan = [0.85, 0.70, 0.55, 1.0];
    let dark_brown = [0.55, 0.26, 0.07, 1.0];
    let black = [0.05, 0.05, 0.05, 1.0];
    let white = [0.95, 0.95, 0.95, 1.0];

    BodyPlan {
        parts: vec![
            // Body
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.15, 0.11, 0.25) },
                offset: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                color: brown,
                segments: 16,
            },
            // Belly highlight
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.12, 0.08, 0.20) },
                offset: Vec3::new(0.0, -0.04, 0.0),
                rotation: Quat::IDENTITY,
                color: tan,
                segments: 14,
            },
            // Head
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.13, 0.14, 0.13) },
                offset: Vec3::new(0.0, 0.06, 0.28),
                rotation: Quat::IDENTITY,
                color: brown,
                segments: 16,
            },
            // Left eye
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.035, 0.04, 0.02) },
                offset: Vec3::new(-0.09, 0.10, 0.36),
                rotation: Quat::IDENTITY,
                color: black,
                segments: 10,
            },
            // Right eye
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.035, 0.04, 0.02) },
                offset: Vec3::new(0.09, 0.10, 0.36),
                rotation: Quat::IDENTITY,
                color: black,
                segments: 10,
            },
            // Nose
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.03, 0.025, 0.03) },
                offset: Vec3::new(0.0, 0.02, 0.40),
                rotation: Quat::IDENTITY,
                color: black,
                segments: 10,
            },
            // Snout
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.06, top_r: 0.03, height: 0.08 },
                offset: Vec3::new(0.0, 0.0, 0.36),
                rotation: Quat::from_rotation_x(-PI / 2.0),
                color: white,
                segments: 14,
            },
            // Left ear
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.04, top_r: 0.015, height: 0.08 },
                offset: Vec3::new(-0.06, 0.20, 0.26),
                rotation: Quat::from_rotation_z(0.15),
                color: brown,
                segments: 12,
            },
            // Right ear
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.04, top_r: 0.015, height: 0.08 },
                offset: Vec3::new(0.06, 0.20, 0.26),
                rotation: Quat::from_rotation_z(-0.15),
                color: brown,
                segments: 12,
            },
            // Tail — bushy, curving upward
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.10, 0.12, 0.22) },
                offset: Vec3::new(0.0, 0.14, -0.32),
                rotation: Quat::from_rotation_x(0.4),
                color: dark_brown,
                segments: 16,
            },
        ],
        leg: LegDef {
            bottom_r: 0.035,
            top_r: 0.045,
            height: 0.18,
            segments: 14,
            color: dark_brown,
            front_attach: Vec3::new(0.10, -0.11, 0.15),
            back_attach: Vec3::new(0.10, -0.11, -0.15),
        },
        leg_height: 0.18,
        motion: MotionParams { min_speed: 3.5, max_speed: 8.0, _leg_freq: 10.0 },
    }
}

fn dog_plan() -> BodyPlan {
    let light_gray = [0.78, 0.78, 0.78, 1.0];
    let white = [0.92, 0.92, 0.92, 1.0];
    let dark_gray = [0.40, 0.40, 0.40, 1.0];
    let black = [0.05, 0.05, 0.05, 1.0];
    let pink = [0.85, 0.55, 0.55, 1.0];

    BodyPlan {
        parts: vec![
            // Body
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.26, 0.23, 0.42) },
                offset: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                color: light_gray,
                segments: 16,
            },
            // Head
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.20, 0.18, 0.21) },
                offset: Vec3::new(0.0, 0.12, 0.48),
                rotation: Quat::IDENTITY,
                color: white,
                segments: 16,
            },
            // Snout
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.10, top_r: 0.06, height: 0.16 },
                offset: Vec3::new(0.0, 0.06, 0.64),
                rotation: Quat::from_rotation_x(-PI / 2.0),
                color: white,
                segments: 14,
            },
            // Nose
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.04, 0.03, 0.03) },
                offset: Vec3::new(0.0, 0.08, 0.72),
                rotation: Quat::IDENTITY,
                color: black,
                segments: 10,
            },
            // Left eye
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.04, 0.04, 0.02) },
                offset: Vec3::new(-0.12, 0.18, 0.62),
                rotation: Quat::IDENTITY,
                color: black,
                segments: 10,
            },
            // Right eye
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.04, 0.04, 0.02) },
                offset: Vec3::new(0.12, 0.18, 0.62),
                rotation: Quat::IDENTITY,
                color: black,
                segments: 10,
            },
            // Left ear (floppy, angled down)
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.06, 0.12, 0.08) },
                offset: Vec3::new(-0.16, 0.18, 0.42),
                rotation: Quat::from_rotation_z(0.5),
                color: dark_gray,
                segments: 14,
            },
            // Right ear (floppy, angled down)
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.06, 0.12, 0.08) },
                offset: Vec3::new(0.16, 0.18, 0.42),
                rotation: Quat::from_rotation_z(-0.5),
                color: dark_gray,
                segments: 14,
            },
            // Tongue
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.03, 0.01, 0.05) },
                offset: Vec3::new(0.0, 0.0, 0.70),
                rotation: Quat::from_rotation_x(0.3),
                color: pink,
                segments: 10,
            },
            // Tail (thin, upward)
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.04, top_r: 0.02, height: 0.30 },
                offset: Vec3::new(0.0, 0.20, -0.42),
                rotation: Quat::from_rotation_x(-0.4),
                color: light_gray,
                segments: 14,
            },
        ],
        leg: LegDef {
            bottom_r: 0.055,
            top_r: 0.075,
            height: 0.34,
            segments: 14,
            color: [0.62, 0.62, 0.62, 1.0],
            front_attach: Vec3::new(0.16, -0.23, 0.25),
            back_attach: Vec3::new(0.16, -0.23, -0.25),
        },
        leg_height: 0.34,
        motion: MotionParams { min_speed: 2.0, max_speed: 5.5, _leg_freq: 6.5 },
    }
}

fn horse_plan() -> BodyPlan {
    let brown = [0.58, 0.38, 0.18, 1.0];
    let dark_brown = [0.30, 0.18, 0.08, 1.0];
    let black = [0.05, 0.05, 0.05, 1.0];

    BodyPlan {
        parts: vec![
            // Body
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.45, 0.38, 0.95) },
                offset: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                color: brown,
                segments: 18,
            },
            // Neck
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.22, top_r: 0.16, height: 0.55 },
                offset: Vec3::new(0.0, 0.35, 0.75),
                rotation: Quat::from_rotation_x(-0.5),
                color: brown,
                segments: 16,
            },
            // Head
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.16, 0.20, 0.30) },
                offset: Vec3::new(0.0, 0.60, 1.05),
                rotation: Quat::from_rotation_x(0.2),
                color: brown,
                segments: 16,
            },
            // Snout/muzzle
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.12, top_r: 0.10, height: 0.18 },
                offset: Vec3::new(0.0, 0.50, 1.30),
                rotation: Quat::from_rotation_x(-PI / 2.0 + 0.2),
                color: brown,
                segments: 14,
            },
            // Left eye
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.04, 0.05, 0.03) },
                offset: Vec3::new(-0.14, 0.66, 1.12),
                rotation: Quat::IDENTITY,
                color: black,
                segments: 10,
            },
            // Right eye
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.04, 0.05, 0.03) },
                offset: Vec3::new(0.14, 0.66, 1.12),
                rotation: Quat::IDENTITY,
                color: black,
                segments: 10,
            },
            // Left ear
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.04, top_r: 0.02, height: 0.12 },
                offset: Vec3::new(-0.08, 0.80, 1.00),
                rotation: Quat::from_rotation_z(0.2),
                color: dark_brown,
                segments: 12,
            },
            // Right ear
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.04, top_r: 0.02, height: 0.12 },
                offset: Vec3::new(0.08, 0.80, 1.00),
                rotation: Quat::from_rotation_z(-0.2),
                color: dark_brown,
                segments: 12,
            },
            // Mane ridge (series of small bumps along neck)
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.04, 0.08, 0.35) },
                offset: Vec3::new(0.0, 0.52, 0.80),
                rotation: Quat::from_rotation_x(-0.3),
                color: dark_brown,
                segments: 14,
            },
            // Tail (long, flowing)
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.06, top_r: 0.03, height: 0.60 },
                offset: Vec3::new(0.0, -0.05, -1.10),
                rotation: Quat::from_rotation_x(0.6),
                color: dark_brown,
                segments: 14,
            },
            // Nostrils
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.03, 0.02, 0.02) },
                offset: Vec3::new(0.0, 0.46, 1.38),
                rotation: Quat::IDENTITY,
                color: black,
                segments: 8,
            },
        ],
        leg: LegDef {
            bottom_r: 0.07,
            top_r: 0.10,
            height: 0.70,
            segments: 14,
            color: [0.14, 0.10, 0.05, 1.0],
            front_attach: Vec3::new(0.28, -0.38, 0.55),
            back_attach: Vec3::new(0.28, -0.38, -0.55),
        },
        leg_height: 0.70,
        motion: MotionParams { min_speed: 3.0, max_speed: 8.0, _leg_freq: 4.5 },
    }
}

fn raccoon_plan() -> BodyPlan {
    let gray = [0.58, 0.58, 0.58, 1.0];
    let dark_gray = [0.30, 0.30, 0.30, 1.0];
    let black = [0.08, 0.08, 0.08, 1.0];
    let white = [0.90, 0.90, 0.90, 1.0];
    let light_gray = [0.75, 0.75, 0.75, 1.0];

    BodyPlan {
        parts: vec![
            // Body
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.21, 0.16, 0.30) },
                offset: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                color: gray,
                segments: 16,
            },
            // Head
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.15, 0.14, 0.15) },
                offset: Vec3::new(0.0, 0.06, 0.34),
                rotation: Quat::IDENTITY,
                color: light_gray,
                segments: 16,
            },
            // Dark eye mask — left
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.07, 0.05, 0.04) },
                offset: Vec3::new(-0.08, 0.10, 0.44),
                rotation: Quat::IDENTITY,
                color: black,
                segments: 12,
            },
            // Dark eye mask — right
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.07, 0.05, 0.04) },
                offset: Vec3::new(0.08, 0.10, 0.44),
                rotation: Quat::IDENTITY,
                color: black,
                segments: 12,
            },
            // Left eye (on mask)
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.025, 0.03, 0.015) },
                offset: Vec3::new(-0.08, 0.11, 0.47),
                rotation: Quat::IDENTITY,
                color: white,
                segments: 8,
            },
            // Right eye (on mask)
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.025, 0.03, 0.015) },
                offset: Vec3::new(0.08, 0.11, 0.47),
                rotation: Quat::IDENTITY,
                color: white,
                segments: 8,
            },
            // White forehead stripe
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.03, 0.04, 0.06) },
                offset: Vec3::new(0.0, 0.16, 0.40),
                rotation: Quat::IDENTITY,
                color: white,
                segments: 10,
            },
            // Pointed snout
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.07, top_r: 0.03, height: 0.10 },
                offset: Vec3::new(0.0, 0.02, 0.46),
                rotation: Quat::from_rotation_x(-PI / 2.0),
                color: light_gray,
                segments: 14,
            },
            // Nose
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.025, 0.02, 0.02) },
                offset: Vec3::new(0.0, 0.03, 0.51),
                rotation: Quat::IDENTITY,
                color: black,
                segments: 8,
            },
            // Left ear
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.035, top_r: 0.015, height: 0.07 },
                offset: Vec3::new(-0.09, 0.20, 0.32),
                rotation: Quat::from_rotation_z(0.2),
                color: dark_gray,
                segments: 12,
            },
            // Right ear
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.035, top_r: 0.015, height: 0.07 },
                offset: Vec3::new(0.09, 0.20, 0.32),
                rotation: Quat::from_rotation_z(-0.2),
                color: dark_gray,
                segments: 12,
            },
            // Tail segment 1 (gray)
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.07, top_r: 0.06, height: 0.10 },
                offset: Vec3::new(0.0, 0.02, -0.35),
                rotation: Quat::from_rotation_x(PI / 2.0 + 0.3),
                color: gray,
                segments: 12,
            },
            // Tail segment 2 (dark ring)
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.065, top_r: 0.055, height: 0.08 },
                offset: Vec3::new(0.0, 0.05, -0.44),
                rotation: Quat::from_rotation_x(PI / 2.0 + 0.4),
                color: dark_gray,
                segments: 12,
            },
            // Tail segment 3 (gray)
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.055, top_r: 0.045, height: 0.08 },
                offset: Vec3::new(0.0, 0.10, -0.52),
                rotation: Quat::from_rotation_x(PI / 2.0 + 0.5),
                color: gray,
                segments: 12,
            },
            // Tail segment 4 (dark ring)
            PartDef {
                shape: Shape::Cylinder { bottom_r: 0.045, top_r: 0.035, height: 0.08 },
                offset: Vec3::new(0.0, 0.16, -0.59),
                rotation: Quat::from_rotation_x(PI / 2.0 + 0.6),
                color: dark_gray,
                segments: 12,
            },
            // Tail tip (dark)
            PartDef {
                shape: Shape::Ellipsoid { radii: Vec3::new(0.035, 0.035, 0.04) },
                offset: Vec3::new(0.0, 0.22, -0.64),
                rotation: Quat::IDENTITY,
                color: black,
                segments: 10,
            },
        ],
        leg: LegDef {
            bottom_r: 0.045,
            top_r: 0.06,
            height: 0.24,
            segments: 14,
            color: [0.32, 0.32, 0.32, 1.0],
            front_attach: Vec3::new(0.14, -0.16, 0.18),
            back_attach: Vec3::new(0.14, -0.16, -0.18),
        },
        leg_height: 0.24,
        motion: MotionParams { min_speed: 1.2, max_speed: 3.0, _leg_freq: 5.5 },
    }
}

/// Build body plans in type-index order: [squirrel=0, dog=1, horse=2, raccoon=3]
fn all_body_plans() -> [BodyPlan; 4] {
    [squirrel_plan(), dog_plan(), horse_plan(), raccoon_plan()]
}

/// Generate the merged body mesh from a body plan.
fn build_body_mesh(plan: &BodyPlan) -> MeshData {
    let parts: Vec<(MeshData, Transform)> = plan
        .parts
        .iter()
        .map(|p| {
            let mesh = match &p.shape {
                Shape::Ellipsoid { radii } => make_ellipsoid(*radii, p.segments, p.color),
                Shape::Cylinder { bottom_r, top_r, height } => {
                    make_tapered_cylinder(*bottom_r, *top_r, *height, p.segments, p.color)
                }
            };
            let xform = Transform::from_translation(p.offset).with_rotation(p.rotation);
            (mesh, xform)
        })
        .collect();
    merge_mesh_data(&parts)
}

/// Generate a single leg mesh from a leg definition.
fn build_leg_mesh(leg: &LegDef) -> MeshData {
    make_tapered_cylinder(leg.bottom_r, leg.top_r, leg.height, leg.segments, leg.color)
}

// ── Data structures ────────────────────────────────────────────────────

/// Per-animal runtime data (position, AI state).
pub struct AnimalData {
    pub position: Vec3,
    pub yaw: f32,
    pub speed: f32,
    pub turn_timer: f32,
    pub animal_type: u32,
    pub local_time: f32,
    pub is_idle: bool,
    /// True if this animal uses a scene-based animated model.
    pub is_scene: bool,
}

/// Resource holding all animal data and part entity references.
#[derive(Resource)]
pub struct Animals {
    pub data: Vec<AnimalData>,
    /// Flat array: `[animal_index * 5 + part]` where part 0=body, 1=FL, 2=FR, 3=BL, 4=BR
    pub part_entities: Vec<Entity>,
    /// Leg height per type index, for ground offset calculation.
    pub leg_heights: [f32; 4],
}

/// Marker on scene root entities so we can find them to link animations.
#[derive(Component)]
pub struct AnimatedAnimal {
    pub animal_index: usize,
}

/// Resource storing animation graph handles and node indices per animated type.
#[derive(Resource)]
pub struct AnimalAnimations {
    /// (graph_handle, walk_node_index) per animated type key
    pub horse: (Handle<AnimationGraph>, AnimationNodeIndex),
    pub raccoon: (Handle<AnimationGraph>, AnimationNodeIndex),
    pub dinosaur: (Handle<AnimationGraph>, AnimationNodeIndex),
}

/// Marker component on each animal entity.
#[derive(Component)]
pub struct AnimalPart {
    pub animal_index: usize,
    pub part_type: AnimalPartType,
}

#[derive(Clone, Copy, PartialEq)]
pub enum AnimalPartType {
    Body,
    FrontLeftLeg,
    FrontRightLeg,
    BackLeftLeg,
    BackRightLeg,
}

// ── Plugin ─────────────────────────────────────────────────────────────

pub struct AnimalPlugin;

impl Plugin for AnimalPlugin {
    fn build(&self, app: &mut App) {
        // DEPENDENCY: requires ChunkManagerPlugin (for ground height queries).
        // spawn_animals runs at Startup and uses terrain::terrain_height_at()
        // directly (ChunkManager may not have chunks loaded yet at Startup).
        // update_animals runs in Update and queries ChunkManager for ground
        // height, falling back to terrain generation if chunks aren't loaded.
        app.add_systems(Update, spawn_animals.in_set(crate::WorldSpawnSet))
            // setup_animal_animations must run before update_animals because
            // it attaches AnimationGraphs that update_animals then controls.
            // setup_animal_animations runs always (one-time graph attachment
            // after scene loads). update_animals is gated to Gameplay only.
            .add_systems(Update, setup_animal_animations
                .run_if(in_state(GameState::Gameplay)))
            .add_systems(
                Update,
                update_animals
                    .run_if(in_state(GameState::Gameplay))
                    .run_if(not(resource_exists::<FileDialogOpen>)),
            );
    }
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Get ground height for an animal at the given world XZ.
fn animal_ground_height(x: f32, z: f32, cm: Option<&ChunkManager>) -> f32 {
    if let Some(manager) = cm {
        let bx = x.floor() as i32;
        let bz = z.floor() as i32;
        let sy = terrain::surface_y(bx, bz);
        for by in (sy - 20..=sy + 2).rev() {
            if manager.block_at(IVec3::new(bx, by, bz)) != BlockType::AIR {
                return (by + 1) as f32;
            }
        }
    }
    terrain::terrain_height_at(x, z)
}

// ── Startup system ─────────────────────────────────────────────────────

fn spawn_animals(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
    world_id: Res<crate::WorldInstanceId>,
) {
    let mut rng = rand::thread_rng();
    let plans = all_body_plans();

    // Material for vertex-colored glTF models (raccoon)
    let gltf_vcol_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        double_sided: true,
        cull_mode: None,
        ..default()
    });

    // Load glTF meshes for animals that use 3D models.
    // Each model has a GLTF_SCALE chosen so the model's bounding box maps to
    // a reasonable in-game size, and a GLTF_Y_OFFSET that compensates for
    // the model's origin not being at ground level (feet).
    //
    // Squirrel (type 0): full-res model with vertex colors
    //   Bounds: X[-0.625,0.625] Y[-0.724,2.022] Z[-1.206,1.351]
    let squirrel_gltf_mesh: Handle<Mesh> =
        asset_server.load(
            GltfAssetLabel::Primitive { mesh: 0, primitive: 0 }
                .from_asset("animals/squirrel.glb"),
        );
    const SQUIRREL_GLTF_SCALE: f32 = 0.18;
    const SQUIRREL_GLTF_Y_OFFSET: f32 = 0.724 * SQUIRREL_GLTF_SCALE;

    // Dog (type 1): textured model from assets/animals/dog.glb
    //   Bounds: X[-1.17,1.17] Y[-0.04,5.46] Z[-5.14,4.88]
    //   Has textures — use its own material
    //   Total length ~10, target ~0.8 units → scale ~0.08
    let dog_gltf_mesh: Handle<Mesh> =
        asset_server.load(
            GltfAssetLabel::Primitive { mesh: 0, primitive: 0 }
                .from_asset("animals/dog.glb"),
        );
    let dog_gltf_mat: Handle<StandardMaterial> =
        asset_server.load(
            GltfAssetLabel::Material { index: 0, is_scale_inverted: false }
                .from_asset("animals/dog.glb"),
        );
    const DOG_GLTF_SCALE: f32 = 0.14;
    // Feet at Y=-0.04, negligible offset
    const DOG_GLTF_Y_OFFSET: f32 = 0.04 * DOG_GLTF_SCALE;

    // Horse (type 2): animated scene from assets/animals/horse.glb
    //   Bounds: X[-0.26,0.26] Y[-0.004,1.90] Z[-0.70,1.40]
    //   Has 1 animation: "Horse_walk"
    let horse_scene: Handle<Scene> =
        asset_server.load(GltfAssetLabel::Scene(0).from_asset("animals/horse.glb"));
    let horse_anim_clip: Handle<AnimationClip> =
        asset_server.load(GltfAssetLabel::Animation(0).from_asset("animals/horse.glb"));
    let (horse_graph, horse_walk_node) = AnimationGraph::from_clip(horse_anim_clip);
    let horse_graph_handle = graphs.add(horse_graph);
    const HORSE_GLTF_SCALE: f32 = 1.8;
    const HORSE_GLTF_Y_OFFSET: f32 = 0.004 * HORSE_GLTF_SCALE;

    // Raccoon (type 3): animated scene from assets/animals/raccoon.glb
    //   Bounds: X[-0.83,0.85] Y[-0.05,2.56] Z[-3.58,2.70]
    //   Has 1 animation: "Animation"
    let raccoon_scene: Handle<Scene> =
        asset_server.load(GltfAssetLabel::Scene(0).from_asset("animals/raccoon.glb"));
    let raccoon_anim_clip: Handle<AnimationClip> =
        asset_server.load(GltfAssetLabel::Animation(0).from_asset("animals/raccoon.glb"));
    let (raccoon_graph, raccoon_walk_node) = AnimationGraph::from_clip(raccoon_anim_clip);
    let raccoon_graph_handle = graphs.add(raccoon_graph);
    const RACCOON_GLTF_SCALE: f32 = 0.18;
    const RACCOON_GLTF_Y_OFFSET: f32 = 0.05 * RACCOON_GLTF_SCALE;

    // Chicken (type 4): textured model from assets/animals/chicken.glb
    //   Bounds: X[-0.88,0.85] Y[-0.03,3.84] Z[-1.95,1.95]
    //   57K tris, textured
    let chicken_gltf_mesh: Handle<Mesh> =
        asset_server.load(
            GltfAssetLabel::Primitive { mesh: 0, primitive: 0 }
                .from_asset("animals/chicken.glb"),
        );
    let chicken_gltf_mat: Handle<StandardMaterial> =
        asset_server.load(
            GltfAssetLabel::Material { index: 0, is_scale_inverted: false }
                .from_asset("animals/chicken.glb"),
        );
    const CHICKEN_GLTF_SCALE: f32 = 0.13;
    const CHICKEN_GLTF_Y_OFFSET: f32 = 0.03 * CHICKEN_GLTF_SCALE;

    // Dinosaur (type 5): animated scene from assets/animals/dinosaur.glb
    //   Bounds: X[-3.26,-1.66] Y[-0.006,3.13] Z[-8.45,0.48]
    //   14966 tris, 1 animation: "run", FK skeleton
    //   Model is offset in X — will be centered by scene. Height ~3.1, make it BIG (~5 blocks tall)
    let dino_scene: Handle<Scene> =
        asset_server.load(GltfAssetLabel::Scene(0).from_asset("animals/dinosaur.glb"));
    let dino_anim_clip: Handle<AnimationClip> =
        asset_server.load(GltfAssetLabel::Animation(0).from_asset("animals/dinosaur.glb"));
    let (dino_graph, dino_run_node) = AnimationGraph::from_clip(dino_anim_clip);
    let dino_graph_handle = graphs.add(dino_graph);
    const DINO_GLTF_SCALE: f32 = 1.6;
    const DINO_GLTF_Y_OFFSET: f32 = 0.006 * DINO_GLTF_SCALE;

    // Build shared mesh handles per type (procedural)
    let mut body_mesh_handles: Vec<Handle<Mesh>> = Vec::with_capacity(4);
    let mut leg_mesh_handles: Vec<Handle<Mesh>> = Vec::with_capacity(4);
    let mut leg_heights = [0.0_f32; 4];

    for (i, plan) in plans.iter().enumerate() {
        let body_data = build_body_mesh(plan);
        body_mesh_handles.push(meshes.add(build_final_mesh(&body_data)));

        let leg_data = build_leg_mesh(&plan.leg);
        leg_mesh_handles.push(meshes.add(build_final_mesh(&leg_data)));

        leg_heights[i] = plan.leg_height;
    }

    // 1 vertex-colored body material shared by procedural types
    let body_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });

    // 4 leg materials (solid color per type)
    let mut leg_mats: Vec<Handle<StandardMaterial>> = Vec::with_capacity(4);
    for plan in &plans {
        let c = plan.leg.color;
        leg_mats.push(materials.add(StandardMaterial {
            base_color: Color::linear_rgba(c[0], c[1], c[2], c[3]),
            ..default()
        }));
    }

    // 75 regular animals (15 per type × 5 types) + 1 dinosaur.
    // Spawned in a ring around origin at varying distances.
    // Dinosaur spawns closer (30-80 blocks) to ensure the player encounters it.
    // Regular animals spread wider (15-180 blocks) to populate the landscape.
    let regular_animals = 75u32;
    let num_regular_types = 5u32;
    let total_animals = regular_animals + 1;
    let mut animal_data: Vec<AnimalData> = Vec::with_capacity(total_animals as usize);
    let mut part_entities: Vec<Entity> = Vec::with_capacity(total_animals as usize * 5);

    for i in 0..total_animals {
        let animal_type = if i < regular_animals { i % num_regular_types } else { 5 }; // 5 = dinosaur
        let t = animal_type as usize;

        let angle = (i as f32) / (total_animals as f32) * PI * 2.0 + rng.gen_range(-0.5_f32..0.5);
        let dist = if animal_type == 5 {
            rng.gen_range(30.0_f32..80.0) // dinosaur spawns somewhat close
        } else {
            rng.gen_range(15.0_f32..180.0)
        };
        let x = angle.cos() * dist;
        let z = angle.sin() * dist;
        let ground_y = terrain::terrain_height_at(x, z);

        let (lo, hi, _) = TYPE_MOTION[t];

        let is_scene = animal_type == 2 || animal_type == 3 || animal_type == 5; // horse + raccoon + dinosaur

        animal_data.push(AnimalData {
            position: Vec3::new(x, ground_y, z),
            yaw: rng.gen_range(0.0..PI * 2.0),
            speed: rng.gen_range(lo..hi),
            turn_timer: rng.gen_range(1.0_f32..6.0),
            animal_type,
            local_time: rng.gen_range(0.0_f32..20.0),
            is_idle: false,
            is_scene,
        });

        let animal_idx = i as usize;

        if is_scene {
            // Animated scene-based animal (horse, raccoon)
            let (scene, gltf_scale) = match animal_type {
                2 => (horse_scene.clone(), HORSE_GLTF_SCALE),
                3 => (raccoon_scene.clone(), RACCOON_GLTF_SCALE),
                5 => (dino_scene.clone(), DINO_GLTF_SCALE),
                _ => unreachable!(),
            };

            let body_entity = commands
                .spawn((
                    crate::WorldEntity,
                    crate::WorldScoped(world_id.0),
                    AnimatedAnimal { animal_index: animal_idx },
                    AnimalPart {
                        animal_index: animal_idx,
                        part_type: AnimalPartType::Body,
                    },
                    SceneRoot(scene),
                    Transform::from_scale(Vec3::splat(gltf_scale)),
                ))
                .id();
            part_entities.push(body_entity);
        } else {
            // Static mesh animal (squirrel, dog, chicken)
            let (gltf_mesh, gltf_material, gltf_scale): (Handle<Mesh>, Handle<StandardMaterial>, f32) = match animal_type {
                0 => (squirrel_gltf_mesh.clone(), gltf_vcol_mat.clone(), SQUIRREL_GLTF_SCALE),
                1 => (dog_gltf_mesh.clone(), dog_gltf_mat.clone(), DOG_GLTF_SCALE),
                4 => (chicken_gltf_mesh.clone(), chicken_gltf_mat.clone(), CHICKEN_GLTF_SCALE),
                _ => unreachable!(),
            };

            let body_entity = commands
                .spawn((
                    crate::WorldEntity,
                    crate::WorldScoped(world_id.0),
                    AnimalPart {
                        animal_index: animal_idx,
                        part_type: AnimalPartType::Body,
                    },
                    Mesh3d(gltf_mesh),
                    MeshMaterial3d(gltf_material),
                    Transform::from_scale(Vec3::splat(gltf_scale)),
                ))
                .id();
            part_entities.push(body_entity);
        }
        // Placeholder slots for 4 legs
        for _ in 0..4 {
            part_entities.push(Entity::PLACEHOLDER);
        }
    }

    commands.insert_resource(Animals {
        data: animal_data,
        part_entities,
        leg_heights,
    });

    commands.insert_resource(AnimalAnimations {
        horse: (horse_graph_handle, horse_walk_node),
        raccoon: (raccoon_graph_handle, raccoon_walk_node),
        dinosaur: (dino_graph_handle, dino_run_node),
    });
}

// ── Animation setup system ─────────────────────────────────────────────

/// When AnimationPlayers appear (after scene loads), attach the graph and start playing.
fn setup_animal_animations(
    mut commands: Commands,
    animations: Option<Res<AnimalAnimations>>,
    animals: Option<Res<Animals>>,
    animated_animals: Query<(Entity, &AnimatedAnimal)>,
    children_query: Query<&Children>,
    mut players: Query<(Entity, &mut AnimationPlayer), Added<AnimationPlayer>>,
) {
    let Some(animations) = animations else { return };
    let Some(animals) = animals else { return };

    for (player_entity, mut player) in &mut players {
        // Walk up to find which AnimatedAnimal this player belongs to
        for (root_entity, animated) in &animated_animals {
            let is_descendant = children_query
                .iter_descendants(root_entity)
                .any(|child| child == player_entity);

            if is_descendant || root_entity == player_entity {
                let animal_type = animals.data[animated.animal_index].animal_type;
                let (graph_handle, walk_node) = match animal_type {
                    2 => (&animations.horse.0, animations.horse.1),
                    3 => (&animations.raccoon.0, animations.raccoon.1),
                    5 => (&animations.dinosaur.0, animations.dinosaur.1),
                    _ => continue,
                };

                // Attach graph and start walk animation looping
                commands
                    .entity(player_entity)
                    .insert(AnimationGraphHandle(graph_handle.clone()));
                player.play(walk_node).repeat();
                break;
            }
        }
    }
}

// ── Update system ──────────────────────────────────────────────────────

/// Motion parameters per type: (min_speed, max_speed, leg_freq).
/// Speeds are in blocks/second. leg_freq is the animation cycle rate (Hz).
///
/// Tuned relative to player_speed (22.0): most animals are slower than the
/// player so they can be observed but not threatening. Dinosaurs (type 5)
/// can match walking speed to create pressure.
///
/// WARNING: these should eventually move to DevSettings for runtime tuning.
/// Currently hardcoded because DevSettings is not yet wired into animal systems.
const TYPE_MOTION: [(f32, f32, f32); 6] = [
    (3.5, 8.0, 10.0),  // squirrel — fast, jittery
    (2.0, 5.5, 6.5),   // dog — medium pace
    (3.0, 8.0, 4.5),   // horse — moderate speed, slow gait
    (1.2, 3.0, 5.5),   // raccoon — slow waddle
    (1.5, 4.0, 8.0),   // chicken — slow body, fast legs
    (4.0, 10.0, 5.0),  // dinosaur — fast and menacing
];

fn update_animals(
    time: Res<Time>,
    animals: Option<ResMut<Animals>>,
    chunk_manager: Option<Res<ChunkManager>>,
    mut transforms: Query<&mut Transform>,
    _animated_animals: Query<(Entity, &AnimatedAnimal)>,
    children_query: Query<&Children>,
    mut anim_players: Query<&mut AnimationPlayer>,
    animations: Option<Res<AnimalAnimations>>,
) {
    let Some(mut animals) = animals else {
        #[cfg(debug_assertions)]
        {
            static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                bevy::log::info!("update_animals skipped: Animals resource not yet initialized");
            }
        }
        return;
    };
    let dt = time.delta_secs();
    let cm = chunk_manager.as_deref();
    let mut rng = rand::thread_rng();

    // --- AI update (unchanged) ---
    for data in &mut animals.data {
        data.local_time += dt;
        data.turn_timer -= dt;

        let t = data.animal_type as usize;
        let (lo, hi, _) = TYPE_MOTION[t];

        if data.turn_timer <= 0.0 {
            if rng.gen::<f32>() < 0.20 {
                data.is_idle = true;
                data.speed = 0.0;
                data.turn_timer = rng.gen_range(0.5_f32..2.5);
            } else {
                data.is_idle = false;
                data.speed = rng.gen_range(lo..hi);
                data.yaw += rng.gen_range(-PI * 0.5..PI * 0.5);
                data.turn_timer = rng.gen_range(2.0_f32..7.0);
            }
        }

        if !data.is_idle {
            let new_x = data.position.x + data.yaw.sin() * data.speed * dt;
            let new_z = data.position.z + data.yaw.cos() * data.speed * dt;
            let ground_y = animal_ground_height(new_x, new_z, cm);

            // Cliff detection: if the ground ahead is >1.2 blocks higher than
            // current position, the animal turns away. 1.2 is slightly above
            // 1 full block — animals can step up 1-block ledges but not climb
            // walls. This also prevents animals from walking off steep cliffs
            // (the check is one-directional: height *increase* only).
            if ground_y - data.position.y > 1.2 {
                data.yaw += rng.gen_range(PI * 0.5..PI);
                data.turn_timer = rng.gen_range(1.0_f32..3.0);
            } else {
                data.position.x = new_x;
                data.position.z = new_z;
            }
        }

        data.position.y = animal_ground_height(data.position.x, data.position.z, cm);
    }

    // --- Transform update ---
    // glTF model constants — duplicated from spawn_animals because both
    // systems need them. Scale values are chosen so the model's bounding
    // box fits the intended in-game size (e.g., squirrel: 2.7 unit model
    // × 0.18 = ~0.5 blocks tall). Y offsets compensate for models whose
    // origin isn't at their feet (e.g., dog feet at Y=-0.04 in model space).
    // WARNING: these are duplicated — changing one without the other will
    // cause spawn/update mismatch. Should be consolidated into a shared const.
    const SQUIRREL_GLTF_SCALE: f32 = 0.18;
    const SQUIRREL_GLTF_Y_OFFSET: f32 = 0.724 * SQUIRREL_GLTF_SCALE;
    const DOG_GLTF_SCALE: f32 = 0.14;
    const DOG_GLTF_Y_OFFSET: f32 = 0.04 * DOG_GLTF_SCALE;
    const HORSE_GLTF_SCALE: f32 = 1.8;
    const HORSE_GLTF_Y_OFFSET: f32 = 0.004 * HORSE_GLTF_SCALE;
    const RACCOON_GLTF_SCALE: f32 = 0.18;
    const RACCOON_GLTF_Y_OFFSET: f32 = 0.05 * RACCOON_GLTF_SCALE;
    const CHICKEN_GLTF_SCALE: f32 = 0.13;
    const CHICKEN_GLTF_Y_OFFSET: f32 = 0.03 * CHICKEN_GLTF_SCALE;
    const DINO_GLTF_SCALE: f32 = 1.6;
    const DINO_GLTF_Y_OFFSET: f32 = 0.006 * DINO_GLTF_SCALE;

    for (idx, data) in animals.data.iter().enumerate() {
        let t = data.animal_type as usize;
        let (_, _, leg_freq) = TYPE_MOTION[t];

        let is_moving = !data.is_idle;
        let phase = data.local_time * leg_freq;
        let ground = data.position.y;
        let body_entity = animals.part_entities[idx * 5];

        let (gltf_scale, gltf_y_offset, gltf_yaw_offset) = match data.animal_type {
            0 => (SQUIRREL_GLTF_SCALE, SQUIRREL_GLTF_Y_OFFSET, 0.0),
            1 => (DOG_GLTF_SCALE, DOG_GLTF_Y_OFFSET, 0.0),
            2 => (HORSE_GLTF_SCALE, HORSE_GLTF_Y_OFFSET, 0.0),
            3 => (RACCOON_GLTF_SCALE, RACCOON_GLTF_Y_OFFSET, 0.0),
            4 => (CHICKEN_GLTF_SCALE, CHICKEN_GLTF_Y_OFFSET, 0.0),
            5 => (DINO_GLTF_SCALE, DINO_GLTF_Y_OFFSET, 0.0),
            _ => unreachable!(),
        };

        if data.is_scene {
            // Scene-based animated animal: set position/rotation/scale on root,
            // skeletal animation handles the walk cycle
            let body_y = ground + gltf_y_offset;
            if let Ok(mut transform) = transforms.get_mut(body_entity) {
                *transform = Transform::from_translation(Vec3::new(
                    data.position.x,
                    body_y,
                    data.position.z,
                ))
                .with_rotation(Quat::from_rotation_y(data.yaw + gltf_yaw_offset))
                .with_scale(Vec3::splat(gltf_scale));
            }

            // Control animation playback speed based on movement
            if let Some(ref anims) = animations {
                let walk_node = match data.animal_type {
                    2 => anims.horse.1,
                    3 => anims.raccoon.1,
                    5 => anims.dinosaur.1,
                    _ => continue,
                };

                // Find the AnimationPlayer in this scene's descendants
                for child in children_query.iter_descendants(body_entity) {
                    if let Ok(mut player) = anim_players.get_mut(child) {
                        if is_moving {
                            // Scale animation speed with movement speed.
                            // 4.0 = reference speed at which animation plays at 1x.
                            // Clamped to [0.5, 2.0] to prevent unnaturally slow/fast
                            // animations at extreme speeds.
                            let speed_factor = (data.speed / 4.0).clamp(0.5, 2.0);
                            if let Some(anim) = player.animation_mut(walk_node) {
                                if anim.is_paused() {
                                    anim.resume();
                                }
                                anim.set_speed(speed_factor);
                            }
                        } else if let Some(anim) = player.animation_mut(walk_node) {
                            anim.pause();
                        }
                        break;
                    }
                }
            }
        } else {
            // Static mesh animal: procedural body animation.
            // bob (0.06): vertical bounce synced to leg cycle — conveys locomotion
            // sway (0.04): slow lateral oscillation (half leg freq) — adds weight
            // lean (-0.08): forward pitch proportional to speed — faster = leaning forward
            //   capped at speed/6.0 so max lean is -0.08 rad (~4.6°) at 6+ blocks/sec
            // roll (0.03): slight body roll at half freq — secondary motion for life
            let body_y = ground + gltf_y_offset + if is_moving { phase.sin() * 0.06 } else { 0.0 };
            let sway = if is_moving { (phase * 0.5).sin() * 0.04 } else { 0.0 };
            let lean = if is_moving { -0.08 * (data.speed / 6.0).min(1.0) } else { 0.0 };
            let roll = if is_moving { (phase * 0.5).cos() * 0.03 } else { 0.0 };

            if let Ok(mut transform) = transforms.get_mut(body_entity) {
                let body_rot = Quat::from_rotation_y(data.yaw + gltf_yaw_offset + sway)
                    * Quat::from_rotation_x(lean)
                    * Quat::from_rotation_z(roll);
                *transform = Transform::from_translation(Vec3::new(
                    data.position.x,
                    body_y,
                    data.position.z,
                ))
                .with_rotation(body_rot)
                .with_scale(Vec3::splat(gltf_scale));
            }
        }
    }
}
