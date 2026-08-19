//! Amanatides–Woo DDA traversal of the voxel grid.
//!
//! Used for block targeting (what the crosshair is pointing at) and for the
//! vertical probes in player collision.

use bevy::prelude::*;

use crate::block_types::BlockType;
use crate::chunk_manager::ChunkManager;

/// A solid block hit by a ray.
pub struct RayHit {
    /// World position of the block that was hit.
    pub block: IVec3,
    /// Face normal of the surface that was crossed to enter `block`. Points
    /// back toward the ray origin, so `block + normal` is the empty cell a new
    /// block should be placed into. Zero when the origin is already inside a
    /// solid block.
    pub normal: IVec3,
    /// Distance along the ray to the entry point. Not currently read — block
    /// selection only needs the hit cell — but it is the natural place for a
    /// reach falloff or a distance-scaled highlight to come from.
    #[allow(dead_code)]
    pub distance: f32,
}

/// Walks the voxel grid along `direction` and returns the first non-air block
/// within `max_distance` world units.
///
/// `direction` need not be normalised, but `distance` is only meaningful in
/// world units when it is.
pub fn cast_ray(
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    chunks: &ChunkManager,
) -> Option<RayHit> {
    // Current voxel, and the step direction along each axis.
    let mut voxel = origin.floor().as_ivec3();
    let step = IVec3::new(
        if direction.x >= 0.0 { 1 } else { -1 },
        if direction.y >= 0.0 { 1 } else { -1 },
        if direction.z >= 0.0 { 1 } else { -1 },
    );

    // Ray distance covered by one full voxel step along each axis. An axis with
    // zero direction never advances, hence the infinity.
    let t_delta = Vec3::new(
        if direction.x != 0.0 {
            (1.0 / direction.x).abs()
        } else {
            f32::INFINITY
        },
        if direction.y != 0.0 {
            (1.0 / direction.y).abs()
        } else {
            f32::INFINITY
        },
        if direction.z != 0.0 {
            (1.0 / direction.z).abs()
        } else {
            f32::INFINITY
        },
    );

    // Ray distance to the next voxel boundary on each axis.
    let next_boundary = |i: usize| -> f32 {
        let (o, d, v, s) = (origin[i], direction[i], voxel[i], step[i]);
        if d == 0.0 {
            return f32::INFINITY;
        }
        let boundary = if s > 0 { v as f32 + 1.0 } else { v as f32 };
        (boundary - o) / d
    };
    let mut t_max = Vec3::new(next_boundary(0), next_boundary(1), next_boundary(2));

    let mut normal = IVec3::ZERO;
    let mut distance = 0.0;

    loop {
        if chunks.block_at(voxel) != BlockType::AIR {
            return Some(RayHit {
                block: voxel,
                normal,
                distance,
            });
        }

        // Advance along whichever axis reaches its next boundary first.
        let axis = if t_max.x < t_max.y && t_max.x < t_max.z {
            0
        } else if t_max.y < t_max.z {
            1
        } else {
            2
        };

        if t_max[axis] > max_distance {
            return None;
        }

        distance = t_max[axis];
        voxel[axis] += step[axis];
        t_max[axis] += t_delta[axis];

        normal = IVec3::ZERO;
        normal[axis] = -step[axis];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Well above any generated terrain, so `block_at` falls through to air and
    /// only explicitly placed blocks are solid.
    const SKY_Y: i32 = 100;

    fn world_with(blocks: &[(IVec3, BlockType)]) -> ChunkManager {
        let mut cm = ChunkManager::default();
        for &(pos, kind) in blocks {
            cm.set_block(pos, kind);
        }
        cm
    }

    #[test]
    fn hits_the_first_solid_block_and_reports_the_entry_face() {
        let target = IVec3::new(0, SKY_Y, 0);
        let cm = world_with(&[(target, BlockType::STONE)]);

        // Travelling along -Z from z = 5 toward a block occupying z in 0..1.
        let hit = cast_ray(
            Vec3::new(0.5, SKY_Y as f32 + 0.5, 5.0),
            Vec3::NEG_Z,
            10.0,
            &cm,
        )
        .expect("ray should reach the block");

        assert_eq!(hit.block, target);
        // Entered through the +Z face, so the normal points back down the ray.
        assert_eq!(hit.normal, IVec3::Z);
        // ... which makes the adjacent empty cell the one a placed block goes in.
        assert_eq!(hit.block + hit.normal, IVec3::new(0, SKY_Y, 1));
        assert!((hit.distance - 4.0).abs() < 1e-3, "got {}", hit.distance);
    }

    #[test]
    fn stops_at_max_distance() {
        let cm = world_with(&[(IVec3::new(0, SKY_Y, 0), BlockType::STONE)]);
        let hit = cast_ray(
            Vec3::new(0.5, SKY_Y as f32 + 0.5, 5.0),
            Vec3::NEG_Z,
            2.0,
            &cm,
        );
        assert!(hit.is_none(), "block is 4 units away, reach is 2");
    }

    #[test]
    fn passes_through_empty_space() {
        let cm = world_with(&[]);
        assert!(cast_ray(Vec3::new(0.5, SKY_Y as f32, 0.5), Vec3::X, 32.0, &cm).is_none());
    }

    #[test]
    fn nearest_block_wins_when_several_are_in_line() {
        let near = IVec3::new(0, SKY_Y, 2);
        let far = IVec3::new(0, SKY_Y, 0);
        let cm = world_with(&[(near, BlockType::STONE), (far, BlockType::DIRT)]);

        let hit = cast_ray(
            Vec3::new(0.5, SKY_Y as f32 + 0.5, 6.0),
            Vec3::NEG_Z,
            16.0,
            &cm,
        )
        .expect("should hit the nearer block");
        assert_eq!(hit.block, near);
    }
}
