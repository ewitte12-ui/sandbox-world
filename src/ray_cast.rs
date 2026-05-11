use bevy::prelude::*;

use crate::block_types::BlockType;
use crate::chunk_manager::ChunkManager;

/// Result of a ray cast hitting a solid block.
pub struct RayHit {
    /// Block position that was hit.
    pub pos: IVec3,
    /// Face normal of the hit surface.
    pub normal: IVec3,
    /// Distance along the ray to the hit point.
    pub t: f32,
}

/// DDA ray cast through the voxel grid. Returns the first non-air block hit
/// within `max_dist` world units, along with the face normal and distance.
///
/// Ported from the Swift `castRay` implementation.
pub fn cast_ray(
    origin: Vec3,
    direction: Vec3,
    max_dist: f32,
    chunk_manager: &ChunkManager,
) -> Option<RayHit> {
    let dir = direction;

    let mut ix = origin.x.floor() as i32;
    let mut iy = origin.y.floor() as i32;
    let mut iz = origin.z.floor() as i32;

    let step_x: i32 = if dir.x >= 0.0 { 1 } else { -1 };
    let step_y: i32 = if dir.y >= 0.0 { 1 } else { -1 };
    let step_z: i32 = if dir.z >= 0.0 { 1 } else { -1 };

    let t_delta_x = if dir.x != 0.0 {
        (1.0 / dir.x).abs()
    } else {
        f32::INFINITY
    };
    let t_delta_y = if dir.y != 0.0 {
        (1.0 / dir.y).abs()
    } else {
        f32::INFINITY
    };
    let t_delta_z = if dir.z != 0.0 {
        (1.0 / dir.z).abs()
    } else {
        f32::INFINITY
    };

    let mut t_max_x = if dir.x >= 0.0 {
        ((ix + 1) as f32 - origin.x) / dir.x
    } else {
        (ix as f32 - origin.x) / dir.x
    };
    let mut t_max_y = if dir.y >= 0.0 {
        ((iy + 1) as f32 - origin.y) / dir.y
    } else {
        (iy as f32 - origin.y) / dir.y
    };
    let mut t_max_z = if dir.z >= 0.0 {
        ((iz + 1) as f32 - origin.z) / dir.z
    } else {
        (iz as f32 - origin.z) / dir.z
    };

    let mut normal = IVec3::ZERO;
    let max_steps = (max_dist * 2.0) as i32 + 10;

    for _ in 0..max_steps {
        let block = chunk_manager.block_at(IVec3::new(ix, iy, iz));
        if block != BlockType::AIR {
            let t = (t_max_x - t_delta_x)
                .min(t_max_y - t_delta_y)
                .min(t_max_z - t_delta_z)
                .max(0.0);
            return Some(RayHit {
                pos: IVec3::new(ix, iy, iz),
                normal,
                t,
            });
        }

        if t_max_x < t_max_y {
            if t_max_x < t_max_z {
                if t_max_x > max_dist {
                    return None;
                }
                ix += step_x;
                t_max_x += t_delta_x;
                normal = IVec3::new(-step_x, 0, 0);
            } else {
                if t_max_z > max_dist {
                    return None;
                }
                iz += step_z;
                t_max_z += t_delta_z;
                normal = IVec3::new(0, 0, -step_z);
            }
        } else {
            if t_max_y < t_max_z {
                if t_max_y > max_dist {
                    return None;
                }
                iy += step_y;
                t_max_y += t_delta_y;
                normal = IVec3::new(0, -step_y, 0);
            } else {
                if t_max_z > max_dist {
                    return None;
                }
                iz += step_z;
                t_max_z += t_delta_z;
                normal = IVec3::new(0, 0, -step_z);
            }
        }
    }

    None
}
