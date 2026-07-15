//! Procedural buildings, baked into chunk generation (same pattern as
//! trees in terrain.rs).
//!
//! HISTORY: buildings used to be inserted into ChunkManager.modifications
//! by a WorldSpawnSet system. That conflated procedural content with
//! player edits: every save carried ~4-5k building entries, the lantern
//! scan iterated them forever, and "modifications are player intent" was
//! false. Baking them into Chunk::generate keeps saves player-only and is
//! deterministic per chunk position. Old saves that still contain building
//! blocks as modifications remain loadable — they overwrite the generated
//! building with identical values.

use crate::block_types::BlockType;
use crate::chunk::{CHUNK_SIZE, CHUNK_VOLUME};
use crate::terrain::surface_y;

/// Predefined building spots in world (x, z) coordinates.
/// The spot is the CENTER reference: the building's origin corner is at
/// (spot - 5, spot - 5) and its base sits at surface_y(spot) + 1 —
/// exactly the legacy modification-based placement, so existing worlds
/// keep their buildings in the same place.
const BUILDING_SPOTS: [(f32, f32); 6] = [
    (50.0, 30.0),
    (90.0, -15.0),
    (-55.0, 45.0),
    (25.0, 85.0),
    (-30.0, -65.0),
    (110.0, 55.0),
];

const BUILDING_WIDTH: i32 = 10;
const BUILDING_DEPTH: i32 = 10;
const BUILDING_HEIGHT: i32 = 7;

/// Write building blocks intersecting this chunk into its block array.
/// Runs after tree placement in Chunk::generate_tracked, so buildings
/// override trees — matching the legacy behavior where building
/// modifications were applied on top of generated chunks.
///
/// Returns the boundary mask (ChunkNeighbors face order: -X, +X, -Y, +Y,
/// -Z, +Z) of outermost layers written, so already-meshed neighbors that
/// used terrain-predicted padding get queued for a remesh.
pub fn place_buildings_in_chunk(
    blocks: &mut [BlockType; CHUNK_VOLUME],
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
) -> [bool; 6] {
    let base_x = chunk_x * CHUNK_SIZE;
    let base_y = chunk_y * CHUNK_SIZE;
    let base_z = chunk_z * CHUNK_SIZE;
    let mut boundary_mask = [false; 6];

    for &(wx, wz) in &BUILDING_SPOTS {
        // Legacy placement: surface sampled at the SPOT, origin offset -5.
        let origin_x = wx as i32 - 5;
        let origin_z = wz as i32 - 5;

        // Cheap horizontal reject before paying for surface_y.
        if origin_x + BUILDING_WIDTH <= base_x
            || origin_x >= base_x + CHUNK_SIZE
            || origin_z + BUILDING_DEPTH <= base_z
            || origin_z >= base_z + CHUNK_SIZE
        {
            continue;
        }

        let origin_y = surface_y(wx as i32, wz as i32) + 1;
        if origin_y + BUILDING_HEIGHT <= base_y || origin_y >= base_y + CHUNK_SIZE {
            continue;
        }

        for dx in 0..BUILDING_WIDTH {
            for dz in 0..BUILDING_DEPTH {
                for dy in 0..BUILDING_HEIGHT {
                    // Classification identical to the legacy algorithm.
                    let is_wall = dx == 0
                        || dx == BUILDING_WIDTH - 1
                        || dz == 0
                        || dz == BUILDING_DEPTH - 1;
                    let is_roof = dy == BUILDING_HEIGHT - 1;

                    let door_min_x = (BUILDING_WIDTH - 4) / 2;
                    let door_max_x = door_min_x + 3;
                    let is_door = dz == 0 && dx >= door_min_x && dx <= door_max_x && dy < 5;

                    let block = if is_roof {
                        BlockType::STONE
                    } else if is_wall && !is_door {
                        BlockType::WOOD
                    } else {
                        // Door opening and interior: leave terrain untouched
                        // (legacy inserted no modification here).
                        continue;
                    };

                    let lx = origin_x + dx - base_x;
                    let ly = origin_y + dy - base_y;
                    let lz = origin_z + dz - base_z;
                    if !(0..CHUNK_SIZE).contains(&lx)
                        || !(0..CHUNK_SIZE).contains(&ly)
                        || !(0..CHUNK_SIZE).contains(&lz)
                    {
                        continue;
                    }

                    // Overwrites terrain AND trees — legacy modifications
                    // had highest priority; running after tree placement
                    // reproduces that.
                    blocks[crate::chunk::Chunk::index(lx, ly, lz)] = block;

                    if lx == 0 { boundary_mask[0] = true; }
                    if lx == CHUNK_SIZE - 1 { boundary_mask[1] = true; }
                    if ly == 0 { boundary_mask[2] = true; }
                    if ly == CHUNK_SIZE - 1 { boundary_mask[3] = true; }
                    if lz == 0 { boundary_mask[4] = true; }
                    if lz == CHUNK_SIZE - 1 { boundary_mask[5] = true; }
                }
            }
        }
    }

    boundary_mask
}
