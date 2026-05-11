use bevy::prelude::*;

use crate::block_types::BlockType;
use crate::chunk_manager::ChunkManager;
use crate::terrain::surface_y;

/// Plugin that places procedural buildings at fixed world locations on startup.
pub struct BuildingsPlugin;

impl Plugin for BuildingsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, place_buildings.in_set(crate::WorldSpawnSet));
    }
}

/// Predefined building spots in world (x, z) coordinates.
const BUILDING_SPOTS: [(f32, f32); 6] = [
    (50.0, 30.0),
    (90.0, -15.0),
    (-55.0, 45.0),
    (25.0, 85.0),
    (-30.0, -65.0),
    (110.0, 55.0),
];

/// Startup system: insert building block modifications into ChunkManager.
fn place_buildings(mut chunk_manager: ResMut<ChunkManager>) {
    for &(wx, wz) in &BUILDING_SPOTS {
        let base_y = surface_y(wx as i32, wz as i32);
        place_one_building(
            &mut chunk_manager,
            wx as i32 - 5,
            wz as i32 - 5,
            base_y + 1,
            10,
            10,
            7,
        );
    }
}

/// Place a single building as block modifications.
fn place_one_building(
    chunk_manager: &mut ChunkManager,
    bx: i32,
    bz: i32,
    base_y: i32,
    width: i32,
    depth: i32,
    height: i32,
) {
    for dx in 0..width {
        for dz in 0..depth {
            for dy in 0..height {
                let x = bx + dx;
                let y = base_y + dy;
                let z = bz + dz;

                let is_wall = dx == 0 || dx == width - 1 || dz == 0 || dz == depth - 1;
                let is_roof = dy == height - 1;

                let door_min_x = (width - 4) / 2;
                let door_max_x = door_min_x + 3;
                let is_door = dz == 0 && dx >= door_min_x && dx <= door_max_x && dy < 5;

                if is_roof {
                    chunk_manager
                        .modifications
                        .insert(IVec3::new(x, y, z), BlockType::STONE);
                } else if is_wall && !is_door {
                    chunk_manager
                        .modifications
                        .insert(IVec3::new(x, y, z), BlockType::WOOD);
                }
            }
        }
    }
}
