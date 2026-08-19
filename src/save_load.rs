//! Persisting a world between sessions.
//!
//! # What is saved
//!
//! Only what cannot be regenerated. Terrain, trees and buildings are pure
//! functions of world position, so re-running generation reproduces them
//! exactly; storing them would mean writing megabytes to recreate something the
//! CPU rebuilds in milliseconds. What genuinely cannot be recovered is the
//! player's *divergence* from that generated world — the blocks they broke and
//! placed — plus where they were standing and what time of day it was.
//!
//! # Format
//!
//! bincode over serde, versioned with an explicit `FORMAT_VERSION`. A save
//! whose version this build does not recognise is refused rather than
//! misinterpreted: a struct layout change makes bincode read whatever bytes
//! happen to be there, and a silently mangled world is worse than a missing one.
//!
//! # Web
//!
//! There is no filesystem in the browser, so saving is a no-op there and
//! loading always reports "no save". This mirrors how `settings.rs` behaves and
//! is deliberate: a partially working save that silently loses worlds would be
//! worse than an honest absence. Browser persistence would mean localStorage
//! via web-sys, which is a separate piece of work.

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::GameState;
use crate::block_types::BlockType;
use crate::chunk_manager::ChunkManager;
use crate::lighting::SunCycle;
use crate::player::Player;

/// Bump on any change to the structs below. Loading refuses anything else.
const FORMAT_VERSION: u32 = 1;

#[cfg(not(target_arch = "wasm32"))]
const SAVE_FILE: &str = ".sandbox_world_save.bin";

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct SaveGame {
    pub version: u32,
    pub player: PlayerSave,
    /// Sun angle in radians, so a world reopens at the time it was left.
    pub sun_angle: f32,
    /// Every block the player changed, as (position, block index).
    ///
    /// A sorted `Vec` rather than a `HashMap`: iteration order of a hash map is
    /// unspecified, so the same world would serialise to different bytes each
    /// time, which makes saves impossible to diff or checksum.
    pub modifications: Vec<([i32; 3], u8)>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct PlayerSave {
    /// Eye position, matching `Transform::translation`.
    pub position: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub selected_block: u8,
    pub home_position: Option<[f32; 3]>,
}

impl SaveGame {
    fn capture(
        player: &Player,
        transform: &Transform,
        sun: &SunCycle,
        chunks: &ChunkManager,
    ) -> Self {
        let mut modifications: Vec<([i32; 3], u8)> = chunks
            .modifications
            .iter()
            .map(|(pos, block)| ([pos.x, pos.y, pos.z], block.index()))
            .collect();
        modifications.sort_unstable();

        Self {
            version: FORMAT_VERSION,
            player: PlayerSave {
                position: transform.translation.to_array(),
                yaw: player.yaw,
                pitch: player.pitch,
                selected_block: player.selected_block.index(),
                home_position: player.home_position.map(|p| p.to_array()),
            },
            sun_angle: sun.angle,
            modifications,
        }
    }
}

/// Holds a save between "the player pressed Continue" and the world existing.
#[derive(Resource, Default)]
pub struct PendingLoad(pub Option<SaveGame>);

pub struct SaveLoadPlugin;

impl Plugin for SaveLoadPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PendingLoad>()
            // Ordering is load-bearing. Capturing state needs the player entity
            // and the modification map, and both are destroyed on the way out
            // of Gameplay — by `ui::teardown_world` and
            // `chunk_manager::reset_for_new_world` respectively. Autosave must
            // therefore run before either of them.
            .add_systems(
                OnExit(GameState::Gameplay),
                autosave
                    .before(crate::ui::teardown_world)
                    .before(crate::chunk_manager::reset_for_new_world),
            )
            // Applying the save must follow `spawn_player`, which is what
            // creates the entity being positioned.
            .add_systems(
                OnEnter(GameState::Gameplay),
                apply_pending_load.after(crate::player::spawn_player),
            );
    }
}

fn autosave(
    chunks: Res<ChunkManager>,
    sun: Res<SunCycle>,
    player: Option<Single<(&Transform, &Player)>>,
) {
    let Some(player) = player else {
        return;
    };
    let (transform, player) = *player;
    let save = SaveGame::capture(player, transform, &sun, &chunks);

    match write_save(&save) {
        Ok(true) => info!(
            "autosaved: {} block edits, sun angle {:.2}",
            save.modifications.len(),
            save.sun_angle
        ),
        Ok(false) => {}
        Err(error) => warn!("autosave failed: {error}"),
    }
}

fn apply_pending_load(
    mut pending: ResMut<PendingLoad>,
    mut chunks: ResMut<ChunkManager>,
    mut sun: ResMut<SunCycle>,
    player: Option<Single<(&mut Transform, &mut Player)>>,
) {
    let Some(save) = pending.0.take() else {
        return;
    };
    let Some(player) = player else {
        return;
    };
    let (mut transform, mut player) = player.into_inner();

    restore_modifications(&save, &mut chunks);

    transform.translation = Vec3::from_array(save.player.position);
    player.yaw = save.player.yaw;
    player.pitch = save.player.pitch;
    player.selected_block = BlockType::from_u8(save.player.selected_block);
    player.home_position = save.player.home_position.map(Vec3::from_array);
    sun.angle = save.sun_angle;

    info!(
        "loaded save: {} block edits, player at {:?}",
        save.modifications.len(),
        transform.translation
    );
}

/// Replays the player's block edits into a fresh `ChunkManager`.
///
/// Must happen before any chunk is generated: chunk generation consults this
/// map, so a chunk built first and edited after would need an immediate remesh
/// and the world would visibly rebuild itself in front of the player.
fn restore_modifications(save: &SaveGame, chunks: &mut ChunkManager) {
    for (pos, block) in &save.modifications {
        chunks.set_block(
            IVec3::new(pos[0], pos[1], pos[2]),
            BlockType::from_u8(*block),
        );
    }
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
fn save_path() -> std::path::PathBuf {
    crate::platform::home_dir().join(SAVE_FILE)
}

/// Writes the save. `Ok(false)` means "this platform has no filesystem", which
/// is a normal outcome on web rather than an error.
fn write_save(save: &SaveGame) -> Result<bool, String> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = save;
        return Ok(false);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let bytes = bincode::serialize(save).map_err(|e| e.to_string())?;
        std::fs::write(save_path(), bytes).map_err(|e| e.to_string())?;
        Ok(true)
    }
}

/// Reads the save, if one exists and this build understands it.
pub fn read_save() -> Option<SaveGame> {
    #[cfg(target_arch = "wasm32")]
    {
        None
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let bytes = std::fs::read(save_path()).ok()?;
        let save: SaveGame = match bincode::deserialize(&bytes) {
            Ok(save) => save,
            Err(error) => {
                warn!("save file is unreadable, ignoring it: {error}");
                return None;
            }
        };
        if save.version != FORMAT_VERSION {
            warn!(
                "save is format v{} but this build writes v{FORMAT_VERSION}; ignoring it",
                save.version
            );
            return None;
        }
        Some(save)
    }
}

/// Whether a loadable save exists, for deciding if the menu offers Continue.
pub fn save_exists() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        save_path().exists()
    }
}

/// Removes the save so the next session starts from fresh terrain.
pub fn delete_save() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = std::fs::remove_file(save_path());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> SaveGame {
        SaveGame {
            version: FORMAT_VERSION,
            player: PlayerSave {
                position: [1.5, 20.0, -3.5],
                yaw: 0.75,
                pitch: -0.25,
                selected_block: BlockType::STONE.index(),
                home_position: Some([0.5, 6.0, 0.5]),
            },
            sun_angle: 1.25,
            modifications: vec![([0, 5, 0], 1), ([2, 6, -1], 3)],
        }
    }

    /// The whole point of the format, end to end: edit a world, capture it,
    /// push it through bytes, replay it into a fresh world, and get the same
    /// blocks back. Serialisation tests alone would not catch a capture that
    /// reads the wrong field or a restore that drops edits on the floor.
    #[test]
    fn a_world_survives_capture_serialise_restore() {
        // A world the player has dug into and built on. High up, where terrain
        // generation yields air, so every solid block here is theirs.
        let mut original = ChunkManager::default();
        let edits = [
            (IVec3::new(0, 120, 0), BlockType::STONE),
            (IVec3::new(3, 121, -2), BlockType::WOOD),
            (IVec3::new(-7, 119, 4), BlockType::SAND),
        ];
        for (pos, block) in edits {
            original.set_block(pos, block);
        }

        // Built by mutation, not struct-update syntax: `Player` keeps a private
        // collision field, so only its own module can use a struct literal.
        let mut player = Player::default();
        player.yaw = 1.25;
        player.pitch = -0.4;
        player.selected_block = BlockType::LEAVES;
        let transform = Transform::from_xyz(12.5, 121.0, -8.25);
        let sun = SunCycle {
            angle: 2.5,
            ..Default::default()
        };

        let save = SaveGame::capture(&player, &transform, &sun, &original);
        let bytes = bincode::serialize(&save).expect("serialize");
        let restored: SaveGame = bincode::deserialize(&bytes).expect("deserialize");

        // Replay into a world that has never been touched.
        let mut fresh = ChunkManager::default();
        restore_modifications(&restored, &mut fresh);

        for (pos, block) in edits {
            assert_eq!(
                fresh.block_at(pos),
                block,
                "edit at {pos:?} did not survive"
            );
        }
        assert_eq!(restored.player.position, [12.5, 121.0, -8.25]);
        assert_eq!(restored.player.selected_block, BlockType::LEAVES.index());
        assert!((restored.sun_angle - 2.5).abs() < 1e-6);
    }

    /// Digging a block out is an edit like any other, and must persist as air
    /// rather than reverting to whatever terrain generation would produce.
    #[test]
    fn removed_blocks_stay_removed() {
        let mut world = ChunkManager::default();
        // Ground level near the origin, where generation produces solid terrain.
        let dug = IVec3::new(0, 0, 0);
        assert_ne!(
            world.block_at(dug),
            BlockType::AIR,
            "test needs solid ground"
        );
        world.set_block(dug, BlockType::AIR);

        let save = SaveGame::capture(
            &Player::default(),
            &Transform::default(),
            &SunCycle::default(),
            &world,
        );
        let restored: SaveGame = bincode::deserialize(&bincode::serialize(&save).unwrap()).unwrap();

        let mut fresh = ChunkManager::default();
        restore_modifications(&restored, &mut fresh);
        assert_eq!(
            fresh.block_at(dug),
            BlockType::AIR,
            "a dug-out block came back after loading"
        );
    }

    #[test]
    fn round_trips_through_bincode() {
        let original = sample();
        let bytes = bincode::serialize(&original).expect("serialize");
        let restored: SaveGame = bincode::deserialize(&bytes).expect("deserialize");
        assert_eq!(original, restored);
    }

    /// Saves must be byte-stable for the same world state, or they cannot be
    /// diffed or checksummed. `modifications` is sorted on capture precisely
    /// because hash-map iteration order is not.
    #[test]
    fn identical_state_produces_identical_bytes() {
        let a = bincode::serialize(&sample()).unwrap();
        let b = bincode::serialize(&sample()).unwrap();
        assert_eq!(a, b);
    }

    /// A block index beyond the registry must clamp to air rather than index
    /// past the end of the texture array.
    #[test]
    fn out_of_range_block_indices_are_clamped_on_load() {
        assert_eq!(BlockType::from_u8(250), BlockType::AIR);
    }

    #[test]
    fn modifications_survive_a_round_trip_in_order() {
        let mut save = sample();
        save.modifications = vec![([5, 1, 5], 3), ([0, 0, 0], 1), ([-2, 9, 4], 2)];
        save.modifications.sort_unstable();

        let bytes = bincode::serialize(&save).unwrap();
        let restored: SaveGame = bincode::deserialize(&bytes).unwrap();

        assert_eq!(restored.modifications[0].0, [-2, 9, 4]);
        assert_eq!(restored.modifications.len(), 3);
        assert!(
            restored.modifications.windows(2).all(|w| w[0] <= w[1]),
            "ordering must be preserved"
        );
    }
}
