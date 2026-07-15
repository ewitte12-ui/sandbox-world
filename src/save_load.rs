use std::path::PathBuf;

use bevy::prelude::*;
use bevy::tasks::{futures::check_ready, IoTaskPool, Task};
use serde::{Deserialize, Serialize};

use crate::block_types::BlockType;
use crate::chunk_manager::ChunkManager;
use crate::player::Player;
use crate::GameState;

/// Run condition: returns true when the menu overlay is NOT open.
/// Save/load systems must not run while the overlay is active.
fn menu_overlay_closed(menu_state: Option<Res<crate::ui::MenuState>>) -> bool {
    !menu_state.map(|m| m.is_open).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Save formats
//
// V2 (current, written): binary — 8-byte magic "MWSAVE02" + bincode of
// SaveDataV2. Modifications are grouped per chunk with packed u16 local
// indices: ~3.5 bytes/modification vs ~60 in the legacy pretty JSON, and
// no text parsing on load. Format detection is by CONTENT (magic sniff),
// never by file extension, so a legacy .json copied to the staged path
// still loads.
//
// Legacy (read-only): pretty JSON SaveData. Existing saves keep working;
// the first save after this change writes V2 at the new default path.
// ---------------------------------------------------------------------------

/// Magic prefix identifying a V2 binary save.
const SAVE_MAGIC: &[u8; 8] = b"MWSAVE02";

/// Legacy JSON schema (read-only — kept for existing save files).
#[derive(Serialize, Deserialize)]
struct SaveData {
    player_position: [f32; 3],
    player_yaw: f32,
    player_pitch: f32,
    home_position: Option<[f32; 3]>,
    modifications: Vec<BlockModification>,
    /// Path to the screenshot captured at save time.
    /// Used as the title menu background. Older saves without this field
    /// deserialize as None — the menu falls back to the default background.
    #[serde(default)]
    last_menu_background_image_path: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct BlockModification {
    x: i32,
    y: i32,
    z: i32,
    block_type: u8,
}

/// V2 player snapshot. Optional in SaveDataV2 so screenshot-path-only
/// metadata writes (no live player) don't fabricate a position.
#[derive(Serialize, Deserialize, Clone, Copy)]
struct PlayerStateV2 {
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    home_position: Option<[f32; 3]>,
}

/// One chunk's modifications: packed local index (x + y·16 + z·256,
/// 0..4095) + block type.
#[derive(Serialize, Deserialize)]
struct ChunkModsV2 {
    pos: (i32, i32, i32),
    mods: Vec<(u16, u8)>,
}

#[derive(Serialize, Deserialize)]
struct SaveDataV2 {
    player: Option<PlayerStateV2>,
    chunks: Vec<ChunkModsV2>,
    last_menu_background_image_path: Option<String>,
}

/// Format-independent runtime view of a save. Every reader in the game
/// (player restore, world-mod apply, menu background) consumes this — the
/// on-disk format is a private concern of this module.
pub struct LoadedSave {
    pub player: Option<PlayerStateV2Public>,
    /// Flat (world_pos, raw_block_type) list, order unspecified.
    pub modifications: Vec<(IVec3, u8)>,
    pub background_path: Option<String>,
}

/// Public mirror of PlayerStateV2 (keeps the serde type private).
#[derive(Clone, Copy)]
pub struct PlayerStateV2Public {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub home_position: Option<Vec3>,
}

fn pack_local(p: IVec3) -> (i32, i32, i32, u16) {
    let cs = crate::chunk::CHUNK_SIZE;
    let cx = p.x.div_euclid(cs);
    let cy = p.y.div_euclid(cs);
    let cz = p.z.div_euclid(cs);
    let lx = p.x.rem_euclid(cs);
    let ly = p.y.rem_euclid(cs);
    let lz = p.z.rem_euclid(cs);
    (cx, cy, cz, (lx + ly * cs + lz * cs * cs) as u16)
}

fn unpack_local(chunk: (i32, i32, i32), idx: u16) -> IVec3 {
    let cs = crate::chunk::CHUNK_SIZE;
    let i = idx as i32;
    IVec3::new(
        chunk.0 * cs + i % cs,
        chunk.1 * cs + (i / cs) % cs,
        chunk.2 * cs + i / (cs * cs),
    )
}

impl SaveDataV2 {
    fn from_parts(
        player: Option<PlayerStateV2>,
        modifications: impl Iterator<Item = (IVec3, u8)>,
        background_path: Option<String>,
    ) -> Self {
        use std::collections::HashMap;
        let mut by_chunk: HashMap<(i32, i32, i32), Vec<(u16, u8)>> = HashMap::new();
        for (pos, bt) in modifications {
            let (cx, cy, cz, idx) = pack_local(pos);
            by_chunk.entry((cx, cy, cz)).or_default().push((idx, bt));
        }
        let mut chunks: Vec<ChunkModsV2> = by_chunk
            .into_iter()
            .map(|(pos, mods)| ChunkModsV2 { pos, mods })
            .collect();
        // Deterministic output (HashMap order varies run to run).
        chunks.sort_by_key(|c| c.pos);
        SaveDataV2 {
            player,
            chunks,
            last_menu_background_image_path: background_path,
        }
    }

    fn into_loaded(self) -> LoadedSave {
        let mut modifications = Vec::new();
        for chunk in &self.chunks {
            for &(idx, bt) in &chunk.mods {
                // Packed index is 0..4095 by construction; a corrupt value
                // would still unpack in-range for x/y and clamp z upward —
                // reject anything out of range instead.
                if idx as usize >= crate::chunk::CHUNK_VOLUME {
                    continue;
                }
                modifications.push((unpack_local(chunk.pos, idx), bt));
            }
        }
        LoadedSave {
            player: self.player.map(|p| PlayerStateV2Public {
                position: Vec3::from_array(p.position),
                yaw: p.yaw,
                pitch: p.pitch,
                home_position: p.home_position.map(Vec3::from_array),
            }),
            modifications,
            background_path: self.last_menu_background_image_path,
        }
    }
}

/// Read and decode a save file at `path`, sniffing the format by content.
/// Returns None (with a warn) on read/parse failure.
pub fn read_save(path: &std::path::Path) -> Option<LoadedSave> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return None,
    };
    if bytes.len() >= SAVE_MAGIC.len() && &bytes[..SAVE_MAGIC.len()] == SAVE_MAGIC {
        match bincode::deserialize::<SaveDataV2>(&bytes[SAVE_MAGIC.len()..]) {
            Ok(v2) => Some(v2.into_loaded()),
            Err(e) => {
                warn!("Failed to decode binary save {}: {}", path.display(), e);
                None
            }
        }
    } else {
        // Legacy JSON. Partial-tolerant: a metadata-only file (just the
        // screenshot path) yields player=None + no mods.
        let text = String::from_utf8_lossy(&bytes);
        match serde_json::from_str::<SaveData>(&text) {
            Ok(legacy) => Some(LoadedSave {
                player: Some(PlayerStateV2Public {
                    position: Vec3::from_array(legacy.player_position),
                    yaw: legacy.player_yaw,
                    pitch: legacy.player_pitch,
                    home_position: legacy.home_position.map(Vec3::from_array),
                }),
                modifications: legacy
                    .modifications
                    .iter()
                    .map(|m| (IVec3::new(m.x, m.y, m.z), m.block_type))
                    .collect(),
                background_path: legacy.last_menu_background_image_path,
            }),
            Err(_) => {
                // Metadata-only or partial legacy file.
                #[derive(Deserialize)]
                struct PartialLegacy {
                    #[serde(default)]
                    last_menu_background_image_path: Option<String>,
                }
                match serde_json::from_str::<PartialLegacy>(&text) {
                    Ok(p) => Some(LoadedSave {
                        player: None,
                        modifications: Vec::new(),
                        background_path: p.last_menu_background_image_path,
                    }),
                    Err(e) => {
                        warn!("Failed to parse save file {}: {}", path.display(), e);
                        None
                    }
                }
            }
        }
    }
}

/// Read the default save, preferring the V2 path and falling back to the
/// legacy JSON path (pre-migration installs).
pub fn read_default_save() -> Option<LoadedSave> {
    read_save(&save_path()).or_else(|| read_save(&legacy_save_path()))
}

/// Encode + write a V2 save. Returns false (with a warn) on failure.
fn write_save_v2(path: &std::path::Path, save: &SaveDataV2) -> bool {
    match bincode::serialize(save) {
        Ok(mut payload) => {
            let mut bytes = SAVE_MAGIC.to_vec();
            bytes.append(&mut payload);
            match std::fs::write(path, bytes) {
                Ok(()) => true,
                Err(e) => {
                    warn!("Failed to write save {}: {}", path.display(), e);
                    false
                }
            }
        }
        Err(e) => {
            warn!("Failed to encode save: {}", e);
            false
        }
    }
}

/// Controls whether saved data is loaded on Gameplay entry.
#[derive(Resource, Clone, Copy, PartialEq, Eq, Debug)]
pub enum StartMode {
    /// Load saved chunk modifications and player position.
    Continue,
    /// Ignore saves — generate a clean world. Saves are NOT deleted;
    /// they are simply not applied for this session.
    NewGame,
}

impl Default for StartMode {
    fn default() -> Self {
        Self::Continue
    }
}

/// Tracks whether the auto-load has been performed.
/// Reset on entering Menu so auto-load runs again on next Gameplay entry.
#[derive(Resource)]
pub struct AutoLoadState {
    pub loaded: bool,
}

/// Resource flag requesting a native load-file dialog.
///
/// ARCHITECTURAL INVARIANT: this is the ONLY public API for user-initiated
/// loads. UI sets the flag; a dedicated system (`open_file_dialog`) consumes
/// it and opens the OS dialog. The resource is not tied to any menu entity's
/// lifetime — it survives menu transitions and works from both the title
/// menu (Menu state) and the in-game overlay (Gameplay state with menu open).
#[derive(Resource, Default)]
pub struct OpenLoadDialog {
    pub pending: bool,
}

/// Resource flag requesting a native save-file dialog.
/// Same decoupling as OpenLoadDialog — set by UI, consumed by a dedicated
/// system that outlives the menu entity.
#[derive(Resource, Default)]
pub struct OpenSaveDialog {
    pub pending: bool,
}

/// Marker resource that exists while any native file dialog is open.
/// Gameplay systems gate on `not(resource_exists::<FileDialogOpen>)` so
/// they are fully paused while the user is interacting with the OS dialog.
/// Inserted by open_file_dialog / open_save_dialog, removed by poll_* when
/// the dialog completes (either with a selection or cancellation).
#[derive(Resource)]
pub struct FileDialogOpen;

/// Holds the async load-file dialog task while it's running.
#[derive(Resource)]
struct FileDialogTask {
    task: Task<Option<PathBuf>>,
}

/// Holds the async save-file dialog task and the encoded data to write.
#[derive(Resource)]
struct SaveDialogTask {
    task: Task<Option<PathBuf>>,
    /// V2 save bytes, encoded before the dialog opens so we capture the
    /// game state at the moment the user clicked Save.
    bytes: Vec<u8>,
}

pub struct SaveLoadPlugin;

impl Plugin for SaveLoadPlugin {
    fn build(&self, app: &mut App) {
        // DEPENDENCY: requires PlayerPlugin and ChunkManagerPlugin to be
        // registered first (see main.rs ordering comment). auto_load_game
        // guards on the player entity existing before applying save data.
        app.insert_resource(AutoLoadState { loaded: false })
            .init_resource::<StartMode>()
            .init_resource::<OpenLoadDialog>()
            .init_resource::<OpenSaveDialog>()
            // auto_load_game runs always (internal, not user-initiated).
            // save_game_on_key (F5) is gated to Gameplay only (internal shortcut).
            //
            // ARCHITECTURAL INVARIANT: all user-initiated save/load actions
            // from menus go through file dialogs. UI sets OpenLoadDialog /
            // OpenSaveDialog flags; the open_* systems consume them and
            // launch async OS dialogs. These systems are NOT state-gated —
            // they must work from both the title menu (Menu state) and the
            // in-game overlay (Gameplay state with menu open).
            .add_systems(Update, auto_load_game
                .run_if(in_state(GameState::Gameplay))
                .run_if(menu_overlay_closed))
            .add_systems(Update, (open_file_dialog, open_save_dialog))
            .add_systems(Update, (poll_file_dialog, poll_save_dialog))
            .add_systems(
                Update,
                save_game_on_key
                    .run_if(in_state(GameState::Gameplay))
                    .run_if(not(resource_exists::<FileDialogOpen>))
                    .run_if(menu_overlay_closed),
            );
    }
}

/// Automatically load saved game state on the first frame where the player exists.
/// Skipped entirely when StartMode::NewGame — the world generates fresh.
fn auto_load_game(
    mut state: ResMut<AutoLoadState>,
    mut chunk_manager: ResMut<ChunkManager>,
    player_query: Query<(), With<Player>>,
    start_mode: Res<StartMode>,
) {
    if state.loaded {
        return;
    }

    // Wait until the player entity is spawned (spawn_player restores position).
    if player_query.iter().next().is_none() {
        return;
    }

    state.loaded = true;

    if *start_mode == StartMode::NewGame {
        info!("New Game — skipping saved chunk modifications");
        return;
    }

    apply_save_data(&mut chunk_manager);
}

/// Load game state from the default save and apply it.
fn apply_save_data(
    chunk_manager: &mut ResMut<ChunkManager>,
) {
    let Some(save) = read_default_save() else {
        info!("No save file found, starting fresh.");
        return;
    };

    // --- Pass 1: bucket modifications by chunk, identify 100%-AIR chunks ---
    // A chunk whose saved modifications are ALL AIR and contain no solid
    // blocks is likely corruption (e.g., a mass-despawn wrote AIR over
    // generated terrain). These are skipped to prevent terrain destruction.
    use std::collections::HashMap;
    let cs = crate::chunk::CHUNK_SIZE;

    struct ChunkBucket { air: u32, solid: u32 }
    let mut buckets: HashMap<(i32, i32, i32), ChunkBucket> = HashMap::new();

    for &(pos, block_type) in &save.modifications {
        let key = (pos.x.div_euclid(cs), pos.y.div_euclid(cs), pos.z.div_euclid(cs));
        let bucket = buckets.entry(key).or_insert(ChunkBucket { air: 0, solid: 0 });
        if BlockType::from_u8(block_type) == BlockType::AIR {
            bucket.air += 1;
        } else {
            bucket.solid += 1;
        }
    }

    // Collect the set of corrupt (100%-AIR) chunk keys.
    let corrupt_chunks: std::collections::HashSet<(i32, i32, i32)> = buckets.iter()
        .filter(|(_, b)| b.solid == 0 && b.air > 0)
        .map(|(&k, _)| k)
        .collect();

    // --- Pass 2: apply modifications, skipping corrupt chunks ---
    let mut applied = 0u32;
    let mut skipped = 0u32;

    for &(pos, block_type) in &save.modifications {
        let key = (pos.x.div_euclid(cs), pos.y.div_euclid(cs), pos.z.div_euclid(cs));
        if corrupt_chunks.contains(&key) {
            skipped += 1;
            continue;
        }
        chunk_manager.set_block(pos, BlockType::from_u8(block_type));
        applied += 1;
    }

    // Player position, yaw, pitch, and home_position are restored by
    // spawn_player (via load_player_from_save) at exact saved coordinates.
    // No +2 Y offset needed — the WorldReady system gates physics until
    // the player's chunk is loaded, preventing fall-through.

    // --- Deferred logging (after all mutations complete) ---
    if !corrupt_chunks.is_empty() {
        let mut sorted: Vec<_> = corrupt_chunks.iter().collect();
        sorted.sort();
        for &&(cx, cy, cz) in &sorted {
            let b = &buckets[&(cx, cy, cz)];
            warn!(
                "Save corruption guard: skipped chunk ({},{},{}) — {} AIR-only mods \
                 (would overwrite generated terrain)",
                cx, cy, cz, b.air,
            );
        }
        warn!(
            "Save corruption guard: {} chunks skipped ({} mods dropped), {} mods applied",
            corrupt_chunks.len(), skipped, applied,
        );
    }

    info!(
        "Loaded save: {} modifications ({} applied, {} skipped)",
        save.modifications.len(), applied, skipped,
    );

    #[cfg(debug_assertions)]
    {
        let mut sorted: Vec<_> = buckets.iter().collect();
        sorted.sort_by_key(|(&k, _)| k);
        for (&(cx, cy, cz), b) in &sorted {
            let total = b.air + b.solid;
            let status = if corrupt_chunks.contains(&(cx, cy, cz)) {
                " ⚠ SKIPPED (100% AIR)"
            } else {
                ""
            };
            bevy::log::info!(
                "Save chunk ({},{},{}): {} mods ({} solid, {} AIR){}",
                cx, cy, cz, total, b.solid, b.air, status,
            );
        }
    }
}

/// Write the quick-save (default path) from current in-memory state.
///
/// SINGLE SOURCE OF TRUTH for the save schema: main.rs auto_save_on_exit
/// and the ui.rs quit handlers call this instead of hand-rolling their own
/// serializer structs. (Historical bug this prevents: ui.rs's private copy
/// of the schema lacked last_menu_background_image_path, so quitting from
/// the menu silently dropped the screenshot path and the next launch lost
/// its menu background.)
///
/// Returns true if the file was written.
pub fn write_quick_save(
    cm: &ChunkManager,
    player_query: &Query<(&Transform, &Player)>,
) -> bool {
    let path = save_path();
    let Some(save) = collect_save_data(cm, player_query, &path) else {
        warn!("Quick-save skipped: no player entity.");
        return false;
    };
    if write_save_v2(&path, &save) {
        let mods: usize = save.chunks.iter().map(|c| c.mods.len()).sum();
        info!("Saved {} modifications to {}", mods, path.display());
        true
    } else {
        false
    }
}

/// Quick-save when F5 is pressed — writes to the default save path.
/// This is an internal gameplay shortcut, not a user-initiated menu action.
fn save_game_on_key(
    keys: Res<ButtonInput<KeyCode>>,
    chunk_manager: Option<Res<ChunkManager>>,
    player_query: Query<(&Transform, &Player)>,
    mut commands: Commands,
) {
    if !keys.just_pressed(KeyCode::F5) {
        return;
    }

    let Some(cm) = chunk_manager.as_deref() else {
        return;
    };
    if write_quick_save(cm, &player_query) {
        request_screenshot(&mut commands, &save_path());
        info!("Game saved!");
    }
}

// ---------------------------------------------------------------------------
// Native file dialog for "Load Saved Game"
// ---------------------------------------------------------------------------

/// Consumes the OpenLoadDialog flag and launches a native OS file dialog on
/// a background thread. Not tied to any menu entity — the resource survives
/// menu transitions. The result is polled by poll_file_dialog.
fn open_file_dialog(
    mut request: ResMut<OpenLoadDialog>,
    mut commands: Commands,
    existing: Option<Res<FileDialogTask>>,
) {
    if !request.pending {
        return;
    }
    request.pending = false;

    // Don't open a second dialog if one is already running
    if existing.is_some() {
        return;
    }

    let start_dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));

    let task = IoTaskPool::get().spawn(async move {
        let handle = rfd::AsyncFileDialog::new()
            .set_title(&format!("Load {} Save", crate::GAME_NAME))
            // "json" kept for pre-V2 saves — the reader sniffs by content.
            .add_filter(&format!("{} Save", crate::GAME_NAME), &["mwsave", "json"])
            .set_directory(&start_dir)
            .pick_file()
            .await;

        handle.map(|h| h.path().to_path_buf())
    });

    commands.insert_resource(FileDialogTask { task });
    commands.insert_resource(FileDialogOpen);
}

/// Polls a running file dialog task. When the user selects a file, stages
/// the save at the default path and schedules a LoadGame reload through the
/// SAME teardown/init machinery the Play button uses. If the user cancels,
/// nothing changes.
///
/// WHY the full teardown wiring: without setting TeardownIntent and bumping
/// WorldInstanceId, guard_clean_world_on_entry never emits TeardownIssued,
/// the fence never triggers WorldInitRequested, and the WorldSpawnSet
/// systems (player, sun, sky, animals) never run. From the title menu that
/// was a permanent hang on the loading screen; from the in-game overlay it
/// merged the loaded save's modifications into the live world instead of
/// replacing it.
fn poll_file_dialog(
    mut commands: Commands,
    mut dialog: Option<ResMut<FileDialogTask>>,
    mut menu_state: ResMut<crate::ui::MenuState>,
    menu_query: Query<Entity, With<crate::ui::SettingsMenu>>,
    mut cursor_query: Query<&mut bevy::window::CursorOptions, With<bevy::window::PrimaryWindow>>,
    state: Res<State<GameState>>,
    mut start_mode: ResMut<StartMode>,
    mut teardown_intent: ResMut<crate::TeardownIntent>,
    mut world_instance: ResMut<crate::WorldInstanceId>,
    mut pending_teardown: ResMut<crate::PendingTeardown>,
    mut pending_reload: ResMut<crate::PendingReload>,
) {
    let Some(ref mut dialog) = dialog else {
        return;
    };

    let Some(result) = check_ready(&mut dialog.task) else {
        return; // still waiting for user
    };

    // Task is done — remove task and dialog-open marker regardless of result
    commands.remove_resource::<FileDialogTask>();
    commands.remove_resource::<FileDialogOpen>();

    let Some(path) = result else {
        info!("File dialog cancelled.");
        return; // user cancelled
    };

    // Validate and stage the chosen file. On failure the current session
    // continues untouched — no teardown is scheduled.
    if !stage_save_file(&path) {
        return;
    }

    // Route through the standard teardown/rebuild machinery (mirrors the
    // Play button). auto_save_on_exit skips the LoadGame intent so it can't
    // overwrite the staged file; reset_chunk_state_on_teardown clears the
    // old world's modifications; spawn_player + auto_load_game then restore
    // everything from the staged default save.
    *start_mode = StartMode::Continue;
    let old_id = world_instance.0;
    world_instance.0 = old_id + 1;
    *pending_teardown = crate::PendingTeardown {
        old_id,
        kind: crate::TeardownIntent::LoadGame,
    };
    *teardown_intent = crate::TeardownIntent::LoadGame;

    // Close menu, grab cursor, and defer the transition (see PendingReload).
    menu_state.is_open = false;
    for entity in &menu_query {
        commands.entity(entity).despawn();
    }
    if let Ok(mut cursor) = cursor_query.single_mut() {
        cursor.grab_mode = bevy::window::CursorGrabMode::Locked;
        cursor.visible = false;
        menu_state.cursor_captured = true;
    }
    // Screenshot only when a live world exists (in-game overlay path).
    if *state.get() == GameState::Gameplay {
        crate::request_menu_screenshot(&mut commands);
    }
    *pending_reload = crate::PendingReload {
        active: true,
        frames: crate::RELOAD_DEFER_FRAMES,
    };
}

/// Validate a user-chosen save file and stage it at the default save path,
/// where spawn_player (load_player_from_save) and auto_load_game read it
/// during world init. Returns false if the file can't be read, parsed, or
/// copied — the caller must then abort the load.
///
/// Modifications are NOT applied here: auto_load_game applies them from the
/// staged file after the world rebuild, which also runs them through the
/// save-corruption guard in apply_save_data.
fn stage_save_file(path: &std::path::Path) -> bool {
    // Format sniffed by content — a legacy .json picked in the dialog
    // stages fine and is auto-detected when read back.
    let Some(save) = read_save(path) else {
        warn!("Failed to read/parse save file {}", path.display());
        return false;
    };

    let default_path = save_path();
    if path != default_path {
        if let Err(e) = std::fs::copy(path, &default_path) {
            warn!(
                "Failed to stage save file {} → {}: {}",
                path.display(),
                default_path.display(),
                e,
            );
            return false;
        }
    }

    info!(
        "Staged save from {}: {} modifications",
        path.display(),
        save.modifications.len()
    );
    true
}

/// Build a V2 save snapshot from the current game state.
fn collect_save_data(
    cm: &ChunkManager,
    player_query: &Query<(&Transform, &Player)>,
    save_file: &std::path::Path,
) -> Option<SaveDataV2> {
    let (transform, player) = player_query.iter().next()?;
    let screenshot = screenshot_path_for(save_file);
    Some(SaveDataV2::from_parts(
        Some(PlayerStateV2 {
            position: transform.translation.to_array(),
            yaw: player.yaw,
            pitch: player.pitch,
            home_position: player.home_position.map(|p| p.to_array()),
        }),
        cm.modifications.iter().map(|(pos, &block)| (*pos, block.index())),
        Some(screenshot.to_string_lossy().into_owned()),
    ))
}

/// Generate a default save file name with a timestamp.
fn default_save_name() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("sandbox_world_{}.mwsave", secs)
}

// ---------------------------------------------------------------------------
// Native save-file dialog
// ---------------------------------------------------------------------------

/// Consumes the OpenSaveDialog flag, serializes the current game state, and
/// launches a native save-file dialog on a background thread. Not tied to
/// any menu entity — the resource survives menu transitions.
fn open_save_dialog(
    mut request: ResMut<OpenSaveDialog>,
    mut commands: Commands,
    existing: Option<Res<SaveDialogTask>>,
    chunk_manager: Option<Res<ChunkManager>>,
    player_query: Query<(&Transform, &Player)>,
) {
    if !request.pending {
        return;
    }
    request.pending = false;

    if existing.is_some() {
        return;
    }

    let Some(cm) = chunk_manager.as_deref() else {
        return;
    };
    // Path isn't known yet (user picks via dialog), so use a placeholder.
    // The actual screenshot path is set when poll_save_dialog writes the file.
    let placeholder = save_path();
    let Some(save) = collect_save_data(cm, &player_query, &placeholder) else {
        warn!("Cannot save: no player entity.");
        return;
    };
    let bytes = match bincode::serialize(&save) {
        Ok(payload) => {
            let mut b = SAVE_MAGIC.to_vec();
            b.extend(payload);
            b
        }
        Err(e) => {
            warn!("Failed to serialize save data: {}", e);
            return;
        }
    };

    let start_dir = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
    let file_name = default_save_name();

    let task = IoTaskPool::get().spawn(async move {
        let handle = rfd::AsyncFileDialog::new()
            .set_title(&format!("Save {} Game", crate::GAME_NAME))
            .add_filter(&format!("{} Save", crate::GAME_NAME), &["mwsave"])
            .set_directory(&start_dir)
            .set_file_name(&file_name)
            .save_file()
            .await;

        handle.map(|h| h.path().to_path_buf())
    });

    commands.insert_resource(SaveDialogTask { task, bytes });
    commands.insert_resource(FileDialogOpen);
}

/// Polls a running save-file dialog. When the user confirms a path, writes
/// the pre-serialized JSON to that file. If the user cancels, cleans up
/// with no side effects. rfd handles OS-level overwrite confirmation.
fn poll_save_dialog(
    mut commands: Commands,
    mut dialog: Option<ResMut<SaveDialogTask>>,
) {
    let Some(ref mut dialog) = dialog else {
        return;
    };

    let Some(result) = check_ready(&mut dialog.task) else {
        return;
    };

    // Grab the bytes before removing task and dialog-open marker
    let bytes = dialog.bytes.clone();
    commands.remove_resource::<SaveDialogTask>();
    commands.remove_resource::<FileDialogOpen>();

    let Some(mut path) = result else {
        info!("Save dialog cancelled.");
        return;
    };

    // Enforce .mwsave extension if the user didn't type one
    if path.extension().is_none() {
        path.set_extension("mwsave");
    }

    match std::fs::write(&path, &bytes) {
        Ok(()) => {
            request_screenshot(&mut commands, &path);
            info!("Game saved to: {}", path.display());
        }
        Err(e) => warn!("Failed to write save file {}: {}", path.display(), e),
    }
}

/// Default quick-save path (V2 binary).
pub fn save_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".metalworld_save.mwsave")
}

/// Pre-V2 default path — read-only fallback so existing installs keep
/// their world on first launch after the format change.
fn legacy_save_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".metalworld_save.json")
}

/// Derive the screenshot path from a save file path.
/// e.g. "foo.mwsave" → "foo.png"
fn screenshot_path_for(save: &std::path::Path) -> PathBuf {
    save.with_extension("png")
}

/// Persist the menu-background screenshot path into the default save,
/// preserving everything else (read-modify-write across either format;
/// always writes V2). Creates a metadata-only save if none exists yet, so
/// the menu can find the screenshot even before the first real save.
pub fn persist_screenshot_path(screenshot: &std::path::Path) {
    let existing = read_default_save();
    let bg = Some(screenshot.to_string_lossy().into_owned());
    let v2 = match existing {
        Some(loaded) => SaveDataV2::from_parts(
            loaded.player.map(|p| PlayerStateV2 {
                position: p.position.to_array(),
                yaw: p.yaw,
                pitch: p.pitch,
                home_position: p.home_position.map(|h| h.to_array()),
            }),
            loaded.modifications.into_iter(),
            bg,
        ),
        None => SaveDataV2 {
            player: None,
            chunks: Vec::new(),
            last_menu_background_image_path: bg,
        },
    };
    if write_save_v2(&save_path(), &v2) {
        #[cfg(debug_assertions)]
        bevy::log::info!("Screenshot path persisted to {}", save_path().display());
    }
}

use bevy::render::view::screenshot::{save_to_disk, Screenshot};

/// Request a screenshot capture for the current frame. The screenshot is
/// captured asynchronously by Bevy's renderer — no blocking IO during gameplay.
/// Uses Bevy's built-in `save_to_disk` observer which handles async write.
fn request_screenshot(commands: &mut Commands, save_file: &std::path::Path) {
    let path = screenshot_path_for(save_file);
    #[cfg(debug_assertions)]
    bevy::log::info!(
        "Screenshot capture requested → {}",
        path.display(),
    );
    commands.spawn(Screenshot::primary_window())
        .observe(save_to_disk(path));
}


#[cfg(test)]
mod tests {
    use super::*;

    /// pack/unpack must be inverse for negative and positive coords.
    #[test]
    fn local_packing_roundtrips() {
        for &p in &[
            IVec3::new(0, 0, 0),
            IVec3::new(15, 15, 15),
            IVec3::new(-1, -1, -1),
            IVec3::new(-16, 31, -47),
            IVec3::new(123, -456, 789),
        ] {
            let (cx, cy, cz, idx) = pack_local(p);
            assert!((idx as usize) < crate::chunk::CHUNK_VOLUME);
            assert_eq!(unpack_local((cx, cy, cz), idx), p, "roundtrip failed for {p}");
        }
    }

    /// V2 write → read must reproduce player state and every modification.
    #[test]
    fn v2_save_roundtrips() {
        let dir = std::env::temp_dir().join("metalworld_test_saves");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("roundtrip.mwsave");

        let mods = vec![
            (IVec3::new(5, 12, -3), 3u8),
            (IVec3::new(-20, 4, 100), 8u8),
            (IVec3::new(0, 0, 0), 1u8),
        ];
        let v2 = SaveDataV2::from_parts(
            Some(PlayerStateV2 {
                position: [1.5, 20.0, -3.25],
                yaw: 0.7,
                pitch: -0.2,
                home_position: Some([4.0, 8.0, 15.0]),
            }),
            mods.iter().copied(),
            Some("shot.png".to_string()),
        );
        assert!(write_save_v2(&path, &v2));

        let loaded = read_save(&path).expect("read_save failed");
        let player = loaded.player.expect("player missing");
        assert_eq!(player.position, Vec3::new(1.5, 20.0, -3.25));
        assert_eq!(player.yaw, 0.7);
        assert_eq!(player.home_position, Some(Vec3::new(4.0, 8.0, 15.0)));
        assert_eq!(loaded.background_path.as_deref(), Some("shot.png"));

        let mut got = loaded.modifications.clone();
        got.sort_by_key(|(p, _)| (p.x, p.y, p.z));
        let mut want = mods.clone();
        want.sort_by_key(|(p, _)| (p.x, p.y, p.z));
        assert_eq!(got, want);

        let _ = std::fs::remove_file(&path);
    }

    /// Legacy JSON saves (full and metadata-only) must still load — by
    /// content sniff, regardless of extension.
    #[test]
    fn legacy_json_still_loads() {
        let dir = std::env::temp_dir().join("metalworld_test_saves");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("legacy.mwsave"); // wrong extension on purpose

        let json = r#"{
            "player_position": [10.0, 21.0, -5.0],
            "player_yaw": 1.0,
            "player_pitch": 0.1,
            "home_position": null,
            "modifications": [
                {"x": 1, "y": 2, "z": 3, "block_type": 5},
                {"x": -4, "y": 0, "z": 9, "block_type": 0}
            ],
            "last_menu_background_image_path": "bg.png"
        }"#;
        std::fs::write(&path, json).unwrap();

        let loaded = read_save(&path).expect("legacy read failed");
        assert_eq!(loaded.player.unwrap().position, Vec3::new(10.0, 21.0, -5.0));
        assert_eq!(loaded.modifications.len(), 2);
        assert!(loaded
            .modifications
            .contains(&(IVec3::new(1, 2, 3), 5)));
        assert_eq!(loaded.background_path.as_deref(), Some("bg.png"));

        // Metadata-only legacy file (screenshot path only).
        std::fs::write(&path, r#"{"last_menu_background_image_path": "only.png"}"#).unwrap();
        let meta = read_save(&path).expect("metadata-only read failed");
        assert!(meta.player.is_none());
        assert!(meta.modifications.is_empty());
        assert_eq!(meta.background_path.as_deref(), Some("only.png"));

        let _ = std::fs::remove_file(&path);
    }
}
