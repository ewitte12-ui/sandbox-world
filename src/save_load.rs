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

/// Holds the async save-file dialog task and the serialized data to write.
#[derive(Resource)]
struct SaveDialogTask {
    task: Task<Option<PathBuf>>,
    /// JSON save data, serialized before the dialog opens so we capture
    /// the game state at the moment the user clicked Save.
    json: String,
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

/// Load game state from the save file and apply it.
fn apply_save_data(
    chunk_manager: &mut ResMut<ChunkManager>,
) {
    let path = save_path();
    let data = match std::fs::read_to_string(&path) {
        Ok(d) => d,
        Err(_) => {
            info!("No save file found, starting fresh.");
            return;
        }
    };
    let save: SaveData = match serde_json::from_str(&data) {
        Ok(s) => s,
        Err(e) => {
            warn!("Failed to parse save file: {}", e);
            return;
        }
    };

    // --- Pass 1: bucket modifications by chunk, identify 100%-AIR chunks ---
    // A chunk whose saved modifications are ALL AIR and contain no solid
    // blocks is likely corruption (e.g., a mass-despawn wrote AIR over
    // generated terrain). These are skipped to prevent terrain destruction.
    use std::collections::HashMap;
    let cs = crate::chunk::CHUNK_SIZE;

    struct ChunkBucket { air: u32, solid: u32 }
    let mut buckets: HashMap<(i32, i32, i32), ChunkBucket> = HashMap::new();

    for m in &save.modifications {
        let key = (m.x.div_euclid(cs), m.y.div_euclid(cs), m.z.div_euclid(cs));
        let bucket = buckets.entry(key).or_insert(ChunkBucket { air: 0, solid: 0 });
        if BlockType::from_u8(m.block_type) == BlockType::AIR {
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

    for m in &save.modifications {
        let key = (m.x.div_euclid(cs), m.y.div_euclid(cs), m.z.div_euclid(cs));
        if corrupt_chunks.contains(&key) {
            skipped += 1;
            continue;
        }
        let block = BlockType::from_u8(m.block_type);
        chunk_manager.set_block(IVec3::new(m.x, m.y, m.z), block);
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
    match serde_json::to_string_pretty(&save) {
        Ok(data) => match std::fs::write(&path, data) {
            Ok(()) => {
                info!("Saved {} modifications to {}", save.modifications.len(), path.display());
                true
            }
            Err(e) => {
                warn!("Quick-save write failed ({}): {}", path.display(), e);
                false
            }
        },
        Err(e) => {
            warn!("Quick-save serialization failed: {}", e);
            false
        }
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
            .add_filter(&format!("{} Save", crate::GAME_NAME), &["json"])
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
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            warn!("Failed to read save file {}: {}", path.display(), e);
            return false;
        }
    };
    let save: SaveData = match serde_json::from_str(&data) {
        Ok(s) => s,
        Err(e) => {
            warn!("Failed to parse save file {}: {}", path.display(), e);
            return false;
        }
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

/// Build a SaveData snapshot from the current game state.
fn collect_save_data(
    cm: &ChunkManager,
    player_query: &Query<(&Transform, &Player)>,
    save_file: &std::path::Path,
) -> Option<SaveData> {
    let (transform, player) = player_query.iter().next()?;
    let modifications = cm
        .modifications
        .iter()
        .map(|(pos, &block)| BlockModification {
            x: pos.x,
            y: pos.y,
            z: pos.z,
            block_type: block.index(),
        })
        .collect();
    let screenshot = screenshot_path_for(save_file);
    Some(SaveData {
        player_position: transform.translation.to_array(),
        player_yaw: player.yaw,
        player_pitch: player.pitch,
        home_position: player.home_position.map(|p| p.to_array()),
        modifications,
        last_menu_background_image_path: Some(screenshot.to_string_lossy().into_owned()),
    })
}

/// Generate a default save file name with a timestamp.
fn default_save_name() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("sandbox_world_{}.json", secs)
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
    let json = match serde_json::to_string_pretty(&save) {
        Ok(j) => j,
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
            .add_filter(&format!("{} Save", crate::GAME_NAME), &["json"])
            .set_directory(&start_dir)
            .set_file_name(&file_name)
            .save_file()
            .await;

        handle.map(|h| h.path().to_path_buf())
    });

    commands.insert_resource(SaveDialogTask { task, json });
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

    // Grab the json before removing task and dialog-open marker
    let json = dialog.json.clone();
    commands.remove_resource::<SaveDialogTask>();
    commands.remove_resource::<FileDialogOpen>();

    let Some(mut path) = result else {
        info!("Save dialog cancelled.");
        return;
    };

    // Enforce .json extension if the user didn't type one
    if path.extension().is_none() {
        path.set_extension("json");
    }

    match std::fs::write(&path, &json) {
        Ok(()) => {
            request_screenshot(&mut commands, &path);
            info!("Game saved to: {}", path.display());
        }
        Err(e) => warn!("Failed to write save file {}: {}", path.display(), e),
    }
}

fn save_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".metalworld_save.json")
}

/// Derive the screenshot path from a save file path.
/// e.g. "foo.json" → "foo.png"
fn screenshot_path_for(save: &std::path::Path) -> PathBuf {
    save.with_extension("png")
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

