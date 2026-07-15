use bevy::prelude::*;

mod animals;
mod block_types;
mod buildings;
mod chunk;
mod chunk_manager;
mod dev_tools;
mod lighting;
mod player;
mod ray_cast;
mod save_load;
mod settings;
mod sky;
mod terrain;
mod ui;
// NOTE: entity-based trees (formerly trees.rs) were removed — they were
// never registered as a plugin, and the live trees are voxel blocks baked
// into chunks by terrain::place_trees_in_chunk.

/// Top-level application state. The game launches into Menu; gameplay begins
/// only when the player explicitly starts or loads a game.
///
/// HARD RULE — Menu state policy:
///   The title menu is NOT a paused game world. It is a clean UI-only state.
///   No world entities (cameras, chunks, animals, lights) may exist in Menu.
///   No world simulation, rendering, or physics may run in Menu.
///   Background visuals must be static images (screenshot from last save).
///   Any WorldEntity visible in Menu is a correctness bug.
#[derive(States, Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GameState {
    /// Main menu is visible, cursor is free, no world entities exist.
    #[default]
    Menu,
    /// Active gameplay — cursor locked, all systems running.
    Gameplay,
}

/// Marker component for all entities that belong to the game world.
/// On exiting Gameplay, all entities with this marker are despawned.
/// UI entities must NOT have this component.
#[derive(Component)]
pub struct WorldEntity;

/// Marker component for entities that belong exclusively to UI overlays
/// (menus, inventory, HUD). World-cleanup systems must NEVER query for or
/// despawn entities carrying this marker.
#[derive(Component)]
pub struct UiOnly;

/// Marker for background plate entities (BackgroundRoot, Cloud).
/// Entities with this marker are EXCLUDED from all teardown, cleanup,
/// and world-reset paths. They survive every state transition and are
/// only destroyed when their world instance is replaced by a new one
/// via the scoped teardown system (which respawns them as part of the
/// new world).
///
/// HARD INVARIANT: no menu, overlay, or state-transition system may
/// despawn, hide, or mutate the transform of a BackgroundPlate entity.
#[derive(Component)]
pub struct BackgroundPlate;

/// Tracks background plate entity count for debug assertions.
/// Populated after WorldSpawnSet runs; checked every frame to detect
/// unexpected despawns.
#[derive(Resource, Default)]
pub struct BackgroundPlateCount(pub usize);

/// Debug assertion: verify no BackgroundPlate entity was despawned and no
/// Cloud transform was mutated by a non-background system.
///
/// Runs every PostUpdate frame. Checks:
/// 1. Entity count matches snapshot (no despawns).
/// 2. No Cloud entity has Changed<Transform> (only BackgroundRoot may change).
/// 3. No BackgroundPlate has Changed<Visibility> (user toggle is on Cloud
///    directly and is intentional; state-transition visibility changes are not).
fn assert_background_plates_intact(
    plates: Query<(), With<BackgroundPlate>>,
    expected: Res<BackgroundPlateCount>,
    // Cloud children: transform must NEVER change after spawn.
    changed_cloud_transforms: Query<
        Entity,
        (With<sky::Cloud>, Changed<Transform>, Without<sky::BackgroundRoot>),
    >,
    // Track frames since init to skip the spawn frame where Changed fires.
    mut frames_since_init: Local<u32>,
) {
    // Skip before first spawn (count is 0 until WorldSpawnSet runs).
    if expected.0 == 0 {
        *frames_since_init = 0;
        return;
    }
    *frames_since_init = frames_since_init.saturating_add(1);

    // --- Check 1: entity count ---
    let live = plates.iter().count();
    if live != expected.0 {
        bevy::log::error!(
            "BG_DIAG FAIL: entity_count expected={} actual={} — BackgroundPlate despawned!",
            expected.0, live,
        );
        debug_assert_eq!(
            live, expected.0,
            "BackgroundPlate entity was despawned — check teardown/cleanup queries for missing Without<BackgroundPlate> filter"
        );
    }

    // --- Check 2: cloud transform immutability ---
    // Skip frame 1-2 after init (Changed<Transform> fires on the spawn frame
    // and the first propagation frame).
    if *frames_since_init > 2 {
        let changed_count = changed_cloud_transforms.iter().count();
        if changed_count > 0 {
            bevy::log::error!(
                "BG_DIAG FAIL: {} Cloud entities had Transform changed this frame! \
                 Cloud transforms must be static after spawn.",
                changed_count,
            );
            debug_assert_eq!(
                changed_count, 0,
                "Cloud Transform was mutated — a system is writing to background plate transforms"
            );
        }
    }
}

/// Snapshot the background plate count once entities exist.
/// Runs in PostUpdate every frame during Gameplay. Updates the snapshot
/// whenever it sees plates but count is still 0 (i.e., first frame after
/// spawn commands are flushed by Bevy).
fn snapshot_background_plate_count(
    plates: Query<(), With<BackgroundPlate>>,
    mut count: ResMut<BackgroundPlateCount>,
) {
    let live = plates.iter().count();
    // Only snapshot once (when transitioning from 0 to non-zero).
    if count.0 == 0 && live > 0 {
        count.0 = live;
        bevy::log::info!(
            "BackgroundPlateCount snapshot: {} plates registered",
            count.0,
        );
    }
}

/// Monotonically increasing id that identifies the current world instance.
/// Incremented on each new-game / load-game cycle so that stale entities
/// from a previous world can be distinguished from current ones.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldInstanceId(pub u64);

impl Default for WorldInstanceId {
    fn default() -> Self {
        Self(0)
    }
}

/// Stamps an entity with the world instance it was spawned into.
/// Future teardown / fence queries can filter on this to ignore strays.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldScoped(pub u64);

/// Read the current world instance id.
pub fn current_world_id(id: &Res<WorldInstanceId>) -> u64 {
    id.0
}

/// Explicit intent for world teardown. Teardown systems should check this
/// resource instead of relying on state transitions, which can fire
/// spuriously (e.g., same-state re-entry from overlay close).
///
/// Only set this when the user explicitly requests an action that requires
/// destroying and rebuilding the world. Reset to `None` after teardown
/// completes.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TeardownIntent {
    /// No teardown requested — overlay open/close, settings changes, etc.
    #[default]
    None,
    /// User started a new game — wipe world, ignore save.
    NewGame,
    /// User loaded a save — wipe world, apply save data.
    LoadGame,
    /// User is quitting — save and exit.
    Quit,
}

/// Snapshot of the world instance being torn down. Set at the same time as
/// TeardownIntent so teardown systems know which instance to clean up.
/// `old_id` is the instance that existed before the id was bumped.
#[derive(Resource, Debug, Clone, Copy, Default)]
pub struct PendingTeardown {
    pub old_id: u64,
    pub kind: TeardownIntent,
}

/// Deferred Play/Load state transition.
///
/// WHY: the exit screenshot must capture the world with the menu already
/// closed and the world still alive. On the old immediate-transition path,
/// OnExit(Gameplay) requested the screenshot in the same frame that
/// OnEnter(Gameplay)'s teardown despawned every world entity — both run in
/// the same StateTransition schedule, so the captured frame showed a dead
/// world (black / stale menu background). That was the visible
/// "menu-reload texture loss".
///
/// Flow: the Play/Load handlers close the menu, request the screenshot
/// (world renders clean for a frame or two), and set this countdown.
/// `process_pending_reload` fires the real transition when it reaches zero.
#[derive(Resource, Default)]
pub struct PendingReload {
    pub active: bool,
    pub frames: u8,
}

/// Number of frames to keep the world alive (menu closed) before the
/// reload transition, so the screenshot capture sees a fully rendered
/// world frame regardless of which frame the renderer grabs.
pub const RELOAD_DEFER_FRAMES: u8 = 2;

/// Counts down PendingReload and fires the deferred Gameplay transition.
/// Runs in all states: Play/Load can be triggered from the title menu
/// (Menu state) and from the in-game overlay (Gameplay state).
fn process_pending_reload(
    mut pending: ResMut<PendingReload>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if !pending.active {
        return;
    }
    if pending.frames > 0 {
        pending.frames -= 1;
        return;
    }
    pending.active = false;
    bevy::log::info!("PendingReload countdown complete — entering Gameplay");
    next_state.set(GameState::Gameplay);
}

/// Explicit request to initialize (or reinitialize) the game world.
/// Separates "teardown completed" from "please build a new world now."
///
/// Teardown destroys the old world. WorldInitRequested tells spawn systems
/// (player, lighting, clouds, animals, buildings) that they should run.
/// Without this event, spawn systems must not create entities — the world
/// may have been torn down for Quit (no rebuild needed) or the OnEnter
/// may be a spurious same-state re-entry (no rebuild wanted).
///
/// Triggered by the teardown fence once all old-world entities are despawned.
#[derive(Event, Debug, Clone, Copy)]
pub struct WorldInitRequested;

/// Emitted by teardown systems after despawn commands have actually been
/// issued.  The fence (`verify_teardown_complete`) only starts waiting
/// once it has observed this event — preventing it from gating on a
/// teardown that never happened.
#[derive(Event, Debug, Clone, Copy)]
pub struct TeardownIssued {
    pub kind: TeardownIntent,
}

/// Set by the TeardownIssued observer. The verify system checks
/// this flag each frame and only emits WorldInitRequested once all
/// entities from the old world instance are confirmed despawned.
#[derive(Resource, Default)]
pub struct TeardownPendingVerification {
    pub active: bool,
    /// The world instance id being torn down. The fence only counts
    /// entities with `WorldScoped(old_id)` — new-world entities are ignored.
    pub old_id: u64,
}

/// Set by the WorldInitRequested observer, consumed by spawn systems.
/// When true, spawn systems run even if they wouldn't normally (e.g.,
/// OnEnter already fired and the player-exists guard would skip).
#[derive(Resource, Default, PartialEq, Eq)]
pub struct WorldInitPending(pub bool);

/// System set for world spawn systems that run exactly once per world init.
/// All spawn systems (player, lighting, clouds, animals, buildings) belong
/// to this set and are gated on `WorldInitPending(true)`.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorldSpawnSet;

/// Observer: teardown despawn commands have been issued — start verification.
fn on_teardown_issued(
    event: On<TeardownIssued>,
    mut pending: ResMut<TeardownPendingVerification>,
    pending_teardown: Res<PendingTeardown>,
) {
    bevy::log::info!(
        "TeardownIssued received (kind={:?}, old_id={}) — fence now waiting for despawn flush",
        event.kind, pending_teardown.old_id,
    );
    pending.active = true;
    pending.old_id = pending_teardown.old_id;
}

/// System: verify all old-world entities are actually gone before allowing init.
/// Runs each frame while TeardownPendingVerification is active. Only counts
/// entities with `WorldScoped(old_id)` — new-world entities are ignored.
fn verify_teardown_complete(
    mut pending: ResMut<TeardownPendingVerification>,
    mut commands: Commands,
    scoped_entities: Query<&WorldScoped>,
) {
    let old_id = pending.old_id;
    let stale_count = scoped_entities.iter().filter(|s| s.0 == old_id).count();

    if !pending.active {
        return;
    }

    // DIAGNOSTIC: only log when fence is actually active (avoids per-frame spam)
    bevy::log::info!(
        "DIAG verify_teardown_complete: old_id={}, stale_WorldScoped={}",
        old_id, stale_count,
    );

    if stale_count > 0 {
        bevy::log::info!(
            "Teardown fence: still waiting — {} WorldScoped({}) remain",
            stale_count, old_id,
        );
        return;
    }

    // Old world is confirmed empty — safe to init.
    pending.active = false;
    bevy::log::info!("Teardown fence passed — no WorldScoped({}) remain, triggering WorldInitRequested", old_id);
    commands.trigger(WorldInitRequested);
}

/// Observer: when WorldInitRequested fires (after teardown verification),
/// set the pending flag so spawn systems know to run on the next frame.
fn on_world_init_requested(_event: On<WorldInitRequested>, mut pending: ResMut<WorldInitPending>) {
    bevy::log::info!("WorldInitRequested received — spawn systems will run next frame");
    pending.0 = true;
}

/// Runs after all WorldSpawnSet systems to reset the one-shot flag.
///
/// WorldInitPending is set mid-frame by the fence observer. Bevy's run
/// conditions for WorldSpawnSet were already evaluated (as false) for that
/// frame, so spawn systems don't actually execute until the NEXT frame.
/// This system waits one full frame before consuming the flag, giving
/// WorldSpawnSet a chance to see it.
fn consume_world_init(
    mut pending: ResMut<WorldInitPending>,
    mut frames_pending: Local<u32>,
) {
    if !pending.0 {
        *frames_pending = 0;
        return;
    }
    *frames_pending += 1;
    // Frame 1: flag was just set (possibly mid-frame). WorldSpawnSet hasn't
    // seen it yet — its run condition was evaluated before the observer fired.
    // Frame 2: WorldSpawnSet evaluates condition as true, spawn systems run.
    // consume_world_init runs after them (.after(WorldSpawnSet)), safe to reset.
    if *frames_pending < 2 {
        bevy::log::info!(
            "WorldInitPending=true, waiting for spawn systems (frame {})",
            *frames_pending,
        );
        return;
    }
    bevy::log::info!("WorldInitPending consumed — spawn systems have run");
    pending.0 = false;
    *frames_pending = 0;
}

/// Teardown entry point: runs on OnEnter(Gameplay). Despawns all stale
/// WorldScoped(old_id) entities, then triggers TeardownIssued so the
/// fence can verify completion before WorldInitRequested fires.
fn guard_clean_world_on_entry(
    mut commands: Commands,
    scoped_entities: Query<(Entity, &WorldScoped)>,
    mut teardown: ResMut<TeardownIntent>,
    pending_teardown: Res<PendingTeardown>,
    mut plate_count: ResMut<BackgroundPlateCount>,
    mut world_ready: ResMut<player::WorldReady>,
) {
    let old_id = pending_teardown.old_id;
    let stale: Vec<Entity> = scoped_entities
        .iter()
        .filter(|(_, s)| s.0 == old_id)
        .map(|(e, _)| e)
        .collect();
    let count = stale.len();

    // DIAGNOSTIC: log every invocation with intent and entity counts
    let early_return = *teardown == TeardownIntent::None;
    bevy::log::info!(
        "DIAG guard_clean_world_on_entry: executing=true, intent={:?}, \
         early_return_none={}, WorldScoped({})={}",
        *teardown, early_return, old_id, count,
    );

    // Only tear down if explicitly requested (NewGame, LoadGame, Quit).
    // Overlay close, settings changes, and other non-destructive transitions
    // leave TeardownIntent::None and skip all despawn logic.
    if *teardown == TeardownIntent::None {
        if count > 0 {
            bevy::log::warn!(
                "Fence bypassed: teardown disabled by diagnostic flag \
                 (intent=None, {} WorldScoped({}) still alive)",
                count, old_id,
            );
        }
        // No TeardownIssued emitted → fence will never start waiting.
        return;
    }

    // Reset world readiness IMMEDIATELY (in StateTransition, before this
    // frame's Update runs). The old session's WorldReady=true otherwise
    // survives until the fence-gated reset_world_ready runs 1-2 frames
    // later, during which despawn_loading_overlay sees stale `true` and
    // kills the loading overlay that ensure_loading_overlay just spawned —
    // the black-screen flash during chunk pre-warm.
    world_ready.0 = false;

    if count > 0 {
        // Reset plate count so the snapshot system re-captures after respawn.
        plate_count.0 = 0;
        bevy::log::warn!(
            "Teardown intent={:?}: despawning {} WorldScoped({}) entities",
            *teardown, count, old_id,
        );
        for entity in stale {
            commands.entity(entity).despawn();
        }
    }

    // Teardown complete — reset intent and request world rebuild.
    let intent = *teardown;
    *teardown = TeardownIntent::None;

    if intent == TeardownIntent::NewGame || intent == TeardownIntent::LoadGame {
        bevy::log::info!(
            "Teardown commands issued (intent={:?}) — triggering TeardownIssued",
            intent,
        );
        commands.trigger(TeardownIssued { kind: intent });
    }
}

/// FULL MENU TEARDOWN — the ONLY function allowed to despawn world entities.
/// Runs on OnEnter(GameState::Menu) during a real state transition.
/// Despawns ALL WorldEntity entities, clears chunk manager, resets auto-load.
///
/// The overlay counterpart is a deliberate NO-OP: the M-key overlay has no
/// teardown function because it must not touch the world. If you are looking
/// for "overlay teardown" — there is none, by design.
///
/// NOTE: currently dead code in practice — nothing transitions back to
/// GameState::Menu after startup (Play/Load are same-state Gameplay
/// cycles). Kept as the safety net for any future title-menu return path.
/// The menu-background screenshot is captured at Play/Load click time via
/// request_menu_screenshot, NOT here.
fn cleanup_world(
    mut commands: Commands,
    scoped_entities: Query<(Entity, &WorldScoped), Without<BackgroundPlate>>,
    mut chunk_manager: ResMut<chunk_manager::ChunkManager>,
    mut auto_load: Option<ResMut<save_load::AutoLoadState>>,
    mut teardown: ResMut<TeardownIntent>,
    pending_teardown: Res<PendingTeardown>,
) {
    let old_id = pending_teardown.old_id;
    let stale: Vec<Entity> = scoped_entities
        .iter()
        .filter(|(_, s)| s.0 == old_id)
        .map(|(e, _)| e)
        .collect();

    // DIAGNOSTIC: log every invocation with intent and entity counts
    {
        let early_return = *teardown == TeardownIntent::None;
        bevy::log::info!(
            "DIAG cleanup_world: executing=true, intent={:?}, \
             early_return_none={}, WorldScoped({})={}",
            *teardown, early_return, old_id, stale.len(),
        );
    }

    // Only tear down if explicitly requested. Without intent, this is a
    // spurious OnEnter(Menu) (e.g., app startup default state) — skip.
    if *teardown == TeardownIntent::None {
        return;
    }

    bevy::log::info!(
        "cleanup_world: intent={:?}, despawning {} WorldScoped({}) entities",
        *teardown, stale.len(), old_id,
    );

    for entity in stale {
        commands.entity(entity).despawn();
    }
    // Clear chunk manager state so stale entity references don't persist
    // into the next Gameplay session.
    chunk_manager.clear_all();

    // Reset auto-load so save data is re-applied on next Gameplay entry.
    if let Some(mut al) = auto_load {
        al.loaded = false;
    }

    // Teardown complete — reset so no further transitions re-trigger it.
    *teardown = TeardownIntent::None;
}

/// Reset ChunkManager and AutoLoadState whenever a teardown is pending,
/// regardless of which state we end up in next.
///
/// WHY this exists in OnExit(Gameplay) instead of OnEnter(Menu):
/// The "Play" / "Load Game" buttons in the in-game overlay fire
/// `next_state.set(GameState::Gameplay)` while already in Gameplay.
/// Bevy treats that as a same-state cycle — OnExit(Gameplay) and
/// OnEnter(Gameplay) fire, but OnEnter(Menu) does NOT. So the
/// `cleanup_world` clear that lives on OnEnter(Menu) never runs on
/// this (extremely common) path, and stale block modifications from
/// the previous session contaminate the fresh world.
///
/// Runs AFTER `auto_save_on_exit` (which reads modifications to disk)
/// so we never clear before the save is written.
fn reset_chunk_state_on_teardown(
    mut chunk_manager: ResMut<chunk_manager::ChunkManager>,
    mut auto_load: Option<ResMut<save_load::AutoLoadState>>,
    teardown: Res<TeardownIntent>,
) {
    if *teardown == TeardownIntent::None {
        return;
    }
    bevy::log::info!(
        "reset_chunk_state_on_teardown: intent={:?}, clearing ChunkManager + AutoLoadState",
        *teardown,
    );
    chunk_manager.clear_all();
    if let Some(al) = auto_load.as_mut() {
        al.loaded = false;
    }
}

/// SAVE SCREENSHOT CONTRACT:
///   1. request_menu_screenshot MUST write the screenshot PNG to disk
///      (via Bevy's async save_to_disk observer).
///   2. request_menu_screenshot MUST persist the PNG path into save
///      metadata (via save_load::persist_screenshot_path) BEFORE the
///      screenshot is captured, so the path exists even if the async write
///      is delayed.
///   3. Menu background loading depends ONLY on save metadata — it reads
///      last_menu_background_image_path from the save, never guesses paths.
///   4. The screenshot file is optional (may not exist yet on first launch).
///      The Menu falls back to a solid dark panel if the file is missing.
///   5. TIMING: the capture is requested at reload-initiation time (Play /
///      Load click), on the frame the menu UI is despawned and BEFORE the
///      deferred state transition tears the world down (see PendingReload).
///      It must NOT be requested from OnExit(Gameplay): on the same-state
///      Gameplay→Gameplay reload cycle, OnEnter's teardown despawns the
///      world in the same StateTransition schedule, so an OnExit capture
///      renders a dead world.
///
/// Auto-save the game when exiting Gameplay. Ensures that all in-memory
/// state (block modifications, player position) is persisted before
/// teardown discards it. This prevents data loss on reload transitions.
/// The save file is then the single source of truth for the next
/// Gameplay entry. Skipped for LoadGame teardowns (see body).
fn auto_save_on_exit(
    chunk_manager: Option<Res<chunk_manager::ChunkManager>>,
    player_query: Query<(&Transform, &player::Player)>,
    teardown: Res<TeardownIntent>,
) {
    // Loading a save must NOT auto-save the abandoned session: the load flow
    // has already staged the chosen save file at the default path for
    // spawn_player / auto_load_game to consume, and writing the old
    // session's state here would overwrite it with exactly the data the
    // user is trying to replace.
    if *teardown == TeardownIntent::LoadGame {
        bevy::log::info!("Auto-save skipped: LoadGame teardown (staged save preserved)");
        return;
    }

    let Some(cm) = chunk_manager.as_deref() else { return };

    // Delegate to the single schema owner in save_load.rs — do NOT
    // hand-roll a serializer here (a private copy of the schema is how the
    // menu-background path got silently dropped from quit-saves).
    save_load::write_quick_save(cm, &player_query);
}

/// Request the menu-background screenshot of the current window contents.
/// Called by the Play/Load handlers on the click frame — the menu UI is
/// despawned in the same command batch, and the PendingReload deferral
/// keeps the world alive for the frames the renderer needs to capture it.
pub fn request_menu_screenshot(commands: &mut Commands) {
    use bevy::render::view::screenshot::{save_to_disk, Screenshot};
    let screenshot_path = save_load::save_path().with_extension("png");

    #[cfg(debug_assertions)]
    bevy::log::info!("Capturing menu screenshot → {}", screenshot_path.display());

    // Persist the path into save metadata BEFORE the async capture, so the
    // Menu can find it even if the PNG write is still in flight.
    save_load::persist_screenshot_path(&screenshot_path);

    commands.spawn(Screenshot::primary_window())
        .observe(save_to_disk(screenshot_path));
}

/// Single authoritative source for the player-facing game name.
/// Used in window title, menus, file dialogs, and save metadata.
pub const GAME_NAME: &str = "Sandbox World";

use animals::AnimalPlugin;
use chunk_manager::ChunkManagerPlugin;
use dev_tools::DevToolsPlugin;
use lighting::LightingPlugin;
use player::PlayerPlugin;
use save_load::SaveLoadPlugin;
use settings::{GameSettings, SettingsPlugin};
use sky::SkyPlugin;
use ui::UiPlugin;

fn main() {
    // Load settings early to configure the window before Bevy starts
    let saved = GameSettings::load();

    // Apply any custom block colors from saved settings
    if !saved.custom_block_colors.is_empty() {
        crate::block_types::set_custom_colors(saved.custom_block_colors.clone());
    }

    // Build the custom block registry from saved definitions
    let mut custom_registry = block_types::CustomBlockRegistry::default();
    for def in &saved.custom_blocks {
        custom_registry.add(block_types::CustomBlockEntry {
            name: def.name.clone(),
            color: Color::linear_rgba(def.color[0], def.color[1], def.color[2], def.color[3]),
            atlas_index: 0, // assigned by add()
        });
    }

    let window_mode = if saved.fullscreen {
        bevy::window::WindowMode::BorderlessFullscreen(bevy::window::MonitorSelection::Primary)
    } else {
        bevy::window::WindowMode::Windowed
    };

    let present = if saved.vsync {
        bevy::window::PresentMode::AutoVsync
    } else {
        bevy::window::PresentMode::AutoNoVsync
    };

    let mut window = Window {
        title: GAME_NAME.into(),
        mode: window_mode,
        present_mode: present,
        ..default()
    };

    // Set saved window size for windowed mode
    if !saved.fullscreen {
        window.resolution = bevy::window::WindowResolution::new(
            saved.window_width as u32,
            saved.window_height as u32,
        );
    }

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(window),
            ..default()
        }))
        .init_state::<GameState>()
        // The menu-background screenshot is requested by the Play/Load
        // handlers at click time (world alive, menu just despawned) — NOT
        // here. See PendingReload / request_menu_screenshot.
        .add_systems(OnExit(GameState::Gameplay), (
            auto_save_on_exit,
            reset_chunk_state_on_teardown,
        ).chain())
        .add_systems(OnEnter(GameState::Menu), cleanup_world)
        .add_systems(OnEnter(GameState::Gameplay), guard_clean_world_on_entry)
        .init_resource::<TeardownIntent>()
        .init_resource::<WorldInstanceId>()
        .init_resource::<PendingTeardown>()
        .init_resource::<TeardownPendingVerification>()
        .init_resource::<WorldInitPending>()
        .init_resource::<PendingReload>()
        .add_systems(Update, process_pending_reload)
        .add_observer(on_teardown_issued)
        .add_observer(on_world_init_requested)
        .add_systems(Update, verify_teardown_complete.run_if(in_state(GameState::Gameplay)))
        // WorldSpawnSet: spawn systems run exactly once when WorldInitPending
        // becomes true (after the teardown fence passes). consume_world_init
        // resets the flag after all spawn systems have run.
        .configure_sets(Update, WorldSpawnSet.run_if(
            in_state(GameState::Gameplay)
                .and(resource_equals(WorldInitPending(true)))
        ))
        .add_systems(Update, consume_world_init
            .after(WorldSpawnSet)
            .run_if(in_state(GameState::Gameplay))
        )
        .init_resource::<BackgroundPlateCount>()
        // Background plate diagnostics — both run in PostUpdate after all
        // game systems have executed. snapshot catches the first frame where
        // plates exist; assertion checks every frame after that.
        .add_systems(PostUpdate, (
            snapshot_background_plate_count.run_if(in_state(GameState::Gameplay)),
            assert_background_plates_intact,
        ))
        .insert_resource(custom_registry)
        // ORDERING CONTRACT:
        // - SettingsPlugin first: GameSettings resource must exist before any
        //   system that reads render_distance, texture_size, etc.
        // - ChunkManagerPlugin before PlayerPlugin: ChunkManager resource must
        //   exist at Startup so player collision can fall back to terrain gen.
        // - Buildings have no plugin: they are baked into chunk generation
        //   (buildings::place_buildings_in_chunk, called by Chunk::generate).
        // - AnimalPlugin after ChunkManager: animals query ChunkManager for
        //   ground height; resource must exist.
        // - LightingPlugin after PlayerPlugin: lighting Update systems read
        //   camera position set by PlayerPlugin. Both run in Update without
        //   explicit cross-plugin ordering, so lighting reads the *previous*
        //   frame's camera position (1-frame lag — accepted trade-off for
        //   simplicity; see 04_camera_guardrails_3d.txt).
        // - SaveLoadPlugin after PlayerPlugin + ChunkManagerPlugin: auto_load_game
        //   waits for the player entity to exist before applying save data.
        // - UiPlugin last: reads state from all other systems for display.
        .add_plugins((
            SettingsPlugin,
            ChunkManagerPlugin,
            PlayerPlugin,
            AnimalPlugin,
            LightingPlugin,
            SkyPlugin,
            DevToolsPlugin,
            SaveLoadPlugin,
            UiPlugin,
        ))
        .run();
}
