//! Sandbox World — a Minecraft-style voxel game on Bevy 0.19.
//!
//! A successor to the Bevy 0.18 project at ~/Documents/claude/metalworld-bevy,
//! which serves as a reference for design decisions and tuning rather than as
//! code to reproduce. See PLAN.md for the phase breakdown.
//!
//! Terrain generation, chunk streaming and greedy meshing, voxel lighting with a
//! day/night cycle, a first-person player, wandering animals, a title menu, and
//! saved worlds.

// Release builds on Windows launch without a console window. Without this a
// double-clicked .exe opens a black console beside the game — the same wart the
// macOS .app bundle exists to avoid. Debug builds keep the console, since that
// is where the log goes and losing it would make `cargo run` silent.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod animals;
mod block_types;
mod buildings;
mod chunk;
mod chunk_manager;
mod dev_tools;
mod lighting;
mod platform;
mod player;
mod ray_cast;
mod save_load;
mod settings;
mod sky;
mod terrain;
mod ui;
mod voxel_light;

use bevy::prelude::*;

pub const GAME_NAME: &str = "Sandbox World";

// ---------------------------------------------------------------------------
// Crate-root state and markers
//
// The state enum and the world-scoping markers every module stamps onto its
// entities. `ui::teardown_world` is what gives `WorldScoped` its meaning.
// ---------------------------------------------------------------------------

/// HARD RULE — Menu state policy:
///   The title menu is NOT a paused game world. It is a clean UI-only state.
///   No world entities (cameras, chunks, animals, lights) may exist in Menu.
///   No world simulation, rendering, or physics may run in Menu.
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

/// Monotonically increasing id that identifies the current world instance.
/// Incremented on each new-game / load-game cycle so that stale entities
/// from a previous world can be distinguished from current ones.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WorldInstanceId(pub u64);

/// Stamps an entity with the world instance it was spawned into.
/// Teardown / fence queries filter on this to ignore strays.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldScoped(pub u64);

/// Deferred world-reload countdown. Phase 5 drives this from the menu; the
/// voxel core only reads it to suppress chunk spawning mid-transition.
#[derive(Resource, Default)]
pub struct PendingReload {
    pub active: bool,
    pub frames: u8,
}

/// Number of frames to keep the world alive (menu closed) before the
/// reload transition, so a screenshot capture sees a fully rendered frame.
pub const RELOAD_DEFER_FRAMES: u8 = 2;

fn main() {
    // Only the wasm block below mutates this.
    #[cfg_attr(not(target_arch = "wasm32"), allow(unused_mut))]
    let mut window = Window {
        title: GAME_NAME.into(),
        ..default()
    };

    // Screenshot runs capture the swapchain of the real window. macOS stops
    // presenting an occluded window, so a capture taken while the terminal is in
    // front grabs an empty buffer — a pure-black frame with no clear colour,
    // which looks exactly like a render failure and is not one. Keep the window
    // on top for the duration of a capture run.
    if std::env::var("SW2_SCREENSHOT").is_ok() {
        window.window_level = bevy::window::WindowLevel::AlwaysOnTop;
        window.focused = true;
    }

    // On web the page owns the canvas: winit binds to the element below and
    // tracks its parent's size, so window mode and resolution do not apply.
    // Sizing is CSS-driven in web/index.html.
    #[cfg(target_arch = "wasm32")]
    {
        window.canvas = Some("#sandbox-world-canvas".into());
        window.fit_canvas_to_parent = true;
    }

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(window),
            ..default()
        }))
        .init_state::<GameState>()
        .init_resource::<WorldInstanceId>()
        .init_resource::<PendingReload>()
        .init_resource::<dev_tools::DevSettings>()
        .init_resource::<dev_tools::OptimizationFlags>()
        .init_resource::<block_types::CustomBlockRegistry>()
        // ORDERING CONTRACT (see CLAUDE.md): SettingsPlugin first — GameSettings
        // must exist before anything reads render_distance / texture_size.
        // ChunkManagerPlugin before PlayerPlugin: the ChunkManager resource must
        // exist before player collision queries it.
        // LightingPlugin before SkyPlugin: the sky reads SunCycle, and the fog
        // colour in turn reads the ClearColor the sky writes.
        .add_plugins((
            settings::SettingsPlugin,
            chunk_manager::ChunkManagerPlugin,
            player::PlayerPlugin,
            lighting::LightingPlugin,
            sky::SkyPlugin,
            // After ChunkManagerPlugin: wandering queries it for ground height.
            animals::AnimalPlugin,
            // Last: the HUD reads state the other plugins produce, and the menu
            // owns the teardown that runs when the world goes away.
            ui::UiPlugin,
            // After PlayerPlugin and ChunkManagerPlugin: autosave captures the
            // player entity and the modification map, and load writes to both.
            save_load::SaveLoadPlugin,
        ))
        .add_systems(Update, (screenshot_after_warmup, state_cycle_probe))
        .run();
}

/// Drives Menu → Gameplay → Menu → Gameplay on a timer when `SW2_CYCLE=1` is
/// set, logging how many world entities survive each teardown.
///
/// Teardown leaks are invisible to a unit test of the teardown system itself:
/// what goes wrong in the real app is a system spawning world content *after*
/// the fence has run, or a plugin forgetting `WorldScoped` entirely. Only a real
/// state cycle catches that, and the count must return to zero every time.
fn state_cycle_probe(
    time: Res<Time>,
    state: Res<State<GameState>>,
    mut next: ResMut<NextState<GameState>>,
    mut screen: ResMut<ui::MenuScreen>,
    world_entities: Query<(), With<WorldScoped>>,
    mut step: Local<usize>,
) {
    if std::env::var("SW2_CYCLE").is_err() {
        return;
    }
    // Menu, play, back to menu, play again — two full teardowns.
    const SCHEDULE: [(f32, GameState); 4] = [
        (2.0, GameState::Gameplay),
        (6.0, GameState::Menu),
        (8.0, GameState::Gameplay),
        (12.0, GameState::Menu),
    ];

    let Some(&(at, target)) = SCHEDULE.get(*step) else {
        return;
    };
    if time.elapsed_secs() < at {
        return;
    }
    *step += 1;

    info!(
        "cycle: t={at}s {:?} -> {target:?} | world entities before transition = {}",
        state.get(),
        world_entities.iter().count(),
    );
    *screen = ui::MenuScreen::Main;
    next.set(target);
}

/// Verification hook. With `SW2_SCREENSHOT=<path>` set, captures the primary
/// window once chunks have streamed in, so a change can be checked without a
/// human at the keyboard. Inert without the env var.
///
/// It logs alongside the image on purpose. A capture of an occluded window comes
/// back all black on macOS, and an embedded camera renders a perfectly plausible
/// scene — neither is diagnosable from the picture alone, so the numbers are
/// what actually tell you whether the frame means anything.
fn screenshot_after_warmup(
    mut commands: Commands,
    time: Res<Time>,
    mut fired: Local<bool>,
    meshed: Query<&ViewVisibility, With<Mesh3d>>,
    cam: Query<&GlobalTransform, With<Camera3d>>,
    chunks: Res<chunk_manager::ChunkManager>,
    player_q: Query<(&Transform, &player::Player)>,
) {
    use bevy::render::view::screenshot::{Screenshot, save_to_disk};

    let Ok(path) = std::env::var("SW2_SCREENSHOT") else {
        return;
    };
    // Capture time is adjustable so a run can be caught mid-state-cycle rather
    // than only after the world has settled.
    let at = std::env::var("SW2_SCREENSHOT_AT")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(8.0);
    if *fired || time.elapsed_secs() < at {
        return;
    }
    *fired = true;

    let total = meshed.iter().count();
    let visible = meshed.iter().filter(|v| v.get()).count();
    for transform in &cam {
        info!(
            "screenshot: camera at {:?} | mesh entities={total} visible={visible}",
            transform.translation(),
        );
    }
    // Is the player standing in clear air, or embedded in geometry?
    //
    // Reported against absolute block Y, not an offset from the feet: foot_y
    // lands on a block boundary when standing, so a relative index flips by one
    // on float rounding and makes "standing on grass" look like "buried in it".
    for (transform, player) in &player_q {
        let feet = transform.translation - Vec3::Y * player.eye_height;
        let base = feet.y.floor() as i32;
        let column: Vec<_> = (base - 1..=base + 2)
            .map(|y| {
                let p = IVec3::new(feet.x.floor() as i32, y, feet.z.floor() as i32);
                (y, chunks.block_at(p).name())
            })
            .collect();
        info!(
            "screenshot: feet at y={:.3} | column (y, block) = {column:?}",
            feet.y
        );
    }

    commands
        .spawn(Screenshot::primary_window())
        .observe(save_to_disk(path));
}
