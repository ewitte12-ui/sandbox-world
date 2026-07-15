use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use std::f32::consts::PI;

use crate::block_types::BlockType;
use crate::chunk_manager::ChunkManager;
use crate::dev_tools::DevSettings;
use crate::ray_cast::cast_ray;
use crate::terrain;
use crate::ui::InventoryState;
use crate::GameState;
use crate::save_load::FileDialogOpen;

// ---------------------------------------------------------------------------
// Run conditions
// ---------------------------------------------------------------------------

/// Run condition: returns true when the settings/menu overlay is closed.
fn menu_closed(menu_state: Option<Res<crate::ui::MenuState>>) -> bool {
    !menu_state.map(|m| m.is_open).unwrap_or(false)
}

/// Run condition: returns true when the inventory overlay is closed.
fn inventory_closed(inv_state: Option<Res<InventoryState>>) -> bool {
    !inv_state.map(|i| i.is_open).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Save restore
// ---------------------------------------------------------------------------

/// Load player state from the default save (any format — save_load sniffs
/// by content). Returns the restored Player and position. If no save or
/// no player snapshot exists, returns defaults (new game at origin).
fn load_player_from_save() -> (Player, Vec3) {
    let Some(state) = crate::save_load::read_default_save().and_then(|s| s.player) else {
        #[cfg(debug_assertions)]
        bevy::log::info!("No save/player snapshot — starting new game at default position");
        return (Player::default(), Vec3::new(0.0, 20.0, 0.0));
    };

    let mut player = Player::default();
    player.yaw = state.yaw;
    player.pitch = state.pitch;
    player.home_position = state.home_position;

    #[cfg(debug_assertions)]
    bevy::log::info!(
        "Restored player from save: pos=({:.1},{:.1},{:.1}) yaw={:.2} pitch={:.2}",
        state.position.x, state.position.y, state.position.z, player.yaw, player.pitch,
    );

    (player, state.position)
}

// ---------------------------------------------------------------------------
// World readiness
// ---------------------------------------------------------------------------

/// Indicates whether the world is ready for physics. Set to false on entering
/// Gameplay, set to true once chunks near the player are loaded and the save
/// has been applied. Physics systems are gated behind this.
#[derive(Resource)]
pub struct WorldReady(pub bool);

impl Default for WorldReady {
    fn default() -> Self {
        Self(false)
    }
}

/// Run condition: returns true only when the world is ready for physics.
fn world_ready(ready: Option<Res<WorldReady>>) -> bool {
    ready.map(|r| r.0).unwrap_or(false)
}

/// Run condition: false while a Play/Load reload countdown is active.
/// During the deferral the menu is closed and the cursor is grabbed, so
/// without this gate a mouse button still held from the menu click would
/// reach block_interact and edit the world right before it is torn down
/// (and right before the menu-background screenshot is captured).
fn reload_not_pending(pending: Res<crate::PendingReload>) -> bool {
    !pending.active
}

/// System: check if chunks near the player are loaded and mark world as ready.
fn check_world_ready(
    mut ready: ResMut<WorldReady>,
    mut player_query: Query<(&mut Transform, &Player, &mut Camera)>,
    chunk_manager: Option<Res<ChunkManager>>,
) {
    if ready.0 {
        return;
    }
    let Some(cm) = chunk_manager else { return };
    let Some((mut transform, player, mut camera)) = player_query.iter_mut().next() else {
        return;
    };

    let pos = transform.translation;
    let chunk_x = (pos.x as i32).div_euclid(crate::chunk::CHUNK_SIZE);
    let chunk_y = (pos.y as i32).div_euclid(crate::chunk::CHUNK_SIZE);
    let chunk_z = (pos.z as i32).div_euclid(crate::chunk::CHUNK_SIZE);

    // Require the player's chunk AND the 8 horizontal neighbors to be
    // meshed before activating the camera. The single-chunk gate that
    // lived here previously meant adjacent chunks could still be in
    // flight on AsyncComputeTaskPool when the camera turned on,
    // producing a visible "hole in the ground at spawn" that filled in
    // over the next several frames. Jumping appeared to fix it only
    // because time passed and async tasks completed.
    let mut all_loaded = true;
    'outer: for dx in -1..=1 {
        for dz in -1..=1 {
            let neighbor = crate::chunk::ChunkPos(chunk_x + dx, chunk_y, chunk_z + dz);
            if !cm.chunks.contains_key(&neighbor) {
                all_loaded = false;
                break 'outer;
            }
        }
    }
    // Also require the chunk directly below the player so the ground
    // beneath spawn is meshed before the camera activates. Skip this
    // check if that chunk is below the bedrock floor (-7) that
    // load_chunks refuses to enqueue — otherwise the gate would
    // deadlock for any save whose Y puts the player at chunk_y <= -7.
    let below_y = chunk_y - 1;
    let below_loaded = if below_y < -7 {
        true
    } else {
        cm.chunks.contains_key(&crate::chunk::ChunkPos(chunk_x, below_y, chunk_z))
    };

    if all_loaded && below_loaded {
        // Snap player to solid ground before the first physics frame.
        // Without this, the player spawns at save-file Y (or default Y=20)
        // and free-falls until player_collision catches them.
        let ground = find_ground_height(pos, player.eye_height, Some(cm.as_ref()));
        if ground > f32::NEG_INFINITY {
            transform.translation.y = ground;
        }

        // Activate the camera now that the player is grounded and the
        // surrounding ground is fully meshed.
        camera.is_active = true;

        ready.0 = true;
        #[cfg(debug_assertions)]
        bevy::log::info!(
            "World ready — player chunk ({},{},{}) + 3x3 neighbors + below loaded, snapped Y {:.1} → {:.1}",
            chunk_x, chunk_y, chunk_z, pos.y, transform.translation.y,
        );
    }
}

/// Reset world readiness when entering Gameplay.
/// Skipped if a player already exists (spurious same-state re-entry).
fn reset_world_ready(mut ready: ResMut<WorldReady>, player: Query<(), With<Player>>) {
    if !player.is_empty() {
        return;
    }
    ready.0 = false;
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Marker + state for the player-controlled FPS camera entity.
///
/// Tuning values (speed, sensitivity, gravity, etc.) are NOT stored here.
/// They live in DevSettings and are read each frame, so runtime changes
/// to DevSettings take effect immediately. Player holds only per-entity
/// state that varies during gameplay.
#[derive(Component)]
pub struct Player {
    pub yaw: f32,
    pub pitch: f32,
    pub eye_height: f32,
    pub standing_eye_height: f32,
    pub crouch_eye_height: f32,
    pub vertical_velocity: f32,
    pub is_on_ground: bool,
    pub is_sneaking: bool,
    pub selected_block: BlockType,
    pub home_position: Option<Vec3>,
    pub keyboard_sensitivity: f32,
}

impl Default for Player {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: -0.1, // slight downward tilt so the player sees terrain on spawn
            eye_height: 1.8,
            standing_eye_height: 1.8,
            crouch_eye_height: 1.2,
            vertical_velocity: 0.0,
            is_on_ground: false,
            is_sneaking: false,
            selected_block: BlockType::STONE,
            home_position: None,
            keyboard_sensitivity: 3.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Constants — geometry and physics constraints only.
//
// SINGLE SOURCE OF TRUTH: all gameplay-tunable values (gravity, jump velocity,
// reach, break/place intervals, speed multipliers) live in DevSettings
// (dev_tools.rs). Player systems read from Res<DevSettings> at runtime.
// This eliminates the prior duplication where player.rs had its own constants
// that shadowed DevSettings defaults.
//
// The constants below are NOT tuning values — they are geometric constraints
// or physics limits that define the player's collision shape, camera math,
// and verticality safety bounds. They remain here because they are tightly
// coupled to the collision code and have no reason to be adjusted at runtime.
//
// - PLAYER_RADIUS 0.4: slightly under half-block so the player fits
//   through 1-wide corridors with margin for float imprecision.
// - PLAYER_WALL_RADIUS 0.5: wider than PLAYER_RADIUS to prevent camera
//   from clipping into walls while the body still fits through gaps.
// - PITCH_LIMIT: 0.01 rad below π/2 to avoid gimbal lock singularity
//   in the YXZ Euler rotation.
// - CROUCH_TRANSITION_SPEED: eye-height interpolation rate (blocks/sec).
// - MAX_CONTINUOUS_BREAKS: caps held-click mining to 5 blocks, requiring
//   a re-click to continue — prevents accidental excavation.
// - TERMINAL_VELOCITY: max downward speed (see Verticality constraints).
// - VOID_KILL_PLANE_Y: safety floor (see Verticality constraints).
// ---------------------------------------------------------------------------

const CROUCH_TRANSITION_SPEED: f32 = 10.0;
const PLAYER_RADIUS: f32 = 0.4;
const PLAYER_WALL_RADIUS: f32 = 0.5;
const PITCH_LIMIT: f32 = PI / 2.0 - 0.01;
const MAX_CONTINUOUS_BREAKS: u32 = 5;

// ---------------------------------------------------------------------------
// Verticality constraints
//
// CONTRACT — VERTICALITY INVARIANTS:
//   1. The player's downward velocity never exceeds TERMINAL_VELOCITY in
//      magnitude. This prevents collision-skipping at extreme speeds and
//      keeps long falls feeling controlled rather than instantaneous.
//   2. The player's Y position never goes below VOID_KILL_PLANE_Y. If it
//      does, the player is teleported to a safe recovery point. This
//      guarantees that falling into unloaded void is always recoverable.
//   3. Recovery must always exist: if the player has placed a bed
//      (home_position), they return there. Otherwise, they return to the
//      world origin at terrain height. Downward mistakes cost time (the
//      fall + respawn), never inevitability (permanent softlock).
//   4. Terminal velocity is a physics constraint, not gameplay tuning.
//      Changing it affects collision reliability, not game feel.
//
// CONTRACT — NO-SOFTLOCK INVARIANT:
//   The player can never be in a state with no escape. This is enforced by
//   three layered mechanisms:
//     a. H key always teleports to safety (bed or spawn). No precondition
//        required — works even if trapped, even if no bed was placed.
//     b. Bedrock cannot be placed by the player. Since bedrock is the only
//        unbreakable block, preventing placement means the player can always
//        break out of any player-constructed enclosure.
//     c. Void kill plane (VOID_KILL_PLANE_Y) catches falls through unloaded
//        chunks and teleports to safety automatically.
//   These three mechanisms are independent — any one of them is sufficient
//   to prevent a permanent softlock.
// ---------------------------------------------------------------------------

/// Maximum downward velocity in blocks/sec (negative = downward).
/// At gravity -28.0, reached after ~1.4 seconds of freefall.
///
/// 40 b/s gives comfortable fall timing: surface-to-bedrock (100 blocks)
/// takes ~4 seconds total, giving the player time to react. Also keeps
/// per-frame displacement under 1.3 blocks at 30fps, well within the
/// collision scan range (±20 blocks).
const TERMINAL_VELOCITY: f32 = -40.0;

/// Y coordinate below which the player is considered to have fallen into
/// void and is teleported to safety.
///
/// Set to -150: well below the deepest possible bedrock (terrain surface
/// tops at Y≈10, bedrock at depth 100 → Y≈-90). Normal gameplay never
/// reaches this — it only triggers if the player falls through unloaded
/// chunks or encounters an edge case in collision.
const VOID_KILL_PLANE_Y: f32 = -150.0;

// ---------------------------------------------------------------------------
// Surface classification
//
// Every solid block adjacent to the player's collision cylinder is classified
// into exactly one SurfaceType before any collision response is applied.
// This makes the role each block plays explicit and prevents collision logic
// from implicitly assuming a surface role.
//
// CONTRACT per surface type:
//   Ground  — block top is less than 1 full block above the player's feet
//             (step_height < STEP_UP_THRESHOLD). Every surface below this
//             threshold is fully walkable — no half-affordance gap.
//             RESPONSE: snap player Y to block top, zero vertical velocity,
//             set is_on_ground = true. Gravity stops.
//   Wall    — block vertically overlaps the player body (between feet and
//             head) and is neither Ground nor Ceiling.
//             RESPONSE: push player horizontally (X/Z only). Never modifies
//             Y or vertical velocity.
//   Ceiling — block bottom is at or above the player's head.
//             RESPONSE: cap player Y below the block, zero upward velocity.
//             Gravity continues (is_on_ground stays false).
//   Slope   — reserved for future use. No blocks currently produce this.
//             When implemented, should behave as Ground with a movement
//             speed modifier.
// ---------------------------------------------------------------------------

/// Explicit surface role for a block contact with the player's collision
/// cylinder. Determined by [`classify_block_contact`] before any collision
/// response is applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SurfaceType {
    /// Block top is less than 1 full block above the player's feet (see
    /// STEP_UP_THRESHOLD). Gravity stops. Player snaps to block top.
    Ground,
    /// Block overlaps the player body vertically but is neither Ground nor
    /// Ceiling. Blocks horizontal movement only — never modifies Y.
    Wall,
    /// Block bottom is at or above the player's head. Caps upward movement.
    /// Gravity continues.
    Ceiling,
    /// Reserved for angled surfaces. Not currently generated in this voxel
    /// world (all blocks are axis-aligned cubes).
    Slope,
}

/// Maximum step-up height: the player can auto-climb any surface whose top
/// is less than one full block above their feet.
///
/// CONTRACT — WALKABILITY INVARIANT:
///   let step = block_top - foot_y;
///   step < STEP_UP_THRESHOLD  →  SurfaceType::Ground (walkable without jump)
///   step >= STEP_UP_THRESHOLD →  SurfaceType::Wall    (must jump to clear)
///
/// Set to 1.0 minus a small float-safe margin (0.02). This makes the rule
/// deterministic and complete: every surface below a 1-block step is walkable,
/// every surface at or above 1 block requires jumping. There is no gap.
///
/// Previously 0.65 — an arbitrary value ported from Swift that left surfaces
/// between 0.65 and 1.0 blocks above foot_y in a half-affordance gap (visually
/// climbable but functionally blocked). In the current integer-aligned voxel
/// world all step heights are 0 or ≥1, so the old value produced identical
/// gameplay. But 0.98 is:
///   - Clearer: the rule is "less than 1 block = walkable", not "less than
///     0.65 blocks = walkable".
///   - Robust: at low framerates gravity can push foot_y well below the
///     current block_top before the ground snap fires. The old 0.65 margin
///     failed at ~20fps; the 0.98 margin survives down to ~8fps.
///   - Future-safe: if slopes or half-blocks are added, surfaces at 0.5 or
///     0.7 blocks will be walkable with no code change.
///
/// The 0.02 margin prevents a block exactly 1.0 above foot_y from being
/// classified as Ground (that would let the player walk up full blocks
/// without jumping).
const STEP_UP_THRESHOLD: f32 = 1.0 - 0.02;

/// Small epsilon to prevent ceiling detection from catching on the block the
/// player is already inside due to float imprecision.
const CEILING_EPSILON: f32 = 0.05;

/// Classify a solid block's surface role relative to the player cylinder.
///
/// Classification is based purely on the vertical relationship between the
/// block and the player — horizontal proximity is checked separately by each
/// collision function. A single block has exactly one classification at any
/// given player position.
///
/// The same physical block may be Ground when approached from above and Wall
/// when approached from the side. This is correct: the classification captures
/// the block's role in the *current* frame, not a permanent property.
fn classify_block_contact(block_y: i32, foot_y: f32, head_y: f32) -> SurfaceType {
    let block_top = (block_y as f32) + 1.0;
    let block_bottom = block_y as f32;

    // Ground: block top is less than 1 full block above the player's feet.
    // Any surface below this threshold is walkable without jumping.
    if block_top <= foot_y + STEP_UP_THRESHOLD {
        return SurfaceType::Ground;
    }

    // Ceiling: block bottom is at or above the player's head (with epsilon).
    // This block caps upward movement.
    if block_bottom >= head_y - CEILING_EPSILON {
        return SurfaceType::Ceiling;
    }

    // Wall: block overlaps the player body vertically. It sits between
    // the step-up threshold and the head — blocking horizontal movement.
    SurfaceType::Wall
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// Tracks mouse hold state for continuous block interaction.
#[derive(Resource, Default)]
pub struct InteractionState {
    pub is_left_held: bool,
    pub is_right_held: bool,
    pub break_timer: f32,
    pub place_timer: f32,
    pub break_count: u32,
}

// ---------------------------------------------------------------------------
// HUD marker components
// ---------------------------------------------------------------------------

#[derive(Component)]
struct CrosshairMarker;

#[derive(Component)]
struct SelectedBlockText;

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InteractionState>()
            .init_resource::<WorldReady>()
            .add_systems(Startup, spawn_hud)
            .add_systems(Update, (spawn_player, reset_world_ready).in_set(crate::WorldSpawnSet))
            .add_systems(Update, check_world_ready
                .run_if(in_state(GameState::Gameplay)))
            // ORDERING CONTRACT: these systems are chained (run sequentially)
            // because each depends on state set by the previous:
            //   1. player_look       — reads mouse input, writes yaw/pitch
            //   2. player_move       — reads yaw (for direction), writes translation + velocity
            //   3. player_collision  — reads translation, resolves floor/ceiling/wall overlaps
            //   4. apply_player_transform — reads yaw/pitch, writes Transform.rotation
            //   5. block_select      — reads key input, writes selected_block
            //   6. block_interact    — reads transform + selected_block, modifies world
            //   7. teleport_home     — may override transform (must run after collision)
            //   8. update_selected_block_text — reads selected_block for HUD display
            .add_systems(
                Update,
                (
                    player_look,
                    player_move,
                    player_collision,
                    apply_player_transform,
                    block_select,
                    block_interact,
                    teleport_home,
                    update_selected_block_text,
                )
                    .chain()
                    .run_if(in_state(GameState::Gameplay))
                    .run_if(not(resource_exists::<FileDialogOpen>))
                    .run_if(menu_closed)
                    .run_if(inventory_closed)
                    .run_if(world_ready)
                    .run_if(reload_not_pending),
            );
    }
}

// ---------------------------------------------------------------------------
// Startup: spawn the player entity + grab cursor
// ---------------------------------------------------------------------------

fn spawn_player(
    mut commands: Commands,
    game_settings: Option<Res<crate::settings::GameSettings>>,
    existing_cameras: Query<Entity, (With<Camera3d>, With<crate::WorldEntity>)>,
    existing_player: Query<(), With<Player>>,
    start_mode: Res<crate::save_load::StartMode>,
    world_id: Res<crate::WorldInstanceId>,
) {
    // GUARD: if a Player entity already exists, skip — the world is
    // already populated. Never despawn or re-spawn the player during
    // live gameplay.
    if !existing_player.is_empty() {
        return;
    }

    // Ensure exactly one world camera — despawn any stale cameras from
    // a previous Gameplay session before spawning a fresh one.
    // Scoped to WorldEntity so UI cameras (MenuCamera, RenderScaleBlit) are never hit.
    for entity in &existing_cameras {
        commands.entity(entity).despawn();
    }

    // Restore player state from save or start fresh depending on StartMode.
    let (mut player, start_pos) = if *start_mode == crate::save_load::StartMode::NewGame {
        #[cfg(debug_assertions)]
        bevy::log::info!("New Game — spawning player at default position");
        (Player::default(), Vec3::new(0.0, 20.0, 0.0))
    } else {
        load_player_from_save()
    };

    // Zero out velocity on resume — prevents carrying momentum across
    // Menu→Gameplay transitions.
    player.vertical_velocity = 0.0;

    let mut cam = commands.spawn((
        crate::WorldEntity,
        crate::WorldScoped(world_id.0),
        player,
        Camera3d::default(),
        Camera {
            order: 0, // World camera renders first. UI/blit cameras use higher orders.
            // Inactive until check_world_ready snaps the player to ground.
            // Prevents a visible "spawn in air → snap" flash.
            is_active: false,
            ..default()
        },
        Transform::from_xyz(start_pos.x, start_pos.y, start_pos.z),
    ));

    // Apply initial AA/SSAO/SMAA at spawn time so they take effect on the first frame.
    // The apply_render_pipeline_settings system in lighting.rs handles live changes.
    if let Some(settings) = &game_settings {
        // SSAO (requires Msaa::Off)
        if settings.ssao_enabled {
            cam.insert(Msaa::Off);
            let quality = match settings.ssao_quality.as_str() {
                "low" => bevy::pbr::ScreenSpaceAmbientOcclusionQualityLevel::Low,
                "high" => bevy::pbr::ScreenSpaceAmbientOcclusionQualityLevel::High,
                "ultra" => bevy::pbr::ScreenSpaceAmbientOcclusionQualityLevel::Ultra,
                _ => bevy::pbr::ScreenSpaceAmbientOcclusionQualityLevel::Medium,
            };
            cam.insert(bevy::pbr::ScreenSpaceAmbientOcclusion {
                quality_level: quality,
                ..default()
            });
        } else {
            // Anti-aliasing
            match settings.anti_aliasing.as_str() {
                "msaa2" => { cam.insert(Msaa::Sample2); }
                "msaa4" => { cam.insert(Msaa::Sample4); }
                "taa" => {
                    cam.insert(Msaa::Off);
                    cam.insert(bevy::anti_alias::taa::TemporalAntiAliasing::default());
                }
                _ => { cam.insert(Msaa::Off); }
            }
        }

        // SMAA
        match settings.smaa_mode.as_str() {
            "low" => { cam.insert(bevy::anti_alias::smaa::Smaa { preset: bevy::anti_alias::smaa::SmaaPreset::Low }); }
            "medium" => { cam.insert(bevy::anti_alias::smaa::Smaa { preset: bevy::anti_alias::smaa::SmaaPreset::Medium }); }
            "high" => { cam.insert(bevy::anti_alias::smaa::Smaa { preset: bevy::anti_alias::smaa::SmaaPreset::High }); }
            "ultra" => { cam.insert(bevy::anti_alias::smaa::Smaa { preset: bevy::anti_alias::smaa::SmaaPreset::Ultra }); }
            _ => {}
        }
    } else {
        // No settings — default to TAA
        cam.insert(Msaa::Off);
        cam.insert(bevy::anti_alias::taa::TemporalAntiAliasing::default());
    }
    // Cursor starts free (menu open). Grab happens when the player enters
    // gameplay via ui.rs (Play/Load button or closing the menu).
}

/// Despawn the player camera and all 3D rendering entities when leaving
/// gameplay. Cameras must be freshly spawned on re-entry — no reuse.
fn despawn_player(
    mut commands: Commands,
    player_query: Query<Entity, With<Player>>,
) {
    for entity in &player_query {
        commands.entity(entity).despawn();
    }
}

// ---------------------------------------------------------------------------
// Startup: spawn crosshair + selected block HUD
// ---------------------------------------------------------------------------

fn spawn_hud(mut commands: Commands) {
    // Crosshair: a small "+" at the center of the screen
    commands
        .spawn((
            CrosshairMarker,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Percent(50.0),
                top: Val::Percent(50.0),
                ..default()
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("+"),
                TextFont {
                    font_size: 24.0,
                    ..default()
                },
                TextColor(Color::WHITE),
                Node {
                    // Offset to center the "+" character
                    left: Val::Px(-6.0),
                    top: Val::Px(-12.0),
                    ..default()
                },
            ));
        });

    // Selected block display at bottom center
    commands.spawn((
        SelectedBlockText,
        Text::new("[3] Stone"),
        TextFont {
            font_size: 20.0,
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(20.0),
            left: Val::Percent(50.0),
            // Approximate centering
            margin: UiRect {
                left: Val::Px(-60.0),
                ..default()
            },
            ..default()
        },
    ));
}

// ---------------------------------------------------------------------------
// System: player_look — mouse movement updates yaw / pitch
// ---------------------------------------------------------------------------

fn player_look(
    mouse_motion: Res<AccumulatedMouseMotion>,
    dev: Res<DevSettings>,
    mut query: Query<&mut Player>,
) {
    let delta = mouse_motion.delta;
    if delta == Vec2::ZERO {
        return;
    }

    for mut player in &mut query {
        player.yaw -= delta.x * dev.mouse_sensitivity;
        player.pitch -= delta.y * dev.mouse_sensitivity;
        player.pitch = player.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }
}

// ---------------------------------------------------------------------------
// System: player_move — WASD + jump + gravity + crouch
// ---------------------------------------------------------------------------

fn player_move(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    chunk_manager: Option<Res<ChunkManager>>,
    dev: Res<DevSettings>,
    mut query: Query<(&mut Transform, &mut Player)>,
) {
    let dt = time.delta_secs();
    if dt == 0.0 {
        return;
    }

    for (mut transform, mut player) in &mut query {
        // Sneaking state
        player.is_sneaking = keys.pressed(KeyCode::ShiftLeft);

        // Smooth eye-height transition (crouch / stand)
        let target_eye_height = if player.is_sneaking {
            player.crouch_eye_height
        } else {
            player.standing_eye_height
        };
        if player.eye_height < target_eye_height {
            player.eye_height =
                (player.eye_height + CROUCH_TRANSITION_SPEED * dt).min(target_eye_height);
        } else if player.eye_height > target_eye_height {
            player.eye_height =
                (player.eye_height - CROUCH_TRANSITION_SPEED * dt).max(target_eye_height);
        }

        let mut speed = if player.is_sneaking {
            dev.player_speed * dev.sneak_multiplier
        } else {
            dev.player_speed
        };

        // Sprint with Ctrl (not while sneaking)
        if keys.pressed(KeyCode::ControlLeft) && !player.is_sneaking {
            speed *= dev.sprint_multiplier;
        }

        // Option/Alt key = 25% speed slow walk
        if keys.pressed(KeyCode::AltLeft) || keys.pressed(KeyCode::AltRight) {
            speed *= 0.25;
        }

        // Directional vectors from yaw only (flat movement).
        let forward_3d = forward_from_angles(player.yaw, player.pitch);
        let flat_forward = Vec3::new(forward_3d.x, 0.0, forward_3d.z).normalize_or_zero();
        let right = forward_3d.cross(Vec3::Y).normalize_or_zero();

        // WASD
        if keys.pressed(KeyCode::KeyW) {
            transform.translation += flat_forward * speed * dt;
        }
        if keys.pressed(KeyCode::KeyS) {
            transform.translation -= flat_forward * speed * dt;
        }
        if keys.pressed(KeyCode::KeyA) {
            transform.translation -= right * speed * dt;
        }
        if keys.pressed(KeyCode::KeyD) {
            transform.translation += right * speed * dt;
        }

        // Edge-guard: prevent falling off edges while sneaking.
        // Only active when grounded + sneaking (shift held). Checks the 4
        // cardinal edges of the player's footprint for air below. If an edge
        // has no ground support, pushes the player back toward solid ground.
        // LIMITATION: does not check diagonal corners — the player can still
        // slip off exact corner edges. This matches Minecraft's sneak behavior.
        if player.is_sneaking && player.is_on_ground {
            let foot_y = (transform.translation.y - player.eye_height).floor() as i32 - 1;
            let cm = chunk_manager.as_deref();
            // Check if there's ground below each edge of the player's footprint
            let check_offsets = [
                Vec3::new(PLAYER_RADIUS, 0.0, 0.0),
                Vec3::new(-PLAYER_RADIUS, 0.0, 0.0),
                Vec3::new(0.0, 0.0, PLAYER_RADIUS),
                Vec3::new(0.0, 0.0, -PLAYER_RADIUS),
            ];
            for offset in &check_offsets {
                let check_pos = transform.translation + *offset;
                let bx = check_pos.x.floor() as i32;
                let bz = check_pos.z.floor() as i32;
                let block_below = block_at_world(bx, foot_y, bz, cm);
                if block_below == crate::block_types::BlockType::AIR {
                    // No ground here — push back from this edge
                    let block_center_x = bx as f32 + 0.5;
                    let block_center_z = bz as f32 + 0.5;
                    let dx = transform.translation.x - block_center_x;
                    let dz = transform.translation.z - block_center_z;
                    // Push the player back toward solid ground
                    if dx.abs() > dz.abs() {
                        if dx > 0.0 {
                            transform.translation.x = transform.translation.x.max(bx as f32 + 1.0 + PLAYER_RADIUS);
                        } else {
                            transform.translation.x = transform.translation.x.min(bx as f32 - PLAYER_RADIUS);
                        }
                    } else if dz > 0.0 {
                        transform.translation.z = transform.translation.z.max(bz as f32 + 1.0 + PLAYER_RADIUS);
                    } else {
                        transform.translation.z = transform.translation.z.min(bz as f32 - PLAYER_RADIUS);
                    }
                }
            }
        }

        // Arrow key turning
        let kb_turn = player.keyboard_sensitivity * dt;
        if keys.pressed(KeyCode::ArrowLeft) {
            player.yaw -= kb_turn;
        }
        if keys.pressed(KeyCode::ArrowRight) {
            player.yaw += kb_turn;
        }
        if keys.pressed(KeyCode::ArrowUp) {
            player.pitch -= kb_turn;
        }
        if keys.pressed(KeyCode::ArrowDown) {
            player.pitch += kb_turn;
        }
        player.pitch = player.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);

        // Jump
        if keys.pressed(KeyCode::Space) && player.is_on_ground {
            player.vertical_velocity = dev.jump_velocity;
            player.is_on_ground = false;
        }

        // Gravity — applied unconditionally every frame. The subsequent
        // player_collision system detects floor contact and zeroes velocity.
        // This "apply then correct" approach (semi-implicit Euler) is simpler
        // than conditional gravity and avoids edge cases at slope transitions.
        // Trade-off: one frame of gravity is applied before grounded detection,
        // but collision snaps position back, so visually the player never sinks.
        player.vertical_velocity += dev.gravity * dt;

        // CONTRACT: terminal velocity — cap downward speed to prevent
        // collision-skipping and keep long falls controllable.
        player.vertical_velocity = player.vertical_velocity.max(TERMINAL_VELOCITY);

        transform.translation.y += player.vertical_velocity * dt;
    }
}

// ---------------------------------------------------------------------------
// System: player_collision — classified surface contact resolution
//
// INVARIANT: after this system runs, the player's position never overlaps
// any solid block AND is never below VOID_KILL_PLANE_Y. Every contact is
// classified via classify_block_contact() before behavior is applied.
//
// Resolution order: Ground → Ceiling → Wall → Void recovery.
// This order matters because Ground sets is_on_ground (used by jump gating
// and edge-guard), Ceiling must run before Wall so vertical capping doesn't
// interfere with horizontal push, and void recovery is the last-resort
// safety net after all other resolution has been attempted.
//
// Each resolution phase only applies behavior matching its SurfaceType:
//   Ground  → snap Y, zero velocity, set is_on_ground
//   Ceiling → cap Y, zero upward velocity
//   Wall    → push X/Z only
//   Void    → teleport to recovery point (bed or spawn)
// ---------------------------------------------------------------------------

fn player_collision(
    chunk_manager: Option<Res<ChunkManager>>,
    mut query: Query<(&mut Transform, &mut Player)>,
) {
    for (mut transform, mut player) in &mut query {
        let pos = transform.translation;

        // --- SurfaceType::Ground ---
        let ground = find_ground_height(pos, player.eye_height, chunk_manager.as_deref());
        if pos.y <= ground {
            transform.translation.y = ground;
            player.vertical_velocity = 0.0;
            player.is_on_ground = true;
        } else {
            // No Ground contact — gravity continues (CONTRACT: gravity
            // only stops on SurfaceType::Ground).
            player.is_on_ground = false;
        }

        // --- SurfaceType::Ceiling ---
        let ceiling = find_ceiling_height(
            transform.translation,
            player.eye_height,
            chunk_manager.as_deref(),
        );
        if ceiling < f32::INFINITY && transform.translation.y >= ceiling {
            transform.translation.y = ceiling - 0.01;
            if player.vertical_velocity > 0.0 {
                player.vertical_velocity = 0.0;
            }
        }

        // --- SurfaceType::Wall ---
        // CONTRACT: only modifies X/Z, never Y or vertical velocity.
        resolve_wall_contacts(
            &mut transform.translation,
            player.eye_height,
            PLAYER_WALL_RADIUS,
            chunk_manager.as_deref(),
        );

        // --- Void recovery ---
        // CONTRACT: the player can never be permanently lost below the world.
        // If they fall below the kill plane (below all possible bedrock),
        // teleport them to a safe recovery point. This is the last-resort
        // safety net — normal gameplay should never reach here.
        if transform.translation.y < VOID_KILL_PLANE_Y {
            let recovery = if let Some(home) = player.home_position {
                // Bed exists — return to bed (home_position is already set
                // to the correct camera Y above the pillow).
                home
            } else {
                // No bed — return to world origin, above terrain.
                let ground = terrain::terrain_height_at(0.0, 0.0);
                Vec3::new(0.0, ground + 1.0 + player.eye_height, 0.0)
            };
            transform.translation = recovery;
            player.vertical_velocity = 0.0;
            player.is_on_ground = false; // let ground snap resolve on next frame
            info!(
                "Void recovery: player fell below Y={}, teleported to ({:.0}, {:.0}, {:.0})",
                VOID_KILL_PLANE_Y, recovery.x, recovery.y, recovery.z
            );
        }
    }
}

// ---------------------------------------------------------------------------
// System: apply_player_transform — sync rotation from yaw/pitch
//
// Roll is always 0.0, guaranteeing the horizon stays level (no camera tilt).
// YXZ order: yaw (Y) applied first, then pitch (X). This matches standard
// FPS camera conventions and avoids gimbal lock at the poles (pitch is
// clamped to ±PITCH_LIMIT, which is 0.01 rad short of ±π/2).
// ---------------------------------------------------------------------------

fn apply_player_transform(mut query: Query<(&mut Transform, &Player)>) {
    for (mut transform, player) in &mut query {
        transform.rotation = Quat::from_euler(EulerRot::YXZ, player.yaw, player.pitch, 0.0);
    }
}

// ---------------------------------------------------------------------------
// System: block_select — number keys 1-9 change selected block
// ---------------------------------------------------------------------------

fn block_select(
    keys: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut Player>,
) {
    // Bedrock is intentionally excluded — it is unbreakable, so allowing
    // placement would let the player create inescapable enclosures (softlock).
    // See NO-SOFTLOCK INVARIANT in teleport_home.
    let mappings: [(KeyCode, BlockType); 10] = [
        (KeyCode::Digit1, BlockType::GRASS),
        (KeyCode::Digit2, BlockType::DIRT),
        (KeyCode::Digit3, BlockType::STONE),
        (KeyCode::Digit4, BlockType::SAND),
        (KeyCode::Digit5, BlockType::WOOD),
        (KeyCode::Digit6, BlockType::DIAMOND),
        (KeyCode::Digit7, BlockType::LEAVES),
        (KeyCode::Digit8, BlockType::LANTERN),
        (KeyCode::Digit9, BlockType::BED),
        (KeyCode::Digit0, BlockType::STONE_BRICK),
    ];

    for mut player in &mut query {
        for (key, block) in &mappings {
            if keys.just_pressed(*key) {
                player.selected_block = *block;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// System: block_interact — left/right click to break/place blocks
// ---------------------------------------------------------------------------

fn block_interact(
    mouse: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    mut interaction: ResMut<InteractionState>,
    mut chunk_manager: Option<ResMut<ChunkManager>>,
    dev: Res<DevSettings>,
    mut query: Query<(&Transform, &mut Player)>,
) {
    let dt = time.delta_secs();
    let Some(ref mut cm) = chunk_manager else {
        return;
    };

    // Track held state
    interaction.is_left_held = mouse.pressed(MouseButton::Left);
    interaction.is_right_held = mouse.pressed(MouseButton::Right);

    // Reset break count when left is released
    if mouse.just_released(MouseButton::Left) {
        interaction.break_count = 0;
    }

    // Update timers
    interaction.break_timer += dt;
    interaction.place_timer += dt;

    for (transform, mut player) in &mut query {
        let origin = transform.translation;
        let direction = forward_from_angles(player.yaw, player.pitch);

        // --- Left click: break block ---
        let should_break = mouse.just_pressed(MouseButton::Left)
            || (interaction.is_left_held
                && interaction.break_timer >= dev.break_interval
                && interaction.break_count < MAX_CONTINUOUS_BREAKS);

        if should_break {
            if let Some(hit) = cast_ray(origin, direction, dev.reach, cm) {
                if cm.block_at(hit.pos) != BlockType::BEDROCK {
                    cm.set_block(hit.pos, BlockType::AIR);
                    interaction.break_timer = 0.0;
                    interaction.break_count += 1;
                }
            }
        }

        // --- Right click: place block ---
        let should_place = mouse.just_pressed(MouseButton::Right)
            || (interaction.is_right_held
                && interaction.place_timer >= dev.place_interval
                && player.selected_block != BlockType::BED);

        if should_place {
            if let Some(hit) = cast_ray(origin, direction, dev.reach, cm) {
                let place_pos = hit.pos + hit.normal;

                // Bedrock placement is blocked to prevent softlocks — the
                // player cannot break bedrock, so placing it would create
                // inescapable enclosures. See NO-SOFTLOCK INVARIANT.
                if player.selected_block == BlockType::BEDROCK {
                    // silently reject
                } else if player.selected_block == BlockType::BED {
                    // Place a 3x2 bed structure oriented based on player facing
                    place_bed(&mut player, place_pos, cm);
                } else {
                    // Check the block is not inside the player cylinder
                    if !block_overlaps_player(place_pos, origin, player.eye_height) {
                        cm.set_block(place_pos, player.selected_block);
                    }
                }
                interaction.place_timer = 0.0;
            }
        }
    }
}

/// Place a 3x2 bed structure. The bed is 3 blocks long (foot to head),
/// 2 blocks wide, oriented in the direction the player is facing.
/// The pillow blocks are at the head end.
fn place_bed(player: &mut Player, base_pos: IVec3, cm: &mut ChunkManager) {
    // Determine facing direction (snap to cardinal)
    let (sin_yaw, cos_yaw) = player.yaw.sin_cos();
    let (dir_x, dir_z) = if sin_yaw.abs() > cos_yaw.abs() {
        // Facing along X axis
        if sin_yaw > 0.0 {
            (1, 0)
        } else {
            (-1, 0)
        }
    } else {
        // Facing along Z axis
        if cos_yaw > 0.0 {
            (0, 1)
        } else {
            (0, -1)
        }
    };

    // Perpendicular direction for width
    let (perp_x, perp_z) = (-dir_z, dir_x);

    // Place 3 rows deep (0=foot, 1=middle, 2=head/pillow) x 2 wide
    for depth in 0..3 {
        for width in 0..2 {
            let x = base_pos.x + dir_x * depth + perp_x * width;
            let y = base_pos.y;
            let z = base_pos.z + dir_z * depth + perp_z * width;
            let pos = IVec3::new(x, y, z);

            if depth == 2 {
                cm.set_block(pos, BlockType::PILLOW);
            } else {
                cm.set_block(pos, BlockType::BED);
            }
        }
    }

    // Set home position above the center of the pillow row
    let pillow_center_x = base_pos.x as f32 + dir_x as f32 * 2.0 + perp_x as f32 * 0.5;
    let pillow_center_z = base_pos.z as f32 + dir_z as f32 * 2.0 + perp_z as f32 * 0.5;
    player.home_position = Some(Vec3::new(
        pillow_center_x,
        base_pos.y as f32 + 1.0 + player.eye_height,
        pillow_center_z,
    ));
}

/// Check if placing a block at `block_pos` would overlap the player's collision cylinder.
/// Prevents the player from entombing themselves by placing a block inside their body.
/// Uses the same cylinder model as wall collision (PLAYER_RADIUS wide, foot_y to head_y).
fn block_overlaps_player(block_pos: IVec3, player_pos: Vec3, eye_height: f32) -> bool {
    let foot_y = player_pos.y - eye_height;
    let head_y = player_pos.y;
    let block_y = block_pos.y as f32;

    // Vertical overlap check
    if block_y + 1.0 <= foot_y || block_y >= head_y {
        return false;
    }

    // Horizontal cylinder overlap check
    let bx = block_pos.x as f32;
    let bz = block_pos.z as f32;
    let near_x = bx.max((bx + 1.0).min(player_pos.x));
    let near_z = bz.max((bz + 1.0).min(player_pos.z));
    let dx = player_pos.x - near_x;
    let dz = player_pos.z - near_z;
    dx * dx + dz * dz < PLAYER_RADIUS * PLAYER_RADIUS
}

// ---------------------------------------------------------------------------
// System: teleport_home — H key teleports to safety
//
// CONTRACT — NO-SOFTLOCK INVARIANT:
//   The player can ALWAYS escape any position by pressing H. This is the
//   primary anti-softlock mechanism and must never be gated on a condition
//   the player cannot satisfy while trapped.
//
//   Priority:
//     1. Bed (home_position) — if the player placed a bed, go there.
//     2. Spawn point — world origin at terrain height. Always valid because
//        terrain generation is deterministic and the origin always has ground.
//
//   This guarantees recovery even if the player has never placed a bed,
//   has sealed themselves in, or has fallen into unloaded chunks.
// ---------------------------------------------------------------------------

fn teleport_home(
    keys: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut Transform, &mut Player)>,
) {
    if !keys.just_pressed(KeyCode::KeyH) {
        return;
    }

    for (mut transform, mut player) in &mut query {
        let destination = if let Some(home) = player.home_position {
            home
        } else {
            // No bed — fall back to world origin above terrain.
            let ground = terrain::terrain_height_at(0.0, 0.0);
            Vec3::new(0.0, ground + 1.0 + player.eye_height, 0.0)
        };
        transform.translation = destination;
        player.vertical_velocity = 0.0;
    }
}

// ---------------------------------------------------------------------------
// System: update_selected_block_text — keep HUD text in sync
// ---------------------------------------------------------------------------

fn update_selected_block_text(
    query: Query<&Player>,
    mut text_query: Query<&mut Text, With<SelectedBlockText>>,
    custom_registry: Res<crate::block_types::CustomBlockRegistry>,
) {
    let Some(player) = query.iter().next() else {
        return;
    };

    let (num, name) = block_display_name(player.selected_block, &custom_registry);

    for mut text in &mut text_query {
        **text = format!("[{}] {}", num, name);
    }
}

/// Returns the hotbar number and display name for a block type.
/// Checks the custom block registry for runtime-added blocks.
fn block_display_name(
    block: BlockType,
    registry: &crate::block_types::CustomBlockRegistry,
) -> (u8, String) {
    // Check custom registry first (runtime-added blocks)
    if block.is_custom() {
        if let Some(entry) = registry.get(block.index()) {
            return (0, entry.name.clone());
        }
        return (0, format!("Custom {}", block.index()));
    }

    // Built-in blocks with hotbar numbers
    let (num, name) = match block {
        BlockType::GRASS => (1, "Grass"),
        BlockType::DIRT => (2, "Dirt"),
        BlockType::STONE => (3, "Stone"),
        BlockType::SAND => (4, "Sand"),
        BlockType::WOOD => (5, "Wood"),
        BlockType::DIAMOND => (6, "Diamond"),
        BlockType::LEAVES => (7, "Leaves"),
        BlockType::LANTERN => (8, "Lantern"),
        BlockType::BED => (9, "Bed"),
        BlockType::STONE_BRICK => (0, "Stone Brick"),
        _ => (0, "Air"),
    };
    (num, name.to_string())
}

// ---------------------------------------------------------------------------
// Helper: forward vector from yaw + pitch (matches Swift Camera.forward)
// ---------------------------------------------------------------------------

/// Camera forward direction in Bevy's coordinate system (-Z is forward).
fn forward_from_angles(yaw: f32, pitch: f32) -> Vec3 {
    Vec3::new(
        -(pitch.cos() * yaw.sin()),
        pitch.sin(),
        -(pitch.cos() * yaw.cos()),
    )
}

// ---------------------------------------------------------------------------
// Collision: find_ground_height — SurfaceType::Ground detection
//
// Scans for the highest solid block classified as Ground within the player's
// collision cylinder. Returns the camera Y at which the player's feet would
// rest on that block, or NEG_INFINITY if no ground exists (freefall/void).
//
// Only blocks classified as SurfaceType::Ground (block_top less than 1 full
// block above foot_y) are considered. Blocks at or above the 1-block
// threshold are Wall and handled separately.
//
// WALKABILITY INVARIANT: every surface below a 1-block step is fully
// reachable. There is no half-affordance gap — if geometry is visible
// and less than 1 block above the player's feet, it is walkable.
//
// Scan range: 20 blocks above and 100 blocks below the terrain surface_y.
// ---------------------------------------------------------------------------

fn find_ground_height(cam_pos: Vec3, eye_height: f32, cm: Option<&ChunkManager>) -> f32 {
    let foot_y = cam_pos.y - eye_height;
    let head_y = cam_pos.y;
    let mut best = f32::NEG_INFINITY;

    let cx = cam_pos.x.floor() as i32;
    let cz = cam_pos.z.floor() as i32;
    let foot_block = foot_y.floor() as i32;

    for dx in -1..=1 {
        for dz in -1..=1 {
            let bx = cx + dx;
            let bz = cz + dz;
            let fbx = bx as f32;
            let fbz = bz as f32;

            // Skip columns outside the player's horizontal cylinder
            let near_x = fbx.max((fbx + 1.0).min(cam_pos.x));
            let near_z = fbz.max((fbz + 1.0).min(cam_pos.z));
            let ddx = cam_pos.x - near_x;
            let ddz = cam_pos.z - near_z;
            if ddx * ddx + ddz * ddz >= PLAYER_RADIUS * PLAYER_RADIUS {
                continue;
            }

            let sy = terrain::surface_y(bx, bz);
            let scan_top = (foot_block + 1).min(sy + 20);
            let scan_bottom = (foot_block - 20).max(sy - 100);

            // Scan downward from foot level to find the first solid block
            let mut by = scan_top;
            while by >= scan_bottom {
                let block = block_at_world(bx, by, bz, cm);
                if block != crate::block_types::BlockType::AIR {
                    // Classify before applying Ground behavior
                    if classify_block_contact(by, foot_y, head_y) == SurfaceType::Ground {
                        let block_top = (by as f32) + 1.0;
                        best = best.max(block_top + eye_height);
                        break;
                    }
                }
                by -= 1;
            }
        }
    }

    best
}

// ---------------------------------------------------------------------------
// Collision: find_ceiling_height — SurfaceType::Ceiling detection
//
// Returns the Y of the lowest solid block classified as Ceiling within the
// player's collision cylinder. Returns INFINITY if no ceiling exists.
// Scans 4 blocks above head level — enough for jump clearance checks.
// Only blocks classified as SurfaceType::Ceiling are considered.
// ---------------------------------------------------------------------------

fn find_ceiling_height(cam_pos: Vec3, eye_height: f32, cm: Option<&ChunkManager>) -> f32 {
    let foot_y = cam_pos.y - eye_height;
    let head_y = cam_pos.y;
    let mut lowest = f32::INFINITY;

    let cx = cam_pos.x.floor() as i32;
    let cz = cam_pos.z.floor() as i32;
    let head_block = cam_pos.y.floor() as i32;

    for dx in -1..=1 {
        for dz in -1..=1 {
            let bx = cx + dx;
            let bz = cz + dz;

            for dy in 0..=3 {
                let by = head_block + dy;
                let block = block_at_world(bx, by, bz, cm);
                if block == crate::block_types::BlockType::AIR {
                    continue;
                }

                // Classify before applying Ceiling behavior
                if classify_block_contact(by, foot_y, head_y) != SurfaceType::Ceiling {
                    continue;
                }

                let fbx = bx as f32;
                let fbz = bz as f32;
                let near_x = fbx.max((fbx + 1.0).min(cam_pos.x));
                let near_z = fbz.max((fbz + 1.0).min(cam_pos.z));
                let ddx = cam_pos.x - near_x;
                let ddz = cam_pos.z - near_z;
                if ddx * ddx + ddz * ddz < PLAYER_RADIUS * PLAYER_RADIUS {
                    lowest = lowest.min(by as f32);
                }
            }
        }
    }

    lowest
}

// ---------------------------------------------------------------------------
// Collision: resolve_wall_contacts — SurfaceType::Wall resolution
//
// Pushes the player horizontally out of any solid blocks classified as
// Wall (or Ground) that overlap their collision cylinder.
//
// CONTRACT: only modifies X/Z position, never Y — walls block horizontal
// movement only. This is the defining behavioral constraint of SurfaceType::Wall.
//
// WHY Ground blocks are included: when the player approaches a step-height
// block from the side, classify returns Ground (it's steppable from above).
// But horizontally the player is pressing into its face, so wall push-out
// must apply. Once find_ground_height snaps the player on top of the block,
// the vertical overlap check (foot_y < block_top && head_y > block_bottom)
// naturally excludes it from future wall processing.
//
// Ceiling blocks are excluded because they sit above head_y and the
// vertical overlap check filters them out.
// ---------------------------------------------------------------------------

fn resolve_wall_contacts(
    position: &mut Vec3,
    eye_height: f32,
    radius: f32,
    cm: Option<&ChunkManager>,
) {
    let foot_y = position.y - eye_height;
    let head_y = position.y;
    let cx = position.x.floor() as i32;
    let cz = position.z.floor() as i32;
    let foot_block = foot_y.floor() as i32;

    for dx in -1..=1 {
        for dz in -1..=1 {
            let bx = cx + dx;
            let bz = cz + dz;

            for dy in 0..=3 {
                let by = foot_block + dy;
                let block = block_at_world(bx, by, bz, cm);
                if block == crate::block_types::BlockType::AIR {
                    continue;
                }

                // Classify: only process Wall and Ground contacts.
                // Ceiling contacts are above the head and excluded.
                let surface = classify_block_contact(by, foot_y, head_y);
                if surface == SurfaceType::Ceiling {
                    continue;
                }

                let fbx = bx as f32;
                let fby = by as f32;
                let fbz = bz as f32;

                // Must overlap vertically (this naturally filters Ground
                // blocks the player is already standing on, since their
                // block_top == foot_y after ground snap).
                if !(foot_y < fby + 1.0 && head_y > fby) {
                    continue;
                }
                // Broad AABB check
                if (position.x - fbx - 0.5).abs() >= radius + 0.5
                    || (position.z - fbz - 0.5).abs() >= radius + 0.5
                {
                    continue;
                }

                // Cylinder push-out (X/Z only — CONTRACT: Wall never modifies Y)
                let near_x = fbx.max((fbx + 1.0).min(position.x));
                let near_z = fbz.max((fbz + 1.0).min(position.z));
                let ddx = position.x - near_x;
                let ddz = position.z - near_z;
                let dist_sq = ddx * ddx + ddz * ddz;
                if dist_sq >= radius * radius || dist_sq <= 0.00001 {
                    continue;
                }
                let dist = dist_sq.sqrt();
                let push = radius - dist;
                position.x += ddx / dist * push;
                position.z += ddz / dist * push;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: read a block, falling back to terrain generation when ChunkManager
// is unavailable or the chunk is not loaded.
// ---------------------------------------------------------------------------

fn block_at_world(
    x: i32,
    y: i32,
    z: i32,
    cm: Option<&ChunkManager>,
) -> crate::block_types::BlockType {
    if let Some(manager) = cm {
        return manager.block_at(IVec3::new(x, y, z));
    }
    // ChunkManager not yet available — fall back to terrain generation so
    // collision still works before chunks finish loading.
    crate::terrain::natural_block_at(x, y, z)
}
