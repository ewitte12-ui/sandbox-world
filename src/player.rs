//! Player: first-person camera, movement, voxel collision, block interaction.
//!
//! The collision model is a vertical cylinder resolved against the voxel grid,
//! with each nearby solid block classified into exactly one contact role per
//! frame. That classification is the heart of `guardrails/06_movement_contact_
//! contract.txt` — ground, wall and ceiling are genuinely different things and
//! are never collapsed into a single "solid" test.

use std::f32::consts::PI;

use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use crate::block_types::BlockType;
use crate::chunk_manager::ChunkManager;
use crate::dev_tools::DevSettings;
use crate::ray_cast::cast_ray;
use crate::terrain;
use crate::{GameState, WorldEntity, WorldInstanceId, WorldScoped};

// ---------------------------------------------------------------------------
// Geometry and physics constraints.
//
// SINGLE SOURCE OF TRUTH: every gameplay-tunable value (speed, gravity, jump
// velocity, reach, interaction intervals) lives in `DevSettings`. The constants
// here are geometric or physical invariants that collision correctness depends
// on — they are not knobs.
// ---------------------------------------------------------------------------

/// Half-width of the player cylinder. Slightly under half a block so a 1-wide
/// corridor is passable with margin for float error.
const PLAYER_RADIUS: f32 = 0.4;

/// Wider radius used to keep the *camera* out of walls while the body still
/// fits through gaps. Without this the near plane clips into geometry.
const PLAYER_WALL_RADIUS: f32 = 0.5;

/// Just under a right angle, to stay off the YXZ Euler gimbal singularity.
const PITCH_LIMIT: f32 = PI / 2.0 - 0.01;

/// Eye-height interpolation rate when crouching, in blocks/second.
const CROUCH_TRANSITION_SPEED: f32 = 10.0;

/// WALKABILITY INVARIANT:
///   step = block_top - foot_y
///   step <  STEP_UP_THRESHOLD  →  Ground (walk up, no jump)
///   step >= STEP_UP_THRESHOLD  →  Wall   (must jump)
///
/// One block minus a float-safe margin. The rule is deliberately total: there
/// is no height that is neither walkable nor jumpable, which is what
/// `guardrails/02` means by "no half-affordances". The margin also keeps the
/// classification stable at low framerates, where gravity can drop `foot_y`
/// well below the block top before the ground snap runs.
const STEP_UP_THRESHOLD: f32 = 1.0 - 0.02;

/// Keeps ceiling detection from catching on the block the player is already
/// intersecting through float imprecision.
const CEILING_EPSILON: f32 = 0.05;

/// VERTICALITY INVARIANT: capped downward speed. This is a collision-safety
/// limit, not game feel — too fast and a frame can step straight through a
/// block. Do not treat it as a tuning value.
const TERMINAL_VELOCITY: f32 = -60.0;

/// VERTICALITY INVARIANT / NO-SOFTLOCK: falling below this triggers recovery.
/// A downward mistake must always cost time, never become permanent.
const VOID_KILL_PLANE_Y: f32 = -80.0;

/// Held-click mining stops after this many blocks, so a stuck button cannot
/// excavate a tunnel.
const MAX_CONTINUOUS_BREAKS: u32 = 5;

// ---------------------------------------------------------------------------
// Components and resources
// ---------------------------------------------------------------------------

/// The player. Required components pull in the camera rig, so spawning a
/// `Player` is enough to get a working first-person view.
#[derive(Component)]
#[require(Camera3d, Transform)]
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
    /// Recovery point set by sleeping in a bed. `None` falls back to spawn.
    pub home_position: Option<Vec3>,
    /// Foot height before this frame's vertical integration, so collision can
    /// sweep the path actually travelled rather than probing a fixed depth.
    prev_foot_y: f32,
}

impl Default for Player {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            // Slight downward tilt so terrain is in view on spawn.
            pitch: -0.1,
            eye_height: 1.8,
            standing_eye_height: 1.8,
            crouch_eye_height: 1.2,
            vertical_velocity: 0.0,
            is_on_ground: false,
            is_sneaking: false,
            selected_block: BlockType::STONE,
            home_position: None,
            prev_foot_y: f32::MAX,
        }
    }
}

/// Gates physics until the terrain under the player exists. Stepping before
/// then drops the player through a world that has not streamed in yet.
#[derive(Resource, Default)]
pub struct WorldReady(pub bool);

/// The block currently under the crosshair, refreshed every frame.
#[derive(Resource, Default)]
pub struct SelectedBlock(pub Option<TargetedBlock>);

#[derive(Clone, Copy)]
pub struct TargetedBlock {
    /// The solid block being looked at.
    pub block: IVec3,
    /// Empty cell adjacent to the hit face — where a placed block would go.
    pub place_at: IVec3,
    pub kind: BlockType,
}

/// Click-hold bookkeeping for continuous break/place.
#[derive(Resource, Default)]
pub struct InteractionState {
    break_cooldown: f32,
    place_cooldown: f32,
    continuous_breaks: u32,
}

#[derive(Component)]
struct Crosshair;

/// A solid block's role relative to the player this frame.
///
/// The same block is Ground when approached from above and Wall when
/// approached from the side; the role describes the current contact, not a
/// property of the block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SurfaceType {
    /// Top is under a full step above the feet: gravity stops, feet snap up.
    Ground,
    /// Overlaps the body vertically: blocks horizontal movement, never Y.
    Wall,
    /// Bottom is at or above the head: caps upward motion, gravity continues.
    Ceiling,
}

fn classify_contact(block_y: i32, foot_y: f32, head_y: f32) -> SurfaceType {
    let block_top = block_y as f32 + 1.0;
    let block_bottom = block_y as f32;

    if block_top <= foot_y + STEP_UP_THRESHOLD {
        SurfaceType::Ground
    } else if block_bottom >= head_y - CEILING_EPSILON {
        SurfaceType::Ceiling
    } else {
        SurfaceType::Wall
    }
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<WorldReady>()
            .init_resource::<SelectedBlock>()
            .init_resource::<InteractionState>()
            .add_systems(
                OnEnter(GameState::Gameplay),
                (spawn_player, spawn_crosshair),
            )
            // SESSION RESET: both resources outlive the player entity. A stale
            // `WorldReady(true)` would let the next session's physics run
            // before its terrain exists, dropping the player through the floor.
            .add_systems(
                OnExit(GameState::Gameplay),
                (reset_for_new_world, release_cursor),
            )
            .add_systems(
                Update,
                (
                    grab_cursor,
                    check_world_ready,
                    player_look,
                    // Movement is a strict pipeline: propose a motion, resolve
                    // it against the world, then derive the camera from the
                    // resolved position. Running these out of order is what
                    // produces the camera jitter `guardrails/04` forbids.
                    (
                        player_move,
                        player_collision,
                        update_eye_height,
                        void_recovery,
                    )
                        .chain(),
                    (block_select, block_interact).chain(),
                )
                    .run_if(in_state(GameState::Gameplay)),
            );
    }
}

/// Picks a spawn column with room to stand, searching outward from the origin.
///
/// The origin is not automatically safe: tree placement is independent of
/// terrain height, so the world can and does grow a trunk right through
/// (0, 0). Spawning inside one leaves the player embedded in geometry, and the
/// wall resolver then ejects them sideways on the first frame — a visible lurch
/// before the player has touched the controls.
fn find_spawn() -> Vec3 {
    const MAX_RING: i32 = 24;
    // Comfortably clear of a trunk, scaled by the tree's own size.
    const TREE_CLEARANCE: f32 = 2.5;

    for ring in 0..MAX_RING {
        for dx in -ring..=ring {
            for dz in -ring..=ring {
                // Only the perimeter of each ring is new.
                if ring > 0 && dx.abs() != ring && dz.abs() != ring {
                    continue;
                }
                // Block centre, so the player is not balanced on a seam.
                let (x, z) = (dx as f32 + 0.5, dz as f32 + 0.5);
                if terrain::is_clear_of_trees(x, z, TREE_CLEARANCE) {
                    return Vec3::new(x, terrain::surface_y(dx, dz) as f32 + 1.0, z);
                }
            }
        }
    }
    // Every column within range is wooded; the wall resolver will sort it out.
    Vec3::new(0.5, terrain::surface_y(0, 0) as f32 + 1.0, 0.5)
}

pub(crate) fn spawn_player(mut commands: Commands, world_id: Res<WorldInstanceId>) {
    let feet = find_spawn();
    let player = Player::default();
    info!("player spawn: feet at {feet:?}");

    commands.spawn((
        Transform::from_translation(feet + Vec3::Y * player.eye_height),
        player,
        WorldEntity,
        WorldScoped(world_id.0),
    ));
}

fn reset_for_new_world(mut ready: ResMut<WorldReady>, mut selected: ResMut<SelectedBlock>) {
    ready.0 = false;
    selected.0 = None;
}

fn spawn_crosshair(mut commands: Commands) {
    commands.spawn((
        Crosshair,
        Node {
            position_type: PositionType::Absolute,
            left: Val::Percent(50.0),
            top: Val::Percent(50.0),
            width: Val::Px(4.0),
            height: Val::Px(4.0),
            margin: UiRect::all(Val::Px(-2.0)),
            // 0.19 folds border radius into Node rather than a sibling component.
            border_radius: BorderRadius::all(Val::Px(2.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.85)),
    ));
}

fn grab_cursor(mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>) {
    if cursor.grab_mode != CursorGrabMode::Locked {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }
}

/// Hands the cursor back on the way out of gameplay.
///
/// This is the counterpart `grab_cursor` needs: that system only ever *locks*,
/// so without a release the pointer stays pinned to the window centre and
/// invisible once Escape opens the menu. Nothing crashes and the menu draws
/// perfectly — but `bevy_ui_widgets` are pointer-driven, so with no pointer to
/// move there is nothing to hover or click, and a fully live menu is
/// indistinguishable from a frozen game.
///
/// It belongs on the state exit rather than on menu entry because cursor policy
/// is this plugin's, and `OnExit(Gameplay)` covers every way out of the world,
/// not just the Escape key.
fn release_cursor(mut cursor: Single<&mut CursorOptions, With<PrimaryWindow>>) {
    cursor.grab_mode = CursorGrabMode::None;
    cursor.visible = true;
}

/// Physics waits for the ground beneath the player to actually exist.
fn check_world_ready(
    mut ready: ResMut<WorldReady>,
    chunks: Res<ChunkManager>,
    player: Option<Single<&Transform, With<Player>>>,
) {
    if ready.0 {
        return;
    }
    let Some(transform) = player else {
        return;
    };
    // A loaded column under the feet is the real precondition — not a timer,
    // and not "chunks are idle", which is also true before streaming starts.
    let below = transform.translation.floor().as_ivec3() - IVec3::Y;
    if chunks.block_at(below) != BlockType::AIR || chunks.streaming_idle() {
        ready.0 = true;
    }
}

fn player_look(
    mouse: Res<AccumulatedMouseMotion>,
    dev: Res<DevSettings>,
    mut player: Single<(&mut Transform, &mut Player)>,
) {
    let (ref mut transform, ref mut player) = *player;

    if mouse.delta != Vec2::ZERO {
        player.yaw -= mouse.delta.x * dev.mouse_sensitivity;
        player.pitch =
            (player.pitch - mouse.delta.y * dev.mouse_sensitivity).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    transform.rotation = Quat::from_euler(EulerRot::YXZ, player.yaw, player.pitch, 0.0);
}

/// Applies input and gravity, producing a proposed position. Resolution against
/// the world is `player_collision`'s job — this system never inspects blocks.
fn player_move(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    dev: Res<DevSettings>,
    ready: Res<WorldReady>,
    mut player: Single<(&mut Transform, &mut Player)>,
) {
    let (ref mut transform, ref mut player) = *player;
    let dt = time.delta_secs();

    // Horizontal basis from yaw only: looking up must not slow you down.
    let (sin_yaw, cos_yaw) = player.yaw.sin_cos();
    let forward = Vec3::new(-sin_yaw, 0.0, -cos_yaw);
    let right = Vec3::new(cos_yaw, 0.0, -sin_yaw);

    let mut wish = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        wish += forward;
    }
    if keys.pressed(KeyCode::KeyS) {
        wish -= forward;
    }
    if keys.pressed(KeyCode::KeyD) {
        wish += right;
    }
    if keys.pressed(KeyCode::KeyA) {
        wish -= right;
    }

    player.is_sneaking = keys.pressed(KeyCode::ShiftLeft);
    let speed = dev.player_speed
        * if player.is_sneaking {
            dev.sneak_multiplier
        } else if keys.pressed(KeyCode::ControlLeft) {
            dev.sprint_multiplier
        } else {
            1.0
        };

    if wish != Vec3::ZERO {
        transform.translation += wish.normalize() * speed * dt;
    }

    if !ready.0 {
        // Hold position until the ground exists, or the player falls through a
        // world that has not streamed in yet.
        player.vertical_velocity = 0.0;
        return;
    }

    if keys.just_pressed(KeyCode::Space) && player.is_on_ground {
        player.vertical_velocity = dev.jump_velocity;
        player.is_on_ground = false;
    }

    // Remembered before the step so collision can sweep the whole path. A
    // single frame can cover a lot of ground: Bevy clamps the frame delta to
    // 0.25s, which at terminal velocity is still ~15 blocks, and an asset-load
    // hitch reaches that clamp easily.
    player.prev_foot_y = transform.translation.y - player.eye_height;

    player.vertical_velocity = (player.vertical_velocity + dev.gravity * dt).max(TERMINAL_VELOCITY);
    transform.translation.y += player.vertical_velocity * dt;
}

/// Resolves the proposed position against the voxel grid.
///
/// Vertical and horizontal are resolved separately and in that order, because
/// the contact classification depends on foot height: settling onto the ground
/// first means a block that would have been a Wall mid-fall is correctly seen
/// as Ground once standing on it.
fn player_collision(
    chunks: Res<ChunkManager>,
    ready: Res<WorldReady>,
    mut player: Single<(&mut Transform, &mut Player)>,
) {
    if !ready.0 {
        return;
    }
    let (ref mut transform, ref mut player) = *player;

    let foot_y = transform.translation.y - player.eye_height;
    let head_y = transform.translation.y;

    // --- Vertical ---
    let feet = Vec3::new(transform.translation.x, foot_y, transform.translation.z);

    if player.vertical_velocity <= 0.0 {
        // `highest_support` only reports a surface the feet actually reached or
        // passed through this frame, so a hit always means "landed".
        if let Some(ground_top) = highest_support(feet, player.prev_foot_y, &chunks) {
            transform.translation.y = ground_top + player.eye_height;
            player.vertical_velocity = 0.0;
            player.is_on_ground = true;
        } else {
            player.is_on_ground = false;
        }
    } else if let Some(ceiling_bottom) = lowest_ceiling(feet, head_y, &chunks) {
        // Head-bump: stop rising, keep falling logic intact.
        if head_y >= ceiling_bottom - CEILING_EPSILON {
            transform.translation.y = ceiling_bottom - CEILING_EPSILON;
            player.vertical_velocity = 0.0;
        }
    }

    // --- Horizontal ---
    // Recomputed after the vertical snap so classification sees final feet.
    let foot_y = transform.translation.y - player.eye_height;
    let head_y = transform.translation.y;
    resolve_walls(transform, foot_y, head_y, &chunks);

    // --- Anti-stuck ---
    // NO-SOFTLOCK INVARIANT. If the feet ended up *inside* solid geometry, lift
    // to the top of that block. Wall resolution cannot rescue this case: a
    // player standing at a block's exact centre has no "out" direction, so the
    // push-out declines to guess and they would otherwise stay embedded.
    //
    // Reachable without doing anything wrong — spawning onto a building floor,
    // which the spawn search cannot see because buildings are stamped in during
    // chunk generation. This is a recovery path, not a movement one: normal wall
    // collision stops the player before they can walk into a block.
    let foot_y = transform.translation.y - player.eye_height;
    let inside = IVec3::new(
        transform.translation.x.floor() as i32,
        foot_y.floor() as i32,
        transform.translation.z.floor() as i32,
    );
    if chunks.block_at(inside) != BlockType::AIR {
        transform.translation.y = (inside.y as f32 + 1.0) + player.eye_height;
        player.vertical_velocity = 0.0;
        player.is_on_ground = true;
    }
}

/// Top surface of the highest solid block directly under the player cylinder,
/// searched down a few blocks from the feet.
fn highest_support(feet: Vec3, prev_foot_y: f32, chunks: &ChunkManager) -> Option<f32> {
    // Sweep the whole path the feet travelled this frame, not a fixed probe
    // depth. A fixed probe is a tunnelling bug waiting for the first frame
    // spike — 60 glTF models finishing at once is enough — after which the
    // player passes straight through the floor and ends up inside the terrain.
    let start_y = prev_foot_y.min(f32::MAX).max(feet.y);
    let y_hi = (start_y + STEP_UP_THRESHOLD).floor() as i32;
    let y_lo = feet.y.floor() as i32 - 1;
    // Bound the work after an enormous fall; anything deeper is the void
    // recovery's problem, not collision's.
    let y_lo = y_lo.max(y_hi - 64);

    let mut best: Option<f32> = None;
    for (dx, dz) in corner_offsets(PLAYER_RADIUS) {
        let x = (feet.x + dx).floor() as i32;
        let z = (feet.z + dz).floor() as i32;
        for y in (y_lo..=y_hi).rev() {
            if chunks.block_at(IVec3::new(x, y, z)) == BlockType::AIR {
                continue;
            }
            let top = y as f32 + 1.0;
            // A surface above where the feet began is a wall, not support;
            // keep descending past it rather than snapping up onto it.
            if top <= start_y + STEP_UP_THRESHOLD {
                best = Some(best.map_or(top, |b: f32| b.max(top)));
                break;
            }
        }
    }
    best
}

/// Bottom surface of the lowest solid block above the player's head.
fn lowest_ceiling(feet: Vec3, head_y: f32, chunks: &ChunkManager) -> Option<f32> {
    let mut best: Option<f32> = None;
    for (dx, dz) in corner_offsets(PLAYER_RADIUS) {
        let x = (feet.x + dx).floor() as i32;
        let z = (feet.z + dz).floor() as i32;
        let start = head_y.floor() as i32;
        for y in start..=start + 2 {
            if chunks.block_at(IVec3::new(x, y, z)) != BlockType::AIR {
                let bottom = y as f32;
                if bottom >= head_y - CEILING_EPSILON {
                    best = Some(best.map_or(bottom, |b: f32| b.min(bottom)));
                }
                break;
            }
        }
    }
    best
}

/// Pushes the player out of any block classified as `Wall`, one axis at a time.
///
/// Ground and Ceiling contacts are deliberately ignored here: walls block
/// horizontal movement *only* and must never move the player vertically.
fn resolve_walls(transform: &mut Transform, foot_y: f32, head_y: f32, chunks: &ChunkManager) {
    let r = PLAYER_WALL_RADIUS;

    // Every block column the cylinder could overlap, and every body level.
    let y_range = (foot_y.floor() as i32)..=((head_y - 0.01).floor() as i32);

    for by in y_range {
        for (dx, dz) in corner_offsets(r) {
            let bx = (transform.translation.x + dx).floor() as i32;
            let bz = (transform.translation.z + dz).floor() as i32;

            if chunks.block_at(IVec3::new(bx, by, bz)) == BlockType::AIR {
                continue;
            }
            if classify_contact(by, foot_y, head_y) != SurfaceType::Wall {
                continue;
            }

            // Offset from the block centre, and how far along each axis the
            // player would have to move to clear it.
            let px = transform.translation.x - (bx as f32 + 0.5);
            let pz = transform.translation.z - (bz as f32 + 0.5);
            let overlap_x = 0.5 + r - px.abs();
            let overlap_z = 0.5 + r - pz.abs();

            if overlap_x <= 0.0 || overlap_z <= 0.0 {
                continue;
            }

            // Push out along whichever axis is least penetrated, which keeps
            // sliding along a wall smooth instead of snagging on seams.
            //
            // The sign has to be chosen explicitly rather than with `signum`:
            // for a player standing exactly on a block's centre line the offset
            // is +0.0, and `f32::signum` reports that as +1.0 — silently
            // ejecting them a full block in an arbitrary direction instead of
            // leaving them be. That only arises when the player is embedded in
            // geometry, but "embedded" must not become "teleported".
            let axis_x = overlap_x < overlap_z;
            let offset = if axis_x { px } else { pz };
            if offset == 0.0 {
                // Dead centre: neither direction is more correct than the
                // other. Defer to the next frame rather than guess — the other
                // axis, or the ground snap, will usually resolve it first.
                continue;
            }
            let push = if axis_x { overlap_x } else { overlap_z } * offset.signum();
            if axis_x {
                transform.translation.x += push;
            } else {
                transform.translation.z += push;
            }
        }
    }
}

/// The four cylinder extremes sampled for collision.
fn corner_offsets(r: f32) -> [(f32, f32); 4] {
    [(r, r), (r, -r), (-r, r), (-r, -r)]
}

fn update_eye_height(time: Res<Time>, mut player: Single<&mut Player>) {
    let target = if player.is_sneaking {
        player.crouch_eye_height
    } else {
        player.standing_eye_height
    };
    let delta = CROUCH_TRANSITION_SPEED * time.delta_secs();
    player.eye_height += (target - player.eye_height).clamp(-delta, delta);
}

/// NO-SOFTLOCK INVARIANT: falling out of the world always returns the player
/// somewhere safe. A mistake costs the fall and the respawn, never the session.
fn void_recovery(mut player: Single<(&mut Transform, &mut Player)>) {
    let (ref mut transform, ref mut player) = *player;
    if transform.translation.y > VOID_KILL_PLANE_Y {
        return;
    }

    let recovery = player.home_position.unwrap_or_else(|| {
        let y = terrain::surface_y(0, 0) as f32 + 1.0;
        Vec3::new(0.5, y, 0.5)
    });
    transform.translation = recovery + Vec3::Y * player.eye_height;
    player.vertical_velocity = 0.0;
    player.is_on_ground = false;
    warn!("player fell below the void plane — recovered to {recovery:?}");
}

fn block_select(
    chunks: Res<ChunkManager>,
    dev: Res<DevSettings>,
    player: Single<(&Transform, &Player)>,
    mut selected: ResMut<SelectedBlock>,
) {
    let (transform, _) = *player;
    let hit = cast_ray(
        transform.translation,
        transform.forward().as_vec3(),
        dev.reach,
        &chunks,
    );

    selected.0 = hit.map(|hit| TargetedBlock {
        block: hit.block,
        place_at: hit.block + hit.normal,
        kind: chunks.block_at(hit.block),
    });
}

fn block_interact(
    time: Res<Time>,
    buttons: Res<ButtonInput<MouseButton>>,
    dev: Res<DevSettings>,
    selected: Res<SelectedBlock>,
    mut chunks: ResMut<ChunkManager>,
    mut state: ResMut<InteractionState>,
    player: Single<(&Transform, &Player)>,
) {
    let dt = time.delta_secs();
    state.break_cooldown = (state.break_cooldown - dt).max(0.0);
    state.place_cooldown = (state.place_cooldown - dt).max(0.0);

    if buttons.just_pressed(MouseButton::Left) {
        state.continuous_breaks = 0;
    }
    if !buttons.pressed(MouseButton::Left) {
        state.continuous_breaks = 0;
    }

    let Some(target) = selected.0 else {
        return;
    };
    let (transform, player) = *player;

    // Break
    if buttons.pressed(MouseButton::Left)
        && state.break_cooldown == 0.0
        && state.continuous_breaks < MAX_CONTINUOUS_BREAKS
        && target.kind != BlockType::BEDROCK
    {
        chunks.set_block(target.block, BlockType::AIR);
        state.break_cooldown = dev.break_interval;
        state.continuous_breaks += 1;
    }

    // Place — never inside the player's own cylinder, which would trap them.
    if buttons.pressed(MouseButton::Right) && state.place_cooldown == 0.0 {
        let foot_y = transform.translation.y - player.eye_height;
        if !block_overlaps_player(target.place_at, transform.translation, foot_y)
            && chunks.block_at(target.place_at) == BlockType::AIR
        {
            chunks.set_block(target.place_at, player.selected_block);
            state.place_cooldown = dev.place_interval;
        }
    }
}

/// Same cylinder model as wall collision, so "can I place here" and "would I be
/// stuck" agree by construction.
fn block_overlaps_player(block: IVec3, eye: Vec3, foot_y: f32) -> bool {
    let head_y = eye.y;
    let block_bottom = block.y as f32;
    let block_top = block_bottom + 1.0;

    if block_top <= foot_y || block_bottom >= head_y {
        return false;
    }

    // Closest point on the block footprint to the cylinder axis.
    let cx = eye.x.clamp(block.x as f32, block.x as f32 + 1.0);
    let cz = eye.z.clamp(block.z as f32, block.z as f32 + 1.0);
    let dx = eye.x - cx;
    let dz = eye.z - cz;

    dx * dx + dz * dz < PLAYER_RADIUS * PLAYER_RADIUS
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;

    /// The cursor must come back when gameplay ends. `grab_cursor` only locks,
    /// so a missing release leaves the menu drawn but unusable — the pointer is
    /// pinned to the window centre and hidden, and every pointer-driven widget
    /// is unreachable. The app looks hung when it is merely holding the mouse.
    #[test]
    fn leaving_gameplay_hands_the_cursor_back() {
        let mut world = World::new();
        world.spawn((
            PrimaryWindow,
            CursorOptions {
                grab_mode: CursorGrabMode::Locked,
                visible: false,
                ..default()
            },
        ));

        world.run_system_once(release_cursor).unwrap();

        let cursor = world
            .query_filtered::<&CursorOptions, With<PrimaryWindow>>()
            .single(&world)
            .expect("primary window cursor");
        assert_eq!(cursor.grab_mode, CursorGrabMode::None);
        assert!(cursor.visible, "cursor must be visible in the menu");
    }

    // A player standing with feet at y = 10.0.
    const FOOT_Y: f32 = 10.0;
    const HEAD_Y: f32 = FOOT_Y + 1.8;

    /// WALKABILITY INVARIANT: the classification is total — every solid block
    /// overlapping the player is exactly one of Ground, Wall or Ceiling, and the
    /// boundary between "step up" and "must jump" sits at one full block.
    #[test]
    fn flush_floor_is_ground() {
        // Block spans 9..10, so its top is exactly at the feet.
        assert_eq!(classify_contact(9, FOOT_Y, HEAD_Y), SurfaceType::Ground);
    }

    #[test]
    fn a_full_block_step_is_a_wall_not_a_free_climb() {
        // Top is 1.0 above the feet, i.e. at the threshold — must be jumped.
        assert_eq!(classify_contact(10, FOOT_Y, HEAD_Y), SurfaceType::Wall);
    }

    #[test]
    fn block_overlapping_the_torso_is_a_wall() {
        assert_eq!(classify_contact(11, FOOT_Y, HEAD_Y), SurfaceType::Wall);
    }

    #[test]
    fn block_above_the_head_is_a_ceiling() {
        assert_eq!(classify_contact(12, FOOT_Y, HEAD_Y), SurfaceType::Ceiling);
    }

    /// No half-affordances: sweeping the feet through a block's height must
    /// never produce a gap where the surface is neither walkable nor jumpable.
    #[test]
    fn classification_is_total_across_a_block_height() {
        let mut foot = 10.0_f32;
        while foot < 11.0 {
            let head = foot + 1.8;
            for block_y in 8..=13 {
                // Every call returns one of the three roles; the point is that
                // none of them panics or falls through a gap in the rules.
                let role = classify_contact(block_y, foot, head);
                assert!(matches!(
                    role,
                    SurfaceType::Ground | SurfaceType::Wall | SurfaceType::Ceiling
                ));
            }
            foot += 0.05;
        }
    }

    /// A block sharing the player's own column is the degenerate case for
    /// push-out: the offset from the block centre is exactly zero, so there is
    /// no "away" direction. `f32::signum` reports +0.0 as +1.0, which would
    /// silently teleport the player a whole block sideways. Resolution may
    /// leave the player where they are, but it must never launch them.
    #[test]
    fn a_centred_block_never_teleports_the_player() {
        let mut chunks = ChunkManager::default();
        let feet_y = 100.0_f32;
        let head_y = feet_y + 1.8;
        // Solid block occupying the player's own column at torso height.
        chunks.set_block(IVec3::new(0, feet_y as i32, 0), BlockType::STONE);

        let mut transform = Transform::from_xyz(0.5, head_y, 0.5);
        let before = transform.translation;
        resolve_walls(&mut transform, feet_y, head_y, &chunks);

        let moved = (transform.translation - before).length();
        assert!(
            moved < 0.5,
            "player was displaced {moved} blocks by a centred block"
        );
    }

    /// A frame spike must not let the player fall through the world. Bevy
    /// clamps the frame delta to 0.25s, which at terminal velocity still covers
    /// ~15 blocks, and an asset-load hitch reaches that clamp easily. A probe
    /// that only looks a fixed depth below the feet misses the floor entirely
    /// and the player ends up buried in terrain.
    #[test]
    fn a_long_fall_in_one_frame_still_lands() {
        let mut chunks = ChunkManager::default();
        let ground_y = 100;
        for dx in -1..=1 {
            for dz in -1..=1 {
                chunks.set_block(IVec3::new(dx, ground_y, dz), BlockType::STONE);
            }
        }
        let ground_top = ground_y as f32 + 1.0;

        // Feet started well above the floor and ended well below it: the whole
        // descent happened inside a single frame.
        let prev_foot_y = ground_top + 14.0;
        let ended_at = ground_top - 6.0;
        let feet = Vec3::new(0.5, ended_at, 0.5);

        let support = highest_support(feet, prev_foot_y, &chunks)
            .expect("swept search must find the floor it passed through");
        assert!(
            (support - ground_top).abs() < 1e-3,
            "landed on {support}, expected {ground_top}"
        );
    }

    /// The sweep must not become a magnet: a player falling high above the
    /// ground should keep falling, not snap down to distant terrain.
    #[test]
    fn falling_far_above_ground_does_not_snap_down() {
        let mut chunks = ChunkManager::default();
        chunks.set_block(IVec3::new(0, 10, 0), BlockType::STONE);

        // Drifting down from 60.5 to 60.0 — nowhere near the block at y=10.
        let feet = Vec3::new(0.5, 60.0, 0.5);
        assert!(highest_support(feet, 60.5, &chunks).is_none());
    }

    #[test]
    fn cannot_place_a_block_inside_yourself() {
        let eye = Vec3::new(0.5, 10.0, 0.5);
        let foot_y = eye.y - 1.8;
        // Block occupying the player's own column, at torso height.
        assert!(block_overlaps_player(IVec3::new(0, 9, 0), eye, foot_y));
    }

    #[test]
    fn placing_clear_of_the_body_is_allowed() {
        let eye = Vec3::new(0.5, 10.0, 0.5);
        let foot_y = eye.y - 1.8;
        // Well to the side.
        assert!(!block_overlaps_player(IVec3::new(5, 9, 5), eye, foot_y));
        // Directly overhead, above the head.
        assert!(!block_overlaps_player(IVec3::new(0, 11, 0), eye, foot_y));
        // Directly underfoot, below the feet.
        assert!(!block_overlaps_player(IVec3::new(0, 7, 0), eye, foot_y));
    }
}
