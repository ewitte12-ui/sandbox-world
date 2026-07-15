use std::time::Instant;

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

use crate::chunk_manager::ChunkManager;
use crate::settings::GameSettings;
use crate::GameState;

/// Runtime-tweakable developer settings — the SINGLE SOURCE OF TRUTH for
/// all gameplay tuning values.
///
/// RATIONALE: previously, player.rs duplicated these values as local
/// constants, creating two sources of truth that could silently diverge.
/// Now player systems read directly from this resource, so changing a value
/// here takes effect immediately at runtime (e.g., via a future dev UI).
///
/// Tuning rationale (ported from the original Swift version):
/// - gravity -28.0: stronger than realistic (-9.8) for snappy game-feel;
///   paired with jump_velocity to give ~2.6 block jump height.
/// - jump_velocity 12.0: clears a 2-block wall but not 3.
/// - reach 8.0: matches Minecraft creative-mode reach.
/// - break/place_interval: minimum seconds between held-click actions.
/// - sprint_multiplier 1.5 / sneak_multiplier 0.2: relative to player_speed.
///
/// NOTE: animal tuning values (animal_count, etc.) are not yet read by
/// animals.rs — that system still uses its own constants.
#[derive(Resource)]
pub struct DevSettings {
    pub player_speed: f32,
    pub sprint_multiplier: f32,
    pub sneak_multiplier: f32,
    pub jump_velocity: f32,
    pub gravity: f32,
    pub mouse_sensitivity: f32,
    /// Length of a full day/night cycle in seconds (read by update_sun_cycle).
    pub day_cycle_duration: f32,
    /// Lantern point-light radius in blocks (read by update_lantern_lights;
    /// fps_120_mode scales it down — see adapt_shadows_for_fps_mode).
    pub lantern_radius: f32,
    pub break_interval: f32,
    pub place_interval: f32,
    pub reach: f32,
    pub animal_count: u32,
    /// Max completed chunk-generation tasks meshed per frame on the main
    /// thread. Caps the per-frame hitch during initial load / fast movement.
    /// Also budgets the deferred remesh queue (neighbor-load and registry
    /// invalidations) in remesh_dirty_chunks.
    pub max_chunk_meshes_per_frame: u32,
    /// Vertex ambient-occlusion strength: 0.0 = off, 1.0 = full corner
    /// darkening. Takes effect on newly (re)meshed chunks.
    pub ao_strength: f32,
    pub show_fps: bool,
    /// Debug: draw chunk bounding boxes colored by face density. Toggle with F4.
    pub show_chunk_bounds: bool,
    /// Debug: color greedy-merged quads cyan to distinguish from naive 1×1 quads.
    /// Toggle with F5. Only visible when enable_greedy_meshing is true.
    pub highlight_greedy_quads: bool,
}

impl Default for DevSettings {
    fn default() -> Self {
        Self {
            player_speed: 22.0,
            sprint_multiplier: 1.5,
            sneak_multiplier: 0.2,
            jump_velocity: 12.0,
            gravity: -28.0,
            mouse_sensitivity: 0.0007,
            day_cycle_duration: 600.0,
            // 12.0 matches the visual result the hardcoded PointLight range
            // produced before this field was wired up.
            lantern_radius: 12.0,
            break_interval: 0.15,
            place_interval: 0.18,
            reach: 8.0,
            animal_count: 60,
            max_chunk_meshes_per_frame: 8,
            ao_strength: 1.0,
            show_fps: true,
            show_chunk_bounds: false,
            highlight_greedy_quads: false,
        }
    }
}

/// Centralized safety switches for optimization features.
///
/// HARD RULE: Correctness gates all performance work. Missing blocks or
/// textures invalidate every optimization — no exception. Performance
/// tuning without a visually stable baseline (src_correctness_baseline/)
/// is forbidden. If the baseline is broken, fix it first.
/// No optimization code may run unless its flag is true.
///
/// The one remaining flag: greedy meshing (single-axis U merge with a
/// debug kill-switch — see chunk.rs GREEDY_INVARIANT_VIOLATED).
///
/// The other flags this struct used to declare (enable_face_culling,
/// enable_chunk_culling, enable_render_scale, enable_120fps_mode) were
/// never read by any system: face culling and chunk unloading are always
/// on (they are correctness features, not optimizations), and render
/// scale / 120fps mode are driven by GameSettings (render_scale,
/// fps_120_mode). They were removed rather than wired — a safety switch
/// that doesn't switch anything is worse than none.
///
/// HARD RULES that outlive the removed flags:
///   - Bevy's built-in visibility + frustum culling stays on. Never
///     insert NoFrustumCulling on chunk entities.
///   - Correctness gates all performance work: if any block or texture
///     disappears after enabling greedy meshing, disable it immediately.
#[derive(Resource)]
pub struct OptimizationFlags {
    /// Merge adjacent coplanar same-type/same-AO faces into larger quads.
    /// ON by default since the texture-array material (UV repeat across
    /// merged runs) removed the atlas-UV blocker; the debug kill-switch
    /// still auto-disables it if a meshing invariant trips.
    pub enable_greedy_meshing: bool,
}

impl Default for OptimizationFlags {
    fn default() -> Self {
        Self {
            enable_greedy_meshing: true,
        }
    }
}

/// Component tagging the FPS display text.
#[derive(Component)]
pub struct FpsText;

/// Tracks the start time of each frame's Update schedule for render time measurement.
#[derive(Resource)]
struct FrameStartTime {
    instant: Instant,
    /// Smoothed render-only time in ms (excludes vsync wait).
    smoothed_render_ms: f64,
}

impl Default for FrameStartTime {
    fn default() -> Self {
        Self {
            instant: Instant::now(),
            smoothed_render_ms: 5.0,
        }
    }
}

/// CPU-side frame pacing: spin-waits until `target_frame_time` has elapsed
/// since the previous frame, giving a consistent frame cadence independent
/// of vsync.
#[derive(Resource)]
pub struct FrameLimiter {
    pub target_frame_time: std::time::Duration,
    pub last_frame_end: Instant,
}

impl Default for FrameLimiter {
    fn default() -> Self {
        Self {
            target_frame_time: std::time::Duration::from_secs_f64(1.0 / 120.0),
            last_frame_end: Instant::now(),
        }
    }
}

/// Tracks render performance baseline for debug regression detection.
/// Only meaningful in debug builds.
#[derive(Resource)]
struct RenderPerfBaseline {
    /// Smoothed render ms captured before render-scale was enabled.
    baseline_ms: f64,
    /// The render_scale value when baseline was last captured.
    baseline_scale: f32,
}

impl Default for RenderPerfBaseline {
    fn default() -> Self {
        Self {
            baseline_ms: 0.0,
            baseline_scale: 1.0,
        }
    }
}

pub struct DevToolsPlugin;

impl Plugin for DevToolsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DevSettings>()
            .init_resource::<OptimizationFlags>()
            .init_resource::<FrameStartTime>()
            .init_resource::<FrameLimiter>()
            .init_resource::<RenderPerfBaseline>()
            .add_plugins(FrameTimeDiagnosticsPlugin::default())
            // mark_frame_start runs in First (before all Update systems) to
            // capture the timestamp. update_fps_display runs in Last (after
            // all Update systems) to measure the elapsed CPU work time,
            // excluding vsync wait which happens between Last and First.
            .add_systems(First, mark_frame_start)
            // Headless verification harness (env-gated, no-op otherwise):
            //   METALWORLD_AUTOSTART=1  — click "Play" programmatically on
            //                             the first frame (New Game).
            //   METALWORLD_SHOT=<path>  — once the world is ready, wait ~1s,
            //                             save a screenshot there, then exit.
            // Lets scripted runs drive the real meshing/render pipeline and
            // produce an inspectable image without a human clicking.
            .add_systems(Update, (dev_autostart, dev_auto_screenshot))
            .add_systems(Startup, spawn_fps_display)
            .add_systems(Last, update_fps_display)
            .add_systems(Last, frame_limiter_system.after(update_fps_display))
            .add_systems(Update, toggle_fps_display)
            .add_systems(Last, detect_render_scale_regression);

        #[cfg(debug_assertions)]
        {
            app.add_systems(Startup, spawn_debug_state_overlay)
                .add_systems(Update, (debug_toggle_chunk_viz, debug_draw_chunk_bounds, update_debug_state_overlay))
                .add_systems(Last, debug_log_visibility_counts)
                .add_systems(Update, debug_assert_clean_menu
                    .run_if(in_state(GameState::Menu)))
                .add_systems(Update, debug_assert_world_intact_during_gameplay
                    .run_if(in_state(GameState::Gameplay)))
                .add_systems(Last, debug_assert_single_world_camera)
                .add_systems(OnEnter(GameState::Gameplay), debug_log_resource_counts)
                .add_systems(Update, debug_overlay_fps_regression
                    .run_if(in_state(GameState::Gameplay)))
                .add_systems(Update, debug_spatial_validation
                    .run_if(in_state(GameState::Gameplay)))
                .add_systems(Update, dev_hotkey_world_reset
                    .run_if(in_state(GameState::Gameplay)));
        }
    }
}

/// Runs at the very start of each frame — records the instant before any systems run.
fn mark_frame_start(mut timer: ResMut<FrameStartTime>) {
    timer.instant = Instant::now();
}

/// CPU-side frame pacing: sleeps the remaining time to hit the target frame duration.
fn frame_limiter_system(
    settings: Res<GameSettings>,
    mut limiter: ResMut<FrameLimiter>,
) {
    if !settings.fps_120_mode {
        limiter.last_frame_end = Instant::now();
        return;
    }

    let elapsed = limiter.last_frame_end.elapsed();
    if elapsed < limiter.target_frame_time {
        std::thread::sleep(limiter.target_frame_time - elapsed);
    }
    limiter.last_frame_end = Instant::now();
}

fn spawn_fps_display(mut commands: Commands) {
    commands.spawn((
        FpsText,
        Text::new("FPS: --"),
        TextFont {
            font_size: 16.0,
            ..default()
        },
        TextColor(Color::linear_rgb(1.0, 1.0, 0.0)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
    ));
}

/// Runs at the very end of each frame — measures actual CPU work time (excludes vsync wait).
fn update_fps_display(
    diagnostics: Res<DiagnosticsStore>,
    dev: Res<DevSettings>,
    chunk_manager: Option<Res<ChunkManager>>,
    mut timer: ResMut<FrameStartTime>,
    mut text_query: Query<(&mut Text, &mut Visibility), With<FpsText>>,
    state: Res<State<GameState>>,
    game_settings: Option<Res<crate::settings::GameSettings>>,
    camera_rt_q: Query<&bevy::camera::RenderTarget, With<Camera3d>>,
) {
    // Measure CPU render time: time from First to Last (excludes vsync wait which happens after)
    let render_ms = timer.instant.elapsed().as_secs_f64() * 1000.0;
    timer.smoothed_render_ms = timer.smoothed_render_ms * 0.9 + render_ms * 0.1;

    let show = dev.show_fps && *state.get() == GameState::Gameplay;

    for (mut text, mut visibility) in &mut text_query {
        *visibility = if show {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };

        if !show {
            continue;
        }

        // Vsync-limited FPS
        let fps = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|d| d.smoothed())
            .unwrap_or(0.0);

        // Render-only FPS = 1000 / render_ms (what the GPU could do without vsync)
        let render_fps = 1000.0 / timer.smoothed_render_ms.max(0.1);

        let chunks_loaded = chunk_manager
            .as_ref()
            .map(|cm| cm.chunks.len())
            .unwrap_or(0);

        let mem_mb = get_memory_usage_mb();

        // Render-scale diagnostics: detect output mode and extra blit pass.
        let scale = game_settings.as_ref().map(|s| s.render_scale).unwrap_or(1.0);
        let (render_mode, blit_pass) = if let Some(rt) = camera_rt_q.iter().next() {
            match rt {
                bevy::camera::RenderTarget::Window(_) => ("Window", false),
                bevy::camera::RenderTarget::Image(_) => ("RTT", true),
                _ => ("Other", false),
            }
        } else {
            ("None", false)
        };

        **text = format!(
            "FPS: {:.0} | Render: {:.0} ({:.1}ms) | Chunks: {} | Mem: {:.0}MB\n\
             Scale: {:.0}% | Output: {} | Blit: {}",
            fps, render_fps, timer.smoothed_render_ms, chunks_loaded, mem_mb,
            scale * 100.0, render_mode, if blit_pass { "Yes" } else { "No" }
        );
    }
}

fn toggle_fps_display(keys: Res<ButtonInput<KeyCode>>, mut dev: ResMut<DevSettings>) {
    if keys.just_pressed(KeyCode::F3) {
        dev.show_fps = !dev.show_fps;
    }
}

/// Get the current process resident memory usage in MB.
fn get_memory_usage_mb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        use std::mem::MaybeUninit;
        extern "C" {
            fn mach_task_self() -> u32;
            fn task_info(
                target_task: u32,
                flavor: u32,
                task_info_out: *mut libc_task_basic_info,
                task_info_count: *mut u32,
            ) -> i32;
        }
        #[repr(C)]
        struct libc_task_basic_info {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [u32; 2],
            system_time: [u32; 2],
            policy: i32,
            suspend_count: i32,
        }
        const MACH_TASK_BASIC_INFO: u32 = 20;
        let mut info = MaybeUninit::<libc_task_basic_info>::uninit();
        let mut count = (std::mem::size_of::<libc_task_basic_info>() / 4) as u32;
        let kr = unsafe {
            task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                info.as_mut_ptr(),
                &mut count,
            )
        };
        if kr == 0 {
            let info = unsafe { info.assume_init() };
            return info.resident_size as f64 / (1024.0 * 1024.0);
        }
        0.0
    }
    #[cfg(not(target_os = "macos"))]
    {
        0.0
    }
}

/// Debug-only: detects when enabling render-scale or fps_120_mode causes a
/// performance regression (render time increases instead of decreasing).
/// Captures a baseline when at scale 1.0, then warns if render time increases
/// by more than 20% after lowering render_scale.
fn detect_render_scale_regression(
    game_settings: Option<Res<GameSettings>>,
    timer: Res<FrameStartTime>,
    mut baseline: ResMut<RenderPerfBaseline>,
) {
    let Some(settings) = game_settings else { return };
    let current_scale = settings.render_scale;
    let current_ms = timer.smoothed_render_ms;

    // When at full scale, continuously update the baseline.
    if (current_scale - 1.0).abs() < f32::EPSILON {
        if current_ms > 1.0 {
            baseline.baseline_ms = current_ms;
        }
        baseline.baseline_scale = 1.0;
        return;
    }

    // Scale just changed from 1.0 to something lower — record transition.
    if (baseline.baseline_scale - 1.0).abs() < f32::EPSILON && current_scale < 1.0 {
        baseline.baseline_scale = current_scale;
        return; // allow a few frames to stabilize
    }

    // Only check once the smoothed value has had time to settle and we have
    // a meaningful baseline.
    #[cfg(debug_assertions)]
    if baseline.baseline_ms > 1.0
        && (baseline.baseline_scale - current_scale).abs() < f32::EPSILON
        && current_ms > baseline.baseline_ms * 1.2
    {
        bevy::log::warn!(
            "Render-scale regression detected: render_scale={:.0}% takes {:.1}ms \
             vs {:.1}ms at native resolution. The extra blit pass may cost more \
             than the reduced pixel count saves. Consider disabling render-scale \
             on this GPU.",
            current_scale * 100.0,
            current_ms,
            baseline.baseline_ms,
        );
        // Reset baseline to avoid spamming the warning every frame.
        baseline.baseline_ms = current_ms;
    }
}

/// Toggle chunk bounds visualization with F4.
#[cfg(debug_assertions)]
fn debug_toggle_chunk_viz(keys: Res<ButtonInput<KeyCode>>, mut dev: ResMut<DevSettings>) {
    if keys.just_pressed(KeyCode::F4) {
        dev.show_chunk_bounds = !dev.show_chunk_bounds;
    }
    if keys.just_pressed(KeyCode::F5) {
        dev.highlight_greedy_quads = !dev.highlight_greedy_quads;
    }
}

/// Draw wireframe bounding boxes for all chunks, colored by face density.
///   Red    = 0 vertices (empty mesh — possible geometry loss)
///   Yellow = suspiciously low (<100 vertices for a chunk with blocks)
///   Green  = normal density
#[cfg(debug_assertions)]
///   Cyan   = high density (>3000 vertices)
///   Magenta = frustum-culled (has geometry but not visible this frame)
fn debug_draw_chunk_bounds(
    dev: Res<DevSettings>,
    chunks: Query<(&Transform, &Mesh3d, Option<&ViewVisibility>), With<crate::chunk_manager::ChunkMarker>>,
    meshes: Res<Assets<Mesh>>,
    mut gizmos: Gizmos,
) {
    if !dev.show_chunk_bounds {
        return;
    }

    // Bevy Gizmos APIs are version-specific. In Bevy 0.18, use gizmos.cube()
    // — gizmos.cuboid() does not exist. Always verify available methods
    // against the current Bevy version before adding gizmo calls.
    let chunk_size = crate::chunk::CHUNK_SIZE as f32;
    let half = chunk_size / 2.0;

    for (transform, mesh3d, view_vis) in &chunks {
        let center = transform.translation + Vec3::splat(half);
        let size = Vec3::splat(chunk_size);

        let vert_count = meshes
            .get(&mesh3d.0)
            .map(|m| m.count_vertices())
            .unwrap_or(0);

        let is_culled = view_vis.map_or(false, |v| !v.get());

        let color = if is_culled && vert_count > 0 {
            Color::linear_rgb(1.0, 0.0, 1.0) // magenta = frustum-culled with geometry
        } else if vert_count == 0 {
            Color::linear_rgb(1.0, 0.0, 0.0) // red = empty, possible bug
        } else if vert_count < 100 {
            Color::linear_rgb(1.0, 1.0, 0.0) // yellow = suspiciously low
        } else if vert_count > 3000 {
            Color::linear_rgb(0.0, 1.0, 1.0) // cyan = high density
        } else {
            Color::linear_rgb(0.0, 1.0, 0.0) // green = normal, visible
        };

        gizmos.cube(
            Transform::from_translation(center).with_scale(size),
            color,
        );
    }
}

/// Debug: log chunk visibility counts from Bevy's built-in frustum culling.
/// Reports how many chunks are visible vs culled each ~2 seconds, and warns
/// if the count never changes (suggesting the camera isn't affecting culling).
#[cfg(debug_assertions)]
fn debug_log_visibility_counts(
    chunks: Query<&ViewVisibility, With<crate::chunk_manager::ChunkMarker>>,
) {
    static FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    static LAST_VISIBLE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

    let frame = FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if frame % 120 != 0 {
        return;
    }

    let mut total = 0u32;
    let mut visible = 0u32;
    for vis in &chunks {
        total += 1;
        if vis.get() {
            visible += 1;
        }
    }

    let prev = LAST_VISIBLE.swap(visible, std::sync::atomic::Ordering::Relaxed);
    let culled = total.saturating_sub(visible);

    bevy::log::info!(
        "Chunk visibility: {}/{} visible, {} culled by frustum",
        visible, total, culled,
    );

    // If visibility count hasn't changed across multiple samples and there
    // are chunks loaded, the camera may not be affecting culling.
    if prev != u32::MAX && prev == visible && total > 0 && culled == 0 {
        bevy::log::info!(
            "All {} chunks visible (unchanged) — camera may be seeing everything",
            total,
        );
    }
}

/// Debug: verify that no world entities or cameras exist during Menu state.
/// Runs only in GameState::Menu. Logs violations — does not panic.
#[cfg(debug_assertions)]
fn debug_assert_clean_menu(
    world_entities: Query<Entity, With<crate::WorldEntity>>,
    cameras_3d: Query<Entity, With<Camera3d>>,
    cameras_2d: Query<Entity, With<Camera2d>>,
    chunks: Query<Entity, With<crate::chunk_manager::ChunkMarker>>,
    animals: Query<Entity, With<crate::animals::AnimalPart>>,
    ui_nodes: Query<Entity, With<Node>>,
) {
    // Throttle to once per second (~60 frames).
    static FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let frame = FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if frame % 60 != 0 {
        return;
    }

    // Must have exactly one UI camera.
    let cam_2d_count = cameras_2d.iter().count();
    if cam_2d_count != 1 {
        bevy::log::warn!(
            "Menu state has {} Camera2d entities — should be exactly 1",
            cam_2d_count,
        );
    }

    // Must have at least one visible UI node (background + menu).
    let node_count = ui_nodes.iter().count();
    if node_count == 0 {
        bevy::log::warn!(
            "Menu state has 0 UI nodes — menu is invisible",
        );
    }

    // Must NOT have any 3D/world cameras.
    let cam_3d_count = cameras_3d.iter().count();
    if cam_3d_count > 0 {
        bevy::log::warn!(
            "Menu state has {} Camera3d entities — should be 0",
            cam_3d_count,
        );
    }

    // Must NOT have any world entities.
    let world_count = world_entities.iter().count();
    if world_count > 0 {
        bevy::log::warn!(
            "Menu state has {} WorldEntity entities — should be 0",
            world_count,
        );
    }

    let chunk_count = chunks.iter().count();
    if chunk_count > 0 {
        bevy::log::warn!(
            "Menu state has {} chunk entities — should be 0",
            chunk_count,
        );
    }

    let animal_count = animals.iter().count();
    if animal_count > 0 {
        bevy::log::warn!(
            "Menu state has {} animal entities — should be 0",
            animal_count,
        );
    }
}

/// Marker for the debug state overlay text.
#[cfg(debug_assertions)]
#[derive(Component)]
struct DebugStateOverlay;

/// Spawn a text node in the top-right corner showing app state diagnostics.
#[cfg(debug_assertions)]
fn spawn_debug_state_overlay(mut commands: Commands) {
    commands.spawn((
        DebugStateOverlay,
        Text::new(""),
        TextFont { font_size: 14.0, ..default() },
        TextColor(Color::linear_rgb(1.0, 1.0, 0.5)),
        Node {
            position_type: PositionType::Absolute,
            right: Val::Px(8.0),
            top: Val::Px(8.0),
            ..default()
        },
    ));
}

/// Update the debug overlay with current state, camera, and background info.
#[cfg(debug_assertions)]
fn update_debug_state_overlay(
    state: Res<State<GameState>>,
    cameras_2d: Query<(), With<Camera2d>>,
    cameras_3d: Query<(), With<Camera3d>>,
    bg_query: Query<(), With<crate::ui::MenuBackground>>,
    mut text_query: Query<&mut Text, With<DebugStateOverlay>>,
) {
    for mut text in &mut text_query {
        let state_name = match state.get() {
            GameState::Menu => "Menu",
            GameState::Gameplay => "Gameplay",
        };
        let cam_2d = cameras_2d.iter().count();
        let cam_3d = cameras_3d.iter().count();
        let has_bg = !bg_query.is_empty();

        **text = format!(
            "State: {} | Cam2D: {} | Cam3D: {} | MenuBG: {}",
            state_name, cam_2d, cam_3d,
            if has_bg { "yes" } else { "no" },
        );
    }
}

/// Debug: verify that world entities are never torn down during Gameplay.
/// Settings is an overlay — the world must remain fully intact.
/// Any loss of the player camera or world entities during Gameplay
/// (including while Settings is open) is a correctness bug.
#[cfg(debug_assertions)]
fn debug_assert_world_intact_during_gameplay(
    cameras_3d: Query<(), With<Camera3d>>,
    world_entities: Query<(), With<crate::WorldEntity>>,
) {
    // Throttle to every 60 frames.
    static FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let frame = FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if frame % 60 != 0 {
        return;
    }

    if cameras_3d.iter().count() == 0 {
        bevy::log::warn!(
            "Gameplay state has 0 Camera3d entities — world camera was \
             despawned during Gameplay (possible settings teardown bug)"
        );
    }

    if world_entities.iter().count() == 0 {
        bevy::log::warn!(
            "Gameplay state has 0 WorldEntity entities — world was torn \
             down during Gameplay (teardown is only allowed on entering Menu)"
        );
    }
}

/// Debug: assert at most one active Camera3d exists at any time.
/// Multiple world cameras cause render ambiguity warnings and visual glitches.
#[cfg(debug_assertions)]
fn debug_assert_single_world_camera(
    cameras: Query<(Entity, &Camera), With<Camera3d>>,
) {
    // Throttle to every 60 frames.
    static FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let frame = FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if frame % 60 != 0 {
        return;
    }

    let active: Vec<_> = cameras.iter()
        .filter(|(_, cam)| cam.is_active)
        .collect();

    if active.len() > 1 {
        bevy::log::warn!(
            "Multiple active Camera3d entities detected ({}) — expected at most 1:",
            active.len(),
        );
        for (entity, cam) in &active {
            bevy::log::warn!(
                "  Camera3d entity {:?}: order={}",
                entity, cam.order,
            );
        }
    }
}

/// Debug: log entity and resource counts on each Gameplay entry to detect
/// accumulation across Menu→Gameplay transitions. If counts grow each
/// time, something is leaking.
#[cfg(debug_assertions)]
fn debug_log_resource_counts(
    cameras_3d: Query<Entity, With<Camera3d>>,
    cameras_2d: Query<Entity, With<Camera2d>>,
    world_entities: Query<Entity, With<crate::WorldEntity>>,
    chunks: Query<Entity, With<crate::chunk_manager::ChunkMarker>>,
    animals: Query<Entity, With<crate::animals::AnimalPart>>,
    meshes: Res<Assets<Mesh>>,
    materials: Res<Assets<StandardMaterial>>,
    images: Res<Assets<Image>>,
) {
    static ENTRY_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let entry = ENTRY_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

    let cam_3d = cameras_3d.iter().count();
    let cam_2d = cameras_2d.iter().count();
    let world = world_entities.iter().count();
    let chunk = chunks.iter().count();
    let animal = animals.iter().count();
    let mesh_count = meshes.len();
    let mat_count = materials.len();
    let img_count = images.len();

    bevy::log::info!(
        "=== Gameplay entry #{} resource snapshot ===\n\
         Cam3D: {} | Cam2D: {} | WorldEntity: {} | Chunks: {} | Animals: {}\n\
         Meshes: {} | Materials: {} | Images: {}",
        entry, cam_3d, cam_2d, world, chunk, animal,
        mesh_count, mat_count, img_count,
    );

    // Warn if counts suggest leaks (after first entry, cameras should be
    // exactly 1 Camera3d and 0 Camera2d, chunks/animals should be 0 at
    // the moment of entry before systems run).
    if cam_3d > 1 {
        bevy::log::warn!("LEAK: {} Camera3d entities on entry (expected 1)", cam_3d);
    }
    if cam_2d > 0 {
        bevy::log::warn!("LEAK: {} Camera2d entities on Gameplay entry (expected 0)", cam_2d);
    }
    if entry > 1 && chunk > 0 {
        bevy::log::warn!(
            "LEAK: {} chunk entities from previous session still exist on entry #{}",
            chunk, entry,
        );
    }
    if entry > 1 && animal > 0 {
        bevy::log::warn!(
            "LEAK: {} animal entities from previous session still exist on entry #{}",
            animal, entry,
        );
    }
}

/// Debug: detect FPS regression caused by menu overlay open/close cycles.
/// Records render_ms before opening and compares after closing. Logs an
/// error if performance degrades, indicating a cumulative leak.
#[cfg(debug_assertions)]
fn debug_overlay_fps_regression(
    menu_state: Res<crate::ui::MenuState>,
    timer: Res<FrameStartTime>,
    cameras_3d: Query<(Entity, &Camera), With<Camera3d>>,
    cameras_2d: Query<(Entity, &Camera), With<Camera2d>>,
    world_entities: Query<(), With<crate::WorldEntity>>,
    chunks: Query<(), With<crate::chunk_manager::ChunkMarker>>,
    animals: Query<(), With<crate::animals::AnimalPart>>,
    all_entities: Query<Entity>,
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    static STATE: Mutex<Option<OverlayPerfState>> = Mutex::new(None);
    static DEBOUNCE: AtomicU64 = AtomicU64::new(0);

    struct OverlayPerfState {
        pre_open_ms: f64,
        was_open: bool,
        cycle_count: u32,
        pre_entity_total: u32,
        pre_world_count: u32,
        pre_chunk_count: u32,
        pre_animal_count: u32,
    }

    let current_ms = timer.smoothed_render_ms;
    let is_open = menu_state.is_open;

    let Ok(mut guard) = STATE.lock() else { return };
    let total = all_entities.iter().count() as u32;
    let world_count = world_entities.iter().count() as u32;
    let chunk_count = chunks.iter().count() as u32;
    let animal_count = animals.iter().count() as u32;

    let state = guard.get_or_insert(OverlayPerfState {
        pre_open_ms: current_ms,
        was_open: false,
        cycle_count: 0,
        pre_entity_total: total,
        pre_world_count: world_count,
        pre_chunk_count: chunk_count,
        pre_animal_count: animal_count,
    });

    let debounce = DEBOUNCE.load(Ordering::Relaxed);

    if is_open && !state.was_open {
        // Overlay just opened — capture pre-open counts.
        state.pre_open_ms = current_ms;
        state.pre_entity_total = total;
        state.pre_world_count = world_count;
        state.pre_chunk_count = chunk_count;
        state.pre_animal_count = animal_count;
        state.was_open = true;
        DEBOUNCE.store(0, Ordering::Relaxed);
    } else if !is_open && state.was_open {
        // Overlay just closed — start debounce.
        state.was_open = false;
        state.cycle_count += 1;
        DEBOUNCE.store(1, Ordering::Relaxed);
    } else if !is_open && debounce > 0 {
        // Debouncing after close — wait 60 frames for performance to stabilize.
        let d = DEBOUNCE.fetch_add(1, Ordering::Relaxed);
        if d >= 60 {
            DEBOUNCE.store(0, Ordering::Relaxed);

            // Log camera state after each overlay cycle.
            let cam3d_count = cameras_3d.iter().count();
            let cam2d_count = cameras_2d.iter().count();
            bevy::log::info!(
                "Post-overlay cycle #{}: Camera3d={} Camera2d={}",
                state.cycle_count, cam3d_count, cam2d_count,
            );
            for (entity, cam) in cameras_3d.iter() {
                bevy::log::info!(
                    "  Camera3d {:?}: order={}, active={}",
                    entity, cam.order, cam.is_active,
                );
            }
            for (entity, cam) in cameras_2d.iter() {
                bevy::log::info!(
                    "  Camera2d {:?}: order={}, active={}",
                    entity, cam.order, cam.is_active,
                );
            }
            if cam3d_count != 1 {
                bevy::log::warn!(
                    "CAMERA LEAK: expected 1 Camera3d, found {} after overlay cycle #{}",
                    cam3d_count, state.cycle_count,
                );
            }
            if cam2d_count != 0 {
                bevy::log::warn!(
                    "CAMERA LEAK: expected 0 Camera2d during Gameplay, found {} after overlay cycle #{}",
                    cam2d_count, state.cycle_count,
                );
            }

            // Entity count delta — detect leaks.
            let delta_total = total as i32 - state.pre_entity_total as i32;
            let delta_world = world_count as i32 - state.pre_world_count as i32;
            let delta_chunks = chunk_count as i32 - state.pre_chunk_count as i32;
            let delta_animals = animal_count as i32 - state.pre_animal_count as i32;

            if delta_total != 0 {
                bevy::log::info!(
                    "Entity delta after overlay cycle #{}: total {:+} \
                     (world {:+}, chunks {:+}, animals {:+})",
                    state.cycle_count, delta_total,
                    delta_world, delta_chunks, delta_animals,
                );
                // Non-world entity changes (e.g. UI nodes) are expected to net zero.
                // World entity changes during overlay are a leak.
                if delta_world != 0 {
                    bevy::log::warn!(
                        "ENTITY LEAK: WorldEntity count changed by {:+} during overlay cycle #{}",
                        delta_world, state.cycle_count,
                    );
                }
            }

            let pre = state.pre_open_ms;
            let post = current_ms;
            if pre > 1.0 && post > pre * 1.15 {
                bevy::log::error!(
                    "FPS REGRESSION after overlay cycle #{}: \
                     {:.1}ms before → {:.1}ms after ({:+.0}%) — \
                     possible resource leak from menu open/close",
                    state.cycle_count, pre, post,
                    ((post / pre) - 1.0) * 100.0,
                );
            }
        }
    }
}

/// Debug: validate spatial correctness of chunks and camera.
/// Logs transforms for the chunk nearest to origin and the active camera.
/// Draws a reference cube at world origin to confirm the render pipeline
/// is working and chunks are inside the frustum.
#[cfg(debug_assertions)]
fn debug_spatial_validation(
    dev: Res<DevSettings>,
    chunks: Query<(&Transform, &crate::chunk_manager::ChunkMarker)>,
    camera_q: Query<(&GlobalTransform, &Projection), With<Camera3d>>,
    mut gizmos: Gizmos,
) {
    // Throttle logging to every 300 frames (~5s).
    static FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let frame = FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let should_log = frame % 300 == 0;

    // Reference cube gated behind the same toggle as chunk bounds.
    if dev.show_chunk_bounds {
        gizmos.cube(
            Transform::from_translation(Vec3::new(0.0, 10.0, 0.0))
                .with_scale(Vec3::splat(2.0)),
            Color::linear_rgb(1.0, 1.0, 0.0), // yellow
        );
    }

    if !should_log {
        return;
    }

    // Find the chunk closest to origin.
    let mut nearest: Option<(&Transform, &crate::chunk_manager::ChunkMarker)> = None;
    let mut nearest_dist = f32::MAX;
    for (transform, marker) in &chunks {
        let dist = transform.translation.length();
        if dist < nearest_dist {
            nearest_dist = dist;
            nearest = Some((transform, marker));
        }
    }

    if let Some((transform, marker)) = nearest {
        let t = transform.translation;
        let s = transform.scale;
        bevy::log::info!(
            "Nearest chunk to origin: ({},{},{}) \
             transform=({:.1},{:.1},{:.1}) scale=({:.3},{:.3},{:.3}) \
             dist={:.1} scale_valid={}",
            marker.pos.0, marker.pos.1, marker.pos.2,
            t.x, t.y, t.z,
            s.x, s.y, s.z,
            nearest_dist,
            s.x.is_finite() && s.y.is_finite() && s.z.is_finite()
                && s.x > 0.0 && s.y > 0.0 && s.z > 0.0,
        );
    } else {
        bevy::log::warn!("No chunk entities found for spatial validation");
    }

    // Log camera transform and projection.
    if let Some((gt, proj)) = camera_q.iter().next() {
        let ct = gt.translation();
        let (fov_info, near, far) = match proj {
            Projection::Perspective(p) => (
                format!("fov={:.1}°", p.fov.to_degrees()),
                p.near,
                p.far,
            ),
            Projection::Orthographic(o) => (
                format!("ortho scale={:.1}", o.scale),
                o.near,
                o.far,
            ),
            _ => (
                "custom".to_string(),
                0.0,
                0.0,
            ),
        };
        bevy::log::info!(
            "Camera: pos=({:.1},{:.1},{:.1}) {} near={:.2} far={:.1} \
             pos_valid={}",
            ct.x, ct.y, ct.z,
            fov_info, near, far,
            ct.is_finite(),
        );
    } else {
        bevy::log::warn!("No Camera3d found for spatial validation");
    }
}

/// DEV-ONLY: Ctrl+Shift+R (Cmd+Shift+R on macOS) resets the world in-place.
/// Clears all chunks, modifications, and entities, then regenerates from
/// scratch with saves ignored. Does NOT delete save files on disk.
/// TODO: remove before release.
#[cfg(debug_assertions)]
fn dev_hotkey_world_reset(
    keys: Res<ButtonInput<KeyCode>>,
    menu_state: Option<Res<crate::ui::MenuState>>,
    mut commands: Commands,
    world_entities: Query<Entity, With<crate::WorldEntity>>,
    mut chunk_manager: ResMut<crate::chunk_manager::ChunkManager>,
    mut start_mode: ResMut<crate::save_load::StartMode>,
    mut auto_load: ResMut<crate::save_load::AutoLoadState>,
    mut world_ready: ResMut<crate::player::WorldReady>,
    world_id: Res<crate::WorldInstanceId>,
) {
    // Block while menu overlay is open.
    if menu_state.map(|m| m.is_open).unwrap_or(false) {
        return;
    }

    // Ctrl+Shift+R (or Cmd+Shift+R on macOS).
    let ctrl = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight)
        || keys.pressed(KeyCode::SuperLeft) || keys.pressed(KeyCode::SuperRight);
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);

    if !(ctrl && shift && keys.just_pressed(KeyCode::KeyR)) {
        return;
    }

    bevy::log::warn!(
        "=== WORLD RESET VIA HOTKEY — SAVE DATA IGNORED FOR THIS SESSION ==="
    );

    // 1. Set StartMode so auto_load_game skips saved modifications.
    *start_mode = crate::save_load::StartMode::NewGame;

    // 2. Pause physics until new chunks load.
    world_ready.0 = false;

    // 3. Despawn all world entities (chunks, animals, lights, player camera).
    let mut count = 0u32;
    for entity in &world_entities {
        commands.entity(entity).despawn();
        count += 1;
    }

    // 4. Clear all in-memory chunk state.
    chunk_manager.clear_all();
    // Note: modifications are NOT cleared from the save file on disk,
    // only from the in-memory ChunkManager.
    chunk_manager.modifications.clear();

    // 5. Reset auto-load so it re-runs (but StartMode::NewGame will skip it).
    auto_load.loaded = false;

    // 6. Respawn player at default position by triggering OnEnter(Gameplay).
    // We force a state round-trip: Gameplay → Menu → Gameplay.
    // This is handled by the existing state machine — spawn_player,
    // spawn_animals, spawn_clouds, setup_lighting all run on OnEnter(Gameplay).
    // For an in-place reset without state transition, we spawn the player directly.
    commands.spawn((
        crate::WorldEntity,
        crate::WorldScoped(world_id.0),
        crate::player::Player::default(),
        Camera3d::default(),
        Camera {
            order: 0,
            ..default()
        },
        Transform::from_xyz(0.0, 20.0, 0.0),
    ));

    bevy::log::warn!(
        "World reset complete: {} entities despawned, chunks cleared, player respawned at origin",
        count,
    );
}


// ---------------------------------------------------------------------------
// Headless verification harness (env-gated)
// ---------------------------------------------------------------------------

/// Mimics the Play button when METALWORLD_AUTOSTART is set: schedules a
/// NewGame teardown/reload exactly like ui.rs handle_play_and_load_buttons
/// (same resources, same PendingReload deferral), once, on the first frame.
#[allow(clippy::too_many_arguments)]
fn dev_autostart(
    mut fired: Local<bool>,
    mut commands: Commands,
    menu_query: Query<Entity, With<crate::ui::SettingsMenu>>,
    mut menu_state: ResMut<crate::ui::MenuState>,
    mut start_mode: ResMut<crate::save_load::StartMode>,
    mut teardown_intent: ResMut<crate::TeardownIntent>,
    mut world_instance: ResMut<crate::WorldInstanceId>,
    mut pending_teardown: ResMut<crate::PendingTeardown>,
    mut pending_reload: ResMut<crate::PendingReload>,
) {
    if *fired || std::env::var_os("METALWORLD_AUTOSTART").is_none() {
        return;
    }
    *fired = true;

    bevy::log::warn!("METALWORLD_AUTOSTART: entering Gameplay (NewGame) without user input");
    *start_mode = crate::save_load::StartMode::NewGame;
    let old_id = world_instance.0;
    world_instance.0 = old_id + 1;
    *pending_teardown = crate::PendingTeardown {
        old_id,
        kind: crate::TeardownIntent::NewGame,
    };
    *teardown_intent = crate::TeardownIntent::NewGame;
    menu_state.is_open = false;
    for entity in &menu_query {
        commands.entity(entity).despawn();
    }
    *pending_reload = crate::PendingReload {
        active: true,
        frames: crate::RELOAD_DEFER_FRAMES,
    };
}

/// When METALWORLD_SHOT=<path> is set: after WorldReady flips true, waits
/// ~60 frames for lighting/streaming to settle, captures a screenshot to
/// <path>, then exits 90 frames later (giving the async PNG write time).
fn dev_auto_screenshot(
    mut frames_ready: Local<u32>,
    mut requested: Local<bool>,
    world_ready: Option<Res<crate::player::WorldReady>>,
    chunk_manager: Option<Res<ChunkManager>>,
    mut commands: Commands,
) {
    let Some(path) = std::env::var_os("METALWORLD_SHOT") else {
        return;
    };
    if !world_ready.map(|r| r.0).unwrap_or(false) {
        return;
    }
    // Wait for chunk streaming/remesh quiescence so the capture shows the
    // fully assembled world, not a mid-stream snapshot.
    if !*requested && !chunk_manager.map(|cm| cm.streaming_idle()).unwrap_or(true) {
        *frames_ready = 0;
        return;
    }
    *frames_ready += 1;
    if *frames_ready >= 60 && !*requested {
        *requested = true;
        use bevy::render::view::screenshot::{save_to_disk, Screenshot};
        let path = std::path::PathBuf::from(path);
        bevy::log::warn!("METALWORLD_SHOT: capturing {}", path.display());
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }
    if *frames_ready >= 150 {
        bevy::log::warn!("METALWORLD_SHOT: exiting after capture");
        std::process::exit(0);
    }
}
