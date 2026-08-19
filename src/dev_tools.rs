//! Developer tuning knobs.
//!
//! PHASE 1 PARTIAL PORT. This module currently holds only the two resources
//! that `chunk_manager` reads. The rest of the 0.18 `dev_tools.rs` — the
//! developer tab UI, perf monitor, gizmo debug viz, camera-leak diagnostics —
//! arrives in Phase 5 and extends this file rather than replacing it.
//!
//! When porting the remainder, note that the 0.18 overlay diagnostic took an
//! unfiltered `all_entities: Query<Entity>` alongside two `Res<..>` params.
//! Under Bevy 0.19 resources are entity components, so that query both
//! miscounts (it now matches resource entities) and conflicts with the
//! resource access. It needs `Query<Entity, Without<IsResource>>`.

use bevy::prelude::*;

/// Tweakable constants. Nothing in the game may hardcode these values.
///
/// Tuning rationale (ported from the original Swift version):
/// - gravity -28.0: stronger than realistic (-9.8) for snappy game-feel;
///   paired with jump_velocity to give ~2.6 block jump height.
/// - jump_velocity 12.0: clears a 2-block wall but not 3.
/// - reach 8.0: matches Minecraft creative-mode reach.
/// - break/place_interval: minimum seconds between held-click actions.
/// - sprint_multiplier 1.5 / sneak_multiplier 0.2: relative to player_speed.
// Some knobs have no consumer yet: lanterns are unimplemented, and the FPS and
// chunk-bounds toggles await the developer overlay. They stay because this
// struct is the single source of truth for tuning — a value that lives here and
// nowhere else cannot drift out of sync with a hardcoded copy.
#[allow(dead_code)]
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
    /// Master switch for baked voxel lighting (per-voxel sky/block light
    /// flood fill — see voxel_light.rs). When false, falls back to the
    /// legacy lantern PointLight pool for A/B comparison. Toggling needs
    /// NO remesh: meshes always carry light levels, the shader uniform
    /// decides whether they apply.
    pub voxel_lighting: bool,
    /// Floor for the combined voxel-light factor so unlit areas render
    /// very dark instead of pitch black.
    pub voxel_light_min: f32,
    /// Sky-light intensity at deepest night (moon/star floor for the
    /// sun_intensity uniform).
    pub voxel_sky_night: f32,
    /// Max per-frame chunk light recomputes in process_light_queue
    /// (cascades are budgeted like the deferred remesh queue; ×8 while
    /// the loading overlay is up).
    pub max_light_updates_per_frame: u32,
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
            voxel_lighting: true,
            voxel_light_min: 0.04,
            voxel_sky_night: 0.10,
            max_light_updates_per_frame: 32,
            show_fps: true,
            show_chunk_bounds: false,
            highlight_greedy_quads: false,
        }
    }
}

/// Centralized safety switches for optimization features.
///
/// HARD RULE: Correctness gates all performance work. Missing blocks or
/// textures invalidate every optimization — no exception.
/// No optimization code may run unless its flag is true.
///
/// HARD RULES:
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
