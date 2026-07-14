use std::f32::consts::PI;

use bevy::light::{CascadeShadowConfig, CascadeShadowConfigBuilder, DirectionalLightShadowMap};
use bevy::pbr::{DistanceFog, FogFalloff, ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionQualityLevel};
use bevy::prelude::*;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::render::view::ColorGrading;
use bevy::anti_alias::smaa::{Smaa, SmaaPreset};
use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::render::render_resource::{Extent3d, TextureFormat, TextureUsages, TextureDimension};

use bevy::camera::{ImageRenderTarget, RenderTarget};
use bevy::window::PrimaryWindow;

use crate::block_types::BlockType;
use crate::chunk_manager::ChunkManager;

// ---------------------------------------------------------------------------
// Camera import policy (Bevy 0.18)
// ---------------------------------------------------------------------------
// Do not import camera types from bevy::render::* — those modules are private.
// The camera public API lives under bevy::camera (re-exported via bevy::prelude).
// If a field does not appear on the Camera struct in bevy::camera, it does not
// exist in this version of Bevy and must not be used. Older examples and docs
// referencing bevy::render::camera will produce E0603.

// ---------------------------------------------------------------------------
// RenderTarget comparison policy
// ---------------------------------------------------------------------------
// Never compare RenderTarget values using == or != — the type does not
// implement PartialEq. Treat render targets as opaque handles. Always
// inspect them via pattern matching (matches!, if let) on variants like
// RenderTarget::Window(_) or RenderTarget::Image(irt).

// ---------------------------------------------------------------------------
// Render-scale policy
// ---------------------------------------------------------------------------
// Camera.viewport is FORBIDDEN for render-scale or performance optimization.
// Viewport reduces the visible output area (letterboxing), it does not lower
// internal resolution. Viewport may only be used for split-screen or minimaps.
//
// All render scaling must use off-screen render targets at reduced resolution,
// blitted/upscaled to the full window surface via a secondary camera + sprite.
// See apply_render_scale() below for the canonical implementation.

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct LightingPlugin;

impl Plugin for LightingPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(DirectionalLightShadowMap { size: 4096 })
            .init_resource::<SunCycle>()
            .add_systems(Update, setup_lighting.in_set(crate::WorldSpawnSet))
            // NOTE: these Update systems are NOT chained and run in arbitrary
            // order within the Update stage. They all read camera position from
            // the previous frame (PlayerPlugin updates camera in a separate
            // chained set). This 1-frame lag is acceptable because lighting
            // changes are gradual and visually imperceptible at 60fps.
            .add_systems(Update, (update_sun_cycle, update_lantern_lights, apply_color_grading, apply_camera_settings, apply_render_pipeline_settings, apply_render_scale, adapt_shadows_for_fps_mode)
                .run_if(in_state(crate::GameState::Gameplay)));
    }
}

// ---------------------------------------------------------------------------
// Resources & components
// ---------------------------------------------------------------------------

/// Tracks the sun's position in a 10-minute day/night cycle.
#[derive(Resource)]
pub struct SunCycle {
    /// Current sun angle in radians (0 = sunrise, PI/2 = noon, PI = sunset, 2*PI = next sunrise).
    pub angle: f32,
    /// Day duration in seconds (default 600 = 10 minutes).
    pub day_duration: f32,
    /// Total elapsed time in seconds.
    pub total_time: f32,
}

impl Default for SunCycle {
    fn default() -> Self {
        Self {
            angle: 0.4, // pleasant morning angle
            day_duration: 600.0,
            total_time: 0.0,
        }
    }
}

/// Marker for the directional "sun" light entity.
#[derive(Component)]
pub struct SunMarker;

/// Marker for the visible sun disc in the sky.
#[derive(Component)]
pub struct SunDisc;

/// Marker for lantern-spawned point lights.
#[derive(Component)]
pub struct LanternLight;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Standard smoothstep interpolation, clamped to [0, 1].
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Linearly interpolate between two `Color` values (assumes linear RGB).
fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    let a = a.to_linear();
    let b = b.to_linear();
    Color::linear_rgba(
        a.red + (b.red - a.red) * t,
        a.green + (b.green - a.green) * t,
        a.blue + (b.blue - a.blue) * t,
        1.0,
    )
}

// ---------------------------------------------------------------------------
// Startup system
// ---------------------------------------------------------------------------

fn setup_lighting(mut commands: Commands, world_id: Res<crate::WorldInstanceId>) {
    commands.spawn((
        crate::WorldEntity,
        crate::WorldScoped(world_id.0),
        SunMarker,
        DirectionalLight {
            illuminance: 80_000.0,
            shadows_enabled: true,
            color: Color::linear_rgb(1.0, 0.95, 0.85),
            shadow_depth_bias: 0.02,
            shadow_normal_bias: 1.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.4, 0.0)),
        // 4-cascade shadow config for quality at varying distances.
        CascadeShadowConfigBuilder {
            num_cascades: 4,
            minimum_distance: 0.1,
            maximum_distance: 200.0,
            ..default()
        }
        .build(),
    ));
}

/// Despawn all lighting entities (sun, lanterns) when leaving gameplay.
fn teardown_lighting(
    mut commands: Commands,
    sun: Query<Entity, With<SunMarker>>,
    lanterns: Query<Entity, With<LanternLight>>,
) {
    for entity in &sun {
        commands.entity(entity).despawn();
    }
    for entity in &lanterns {
        commands.entity(entity).despawn();
    }
}

// ---------------------------------------------------------------------------
// Sun cycle system
// ---------------------------------------------------------------------------

fn update_sun_cycle(
    time: Res<Time>,
    mut sun_cycle: ResMut<SunCycle>,
    mut sun_query: Query<(&mut DirectionalLight, &mut Transform), With<SunMarker>>,
    mut ambient: ResMut<GlobalAmbientLight>,
    cameras_without_fog: Query<Entity, (With<Camera3d>, Without<DistanceFog>)>,
    mut fog_query: Query<&mut DistanceFog>,
    game_settings: Option<Res<crate::settings::GameSettings>>,
    dev: Res<crate::dev_tools::DevSettings>,
    mut commands: Commands,
) {
    let dt = time.delta_secs();

    // Day length is a live dev tweakable.
    sun_cycle.day_duration = dev.day_cycle_duration.max(1.0);

    // Advance angle
    sun_cycle.total_time += dt;
    sun_cycle.angle += (2.0 * PI / sun_cycle.day_duration) * dt;
    if sun_cycle.angle > 2.0 * PI {
        sun_cycle.angle -= 2.0 * PI;
    }

    let angle = sun_cycle.angle;

    // Sun direction (from Swift Renderer.swift line 935)
    let sun_direction = Vec3::new(angle.cos() * 0.5, angle.sin(), angle.cos() * 0.866).normalize();

    let sun_height = sun_direction.y;
    let day_factor = smoothstep(-0.15, 0.2, sun_height);

    // --- Directional light ---
    for (mut light, mut transform) in &mut sun_query {
        *transform = Transform::default().looking_to(-sun_direction, Vec3::Y);

        let shadow_mul = game_settings.as_ref().map(|s| s.shadow_intensity).unwrap_or(1.0);
        if sun_height > 0.0 {
            // Very low sun — just enough to cast shadows, not wash out colors
            light.illuminance = sun_height * 5_000.0 * shadow_mul;
            light.color = Color::WHITE;
        } else {
            light.illuminance = 200.0 * shadow_mul;
            light.color = Color::linear_rgb(0.5, 0.55, 0.8);
        }
    }

    // --- Ambient light: high so blocks show true color, only shadows darken them ---
    let brightness_mul = game_settings.as_ref().map(|s| s.brightness).unwrap_or(1.0);
    let day_ambient = Color::WHITE;
    let night_ambient = Color::linear_rgb(0.15, 0.17, 0.25);
    ambient.color = lerp_color(night_ambient, day_ambient, day_factor);
    ambient.brightness = (200.0 + (800.0 - 200.0) * day_factor) * brightness_mul;

    // --- Fog ---
    let day_fog_color = Color::linear_rgb(0.65, 0.75, 0.90);
    let night_fog_color = Color::linear_rgb(0.02, 0.03, 0.06);
    let fog_color = lerp_color(night_fog_color, day_fog_color, day_factor);

    // Fog density scales with render distance so blocks fade before the chunk edge.
    // At render_distance=5 (80 blocks), we want ~90% fog at ~70 blocks.
    // ExponentialSquared: fog = 1 - exp(-density * dist^2)
    // 0.9 = 1 - exp(-d * 70^2) => d = -ln(0.1) / 4900 ≈ 0.00047
    let rd_blocks = game_settings
        .as_ref()
        .map(|s| s.render_distance as f32 * 16.0)
        .unwrap_or(80.0);
    let fade_dist = rd_blocks * 0.85; // start fading at 85% of render distance
    let fog_density = 2.3 / (fade_dist * fade_dist); // -ln(0.1) ≈ 2.3

    // Attach fog to cameras that don't have it yet
    for entity in &cameras_without_fog {
        commands.entity(entity).insert(DistanceFog {
            color: fog_color,
            falloff: FogFalloff::ExponentialSquared {
                density: fog_density,
            },
            ..default()
        });
    }

    // Update existing fog
    for mut fog in &mut fog_query {
        fog.color = fog_color;
        fog.falloff = FogFalloff::ExponentialSquared {
            density: fog_density,
        };
    }
}

// ---------------------------------------------------------------------------
// Lantern lights system
// ---------------------------------------------------------------------------

/// Maximum number of lantern point lights active at once.
/// 64 is the practical limit before per-frame light culling causes
/// visible popping. Each PointLight costs ~0.1ms of GPU shadow time
/// (shadows_enabled=false mitigates this, but draw calls still add up).
const MAX_LANTERN_LIGHTS: usize = 64;

/// Distance (in blocks) within which lantern lights are spawned.
/// Matches the default render distance (5 chunks × 16 = 80 blocks)
/// so lanterns at the chunk edge are still lit.
const LANTERN_RANGE: f32 = 80.0;

fn update_lantern_lights(
    mut commands: Commands,
    chunk_manager: Option<Res<ChunkManager>>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    existing: Query<(Entity, &Transform), With<LanternLight>>,
    world_id: Res<crate::WorldInstanceId>,
    dev: Res<crate::dev_tools::DevSettings>,
    mut last_scan: Local<Option<(u64, IVec3)>>,
) {
    let Some(cam_gt) = cameras.iter().next() else {
        return;
    };
    let cam_pos = cam_gt.translation();

    // Lantern membership only changes when the modification set changes
    // (mods_version, bumped by set_block/clear_all) or the camera crosses a
    // block boundary (range checks). Skip the full modification scan +
    // O(lights × lanterns) matching otherwise.
    let mods_version = chunk_manager.as_ref().map(|cm| cm.mods_version).unwrap_or(0);
    let cam_block = cam_pos.floor().as_ivec3();
    let scan_key = (mods_version, cam_block);
    if *last_scan == Some(scan_key) {
        return;
    }
    *last_scan = Some(scan_key);

    // Collect lantern world positions from modifications within range.
    let mut lantern_positions: Vec<Vec3> = Vec::new();
    if let Some(cm) = &chunk_manager {
        for (pos, &block) in &cm.modifications {
            if block == BlockType::LANTERN {
                // Light at block center — shadows_enabled=false so light passes through
                // the lantern block's own faces to illuminate all surrounding blocks
                let wp = Vec3::new(pos.x as f32 + 0.5, pos.y as f32 + 0.5, pos.z as f32 + 0.5);
                if wp.distance(cam_pos) < LANTERN_RANGE {
                    lantern_positions.push(wp);
                }
            }
        }
    }

    // Despawn lights whose lantern block is no longer present/nearby.
    for (entity, transform) in &existing {
        let pos = transform.translation;
        if !lantern_positions.iter().any(|lp| lp.distance(pos) < 1.0) {
            commands.entity(entity).despawn();
        }
    }

    // Spawn lights for new lanterns (respect cap).
    let mut count = existing.iter().count();
    for lp in &lantern_positions {
        if count >= MAX_LANTERN_LIGHTS {
            break;
        }
        if existing
            .iter()
            .any(|(_, t)| t.translation.distance(*lp) < 1.0)
        {
            continue; // already has a light
        }
        commands.spawn((
            crate::WorldEntity,
            crate::WorldScoped(world_id.0),
            LanternLight,
            PointLight {
                color: Color::linear_rgb(1.0, 0.85, 0.55), // warm glow
                intensity: 30_000.0,  // subtle warm glow, not overpowering
                range: dev.lantern_radius,
                radius: 0.2,
                shadows_enabled: false,
                ..default()
            },
            Transform::from_translation(*lp),
        ));
        count += 1;
    }
}

// ---------------------------------------------------------------------------
// Color grading system — applies gamma and contrast from settings
// ---------------------------------------------------------------------------

fn apply_color_grading(
    game_settings: Option<Res<crate::settings::GameSettings>>,
    mut cameras: Query<&mut ColorGrading, With<Camera3d>>,
    added_cameras: Query<(), Added<Camera3d>>,
) {
    let Some(settings) = game_settings else {
        return;
    };
    // Only write when settings changed or a camera was just spawned
    // (world reload) — unconditional writes mark ColorGrading changed
    // every frame for no reason.
    if !settings.is_changed() && added_cameras.is_empty() {
        return;
    }

    for mut grading in &mut cameras {
        grading.global.post_saturation = 1.0;
        // Apply gamma and contrast to all sections
        let gamma = settings.gamma;
        let contrast = settings.contrast;
        grading.shadows.gamma = gamma;
        grading.shadows.contrast = contrast;
        grading.midtones.gamma = gamma;
        grading.midtones.contrast = contrast;
        grading.highlights.gamma = gamma;
        grading.highlights.contrast = contrast;
    }
}

// ---------------------------------------------------------------------------
// Camera settings system — applies exposure, tonemapping, and FOV from settings
// ---------------------------------------------------------------------------

fn apply_camera_settings(
    game_settings: Option<Res<crate::settings::GameSettings>>,
    mut cameras: Query<(Entity, &mut Projection), With<Camera3d>>,
    mut commands: Commands,
    added_cameras: Query<(), Added<Camera3d>>,
) {
    let Some(settings) = game_settings else {
        return;
    };
    // Only write when settings changed or a camera was just spawned (world
    // reload). The old unconditional path re-inserted Exposure + Tonemapping
    // components and dirtied Projection on every camera every frame.
    if !settings.is_changed() && added_cameras.is_empty() {
        return;
    }

    for (entity, mut projection) in &mut cameras {
        // FOV
        if let Projection::Perspective(ref mut persp) = *projection {
            persp.fov = settings.fov.to_radians();
        }

        // Exposure (setting is an offset from the default 9.7 EV100)
        commands.entity(entity).insert(bevy::camera::Exposure { ev100: 9.7 + settings.exposure });

        // Tonemapping
        let tonemapping = match settings.tonemapping.as_str() {
            "reinhard" => Tonemapping::Reinhard,
            "aces" => Tonemapping::AcesFitted,
            "agx" => Tonemapping::AgX,
            "tony" => Tonemapping::TonyMcMapface,
            "blender" => Tonemapping::BlenderFilmic,
            _ => Tonemapping::None,
        };
        commands.entity(entity).insert(tonemapping);
    }
}

// ---------------------------------------------------------------------------
// Render pipeline settings — applies AA, SSAO, SMAA live from settings
// ---------------------------------------------------------------------------

fn apply_render_pipeline_settings(
    game_settings: Option<Res<crate::settings::GameSettings>>,
    cameras: Query<Entity, With<Camera3d>>,
    mut commands: Commands,
    added_cameras: Query<(), Added<Camera3d>>,
) {
    let Some(settings) = game_settings else {
        return;
    };

    // Run when settings change, or when a camera was just spawned (world
    // reload) so the fresh camera doesn't rely solely on spawn_player's
    // initial insert. spawn_player applies the same values, so the
    // double-apply on the spawn frame is harmless.
    if !settings.is_changed() && added_cameras.is_empty() {
        return;
    }

    for entity in &cameras {
        // --- SSAO (must be handled before MSAA since it requires Msaa::Off) ---
        if settings.ssao_enabled {
            commands.entity(entity).insert(Msaa::Off);
            let quality = match settings.ssao_quality.as_str() {
                "low" => ScreenSpaceAmbientOcclusionQualityLevel::Low,
                "high" => ScreenSpaceAmbientOcclusionQualityLevel::High,
                "ultra" => ScreenSpaceAmbientOcclusionQualityLevel::Ultra,
                _ => ScreenSpaceAmbientOcclusionQualityLevel::Medium,
            };
            commands.entity(entity).insert(ScreenSpaceAmbientOcclusion {
                quality_level: quality,
                ..default()
            });
        } else {
            commands.entity(entity).remove::<ScreenSpaceAmbientOcclusion>();

            // --- Anti-aliasing (only when SSAO is off, since SSAO forces Msaa::Off) ---
            match settings.anti_aliasing.as_str() {
                "msaa2" => {
                    commands.entity(entity).insert(Msaa::Sample2);
                    commands.entity(entity).remove::<TemporalAntiAliasing>();
                }
                "msaa4" => {
                    commands.entity(entity).insert(Msaa::Sample4);
                    commands.entity(entity).remove::<TemporalAntiAliasing>();
                }
                "taa" => {
                    commands.entity(entity).insert(Msaa::Off);
                    commands.entity(entity).insert(TemporalAntiAliasing::default());
                }
                _ => {
                    commands.entity(entity).remove::<TemporalAntiAliasing>();
                    commands.entity(entity).insert(Msaa::Off);
                }
            }
        }

        // --- SMAA ---
        match settings.smaa_mode.as_str() {
            "low" => { commands.entity(entity).insert(Smaa { preset: SmaaPreset::Low }); }
            "medium" => { commands.entity(entity).insert(Smaa { preset: SmaaPreset::Medium }); }
            "high" => { commands.entity(entity).insert(Smaa { preset: SmaaPreset::High }); }
            "ultra" => { commands.entity(entity).insert(Smaa { preset: SmaaPreset::Ultra }); }
            _ => { commands.entity(entity).remove::<Smaa>(); }
        }
    }
}

/// Marker for the 2D camera that blits the scaled render target to the screen.
#[derive(Component)]
struct RenderScaleBlit;

/// Marker for the sprite that displays the scaled render target.
#[derive(Component)]
struct RenderScaleSprite;

/// Tracks the off-screen render target used for DSR-style render scaling.
#[derive(Resource)]
struct RenderScaleTarget {
    image: Handle<Image>,
    /// The physical size the image was last created at.
    current_size: UVec2,
}

/// DSR-style render scaling via the RenderTarget component.
///
/// PERFORMANCE WARNING:
///   Render-scale introduces an extra fullscreen blit pass (render to image,
///   then upscale to window via a 2D sprite). On fast GPUs (e.g. M4 Max) the
///   overhead of this extra pass can exceed the savings from rendering fewer
///   pixels, making it slower than native resolution. Render-scale should only
///   be enabled when render_scale < 1.0; at 1.0 the RTT pipeline is fully
///   torn down and the 3D camera renders directly to the window.
///
/// BEVY 0.18 API NOTE:
///   - `RenderTarget` is a Component (not a field on Camera). Import from
///     `bevy::camera::RenderTarget`, NOT from `bevy::render::camera`.
///   - To render to an image, insert `RenderTarget::Image(ImageRenderTarget { .. })`
///     on the camera entity. To restore window output, insert `RenderTarget::default()`.
///   - `Camera.output_mode` (CameraOutputMode) controls write/skip behavior,
///     not where the camera renders.
///
/// GUARDRAIL — Camera.viewport is NOT suitable for render-scale / DSR:
///   - Viewport reduces the visible output area (letterboxing / shrinking),
///     it does not render at a lower resolution and upscale.
///   - Render-scale is implemented here via an off-screen render target
///     rendered at a reduced resolution, then blitted/upscaled to the
///     full-size window surface by a secondary 2D camera + sprite.
fn apply_render_scale(
    game_settings: Option<Res<crate::settings::GameSettings>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(Entity, &RenderTarget), With<Camera3d>>,
    mut images: ResMut<Assets<Image>>,
    mut commands: Commands,
    target: Option<ResMut<RenderScaleTarget>>,
    blit_camera_q: Query<Entity, With<RenderScaleBlit>>,
    blit_sprite_q: Query<Entity, With<RenderScaleSprite>>,
) {
    let Some(settings) = game_settings else { return };
    let Ok(window) = windows.single() else { return };

    let scale = settings.render_scale.clamp(0.5, 1.0);

    // At full scale, render directly to the window with no intermediate target.
    // Early-out if already in window mode (no resource = nothing to tear down).
    if (scale - 1.0).abs() < f32::EPSILON {
        if target.is_none() {
            return;
        }
        #[cfg(debug_assertions)]
        bevy::log::warn!(
            "render_scale is 1.0 but RTT pipeline is still active — \
             tearing down to avoid unnecessary fullscreen blit pass"
        );
        // Tear down: remove blit camera, sprite, and render target resource.
        for entity in &blit_camera_q {
            commands.entity(entity).despawn();
        }
        for entity in &blit_sprite_q {
            commands.entity(entity).despawn();
        }
        commands.remove_resource::<RenderScaleTarget>();
        // Restore 3D camera to render to the window.
        for (entity, rt) in &cameras {
            if !matches!(rt, RenderTarget::Window(_)) {
                commands.entity(entity).insert(RenderTarget::default());
            }
        }
        return;
    }

    let phys_w = window.physical_width();
    let phys_h = window.physical_height();
    if phys_w == 0 || phys_h == 0 {
        return;
    }

    let scaled_w = ((phys_w as f32 * scale).round() as u32).max(1);
    let scaled_h = ((phys_h as f32 * scale).round() as u32).max(1);
    let scaled_size = UVec2::new(scaled_w, scaled_h);

    // Create or resize the render target image. Track whether anything changed
    // so we can skip redundant sprite/camera updates below.
    let mut needs_update = false;
    let image_handle = if let Some(mut existing) = target {
        if existing.current_size != scaled_size {
            if let Some(image) = images.get_mut(&existing.image) {
                image.resize(Extent3d {
                    width: scaled_w,
                    height: scaled_h,
                    depth_or_array_layers: 1,
                });
            }
            existing.current_size = scaled_size;
            needs_update = true;
        }
        existing.image.clone()
    } else {
        // First-time setup: create the render target image with linear filtering
        // so the upscaled blit is smooth (no pixelation).
        use bevy::image::{ImageFilterMode, ImageSampler, ImageSamplerDescriptor};

        let mut image = Image::new_fill(
            Extent3d {
                width: scaled_w,
                height: scaled_h,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &[0, 0, 0, 255],
            TextureFormat::Bgra8UnormSrgb,
            default(),
        );
        image.texture_descriptor.usage =
            TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_DST
            | TextureUsages::RENDER_ATTACHMENT;
        image.sampler = ImageSampler::Descriptor(ImageSamplerDescriptor {
            mag_filter: ImageFilterMode::Linear,
            min_filter: ImageFilterMode::Linear,
            ..default()
        });

        let handle = images.add(image);

        // Spawn the blit sprite (full-screen quad textured with the render target).
        // custom_size uses logical window dimensions because the 2D camera's
        // orthographic projection operates in logical coordinates.
        //
        // Deliberately NOT WorldEntity/WorldScoped: the blit sprite and camera
        // are render infrastructure (like MenuCamera), not world content.
        // Marking them world-scoped meant teardown despawned them while the
        // RenderScaleTarget resource survived — after a reload the early-out
        // below then skipped rebuilding the pipeline, silently disabling
        // render scaling.
        commands.spawn((
            crate::UiOnly,
            RenderScaleSprite,
            Sprite {
                image: handle.clone(),
                custom_size: Some(Vec2::new(window.width(), window.height())),
                ..default()
            },
        ));

        // Spawn a 2D camera to display the blit sprite, ordered after the 3D camera.
        // IsDefaultUiCamera ensures all UI renders on this camera (directly to
        // the window at full resolution), NOT on the 3D camera whose render
        // target is the scaled-down image. This keeps UI text crisp.
        commands.spawn((
            crate::UiOnly,
            RenderScaleBlit,
            Camera2d,
            Camera {
                order: 1,
                ..default()
            },
            IsDefaultUiCamera,
        ));

        commands.insert_resource(RenderScaleTarget {
            image: handle.clone(),
            current_size: scaled_size,
        });

        needs_update = true;
        handle
    };

    // Point the 3D camera at the render target image. Runs every frame (the
    // matches! check is cheap) rather than only when needs_update: a fresh
    // world camera spawned by a Play/Load reload must be retargeted even
    // though the render target itself is unchanged.
    // NOTE: RenderTarget does NOT implement PartialEq. Always use pattern
    // matching (matches!, if let) to inspect variants. Direct == / !=
    // comparisons will not compile.
    for (entity, rt) in &cameras {
        let already_set = matches!(
            rt,
            RenderTarget::Image(irt) if irt.handle == image_handle
        );
        if !already_set {
            commands.entity(entity).insert(RenderTarget::Image(ImageRenderTarget {
                handle: image_handle.clone(),
                scale_factor: 1.0,
            }));
        }
    }

    // Skip sprite updates if the render target already exists at the right size.
    if !needs_update {
        return;
    }

    // Update the blit sprite size to match the current logical window dimensions.
    let logical_size = Vec2::new(window.width(), window.height());
    for entity in &blit_sprite_q {
        commands.entity(entity).insert(Sprite {
            image: image_handle.clone(),
            custom_size: Some(logical_size),
            ..default()
        });
    }
}

/// Reduces shadow quality when fps_120_mode is enabled to hit higher frame
/// rates on tile-based GPUs (Apple Silicon). At 60 FPS the full shadow
/// pipeline is used; at 120 FPS targets, shadow map resolution is halved
/// and cascade count is reduced.
///
/// No lighting changes are applied for 60 FPS mode — this only activates
/// when fps_120_mode == true.
fn adapt_shadows_for_fps_mode(
    game_settings: Option<Res<crate::settings::GameSettings>>,
    dev: Res<crate::dev_tools::DevSettings>,
    mut shadow_map: ResMut<DirectionalLightShadowMap>,
    mut sun_query: Query<&mut CascadeShadowConfig, With<SunMarker>>,
    mut lanterns: Query<&mut PointLight, With<LanternLight>>,
) {
    let Some(settings) = game_settings else { return };
    if !settings.is_changed() {
        return;
    }

    if settings.fps_120_mode {
        // Halve shadow map resolution: 4096 → 2048
        if shadow_map.size != 2048 {
            shadow_map.size = 2048;
        }

        // Reduce cascades: 4 → 2 with shorter max distance
        for mut config in &mut sun_query {
            let reduced = CascadeShadowConfigBuilder {
                num_cascades: 2,
                minimum_distance: 0.1,
                maximum_distance: 100.0,
                ..default()
            }
            .build();
            *config = reduced;
        }

        // Reduce lantern light range to cut per-pixel evaluations
        for mut light in &mut lanterns {
            light.range = dev.lantern_radius * 0.66;
        }
    } else {
        // Restore full quality for 60 FPS mode
        if shadow_map.size != 4096 {
            shadow_map.size = 4096;
        }

        for mut config in &mut sun_query {
            let full = CascadeShadowConfigBuilder {
                num_cascades: 4,
                minimum_distance: 0.1,
                maximum_distance: 200.0,
                ..default()
            }
            .build();
            *config = full;
        }

        for mut light in &mut lanterns {
            light.range = dev.lantern_radius;
        }
    }
}
