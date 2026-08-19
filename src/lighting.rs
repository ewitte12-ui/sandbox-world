//! Sun, day/night cycle, and the coupling between the cycle and baked voxel light.
//!
//! Two lighting systems run at once and must not be confused:
//!
//! * **Baked voxel light** — a per-voxel sky/block flood fill (`voxel_light.rs`)
//!   whose 0..15 levels are written into mesh vertex attributes at mesh time.
//!   This is what makes caves dark and windows bright.
//! * **Real-time PBR light** — the `DirectionalLight` sun and ambient term,
//!   which shade surfaces normally.
//!
//! The sun moving must never trigger a remesh. It doesn't: meshes carry *raw*
//! light levels, and the day/night curve reaches them through a single uniform
//! (`light_params.x`) that `update_voxel_light_material` writes. That is the
//! only per-frame coupling between the cycle and the chunk meshes.

use std::f32::consts::PI;

use bevy::light::CascadeShadowConfigBuilder;
use bevy::pbr::DistanceFog;
use bevy::prelude::*;

use crate::chunk_manager::{BlockArrayMaterial, ChunkMaterial};
use crate::dev_tools::DevSettings;
use crate::{GameState, WorldEntity, WorldInstanceId, WorldScoped};

/// Position of the sun along its arc.
#[derive(Resource)]
pub struct SunCycle {
    /// Radians: 0 = sunrise, PI/2 = noon, PI = sunset.
    pub angle: f32,
    /// Seconds per full cycle. Mirrors `DevSettings::day_cycle_duration`.
    pub day_duration: f32,
    pub elapsed: f32,
}

impl Default for SunCycle {
    fn default() -> Self {
        Self {
            // A pleasant morning angle, so a fresh world opens in daylight.
            angle: 0.4,
            day_duration: 600.0,
            elapsed: 0.0,
        }
    }
}

impl SunCycle {
    /// Unit vector pointing from the world toward the sun.
    ///
    /// The arc is deliberately tilted rather than a straight overhead sweep —
    /// a sun that passes exactly through the zenith makes every wall of a
    /// cube-shaped world flip between fully lit and fully dark at the same
    /// moment, which reads as a strobe rather than a sunrise.
    pub fn direction(&self) -> Vec3 {
        Vec3::new(
            self.angle.cos() * 0.5,
            self.angle.sin(),
            self.angle.cos() * 0.866,
        )
        .normalize()
    }

    /// 0 at night, 1 in full day, smoothly blended across the horizon.
    ///
    /// The lower edge sits *below* the horizon so the light fades out during
    /// dusk instead of snapping off the instant the sun crosses y = 0.
    pub fn day_factor(&self) -> f32 {
        smoothstep(-0.15, 0.2, self.direction().y)
    }
}

/// The directional light acting as the sun.
#[derive(Component)]
pub struct Sun;

pub struct LightingPlugin;

impl Plugin for LightingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SunCycle>()
            .add_systems(OnEnter(GameState::Gameplay), spawn_sun)
            .add_systems(
                Update,
                (
                    advance_sun_cycle,
                    (
                        update_sun,
                        update_ambient,
                        update_voxel_light_material,
                        update_fog,
                    ),
                )
                    .chain()
                    .run_if(in_state(GameState::Gameplay)),
            );
    }
}

fn spawn_sun(mut commands: Commands, world_id: Res<WorldInstanceId>) {
    commands.spawn((
        Sun,
        DirectionalLight {
            shadow_maps_enabled: true,
            ..default()
        },
        // Cascades tuned to the render distance rather than Bevy's default,
        // which is sized for a much smaller scene and wastes most of its
        // resolution far behind the player.
        CascadeShadowConfigBuilder {
            num_cascades: 3,
            maximum_distance: 160.0,
            first_cascade_far_bound: 24.0,
            ..default()
        }
        .build(),
        Transform::default(),
        WorldEntity,
        WorldScoped(world_id.0),
    ));
}

fn advance_sun_cycle(time: Res<Time>, dev: Res<DevSettings>, mut cycle: ResMut<SunCycle>) {
    cycle.day_duration = dev.day_cycle_duration.max(1.0);
    cycle.elapsed += time.delta_secs();
    cycle.angle = (cycle.angle + (2.0 * PI / cycle.day_duration) * time.delta_secs()) % (2.0 * PI);
}

fn update_sun(
    cycle: Res<SunCycle>,
    mut sun: Single<(&mut DirectionalLight, &mut Transform), With<Sun>>,
) {
    let (ref mut light, ref mut transform) = *sun;
    let direction = cycle.direction();

    // The light travels *toward* the scene, hence the negation.
    **transform = Transform::default().looking_to(-direction, Vec3::Y);

    if direction.y > 0.0 {
        // Scaled by height rather than held constant, so dawn and dusk are dim
        // and low-angled instead of a full-strength light lying on its side.
        light.illuminance = direction.y * 5_000.0;
        light.color = Color::WHITE;
    } else {
        // Moonlight: faint, and tinted cold so night reads as night even
        // though the voxel sky-light floor keeps it navigable.
        light.illuminance = 120.0;
        light.color = Color::srgb(0.6, 0.7, 1.0);
    }
}

fn update_ambient(cycle: Res<SunCycle>, mut ambient: ResMut<GlobalAmbientLight>) {
    let day = cycle.day_factor();
    ambient.brightness = 12.0 + 140.0 * day;
    ambient.color = Color::srgb(0.45 + 0.35 * day, 0.50 + 0.35 * day, 0.65 + 0.20 * day);
}

/// Pushes the day/night curve into the chunk material's uniform.
///
/// This is the entire cost of a moving sun: one `Vec4` write, no remesh, no
/// per-chunk work. The value is quantised so that a slowly drifting sun does
/// not mark the material changed every single frame — without it, the asset is
/// dirtied 60 times a second and its bind group re-prepared for a change no one
/// can see.
fn update_voxel_light_material(
    cycle: Res<SunCycle>,
    dev: Res<DevSettings>,
    chunk_material: Option<Res<ChunkMaterial>>,
    mut materials: ResMut<Assets<BlockArrayMaterial>>,
    mut last: Local<Option<Vec4>>,
) {
    let Some(material_handle) = chunk_material else {
        return;
    };

    // Sky light never reaches zero: the moon/star floor keeps outdoor areas
    // navigable at night, which `guardrails/03` requires of any recovery path.
    let night = dev.voxel_sky_night.clamp(0.0, 1.0);
    let sun_intensity = night + (1.0 - night) * cycle.day_factor();

    let params = Vec4::new(
        (sun_intensity * 256.0).round() / 256.0,
        if dev.voxel_lighting { 1.0 } else { 0.0 },
        dev.voxel_light_min.clamp(0.0, 1.0),
        0.0,
    );

    if *last == Some(params) {
        return;
    }
    *last = Some(params);

    if let Some(mut material) = materials.get_mut(&material_handle.handle) {
        material.extension.light_params = params;
    }
}

/// Fades the far edge of the loaded world into the sky colour, so chunk
/// streaming appears as haze rather than as a hard wall of popping geometry.
fn update_fog(
    cycle: Res<SunCycle>,
    clear_color: Res<ClearColor>,
    mut commands: Commands,
    cameras: Query<Entity, (With<Camera3d>, Without<DistanceFog>)>,
    mut fogs: Query<&mut DistanceFog>,
) {
    // Matching fog to the sky is what sells the illusion; a fixed grey would
    // glow against a night sky.
    let color = clear_color.0;
    let day = cycle.day_factor();
    let start = 90.0 + 30.0 * day;

    for entity in &cameras {
        commands.entity(entity).insert(DistanceFog {
            color,
            falloff: FogFalloff::Linear {
                start,
                end: start + 90.0,
            },
            ..default()
        });
    }
    for mut fog in &mut fogs {
        fog.color = color;
        fog.falloff = FogFalloff::Linear {
            start,
            end: start + 90.0,
        };
    }
}

pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cycle_at(angle: f32) -> SunCycle {
        SunCycle { angle, ..default() }
    }

    #[test]
    fn noon_is_full_day_and_midnight_is_night() {
        assert!(
            cycle_at(PI / 2.0).day_factor() > 0.99,
            "noon should be full day"
        );
        assert!(
            cycle_at(3.0 * PI / 2.0).day_factor() < 0.01,
            "midnight should be night"
        );
    }

    #[test]
    fn sun_is_above_the_horizon_during_the_day() {
        assert!(cycle_at(PI / 2.0).direction().y > 0.9);
        assert!(cycle_at(3.0 * PI / 2.0).direction().y < -0.9);
    }

    /// The cycle must be continuous across the wrap point — a jump would show
    /// up as the sky and every surface flickering once per day.
    #[test]
    fn day_factor_is_continuous_across_the_wrap() {
        let before = cycle_at(2.0 * PI - 0.001).day_factor();
        let after = cycle_at(0.0).day_factor();
        assert!((before - after).abs() < 0.01, "{before} vs {after}");
    }

    #[test]
    fn day_factor_is_monotonic_through_sunrise() {
        let mut previous = 0.0;
        // Sweep from below the horizon up to noon.
        for step in 0..=50 {
            let angle = -0.4 + (PI / 2.0 + 0.4) * (step as f32 / 50.0);
            let factor = cycle_at(angle).day_factor();
            assert!(factor >= previous - 1e-6, "dipped at angle {angle}");
            previous = factor;
        }
    }
}
