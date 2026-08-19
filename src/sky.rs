//! Procedural sky colour and drifting cloud plates.
//!
//! # Why this is hand-rolled rather than `bevy_light::Atmosphere`
//!
//! Bevy 0.19 ships a physically-based atmosphere, and it would look better than
//! a lerped gradient. It is not usable here: `AtmospherePlugin` requires compute
//! shaders and storage textures, and refuses to load without them —
//! WebGL2 has neither. Since the web build is a first-class target (see
//! `web/build.sh`), adopting it would mean the browser silently losing its sky
//! while the desktop build kept one. Revisit if the web build ever moves to
//! WebGPU.
//!
//! # Parallax without per-frame work
//!
//! Clouds are children of a single `BackgroundRoot`. Exactly one entity is
//! written each frame — the root, positioned at `camera_xz * PARALLAX_FACTOR` —
//! and Bevy's transform propagation moves every cloud from there. No system
//! ever touches an individual cloud after spawn, and there is no velocity or
//! timer state that can drift out of sync if the schedule pauses.

use bevy::light::NotShadowCaster;
use bevy::prelude::*;

use crate::lighting::{SunCycle, smoothstep};
use crate::{GameState, WorldEntity, WorldInstanceId, WorldScoped};

/// Fraction of camera motion the clouds follow. 0 pins them to the world,
/// 1 glues them to the camera; 0.05 reads as "very far away".
const PARALLAX_FACTOR: f32 = 0.05;

const CLOUD_COUNT: usize = 30;

/// Marker for a cloud plate. Nothing queries these after spawn.
#[derive(Component)]
pub struct Cloud;

/// The single camera-tracking parent of every background plate.
#[derive(Component)]
pub struct BackgroundRoot;

pub struct SkyPlugin;

impl Plugin for SkyPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(Color::linear_rgb(0.10, 0.18, 0.48)))
            .add_systems(OnEnter(GameState::Gameplay), spawn_clouds)
            .add_systems(
                Update,
                (update_sky_color, track_background_root).run_if(in_state(GameState::Gameplay)),
            );
    }
}

/// Drives `ClearColor` from the sun's height.
///
/// `update_fog` in `lighting.rs` reads this same colour, so haze at the edge of
/// the loaded world always matches the sky behind it.
fn update_sky_color(cycle: Res<SunCycle>, mut clear_color: ResMut<ClearColor>) {
    let sun_height = cycle.direction().y;
    let day = smoothstep(-0.15, 0.2, sun_height);

    let day_zenith = Vec3::new(0.10, 0.18, 0.48);
    let night_zenith = Vec3::new(0.005, 0.008, 0.025);
    let mut sky = night_zenith.lerp(day_zenith, day);

    // Warm the sky while the sun is near the horizon, from either side. The
    // quartic keeps the tint tight around sunrise/sunset instead of smearing
    // orange across the whole afternoon.
    if (-0.1..0.3).contains(&sun_height) {
        let amount = (1.0 - sun_height.abs()).powi(4) * 0.6;
        sky += Vec3::new(0.4, 0.15, 0.05) * amount;
    }

    clear_color.0 = Color::linear_rgb(sky.x.max(0.0), sky.y.max(0.0), sky.z.max(0.0));
}

fn track_background_root(
    camera: Option<Single<&GlobalTransform, With<Camera3d>>>,
    root: Option<Single<&mut Transform, With<BackgroundRoot>>>,
) {
    let (Some(camera), Some(mut root)) = (camera, root) else {
        return;
    };
    let position = camera.translation();
    // Y is untouched: cloud altitude lives in each plate's local offset, so the
    // ceiling stays put as the player climbs.
    root.translation.x = position.x * PARALLAX_FACTOR;
    root.translation.z = position.z * PARALLAX_FACTOR;
}

fn spawn_clouds(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    world_id: Res<WorldInstanceId>,
) {
    let plate = meshes.add(Plane3d::default().mesh().size(1.0, 1.0));

    // One shared material: the plates differ only in transform, so 30 copies
    // would be 30 bind groups for one appearance.
    let material = materials.add(StandardMaterial {
        base_color: Color::linear_rgba(0.45, 0.46, 0.48, 0.5),
        alpha_mode: AlphaMode::Add,
        unlit: true,
        // Plates are single-sided planes and the player will fly under them.
        cull_mode: None,
        ..default()
    });

    commands
        .spawn((
            BackgroundRoot,
            Transform::default(),
            Visibility::Inherited,
            WorldEntity,
            WorldScoped(world_id.0),
        ))
        .with_children(|root| {
            for i in 0..CLOUD_COUNT {
                let f = i as f32;
                let x = (hash(f * 13.7, f * 31.1) - 0.5) * 800.0;
                let z = (hash(f * 17.3, f * 88.2) - 0.5) * 800.0;
                let y = 350.0 + hash(f * 55.3, f * 72.1) * 100.0;
                let width = 30.0 + hash(f * 99.1, f * 5.7) * 60.0;

                root.spawn((
                    Cloud,
                    Mesh3d(plate.clone()),
                    MeshMaterial3d(material.clone()),
                    Transform::from_xyz(x, y, z).with_scale(Vec3::new(width, 1.0, width * 0.7)),
                    NotShadowCaster,
                    WorldEntity,
                    WorldScoped(world_id.0),
                ));
            }
        });
}

/// Deterministic value hash, so a world's clouds are identical every run.
#[allow(clippy::excessive_precision)]
fn hash(x: f32, z: f32) -> f32 {
    let s = (x * 127.1 + z * 311.7).sin() * 43758.5453;
    s - s.floor()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic_and_unit_range() {
        for i in 0..64 {
            let f = i as f32;
            let a = hash(f * 13.7, f * 31.1);
            assert_eq!(a, hash(f * 13.7, f * 31.1), "hash must be stable");
            assert!((0.0..1.0).contains(&a), "hash out of range: {a}");
        }
    }
}
