use bevy::light::NotShadowCaster;
use bevy::prelude::*;

use crate::lighting::SunCycle;

/// Plugin providing a procedural sky gradient (via ClearColor) and static
/// cloud plates with camera-driven parallax.
///
/// # Parallax architecture
///
/// Clouds are children of a single `BackgroundRoot` entity. The root's
/// transform is set each frame to `camera_xz * PARALLAX_FACTOR` — this is
/// the ONLY per-frame mutation in the entire sky system, and it touches
/// exactly one infrastructure entity (the root), never the 30 cloud plates.
///
/// Bevy's hierarchical transform propagation then shifts every cloud's
/// `GlobalTransform` by the root offset automatically. Because the factor
/// is small (0.05), clouds drift at 5 % of camera speed, producing a
/// gentle parallax against nearby terrain and chunks.
///
/// If Update systems are paused, the root stays at its last position and
/// clouds remain correctly placed — no timer or velocity state to go stale.
pub struct SkyPlugin;

/// Parallax depth factor applied to camera XZ position.
/// 0.0 = clouds locked to world origin (no tracking).
/// 1.0 = clouds move 1:1 with camera (no parallax).
/// 0.05 = clouds drift at 5 % of camera speed (subtle parallax).
const PARALLAX_FACTOR: f32 = 0.05;

impl Plugin for SkyPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(Color::linear_rgb(0.10, 0.18, 0.48)))
            .add_systems(Update, spawn_clouds.in_set(crate::WorldSpawnSet))
            .add_systems(Update, (update_sky, track_background_root)
                .run_if(in_state(crate::GameState::Gameplay)));
    }
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Marker for a cloud entity. Clouds are static plates parented to
/// `BackgroundRoot`. No per-frame system queries or mutates them — their
/// `GlobalTransform` changes only via Bevy's hierarchical propagation
/// when the root moves.
#[derive(Component)]
pub struct Cloud;

/// Camera-tracking root for all background plates. ONE entity, positioned
/// each frame at `camera_xz * PARALLAX_FACTOR`. All cloud entities are
/// children — their global transforms shift automatically via Bevy's
/// transform propagation without any system iterating over them.
#[derive(Component)]
pub struct BackgroundRoot;

// ---------------------------------------------------------------------------
// Systems
// ---------------------------------------------------------------------------

/// Update `ClearColor` to reflect the sky zenith colour based on sun height.
fn update_sky(sun_cycle: Option<Res<SunCycle>>, mut clear_color: ResMut<ClearColor>) {
    let angle = sun_cycle.map(|s| s.angle).unwrap_or(0.4);

    let sun_dir = Vec3::new(angle.cos() * 0.5, angle.sin(), angle.cos() * 0.866).normalize();
    let sun_height = sun_dir.y;

    let day_factor = smoothstep(-0.15, 0.2, sun_height);

    let day_zenith = Vec3::new(0.10, 0.18, 0.48);
    let night_zenith = Vec3::new(0.005, 0.008, 0.025);
    let zenith = night_zenith.lerp(day_zenith, day_factor);

    let mut sky = zenith;
    if sun_height > -0.1 && sun_height < 0.3 {
        let sunset_amount = (1.0 - sun_height.abs()).powi(4) * 0.6;
        let sunset_tint = Vec3::new(0.4, 0.15, 0.05);
        sky = sky.lerp(sky + sunset_tint, sunset_amount);
    }

    clear_color.0 = Color::linear_rgb(sky.x.max(0.0), sky.y.max(0.0), sky.z.max(0.0));
}

/// Track the active camera and position the BackgroundRoot at a fraction
/// of the camera's XZ position. This is the sole per-frame mutation in
/// the sky system — it writes ONE entity (the root), never the clouds.
///
/// Math:
///   root.translation = (cam.x * F, 0, cam.z * F)
///   where F = PARALLAX_FACTOR (0.05)
///
/// Each cloud's GlobalTransform becomes:
///   (root.translation + cloud.local_translation)
///
/// When the camera moves by `delta`, clouds shift by `delta * F` in world
/// space, producing parallax: nearby terrain scrolls fast, distant clouds
/// drift slowly.
fn track_background_root(
    cameras: Query<&GlobalTransform, (With<Camera3d>, With<crate::WorldEntity>)>,
    mut roots: Query<&mut Transform, With<BackgroundRoot>>,
) {
    let Ok(cam_gt) = cameras.single() else { return };
    let Ok(mut root_tf) = roots.single_mut() else { return };

    let cam_pos = cam_gt.translation();
    root_tf.translation = Vec3::new(
        cam_pos.x * PARALLAX_FACTOR,
        0.0, // Y stays at origin — cloud altitude is baked into local offsets
        cam_pos.z * PARALLAX_FACTOR,
    );
    // No rotation — clouds keep world-space orientation regardless of
    // which way the camera faces.
}

/// Spawn the BackgroundRoot and 30 static cloud plates as its children.
///
/// Cloud local transforms are deterministic (fixed offsets and scales).
/// After this system runs, no system ever queries `Cloud` entities again.
/// Parallax comes entirely from the root's position being updated by
/// `track_background_root`.
fn spawn_clouds(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    world_id: Res<crate::WorldInstanceId>,
) {
    // Spawn the root at origin. track_background_root will position it.
    let root = commands.spawn((
        crate::WorldEntity,
        crate::WorldScoped(world_id.0),
        crate::BackgroundPlate,
        BackgroundRoot,
        Transform::default(),
        Visibility::Inherited,
    )).id();

    let mesh_handle = meshes.add(Plane3d::default().mesh().size(1.0, 1.0));

    let cloud_count = 30;
    for i in 0..cloud_count {
        let fi = i as f32;
        let h1 = hash_f32(fi * 13.7, fi * 31.1);
        let h2 = hash_f32(fi * 17.3, fi * 88.2);
        let h3 = hash_f32(fi * 55.3, fi * 72.1);
        let h4 = hash_f32(fi * 99.1, fi * 5.7);

        // Deterministic local offset relative to the root.
        // XZ: spread over 800×800 area.
        // Y: 350-450 (absolute altitude baked as local offset from root Y=0).
        let x = (h1 - 0.5) * 800.0;
        let z = (h2 - 0.5) * 800.0;
        let y = 350.0 + h3 * 100.0;
        let scale_xz = 30.0 + h4 * 60.0;
        let scale_z = scale_xz * 0.7;

        let material = materials.add(StandardMaterial {
            base_color: Color::linear_rgba(0.45, 0.46, 0.48, 0.5),
            alpha_mode: AlphaMode::Add,
            unlit: true,
            cull_mode: None,
            ..default()
        });

        let cloud = commands.spawn((
            crate::WorldEntity,
            crate::WorldScoped(world_id.0),
            crate::BackgroundPlate,
            Cloud,
            Mesh3d(mesh_handle.clone()),
            MeshMaterial3d(material),
            Transform::from_xyz(x, y, z).with_scale(Vec3::new(scale_xz, 1.0, scale_z)),
            NotShadowCaster,
        )).id();

        commands.entity(root).add_child(cloud);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[allow(clippy::excessive_precision)]
fn hash_f32(px: f32, pz: f32) -> f32 {
    let v = px * 127.1 + pz * 311.7;
    let s = v.sin() * 43758.5453;
    s - s.floor()
}
