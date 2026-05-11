use bevy::prelude::*;

use crate::terrain::terrain_height_at;

/// Plugin that places procedural trees around the player using box entities.
pub struct TreePlugin;

impl Plugin for TreePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_tree_manager)
            .add_systems(Update, update_trees.run_if(in_state(crate::GameState::Gameplay)));
    }
}

// --- Constants ---

/// Grid spacing between potential tree positions.
const TREE_STEP: f32 = 18.0;
/// Radius around the player in which trees are generated.
const TREE_RADIUS: f32 = 220.0;
/// Minimum terrain height for tree placement.
const TREE_MIN_Y: f32 = 3.4;
/// Maximum terrain height for tree placement.
const TREE_MAX_Y: f32 = 92.4;
/// Placement probability threshold (lower = more trees).
const TREE_PLACEMENT_THRESHOLD: f32 = 0.28;
/// Distance the player must move before trees are regenerated.
const TREE_UPDATE_DISTANCE: f32 = 50.0;
/// Maximum number of trees to spawn.
const MAX_TREES: usize = 500;

// --- Components ---

/// Marker component for tree part entities.
#[derive(Component)]
pub struct TreeMarker;

// --- Resources ---

/// Manages tree entity spawning and shared rendering handles.
#[derive(Resource)]
pub struct TreeManager {
    last_update_pos: Vec3,
    tree_entities: Vec<Entity>,
    // Shared mesh/material handles
    box_mesh: Handle<Mesh>,
    trunk_material: Handle<StandardMaterial>,
    canopy1_material: Handle<StandardMaterial>,
    canopy2_material: Handle<StandardMaterial>,
    canopy3_material: Handle<StandardMaterial>,
}

// --- Hash ---

/// Deterministic hash for tree placement, matching the Swift implementation.
#[allow(clippy::excessive_precision)]
fn h1(px: f32, pz: f32) -> f32 {
    let v = px * 127.1 + pz * 311.7;
    let s = v.sin() * 43758.5453;
    s - s.floor()
}

// --- Systems ---

/// Startup system: create shared mesh and material handles, insert TreeManager resource.
fn setup_tree_manager(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let box_mesh = meshes.add(Cuboid::new(1.0, 1.0, 1.0));

    let trunk_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.36, 0.20, 0.07),
        ..default()
    });
    let canopy1_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.10, 0.36, 0.08),
        ..default()
    });
    let canopy2_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.13, 0.44, 0.10),
        ..default()
    });
    let canopy3_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.17, 0.52, 0.13),
        ..default()
    });

    commands.insert_resource(TreeManager {
        last_update_pos: Vec3::new(f32::MAX, f32::MAX, f32::MAX),
        tree_entities: Vec::new(),
        box_mesh,
        trunk_material,
        canopy1_material,
        canopy2_material,
        canopy3_material,
    });
}

/// Update system: regenerate trees when the player moves far enough.
fn update_trees(
    mut commands: Commands,
    mut tree_manager: ResMut<TreeManager>,
    camera_query: Query<&GlobalTransform, With<Camera3d>>,
    world_id: Res<crate::WorldInstanceId>,
) {
    let Ok(cam_transform) = camera_query.single() else {
        return;
    };
    let player_pos = cam_transform.translation();

    let dx = player_pos.x - tree_manager.last_update_pos.x;
    let dz = player_pos.z - tree_manager.last_update_pos.z;
    if (dx * dx + dz * dz).sqrt() < TREE_UPDATE_DISTANCE {
        return;
    }

    // Despawn all existing tree entities
    for entity in tree_manager.tree_entities.drain(..) {
        if let Ok(mut cmd) = commands.get_entity(entity) {
            cmd.despawn();
        }
    }

    tree_manager.last_update_pos = player_pos;

    // Generate trees on a grid around the player
    let half = TREE_RADIUS;
    let i_min = ((player_pos.x - half) / TREE_STEP).floor() as i32;
    let i_max = ((player_pos.x + half) / TREE_STEP).ceil() as i32;
    let j_min = ((player_pos.z - half) / TREE_STEP).floor() as i32;
    let j_max = ((player_pos.z + half) / TREE_STEP).ceil() as i32;

    let mut tree_count: usize = 0;

    for i in i_min..=i_max {
        if tree_count >= MAX_TREES {
            break;
        }
        for j in j_min..=j_max {
            if tree_count >= MAX_TREES {
                break;
            }

            let cx = i as f32 * TREE_STEP;
            let cz = j as f32 * TREE_STEP;

            // 5 hash values for this cell
            let r1 = h1(cx * 0.013 + 42.1, cz * 0.013 + 13.7);
            let r2 = h1(cx * 0.013 + 17.3, cz * 0.013 + 88.2);
            let r3 = h1(cx * 0.013 + 99.1, cz * 0.013 + 5.7);
            let r4 = h1(cx * 0.013 + 55.3, cz * 0.013 + 72.1);
            let r5 = h1(cx * 0.013 + 181.3, cz * 0.013 + 37.9);

            // Placement chance
            if r1 >= TREE_PLACEMENT_THRESHOLD {
                continue;
            }

            // Jittered world position
            let wx = cx + (r2 - 0.5) * TREE_STEP * 0.85;
            let wz = cz + (r3 - 0.5) * TREE_STEP * 0.85;

            // Distance check
            let dist_x = wx - player_pos.x;
            let dist_z = wz - player_pos.z;
            if dist_x * dist_x + dist_z * dist_z > TREE_RADIUS * TREE_RADIUS {
                continue;
            }

            // Terrain height check
            let ground_y = terrain_height_at(wx, wz);
            if !(TREE_MIN_Y..TREE_MAX_Y).contains(&ground_y) {
                continue;
            }

            let s = 1.0 + r4 * 1.5; // scale 1.0 – 2.5
            let yaw = r5 * std::f32::consts::PI * 2.0;

            // Geometry
            let trunk_h = 4.0 * s;
            let trunk_r = 0.5 * s;

            let trunk_cy = ground_y + trunk_h * 0.5;
            let canopy1_cy = ground_y + trunk_h * 0.72;
            let canopy2_cy = ground_y + trunk_h + 1.0 * s;
            let canopy3_cy = ground_y + trunk_h + 2.4 * s;

            let rotation = Quat::from_rotation_y(yaw);
            let mesh = tree_manager.box_mesh.clone();

            // Trunk
            let trunk = commands
                .spawn((
                    crate::WorldEntity,
                    crate::WorldScoped(world_id.0),
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(tree_manager.trunk_material.clone()),
                    Transform::from_xyz(wx, trunk_cy, wz)
                        .with_rotation(rotation)
                        .with_scale(Vec3::new(trunk_r * 2.0, trunk_h, trunk_r * 2.0)),
                    TreeMarker,
                ))
                .id();

            // Canopy 1
            let canopy1 = commands
                .spawn((
                    crate::WorldEntity,
                    crate::WorldScoped(world_id.0),
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(tree_manager.canopy1_material.clone()),
                    Transform::from_xyz(wx, canopy1_cy, wz)
                        .with_rotation(rotation)
                        .with_scale(Vec3::new(4.4 * s, 1.6 * s, 4.4 * s)),
                    TreeMarker,
                ))
                .id();

            // Canopy 2
            let canopy2 = commands
                .spawn((
                    crate::WorldEntity,
                    crate::WorldScoped(world_id.0),
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(tree_manager.canopy2_material.clone()),
                    Transform::from_xyz(wx, canopy2_cy, wz)
                        .with_rotation(rotation)
                        .with_scale(Vec3::new(3.2 * s, 1.6 * s, 3.2 * s)),
                    TreeMarker,
                ))
                .id();

            // Canopy 3
            let canopy3 = commands
                .spawn((
                    crate::WorldEntity,
                    crate::WorldScoped(world_id.0),
                    Mesh3d(mesh.clone()),
                    MeshMaterial3d(tree_manager.canopy3_material.clone()),
                    Transform::from_xyz(wx, canopy3_cy, wz)
                        .with_rotation(rotation)
                        .with_scale(Vec3::new(2.0 * s, 1.8 * s, 2.0 * s)),
                    TreeMarker,
                ))
                .id();

            tree_manager
                .tree_entities
                .extend_from_slice(&[trunk, canopy1, canopy2, canopy3]);

            tree_count += 1;
        }
    }
}
