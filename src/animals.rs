//! Wandering animals.
//!
//! Each animal is a single entity: a glTF scene root carrying an [`Animal`]
//! component, positioned by its own `Transform`. There is no side table of
//! animal state indexed in parallel with entities — the ECS *is* the storage,
//! so an animal cannot end up half-despawned with a stale index pointing at it.
//!
//! Three of the six models ship with a rigged walk cycle (horse, raccoon,
//! dinosaur); the rest are static meshes. Rather than build procedural skeletons
//! for the static ones, they get a gait bob applied to the scene root — enough
//! that they do not slide across the ground like furniture, and a fraction of
//! the machinery.

use bevy::prelude::*;
// Bevy 0.19 loads glTF scenes as `WorldAsset`, spawned via `WorldAssetRoot`.
// This is the 0.18 `Scene`/`SceneRoot` pair after `bevy_scene` was split into
// `bevy_world_serialization`.
use bevy::world_serialization::{WorldAssetRoot, WorldInstanceReady};
use rand::Rng;

use crate::block_types::BlockType;
use crate::chunk_manager::ChunkManager;
use crate::dev_tools::DevSettings;
use crate::terrain;
use crate::{GameState, WorldEntity, WorldInstanceId, WorldScoped};

/// How far from the origin animals are scattered at world start.
const SPAWN_RADIUS: f32 = 90.0;

/// Animals stay within this distance of the origin, turning back at the edge.
/// Without it they random-walk out of the loaded world and stop being part of
/// the game.
const WANDER_RADIUS: f32 = 220.0;

/// Seconds between wander decisions, randomised per animal within this range.
const DECISION_INTERVAL: (f32, f32) = (1.5, 5.0);

/// How quickly an animal turns toward its chosen heading, in radians/second.
/// Low enough to read as an animal turning rather than a turret snapping.
const TURN_RATE: f32 = 2.5;

/// Vertical follow rate for terrain, in blocks/second. Animals lerp onto the
/// ground instead of teleporting, so a step up reads as a stride.
const GROUND_FOLLOW_RATE: f32 = 8.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Species {
    Squirrel,
    Dog,
    Horse,
    Raccoon,
    Chicken,
    Dinosaur,
}

impl Species {
    pub const ALL: [Species; 6] = [
        Species::Squirrel,
        Species::Dog,
        Species::Horse,
        Species::Raccoon,
        Species::Chicken,
        Species::Dinosaur,
    ];

    fn def(self) -> SpeciesDef {
        // Scales convert each model's native units to blocks, and were found by
        // eye against a 1-block cube. y_offset lifts the model so its feet, not
        // its origin, rest on the ground.
        match self {
            // (min_speed, max_speed) are in blocks/second, all below the
            // player's 22 so animals can be approached and watched. The
            // dinosaur is the exception: it keeps pace, which is the point.
            Species::Squirrel => SpeciesDef {
                model: "animals/squirrel.glb",
                scale: 0.18,
                y_offset: 0.724 * 0.18,
                speed: (3.5, 8.0),
                gait_hz: 10.0,
                clip: None,
            },
            Species::Dog => SpeciesDef {
                model: "animals/dog.glb",
                scale: 0.14,
                y_offset: 0.04 * 0.14,
                speed: (2.0, 5.5),
                gait_hz: 6.5,
                clip: None,
            },
            Species::Horse => SpeciesDef {
                model: "animals/horse.glb",
                scale: 1.8,
                y_offset: 0.004 * 1.8,
                speed: (3.0, 8.0),
                gait_hz: 4.5,
                clip: Some(0),
            },
            Species::Raccoon => SpeciesDef {
                model: "animals/raccoon.glb",
                scale: 0.18,
                y_offset: 0.05 * 0.18,
                speed: (1.2, 3.0),
                gait_hz: 5.5,
                clip: Some(0),
            },
            Species::Chicken => SpeciesDef {
                model: "animals/chicken.glb",
                scale: 0.13,
                y_offset: 0.03 * 0.13,
                speed: (1.5, 4.0),
                gait_hz: 8.0,
                clip: None,
            },
            Species::Dinosaur => SpeciesDef {
                model: "animals/dinosaur.glb",
                scale: 1.6,
                y_offset: 0.006 * 1.6,
                speed: (4.0, 10.0),
                gait_hz: 5.0,
                clip: Some(0),
            },
        }
    }
}

struct SpeciesDef {
    model: &'static str,
    scale: f32,
    y_offset: f32,
    speed: (f32, f32),
    /// Gait cycle rate for the bob applied to models with no rigged animation.
    gait_hz: f32,
    /// Index of the walk clip inside the glTF, when the model has one.
    clip: Option<usize>,
}

/// A wandering animal. The entity's `Transform` is its position and facing;
/// this component holds only what the transform cannot express.
#[derive(Component)]
pub struct Animal {
    pub species: Species,
    /// Heading the animal is turning toward, in radians.
    target_yaw: f32,
    /// Current ground speed in blocks/second; zero while idling.
    speed: f32,
    /// Countdown to the next wander decision.
    decision_timer: f32,
    /// Phase accumulator for the gait bob, so animals are not in lockstep.
    gait_phase: f32,
    /// Vertical offset applied on top of terrain height, from the gait bob.
    bob: f32,
}

/// Graph handles for the species whose models carry a walk clip. Built once at
/// startup; the observer below attaches them as each scene finishes loading.
#[derive(Resource)]
struct AnimalAnimations {
    /// (species, graph, node) for every species with a rigged clip.
    entries: Vec<(Species, Handle<AnimationGraph>, AnimationNodeIndex)>,
}

pub struct AnimalPlugin;

impl Plugin for AnimalPlugin {
    fn build(&self, app: &mut App) {
        // DEPENDENCY: ChunkManagerPlugin must be registered first — wandering
        // queries ChunkManager for ground height (falling back to terrain
        // generation for chunks that have not streamed in yet).
        app.add_systems(
            OnEnter(GameState::Gameplay),
            (load_animations, spawn_animals).chain(),
        )
        .add_systems(
            Update,
            (wander, animal_probe).run_if(in_state(GameState::Gameplay)),
        )
        .add_observer(attach_animation);
    }
}

fn load_animations(
    mut commands: Commands,
    assets: Res<AssetServer>,
    mut graphs: ResMut<Assets<AnimationGraph>>,
) {
    let mut entries = Vec::new();
    for species in Species::ALL {
        let def = species.def();
        let Some(clip_index) = def.clip else { continue };
        let clip = assets.load(GltfAssetLabel::Animation(clip_index).from_asset(def.model));
        let (graph, node) = AnimationGraph::from_clip(clip);
        entries.push((species, graphs.add(graph), node));
    }
    commands.insert_resource(AnimalAnimations { entries });
}

fn spawn_animals(
    mut commands: Commands,
    assets: Res<AssetServer>,
    dev: Res<DevSettings>,
    world_id: Res<WorldInstanceId>,
) {
    let mut rng = rand::thread_rng();

    for _ in 0..dev.animal_count {
        let species = Species::ALL[rng.gen_range(0..Species::ALL.len())];
        let def = species.def();

        // Scatter on a disc rather than a square, so density is even and there
        // is no visible corner clustering.
        let angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let radius = SPAWN_RADIUS * rng.gen_range(0.0_f32..1.0).sqrt();
        let (x, z) = (angle.cos() * radius, angle.sin() * radius);
        let y = terrain::surface_y(x.floor() as i32, z.floor() as i32) as f32 + 1.0;

        let yaw = rng.gen_range(0.0..std::f32::consts::TAU);

        commands.spawn((
            Animal {
                species,
                target_yaw: yaw,
                speed: 0.0,
                decision_timer: rng.gen_range(0.0..DECISION_INTERVAL.1),
                // Randomised so a herd does not bob in unison.
                gait_phase: rng.gen_range(0.0..std::f32::consts::TAU),
                bob: 0.0,
            },
            WorldAssetRoot(assets.load(GltfAssetLabel::Scene(0).from_asset(def.model))),
            Transform::from_xyz(x, y + def.y_offset, z)
                .with_rotation(Quat::from_rotation_y(yaw))
                .with_scale(Vec3::splat(def.scale)),
            WorldEntity,
            WorldScoped(world_id.0),
        ));
    }
}

/// Starts the walk cycle on each animal's `AnimationPlayer` once its glTF scene
/// has finished spawning.
///
/// This reacts to [`WorldInstanceReady`] rather than to `Add<AnimationPlayer>`,
/// and that distinction is the entire bug it fixes. `WorldInstanceSpawner`
/// spawns every scene entity first (`spawn_sync_internal`) and only afterwards
/// attaches the scene root to the entity that asked for it
/// (`set_instance_parent_sync`). An `Add<AnimationPlayer>` observer therefore
/// fires while the player is still an orphan subtree: walking up its `ChildOf`
/// chain never reaches the `Animal`, the species lookup returns `None`, and the
/// clip is silently never attached — every rigged animal stands still forever.
/// `WorldInstanceReady` is triggered *after* parenting, on the animal itself,
/// so the hierarchy is whole by the time we go looking.
fn attach_animation(
    ready: On<WorldInstanceReady>,
    mut commands: Commands,
    animations: Option<Res<AnimalAnimations>>,
    children: Query<&Children>,
    animals: Query<&Animal>,
    mut players: Query<&mut AnimationPlayer>,
) {
    let Some(animations) = animations else { return };

    // Scenes are spawned for more than animals; only animals declare clips.
    let Ok(animal) = animals.get(ready.entity) else {
        return;
    };
    let Some((_, graph, node)) = animations
        .entries
        .iter()
        .find(|(candidate, _, _)| *candidate == animal.species)
    else {
        return;
    };

    // The player sits at whatever depth the model's rig puts it, so search the
    // whole subtree rather than assuming a fixed shape.
    for entity in children.iter_descendants(ready.entity) {
        if let Ok(mut player) = players.get_mut(entity) {
            // Both components must land on the *same* entity: `advance_animations`
            // queries `(&mut AnimationPlayer, &AnimationGraphHandle)` and skips
            // any player that has no graph.
            commands
                .entity(entity)
                .insert(AnimationGraphHandle(graph.clone()));
            player.play(*node).repeat();
        }
    }
}

/// Unit heading for a yaw, in the convention the models are authored in.
///
/// These glTF models face **+Z**, not Bevy's camera-style -Z forward: they are
/// Blender exports, and Blender's +Y "front" becomes +Z under the glTF axis
/// conversion. Driving movement along -Z made every animal walk backwards.
/// It also inverted the wander-radius containment in `wander`, which solves for
/// this same convention — so an animal reaching the edge was steered *away*
/// from the origin and wandered out of the loaded world.
fn heading_vector(yaw: f32) -> Vec3 {
    Vec3::new(yaw.sin(), 0.0, yaw.cos())
}

fn wander(
    time: Res<Time>,
    chunks: Res<ChunkManager>,
    mut animals: Query<(&mut Animal, &mut Transform)>,
) {
    let dt = time.delta_secs();
    let mut rng = rand::thread_rng();

    for (mut animal, mut transform) in &mut animals {
        let def = animal.species.def();

        // --- Decide ---
        animal.decision_timer -= dt;
        if animal.decision_timer <= 0.0 {
            animal.decision_timer = rng.gen_range(DECISION_INTERVAL.0..DECISION_INTERVAL.1);
            // Roughly a third of decisions are to stop and look around, which
            // is what stops the herd from looking like it is on rails.
            if rng.gen_bool(0.3) {
                animal.speed = 0.0;
            } else {
                animal.speed = rng.gen_range(def.speed.0..def.speed.1);
                animal.target_yaw = rng.gen_range(0.0..std::f32::consts::TAU);
            }
        }

        // Turn back toward the origin at the edge of the wander area, rather
        // than letting a random walk carry animals out of the loaded world.
        let from_origin = Vec2::new(transform.translation.x, transform.translation.z);
        if from_origin.length() > WANDER_RADIUS {
            animal.target_yaw = (-from_origin.x).atan2(-from_origin.y);
            animal.speed = animal.speed.max(def.speed.0);
        }

        // --- Turn ---
        let (current_yaw, ..) = transform.rotation.to_euler(EulerRot::YXZ);
        // Shortest angular path, so an animal never spins the long way round.
        let delta = wrap_angle(animal.target_yaw - current_yaw);
        let yaw = current_yaw + delta.clamp(-TURN_RATE * dt, TURN_RATE * dt);

        // --- Move ---
        let forward = heading_vector(yaw);
        let step = forward * animal.speed * dt;
        let proposed = transform.translation + step;

        // --- Follow the ground ---
        let ground = ground_height(proposed.x, proposed.z, &chunks);

        // A step the animal cannot climb is a wall: stop rather than walk into
        // it, and pick a new heading at the next decision.
        let rise = ground - (transform.translation.y - animal.bob - def.y_offset);
        if rise > 1.2 {
            animal.speed = 0.0;
            animal.decision_timer = animal.decision_timer.min(0.3);
        } else {
            transform.translation.x = proposed.x;
            transform.translation.z = proposed.z;
        }

        // --- Gait ---
        // Models with a rigged clip are animated by the AnimationPlayer; the
        // static ones get a bob here so they are not rigid while moving.
        animal.bob = if def.clip.is_none() && animal.speed > 0.0 {
            animal.gait_phase += dt * def.gait_hz;
            // Amplitude scales with the model so a horse does not bob like a
            // squirrel. Absolute value gives a hop rather than a float.
            (animal.gait_phase.sin().abs()) * 0.08 * def.scale.max(0.5)
        } else {
            0.0
        };

        let target_y = ground + def.y_offset + animal.bob;
        let current = transform.translation.y;
        transform.translation.y =
            current + (target_y - current) * (GROUND_FOLLOW_RATE * dt).min(1.0);
        transform.rotation = Quat::from_rotation_y(yaw);
    }
}

/// Verification hook. With `SW2_ANIMAL_PROBE=1`, reports once after the scenes
/// have loaded whether each rigged species actually ended up playing its clip.
/// Inert without the env var.
///
/// This exists because the failure it checks for is silent and untestable from
/// the outside: the clip attaches (or does not) depending on whether the glTF
/// hierarchy happens to be parented at the instant an observer fires, and a
/// unit test of the species table passes either way. The counts below
/// distinguish "no player was ever found" from "player found but no graph" from
/// "graph attached but not playing" — three different bugs that look identical
/// in the window.
fn animal_probe(
    time: Res<Time>,
    mut fired: Local<bool>,
    animals: Query<(Entity, &Animal)>,
    children: Query<&Children>,
    players: Query<(&AnimationPlayer, Option<&AnimationGraphHandle>)>,
) {
    if *fired || std::env::var("SW2_ANIMAL_PROBE").is_err() {
        return;
    }
    let at = std::env::var("SW2_ANIMAL_PROBE_AT")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(10.0);
    if time.elapsed_secs() < at {
        return;
    }
    *fired = true;

    for species in Species::ALL {
        let expects_clip = species.def().clip.is_some();
        let (mut total, mut found, mut with_graph, mut playing) = (0, 0, 0, 0);

        for (entity, animal) in &animals {
            if animal.species != species {
                continue;
            }
            total += 1;
            for descendant in children.iter_descendants(entity) {
                if let Ok((player, graph)) = players.get(descendant) {
                    found += 1;
                    if graph.is_some() {
                        with_graph += 1;
                    }
                    if player.playing_animations().next().is_some() {
                        playing += 1;
                    }
                }
            }
        }

        info!(
            "animal probe: {species:?} count={total} rigged={expects_clip} \
             players={found} with_graph={with_graph} playing={playing}"
        );
        if expects_clip && total > 0 && playing == 0 {
            warn!("animal probe: {species:?} declares a clip but nothing is playing");
        }
    }
}

/// Ground height under a world XZ, preferring loaded chunk data and falling
/// back to the terrain function for chunks that have not streamed in.
fn ground_height(x: f32, z: f32, chunks: &ChunkManager) -> f32 {
    let (bx, bz) = (x.floor() as i32, z.floor() as i32);
    let surface = terrain::surface_y(bx, bz);

    // Scan a window around the predicted surface so player-built structures and
    // excavations are respected, not just the original terrain.
    for by in (surface - 20..=surface + 4).rev() {
        if chunks.block_at(IVec3::new(bx, by, bz)) != BlockType::AIR {
            return by as f32 + 1.0;
        }
    }
    surface as f32 + 1.0
}

/// Maps an angle into `[-PI, PI)`, so a turn always takes the shorter way round.
///
/// The half-open end is on the negative side because `rem_euclid` yields
/// `[0, TAU)`. Exactly half a turn therefore comes back as `-PI`; either sign is
/// equally correct there, since both directions are the same distance.
fn wrap_angle(a: f32) -> f32 {
    (a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn every_species_has_a_model_and_sane_tuning() {
        for species in Species::ALL {
            let def = species.def();
            assert!(def.model.ends_with(".glb"), "{species:?}");
            assert!(def.scale > 0.0, "{species:?} scale");
            assert!(
                def.speed.0 > 0.0 && def.speed.0 < def.speed.1,
                "{species:?} speed range {:?}",
                def.speed
            );
        }
    }

    /// Only the models that actually ship a clip may claim one — asking for a
    /// missing glTF animation yields a handle that never resolves, and the
    /// animal silently stands still forever.
    #[test]
    fn only_rigged_species_declare_a_clip() {
        use Species::*;
        for species in Species::ALL {
            let expected = matches!(species, Horse | Raccoon | Dinosaur);
            assert_eq!(
                species.def().clip.is_some(),
                expected,
                "{species:?} clip declaration disagrees with the shipped model"
            );
        }
    }

    /// The models face +Z, so travel must be along +Z. A -Z heading here is
    /// precisely the "every animal walks backwards" bug.
    #[test]
    fn heading_follows_the_plus_z_the_models_face() {
        let at_rest = heading_vector(0.0);
        assert!((at_rest - Vec3::Z).length() < 1e-5, "{at_rest:?}");

        // A quarter turn puts the heading on +X.
        let quarter = heading_vector(PI / 2.0);
        assert!((quarter - Vec3::X).length() < 1e-5, "{quarter:?}");

        assert!(heading_vector(1.234).length() - 1.0 < 1e-5);
    }

    /// The containment rule in `wander` solves for a heading using the same
    /// convention `heading_vector` supplies. If the two ever disagree the
    /// boundary pushes animals *out* of the world instead of turning them back
    /// — invisible until an animal actually reaches the edge, and then the
    /// whole population quietly leaves.
    #[test]
    fn boundary_heading_turns_an_animal_back_toward_the_origin() {
        for (x, z) in [
            (WANDER_RADIUS + 5.0, 0.0),
            (0.0, WANDER_RADIUS + 5.0),
            (-WANDER_RADIUS - 1.0, 0.0),
            (0.0, -WANDER_RADIUS - 1.0),
            (-WANDER_RADIUS, -WANDER_RADIUS),
            (WANDER_RADIUS * 0.8, -WANDER_RADIUS * 0.9),
        ] {
            let from_origin = Vec2::new(x, z);
            // Same expression as the containment branch of `wander`.
            let yaw = (-from_origin.x).atan2(-from_origin.y);
            let step = heading_vector(yaw);

            let before = from_origin.length();
            let after = Vec2::new(x + step.x, z + step.z).length();
            assert!(
                after < before,
                "at ({x}, {z}) the boundary heading moved the animal from \
                 {before} to {after} blocks from the origin"
            );
        }
    }

    #[test]
    fn wrap_angle_maps_into_half_open_turn() {
        for (input, expected) in [
            (0.0, 0.0),
            (PI / 2.0, PI / 2.0),
            (-PI / 2.0, -PI / 2.0),
            // Just past half a turn must come back as a small negative angle,
            // so a turning animal takes the short way round.
            (PI + 0.1, -PI + 0.1),
            // Exactly half a turn lands on the closed end of the range.
            (3.0 * PI, -PI),
        ] {
            let got = wrap_angle(input);
            assert!(
                (got - expected).abs() < 1e-4,
                "wrap_angle({input}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn wrap_angle_always_lands_in_range() {
        let mut a = -40.0_f32;
        while a < 40.0 {
            let w = wrap_angle(a);
            assert!(
                (-PI - 1e-4..PI + 1e-4).contains(&w),
                "wrap_angle({a}) = {w}"
            );
            a += 0.37;
        }
    }
}
