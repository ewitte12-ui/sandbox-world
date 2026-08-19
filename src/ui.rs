//! Title menu, settings screen, and the in-game HUD.
//!
//! # Menu state policy (hard rule)
//!
//! The menu is **not** a paused game world. It is a clean, UI-only state: no
//! world entities, no simulation, no world rendering. Everything spawned into
//! the world carries [`WorldScoped`], and leaving `Gameplay` despawns all of it.
//! A chunk or an animal surviving into the menu is a correctness bug, not a
//! cosmetic one — it means the next world will be built on top of the last one.
//!
//! # Widgets
//!
//! Interaction comes from `bevy_ui_widgets`, which is headless: `Button`,
//! `Slider` and `Checkbox` supply pointer, drag and keyboard behaviour and emit
//! `Activate` / `ValueChange` notifications, while every pixel of styling is
//! ours. That split is the reason this file is a few hundred lines rather than a
//! few thousand — none of it is re-implementing drag maths or focus handling.

use bevy::prelude::*;
// `Checked` is a shared UI interaction state, not a widget-crate type.
use bevy::ui::Checked;
use bevy::ui_widgets::{
    Activate, Checkbox, Slider, SliderRange, SliderThumb, SliderValue, ValueChange,
};

use crate::player::{Player, SelectedBlock};
use crate::settings::GameSettings;
use crate::{GameState, WorldScoped};

// Palette. Deliberately muted so the world reads as the bright thing and the
// menu as chrome around it.
const PANEL: Color = Color::srgba(0.07, 0.08, 0.11, 0.92);
const INK: Color = Color::srgb(0.86, 0.88, 0.93);
const INK_DIM: Color = Color::srgb(0.55, 0.59, 0.67);
const ACCENT: Color = Color::srgb(0.36, 0.62, 0.45);
const CONTROL: Color = Color::srgb(0.16, 0.18, 0.23);
const CONTROL_HOVER: Color = Color::srgb(0.22, 0.25, 0.31);

/// Which menu screen is showing. Only meaningful in [`GameState::Menu`].
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum MenuScreen {
    #[default]
    Main,
    Settings,
}

/// Marker for every entity belonging to the menu UI, including its camera.
/// Never carries [`WorldScoped`] — menu and world lifetimes are independent.
#[derive(Component)]
struct MenuUi;

/// Marker for the in-game HUD root.
#[derive(Component)]
struct Hud;

#[derive(Component)]
struct FpsLabel;

#[derive(Component)]
struct BlockLabel;

/// What a menu button does when activated.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum MenuAction {
    /// Resume the saved world.
    Continue,
    /// Discard any save and generate a fresh world.
    NewWorld,
    OpenSettings,
    BackToMain,
    /// Native only: a browser tab cannot close itself, so the button that
    /// constructs this is compiled out on wasm.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    Quit,
}

/// Which setting a slider or checkbox is bound to.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum SettingBinding {
    RenderDistance,
    Fov,
    Vsync,
    Clouds,
}

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MenuScreen>()
            .add_systems(OnEnter(GameState::Menu), spawn_menu)
            // Settings are only editable from the menu, so leaving it is the
            // moment to persist them. Writing on every change instead would mean
            // a file write per frame for the whole time a slider is dragged.
            .add_systems(OnExit(GameState::Menu), (despawn_menu, persist_settings))
            // TEARDOWN FENCE: the world is destroyed on the way out of
            // Gameplay, before the menu exists, so no world entity can ever be
            // observed from the menu.
            .add_systems(OnExit(GameState::Gameplay), teardown_world)
            .add_systems(OnEnter(GameState::Gameplay), spawn_hud)
            .add_systems(OnExit(GameState::Gameplay), despawn_hud)
            .add_systems(
                Update,
                rebuild_menu_on_screen_change.run_if(in_state(GameState::Menu)),
            )
            .add_systems(
                Update,
                (update_hud, leave_to_menu).run_if(in_state(GameState::Gameplay)),
            );
    }
}

// ---------------------------------------------------------------------------
// World teardown
// ---------------------------------------------------------------------------

/// Despawns every world entity on the way out of `Gameplay`.
///
/// Scoped by the [`WorldScoped`] marker rather than by sweeping all entities.
/// That matters more in Bevy 0.19 than it used to: resources are stored as
/// components on entities now, so a blanket `Query<Entity>` sweep would delete
/// `GameSettings`, `ChunkManager` and every other resource along with the world.
/// A marker query cannot, because resource entities never carry the marker.
pub(crate) fn teardown_world(
    mut commands: Commands,
    mut world_id: ResMut<crate::WorldInstanceId>,
    // Roots only. `despawn` takes the whole hierarchy with it, so scoped
    // children — cloud plates under their parallax root, glTF nodes under an
    // animal — are already gone by the time the loop would reach them, and
    // despawning them again logs a "that entity is invalid" warning per child.
    roots: Query<Entity, (With<WorldScoped>, Without<ChildOf>)>,
) {
    let mut count = 0;
    for entity in &roots {
        commands.entity(entity).despawn();
        count += 1;
    }
    // A fresh id for the next session, so anything still holding the old one —
    // an in-flight chunk task, say — is identifiably stale rather than quietly
    // adopted by the new world.
    world_id.0 += 1;
    info!("world teardown: despawned {count} world entities");
}

// ---------------------------------------------------------------------------
// Menu
// ---------------------------------------------------------------------------

fn spawn_menu(mut commands: Commands, screen: Res<MenuScreen>, settings: Res<GameSettings>) {
    // The menu needs its own camera: the world's 3D camera belongs to the
    // player, which the teardown above has just despawned.
    commands.spawn((MenuUi, Camera2d));

    commands
        .spawn((
            MenuUi,
            Node {
                width: percent(100),
                height: percent(100),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(Color::srgb(0.05, 0.06, 0.09)),
        ))
        .with_children(|root| {
            root.spawn((
                Node {
                    width: px(420),
                    padding: UiRect::all(px(28)),
                    flex_direction: FlexDirection::Column,
                    row_gap: px(14),
                    border_radius: BorderRadius::all(px(10)),
                    ..default()
                },
                BackgroundColor(PANEL),
            ))
            .with_children(|panel| {
                panel.spawn((
                    Text::new(crate::GAME_NAME),
                    TextFont {
                        font_size: FontSize::Px(34.0),
                        ..default()
                    },
                    TextColor(INK),
                ));

                match *screen {
                    MenuScreen::Main => build_main_menu(panel),
                    MenuScreen::Settings => build_settings_menu(panel, &settings),
                }
            });
        });
}

fn build_main_menu(panel: &mut ChildSpawnerCommands) {
    panel.spawn((
        Text::new("A voxel sandbox"),
        TextFont {
            font_size: FontSize::Px(15.0),
            ..default()
        },
        TextColor(INK_DIM),
        Node {
            margin: UiRect::bottom(px(10)),
            ..default()
        },
    ));

    // Continue only appears when there is something to continue, so the button
    // never promises a world that does not exist.
    if crate::save_load::save_exists() {
        menu_button(panel, "Continue", MenuAction::Continue);
    }
    menu_button(panel, "New World", MenuAction::NewWorld);
    menu_button(panel, "Settings", MenuAction::OpenSettings);
    #[cfg(not(target_arch = "wasm32"))]
    menu_button(panel, "Quit", MenuAction::Quit);
}

fn build_settings_menu(panel: &mut ChildSpawnerCommands, settings: &GameSettings) {
    slider_row(
        panel,
        "Render distance",
        SettingBinding::RenderDistance,
        settings.render_distance as f32,
        2.0..=16.0,
    );
    slider_row(
        panel,
        "Field of view",
        SettingBinding::Fov,
        settings.fov,
        60.0..=110.0,
    );
    checkbox_row(panel, "VSync", SettingBinding::Vsync, settings.vsync);
    checkbox_row(
        panel,
        "Clouds",
        SettingBinding::Clouds,
        settings.clouds_enabled,
    );
    menu_button(panel, "Back", MenuAction::BackToMain);
}

fn menu_button(parent: &mut ChildSpawnerCommands, label: &str, action: MenuAction) {
    parent
        .spawn((
            bevy::ui_widgets::Button,
            action,
            Node {
                padding: UiRect::axes(px(16), px(11)),
                justify_content: JustifyContent::Center,
                border_radius: BorderRadius::all(px(6)),
                ..default()
            },
            BackgroundColor(CONTROL),
            children![(
                Text::new(label),
                TextFont {
                    font_size: FontSize::Px(18.0),
                    ..default()
                },
                TextColor(INK),
            )],
        ))
        // Hover feedback is ours to draw: the widget crate supplies behaviour,
        // not appearance.
        .observe(
            |over: On<Pointer<Over>>, mut colors: Query<&mut BackgroundColor>| {
                if let Ok(mut color) = colors.get_mut(over.entity) {
                    color.0 = CONTROL_HOVER;
                }
            },
        )
        .observe(
            |out: On<Pointer<Out>>, mut colors: Query<&mut BackgroundColor>| {
                if let Ok(mut color) = colors.get_mut(out.entity) {
                    color.0 = CONTROL;
                }
            },
        )
        .observe(on_button_activated);
}

fn on_button_activated(
    activate: On<Activate>,
    actions: Query<&MenuAction>,
    mut screen: ResMut<MenuScreen>,
    mut next_state: ResMut<NextState<GameState>>,
    mut pending: ResMut<crate::save_load::PendingLoad>,
    settings: Res<GameSettings>,
    mut exit: MessageWriter<AppExit>,
) {
    let Ok(action) = actions.get(activate.entity) else {
        return;
    };
    match action {
        MenuAction::Continue => {
            // Read the save now, while the menu still exists. The world is
            // built on the way into Gameplay and consumes whatever is here.
            pending.0 = crate::save_load::read_save();
            next_state.set(GameState::Gameplay);
        }
        MenuAction::NewWorld => {
            // Discard the old world outright, otherwise the autosave on the way
            // back to the menu would merge this session's edits into it.
            crate::save_load::delete_save();
            pending.0 = None;
            next_state.set(GameState::Gameplay);
        }
        MenuAction::OpenSettings => *screen = MenuScreen::Settings,
        MenuAction::BackToMain => *screen = MenuScreen::Main,
        MenuAction::Quit => {
            // Quitting skips OnExit(Menu), so persist here too.
            settings.save();
            exit.write(AppExit::Success);
        }
    }
}

fn slider_row(
    parent: &mut ChildSpawnerCommands,
    label: &str,
    binding: SettingBinding,
    value: f32,
    range: std::ops::RangeInclusive<f32>,
) {
    parent
        .spawn((Node {
            flex_direction: FlexDirection::Column,
            row_gap: px(5),
            ..default()
        },))
        .with_children(|row| {
            row.spawn((
                Text::new(format!("{label}: {value:.0}")),
                TextFont {
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(INK_DIM),
                SettingLabel(binding),
            ));

            row.spawn((
                Slider::default(),
                binding,
                SliderValue(value),
                SliderRange::new(*range.start(), *range.end()),
                Node {
                    height: px(16),
                    align_items: AlignItems::Center,
                    border_radius: BorderRadius::all(px(8)),
                    ..default()
                },
                BackgroundColor(CONTROL),
                children![(
                    SliderThumb,
                    Node {
                        width: px(16),
                        height: px(16),
                        border_radius: BorderRadius::all(px(8)),
                        ..default()
                    },
                    BackgroundColor(ACCENT),
                )],
            ))
            .observe(on_slider_changed);
        });
}

/// Marks the text that mirrors a slider's current value.
#[derive(Component)]
struct SettingLabel(SettingBinding);

fn on_slider_changed(
    change: On<ValueChange<f32>>,
    mut commands: Commands,
    bindings: Query<&SettingBinding>,
    mut settings: ResMut<GameSettings>,
    mut labels: Query<(&mut Text, &SettingLabel)>,
) {
    let Ok(binding) = bindings.get(change.source) else {
        return;
    };
    commands
        .entity(change.source)
        .insert(SliderValue(change.value));

    let (name, shown) = match binding {
        SettingBinding::RenderDistance => {
            settings.render_distance = change.value.round() as i32;
            ("Render distance", settings.render_distance as f32)
        }
        SettingBinding::Fov => {
            settings.fov = change.value;
            ("Field of view", settings.fov)
        }
        // Checkboxes do not emit ValueChange<f32>.
        _ => return,
    };

    for (mut text, label) in &mut labels {
        if label.0 == *binding {
            **text = format!("{name}: {shown:.0}");
        }
    }
}

fn checkbox_row(
    parent: &mut ChildSpawnerCommands,
    label: &str,
    binding: SettingBinding,
    checked: bool,
) {
    let mut row = parent.spawn((
        Checkbox,
        binding,
        Node {
            column_gap: px(10),
            align_items: AlignItems::Center,
            padding: UiRect::vertical(px(4)),
            ..default()
        },
        children![
            (
                CheckboxBox,
                Node {
                    width: px(18),
                    height: px(18),
                    border_radius: BorderRadius::all(px(4)),
                    ..default()
                },
                BackgroundColor(if checked { ACCENT } else { CONTROL }),
            ),
            (
                Text::new(label),
                TextFont {
                    font_size: FontSize::Px(15.0),
                    ..default()
                },
                TextColor(INK),
            ),
        ],
    ));
    if checked {
        row.insert(Checked);
    }
    row.observe(on_checkbox_changed);
}

/// The drawn square of a checkbox; recoloured when the state flips.
#[derive(Component)]
struct CheckboxBox;

fn on_checkbox_changed(
    change: On<ValueChange<bool>>,
    mut commands: Commands,
    bindings: Query<&SettingBinding>,
    children: Query<&Children>,
    mut boxes: Query<&mut BackgroundColor, With<CheckboxBox>>,
    mut settings: ResMut<GameSettings>,
) {
    let Ok(binding) = bindings.get(change.source) else {
        return;
    };
    match binding {
        SettingBinding::Vsync => settings.vsync = change.value,
        SettingBinding::Clouds => settings.clouds_enabled = change.value,
        _ => return,
    }

    // `Checked` is a marker component, so the widget's state and its appearance
    // are updated together here rather than polled every frame.
    if change.value {
        commands.entity(change.source).insert(Checked);
    } else {
        commands.entity(change.source).remove::<Checked>();
    }
    for child in children.iter_descendants(change.source) {
        if let Ok(mut color) = boxes.get_mut(child) {
            color.0 = if change.value { ACCENT } else { CONTROL };
        }
    }
}

/// Rebuilds the panel when the user moves between Main and Settings.
fn rebuild_menu_on_screen_change(
    mut commands: Commands,
    screen: Res<MenuScreen>,
    settings: Res<GameSettings>,
    existing: Query<Entity, With<MenuUi>>,
) {
    if !screen.is_changed() || screen.is_added() {
        return;
    }
    for entity in &existing {
        commands.entity(entity).despawn();
    }
    spawn_menu(commands.reborrow(), screen, settings);
}

fn persist_settings(settings: Res<GameSettings>) {
    settings.save();
}

fn despawn_menu(mut commands: Commands, menu: Query<Entity, With<MenuUi>>) {
    for entity in &menu {
        commands.entity(entity).despawn();
    }
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------

fn spawn_hud(mut commands: Commands) {
    commands.spawn((
        Hud,
        Node {
            position_type: PositionType::Absolute,
            left: px(12),
            top: px(10),
            flex_direction: FlexDirection::Column,
            row_gap: px(3),
            ..default()
        },
        children![
            (
                Text::new("-- fps"),
                TextFont {
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(INK_DIM),
                FpsLabel,
            ),
            (
                Text::new(""),
                TextFont {
                    font_size: FontSize::Px(14.0),
                    ..default()
                },
                TextColor(INK),
                BlockLabel,
            ),
        ],
    ));
}

fn despawn_hud(mut commands: Commands, hud: Query<Entity, With<Hud>>) {
    for entity in &hud {
        commands.entity(entity).despawn();
    }
}

fn update_hud(
    time: Res<Time>,
    selected: Res<SelectedBlock>,
    player: Option<Single<&Player>>,
    mut smoothed: Local<f32>,
    mut fps: Query<&mut Text, (With<FpsLabel>, Without<BlockLabel>)>,
    mut block: Query<&mut Text, (With<BlockLabel>, Without<FpsLabel>)>,
) {
    // Exponential smoothing: a raw per-frame reciprocal is unreadable, and this
    // needs no history buffer.
    let dt = time.delta_secs();
    if dt > 0.0 {
        let instant = 1.0 / dt;
        *smoothed = if *smoothed == 0.0 {
            instant
        } else {
            *smoothed * 0.9 + instant * 0.1
        };
    }
    for mut text in &mut fps {
        **text = format!("{:.0} fps", *smoothed);
    }

    for mut text in &mut block {
        let holding = player
            .as_ref()
            .map(|p| p.selected_block.name())
            .unwrap_or("");
        **text = match selected.0 {
            Some(target) => format!("holding {holding}   |   looking at {}", target.kind.name()),
            None => format!("holding {holding}"),
        };
    }
}

/// Keys that leave gameplay for the menu.
///
/// `M` is carried over from the 0.18 build, where it was the menu key; Escape
/// is the conventional one and is what this build shipped with. Both are
/// accepted rather than picking a winner, since either is a reasonable reach.
///
/// Note this is a full *exit*, not the 0.18 pause overlay: the 0.18 menu opened
/// without a state transition and left the world resident underneath, whereas
/// here the teardown fence runs on the way out, so the world is gone by the
/// time the menu appears and returns via Continue.
const MENU_KEYS: [KeyCode; 2] = [KeyCode::Escape, KeyCode::KeyM];

/// Leaves gameplay for the menu. Teardown runs on the state exit, so the world
/// is gone by the time the menu appears.
fn leave_to_menu(
    keys: Res<ButtonInput<KeyCode>>,
    mut screen: ResMut<MenuScreen>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if keys.any_just_pressed(MENU_KEYS) {
        *screen = MenuScreen::Main;
        next_state.set(GameState::Menu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;

    #[derive(Resource, PartialEq, Debug)]
    struct SurvivesTeardown(u32);

    /// Both menu keys must reach the menu. M is the 0.18 binding and Escape is
    /// what this build shipped with; a regression that silently drops either
    /// one is invisible until someone reaches for the key they remember.
    #[test]
    fn either_menu_key_leaves_gameplay() {
        for key in MENU_KEYS {
            let mut world = World::new();
            let mut input = ButtonInput::<KeyCode>::default();
            input.press(key);
            world.insert_resource(input);
            world.insert_resource(MenuScreen::Settings);
            world.init_resource::<NextState<GameState>>();

            world.run_system_once(leave_to_menu).unwrap();

            assert!(
                matches!(
                    *world.resource::<NextState<GameState>>(),
                    NextState::Pending(GameState::Menu)
                ),
                "{key:?} did not request the menu"
            );
            // The menu must open on its main screen, not wherever it was left.
            assert_eq!(*world.resource::<MenuScreen>(), MenuScreen::Main);
        }
    }

    /// An unbound key must not open the menu — otherwise the assertion above
    /// would pass for a system that ignores its input entirely.
    #[test]
    fn an_unbound_key_does_not_leave_gameplay() {
        let mut world = World::new();
        let mut input = ButtonInput::<KeyCode>::default();
        input.press(KeyCode::KeyW);
        world.insert_resource(input);
        world.insert_resource(MenuScreen::Main);
        world.init_resource::<NextState<GameState>>();

        world.run_system_once(leave_to_menu).unwrap();

        assert!(matches!(
            *world.resource::<NextState<GameState>>(),
            NextState::Unchanged
        ));
    }

    /// The teardown hazard specific to Bevy 0.19: resources are stored as
    /// components on entities, so a teardown that swept entities broadly would
    /// delete `GameSettings`, `ChunkManager` and everything else along with the
    /// world — and the next world would be built on a gutted App. Scoping by
    /// `WorldScoped` is what makes that impossible, and this pins it.
    #[test]
    fn teardown_despawns_world_entities_and_spares_resources() {
        let mut world = World::new();
        world.insert_resource(SurvivesTeardown(7));
        world.insert_resource(crate::WorldInstanceId(0));

        let in_world = world.spawn(WorldScoped(0)).id();
        let also_in_world = world.spawn(WorldScoped(0)).id();
        // Menu/UI entities are deliberately unscoped and must survive.
        let ui_entity = world.spawn_empty().id();

        world.run_system_once(teardown_world).unwrap();

        assert!(world.get_entity(in_world).is_err(), "world entity survived");
        assert!(
            world.get_entity(also_in_world).is_err(),
            "world entity survived"
        );
        assert!(
            world.get_entity(ui_entity).is_ok(),
            "unscoped entity was despawned"
        );
        assert_eq!(
            world.get_resource::<SurvivesTeardown>(),
            Some(&SurvivesTeardown(7)),
            "teardown destroyed a resource"
        );
        assert_eq!(
            world.get_resource::<crate::WorldInstanceId>(),
            Some(&crate::WorldInstanceId(1)),
            "the next session must get a fresh world id"
        );
    }

    /// Teardown runs on every exit from Gameplay, including repeated
    /// menu/play cycles. It must stay idempotent and must not accumulate.
    #[test]
    fn repeated_teardown_is_stable() {
        let mut world = World::new();
        world.insert_resource(SurvivesTeardown(1));
        world.insert_resource(crate::WorldInstanceId(0));

        for cycle in 0..3 {
            world.spawn(WorldScoped(cycle));
            world.spawn(WorldScoped(cycle));
            world.run_system_once(teardown_world).unwrap();

            let remaining = world
                .query_filtered::<Entity, With<WorldScoped>>()
                .iter(&world)
                .count();
            assert_eq!(remaining, 0, "cycle {cycle} left world entities behind");
        }
        assert_eq!(
            world.get_resource::<SurvivesTeardown>(),
            Some(&SurvivesTeardown(1))
        );
        // One fresh id per session, never reused.
        assert_eq!(
            world.get_resource::<crate::WorldInstanceId>(),
            Some(&crate::WorldInstanceId(3))
        );
    }
}
