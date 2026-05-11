// ---------------------------------------------------------------------------
// Settings overlay policy
// ---------------------------------------------------------------------------
// Settings is a UI OVERLAY, not a state transition. It must never:
//   - Despawn world entities (chunks, cameras, animals, lights)
//   - Spawn or modify cameras
//   - Reset physics, player transforms, or world state
// The world must remain fully intact and renderable while Settings is open.
// Only GameState::Menu is allowed to tear down the world (via cleanup_world).
// Any code that despawns WorldEntity from Settings is a correctness bug.

use bevy::prelude::*;
use bevy::input::mouse::{AccumulatedMouseScroll, MouseScrollUnit};
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow, PresentMode, WindowMode};

use crate::block_types::BlockType;
use crate::GameState;
use crate::player::Player;
use crate::settings::GameSettings;
use crate::sky::Cloud;

// ---------------------------------------------------------------------------
// Menu background
// ---------------------------------------------------------------------------

/// Marker for the full-screen menu background image.
#[derive(Component)]
pub struct MenuBackground;

/// Holds the background image handle to prevent it from being dropped
/// when the MenuBackground entity is despawned during menu rebuilds.
/// The handle keeps the Image asset alive for the entire Menu state.
/// Also tracks retry state for async screenshot loading.
#[derive(Resource)]
struct MenuBackgroundHandle {
    handle: Option<Handle<Image>>,
    /// True if initial load returned None (screenshot may not exist on disk yet).
    needs_retry: bool,
    /// Frame counter for retry throttling.
    retry_frames: u32,
}

impl Default for MenuBackgroundHandle {
    fn default() -> Self {
        Self {
            handle: None,
            needs_retry: false,
            retry_frames: 0,
        }
    }
}

/// Marker for the 2D camera used during Menu state.
/// This camera renders UI nodes and the background image.
/// It is despawned when entering Gameplay (the Player's Camera3d takes over).
///
/// DIAGNOSIS: A blank blue screen on startup means no camera is active.
/// Bevy renders the ClearColor but no UI without a camera. This 2D camera
/// ensures the menu is always visible regardless of world camera state.
#[derive(Component)]
struct MenuCamera;

/// Try to load the screenshot from the most recent save file for use as
/// the title menu background. Returns None if no save or screenshot exists.
fn load_menu_background_image(images: &mut Assets<Image>) -> Option<Handle<Image>> {
    // Read the default save file to get the screenshot path.
    let save_path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".metalworld_save.json");
    let data = match std::fs::read_to_string(&save_path) {
        Ok(d) => d,
        Err(_) => {
            #[cfg(debug_assertions)]
            bevy::log::info!("Menu background: no save file at {}", save_path.display());
            return None;
        }
    };

    // Parse just enough to extract the image path.
    #[derive(serde::Deserialize)]
    struct Partial {
        #[serde(default)]
        last_menu_background_image_path: Option<String>,
    }
    let partial: Partial = serde_json::from_str(&data).ok()?;
    let img_path = match partial.last_menu_background_image_path {
        Some(p) if !p.is_empty() => p,
        _ => {
            #[cfg(debug_assertions)]
            bevy::log::info!("Menu background: save has no screenshot path");
            return None;
        }
    };

    // Verify the file exists before reading.
    #[cfg(debug_assertions)]
    if !std::path::Path::new(&img_path).exists() {
        bevy::log::warn!(
            "Menu background: screenshot file does not exist: {}",
            img_path,
        );
    }

    // Load the image from disk.
    let img_data = match std::fs::read(&img_path) {
        Ok(d) => d,
        Err(e) => {
            #[cfg(debug_assertions)]
            bevy::log::warn!("Menu background: failed to read {}: {}", img_path, e);
            return None;
        }
    };
    let dyn_img = match image::load_from_memory(&img_data) {
        Ok(i) => i,
        Err(e) => {
            #[cfg(debug_assertions)]
            bevy::log::warn!("Menu background: failed to decode {}: {}", img_path, e);
            return None;
        }
    };
    let rgba = dyn_img.to_rgba8();
    let (w, h) = rgba.dimensions();

    #[cfg(debug_assertions)]
    bevy::log::info!("Menu background: loaded {} ({}x{})", img_path, w, h);

    let bevy_img = Image::new(
        bevy::render::render_resource::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        bevy::render::render_resource::TextureDimension::D2,
        rgba.into_raw(),
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::MAIN_WORLD | bevy::asset::RenderAssetUsages::RENDER_WORLD,
    );
    let handle = images.add(bevy_img);
    #[cfg(debug_assertions)]
    bevy::log::info!("Menu background: asset handle {:?} created", handle);
    Some(handle)
}

/// Always spawn a full-screen background node for the menu.
/// If a saved screenshot exists, render it as an image.
/// Otherwise, render a solid dark gradient panel as fallback.
fn spawn_menu_background(commands: &mut Commands, image: Option<Handle<Image>>) {
    let bg_node = Node {
        position_type: PositionType::Absolute,
        left: Val::Px(0.0),
        top: Val::Px(0.0),
        width: Val::Percent(100.0),
        height: Val::Percent(100.0),
        ..default()
    };

    if let Some(handle) = image {
        commands.spawn((
            crate::UiOnly,
            MenuBackground,
            ImageNode {
                image: handle,
                ..default()
            },
            bg_node,
            ZIndex(-1),
        ));
    } else {
        // Fallback: solid dark panel when no save screenshot exists.
        #[cfg(debug_assertions)]
        bevy::log::info!("Menu background: using solid color fallback (no screenshot available)");
        commands.spawn((
            crate::UiOnly,
            MenuBackground,
            bg_node,
            BackgroundColor(Color::linear_rgb(0.04, 0.04, 0.08)),
            ZIndex(-1),
        ));
    }
}

/// Retry loading the menu background if the initial attempt failed.
/// Updates the existing entity's image in-place — never despawns/respawns.
fn retry_menu_background(
    mut images: ResMut<Assets<Image>>,
    mut bg_handle: ResMut<MenuBackgroundHandle>,
    mut bg_query: Query<(Option<&mut ImageNode>, Option<&mut BackgroundColor>), With<MenuBackground>>,
) {
    if !bg_handle.needs_retry {
        return;
    }

    bg_handle.retry_frames += 1;

    // Retry every 30 frames (~0.5s at 60fps).
    if bg_handle.retry_frames % 30 != 0 {
        return;
    }

    // Give up after ~5 seconds (300 frames).
    if bg_handle.retry_frames > 300 {
        #[cfg(debug_assertions)]
        bevy::log::info!("Menu background: giving up retry after 5s");
        bg_handle.needs_retry = false;
        return;
    }

    let bg_image = load_menu_background_image(&mut images);
    if let Some(handle) = bg_image {
        #[cfg(debug_assertions)]
        bevy::log::info!("Menu background: retry succeeded on frame {}", bg_handle.retry_frames);

        // Update the existing entity in-place — no despawn/respawn.
        for (image_node, bg_color) in &mut bg_query {
            if let Some(mut img) = image_node {
                img.image = handle.clone();
            }
            // Remove fallback color if present (set to transparent).
            if let Some(mut color) = bg_color {
                color.0 = Color::NONE;
            }
        }
        bg_handle.handle = Some(handle);
        bg_handle.needs_retry = false;
    }
}

/// Hide the menu background and menu camera when entering gameplay.
/// The Player's Camera3d takes over rendering in Gameplay state.
/// Entities are hidden (not despawned) so they can be re-shown on return.
fn hide_menu_background(
    mut bg_query: Query<&mut Visibility, With<MenuBackground>>,
    mut cam_query: Query<&mut Camera, With<MenuCamera>>,
) {
    for mut vis in &mut bg_query {
        *vis = Visibility::Hidden;
    }
    for mut cam in &mut cam_query {
        cam.is_active = false;
    }
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------
//
// IMPORTANT: Each UI marker component must have exactly ONE source of Default:
//   - #[derive(Component, Default)]  — OR —
//   - #[derive(Component)] + inclusion in the impl_default!() macro at the
//     bottom of this file.
// Using both causes E0119 (conflicting Default implementations).

#[derive(Component)]
pub struct SettingsMenu;

#[derive(Component)]
struct TabButton(SettingsTab);

#[derive(Component)]
struct TabContent;

// Main menu button components
#[derive(Component, Default)]
struct PlayButton;
#[derive(Component, Default)]
struct LoadLastGameButton;
#[derive(Component, Default)]
struct LoadSavedGameButton;
#[derive(Component, Default)]
struct SaveGameButton;
#[derive(Component, Default)]
struct SettingsButton;
#[derive(Component, Default)]
struct ControlsMenuButton;
#[derive(Component, Default)]
struct MainQuitButton;
#[derive(Component, Default)]
struct BackButton;

// Setting toggle/action buttons
#[derive(Component)]
struct FullscreenButton;
#[derive(Component)]
struct VsyncButton;
#[derive(Component)]
struct Fps120Button;
#[derive(Component)]
struct BrightnessUpButton;
#[derive(Component)]
struct BrightnessDownButton;
#[derive(Component)]
struct ShadowUpButton;
#[derive(Component)]
struct ShadowDownButton;
#[derive(Component)]
struct DrawDistUpButton;
#[derive(Component)]
struct DrawDistDownButton;
#[derive(Component)]
struct CloudsButton;
#[derive(Component)]
struct TexSizeUpButton;
#[derive(Component)]
struct TexSizeDownButton;
#[derive(Component)]
struct SunUpButton;
#[derive(Component)]
struct SunDownButton;
#[derive(Component)]
struct GammaUpButton;
#[derive(Component)]
struct GammaDownButton;
#[derive(Component)]
struct ContrastUpButton;
#[derive(Component)]
struct ContrastDownButton;
#[derive(Component)]
struct AnisoUpButton;
#[derive(Component)]
struct AnisoDownButton;
#[derive(Component)]
struct AaCycleButton;
#[derive(Component, Default)]
struct ExposureUpButton;
#[derive(Component, Default)]
struct ExposureDownButton;
#[derive(Component)]
struct TonemappingCycleButton;
#[derive(Component, Default)]
struct FovUpButton;
#[derive(Component, Default)]
struct FovDownButton;
#[derive(Component, Default)]
struct SsaoButton;
#[derive(Component)]
struct SsaoQualityCycleButton;
#[derive(Component)]
struct SmaaCycleButton;
#[derive(Component)]
struct RenderScaleUpButton;
#[derive(Component)]
struct RenderScaleDownButton;

#[derive(Component)]
struct LoadTextureButton {
    block_name: String,
    block_idx: u8,
}

#[derive(Component)]
struct RemoveTextureButton {
    block_name: String,
    block_idx: u8,
}

#[derive(Component, Default)]
struct AddCustomBlockButton;

/// Holds the async dialog for adding a new custom block.
#[derive(Resource)]
struct AddBlockDialogTask {
    task: bevy::tasks::Task<Option<(std::path::PathBuf, Vec<u8>)>>,
}

/// Holds a pending custom block name entered by the user. For simplicity,
/// auto-generated from the filename.
#[derive(Component)]
struct RemoveCustomBlockButton {
    atlas_index: u8,
}

/// Holds an async file dialog task for texture selection + the target block.
#[derive(Resource)]
struct TextureDialogTask {
    task: bevy::tasks::Task<Option<(std::path::PathBuf, Vec<u8>)>>,
    block_name: String,
    block_idx: u8,
}

/// Minimum pixel dimension for a valid block texture. Anything smaller is
/// likely corrupt or not a real texture.
const MIN_TEXTURE_SIZE: u32 = 8;

/// Holds a texture loading error message to display in the settings UI.
/// Set on validation failure, cleared on next successful load or dialog open.
#[derive(Resource, Default)]
pub struct TextureLoadError {
    pub message: Option<String>,
}

/// Tracks per-block preview image handles for the texture settings UI.
/// These are separate from the atlas — used only for the menu preview
/// thumbnails so the user sees the original source image without atlas
/// border darkening. Handles are removed when the texture is removed.
#[derive(Resource, Default)]
pub struct TexturePreviews {
    /// block_name → Image handle (UI-only, not used in rendering)
    handles: std::collections::HashMap<String, Handle<Image>>,
}

// Inventory components
#[derive(Component)]
struct InventoryPanel;

#[derive(Component)]
struct InventoryItem(BlockType);

// Value display labels
#[derive(Component)]
struct BrightnessLabel;
#[derive(Component)]
struct ShadowLabel;
#[derive(Component)]
struct DrawDistLabel;
#[derive(Component)]
struct TexSizeLabel;
#[derive(Component)]
struct SunLabel;
#[derive(Component)]
struct GammaLabel;
#[derive(Component)]
struct ContrastLabel;
#[derive(Component)]
struct AnisoLabel;
#[derive(Component, Default)]
struct ExposureLabel;
#[derive(Component, Default)]
struct FovLabel;
#[derive(Component)]
struct RenderScaleLabel;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MenuScreen {
    MainMenu,
    Settings,
    Controls,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SettingsTab {
    Display,
    Graphics,
    BlockTextures,
}

#[derive(Resource)]
pub struct MenuState {
    pub is_open: bool,
    pub cursor_captured: bool,
    screen: MenuScreen,
    settings_tab: SettingsTab,
}

impl Default for MenuState {
    fn default() -> Self {
        Self {
            is_open: false,
            cursor_captured: false,
            screen: MenuScreen::MainMenu,
            settings_tab: SettingsTab::Display,
        }
    }
}

#[derive(Resource, Default)]
pub struct InventoryState {
    pub is_open: bool,
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<MenuState>()
            .init_resource::<MenuBackgroundHandle>()
            .init_resource::<InventoryState>()
            .init_resource::<TexturePreviews>()
            .init_resource::<TextureLoadError>()
            .add_systems(OnEnter(GameState::Menu), open_menu_on_launch)
            .add_systems(OnEnter(GameState::Gameplay), hide_menu_background)
            .add_systems(Update, retry_menu_background.run_if(in_state(GameState::Menu)))
            .add_systems(Update, enforce_cursor_lock)
            .add_systems(Update, window_and_cursor_system)
            .add_systems(Update, toggle_menu)
            .add_systems(Update, handle_play_and_load_buttons)
            .add_systems(Update, handle_main_menu_buttons)
            .add_systems(Update, handle_back_button)
            .add_systems(Update, handle_tab_buttons)
            .add_systems(Update, (handle_setting_buttons, handle_120fps_button))
            .add_systems(PostUpdate, apply_fps_mode_changes)
            .add_systems(Update, handle_render_scale_buttons)
            .add_systems(Update, handle_tex_size_buttons)
            .add_systems(Update, handle_graphics_settings_buttons)
            .add_systems(Update, handle_load_texture_buttons)
            .add_systems(Update, poll_texture_dialog)
            .add_systems(Update, handle_remove_texture_buttons)
            .add_systems(Update, (handle_add_block_button, poll_add_block_dialog, handle_remove_custom_block))
            .add_systems(Update, (toggle_inventory, handle_inventory_clicks))
            .add_systems(Update, ui_scroll_mouse_wheel);
    }
}

// ---------------------------------------------------------------------------
// Startup: open the main menu with cursor free. The game launches into
// GameState::Menu — gameplay begins only when the player clicks Play or
// loads a save. No auto-transition.
// ---------------------------------------------------------------------------

fn open_menu_on_launch(
    mut commands: Commands,
    mut query: Query<&mut CursorOptions, With<PrimaryWindow>>,
    mut menu_state: ResMut<MenuState>,
    game_settings: Res<GameSettings>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut images: ResMut<Assets<Image>>,
    mut bg_handle: ResMut<MenuBackgroundHandle>,
    existing_menu_cams: Query<Entity, With<MenuCamera>>,
    existing_bgs: Query<Entity, With<MenuBackground>>,
) {
    // Cursor free for menu interaction
    if let Ok(mut cursor) = query.single_mut() {
        release_cursor(&mut cursor);
    }
    menu_state.is_open = true;
    menu_state.cursor_captured = false;
    menu_state.screen = MenuScreen::MainMenu;

    // Re-show existing menu camera if it was hidden, otherwise spawn fresh.
    if existing_menu_cams.is_empty() {
        commands.spawn((
            crate::UiOnly,
            MenuCamera,
            Camera2d,
            Camera {
                order: 0,
                clear_color: ClearColorConfig::Custom(
                    Color::linear_rgb(0.02, 0.02, 0.04),
                ),
                ..default()
            },
        ));
    } else {
        // Re-activate hidden camera via commands (avoids mutable query conflict).
        for entity in &existing_menu_cams {
            commands.entity(entity).insert(Camera {
                order: 0,
                is_active: true,
                clear_color: ClearColorConfig::Custom(
                    Color::linear_rgb(0.02, 0.02, 0.04),
                ),
                ..default()
            });
        }
    }

    // Re-show existing background if it was hidden, otherwise spawn fresh.
    if existing_bgs.is_empty() {
        let bg_image = load_menu_background_image(&mut images);
        bg_handle.handle = bg_image.clone();
        bg_handle.needs_retry = bg_image.is_none();
        bg_handle.retry_frames = 0;
        spawn_menu_background(&mut commands, bg_image);
    } else {
        // Re-show hidden background.
        for entity in &existing_bgs {
            commands.entity(entity).insert(Visibility::Inherited);
        }
        // Try to load a fresh screenshot (may have been taken on last exit).
        let bg_image = load_menu_background_image(&mut images);
        if let Some(handle) = &bg_image {
            bg_handle.handle = Some(handle.clone());
            bg_handle.needs_retry = false;
        } else {
            bg_handle.needs_retry = true;
            bg_handle.retry_frames = 0;
        }
    }

    let is_fullscreen = windows
        .iter()
        .next()
        .map(|w| !matches!(w.mode, WindowMode::Windowed))
        .unwrap_or(false);
    spawn_current_screen(&mut commands, &menu_state, &game_settings, is_fullscreen, None, None, None, None);
}

// ---------------------------------------------------------------------------
// Update: re-apply cursor lock whenever the window is focused during gameplay.
// Idempotent — safe to run every frame. Skips when menu or inventory is open
// so those systems can show the cursor without conflict.
// ---------------------------------------------------------------------------

fn enforce_cursor_lock(
    menu_state: Res<MenuState>,
    inv_state: Res<InventoryState>,
    dialog_open: Option<Res<crate::save_load::FileDialogOpen>>,
    mut query: Query<(&Window, &mut CursorOptions), With<PrimaryWindow>>,
) {
    let Ok((window, mut cursor)) = query.single_mut() else {
        return;
    };

    // Don't fight menu/inventory/file dialogs — they need the cursor visible.
    if menu_state.is_open || inv_state.is_open || dialog_open.is_some() {
        return;
    }

    // Only act when the window has focus. When unfocused, the OS owns the
    // cursor and re-locking would fight the window manager.
    if !window.focused {
        return;
    }

    // Idempotent: only write if state actually drifted.
    if cursor.grab_mode != CursorGrabMode::Locked || cursor.visible {
        cursor.grab_mode = CursorGrabMode::Locked;
        cursor.visible = false;
    }
}

// ---------------------------------------------------------------------------
// Mouse wheel scrolling for Bevy UI nodes with Overflow::scroll_y().
// Bevy 0.18 provides the layout/clipping but does not wire mouse wheel
// input to ScrollPosition automatically. This system does that for all
// scrollable nodes in the UI.
// ---------------------------------------------------------------------------

/// Pixels scrolled per mouse wheel line (platform-independent).
const SCROLL_LINE_HEIGHT: f32 = 32.0;

/// Drives ScrollPosition from mouse wheel / trackpad input for all UI nodes
/// with Overflow::scroll_y(). Bevy 0.18 provides the layout/clipping but
/// does not wire input to ScrollPosition automatically.
fn ui_scroll_mouse_wheel(
    scroll_input: Res<AccumulatedMouseScroll>,
    mut scrollable: Query<(&ComputedNode, &mut ScrollPosition, &Node)>,
) {
    if scroll_input.delta == Vec2::ZERO {
        return;
    }

    let dy = match scroll_input.unit {
        MouseScrollUnit::Line => -scroll_input.delta.y * SCROLL_LINE_HEIGHT,
        MouseScrollUnit::Pixel => -scroll_input.delta.y,
    };
    if dy == 0.0 {
        return;
    }

    for (computed, mut scroll_pos, node) in &mut scrollable {
        // Only scroll nodes that have vertical overflow enabled
        if node.overflow.y != OverflowAxis::Scroll {
            continue;
        }

        // Content height minus visible height = max scroll range
        let content_height = computed.content_size().y;
        let visible_height = computed.size().y;
        let max_scroll = (content_height - visible_height).max(0.0);

        scroll_pos.0.y = (scroll_pos.0.y + dy).clamp(0.0, max_scroll);
    }
}

// ---------------------------------------------------------------------------
// Cursor helpers
// ---------------------------------------------------------------------------

fn release_cursor(cursor: &mut CursorOptions) {
    cursor.grab_mode = CursorGrabMode::None;
    cursor.visible = true;
}

fn grab_cursor(cursor: &mut CursorOptions) {
    // Locked hides the cursor and reports raw mouse deltas (ideal for FPS).
    // On platforms that don't support Locked, Bevy falls back to Confined
    // automatically.
    cursor.grab_mode = CursorGrabMode::Locked;
    cursor.visible = false;
}

// ---------------------------------------------------------------------------
// Cursor + keyboard quit system
// ---------------------------------------------------------------------------

fn window_and_cursor_system(
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut menu_state: ResMut<MenuState>,
    inv_state: Res<InventoryState>,
    mut query: Query<(&mut Window, &mut CursorOptions), With<PrimaryWindow>>,
    mut game_settings: ResMut<GameSettings>,
    chunk_manager: Option<Res<crate::chunk_manager::ChunkManager>>,
    player_query: Query<(&Transform, &crate::player::Player)>,
) {
    let Ok((window, mut cursor)) = query.single_mut() else {
        return;
    };

    if menu_state.is_open && keys.just_pressed(KeyCode::KeyQ) {
        save_window_settings(&window, &mut game_settings);
        save_game_state(chunk_manager.as_deref(), &player_query);
        release_cursor(&mut cursor);
        std::process::exit(0);
    }

    if menu_state.is_open || inv_state.is_open {
        return;
    }

    // Re-grab if the OS released the cursor (e.g., after alt-tab or focus loss).
    // The initial grab happens at Startup in lock_cursor_on_launch.
    if !menu_state.cursor_captured {
        if window.focused
            && (mouse.just_pressed(MouseButton::Left) || mouse.just_pressed(MouseButton::Right))
        {
            grab_cursor(&mut cursor);
            menu_state.cursor_captured = true;
        }
    } else if window.focused && cursor.grab_mode == CursorGrabMode::None {
        grab_cursor(&mut cursor);
    }
}

// ---------------------------------------------------------------------------
// Menu toggle (M / Escape)
// ---------------------------------------------------------------------------

fn toggle_menu(
    keys: Res<ButtonInput<KeyCode>>,
    mut menu_state: ResMut<MenuState>,
    mut commands: Commands,
    menu_query: Query<Entity, With<SettingsMenu>>,
    mut cursor_query: Query<&mut CursorOptions, With<PrimaryWindow>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    game_settings: Res<GameSettings>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if keys.just_pressed(KeyCode::KeyM) {
        if menu_state.is_open {
            // Close menu — ensure we're in Gameplay (handles both the initial
            // launch menu and the in-game overlay).
            // During gameplay, this is a no-op (already in Gameplay).
            // From the title menu, this transitions Menu → Gameplay.
            #[cfg(debug_assertions)]
            bevy::log::info!(
                "Menu toggle: closing (screen={:?}), setting GameState::Gameplay (trigger=M key)",
                menu_state.screen,
            );
            close_menu(&mut menu_state, &mut commands, &menu_query, &mut cursor_query);
            // PendingIfNeq: when already in Gameplay (overlay close), Bevy skips
            // the OnExit/OnEnter cycle. From the title Menu, this is a real
            // Menu→Gameplay transition as intended. Using .set() here would
            // trigger a same-state Gameplay→Gameplay cycle that despawns the world.
            *next_state = NextState::PendingIfNeq(GameState::Gameplay);
        } else {
            // Open to main menu — NO state transition, just UI overlay.
            #[cfg(debug_assertions)]
            bevy::log::info!("Menu toggle: opening overlay (trigger=M key, no state change)");
            menu_state.is_open = true;
            menu_state.screen = MenuScreen::MainMenu;
            if let Ok(mut cursor) = cursor_query.single_mut() {
                release_cursor(&mut cursor);
            }
            let is_fullscreen = windows
                .iter()
                .next()
                .map(|w| !matches!(w.mode, WindowMode::Windowed))
                .unwrap_or(false);
            spawn_current_screen(&mut commands, &menu_state, &game_settings, is_fullscreen, None, None, None, None);
        }
        return;
    }

    if menu_state.is_open && keys.just_pressed(KeyCode::Escape) {
        match menu_state.screen {
            MenuScreen::Settings | MenuScreen::Controls => {
                // Go back to main menu — NO state transition.
                #[cfg(debug_assertions)]
                bevy::log::info!(
                    "Menu nav: {:?} → MainMenu (trigger=Escape, no state change)",
                    menu_state.screen,
                );
                for entity in &menu_query {
                    commands.entity(entity).despawn();
                }
                menu_state.screen = MenuScreen::MainMenu;
                let is_fullscreen = windows
                    .iter()
                    .next()
                    .map(|w| !matches!(w.mode, WindowMode::Windowed))
                    .unwrap_or(false);
                spawn_current_screen(&mut commands, &menu_state, &game_settings, is_fullscreen, None, None, None, None);
            }
            MenuScreen::MainMenu => {
                #[cfg(debug_assertions)]
                bevy::log::info!("Menu close: MainMenu → Gameplay (trigger=Escape)");
                close_menu(&mut menu_state, &mut commands, &menu_query, &mut cursor_query);
                // PendingIfNeq: same-state no-op when closing overlay during Gameplay.
                *next_state = NextState::PendingIfNeq(GameState::Gameplay);
            }
        }
    }
}

fn close_menu(
    menu_state: &mut ResMut<MenuState>,
    commands: &mut Commands,
    menu_query: &Query<Entity, With<SettingsMenu>>,
    cursor_query: &mut Query<&mut CursorOptions, With<PrimaryWindow>>,
) {
    menu_state.is_open = false;
    for entity in menu_query {
        commands.entity(entity).despawn();
    }
    if let Ok(mut cursor) = cursor_query.single_mut() {
        grab_cursor(&mut cursor);
        menu_state.cursor_captured = true;
    }
}

// ---------------------------------------------------------------------------
// Spawn the current screen
// ---------------------------------------------------------------------------

fn spawn_current_screen(
    commands: &mut Commands,
    menu_state: &MenuState,
    settings: &GameSettings,
    is_fullscreen: bool,
    atlas: Option<&crate::chunk_manager::BlockAtlas>,
    previews: Option<&TexturePreviews>,
    tex_error: Option<&TextureLoadError>,
    custom_registry: Option<&crate::block_types::CustomBlockRegistry>,
) {
    match menu_state.screen {
        MenuScreen::MainMenu => spawn_main_menu(commands),
        MenuScreen::Settings => spawn_settings_menu(commands, menu_state, settings, is_fullscreen, atlas, previews, tex_error, custom_registry),
        MenuScreen::Controls => spawn_controls_screen(commands),
    }
}

// ---------------------------------------------------------------------------
// Main Menu Screen
// ---------------------------------------------------------------------------

fn spawn_main_menu(commands: &mut Commands) {
    commands
        .spawn((
            crate::UiOnly,
            SettingsMenu,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Percent(30.0),
                top: Val::Percent(15.0),
                width: Val::Percent(40.0),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                padding: UiRect::all(Val::Px(24.0)),
                ..default()
            },
            BackgroundColor(Color::linear_rgba(0.08, 0.08, 0.12, 0.95)),
        ))
        .with_children(|root| {
            // Title
            root.spawn((
                Text::new(crate::GAME_NAME),
                TextFont { font_size: 36.0, ..default() },
                TextColor(Color::linear_rgb(0.9, 0.85, 0.5)),
                Node { margin: UiRect::bottom(Val::Px(8.0)), ..default() },
            ));

            // Divider
            root.spawn((
                Node {
                    width: Val::Percent(80.0),
                    height: Val::Px(2.0),
                    margin: UiRect::bottom(Val::Px(20.0)),
                    ..default()
                },
                BackgroundColor(Color::linear_rgba(0.5, 0.5, 0.5, 0.5)),
            ));

            // Menu buttons
            main_menu_button::<PlayButton>(root, "Play");
            main_menu_button::<LoadLastGameButton>(root, "Load Last Game");
            main_menu_button::<LoadSavedGameButton>(root, "Load Saved Game");
            main_menu_button::<SaveGameButton>(root, "Save");
            main_menu_button::<SettingsButton>(root, "Settings");
            main_menu_button::<ControlsMenuButton>(root, "Controls (?)");
            main_menu_button::<MainQuitButton>(root, "Quit");
        });
}

fn main_menu_button<M: Component + Default>(parent: &mut ChildSpawnerCommands, label: &str) {
    parent
        .spawn((
            M::default(),
            Button,
            Node {
                width: Val::Percent(70.0),
                padding: UiRect::axes(Val::Px(20.0), Val::Px(12.0)),
                margin: UiRect::bottom(Val::Px(8.0)),
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(Color::linear_rgba(0.2, 0.2, 0.28, 1.0)),
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(label),
                TextFont { font_size: 20.0, ..default() },
                TextColor(Color::WHITE),
            ));
        });
}

// ---------------------------------------------------------------------------
// Main menu button handler
// ---------------------------------------------------------------------------

/// Handles the Play and Load buttons.
/// Play enters gameplay directly. Both Load buttons open a native file
/// dialog — there is no hardcoded-path load from menu actions.
fn handle_play_and_load_buttons(
    play_q: Query<&Interaction, (Changed<Interaction>, With<PlayButton>)>,
    load_last_q: Query<&Interaction, (Changed<Interaction>, With<LoadLastGameButton>)>,
    load_saved_q: Query<&Interaction, (Changed<Interaction>, With<LoadSavedGameButton>)>,
    mut menu_state: ResMut<MenuState>,
    mut commands: Commands,
    menu_query: Query<Entity, With<SettingsMenu>>,
    mut windows: Query<(&Window, &mut CursorOptions), With<PrimaryWindow>>,
    mut load_dialog: ResMut<crate::save_load::OpenLoadDialog>,
    mut next_state: ResMut<NextState<GameState>>,
    mut start_mode: ResMut<crate::save_load::StartMode>,
    mut teardown_intent: ResMut<crate::TeardownIntent>,
    mut world_instance: ResMut<crate::WorldInstanceId>,
    mut pending_teardown: ResMut<crate::PendingTeardown>,
) {
    // Play = New Game — generate a fresh world, ignore saved modifications.
    for i in &play_q {
        if *i == Interaction::Pressed {
            *start_mode = crate::save_load::StartMode::NewGame;
            // Snapshot old instance and bump to new id before setting intent.
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
            if let Ok((_, mut cursor)) = windows.single_mut() {
                grab_cursor(&mut cursor);
                menu_state.cursor_captured = true;
            }
            #[cfg(debug_assertions)]
            bevy::log::info!("State transition: Menu → Gameplay (trigger=Play/NewGame, teardown=NewGame)");
            next_state.set(GameState::Gameplay);
            return;
        }
    }

    // Load Last Game — continue from saved state.
    for i in &load_last_q {
        if *i == Interaction::Pressed {
            *start_mode = crate::save_load::StartMode::Continue;
            load_dialog.pending = true;
            return;
        }
    }

    // Load Saved Game — opens file dialog, continues from chosen save.
    for i in &load_saved_q {
        if *i == Interaction::Pressed {
            *start_mode = crate::save_load::StartMode::Continue;
            load_dialog.pending = true;
            return;
        }
    }
}

fn handle_main_menu_buttons(
    save_q: Query<&Interaction, (Changed<Interaction>, With<SaveGameButton>)>,
    settings_q: Query<&Interaction, (Changed<Interaction>, With<SettingsButton>)>,
    controls_q: Query<&Interaction, (Changed<Interaction>, With<ControlsMenuButton>)>,
    quit_q: Query<&Interaction, (Changed<Interaction>, With<MainQuitButton>)>,
    mut menu_state: ResMut<MenuState>,
    mut commands: Commands,
    menu_query: Query<Entity, With<SettingsMenu>>,
    mut windows: Query<(&Window, &mut CursorOptions), With<PrimaryWindow>>,
    mut game_settings: ResMut<GameSettings>,
    chunk_manager: Option<Res<crate::chunk_manager::ChunkManager>>,
    player_query: Query<(&Transform, &crate::player::Player)>,
    block_atlas: Option<Res<crate::chunk_manager::BlockAtlas>>,
    mut save_dialog: ResMut<crate::save_load::OpenSaveDialog>,
    tex_previews: Res<TexturePreviews>,
    custom_registry: Res<crate::block_types::CustomBlockRegistry>,
) {

    // Save — emits event consumed by open_save_dialog in save_load.rs
    for i in &save_q {
        if *i == Interaction::Pressed {
            save_dialog.pending = true;
            return;
        }
    }

    // Settings
    for i in &settings_q {
        if *i == Interaction::Pressed {
            for entity in &menu_query {
                commands.entity(entity).despawn();
            }
            menu_state.screen = MenuScreen::Settings;
            let is_fullscreen = windows
                .iter()
                .next()
                .map(|(w, _)| !matches!(w.mode, WindowMode::Windowed))
                .unwrap_or(false);
            spawn_current_screen(&mut commands, &menu_state, &game_settings, is_fullscreen, block_atlas.as_deref(), Some(&tex_previews), None, Some(&custom_registry));
            return;
        }
    }

    // Controls
    for i in &controls_q {
        if *i == Interaction::Pressed {
            for entity in &menu_query {
                commands.entity(entity).despawn();
            }
            menu_state.screen = MenuScreen::Controls;
            let is_fullscreen = windows
                .iter()
                .next()
                .map(|(w, _)| !matches!(w.mode, WindowMode::Windowed))
                .unwrap_or(false);
            spawn_current_screen(&mut commands, &menu_state, &game_settings, is_fullscreen, None, None, None, None);
            return;
        }
    }

    // Quit
    for i in &quit_q {
        if *i == Interaction::Pressed {
            if let Ok((window, mut cursor)) = windows.single_mut() {
                save_window_settings(&window, &mut game_settings);
                release_cursor(&mut cursor);
            }
            save_game_state(chunk_manager.as_deref(), &player_query);
            std::process::exit(0);
        }
    }
}

// ---------------------------------------------------------------------------
// Back button handler
// ---------------------------------------------------------------------------

fn handle_back_button(
    back_q: Query<&Interaction, (Changed<Interaction>, With<BackButton>)>,
    mut menu_state: ResMut<MenuState>,
    mut commands: Commands,
    menu_query: Query<Entity, With<SettingsMenu>>,
    windows: Query<(&Window, &CursorOptions), With<PrimaryWindow>>,
    game_settings: Res<GameSettings>,
) {
    for i in &back_q {
        if *i == Interaction::Pressed {
            for entity in &menu_query {
                commands.entity(entity).despawn();
            }
            menu_state.screen = MenuScreen::MainMenu;
            let is_fullscreen = windows
                .iter()
                .next()
                .map(|(w, _)| !matches!(w.mode, WindowMode::Windowed))
                .unwrap_or(false);
            spawn_current_screen(&mut commands, &menu_state, &game_settings, is_fullscreen, None, None, None, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Settings Menu Screen
// ---------------------------------------------------------------------------

fn spawn_settings_menu(
    commands: &mut Commands,
    menu_state: &MenuState,
    settings: &GameSettings,
    is_fullscreen: bool,
    atlas: Option<&crate::chunk_manager::BlockAtlas>,
    previews: Option<&TexturePreviews>,
    tex_error: Option<&TextureLoadError>,
    custom_registry: Option<&crate::block_types::CustomBlockRegistry>,
) {
    commands
        .spawn((
            crate::UiOnly,
            SettingsMenu,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Percent(20.0),
                top: Val::Percent(8.0),
                width: Val::Percent(60.0),
                height: Val::Percent(84.0),
                flex_direction: FlexDirection::Column,
                ..default()
            },
            BackgroundColor(Color::linear_rgba(0.08, 0.08, 0.12, 0.95)),
        ))
        .with_children(|root| {
            // Title bar with back button
            root.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                padding: UiRect::all(Val::Px(16.0)),
                ..default()
            })
            .with_children(|title_row| {
                // Back button
                title_row
                    .spawn((
                        BackButton,
                        Button,
                        Node {
                            padding: UiRect::axes(Val::Px(12.0), Val::Px(6.0)),
                            margin: UiRect::right(Val::Px(16.0)),
                            ..default()
                        },
                        BackgroundColor(Color::linear_rgba(0.3, 0.3, 0.4, 1.0)),
                    ))
                    .with_children(|btn| {
                        btn.spawn((
                            Text::new("< Back"),
                            TextFont { font_size: 16.0, ..default() },
                            TextColor(Color::WHITE),
                        ));
                    });

                title_row.spawn((
                    Text::new("Settings"),
                    TextFont { font_size: 28.0, ..default() },
                    TextColor(Color::WHITE),
                ));
            });

            // Tab bar
            root.spawn(Node {
                flex_direction: FlexDirection::Row,
                padding: UiRect::horizontal(Val::Px(16.0)),
                margin: UiRect::bottom(Val::Px(4.0)),
                ..default()
            })
            .with_children(|tab_bar| {
                for tab in [SettingsTab::Display, SettingsTab::Graphics, SettingsTab::BlockTextures] {
                    let is_active = tab == menu_state.settings_tab;
                    let bg = if is_active {
                        Color::linear_rgba(0.3, 0.3, 0.4, 1.0)
                    } else {
                        Color::linear_rgba(0.15, 0.15, 0.2, 1.0)
                    };
                    let label = match tab {
                        SettingsTab::Display => "Display",
                        SettingsTab::Graphics => "Graphics",
                        SettingsTab::BlockTextures => "Block Textures",
                    };
                    tab_bar
                        .spawn((
                            TabButton(tab),
                            Button,
                            Node {
                                padding: UiRect::axes(Val::Px(16.0), Val::Px(8.0)),
                                margin: UiRect::right(Val::Px(4.0)),
                                ..default()
                            },
                            BackgroundColor(bg),
                        ))
                        .with_children(|btn| {
                            btn.spawn((
                                Text::new(label),
                                TextFont { font_size: 16.0, ..default() },
                                TextColor(Color::WHITE),
                            ));
                        });
                }
            });

            // Tab content area
            let atlas_handle = atlas.map(|a| a.image_handle.clone());
            let atlas_tile_size = atlas.map(|a| a.tile_size).unwrap_or(64);
            root.spawn((
                TabContent,
                Node {
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(Val::Px(16.0)),
                    flex_grow: 1.0,
                    overflow: Overflow::scroll_y(),
                    ..default()
                },
            ))
            .with_children(|content| {
                spawn_settings_tab_content(content, menu_state.settings_tab, settings, is_fullscreen, atlas_handle.as_ref(), atlas_tile_size, previews, tex_error, custom_registry);
            });
        });
}

fn spawn_settings_tab_content(
    parent: &mut ChildSpawnerCommands,
    tab: SettingsTab,
    settings: &GameSettings,
    is_fullscreen: bool,
    atlas_handle: Option<&Handle<Image>>,
    atlas_tile_size: u32,
    previews: Option<&TexturePreviews>,
    tex_error: Option<&TextureLoadError>,
    custom_registry: Option<&crate::block_types::CustomBlockRegistry>,
) {
    match tab {
        SettingsTab::Display => spawn_display_tab(parent, settings, is_fullscreen),
        SettingsTab::Graphics => spawn_graphics_tab(parent, settings),
        SettingsTab::BlockTextures => spawn_textures_tab(parent, settings, atlas_handle, atlas_tile_size, previews, tex_error, custom_registry),
    }
}

// ---------------------------------------------------------------------------
// Controls Screen
// ---------------------------------------------------------------------------

fn spawn_controls_screen(commands: &mut Commands) {
    commands
        .spawn((
            crate::UiOnly,
            SettingsMenu,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Percent(25.0),
                top: Val::Percent(10.0),
                width: Val::Percent(50.0),
                height: Val::Percent(80.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(16.0)),
                overflow: Overflow::scroll_y(),
                ..default()
            },
            BackgroundColor(Color::linear_rgba(0.08, 0.08, 0.12, 0.95)),
        ))
        .with_children(|root| {
            // Title bar with back button
            root.spawn(Node {
                flex_direction: FlexDirection::Row,
                align_items: AlignItems::Center,
                margin: UiRect::bottom(Val::Px(16.0)),
                ..default()
            })
            .with_children(|title_row| {
                title_row
                    .spawn((
                        BackButton,
                        Button,
                        Node {
                            padding: UiRect::axes(Val::Px(12.0), Val::Px(6.0)),
                            margin: UiRect::right(Val::Px(16.0)),
                            ..default()
                        },
                        BackgroundColor(Color::linear_rgba(0.3, 0.3, 0.4, 1.0)),
                    ))
                    .with_children(|btn| {
                        btn.spawn((
                            Text::new("< Back"),
                            TextFont { font_size: 16.0, ..default() },
                            TextColor(Color::WHITE),
                        ));
                    });

                title_row.spawn((
                    Text::new("Controls"),
                    TextFont { font_size: 28.0, ..default() },
                    TextColor(Color::WHITE),
                ));
            });

            spawn_controls_content(root);
        });
}

fn spawn_controls_content(parent: &mut ChildSpawnerCommands) {
    let bindings = [
        ("WASD", "Move"),
        ("Mouse", "Look"),
        ("Space", "Jump"),
        ("Shift", "Sneak"),
        ("Ctrl", "Sprint"),
        ("Alt", "Slow Walk"),
        ("Arrow Keys", "Turn Camera"),
        ("1-9", "Select Block"),
        ("Left Click", "Break Block"),
        ("Right Click", "Place Block"),
        ("I", "Inventory"),
        ("H", "Teleport Home"),
        ("F3", "Toggle FPS"),
        ("F5", "Save Game"),
        ("M / Escape", "Menu"),
        ("Q", "Quit (in menu)"),
    ];

    for (key, action) in &bindings {
        parent.spawn(Node {
            flex_direction: FlexDirection::Row,
            justify_content: JustifyContent::SpaceBetween,
            width: Val::Percent(100.0),
            margin: UiRect::bottom(Val::Px(4.0)),
            ..default()
        })
        .with_children(|row| {
            row.spawn((
                Text::new(*key),
                TextFont { font_size: 16.0, ..default() },
                TextColor(Color::linear_rgb(0.9, 0.9, 0.5)),
            ));
            row.spawn((
                Text::new(*action),
                TextFont { font_size: 16.0, ..default() },
                TextColor(Color::linear_rgb(0.8, 0.8, 0.8)),
            ));
        });
    }
}

// ---------------------------------------------------------------------------
// Display tab
// ---------------------------------------------------------------------------

fn spawn_display_tab(parent: &mut ChildSpawnerCommands, settings: &GameSettings, is_fullscreen: bool) {
    section_header(parent, "Display");

    // Fullscreen
    toggle_button::<FullscreenButton>(
        parent,
        "Fullscreen",
        is_fullscreen,
    );

    // VSync
    toggle_button::<VsyncButton>(
        parent,
        "VSync",
        settings.vsync,
    );

    // 120 FPS Mode
    toggle_button::<Fps120Button>(
        parent,
        "120 FPS Mode",
        settings.fps_120_mode,
    );

    // Render Scale
    value_row::<RenderScaleUpButton, RenderScaleDownButton, RenderScaleLabel>(
        parent,
        "Render Scale",
        &format!("{}%", (settings.render_scale * 100.0).round() as u32),
    );

    // Brightness
    value_row::<BrightnessUpButton, BrightnessDownButton, BrightnessLabel>(
        parent,
        "Brightness",
        &format!("{:.1}", settings.brightness),
    );

    // Gamma
    value_row::<GammaUpButton, GammaDownButton, GammaLabel>(
        parent,
        "Gamma",
        &format!("{:.1}", settings.gamma),
    );

    // Contrast
    value_row::<ContrastUpButton, ContrastDownButton, ContrastLabel>(
        parent,
        "Contrast",
        &format!("{:.1}", settings.contrast),
    );

    // Exposure
    value_row::<ExposureUpButton, ExposureDownButton, ExposureLabel>(
        parent,
        "Exposure (EV)",
        &format!("{:.1}", settings.exposure),
    );

    // Tonemapping (cycle button)
    let tm_label = match settings.tonemapping.as_str() {
        "reinhard" => "Reinhard",
        "aces" => "ACES",
        "agx" => "AgX",
        "tony" => "TonyMcMapface",
        "blender" => "BlenderFilmic",
        _ => "None",
    };
    parent
        .spawn((
            TonemappingCycleButton,
            Button,
            Node {
                padding: UiRect::axes(Val::Px(12.0), Val::Px(8.0)),
                margin: UiRect::bottom(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::linear_rgba(0.2, 0.2, 0.28, 1.0)),
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(format!("Tonemapping: {}", tm_label)),
                TextFont { font_size: 17.0, ..default() },
                TextColor(Color::WHITE),
            ));
        });
}

// ---------------------------------------------------------------------------
// Graphics tab
// ---------------------------------------------------------------------------

fn spawn_graphics_tab(parent: &mut ChildSpawnerCommands, settings: &GameSettings) {
    section_header(parent, "Graphics");

    // Draw distance
    value_row::<DrawDistUpButton, DrawDistDownButton, DrawDistLabel>(
        parent,
        "Draw Distance",
        &format!("{} chunks", settings.render_distance),
    );

    // Texture resolution
    value_row::<TexSizeUpButton, TexSizeDownButton, TexSizeLabel>(
        parent,
        "Texture Size",
        &format!("{}px", settings.texture_size),
    );

    // Sun intensity
    value_row::<SunUpButton, SunDownButton, SunLabel>(
        parent,
        "Sun Intensity",
        &format!("{:.1}", settings.shadow_intensity),
    );

    // Anisotropic filtering
    let aniso_label = if settings.anisotropic_filtering <= 1 {
        "Off".to_string()
    } else {
        format!("{}x", settings.anisotropic_filtering)
    };
    value_row::<AnisoUpButton, AnisoDownButton, AnisoLabel>(
        parent,
        "Anisotropic Filter",
        &aniso_label,
    );

    // Anti-aliasing (cycle button)
    let aa_label = match settings.anti_aliasing.as_str() {
        "msaa2" => "MSAA 2x",
        "msaa4" => "MSAA 4x",
        "taa" => "TAA",
        _ => "Off",
    };
    parent
        .spawn((
            AaCycleButton,
            Button,
            Node {
                padding: UiRect::axes(Val::Px(12.0), Val::Px(8.0)),
                margin: UiRect::bottom(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::linear_rgba(0.2, 0.2, 0.28, 1.0)),
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(format!("Anti-Aliasing: {}", aa_label)),
                TextFont { font_size: 17.0, ..default() },
                TextColor(Color::WHITE),
            ));
        });

    // FOV
    value_row::<FovUpButton, FovDownButton, FovLabel>(
        parent,
        "FOV",
        &format!("{:.0}", settings.fov),
    );

    // SSAO
    toggle_button::<SsaoButton>(
        parent,
        "SSAO",
        settings.ssao_enabled,
    );

    // SSAO Quality (cycle button)
    let ssao_q_label = match settings.ssao_quality.as_str() {
        "low" => "Low",
        "high" => "High",
        "ultra" => "Ultra",
        _ => "Medium",
    };
    parent
        .spawn((
            SsaoQualityCycleButton,
            Button,
            Node {
                padding: UiRect::axes(Val::Px(12.0), Val::Px(8.0)),
                margin: UiRect::bottom(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::linear_rgba(0.2, 0.2, 0.28, 1.0)),
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(format!("SSAO Quality: {}", ssao_q_label)),
                TextFont { font_size: 17.0, ..default() },
                TextColor(Color::WHITE),
            ));
        });

    // SMAA (cycle button)
    let smaa_label = match settings.smaa_mode.as_str() {
        "low" => "Low",
        "medium" => "Medium",
        "high" => "High",
        "ultra" => "Ultra",
        _ => "Off",
    };
    parent
        .spawn((
            SmaaCycleButton,
            Button,
            Node {
                padding: UiRect::axes(Val::Px(12.0), Val::Px(8.0)),
                margin: UiRect::bottom(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::linear_rgba(0.2, 0.2, 0.28, 1.0)),
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(format!("SMAA: {}", smaa_label)),
                TextFont { font_size: 17.0, ..default() },
                TextColor(Color::WHITE),
            ));
        });

    // Clouds
    toggle_button::<CloudsButton>(
        parent,
        "Clouds",
        settings.clouds_enabled,
    );
}

// ---------------------------------------------------------------------------
// Textures tab
// ---------------------------------------------------------------------------

fn spawn_textures_tab(
    parent: &mut ChildSpawnerCommands,
    settings: &GameSettings,
    atlas_handle: Option<&Handle<Image>>,
    atlas_tile_size: u32,
    previews: Option<&TexturePreviews>,
    error: Option<&TextureLoadError>,
    custom_registry: Option<&crate::block_types::CustomBlockRegistry>,
) {
    section_header(parent, "Block Textures");

    parent.spawn((
        Text::new("Select image files to customize block textures"),
        TextFont { font_size: 14.0, ..default() },
        TextColor(Color::linear_rgb(0.7, 0.7, 0.7)),
        Node { margin: UiRect::bottom(Val::Px(12.0)), ..default() },
    ));

    // Show error message if a texture load failed
    if let Some(err) = error.and_then(|e| e.message.as_deref()) {
        parent.spawn((
            Text::new(err.to_string()),
            TextFont { font_size: 14.0, ..default() },
            TextColor(Color::linear_rgb(1.0, 0.3, 0.3)),
            Node { margin: UiRect::bottom(Val::Px(12.0)), ..default() },
        ));
    }

    let block_types: &[(&str, BlockType)] = &[
        ("Grass", BlockType::GRASS),
        ("Dirt", BlockType::DIRT),
        ("Stone", BlockType::STONE),
        ("Sand", BlockType::SAND),
        ("Wood", BlockType::WOOD),
        ("Diamond", BlockType::DIAMOND),
        ("Lantern", BlockType::LANTERN),
        ("Leaves", BlockType::LEAVES),
        ("StoneBrick", BlockType::STONE_BRICK),
    ];

    let tiles_per_row = crate::chunk_manager::ATLAS_TILES_PER_ROW;

    // Wrapping grid container — cards flow left-to-right, wrap to next row.
    // The parent tab content has overflow: scroll_y() so the grid scrolls
    // vertically regardless of how many entries there are.
    parent.spawn(Node {
        flex_direction: FlexDirection::Row,
        flex_wrap: FlexWrap::Wrap,
        column_gap: Val::Px(10.0),
        row_gap: Val::Px(10.0),
        ..default()
    }).with_children(|grid| {
        for &(name, block_type) in block_types {
            let block_idx = block_type.index();
            let has_texture = settings.block_textures.contains_key(name);

            // Highlight color: custom texture = blue border, default = dark
            let card_bg = if has_texture {
                Color::linear_rgba(0.15, 0.18, 0.30, 1.0)
            } else {
                Color::linear_rgba(0.12, 0.12, 0.16, 1.0)
            };

            grid.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    align_items: AlignItems::Center,
                    padding: UiRect::all(Val::Px(6.0)),
                    width: Val::Px(110.0),
                    ..default()
                },
                BackgroundColor(card_bg),
            )).with_children(|card| {
                // Block name
                card.spawn((
                    Text::new(name),
                    TextFont { font_size: 13.0, ..default() },
                    TextColor(Color::WHITE),
                    Node { margin: UiRect::bottom(Val::Px(4.0)), ..default() },
                ));

                // Preview thumbnail
                let preview_handle = previews.and_then(|p| p.handles.get(name));
                if let Some(phandle) = preview_handle {
                    card.spawn((
                        ImageNode { image: phandle.clone(), ..default() },
                        Node {
                            width: Val::Px(64.0),
                            height: Val::Px(64.0),
                            margin: UiRect::bottom(Val::Px(4.0)),
                            ..default()
                        },
                    ));
                } else if let Some(handle) = atlas_handle {
                    let tile_x = (block_idx as u32) % tiles_per_row;
                    let tile_y = (block_idx as u32) / tiles_per_row;
                    let ts = atlas_tile_size as f32;
                    let rect = Rect::new(
                        tile_x as f32 * ts,
                        tile_y as f32 * ts,
                        (tile_x + 1) as f32 * ts,
                        (tile_y + 1) as f32 * ts,
                    );
                    card.spawn((
                        ImageNode {
                            image: handle.clone(),
                            rect: Some(rect),
                            ..default()
                        },
                        Node {
                            width: Val::Px(64.0),
                            height: Val::Px(64.0),
                            margin: UiRect::bottom(Val::Px(4.0)),
                            ..default()
                        },
                    ));
                } else {
                    card.spawn((
                        Node {
                            width: Val::Px(64.0),
                            height: Val::Px(64.0),
                            margin: UiRect::bottom(Val::Px(4.0)),
                            ..default()
                        },
                        BackgroundColor(block_type.color()),
                    ));
                }

                // Status line
                let status = if has_texture {
                    settings.block_textures.get(name)
                        .map(|p| {
                            let fname = p.rsplit('/').next().unwrap_or(p);
                            // Truncate long filenames
                            if fname.len() > 12 {
                                format!("{}…", &fname[..11])
                            } else {
                                fname.to_string()
                            }
                        })
                        .unwrap_or_else(|| "custom".to_string())
                } else {
                    "default".to_string()
                };
                card.spawn((
                    Text::new(status),
                    TextFont { font_size: 11.0, ..default() },
                    TextColor(Color::linear_rgb(0.5, 0.5, 0.5)),
                    Node { margin: UiRect::bottom(Val::Px(4.0)), ..default() },
                ));

                // Action buttons row
                card.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(4.0),
                    ..default()
                }).with_children(|btns| {
                    // Load button
                    btns.spawn((
                        LoadTextureButton { block_name: name.to_string(), block_idx },
                        Button,
                        Node {
                            padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                            ..default()
                        },
                        BackgroundColor(Color::linear_rgba(0.2, 0.2, 0.3, 1.0)),
                    )).with_children(|btn| {
                        btn.spawn((
                            Text::new("Load"),
                            TextFont { font_size: 12.0, ..default() },
                            TextColor(Color::WHITE),
                        ));
                    });

                    // Remove button (only if custom texture)
                    if has_texture {
                        btns.spawn((
                            RemoveTextureButton { block_name: name.to_string(), block_idx },
                            Button,
                            Node {
                                padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)),
                                ..default()
                            },
                            BackgroundColor(Color::linear_rgba(0.4, 0.15, 0.15, 1.0)),
                        )).with_children(|btn| {
                            btn.spawn((
                                Text::new("✕"),
                                TextFont { font_size: 12.0, ..default() },
                                TextColor(Color::WHITE),
                            ));
                        });
                    }
                });
            });
        }
    });

    // --- Custom blocks section ---
    section_header(parent, "Custom Blocks");

    let registry = custom_registry.filter(|r| r.count() > 0);
    if let Some(reg) = registry {
        parent.spawn(Node {
            flex_direction: FlexDirection::Row,
            flex_wrap: FlexWrap::Wrap,
            column_gap: Val::Px(10.0),
            row_gap: Val::Px(10.0),
            margin: UiRect::bottom(Val::Px(10.0)),
            ..default()
        }).with_children(|grid| {
            let tiles_per_row = crate::chunk_manager::ATLAS_TILES_PER_ROW;
            for entry in reg.iter() {
                let idx = entry.atlas_index;
                grid.spawn((
                    Node {
                        flex_direction: FlexDirection::Column,
                        align_items: AlignItems::Center,
                        padding: UiRect::all(Val::Px(6.0)),
                        width: Val::Px(110.0),
                        ..default()
                    },
                    BackgroundColor(Color::linear_rgba(0.15, 0.22, 0.18, 1.0)),
                )).with_children(|card| {
                    card.spawn((
                        Text::new(&entry.name),
                        TextFont { font_size: 13.0, ..default() },
                        TextColor(Color::WHITE),
                        Node { margin: UiRect::bottom(Val::Px(4.0)), ..default() },
                    ));

                    // Preview from atlas
                    if let Some(handle) = atlas_handle {
                        let tile_x = (idx as u32) % tiles_per_row;
                        let tile_y = (idx as u32) / tiles_per_row;
                        let ts = atlas_tile_size as f32;
                        let rect = Rect::new(
                            tile_x as f32 * ts,
                            tile_y as f32 * ts,
                            (tile_x + 1) as f32 * ts,
                            (tile_y + 1) as f32 * ts,
                        );
                        card.spawn((
                            ImageNode { image: handle.clone(), rect: Some(rect), ..default() },
                            Node {
                                width: Val::Px(64.0),
                                height: Val::Px(64.0),
                                margin: UiRect::bottom(Val::Px(4.0)),
                                ..default()
                            },
                        ));
                    }

                    // Remove button
                    card.spawn((
                        RemoveCustomBlockButton { atlas_index: idx },
                        Button,
                        Node { padding: UiRect::axes(Val::Px(8.0), Val::Px(4.0)), ..default() },
                        BackgroundColor(Color::linear_rgba(0.4, 0.15, 0.15, 1.0)),
                    )).with_children(|btn| {
                        btn.spawn((
                            Text::new("Remove"),
                            TextFont { font_size: 12.0, ..default() },
                            TextColor(Color::WHITE),
                        ));
                    });
                });
            }
        });
    }

    // Add Block button (only if there's room in the atlas)
    let has_room = custom_registry.map(|r| r.has_room()).unwrap_or(true);
    if has_room {
        parent.spawn((
            AddCustomBlockButton,
            Button,
            Node {
                padding: UiRect::axes(Val::Px(16.0), Val::Px(8.0)),
                margin: UiRect::top(Val::Px(8.0)),
                ..default()
            },
            BackgroundColor(Color::linear_rgba(0.15, 0.30, 0.15, 1.0)),
        )).with_children(|btn| {
            btn.spawn((
                Text::new("+ Add Custom Block"),
                TextFont { font_size: 14.0, ..default() },
                TextColor(Color::WHITE),
            ));
        });
    } else {
        parent.spawn((
            Text::new(format!("Maximum {} custom blocks reached", crate::block_types::MAX_CUSTOM_BLOCKS)),
            TextFont { font_size: 13.0, ..default() },
            TextColor(Color::linear_rgb(0.5, 0.5, 0.5)),
            Node { margin: UiRect::top(Val::Px(8.0)), ..default() },
        ));
    }
}

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------

fn section_header(parent: &mut ChildSpawnerCommands, text: &str) {
    parent.spawn((
        Text::new(text),
        TextFont { font_size: 22.0, ..default() },
        TextColor(Color::linear_rgb(0.8, 0.8, 0.2)),
        Node { margin: UiRect::bottom(Val::Px(12.0)), ..default() },
    ));
}

fn toggle_button<M: Component + Default>(
    parent: &mut ChildSpawnerCommands,
    label: &str,
    is_on: bool,
) {
    let checkbox = if is_on { "[x]" } else { "[ ]" };
    parent
        .spawn((
            M::default(),
            Button,
            Node {
                padding: UiRect::axes(Val::Px(12.0), Val::Px(8.0)),
                margin: UiRect::bottom(Val::Px(6.0)),
                ..default()
            },
            BackgroundColor(Color::linear_rgba(0.2, 0.2, 0.28, 1.0)),
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(format!("{} {}", checkbox, label)),
                TextFont { font_size: 17.0, ..default() },
                TextColor(Color::WHITE),
            ));
        });
}

fn value_row<Up: Component + Default, Down: Component + Default, Label: Component + Default>(
    parent: &mut ChildSpawnerCommands,
    name: &str,
    value: &str,
) {
    parent.spawn(Node {
        flex_direction: FlexDirection::Row,
        align_items: AlignItems::Center,
        margin: UiRect::bottom(Val::Px(6.0)),
        ..default()
    })
    .with_children(|row| {
        // Label
        row.spawn((
            Text::new(format!("{}:  ", name)),
            TextFont { font_size: 17.0, ..default() },
            TextColor(Color::linear_rgb(0.85, 0.85, 0.85)),
        ));

        // Minus button
        row.spawn((
            Down::default(),
            Button,
            Node {
                padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                margin: UiRect::right(Val::Px(4.0)),
                ..default()
            },
            BackgroundColor(Color::linear_rgba(0.3, 0.2, 0.2, 1.0)),
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new("-"),
                TextFont { font_size: 18.0, ..default() },
                TextColor(Color::WHITE),
            ));
        });

        // Value display
        row.spawn((
            Label::default(),
            Text::new(value),
            TextFont { font_size: 17.0, ..default() },
            TextColor(Color::linear_rgb(1.0, 1.0, 0.6)),
            Node { min_width: Val::Px(60.0), ..default() },
        ));

        // Plus button
        row.spawn((
            Up::default(),
            Button,
            Node {
                padding: UiRect::axes(Val::Px(10.0), Val::Px(4.0)),
                margin: UiRect::left(Val::Px(4.0)),
                ..default()
            },
            BackgroundColor(Color::linear_rgba(0.2, 0.3, 0.2, 1.0)),
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new("+"),
                TextFont { font_size: 18.0, ..default() },
                TextColor(Color::WHITE),
            ));
        });
    });
}

// ---------------------------------------------------------------------------
// Helper to rebuild current menu screen
// ---------------------------------------------------------------------------

fn rebuild_menu(
    commands: &mut Commands,
    menu_state: &MenuState,
    menu_query: &Query<Entity, With<SettingsMenu>>,
    game_settings: &GameSettings,
    windows: &Query<(&Window, &CursorOptions), With<PrimaryWindow>>,
    atlas: Option<&crate::chunk_manager::BlockAtlas>,
    previews: Option<&TexturePreviews>,
    tex_error: Option<&TextureLoadError>,
    custom_registry: Option<&crate::block_types::CustomBlockRegistry>,
) {
    for entity in menu_query {
        commands.entity(entity).despawn();
    }
    let is_fullscreen = windows
        .iter()
        .next()
        .map(|(w, _)| !matches!(w.mode, WindowMode::Windowed))
        .unwrap_or(false);
    spawn_current_screen(commands, menu_state, game_settings, is_fullscreen, atlas, previews, tex_error, custom_registry);
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

fn handle_tab_buttons(
    tab_query: Query<(&Interaction, &TabButton), Changed<Interaction>>,
    mut menu_state: ResMut<MenuState>,
    mut commands: Commands,
    menu_query: Query<Entity, With<SettingsMenu>>,
    windows: Query<(&Window, &CursorOptions), With<PrimaryWindow>>,
    game_settings: Res<GameSettings>,
    block_atlas: Option<Res<crate::chunk_manager::BlockAtlas>>,
    tex_previews: Res<TexturePreviews>,
    custom_registry: Res<crate::block_types::CustomBlockRegistry>,
) {
    for (interaction, tab_btn) in &tab_query {
        if *interaction == Interaction::Pressed && tab_btn.0 != menu_state.settings_tab {
            menu_state.settings_tab = tab_btn.0;
            rebuild_menu(&mut commands, &menu_state, &menu_query, &game_settings, &windows, block_atlas.as_deref(), Some(&tex_previews), None, Some(&custom_registry));
        }
    }
}

// ---------------------------------------------------------------------------
// Setting button handlers
// ---------------------------------------------------------------------------
// RULE: UI interaction handlers must be instant. They may only update
// GameSettings fields and rebuild the menu for visual feedback. No rendering,
// window, GPU, or pipeline logic may run directly on click. Heavy work
// (present mode, render targets, shadow config) must be deferred to reactive
// systems that use is_changed() on GameSettings (e.g. apply_fps_mode_changes
// in PostUpdate).

fn handle_setting_buttons(
    fullscreen_q: Query<&Interaction, (Changed<Interaction>, With<FullscreenButton>)>,
    vsync_q: Query<&Interaction, (Changed<Interaction>, With<VsyncButton>)>,
    bright_up_q: Query<&Interaction, (Changed<Interaction>, With<BrightnessUpButton>)>,
    bright_dn_q: Query<&Interaction, (Changed<Interaction>, With<BrightnessDownButton>)>,
    shadow_up_q: Query<&Interaction, (Changed<Interaction>, With<ShadowUpButton>)>,
    shadow_dn_q: Query<&Interaction, (Changed<Interaction>, With<ShadowDownButton>)>,
    dist_up_q: Query<&Interaction, (Changed<Interaction>, With<DrawDistUpButton>)>,
    dist_dn_q: Query<&Interaction, (Changed<Interaction>, With<DrawDistDownButton>)>,
    clouds_q: Query<&Interaction, (Changed<Interaction>, With<CloudsButton>)>,
    mut windows: Query<(&mut Window, &mut CursorOptions), With<PrimaryWindow>>,
    mut game_settings: ResMut<GameSettings>,
    mut menu_state: ResMut<MenuState>,
    mut commands: Commands,
    menu_query: Query<Entity, With<SettingsMenu>>,
    mut cloud_vis: Query<&mut Visibility, With<Cloud>>,
    block_atlas: Option<Res<crate::chunk_manager::BlockAtlas>>,
) {
    let mut changed = false;

    // Fullscreen
    for i in &fullscreen_q {
        if *i == Interaction::Pressed {
            if let Ok((mut window, _)) = windows.single_mut() {
                window.mode = match window.mode {
                    WindowMode::Windowed => {
                        WindowMode::BorderlessFullscreen(bevy::window::MonitorSelection::Primary)
                    }
                    _ => WindowMode::Windowed,
                };
                menu_state.cursor_captured = false;
            }
            changed = true;
        }
    }

    // VSync — only toggle the setting; present_mode is applied by
    // apply_fps_mode_changes in PostUpdate to keep this handler lightweight.
    for i in &vsync_q {
        if *i == Interaction::Pressed {
            game_settings.vsync = !game_settings.vsync;
            changed = true;
        }
    }

    // Brightness
    for i in &bright_up_q {
        if *i == Interaction::Pressed {
            game_settings.brightness = (game_settings.brightness + 0.1).min(2.0);
            changed = true;
        }
    }
    for i in &bright_dn_q {
        if *i == Interaction::Pressed {
            game_settings.brightness = (game_settings.brightness - 0.1).max(0.1);
            changed = true;
        }
    }

    // Shadow intensity
    for i in &shadow_up_q {
        if *i == Interaction::Pressed {
            game_settings.shadow_intensity = (game_settings.shadow_intensity + 0.1).min(2.0);
            changed = true;
        }
    }
    for i in &shadow_dn_q {
        if *i == Interaction::Pressed {
            game_settings.shadow_intensity = (game_settings.shadow_intensity - 0.1).max(0.0);
            changed = true;
        }
    }

    // Draw distance
    for i in &dist_up_q {
        if *i == Interaction::Pressed {
            game_settings.render_distance = (game_settings.render_distance + 1).min(16);
            changed = true;
        }
    }
    for i in &dist_dn_q {
        if *i == Interaction::Pressed {
            game_settings.render_distance = (game_settings.render_distance - 1).max(2);
            changed = true;
        }
    }

    // Clouds
    for i in &clouds_q {
        if *i == Interaction::Pressed {
            game_settings.clouds_enabled = !game_settings.clouds_enabled;
            let vis = if game_settings.clouds_enabled {
                Visibility::Inherited
            } else {
                Visibility::Hidden
            };
            for mut v in &mut cloud_vis {
                *v = vis;
            }
            changed = true;
        }
    }

    // Rebuild menu to reflect changes
    if changed {
        if let Ok((window, _)) = windows.single_mut() {
            save_window_settings(&window, &mut game_settings);
        }

        for entity in &menu_query {
            commands.entity(entity).despawn();
        }

        let is_fullscreen = windows
            .iter()
            .next()
            .map(|(w, _)| !matches!(w.mode, WindowMode::Windowed))
            .unwrap_or(false);

        spawn_current_screen(&mut commands, &menu_state, &game_settings, is_fullscreen, block_atlas.as_deref(), None, None, None);
    }
}

// ---------------------------------------------------------------------------
// 120 FPS button handler
// ---------------------------------------------------------------------------

/// Lightweight button handler: only updates GameSettings and rebuilds the
/// menu for immediate visual feedback. Heavy side effects (vsync, present
/// mode) are handled by apply_fps_mode_changes via change detection.
fn handle_120fps_button(
    interaction_q: Query<&Interaction, (Changed<Interaction>, With<Fps120Button>)>,
    mut game_settings: ResMut<GameSettings>,
    windows: Query<(&Window, &CursorOptions), With<PrimaryWindow>>,
    menu_state: Res<MenuState>,
    mut commands: Commands,
    menu_query: Query<Entity, With<SettingsMenu>>,
) {
    #[cfg(debug_assertions)]
    let _start = std::time::Instant::now();

    let mut changed = false;

    for interaction in &interaction_q {
        if *interaction == Interaction::Pressed {
            game_settings.fps_120_mode = !game_settings.fps_120_mode;
            game_settings.vsync = true;
            if !game_settings.render_scale_user_override {
                game_settings.render_scale = if game_settings.fps_120_mode {
                    0.75
                } else {
                    1.0
                };
            }
            changed = true;
        }
    }

    if changed {
        game_settings.save();

        // Rebuild menu so the checkbox reflects the new state immediately.
        for entity in &menu_query {
            commands.entity(entity).despawn();
        }
        let is_fullscreen = windows
            .iter()
            .next()
            .map(|(w, _)| !matches!(w.mode, WindowMode::Windowed))
            .unwrap_or(false);
        spawn_current_screen(&mut commands, &menu_state, &game_settings, is_fullscreen, None, None, None, None);
    }

    #[cfg(debug_assertions)]
    {
        let elapsed = _start.elapsed();
        if elapsed.as_millis() > 2 {
            bevy::log::warn!(
                "handle_120fps_button took {}ms — UI handlers should be <2ms. \
                 Move heavy work to a reactive PostUpdate system.",
                elapsed.as_millis()
            );
        }
    }
}

/// Runs in PostUpdate: reacts to GameSettings changes to apply window-level
/// side effects (vsync, present mode). Decoupled from UI interaction handlers
/// so button systems remain lightweight and window mutations don't stall UI.
fn apply_fps_mode_changes(
    game_settings: Res<GameSettings>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    if !game_settings.is_changed() {
        return;
    }

    if let Ok(mut window) = windows.single_mut() {
        if game_settings.vsync {
            window.present_mode = PresentMode::AutoVsync;
        } else {
            window.present_mode = PresentMode::AutoNoVsync;
        }
    }
}

// ---------------------------------------------------------------------------
// Render scale button handler
// ---------------------------------------------------------------------------

fn handle_render_scale_buttons(
    up_q: Query<&Interaction, (Changed<Interaction>, With<RenderScaleUpButton>)>,
    dn_q: Query<&Interaction, (Changed<Interaction>, With<RenderScaleDownButton>)>,
    mut game_settings: ResMut<GameSettings>,
    menu_state: Res<MenuState>,
    mut commands: Commands,
    menu_query: Query<Entity, With<SettingsMenu>>,
    windows: Query<(&Window, &CursorOptions), With<PrimaryWindow>>,
) {
    let mut changed = false;

    for i in &up_q {
        if *i == Interaction::Pressed {
            game_settings.render_scale = (game_settings.render_scale + 0.05).min(1.0);
            game_settings.render_scale_user_override = true;
            changed = true;
        }
    }
    for i in &dn_q {
        if *i == Interaction::Pressed {
            game_settings.render_scale = (game_settings.render_scale - 0.05).max(0.5);
            game_settings.render_scale_user_override = true;
            changed = true;
        }
    }

    if changed {
        game_settings.save();
        rebuild_menu(&mut commands, &menu_state, &menu_query, &game_settings, &windows, None, None, None, None);
    }
}

// ---------------------------------------------------------------------------
// Texture size button handler
// ---------------------------------------------------------------------------

fn handle_tex_size_buttons(
    up_q: Query<&Interaction, (Changed<Interaction>, With<TexSizeUpButton>)>,
    dn_q: Query<&Interaction, (Changed<Interaction>, With<TexSizeDownButton>)>,
    sun_up_q: Query<&Interaction, (Changed<Interaction>, With<SunUpButton>)>,
    sun_dn_q: Query<&Interaction, (Changed<Interaction>, With<SunDownButton>)>,
    gamma_up_q: Query<&Interaction, (Changed<Interaction>, With<GammaUpButton>)>,
    gamma_dn_q: Query<&Interaction, (Changed<Interaction>, With<GammaDownButton>)>,
    contrast_up_q: Query<&Interaction, (Changed<Interaction>, With<ContrastUpButton>)>,
    contrast_dn_q: Query<&Interaction, (Changed<Interaction>, With<ContrastDownButton>)>,
    aniso_up_q: Query<&Interaction, (Changed<Interaction>, With<AnisoUpButton>)>,
    aniso_dn_q: Query<&Interaction, (Changed<Interaction>, With<AnisoDownButton>)>,
    aa_q: Query<&Interaction, (Changed<Interaction>, With<AaCycleButton>)>,
    mut game_settings: ResMut<GameSettings>,
    menu_state: Res<MenuState>,
    mut commands: Commands,
    menu_query: Query<Entity, With<SettingsMenu>>,
    windows: Query<(&Window, &CursorOptions), With<PrimaryWindow>>,
) {
    let tex_steps: [u32; 4] = [64, 128, 256, 512];
    let mut changed = false;

    for i in &up_q {
        if *i == Interaction::Pressed {
            let cur = game_settings.texture_size;
            if let Some(&next) = tex_steps.iter().find(|&&s| s > cur) {
                game_settings.texture_size = next;
                changed = true;
            }
        }
    }
    for i in &dn_q {
        if *i == Interaction::Pressed {
            let cur = game_settings.texture_size;
            if let Some(&prev) = tex_steps.iter().rev().find(|&&s| s < cur) {
                game_settings.texture_size = prev;
                changed = true;
            }
        }
    }

    // Gamma
    for i in &gamma_up_q {
        if *i == Interaction::Pressed {
            game_settings.gamma = (game_settings.gamma + 0.1).min(3.0);
            changed = true;
        }
    }
    for i in &gamma_dn_q {
        if *i == Interaction::Pressed {
            game_settings.gamma = (game_settings.gamma - 0.1).max(0.1);
            changed = true;
        }
    }

    // Contrast
    for i in &contrast_up_q {
        if *i == Interaction::Pressed {
            game_settings.contrast = (game_settings.contrast + 0.1).min(3.0);
            changed = true;
        }
    }
    for i in &contrast_dn_q {
        if *i == Interaction::Pressed {
            game_settings.contrast = (game_settings.contrast - 0.1).max(0.1);
            changed = true;
        }
    }

    // Anisotropic filtering (steps: 1, 2, 4, 8, 16)
    let aniso_steps: [u16; 5] = [1, 2, 4, 8, 16];
    for i in &aniso_up_q {
        if *i == Interaction::Pressed {
            let cur = game_settings.anisotropic_filtering;
            if let Some(&next) = aniso_steps.iter().find(|&&s| s > cur) {
                game_settings.anisotropic_filtering = next;
                changed = true;
            }
        }
    }
    for i in &aniso_dn_q {
        if *i == Interaction::Pressed {
            let cur = game_settings.anisotropic_filtering;
            if let Some(&prev) = aniso_steps.iter().rev().find(|&&s| s < cur) {
                game_settings.anisotropic_filtering = prev;
                changed = true;
            }
        }
    }

    // Anti-aliasing cycle: off -> msaa2 -> msaa4 -> taa -> off
    for i in &aa_q {
        if *i == Interaction::Pressed {
            game_settings.anti_aliasing = match game_settings.anti_aliasing.as_str() {
                "off" => "msaa2".to_string(),
                "msaa2" => "msaa4".to_string(),
                "msaa4" => "taa".to_string(),
                "taa" => "off".to_string(),
                _ => "off".to_string(),
            };
            changed = true;
        }
    }

    // Sun intensity
    for i in &sun_up_q {
        if *i == Interaction::Pressed {
            game_settings.shadow_intensity = (game_settings.shadow_intensity + 0.1).min(3.0);
            changed = true;
        }
    }
    for i in &sun_dn_q {
        if *i == Interaction::Pressed {
            game_settings.shadow_intensity = (game_settings.shadow_intensity - 0.1).max(0.0);
            changed = true;
        }
    }

    if changed {
        game_settings.save();
        rebuild_menu(&mut commands, &menu_state, &menu_query, &game_settings, &windows, None, None, None, None);
    }
}

// ---------------------------------------------------------------------------
// Graphics settings button handlers (exposure, tonemapping, FOV, SSAO, SMAA)
// ---------------------------------------------------------------------------

fn handle_graphics_settings_buttons(
    exposure_up_q: Query<&Interaction, (Changed<Interaction>, With<ExposureUpButton>)>,
    exposure_dn_q: Query<&Interaction, (Changed<Interaction>, With<ExposureDownButton>)>,
    tm_q: Query<&Interaction, (Changed<Interaction>, With<TonemappingCycleButton>)>,
    fov_up_q: Query<&Interaction, (Changed<Interaction>, With<FovUpButton>)>,
    fov_dn_q: Query<&Interaction, (Changed<Interaction>, With<FovDownButton>)>,
    ssao_q: Query<&Interaction, (Changed<Interaction>, With<SsaoButton>)>,
    ssao_quality_q: Query<&Interaction, (Changed<Interaction>, With<SsaoQualityCycleButton>)>,
    smaa_q: Query<&Interaction, (Changed<Interaction>, With<SmaaCycleButton>)>,
    mut game_settings: ResMut<GameSettings>,
    menu_state: Res<MenuState>,
    mut commands: Commands,
    menu_query: Query<Entity, With<SettingsMenu>>,
    windows: Query<(&Window, &CursorOptions), With<PrimaryWindow>>,
    custom_registry: Res<crate::block_types::CustomBlockRegistry>,
) {
    let mut changed = false;

    // Exposure
    for i in &exposure_up_q {
        if *i == Interaction::Pressed {
            game_settings.exposure = (game_settings.exposure + 0.5).min(3.0);
            changed = true;
        }
    }
    for i in &exposure_dn_q {
        if *i == Interaction::Pressed {
            game_settings.exposure = (game_settings.exposure - 0.5).max(-3.0);
            changed = true;
        }
    }

    // Tonemapping cycle: none -> reinhard -> aces -> agx -> tony -> blender -> none
    for i in &tm_q {
        if *i == Interaction::Pressed {
            game_settings.tonemapping = match game_settings.tonemapping.as_str() {
                "none" => "reinhard".to_string(),
                "reinhard" => "aces".to_string(),
                "aces" => "agx".to_string(),
                "agx" => "tony".to_string(),
                "tony" => "blender".to_string(),
                "blender" => "none".to_string(),
                _ => "none".to_string(),
            };
            changed = true;
        }
    }

    // FOV
    for i in &fov_up_q {
        if *i == Interaction::Pressed {
            game_settings.fov = (game_settings.fov + 5.0).min(120.0);
            changed = true;
        }
    }
    for i in &fov_dn_q {
        if *i == Interaction::Pressed {
            game_settings.fov = (game_settings.fov - 5.0).max(40.0);
            changed = true;
        }
    }

    // SSAO toggle
    for i in &ssao_q {
        if *i == Interaction::Pressed {
            game_settings.ssao_enabled = !game_settings.ssao_enabled;
            changed = true;
        }
    }

    // SSAO Quality cycle: low -> medium -> high -> ultra -> low
    for i in &ssao_quality_q {
        if *i == Interaction::Pressed {
            game_settings.ssao_quality = match game_settings.ssao_quality.as_str() {
                "low" => "medium".to_string(),
                "medium" => "high".to_string(),
                "high" => "ultra".to_string(),
                "ultra" => "low".to_string(),
                _ => "medium".to_string(),
            };
            changed = true;
        }
    }

    // SMAA cycle: off -> low -> medium -> high -> ultra -> off
    for i in &smaa_q {
        if *i == Interaction::Pressed {
            game_settings.smaa_mode = match game_settings.smaa_mode.as_str() {
                "off" => "low".to_string(),
                "low" => "medium".to_string(),
                "medium" => "high".to_string(),
                "high" => "ultra".to_string(),
                "ultra" => "off".to_string(),
                _ => "off".to_string(),
            };
            changed = true;
        }
    }

    if changed {
        game_settings.save();
        rebuild_menu(&mut commands, &menu_state, &menu_query, &game_settings, &windows, None, None, None, None);
    }
}

// ---------------------------------------------------------------------------
// Load texture button handler (opens file dialog)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Load texture button handler (reads from textures/ directory)
// ---------------------------------------------------------------------------

/// Opens a native file dialog when a "Load..." texture button is clicked.
/// The dialog runs asynchronously on the IO thread pool. The selected file
/// is read on that thread and the bytes are returned via TextureDialogTask.
/// The actual atlas update happens in poll_texture_dialog.
fn handle_load_texture_buttons(
    load_btn_q: Query<(&Interaction, &LoadTextureButton), Changed<Interaction>>,
    mut commands: Commands,
    existing: Option<Res<TextureDialogTask>>,
    mut tex_error: ResMut<TextureLoadError>,
) {
    // Don't open a second dialog if one is already running
    if existing.is_some() {
        return;
    }

    for (interaction, btn) in &load_btn_q {
        if *interaction != Interaction::Pressed {
            continue;
        }

        // Clear any previous error when opening a new dialog
        tex_error.message = None;

        let block_name = btn.block_name.clone();
        let block_idx = btn.block_idx;
        let start_dir = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));

        let task = bevy::tasks::IoTaskPool::get().spawn(async move {
            let handle = rfd::AsyncFileDialog::new()
                .set_title("Select Block Texture")
                .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "tga"])
                .set_directory(&start_dir)
                .pick_file()
                .await;

            let handle = handle?;
            let path = handle.path().to_path_buf();
            let data = std::fs::read(&path).ok()?;
            Some((path, data))
        });

        commands.insert_resource(TextureDialogTask {
            task,
            block_name,
            block_idx,
        });
        return;
    }
}

/// Polls the texture file dialog. When the user selects a file, decodes the
/// image, copies it into the atlas, saves the path in settings, and rebuilds
/// the menu to reflect the change. Cancel is handled gracefully (no-op).
fn poll_texture_dialog(
    mut commands: Commands,
    mut dialog: Option<ResMut<TextureDialogTask>>,
    mut game_settings: ResMut<GameSettings>,
    menu_state: Res<MenuState>,
    menu_query: Query<Entity, With<SettingsMenu>>,
    windows: Query<(&Window, &CursorOptions), With<PrimaryWindow>>,
    mut images: ResMut<Assets<Image>>,
    mut block_atlas: Option<ResMut<crate::chunk_manager::BlockAtlas>>,
    mut previews: ResMut<TexturePreviews>,
    mut tex_error: ResMut<TextureLoadError>,
) {
    let Some(ref mut dialog) = dialog else {
        return;
    };

    let Some(result) = bevy::tasks::futures::check_ready(&mut dialog.task) else {
        return;
    };

    let block_name = dialog.block_name.clone();
    let block_idx = dialog.block_idx;
    commands.remove_resource::<TextureDialogTask>();

    let Some((path, img_data)) = result else {
        info!("Texture dialog cancelled.");
        return;
    };

    // --- Validation ---
    let file_name = path.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string());

    let dyn_img = match image::load_from_memory(&img_data) {
        Ok(img) => img,
        Err(e) => {
            let msg = format!("Not a valid image: {} ({})", file_name, e);
            warn!("{}", msg);
            tex_error.message = Some(msg);
            let atlas_ref = block_atlas.as_deref();
            rebuild_menu(&mut commands, &menu_state, &menu_query, &game_settings, &windows, atlas_ref, Some(&previews), Some(&tex_error), None);
            return;
        }
    };

    let (w, h) = (dyn_img.width(), dyn_img.height());
    if w < MIN_TEXTURE_SIZE || h < MIN_TEXTURE_SIZE {
        let msg = format!(
            "Image too small: {} is {}x{} (minimum {}x{})",
            file_name, w, h, MIN_TEXTURE_SIZE, MIN_TEXTURE_SIZE
        );
        warn!("{}", msg);
        tex_error.message = Some(msg);
        let atlas_ref = block_atlas.as_deref();
        rebuild_menu(&mut commands, &menu_state, &menu_query, &game_settings, &windows, atlas_ref, Some(&previews), Some(&tex_error), None);
        return;
    }

    // Validation passed — clear any previous error
    tex_error.message = None;

    let ts = block_atlas.as_ref().map(|a| a.tile_size).unwrap_or(64);
    let resized = dyn_img.resize_exact(ts, ts, image::imageops::FilterType::Lanczos3);
    let rgba = resized.to_rgba8();

    if let Some(ref mut atlas) = block_atlas {
        if let Some(img) = images.get_mut(&atlas.image_handle) {
            if let Some(data) = img.data.as_mut() {
                crate::chunk_manager::copy_image_to_atlas_tile(
                    data,
                    block_idx,
                    &rgba,
                    atlas.tile_size,
                    atlas.atlas_size,
                );
                atlas.loaded_textures.insert(block_idx);
            }
        }
    }

    // Create a UI-only preview image from the source file (without atlas
    // border darkening). Resized to 64px for reasonable memory usage.
    // Remove old preview handle if one exists (prevents leaking).
    let preview_size = 64u32;
    let preview_rgba = dyn_img
        .resize(preview_size, preview_size, image::imageops::FilterType::Lanczos3)
        .to_rgba8();
    let (pw, ph) = preview_rgba.dimensions();
    let preview_image = Image::new(
        bevy::render::render_resource::Extent3d {
            width: pw,
            height: ph,
            depth_or_array_layers: 1,
        },
        bevy::render::render_resource::TextureDimension::D2,
        preview_rgba.into_raw(),
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        bevy::asset::RenderAssetUsages::MAIN_WORLD | bevy::asset::RenderAssetUsages::RENDER_WORLD,
    );
    if let Some(old) = previews.handles.remove(&block_name) {
        images.remove(&old);
    }
    previews.handles.insert(block_name.clone(), images.add(preview_image));

    let path_str = path.to_string_lossy().to_string();
    game_settings.block_textures.insert(block_name.clone(), path_str.clone());
    game_settings.save();
    info!("Loaded texture for {} from {}", block_name, path_str);

    let atlas_ref = block_atlas.as_deref();
    rebuild_menu(&mut commands, &menu_state, &menu_query, &game_settings, &windows, atlas_ref, Some(&previews), Some(&tex_error), None);
}

// ---------------------------------------------------------------------------
// Remove texture button handler
// ---------------------------------------------------------------------------

fn handle_remove_texture_buttons(
    remove_btn_q: Query<(&Interaction, &RemoveTextureButton), Changed<Interaction>>,
    mut game_settings: ResMut<GameSettings>,
    menu_state: Res<MenuState>,
    mut commands: Commands,
    menu_query: Query<Entity, With<SettingsMenu>>,
    windows: Query<(&Window, &CursorOptions), With<PrimaryWindow>>,
    mut images: ResMut<Assets<Image>>,
    mut block_atlas: Option<ResMut<crate::chunk_manager::BlockAtlas>>,
    mut previews: ResMut<TexturePreviews>,
) {
    let mut changed = false;

    for (interaction, btn) in &remove_btn_q {
        if *interaction != Interaction::Pressed {
            continue;
        }

        game_settings.block_textures.remove(&btn.block_name);

        // Remove the preview image handle (frees memory)
        if let Some(old) = previews.handles.remove(&btn.block_name) {
            images.remove(&old);
        }

        // Restore the solid color tile in the atlas
        let block_type = block_type_from_name(&btn.block_name);
        if let Some(ref mut atlas) = block_atlas {
            if let Some(img) = images.get_mut(&atlas.image_handle) {
                if let Some(data) = img.data.as_mut() {
                    crate::chunk_manager::fill_atlas_tile(
                        data,
                        btn.block_idx,
                        block_type.color(),
                        atlas.tile_size,
                        atlas.atlas_size,
                    );
                    atlas.loaded_textures.remove(&btn.block_idx);
                }
            }
        }

        game_settings.save();
        changed = true;
    }

    if changed {
        let atlas_ref = block_atlas.as_deref();
        rebuild_menu(&mut commands, &menu_state, &menu_query, &game_settings, &windows, atlas_ref, Some(&previews), None, None);
    }
}

fn block_type_from_name(name: &str) -> BlockType {
    match name {
        "Grass" => BlockType::GRASS,
        "Dirt" => BlockType::DIRT,
        "Stone" => BlockType::STONE,
        "Sand" => BlockType::SAND,
        "Wood" => BlockType::WOOD,
        "Diamond" => BlockType::DIAMOND,
        "Lantern" => BlockType::LANTERN,
        "Leaves" => BlockType::LEAVES,
        "StoneBrick" => BlockType::STONE_BRICK,
        _ => BlockType::AIR,
    }
}

// ---------------------------------------------------------------------------
// Save helper
// ---------------------------------------------------------------------------

fn save_window_settings(window: &Window, settings: &mut GameSettings) {
    settings.fullscreen = !matches!(window.mode, WindowMode::Windowed);
    if !settings.fullscreen {
        settings.window_width = window.resolution.width();
        settings.window_height = window.resolution.height();
    }
    settings.save();
}

// ---------------------------------------------------------------------------
// Add Custom Block
// ---------------------------------------------------------------------------

/// Opens a file dialog to select a texture for a new custom block.
fn handle_add_block_button(
    query: Query<&Interaction, (Changed<Interaction>, With<AddCustomBlockButton>)>,
    mut commands: Commands,
    existing: Option<Res<AddBlockDialogTask>>,
    registry: Res<crate::block_types::CustomBlockRegistry>,
) {
    if existing.is_some() || !registry.has_room() {
        return;
    }

    for interaction in &query {
        if *interaction != Interaction::Pressed {
            continue;
        }

        let start_dir = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
        let task = bevy::tasks::IoTaskPool::get().spawn(async move {
            let handle = rfd::AsyncFileDialog::new()
                .set_title("Select Texture for New Block")
                .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "tga"])
                .set_directory(&start_dir)
                .pick_file()
                .await;

            let handle = handle?;
            let path = handle.path().to_path_buf();
            let data = std::fs::read(&path).ok()?;
            Some((path, data))
        });

        commands.insert_resource(AddBlockDialogTask { task });
        return;
    }
}

/// Polls the add-block file dialog. On success, creates a new custom block
/// definition, loads its texture into the atlas, and persists to settings.
fn poll_add_block_dialog(
    mut commands: Commands,
    mut dialog: Option<ResMut<AddBlockDialogTask>>,
    mut registry: ResMut<crate::block_types::CustomBlockRegistry>,
    mut game_settings: ResMut<GameSettings>,
    menu_state: Res<MenuState>,
    menu_query: Query<Entity, With<SettingsMenu>>,
    windows: Query<(&Window, &CursorOptions), With<PrimaryWindow>>,
    mut images: ResMut<Assets<Image>>,
    mut block_atlas: Option<ResMut<crate::chunk_manager::BlockAtlas>>,
    mut tex_error: ResMut<TextureLoadError>,
) {
    let Some(ref mut dialog) = dialog else { return };
    let Some(result) = bevy::tasks::futures::check_ready(&mut dialog.task) else { return };

    commands.remove_resource::<AddBlockDialogTask>();

    let Some((path, img_data)) = result else {
        info!("Add block dialog cancelled.");
        return;
    };

    let file_name = path.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "custom".to_string());

    let dyn_img = match image::load_from_memory(&img_data) {
        Ok(img) => img,
        Err(e) => {
            let msg = format!("Not a valid image: {} ({})", file_name, e);
            warn!("{}", msg);
            tex_error.message = Some(msg);
            return;
        }
    };

    // Derive block name from filename (without extension)
    let block_name = path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| format!("Custom{}", registry.count() + 1));

    // Compute average color for fallback
    let thumb = dyn_img.resize(1, 1, image::imageops::FilterType::Lanczos3).to_rgba8();
    let px = thumb.get_pixel(0, 0);
    let avg_color = [px[0] as f32 / 255.0, px[1] as f32 / 255.0, px[2] as f32 / 255.0, px[3] as f32 / 255.0];

    // Verify the atlas is available before committing. The atlas texture
    // must be written BEFORE the registry is updated so that chunks
    // remeshed by on_block_registry_changed already have the correct
    // texture data on the GPU.
    let Some(ref mut atlas) = block_atlas else {
        tex_error.message = Some("Block atlas not available".to_string());
        return;
    };
    let Some(img) = images.get_mut(&atlas.image_handle) else {
        tex_error.message = Some("Atlas image not available".to_string());
        return;
    };

    // Write texture into the atlas tile FIRST.
    // images.get_mut() emits AssetEvent::Modified → Bevy re-uploads the
    // atlas texture to the GPU on the same frame. No restart required.
    let Some(atlas_index) = registry.next_index() else {
        tex_error.message = Some("No room for more custom blocks".to_string());
        return;
    };

    let ts = atlas.tile_size;
    let resized = dyn_img.resize_exact(ts, ts, image::imageops::FilterType::Lanczos3);
    let rgba = resized.to_rgba8();

    let Some(data) = img.data.as_mut() else {
        tex_error.message = Some("Atlas image data not available".to_string());
        return;
    };
    crate::chunk_manager::copy_image_to_atlas_tile(
        data,
        atlas_index,
        &rgba,
        atlas.tile_size,
        atlas.atlas_size,
    );
    atlas.loaded_textures.insert(atlas_index);

    // Now register in the runtime registry. The atlas tile is already
    // populated, so when on_block_registry_changed triggers remeshing,
    // the GPU texture is correct.
    registry.add(crate::block_types::CustomBlockEntry {
        name: block_name.clone(),
        color: Color::linear_rgba(avg_color[0], avg_color[1], avg_color[2], avg_color[3]),
        atlas_index,
    });

    // Persist to settings
    game_settings.custom_blocks.push(crate::settings::CustomBlockDef {
        name: block_name.clone(),
        texture_path: path.to_string_lossy().to_string(),
        color: avg_color,
    });
    game_settings.save();

    // Chunk remeshing is handled reactively by on_block_registry_changed
    // in chunk_manager.rs (triggered by the registry mutation above).

    tex_error.message = None;
    info!("Added custom block '{}' at atlas index {}", block_name, atlas_index);

    let atlas_ref = block_atlas.as_deref();
    rebuild_menu(
        &mut commands, &menu_state, &menu_query, &game_settings, &windows,
        atlas_ref, None, None, Some(&registry),
    );
}

/// Removes a custom block from the registry and settings.
fn handle_remove_custom_block(
    query: Query<(&Interaction, &RemoveCustomBlockButton), Changed<Interaction>>,
    mut registry: ResMut<crate::block_types::CustomBlockRegistry>,
    mut game_settings: ResMut<GameSettings>,
    mut commands: Commands,
    menu_state: Res<MenuState>,
    menu_query: Query<Entity, With<SettingsMenu>>,
    windows: Query<(&Window, &CursorOptions), With<PrimaryWindow>>,
    mut images: ResMut<Assets<Image>>,
    mut block_atlas: Option<ResMut<crate::chunk_manager::BlockAtlas>>,
) {
    for (interaction, btn) in &query {
        if *interaction != Interaction::Pressed {
            continue;
        }

        // Clear the atlas tile BEFORE removing from registry so the GPU
        // texture is correct when on_block_registry_changed triggers remeshing.
        if let Some(ref mut atlas) = block_atlas {
            if let Some(img) = images.get_mut(&atlas.image_handle) {
                if let Some(data) = img.data.as_mut() {
                    crate::chunk_manager::fill_atlas_tile(
                        data,
                        btn.atlas_index,
                        Color::WHITE,
                        atlas.tile_size,
                        atlas.atlas_size,
                    );
                    atlas.loaded_textures.remove(&btn.atlas_index);
                }
            }
        }

        // Remove from registry (triggers on_block_registry_changed for remesh)
        let Some(removed) = registry.remove_by_index(btn.atlas_index) else {
            continue;
        };

        // Remove from settings by name (not by Vec index, which may have
        // shifted after prior mid-list removals).
        if let Some(pos) = game_settings.custom_blocks.iter().position(|d| d.name == removed.name) {
            game_settings.custom_blocks.remove(pos);
        }

        // Chunk remeshing is handled reactively by on_block_registry_changed
        // in chunk_manager.rs (triggered by the registry mutation above).

        game_settings.save();
        info!("Removed custom block at index {}", btn.atlas_index);

        let atlas_ref = block_atlas.as_deref();
        rebuild_menu(
            &mut commands, &menu_state, &menu_query, &game_settings, &windows,
            atlas_ref, None, None, Some(&registry),
        );
        return;
    }
}

// ---------------------------------------------------------------------------
// Inventory toggle (I key)
// ---------------------------------------------------------------------------

fn toggle_inventory(
    keys: Res<ButtonInput<KeyCode>>,
    mut inv_state: ResMut<InventoryState>,
    menu_state: Res<MenuState>,
    mut commands: Commands,
    panel_query: Query<Entity, With<InventoryPanel>>,
    mut cursor_query: Query<&mut CursorOptions, With<PrimaryWindow>>,
    game_settings: Res<GameSettings>,
    atlas: Option<Res<crate::chunk_manager::BlockAtlas>>,
    custom_registry: Res<crate::block_types::CustomBlockRegistry>,
) {
    // Also close inventory on Escape
    if inv_state.is_open && keys.just_pressed(KeyCode::Escape) {
        inv_state.is_open = false;
        for entity in &panel_query {
            commands.entity(entity).despawn();
        }
        if let Ok(mut cursor) = cursor_query.single_mut() {
            grab_cursor(&mut cursor);
        }
        return;
    }

    if !keys.just_pressed(KeyCode::KeyI) {
        return;
    }
    if menu_state.is_open {
        return;
    }

    inv_state.is_open = !inv_state.is_open;

    if let Ok(mut cursor) = cursor_query.single_mut() {
        if inv_state.is_open {
            release_cursor(&mut cursor);
        } else {
            grab_cursor(&mut cursor);
        }
    }

    if inv_state.is_open {
        spawn_inventory(&mut commands, &game_settings, atlas.as_deref(), Some(&custom_registry));
    } else {
        for entity in &panel_query {
            commands.entity(entity).despawn();
        }
    }
}

fn spawn_inventory(
    commands: &mut Commands,
    _settings: &GameSettings,
    atlas: Option<&crate::chunk_manager::BlockAtlas>,
    custom_registry: Option<&crate::block_types::CustomBlockRegistry>,
) {
    commands
        .spawn((
            crate::UiOnly,
            InventoryPanel,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Percent(30.0),
                top: Val::Percent(20.0),
                width: Val::Percent(40.0),
                height: Val::Percent(60.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(16.0)),
                overflow: Overflow::scroll_y(),
                ..default()
            },
            BackgroundColor(Color::linear_rgba(0.08, 0.08, 0.12, 0.95)),
        ))
        .with_children(|root| {
            // Title
            root.spawn((
                Text::new("Inventory - Blocks (I to close)"),
                TextFont {
                    font_size: 24.0,
                    ..default()
                },
                TextColor(Color::WHITE),
                Node {
                    margin: UiRect::bottom(Val::Px(16.0)),
                    ..default()
                },
            ));

            // Grid of blocks
            root.spawn(Node {
                flex_direction: FlexDirection::Row,
                flex_wrap: FlexWrap::Wrap,
                ..default()
            })
            .with_children(|grid| {
                let blocks = [
                    BlockType::GRASS,
                    BlockType::DIRT,
                    BlockType::STONE,
                    BlockType::SAND,
                    BlockType::WOOD,
                    BlockType::DIAMOND,
                    BlockType::BEDROCK,
                    BlockType::LANTERN,
                    BlockType::BED,
                    BlockType::PILLOW,
                    BlockType::LEAVES,
                    BlockType::STONE_BRICK,
                ];

                let tile_size = atlas.map(|a| a.tile_size as f32).unwrap_or(64.0);

                for block in blocks {
                    let name = block.name();
                    let block_idx = block.index() as u32;
                    let tiles_per_row = crate::block_types::ATLAS_TILES_PER_ROW;
                    let tile_x = block_idx % tiles_per_row;
                    let tile_y = block_idx / tiles_per_row;

                    grid.spawn((
                        InventoryItem(block),
                        Button,
                        Node {
                            flex_direction: FlexDirection::Column,
                            align_items: AlignItems::Center,
                            padding: UiRect::all(Val::Px(8.0)),
                            margin: UiRect::all(Val::Px(4.0)),
                            width: Val::Px(80.0),
                            ..default()
                        },
                        BackgroundColor(Color::linear_rgba(0.2, 0.2, 0.25, 1.0)),
                    ))
                    .with_children(|item| {
                        // Show atlas tile (texture or solid color from the atlas)
                        if let Some(atlas_res) = atlas {
                            let rect = Rect::new(
                                tile_x as f32 * tile_size,
                                tile_y as f32 * tile_size,
                                (tile_x + 1) as f32 * tile_size,
                                (tile_y + 1) as f32 * tile_size,
                            );
                            item.spawn((
                                ImageNode {
                                    image: atlas_res.image_handle.clone(),
                                    rect: Some(rect),
                                    ..default()
                                },
                                Node {
                                    width: Val::Px(48.0),
                                    height: Val::Px(48.0),
                                    margin: UiRect::bottom(Val::Px(4.0)),
                                    ..default()
                                },
                            ));
                        } else {
                            // Fallback: solid color swatch
                            item.spawn((
                                Node {
                                    width: Val::Px(48.0),
                                    height: Val::Px(48.0),
                                    margin: UiRect::bottom(Val::Px(4.0)),
                                    ..default()
                                },
                                BackgroundColor(block.color()),
                            ));
                        }
                        // Name
                        item.spawn((
                            Text::new(name),
                            TextFont {
                                font_size: 12.0,
                                ..default()
                            },
                            TextColor(Color::WHITE),
                        ));
                    });
                }

                // Custom blocks from the registry
                if let Some(reg) = custom_registry {
                    for entry in reg.iter() {
                        let block = BlockType::from_u8(entry.atlas_index);
                        let tiles_per_row = crate::block_types::ATLAS_TILES_PER_ROW;
                        let tile_x = (entry.atlas_index as u32) % tiles_per_row;
                        let tile_y = (entry.atlas_index as u32) / tiles_per_row;

                        grid.spawn((
                            InventoryItem(block),
                            Button,
                            Node {
                                flex_direction: FlexDirection::Column,
                                align_items: AlignItems::Center,
                                padding: UiRect::all(Val::Px(8.0)),
                                margin: UiRect::all(Val::Px(4.0)),
                                width: Val::Px(80.0),
                                ..default()
                            },
                            BackgroundColor(Color::linear_rgba(0.2, 0.25, 0.2, 1.0)),
                        ))
                        .with_children(|item| {
                            if let Some(atlas_res) = atlas {
                                let rect = Rect::new(
                                    tile_x as f32 * tile_size,
                                    tile_y as f32 * tile_size,
                                    (tile_x + 1) as f32 * tile_size,
                                    (tile_y + 1) as f32 * tile_size,
                                );
                                item.spawn((
                                    ImageNode {
                                        image: atlas_res.image_handle.clone(),
                                        rect: Some(rect),
                                        ..default()
                                    },
                                    Node {
                                        width: Val::Px(64.0),
                                        height: Val::Px(64.0),
                                        margin: UiRect::bottom(Val::Px(4.0)),
                                        ..default()
                                    },
                                ));
                            }
                            item.spawn((
                                Text::new(&entry.name),
                                TextFont { font_size: 12.0, ..default() },
                                TextColor(Color::WHITE),
                            ));
                        });
                    }
                }
            });
        });
}

// ---------------------------------------------------------------------------
// Inventory click handler
// ---------------------------------------------------------------------------

fn handle_inventory_clicks(
    inv_items: Query<(&Interaction, &InventoryItem), Changed<Interaction>>,
    mut player_query: Query<&mut Player>,
    mut inv_state: ResMut<InventoryState>,
    mut commands: Commands,
    panel_query: Query<Entity, With<InventoryPanel>>,
    mut cursor_query: Query<&mut CursorOptions, With<PrimaryWindow>>,
    mut menu_state: ResMut<MenuState>,
) {
    for (interaction, item) in &inv_items {
        if *interaction == Interaction::Pressed {
            // Set selected block
            for mut player in &mut player_query {
                player.selected_block = item.0;
            }

            // Close inventory
            inv_state.is_open = false;
            for entity in &panel_query {
                commands.entity(entity).despawn();
            }

            // Re-grab cursor
            if let Ok(mut cursor) = cursor_query.single_mut() {
                grab_cursor(&mut cursor);
                menu_state.cursor_captured = true;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Derive Default for marker components
// ---------------------------------------------------------------------------
// Components listed in impl_default! must NOT #[derive(Default)].
// Default is provided exclusively by this macro — combining both causes E0119
// (conflicting implementations of trait `Default`).

macro_rules! impl_default {
    ($($t:ty),*) => { $(impl Default for $t { fn default() -> Self { Self } })* };
}

impl_default!(
    FullscreenButton, VsyncButton, Fps120Button,
    BrightnessUpButton, BrightnessDownButton,
    ShadowUpButton, ShadowDownButton,
    DrawDistUpButton, DrawDistDownButton,
    TexSizeUpButton, TexSizeDownButton,
    SunUpButton, SunDownButton,
    GammaUpButton, GammaDownButton,
    ContrastUpButton, ContrastDownButton,
    CloudsButton,
    BrightnessLabel, ShadowLabel, DrawDistLabel, TexSizeLabel, SunLabel,
    GammaLabel, ContrastLabel, AnisoLabel,
    AnisoUpButton, AnisoDownButton, AaCycleButton,
    RenderScaleUpButton, RenderScaleDownButton, RenderScaleLabel
);

/// Save game state (player position + block modifications) before quitting.
fn save_game_state(
    chunk_manager: Option<&crate::chunk_manager::ChunkManager>,
    player_query: &Query<(&Transform, &crate::player::Player)>,
) {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    struct SaveData {
        player_position: [f32; 3],
        player_yaw: f32,
        player_pitch: f32,
        home_position: Option<[f32; 3]>,
        modifications: Vec<SaveMod>,
    }
    #[derive(Serialize, Deserialize)]
    struct SaveMod {
        x: i32,
        y: i32,
        z: i32,
        block_type: u8,
    }

    let Some(cm) = chunk_manager else { return };
    let Some((transform, player)) = player_query.iter().next() else {
        return;
    };

    let modifications: Vec<SaveMod> = cm
        .modifications
        .iter()
        .map(|(pos, &block)| SaveMod {
            x: pos.x,
            y: pos.y,
            z: pos.z,
            block_type: block.index(),
        })
        .collect();

    let save = SaveData {
        player_position: transform.translation.to_array(),
        player_yaw: player.yaw,
        player_pitch: player.pitch,
        home_position: player.home_position.map(|p| p.to_array()),
        modifications,
    };

    let path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".metalworld_save.json");
    if let Ok(data) = serde_json::to_string_pretty(&save) {
        let _ = std::fs::write(path, data);
        info!("Game auto-saved on quit");
    }
}
