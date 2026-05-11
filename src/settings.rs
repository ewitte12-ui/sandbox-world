use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Persistent game settings, saved to ~/.metalworld_settings.json
/// (filename kept for backwards compatibility with existing installs)
///
/// NOTE: mouse_sensitivity and player_speed live in DevSettings, not here.
/// GameSettings holds display/rendering preferences that are persisted to
/// disk. Gameplay tuning values are in DevSettings (the single source of truth).
#[derive(Resource, Serialize, Deserialize, Clone)]
pub struct GameSettings {
    /// Retained for backwards-compat with existing settings files; ignored at
    /// runtime — mouse sensitivity is read from DevSettings.
    #[serde(default = "default_mouse_sensitivity")]
    #[allow(dead_code)]
    mouse_sensitivity: f32,
    pub show_grid: bool,
    pub contrast: f32,
    pub gamma: f32,
    pub brightness: f32,
    pub render_distance: i32,
    #[serde(default = "default_fov")]
    pub fov: f32,
    #[serde(default = "default_exposure")]
    pub exposure: f32,
    #[serde(default = "default_tonemapping")]
    pub tonemapping: String,
    #[serde(default = "default_ssao")]
    pub ssao_enabled: bool,
    #[serde(default = "default_ssao_quality")]
    pub ssao_quality: String,
    #[serde(default = "default_smaa")]
    pub smaa_mode: String,
    #[serde(default = "default_fullscreen")]
    pub fullscreen: bool,
    #[serde(default = "default_window_width")]
    pub window_width: f32,
    #[serde(default = "default_window_height")]
    pub window_height: f32,
    #[serde(default = "default_vsync")]
    pub vsync: bool,
    #[serde(default = "default_shadow_intensity")]
    pub shadow_intensity: f32,
    #[serde(default = "default_clouds")]
    pub clouds_enabled: bool,
    /// Texture resolution per block face (64, 128, 256, or 512).
    #[serde(default = "default_texture_size")]
    pub texture_size: u32,
    /// Anisotropic filtering level (1=off, 2, 4, 8, 16).
    #[serde(default = "default_aniso")]
    pub anisotropic_filtering: u16,
    /// Anti-aliasing mode: "off", "msaa2", "msaa4", "taa"
    #[serde(default = "default_aa")]
    pub anti_aliasing: String,
    #[serde(default)]
    pub fps_120_mode: bool,
    /// Render scale factor (0.5–1.0).
    #[serde(default = "default_render_scale")]
    pub render_scale: f32,
    /// Whether the user has manually adjusted render_scale.
    #[serde(default)]
    pub render_scale_user_override: bool,
    #[serde(default)]
    pub custom_block_colors: HashMap<String, [f32; 4]>,
    #[serde(default)]
    pub block_textures: HashMap<String, String>,
    /// User-defined custom block types. Each entry occupies one atlas slot
    /// starting at index 13 (after the built-in block types 0–12). The
    /// current 4×4 atlas supports up to 3 custom blocks.
    #[serde(default)]
    pub custom_blocks: Vec<CustomBlockDef>,
}

/// Persisted definition of a user-created block type.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CustomBlockDef {
    /// Display name chosen by the user.
    pub name: String,
    /// Absolute path to the texture image file.
    pub texture_path: String,
    /// Fallback color (average of the texture, computed on creation).
    pub color: [f32; 4],
}

fn default_fov() -> f32 { 70.0 }
fn default_exposure() -> f32 { 0.0 }
fn default_tonemapping() -> String { "none".to_string() }
fn default_ssao() -> bool { false }
fn default_ssao_quality() -> String { "medium".to_string() }
fn default_smaa() -> String { "off".to_string() }
fn default_fullscreen() -> bool { true }
fn default_window_width() -> f32 { 1280.0 }
fn default_window_height() -> f32 { 720.0 }
fn default_vsync() -> bool { false }
fn default_shadow_intensity() -> f32 { 1.0 }
fn default_clouds() -> bool { true }
fn default_texture_size() -> u32 { 64 }
fn default_aniso() -> u16 { 1 }
fn default_aa() -> String { "taa".to_string() }
fn default_render_scale() -> f32 { 1.0 }
fn default_mouse_sensitivity() -> f32 { 0.0007 }

impl Default for GameSettings {
    fn default() -> Self {
        Self {
            mouse_sensitivity: 0.0007, // serde compat only; runtime value is in DevSettings
            show_grid: false,
            contrast: 1.0,
            gamma: 1.0,
            brightness: 1.0,
            render_distance: 5,
            fov: 70.0,
            exposure: 0.0,
            tonemapping: "none".to_string(),
            ssao_enabled: false,
            ssao_quality: "medium".to_string(),
            smaa_mode: "off".to_string(),
            fullscreen: true,
            window_width: 1280.0,
            window_height: 720.0,
            vsync: false,
            shadow_intensity: 1.0,
            clouds_enabled: true,
            texture_size: 64,
            anisotropic_filtering: 1,
            anti_aliasing: "taa".to_string(),
            fps_120_mode: false,
            render_scale: 1.0,
            render_scale_user_override: false,
            custom_block_colors: HashMap::new(),
            block_textures: HashMap::new(),
            custom_blocks: Vec::new(),
        }
    }
}

impl GameSettings {
    pub fn load() -> Self {
        let path = Self::settings_path();
        let mut settings = match std::fs::read_to_string(&path) {
            Ok(data) => serde_json::from_str(&data).unwrap_or_default(),
            Err(_) => Self::default(),
        };
        settings.sanitize();
        settings
    }

    /// Clamp and fix any out-of-range or invalid field values.
    pub fn sanitize(&mut self) {
        if self.render_scale.is_nan() || self.render_scale.is_infinite() {
            self.render_scale = 1.0;
        }
        self.render_scale = self.render_scale.clamp(0.5, 1.0);
    }

    pub fn save(&self) {
        let path = Self::settings_path();
        if let Ok(data) = serde_json::to_string_pretty(self) {
            let _ = std::fs::write(path, data);
        }
    }

    fn settings_path() -> std::path::PathBuf {
        dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".metalworld_settings.json")
    }
}

pub struct SettingsPlugin;

impl Plugin for SettingsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(GameSettings::load())
            .add_systems(Update, sanitize_settings.run_if(resource_changed::<GameSettings>));
    }
}

fn sanitize_settings(mut settings: ResMut<GameSettings>) {
    settings.sanitize();
}
