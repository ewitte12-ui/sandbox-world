use bevy::prelude::*;
use block_mesh::{MergeVoxel, Voxel, VoxelVisibility};
use std::collections::HashMap;
use std::sync::RwLock;

// ---------------------------------------------------------------------------
// Authoritative block capacity constant. The block texture is a 2D array
// with exactly this many layers (one per block type); the registry limit
// and every block index derive from this single value.
// ---------------------------------------------------------------------------

/// Maximum number of distinct block types (built-in + custom).
/// The block texture array has exactly this many layers.
/// A block's `index()` is its texture-array layer, so this also bounds the
/// layer count (see chunk_manager::BLOCK_LAYER_COUNT).
pub const MAX_BLOCK_TYPES: u8 = 64;

/// Number of built-in (non-custom) block types (Air through StoneBrick).
pub const BUILTIN_BLOCK_COUNT: u8 = 13;

/// First layer index available for user-created custom blocks.
pub const CUSTOM_BLOCK_START: u8 = BUILTIN_BLOCK_COUNT;

/// Maximum number of custom blocks (total capacity minus built-in).
pub const MAX_CUSTOM_BLOCKS: usize = (MAX_BLOCK_TYPES - CUSTOM_BLOCK_START) as usize;

// Compile-time safety checks — these prevent silent breakage if constants
// are changed without updating dependent values.
const _: () = assert!(
    CUSTOM_BLOCK_START < MAX_BLOCK_TYPES,
    "CUSTOM_BLOCK_START must be less than MAX_BLOCK_TYPES"
);
const _: () = assert!(
    BUILTIN_BLOCK_COUNT <= CUSTOM_BLOCK_START,
    "BUILTIN_BLOCK_COUNT must not exceed CUSTOM_BLOCK_START"
);

// ---------------------------------------------------------------------------
// BlockType — newtype over u8
//
// Using a newtype instead of an enum allows the block ID space to scale
// without adding enum variants. Built-in types are associated constants.
// Custom blocks occupy CUSTOM_BLOCK_START..MAX_BLOCK_TYPES and are defined
// at runtime via CustomBlockRegistry.
// ---------------------------------------------------------------------------

/// A block type identified by a u8 index into the texture atlas.
/// Built-in types are associated constants; custom types are dynamically
/// registered indices in the range CUSTOM_BLOCK_START..MAX_BLOCK_TYPES.
///
/// The inner field is private to prevent construction of out-of-range
/// indices. All external access goes through `index()` (read) and
/// `from_u8()` (construct with clamping).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct BlockType(u8);

// Built-in block constants (indices 0–12, matching the original enum)
impl BlockType {
    pub const AIR: Self = Self(0);
    pub const GRASS: Self = Self(1);
    pub const DIRT: Self = Self(2);
    pub const STONE: Self = Self(3);
    pub const SAND: Self = Self(4);
    pub const WOOD: Self = Self(5);
    pub const DIAMOND: Self = Self(6);
    pub const BEDROCK: Self = Self(7);
    pub const LANTERN: Self = Self(8);
    pub const BED: Self = Self(9);
    pub const PILLOW: Self = Self(10);
    pub const LEAVES: Self = Self(11);
    pub const STONE_BRICK: Self = Self(12);
}

impl Default for BlockType {
    fn default() -> Self {
        Self::AIR
    }
}

// ---------------------------------------------------------------------------
// Custom block registry
// ---------------------------------------------------------------------------

/// Runtime registry of user-created block types. Maps atlas indices
/// (CUSTOM_BLOCK_START..) to their definitions. Loaded from GameSettings
/// at startup; modified by the texture menu at runtime.
///
/// ARCHITECTURAL INVARIANT: the `blocks` field is private. All mutations
/// go through `add()` and `remove()` methods. This ensures:
///   1. Bevy's `Res::is_changed()` fires on `ResMut` access (the only
///      way to call mutation methods), triggering `on_block_registry_changed`.
///   2. Future systems that read the registry via `Res<CustomBlockRegistry>`
///      (immutable) will always see consistent data — they cannot mutate
///      it and bypass the reactive invalidation chain.
///   3. Adding a new system that caches block data without reacting to
///      changes will produce stale results that are visually obvious
///      (wrong textures), not silently corrupt.
#[derive(Resource, Default)]
pub struct CustomBlockRegistry {
    blocks: Vec<CustomBlockEntry>,
}

/// A custom block in the runtime registry.
#[derive(Clone, Debug)]
pub struct CustomBlockEntry {
    pub name: String,
    pub color: Color,
    pub atlas_index: u8,
}

impl CustomBlockRegistry {
    /// Returns the entry for a given atlas index, if it's a custom block.
    pub fn get(&self, atlas_index: u8) -> Option<&CustomBlockEntry> {
        if atlas_index < CUSTOM_BLOCK_START {
            return None;
        }
        self.blocks.get((atlas_index - CUSTOM_BLOCK_START) as usize)
    }

    /// Iterate over all registered custom blocks (read-only).
    pub fn iter(&self) -> impl Iterator<Item = &CustomBlockEntry> {
        self.blocks.iter()
    }

    /// Number of custom blocks currently registered.
    pub fn count(&self) -> usize {
        self.blocks.len()
    }

    /// True if there's room for another custom block.
    pub fn has_room(&self) -> bool {
        self.blocks.len() < MAX_CUSTOM_BLOCKS
    }

    /// The next available atlas index, or None if full.
    /// Scans for the first unused index in the custom range, so add/remove
    /// cycles correctly reuse freed slots without index collisions.
    pub fn next_index(&self) -> Option<u8> {
        if !self.has_room() {
            return None;
        }
        for idx in CUSTOM_BLOCK_START..MAX_BLOCK_TYPES {
            if !self.blocks.iter().any(|b| b.atlas_index == idx) {
                return Some(idx);
            }
        }
        None
    }

    /// Register a new custom block. Returns the assigned atlas index,
    /// or None if the registry is full. Must be called via ResMut to
    /// trigger Bevy change detection.
    pub fn add(&mut self, entry: CustomBlockEntry) -> Option<u8> {
        let idx = self.next_index()?;
        let mut entry = entry;
        entry.atlas_index = idx;
        self.blocks.push(entry);
        Some(idx)
    }

    /// Remove a custom block by atlas index. Returns the removed entry,
    /// or None if the index was invalid. Must be called via ResMut to
    /// trigger Bevy change detection.
    pub fn remove_by_index(&mut self, atlas_index: u8) -> Option<CustomBlockEntry> {
        if atlas_index < CUSTOM_BLOCK_START {
            return None;
        }
        let slot = (atlas_index - CUSTOM_BLOCK_START) as usize;
        if slot >= self.blocks.len() {
            return None;
        }
        Some(self.blocks.remove(slot))
    }
}

// ---------------------------------------------------------------------------
// Color overrides
// ---------------------------------------------------------------------------

static CUSTOM_COLORS: RwLock<Option<HashMap<String, [f32; 4]>>> = RwLock::new(None);

/// Set custom block color overrides. Can be called multiple times to update.
pub fn set_custom_colors(colors: HashMap<String, [f32; 4]>) {
    if let Ok(mut guard) = CUSTOM_COLORS.write() {
        *guard = if colors.is_empty() { None } else { Some(colors) };
    }
}

// ---------------------------------------------------------------------------
// BlockType methods
// ---------------------------------------------------------------------------

/// All built-in block types in index order (excluding Air).
pub const ALL_BUILTIN_BLOCKS: [BlockType; 12] = [
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

/// All built-in block types including Air (for atlas fill loops).
pub const ALL_BUILTIN_BLOCKS_WITH_AIR: [BlockType; 13] = [
    BlockType::AIR,
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

impl BlockType {
    /// Returns the display name for built-in block types.
    /// Custom blocks return "Custom" — use CustomBlockRegistry for their
    /// actual names.
    pub fn name(&self) -> &'static str {
        match self.0 {
            0 => "Air",
            1 => "Grass",
            2 => "Dirt",
            3 => "Stone",
            4 => "Sand",
            5 => "Wood",
            6 => "Diamond",
            7 => "Bedrock",
            8 => "Lantern",
            9 => "Bed",
            10 => "Pillow",
            11 => "Leaves",
            12 => "StoneBrick",
            _ => "Custom",
        }
    }

    /// Returns the default block color.
    pub fn default_color(&self) -> Color {
        match self.0 {
            0 => Color::NONE,
            1 => Color::linear_rgba(0.26, 0.54, 0.16, 1.0),
            2 => Color::linear_rgba(0.44, 0.30, 0.18, 1.0),
            3 => Color::linear_rgba(0.52, 0.52, 0.56, 1.0),
            4 => Color::linear_rgba(0.82, 0.76, 0.54, 1.0),
            5 => Color::linear_rgba(0.52, 0.36, 0.20, 1.0),
            6 => Color::linear_rgba(0.40, 0.82, 0.92, 1.0),
            7 => Color::linear_rgba(0.18, 0.18, 0.20, 1.0),
            8 => Color::linear_rgba(1.00, 0.85, 0.45, 1.0),
            9 => Color::linear_rgba(0.70, 0.15, 0.15, 1.0),
            10 => Color::linear_rgba(0.92, 0.92, 0.95, 1.0),
            11 => Color::linear_rgba(0.13, 0.44, 0.10, 1.0),
            12 => Color::linear_rgba(0.58, 0.56, 0.52, 1.0),
            // Custom blocks default to white; actual color comes from atlas.
            _ => Color::WHITE,
        }
    }

    /// Returns the block color, checking custom overrides first.
    pub fn color(&self) -> Color {
        if let Ok(guard) = CUSTOM_COLORS.read() {
            if let Some(ref colors) = *guard {
                if let Some(rgba) = colors.get(self.name()) {
                    return Color::linear_rgba(rgba[0], rgba[1], rgba[2], rgba[3]);
                }
            }
        }
        self.default_color()
    }

    /// Returns the atlas index for this block type. Always < MAX_BLOCK_TYPES.
    #[inline]
    pub fn index(&self) -> u8 {
        self.0
    }

    /// Returns true if this is a custom (non-built-in) block.
    pub fn is_custom(&self) -> bool {
        self.0 >= CUSTOM_BLOCK_START
    }

    /// Returns true for all solid (non-air) blocks.
    pub fn is_opaque(&self) -> bool {
        *self != BlockType::AIR
    }

    /// Returns true if this block emits light.
    pub fn is_emissive(&self) -> bool {
        *self == BlockType::LANTERN
    }

    /// Construct from a raw u8 (e.g., from save data). Values beyond
    /// MAX_BLOCK_TYPES are clamped to AIR to prevent out-of-atlas indexing.
    pub fn from_u8(v: u8) -> Self {
        if v >= MAX_BLOCK_TYPES {
            Self::AIR
        } else {
            Self(v)
        }
    }
}

impl Voxel for BlockType {
    fn get_visibility(&self) -> VoxelVisibility {
        if *self == BlockType::AIR {
            VoxelVisibility::Empty
        } else {
            VoxelVisibility::Opaque
        }
    }
}

impl MergeVoxel for BlockType {
    type MergeValue = Self;

    fn merge_value(&self) -> Self::MergeValue {
        *self
    }
}
