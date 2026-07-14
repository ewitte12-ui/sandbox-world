use std::collections::{HashMap, HashSet};

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    tasks::{futures::check_ready, AsyncComputeTaskPool, Task},
};

use crate::block_types::BlockType;
use crate::chunk::{Chunk, ChunkNeighbors, ChunkPos, CHUNK_SIZE};
use crate::terrain::natural_block_at;

/// Default render distance in chunks (5 chunks = 80 blocks radius).
pub const DEFAULT_RENDER_DISTANCE: i32 = 5;

/// Re-export atlas layout from block_types (the single source of truth).
pub use crate::block_types::ATLAS_TILES_PER_ROW;

/// Shared material handle for all chunk meshes (with grid texture).
#[derive(Resource)]
pub struct ChunkMaterial {
    pub handle: Handle<StandardMaterial>,
}

/// Resource holding the texture atlas image handle so tiles can be updated at runtime.
#[derive(Resource)]
pub struct BlockAtlas {
    pub image_handle: Handle<Image>,
    /// Tracks which block types have a loaded PNG texture (by u8 index).
    pub loaded_textures: HashSet<u8>,
    /// Current tile size in pixels.
    pub tile_size: u32,
    /// Total atlas size in pixels.
    pub atlas_size: u32,
}

/// Marker component for chunk entities in the world.
#[derive(Component)]
pub struct ChunkMarker {
    pub pos: ChunkPos,
}

/// Component holding a background chunk generation task. The task yields
/// the chunk plus a boundary mask: which outermost layers contain
/// non-terrain content (trees), in ChunkNeighbors face order. See
/// Chunk::generate_tracked.
#[derive(Component)]
struct ComputeChunk {
    pos: ChunkPos,
    task: Task<(Chunk, [bool; 6])>,
}

/// Resource managing the chunk lifecycle: loading, unloading, and block queries.
///
/// OWNERSHIP:
/// - `chunk_data`: owned exclusively by ChunkManager. Written by handle_chunk_tasks
///   (on initial load) and remesh_dirty_chunks (on modification). Read by block_at()
///   and neighbor meshing. No other system writes to chunk_data.
/// - `modifications`: the source of truth for player-placed/removed blocks.
///   Written by set_block() (called from player block_interact and save_load).
///   Modifications are applied on top of procedural terrain during chunk gen
///   and remeshing — they persist even when chunks are unloaded and reloaded.
/// - `dirty_chunks`: transient set consumed each frame by remesh_dirty_chunks.
///   Populated by set_block() when a block is modified.
#[derive(Resource)]
pub struct ChunkManager {
    /// Map from chunk position to spawned entity.
    pub chunks: HashMap<ChunkPos, Entity>,
    /// Chunk data stored for block lookups (separate from ECS for fast access).
    pub chunk_data: HashMap<ChunkPos, Chunk>,
    /// Render distance in chunks.
    pub render_distance: i32,
    /// Player-placed or removed block modifications (world coordinates).
    pub modifications: HashMap<IVec3, BlockType>,
    /// Set of chunk positions currently being generated.
    pending: HashMap<ChunkPos, Entity>,
    /// Chunks needing an IMMEDIATE remesh (player edited a block). Fully
    /// drained every frame — edits are ≤7 chunks and must show same-frame.
    dirty_chunks: HashSet<ChunkPos>,
    /// Chunks needing a DEFERRED remesh (a neighbor loaded with
    /// non-terrain boundary content, or the block registry changed).
    /// Cosmetic-latency work: processed a budgeted batch per frame so a
    /// bulk invalidation can't freeze a frame.
    deferred_remesh: HashSet<ChunkPos>,
    /// Bumped whenever `modifications` changes (set_block / clear_all).
    /// Lets per-frame consumers (lantern lights) skip work when nothing changed.
    pub mods_version: u64,
    /// Bumped whenever loaded-chunk bookkeeping is invalidated wholesale
    /// (clear_all, defensive entity-loss removal). Lets load_chunks skip
    /// its full render-bubble scan when the player hasn't moved.
    pub load_generation: u64,
}

impl Default for ChunkManager {
    fn default() -> Self {
        Self {
            chunks: HashMap::new(),
            chunk_data: HashMap::new(),
            render_distance: DEFAULT_RENDER_DISTANCE,
            modifications: HashMap::new(),
            pending: HashMap::new(),
            dirty_chunks: HashSet::new(),
            deferred_remesh: HashSet::new(),
            mods_version: 0,
            load_generation: 0,
        }
    }
}

impl ChunkManager {
    /// Clear all chunk state for a clean Gameplay re-entry.
    /// Called by cleanup_world on entering Menu.
    ///
    /// Must reset every field that could leak across a Menu↔Gameplay
    /// boundary. Missing fields here cause stale block modifications to
    /// reappear in fresh NewGame sessions, and the symptom is "no
    /// textures near spawn until I walk far" because previous-session
    /// edits get baked into the new world via apply_modifications.
    pub fn clear_all(&mut self) {
        self.chunks.clear();
        self.chunk_data.clear();
        self.pending.clear();
        self.dirty_chunks.clear();
        self.deferred_remesh.clear();
        self.modifications.clear();
        // Invalidate consumers' cached views of modifications / loaded chunks
        // so lantern lights re-scan and load_chunks re-fills the bubble.
        self.mods_version += 1;
        self.load_generation += 1;

        // Reset the greedy-meshing kill-switch so a prior session's
        // tripped invariant doesn't persist into the next Gameplay.
        #[cfg(debug_assertions)]
        crate::chunk::GREEDY_INVARIANT_VIOLATED
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Look up the block at a world position, checking modifications first,
    /// then loaded chunk data, falling back to terrain generation.
    pub fn block_at(&self, world_pos: IVec3) -> BlockType {
        // Check modifications first
        if let Some(&block) = self.modifications.get(&world_pos) {
            return block;
        }
        // Check loaded chunk data
        let cp = world_to_chunk_pos(world_pos);
        if let Some(chunk) = self.chunk_data.get(&cp) {
            let local = world_to_local(world_pos);
            return chunk.get_block(local.x, local.y, local.z);
        }
        // Fall back to procedural generation
        natural_block_at(world_pos.x, world_pos.y, world_pos.z)
    }

    /// Record a block modification and mark the containing chunk (and potentially neighbors) as dirty.
    ///
    /// WHY dirty neighbors: mesh generation uses a 1-block padding of neighbor
    /// data for face culling (see Chunk::build_mesh). If a block on a chunk
    /// boundary changes, the adjacent chunk's boundary faces may now be visible
    /// or hidden. Without dirtying the neighbor, seams/holes appear at chunk edges.
    pub fn set_block(&mut self, world_pos: IVec3, block_type: BlockType) {
        self.modifications.insert(world_pos, block_type);
        self.mods_version += 1;

        let chunk_pos = world_to_chunk_pos(world_pos);
        self.dirty_chunks.insert(chunk_pos);
        let local = world_to_local(world_pos);

        // Keep loaded chunk data in sync so remesh_dirty_chunks can rebuild
        // meshes straight from chunk_data instead of regenerating the chunk
        // from terrain noise on the main thread.
        if let Some(chunk) = self.chunk_data.get_mut(&chunk_pos) {
            chunk.blocks[Chunk::index(local.x, local.y, local.z)] = block_type;
        }

        // If the block is on a chunk boundary, also dirty the neighbor chunk
        if local.x == 0 {
            self.dirty_chunks
                .insert(ChunkPos(chunk_pos.0 - 1, chunk_pos.1, chunk_pos.2));
        }
        if local.x == CHUNK_SIZE - 1 {
            self.dirty_chunks
                .insert(ChunkPos(chunk_pos.0 + 1, chunk_pos.1, chunk_pos.2));
        }
        if local.y == 0 {
            self.dirty_chunks
                .insert(ChunkPos(chunk_pos.0, chunk_pos.1 - 1, chunk_pos.2));
        }
        if local.y == CHUNK_SIZE - 1 {
            self.dirty_chunks
                .insert(ChunkPos(chunk_pos.0, chunk_pos.1 + 1, chunk_pos.2));
        }
        if local.z == 0 {
            self.dirty_chunks
                .insert(ChunkPos(chunk_pos.0, chunk_pos.1, chunk_pos.2 - 1));
        }
        if local.z == CHUNK_SIZE - 1 {
            self.dirty_chunks
                .insert(ChunkPos(chunk_pos.0, chunk_pos.1, chunk_pos.2 + 1));
        }
    }

    /// Queue every loaded chunk for a remesh (e.g. after block color /
    /// registry changes). Goes through the DEFERRED set: with ~1k loaded
    /// chunks, remeshing them all in one frame froze the game for seconds;
    /// budgeted processing updates them progressively over ~a second.
    pub fn mark_all_dirty(&mut self) {
        for &pos in self.chunks.keys() {
            self.deferred_remesh.insert(pos);
        }
    }

}

/// Convert a world-space position to chunk coordinates.
pub fn world_to_chunk_pos(world_pos: IVec3) -> ChunkPos {
    ChunkPos(
        world_pos.x.div_euclid(CHUNK_SIZE),
        world_pos.y.div_euclid(CHUNK_SIZE),
        world_pos.z.div_euclid(CHUNK_SIZE),
    )
}

/// Convert a world-space position to local chunk coordinates (0..15).
pub fn world_to_local(world_pos: IVec3) -> IVec3 {
    IVec3::new(
        world_pos.x.rem_euclid(CHUNK_SIZE),
        world_pos.y.rem_euclid(CHUNK_SIZE),
        world_pos.z.rem_euclid(CHUNK_SIZE),
    )
}

/// Plugin that manages chunk loading, meshing, and unloading.
pub struct ChunkManagerPlugin;

impl Plugin for ChunkManagerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkManager>()
            .add_systems(Startup, setup_chunk_material)
            // ORDERING CONTRACT: chained because each stage depends on the previous:
            //   1. load_chunks           — spawns async gen tasks for missing chunks
            //   2. handle_chunk_tasks    — polls completed tasks, inserts chunk data + mesh
            //   3. remesh_dirty_chunks   — rebuilds meshes for modified chunks
            //   4. unload_chunks         — despawns chunks beyond render distance
            //   5. apply_texture_size_change — rebuilds atlas if texture size changed
            //   6. apply_aniso_filter_change — updates sampler if filter setting changed
            // Steps 5-6 must run after meshing so new meshes get the correct material.
            .add_systems(
                Update,
                (
                    on_block_registry_changed,
                    load_chunks,
                    handle_chunk_tasks,
                    remesh_dirty_chunks,
                    unload_chunks,
                    apply_texture_size_change,
                    apply_aniso_filter_change,
                )
                    .chain()
                    .run_if(in_state(crate::GameState::Gameplay)),
            );
    }
}

/// Reactive system: when the block registry changes (block added/removed),
/// mark all loaded chunks dirty so they remesh with updated atlas data.
/// Uses Bevy's built-in change detection on the Resource — no custom events,
/// no polling. Runs before load_chunks in the chain so dirty flags are set
/// before remesh_dirty_chunks consumes them.
fn on_block_registry_changed(
    registry: Res<crate::block_types::CustomBlockRegistry>,
    mut manager: ResMut<ChunkManager>,
) {
    if registry.is_changed() && !registry.is_added() {
        manager.mark_all_dirty();
        info!("Block registry changed — marked all chunks for remesh");
    }
}

/// Fill a single tile in the atlas data with a solid color + grid border.
pub fn fill_atlas_tile(data: &mut [u8], block_idx: u8, color: Color, tile_size: u32, atlas_width: u32) {
    if block_idx >= crate::block_types::MAX_BLOCK_TYPES {
        return;
    }
    let tiles_per_row = ATLAS_TILES_PER_ROW;

    let tile_x = (block_idx as u32) % tiles_per_row;
    let tile_y = (block_idx as u32) / tiles_per_row;
    let base_px = tile_x * tile_size;
    let base_py = tile_y * tile_size;

    let LinearRgba { red, green, blue, alpha } = color.to_linear();
    // Convert linear to sRGB for Rgba8UnormSrgb storage
    let r = (linear_to_srgb(red) * 255.0) as u8;
    let g = (linear_to_srgb(green) * 255.0) as u8;
    let b = (linear_to_srgb(blue) * 255.0) as u8;
    let a = (alpha * 255.0) as u8;

    let border = 3u32;

    for py in 0..tile_size {
        for px in 0..tile_size {
            let is_border = px < border || px >= tile_size - border
                || py < border || py >= tile_size - border;
            let ix = base_px + px;
            let iy = base_py + py;
            let i = ((iy * atlas_width + ix) * 4) as usize;
            if is_border {
                // Subtle darkened border
                data[i] = ((r as f32) * 0.65) as u8;
                data[i + 1] = ((g as f32) * 0.65) as u8;
                data[i + 2] = ((b as f32) * 0.65) as u8;
                data[i + 3] = a;
            } else {
                data[i] = r;
                data[i + 1] = g;
                data[i + 2] = b;
                data[i + 3] = a;
            }
        }
    }
}

/// Convert a linear color component to sRGB.
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Build the texture atlas and shared material for all chunk meshes.
fn setup_chunk_material(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    game_settings: Res<crate::settings::GameSettings>,
) {
    use bevy::image::ImageSampler;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

    let tile_size = game_settings.texture_size.clamp(64, 512);
    let atlas_size = ATLAS_TILES_PER_ROW * tile_size;
    let mut data = vec![255u8; (atlas_size * atlas_size * 4) as usize];

    let all_blocks = [
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

    for &block in &all_blocks {
        fill_atlas_tile(&mut data, block.index(), block.color(), tile_size, atlas_size);
    }
    for idx in crate::block_types::CUSTOM_BLOCK_START..crate::block_types::MAX_BLOCK_TYPES {
        fill_atlas_tile(&mut data, idx, Color::WHITE, tile_size, atlas_size);
    }

    // Load block textures from two sources (settings path takes priority):
    // 1. game_settings.block_textures — user-selected via the in-game file
    //    dialog. This is the primary, user-facing workflow.
    // 2. textures/ directory — developer convenience for bundled/default
    //    textures. NOT a user-facing workflow; users should never need to
    //    copy files here manually.
    let mut loaded_textures = HashSet::new();
    for &block in &all_blocks {
        if block == BlockType::AIR {
            continue;
        }
        let name = block.name();
        let name_lower = name.to_lowercase();

        // Check settings first (user-selected path from file dialog)
        let settings_path = game_settings.block_textures.get(name);

        // Try settings path, then fall back to textures/ directory
        let try_paths: Vec<String> = if let Some(sp) = settings_path {
            vec![sp.clone(), format!("textures/{}.png", name_lower)]
        } else {
            vec![format!("textures/{}.png", name_lower)]
        };

        for tex_path in &try_paths {
            if let Ok(img_data) = std::fs::read(tex_path) {
                if let Ok(dyn_img) = image::load_from_memory(&img_data) {
                    let resized = dyn_img.resize_exact(
                        tile_size,
                        tile_size,
                        image::imageops::FilterType::Lanczos3,
                    );
                    let rgba = resized.to_rgba8();
                    copy_image_to_atlas_tile(&mut data, block.index(), &rgba, tile_size, atlas_size);
                    loaded_textures.insert(block.index());
                    info!("Loaded texture: {} ({}x{})", tex_path, tile_size, tile_size);
                    break; // first successful source wins
                }
            }
        }
    }

    // Load textures for custom blocks from their saved paths
    for def in &game_settings.custom_blocks {
        let idx = crate::block_types::CUSTOM_BLOCK_START + game_settings.custom_blocks.iter().position(|d| d.name == def.name).unwrap_or(0) as u8;
        if let Ok(img_data) = std::fs::read(&def.texture_path) {
            if let Ok(dyn_img) = image::load_from_memory(&img_data) {
                let resized = dyn_img.resize_exact(
                    tile_size,
                    tile_size,
                    image::imageops::FilterType::Lanczos3,
                );
                let rgba = resized.to_rgba8();
                copy_image_to_atlas_tile(&mut data, idx, &rgba, tile_size, atlas_size);
                loaded_textures.insert(idx);
                info!("Loaded custom block texture: {} ({}x{})", def.name, tile_size, tile_size);
            }
        }
    }

    let (atlas_chain, mip_count) = generate_atlas_mip_chain(data, atlas_size);

    // Construct via new_uninit + manual data assignment: Image::new's
    // `debug_assert_eq!(size.volume() * pixel_size, data.len())` would
    // panic in debug builds because our chain is ~4/3× the mip0 byte
    // length once mipmaps are appended.
    let mut image = Image::new_uninit(
        Extent3d {
            width: atlas_size,
            height: atlas_size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.mip_level_count = mip_count;
    image.data = Some(atlas_chain);

    // Nearest mag + Linear min/mipmap: crisp voxel pixels up close, smooth
    // far-distance sampling that eliminates the sub-pixel texel-flip
    // shimmer Nearest min-filter produces. Anisotropic filtering needs
    // mipmaps to do anything, which is why we now generate them above.
    let aniso = game_settings.anisotropic_filtering.max(1);
    let sampler_desc = make_atlas_sampler_desc(aniso);
    image.sampler = ImageSampler::Descriptor(sampler_desc);

    let image_handle = images.add(image);

    // Build an emissive atlas: black everywhere except the lantern tile (index 8),
    // which gets a bright warm glow. This makes lanterns visibly bright without
    // affecting other blocks or requiring a separate material.
    let mut emissive_data = vec![0u8; (atlas_size * atlas_size * 4) as usize];
    fill_atlas_tile(
        &mut emissive_data,
        BlockType::LANTERN.index(),
        Color::linear_rgba(1.0, 0.85, 0.45, 1.0),
        tile_size,
        atlas_size,
    );
    let emissive_image = Image::new(
        Extent3d {
            width: atlas_size,
            height: atlas_size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        emissive_data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    let emissive_handle = images.add(emissive_image);

    // DEBUG: uncomment the block below to replace the atlas material with
    // a solid magenta unlit material. If chunks appear magenta, geometry is
    // correct and the bug is in textures/materials. If chunks are still
    // invisible, the bug is in mesh generation or visibility.
    // TODO: remove after diagnosis.
    #[cfg(debug_assertions)]
    let debug_override_material = false; // flip to true to activate

    #[cfg(debug_assertions)]
    let material = if debug_override_material {
        materials.add(StandardMaterial {
            base_color: Color::linear_rgb(1.0, 0.0, 1.0),
            unlit: true,
            ..default()
        })
    } else {
        materials.add(StandardMaterial {
            base_color_texture: Some(image_handle.clone()),
            base_color: Color::WHITE,
            emissive: bevy::color::LinearRgba::new(6.0, 5.1, 2.7, 1.0),
            emissive_texture: Some(emissive_handle),
            reflectance: 0.0,
            ..default()
        })
    };
    #[cfg(not(debug_assertions))]
    let material = materials.add(StandardMaterial {
        base_color_texture: Some(image_handle.clone()),
        base_color: Color::WHITE,
        emissive: bevy::color::LinearRgba::new(6.0, 5.1, 2.7, 1.0),
        emissive_texture: Some(emissive_handle),
        reflectance: 0.0,
        ..default()
    });

    // Debug: validate all asset handles are valid and bound.
    #[cfg(debug_assertions)]
    {
        let mat_valid = materials.get(&material).is_some();
        let atlas_valid = images.get(&image_handle).is_some();
        let atlas_size_px = images.get(&image_handle).map(|img| {
            let e = &img.texture_descriptor.size;
            (e.width, e.height)
        });

        bevy::log::info!(
            "Asset validation: material={} atlas_image={} atlas_size={:?} \
             tile_size={} loaded_textures={}/{}",
            if mat_valid { "OK" } else { "MISSING" },
            if atlas_valid { "OK" } else { "MISSING" },
            atlas_size_px.unwrap_or((0, 0)),
            tile_size,
            loaded_textures.len(),
            crate::block_types::BUILTIN_BLOCK_COUNT,
        );

        debug_assert!(mat_valid, "Chunk material handle is invalid after creation");
        debug_assert!(atlas_valid, "Atlas image handle is invalid after creation");

        // Verify the material's texture handles resolve.
        if let Some(mat) = materials.get(&material) {
            let base_ok = mat.base_color_texture.as_ref()
                .map(|h| images.get(h).is_some()).unwrap_or(false);
            let emissive_ok = mat.emissive_texture.as_ref()
                .map(|h| images.get(h).is_some()).unwrap_or(true);
            bevy::log::info!(
                "Material textures: base_color={} emissive={}",
                if base_ok { "OK" } else { "MISSING" },
                if emissive_ok { "OK" } else { "MISSING" },
            );
            debug_assert!(base_ok, "base_color_texture handle does not resolve to an Image");
        }
    }

    commands.insert_resource(ChunkMaterial { handle: material });
    commands.insert_resource(BlockAtlas {
        image_handle,
        loaded_textures,
        tile_size,
        atlas_size,
    });
}

/// Copy raw RGBA8 pixel data into a tile slot of the atlas.
pub fn copy_image_to_atlas_tile(atlas_data: &mut [u8], block_idx: u8, rgba: &image::RgbaImage, tile_size: u32, atlas_width: u32) {
    if block_idx >= crate::block_types::MAX_BLOCK_TYPES {
        return;
    }
    let tiles_per_row = ATLAS_TILES_PER_ROW;

    let tile_x = (block_idx as u32) % tiles_per_row;
    let tile_y = (block_idx as u32) / tiles_per_row;
    let base_px = tile_x * tile_size;
    let base_py = tile_y * tile_size;

    let border = 3u32;

    for py in 0..tile_size {
        for px in 0..tile_size {
            let ix = base_px + px;
            let iy = base_py + py;
            let i = ((iy * atlas_width + ix) * 4) as usize;

            let is_border = px < border || px >= tile_size - border
                || py < border || py >= tile_size - border;

            if is_border {
                // Darken texture pixels at border
                let src = rgba.get_pixel(px, py);
                atlas_data[i] = ((src[0] as f32) * 0.65) as u8;
                atlas_data[i + 1] = ((src[1] as f32) * 0.65) as u8;
                atlas_data[i + 2] = ((src[2] as f32) * 0.65) as u8;
                atlas_data[i + 3] = src[3];
            } else {
                let src_pixel = rgba.get_pixel(px, py);
                atlas_data[i] = src_pixel[0];
                atlas_data[i + 1] = src_pixel[1];
                atlas_data[i + 2] = src_pixel[2];
                atlas_data[i + 3] = src_pixel[3];
            }
        }
    }
}

/// Generate a complete mip chain for the atlas image.
///
/// Each mip level is built **per tile**: every mip-N pixel inside a tile
/// is the average of a `ratio × ratio` block of pixels from that same
/// tile in mip 0 (where `ratio = base_tile_size / cur_tile_size`). This
/// matters because the alternative — box-filtering each mip from the
/// previous mip — averages across tile boundaries at every level and
/// rapidly contaminates each tile's color with its neighbors. Per-tile
/// downsampling keeps every tile's pyramid pure; the only neighbor
/// bleed at runtime comes from the Linear sampler's 2×2 blend at the
/// outermost texel, which the chunk.rs UV inset covers.
///
/// Chain stops when the atlas would drop below `min_atlas_size` (32 px)
/// or `base_tile_size` is no longer divisible by the next ratio. At
/// `min_atlas_size = 32`, each tile is 4×4 px — enough to keep distinct
/// per-block color at the horizon while giving the GPU 4-5 mip levels
/// of LOD headroom (vs the 1-2 levels the previous cap allowed).
///
/// Averaging is in linear color space (the atlas is `Rgba8UnormSrgb`)
/// to avoid the gamma-darkening artifact of naive sRGB averaging.
fn generate_atlas_mip_chain(base_data: Vec<u8>, base_size: u32) -> (Vec<u8>, u32) {
    let min_atlas_size: u32 = 32;
    let tiles_per_row = ATLAS_TILES_PER_ROW;
    let base_tile_size = base_size / tiles_per_row;
    if base_tile_size * tiles_per_row != base_size {
        // Atlas dimensions don't tile cleanly — refuse to build mips
        // rather than risk subtly misaligned sampling.
        return (base_data, 1);
    }

    let mut chain = base_data;
    let mut mip_count: u32 = 1;

    // sRGB-to-linear LUT for byte values 0..=255.
    let mut srgb_to_lin = [0.0f32; 256];
    for (i, slot) in srgb_to_lin.iter_mut().enumerate() {
        let c = i as f32 / 255.0;
        *slot = if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
    }

    let mut ratio: u32 = 2;
    loop {
        if base_tile_size % ratio != 0 {
            break;
        }
        let cur_tile_size = base_tile_size / ratio;
        let cur_atlas_size = cur_tile_size * tiles_per_row;
        if cur_atlas_size < min_atlas_size {
            break;
        }

        let mip_byte_len = (cur_atlas_size * cur_atlas_size * 4) as usize;
        let mut mip_data = vec![0u8; mip_byte_len];
        let samples_per_pixel = (ratio * ratio) as f32;
        let inv_samples = 1.0 / samples_per_pixel;

        for tile_y in 0..tiles_per_row {
            for tile_x in 0..tiles_per_row {
                let src_tx = tile_x * base_tile_size;
                let src_ty = tile_y * base_tile_size;
                let dst_tx = tile_x * cur_tile_size;
                let dst_ty = tile_y * cur_tile_size;

                for py in 0..cur_tile_size {
                    for px in 0..cur_tile_size {
                        let dst_x = dst_tx + px;
                        let dst_y = dst_ty + py;
                        let dst_idx = ((dst_y * cur_atlas_size + dst_x) * 4) as usize;

                        let mut sum_rgb = [0.0f32; 3];
                        let mut sum_a: u32 = 0;
                        for sy in 0..ratio {
                            for sx in 0..ratio {
                                let src_x = src_tx + px * ratio + sx;
                                let src_y = src_ty + py * ratio + sy;
                                let si = ((src_y * base_size + src_x) * 4) as usize;
                                sum_rgb[0] += srgb_to_lin[chain[si] as usize];
                                sum_rgb[1] += srgb_to_lin[chain[si + 1] as usize];
                                sum_rgb[2] += srgb_to_lin[chain[si + 2] as usize];
                                sum_a += chain[si + 3] as u32;
                            }
                        }
                        for c in 0..3 {
                            let lin = sum_rgb[c] * inv_samples;
                            mip_data[dst_idx + c] = (linear_to_srgb(lin) * 255.0).round().clamp(0.0, 255.0) as u8;
                        }
                        mip_data[dst_idx + 3] = (sum_a as f32 * inv_samples).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }

        chain.extend_from_slice(&mip_data);
        mip_count += 1;

        // Next mip down — only if it's still cleanly divisible.
        let Some(next_ratio) = ratio.checked_mul(2) else { break };
        ratio = next_ratio;
    }

    (chain, mip_count)
}

/// Build the sampler descriptor used for the block atlas. Mag = Nearest
/// preserves the crisp pixel look at close range; Min + Mipmap = Linear
/// kills distance shimmer (sub-pixel texel-flip aliasing) and lets
/// anisotropic filtering do something useful — aniso requires mipmaps.
fn make_atlas_sampler_desc(aniso: u16) -> bevy::image::ImageSamplerDescriptor {
    use bevy::image::{ImageFilterMode, ImageSamplerDescriptor};
    let mut desc = ImageSamplerDescriptor {
        mag_filter: ImageFilterMode::Nearest,
        min_filter: ImageFilterMode::Linear,
        mipmap_filter: ImageFilterMode::Linear,
        ..default()
    };
    if aniso > 1 {
        desc.set_anisotropic_filter(aniso);
    }
    desc
}

/// System: check player position and spawn async generation tasks for chunks in range.
fn load_chunks(
    mut commands: Commands,
    mut manager: ResMut<ChunkManager>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    game_settings: Option<Res<crate::settings::GameSettings>>,
    world_id: Res<crate::WorldInstanceId>,
    mut last_scan: Local<Option<(ChunkPos, i32, u64)>>,
) {
    // Use the camera position as the player position
    let camera_pos = match cameras.iter().next() {
        Some(gt) => gt.translation(),
        None => return,
    };

    let player_chunk = ChunkPos(
        (camera_pos.x as i32).div_euclid(CHUNK_SIZE),
        (camera_pos.y as i32).div_euclid(CHUNK_SIZE),
        (camera_pos.z as i32).div_euclid(CHUNK_SIZE),
    );

    // Use settings render_distance if available, otherwise use stored value
    if let Some(ref gs) = game_settings {
        if manager.render_distance != gs.render_distance {
            manager.render_distance = gs.render_distance;
        }
    }
    let rd = manager.render_distance;

    // Missing chunks can only appear when the player chunk changes, the
    // render distance changes, or the loaded-chunk bookkeeping is invalidated
    // (load_generation bump). Skip the full render-bubble scan otherwise —
    // it's ~1-2k HashMap probes per frame for nothing.
    let scan_key = (player_chunk, rd, manager.load_generation);
    if *last_scan == Some(scan_key) {
        return;
    }

    let thread_pool = AsyncComputeTaskPool::get();

    // Vertical loading range is asymmetric: only 2 chunks below the player
    // (32 blocks — enough to see nearby caves) vs full render_distance above.
    // Underground chunks far below are invisible and waste generation time.
    // Floor at Y=-7 because bedrock is at depth 100 (~Y=-90 for surface at
    // Y=10), and chunk Y=-7 starts at block Y=-112 — well below bedrock.
    let y_min = (player_chunk.1 - 2).max(-7);
    let y_max = player_chunk.1 + rd;

    // Collect candidate chunk positions, then sort by chebyshev distance
    // from the player chunk so chunks nearest the camera are enqueued
    // FIRST on AsyncComputeTaskPool. Iteration order matters: the pool
    // grabs tasks in submission order, so a flat row-major sweep would
    // start with the far corner of the render bubble and leave the
    // central spawn ground for last. The "hole in the ground at spawn"
    // symptom is partly that ordering.
    let mut candidates: Vec<ChunkPos> = Vec::new();
    for cy in y_min..=y_max {
        for cz in (player_chunk.2 - rd)..=(player_chunk.2 + rd) {
            for cx in (player_chunk.0 - rd)..=(player_chunk.0 + rd) {
                let pos = ChunkPos(cx, cy, cz);
                if manager.chunks.contains_key(&pos) || manager.pending.contains_key(&pos) {
                    continue;
                }
                candidates.push(pos);
            }
        }
    }
    candidates.sort_by_key(|p| {
        let dx = (p.0 - player_chunk.0).abs();
        let dy = (p.1 - player_chunk.1).abs();
        let dz = (p.2 - player_chunk.2).abs();
        dx.max(dy).max(dz)
    });

    for pos in candidates {
        // Task entities carry the world markers from birth so world teardown
        // despawns in-flight generation tasks too. Without these markers,
        // tasks spawned before a Play/Load reload survive into the next
        // session, complete alongside the new session's duplicate tasks for
        // the same positions, and leave orphaned ghost chunk meshes.
        let entity = commands
            .spawn((crate::WorldEntity, crate::WorldScoped(world_id.0)))
            .id();
        let task = thread_pool.spawn(async move { Chunk::generate_tracked(pos) });

        commands.entity(entity).insert(ComputeChunk { pos, task });
        manager.pending.insert(pos, entity);
        #[cfg(debug_assertions)]
        bevy::log::trace!(
            "Chunk SPAWN: ({},{},{}) entity={:?} (pending)",
            pos.0, pos.1, pos.2, entity,
        );
    }

    *last_scan = Some(scan_key);
}

/// Apply player modifications to a freshly generated chunk. Used by both
/// handle_chunk_tasks (initial load) and remesh_dirty_chunks (runtime) to
/// ensure both paths produce identical block data.
/// Returns a mask of boundary layers a modification was written into
/// (ChunkNeighbors face order) — used to decide which already-meshed
/// neighbors must remesh because they meshed against terrain-predicted
/// padding for this chunk.
fn apply_modifications(
    chunk: &mut Chunk,
    modifications: &HashMap<IVec3, BlockType>,
) -> [bool; 6] {
    let base_x = chunk.pos.0 * CHUNK_SIZE;
    let base_y = chunk.pos.1 * CHUNK_SIZE;
    let base_z = chunk.pos.2 * CHUNK_SIZE;
    let mut boundary_mask = [false; 6];

    let mut write = |chunk: &mut Chunk, lx: i32, ly: i32, lz: i32, bt: BlockType| {
        chunk.blocks[Chunk::index(lx, ly, lz)] = bt;
        if lx == 0 { boundary_mask[0] = true; }
        if lx == CHUNK_SIZE - 1 { boundary_mask[1] = true; }
        if ly == 0 { boundary_mask[2] = true; }
        if ly == CHUNK_SIZE - 1 { boundary_mask[3] = true; }
        if lz == 0 { boundary_mask[4] = true; }
        if lz == CHUNK_SIZE - 1 { boundary_mask[5] = true; }
    };

    // Iterate whichever side is smaller: probing the map 4096 times per
    // chunk dominates initial world load when only a handful of player
    // edits exist. Both paths produce identical results.
    if modifications.len() < CHUNK_VOLUME_LOOKUPS {
        for (&wp, &bt) in modifications {
            let lx = wp.x - base_x;
            let ly = wp.y - base_y;
            let lz = wp.z - base_z;
            if (0..CHUNK_SIZE).contains(&lx)
                && (0..CHUNK_SIZE).contains(&ly)
                && (0..CHUNK_SIZE).contains(&lz)
            {
                write(chunk, lx, ly, lz, bt);
            }
        }
    } else {
        for lx in 0..CHUNK_SIZE {
            for ly in 0..CHUNK_SIZE {
                for lz in 0..CHUNK_SIZE {
                    let wp = IVec3::new(base_x + lx, base_y + ly, base_z + lz);
                    if let Some(&bt) = modifications.get(&wp) {
                        write(chunk, lx, ly, lz, bt);
                    }
                }
            }
        }
    }

    boundary_mask
}

/// Crossover point for apply_modifications: below this many total
/// modifications, iterating the modification map beats probing it once
/// per block position (16³ = 4096 probes).
const CHUNK_VOLUME_LOOKUPS: usize = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;

/// System: poll completed chunk generation tasks, build meshes, and insert into the world.
fn handle_chunk_tasks(
    mut commands: Commands,
    mut manager: ResMut<ChunkManager>,
    mut tasks: Query<(Entity, &mut ComputeChunk)>,
    mut meshes: ResMut<Assets<Mesh>>,
    chunk_material: Option<Res<ChunkMaterial>>,
    mut opt_flags: ResMut<crate::dev_tools::OptimizationFlags>,
    dev: Res<crate::dev_tools::DevSettings>,
    world_id: Res<crate::WorldInstanceId>,
) {
    // Debug: auto-disable greedy meshing if an invariant was violated.
    #[cfg(debug_assertions)]
    if crate::chunk::GREEDY_INVARIANT_VIOLATED.load(std::sync::atomic::Ordering::Relaxed) {
        if opt_flags.enable_greedy_meshing {
            bevy::log::warn!("Greedy meshing kill-switch triggered — falling back to naive meshing");
            opt_flags.enable_greedy_meshing = false;
        }
    }

    let Some(chunk_mat) = chunk_material else {
        return;
    };

    // Debug: verify all pending entries reference valid entities.
    // Stale entries indicate a despawn path that didn't clean up pending.
    #[cfg(debug_assertions)]
    {
        let stale: Vec<_> = manager.pending.iter()
            .filter(|(_, &ent)| commands.get_entity(ent).is_err())
            .map(|(&pos, _)| pos)
            .collect();
        for pos in &stale {
            bevy::log::warn!(
                "Stale pending entry for chunk ({},{},{}) — entity was despawned",
                pos.0, pos.1, pos.2,
            );
            manager.pending.remove(pos);
        }
    }

    // Cap how many completed chunks are meshed per frame. Mesh building runs
    // on the main thread; with no cap, a burst of completed generation tasks
    // (initial load, fast movement) meshes dozens of chunks in one frame and
    // produces a visible hitch. Remaining ready tasks are picked up on
    // subsequent frames.
    let mesh_budget = dev.max_chunk_meshes_per_frame.max(1) as usize;
    let mut meshed_this_frame = 0usize;

    for (entity, mut compute) in &mut tasks {
        if meshed_this_frame >= mesh_budget {
            break;
        }
        if let Some((mut chunk, tree_mask)) = check_ready(&mut compute.task) {
            let pos = compute.pos;
            meshed_this_frame += 1;

            let mods_mask = apply_modifications(&mut chunk, &manager.modifications);
            // Boundary layers that diverge from raw terrain (trees or player
            // edits). Neighbors already meshed used terrain-predicted padding
            // for this chunk — only they, and only on divergent faces, need
            // a (deferred) remesh.
            let boundary_mask: [bool; 6] = std::array::from_fn(|i| tree_mask[i] || mods_mask[i]);

            // Debug: log block composition for chunks near origin to detect
            // corrupted save data overriding valid terrain generation.
            #[cfg(debug_assertions)]
            if pos.0.abs() <= 1 && pos.1.abs() <= 1 && pos.2.abs() <= 1 {
                let air = chunk.blocks.iter().filter(|b| **b == BlockType::AIR).count();
                let solid = chunk.blocks.len() - air;
                let mod_count = manager.modifications.iter()
                    .filter(|(p, _)| {
                        let cx = p.x.div_euclid(CHUNK_SIZE);
                        let cy = p.y.div_euclid(CHUNK_SIZE);
                        let cz = p.z.div_euclid(CHUNK_SIZE);
                        cx == pos.0 && cy == pos.1 && cz == pos.2
                    })
                    .count();
                bevy::log::info!(
                    "Chunk ({},{},{}) built: {}/{} solid, {} mods applied",
                    pos.0, pos.1, pos.2, solid, chunk.blocks.len(), mod_count,
                );
                if solid == 0 {
                    bevy::log::warn!(
                        "Chunk ({},{},{}) near origin is 100% AIR after modifications — \
                         save data may have cleared all blocks",
                        pos.0, pos.1, pos.2,
                    );
                }
            }

            let loaded_neighbors = ChunkNeighbors {
                neighbors: [
                    manager.chunk_data.get(&ChunkPos(pos.0 - 1, pos.1, pos.2)),
                    manager.chunk_data.get(&ChunkPos(pos.0 + 1, pos.1, pos.2)),
                    manager.chunk_data.get(&ChunkPos(pos.0, pos.1 - 1, pos.2)),
                    manager.chunk_data.get(&ChunkPos(pos.0, pos.1 + 1, pos.2)),
                    manager.chunk_data.get(&ChunkPos(pos.0, pos.1, pos.2 - 1)),
                    manager.chunk_data.get(&ChunkPos(pos.0, pos.1, pos.2 + 1)),
                ],
            };
            let mesh = chunk.build_mesh(
                &loaded_neighbors,
                opt_flags.enable_greedy_meshing,
                dev.highlight_greedy_quads,
                dev.ao_strength,
            );

            // NOTE: a solid chunk with 0 vertices is NORMAL here — fully
            // buried chunks cull all faces against the terrain-predicted
            // padding shell. build_mesh's debug invariant handles the real
            // error case (exposed solid faces producing no geometry).

            let world_x = (pos.0 * CHUNK_SIZE) as f32;
            let world_y = (pos.1 * CHUNK_SIZE) as f32;
            let world_z = (pos.2 * CHUNK_SIZE) as f32;

            // Verify the entity still exists before issuing commands.
            // The entity may have been despawned by unload_chunks or
            // cleanup_world if the chunk went out of range or the state
            // transitioned while the async task was in flight.
            let Ok(mut ec) = commands.get_entity(entity) else {
                manager.pending.remove(&pos);
                continue;
            };

            let mesh_handle = meshes.add(mesh);
            let material_handle = chunk_mat.handle.clone();

            #[cfg(debug_assertions)]
            let _vert_count = meshes.get(&mesh_handle).map(|m| m.count_vertices()).unwrap_or(0);

            // Chunk entities must always use Bevy's built-in check_visibility
            // for frustum culling — it is already optimal. Do NOT insert
            // NoFrustumCulling; it disables culling entirely and is a
            // performance regression. Bevy computes Aabb from the mesh
            // automatically and caches it per entity.
            ec.remove::<ComputeChunk>().insert((
                crate::WorldEntity,
                crate::WorldScoped(world_id.0),
                ChunkMarker { pos },
                Mesh3d(mesh_handle),
                MeshMaterial3d(material_handle),
                Transform::from_xyz(world_x, world_y, world_z),
            ));

            manager.chunk_data.insert(pos, chunk);
            manager.chunks.insert(pos, entity);
            manager.pending.remove(&pos);

            // Queue deferred remeshes for already-meshed neighbors on faces
            // where this chunk diverges from raw terrain. They meshed
            // against terrain-predicted padding (build_mesh shell fill) and
            // are correct everywhere else, so untouched-terrain boundaries
            // (the vast majority) trigger nothing.
            const FACE_OFFSETS: [(i32, i32, i32); 6] = [
                (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
            ];
            for (face, &(dx, dy, dz)) in FACE_OFFSETS.iter().enumerate() {
                if !boundary_mask[face] {
                    continue;
                }
                let npos = ChunkPos(pos.0 + dx, pos.1 + dy, pos.2 + dz);
                if manager.chunks.contains_key(&npos) {
                    manager.deferred_remesh.insert(npos);
                }
            }

            #[cfg(debug_assertions)]
            bevy::log::trace!(
                "Chunk MESH: ({},{},{}) entity={:?} verts={}",
                pos.0, pos.1, pos.2, entity, _vert_count,
            );
        }
    }
}

/// System: remesh chunks that have been dirtied by block modifications.
/// Uses the same mesh generation logic as handle_chunk_tasks (initial load)
/// — same build_mesh(), same material, same UV computation. The only
/// difference is that the mesh asset is updated in-place rather than
/// creating a new handle.
fn remesh_dirty_chunks(
    mut commands: Commands,
    mut manager: ResMut<ChunkManager>,
    mut meshes: ResMut<Assets<Mesh>>,
    mesh_query: Query<(Entity, &Mesh3d, Option<&MeshMaterial3d<StandardMaterial>>), With<ChunkMarker>>,
    chunk_material: Option<Res<ChunkMaterial>>,
    opt_flags: Res<crate::dev_tools::OptimizationFlags>,
    dev: Res<crate::dev_tools::DevSettings>,
) {
    if manager.dirty_chunks.is_empty() && manager.deferred_remesh.is_empty() {
        return;
    }

    // Immediate set (player edits): fully drained — must show same-frame,
    // and edits touch ≤7 chunks. Deferred set (neighbor loads, registry
    // changes): budgeted, so bulk invalidations spread across frames
    // instead of freezing one.
    let mut work: Vec<ChunkPos> = manager.dirty_chunks.drain().collect();
    // An edit-dirtied chunk supersedes any deferred entry for the same pos.
    for p in &work {
        manager.deferred_remesh.remove(p);
    }
    let budget = dev.max_chunk_meshes_per_frame.max(1) as usize;
    if work.len() < budget && !manager.deferred_remesh.is_empty() {
        let take = budget - work.len();
        let batch: Vec<ChunkPos> = manager
            .deferred_remesh
            .iter()
            .copied()
            .filter(|p| !work.contains(p))
            .take(take)
            .collect();
        for p in &batch {
            manager.deferred_remesh.remove(p);
        }
        work.extend(batch);
    }

    for pos in work {
        // Only remesh chunks that are actually loaded
        let Some(&entity) = manager.chunks.get(&pos) else {
            continue;
        };

        // Verify the entity still exists before doing expensive mesh work.
        // The entity may have been despawned by unload_chunks or cleanup_world.
        if mesh_query.get(entity).is_err() {
            manager.chunks.remove(&pos);
            manager.chunk_data.remove(&pos);
            // A hole opened inside the render bubble — let load_chunks
            // rescan and refill it even if the player hasn't moved.
            manager.load_generation += 1;
            continue;
        }

        // chunk_data is the source of truth here: set_block writes modified
        // blocks through to it, and handle_chunk_tasks stores fully modified
        // chunks at load. Rebuilding the mesh from stored data avoids the
        // old path's full terrain regeneration (noise + trees + 4096-probe
        // modification pass) on the main thread for every dirtied chunk.
        let mesh = if let Some(chunk) = manager.chunk_data.get(&pos) {
            let neighbors = ChunkNeighbors {
                neighbors: [
                    manager.chunk_data.get(&ChunkPos(pos.0 - 1, pos.1, pos.2)),
                    manager.chunk_data.get(&ChunkPos(pos.0 + 1, pos.1, pos.2)),
                    manager.chunk_data.get(&ChunkPos(pos.0, pos.1 - 1, pos.2)),
                    manager.chunk_data.get(&ChunkPos(pos.0, pos.1 + 1, pos.2)),
                    manager.chunk_data.get(&ChunkPos(pos.0, pos.1, pos.2 - 1)),
                    manager.chunk_data.get(&ChunkPos(pos.0, pos.1, pos.2 + 1)),
                ],
            };
            chunk.build_mesh(
                &neighbors,
                opt_flags.enable_greedy_meshing,
                dev.highlight_greedy_quads,
                dev.ao_strength,
            )
        } else {
            // Defensive fallback: tracked chunk without stored data (should
            // not happen — chunks and chunk_data are inserted together).
            // Regenerate exactly like the initial-load path.
            let mut chunk = Chunk::generate(pos);
            let _ = apply_modifications(&mut chunk, &manager.modifications);
            let mesh = {
                let neighbors = ChunkNeighbors {
                    neighbors: [
                        manager.chunk_data.get(&ChunkPos(pos.0 - 1, pos.1, pos.2)),
                        manager.chunk_data.get(&ChunkPos(pos.0 + 1, pos.1, pos.2)),
                        manager.chunk_data.get(&ChunkPos(pos.0, pos.1 - 1, pos.2)),
                        manager.chunk_data.get(&ChunkPos(pos.0, pos.1 + 1, pos.2)),
                        manager.chunk_data.get(&ChunkPos(pos.0, pos.1, pos.2 - 1)),
                        manager.chunk_data.get(&ChunkPos(pos.0, pos.1, pos.2 + 1)),
                    ],
                };
                chunk.build_mesh(
                    &neighbors,
                    opt_flags.enable_greedy_meshing,
                    dev.highlight_greedy_quads,
                    dev.ao_strength,
                )
            };
            manager.chunk_data.insert(pos, chunk);
            mesh
        };

        // Update the mesh asset in-place, and ensure the material is present.
        // This unifies the runtime path with the initial-load path — both end
        // up with the same Mesh3d + MeshMaterial3d pointing to the shared
        // atlas material.
        if let Ok((ent, mesh3d, maybe_mat)) = mesh_query.get(entity) {
            let _ = meshes.insert(&mesh3d.0, mesh);
            // Re-apply material if it was somehow removed (defensive — ensures
            // the runtime path cannot produce a materialless chunk).
            if maybe_mat.is_none() {
                if let Some(ref mat) = chunk_material {
                    commands.entity(ent).insert(MeshMaterial3d(mat.handle.clone()));
                }
            }
        }
    }
}

/// System: despawn chunks that are too far from the player.
fn unload_chunks(
    mut commands: Commands,
    mut manager: ResMut<ChunkManager>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
) {
    let camera_pos = match cameras.iter().next() {
        Some(gt) => gt.translation(),
        None => return,
    };

    let player_chunk = ChunkPos(
        (camera_pos.x as i32).div_euclid(CHUNK_SIZE),
        (camera_pos.y as i32).div_euclid(CHUNK_SIZE),
        (camera_pos.z as i32).div_euclid(CHUNK_SIZE),
    );

    let rd = manager.render_distance + 1; // unload one chunk beyond render distance
    let mut to_remove = Vec::new();

    for (&pos, &entity) in &manager.chunks {
        if (pos.0 - player_chunk.0).abs() > rd
            || (pos.1 - player_chunk.1).abs() > rd
            || (pos.2 - player_chunk.2).abs() > rd
        {
            // Do not unload a chunk that has an active generation task.
            // Despawning the entity would invalidate the ComputeChunk task,
            // causing handle_chunk_tasks to issue commands to a dead entity.
            // The chunk will be unloaded on a future frame after the task completes.
            if manager.pending.contains_key(&pos) {
                continue;
            }
            if let Ok(mut ec) = commands.get_entity(entity) {
                ec.despawn();
                #[cfg(debug_assertions)]
                bevy::log::trace!(
                    "Chunk DESPAWN: ({},{},{}) entity={:?} (unload)",
                    pos.0, pos.1, pos.2, entity,
                );
            }
            to_remove.push(pos);
        }
    }

    for pos in to_remove {
        manager.chunks.remove(&pos);
        manager.chunk_data.remove(&pos);
        // Also clear pending and dirty state for this position to prevent
        // later systems from issuing commands to the despawned entity.
        if let Some(pending_entity) = manager.pending.remove(&pos) {
            // Despawn the pending generation task entity too.
            if let Ok(mut ec) = commands.get_entity(pending_entity) {
                ec.despawn();
            }
        }
        manager.dirty_chunks.remove(&pos);
        manager.deferred_remesh.remove(&pos);
    }
}

/// System: detect texture_size changes and rebuild the atlas image in-place.
fn apply_texture_size_change(
    game_settings: Option<Res<crate::settings::GameSettings>>,
    mut block_atlas: Option<ResMut<BlockAtlas>>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(settings) = game_settings else { return };
    if !settings.is_changed() {
        return;
    }
    let Some(ref mut atlas) = block_atlas else { return };
    let new_tile_size = settings.texture_size.clamp(64, 512);
    if new_tile_size == atlas.tile_size {
        return;
    }

    let new_atlas_size = ATLAS_TILES_PER_ROW * new_tile_size;
    let mut data = vec![255u8; (new_atlas_size * new_atlas_size * 4) as usize];

    let all_blocks = [
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

    // Fill solid color tiles
    for &block in &all_blocks {
        fill_atlas_tile(&mut data, block.index(), block.color(), new_tile_size, new_atlas_size);
    }
    for idx in crate::block_types::CUSTOM_BLOCK_START..crate::block_types::MAX_BLOCK_TYPES {
        fill_atlas_tile(&mut data, idx, Color::WHITE, new_tile_size, new_atlas_size);
    }

    // Re-load block textures (settings path = user-selected via dialog,
    // textures/ directory = developer convenience only)
    let mut new_loaded = HashSet::new();
    for &block in &all_blocks {
        if block == BlockType::AIR {
            continue;
        }
        let name = block.name();
        let name_lower = name.to_lowercase();
        let settings_path = settings.block_textures.get(name);
        let try_paths: Vec<String> = if let Some(sp) = settings_path {
            vec![sp.clone(), format!("textures/{}.png", name_lower)]
        } else {
            vec![format!("textures/{}.png", name_lower)]
        };
        for tex_path in &try_paths {
            if let Ok(img_data) = std::fs::read(tex_path) {
                if let Ok(dyn_img) = image::load_from_memory(&img_data) {
                    let resized = dyn_img.resize_exact(
                        new_tile_size,
                        new_tile_size,
                        image::imageops::FilterType::Lanczos3,
                    );
                    let rgba = resized.to_rgba8();
                    copy_image_to_atlas_tile(&mut data, block.index(), &rgba, new_tile_size, new_atlas_size);
                    new_loaded.insert(block.index());
                    break;
                }
            }
        }
    }

    // Re-load custom block textures (same logic as setup_chunk_material).
    // Without this, custom blocks would lose their textures when the user
    // changes texture resolution — the atlas rebuild only handled built-ins.
    for def in &settings.custom_blocks {
        let idx = crate::block_types::CUSTOM_BLOCK_START
            + settings.custom_blocks.iter().position(|d| d.name == def.name).unwrap_or(0) as u8;
        if idx >= crate::block_types::MAX_BLOCK_TYPES {
            continue;
        }
        if let Ok(img_data) = std::fs::read(&def.texture_path) {
            if let Ok(dyn_img) = image::load_from_memory(&img_data) {
                let resized = dyn_img.resize_exact(
                    new_tile_size,
                    new_tile_size,
                    image::imageops::FilterType::Lanczos3,
                );
                let rgba = resized.to_rgba8();
                copy_image_to_atlas_tile(&mut data, idx, &rgba, new_tile_size, new_atlas_size);
                new_loaded.insert(idx);
            }
        }
    }

    // Replace the image data in-place (with regenerated mip chain).
    if let Some(image) = images.get_mut(&atlas.image_handle) {
        use bevy::image::ImageSampler;
        use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
        let (atlas_chain, mip_count) = generate_atlas_mip_chain(data, new_atlas_size);
        // See note in setup_chunk_material: Image::new would panic on
        // the chain length vs mip0 extent in debug builds.
        let mut new_image = Image::new_uninit(
            Extent3d {
                width: new_atlas_size,
                height: new_atlas_size,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            TextureFormat::Rgba8UnormSrgb,
            bevy::asset::RenderAssetUsages::MAIN_WORLD | bevy::asset::RenderAssetUsages::RENDER_WORLD,
        );
        new_image.texture_descriptor.mip_level_count = mip_count;
        new_image.data = Some(atlas_chain);
        let aniso = settings.anisotropic_filtering.max(1);
        new_image.sampler = ImageSampler::Descriptor(make_atlas_sampler_desc(aniso));
        *image = new_image;
    }

    atlas.tile_size = new_tile_size;
    atlas.atlas_size = new_atlas_size;
    atlas.loaded_textures = new_loaded;
    info!("Atlas rebuilt: {}px tiles ({}x{} atlas)", new_tile_size, new_atlas_size, new_atlas_size);
}

/// System: apply anisotropic filtering changes to the atlas sampler at runtime.
fn apply_aniso_filter_change(
    game_settings: Option<Res<crate::settings::GameSettings>>,
    block_atlas: Option<Res<BlockAtlas>>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(settings) = game_settings else { return };
    if !settings.is_changed() {
        return;
    }
    let Some(atlas) = block_atlas else { return };

    // Only update sampler, not rebuild the whole atlas
    if let Some(image) = images.get_mut(&atlas.image_handle) {
        use bevy::image::ImageSampler;
        let aniso = settings.anisotropic_filtering.max(1);
        image.sampler = ImageSampler::Descriptor(make_atlas_sampler_desc(aniso));
    }
}
