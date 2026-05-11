use std::collections::{HashMap, HashSet};

use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    tasks::{futures::check_ready, AsyncComputeTaskPool, Task},
};

use crate::block_types::BlockType;
use crate::chunk::{Chunk, ChunkNeighbors, ChunkPos, CHUNK_SIZE};
use crate::terrain::{natural_block_at, surface_y};

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

/// Component holding a background chunk generation task.
#[derive(Component)]
struct ComputeChunk {
    pos: ChunkPos,
    task: Task<Chunk>,
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
/// - `surface_cache`: lazily populated by surface_y_cached(). Never invalidated
///   because the procedural surface_y is deterministic and never changes.
///   Player block modifications don't affect surface_y (it's terrain-only).
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
    /// Cached surface Y values per (x, z) column — never invalidated (deterministic).
    pub surface_cache: HashMap<(i32, i32), i32>,
    /// Set of chunk positions currently being generated.
    pending: HashMap<ChunkPos, Entity>,
    /// Set of chunk positions that need remeshing (block was modified).
    dirty_chunks: HashSet<ChunkPos>,
}

impl Default for ChunkManager {
    fn default() -> Self {
        Self {
            chunks: HashMap::new(),
            chunk_data: HashMap::new(),
            render_distance: DEFAULT_RENDER_DISTANCE,
            modifications: HashMap::new(),
            surface_cache: HashMap::new(),
            pending: HashMap::new(),
            dirty_chunks: HashSet::new(),
        }
    }
}

impl ChunkManager {
    /// Clear all chunk state for a clean Gameplay re-entry.
    /// Called by cleanup_world on entering Menu.
    pub fn clear_all(&mut self) {
        self.chunks.clear();
        self.chunk_data.clear();
        self.pending.clear();
        self.dirty_chunks.clear();
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

        let chunk_pos = world_to_chunk_pos(world_pos);
        self.dirty_chunks.insert(chunk_pos);

        // If the block is on a chunk boundary, also dirty the neighbor chunk
        let local = world_to_local(world_pos);
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

    /// Mark all loaded chunks as dirty so they get remeshed (e.g. after color changes).
    pub fn mark_all_dirty(&mut self) {
        for &pos in self.chunks.keys() {
            self.dirty_chunks.insert(pos);
        }
    }

    /// Get the cached surface Y for a column, computing and caching if needed.
    pub fn surface_y_cached(&mut self, x: i32, z: i32) -> i32 {
        *self
            .surface_cache
            .entry((x, z))
            .or_insert_with(|| surface_y(x, z))
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
    use bevy::image::{ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
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

    let mut image = Image::new(
        Extent3d {
            width: atlas_size,
            height: atlas_size,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );

    // Anisotropic filtering: minimum 1 (no aniso) to avoid wgpu validation
    // errors. Values >1 enable aniso for better texture quality at oblique
    // angles. Nearest filtering preserves the pixel-art/voxel grid aesthetic.
    let aniso = game_settings.anisotropic_filtering.max(1);
    let mut sampler_desc = ImageSamplerDescriptor {
        mag_filter: ImageFilterMode::Nearest,
        min_filter: ImageFilterMode::Nearest,
        ..default()
    };
    if aniso > 1 {
        sampler_desc.set_anisotropic_filter(aniso);
    }
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

/// System: check player position and spawn async generation tasks for chunks in range.
fn load_chunks(
    mut commands: Commands,
    mut manager: ResMut<ChunkManager>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    game_settings: Option<Res<crate::settings::GameSettings>>,
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
        manager.render_distance = gs.render_distance;
    }
    let rd = manager.render_distance;
    let thread_pool = AsyncComputeTaskPool::get();

    // Vertical loading range is asymmetric: only 2 chunks below the player
    // (32 blocks — enough to see nearby caves) vs full render_distance above.
    // Underground chunks far below are invisible and waste generation time.
    // Floor at Y=-7 because bedrock is at depth 100 (~Y=-90 for surface at
    // Y=10), and chunk Y=-7 starts at block Y=-112 — well below bedrock.
    let y_min = (player_chunk.1 - 2).max(-7);
    let y_max = player_chunk.1 + rd;
    for cy in y_min..=y_max {
        for cz in (player_chunk.2 - rd)..=(player_chunk.2 + rd) {
            for cx in (player_chunk.0 - rd)..=(player_chunk.0 + rd) {
                let pos = ChunkPos(cx, cy, cz);
                if manager.chunks.contains_key(&pos) || manager.pending.contains_key(&pos) {
                    continue;
                }

                let entity = commands.spawn_empty().id();
                let task = thread_pool.spawn(async move { Chunk::generate(pos) });

                commands.entity(entity).insert(ComputeChunk { pos, task });
                manager.pending.insert(pos, entity);
                #[cfg(debug_assertions)]
                bevy::log::trace!(
                    "Chunk SPAWN: ({},{},{}) entity={:?} (pending)",
                    pos.0, pos.1, pos.2, entity,
                );
            }
        }
    }
}

/// Apply player modifications to a freshly generated chunk. Used by both
/// handle_chunk_tasks (initial load) and remesh_dirty_chunks (runtime) to
/// ensure both paths produce identical block data.
fn apply_modifications(chunk: &mut Chunk, modifications: &HashMap<IVec3, BlockType>) {
    let base_x = chunk.pos.0 * CHUNK_SIZE;
    let base_y = chunk.pos.1 * CHUNK_SIZE;
    let base_z = chunk.pos.2 * CHUNK_SIZE;
    for lx in 0..CHUNK_SIZE {
        for ly in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let wp = IVec3::new(base_x + lx, base_y + ly, base_z + lz);
                if let Some(&bt) = modifications.get(&wp) {
                    chunk.blocks[Chunk::index(lx, ly, lz)] = bt;
                }
            }
        }
    }
}

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

    for (entity, mut compute) in &mut tasks {
        if let Some(mut chunk) = check_ready(&mut compute.task) {
            let pos = compute.pos;

            apply_modifications(&mut chunk, &manager.modifications);

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

            let no_neighbors = ChunkNeighbors {
                neighbors: [
                    manager.chunk_data.get(&ChunkPos(pos.0 - 1, pos.1, pos.2)),
                    manager.chunk_data.get(&ChunkPos(pos.0 + 1, pos.1, pos.2)),
                    manager.chunk_data.get(&ChunkPos(pos.0, pos.1 - 1, pos.2)),
                    manager.chunk_data.get(&ChunkPos(pos.0, pos.1 + 1, pos.2)),
                    manager.chunk_data.get(&ChunkPos(pos.0, pos.1, pos.2 - 1)),
                    manager.chunk_data.get(&ChunkPos(pos.0, pos.1, pos.2 + 1)),
                ],
            };
            let mesh = chunk.build_mesh(&no_neighbors, opt_flags.enable_greedy_meshing, dev.highlight_greedy_quads);

            // Debug: warn if a non-air chunk produces an empty mesh.
            #[cfg(debug_assertions)]
            {
                let has_solid = chunk.blocks.iter().any(|b| *b != BlockType::AIR);
                if has_solid && mesh.count_vertices() == 0 {
                    bevy::log::warn!(
                        "Chunk ({},{},{}) has solid blocks but mesh has 0 vertices \
                         — will render as invisible",
                        pos.0, pos.1, pos.2,
                    );
                }
            }

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
    if manager.dirty_chunks.is_empty() {
        return;
    }

    let dirty: Vec<ChunkPos> = manager.dirty_chunks.drain().collect();

    for pos in dirty {
        // Only remesh chunks that are actually loaded
        let Some(&entity) = manager.chunks.get(&pos) else {
            continue;
        };

        // Verify the entity still exists before doing expensive mesh work.
        // The entity may have been despawned by unload_chunks or cleanup_world.
        if mesh_query.get(entity).is_err() {
            manager.chunks.remove(&pos);
            manager.chunk_data.remove(&pos);
            continue;
        }

        // Regenerate the chunk from scratch, then apply modifications
        // (same apply_modifications used by handle_chunk_tasks)
        let mut chunk = Chunk::generate(pos);
        apply_modifications(&mut chunk, &manager.modifications);

        // Clone neighbor data to avoid borrow conflict when we later insert into chunk_data
        let neighbor_positions = [
            ChunkPos(pos.0 - 1, pos.1, pos.2),
            ChunkPos(pos.0 + 1, pos.1, pos.2),
            ChunkPos(pos.0, pos.1 - 1, pos.2),
            ChunkPos(pos.0, pos.1 + 1, pos.2),
            ChunkPos(pos.0, pos.1, pos.2 - 1),
            ChunkPos(pos.0, pos.1, pos.2 + 1),
        ];
        let neighbor_chunks: Vec<Option<Chunk>> = neighbor_positions
            .iter()
            .map(|np| {
                manager.chunk_data.get(np).map(|c| Chunk {
                    blocks: c.blocks,
                    pos: c.pos,
                })
            })
            .collect();

        let neighbors = ChunkNeighbors {
            neighbors: [
                neighbor_chunks[0].as_ref(),
                neighbor_chunks[1].as_ref(),
                neighbor_chunks[2].as_ref(),
                neighbor_chunks[3].as_ref(),
                neighbor_chunks[4].as_ref(),
                neighbor_chunks[5].as_ref(),
            ],
        };
        let mesh = chunk.build_mesh(&neighbors, opt_flags.enable_greedy_meshing, dev.highlight_greedy_quads);

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

        // Update stored chunk data so future lookups reflect modifications
        manager.chunk_data.insert(pos, chunk);
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

    // Replace the image data in-place
    if let Some(image) = images.get_mut(&atlas.image_handle) {
        use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
        *image = Image::new(
            Extent3d {
                width: new_atlas_size,
                height: new_atlas_size,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            data,
            TextureFormat::Rgba8UnormSrgb,
            bevy::asset::RenderAssetUsages::MAIN_WORLD | bevy::asset::RenderAssetUsages::RENDER_WORLD,
        );
        use bevy::image::{ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
        let aniso = settings.anisotropic_filtering.max(1);
        let mut sampler_desc = ImageSamplerDescriptor {
            mag_filter: ImageFilterMode::Nearest,
            min_filter: ImageFilterMode::Nearest,
            ..default()
        };
        if aniso > 1 {
            sampler_desc.set_anisotropic_filter(aniso);
        }
        image.sampler = ImageSampler::Descriptor(sampler_desc);
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
        use bevy::image::{ImageFilterMode, ImageSampler, ImageSamplerDescriptor};
        let aniso = settings.anisotropic_filtering.max(1);
        let mut sampler_desc = ImageSamplerDescriptor {
            mag_filter: ImageFilterMode::Nearest,
            min_filter: ImageFilterMode::Nearest,
            ..default()
        };
        if aniso > 1 {
            sampler_desc.set_anisotropic_filter(aniso);
        }
        image.sampler = ImageSampler::Descriptor(sampler_desc);
    }
}
