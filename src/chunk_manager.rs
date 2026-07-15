use std::collections::{HashMap, HashSet};

use bevy::{
    asset::RenderAssetUsages,
    pbr::{ExtendedMaterial, MaterialExtension},
    prelude::*,
    render::render_resource::AsBindGroup,
    shader::ShaderRef,
    tasks::{futures::check_ready, AsyncComputeTaskPool, Task},
};

use crate::block_types::BlockType;
use crate::chunk::{Chunk, ChunkNeighbors, ChunkPos, CHUNK_SIZE};
use crate::terrain::natural_block_at;

/// Default render distance in chunks (5 chunks = 80 blocks radius).
pub const DEFAULT_RENDER_DISTANCE: i32 = 5;

/// Number of layers in the block texture array — one per block type.
pub const BLOCK_LAYER_COUNT: u32 = crate::block_types::MAX_BLOCK_TYPES as u32;

/// StandardMaterial extension that samples base color from a 2D array
/// texture (one layer per block type) instead of a tiled atlas. The mesh
/// supplies the layer index in UV_1.x (see chunk.rs build_mesh) and the
/// fragment shader (assets/shaders/block_array.wgsl) does the array
/// sample, plus per-layer emissive gating for the lantern.
///
/// WHY an array over the old atlas: no UV-inset bleed hacks, correct
/// per-layer hardware-addressable mips, and UV repeat on greedy-merged
/// quads — which a shared atlas cannot express.
#[derive(Asset, AsBindGroup, Reflect, Debug, Clone)]
pub struct BlockArrayExtension {
    // Extension bindings start at 100 (0-99 reserved for StandardMaterial).
    #[texture(100, dimension = "2d_array")]
    #[sampler(101)]
    pub layers: Handle<Image>,
}

impl MaterialExtension for BlockArrayExtension {
    fn fragment_shader() -> ShaderRef {
        "shaders/block_array.wgsl".into()
    }
}

/// The concrete chunk material type: full PBR pipeline (fog, shadows,
/// tonemapping) with the array-texture base color swap.
pub type BlockArrayMaterial = ExtendedMaterial<StandardMaterial, BlockArrayExtension>;

/// Shared material handle for all chunk meshes.
#[derive(Resource)]
pub struct ChunkMaterial {
    pub handle: Handle<BlockArrayMaterial>,
}

/// Resource holding the block layer-array image so layers can be updated
/// at runtime (texture menu). Data layout is LAYER-MAJOR (matching
/// `TextureDataOrder::LayerMajor`): each layer's full mip chain is
/// contiguous, so a runtime tile update rewrites one layer's slice and
/// regenerates only that layer's mips.
#[derive(Resource)]
pub struct BlockAtlas {
    pub image_handle: Handle<Image>,
    /// Tracks which block types have a loaded PNG texture (by u8 index).
    pub loaded_textures: HashSet<u8>,
    /// Current tile (layer) size in pixels. Always a power of two.
    pub tile_size: u32,
    /// Mip levels per layer (down to 1×1).
    pub mip_count: u32,
    /// Anisotropic filtering level the sampler was last built with.
    pub aniso: u16,
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

    /// True when no generation tasks are in flight and no remesh work is
    /// queued — used by the dev screenshot harness to wait for a fully
    /// streamed-in world before capturing.
    pub fn streaming_idle(&self) -> bool {
        self.pending.is_empty() && self.dirty_chunks.is_empty() && self.deferred_remesh.is_empty()
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
            // Registers the chunk material type (StandardMaterial extended
            // with the block layer-array sampler).
            .add_plugins(MaterialPlugin::<BlockArrayMaterial>::default())
            .add_systems(Startup, setup_chunk_material)
            // ORDERING CONTRACT: chained because each stage depends on the previous:
            //   1. load_chunks           — spawns async gen tasks for missing chunks
            //   2. handle_chunk_tasks    — polls completed tasks, inserts chunk data + mesh
            //   3. remesh_dirty_chunks   — rebuilds meshes for modified chunks
            //   4. unload_chunks         — despawns chunks beyond render distance
            //   5. apply_texture_settings_change — rebuilds layers on texture_size
            //      change, swaps sampler on anisotropy change
            // Step 5 must run after meshing so new meshes get the correct material.
            .add_systems(
                Update,
                (
                    on_block_registry_changed,
                    load_chunks,
                    handle_chunk_tasks,
                    remesh_dirty_chunks,
                    unload_chunks,
                    apply_texture_settings_change,
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

// ---------------------------------------------------------------------------
// Layer-array data layout (LAYER-MAJOR: per-layer contiguous mip chains)
// ---------------------------------------------------------------------------

/// Normalize a requested tile size to a power of two in [64, 512]
/// (rounds down) so the mip chain divides cleanly to 1×1.
pub fn normalize_tile_size(size: u32) -> u32 {
    let s = size.clamp(64, 512);
    1 << (31 - s.leading_zeros())
}

/// Mip levels for a power-of-two tile, down to 1×1 (64 px → 7 levels).
pub fn mip_count_for(tile_size: u32) -> u32 {
    tile_size.trailing_zeros() + 1
}

/// Bytes of one layer's full mip chain.
fn layer_chain_bytes(tile_size: u32, mip_count: u32) -> usize {
    (0..mip_count)
        .map(|m| {
            let s = (tile_size >> m).max(1);
            (s * s * 4) as usize
        })
        .sum()
}

/// Write a solid color + darkened 3 px grid border into a layer's mip 0,
/// then regenerate that layer's mip chain.
pub fn write_layer_solid(
    data: &mut [u8],
    layer: u8,
    color: Color,
    tile_size: u32,
    mip_count: u32,
) {
    if layer as u32 >= BLOCK_LAYER_COUNT {
        return;
    }
    let chain = layer_chain_bytes(tile_size, mip_count);
    let slice = &mut data[layer as usize * chain..(layer as usize + 1) * chain];

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
            let i = ((py * tile_size + px) * 4) as usize;
            if is_border {
                // Subtle darkened border
                slice[i] = ((r as f32) * 0.65) as u8;
                slice[i + 1] = ((g as f32) * 0.65) as u8;
                slice[i + 2] = ((b as f32) * 0.65) as u8;
                slice[i + 3] = a;
            } else {
                slice[i] = r;
                slice[i + 1] = g;
                slice[i + 2] = b;
                slice[i + 3] = a;
            }
        }
    }
    regenerate_layer_mips(slice, tile_size, mip_count);
}

/// Copy RGBA8 pixel data (tile_size²) into a layer's mip 0 with the
/// darkened 3 px border, then regenerate that layer's mip chain.
pub fn write_layer_rgba(
    data: &mut [u8],
    layer: u8,
    rgba: &image::RgbaImage,
    tile_size: u32,
    mip_count: u32,
) {
    if layer as u32 >= BLOCK_LAYER_COUNT {
        return;
    }
    let chain = layer_chain_bytes(tile_size, mip_count);
    let slice = &mut data[layer as usize * chain..(layer as usize + 1) * chain];

    let border = 3u32;
    for py in 0..tile_size {
        for px in 0..tile_size {
            let i = ((py * tile_size + px) * 4) as usize;
            let src = rgba.get_pixel(px, py);
            let is_border = px < border || px >= tile_size - border
                || py < border || py >= tile_size - border;
            if is_border {
                slice[i] = ((src[0] as f32) * 0.65) as u8;
                slice[i + 1] = ((src[1] as f32) * 0.65) as u8;
                slice[i + 2] = ((src[2] as f32) * 0.65) as u8;
                slice[i + 3] = src[3];
            } else {
                slice[i] = src[0];
                slice[i + 1] = src[1];
                slice[i + 2] = src[2];
                slice[i + 3] = src[3];
            }
        }
    }
    regenerate_layer_mips(slice, tile_size, mip_count);
}

/// Rebuild mips 1..N of a single layer chain from its mip 0 by successive
/// 2×2 box filtering in LINEAR space (naive sRGB averaging darkens).
/// Chaining mip-from-previous-mip is exact here because a layer contains
/// only its own block's texels — the cross-tile contamination that forced
/// the old atlas to downsample every mip from mip 0 cannot happen.
fn regenerate_layer_mips(layer_slice: &mut [u8], tile_size: u32, mip_count: u32) {
    // sRGB-to-linear LUT for byte values 0..=255.
    let mut srgb_to_lin = [0.0f32; 256];
    for (i, slot) in srgb_to_lin.iter_mut().enumerate() {
        let c = i as f32 / 255.0;
        *slot = if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
    }

    let mut src_offset = 0usize;
    for m in 1..mip_count {
        let src_size = (tile_size >> (m - 1)).max(1);
        let dst_size = (tile_size >> m).max(1);
        let dst_offset = src_offset + (src_size * src_size * 4) as usize;

        for dy in 0..dst_size {
            for dx in 0..dst_size {
                let mut sum_rgb = [0.0f32; 3];
                let mut sum_a: u32 = 0;
                for sy in 0..2u32 {
                    for sx in 0..2u32 {
                        let s_x = (dx * 2 + sx).min(src_size - 1);
                        let s_y = (dy * 2 + sy).min(src_size - 1);
                        let si = src_offset + ((s_y * src_size + s_x) * 4) as usize;
                        sum_rgb[0] += srgb_to_lin[layer_slice[si] as usize];
                        sum_rgb[1] += srgb_to_lin[layer_slice[si + 1] as usize];
                        sum_rgb[2] += srgb_to_lin[layer_slice[si + 2] as usize];
                        sum_a += layer_slice[si + 3] as u32;
                    }
                }
                let di = dst_offset + ((dy * dst_size + dx) * 4) as usize;
                for c in 0..3 {
                    layer_slice[di + c] =
                        (linear_to_srgb(sum_rgb[c] * 0.25) * 255.0).round().clamp(0.0, 255.0) as u8;
                }
                layer_slice[di + 3] = ((sum_a as f32) * 0.25).round().clamp(0.0, 255.0) as u8;
            }
        }
        src_offset = dst_offset;
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

/// Build the block layer-array image and shared material for all chunk
/// meshes. Single source of truth for layer content — the texture-size
/// rebuild path calls the same builder (the old code had two ~150-line
/// copies of the atlas fill that had already drifted).
fn setup_chunk_material(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<BlockArrayMaterial>>,
    game_settings: Res<crate::settings::GameSettings>,
) {
    let tile_size = normalize_tile_size(game_settings.texture_size);
    let (data, mip_count, loaded_textures) = build_layers_data(&game_settings, tile_size);

    let aniso = game_settings.anisotropic_filtering.max(1);
    let image = make_layer_array_image(data, tile_size, mip_count, aniso);
    let image_handle = images.add(image);

    let material = materials.add(BlockArrayMaterial {
        base: StandardMaterial {
            base_color: Color::WHITE,
            // Warm lantern glow: the shader zeroes this for every layer
            // except the lantern's (replaces the old emissive atlas).
            emissive: bevy::color::LinearRgba::new(6.0, 5.1, 2.7, 1.0),
            reflectance: 0.0,
            ..default()
        },
        extension: BlockArrayExtension {
            layers: image_handle.clone(),
        },
    });

    #[cfg(debug_assertions)]
    {
        let mat_valid = materials.get(&material).is_some();
        let img_valid = images.get(&image_handle).is_some();
        bevy::log::info!(
            "Asset validation: material={} layer_array={} tile_size={} mips={} loaded_textures={}/{}",
            if mat_valid { "OK" } else { "MISSING" },
            if img_valid { "OK" } else { "MISSING" },
            tile_size,
            mip_count,
            loaded_textures.len(),
            crate::block_types::BUILTIN_BLOCK_COUNT,
        );
        debug_assert!(mat_valid, "Chunk material handle is invalid after creation");
        debug_assert!(img_valid, "Layer-array image handle is invalid after creation");
    }

    commands.insert_resource(ChunkMaterial { handle: material });
    commands.insert_resource(BlockAtlas {
        image_handle,
        loaded_textures,
        tile_size,
        mip_count,
        aniso,
    });
}

/// Build the full layer-major data blob for all block layers: solid colors
/// + grid borders for every block type, overwritten by PNG textures where
/// configured (settings paths first, then the textures/ dev directory).
/// Returns (data, mip_count, loaded_texture_indices).
fn build_layers_data(
    settings: &crate::settings::GameSettings,
    tile_size: u32,
) -> (Vec<u8>, u32, HashSet<u8>) {
    let mip_count = mip_count_for(tile_size);
    let chain = layer_chain_bytes(tile_size, mip_count);
    let mut data = vec![255u8; chain * BLOCK_LAYER_COUNT as usize];

    for &block in &crate::block_types::ALL_BUILTIN_BLOCKS_WITH_AIR {
        write_layer_solid(&mut data, block.index(), block.color(), tile_size, mip_count);
    }
    for idx in crate::block_types::CUSTOM_BLOCK_START..crate::block_types::MAX_BLOCK_TYPES {
        write_layer_solid(&mut data, idx, Color::WHITE, tile_size, mip_count);
    }

    // Load block textures from two sources (settings path takes priority):
    // 1. settings.block_textures — user-selected via the in-game file dialog.
    // 2. textures/ directory — developer convenience for bundled defaults.
    let mut loaded_textures = HashSet::new();
    for &block in &crate::block_types::ALL_BUILTIN_BLOCKS {
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
                        tile_size,
                        tile_size,
                        image::imageops::FilterType::Lanczos3,
                    );
                    write_layer_rgba(&mut data, block.index(), &resized.to_rgba8(), tile_size, mip_count);
                    loaded_textures.insert(block.index());
                    info!("Loaded texture: {} ({}x{})", tex_path, tile_size, tile_size);
                    break; // first successful source wins
                }
            }
        }
    }

    // Custom block textures from their saved paths.
    for (slot, def) in settings.custom_blocks.iter().enumerate() {
        let idx = crate::block_types::CUSTOM_BLOCK_START as usize + slot;
        if idx >= crate::block_types::MAX_BLOCK_TYPES as usize {
            break;
        }
        if let Ok(img_data) = std::fs::read(&def.texture_path) {
            if let Ok(dyn_img) = image::load_from_memory(&img_data) {
                let resized = dyn_img.resize_exact(
                    tile_size,
                    tile_size,
                    image::imageops::FilterType::Lanczos3,
                );
                write_layer_rgba(&mut data, idx as u8, &resized.to_rgba8(), tile_size, mip_count);
                loaded_textures.insert(idx as u8);
                info!("Loaded custom block texture: {} ({}x{})", def.name, tile_size, tile_size);
            }
        }
    }

    (data, mip_count, loaded_textures)
}

/// Assemble the D2 array Image from layer-major data.
fn make_layer_array_image(data: Vec<u8>, tile_size: u32, mip_count: u32, aniso: u16) -> Image {
    use bevy::image::ImageSampler;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat, TextureDataOrder};

    // new_uninit + manual assignment: Image::new's debug_assert compares
    // data length against mip 0 alone and would reject the mip chain.
    let mut image = Image::new_uninit(
        Extent3d {
            width: tile_size,
            height: tile_size,
            depth_or_array_layers: BLOCK_LAYER_COUNT,
        },
        TextureDimension::D2,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.mip_level_count = mip_count;
    // Our data is packed layer-major (each layer's mip chain contiguous);
    // tell the uploader explicitly rather than relying on the default.
    image.data_order = TextureDataOrder::LayerMajor;
    image.data = Some(data);
    image.sampler = ImageSampler::Descriptor(make_layer_sampler_desc(aniso));
    image
}


/// Sampler for the block layer array. Mag = Nearest preserves the crisp
/// pixel look up close; Min + Mipmap = Linear kills distance shimmer.
/// Address mode = Repeat so greedy-merged quads tile their block texture
/// (UVs run 0..w along merged runs). NOTE: enabling anisotropy forces all
/// filters to Linear (wgpu validity requirement — bevy's helper does this),
/// trading crisp closeups for smooth grazing angles.
fn make_layer_sampler_desc(aniso: u16) -> bevy::image::ImageSamplerDescriptor {
    use bevy::image::{ImageAddressMode, ImageFilterMode, ImageSamplerDescriptor};
    let mut desc = ImageSamplerDescriptor {
        address_mode_u: ImageAddressMode::Repeat,
        address_mode_v: ImageAddressMode::Repeat,
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
    pending_reload: Res<crate::PendingReload>,
    mut last_scan: Local<Option<(ChunkPos, i32, u64)>>,
) {
    // No new generation tasks while a Play/Load reload is counting down:
    // the WorldInstanceId is already bumped, so a task spawned here would
    // carry the NEW id, survive teardown, and duplicate a chunk the fresh
    // session generates for itself (its `pending` entry is wiped by
    // clear_all, so nothing would dedupe it).
    if pending_reload.active {
        return;
    }

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
    world_ready: Option<Res<crate::player::WorldReady>>,
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
    //
    // While the world ISN'T ready yet (loading overlay up, player camera
    // inactive) there is nothing on screen a hitch could disturb — mesh 8×
    // as aggressively so large render distances fill in seconds instead of
    // visibly assembling in front of the player after the gate opens.
    let ready = world_ready.map(|r| r.0).unwrap_or(true);
    let base_budget = dev.max_chunk_meshes_per_frame.max(1) as usize;
    let mesh_budget = if ready { base_budget } else { base_budget * 8 };
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
            // Do NOT re-insert WorldEntity/WorldScoped here — the entity
            // carries them from its spawn in load_chunks. Re-stamping with
            // the CURRENT WorldInstanceId would let a task that completes
            // during the PendingReload deferral (after the id bump, before
            // teardown) adopt the NEW world's id and survive teardown as a
            // ghost mesh carrying the abandoned session's data.
            ec.remove::<ComputeChunk>().insert((
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
    mesh_query: Query<(Entity, &Mesh3d, Option<&MeshMaterial3d<BlockArrayMaterial>>), With<ChunkMarker>>,
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

/// React to texture settings changes: rebuilds all layers when
/// texture_size changes, or just swaps the sampler when only the
/// anisotropic filtering level changed. Replaces the two previous systems
/// (apply_texture_size_change / apply_aniso_filter_change), whose size
/// path was a full copy of the setup builder.
fn apply_texture_settings_change(
    game_settings: Option<Res<crate::settings::GameSettings>>,
    mut block_atlas: Option<ResMut<BlockAtlas>>,
    mut images: ResMut<Assets<Image>>,
) {
    let Some(settings) = game_settings else { return };
    if !settings.is_changed() {
        return;
    }
    let Some(ref mut atlas) = block_atlas else { return };

    let new_tile = normalize_tile_size(settings.texture_size);
    let new_aniso = settings.anisotropic_filtering.max(1);

    if new_tile != atlas.tile_size {
        // Full rebuild through the same builder setup uses.
        let (data, mip_count, loaded) = build_layers_data(&settings, new_tile);
        if let Some(image) = images.get_mut(&atlas.image_handle) {
            *image = make_layer_array_image(data, new_tile, mip_count, new_aniso);
        }
        atlas.tile_size = new_tile;
        atlas.mip_count = mip_count;
        atlas.loaded_textures = loaded;
        atlas.aniso = new_aniso;
        info!("Block layers rebuilt: {}px tiles, {} mips", new_tile, mip_count);
    } else if new_aniso != atlas.aniso {
        // Sampler-only change: no data rebuild.
        if let Some(image) = images.get_mut(&atlas.image_handle) {
            use bevy::image::ImageSampler;
            image.sampler = ImageSampler::Descriptor(make_layer_sampler_desc(new_aniso));
        }
        atlas.aniso = new_aniso;
    }
}
