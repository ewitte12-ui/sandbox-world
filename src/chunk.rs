use bevy::{
    asset::RenderAssetUsages, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
};
use block_mesh::{
    ndshape::{ConstShape, ConstShape3u32},
    visible_block_faces, UnitQuadBuffer, RIGHT_HANDED_Y_UP_CONFIG,
};

use std::collections::{HashMap, HashSet};

use crate::block_types::BlockType;
use crate::chunk_manager::{ATLAS_TILES_PER_ROW};
use crate::terrain::{natural_block_at, place_trees_in_chunk};

/// Debug: set to true by build_mesh if a greedy meshing invariant is violated.
/// Checked by chunk_manager systems to auto-disable greedy meshing.
#[cfg(debug_assertions)]
pub static GREEDY_INVARIANT_VIOLATED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub const CHUNK_SIZE: i32 = 16;
pub const CHUNK_VOLUME: usize = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;

/// Padded chunk shape: 18x18x18 (16 + 1 padding on each side for neighbor data).
type PaddedChunkShape = ConstShape3u32<18, 18, 18>;

/// Chunk coordinates in chunk-space (not world-space).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ChunkPos(pub i32, pub i32, pub i32);

/// Provides neighboring chunk block data for seamless meshing across boundaries.
pub struct ChunkNeighbors<'a> {
    /// -X, +X, -Y, +Y, -Z, +Z neighbor chunks (None if not loaded)
    pub neighbors: [Option<&'a Chunk>; 6],
}

/// A 16x16x16 voxel chunk.
pub struct Chunk {
    pub blocks: [BlockType; CHUNK_VOLUME],
    pub pos: ChunkPos,
}

impl Chunk {
    /// Flat index into the blocks array from local coordinates.
    #[inline]
    pub fn index(x: i32, y: i32, z: i32) -> usize {
        (x + y * CHUNK_SIZE + z * CHUNK_SIZE * CHUNK_SIZE) as usize
    }

    /// Get the block at local chunk coordinates, returning Air for out-of-bounds.
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockType {
        if !(0..CHUNK_SIZE).contains(&x)
            || !(0..CHUNK_SIZE).contains(&y)
            || !(0..CHUNK_SIZE).contains(&z)
        {
            return BlockType::AIR;
        }
        self.blocks[Self::index(x, y, z)]
    }

    /// Generate a chunk by filling blocks from terrain generation.
    pub fn generate(pos: ChunkPos) -> Self {
        let mut blocks = [BlockType::AIR; CHUNK_VOLUME];
        let base_x = pos.0 * CHUNK_SIZE;
        let base_y = pos.1 * CHUNK_SIZE;
        let base_z = pos.2 * CHUNK_SIZE;

        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let wx = base_x + x;
                    let wy = base_y + y;
                    let wz = base_z + z;
                    blocks[Self::index(x, y, z)] = natural_block_at(wx, wy, wz);
                }
            }
        }

        place_trees_in_chunk(&mut blocks, pos.0, pos.1, pos.2);

        Chunk { blocks, pos }
    }

    /// Build a Bevy Mesh from this chunk's block data.
    ///
    /// Uses block_mesh's `visible_block_faces` for per-block face culling:
    /// a face is emitted only if the adjacent block is Air. This is the
    /// occlusion contract — fully interior blocks produce zero geometry.
    /// The padded 18³ buffer includes 1 block of neighbor data on each
    /// side so that faces at chunk boundaries are correctly culled against
    /// the neighboring chunk's blocks (or Air if neighbor is unloaded).
    ///
    /// RENDERING NOTE: all vertex positions are in float chunk-local space
    /// (0.0..16.0). The chunk's world offset is applied via Transform in
    /// handle_chunk_tasks. UVs map into a texture atlas using float division
    /// (tile_uv_size = 0.25 for a 4×4 atlas). The 0.01 epsilon in face-axis
    /// detection avoids misclassification from float imprecision in quad
    /// vertex positions.
    pub fn build_mesh(&self, neighbors: &ChunkNeighbors, greedy: bool, highlight_greedy: bool) -> Mesh {
        // Build padded 18x18x18 voxel buffer
        let mut padded = [BlockType::AIR; PaddedChunkShape::SIZE as usize];

        // Fill interior (1..17 in each axis maps to 0..16 in chunk local)
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let pi = PaddedChunkShape::linearize([
                        (x + 1) as u32,
                        (y + 1) as u32,
                        (z + 1) as u32,
                    ]);
                    padded[pi as usize] = self.blocks[Self::index(x, y, z)];
                }
            }
        }

        // Fill padding from neighbors
        // Neighbor order: -X, +X, -Y, +Y, -Z, +Z
        // -X face (padded x=0, chunk x=15 of neighbor)
        if let Some(neg_x) = neighbors.neighbors[0] {
            for z in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    let pi = PaddedChunkShape::linearize([0, (y + 1) as u32, (z + 1) as u32]);
                    padded[pi as usize] = neg_x.get_block(CHUNK_SIZE - 1, y, z);
                }
            }
        }
        // +X face (padded x=17, chunk x=0 of neighbor)
        if let Some(pos_x) = neighbors.neighbors[1] {
            for z in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    let pi = PaddedChunkShape::linearize([17, (y + 1) as u32, (z + 1) as u32]);
                    padded[pi as usize] = pos_x.get_block(0, y, z);
                }
            }
        }
        // -Y face (padded y=0, chunk y=15 of neighbor)
        if let Some(neg_y) = neighbors.neighbors[2] {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let pi = PaddedChunkShape::linearize([(x + 1) as u32, 0, (z + 1) as u32]);
                    padded[pi as usize] = neg_y.get_block(x, CHUNK_SIZE - 1, z);
                }
            }
        }
        // +Y face (padded y=17, chunk y=0 of neighbor)
        if let Some(pos_y) = neighbors.neighbors[3] {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let pi = PaddedChunkShape::linearize([(x + 1) as u32, 17, (z + 1) as u32]);
                    padded[pi as usize] = pos_y.get_block(x, 0, z);
                }
            }
        }
        // -Z face (padded z=0, chunk z=15 of neighbor)
        if let Some(neg_z) = neighbors.neighbors[4] {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let pi = PaddedChunkShape::linearize([(x + 1) as u32, (y + 1) as u32, 0]);
                    padded[pi as usize] = neg_z.get_block(x, y, CHUNK_SIZE - 1);
                }
            }
        }
        // +Z face (padded z=17, chunk z=0 of neighbor)
        if let Some(pos_z) = neighbors.neighbors[5] {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let pi = PaddedChunkShape::linearize([(x + 1) as u32, (y + 1) as u32, 17]);
                    padded[pi as usize] = pos_z.get_block(x, y, 0);
                }
            }
        }

        // Debug: verify AIR padding is non-opaque. If AIR were opaque,
        // all boundary faces would be incorrectly culled.
        #[cfg(debug_assertions)]
        debug_assert!(
            !BlockType::AIR.is_opaque(),
            "BlockType::AIR must not be opaque — boundary faces depend on this"
        );

        // Per-block face culling (not greedy meshing). Greedy meshing would
        // merge adjacent same-type faces into larger quads for fewer draw calls,
        // but would break per-block atlas UV mapping and the grid border effect.
        let mut buffer = UnitQuadBuffer::new();
        visible_block_faces(
            &padded,
            &PaddedChunkShape {},
            [0; 3],
            [17; 3],
            &RIGHT_HANDED_Y_UP_CONFIG.faces,
            &mut buffer,
        );

        // Convert unit quads to mesh vertices
        let mut positions: Vec<[f32; 3]> = Vec::new();
        let mut normals: Vec<[f32; 3]> = Vec::new();
        let mut colors: Vec<[f32; 4]> = Vec::new();
        let mut uvs: Vec<[f32; 2]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let tiles_per_row = ATLAS_TILES_PER_ROW;
        let tile_uv_size = 1.0 / tiles_per_row as f32;

        // Helper: emit a single quad into the mesh buffers.
        // `p0..p3` are in padded space (offset by -1 to chunk-local).
        // `w` and `h` are the quad dimensions in blocks (1 for non-greedy).
        let emit_quad = |
            positions: &mut Vec<[f32; 3]>,
            normals: &mut Vec<[f32; 3]>,
            colors: &mut Vec<[f32; 4]>,
            uvs: &mut Vec<[f32; 2]>,
            indices: &mut Vec<u32>,
            corners: [[f32; 3]; 4],
            normal: [f32; 3],
            face_brightness: f32,
            tile_u0: f32,
            tile_v0: f32,
            ax_u: usize,
            ax_v: usize,
            _w: f32,
            _h: f32,
        | {
            let idx = positions.len() as u32;
            let min_u = corners[0][ax_u];
            let min_v = corners[0][ax_v];
            for c in &corners {
                positions.push([c[0] - 1.0, c[1] - 1.0, c[2] - 1.0]);
                // Tiled UVs: each block-sized cell maps to the full atlas tile.
                let lu = c[ax_u] - min_u;
                let lv = c[ax_v] - min_v;
                uvs.push([
                    tile_u0 + lu * tile_uv_size,
                    tile_v0 + lv * tile_uv_size,
                ]);
                normals.push(normal);
                colors.push([face_brightness, face_brightness, face_brightness, 1.0]);
            }
            for i in &[0u32, 1, 2, 0, 2, 3] {
                indices.push(idx + i);
            }
        };

        let faces = &RIGHT_HANDED_Y_UP_CONFIG.faces;
        for (group, face) in buffer.groups.iter().zip(faces.iter()) {
            let normal = face.signed_normal();
            let normal_arr = [normal.x as f32, normal.y as f32, normal.z as f32];
            let face_brightness: f32 = if normal.y > 0 {
                1.0
            } else if normal.y < 0 {
                0.5
            } else {
                0.82
            };

            // Determine which axes span this face group.
            // All quads in a group share the same face direction.
            let (ax_u, ax_v) = if normal.x.abs() > 0 {
                (2, 1) // YZ face
            } else if normal.y.abs() > 0 {
                (0, 2) // XZ face
            } else {
                (0, 1) // XY face
            };

            if !greedy {
                // Baseline path: one quad per visible face, unchanged from original.
                for unit_quad in group.iter() {
                    let quad = block_mesh::UnorientedQuad::from(*unit_quad);
                    let voxel = padded[PaddedChunkShape::linearize(unit_quad.minimum) as usize];
                    let block_idx = voxel.index() as u32;
                    let tile_x = block_idx % tiles_per_row;
                    let tile_y = block_idx / tiles_per_row;
                    let tile_u0 = tile_x as f32 * tile_uv_size;
                    let tile_v0 = tile_y as f32 * tile_uv_size;
                    let mesh_positions = face.quad_mesh_positions(&quad, 1.0);

                    let idx = positions.len() as u32;
                    let mut min_pos = mesh_positions[0];
                    let mut max_pos = mesh_positions[0];
                    for p in &mesh_positions[1..] {
                        for a in 0..3 {
                            if p[a] < min_pos[a] { min_pos[a] = p[a]; }
                            if p[a] > max_pos[a] { max_pos[a] = p[a]; }
                        }
                    }
                    for pos in &mesh_positions {
                        positions.push([pos[0] - 1.0, pos[1] - 1.0, pos[2] - 1.0]);
                        let extent_u = max_pos[ax_u] - min_pos[ax_u];
                        let extent_v = max_pos[ax_v] - min_pos[ax_v];
                        let lu = if extent_u > 0.01 {
                            (pos[ax_u] - min_pos[ax_u]) / extent_u
                        } else { 0.0 };
                        let lv = if extent_v > 0.01 {
                            (pos[ax_v] - min_pos[ax_v]) / extent_v
                        } else { 0.0 };
                        uvs.push([tile_u0 + lu * tile_uv_size, tile_v0 + lv * tile_uv_size]);
                    }
                    for n in face.quad_mesh_normals() { normals.push(n); }
                    for _ in 0..4 {
                        colors.push([face_brightness, face_brightness, face_brightness, 1.0]);
                    }
                    for i in face.quad_mesh_indices(idx) { indices.push(i); }
                }
            } else {
                // Greedy path: merge adjacent same-type quads along U axis only.
                // Collect visible faces into a sparse map keyed by (normal_pos, v),
                // then scan along U to merge consecutive same-type runs.

                // Build a map: (slice_coord, v_coord) -> sorted list of (u_coord, block_type)
                let norm_ax = if normal.x.abs() > 0 { 0usize }
                    else if normal.y.abs() > 0 { 1 } else { 2 };

                // Collect all visible faces for this direction into a grid.
                // Key: (slice_in_normal_axis, v_coord) -> Vec<(u_coord, BlockType)>
                let mut rows: HashMap<(u32, u32), Vec<(u32, BlockType)>> = HashMap::new();

                for unit_quad in group.iter() {
                    let min = unit_quad.minimum;
                    let voxel = padded[PaddedChunkShape::linearize(min) as usize];

                    // Skip emissive/transparent blocks — never merge them.
                    if voxel.is_emissive() || !voxel.is_opaque() {
                        // Emit as a single 1×1 quad.
                        let quad = block_mesh::UnorientedQuad::from(*unit_quad);
                        let block_idx = voxel.index() as u32;
                        let tile_x = block_idx % tiles_per_row;
                        let tile_y = block_idx / tiles_per_row;
                        let mesh_positions = face.quad_mesh_positions(&quad, 1.0);
                        emit_quad(
                            &mut positions, &mut normals, &mut colors, &mut uvs, &mut indices,
                            [mesh_positions[0], mesh_positions[1], mesh_positions[2], mesh_positions[3]],
                            normal_arr, face_brightness,
                            tile_x as f32 * tile_uv_size, tile_y as f32 * tile_uv_size,
                            ax_u, ax_v, 1.0, 1.0,
                        );
                        continue;
                    }

                    let slice_coord = min[norm_ax];
                    let v_coord = min[ax_v];
                    let u_coord = min[ax_u];
                    rows.entry((slice_coord, v_coord))
                        .or_default()
                        .push((u_coord, voxel));
                }

                // Debug: track which faces the greedy merge covers.
                #[cfg(debug_assertions)]
                let mut greedy_covered: HashSet<(u32, u32, u32)> = HashSet::new();
                #[cfg(debug_assertions)]
                let input_face_count = rows.values().map(|r| r.len()).sum::<usize>();

                // For each row, sort by U and merge consecutive same-type runs.
                for ((_slice, _v), mut row) in rows {
                    row.sort_by_key(|&(u, _)| u);

                    let mut i = 0;
                    while i < row.len() {
                        let (start_u, block) = row[i];
                        let mut end_u = start_u;

                        // Extend along U while:
                        //   - next cell is adjacent (no gap)
                        //   - same block type
                        //   - stays within chunk interior (padded 1..16, not into padding)
                        //   - same emissive status (never merge lit + unlit)
                        //   - same opacity (defensive — both should be opaque here)
                        while i + 1 < row.len()
                            && row[i + 1].0 == end_u + 1
                            && row[i + 1].1 == block
                            && end_u + 1 <= 16  // never extend into padding (padded index 17)
                            && row[i + 1].1.is_emissive() == block.is_emissive()
                            && row[i + 1].1.is_opaque() == block.is_opaque()
                        {
                            end_u = row[i + 1].0;
                            i += 1;
                        }
                        i += 1;

                        let w = (end_u - start_u + 1) as f32;

                        // Debug: validate the merged run.
                        #[cfg(debug_assertions)]
                        {
                            for u in start_u..=end_u {
                                // Every cell in the run must be the same type.
                                let mut coord = [0u32; 3];
                                coord[norm_ax] = _slice;
                                coord[ax_u] = u;
                                coord[ax_v] = _v;
                                let voxel = padded[PaddedChunkShape::linearize(coord) as usize];
                                if voxel != block {
                                    bevy::log::warn!(
                                        "Greedy merge type mismatch in chunk ({},{},{}): \
                                         run block={}, found={} at u={}, slice={}, v={}, \
                                         norm_ax={}, run_len={} — disabling greedy meshing",
                                        self.pos.0, self.pos.1, self.pos.2,
                                        block.name(), voxel.name(),
                                        u, _slice, _v, norm_ax, end_u - start_u + 1,
                                    );
                                    GREEDY_INVARIANT_VIOLATED.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                                if voxel == BlockType::AIR {
                                    bevy::log::warn!(
                                        "Greedy quad covers AIR at u={}, slice={}, v={} \
                                         in chunk ({},{},{}) — disabling greedy meshing",
                                        u, _slice, _v,
                                        self.pos.0, self.pos.1, self.pos.2,
                                    );
                                    GREEDY_INVARIANT_VIOLATED.store(true, std::sync::atomic::Ordering::Relaxed);
                                }
                                greedy_covered.insert((_slice, u, _v));
                            }
                        }
                        let block_idx = block.index() as u32;
                        let tile_x = block_idx % tiles_per_row;
                        let tile_y = block_idx / tiles_per_row;
                        let tile_u0 = tile_x as f32 * tile_uv_size;
                        let tile_v0 = tile_y as f32 * tile_uv_size;

                        // Build corner positions in padded space.
                        let mut c0 = [0.0f32; 3];
                        let mut c1 = [0.0f32; 3];
                        let mut c2 = [0.0f32; 3];
                        let mut c3 = [0.0f32; 3];

                        // The face sits at the boundary of the voxel in normal direction.
                        // visible_block_faces places quads at min..min+1, so the
                        // normal-axis position is already correct from min.
                        c0[norm_ax] = _slice as f32;
                        c1[norm_ax] = _slice as f32;
                        c2[norm_ax] = _slice as f32;
                        c3[norm_ax] = _slice as f32;

                        c0[ax_u] = start_u as f32;
                        c0[ax_v] = _v as f32;
                        c1[ax_u] = start_u as f32 + w;
                        c1[ax_v] = _v as f32;
                        c2[ax_u] = start_u as f32 + w;
                        c2[ax_v] = _v as f32 + 1.0;
                        c3[ax_u] = start_u as f32;
                        c3[ax_v] = _v as f32 + 1.0;

                        let greedy_brightness = if highlight_greedy && w > 1.0 {
                            // Merged quad: use cyan tint to distinguish from 1×1.
                            -1.0 // sentinel; handled below
                        } else {
                            face_brightness
                        };
                        emit_quad(
                            &mut positions, &mut normals, &mut colors, &mut uvs, &mut indices,
                            [c0, c1, c2, c3],
                            normal_arr, greedy_brightness,
                            tile_u0, tile_v0,
                            ax_u, ax_v, w, 1.0,
                        );
                        // Override colors for highlighted greedy quads.
                        if highlight_greedy && w > 1.0 {
                            let len = colors.len();
                            for ci in (len - 4)..len {
                                colors[ci] = [0.0, 1.0, 1.0, 1.0]; // cyan
                            }
                        }
                    }
                }

                // Debug: verify every input face was covered by a greedy quad.
                #[cfg(debug_assertions)]
                if greedy_covered.len() != input_face_count {
                    bevy::log::warn!(
                        "Greedy merge coverage mismatch in chunk ({},{},{}): \
                         {} input faces but {} covered by merged quads, \
                         norm_ax={} — {} faces lost — disabling greedy meshing",
                        self.pos.0, self.pos.1, self.pos.2,
                        input_face_count, greedy_covered.len(),
                        norm_ax,
                        input_face_count - greedy_covered.len(),
                    );
                    GREEDY_INVARIANT_VIOLATED.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }

        // Debug: log mesh generation stats for every chunk build.
        #[cfg(debug_assertions)]
        {
            let solid_count = self.blocks.iter().filter(|b| **b != BlockType::AIR).count();
            bevy::log::trace!(
                "build_mesh({},{},{}): blocks={}/{} verts={} indices={}",
                self.pos.0, self.pos.1, self.pos.2,
                solid_count, CHUNK_VOLUME,
                positions.len(), indices.len(),
            );
        }

        // Debug correctness invariants — validated before mesh data is moved.
        #[cfg(debug_assertions)]
        {
            let vert_count = positions.len();
            let has_solid = self.blocks.iter().any(|b| *b != BlockType::AIR);

            // 1. Any chunk with solid blocks must produce a non-empty mesh.
            //    A fully buried chunk is an exception (all neighbors opaque),
            //    so we check for exposed faces below.
            if has_solid && vert_count == 0 {
                // Check if any solid block has a non-opaque neighbor
                let mut has_exposed = false;
                'scan: for z in 0..CHUNK_SIZE {
                    for y in 0..CHUNK_SIZE {
                        for x in 0..CHUNK_SIZE {
                            let block = self.blocks[Self::index(x, y, z)];
                            if block == BlockType::AIR { continue; }
                            let adj: [(i32,i32,i32); 6] = [
                                (-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1),
                            ];
                            for (dx,dy,dz) in adj {
                                let ai = PaddedChunkShape::linearize([
                                    (x+1+dx) as u32, (y+1+dy) as u32, (z+1+dz) as u32,
                                ]);
                                if !padded[ai as usize].is_opaque() {
                                    has_exposed = true;
                                    break 'scan;
                                }
                            }
                        }
                    }
                }
                if has_exposed {
                    // Count opaque blocks on each boundary face of the padded buffer
                    // to diagnose which face direction is failing.
                    let face_names = ["-X", "+X", "-Y", "+Y", "-Z", "+Z"];
                    let neighbor_present: [bool; 6] = [
                        neighbors.neighbors[0].is_some(),
                        neighbors.neighbors[1].is_some(),
                        neighbors.neighbors[2].is_some(),
                        neighbors.neighbors[3].is_some(),
                        neighbors.neighbors[4].is_some(),
                        neighbors.neighbors[5].is_some(),
                    ];
                    // Count how many padding cells are opaque per face.
                    let mut opaque_counts = [0u32; 6];
                    for y in 0..CHUNK_SIZE {
                        for z in 0..CHUNK_SIZE {
                            // -X (padded x=0)
                            if padded[PaddedChunkShape::linearize([0, (y+1) as u32, (z+1) as u32]) as usize].is_opaque() { opaque_counts[0] += 1; }
                            // +X (padded x=17)
                            if padded[PaddedChunkShape::linearize([17, (y+1) as u32, (z+1) as u32]) as usize].is_opaque() { opaque_counts[1] += 1; }
                        }
                    }
                    for x in 0..CHUNK_SIZE {
                        for z in 0..CHUNK_SIZE {
                            // -Y (padded y=0)
                            if padded[PaddedChunkShape::linearize([(x+1) as u32, 0, (z+1) as u32]) as usize].is_opaque() { opaque_counts[2] += 1; }
                            // +Y (padded y=17)
                            if padded[PaddedChunkShape::linearize([(x+1) as u32, 17, (z+1) as u32]) as usize].is_opaque() { opaque_counts[3] += 1; }
                        }
                    }
                    for x in 0..CHUNK_SIZE {
                        for y in 0..CHUNK_SIZE {
                            // -Z (padded z=0)
                            if padded[PaddedChunkShape::linearize([(x+1) as u32, (y+1) as u32, 0]) as usize].is_opaque() { opaque_counts[4] += 1; }
                            // +Z (padded z=17)
                            if padded[PaddedChunkShape::linearize([(x+1) as u32, (y+1) as u32, 17]) as usize].is_opaque() { opaque_counts[5] += 1; }
                        }
                    }

                    bevy::log::warn!(
                        "Chunk ({},{},{}) has exposed solid blocks but produced \
                         0 vertices — missing geometry. Neighbors: \
                         {}={}/{} {}={}/{} {}={}/{} {}={}/{} {}={}/{} {}={}/{}",
                        self.pos.0, self.pos.1, self.pos.2,
                        face_names[0], neighbor_present[0], opaque_counts[0],
                        face_names[1], neighbor_present[1], opaque_counts[1],
                        face_names[2], neighbor_present[2], opaque_counts[2],
                        face_names[3], neighbor_present[3], opaque_counts[3],
                        face_names[4], neighbor_present[4], opaque_counts[4],
                        face_names[5], neighbor_present[5], opaque_counts[5],
                    );
                }
            }

            // 2. All UVs must be finite and non-negative.
            for (i, uv) in uvs.iter().enumerate() {
                if !uv[0].is_finite() || !uv[1].is_finite()
                    || uv[0] < 0.0 || uv[1] < 0.0
                {
                    bevy::log::warn!(
                        "Invalid UV at vertex {} in chunk ({},{},{}): ({:.4},{:.4})",
                        i, self.pos.0, self.pos.1, self.pos.2, uv[0], uv[1],
                    );
                    break; // log once per chunk, not per vertex
                }
            }

            // 3. Attribute array lengths must match (no vertex missing a UV/normal/color).
            debug_assert_eq!(vert_count, uvs.len(),
                "Chunk ({},{},{}): {} positions vs {} UVs",
                self.pos.0, self.pos.1, self.pos.2, vert_count, uvs.len());
            debug_assert_eq!(vert_count, normals.len(),
                "Chunk ({},{},{}): {} positions vs {} normals",
                self.pos.0, self.pos.1, self.pos.2, vert_count, normals.len());
            debug_assert_eq!(vert_count, colors.len(),
                "Chunk ({},{},{}): {} positions vs {} colors",
                self.pos.0, self.pos.1, self.pos.2, vert_count, colors.len());
        }

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_indices(Indices::U32(indices));
        mesh
    }
}
