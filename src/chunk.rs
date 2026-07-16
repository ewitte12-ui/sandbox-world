use bevy::{
    asset::RenderAssetUsages, mesh::Indices, prelude::*, render::render_resource::PrimitiveTopology,
};
use block_mesh::{
    ndshape::{ConstShape, ConstShape3u32},
    visible_block_faces, UnitQuadBuffer, RIGHT_HANDED_Y_UP_CONFIG,
};

use std::collections::{HashMap, HashSet};

use crate::block_types::BlockType;
use crate::terrain::place_trees_in_chunk;

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
    /// Packed voxel light (sky high nibble, block low nibble — see
    /// voxel_light). Zeroed at generation; computed on the main thread
    /// when the chunk is inserted (handle_chunk_tasks) and kept current
    /// by process_light_queue. INVARIANT: every chunk in
    /// ChunkManager::chunk_data has had its light computed before its
    /// first mesh build.
    pub light: [u8; CHUNK_VOLUME],
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
        Self::generate_tracked(pos).0
    }

    /// Generate a chunk and report which of its boundary layers contain
    /// non-terrain content (tree blocks), in ChunkNeighbors face order
    /// (-X, +X, -Y, +Y, -Z, +Z). Neighbors already meshed against
    /// terrain-predicted padding (see build_mesh) only need a remesh when
    /// the corresponding bit is set — this is what keeps the neighbor-load
    /// remesh pass from becoming a remesh-everything storm.
    pub fn generate_tracked(pos: ChunkPos) -> (Self, [bool; 6]) {
        let mut blocks = [BlockType::AIR; CHUNK_VOLUME];
        let base_x = pos.0 * CHUNK_SIZE;
        let base_y = pos.1 * CHUNK_SIZE;
        let base_z = pos.2 * CHUNK_SIZE;

        // Surface height depends only on the (x, z) column — compute each
        // column once instead of once per block (16× fewer FBM evaluations
        // per chunk). Produces byte-identical terrain to the per-block path.
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let wx = base_x + x;
                let wz = base_z + z;
                let sy = crate::terrain::surface_y(wx, wz);
                for y in 0..CHUNK_SIZE {
                    let wy = base_y + y;
                    blocks[Self::index(x, y, z)] =
                        crate::terrain::natural_block_at_with_surface(wx, wy, wz, sy);
                }
            }
        }

        let tree_mask = place_trees_in_chunk(&mut blocks, pos.0, pos.1, pos.2);
        // Buildings AFTER trees: legacy building modifications overrode
        // generated chunk content, so buildings must win over tree blocks.
        let building_mask =
            crate::buildings::place_buildings_in_chunk(&mut blocks, pos.0, pos.1, pos.2);
        let boundary_mask: [bool; 6] =
            std::array::from_fn(|i| tree_mask[i] || building_mask[i]);

        (Chunk { blocks, light: [0; CHUNK_VOLUME], pos }, boundary_mask)
    }

    /// Build a Bevy Mesh from this chunk's block data.
    ///
    /// Uses block_mesh's `visible_block_faces` for per-block face culling:
    /// a face is emitted only if the adjacent block is Air. This is the
    /// occlusion contract — fully interior blocks produce zero geometry.
    /// The padded 18³ buffer includes 1 block of neighbor data on each
    /// side so that faces at chunk boundaries are correctly culled against
    /// the neighboring chunk's blocks. Unloaded neighbors are predicted
    /// from deterministic terrain (see the shell pre-fill below) rather
    /// than treated as Air.
    ///
    /// RENDERING NOTE: all vertex positions are in float chunk-local space
    /// (0.0..16.0). The chunk's world offset is applied via Transform in
    /// handle_chunk_tasks. UV_0 is tile-local (0..1 per block face, 0..w
    /// along greedy-merged runs — sampled with a repeating sampler);
    /// UV_1.x carries the block's texture-array layer index, constant per
    /// quad (see chunk_manager::BlockArrayExtension). UV_1.y carries the
    /// vertex SKY light (0..1) and COLOR alpha the vertex BLOCK light
    /// (0..1) — combined in the shader with the live sun-intensity
    /// uniform so day/night needs no remesh. COLOR rgb carries face
    /// brightness × baked ambient occlusion (vertex_ao / ao_strength).
    /// `shell_light`: pass the ShellLight already gathered for this
    /// chunk's light recompute to avoid a second ~324-`surface_y` gather;
    /// None gathers one internally (remesh paths where light is current).
    pub fn build_mesh(
        &self,
        neighbors: &ChunkNeighbors,
        greedy: bool,
        highlight_greedy: bool,
        ao_strength: f32,
        shell_light: Option<&crate::voxel_light::ShellLight>,
    ) -> Mesh {
        let ao_strength = ao_strength.clamp(0.0, 1.0);
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

        // Pre-fill the ENTIRE padding shell (6 face slabs + 12 edges +
        // 8 corners) from deterministic terrain. Two reasons:
        //
        //  - Face slabs: an unloaded neighbor used to read as AIR, so a
        //    buried chunk emitted a full 16×16 wall of boundary quads on
        //    every unloaded side — geometry that was never visible but
        //    persisted forever (nothing remeshed on neighbor load). With
        //    terrain prediction the walls are culled correctly from the
        //    first mesh. Real neighbor data (below) overwrites these cells
        //    when available; divergence from raw terrain (trees, player
        //    edits) is reconciled by the boundary-mask remesh pass in
        //    handle_chunk_tasks.
        //
        //  - Edge/corner cells: never covered by the 6 face neighbors, but
        //    vertex AO samples diagonal blocks. Terrain data here keeps AO
        //    seam-free across chunk borders on hillsides. (Player edits and
        //    trees exactly on a diagonal boundary can still produce a
        //    subtly-off AO corner — cosmetic and rare.)
        //
        // Cost: one surface_y per padded column (cached below) plus one
        // block classification per shell cell (~1.8k cells).
        {
            let base_x = self.pos.0 * CHUNK_SIZE;
            let base_y = self.pos.1 * CHUNK_SIZE;
            let base_z = self.pos.2 * CHUNK_SIZE;
            // Lazily computed per-column surface heights (i32::MIN = unset).
            let mut col_surface = [[i32::MIN; 18]; 18];
            fn surface_at(
                cache: &mut [[i32; 18]; 18],
                base_x: i32,
                base_z: i32,
                px: usize,
                pz: usize,
            ) -> i32 {
                if cache[pz][px] == i32::MIN {
                    cache[pz][px] = crate::terrain::surface_y(
                        base_x + px as i32 - 1,
                        base_z + pz as i32 - 1,
                    );
                }
                cache[pz][px]
            }
            let have_neighbor = [
                neighbors.neighbors[0].is_some(),
                neighbors.neighbors[1].is_some(),
                neighbors.neighbors[2].is_some(),
                neighbors.neighbors[3].is_some(),
                neighbors.neighbors[4].is_some(),
                neighbors.neighbors[5].is_some(),
            ];
            for pz in 0..18u32 {
                for py in 0..18u32 {
                    for px in 0..18u32 {
                        let sx = px == 0 || px == 17;
                        let sy = py == 0 || py == 17;
                        let sz = pz == 0 || pz == 17;
                        let shell_axes = sx as u8 + sy as u8 + sz as u8;
                        if shell_axes == 0 {
                            continue; // interior
                        }
                        // Face-interior cells (exactly one shell axis) of a
                        // LOADED neighbor's slab are fully overwritten by
                        // real data below — skip the terrain prediction.
                        // Edge/corner cells (≥2 shell axes) are never
                        // covered by face neighbors and always need it
                        // (AO samples diagonals).
                        if shell_axes == 1 {
                            let face = if px == 0 { 0 }
                                else if px == 17 { 1 }
                                else if py == 0 { 2 }
                                else if py == 17 { 3 }
                                else if pz == 0 { 4 }
                                else { 5 };
                            if have_neighbor[face] {
                                continue;
                            }
                        }
                        let sy_col =
                            surface_at(&mut col_surface, base_x, base_z, px as usize, pz as usize);
                        let pi = PaddedChunkShape::linearize([px, py, pz]);
                        padded[pi as usize] = crate::terrain::natural_block_at_with_surface(
                            base_x + px as i32 - 1,
                            base_y + py as i32 - 1,
                            base_z + pz as i32 - 1,
                            sy_col,
                        );
                    }
                }
            }
        }

        // Fill padding from neighbors (overwrites the terrain prediction
        // with real data — including player modifications and trees).
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

        // Padded voxel-light grids, mirroring the block padding: interior
        // from this chunk's computed light, shell from neighbor light or
        // terrain-predicted sky (voxel_light::gather_shell_light — the
        // same prediction rule the light BFS itself uses, so mesh
        // sampling and light computation always agree at seams).
        let owned_shell;
        let shell_light = match shell_light {
            Some(shell) => shell,
            None => {
                owned_shell = crate::voxel_light::gather_shell_light(self.pos, neighbors);
                &owned_shell
            }
        };
        let mut padded_sky = shell_light.sky;
        let mut padded_blk = shell_light.blk;
        for z in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let l = self.light[Self::index(x, y, z)];
                    let pi = crate::voxel_light::pidx(
                        (x + 1) as usize,
                        (y + 1) as usize,
                        (z + 1) as usize,
                    );
                    padded_sky[pi] = crate::voxel_light::sky(l);
                    padded_blk[pi] = crate::voxel_light::blk(l);
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

        // Per-block visible-face collection. The greedy branch below merges
        // same-type/same-AO runs of these unit faces into wider quads.
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
        // UV_1.x = texture-array layer (block type index), constant per quad.
        let mut layer_uvs: Vec<[f32; 2]> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        // --- Vertex ambient occlusion -----------------------------------
        // Classic voxel AO ("0fps" formulation): for each face vertex,
        // sample the two edge-adjacent blocks and the corner block on the
        // face's normal side. Level 3 = fully open, 0 = pinched corner.
        // Runs against the padded buffer, whose shell is always populated
        // (real neighbor data or terrain prediction), so terrain AO is
        // seam-free across chunk borders.
        let ao_curve = |level: u8| -> f32 {
            // Brightness multiplier per occlusion level, blended toward
            // 1.0 (no AO) by DevSettings.ao_strength.
            const AO_LEVELS: [f32; 4] = [0.55, 0.72, 0.86, 1.0];
            1.0 + (AO_LEVELS[level as usize] - 1.0) * ao_strength
        };

        // `voxel` is the padded-space coord of the solid block owning the
        // face; `cu`/`cv` select the face corner (false = min side).
        // Samples one block outside the padded buffer resolve as open —
        // that's the 1-block-padding limit; it can only over-brighten a
        // diagonal at a chunk border, never darken.
        let vertex_ao = |voxel: [u32; 3],
                         norm_ax: usize,
                         norm_sign: i32,
                         ax_u: usize,
                         ax_v: usize,
                         cu: bool,
                         cv: bool|
         -> u8 {
            let mut base = [voxel[0] as i32, voxel[1] as i32, voxel[2] as i32];
            base[norm_ax] += norm_sign;
            let du: i32 = if cu { 1 } else { -1 };
            let dv: i32 = if cv { 1 } else { -1 };
            let sample = |p: [i32; 3]| -> bool {
                if p.iter().any(|c| !(0..18).contains(c)) {
                    return false;
                }
                let pi = PaddedChunkShape::linearize([p[0] as u32, p[1] as u32, p[2] as u32]);
                padded[pi as usize].is_opaque()
            };
            let mut s1 = base;
            s1[ax_u] += du;
            let mut s2 = base;
            s2[ax_v] += dv;
            let mut corner = base;
            corner[ax_u] += du;
            corner[ax_v] += dv;
            let (side1, side2, corner) = (sample(s1), sample(s2), sample(corner));
            if side1 && side2 {
                0
            } else {
                3 - (side1 as u8 + side2 as u8 + corner as u8)
            }
        };

        // --- Vertex voxel light (smooth lighting) ------------------------
        // Same sample points as vertex_ao: the face-front cell plus the
        // two edge and one corner cell on the air side of the vertex.
        // Averages the sky/block light of the non-opaque samples (the
        // corner is unreachable when both edges are opaque — matching the
        // AO pinch rule), quantized to u8 (0..=255 ≙ light 0..=15) so
        // greedy merging can compare runs exactly.
        let light_of = |p: [i32; 3]| -> Option<(u8, u8)> {
            if p.iter().any(|c| !(0..18).contains(c)) {
                return None;
            }
            let bi = PaddedChunkShape::linearize([p[0] as u32, p[1] as u32, p[2] as u32]);
            if padded[bi as usize].is_opaque() {
                return None;
            }
            let pi = crate::voxel_light::pidx(p[0] as usize, p[1] as usize, p[2] as usize);
            Some((padded_sky[pi], padded_blk[pi]))
        };
        let vertex_light = |voxel: [u32; 3],
                            norm_ax: usize,
                            norm_sign: i32,
                            ax_u: usize,
                            ax_v: usize,
                            cu: bool,
                            cv: bool|
         -> [u8; 2] {
            let mut base = [voxel[0] as i32, voxel[1] as i32, voxel[2] as i32];
            base[norm_ax] += norm_sign;
            let du: i32 = if cu { 1 } else { -1 };
            let dv: i32 = if cv { 1 } else { -1 };
            let mut s1 = base;
            s1[ax_u] += du;
            let mut s2 = base;
            s2[ax_v] += dv;
            let mut corner = base;
            corner[ax_u] += du;
            corner[ax_v] += dv;

            let (mut sum_s, mut sum_b, mut n) = (0u32, 0u32, 0u32);
            let mut add = |c: Option<(u8, u8)>| {
                if let Some((s, b)) = c {
                    sum_s += s as u32;
                    sum_b += b as u32;
                    n += 1;
                }
            };
            // The face-front cell is non-opaque by face visibility, so
            // n >= 1 whenever the face exists.
            add(light_of(base));
            let l1 = light_of(s1);
            let l2 = light_of(s2);
            add(l1);
            add(l2);
            if l1.is_some() || l2.is_some() {
                add(light_of(corner));
            }
            if n == 0 {
                return [0, 0];
            }
            let q = |sum: u32| -> u8 {
                ((sum as f32 / n as f32) / 15.0 * 255.0).round().clamp(0.0, 255.0) as u8
            };
            [q(sum_s), q(sum_b)]
        };

        // Helper: emit a single quad into the mesh buffers.
        // `corners` are 4 face corners in padded space, in ANY order — the
        // helper classifies them into ring order on (ax_u, ax_v), fixes
        // winding against `normal`, and picks the triangulation diagonal
        // from per-corner AO (split along the DARKER diagonal so the
        // crease passes through the dark corner instead of an X artifact).
        // `ao` holds final brightness multipliers aligned with `corners`;
        // `light` holds quantized (sky, block) voxel light per corner —
        // emitted as UV_1.y (sky) and COLOR alpha (block), both 0..1, so
        // the fragment shader can combine them with the live sun uniform.
        let mut emit_quad = |
            positions: &mut Vec<[f32; 3]>,
            normals: &mut Vec<[f32; 3]>,
            colors: &mut Vec<[f32; 4]>,
            uvs: &mut Vec<[f32; 2]>,
            indices: &mut Vec<u32>,
            corners: [[f32; 3]; 4],
            ao: [f32; 4],
            light: [[u8; 2]; 4],
            normal: [f32; 3],
            face_brightness: f32,
            layer: f32,
            ax_u: usize,
            ax_v: usize,
        | {
            let idx = positions.len() as u32;
            let mut min_u = f32::INFINITY;
            let mut max_u = f32::NEG_INFINITY;
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for c in &corners {
                min_u = min_u.min(c[ax_u]);
                max_u = max_u.max(c[ax_u]);
                min_v = min_v.min(c[ax_v]);
                max_v = max_v.max(c[ax_v]);
            }
            for ((c, a), l) in corners.iter().zip(ao.iter()).zip(light.iter()) {
                positions.push([c[0] - 1.0, c[1] - 1.0, c[2] - 1.0]);
                // Tile-local UVs: 0..1 per block cell, 0..w across a greedy
                // run. The repeating sampler tiles the block's array layer
                // across merged quads — no atlas offset, no inset.
                let lu = c[ax_u] - min_u;
                let lv = c[ax_v] - min_v;
                uvs.push([lu, lv]);
                layer_uvs.push([layer, l[0] as f32 / 255.0]);
                normals.push(normal);
                let b = face_brightness * a;
                colors.push([b, b, b, l[1] as f32 / 255.0]);
            }
            // Ring classification: which pushed corner sits at each
            // (u, v) extreme — ring order (0,0), (1,0), (1,1), (0,1).
            let mid_u = (min_u + max_u) * 0.5;
            let mid_v = (min_v + max_v) * 0.5;
            let mut ring = [0usize; 4];
            for (i, c) in corners.iter().enumerate() {
                let slot = match (c[ax_u] > mid_u, c[ax_v] > mid_v) {
                    (false, false) => 0,
                    (true, false) => 1,
                    (true, true) => 2,
                    (false, true) => 3,
                };
                ring[slot] = i;
            }
            // Winding: if the ring's geometric normal opposes the face
            // normal, swap to keep front faces outward (backface culling).
            let e01 = [
                corners[ring[1]][0] - corners[ring[0]][0],
                corners[ring[1]][1] - corners[ring[0]][1],
                corners[ring[1]][2] - corners[ring[0]][2],
            ];
            let e03 = [
                corners[ring[3]][0] - corners[ring[0]][0],
                corners[ring[3]][1] - corners[ring[0]][1],
                corners[ring[3]][2] - corners[ring[0]][2],
            ];
            let cross = [
                e01[1] * e03[2] - e01[2] * e03[1],
                e01[2] * e03[0] - e01[0] * e03[2],
                e01[0] * e03[1] - e01[1] * e03[0],
            ];
            if cross[0] * normal[0] + cross[1] * normal[1] + cross[2] * normal[2] < 0.0 {
                ring.swap(1, 3);
            }
            // AO diagonal flip: split along the darker (lower-sum) diagonal.
            let tris = if ao[ring[0]] + ao[ring[2]] > ao[ring[1]] + ao[ring[3]] {
                [ring[1], ring[2], ring[3], ring[1], ring[3], ring[0]]
            } else {
                [ring[0], ring[1], ring[2], ring[0], ring[2], ring[3]]
            };
            for t in tris {
                indices.push(idx + t as u32);
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

            let norm_sign = (normal.x + normal.y + normal.z).signum();
            let norm_ax = if normal.x.abs() > 0 { 0usize }
                else if normal.y.abs() > 0 { 1 } else { 2 };

            if !greedy {
                // Baseline path: one quad per visible face.
                for unit_quad in group.iter() {
                    let quad = block_mesh::UnorientedQuad::from(*unit_quad);
                    let voxel = padded[PaddedChunkShape::linearize(unit_quad.minimum) as usize];
                    let layer = voxel.index() as f32;
                    let mesh_positions = face.quad_mesh_positions(&quad, 1.0);

                    let mut min_pos = mesh_positions[0];
                    let mut max_pos = mesh_positions[0];
                    for p in &mesh_positions[1..] {
                        for a in 0..3 {
                            if p[a] < min_pos[a] { min_pos[a] = p[a]; }
                            if p[a] > max_pos[a] { max_pos[a] = p[a]; }
                        }
                    }
                    // Per-corner AO + voxel light, aligned with
                    // mesh_positions order.
                    let mut ao = [1.0f32; 4];
                    let mut light = [[0u8; 2]; 4];
                    for (i, p) in mesh_positions.iter().enumerate() {
                        let cu = p[ax_u] > (min_pos[ax_u] + max_pos[ax_u]) * 0.5;
                        let cv = p[ax_v] > (min_pos[ax_v] + max_pos[ax_v]) * 0.5;
                        ao[i] = ao_curve(vertex_ao(
                            unit_quad.minimum, norm_ax, norm_sign, ax_u, ax_v, cu, cv,
                        ));
                        light[i] = vertex_light(
                            unit_quad.minimum, norm_ax, norm_sign, ax_u, ax_v, cu, cv,
                        );
                    }
                    emit_quad(
                        &mut positions, &mut normals, &mut colors, &mut uvs, &mut indices,
                        [mesh_positions[0], mesh_positions[1], mesh_positions[2], mesh_positions[3]],
                        ao, light, normal_arr, face_brightness,
                        layer, ax_u, ax_v,
                    );
                }
            } else {
                // Greedy path: merge adjacent same-type quads along U axis only.
                // Collect visible faces into a sparse map keyed by (normal_pos, v),
                // then scan along U to merge consecutive same-type runs.
                //
                // Faces also carry their 4-corner AO and voxel-light
                // tuples (ring order) — two faces only merge when BOTH
                // tuples are identical, so the merged quad's interpolated
                // AO/light is exactly what the individual quads would have
                // shown. Uniform open terrain (AO = 3, sky = 15
                // everywhere) merges as before; runs stop at AO or light
                // transitions like wall bases, ledges, and lantern glow.
                type FaceRun = (u32, BlockType, [u8; 4], [[u8; 2]; 4]);
                let mut rows: HashMap<(u32, u32), Vec<FaceRun>> = HashMap::new();

                for unit_quad in group.iter() {
                    let min = unit_quad.minimum;
                    let voxel = padded[PaddedChunkShape::linearize(min) as usize];

                    // Ring-order AO for this face: (0,0), (1,0), (1,1), (0,1).
                    let ao_t = [
                        vertex_ao(min, norm_ax, norm_sign, ax_u, ax_v, false, false),
                        vertex_ao(min, norm_ax, norm_sign, ax_u, ax_v, true, false),
                        vertex_ao(min, norm_ax, norm_sign, ax_u, ax_v, true, true),
                        vertex_ao(min, norm_ax, norm_sign, ax_u, ax_v, false, true),
                    ];
                    // Ring-order voxel light, same corner order as ao_t.
                    let light_t = [
                        vertex_light(min, norm_ax, norm_sign, ax_u, ax_v, false, false),
                        vertex_light(min, norm_ax, norm_sign, ax_u, ax_v, true, false),
                        vertex_light(min, norm_ax, norm_sign, ax_u, ax_v, true, true),
                        vertex_light(min, norm_ax, norm_sign, ax_u, ax_v, false, true),
                    ];

                    // Skip emissive/transparent blocks — never merge them.
                    if voxel.is_emissive() || !voxel.is_opaque() {
                        // Emit as a single 1×1 quad.
                        let quad = block_mesh::UnorientedQuad::from(*unit_quad);
                        let mesh_positions = face.quad_mesh_positions(&quad, 1.0);
                        // Classify corners against the face midpoint so the
                        // AO values align with block_mesh's vertex order.
                        let mut min_p = mesh_positions[0];
                        let mut max_p = mesh_positions[0];
                        for p in &mesh_positions[1..] {
                            for a in 0..3 {
                                if p[a] < min_p[a] { min_p[a] = p[a]; }
                                if p[a] > max_p[a] { max_p[a] = p[a]; }
                            }
                        }
                        let mut ao = [1.0f32; 4];
                        let mut light = [[0u8; 2]; 4];
                        for (i, p) in mesh_positions.iter().enumerate() {
                            let cu = p[ax_u] > (min_p[ax_u] + max_p[ax_u]) * 0.5;
                            let cv = p[ax_v] > (min_p[ax_v] + max_p[ax_v]) * 0.5;
                            let ring_slot = match (cu, cv) {
                                (false, false) => 0,
                                (true, false) => 1,
                                (true, true) => 2,
                                (false, true) => 3,
                            };
                            ao[i] = ao_curve(ao_t[ring_slot]);
                            light[i] = light_t[ring_slot];
                        }
                        emit_quad(
                            &mut positions, &mut normals, &mut colors, &mut uvs, &mut indices,
                            [mesh_positions[0], mesh_positions[1], mesh_positions[2], mesh_positions[3]],
                            ao, light, normal_arr, face_brightness,
                            voxel.index() as f32,
                            ax_u, ax_v,
                        );
                        continue;
                    }

                    let slice_coord = min[norm_ax];
                    let v_coord = min[ax_v];
                    let u_coord = min[ax_u];
                    rows.entry((slice_coord, v_coord))
                        .or_default()
                        .push((u_coord, voxel, ao_t, light_t));
                }

                // Debug: track which faces the greedy merge covers.
                #[cfg(debug_assertions)]
                let mut greedy_covered: HashSet<(u32, u32, u32)> = HashSet::new();
                #[cfg(debug_assertions)]
                let input_face_count = rows.values().map(|r| r.len()).sum::<usize>();

                // For each row, sort by U and merge consecutive same-type runs.
                for ((_slice, _v), mut row) in rows {
                    row.sort_by_key(|&(u, _, _, _)| u);

                    let mut i = 0;
                    while i < row.len() {
                        let (start_u, block, run_ao, run_light) = row[i];
                        let mut end_u = start_u;

                        // Extend along U while:
                        //   - next cell is adjacent (no gap)
                        //   - same block type
                        //   - stays within chunk interior (padded 1..16, not into padding)
                        //   - same emissive status (never merge lit + unlit)
                        //   - same opacity (defensive — both should be opaque here)
                        //   - identical AO and voxel-light tuples (constant along
                        //     the run, so the merged quad interpolates exactly)
                        while i + 1 < row.len()
                            && row[i + 1].0 == end_u + 1
                            && row[i + 1].1 == block
                            && end_u + 1 <= 16  // never extend into padding (padded index 17)
                            && row[i + 1].1.is_emissive() == block.is_emissive()
                            && row[i + 1].1.is_opaque() == block.is_opaque()
                            && row[i + 1].2 == run_ao
                            && row[i + 1].3 == run_light
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
                        // Corners c0..c3 are built in ring order, matching
                        // run_ao/run_light's ring order — and the tuples are
                        // constant along the run, so endpoint values are exact.
                        let ao = [
                            ao_curve(run_ao[0]),
                            ao_curve(run_ao[1]),
                            ao_curve(run_ao[2]),
                            ao_curve(run_ao[3]),
                        ];
                        emit_quad(
                            &mut positions, &mut normals, &mut colors, &mut uvs, &mut indices,
                            [c0, c1, c2, c3],
                            ao, run_light, normal_arr, greedy_brightness,
                            block.index() as f32,
                            ax_u, ax_v,
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
            debug_assert_eq!(vert_count, layer_uvs.len(),
                "Chunk ({},{},{}): {} positions vs {} layer UVs",
                self.pos.0, self.pos.1, self.pos.2, vert_count, layer_uvs.len());
        }

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_1, layer_uvs);
        mesh.insert_indices(Indices::U32(indices));
        mesh
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::{natural_block_at, natural_block_at_with_surface, surface_y};

    const NO_NEIGHBORS: ChunkNeighbors = ChunkNeighbors { neighbors: [None; 6] };

    fn mesh_stats(mesh: &Mesh) -> (usize, usize) {
        let verts = mesh.count_vertices();
        let indices = mesh.indices().map(|i| i.len()).unwrap_or(0);
        (verts, indices)
    }

    /// A fully buried chunk with NO loaded neighbors must produce zero
    /// geometry: the terrain-predicted padding shell culls the boundary
    /// walls. Regression test for the hidden-wall bug (each buried chunk
    /// used to emit up to 6×256 invisible quads that persisted forever).
    #[test]
    fn buried_chunk_produces_no_geometry() {
        // Chunk y=-3 spans blocks -48..-33; surface is 0..10, so every
        // block is >= 33 deep — solid rock, surrounded by solid rock.
        let chunk = Chunk::generate(ChunkPos(0, -3, 0));
        assert!(chunk.blocks.iter().all(|b| *b != BlockType::AIR));
        let mesh = chunk.build_mesh(&NO_NEIGHBORS, false, false, 1.0, None);
        let (verts, _) = mesh_stats(&mesh);
        assert_eq!(verts, 0, "buried chunk emitted {} hidden-wall vertices", verts);
    }

    /// An all-air sky chunk produces zero geometry.
    #[test]
    fn sky_chunk_produces_no_geometry() {
        let chunk = Chunk::generate(ChunkPos(0, 10, 0));
        assert!(chunk.blocks.iter().all(|b| *b == BlockType::AIR));
        let (verts, _) = mesh_stats(&chunk.build_mesh(&NO_NEIGHBORS, false, false, 1.0, None));
        assert_eq!(verts, 0);
    }

    /// A surface chunk produces valid, well-formed geometry in both the
    /// naive and greedy paths: attribute arrays line up, index count is a
    /// multiple of 3, and every index is in range.
    #[test]
    fn surface_chunk_mesh_is_well_formed() {
        let chunk = Chunk::generate(ChunkPos(0, 0, 0));
        for greedy in [false, true] {
            let mesh = chunk.build_mesh(&NO_NEIGHBORS, greedy, false, 1.0, None);
            let (verts, index_count) = mesh_stats(&mesh);
            assert!(verts > 0, "surface chunk produced no geometry (greedy={greedy})");
            assert_eq!(index_count % 3, 0);
            if let Some(indices) = mesh.indices() {
                assert!(
                    indices.iter().all(|i| i < verts),
                    "index out of range (greedy={greedy})"
                );
            }
            let uvs = mesh.attribute(Mesh::ATTRIBUTE_UV_0).unwrap().len();
            let normals = mesh.attribute(Mesh::ATTRIBUTE_NORMAL).unwrap().len();
            let colors = mesh.attribute(Mesh::ATTRIBUTE_COLOR).unwrap().len();
            assert_eq!(verts, uvs);
            assert_eq!(verts, normals);
            assert_eq!(verts, colors);
        }
    }

    /// Greedy meshing must never produce MORE vertices than the naive path.
    #[test]
    fn greedy_never_exceeds_naive() {
        let chunk = Chunk::generate(ChunkPos(3, 0, -2));
        let naive = chunk.build_mesh(&NO_NEIGHBORS, false, false, 1.0, None).count_vertices();
        let greedy = chunk.build_mesh(&NO_NEIGHBORS, true, false, 1.0, None).count_vertices();
        assert!(greedy <= naive, "greedy {} > naive {}", greedy, naive);
    }

    /// ao_strength = 0 must disable all AO darkening: every color channel
    /// equals plain face brightness (1.0 / 0.82 / 0.5 shades only).
    #[test]
    fn ao_strength_zero_matches_flat_shading() {
        let chunk = Chunk::generate(ChunkPos(0, 0, 0));
        let mesh = chunk.build_mesh(&NO_NEIGHBORS, false, false, 0.0, None);
        let Some(bevy::mesh::VertexAttributeValues::Float32x4(colors)) =
            mesh.attribute(Mesh::ATTRIBUTE_COLOR)
        else {
            panic!("missing color attribute");
        };
        for c in colors {
            let is_flat_shade = (c[0] - 1.0).abs() < 1e-5
                || (c[0] - 0.82).abs() < 1e-5
                || (c[0] - 0.5).abs() < 1e-5;
            assert!(is_flat_shade, "unexpected color {:?} with ao_strength=0", c);
        }
    }

    /// The column-cached terrain path must be byte-identical to the
    /// original per-block path (save compatibility).
    #[test]
    fn cached_surface_terrain_is_identical() {
        for x in -20..20 {
            for z in -20..20 {
                let sy = surface_y(x, z);
                for y in (sy - 4)..(sy + 3) {
                    assert_eq!(
                        natural_block_at(x, y, z),
                        natural_block_at_with_surface(x, y, z, sy),
                        "divergence at ({x},{y},{z})"
                    );
                }
            }
        }
    }

    /// Buildings must be baked into generated chunks exactly like the
    /// legacy modification-based placement: WOOD walls, STONE roof, a
    /// 4-wide door opening, interior left as terrain.
    #[test]
    fn building_is_baked_into_chunks() {
        use crate::terrain::surface_y;
        // Spot (50, 30): origin corner (45, 25), base at surface+1.
        let base_y = surface_y(50, 30) + 1;
        let (bx0, bz0) = (45, 25);

        let block_at_world = |wx: i32, wy: i32, wz: i32| -> BlockType {
            let cp = ChunkPos(
                wx.div_euclid(CHUNK_SIZE),
                wy.div_euclid(CHUNK_SIZE),
                wz.div_euclid(CHUNK_SIZE),
            );
            let chunk = Chunk::generate(cp);
            chunk.get_block(
                wx.rem_euclid(CHUNK_SIZE),
                wy.rem_euclid(CHUNK_SIZE),
                wz.rem_euclid(CHUNK_SIZE),
            )
        };

        // Roof (dy=6) is stone everywhere on the footprint.
        assert_eq!(block_at_world(bx0, base_y + 6, bz0), BlockType::STONE);
        assert_eq!(block_at_world(bx0 + 9, base_y + 6, bz0 + 9), BlockType::STONE);
        // Wall corners are wood below the roof.
        assert_eq!(block_at_world(bx0, base_y, bz0), BlockType::WOOD);
        assert_eq!(block_at_world(bx0 + 9, base_y + 2, bz0 + 9), BlockType::WOOD);
        // Door opening: dz=0, dx in 3..=6, dy<5 — must NOT be wall material.
        for dx in 3..=6 {
            let b = block_at_world(bx0 + dx, base_y + 1, bz0);
            assert_ne!(b, BlockType::WOOD, "door blocked at dx={dx}");
            assert_ne!(b, BlockType::STONE, "door blocked at dx={dx}");
        }
        // The building spans chunk x=2/x=3 (blocks 45..54 cross x=48):
        // both sides must report a boundary-divergent face there.
        let cy = base_y.div_euclid(CHUNK_SIZE);
        let cz = bz0.div_euclid(CHUNK_SIZE);
        let (_, mask_left) = Chunk::generate_tracked(ChunkPos(2, cy, cz));
        let (_, mask_right) = Chunk::generate_tracked(ChunkPos(3, cy, cz));
        assert!(mask_left[1], "left chunk +X face should be divergent");
        assert!(mask_right[0], "right chunk -X face should be divergent");
    }

    /// generate() and generate_tracked() must agree, and the boundary mask
    /// must be set whenever a tree block sits on the corresponding
    /// outermost layer.
    #[test]
    fn boundary_mask_matches_tree_content() {
        for (cx, cz) in [(0, 0), (1, 3), (-2, 5), (7, -4), (10, 10)] {
            let pos = ChunkPos(cx, 0, cz);
            let (chunk, mask) = Chunk::generate_tracked(pos);
            // Recompute divergence from pure terrain per face.
            let layer_diverges = |face: usize| -> bool {
                let (fixed_axis, fixed_val) = match face {
                    0 => (0, 0), 1 => (0, CHUNK_SIZE - 1),
                    2 => (1, 0), 3 => (1, CHUNK_SIZE - 1),
                    4 => (2, 0), _ => (2, CHUNK_SIZE - 1),
                };
                for a in 0..CHUNK_SIZE {
                    for b in 0..CHUNK_SIZE {
                        let (lx, ly, lz) = match fixed_axis {
                            0 => (fixed_val, a, b),
                            1 => (a, fixed_val, b),
                            _ => (a, b, fixed_val),
                        };
                        let wx = pos.0 * CHUNK_SIZE + lx;
                        let wy = pos.1 * CHUNK_SIZE + ly;
                        let wz = pos.2 * CHUNK_SIZE + lz;
                        if chunk.blocks[Chunk::index(lx, ly, lz)] != natural_block_at(wx, wy, wz) {
                            return true;
                        }
                    }
                }
                false
            };
            for face in 0..6 {
                assert_eq!(
                    mask[face],
                    layer_diverges(face),
                    "mask mismatch on face {face} of chunk ({cx},0,{cz})"
                );
            }
        }
    }
}
