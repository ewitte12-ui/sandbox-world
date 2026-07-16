//! Minecraft-style per-voxel flood-fill lighting.
//!
//! Two independent 0..15 channels per voxel, packed into one byte on
//! `Chunk::light` (sky in the high nibble, block in the low nibble):
//!
//!  - SKY light: seeded 15 from open sky, propagates straight DOWN without
//!    attenuation while at full strength, spreads sideways/up at −1 per
//!    step. Sun intensity is NOT baked into the level — the shader
//!    modulates the sky channel with a per-frame uniform, so day/night
//!    never forces a remesh.
//!  - BLOCK light: seeded 15 at emissive voxels (lanterns), −1 per step in
//!    every direction.
//!
//! Both are blocked by opaque voxels and are baked into mesh vertices at
//! mesh time (see chunk.rs build_mesh), replacing the old capped pool of
//! 64 lantern PointLights that leaked through walls.
//!
//! ## Cross-chunk model: per-chunk recompute + relaxation
//!
//! A chunk's light is always computed AS A UNIT from (a) its own emissive
//! blocks and (b) fixed boundary inputs — the 1-cell shell copied from
//! loaded neighbors' light, or predicted from deterministic terrain where
//! a neighbor isn't loaded (sky = 15 above `surface_y`, exactly like the
//! terrain-predicted block padding in build_mesh). When a recompute
//! changes a chunk's boundary layers, its neighbors are queued for their
//! own recompute (chunk_manager::process_light_queue, budgeted like the
//! deferred remesh queue).
//!
//! This replaces the classic per-voxel "removal BFS" with fixed-point
//! relaxation: stale light that lost its source decays by at least one
//! level per neighbor round-trip, so removals converge in a bounded
//! number of budgeted iterations (light fades over a few frames instead
//! of vanishing in one — an accepted tradeoff for a much simpler and
//! self-healing algorithm; see `removing_the_source_converges_to_dark`).
//!
//! Known residual: the seam staleness check (`seam_stale`) only fires on
//! a ≥2-level discontinuity, so a neighbor lit from a stale prediction
//! can stay over-bright by exactly 1 level until any nearby edit or
//! reload — invisible in practice (~4% brightness).

use crate::block_types::BlockType;
use crate::chunk::{Chunk, ChunkNeighbors, ChunkPos, CHUNK_SIZE, CHUNK_VOLUME};

/// Maximum light level for both channels.
pub const MAX_LIGHT: u8 = 15;

/// Padded dimension: 16 + 1 boundary-input cell on each side.
const P: usize = (CHUNK_SIZE + 2) as usize;

/// Cells in the padded 18³ grid.
pub const PADDED_VOLUME: usize = P * P * P;

/// Index into an 18³ padded light array. Must be used for every access to
/// `ShellLight` arrays and the padded channel grids so chunk.rs and this
/// module agree on layout.
#[inline]
pub fn pidx(x: usize, y: usize, z: usize) -> usize {
    x + y * P + z * P * P
}

/// Sky channel of a packed light byte.
#[inline]
pub fn sky(l: u8) -> u8 {
    l >> 4
}

/// Block channel of a packed light byte.
#[inline]
pub fn blk(l: u8) -> u8 {
    l & 0x0F
}

/// Pack sky + block channels into one byte.
#[inline]
pub fn pack(sky: u8, blk: u8) -> u8 {
    (sky << 4) | (blk & 0x0F)
}

/// Terrain-predicted sky light for a world cell in UNLOADED space: full
/// sky above the deterministic surface, dark below. The same prediction
/// build_mesh uses for block padding — divergence (trees, buildings,
/// edits) is reconciled when the real neighbor loads (`seam_stale`).
#[inline]
pub fn predicted_sky(wy: i32, surface: i32) -> u8 {
    if wy > surface {
        MAX_LIGHT
    } else {
        0
    }
}

/// Fixed boundary-input light for a chunk recompute: the full 18³ grids
/// with ONLY the 1-cell shell populated (interior cells are zero and are
/// overwritten by the flood fill / the caller).
pub struct ShellLight {
    pub sky: [u8; PADDED_VOLUME],
    pub blk: [u8; PADDED_VOLUME],
}

impl ShellLight {
    /// All-dark shell (no boundary input) — used by tests and as the
    /// build target for `gather_shell_light`.
    pub fn dark() -> Self {
        Self {
            sky: [0; PADDED_VOLUME],
            blk: [0; PADDED_VOLUME],
        }
    }
}

/// Build the boundary-input shell for a chunk: real light from loaded
/// face neighbors, terrain-predicted sky everywhere else (unloaded face
/// slabs, and all edge/corner cells — which face neighbors never cover
/// but vertex light sampling touches, mirroring the AO shell fill).
pub fn gather_shell_light(pos: ChunkPos, neighbors: &ChunkNeighbors) -> ShellLight {
    let mut shell = ShellLight::dark();
    let cs = CHUNK_SIZE;
    let base_x = pos.0 * cs;
    let base_y = pos.1 * cs;
    let base_z = pos.2 * cs;

    // Lazily computed per-column surface heights (i32::MIN = unset), same
    // pattern as build_mesh's block-shell fill.
    let mut col_surface = [[i32::MIN; P]; P];
    let mut surface_at = |px: usize, pz: usize| -> i32 {
        if col_surface[pz][px] == i32::MIN {
            col_surface[pz][px] =
                crate::terrain::surface_y(base_x + px as i32 - 1, base_z + pz as i32 - 1);
        }
        col_surface[pz][px]
    };

    for pz in 0..P {
        for py in 0..P {
            for px in 0..P {
                let sx = px == 0 || px == P - 1;
                let sy = py == 0 || py == P - 1;
                let sz = pz == 0 || pz == P - 1;
                let shell_axes = sx as u8 + sy as u8 + sz as u8;
                if shell_axes == 0 {
                    continue; // interior — filled by the caller
                }
                // Face-interior cells of a LOADED neighbor carry real light.
                if shell_axes == 1 {
                    let (face, lx, ly, lz) = if px == 0 {
                        (0, cs - 1, py as i32 - 1, pz as i32 - 1)
                    } else if px == P - 1 {
                        (1, 0, py as i32 - 1, pz as i32 - 1)
                    } else if py == 0 {
                        (2, px as i32 - 1, cs - 1, pz as i32 - 1)
                    } else if py == P - 1 {
                        (3, px as i32 - 1, 0, pz as i32 - 1)
                    } else if pz == 0 {
                        (4, px as i32 - 1, py as i32 - 1, cs - 1)
                    } else {
                        (5, px as i32 - 1, py as i32 - 1, 0)
                    };
                    if let Some(n) = neighbors.neighbors[face] {
                        let l = n.light[Chunk::index(lx, ly, lz)];
                        let pi = pidx(px, py, pz);
                        shell.sky[pi] = sky(l);
                        shell.blk[pi] = blk(l);
                        continue;
                    }
                }
                // Unloaded neighbor or edge/corner cell: terrain prediction.
                let s = surface_at(px, pz);
                let pi = pidx(px, py, pz);
                shell.sky[pi] = predicted_sky(base_y + py as i32 - 1, s);
                // Predicted block light is always 0 (lanterns are edits).
            }
        }
    }

    shell
}

/// Six face offsets in ChunkNeighbors order (-X, +X, -Y, +Y, -Z, +Z).
pub const FACE_OFFSETS: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// Flood-fill one channel over the padded grid. Shell cells are fixed
/// sources (never written); interior cells are relaxed to their final
/// level. `is_sky` enables the "full-strength straight down" rule.
///
/// Dijkstra-style bucket queue processed from bright to dark: each cell
/// settles at its final level the first time it's popped at that level,
/// so total work is linear in lit cells.
fn propagate(levels: &mut [u8; PADDED_VOLUME], opaque: &[bool; PADDED_VOLUME], is_sky: bool) {
    let mut buckets: [Vec<u32>; 16] = Default::default();
    for (idx, &l) in levels.iter().enumerate() {
        // Level-1 cells can't light anything (would spread 0).
        if l >= 2 {
            buckets[l as usize].push(idx as u32);
        }
    }

    for lv in (2..=MAX_LIGHT as usize).rev() {
        let mut i = 0;
        // The bucket can grow while iterating (sky light propagating down
        // at 15 appends to bucket 15) — index loop, not iterator.
        while i < buckets[lv].len() {
            let idx = buckets[lv][i] as usize;
            i += 1;
            if levels[idx] as usize != lv {
                continue; // stale entry — cell was re-relaxed brighter
            }
            let x = (idx % P) as i32;
            let y = ((idx / P) % P) as i32;
            let z = (idx / (P * P)) as i32;
            for (dx, dy, dz) in FACE_OFFSETS {
                let (nx, ny, nz) = (x + dx, y + dy, z + dz);
                // Only relax interior cells — the shell is fixed input.
                if !(1..=CHUNK_SIZE).contains(&nx)
                    || !(1..=CHUNK_SIZE).contains(&ny)
                    || !(1..=CHUNK_SIZE).contains(&nz)
                {
                    continue;
                }
                let ni = pidx(nx as usize, ny as usize, nz as usize);
                if opaque[ni] {
                    continue;
                }
                let nl = if is_sky && dy == -1 && lv == MAX_LIGHT as usize {
                    MAX_LIGHT
                } else {
                    (lv - 1) as u8
                };
                if nl > levels[ni] {
                    levels[ni] = nl;
                    buckets[nl as usize].push(ni as u32);
                }
            }
        }
    }
}

/// Compute a chunk's packed light grid from its blocks and a fixed
/// boundary-input shell. Pure — this is the unit the relaxation model
/// recomputes; unit tests drive it directly with hand-built shells.
pub fn compute_light(
    blocks: &[BlockType; CHUNK_VOLUME],
    shell: &ShellLight,
) -> [u8; CHUNK_VOLUME] {
    let mut sky_l = shell.sky;
    let mut blk_l = shell.blk;
    let mut opaque = [false; PADDED_VOLUME];

    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let b = blocks[Chunk::index(x, y, z)];
                let pi = pidx((x + 1) as usize, (y + 1) as usize, (z + 1) as usize);
                // Interior starts dark; emissive voxels are block-light
                // sources at full strength (they propagate outward even
                // though the voxel itself is opaque).
                sky_l[pi] = 0;
                blk_l[pi] = if b.is_emissive() { MAX_LIGHT } else { 0 };
                opaque[pi] = b.is_opaque();
            }
        }
    }

    propagate(&mut sky_l, &opaque, true);
    propagate(&mut blk_l, &opaque, false);

    let mut out = [0u8; CHUNK_VOLUME];
    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let pi = pidx((x + 1) as usize, (y + 1) as usize, (z + 1) as usize);
                out[Chunk::index(x, y, z)] = pack(sky_l[pi], blk_l[pi]);
            }
        }
    }
    out
}

/// Recompute a chunk's light in place in the world: boundary inputs from
/// loaded neighbors, terrain prediction elsewhere.
pub fn compute_chunk_light(
    blocks: &[BlockType; CHUNK_VOLUME],
    pos: ChunkPos,
    neighbors: &ChunkNeighbors,
) -> [u8; CHUNK_VOLUME] {
    let shell = gather_shell_light(pos, neighbors);
    compute_light(blocks, &shell)
}

/// Which boundary faces (ChunkNeighbors order) have any differing light
/// value between two light grids — after a recompute, only neighbors
/// behind changed faces need their own recompute.
pub fn boundary_faces_changed(
    old: &[u8; CHUNK_VOLUME],
    new: &[u8; CHUNK_VOLUME],
) -> [bool; 6] {
    let mut changed = [false; 6];
    let last = CHUNK_SIZE - 1;
    for a in 0..CHUNK_SIZE {
        for b in 0..CHUNK_SIZE {
            let checks = [
                (0, Chunk::index(0, a, b)),
                (1, Chunk::index(last, a, b)),
                (2, Chunk::index(a, 0, b)),
                (3, Chunk::index(a, last, b)),
                (4, Chunk::index(a, b, 0)),
                (5, Chunk::index(a, b, last)),
            ];
            for (face, idx) in checks {
                if !changed[face] && old[idx] != new[idx] {
                    changed[face] = true;
                }
            }
        }
    }
    changed
}

/// True when the seam between a freshly lit chunk and an already-lit
/// neighbor shows a ≥2-level discontinuity in either channel — light can
/// only change by 1 per step, so a bigger jump means the neighbor was lit
/// against stale (usually terrain-predicted) inputs and needs a
/// recompute. Opaque cells carry level 0 (or 15 for emissive sources), so
/// e.g. a tree canopy loading next to a sky-lit neighbor trips this.
///
/// Deliberately over-triggers on natural terrain seams (an opaque surface
/// cell next to open sky reads as 0 vs 15): the false positive costs one
/// recompute that produces identical light and queues nothing further.
pub fn seam_stale(chunk: &Chunk, face: usize, neighbor: &Chunk) -> bool {
    let last = CHUNK_SIZE - 1;
    for a in 0..CHUNK_SIZE {
        for b in 0..CHUNK_SIZE {
            let (ci, ni) = match face {
                0 => (Chunk::index(0, a, b), Chunk::index(last, a, b)),
                1 => (Chunk::index(last, a, b), Chunk::index(0, a, b)),
                2 => (Chunk::index(a, 0, b), Chunk::index(a, last, b)),
                3 => (Chunk::index(a, last, b), Chunk::index(a, 0, b)),
                4 => (Chunk::index(a, b, 0), Chunk::index(a, b, last)),
                _ => (Chunk::index(a, b, last), Chunk::index(a, b, 0)),
            };
            let (c, n) = (chunk.light[ci], neighbor.light[ni]);
            if sky(c).abs_diff(sky(n)) >= 2 || blk(c).abs_diff(blk(n)) >= 2 {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn air_blocks() -> [BlockType; CHUNK_VOLUME] {
        [BlockType::AIR; CHUNK_VOLUME]
    }

    fn blk_at(light: &[u8; CHUNK_VOLUME], x: i32, y: i32, z: i32) -> u8 {
        blk(light[Chunk::index(x, y, z)])
    }

    fn sky_at(light: &[u8; CHUNK_VOLUME], x: i32, y: i32, z: i32) -> u8 {
        sky(light[Chunk::index(x, y, z)])
    }

    /// Block light spreads from an emissive voxel at −1 per step
    /// (manhattan distance on the 6-connected grid) and dies at 15 steps.
    #[test]
    fn lantern_light_attenuates_with_distance() {
        let mut blocks = air_blocks();
        blocks[Chunk::index(8, 8, 8)] = BlockType::LANTERN;
        let light = compute_light(&blocks, &ShellLight::dark());

        assert_eq!(blk_at(&light, 8, 8, 8), 15, "source cell");
        assert_eq!(blk_at(&light, 9, 8, 8), 14);
        assert_eq!(blk_at(&light, 15, 8, 8), 8, "7 steps away");
        assert_eq!(blk_at(&light, 15, 15, 8), 1, "14 steps away");
        assert_eq!(blk_at(&light, 15, 15, 15), 0, "21 steps — out of range");
        // Sky channel untouched by a dark shell.
        assert_eq!(sky_at(&light, 8, 8, 8), 0);
    }

    /// An opaque wall spanning the full chunk cross-section stops block
    /// light dead (no wrap path, dark shell).
    #[test]
    fn opaque_wall_blocks_light() {
        let mut blocks = air_blocks();
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                blocks[Chunk::index(10, y, z)] = BlockType::STONE;
            }
        }
        blocks[Chunk::index(5, 8, 8)] = BlockType::LANTERN;
        let light = compute_light(&blocks, &ShellLight::dark());

        assert_eq!(blk_at(&light, 9, 8, 8), 11, "4 steps, open side");
        assert_eq!(blk_at(&light, 10, 8, 8), 0, "inside the wall");
        for x in 11..CHUNK_SIZE {
            assert_eq!(blk_at(&light, x, 8, 8), 0, "behind the wall at x={x}");
        }
    }

    /// Sky light: full strength straight down through a hole in an
    /// otherwise solid ceiling, −1 when spreading sideways under it.
    #[test]
    fn sky_light_descends_through_hole_and_spreads() {
        let mut blocks = air_blocks();
        for x in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                blocks[Chunk::index(x, 15, z)] = BlockType::STONE;
            }
        }
        blocks[Chunk::index(8, 15, 8)] = BlockType::AIR; // the hole

        // Sky 15 across the entire top shell face.
        let mut shell = ShellLight::dark();
        for px in 1..=CHUNK_SIZE as usize {
            for pz in 1..=CHUNK_SIZE as usize {
                shell.sky[pidx(px, P - 1, pz)] = MAX_LIGHT;
            }
        }
        let light = compute_light(&blocks, &shell);

        // Full-strength column under the hole, all the way down.
        for y in 0..=15 {
            assert_eq!(sky_at(&light, 8, y, 8), 15, "column at y={y}");
        }
        // One block sideways under the ceiling: attenuated once.
        assert_eq!(sky_at(&light, 9, 14, 8), 14);
        assert_eq!(sky_at(&light, 10, 14, 8), 13);
        // Ceiling cells themselves are opaque and dark.
        assert_eq!(sky_at(&light, 9, 15, 8), 0);
    }

    /// Cross-chunk relaxation: computing two adjacent chunks against each
    /// other's boundary light converges, light crosses the seam
    /// continuously, and removing the source decays everything back to
    /// dark within a bounded number of round-trips. This is the
    /// correctness argument for replacing the per-voxel removal BFS.
    #[test]
    fn removing_the_source_converges_to_dark() {
        // Underground positions so missing-neighbor prediction is all 0.
        let pos_a = ChunkPos(0, -3, 0);
        let pos_b = ChunkPos(1, -3, 0);
        let mut a = Chunk { blocks: air_blocks(), light: [0; CHUNK_VOLUME], pos: pos_a };
        let mut b = Chunk { blocks: air_blocks(), light: [0; CHUNK_VOLUME], pos: pos_b };
        a.blocks[Chunk::index(14, 8, 8)] = BlockType::LANTERN;

        // Relax the pair until stable (mirrors process_light_queue).
        let mut relax = |a: &mut Chunk, b: &mut Chunk| -> u32 {
            for round in 0..40 {
                let na = compute_chunk_light(
                    &a.blocks,
                    a.pos,
                    &ChunkNeighbors { neighbors: [None, Some(b), None, None, None, None] },
                );
                let a_changed = na != a.light;
                a.light = na;
                let nb = compute_chunk_light(
                    &b.blocks,
                    b.pos,
                    &ChunkNeighbors { neighbors: [Some(a), None, None, None, None, None] },
                );
                let b_changed = nb != b.light;
                b.light = nb;
                if !a_changed && !b_changed {
                    return round;
                }
            }
            panic!("relaxation did not converge in 40 rounds");
        };

        relax(&mut a, &mut b);
        // Light crossed the seam: A(15,8,8)=14 → B(0,8,8)=13.
        assert_eq!(blk(a.light[Chunk::index(15, 8, 8)]), 14);
        assert_eq!(blk(b.light[Chunk::index(0, 8, 8)]), 13);
        // Seam is continuous — no stale discontinuity.
        assert!(!seam_stale(&a, 1, &b));
        assert!(!seam_stale(&b, 0, &a));

        // Remove the lantern. One recompute of A must change A's +X
        // boundary layer — that's what makes process_light_queue re-queue
        // B and drive the decay cascade. (seam_stale does NOT fire here:
        // relaxation decays gradually, so seams stay within 1 level —
        // it's the chunk-LOAD reconciliation detector, not the cascade
        // trigger.)
        a.blocks[Chunk::index(14, 8, 8)] = BlockType::AIR;
        let new_a = compute_chunk_light(
            &a.blocks,
            a.pos,
            &ChunkNeighbors { neighbors: [None, Some(&b), None, None, None, None] },
        );
        assert!(
            boundary_faces_changed(&a.light, &new_a)[1],
            "+X boundary must report a change so the neighbor gets re-queued"
        );
        a.light = new_a;

        // Full relaxation drains every remnant of the removed source.
        relax(&mut a, &mut b);
        assert!(a.light.iter().all(|&l| blk(l) == 0), "A still lit");
        assert!(b.light.iter().all(|&l| blk(l) == 0), "B still lit");
    }

    /// Real-content integration: relax a neighborhood of generated chunks
    /// around building spot (50, 30) and check the light field the whole
    /// feature exists for — enclosed interiors are dark, open ground is
    /// fully sky-lit, and light falls off with depth through the door.
    #[test]
    fn building_interior_is_dark_outside_is_lit() {
        use crate::terrain::surface_y;
        let base_y = surface_y(50, 30) + 1; // building floor level

        // Chunks covering the building (x 45..54, z 25..34, up past the
        // roof) plus a ring so boundary inputs inside the region are real.
        let mut region: std::collections::HashMap<ChunkPos, Chunk> =
            std::collections::HashMap::new();
        let cy_lo = (base_y - 4).div_euclid(CHUNK_SIZE);
        let cy_hi = (base_y + 8).div_euclid(CHUNK_SIZE);
        for cx in 1..=4 {
            for cz in 0..=3 {
                for cy in cy_lo..=cy_hi {
                    let pos = ChunkPos(cx, cy, cz);
                    region.insert(pos, Chunk::generate(pos));
                }
            }
        }

        // Relax until stable (mirrors process_light_queue without budget).
        let positions: Vec<ChunkPos> = region.keys().copied().collect();
        for round in 0..40 {
            let mut any_changed = false;
            for &pos in &positions {
                let neighbors = ChunkNeighbors {
                    neighbors: [
                        region.get(&ChunkPos(pos.0 - 1, pos.1, pos.2)),
                        region.get(&ChunkPos(pos.0 + 1, pos.1, pos.2)),
                        region.get(&ChunkPos(pos.0, pos.1 - 1, pos.2)),
                        region.get(&ChunkPos(pos.0, pos.1 + 1, pos.2)),
                        region.get(&ChunkPos(pos.0, pos.1, pos.2 - 1)),
                        region.get(&ChunkPos(pos.0, pos.1, pos.2 + 1)),
                    ],
                };
                let chunk = &region[&pos];
                let new_light = compute_chunk_light(&chunk.blocks, pos, &neighbors);
                if new_light != chunk.light {
                    any_changed = true;
                    // Split borrow: recompute uses immutable refs, write after.
                    drop(neighbors);
                    region.get_mut(&pos).expect("chunk in region").light = new_light;
                }
            }
            if !any_changed {
                assert!(round > 0, "no light computed at all");
                break;
            }
            assert!(round < 39, "region relaxation did not converge");
        }

        let light_at = |wx: i32, wy: i32, wz: i32| -> u8 {
            let pos = ChunkPos(
                wx.div_euclid(CHUNK_SIZE),
                wy.div_euclid(CHUNK_SIZE),
                wz.div_euclid(CHUNK_SIZE),
            );
            sky(region[&pos].light[Chunk::index(
                wx.rem_euclid(CHUNK_SIZE),
                wy.rem_euclid(CHUNK_SIZE),
                wz.rem_euclid(CHUNK_SIZE),
            )])
        };

        // Open ground just outside the door (dz=0 wall is at z=25): full sky.
        assert_eq!(light_at(49, base_y + 1, 22), 15, "open air outside");
        // Interior back corner (46, ·, 32): reachable only via the door at
        // (48..51, 25) — at least 8 propagation steps of horizontal spread.
        let back = light_at(46, base_y + 1, 32);
        assert!(back <= 7, "back corner should be dim, got {back}");
        // Interior is darker than the doorway-adjacent cell.
        let near_door = light_at(49, base_y + 1, 26);
        assert!(near_door > back, "light must fall off with door distance");
        // Nothing inside reaches full skylight (roof blocks the column).
        for wx in 46..=53 {
            for wz in 26..=33 {
                let l = light_at(wx, base_y + 2, wz);
                assert!(l < 15, "interior ({wx},{wz}) fully sky-lit: {l}");
            }
        }
    }

    /// Packing helpers are lossless over the full 0..15 range.
    #[test]
    fn pack_roundtrips() {
        for s in 0..=MAX_LIGHT {
            for b in 0..=MAX_LIGHT {
                let p = pack(s, b);
                assert_eq!((sky(p), blk(p)), (s, b));
            }
        }
    }
}
