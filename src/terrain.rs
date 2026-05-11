use crate::block_types::BlockType;
use crate::chunk::CHUNK_VOLUME;

// ---------------------------------------------------------------------------
// Noise functions — exact ports from the original Swift renderer.
//
// These use standard GPU-style hash and gradient noise techniques (see
// Inigo Quilez, "Hash without Sine" and Perlin gradient noise references).
// The specific magic constants (127.1, 311.7, 43758.5453, etc.) are
// widely-used hash primes that produce good pseudo-random distribution
// when combined with sin(). They originate from the classic GLSL noise
// one-liners and must NOT be changed — doing so would alter all terrain
// generation and break save-file compatibility.
// ---------------------------------------------------------------------------

/// Hash function for 2D gradient noise, ported from Swift.
#[allow(clippy::excessive_precision)]
fn hash2(p: [f32; 2]) -> [f32; 2] {
    let qx = p[0] * 127.1 + p[1] * 311.7;
    let qy = p[0] * 269.5 + p[1] * 183.3;
    fn sfract(x: f32) -> f32 {
        x - x.floor()
    }
    [
        -1.0 + 2.0 * sfract(qx.sin() * 43758.5453123),
        -1.0 + 2.0 * sfract(qy.sin() * 43758.5453123),
    ]
}

/// 2D gradient noise, ported from Swift.
fn grad_noise(p: [f32; 2]) -> f32 {
    fn sfract(x: f32) -> f32 {
        x - x.floor()
    }
    let ix = p[0].floor();
    let iy = p[1].floor();
    let fx = sfract(p[0]);
    let fy = sfract(p[1]);
    // Quintic Hermite interpolation (Ken Perlin's improved noise, 2002).
    // f(t) = 6t^5 - 15t^4 + 10t^3 — eliminates second-derivative
    // discontinuities that cause visible grid artifacts with cubic interp.
    let ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);

    fn dot2(a: [f32; 2], b: [f32; 2]) -> f32 {
        a[0] * b[0] + a[1] * b[1]
    }

    let a = dot2(hash2([ix, iy]), [fx, fy]);
    let b = dot2(hash2([ix + 1.0, iy]), [fx - 1.0, fy]);
    let c = dot2(hash2([ix, iy + 1.0]), [fx, fy - 1.0]);
    let d = dot2(hash2([ix + 1.0, iy + 1.0]), [fx - 1.0, fy - 1.0]);

    let ab = a + ux * (b - a);
    let cd = c + ux * (d - c);
    ab + uy * (cd - ab)
}

/// Fractal Brownian Motion using gradient noise, ported from Swift.
fn fbm(p: [f32; 2], octaves: i32, persistence: f32, lacunarity: f32) -> f32 {
    let mut val: f32 = 0.0;
    let mut amp: f32 = 0.5;
    let mut freq: f32 = 1.0;
    let mut max_v: f32 = 0.0;
    for _ in 0..octaves {
        val += amp * grad_noise([p[0] * freq, p[1] * freq]);
        max_v += amp;
        amp *= persistence;
        freq *= lacunarity;
    }
    val / max_v
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Returns the terrain height at a given world X/Z position.
/// Exact port of Swift `terrainHeightAt`.
///
/// The terrain is generated in two layers:
/// 1. A coarse "hill mask" (large-scale FBM) that controls WHERE hills appear.
///    Sampled at 1/400 world scale (500 * 0.8) with 3 octaves for broad shapes.
///    smoothstep(0.15, 0.45) creates flat plains (mask=0) vs hilly regions (mask=1).
/// 2. A detail layer (finer FBM) that provides the actual surface undulation.
///    Sampled at 1/150 world scale (500 * 0.3) with 4 octaves for richer detail.
///    Multiplied by the hill mask so flat regions stay flat.
///
/// Result range: 0.0 to 10.0 blocks above sea level.
/// These values are tuned to match the Swift version and must not change
/// without regenerating all saved worlds.
pub fn terrain_height_at(world_x: f32, world_z: f32) -> f32 {
    let noise_scale: f32 = 500.0; // world-space period of the largest noise octave
    let seed: f32 = 42.7; // offset to break symmetry around origin

    // Coarse hill mask: large features, few octaves, low persistence (smooth)
    let px = world_x / (noise_scale * 0.8) + seed;
    let pz = world_z / (noise_scale * 0.8) + seed;
    let n1 = fbm([px, pz], 3, 0.4, 2.0);
    // Map noise [-1,1] → [0,1], then smoothstep to create binary-ish
    // flat-vs-hilly regions. 0.15/0.45 thresholds give ~60% flat plains.
    let hill_mask = smoothstep(0.15, 0.45, n1 * 0.5 + 0.5);

    // Detail heightfield: more octaves, higher persistence (rougher)
    let detail = fbm(
        [
            world_x / (noise_scale * 0.3) + seed * 1.7,
            world_z / (noise_scale * 0.3) + seed * 1.7,
        ],
        4,   // octaves — more than hill mask for finer surface detail
        0.5, // persistence — each octave is half the previous amplitude
        2.1, // lacunarity — slightly above 2.0 to avoid grid alignment
    );
    // Map [-1,1] → [0,1], clamp negative, scale to max 10 blocks, mask by hills
    (detail * 0.5 + 0.5).max(0.0) * 10.0 * hill_mask
}

/// Returns the integer surface Y for a given X/Z column.
pub fn surface_y(x: i32, z: i32) -> i32 {
    let h = terrain_height_at(x as f32 + 0.5, z as f32 + 0.5);
    h.floor() as i32
}

/// 3D gradient noise for underground block selection, ported from Swift.
/// Used at frequency 0.12 (stone/dirt veins ~8 blocks wide) and 0.08
/// (diamond veins ~12 blocks wide). See natural_block_at() for usage.
#[allow(clippy::excessive_precision)]
fn noise3d(x: f32, y: f32, z: f32) -> f32 {
    let ix = x.floor();
    let iy = y.floor();
    let iz = z.floor();
    let fx = x - ix;
    let fy = y - iy;
    let fz = z - iz;
    // Quintic Hermite interpolation — same rationale as grad_noise()
    let ux = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let uy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);
    let uz = fz * fz * fz * (fz * (fz * 6.0 - 15.0) + 10.0);

    fn hash3(px: f32, py: f32, pz: f32) -> f32 {
        let v = px * 127.1 + py * 311.7 + pz * 74.7;
        v.sin() * 43758.5453
    }
    fn grad(px: f32, py: f32, pz: f32, ox: f32, oy: f32, oz: f32) -> f32 {
        let h = hash3(px, py, pz);
        let gx = (h * 1.0).sin() * 2.0 - 1.0;
        let gy = (h * 2.37).sin() * 2.0 - 1.0;
        let gz = (h * 3.71).sin() * 2.0 - 1.0;
        gx * ox + gy * oy + gz * oz
    }

    let n000 = grad(ix, iy, iz, fx, fy, fz);
    let n100 = grad(ix + 1.0, iy, iz, fx - 1.0, fy, fz);
    let n010 = grad(ix, iy + 1.0, iz, fx, fy - 1.0, fz);
    let n110 = grad(ix + 1.0, iy + 1.0, iz, fx - 1.0, fy - 1.0, fz);
    let n001 = grad(ix, iy, iz + 1.0, fx, fy, fz - 1.0);
    let n101 = grad(ix + 1.0, iy, iz + 1.0, fx - 1.0, fy, fz - 1.0);
    let n011 = grad(ix, iy + 1.0, iz + 1.0, fx, fy - 1.0, fz - 1.0);
    let n111 = grad(ix + 1.0, iy + 1.0, iz + 1.0, fx - 1.0, fy - 1.0, fz - 1.0);
    let nx00 = n000 + ux * (n100 - n000);
    let nx10 = n010 + ux * (n110 - n010);
    let nx01 = n001 + ux * (n101 - n001);
    let nx11 = n011 + ux * (n111 - n011);
    let nxy0 = nx00 + uy * (nx10 - nx00);
    let nxy1 = nx01 + uy * (nx11 - nx01);
    nxy0 + uz * (nxy1 - nxy0)
}

/// Positional hash for surface block selection, ported from Swift.
#[allow(clippy::excessive_precision)]
fn pos_hash(x: i32, z: i32) -> f32 {
    let v = x as f32 * 127.1 + z as f32 * 311.7;
    let s = v.sin() * 43758.5453;
    s - s.floor()
}

/// Returns the surface block type for a given X/Z column.
/// Distribution: 75% Grass, 10% Dirt, 10% Stone, 5% Sand.
/// Tuned to give a predominantly green landscape with occasional variety.
pub fn surface_block(x: i32, z: i32) -> BlockType {
    let h = pos_hash(x, z);
    if h < 0.75 {
        BlockType::GRASS
    } else if h < 0.85 {
        BlockType::DIRT
    } else if h < 0.95 {
        BlockType::STONE
    } else {
        BlockType::SAND
    }
}

/// Returns the natural (unmodified) block at world coordinates.
/// Exact port of Swift `naturalBlockAt`.
///
/// Geological strata (depth = blocks below surface):
///   0       — surface block (Grass/Dirt/Stone/Sand per column hash)
///   1       — always Dirt (topsoil layer)
///   2-10    — shallow mix: noise selects Dirt/Sand/Stone (varied near-surface)
///   11-25   — mid stratum: noise selects Dirt or Stone (transitioning to rock)
///   26-99   — deep rock: mostly Stone/Dirt, with rare Diamond veins
///   100+    — Bedrock (indestructible floor, prevents digging into void)
///
/// Diamond rarity increases closer to the surface: the noise threshold
/// starts at 0.6 (very rare) and decreases by 0.08 over the full depth
/// range, making diamonds slightly more common near bedrock. This rewards
/// deep mining without making diamonds trivially abundant.
///
/// The 0.12 noise frequency for block selection creates ~8-block-wide veins.
/// Diamond uses a separate noise sample (0.08 freq, offset by 100) so vein
/// shapes are independent of the stone/dirt pattern.
pub fn natural_block_at(x: i32, y: i32, z: i32) -> BlockType {
    let sy = surface_y(x, z);
    if y > sy {
        return BlockType::AIR;
    }
    let depth = sy - y;

    // Bedrock at depth 100+: prevents players from digging into the void.
    // 100 blocks gives ample mining depth while keeping world bounded.
    if depth >= 100 {
        return BlockType::BEDROCK;
    }
    if depth == 0 {
        return surface_block(x, z);
    }
    if depth == 1 {
        return BlockType::DIRT;
    }

    // Shallow stratum (2-10): mixed materials for visual variety near surface.
    // Noise thresholds 0.3/0.15 give roughly 35% Stone, 15% Sand, 50% Dirt.
    if depth <= 10 {
        let n = noise3d(x as f32 * 0.12, y as f32 * 0.12, z as f32 * 0.12);
        if n > 0.3 {
            return BlockType::STONE;
        }
        if n > 0.15 {
            return BlockType::SAND;
        }
        return BlockType::DIRT;
    }

    // Mid stratum (11-25): transitioning to predominantly stone.
    // Threshold 0.0 gives ~50/50 Stone/Dirt.
    if depth <= 25 {
        let n = noise3d(x as f32 * 0.12, y as f32 * 0.12, z as f32 * 0.12);
        if n > 0.0 {
            return BlockType::STONE;
        }
        return BlockType::DIRT;
    }

    // Deep stratum (26-99): mostly stone with rare diamond veins.
    let n = noise3d(x as f32 * 0.12, y as f32 * 0.12, z as f32 * 0.12);
    // Separate noise sample for diamond veins (different frequency + offset
    // so diamond distribution is independent of stone/dirt boundaries).
    let d_noise = noise3d(
        x as f32 * 0.08 + 100.0,
        y as f32 * 0.08,
        z as f32 * 0.08 + 100.0,
    );
    // depth_frac 0.0 at depth 26, 1.0 at depth 99
    let depth_frac = (depth - 26) as f32 / (99 - 26) as f32;
    // Threshold decreases with depth: 0.6 (rare) → 0.52 (slightly less rare)
    let diamond_threshold: f32 = 0.6 - depth_frac * 0.08;
    if d_noise > diamond_threshold {
        return BlockType::DIAMOND;
    }
    if n > 0.0 {
        return BlockType::STONE;
    }
    BlockType::DIRT
}

// ---------------------------------------------------------------------------
// Tree placement as voxel blocks
// ---------------------------------------------------------------------------

/// Hash function matching trees.rs placement.
#[allow(clippy::excessive_precision)]
fn tree_hash(px: f32, pz: f32) -> f32 {
    let v = px * 127.1 + pz * 311.7;
    let s = v.sin() * 43758.5453;
    s - s.floor()
}

/// Check if a tree exists at grid cell (i, j) and return (world_x, world_z, ground_y, scale).
/// Returns None if no tree at this position.
///
/// Trees are placed on an 18-block grid (one potential tree per 18x18 area)
/// to prevent overlap while keeping density high enough for forests.
/// Each cell has a 28% chance of containing a tree (r1 < 0.28), giving
/// roughly 1 tree per 1150 sq blocks — matching the Swift version's density.
///
/// The tree position is jittered within the cell by up to ±85% of the
/// step size (0.85 * 18 / 2 ≈ ±7.6 blocks) so the grid isn't visible.
///
/// Trees are excluded below height 3.4 (underwater/beach) and above 92.4
/// (near bedrock depth limit, where they'd be buried). These thresholds
/// match the Swift version.
pub fn tree_at_grid(i: i32, j: i32) -> Option<(f32, f32, f32, f32)> {
    let step: f32 = 18.0;
    let cx = i as f32 * step;
    let cz = j as f32 * step;

    // Each hash uses different seed offsets to produce independent values
    let r1 = tree_hash(cx * 0.013 + 42.1, cz * 0.013 + 13.7);
    if r1 >= 0.28 {
        return None;
    }

    let r2 = tree_hash(cx * 0.013 + 17.3, cz * 0.013 + 88.2);
    let r3 = tree_hash(cx * 0.013 + 99.1, cz * 0.013 + 5.7);
    let r4 = tree_hash(cx * 0.013 + 55.3, cz * 0.013 + 72.1);

    // Jitter position within grid cell to hide regularity
    let wx = cx + (r2 - 0.5) * step * 0.85;
    let wz = cz + (r3 - 0.5) * step * 0.85;

    let ground_y = terrain_height_at(wx, wz);
    if ground_y < 3.4 || ground_y >= 92.4 {
        return None;
    }

    let scale = 1.0 + r4 * 1.5; // range 1.0–2.5, varies trunk/canopy size
    Some((wx, wz, ground_y, scale))
}

/// Place tree blocks into a chunk's block array during generation.
/// Call this after filling natural blocks in Chunk::generate().
pub fn place_trees_in_chunk(
    blocks: &mut [BlockType; CHUNK_VOLUME],
    chunk_pos_x: i32,
    chunk_pos_y: i32,
    chunk_pos_z: i32,
) {
    let chunk_size = 16;
    let base_x = chunk_pos_x * chunk_size;
    let base_y = chunk_pos_y * chunk_size;
    let base_z = chunk_pos_z * chunk_size;

    // Check a wide area of tree grid cells that could have trees overlapping this chunk
    let step: f32 = 18.0;
    // Trees can be up to ~12 blocks wide and ~15 blocks tall, so scan extra grid cells
    let search_margin = 20;
    let i_min = ((base_x - search_margin) as f32 / step).floor() as i32;
    let i_max = ((base_x + chunk_size + search_margin) as f32 / step).ceil() as i32;
    let j_min = ((base_z - search_margin) as f32 / step).floor() as i32;
    let j_max = ((base_z + chunk_size + search_margin) as f32 / step).ceil() as i32;

    for i in i_min..=i_max {
        for j in j_min..=j_max {
            if let Some((wx, wz, ground_y, scale)) = tree_at_grid(i, j) {
                place_one_tree(blocks, base_x, base_y, base_z, wx, wz, ground_y, scale);
            }
        }
    }
}

fn place_one_tree(
    blocks: &mut [BlockType; CHUNK_VOLUME],
    base_x: i32,
    base_y: i32,
    base_z: i32,
    wx: f32,
    wz: f32,
    ground_y: f32,
    scale: f32,
) {
    let chunk_size = 16;
    let tree_x = wx.floor() as i32;
    let tree_z = wz.floor() as i32;
    let tree_base_y = ground_y.floor() as i32 + 1; // one above ground

    // Trunk: 1 block wide, height based on scale
    let trunk_height = (4.0 * scale).round() as i32;
    for dy in 0..trunk_height {
        let wy = tree_base_y + dy;
        set_block_if_in_chunk(
            blocks, tree_x, wy, tree_z, base_x, base_y, base_z, chunk_size, BlockType::WOOD,
        );
    }

    // Canopy: ellipsoid of leaves centered above the trunk
    let canopy_bottom = tree_base_y + (trunk_height as f32 * 0.6).round() as i32;
    let canopy_top = tree_base_y + trunk_height + (2.5 * scale).round() as i32;
    let canopy_radius_xz = (2.2 * scale).round() as i32;

    for cy in canopy_bottom..=canopy_top {
        // Radius tapers: widest in middle, narrower at top and bottom
        let mid = (canopy_bottom + canopy_top) as f32 / 2.0;
        let dist_from_mid =
            ((cy as f32 - mid) / ((canopy_top - canopy_bottom) as f32 / 2.0 + 0.1)).abs();
        let radius = (canopy_radius_xz as f32 * (1.0 - dist_from_mid * 0.4)).max(1.0) as i32;

        for dx in -radius..=radius {
            for dz in -radius..=radius {
                // Rough sphere check
                if dx * dx + dz * dz > radius * radius {
                    continue;
                }
                let bx = tree_x + dx;
                let bz = tree_z + dz;
                set_block_if_in_chunk(
                    blocks, bx, cy, bz, base_x, base_y, base_z, chunk_size, BlockType::LEAVES,
                );
            }
        }
    }
}

fn set_block_if_in_chunk(
    blocks: &mut [BlockType; CHUNK_VOLUME],
    wx: i32,
    wy: i32,
    wz: i32,
    base_x: i32,
    base_y: i32,
    base_z: i32,
    chunk_size: i32,
    block: BlockType,
) {
    let lx = wx - base_x;
    let ly = wy - base_y;
    let lz = wz - base_z;
    if lx >= 0 && lx < chunk_size && ly >= 0 && ly < chunk_size && lz >= 0 && lz < chunk_size {
        let idx = (lx + ly * chunk_size + lz * chunk_size * chunk_size) as usize;
        // Only place into air blocks (don't overwrite terrain)
        if blocks[idx] == BlockType::AIR {
            blocks[idx] = block;
        }
    }
}
