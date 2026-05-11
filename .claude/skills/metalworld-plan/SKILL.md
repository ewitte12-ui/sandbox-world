---
name: metalworld-plan
description: |
  Reference plan for the MetalWorld Bevy port. Use when working on any phase of the project,
  implementing features, or needing to understand the architecture and Swift source mappings.
user-invocable: true
disable-model-invocation: false
---

# Plan: Port MetalWorld to Bevy (Rust)

## Context
MetalWorld is a Minecraft-style voxel game currently built in Swift + Metal, locked to macOS. The user wants to rebuild it as a multi-platform game using Bevy (Rust). This plan covers the full port in 7 incremental phases, each producing a runnable game. The architecture shifts from a monolithic renderer with flat-grid instanced voxels to a chunked ECS system with greedy meshing.

## Engine Choice: Bevy 0.18 (Rust)
- **wgpu** backend -> Metal/Vulkan/DX12/WebGPU (all platforms)
- **block-mesh-rs** for greedy voxel meshing (~40M quads/sec)
- Built-in deferred rendering, PBR, automatic instanced batching
- ECS architecture well-suited for voxel games
- Targets: macOS, Windows, Linux, WASM/WebGPU

## Key Architectural Changes from Swift Version

| Aspect | Swift/Metal | Bevy/Rust |
|--------|------------|-----------|
| Voxel storage | Flat grid + modifications dict | 16x16x16 chunks in HashMap |
| Rendering | Per-block instanced cubes (400K) | Greedy-meshed chunk meshes |
| Face culling | faceMask bitmask per instance | Built into greedy meshing |
| Shaders | Metal Shading Language | WGSL |
| Entity system | Manual arrays | Bevy ECS |
| UI | AppKit NSView overlay | Bevy UI nodes |
| Platform | macOS only | All desktop + web |

## Swift Source Files to Reference

| Rust target | Swift source | Key content |
|-------------|-------------|-------------|
| terrain.rs | Camera.swift:9-65 | terrainHeightAt, FBM noise |
| terrain.rs | BlockWorld.swift:189-233 | naturalBlockAt, layer rules |
| chunk.rs + chunk_manager.rs | BlockWorld.swift:260-500 | Generation, exposed tracking, instance buffer |
| player.rs | Camera.swift:130-187 | Movement, jumping, sneaking |
| player.rs | BlockWorld.swift:648-780 | Floor/ceiling/wall collision |
| ray_cast.rs | BlockWorld.swift:559-638 | DDA ray casting |
| animals.rs | AnimalManager.swift | All animal logic |
| trees.rs | TreeManager.swift | All tree logic |
| lighting.rs | Renderer.swift:890-930 | Sun cycle, uniforms |
| lighting.rs | Raytracing.metal | Voxel shadow kernel |
| ui.rs | AppDelegate.swift | Settings menu |

Swift source location: ~/Documents/metalworld/

---

## Phase 1: Chunked Voxel Terrain + FPS Camera
**Runnable result**: Walk around procedurally generated colored block terrain.

### Files
- **main.rs** -- App with DefaultPlugins, register ChunkPlugin + PlayerPlugin
- **block_types.rs** -- enum BlockType (11 variants, same u8 values as Swift), implement block_mesh::Voxel + MergeVoxel traits, per-type fn color() -> Color
- **terrain.rs** -- Port from Camera.swift lines 9-65 (terrainHeightAt, FBM noise) and BlockWorld.swift lines 189-233 (naturalBlockAt, layer rules, 3D noise). Must reproduce terrain exactly for visual parity
- **chunk.rs** -- struct Chunk with [BlockType; 4096], fn generate() from naturalBlockAt, fn build_mesh() using block_mesh::greedy_quads() with neighbor chunk data for seamless boundaries
- **chunk_manager.rs** -- Resource with HashMap<ChunkPos, Entity>, load/unload chunks within radius=5 (80 blocks), async generation via AsyncComputeTaskPool, modifications: HashMap<IVec3, BlockType>
- **player.rs** -- FPS camera with WASD (speed=22), mouse look (sensitivity=0.0007), jumping (v=12, gravity=-28), sneaking (20% speed, eye 1.2), floor/ceiling/wall collision via blockAt(). Port from Camera.swift lines 130-187 and BlockWorld.swift lines 648-780

### Key decisions
- 16x16x16 chunks with greedy meshing replaces 400K instanced cubes
- StandardMaterial with vertex colors for Phase 1 (custom shaders in Phase 4)
- Cursor grabbed during gameplay, released for menu

---

## Phase 2: Block Interaction + Buildings
**Runnable result**: Break/place blocks, crosshair, block type selection, 6 buildings near origin.

### Files
- **ray_cast.rs** -- Port DDA from BlockWorld.swift lines 559-638, reach=8 blocks
- **buildings.rs** -- Port placeBuildings from BlockWorld.swift lines 787-828 (wood walls, stone roof, door opening)
- **player.rs** (modify) -- Add selected_block, left-click break (hold=continuous 0.15s, max 5), right-click place (hold=continuous 0.18s, prevent self-placement), number keys 1-9, bed placement (6-block), teleport home (H key)
- **chunk_manager.rs** (modify) -- Mark chunks dirty on modification, remesh affected + neighbor chunks

---

## Phase 3: Trees + Animals
**Runnable result**: Trees dot landscape, 60 animals wander with walking animation.

### Files
- **trees.rs** -- Port TreeManager.swift: grid step=18, radius=220, 28% placement, 4 boxes per tree (trunk + 3 canopy), hash-based jitter/scale (1.0-2.5x), height band 3.4-92.4
- **animals.rs** -- Port AnimalManager.swift: 4 types x 6 box parts, 60 animals, wandering AI (20% idle, random turns, wall collision), diagonal leg-swing animation (sin(t*freq)*0.04), body bob (sin(t*freq)*0.03)

---

## Phase 4: Deferred Rendering + Sky + Lighting
**Runnable result**: Day/night cycle, procedural sky/clouds, face-shaded blocks, lantern point lights.

### Files
- **lighting.rs** -- Sun cycle (10-min day), direction=normalize(cos(a)*0.5, sin(a), cos(a)*0.866), up to 64 lantern point lights (radius=8.0)
- **sky.rs** -- Procedural sky gradient, sun disc (halo exp=32), animated FBM clouds
- **assets/shaders/block_material.wgsl** -- G-buffer output, face brightness (top=1.0, side=0.82, bottom=0.50), grid overlay (2% UV border), noise color variation +/-6%
- **assets/shaders/deferred_lighting.wgsl** -- PBR sun + 64 point lights, exponential fog (density 0.00003), ACES tone mapping, contrast/gamma/brightness
- **assets/shaders/sky.wgsl** -- Sky + cloud rendering

---

## Phase 5: Voxel Shadows + AO
**Runnable result**: Blocks cast sun shadows, enclosed spaces are dark, smooth shadow quality.

### Files
- **lighting.rs** (modify) -- 3D occupancy texture (64x64x48, 4 blocks/texel, 256x256x192 world coverage), amortized build (8 slices/frame), recenter at 32-block edge margin
- **assets/shaders/shadow_ao.wgsl** -- Compute shader: sun ray march (16 steps x 4 blocks), sky visibility march (12 steps upward), texel-snapped coords, separable Gaussian blur

---

## Phase 6: UI + Settings + Save/Load
**Runnable result**: Settings menu (M key), persistent settings, save/load game.

### Files
- **settings.rs** -- GameSettings (serde): sensitivity, grid toggle, contrast/gamma/brightness, custom colors. JSON to ~/.metalworld_settings.json
- **save_load.rs** -- Modifications + camera position + home. JSON to ~/.metalworld_save.json
- **ui.rs** -- Tabbed settings (Display/Controls/Edit/Game/Developer), crosshair HUD, selected block display, sliders + toggles + color pickers
- **dev_tools.rs** -- Developer tab backend: runtime-tweakable constants, performance monitoring, debug visualizations

### Developer Tab Constants
Movement (player_speed=22, sprint=1.5, sneak=0.2, jump=12, gravity=-28), Camera (sensitivity=0.0007, fov=70), Terrain (octaves=6, persistence=0.5), Rendering (render_distance=5, fog_density=0.00003), Shadows (steps=16, sky_steps=12, ao_radius=12), Lighting (day_cycle=600s, ambient_min=0.08, lantern_radius=8.0), Interaction (break_interval=0.15, place_interval=0.18, reach=8), Animals (count=60, speed=2.0, idle_chance=0.2)

### Performance Monitor
FPS counter, frame time graph, draw calls, chunk stats, entity count, memory, voxel shadow status

### Debug Visualizations
Chunk boundaries (yellow wireframe), collision boxes (green), voxel shadow volume (red), ray cast line, normals overlay, depth buffer view, light-only view

---

## Phase 7: Polish + Block Textures + Performance
**Runnable result**: Full feature parity with Swift version.

### Changes
- Block texture array (256x256 x 11 slices, r8unorm), file loading, blockTextureMask bitmask in shader
- Chunk LOD (greedy for near, simple for far)
- Arrow key turning, Option slow-walk (25%), edge-guard sneak
- Render scale with upscale pass
- WASM build verification
