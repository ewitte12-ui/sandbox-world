# CLAUDE.md

## Project Overview

MetalWorld is a Minecraft-style voxel game being ported from Swift/Metal to Bevy (Rust) for multi-platform support. The full implementation plan is in `.claude/skills/metalworld-plan/SKILL.md`.

## Build & Run Commands

- **Build:** `cargo build`
- **Run:** `cargo run`
- **Check (fast compile check):** `cargo check`
- **Run tests:** `cargo test`
- **Run a single test:** `cargo test test_name`
- **Lint:** `cargo clippy`
- **Format:** `cargo fmt`
- **Format check:** `cargo fmt -- --check`

## Architecture

- **Engine:** Bevy 0.18 with wgpu backend
- **Voxel storage:** 16x16x16 chunks in HashMap, greedy meshed with block-mesh-rs
- **Shaders:** WGSL (assets/shaders/)
- **Entity system:** Bevy ECS
- **UI:** Bevy UI nodes

## Conventions

- One Plugin per system domain (ChunkPlugin, PlayerPlugin, LightingPlugin, etc.)
- Resources for global state, Components for per-entity data, Events for one-shot communication
- Systems named as verb_noun (load_chunks, update_animals)
- No unwrap() on user input or file I/O
- All tweakable constants in DevSettings resource, not hardcoded
- Source port reference: ~/Documents/metalworld/ (Swift version)

## File Structure

```
src/
├── main.rs              # App setup, plugin registration
├── block_types.rs       # BlockType enum, colors, block-mesh traits
├── terrain.rs           # terrainHeightAt(), noise, naturalBlockAt()
├── chunk.rs             # Chunk storage, greedy mesh generation
├── chunk_manager.rs     # Chunk load/unload, modifications, surface cache
├── player.rs            # FPS camera, movement, collision, interaction
├── ray_cast.rs          # DDA ray casting
├── buildings.rs         # Procedural building placement
├── animals.rs           # Animal entities, AI, animation
├── trees.rs             # Tree placement, geometry
├── lighting.rs          # Sun cycle, lanterns, voxel shadows
├── sky.rs               # Procedural sky + clouds
├── ui.rs                # Settings menu, HUD
├── settings.rs          # GameSettings, JSON persistence
├── dev_tools.rs         # Developer tab: tweakable constants, perf monitor, debug viz
└── save_load.rs         # Save/load game state
```
