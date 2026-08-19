# CLAUDE.md

## Project Overview

Sandbox World is a Minecraft-style voxel game on **Bevy 0.19**. The phased build plan is
in `PLAN.md`.

**This is a sequel, not a port.** The Bevy 0.18 project at
`~/Documents/claude/metalworld-bevy/` (crate `metalworld`) is a **template and reference**
— a working prior version to learn from, not a spec to reproduce. Take its design
decisions, tuning values, and hard-won guardrail rationale; do not preserve its code
shape.

**Where Bevy 0.19 does something better, do it the 0.19 way.** Required components,
observers, `Single<...>` queries, relationships, BSN, the stabilized widget set, contact
shadows — prefer the current idiom over transcribing a 0.18 workaround for a limitation
that no longer exists. A module that reads as if it were written for 0.19 from scratch is
the goal. When in doubt, check what the old code was working around before copying it.

The authoritative design constraints are `guardrails/` (traversal, camera stability,
movement contact, tuning rules) — those are engine-independent and carry forward intact.

## Build & Run Commands

- **Build:** `cargo build`
- **Run:** `cargo run`
- **Check (fast compile check):** `cargo check`
- **Run tests:** `cargo test`
- **Run a single test:** `cargo test test_name`
- **Lint:** `cargo clippy`
- **Format:** `cargo fmt`
- **Format check:** `cargo fmt -- --check`

`cargo fmt` and `cargo clippy` are real gates — keep them clean. (An earlier revision of
this file told you to skip them to preserve line-for-line correspondence with the 0.18
tree. That was based on treating this as a port; it is not one, so the exemption is
withdrawn.)

**Check `--release` as well as debug, and wasm as well as native.** They are different
lint configurations, not just different optimisation levels: code used only inside
`#[cfg(debug_assertions)]` warns only in release, and code behind
`#[cfg(not(target_arch = "wasm32"))]` warns only on wasm. All four combinations are
currently warning-free; keep them that way:

```
cargo clippy && cargo clippy --release
~/.cargo/bin/cargo check --target wasm32-unknown-unknown
~/.cargo/bin/cargo check --release --target wasm32-unknown-unknown
```
- **Web build:** `web/build.sh` then `web/serve.sh`

### Toolchain gotcha (this machine)

There are two Rust toolchains installed and **the one on `PATH` cannot build wasm**:

| | `cargo` on PATH | `~/.cargo/bin/cargo` |
|---|---|---|
| source | Homebrew | rustup |
| version | 1.95.0 | 1.97.1 |
| targets | `aarch64-apple-darwin` only | + `wasm32-unknown-unknown` |

Native builds work with either (Bevy 0.19's MSRV is 1.95.0 — the Homebrew toolchain is
exactly at the floor). **Any wasm build must use `~/.cargo/bin/cargo`.** `web/build.sh`
already resolves this itself and fails with a clear message otherwise; invoke it
directly rather than running `cargo build --target wasm32-unknown-unknown` by hand.

`wasm-bindgen` CLI must match the `wasm-bindgen` crate version in `Cargo.lock` exactly
(Bevy 0.19 pulls **0.2.127**); build.sh enforces this. Install with
`~/.cargo/bin/cargo install wasm-bindgen-cli --version <lockfile version>`.

## Architecture

- **Engine:** Bevy 0.19 with wgpu backend; edition 2024, MSRV 1.95
- **Voxel storage:** 16x16x16 chunks in HashMap; face culling via block-mesh-rs,
  single-axis greedy merge (default on), per-vertex AO baked at mesh time; buildings
  baked into chunk generation
- **Voxel lighting:** per-voxel sky+block flood fill (voxel_light.rs), packed nibbles on
  `Chunk::light`; per-chunk recompute + relaxation across seams via budgeted queues in
  chunk_manager; baked per-vertex at mesh time, day/night via a material uniform (no
  remesh)
- **Rendering:** `ExtendedMaterial<StandardMaterial, BlockArrayExtension>` sampling a
  64-layer texture array via `assets/shaders/block_array.wgsl`; mesh carries UV_0
  (tile-local) + UV_1.x (layer index) + UV_1.y (sky light) + COLOR.a (block light)
- **Entity system:** Bevy ECS
- **UI:** Bevy UI nodes

## Bevy 0.19 porting notes

Traps specific to this port (full list in PLAN.md):

- **Resources are entity components now.** `#[derive(Resource)]` implies `Component`;
  a type cannot derive both. Broad queries (`Query<Entity>`, `Query<()>`) now match
  resource entities — the world-teardown systems MUST filter `Without<IsResource>` or
  they will despawn global state.
- **Text rewrite (cosmic-text -> Parley).** `TextFont::font` is a `FontSource`;
  `font_size` is a `FontSize` (`FontSize::Px(16.0)`, not `16.0`).
- **`Assets::get_mut` returns `AssetMut<A>`** — affects the per-frame material uniform
  update in the day/night cycle.
- **`bevy_material` split out of `bevy_pbr`**; validate WGSL `#import bevy_pbr::...`
  paths against the 0.19 shader tree.
- **`Image` pixel access returns `Result<_, TextureAccessError>`** — affects texture
  array assembly.
- Cargo features: `default-features = false` + `["2d", "3d", "ui"]` (default minus
  audio). `webgl2` now comes from `default_platform`, so it is no longer listed for
  wasm explicitly.

## Conventions

- One Plugin per system domain (ChunkPlugin, PlayerPlugin, LightingPlugin, etc.)
- Resources for global state, Components for per-entity data, Events for one-shot
  communication
- Systems named as verb_noun (load_chunks, update_animals)
- No unwrap() on user input or file I/O
- All tweakable constants in DevSettings resource, not hardcoded
- Save format is bincode 1.x and must stay byte-compatible with 0.18 saves; any struct
  change needs a version header

## Plugin ordering contract

Carried over from the 0.18 build — this ordering is load-bearing, not cosmetic:

- **SettingsPlugin first:** GameSettings must exist before anything reads
  render_distance, texture_size, etc.
- **ChunkManagerPlugin before PlayerPlugin:** ChunkManager must exist at Startup so
  player collision can fall back to terrain generation.
- **Buildings have no plugin:** baked into chunk generation
  (`buildings::place_buildings_in_chunk`, called by `Chunk::generate`).
- **AnimalPlugin after ChunkManager:** animals query ChunkManager for ground height.
- **LightingPlugin after PlayerPlugin:** lighting reads the camera position PlayerPlugin
  sets. Both run in Update with no cross-plugin ordering, so lighting reads the
  *previous* frame's camera position (1-frame lag — accepted, see
  `guardrails/04_camera_guardrails_3d.txt`).
- **SaveLoadPlugin after PlayerPlugin + ChunkManagerPlugin:** auto_load_game waits for
  the player entity before applying save data.
- **UiPlugin last:** reads state from all other systems for display.

## Agent Usage

Delegate work to specialized agents instead of doing everything in the main thread:

- **Research agent** (when needed): non-trivial codebase exploration, locating symbols
  across files, or mapping 0.18 source to the 0.19 port. Skip for trivial single-file
  reads. Launch via `Agent` with `subagent_type: "Explore"` (read-only) or
  `"general-purpose"` (broader research).
- **Critic agent** (whenever anything is changing): before finalizing any code change,
  spawn a critic to review the diff for correctness, regressions, and consistency with
  `guardrails/` and the conventions above. Launch via `Agent` with `subagent_type:
  "general-purpose"`, briefing it as an independent reviewer.
- **Coding agent**: for multi-file or non-trivial implementation work, delegate the
  implementation rather than editing in the main thread. Launch via `Agent` with
  `subagent_type: "general-purpose"`.

Use a frontier model for these roles; do not downgrade to a small/fast model.

## File Structure

```
src/
├── main.rs              # App setup, plugin registration, Menu/Gameplay state machine
├── block_types.rs       # BlockType enum, colors, block-mesh traits
├── terrain.rs           # terrainHeightAt(), noise, naturalBlockAt(), voxel tree placement
├── chunk.rs             # Chunk storage, greedy mesh generation
├── chunk_manager.rs     # Chunk load/unload, modifications, remesh queues
├── player.rs            # FPS camera, movement, collision, interaction
├── ray_cast.rs          # DDA ray casting
├── buildings.rs         # Procedural building placement
├── animals.rs           # Animal entities, AI, animation
├── lighting.rs          # Sun cycle, lanterns, voxel shadows, render-scale blit
├── sky.rs               # Procedural sky + clouds
├── ui.rs                # Settings menu, HUD
├── settings.rs          # GameSettings, JSON persistence
├── dev_tools.rs         # Developer tab: tweakable constants, perf monitor, debug viz
├── platform.rs          # Native/wasm platform split
├── voxel_light.rs       # Per-voxel flood-fill lighting (sky + block channels)
└── save_load.rs         # Save/load game state
```

Modules land in phase order (see PLAN.md); the tree above is the target shape, not what
exists today.
