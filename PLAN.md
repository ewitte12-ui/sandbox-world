# Sandbox World — Bevy 0.19

**This is a sequel, not a port.** The Bevy 0.18 project (`metalworld`) is a template and
reference: take its design decisions, tuning values and guardrail rationale, but write
each module the way it should be written for 0.19. Where 0.19 offers a better mechanism
than the 0.18 code used, use the 0.19 one.

The phase list below survives from the original port framing and still describes a sane
build order; read "port X" as "build X, using the 0.18 X as reference".

## Status

- **Phase 0 (scaffold): complete.** Crate `sandbox_world` builds on Bevy 0.19.0 and runs
  natively (Metal, Apple M5) with 0.19's GPU clustering and GPU preprocessing both
  reporting full support. Wasm32 target compiles clean. See "Verified against the real
  0.19 source" below for what the pre-flight retired.
- **Phase 1 (voxel core): complete.** ~4,600 lines ported; the world generates and
  renders — terrain strata, greedy-meshed chunks, the 64-layer texture array through
  `block_array.wgsl`, voxel trees, and procedural buildings. Verified by screenshot,
  not just by log. Zero errors/warnings at runtime.
- **Phase 2 (player + raycast): complete.** First-person controller with cylinder
  collision, dig/place, and crosshair. Written fresh against 0.19 rather than
  transcribed — `#[require(Camera3d, Transform)]` so spawning `Player` brings its own
  camera rig, `Single<...>` instead of `single()` unwrapping, and
  `AccumulatedMouseMotion` instead of manual mouse-event draining. The temporary
  fly-cam is gone. **26 tests pass; clippy and rustfmt are clean.**
- **Phase 3 (lighting + sky): complete.** Day/night cycle, cascaded sun shadows,
  sky colour with sunrise/sunset warming, distance fog matched to the sky, and drifting
  cloud plates. **32 tests pass; clippy and rustfmt clean.**
- **Phase 4 (animals): complete.** Six species wandering on terrain from glTF models,
  with rigged walk cycles where the models provide them. **38 tests pass; clippy and
  rustfmt clean.**
- **Phase 5 (menu, HUD, settings): complete.** Title menu, settings screen with live
  sliders and checkboxes, in-game HUD, and the world teardown fence. `GameState::Menu` is
  the default again. **40 tests pass; clippy and rustfmt clean; zero runtime warnings.**
- **Phase 6 (save/load): complete.** Versioned bincode saves, autosave on leaving a
  world, Continue / New World in the menu, and settings that actually persist.
  **46 tests pass; clippy and rustfmt clean on both targets.**
- **Phase 7 (web build): complete.** Release wasm builds and bundles; all four build
  configurations (native/wasm × debug/release) are warning-free. One real web defect
  found and documented below rather than rushed.

### Phase 7 notes

`web/build.sh` produces a working bundle: **52MB total — 36MB wasm + 16MB assets.**

**Model textures are the payload story, and were already handled.** `optimize_assets.py`
downscales the PNGs embedded in the glTF files, on the staged copy only:

| model | before | after |
|---|---|---|
| chicken | 22.6MB | 2.9MB |
| dog | 22.2MB | 2.8MB |
| raccoon | 14.0MB | 1.5MB |
| dinosaur | 3.6MB | 1.5MB |
| horse | 2.1MB | 1.2MB |
| squirrel | 5.8MB | 5.8MB |
| **total** | **70.3MB** | **15.8MB (78% smaller)** |

Squirrel is untouched — worth a look; its textures are probably in a format `sips` will
not resample.

**Release-only warnings existed and every previous check missed them.** Two pieces of
code are used solely inside `#[cfg(debug_assertions)]` blocks — a `HashSet` import in
`chunk.rs` and the `mut` on the greedy-meshing kill-switch's `ResMut` — so they warn only
in release. Every gate check up to this phase ran debug. **Check `--release` too**; it is
a genuinely different lint configuration.

#### Known defect: no block textures on web

`build_layers_data` fills each texture-array layer with a solid colour from
`BlockType::default_color()`, then overlays PNGs read with **`std::fs::read`** — which
cannot succeed in a browser. The web build therefore renders flat-shaded solid-colour
blocks: coherent and playable, but visibly poorer than native. Native is unaffected
(`loaded_textures=8/13`, the other five having no PNG).

Not fixed here, because the obvious fix is the wrong one. The source textures total
8.3MB (dirt.png alone is 4.9MB), so `include_bytes!` would add all of it to a wasm that
is already 36MB. The proportionate fix is to downscale first: these are only ever sampled
at `texture_size`, which defaults to 128 and caps at 512, so 256px copies would be
~400KB total with no visible difference — and would shrink the repo and speed native
startup too. Options, in order of preference:

1. Pre-downscale the source PNGs (they are far larger than anything samples), then
   `include_bytes!` them — works identically on every platform, no async.
2. Move `textures/` under `assets/` and load through `AssetServer`, which fetches over
   HTTP on web. Correct, but the texture array is built synchronously at startup, so this
   means making material construction wait on asset loads.

#### Wasm size

36MB after `wasm-opt -Oz` is large. It is dominated by Bevy itself, not by game code —
the `2d`/`3d`/`ui` collections pull `default_app` and `default_platform` wholesale,
including `reflect_auto_register`, which registers every reflected type and is a known
size cost. Trimming means replacing the three collections with an explicit feature list,
which is fiddly precisely because the collections are what pull the platform essentials.
Worth doing before shipping, and worth measuring rather than guessing at.

### Phase 6 notes

**The original premise was void.** The port plan said "keep the bincode format
byte-compatible so old saves load" — but a sequel has no old saves. The format is
designed for what this game needs and carries its own `FORMAT_VERSION`; a save whose
version this build does not recognise is refused rather than reinterpreted, because
bincode will happily read whatever bytes are there and a silently mangled world is worse
than a missing one.

**Only divergence is stored.** Terrain, trees and buildings are pure functions of world
position, so generation reproduces them exactly. What cannot be recovered is the player's
*departure* from that generated world: the blocks they broke and placed, where they
stood, and the time of day. An untouched world saves in 38 bytes.

`modifications` is a sorted `Vec`, not the `HashMap` it comes from — hash iteration order
is unspecified, so the same world would otherwise serialise to different bytes every
time, and saves could not be diffed or checksummed. A test pins that.

Ordering in the plugin is load-bearing and stated explicitly: autosave must run
`.before(ui::teardown_world)` and `.before(chunk_manager::reset_for_new_world)`, since
both destroy exactly what it needs to read, and `apply_pending_load` must run
`.after(player::spawn_player)`, which creates the entity it positions.

**The blanket `#![allow(dead_code)]` is gone.** It had been hiding 12 real warnings. Each
is now annotated where it lives with why it exists — the custom-block registry awaiting a
texture menu, `DevSettings` knobs that are the single source of truth for tuning even
before a consumer reads them, `taa_supported`'s hard-won analysis. One turned out to be a
genuine bug rather than dead code: **`GameSettings::save()` was never called anywhere**,
so every settings change was lost on restart. Settings now persist on leaving the menu —
not on change, which would mean a file write per frame while a slider is dragged.

Also fixed a leftover: settings were being written to `~/.metalworld_settings.json`,
named after the 0.18 project.

### Phase 5 notes

**The Parley text migration never happened, because there was nothing to migrate.**
This was the phase the original plan called the biggest risk — 3,300 lines of `ui.rs`
needing `TextFont::font` → `FontSource` and `font_size` → `FontSize`. Writing fresh
against 0.19 meant simply using `FontSize::Px(..)` from the start. The single largest
line item in the port plan evaporated once the work stopped being a port.

`bevy_ui_widgets` (stabilised in 0.19) supplies `Button`, `Slider` and `Checkbox` as
**headless** widgets: they own pointer, drag, focus and keyboard behaviour and emit
`Activate` / `ValueChange` notifications, while every pixel of styling is ours. That is
why the whole menu, settings screen and HUD is a few hundred lines rather than a few
thousand — none of it re-implements drag maths or focus handling.

**Three bugs found, all by the state-cycle probe rather than by tests.**

1. **The second world came up almost empty** — 1061 entities in the first session, 93 in
   the second. `ChunkManager` is a *resource*, so it survived the teardown that despawned
   its chunk entities, and entered the next session still believing every chunk was
   loaded, its `chunks` map pointing at dead entities. `clear_all()` already existed for
   exactly this and was one of the dead-code warnings — nothing called it. Session resets
   now also clear `WorldReady` (a stale `true` would run the next session's physics before
   its terrain existed) and bump `WorldInstanceId`.
2. **Teardown logged a warning per cloud plate.** `despawn` is recursive, so despawning a
   parent takes its children with it, and the loop then tried to despawn each child again.
   Teardown now despawns *roots* (`With<WorldScoped>, Without<ChildOf>`) and lets the
   hierarchy do the rest: 1031 roots for 1061 entities, the difference being the 30 clouds.
3. An **anti-stuck lift** was added defensively: feet inside solid geometry are raised to
   the top of that block. Wall resolution cannot help there — a player at a block's exact
   centre has no "out" direction, which is the case the Phase 3 fix deliberately declines
   to guess at. Reachable by spawning onto a building floor, since buildings are stamped in
   during chunk generation and the spawn search cannot see them.

Honest note on (3): I added it after misreading a diagnostic. The column probe indexed
blocks *relative* to `floor(foot_y)`, and since a standing player's feet sit exactly on a
block boundary, float rounding flips that index by one — making "standing on grass" print
identically to "buried in it". The probe now reports absolute block Y alongside the exact
foot height. The lift is still worth keeping as a no-softlock safety net, but it was not
fixing an observed bug.

**The state-cycle probe (`SW2_CYCLE=1`) earned its keep.** It drives
Menu → Gameplay → Menu → Gameplay and logs surviving world entities at each transition.
Every one of the bugs above is invisible to a unit test of the teardown system: they are
about what *other* systems do across a session boundary. The count must return to 0 in the
menu and both sessions must reach the same entity count — which they now do (0 / 1061 / 0 /
1061).

### Phase 4 notes

Inspecting the models first settled the design: `horse`, `raccoon` and `dinosaur` ship a
rigged clip; `chicken`, `dog` and `squirrel` are static meshes. The 0.18 code met that
split by building procedural ellipsoid-and-cylinder skeletons (~500 lines) for the
static ones. Here they instead get a gait bob on the scene root — enough that they do
not slide across the ground like furniture, for a fraction of the machinery. A test
asserts the clip declarations match the models that actually ship one, since asking for
a missing glTF animation yields a handle that never resolves and an animal that silently
stands still forever.

Structurally this is the biggest departure from the reference. The 0.18 version kept
animals in a `Vec<AnimalData>` plus a flat `part_entities: Vec<Entity>` indexed
`animal_index * 5 + part`, with components holding indices back into those arrays. Here
each animal is simply an entity: `Animal` + `Transform` + `WorldAssetRoot`. No parallel
arrays, so no way to leave a stale index pointing at a despawned animal.

Animation attachment is an **observer** on `Add<AnimationPlayer>` that walks *up* the
`ChildOf` chain via `iter_ancestors`. glTF scenes spawn their hierarchy asynchronously,
so the player entity appears several frames after the animal and is a descendant of it.
The 0.18 code searched every animal's descendants every frame looking for new players —
quadratic, and running forever. Reacting to the insertion costs one upward traversal.

**Bug found and fixed: collision tunnelling.** Adding 60 glTF models introduced a
load hitch, and the player fell straight through the world into the terrain (the
diagnostic column read `Dirt/Dirt/Dirt/Grass`). Ground detection probed a fixed three
blocks below the feet, but Bevy clamps the frame delta to 0.25s, which at terminal
velocity still covers ~15 blocks. The probe simply missed the floor. Ground search is
now **swept** over the path the feet actually travelled this frame, with two regression
tests: one that a long single-frame fall still lands, and one that the sweep does not
become a magnet that snaps a high-falling player down to distant terrain.

Worth noting the shape of that bug: it was latent from Phase 2 and only surfaced when an
unrelated phase made frames slow. Fixed probe depths are a tunnelling bug waiting for a
frame spike.

### glTF in 0.19

Scenes load as `WorldAsset` and spawn via `WorldAssetRoot` — the 0.18 `Scene`/`SceneRoot`
pair renamed when `bevy_scene` was split into `bevy_world_serialization`. Asset labels
are unchanged: `GltfAssetLabel::Scene(0)` / `GltfAssetLabel::Animation(0)`.

The models total 76MB (chicken and dog are ~23MB each), which matters for the web
payload; `web/build.sh` already downscales embedded textures for the browser bundle.

### Phase 3 notes

**Deliberately did NOT adopt `bevy_light::Atmosphere`.** 0.19's physically-based sky
would look better than a lerped gradient, but `AtmospherePlugin` requires compute
shaders and storage textures and refuses to load without them — WebGL2 has neither.
Adopting it would mean the browser silently losing its sky while desktop kept one.
This is the counter-example to "prefer the 0.19 way": check that the new mechanism
actually fits the target platforms first. Revisit if the web build moves to WebGPU.

The sun moving must never trigger a remesh, and doesn't: meshes carry raw 0..15 light
levels and the day/night curve reaches them through one uniform write
(`light_params.x`). That write is quantised to 1/256 so a slowly drifting sun does not
mark the material changed every frame and force its bind group to be re-prepared.

Fog colour is driven from `ClearColor`, which the sky writes — so the haze at the edge
of the loaded world always matches the sky behind it instead of glowing against a night
sky. That is also what turns chunk streaming from a visible wall of popping geometry
into distance haze.

**Bug found and fixed: spawn drift.** The player was ending a no-input run one block
away from where it spawned. Two independent defects were fixed:

1. `resolve_walls` chose its push direction with `f32::signum`, which reports `+0.0` as
   `+1.0`. A player standing exactly on a block's centre line — which the spawn at
   `(0.5, 0.5)` is, exactly — would be ejected a full block in an arbitrary direction.
   Now a zero offset defers instead of guessing, with a regression test
   (`a_centred_block_never_teletports_the_player`).
2. Spawn point was hardcoded to the origin, which tree placement can and does put a
   trunk through. `find_spawn` now searches outward for a column clear of trees, using
   the new `terrain::is_clear_of_trees` — necessary because trees are stamped in during
   chunk generation and are therefore invisible to `ChunkManager::block_at` until the
   chunk actually loads.

Honest caveat: the drift no longer reproduces, but the runtime evidence does not confirm
which of the two was the actual cause — the diagnostic shows the spawn column is
`Grass / Air / Air / Air`, i.e. genuinely clear, which argues against the tree theory for
*this* seed. Both fixes are correct on their own merits and both are now covered.

### Phase 2 notes

What carried over from the reference is the *design*, not the code:

- **`SurfaceType` classification** (Ground / Wall / Ceiling), which is how
  `guardrails/06` is actually enforced — a wall blocks horizontal movement and never
  moves the player vertically, and the same block can be Ground from above and Wall from
  the side.
- **`STEP_UP_THRESHOLD = 0.98`** and its reasoning: one block minus a float-safe margin
  makes the rule total, so no height is left "neither walkable nor jumpable" — the
  no-half-affordances rule from `guardrails/02`. A `classification_is_total_across_a_
  block_height` test pins this.
- **Verticality invariants**: terminal velocity as a *collision-safety* limit (not game
  feel), a void kill plane, and guaranteed recovery so a fall costs time and never the
  session.
- **DevSettings as the single source of tuning.** Constants in `player.rs` are geometric
  or physical invariants only.

Collision resolves vertical before horizontal deliberately: settling onto the ground
first means a block that was a Wall mid-fall is correctly seen as Ground once standing
on it. The movement systems are `.chain()`ed for the deterministic update order
`guardrails/04` requires.

### Phase 1 notes

Scope was the dependency closure of the voxel core, which is wider than the original
plan's four modules: `chunk.rs` needs `buildings` and `voxel_light`, and
`chunk_manager.rs` needs `settings` (+`platform`) plus two resources from `dev_tools`
and `WorldReady` from `player`. `dev_tools.rs` and `player.rs` therefore exist as
faithful *partial* ports carrying only those types; later phases extend them.

Three defects found, all now fixed — see "Bugs found during the port" below.

Temporary scaffolding in `main.rs`, all tagged in-comment with its removal phase:
`fly_camera` + `spawn_fly_camera` (Phase 2), a stand-in `DirectionalLight` (Phase 3,
when `lighting.rs` brings the real sun), `screenshot_after_warmup` (env-gated
`SW2_SCREENSHOT=<path>`, for headless visual verification), `mark_world_ready`, and
`GameState::Gameplay` temporarily marked `#[default]` so the app boots into the world
before the menu exists. `#![allow(dead_code)]` is on until Phase 6.


Recreate "Sandbox World" (the `metalworld` crate at `~/Documents/claude/metalworld-bevy`,
Bevy 0.18, ~15k lines / 18 modules) as a new project on **Bevy 0.19** (released 2026-06-19).

## Strategy

**Port, don't rewrite.** Most of the codebase is engine-agnostic game logic (terrain
noise, chunk storage, greedy meshing, voxel flood-fill lighting, DDA ray casting, the
bincode save format) and copies over nearly unchanged. The Bevy 0.19 churn is
concentrated in three places: app wiring (resources-as-entities), the material/shader
layer, and UI text. Port modules in dependency order with a running app at every phase,
using the old repo as the reference implementation.

Carry over as-is: `guardrails/` docs, `assets/` (textures, shader), `web/` build
scripts, the Cargo profile tuning (wasm-size rationale), and the plugin ordering
contract documented in old `main.rs:749-765`.

## Bugs found during the port

**1. Bevy 0.19 mesh allocator errors on every zero-vertex mesh (upstream bug).**
`bevy_render::mesh::allocator::allocate_meshes` skips slab allocation for an empty mesh
(`if vertex_buffer_size == 0 { continue; }`) but its copy loop has no matching guard, so
`copy_element_data` runs against a never-allocated key and logs
`Use-after-free: attempted to copy element data for an unallocated key`. A voxel world
produces empty meshes constantly — every all-air chunk above the surface, every
fully-occluded interior chunk — which produced **~100 errors/second** (1,936 in a 20s
run) and buried the log.

Fixed on our side by never creating a mesh asset for an empty chunk (the EMPTY-MESH
CONTRACT in `chunk_manager.rs`): a chunk with no geometry carries no `Mesh3d` at all.
The remesh path handles all four transitions, including the one that matters for
gameplay — placing a block in an all-air chunk must attach `Mesh3d`+`MeshMaterial3d`,
and digging out the last block removes them. This is also a genuine optimisation
(fewer assets, fewer renderable entities). Worth reporting upstream.

**2. `Assets::get_mut` now returns `AssetMut<A>`** — predicted, and it duly appeared at
the two texture-array mutation sites. `Some(image)` → `Some(mut image)`; keep `get_mut`
(not `get_mut_untracked`) because the write is what flags the GPU re-upload.

**3. False-positive "100% AIR near origin" warning (inherited from 0.18).** The
diagnostic flagged any all-air chunk in the 3×3×3 neighbourhood of the origin, but
`surface_y(0,0) == 5`, so the entire `pos.1 == 1` layer (world y 16–31) is legitimately
empty sky. It fired every run with a message blaming save data — actively misleading for
Phase 6. Now suppressed for chunks sitting above the terrain surface.

Also worth noting: **edition 2024**, not Bevy, caused one of the three compile errors
(a match-ergonomics tightening on `.filter(|(_, &ent)| ...)`).

### Verification harness caveat (macOS)

`screenshot_after_warmup` captures the real window's swapchain, and macOS stops
presenting an occluded window — so a capture taken while the terminal has focus returns
an **all-black frame, including the background**. That is diagnostic gold, because it is
distinguishable from a genuine render failure: a real "nothing drew" frame still shows
the `ClearColor` (`#2b2c30`), while an unpresented capture is `#000000` everywhere.
Setting `WindowLevel::AlwaysOnTop` during capture runs did not reliably fix it.

Do not read a black capture as a regression without checking the background colour
first, and cross-check against the metrics the same system logs (mesh entity count,
visible count, light count). If reliable automated capture is needed later, render a
second camera to an offscreen `RenderTarget::Image` instead of the window.

Also found while chasing this: **`AmbientLight` is a per-camera component in 0.19, not a
`Resource`.** Nothing in Phase 1 depends on it, but `lighting.rs` may in Phase 3, and
our cameras currently have none.

## Verified against the real 0.19 source (2026-08-10)

Checked in `~/.cargo/registry/src/*/bevy_*-0.19.0/`, which has both 0.18.1 and 0.19.0
unpacked side by side:

- **Bevy 0.19.0 is on crates.io.** MSRV **1.95.0** (we have exactly 1.95.0), edition
  **2024**. `default = ["2d", "3d", "ui", "audio"]`.
- **The block shader should port unchanged.** Every symbol `block_array.wgsl` imports
  still exists in 0.19 with the same path: `pbr_fragment::pbr_input_from_standard_material`
  (identical signature, `(in: VertexOutput, is_front: bool) -> PbrInput`),
  `pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing}`,
  `forward_io`, `prepass_io`, `pbr_deferred_functions::deferred_output`.
  `#{MATERIAL_BIND_GROUP}` is still the extension-binding idiom (0.19's own
  `forward_decal.wgsl` uses it). **Risk 2 below is largely retired.**
- **`ExtendedMaterial` / `MaterialExtension` survived the `bevy_material` split**
  unchanged — still `bevy_pbr::extended_material`, and diffing the trait surface
  0.18.1 → 0.19.0 shows only a cosmetic import-path edit inside Bevy's own file, no
  signature change for implementors.
- **`IsResource` exists** (`bevy_ecs::resource`), so `Without<IsResource>` is the real
  filter for the teardown audit.
- **`Assets::get_mut` → `Option<AssetMut<'_, A>>`** as expected. Note `get_mut_untracked`
  → `Option<&mut A>` also exists, but the day/night uniform *wants* change tracking so
  the material re-uploads — use `get_mut`.
- **`FontSource` and `FontSize`** confirmed in `bevy_text::text`.
- **The mesh-building API `chunk.rs` uses is unchanged**: `Mesh::new(PrimitiveTopology,
  RenderAssetUsages)`, `insert_attribute`, `attribute`, `insert_indices`, and all five
  attributes it writes (`ATTRIBUTE_POSITION/NORMAL/COLOR/UV_0/UV_1`). Only the `use`
  paths may need adjusting for re-export moves.
- Shader file set is otherwise stable: 0.19 adds `unpack_bins.wgsl`, drops
  `pbr_transmission.wgsl` (moved, per the `ScreenSpaceTransmission` migration note).
  Neither is used here.

Net effect: the two scariest items (shader/material layer, render internals) look
benign; the resources-as-entities teardown audit and the text migration remain the real
work.

## What changed in Bevy 0.19 that affects this codebase

### Must handle (breaking)

1. **Resources are now entity components.** `#[derive(Resource)]` implements
   `Component`; a type can no longer derive both. Broad queries (`Query<Entity>`,
   `Query<()>`, `Query<Option<&T>>`) now match resource entities and need
   `Without<IsResource>`.

   **Audited the 0.18 tree — this is narrower than feared.** Teardown is already
   marker-scoped, not a blanket sweep: `cleanup_world` queries
   `Query<(Entity, &WorldScoped), Without<BackgroundPlate>>`, and resource entities
   will never carry `WorldScoped`. So teardown does **not** eat global state.

   Exactly one offender exists: `dev_tools.rs:827`, `all_entities: Query<Entity>` in
   the overlay leak-diagnostic. Two problems there, and the second is the serious one:
   - it will silently count every resource entity into `pre_entity_total`, reporting
     phantom leaks;
   - that same system also takes `Res<MenuState>` and `Res<FrameStartTime>`, so the
     unfiltered `Query<Entity>` now *overlaps* those resource reads — this is the
     access-conflict case the migration guide warns about, and it can panic at
     schedule build rather than merely miscount.

   Fix is one filter: `Query<Entity, Without<IsResource>>`. Still re-audit as each
   module lands, but the port does not need a teardown redesign.
2. **Text system rewritten (cosmic-text → Parley).** `TextFont::font` is now a
   `FontSource` enum, `font_size` is `FontSize::Px(..)`. `ui.rs` (3,316 lines) and
   `dev_tools.rs` are text-heavy — this is the largest mechanical migration, but it's
   find-and-replace-shaped.
3. **Feature flags reshuffled.** Audio is no longer implied by `3d`/`ui` (we don't use
   audio — fine); `bevy_window`/`bevy_input_focus`/`custom_cursor` moved into new
   collections. Rebuild `Cargo.toml` feature lists against the 0.19 docs for both the
   native and wasm targets rather than copying blindly; verify `webgl2`/`web` names.
4. **`Assets::get_mut` returns `AssetMut<A>`.** The day/night cycle mutates the
   material uniform (`light_params`) every frame through `Assets<ExtendedMaterial<..>>`
   — adjust call sites. Side benefit: change events now fire only on real mutation.
5. **`bevy_material` extracted from `bevy_pbr`.** `AlphaMode`, `MaterialProperties`
   moved; `Hdr` moved to `bevy_camera`. Validate `block_array.wgsl`'s
   `#import bevy_pbr::...` paths against the 0.19 shader tree **first thing in
   Phase 1** — the whole visual pipeline gates on this shader.
6. ~~**`Image` pixel access returns `Result<_, TextureAccessError>`.**~~ **Does not
   apply here.** The 64-layer array build (`chunk_manager.rs:667`) uses
   `Image::new_uninit` + direct `data` assignment, never `pixel_bytes` /
   `pixel_data_offset`. `new_uninit`, `TextureDataOrder`, and `TextureDimension::D2`
   are all unchanged in 0.19.
7. **Render graph replaced by ECS schedules.** We have no custom render nodes
   (everything goes through `ExtendedMaterial`), so this should be transparent — but
   it's the riskiest area to verify early since it's a wholesale internal rewrite.
8. **Lights: `shadows_enabled` → `shadow_maps_enabled`**, consistently across
   `DirectionalLight`, `PointLight`, and `SpotLight` (all now in `bevy_light`). Each
   also gains a `contact_shadows_enabled: bool`. Hits `lighting.rs:137` (the sun) and
   `lighting.rs:360` (lanterns). Found by compiling, not by the migration guide — treat
   the guide as incomplete and let the compiler enumerate the rest.
9. **Misc renames:** `Skybox::image` is now `Option<Handle<Image>>`;
   `bevy_scene` → `bevy_world_serialization` (we don't use scenes — no-op);
   `DefaultErrorHandler` → `FallbackErrorHandler`.

### Optional adoption (after parity, Phase 8)

- **`EditableText`** — upstream text input (cursor, selection, clipboard). Replace any
  hand-rolled text entry (world-name field).
- **`bevy_ui_widgets` / feathers (now stable)** — sliders, scrollbars, dropdowns,
  number inputs could shrink the hand-rolled settings menu substantially.
- **`bsn!` scene macro** — ergonomic UI-tree construction; code-only in 0.19.
- **`DiagnosticsOverlayPlugin`** — may replace part of the dev-tools perf monitor.
- **Contact shadows** (`contact_shadows_enabled` per light + `ContactShadows` on
  camera) — cheap shadow detail; add as a settings toggle.
- **App Settings framework (`SettingsPlugin`)** — could replace hand-rolled JSON
  settings persistence; only adopt if it supports our native+wasm split.
- **`Query::contiguous_iter`** — SIMD-friendly iteration if animal/lighting perf ever
  needs it.
- **GPU perf wins are free:** 0.19's batching/culling work (~2.6x on `many_cubes`) and
  20x faster light clustering directly benefit a many-chunk voxel scene.

## Project setup (Phase 0)

- New crate `sandbox_world` in `~/Documents/claude/sw2`; `git init`.
- Check Bevy 0.19's MSRV / edition requirement; bump toolchain if needed.
- `Cargo.toml`: bevy 0.19 with per-target features rebuilt per item 3 above. Keep the
  three-way dependency split (base / native-only / wasm-only) and its comments, and
  the `[profile.release]` tuning (fat LTO, codegen-units 1, strip, **no**
  `panic="abort"` — same reasons as before).
- Other deps: `noise`, `block-mesh 0.2` (engine-agnostic — verify it still compiles on
  current rustc; it has no Bevy coupling), `serde`, `serde_json`, `bincode 1` (save
  compat), `image`, `rand 0.8` + `getrandom/js` for wasm (defer the rand 0.9 /
  getrandom 0.3 migration — it changes the wasm backend story and buys nothing).
- Copy `guardrails/`, `assets/`, `web/`, `.github/`; write `CLAUDE.md` (adapted from
  the old one — same conventions, plugin-per-domain, DevSettings for constants).
- **Milestone:** empty Bevy 0.19 app opens a window with a camera, native **and** wasm.

## Port phases (dependency order = old plugin ordering contract)

Each phase ends with the app running natively; wasm is re-checked at phases 1, 5, 7.

| Phase | Modules | Milestone / notes |
|---|---|---|
| 1. Voxel core | `block_types`, `terrain`, `chunk`, `chunk_manager`, `block_array.wgsl`, texture-array build | Fly-cam over generated terrain. Validate WGSL imports (breaking item 5) and mesh contract (UV_0 tile-local, UV_1 = layer + sky light, COLOR = AO/block light) before porting anything downstream. |
| 2. Player | `player`, `ray_cast` | FPS movement, collision, dig/place. Logic ports clean; input/window API is stable-ish but re-check cursor-grab calls. |
| 3. Lighting & sky | `voxel_light`, `lighting`, `sky` | Day/night, lanterns, budgeted relaxation queues. Flood-fill is pure logic; the material-uniform update hits breaking item 4. `Skybox::image` is now `Option`. |
| 4. World content | `buildings`, `animals` | Buildings stay baked into chunk generation (no plugin). Animals after ChunkManager (ground-height queries). |
| 5. UI & settings | `settings`, `ui`, `dev_tools`, `platform` | The text-migration bulk (breaking item 2). Port hand-rolled UI 1:1 first; widget adoption waits for Phase 8. |
| 6. Persistence | `save_load` | Keep the bincode format byte-compatible so old saves load; add a version header if any struct must change. Auto-save on exit, load waits for player entity (old ordering contract). |
| 7. Web build | `web/build.sh`, payload check | Wasm feature wiring, canvas setup (`#canvas`, fit-to-parent), compare payload size against the old build. |
| 8. Modernization (optional) | — | Items from "Optional adoption" above, one at a time, behind settings toggles where user-visible. |

The `main.rs` state machine (Menu/Gameplay states, teardown fence, `WorldSpawnSet`,
`WorldInitPending`) is rebuilt in Phase 1 and grows with each phase — it's where the
resources-as-entities audit (breaking item 1) lives.

## Risks, ordered

1. **Text migration volume** (item 2) — low risk per change, high volume across
   `ui.rs` (3.3k lines) + `dev_tools.rs`; mechanical but the largest single chunk of
   work in the port. Now the top risk by effort.
2. **Render internals rewrite** (item 7) — no action expected, but unknown-unknowns;
   the Phase 1 milestone smokes it out.
3. ~~**Teardown vs. resource entities**~~ (item 1) — **downgraded after audit.**
   Teardown is marker-scoped and safe; one `Query<Entity>` in `dev_tools.rs` needs
   `Without<IsResource>`. Still write the Menu→Gameplay→Menu regression test.
4. ~~**Shader import drift**~~ (item 5) — **retired.** Verified every imported symbol
   and the `#{MATERIAL_BIND_GROUP}` idiom are unchanged in 0.19.
5. **`block-mesh 0.2` bitrot** — unmaintained but dependency-light; fallback is
   vendoring it or hand-rolling the single-axis greedy merge we actually use.
6. **Wasm feature churn** (item 3) — caught at Phase 1/7 wasm checks.

## Verification

- Port the old test suite alongside each module; `cargo test`, `cargo clippy`,
  `cargo fmt --check` green per phase.
- Guardrails in `guardrails/` remain the design contract (traversal, camera, movement
  contact, tuning rules).
- Manual smoke per milestone: native run + (phases 1/5/7) wasm run via `web/serve.sh`.
- Save-compat check in Phase 6: load a save file produced by the 0.18 build.
