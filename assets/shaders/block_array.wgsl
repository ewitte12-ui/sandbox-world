// Block-array material: StandardMaterial extension that samples the block
// texture from a 2D array texture (one layer per block type) instead of a
// tiled atlas.
//
// WHY an array instead of an atlas:
//   - no UV inset hacks: Linear filtering can never bleed into a
//     neighboring block's texels, because each block type is its own layer
//   - correct per-layer mip chains (no cross-tile contamination)
//   - greedy-merged quads can tile with UV repeat (uv in 0..w), which an
//     atlas fundamentally cannot do
//
// Mesh contract (see chunk.rs build_mesh):
//   UV_0  = tile-local coordinates, 0..1 per block (0..w on greedy runs,
//           sampled with a repeating sampler)
//   UV_1.x = block layer index (constant per quad)
//   COLOR  = face brightness x baked vertex AO
//
// The lantern glow lives here too: the base StandardMaterial carries the
// warm emissive constant, and this shader zeroes it for every layer except
// the lantern's. This replaces the old separate emissive atlas texture.

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

// Extension bindings start at 100 (0-99 reserved for StandardMaterial).
@group(#{MATERIAL_BIND_GROUP}) @binding(100) var block_layers: texture_2d_array<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(101) var block_layers_sampler: sampler;

// Must match BlockType::LANTERN.index() in block_types.rs.
const LANTERN_LAYER: i32 = 8;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // Standard PBR input: applies the base material + vertex color (which
    // carries face brightness x baked AO).
    var pbr_input = pbr_input_from_standard_material(in, is_front);

#ifdef VERTEX_UVS_B
    let layer = i32(round(in.uv_b.x));
#else
    let layer = 0;
#endif

    // Sample outside any control flow so derivatives (mip selection,
    // anisotropy) stay valid.
    let block_color = textureSample(block_layers, block_layers_sampler, in.uv, layer);
    pbr_input.material.base_color = pbr_input.material.base_color * block_color;

    // Emissive constant applies to the lantern layer only.
    if (layer != LANTERN_LAYER) {
        pbr_input.material.emissive = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}
