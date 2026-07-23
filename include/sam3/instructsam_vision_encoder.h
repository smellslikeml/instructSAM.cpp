#pragma once

#include "sam3/gguf_model.h"

#include <cstdint>
#include <vector>

namespace sam3 {

// SAM3 vision encoder (Sam3ViTModel) — 32-layer ViT with 2D RoPE.
//
// Layer structure:
//   - 32 transformer blocks (hidden 1024, heads 16, head_dim 64, MLP 4736)
//   - Windowed attention (24×24) at all layers EXCEPT
//     [7, 15, 23, 31] which use global attention
//   - Pre-norm structure (layer_norm1 → attention → residual, then
//     layer_norm2 → mlp → residual)
//   - 2D axial rotary positional encoding on q + k inside attention
//   - GELU activation in MLP
//   - No layer_scale (layer_scale_init_value=None in config)
//
// Patch embedding: Conv2d(3, 1024, k=14, s=14, bias=False) on 1008×1008
// pixel input → 72×72 patches → 5184 tokens. Then add learnable
// position_embeddings [1, 5184, 1024], then pre-trunk LayerNorm.
//
// Input tensor namespace in our GGUF (converter maps
// model.grounding_model.model.vision_encoder → backbone.vision_backbone):
//   backbone.vision_backbone.trunk.embeddings.patch_embeddings.projection.weight
//   backbone.vision_backbone.trunk.embeddings.position_embeddings
//   backbone.vision_backbone.trunk.layer_norm.{weight,bias}
//   backbone.vision_backbone.trunk.layers.N.attention.{q,k,v,o}_proj.{weight,bias}
//   backbone.vision_backbone.trunk.layers.N.layer_norm{1,2}.{weight,bias}
//   backbone.vision_backbone.trunk.layers.N.mlp.fc1/fc2.{weight,bias}
//
// This is an in-progress port. Milestones in order (see PARITY-STATUS.md):
//   1. Skeleton + validate_all_tensors_present()   ← this session
//   2. Patch embed + pos embed + pre-trunk LN     ← this session
//   3. Layer 0 forward (windowed attn, 2D RoPE)   ← this session (attempt)
//   4. Layers 1-31 (mostly a loop)                 ← follow-up
//   5. FPN neck (3-level output)                   ← follow-up
class InstructsamVisionEncoder {
public:
    explicit InstructsamVisionEncoder(const GgufModel & model);

    // Verify every required tensor is present. Throws on first missing.
    // Returns number of tensors probed.
    size_t validate_all_tensors_present() const;

    // Debug helper — runs JUST the patch embed conv2d. Output should
    // match Sam3ViTPatchEmbeddings(pixel_values) which is [5184, 1024].
    std::vector<float> run_patch_embed_only(
        const std::vector<float> & pixel_values,
        const std::vector<int64_t> & pixel_shape
    ) const;

    // Run patch embed + pos embed + pre-trunk layer_norm on pixel_values.
    // Input: pixel_values [3, 1008, 1008] (single image, RGB, normalized).
    // Output: [72, 72, 1024] in spatial layout (ready for windowed attn).
    // Milestone 2 of the port. Layer forward passes come later.
    std::vector<float> run_prenorm(
        const std::vector<float> & pixel_values,
        const std::vector<int64_t> & pixel_shape
    ) const;

    // Run ONE encoder layer starting from spatial [72*72, 1024] hidden states.
    // Handles windowed attention (24×24) with 2D axial RoPE for layers not
    // in {7, 15, 23, 31}, or global attention (72×72 with scale=1/3) for those.
    // Structure per Sam3ViTLayer.forward: LN1 → windowed attn (with 2D RoPE
    // on Q and K) → residual → LN2 → MLP (GELU) → residual.
    std::vector<float> run_layer(int layer_idx, const std::vector<float> & hidden) const;

    // Full 32-layer forward pass. TODO after layer 0 lands.
    // std::vector<float> run(...) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
