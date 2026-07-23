#pragma once

#include "sam3/gguf_model.h"

#include <cstdint>
#include <vector>

namespace sam3 {

// Prompt cross-attention block that fuses prompt features into
// encoder_hidden_states inside InstructSAM's mask decoder. Mirrors:
//
//   normed = LayerNorm(encoder_hidden_states)
//   attn   = Sam3Attention(Q=normed, K=V=prompt_features, mask=bidirectional)
//   encoder_hidden_states = residual + attn
//
// Weights used (from our GGUF under `segmentation_head.*` after the
// mask_decoder → segmentation_head remap):
//   prompt_cross_attn_norm.{weight,bias}     LayerNorm 256
//   prompt_cross_attn.q_proj.{weight,bias}   [256, 256] + [256]
//   prompt_cross_attn.k_proj.{weight,bias}
//   prompt_cross_attn.v_proj.{weight,bias}
//   prompt_cross_attn.o_proj.{weight,bias}
class InstructsamPromptCrossAttn {
public:
    explicit InstructsamPromptCrossAttn(const GgufModel & model);

    // encoder_hidden_states: [hw=5184, 256]  raw input
    // prompt_features:       [prompt_seq=32, 256]
    // prompt_mask:           [prompt_seq=32]  1.0 = valid, 0.0 = padding
    // Returns post-PCA encoder [hw, 256] = residual + attention_output.
    std::vector<float> run(
        const std::vector<float> & encoder_hidden_states,
        const std::vector<int64_t> & encoder_hidden_states_shape,
        const std::vector<float> & prompt_features,
        const std::vector<int64_t> & prompt_features_shape,
        const std::vector<float> & prompt_mask,
        const std::vector<int64_t> & prompt_mask_shape
    ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
