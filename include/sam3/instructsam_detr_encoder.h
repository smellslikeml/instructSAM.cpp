#pragma once

#include "sam3/gguf_model.h"

#include <cstdint>
#include <vector>

namespace sam3 {

struct DetrEncoderOutput {
    int64_t num_layers = 0;
    int64_t vision_seq = 0;
    int64_t hidden_dim = 0;
    // Per-layer vision hidden states, each [vision_seq, 256]. Same shape.
    std::vector<std::vector<float>> hs;
    // Alias for the final layer output, same buffer as hs.back(). Provided
    // for callers that only need the encoder result.
    std::vector<float> last_hidden_state;
};

// InstructSAM's DETR-style vision-text fusion encoder.
//
// Per-layer flow (PRE-NORM — opposite of the decoder!):
//   normed = layer_norm1(hidden)
//   qk     = normed + vision_pos
//   self_out = self_attn(Q=qk, K=qk, V=normed)
//   hidden = hidden + self_out
//   normed = layer_norm2(hidden)
//   cross_out = cross_attn(Q=normed, K=V=text_features, mask=text_mask)
//   hidden = hidden + cross_out
//   normed = layer_norm3(hidden)
//   hidden = hidden + mlp(normed)
//
// Six layers total. Uses `transformer.encoder.layers.N.*` tensor namespace
// (already present in the GGUF from the converter).
class InstructsamDetrEncoder {
public:
    explicit InstructsamDetrEncoder(const GgufModel & model);

    DetrEncoderOutput run(
        const std::vector<float> & vision_features,
        const std::vector<int64_t> & vision_shape,   // [hw=5184, 256]
        const std::vector<float> & vision_pos,
        const std::vector<int64_t> & vision_pos_shape,
        const std::vector<float> & text_features,
        const std::vector<int64_t> & text_shape,     // [text_seq=32, 256]
        const std::vector<float> & text_mask,
        const std::vector<int64_t> & text_mask_shape // [text_seq]
    ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
