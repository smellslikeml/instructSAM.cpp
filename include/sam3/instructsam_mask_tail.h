#pragma once

#include "sam3/gguf_model.h"

#include <cstdint>
#include <vector>

namespace sam3 {

struct MaskTailOutput {
    int64_t num_queries = 0;   // 10 (kNumQueries)
    int64_t height = 0;        // 288
    int64_t width = 0;         // 288
    std::vector<float> pred_masks;    // [num_queries, height, width]
    std::vector<float> semantic_seg;  // [1, height, width]
};

// InstructSAM mask decoder — TAIL ONLY. Takes:
//   - pixel_embed:      [256, 288, 288] pre-projected pixel features
//                       (output of pixel_decoder FPN + prompt_cross_attn)
//   - decoder_queries:  [10, 256] final decoder hidden states
//
// Produces per-query pred_masks + semantic segmentation.
//
// Deferred (future work — see PARITY-STATUS.md): the pixel_decoder FPN
// itself and the prompt_cross_attn that produces pixel_embed. Those
// components take backbone_features (list) + encoder_hidden_states +
// prompt_features as inputs.
//
// Weights used:
//   mask_decoder.mask_embedder.layers.0/1/2.{weight,bias}   (256→256×3)
//   mask_decoder.instance_projection.{weight,bias}          (Conv1x1 256→256)
//   mask_decoder.semantic_projection.{weight,bias}          (Conv1x1 256→1)
class InstructsamMaskTail {
public:
    explicit InstructsamMaskTail(const GgufModel & model);

    MaskTailOutput run(
        const std::vector<float> & pixel_embed,
        const std::vector<int64_t> & pixel_embed_shape,   // [256, 288, 288]
        const std::vector<float> & decoder_queries,
        const std::vector<int64_t> & decoder_queries_shape  // [10, 256]
    ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
