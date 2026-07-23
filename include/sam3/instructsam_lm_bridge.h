#pragma once

#include "sam3/gguf_model.h"

#include <cstdint>
#include <vector>

namespace sam3 {

// LM-side bridges — 2-layer MLPs with ReLU that project LM-space
// hidden states (2048-dim) down to sam3-space (256-dim).
//
// - mask_hidden_fcs: seg_output_embeddings [N, 10, 2048] → decoder queries
//   [N, 10, 256]. Input is LM hidden states captured at [SEG]-injected
//   mask_queries positions during generation.
//
// - text_hidden_fcs: LM_embed_table[phrase_ids] [N, ≤32, 2048] →
//   text_features [N, ≤32, 256]. Input is LM's input embedding table
//   looked up at phrase token IDs (extracted from generated tokens
//   between <object_ref_start> and <object_ref_end>).
//
// Both use weights `instructsam.{mask,text}_hidden_fcs.0.{0,2}.{weight,bias}`.
// Same architecture — Linear(2048→2048) → ReLU → Linear(2048→256).

struct LmBridgeOutput {
    int64_t batch = 0;
    int64_t seq = 0;
    int64_t out_dim = 0;      // 256
    std::vector<float> data;  // [batch, seq, out_dim] flattened
};

class InstructsamMaskHiddenFcs {
public:
    explicit InstructsamMaskHiddenFcs(const GgufModel & model);
    // input: seg_output_embeddings, shape [batch, seq, 2048]
    LmBridgeOutput run(
        const std::vector<float> & seg_output_embeddings,
        const std::vector<int64_t> & shape
    ) const;
private:
    const GgufModel & model_;
};

class InstructsamTextHiddenFcs {
public:
    explicit InstructsamTextHiddenFcs(const GgufModel & model);
    // input: phrase_embeddings (LM embed lookups), shape [batch, seq, 2048]
    LmBridgeOutput run(
        const std::vector<float> & phrase_embeddings,
        const std::vector<int64_t> & shape
    ) const;
private:
    const GgufModel & model_;
};

}  // namespace sam3
