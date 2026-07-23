#pragma once

#include "sam3/decoder.h"        // reuse DecoderOutput
#include "sam3/gguf_model.h"

#include <vector>

namespace sam3 {

// InstructSAM's decoder variant. Diverges from sam3cpp's stock Decoder in
// three ways:
//   1. Two cross-attention blocks per layer (text_cross_attn + vision_cross_attn)
//      versus stock sam3cpp's single fused cross_attn.
//   2. Tensor naming: InstructSAM uses q_proj/k_proj/v_proj/o_proj
//      (transformers convention) rather than sam3cpp's query_proj/etc.
//   3. Takes TWO memory tensors (text_memory + vision_memory) rather than a
//      single pre-fused memory. Text and vision streams stay separate.
//
// Also consumes the InstructSAM-specific glue tensors during initialization
// (mask_hidden_fcs, text_hidden_fcs projections) — those are used by the
// caller to project external LM hidden states down to decoder-query dim,
// not by the decoder graph itself.
//
// See docs/instructsam/DESIGN.md for the full architectural rationale.
class InstructsamDecoder {
public:
    explicit InstructsamDecoder(const GgufModel & model);

    // Validate that every tensor the decoder graph will need is
    // present in the loaded model. Enumerates all 6 layers, all attention
    // blocks (self + text_cross + vision_cross), all norms + MLPs, plus
    // per-layer box refinement heads. Throws on the first missing tensor.
    //
    // Returns the number of tensors probed; used by the scaffold test
    // to prove Path A phase-2-step-2 prep is complete before the actual
    // ggml graph implementation begins.
    size_t validate_all_tensors_present() const;

    // Full graph implementation — TODO in a subsequent commit.
    // DecoderOutput run(
    //     const std::vector<float> & text_memory,
    //     const std::vector<int64_t> & text_memory_shape,
    //     const std::vector<float> & vision_memory,
    //     const std::vector<int64_t> & vision_memory_shape,
    //     const std::vector<float> & pos_embed,
    //     const std::vector<int64_t> & pos_shape,
    //     const std::vector<float> & seg_hidden_states,       // from LM at mask_queries positions
    //     const std::vector<int64_t> & seg_hidden_states_shape,
    //     const std::vector<float> & phrase_embeddings,       // text_hidden_fcs output
    //     const std::vector<int64_t> & phrase_embeddings_shape
    // ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
