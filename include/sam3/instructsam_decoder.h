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

    // Full decoder graph — 6 layers of self_attn + text_cross_attn +
    // vision_cross_attn + MLP, with post-norm placement. Inputs:
    //   - queries: initial decoder queries [num_queries, 256] — typically
    //     mask_hidden_fcs output on LM hidden states at mask_queries positions
    //   - text_memory: text keys/values [text_seq, 256] — text_hidden_fcs
    //     output on phrase embeddings
    //   - vision_memory: vision keys/values [hw, 256] — output of
    //     detr_encoder (which already fused vision + text upstream)
    //   - vision_pos: vision position embeddings [hw, 256]
    //   - text_mask: attention mask over text_memory [text_seq]
    //
    // Returns per-layer hidden states + reference boxes. Presence-token
    // machinery is omitted (InstructSAM uses per-slot cls_score from
    // dot_product_scoring downstream, not presence prediction here).
    // query_pos: if non-empty, [num_queries, 256] positional embedding added
    // to Q in every attention block of every layer. Real InstructSAM updates
    // query_pos each layer from the layer-N reference_points via ref_point_head;
    // for now this test harness accepts an externally-computed layer-0
    // query_pos and reuses it — good enough for layer-0 numerical parity.
    // Pass empty vector to disable (uses zeros — for structural smoke only).
    // Optional:
    //   query_pos: [10, 256]. Empty → zeros (structural smoke only).
    //   initial_reference_boxes: [10, 4] in (cx,cy,w,h) sigmoid space; needed
    //     to compute vision cross-attn box_rpb bias. Empty → no bias (bad
    //     numerical parity, structural smoke only).
    DecoderOutput run(
        const std::vector<float> & queries,
        const std::vector<int64_t> & queries_shape,
        const std::vector<float> & text_memory,
        const std::vector<int64_t> & text_memory_shape,
        const std::vector<float> & vision_memory,
        const std::vector<int64_t> & vision_memory_shape,
        const std::vector<float> & vision_pos,
        const std::vector<int64_t> & vision_pos_shape,
        const std::vector<float> & text_mask,
        const std::vector<int64_t> & text_mask_shape,
        const std::vector<float> & query_pos = {},
        const std::vector<float> & initial_reference_boxes = {}
    ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
