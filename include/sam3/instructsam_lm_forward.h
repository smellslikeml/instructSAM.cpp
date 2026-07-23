#pragma once

#include "sam3/gguf_model.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sam3 {

// Qwen3-VL text-model forward pass — fork approach for InstructSAM's
// LM half. We control the embed injection semantics directly (unlike
// llama.cpp's embed-mode decode which has an unresolved bug at
// multi-position embed batches > 2 for our use case).
//
// Architecture (from InstructSAM's Qwen3-VL text config):
//   28 layers
//   hidden_size          = 2048
//   num_attention_heads  = 16
//   num_key_value_heads  = 8  (GQA: K/V repeated 2x for attention)
//   head_dim             = 128
//   intermediate_size    = 6144
//   silu (SwiGLU MLP)
//   rms_norm_eps         = 1e-06
//   RoPE 1D (rope_theta = 5000000 for Qwen3, verify from config)
//   Per-layer: input_layernorm (RMS) → self_attn → post_attention_layernorm (RMS) → SwiGLU MLP → residual
//   self_attn: q_proj/k_proj/v_proj (GQA sizes) → q_norm/k_norm PER-HEAD (RMS) → 1D RoPE → attn → o_proj
//   Final: RMSNorm at end
//
// Tensor namespace (llama.cpp GGUF from Path B — different from PyTorch names):
//   token_embd.weight                                       [2048, 151936] (F16)
//   output_norm.weight                                      [2048]         (F32)
//   blk.N.attn_norm.weight                                  [2048]         (input_layernorm)
//   blk.N.attn_q.weight, .bias                              [2048, 2048]
//   blk.N.attn_k.weight, .bias                              [1024, 2048]   (GQA)
//   blk.N.attn_v.weight, .bias                              [1024, 2048]
//   blk.N.attn_output.weight                                [2048, 2048]   (o_proj)
//   blk.N.attn_q_norm.weight                                [128]          (per-head)
//   blk.N.attn_k_norm.weight                                [128]
//   blk.N.ffn_norm.weight                                   [2048]         (post_attention_layernorm)
//   blk.N.ffn_gate.weight                                   [6144, 2048]
//   blk.N.ffn_up.weight                                     [6144, 2048]
//   blk.N.ffn_down.weight                                   [2048, 6144]

class InstructsamLmForward {
public:
    // Load Path B's LM GGUF (instructsam-lm-f16.gguf) via our GgufModel.
    explicit InstructsamLmForward(const GgufModel & model);

    // Validate all 28 layers × tensors + top-level (token_embd, output_norm).
    // Throws on first missing. Returns count probed.
    size_t validate_all_tensors_present() const;

    // Look up a token's INPUT embedding from token_embd.weight.
    // Handy for building the phrase_embeddings input to text_hidden_fcs
    // (piece 4b). Also for constructing the initial embed sequence in the
    // Qwen3 forward pass (mask_start + mask_queries + mask_end).
    // Returns [hidden_size = 2048] f32 vector.
    std::vector<float> embed_for_token(int32_t token_id) const;

    // Run a single-layer forward pass — NOT YET IMPLEMENTED.
    // Signature will be along the lines of:
    //   std::vector<float> run_layer(int layer_idx,
    //                                const std::vector<float> & hidden,  // [seq, 2048]
    //                                int64_t seq_len,
    //                                ...) const;

    // Full forward pass with embedding injection support — NOT YET IMPLEMENTED.
    //   struct LmForwardOutput {
    //       // hidden_states[layer] shape [seq, 2048]. hidden_states.back()
    //       // is the pre-output_norm final layer output.
    //       std::vector<std::vector<float>> hidden_states;
    //   };
    //   LmForwardOutput run(
    //     const std::vector<int32_t> & prompt_token_ids,
    //     const std::vector<float>   & injected_embeds,  // [n_inject, 2048]
    //     int64_t inject_after_position                  // position to insert embeds
    //   ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
