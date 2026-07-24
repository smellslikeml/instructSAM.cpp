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

    // Full forward pass on a sequence of position-embed pairs.
    //
    // Input: `embeds` [seq_len, 2048] — the input embeddings for each
    //   position. Caller looks up token embeds via embed_for_token, and can
    //   substitute any position with a custom embed (e.g. mask_queries for
    //   InstructSAM's injection at ref_end).
    // Input: `positions` [seq_len] — position IDs for RoPE. For a
    //   contiguous decode this is just 0, 1, 2, .... For M-RoPE variants
    //   the caller supplies whatever is right (text-only reduces to 1D).
    //
    // Output: final hidden states [seq_len, 2048] AFTER output_norm.
    //
    // Uses causal attention. No KV cache (recomputes full attention each
    // call — fine for our small ~10-token injection use case; can add
    // caching later for full generation loops if needed).
    std::vector<float> run(
        const std::vector<float> & embeds,
        int64_t seq_len,
        const std::vector<int32_t> & positions
    ) const;

    // Convenience wrapper for InstructSAM's mask_queries injection.
    // Given a prompt token sequence (ending with <|object_ref_end|>) and
    // the 10 mask_queries embeddings, builds the full injected sequence:
    //   [prompt_tokens..., mask_start, mask_queries[0..9], mask_end]
    // Runs the forward pass and returns the 10 hidden states at the
    // mask_queries positions. This is seg_output_embeddings — feeds
    // directly into InstructsamMaskHiddenFcs → decoder queries.
    //
    // mask_queries shape [10, 2048]. mask_start_embed / mask_end_embed
    // shape [2048] each. Typically extracted from InstructSAM's checkpoint.
    std::vector<float> extract_seg_output_embeddings(
        const std::vector<int32_t> & prompt_token_ids,  // must end with <|object_ref_end|>
        const std::vector<float> & mask_queries,        // [10, 2048]
        const std::vector<float> & mask_start_embed,    // [2048]
        const std::vector<float> & mask_end_embed       // [2048]
    ) const;

    // Same as above, but takes a raw prefix of LM-space embeddings
    // (rather than token IDs). Use this when the prefix contains image
    // embeddings from mmproj that don't correspond to any single token —
    // caller stitches [text_embeds, image_embeds, text_embeds] and passes
    // the whole thing here.
    std::vector<float> extract_seg_output_embeddings_from_prefix(
        const std::vector<float> & prefix_embeds,       // [n_prefix, 2048]
        int64_t n_prefix,
        const std::vector<float> & mask_queries,        // [10, 2048]
        const std::vector<float> & mask_start_embed,    // [2048]
        const std::vector<float> & mask_end_embed       // [2048]
    ) const;

    // ── KV cache for prefill-then-decode ─────────────────────────────────
    //
    // Motivating problem: extract_seg_output_embeddings_from_prefix() runs
    // a full O(N²·D) forward on 300+ tokens per call. When the CLI computes
    // 4 phrases sharing the same [text + image + text] prefix, that's 4×
    // duplicate prefix work.
    //
    // KvCache stores per-layer K and V for a fixed prefix. `prefill_prefix`
    // populates it once (~one full-cost forward). Then
    // `extract_seg_output_embeddings_with_cache` runs a decode over just
    // the appended tokens per phrase, attending to prefix K/V from cache
    // + new K/V computed on the fly. Same numerical result as the
    // no-cache version, but the per-phrase forward is O((n_new)² + n_cached·n_new)
    // in attention and O(n_new·D²) in projections — much smaller than
    // O(n_total² · D + n_total · D²).
    struct KvCache {
        // Per-layer flat K and V — each [n_cached, num_kv_heads=8, head_dim=128]
        // as a flat float vector of length n_cached * 1024. K/V here are
        // post-RoPE (K) and raw (V), ready for attention with the
        // concatenated new K/V without further transformation.
        std::vector<std::vector<float>> k;  // size num_layers=28
        std::vector<std::vector<float>> v;
        int64_t n_cached = 0;
    };

    // Run prefill on `prefix_embeds` and return the populated cache.
    // `positions` gives the RoPE position for each of the n_prefix tokens
    // (typically 0..n_prefix-1). Cost: same as one full run() on the prefix.
    KvCache prefill_prefix(
        const std::vector<float> & prefix_embeds,
        int64_t n_prefix,
        const std::vector<int32_t> & positions
    ) const;

    // Same shape/return as extract_seg_output_embeddings_from_prefix, but
    // reuses a pre-computed prefix cache. Only the appended ~14-token
    // sequence is decoded per call, so this can be looped over phrases
    // sharing a common cache without recomputing the prefix each time.
    std::vector<float> extract_seg_output_embeddings_with_cache(
        const KvCache & prefix_cache,
        const std::vector<float> & appended_prefix_embeds,  // [n_ap, 2048]
        int64_t n_appended_prefix,  // e.g. <|object_ref_start|>+phrase+<|object_ref_end|>
        const std::vector<float> & mask_queries,
        const std::vector<float> & mask_start_embed,
        const std::vector<float> & mask_end_embed
    ) const;

    // ── Autoregressive generation support ────────────────────────────────

    // Prefill + return final hidden state (post-output_norm) at the last
    // position, ready to project via `logits_for_hidden` for the first
    // sampled token.
    struct PrefillResult {
        KvCache cache;
        std::vector<float> last_hidden;  // [2048], AFTER output_norm
    };
    PrefillResult prefill_with_last_hidden(
        const std::vector<float> & prefix_embeds,
        int64_t n_prefix,
        const std::vector<int32_t> & positions
    ) const;

    // Decode a single new token (given its embed + absolute position) and
    // EXTEND the cache in place. Returns the token's hidden state after
    // output_norm — feed to logits_for_hidden for next-token prediction.
    std::vector<float> decode_step(
        KvCache & cache_mutable,
        const std::vector<float> & new_embed,  // [2048]
        int32_t new_position
    ) const;

    // LM head: project [2048] hidden → [vocab_size=151936] logits.
    // Qwen3-VL ties the input embedding table for output — no separate
    // output.weight. logits = hidden @ token_embd.weight.T
    std::vector<float> logits_for_hidden(const std::vector<float> & hidden_2048) const;

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
    // Lazy: full [vocab, hidden] = 304 MB f32 fetched once for
    // logits_for_hidden. Reused across all generation steps.
    mutable std::vector<float> embd_cache_f32_;
};

}  // namespace sam3
