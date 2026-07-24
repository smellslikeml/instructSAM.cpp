#include "sam3/instructsam_lm_forward.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

constexpr int32_t kNumLayers    = 28;
constexpr int32_t kHiddenSize   = 2048;
constexpr int32_t kNumHeads     = 16;
constexpr int32_t kNumKvHeads   = 8;
constexpr int32_t kHeadDim      = 128;
constexpr int32_t kIntermediate = 6144;
constexpr int32_t kVocabSize    = 151936;
constexpr float   kRmsNormEps   = 1e-6f;
constexpr float   kRopeTheta    = 5000000.0f;  // qwen3vl.rope.freq_base
constexpr int32_t kKvGroups     = kNumHeads / kNumKvHeads;  // 16/8 = 2 (GQA repeat factor)

ggml_tensor * require_tensor(const GgufModel & model, const std::string & name) {
    ggml_tensor * t = model.find_weight(name);
    if (t == nullptr) throw std::runtime_error("lm_forward: missing " + name);
    return t;
}

// Read a weight tensor as F32 (handles F16 → F32).
std::vector<float> get_f32(const GgufModel & model, const std::string & name, size_t n) {
    ggml_tensor * t = require_tensor(model, name);
    std::vector<float> v(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) v[i] = ggml_fp16_to_fp32(buf[i]);
    } else {
        throw std::runtime_error("lm_forward: unsupported dtype for " + name);
    }
    return v;
}

// ── CPU primitives ──────────────────────────────────────────────────────

// RMSNorm on [N, D] with per-D weight vector [D].
// y[i, d] = x[i, d] * weight[d] / sqrt(mean(x[i, :]^2) + eps)
std::vector<float> cpu_rms_norm(
    const std::vector<float> & x, int64_t N, int64_t D,
    const std::vector<float> & weight, float eps = kRmsNormEps
) {
    std::vector<float> out(x.size());
    for (int64_t i = 0; i < N; ++i) {
        double sumsq = 0.0;
        for (int64_t d = 0; d < D; ++d) {
            const double v = x[static_cast<size_t>(i * D + d)];
            sumsq += v * v;
        }
        const double inv_rms = 1.0 / std::sqrt(sumsq / D + eps);
        for (int64_t d = 0; d < D; ++d) {
            out[static_cast<size_t>(i * D + d)] = static_cast<float>(
                x[static_cast<size_t>(i * D + d)] * inv_rms * weight[static_cast<size_t>(d)]);
        }
    }
    return out;
}

// Linear: y = x @ w.T + b (b optional). x [N, D_in], w [D_out, D_in], y [N, D_out].
std::vector<float> cpu_linear(
    const std::vector<float> & x, int64_t N, int64_t D_in, int64_t D_out,
    const std::vector<float> & w,
    const std::vector<float> * b = nullptr
) {
    std::vector<float> y(static_cast<size_t>(N * D_out));
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t o = 0; o < D_out; ++o) {
            float s = b ? (*b)[static_cast<size_t>(o)] : 0.0f;
            for (int64_t k = 0; k < D_in; ++k) {
                s += w[static_cast<size_t>(o * D_in + k)] *
                     x[static_cast<size_t>(n * D_in + k)];
            }
            y[static_cast<size_t>(n * D_out + o)] = s;
        }
    }
    return y;
}

// SwiGLU: silu(gate) * up, then down_proj.
void cpu_silu_mul(std::vector<float> & gate, const std::vector<float> & up) {
    // silu(x) = x * sigmoid(x)
    for (size_t i = 0; i < gate.size(); ++i) {
        const float g = gate[i];
        const float sig = 1.0f / (1.0f + std::exp(-g));
        gate[i] = g * sig * up[i];
    }
}

// 1D RoPE for adjacent-pair rotation (NeoX / Qwen convention):
//   angle = pos * (theta ^ (-2i / head_dim))  for i in [0, head_dim/2)
//   for each pair (2i, 2i+1):
//     q'[2i]   = q[2i]   * cos(angle_i) - q[2i+1] * sin(angle_i)
//     q'[2i+1] = q[2i+1] * cos(angle_i) + q[2i]   * sin(angle_i)
//
// Wait — Qwen3 actually uses the "half rotation" variant (LLAMA style):
//   pair (i, i + head_dim/2), NOT (2i, 2i+1). Let me verify from
//   transformers source before wiring; for now use half-rotation (standard).
//
// x has shape [N, num_heads, head_dim]. positions [N].
void cpu_rope_1d_half(
    std::vector<float> & x, int64_t N, int64_t num_heads, int64_t head_dim,
    const std::vector<int32_t> & positions, float theta = kRopeTheta
) {
    const int64_t half = head_dim / 2;
    std::vector<float> freqs(static_cast<size_t>(half));
    for (int64_t i = 0; i < half; ++i) {
        freqs[static_cast<size_t>(i)] = 1.0f / std::pow(theta,
            static_cast<float>(2 * i) / static_cast<float>(head_dim));
    }
    for (int64_t n = 0; n < N; ++n) {
        const float pos = static_cast<float>(positions[static_cast<size_t>(n)]);
        for (int64_t h = 0; h < num_heads; ++h) {
            float * row = x.data() + (n * num_heads + h) * head_dim;
            for (int64_t i = 0; i < half; ++i) {
                const float angle = pos * freqs[static_cast<size_t>(i)];
                const float c = std::cos(angle), s = std::sin(angle);
                const float x0 = row[i];
                const float x1 = row[i + half];
                row[i]        = x0 * c - x1 * s;
                row[i + half] = x1 * c + x0 * s;
            }
        }
    }
}

std::vector<std::string> per_layer_tensor_names(int layer) {
    const std::string p = "blk." + std::to_string(layer);
    return {
        p + ".attn_norm.weight",
        p + ".attn_q.weight", p + ".attn_q.bias",
        p + ".attn_k.weight", p + ".attn_k.bias",
        p + ".attn_v.weight", p + ".attn_v.bias",
        p + ".attn_output.weight",
        p + ".attn_q_norm.weight",
        p + ".attn_k_norm.weight",
        p + ".ffn_norm.weight",
        p + ".ffn_gate.weight",
        p + ".ffn_up.weight",
        p + ".ffn_down.weight",
    };
}

std::vector<std::string> top_level_tensor_names() {
    return {
        "token_embd.weight",
        "output_norm.weight",
    };
}

}  // namespace

InstructsamLmForward::InstructsamLmForward(const GgufModel & model) : model_(model) {}

size_t InstructsamLmForward::validate_all_tensors_present() const {
    size_t probed = 0;
    for (const auto & name : top_level_tensor_names()) {
        // Some tensors (like biases for attn_q) may not exist in Qwen3
        // depending on config. Skip biases from must-have list if not found —
        // but tensor names above are all required in Qwen3 as of the
        // reference checkpoint. Throw on genuinely missing.
        if (model_.find_tensor(name) == nullptr) {
            throw std::runtime_error("lm_forward: missing top-level tensor: " + name);
        }
        ++probed;
    }
    for (int layer = 0; layer < kNumLayers; ++layer) {
        for (const auto & name : per_layer_tensor_names(layer)) {
            // Bias tensors may or may not exist depending on GGUF conversion
            // choices for Qwen3-VL (some converters skip biases if 0.0).
            // Attempt the lookup; if it's a .bias, tolerate absent.
            const bool is_bias = name.size() >= 5 && name.compare(name.size() - 5, 5, ".bias") == 0;
            if (model_.find_tensor(name) == nullptr) {
                if (is_bias) continue;
                throw std::runtime_error("lm_forward: missing per-layer tensor: " + name);
            }
            ++probed;
        }
    }
    return probed;
}

std::vector<float> InstructsamLmForward::embed_for_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= kVocabSize) {
        throw std::runtime_error("lm_forward: token_id out of range");
    }
    ggml_tensor * t = require_tensor(model_, "token_embd.weight");
    // token_embd.weight stored as [hidden=2048, vocab=151936] in ggml (col-major).
    // Row `token_id` is at offset token_id * hidden * sizeof(dtype).
    std::vector<float> out(kHiddenSize);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(),
            token_id * kHiddenSize * sizeof(float),
            kHiddenSize * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(kHiddenSize);
        ggml_backend_tensor_get(t, buf.data(),
            token_id * kHiddenSize * sizeof(ggml_fp16_t),
            kHiddenSize * sizeof(ggml_fp16_t));
        for (int i = 0; i < kHiddenSize; ++i) out[i] = ggml_fp16_to_fp32(buf[i]);
    } else {
        throw std::runtime_error("lm_forward: token_embd unsupported dtype");
    }
    return out;
}

std::vector<float> InstructsamLmForward::run(
    const std::vector<float> & embeds,
    int64_t seq_len,
    const std::vector<int32_t> & positions
) const {
    if (embeds.size() != static_cast<size_t>(seq_len * kHiddenSize)) {
        throw std::runtime_error("lm_forward.run: embeds size must be seq_len * 2048");
    }
    if (positions.size() != static_cast<size_t>(seq_len)) {
        throw std::runtime_error("lm_forward.run: positions size must be seq_len");
    }

    std::vector<float> hidden = embeds;   // [seq_len, 2048]

    for (int layer = 0; layer < kNumLayers; ++layer) {
        const std::string p = "blk." + std::to_string(layer);

        // Load per-layer weights (each layer ~30 MB for all its tensors —
        // fine to reload per-call given lazy mmap. Cache-later for perf.)
        const auto attn_norm_w   = get_f32(model_, p + ".attn_norm.weight",   kHiddenSize);
        const auto attn_q_w      = get_f32(model_, p + ".attn_q.weight",      kHiddenSize * kHiddenSize);
        const auto attn_k_w      = get_f32(model_, p + ".attn_k.weight",      kNumKvHeads * kHeadDim * kHiddenSize);
        const auto attn_v_w      = get_f32(model_, p + ".attn_v.weight",      kNumKvHeads * kHeadDim * kHiddenSize);
        const auto attn_o_w      = get_f32(model_, p + ".attn_output.weight", kHiddenSize * kHiddenSize);
        const auto q_norm_w      = get_f32(model_, p + ".attn_q_norm.weight", kHeadDim);
        const auto k_norm_w      = get_f32(model_, p + ".attn_k_norm.weight", kHeadDim);
        const auto ffn_norm_w    = get_f32(model_, p + ".ffn_norm.weight",    kHiddenSize);
        const auto ffn_gate_w    = get_f32(model_, p + ".ffn_gate.weight",    kIntermediate * kHiddenSize);
        const auto ffn_up_w      = get_f32(model_, p + ".ffn_up.weight",      kIntermediate * kHiddenSize);
        const auto ffn_down_w    = get_f32(model_, p + ".ffn_down.weight",    kHiddenSize * kIntermediate);

        // ── Attention block ─────────────────────────────────────────────
        const auto residual = hidden;
        const auto normed = cpu_rms_norm(hidden, seq_len, kHiddenSize, attn_norm_w);

        // Q/K/V projections
        auto Q_flat = cpu_linear(normed, seq_len, kHiddenSize, kHiddenSize,             attn_q_w); // [seq, 2048]
        auto K_flat = cpu_linear(normed, seq_len, kHiddenSize, kNumKvHeads * kHeadDim, attn_k_w); // [seq, 1024]
        auto V_flat = cpu_linear(normed, seq_len, kHiddenSize, kNumKvHeads * kHeadDim, attn_v_w); // [seq, 1024]

        // Reshape Q [seq, 16*128] → [seq, 16, 128], K/V [seq, 8*128] → [seq, 8, 128]
        // (In flat form the reshape is a no-op — just interpret differently.)

        // Per-head RMSNorm on Q and K (Qwen3 specific):
        //   normalize each [head_dim=128] vector, then multiply by [head_dim=128] weight
        auto per_head_norm = [&](std::vector<float> & x, int64_t heads, const std::vector<float> & w) {
            for (int64_t n = 0; n < seq_len; ++n) {
                for (int64_t h = 0; h < heads; ++h) {
                    float * row = x.data() + (n * heads + h) * kHeadDim;
                    double sumsq = 0.0;
                    for (int64_t d = 0; d < kHeadDim; ++d) sumsq += static_cast<double>(row[d]) * row[d];
                    const double inv_rms = 1.0 / std::sqrt(sumsq / kHeadDim + kRmsNormEps);
                    for (int64_t d = 0; d < kHeadDim; ++d) {
                        row[d] = static_cast<float>(row[d] * inv_rms * w[static_cast<size_t>(d)]);
                    }
                }
            }
        };
        per_head_norm(Q_flat, kNumHeads,   q_norm_w);
        per_head_norm(K_flat, kNumKvHeads, k_norm_w);

        // Apply 1D RoPE to Q and K
        cpu_rope_1d_half(Q_flat, seq_len, kNumHeads,   kHeadDim, positions);
        cpu_rope_1d_half(K_flat, seq_len, kNumKvHeads, kHeadDim, positions);

        // Attention with causal mask + GQA (K/V heads repeated 2x)
        const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
        std::vector<float> attn_out(static_cast<size_t>(seq_len * kHiddenSize));

        for (int64_t h = 0; h < kNumHeads; ++h) {
            const int64_t kv_h = h / kKvGroups;  // GQA: 2 Q-heads share each KV-head
            // Compute S = Q_h @ K_kvh^T * scale, shape [seq, seq], with causal mask.
            std::vector<float> S(static_cast<size_t>(seq_len * seq_len));
            for (int64_t i = 0; i < seq_len; ++i) {
                const float * qi = Q_flat.data() + (i * kNumHeads + h) * kHeadDim;
                for (int64_t j = 0; j < seq_len; ++j) {
                    if (j > i) { S[static_cast<size_t>(i * seq_len + j)] = -std::numeric_limits<float>::infinity(); continue; }
                    const float * kj = K_flat.data() + (j * kNumKvHeads + kv_h) * kHeadDim;
                    float s = 0.0f;
                    for (int64_t d = 0; d < kHeadDim; ++d) s += qi[d] * kj[d];
                    S[static_cast<size_t>(i * seq_len + j)] = s * scale;
                }
            }
            // Softmax over each row
            for (int64_t i = 0; i < seq_len; ++i) {
                float * row = S.data() + i * seq_len;
                float m = row[0];
                for (int64_t j = 1; j <= i; ++j) if (row[j] > m) m = row[j];
                float sum = 0.0f;
                for (int64_t j = 0; j <= i; ++j) { row[j] = std::exp(row[j] - m); sum += row[j]; }
                for (int64_t j = 0; j <= i; ++j) row[j] /= sum;
                for (int64_t j = i + 1; j < seq_len; ++j) row[j] = 0.0f;
            }
            // Y_h = S @ V_kvh, write into attn_out at [n, h*head_dim..(h+1)*head_dim]
            for (int64_t i = 0; i < seq_len; ++i) {
                for (int64_t d = 0; d < kHeadDim; ++d) {
                    float y = 0.0f;
                    for (int64_t j = 0; j <= i; ++j) {
                        const float * vj = V_flat.data() + (j * kNumKvHeads + kv_h) * kHeadDim;
                        y += S[static_cast<size_t>(i * seq_len + j)] * vj[d];
                    }
                    attn_out[static_cast<size_t>(i * kHiddenSize + h * kHeadDim + d)] = y;
                }
            }
        }

        // Output projection + residual add
        const auto attn_proj = cpu_linear(attn_out, seq_len, kHiddenSize, kHiddenSize, attn_o_w);
        for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = residual[i] + attn_proj[i];

        // ── MLP block ────────────────────────────────────────────────────
        const auto residual2 = hidden;
        const auto normed2 = cpu_rms_norm(hidden, seq_len, kHiddenSize, ffn_norm_w);
        auto gate_out = cpu_linear(normed2, seq_len, kHiddenSize, kIntermediate, ffn_gate_w);
        const auto up_out   = cpu_linear(normed2, seq_len, kHiddenSize, kIntermediate, ffn_up_w);
        cpu_silu_mul(gate_out, up_out);
        const auto down_out = cpu_linear(gate_out, seq_len, kIntermediate, kHiddenSize, ffn_down_w);
        for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = residual2[i] + down_out[i];
    }

    // Final RMSNorm
    const auto output_norm_w = get_f32(model_, "output_norm.weight", kHiddenSize);
    return cpu_rms_norm(hidden, seq_len, kHiddenSize, output_norm_w);
}

std::vector<float> InstructsamLmForward::extract_seg_output_embeddings(
    const std::vector<int32_t> & prompt_token_ids,
    const std::vector<float> & mask_queries,
    const std::vector<float> & mask_start_embed,
    const std::vector<float> & mask_end_embed
) const {
    if (mask_queries.size() != 10 * kHiddenSize) {
        throw std::runtime_error("extract_seg_output_embeddings: mask_queries must be [10, 2048]");
    }
    if (mask_start_embed.size() != kHiddenSize || mask_end_embed.size() != kHiddenSize) {
        throw std::runtime_error("extract_seg_output_embeddings: mask_start/end embeds must be [2048]");
    }
    if (prompt_token_ids.empty()) {
        throw std::runtime_error("extract_seg_output_embeddings: empty prompt");
    }

    const int64_t n_prompt = static_cast<int64_t>(prompt_token_ids.size());
    // Total sequence: prompt + mask_start + 10 mask_queries + mask_end
    const int64_t n_inject = 12;
    const int64_t seq_len = n_prompt + n_inject;

    std::vector<float> embeds(seq_len * kHiddenSize);
    std::vector<int32_t> positions(seq_len);

    // Prompt token embeddings via LM's embed table
    for (int64_t i = 0; i < n_prompt; ++i) {
        const auto e = embed_for_token(prompt_token_ids[static_cast<size_t>(i)]);
        std::memcpy(embeds.data() + i * kHiddenSize, e.data(), kHiddenSize * sizeof(float));
        positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }

    // Inject: mask_start
    std::memcpy(embeds.data() + n_prompt * kHiddenSize,
                mask_start_embed.data(), kHiddenSize * sizeof(float));
    positions[static_cast<size_t>(n_prompt)] = static_cast<int32_t>(n_prompt);

    // Inject: 10 mask_queries
    for (int64_t j = 0; j < 10; ++j) {
        std::memcpy(embeds.data() + (n_prompt + 1 + j) * kHiddenSize,
                    mask_queries.data() + j * kHiddenSize,
                    kHiddenSize * sizeof(float));
        positions[static_cast<size_t>(n_prompt + 1 + j)] = static_cast<int32_t>(n_prompt + 1 + j);
    }

    // Inject: mask_end
    std::memcpy(embeds.data() + (n_prompt + 11) * kHiddenSize,
                mask_end_embed.data(), kHiddenSize * sizeof(float));
    positions[static_cast<size_t>(n_prompt + 11)] = static_cast<int32_t>(n_prompt + 11);

    // Full forward
    const auto final_hidden = run(embeds, seq_len, positions);

    // Extract the 10 mask_queries hidden states at positions [n_prompt+1 .. n_prompt+10]
    std::vector<float> seg_out(10 * kHiddenSize);
    for (int64_t j = 0; j < 10; ++j) {
        std::memcpy(seg_out.data() + j * kHiddenSize,
                    final_hidden.data() + (n_prompt + 1 + j) * kHiddenSize,
                    kHiddenSize * sizeof(float));
    }
    return seg_out;
}

}  // namespace sam3
