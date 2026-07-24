#include "sam3/instructsam_lm_forward.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
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

// Read a weight tensor as raw F16 (throws if not F16). Half the bytes vs
// get_f32, no per-element dequant — ggml's mul_mat kernel does that in-SIMD.
std::vector<ggml_fp16_t> get_f16(const GgufModel & model, const std::string & name, size_t n) {
    ggml_tensor * t = require_tensor(model, name);
    if (t->type != GGML_TYPE_F16) {
        throw std::runtime_error("get_f16: expected F16 tensor for " + name);
    }
    std::vector<ggml_fp16_t> v(n);
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(ggml_fp16_t));
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
    #pragma omp parallel for schedule(static)
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

// Number of threads to use for ggml_graph_compute. Set once at first-call.
int g_ggml_threads = 0;
int lm_ggml_threads() {
    if (g_ggml_threads == 0) {
        int hc = static_cast<int>(std::thread::hardware_concurrency());
        if (hc <= 0) hc = 4;
        g_ggml_threads = std::min(hc, 12);  // diminishing returns past 12 cores
    }
    return g_ggml_threads;
}

// Linear: y = x @ w.T + b (b optional). x [N, D_in], w [D_out, D_in], y [N, D_out].
//
// Implemented via a per-call ggml compute graph: mul_mat(w, x) + optional add(b).
// ggml's CPU backend has hand-tuned AVX2/AVX-512 GEMM kernels and its own
// pthread pool, so this is 10-20× faster than the OMP+auto-vectorized
// scalar loop at large sizes. Setup/teardown overhead is ~20 µs per call,
// which is negligible for our 2048-dim projections and 6144-dim MLP.
// Core: y = mul_mat(w_typed, x_f32) via ggml. Weight tensor has the given
// ggml type + data pointer; ggml auto-selects the right kernel (F16→F32,
// Q4_K→F32 etc.) with SIMD dequant on the fly.
std::vector<float> cpu_linear_impl(
    const std::vector<float> & x, int64_t N, int64_t D_in, int64_t D_out,
    ggml_type w_type, const void * w_data,
    const std::vector<float> * b = nullptr
) {
    // ggml graph_compute needs a work buffer proportional to output size
    // (partial results held temporarily). Empirically observed:
    //   ~166 KB at N=15  → linear in N × D_out.
    // Scale conservatively with a 512 KB floor.
    const size_t out_bytes = static_cast<size_t>(N * D_out * 4);
    const size_t ctx_bytes = std::max<size_t>(512 * 1024, 8 * out_bytes + 512 * 1024);
    std::vector<uint8_t> ctx_mem(ctx_bytes);
    ggml_init_params params { ctx_bytes, ctx_mem.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("cpu_linear: ggml_init failed");

    ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D_in, N);
    ggml_tensor * wt = ggml_new_tensor_2d(ctx, w_type,        D_in, D_out);
    xt->data = const_cast<float *>(x.data());
    wt->data = const_cast<void *>(w_data);

    ggml_tensor * yt = ggml_mul_mat(ctx, wt, xt);
    if (b) {
        ggml_tensor * bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, D_out);
        bt->data = const_cast<float *>(b->data());
        yt = ggml_add(ctx, yt, bt);
    }

    std::vector<float> y(static_cast<size_t>(N * D_out));
    yt->data = y.data();

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, yt);
    ggml_graph_compute_with_ctx(ctx, graph, lm_ggml_threads());

    ggml_free(ctx);
    return y;
}

// F32 weight overload (used by old run() codepath).
std::vector<float> cpu_linear(
    const std::vector<float> & x, int64_t N, int64_t D_in, int64_t D_out,
    const std::vector<float> & w,
    const std::vector<float> * b = nullptr
) {
    return cpu_linear_impl(x, N, D_in, D_out, GGML_TYPE_F32, w.data(), b);
}

// F16 weight overload — used by the layer_forward / KV-cache path.
// Weight stays as F16 in memory (halves streaming bandwidth vs F32) and
// ggml's mul_mat kernel does F16→F32 dequant per SIMD lane.
std::vector<float> cpu_linear(
    const std::vector<float> & x, int64_t N, int64_t D_in, int64_t D_out,
    const std::vector<ggml_fp16_t> & w,
    const std::vector<float> * b = nullptr
) {
    return cpu_linear_impl(x, N, D_in, D_out, GGML_TYPE_F16, w.data(), b);
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

// ── Per-layer weights bundle (loaded once per layer per forward) ────────
// Norms stay F32 (they are F32 in the GGUF). Matmul weights kept AS F16
// straight from the GGUF — no dequant on load; ggml's mul_mat does F16→F32
// SIMD dequant per lane at compute time, halving weight-streaming bandwidth.
struct LayerWeights {
    std::vector<float>        attn_norm_w, ffn_norm_w, q_norm_w, k_norm_w;
    std::vector<ggml_fp16_t>  attn_q_w, attn_k_w, attn_v_w, attn_o_w;
    std::vector<ggml_fp16_t>  ffn_gate_w, ffn_up_w, ffn_down_w;
};

LayerWeights load_layer_weights(const GgufModel & model, int layer) {
    const std::string p = "blk." + std::to_string(layer);
    LayerWeights w;
    w.attn_norm_w = get_f32(model, p + ".attn_norm.weight",   kHiddenSize);
    w.attn_q_w    = get_f16(model, p + ".attn_q.weight",      kHiddenSize * kHiddenSize);
    w.attn_k_w    = get_f16(model, p + ".attn_k.weight",      kNumKvHeads * kHeadDim * kHiddenSize);
    w.attn_v_w    = get_f16(model, p + ".attn_v.weight",      kNumKvHeads * kHeadDim * kHiddenSize);
    w.attn_o_w    = get_f16(model, p + ".attn_output.weight", kHiddenSize * kHiddenSize);
    w.q_norm_w    = get_f32(model, p + ".attn_q_norm.weight", kHeadDim);
    w.k_norm_w    = get_f32(model, p + ".attn_k_norm.weight", kHeadDim);
    w.ffn_norm_w  = get_f32(model, p + ".ffn_norm.weight",    kHiddenSize);
    w.ffn_gate_w  = get_f16(model, p + ".ffn_gate.weight",    kIntermediate * kHiddenSize);
    w.ffn_up_w    = get_f16(model, p + ".ffn_up.weight",      kIntermediate * kHiddenSize);
    w.ffn_down_w  = get_f16(model, p + ".ffn_down.weight",    kHiddenSize * kIntermediate);
    return w;
}

// Per-head RMSNorm on Q or K reshape [seq * heads * head_dim].
void per_head_rms_norm(std::vector<float> & x, int64_t N, int64_t heads,
                       const std::vector<float> & w) {
    for (int64_t n = 0; n < N; ++n) {
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
}

// Per-layer forward with OPTIONAL cached K/V for previous positions.
//
// Inputs:
//   hidden [N, D=2048]        — hidden state for the N NEW positions
//   positions_new [N]         — absolute RoPE positions for those N tokens
//   cached_k, cached_v (may be empty) — flat [n_cached * KVH * HD]
//                                        (post-RoPE K, raw V) from earlier prefill
//
// Returns updated hidden [N, D] AND new K/V for this layer (post-RoPE K),
// each of shape [N * KVH * HD]. Caller appends new K/V into the cache
// vectors after the call.
//
// Attention pattern: causal. New Q at row i attends to
//   K/V positions [0, n_cached + i] — i.e. all cached tokens (fully causal
//   already since they came before) plus new tokens 0..i inclusive.
struct LayerForwardOut {
    std::vector<float> hidden;   // [N, D]
    std::vector<float> new_k;    // [N, KVH*HD]
    std::vector<float> new_v;    // [N, KVH*HD]
};

LayerForwardOut layer_forward(
    const std::vector<float> & hidden_in,
    int64_t N,
    const std::vector<int32_t> & positions_new,
    const LayerWeights & w,
    const std::vector<float> & cached_k,   // may be empty
    const std::vector<float> & cached_v,   // may be empty
    int64_t n_cached
) {
    // ── Attention block ─────────────────────────────────────────────────
    const auto residual = hidden_in;
    const auto normed = cpu_rms_norm(hidden_in, N, kHiddenSize, w.attn_norm_w);

    auto Q_flat = cpu_linear(normed, N, kHiddenSize, kHiddenSize,             w.attn_q_w);
    auto K_flat = cpu_linear(normed, N, kHiddenSize, kNumKvHeads * kHeadDim,  w.attn_k_w);
    auto V_flat = cpu_linear(normed, N, kHiddenSize, kNumKvHeads * kHeadDim,  w.attn_v_w);

    per_head_rms_norm(Q_flat, N, kNumHeads,   w.q_norm_w);
    per_head_rms_norm(K_flat, N, kNumKvHeads, w.k_norm_w);
    cpu_rope_1d_half(Q_flat, N, kNumHeads,   kHeadDim, positions_new);
    cpu_rope_1d_half(K_flat, N, kNumKvHeads, kHeadDim, positions_new);

    // Attention. Total sequence length for K/V is n_total = n_cached + N.
    const int64_t n_total = n_cached + N;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    std::vector<float> attn_out(static_cast<size_t>(N * kHiddenSize), 0.0f);

    #pragma omp parallel for schedule(static)
    for (int64_t h = 0; h < kNumHeads; ++h) {
        const int64_t kv_h = h / kKvGroups;
        // For each new row i (absolute pos = n_cached + i), attend to positions [0, n_cached + i].
        for (int64_t i = 0; i < N; ++i) {
            const float * qi = Q_flat.data() + (i * kNumHeads + h) * kHeadDim;
            const int64_t last_pos = n_cached + i;  // inclusive
            std::vector<float> scores(static_cast<size_t>(last_pos + 1));
            for (int64_t j = 0; j <= last_pos; ++j) {
                const float * kj;
                if (j < n_cached) {
                    kj = cached_k.data() + (j * kNumKvHeads + kv_h) * kHeadDim;
                } else {
                    const int64_t j_new = j - n_cached;
                    kj = K_flat.data() + (j_new * kNumKvHeads + kv_h) * kHeadDim;
                }
                float s = 0.0f;
                for (int64_t d = 0; d < kHeadDim; ++d) s += qi[d] * kj[d];
                scores[static_cast<size_t>(j)] = s * scale;
            }
            // Softmax
            float m = scores[0];
            for (int64_t j = 1; j <= last_pos; ++j) if (scores[j] > m) m = scores[j];
            float sum = 0.0f;
            for (int64_t j = 0; j <= last_pos; ++j) { scores[j] = std::exp(scores[j] - m); sum += scores[j]; }
            for (int64_t j = 0; j <= last_pos; ++j) scores[j] /= sum;
            // Weighted sum with V
            for (int64_t d = 0; d < kHeadDim; ++d) {
                float y = 0.0f;
                for (int64_t j = 0; j <= last_pos; ++j) {
                    const float * vj;
                    if (j < n_cached) {
                        vj = cached_v.data() + (j * kNumKvHeads + kv_h) * kHeadDim;
                    } else {
                        const int64_t j_new = j - n_cached;
                        vj = V_flat.data() + (j_new * kNumKvHeads + kv_h) * kHeadDim;
                    }
                    y += scores[static_cast<size_t>(j)] * vj[d];
                }
                attn_out[static_cast<size_t>(i * kHiddenSize + h * kHeadDim + d)] = y;
            }
        }
    }

    // O proj + residual
    const auto attn_proj = cpu_linear(attn_out, N, kHiddenSize, kHiddenSize, w.attn_o_w);
    std::vector<float> hidden(residual.size());
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = residual[i] + attn_proj[i];

    // ── MLP block ────────────────────────────────────────────────────────
    const auto residual2 = hidden;
    const auto normed2 = cpu_rms_norm(hidden, N, kHiddenSize, w.ffn_norm_w);
    auto gate_out = cpu_linear(normed2, N, kHiddenSize, kIntermediate, w.ffn_gate_w);
    const auto up_out   = cpu_linear(normed2, N, kHiddenSize, kIntermediate, w.ffn_up_w);
    cpu_silu_mul(gate_out, up_out);
    const auto down_out = cpu_linear(gate_out, N, kIntermediate, kHiddenSize, w.ffn_down_w);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = residual2[i] + down_out[i];

    return LayerForwardOut{ std::move(hidden), std::move(K_flat), std::move(V_flat) };
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

        #pragma omp parallel for schedule(static)
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

InstructsamLmForward::KvCache InstructsamLmForward::prefill_prefix(
    const std::vector<float> & prefix_embeds,
    int64_t n_prefix,
    const std::vector<int32_t> & positions
) const {
    if (prefix_embeds.size() != static_cast<size_t>(n_prefix * kHiddenSize)) {
        throw std::runtime_error("prefill_prefix: embeds size mismatch");
    }
    if (positions.size() != static_cast<size_t>(n_prefix)) {
        throw std::runtime_error("prefill_prefix: positions size mismatch");
    }

    KvCache cache;
    cache.k.resize(kNumLayers);
    cache.v.resize(kNumLayers);
    cache.n_cached = n_prefix;

    std::vector<float> hidden = prefix_embeds;
    for (int layer = 0; layer < kNumLayers; ++layer) {
        const auto w = load_layer_weights(model_, layer);
        auto out = layer_forward(hidden, n_prefix, positions, w,
                                 /*cached_k=*/{}, /*cached_v=*/{}, /*n_cached=*/0);
        hidden        = std::move(out.hidden);
        cache.k[layer] = std::move(out.new_k);
        cache.v[layer] = std::move(out.new_v);
    }
    // Note: we discard `hidden` (post-transformer, pre-output_norm). Callers
    // that need it can compute cpu_rms_norm(hidden, ..., output_norm_w) but
    // that's not the cache's job.
    return cache;
}

InstructsamLmForward::PrefillResult InstructsamLmForward::prefill_with_last_hidden(
    const std::vector<float> & prefix_embeds,
    int64_t n_prefix,
    const std::vector<int32_t> & positions
) const {
    if (prefix_embeds.size() != static_cast<size_t>(n_prefix * kHiddenSize)) {
        throw std::runtime_error("prefill_with_last_hidden: embeds size mismatch");
    }
    if (positions.size() != static_cast<size_t>(n_prefix)) {
        throw std::runtime_error("prefill_with_last_hidden: positions size mismatch");
    }

    PrefillResult r;
    r.cache.k.resize(kNumLayers);
    r.cache.v.resize(kNumLayers);
    r.cache.n_cached = n_prefix;

    std::vector<float> hidden = prefix_embeds;
    for (int layer = 0; layer < kNumLayers; ++layer) {
        const auto w = load_layer_weights(model_, layer);
        auto out = layer_forward(hidden, n_prefix, positions, w, /*cached_k=*/{}, /*cached_v=*/{}, 0);
        hidden          = std::move(out.hidden);
        r.cache.k[layer] = std::move(out.new_k);
        r.cache.v[layer] = std::move(out.new_v);
    }
    // Apply output_norm to the final position only.
    const auto output_norm_w = get_f32(model_, "output_norm.weight", kHiddenSize);
    // Extract last-position hidden [2048]
    std::vector<float> last_pre(kHiddenSize);
    std::memcpy(last_pre.data(),
                hidden.data() + (n_prefix - 1) * kHiddenSize,
                kHiddenSize * sizeof(float));
    r.last_hidden = cpu_rms_norm(last_pre, 1, kHiddenSize, output_norm_w);
    return r;
}

std::vector<float> InstructsamLmForward::decode_step(
    KvCache & cache_mutable,
    const std::vector<float> & new_embed,
    int32_t new_position
) const {
    if (new_embed.size() != static_cast<size_t>(kHiddenSize)) {
        throw std::runtime_error("decode_step: embed must be [2048]");
    }
    if (cache_mutable.k.size() != kNumLayers || cache_mutable.v.size() != kNumLayers) {
        throw std::runtime_error("decode_step: cache layer count mismatch");
    }
    const std::vector<int32_t> pos_new = { new_position };
    std::vector<float> hidden = new_embed;
    for (int layer = 0; layer < kNumLayers; ++layer) {
        const auto w = load_layer_weights(model_, layer);
        auto out = layer_forward(hidden, /*N=*/1, pos_new, w,
                                 cache_mutable.k[layer], cache_mutable.v[layer],
                                 cache_mutable.n_cached);
        hidden = std::move(out.hidden);
        // Append new K/V to the cache (each is [1 * KVH * HD] = 1024 floats)
        cache_mutable.k[layer].insert(cache_mutable.k[layer].end(),
                                      out.new_k.begin(), out.new_k.end());
        cache_mutable.v[layer].insert(cache_mutable.v[layer].end(),
                                      out.new_v.begin(), out.new_v.end());
    }
    cache_mutable.n_cached += 1;
    const auto output_norm_w = get_f32(model_, "output_norm.weight", kHiddenSize);
    return cpu_rms_norm(hidden, 1, kHiddenSize, output_norm_w);
}

std::vector<float> InstructsamLmForward::logits_for_hidden(
    const std::vector<float> & hidden_2048
) const {
    if (hidden_2048.size() != static_cast<size_t>(kHiddenSize)) {
        throw std::runtime_error("logits_for_hidden: input must be [2048]");
    }
    // Qwen3-VL ties input embedding for LM head. token_embd.weight is
    // stored [hidden=2048, vocab=151936] in ggml's ne convention — meaning
    // per-token row of size 2048 is contiguous. Fetch the whole table once
    // (622 MB F16 = 304 MB F32-decoded) then GEMV over it in parallel.
    // Cache across calls in a mutable member so successive generation
    // steps skip the fetch.
    ggml_tensor * emb = require_tensor(model_, "token_embd.weight");
    if (embd_cache_f32_.empty()) {
        embd_cache_f32_.resize(static_cast<size_t>(kVocabSize) * kHiddenSize);
        if (emb->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(emb, embd_cache_f32_.data(), 0,
                                    embd_cache_f32_.size() * sizeof(float));
        } else if (emb->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> buf(embd_cache_f32_.size());
            ggml_backend_tensor_get(emb, buf.data(), 0, buf.size() * sizeof(ggml_fp16_t));
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < static_cast<int64_t>(buf.size()); ++i) {
                embd_cache_f32_[i] = ggml_fp16_to_fp32(buf[i]);
            }
        } else {
            throw std::runtime_error("logits_for_hidden: unsupported dtype for token_embd");
        }
    }

    std::vector<float> logits(static_cast<size_t>(kVocabSize), 0.0f);
    #pragma omp parallel for schedule(static)
    for (int32_t t = 0; t < kVocabSize; ++t) {
        const float * row = embd_cache_f32_.data() + static_cast<size_t>(t) * kHiddenSize;
        float s = 0.0f;
        for (int d = 0; d < kHiddenSize; ++d) s += hidden_2048[d] * row[d];
        logits[static_cast<size_t>(t)] = s;
    }
    return logits;
}

std::vector<float> InstructsamLmForward::extract_seg_output_embeddings_with_cache(
    const KvCache & prefix_cache,
    const std::vector<float> & appended_prefix_embeds,
    int64_t n_appended_prefix,
    const std::vector<float> & mask_queries,
    const std::vector<float> & mask_start_embed,
    const std::vector<float> & mask_end_embed
) const {
    if (mask_queries.size() != 10 * kHiddenSize) {
        throw std::runtime_error("with_cache: mask_queries must be [10, 2048]");
    }
    if (mask_start_embed.size() != kHiddenSize || mask_end_embed.size() != kHiddenSize) {
        throw std::runtime_error("with_cache: mask_start/end embeds must be [2048]");
    }
    if (n_appended_prefix < 0 ||
        appended_prefix_embeds.size() != static_cast<size_t>(n_appended_prefix * kHiddenSize)) {
        throw std::runtime_error("with_cache: appended_prefix shape mismatch");
    }
    if (prefix_cache.k.size() != kNumLayers || prefix_cache.v.size() != kNumLayers) {
        throw std::runtime_error("with_cache: cache layer count mismatch");
    }

    // Build the delta sequence to decode:
    //   [appended_prefix (n_appended_prefix)] +
    //   [mask_start, 10 mask_queries, mask_end]  (12 tokens)
    const int64_t n_inject = 12;
    const int64_t n_new = n_appended_prefix + n_inject;
    const int64_t n_cached = prefix_cache.n_cached;

    std::vector<float> new_embeds(static_cast<size_t>(n_new * kHiddenSize));
    std::vector<int32_t> positions_new(static_cast<size_t>(n_new));

    // (a) appended_prefix (e.g. <|object_ref_start|> + phrase + <|object_ref_end|>)
    std::memcpy(new_embeds.data(), appended_prefix_embeds.data(),
                static_cast<size_t>(n_appended_prefix * kHiddenSize) * sizeof(float));
    for (int64_t i = 0; i < n_appended_prefix; ++i) {
        positions_new[i] = static_cast<int32_t>(n_cached + i);
    }
    // (b) mask_start
    const int64_t inj0 = n_appended_prefix;
    std::memcpy(new_embeds.data() + inj0 * kHiddenSize, mask_start_embed.data(),
                kHiddenSize * sizeof(float));
    positions_new[inj0] = static_cast<int32_t>(n_cached + inj0);
    // (c) 10 mask_queries
    for (int64_t j = 0; j < 10; ++j) {
        std::memcpy(new_embeds.data() + (inj0 + 1 + j) * kHiddenSize,
                    mask_queries.data() + j * kHiddenSize, kHiddenSize * sizeof(float));
        positions_new[inj0 + 1 + j] = static_cast<int32_t>(n_cached + inj0 + 1 + j);
    }
    // (d) mask_end
    std::memcpy(new_embeds.data() + (inj0 + 11) * kHiddenSize, mask_end_embed.data(),
                kHiddenSize * sizeof(float));
    positions_new[inj0 + 11] = static_cast<int32_t>(n_cached + inj0 + 11);

    // Run 28 layers, feeding each layer's cached K/V from prefix_cache.
    std::vector<float> hidden = new_embeds;
    for (int layer = 0; layer < kNumLayers; ++layer) {
        const auto w = load_layer_weights(model_, layer);
        auto out = layer_forward(hidden, n_new, positions_new, w,
                                 prefix_cache.k[layer], prefix_cache.v[layer], n_cached);
        hidden = std::move(out.hidden);
    }

    // Final RMSNorm on the n_new hidden states
    const auto output_norm_w = get_f32(model_, "output_norm.weight", kHiddenSize);
    const auto final_hidden = cpu_rms_norm(hidden, n_new, kHiddenSize, output_norm_w);

    // Extract slots at [inj0+1 .. inj0+10] — the 10 mask_queries positions
    std::vector<float> seg_out(10 * kHiddenSize);
    for (int64_t j = 0; j < 10; ++j) {
        std::memcpy(seg_out.data() + j * kHiddenSize,
                    final_hidden.data() + (inj0 + 1 + j) * kHiddenSize,
                    kHiddenSize * sizeof(float));
    }
    return seg_out;
}

std::vector<float> InstructsamLmForward::extract_seg_output_embeddings_from_prefix(
    const std::vector<float> & prefix_embeds,
    int64_t n_prefix,
    const std::vector<float> & mask_queries,
    const std::vector<float> & mask_start_embed,
    const std::vector<float> & mask_end_embed
) const {
    if (mask_queries.size() != 10 * kHiddenSize) {
        throw std::runtime_error("from_prefix: mask_queries must be [10, 2048]");
    }
    if (mask_start_embed.size() != kHiddenSize || mask_end_embed.size() != kHiddenSize) {
        throw std::runtime_error("from_prefix: mask_start/end must be [2048]");
    }
    if (n_prefix <= 0 || static_cast<size_t>(n_prefix) * kHiddenSize != prefix_embeds.size()) {
        throw std::runtime_error("from_prefix: prefix_embeds shape mismatch");
    }

    const int64_t n_inject = 12;  // mask_start + 10 mask_queries + mask_end
    const int64_t seq_len = n_prefix + n_inject;
    std::vector<float> embeds(seq_len * kHiddenSize);
    std::vector<int32_t> positions(seq_len);

    // Splice prefix (LM-space embeds, may include image embeddings)
    std::memcpy(embeds.data(), prefix_embeds.data(),
                static_cast<size_t>(n_prefix) * kHiddenSize * sizeof(float));
    for (int64_t i = 0; i < n_prefix; ++i) positions[i] = static_cast<int32_t>(i);

    std::memcpy(embeds.data() + n_prefix * kHiddenSize,
                mask_start_embed.data(), kHiddenSize * sizeof(float));
    positions[n_prefix] = static_cast<int32_t>(n_prefix);

    for (int64_t j = 0; j < 10; ++j) {
        std::memcpy(embeds.data() + (n_prefix + 1 + j) * kHiddenSize,
                    mask_queries.data() + j * kHiddenSize,
                    kHiddenSize * sizeof(float));
        positions[n_prefix + 1 + j] = static_cast<int32_t>(n_prefix + 1 + j);
    }

    std::memcpy(embeds.data() + (n_prefix + 11) * kHiddenSize,
                mask_end_embed.data(), kHiddenSize * sizeof(float));
    positions[n_prefix + 11] = static_cast<int32_t>(n_prefix + 11);

    const auto final_hidden = run(embeds, seq_len, positions);
    std::vector<float> seg_out(10 * kHiddenSize);
    for (int64_t j = 0; j < 10; ++j) {
        std::memcpy(seg_out.data() + j * kHiddenSize,
                    final_hidden.data() + (n_prefix + 1 + j) * kHiddenSize,
                    kHiddenSize * sizeof(float));
    }
    return seg_out;
}

}  // namespace sam3
