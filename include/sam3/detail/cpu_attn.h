#pragma once

// Shared multi-head attention helper backed by ggml_flash_attn_ext.
//
// For the SAM3 vision encoder's global-attention layers (7/15/23/31),
// the manual O(N²·D) scalar loop takes ~25s per layer at N=5184. Flash
// attention runs the same math via a fused block kernel that is
// ~10-30× faster on CPU (and doesn't materialize the N×N scores matrix).

#include "sam3/detail/cpu_linear.h"  // cpu_linear_threads()

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sam3 {
namespace detail {

// Multi-head scaled dot-product attention. Inputs are laid out with head_dim
// innermost — matching (batch, n_head, seq, head_dim) row-major memory,
// which is what our vision encoder already builds via to_heads().
//
// ggml_flash_attn_ext requires:
//   Q/K/V type = F16 (mask/softmax scratch space)
//   Q ne = [head_dim, seq_q,  n_head,    batch]
//   K ne = [head_dim, seq_kv, n_head_kv, batch]
//   V ne = [head_dim, seq_kv, n_head_kv, batch]  (NOT transposed)
// The output ne = [head_dim, n_head, seq_q, batch] — note head/seq swap.
// We fetch and reorder back to (batch, n_head, seq_q, head_dim).
inline std::vector<float> flash_attn(
    const std::vector<float> & Q,    // [batch, n_head, seq_q, head_dim] f32
    const std::vector<float> & K,    // [batch, n_head_kv, seq_kv, head_dim] f32
    const std::vector<float> & V,    // same layout as K
    int64_t batch, int64_t n_head, int64_t n_head_kv,
    int64_t seq_q, int64_t seq_kv, int64_t head_dim,
    float scale
) {
    // Q stays as F32 (fast-path requirement in ggml_compute_forward_flash_attn_ext);
    // K/V are F16.
    const size_t nK = static_cast<size_t>(batch) * n_head_kv * seq_kv * head_dim;
    std::vector<ggml_fp16_t> K_f16(nK), V_f16(nK);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(nK); ++i) K_f16[i] = ggml_fp32_to_fp16(K[i]);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(nK); ++i) V_f16[i] = ggml_fp32_to_fp16(V[i]);

    const size_t ctx_bytes = 4 * 1024 * 1024;
    std::vector<uint8_t> ctx_mem(ctx_bytes);
    ggml_init_params params { ctx_bytes, ctx_mem.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("flash_attn: ggml_init failed");

    ggml_tensor * qt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_dim, seq_q,  n_head,    batch);
    ggml_tensor * kt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, seq_kv, n_head_kv, batch);
    ggml_tensor * vt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, seq_kv, n_head_kv, batch);
    qt->data = const_cast<float *>(Q.data());
    kt->data = K_f16.data();
    vt->data = V_f16.data();

    ggml_tensor * yt = ggml_flash_attn_ext(ctx, qt, kt, vt, nullptr, scale, 0.0f, 0.0f);

    // Fetch flash_attn output directly into an intermediate buffer, then
    // reorder into (batch, n_head, seq_q, head_dim) via a plain CPU permute.
    // Doing the transpose via ggml_permute+ggml_cont in the graph crashed
    // on stride asserts, so keep the transpose outside the compute graph.
    std::vector<float> fa_out(static_cast<size_t>(batch) * seq_q * n_head * head_dim);
    yt->data = fa_out.data();

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, yt);
    ggml_graph_compute_with_ctx(ctx, graph, cpu_linear_threads());
    ggml_free(ctx);

    // flash_attn output ne=[D, H, S, B] → memory layout (B, S, H, D).
    // Convert to (B, H, S, D).
    std::vector<float> out(static_cast<size_t>(batch) * n_head * seq_q * head_dim);
    #pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < n_head; ++h) {
            for (int64_t s = 0; s < seq_q; ++s) {
                const size_t src = ((b * seq_q + s) * n_head + h) * head_dim;
                const size_t dst = ((b * n_head + h) * seq_q + s) * head_dim;
                std::memcpy(out.data() + dst, fa_out.data() + src, head_dim * sizeof(float));
            }
        }
    }
    return out;
}

}  // namespace detail
}  // namespace sam3
