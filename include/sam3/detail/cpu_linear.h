#pragma once

// Shared CPU matmul helper backed by ggml_mul_mat.
//
// Used by both the LM forward (F16/Q4_K weights, no bias) and the SAM3
// vision encoder (F32 weights with bias). Same graph shape, different
// argument sets — all overloads dispatch to cpu_linear_impl() which builds
// a per-call ggml compute graph and runs it via the CPU backend.

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace sam3 {
namespace detail {

// Number of threads to use for ggml_graph_compute. Set once at first call.
inline int cpu_linear_threads() {
    static int cached = 0;
    if (cached == 0) {
        int hc = static_cast<int>(std::thread::hardware_concurrency());
        if (hc <= 0) hc = 4;
        cached = std::min(hc, 12);
    }
    return cached;
}

// Core: y = mul_mat(w_typed, x_f32) + optional bias. Weight is passed as
// (ggml_type, raw byte pointer) so this works for F32/F16/Q4_K/Q6_K/etc.
inline std::vector<float> cpu_linear_impl(
    const std::vector<float> & x, int64_t N, int64_t D_in, int64_t D_out,
    ggml_type w_type, const void * w_data,
    const std::vector<float> * b
) {
    // ggml_graph_compute_with_ctx allocates work buffer INSIDE the ctx.
    // Empirically the buffer scales roughly as 8× the output tensor size for
    // blocked GEMM (partial accumulators). 512 KB floor covers the N=1
    // decode-step case.
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
    ggml_graph_compute_with_ctx(ctx, graph, cpu_linear_threads());

    ggml_free(ctx);
    return y;
}

// F32 weight overload.
inline std::vector<float> cpu_linear(
    const std::vector<float> & x, int64_t N, int64_t D_in, int64_t D_out,
    const std::vector<float> & w,
    const std::vector<float> * b = nullptr
) {
    return cpu_linear_impl(x, N, D_in, D_out, GGML_TYPE_F32, w.data(), b);
}

// F32 weight + F32 bias (by-value bias for convenience — SAM3 vision uses
// this since every projection has a bias). Wraps to the pointer overload.
inline std::vector<float> cpu_linear(
    const std::vector<float> & x, int64_t N, int64_t D_in, int64_t D_out,
    const std::vector<float> & w,
    const std::vector<float> & b
) {
    return cpu_linear_impl(x, N, D_in, D_out, GGML_TYPE_F32, w.data(), &b);
}

// F16 weight overload — halves streaming bandwidth vs F32.
inline std::vector<float> cpu_linear(
    const std::vector<float> & x, int64_t N, int64_t D_in, int64_t D_out,
    const std::vector<ggml_fp16_t> & w,
    const std::vector<float> * b = nullptr
) {
    return cpu_linear_impl(x, N, D_in, D_out, GGML_TYPE_F16, w.data(), b);
}

}  // namespace detail
}  // namespace sam3
