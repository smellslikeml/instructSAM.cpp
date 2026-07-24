#pragma once

// Shared CPU conv2d helpers backed by ggml_conv_2d / ggml_conv_transpose_2d.
// Used by the SAM3 vision encoder's FPN neck (Conv1x1, Conv3x3, ConvT k=2 s=2).
//
// Weight tensors are passed as (ggml_type, raw pointer + shape) so this
// works for F32 and F16 weights from the GGUF without a copy.

#include "sam3/detail/cpu_linear.h"  // for cpu_linear_threads()

#include "ggml.h"
#include "ggml-cpu.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sam3 {
namespace detail {

// Conv2d input tensor layout (ggml ne convention): [W, H, C, N=1].
// Weight ne=[KW, KH, IC, OC] which matches how PyTorch [OC, IC, KH, KW]
// weights are stored in GGUF (ne reverses PyTorch shape).
//
// Returns output in ggml row-major order matching ne=[W_out, H_out, OC, 1].
// For SAM3 that means memory layout is (c, h, w) with w fastest — matches
// PyTorch [B, C, H, W] backbone_features format.
inline std::vector<float> ggml_conv2d(
    const std::vector<float> & x, int64_t Cin, int64_t H, int64_t W,
    int64_t Cout, int64_t KH, int64_t KW,
    ggml_type w_type, const void * w_data,
    const std::vector<float> * b,
    int stride, int pad
) {
    const size_t ctx_bytes = 32 * 1024 * 1024;
    std::vector<uint8_t> ctx_mem(ctx_bytes);
    ggml_init_params params { ctx_bytes, ctx_mem.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("ggml_conv2d: ggml_init failed");

    // ggml_conv_2d requires F16 kernel (its im2col step asserts src0 F16).
    // If the caller passes F32 weights, convert on the fly to a stashed F16
    // buffer.
    std::vector<ggml_fp16_t> w_fp16;
    if (w_type == GGML_TYPE_F32) {
        const size_t nw = static_cast<size_t>(Cout) * Cin * KH * KW;
        w_fp16.resize(nw);
        const float * wf = static_cast<const float *>(w_data);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(nw); ++i) {
            w_fp16[i] = ggml_fp32_to_fp16(wf[i]);
        }
        w_type = GGML_TYPE_F16;
        w_data = w_fp16.data();
    }

    // Input: ne=[W, H, Cin, 1]  — treat x as row-major [Cin, H, W] with W fastest
    ggml_tensor * xt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, Cin, 1);
    xt->data = const_cast<float *>(x.data());
    // Kernel: ne=[KW, KH, Cin, Cout]
    ggml_tensor * wt = ggml_new_tensor_4d(ctx, w_type, KW, KH, Cin, Cout);
    wt->data = const_cast<void *>(w_data);

    // ggml_conv_2d_direct (not ggml_conv_2d) — the latter expands into
    // im2col + mul_mat + reshape + permute + cont which needs a HUGE
    // intermediate tensor (~380 MB for our 288×288×256 3×3 conv). The
    // direct kernel skips im2col and streams through a smaller work
    // buffer allocated via graph_plan.
    ggml_tensor * yt = ggml_conv_2d_direct(ctx, wt, xt, stride, stride, pad, pad, 1, 1);

    const int64_t OH = (H + 2 * pad - KH) / stride + 1;
    const int64_t OW = (W + 2 * pad - KW) / stride + 1;
    std::vector<float> y(static_cast<size_t>(Cout * OH * OW));
    yt->data = y.data();  // no_alloc=true → we own output storage

    // Bias broadcasting via ggml_add can be finicky at large sizes; do the
    // add as a plain OMP loop after the conv.
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, yt);
    ggml_cplan cplan = ggml_graph_plan(graph, cpu_linear_threads(), nullptr);
    std::vector<uint8_t> work_buf(cplan.work_size + 64);
    uintptr_t addr = reinterpret_cast<uintptr_t>(work_buf.data());
    uintptr_t pad_al = (64 - (addr % 64)) % 64;
    cplan.work_data = work_buf.data() + pad_al;
    ggml_graph_compute(graph, &cplan);

    if (b) {
        #pragma omp parallel for schedule(static)
        for (int64_t oc = 0; oc < Cout; ++oc) {
            const float bo = (*b)[oc];
            float * chan = y.data() + oc * OH * OW;
            for (int64_t p = 0; p < OH * OW; ++p) chan[p] += bo;
        }
    }
    ggml_free(ctx);
    return y;
}

// ConvTranspose2d with kernel=stride (k=2, s=2 in our use case).
// Weight ne=[KW, KH, Cout, Cin] since PyTorch ConvTranspose stores
// weight as [Cin, Cout, KH, KW] which reverses to ne=[KW, KH, Cout, Cin].
inline std::vector<float> ggml_conv_transpose2d_ks(
    const std::vector<float> & x, int64_t Cin, int64_t H, int64_t W,
    int64_t Cout, int64_t KH, int64_t KW,
    ggml_type w_type, const void * w_data,
    const std::vector<float> * b,
    int stride
) {
    const size_t ctx_bytes = 32 * 1024 * 1024;
    std::vector<uint8_t> ctx_mem(ctx_bytes);
    ggml_init_params params { ctx_bytes, ctx_mem.data(), /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("ggml_conv_transpose2d: ggml_init failed");

    // Same F16-kernel requirement as ggml_conv_2d.
    std::vector<ggml_fp16_t> w_fp16;
    if (w_type == GGML_TYPE_F32) {
        const size_t nw = static_cast<size_t>(Cin) * Cout * KH * KW;
        w_fp16.resize(nw);
        const float * wf = static_cast<const float *>(w_data);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(nw); ++i) {
            w_fp16[i] = ggml_fp32_to_fp16(wf[i]);
        }
        w_type = GGML_TYPE_F16;
        w_data = w_fp16.data();
    }

    ggml_tensor * xt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, Cin, 1);
    xt->data = const_cast<float *>(x.data());
    ggml_tensor * wt = ggml_new_tensor_4d(ctx, w_type, KW, KH, Cout, Cin);
    wt->data = const_cast<void *>(w_data);

    ggml_tensor * yt = ggml_conv_transpose_2d_p0(ctx, wt, xt, stride);

    const int64_t OH = (H - 1) * stride + KH;
    const int64_t OW = (W - 1) * stride + KW;
    std::vector<float> y(static_cast<size_t>(Cout * OH * OW));
    yt->data = y.data();  // no_alloc=true → we own output storage

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, yt);
    ggml_cplan cplan = ggml_graph_plan(graph, cpu_linear_threads(), nullptr);
    std::vector<uint8_t> work_buf(cplan.work_size + 64);
    uintptr_t addr = reinterpret_cast<uintptr_t>(work_buf.data());
    uintptr_t pad_al = (64 - (addr % 64)) % 64;
    cplan.work_data = work_buf.data() + pad_al;
    ggml_graph_compute(graph, &cplan);
    std::memcpy(y.data(), yt->data, y.size() * sizeof(float));

    if (b) {
        #pragma omp parallel for schedule(static)
        for (int64_t oc = 0; oc < Cout; ++oc) {
            const float bo = (*b)[oc];
            float * chan = y.data() + oc * OH * OW;
            for (int64_t p = 0; p < OH * OW; ++p) chan[p] += bo;
        }
    }
    ggml_free(ctx);
    return y;
}

}  // namespace detail
}  // namespace sam3
