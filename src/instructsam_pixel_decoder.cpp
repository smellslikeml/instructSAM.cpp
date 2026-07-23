#include "sam3/instructsam_pixel_decoder.h"

#include "ggml-alloc.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

constexpr int32_t kChannels = 256;
constexpr int32_t kGroupNormGroups = 8;
constexpr float   kGroupNormEps = 1e-5f;

ggml_tensor * require_tensor(const GgufModel & model, const std::string & name) {
    ggml_tensor * t = model.find_weight(name);
    if (t == nullptr) throw std::runtime_error("pixel_decoder: missing tensor: " + name);
    return t;
}

ggml_tensor * ensure_f32(ggml_context * ctx, ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) return t;
    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

// Bias a 4D output [W, H, C, N] with a per-channel bias [C].
ggml_tensor * add_bias_1d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * b, int64_t C) {
    ggml_tensor * b4 = ggml_reshape_4d(ctx, ensure_f32(ctx, b), 1, 1, C, 1);
    return ggml_add(ctx, x, b4);
}

// Conv2d wrapper for PyTorch-format weights.
// PyTorch stores Conv2d weight as [OutC, InC, KH, KW] (C-order contiguous).
// When loaded into GGUF (column-major reversal), ggml sees ne =
// [KW, KH, InC, OutC] — which IS the layout ggml_conv_2d expects. No
// permute needed. (Contrast with sam3cpp's stock conv2d_bias which
// permutes because their MLX-origin weights have a different layout.)
ggml_tensor * conv2d_bias(
    ggml_context * ctx,
    ggml_tensor * w, ggml_tensor * b, ggml_tensor * x,
    int s0, int s1, int p0, int p1
) {
    ggml_tensor * w_f32 = ensure_f32(ctx, w);
    ggml_tensor * y = ggml_conv_2d(ctx, w_f32, x, s0, s1, p0, p1, 1, 1);
    return add_bias_1d(ctx, y, b, w_f32->ne[3]);
}

// Standard GroupNorm (not the interleaved variant sam3cpp uses for MLX
// weights). InstructSAM weights are PyTorch-standard channel layout.
ggml_tensor * group_norm(
    ggml_context * ctx,
    ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias
) {
    ggml_tensor * y = ggml_group_norm(ctx, x, kGroupNormGroups, kGroupNormEps);
    ggml_tensor * w = ggml_reshape_4d(ctx, ensure_f32(ctx, weight), 1, 1, x->ne[2], 1);
    ggml_tensor * b = ggml_reshape_4d(ctx, ensure_f32(ctx, bias),   1, 1, x->ne[2], 1);
    y = ggml_add(ctx, ggml_mul(ctx, y, w), b);
    return y;
}

}  // namespace

InstructsamPixelDecoder::InstructsamPixelDecoder(const GgufModel & model) : model_(model) {}

std::vector<float> InstructsamPixelDecoder::run(
    const std::vector<float> & bb0, const std::vector<int64_t> & bb0_shape,
    const std::vector<float> & bb1, const std::vector<int64_t> & bb1_shape,
    const std::vector<float> & bb2, const std::vector<int64_t> & bb2_shape
) const {
    if (bb0_shape.size() != 3 || bb0_shape[0] != kChannels) throw std::runtime_error("bb0 must be [256, H, W]");
    if (bb1_shape.size() != 3 || bb1_shape[0] != kChannels) throw std::runtime_error("bb1 must be [256, H, W]");
    if (bb2_shape.size() != 3 || bb2_shape[0] != kChannels) throw std::runtime_error("bb2 must be [256, H, W]");

    const int64_t H0 = bb0_shape[1], W0 = bb0_shape[2];
    const int64_t H1 = bb1_shape[1], W1 = bb1_shape[2];
    const int64_t H2 = bb2_shape[1], W2 = bb2_shape[2];
    if (H0 != 2 * H1 || W0 != 2 * W1 || H1 != 2 * H2 || W1 != 2 * W2) {
        throw std::runtime_error("pixel_decoder expects each level to be 2x the next (nearest-neighbor upsample assumption)");
    }

    ggml_backend_t backend = model_.backend();
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_t backends[2] = { backend, nullptr };
    int n_backends = 1;
    if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        cpu_backend = model_.cpu_backend();
        if (cpu_backend != nullptr) { backends[1] = cpu_backend; n_backends = 2; }
    }

    std::vector<float> stage0_host;  // stage0 output at bb1 resolution
    std::vector<float> stage1_host;  // stage1 output at bb0 resolution == final pixel_embed

    // ── Stage 0: upscale(bb2) + bb1 → conv[0] → gn[0] → relu ─────────────
    {
        const size_t graph_size = 16384;
        const size_t ctx_size = ggml_tensor_overhead() * graph_size +
                                ggml_graph_overhead_custom(graph_size, false);
        std::vector<uint8_t> ctx_buf(ctx_size);
        ggml_context * ctx = ggml_init({ctx_size, ctx_buf.data(), true});
        if (!ctx) throw std::runtime_error("ggml_init failed");
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, graph_size, false, true);
        if (!sched) { ggml_free(ctx); throw std::runtime_error("sched_new failed"); }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

        // ggml layout is column-major so shape order is [W, H, C, N].
        ggml_tensor * bb1_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W1, H1, kChannels, 1);
        ggml_tensor * bb2_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W2, H2, kChannels, 1);
        ggml_backend_sched_set_tensor_backend(sched, bb1_t, backend);
        ggml_backend_sched_set_tensor_backend(sched, bb2_t, backend);

        ggml_tensor * up = ggml_upscale(ctx, bb2_t, 2, GGML_SCALE_MODE_NEAREST);
        ggml_tensor * added = ggml_add(ctx, bb1_t, up);
        ggml_tensor * convd = conv2d_bias(
            ctx,
            require_tensor(model_, "segmentation_head.pixel_decoder.conv_layers.0.weight"),
            require_tensor(model_, "segmentation_head.pixel_decoder.conv_layers.0.bias"),
            added,
            /*s0=*/1, /*s1=*/1, /*p0=*/1, /*p1=*/1);
        ggml_tensor * normed = group_norm(
            ctx, convd,
            require_tensor(model_, "segmentation_head.pixel_decoder.norms.0.weight"),
            require_tensor(model_, "segmentation_head.pixel_decoder.norms.0.bias"));
        ggml_tensor * relu = ggml_relu(ctx, normed);
        ggml_tensor * capture = ggml_cont(ctx, relu);
        ggml_build_forward_expand(gf, capture);
        ggml_backend_sched_alloc_graph(sched, gf);

        ggml_backend_tensor_set(bb1_t, bb1.data(), 0, bb1.size() * sizeof(float));
        ggml_backend_tensor_set(bb2_t, bb2.data(), 0, bb2.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(sched); ggml_free(ctx);
            throw std::runtime_error("pixel_decoder stage0 compute failed");
        }
        stage0_host.resize(static_cast<size_t>(ggml_nelements(capture)));
        ggml_backend_tensor_get(capture, stage0_host.data(), 0, stage0_host.size() * sizeof(float));
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }

    // ── Stage 1: upscale(stage0_out) + bb0 → conv[1] → gn[1] → relu ──────
    {
        const size_t graph_size = 16384;
        const size_t ctx_size = ggml_tensor_overhead() * graph_size +
                                ggml_graph_overhead_custom(graph_size, false);
        std::vector<uint8_t> ctx_buf(ctx_size);
        ggml_context * ctx = ggml_init({ctx_size, ctx_buf.data(), true});
        if (!ctx) throw std::runtime_error("ggml_init failed");
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, graph_size, false, true);
        if (!sched) { ggml_free(ctx); throw std::runtime_error("sched_new failed"); }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

        ggml_tensor * bb0_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W0, H0, kChannels, 1);
        ggml_tensor * s0_t  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W1, H1, kChannels, 1);
        ggml_backend_sched_set_tensor_backend(sched, bb0_t, backend);
        ggml_backend_sched_set_tensor_backend(sched, s0_t,  backend);

        ggml_tensor * up = ggml_upscale(ctx, s0_t, 2, GGML_SCALE_MODE_NEAREST);
        ggml_tensor * added = ggml_add(ctx, bb0_t, up);
        ggml_tensor * convd = conv2d_bias(
            ctx,
            require_tensor(model_, "segmentation_head.pixel_decoder.conv_layers.1.weight"),
            require_tensor(model_, "segmentation_head.pixel_decoder.conv_layers.1.bias"),
            added, 1, 1, 1, 1);
        ggml_tensor * normed = group_norm(
            ctx, convd,
            require_tensor(model_, "segmentation_head.pixel_decoder.norms.1.weight"),
            require_tensor(model_, "segmentation_head.pixel_decoder.norms.1.bias"));
        ggml_tensor * relu = ggml_relu(ctx, normed);
        ggml_tensor * capture = ggml_cont(ctx, relu);
        ggml_build_forward_expand(gf, capture);
        ggml_backend_sched_alloc_graph(sched, gf);

        ggml_backend_tensor_set(bb0_t, bb0.data(),          0, bb0.size() * sizeof(float));
        ggml_backend_tensor_set(s0_t,  stage0_host.data(),  0, stage0_host.size() * sizeof(float));
        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(sched); ggml_free(ctx);
            throw std::runtime_error("pixel_decoder stage1 compute failed");
        }
        stage1_host.resize(static_cast<size_t>(ggml_nelements(capture)));
        ggml_backend_tensor_get(capture, stage1_host.data(), 0, stage1_host.size() * sizeof(float));
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }

    return stage1_host;
}

}  // namespace sam3
