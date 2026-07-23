#include "sam3/instructsam_mask_tail.h"

#include "ggml-alloc.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

constexpr int32_t kChannels = 256;
constexpr int32_t kNumQueries = 10;
constexpr int32_t kFinalH = 288;
constexpr int32_t kFinalW = 288;

ggml_tensor * require_tensor(const GgufModel & model, const std::string & name) {
    ggml_tensor * t = model.find_weight(name);
    if (t == nullptr) throw std::runtime_error("mask_tail: missing tensor: " + name);
    return t;
}

// Read a weight tensor as F32 regardless of on-disk dtype (F16 → convert).
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
        throw std::runtime_error("mask_tail: unsupported dtype for " + name);
    }
    return v;
}

// Fused linear + optional ReLU on [batch, in_dim] × [out_dim, in_dim]^T + bias
std::vector<float> cpu_linear(
    const std::vector<float> & x,
    int64_t in_dim, int64_t out_dim, int64_t batch,
    const std::vector<float> & w, const std::vector<float> & b,
    bool with_relu
) {
    std::vector<float> out(static_cast<size_t>(batch * out_dim));
    for (int64_t i = 0; i < batch; ++i) {
        for (int64_t o = 0; o < out_dim; ++o) {
            float s = b[static_cast<size_t>(o)];
            for (int64_t k = 0; k < in_dim; ++k) {
                s += w[static_cast<size_t>(o * in_dim + k)] *
                     x[static_cast<size_t>(i * in_dim + k)];
            }
            out[static_cast<size_t>(i * out_dim + o)] = with_relu ? std::max(0.0f, s) : s;
        }
    }
    return out;
}

// Conv2d with 1x1 kernel is just per-pixel linear over channels:
//   out[o, h, w] = sum_c w[o, c] * in[c, h, w] + b[o]
// Weight tensor layout for Conv2d(in=C, out=O, k=1x1) is [O, C, 1, 1].
std::vector<float> cpu_conv1x1(
    const std::vector<float> & in,      // [in_c, H, W]  row-major
    int64_t in_c, int64_t H, int64_t W,
    int64_t out_c,
    const std::vector<float> & w,       // [out_c, in_c, 1, 1] → [out_c, in_c]
    const std::vector<float> & b        // [out_c]
) {
    const int64_t HW = H * W;
    std::vector<float> out(static_cast<size_t>(out_c * HW));
    for (int64_t o = 0; o < out_c; ++o) {
        const float bo = b[static_cast<size_t>(o)];
        for (int64_t p = 0; p < HW; ++p) {
            float s = bo;
            for (int64_t c = 0; c < in_c; ++c) {
                s += w[static_cast<size_t>(o * in_c + c)] *
                     in[static_cast<size_t>(c * HW + p)];
            }
            out[static_cast<size_t>(o * HW + p)] = s;
        }
    }
    return out;
}

// pred_masks[q, h, w] = sum_c mask_emb[q, c] * inst_emb[c, h, w]
std::vector<float> cpu_einsum_qc_chw_qhw(
    const std::vector<float> & mask_emb,   // [Q, C]
    const std::vector<float> & inst_emb,   // [C, H, W]
    int64_t Q, int64_t C, int64_t H, int64_t W
) {
    const int64_t HW = H * W;
    std::vector<float> out(static_cast<size_t>(Q * HW));
    for (int64_t q = 0; q < Q; ++q) {
        for (int64_t p = 0; p < HW; ++p) {
            float s = 0.0f;
            for (int64_t c = 0; c < C; ++c) {
                s += mask_emb[static_cast<size_t>(q * C + c)] *
                     inst_emb[static_cast<size_t>(c * HW + p)];
            }
            out[static_cast<size_t>(q * HW + p)] = s;
        }
    }
    return out;
}

}  // namespace

InstructsamMaskTail::InstructsamMaskTail(const GgufModel & model) : model_(model) {}

MaskTailOutput InstructsamMaskTail::run(
    const std::vector<float> & pixel_embed,
    const std::vector<int64_t> & pixel_embed_shape,
    const std::vector<float> & decoder_queries,
    const std::vector<int64_t> & decoder_queries_shape
) const {
    if (pixel_embed_shape.size() != 3 ||
        pixel_embed_shape[0] != kChannels ||
        pixel_embed_shape[1] != kFinalH ||
        pixel_embed_shape[2] != kFinalW) {
        throw std::runtime_error("mask_tail: pixel_embed must be [256, 288, 288]");
    }
    if (decoder_queries_shape.size() != 2 ||
        decoder_queries_shape[0] != kNumQueries ||
        decoder_queries_shape[1] != kChannels) {
        throw std::runtime_error("mask_tail: decoder_queries must be [10, 256]");
    }

    // Load weights (one-shot).
    // Converter maps model.grounding_model.model.mask_decoder → segmentation_head.
    const std::string mp = "segmentation_head";
    const auto me0_w = get_f32(model_, mp + ".mask_embedder.layers.0.weight", kChannels * kChannels);
    const auto me0_b = get_f32(model_, mp + ".mask_embedder.layers.0.bias",   kChannels);
    const auto me1_w = get_f32(model_, mp + ".mask_embedder.layers.1.weight", kChannels * kChannels);
    const auto me1_b = get_f32(model_, mp + ".mask_embedder.layers.1.bias",   kChannels);
    const auto me2_w = get_f32(model_, mp + ".mask_embedder.layers.2.weight", kChannels * kChannels);
    const auto me2_b = get_f32(model_, mp + ".mask_embedder.layers.2.bias",   kChannels);

    const auto ip_w = get_f32(model_, mp + ".instance_projection.weight", kChannels * kChannels);
    const auto ip_b = get_f32(model_, mp + ".instance_projection.bias",   kChannels);
    const auto sp_w = get_f32(model_, mp + ".semantic_projection.weight", 1 * kChannels);
    const auto sp_b = get_f32(model_, mp + ".semantic_projection.bias",   1);

    // mask_embedder: 3-layer MLP with ReLU between (not after final)
    auto me = cpu_linear(decoder_queries, kChannels, kChannels, kNumQueries, me0_w, me0_b, /*relu=*/true);
    me      = cpu_linear(me,              kChannels, kChannels, kNumQueries, me1_w, me1_b, /*relu=*/true);
    me      = cpu_linear(me,              kChannels, kChannels, kNumQueries, me2_w, me2_b, /*relu=*/false);

    // instance_projection: Conv1x1 256 → 256 on pixel_embed
    const auto instance_embed = cpu_conv1x1(pixel_embed, kChannels, kFinalH, kFinalW, kChannels, ip_w, ip_b);
    // semantic_projection: Conv1x1 256 → 1 on pixel_embed
    const auto semantic_embed = cpu_conv1x1(pixel_embed, kChannels, kFinalH, kFinalW, 1, sp_w, sp_b);

    // pred_masks = einsum("qc,chw->qhw", me, instance_embed)
    auto pred_masks = cpu_einsum_qc_chw_qhw(me, instance_embed, kNumQueries, kChannels, kFinalH, kFinalW);

    MaskTailOutput out;
    out.num_queries = kNumQueries;
    out.height = kFinalH;
    out.width  = kFinalW;
    out.pred_masks   = std::move(pred_masks);
    out.semantic_seg = std::move(semantic_embed);
    return out;
}

}  // namespace sam3
