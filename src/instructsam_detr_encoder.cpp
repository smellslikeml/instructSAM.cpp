#include "sam3/instructsam_detr_encoder.h"

#include "ggml-alloc.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

constexpr int32_t kModelDim = 256;
constexpr int32_t kHeads = 8;
constexpr int32_t kHeadDim = kModelDim / kHeads;
constexpr int32_t kLayers = 6;
constexpr int32_t kFfnDim = 2048;
constexpr float   kLayerNormEps = 1e-5f;

ggml_tensor * require_tensor(const GgufModel & model, const std::string & name) {
    ggml_tensor * t = model.find_weight(name);
    if (t == nullptr) throw std::runtime_error("detr_encoder: missing tensor: " + name);
    return t;
}

ggml_tensor * ensure_f32(ggml_context * ctx, ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) return t;
    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

ggml_tensor * layer_norm(
    ggml_context * ctx, ggml_tensor * x,
    ggml_tensor * weight, ggml_tensor * bias
) {
    ggml_tensor * y = ggml_norm(ctx, ensure_f32(ctx, x), kLayerNormEps);
    y = ggml_mul(ctx, y, ensure_f32(ctx, weight));
    y = ggml_add(ctx, y, ensure_f32(ctx, bias));
    return y;
}

ggml_tensor * linear(
    ggml_context * ctx,
    ggml_tensor * weight, ggml_tensor * bias, ggml_tensor * x
) {
    ggml_tensor * y = ggml_mul_mat(ctx, weight, x);
    y = ensure_f32(ctx, y);
    if (bias != nullptr) y = ggml_add(ctx, y, ensure_f32(ctx, bias));
    return y;
}

ggml_tensor * mha(
    ggml_context * ctx,
    ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
    int32_t q_len, int32_t kv_len, ggml_tensor * mask
) {
    ggml_tensor * qh = ggml_permute(
        ctx, ggml_cont_3d(ctx, q, kHeadDim, kHeads, q_len), 0, 2, 1, 3);
    ggml_tensor * kh = ggml_permute(
        ctx, ggml_cont_3d(ctx, k, kHeadDim, kHeads, kv_len), 0, 2, 1, 3);
    ggml_tensor * vh = ggml_cont_3d(
        ctx,
        ggml_permute(ctx, ggml_cont_3d(ctx, v, kHeadDim, kHeads, kv_len), 1, 2, 0, 3),
        kv_len, kHeadDim, kHeads);
    ggml_tensor * scores = ensure_f32(ctx, ggml_mul_mat(ctx, kh, qh));
    ggml_tensor * probs = ggml_soft_max_ext(
        ctx, scores, mask,
        1.0f / std::sqrt(static_cast<float>(kHeadDim)),
        0.0f);
    ggml_tensor * attn = ensure_f32(ctx, ggml_mul_mat(ctx, vh, probs));
    ggml_tensor * merged = ggml_permute(ctx, attn, 0, 2, 1, 3);
    return ggml_cont_2d(ctx, merged, kModelDim, q_len);
}

std::string layer_prefix(int layer) {
    return "transformer.encoder.layers." + std::to_string(layer);
}

// Build one encoder layer's forward graph. PRE-NORM structure.
ggml_tensor * build_layer(
    ggml_context * ctx, const GgufModel & model, int layer,
    ggml_tensor * hidden,       // [hw, kModelDim]
    ggml_tensor * vision_pos,   // [hw, kModelDim]
    ggml_tensor * text_mem,     // [text_seq, kModelDim]
    int32_t hw, int32_t text_seq,
    ggml_tensor * text_mask     // [text_seq, hw] f16 additive mask
) {
    const std::string p = layer_prefix(layer);

    // ── Self-attention with vision_pos ───────────────────────────────────
    ggml_tensor * normed1 = layer_norm(ctx, hidden,
        require_tensor(model, p + ".layer_norm1.weight"),
        require_tensor(model, p + ".layer_norm1.bias"));
    ggml_tensor * qk = ggml_add(ctx, normed1, vision_pos);
    ggml_tensor * self_out = mha(ctx,
        linear(ctx, require_tensor(model, p + ".self_attn.q_proj.weight"),
                     require_tensor(model, p + ".self_attn.q_proj.bias"), qk),
        linear(ctx, require_tensor(model, p + ".self_attn.k_proj.weight"),
                     require_tensor(model, p + ".self_attn.k_proj.bias"), qk),
        linear(ctx, require_tensor(model, p + ".self_attn.v_proj.weight"),
                     require_tensor(model, p + ".self_attn.v_proj.bias"), normed1),
        hw, hw, nullptr);
    self_out = linear(ctx,
        require_tensor(model, p + ".self_attn.o_proj.weight"),
        require_tensor(model, p + ".self_attn.o_proj.bias"), self_out);
    hidden = ggml_add(ctx, hidden, self_out);

    // ── Cross-attention: vision queries attend to text ───────────────────
    ggml_tensor * normed2 = layer_norm(ctx, hidden,
        require_tensor(model, p + ".layer_norm2.weight"),
        require_tensor(model, p + ".layer_norm2.bias"));
    ggml_tensor * cross_out = mha(ctx,
        linear(ctx, require_tensor(model, p + ".cross_attn.q_proj.weight"),
                     require_tensor(model, p + ".cross_attn.q_proj.bias"), normed2),
        linear(ctx, require_tensor(model, p + ".cross_attn.k_proj.weight"),
                     require_tensor(model, p + ".cross_attn.k_proj.bias"), text_mem),
        linear(ctx, require_tensor(model, p + ".cross_attn.v_proj.weight"),
                     require_tensor(model, p + ".cross_attn.v_proj.bias"), text_mem),
        hw, text_seq, text_mask);
    cross_out = linear(ctx,
        require_tensor(model, p + ".cross_attn.o_proj.weight"),
        require_tensor(model, p + ".cross_attn.o_proj.bias"), cross_out);
    hidden = ggml_add(ctx, hidden, cross_out);

    // ── MLP ──────────────────────────────────────────────────────────────
    ggml_tensor * normed3 = layer_norm(ctx, hidden,
        require_tensor(model, p + ".layer_norm3.weight"),
        require_tensor(model, p + ".layer_norm3.bias"));
    ggml_tensor * ff = linear(ctx,
        require_tensor(model, p + ".mlp.fc1.weight"),
        require_tensor(model, p + ".mlp.fc1.bias"), normed3);
    ff = ggml_relu(ctx, ff);
    ff = linear(ctx,
        require_tensor(model, p + ".mlp.fc2.weight"),
        require_tensor(model, p + ".mlp.fc2.bias"), ff);
    hidden = ggml_add(ctx, hidden, ff);

    return hidden;
}

}  // namespace

InstructsamDetrEncoder::InstructsamDetrEncoder(const GgufModel & model) : model_(model) {}

DetrEncoderOutput InstructsamDetrEncoder::run(
    const std::vector<float> & vision_features,
    const std::vector<int64_t> & vision_shape,
    const std::vector<float> & vision_pos,
    const std::vector<int64_t> & vision_pos_shape,
    const std::vector<float> & text_features,
    const std::vector<int64_t> & text_shape,
    const std::vector<float> & text_mask,
    const std::vector<int64_t> & text_mask_shape
) const {
    if (vision_shape.size() != 2 || vision_shape[1] != kModelDim)
        throw std::runtime_error("encoder: vision must be [hw, 256]");
    if (vision_pos_shape != vision_shape)
        throw std::runtime_error("encoder: vision_pos shape must match vision");
    if (text_shape.size() != 2 || text_shape[1] != kModelDim)
        throw std::runtime_error("encoder: text must be [seq, 256]");
    if (text_mask_shape.size() != 1 || text_mask_shape[0] != text_shape[0])
        throw std::runtime_error("encoder: text_mask must be [seq]");

    const int32_t hw       = static_cast<int32_t>(vision_shape[0]);
    const int32_t text_seq = static_cast<int32_t>(text_shape[0]);

    // Text mask → f16 additive attention mask [text_seq, hw]
    std::vector<ggml_fp16_t> text_mask_f16(static_cast<size_t>(text_seq * hw));
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int32_t q = 0; q < hw; ++q) {
        for (int32_t k = 0; k < text_seq; ++k) {
            const float valid = text_mask[static_cast<size_t>(k)];
            const float m = (valid > 0.5f) ? 0.0f : neg_inf;
            text_mask_f16[static_cast<size_t>(q * text_seq + k)] = ggml_fp32_to_fp16(m);
        }
    }

    ggml_backend_t backend = model_.backend();
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_t backends[2] = { backend, nullptr };
    int n_backends = 1;
    if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        cpu_backend = model_.cpu_backend();
        if (cpu_backend != nullptr) { backends[1] = cpu_backend; n_backends = 2; }
    }

    DetrEncoderOutput out;
    out.num_layers  = kLayers;
    out.vision_seq  = hw;
    out.hidden_dim  = kModelDim;
    out.hs.resize(static_cast<size_t>(kLayers));

    std::vector<float> current_hidden = vision_features;

    for (int layer = 0; layer < kLayers; ++layer) {
        const size_t graph_size = 32768;
        const size_t ctx_size = ggml_tensor_overhead() * graph_size +
                                ggml_graph_overhead_custom(graph_size, false);
        std::vector<uint8_t> ctx_buf(ctx_size);
        ggml_context * ctx = ggml_init({ctx_size, ctx_buf.data(), true});
        if (!ctx) throw std::runtime_error("ggml_init failed");
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, graph_size, false, true);
        if (!sched) { ggml_free(ctx); throw std::runtime_error("sched_new failed"); }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

        ggml_tensor * hidden_t   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, hw);
        ggml_tensor * vpos_t     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, hw);
        ggml_tensor * text_t     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, text_seq);
        ggml_tensor * tmask_t    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, text_seq, hw);
        for (ggml_tensor * t : {hidden_t, vpos_t, text_t, tmask_t}) {
            ggml_backend_sched_set_tensor_backend(sched, t, backend);
        }

        ggml_tensor * layer_out = build_layer(
            ctx, model_, layer,
            hidden_t, vpos_t, text_t,
            hw, text_seq, tmask_t);

        ggml_tensor * capture = ggml_cont(ctx, layer_out);
        ggml_build_forward_expand(gf, capture);
        ggml_backend_sched_alloc_graph(sched, gf);

        ggml_backend_tensor_set(hidden_t, current_hidden.data(), 0, current_hidden.size() * sizeof(float));
        ggml_backend_tensor_set(vpos_t,   vision_pos.data(),     0, vision_pos.size() * sizeof(float));
        ggml_backend_tensor_set(text_t,   text_features.data(),  0, text_features.size() * sizeof(float));
        ggml_backend_tensor_set(tmask_t,  text_mask_f16.data(),  0, text_mask_f16.size() * sizeof(ggml_fp16_t));

        if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(sched); ggml_free(ctx);
            throw std::runtime_error("encoder layer compute failed");
        }

        out.hs[static_cast<size_t>(layer)].resize(static_cast<size_t>(hw * kModelDim));
        ggml_backend_tensor_get(capture, out.hs[static_cast<size_t>(layer)].data(), 0,
            out.hs[static_cast<size_t>(layer)].size() * sizeof(float));
        current_hidden = out.hs[static_cast<size_t>(layer)];

        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }

    out.last_hidden_state = out.hs.back();
    return out;
}

}  // namespace sam3
