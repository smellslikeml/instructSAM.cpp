#include "sam3/instructsam_prompt_cross_attn.h"

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
constexpr float   kLayerNormEps = 1e-5f;

ggml_tensor * require_tensor(const GgufModel & model, const std::string & name) {
    ggml_tensor * t = model.find_weight(name);
    if (t == nullptr) throw std::runtime_error("prompt_cross_attn: missing tensor: " + name);
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
    int32_t q_len, int32_t kv_len,
    ggml_tensor * mask
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

}  // namespace

InstructsamPromptCrossAttn::InstructsamPromptCrossAttn(const GgufModel & model) : model_(model) {}

std::vector<float> InstructsamPromptCrossAttn::run(
    const std::vector<float> & encoder_hidden_states,
    const std::vector<int64_t> & encoder_shape,
    const std::vector<float> & prompt_features,
    const std::vector<int64_t> & prompt_shape,
    const std::vector<float> & prompt_mask,
    const std::vector<int64_t> & prompt_mask_shape
) const {
    if (encoder_shape.size() != 2 || encoder_shape[1] != kModelDim) {
        throw std::runtime_error("PCA: encoder must be [hw, 256]");
    }
    if (prompt_shape.size() != 2 || prompt_shape[1] != kModelDim) {
        throw std::runtime_error("PCA: prompt_features must be [prompt_seq, 256]");
    }
    if (prompt_mask_shape.size() != 1 || prompt_mask_shape[0] != prompt_shape[0]) {
        throw std::runtime_error("PCA: prompt_mask must be [prompt_seq]");
    }

    const int32_t hw         = static_cast<int32_t>(encoder_shape[0]);
    const int32_t prompt_seq = static_cast<int32_t>(prompt_shape[0]);

    // Convert prompt_mask (1.0=valid, 0.0=padding) to f16 additive mask
    // [prompt_seq, hw]. Padded positions get -inf, valid get 0.
    std::vector<ggml_fp16_t> mask_f16(static_cast<size_t>(prompt_seq * hw));
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int32_t q = 0; q < hw; ++q) {
        for (int32_t k = 0; k < prompt_seq; ++k) {
            const float valid = prompt_mask[static_cast<size_t>(k)];
            const float m = (valid > 0.5f) ? 0.0f : neg_inf;
            mask_f16[static_cast<size_t>(q * prompt_seq + k)] = ggml_fp32_to_fp16(m);
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

    const size_t graph_size = 16384;
    const size_t ctx_size = ggml_tensor_overhead() * graph_size +
                            ggml_graph_overhead_custom(graph_size, false);
    std::vector<uint8_t> ctx_buf(ctx_size);
    ggml_context * ctx = ggml_init({ctx_size, ctx_buf.data(), true});
    if (!ctx) throw std::runtime_error("ggml_init failed");
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, graph_size, false, true);
    if (!sched) { ggml_free(ctx); throw std::runtime_error("sched_new failed"); }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

    ggml_tensor * encoder_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, hw);
    ggml_tensor * prompt_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, prompt_seq);
    ggml_tensor * mask_t    = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, prompt_seq, hw);
    for (ggml_tensor * t : {encoder_t, prompt_t, mask_t}) {
        ggml_backend_sched_set_tensor_backend(sched, t, backend);
    }

    const std::string p = "segmentation_head";
    ggml_tensor * normed = layer_norm(
        ctx, encoder_t,
        require_tensor(model_, p + ".prompt_cross_attn_norm.weight"),
        require_tensor(model_, p + ".prompt_cross_attn_norm.bias"));

    ggml_tensor * q_out = linear(ctx,
        require_tensor(model_, p + ".prompt_cross_attn.q_proj.weight"),
        require_tensor(model_, p + ".prompt_cross_attn.q_proj.bias"), normed);
    ggml_tensor * k_out = linear(ctx,
        require_tensor(model_, p + ".prompt_cross_attn.k_proj.weight"),
        require_tensor(model_, p + ".prompt_cross_attn.k_proj.bias"), prompt_t);
    ggml_tensor * v_out = linear(ctx,
        require_tensor(model_, p + ".prompt_cross_attn.v_proj.weight"),
        require_tensor(model_, p + ".prompt_cross_attn.v_proj.bias"), prompt_t);

    ggml_tensor * attn = mha(ctx, q_out, k_out, v_out, hw, prompt_seq, mask_t);
    ggml_tensor * attn_out = linear(ctx,
        require_tensor(model_, p + ".prompt_cross_attn.o_proj.weight"),
        require_tensor(model_, p + ".prompt_cross_attn.o_proj.bias"), attn);

    // residual add
    ggml_tensor * fused = ggml_add(ctx, encoder_t, attn_out);
    ggml_tensor * capture = ggml_cont(ctx, fused);
    ggml_build_forward_expand(gf, capture);
    ggml_backend_sched_alloc_graph(sched, gf);

    ggml_backend_tensor_set(encoder_t, encoder_hidden_states.data(), 0,
        encoder_hidden_states.size() * sizeof(float));
    ggml_backend_tensor_set(prompt_t, prompt_features.data(), 0,
        prompt_features.size() * sizeof(float));
    ggml_backend_tensor_set(mask_t, mask_f16.data(), 0,
        mask_f16.size() * sizeof(ggml_fp16_t));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_free(sched); ggml_free(ctx);
        throw std::runtime_error("prompt_cross_attn compute failed");
    }

    std::vector<float> out(static_cast<size_t>(ggml_nelements(capture)));
    ggml_backend_tensor_get(capture, out.data(), 0, out.size() * sizeof(float));

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return out;
}

}  // namespace sam3
