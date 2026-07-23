#include "sam3/instructsam_decoder.h"

#include "ggml-alloc.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

// InstructSAM decoder shape constants (from Sam3DetrDecoderLayer + max_seg_nums=10)
constexpr int32_t kModelDim = 256;
constexpr int32_t kHeads = 8;
constexpr int32_t kHeadDim = kModelDim / kHeads;
constexpr int32_t kLayers = 6;
constexpr int32_t kNumQueries = 10;      // InstructSAM's max_seg_nums (per object)
constexpr int32_t kFfnDim = 2048;
constexpr float kLayerNormEps = 1e-5f;

// Helpers copied from sam3cpp's stock decoder.cpp (unnamed namespace)
// — kept local to avoid exporting them from decoder.cpp. When the C++
// codebase grows, these should move to a shared internal header.

ggml_tensor * require_tensor(const GgufModel & model, const std::string & name) {
    ggml_tensor * t = model.find_weight(name);
    if (t == nullptr) {
        throw std::runtime_error("InstructsamDecoder: missing tensor: " + name);
    }
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
    if (bias != nullptr) {
        y = ggml_add(ctx, y, ensure_f32(ctx, bias));
    }
    return y;
}

ggml_tensor * relu(ggml_context * ctx, ggml_tensor * x) {
    return ggml_relu(ctx, x);
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
        kv_len, kHeadDim, kHeads
    );
    ggml_tensor * scores = ensure_f32(ctx, ggml_mul_mat(ctx, kh, qh));
    ggml_tensor * probs = ggml_soft_max_ext(
        ctx, scores, mask,
        1.0f / std::sqrt(static_cast<float>(kHeadDim)),
        0.0f
    );
    ggml_tensor * attn = ensure_f32(ctx, ggml_mul_mat(ctx, vh, probs));
    ggml_tensor * merged = ggml_permute(ctx, attn, 0, 2, 1, 3);
    return ggml_cont_2d(ctx, merged, kModelDim, q_len);
}

std::string layer_prefix(int layer) {
    return "transformer.decoder.layers." + std::to_string(layer);
}

// Per-layer names using InstructSAM's native naming (q_proj / self_attn_layer_norm
// / mlp.fc1 / etc.). The dual-aliased GGUF also provides sam3cpp's stock names,
// so either works — using InstructSAM's since it's what appears in checkpoint
// files and matches the reference-oracle manifest fingerprints.
std::string prefix_attn_qkvo(int layer, const char * attn_name, const char * qkvo, const char * kind) {
    return layer_prefix(layer) + "." + attn_name + "." + qkvo + "." + kind;
}

std::string prefix_norm(int layer, const char * norm_name, const char * kind) {
    return layer_prefix(layer) + "." + norm_name + "." + kind;
}

// Build one decoder layer's forward graph.
// Structure (matches InstructSAM's Sam3DetrDecoderLayer.forward):
//   self_attn(Q=hidden+query_pos, K=hidden+query_pos, V=hidden)  → +residual → self_norm
//   text_cross_attn(Q=hidden+query_pos, K=V=text_mem, mask)      → +residual → text_norm
//   vision_cross_attn(Q=hidden+query_pos, K=vision_mem+vision_pos, V=vision_mem, mask=vision_mask)
//                                                                → +residual → vision_norm
//   MLP(fc1 → activation → fc2)                                  → +residual → mlp_norm
ggml_tensor * build_layer(
    ggml_context * ctx,
    const GgufModel & model,
    int layer,
    ggml_tensor * hidden,          // [kNumQueries, kModelDim]
    ggml_tensor * query_pos,       // [kNumQueries, kModelDim] — currently placeholder zeros
    ggml_tensor * text_memory,     // [text_seq, kModelDim]
    int32_t text_seq,
    ggml_tensor * text_mask,       // [text_seq, kNumQueries] f16 mask
    ggml_tensor * vision_memory,   // [hw, kModelDim]
    ggml_tensor * vision_pos,      // [hw, kModelDim]
    int32_t hw
) {
    const int32_t nq = kNumQueries;

    // ── Self-attention ─────────────────────────────────────────────────
    ggml_tensor * self_qk = ggml_add(ctx, hidden, query_pos);
    ggml_tensor * self_out = mha(ctx,
        linear(ctx, require_tensor(model, prefix_attn_qkvo(layer, "self_attn", "q_proj", "weight")),
                     require_tensor(model, prefix_attn_qkvo(layer, "self_attn", "q_proj", "bias")), self_qk),
        linear(ctx, require_tensor(model, prefix_attn_qkvo(layer, "self_attn", "k_proj", "weight")),
                     require_tensor(model, prefix_attn_qkvo(layer, "self_attn", "k_proj", "bias")), self_qk),
        linear(ctx, require_tensor(model, prefix_attn_qkvo(layer, "self_attn", "v_proj", "weight")),
                     require_tensor(model, prefix_attn_qkvo(layer, "self_attn", "v_proj", "bias")), hidden),
        nq, nq, nullptr);
    self_out = linear(ctx,
        require_tensor(model, prefix_attn_qkvo(layer, "self_attn", "o_proj", "weight")),
        require_tensor(model, prefix_attn_qkvo(layer, "self_attn", "o_proj", "bias")), self_out);
    hidden = layer_norm(ctx, ggml_add(ctx, hidden, self_out),
        require_tensor(model, prefix_norm(layer, "self_attn_layer_norm", "weight")),
        require_tensor(model, prefix_norm(layer, "self_attn_layer_norm", "bias")));

    // ── Text cross-attention ───────────────────────────────────────────
    ggml_tensor * tcqp = ggml_add(ctx, hidden, query_pos);
    ggml_tensor * text_out = mha(ctx,
        linear(ctx, require_tensor(model, prefix_attn_qkvo(layer, "text_cross_attn", "q_proj", "weight")),
                     require_tensor(model, prefix_attn_qkvo(layer, "text_cross_attn", "q_proj", "bias")), tcqp),
        linear(ctx, require_tensor(model, prefix_attn_qkvo(layer, "text_cross_attn", "k_proj", "weight")),
                     require_tensor(model, prefix_attn_qkvo(layer, "text_cross_attn", "k_proj", "bias")), text_memory),
        linear(ctx, require_tensor(model, prefix_attn_qkvo(layer, "text_cross_attn", "v_proj", "weight")),
                     require_tensor(model, prefix_attn_qkvo(layer, "text_cross_attn", "v_proj", "bias")), text_memory),
        nq, text_seq, text_mask);
    text_out = linear(ctx,
        require_tensor(model, prefix_attn_qkvo(layer, "text_cross_attn", "o_proj", "weight")),
        require_tensor(model, prefix_attn_qkvo(layer, "text_cross_attn", "o_proj", "bias")), text_out);
    hidden = layer_norm(ctx, ggml_add(ctx, hidden, text_out),
        require_tensor(model, prefix_norm(layer, "text_cross_attn_layer_norm", "weight")),
        require_tensor(model, prefix_norm(layer, "text_cross_attn_layer_norm", "bias")));

    // ── Vision cross-attention ─────────────────────────────────────────
    ggml_tensor * vcqp = ggml_add(ctx, hidden, query_pos);
    ggml_tensor * vision_k = ggml_add(ctx, vision_memory, vision_pos);
    ggml_tensor * vision_out = mha(ctx,
        linear(ctx, require_tensor(model, prefix_attn_qkvo(layer, "vision_cross_attn", "q_proj", "weight")),
                     require_tensor(model, prefix_attn_qkvo(layer, "vision_cross_attn", "q_proj", "bias")), vcqp),
        linear(ctx, require_tensor(model, prefix_attn_qkvo(layer, "vision_cross_attn", "k_proj", "weight")),
                     require_tensor(model, prefix_attn_qkvo(layer, "vision_cross_attn", "k_proj", "bias")), vision_k),
        linear(ctx, require_tensor(model, prefix_attn_qkvo(layer, "vision_cross_attn", "v_proj", "weight")),
                     require_tensor(model, prefix_attn_qkvo(layer, "vision_cross_attn", "v_proj", "bias")), vision_memory),
        nq, hw, nullptr);  // TODO: attach box_rpb_embed mask when reference-boxes wiring lands
    vision_out = linear(ctx,
        require_tensor(model, prefix_attn_qkvo(layer, "vision_cross_attn", "o_proj", "weight")),
        require_tensor(model, prefix_attn_qkvo(layer, "vision_cross_attn", "o_proj", "bias")), vision_out);
    hidden = layer_norm(ctx, ggml_add(ctx, hidden, vision_out),
        require_tensor(model, prefix_norm(layer, "vision_cross_attn_layer_norm", "weight")),
        require_tensor(model, prefix_norm(layer, "vision_cross_attn_layer_norm", "bias")));

    // ── MLP ─────────────────────────────────────────────────────────────
    ggml_tensor * ff = linear(ctx,
        require_tensor(model, layer_prefix(layer) + ".mlp.fc1.weight"),
        require_tensor(model, layer_prefix(layer) + ".mlp.fc1.bias"), hidden);
    ff = relu(ctx, ff);
    ff = linear(ctx,
        require_tensor(model, layer_prefix(layer) + ".mlp.fc2.weight"),
        require_tensor(model, layer_prefix(layer) + ".mlp.fc2.bias"), ff);
    hidden = layer_norm(ctx, ggml_add(ctx, hidden, ff),
        require_tensor(model, prefix_norm(layer, "mlp_layer_norm", "weight")),
        require_tensor(model, prefix_norm(layer, "mlp_layer_norm", "bias")));

    return hidden;
}

// Per-layer tensor names — unchanged from skeleton; used by validate.
std::vector<std::string> per_layer_tensor_names(int layer) {
    const std::string p = "transformer.decoder.layers." + std::to_string(layer);
    std::vector<std::string> names;
    for (const std::string qkvo : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        names.push_back(p + ".self_attn." + qkvo + ".weight");
        names.push_back(p + ".self_attn." + qkvo + ".bias");
    }
    names.push_back(p + ".self_attn_layer_norm.weight");
    names.push_back(p + ".self_attn_layer_norm.bias");
    for (const std::string qkvo : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        names.push_back(p + ".text_cross_attn." + qkvo + ".weight");
        names.push_back(p + ".text_cross_attn." + qkvo + ".bias");
    }
    names.push_back(p + ".text_cross_attn_layer_norm.weight");
    names.push_back(p + ".text_cross_attn_layer_norm.bias");
    for (const std::string qkvo : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        names.push_back(p + ".vision_cross_attn." + qkvo + ".weight");
        names.push_back(p + ".vision_cross_attn." + qkvo + ".bias");
    }
    names.push_back(p + ".vision_cross_attn_layer_norm.weight");
    names.push_back(p + ".vision_cross_attn_layer_norm.bias");
    for (const std::string fc : {"fc1", "fc2"}) {
        names.push_back(p + ".mlp." + fc + ".weight");
        names.push_back(p + ".mlp." + fc + ".bias");
    }
    names.push_back(p + ".mlp_layer_norm.weight");
    names.push_back(p + ".mlp_layer_norm.bias");
    return names;
}

std::vector<std::string> global_decoder_tensor_names() {
    std::vector<std::string> names;
    const std::string p = "transformer.decoder";
    for (int i = 1; i <= 3; ++i) {
        names.push_back(p + ".box_head.layer" + std::to_string(i) + ".weight");
        names.push_back(p + ".box_head.layer" + std::to_string(i) + ".bias");
    }
    for (int i = 1; i <= 2; ++i) {
        names.push_back(p + ".ref_point_head.layer" + std::to_string(i) + ".weight");
        names.push_back(p + ".ref_point_head.layer" + std::to_string(i) + ".bias");
    }
    for (const std::string axis : {"box_rpb_embed_x", "box_rpb_embed_y"}) {
        for (int i = 1; i <= 2; ++i) {
            names.push_back(p + "." + axis + ".layer" + std::to_string(i) + ".weight");
            names.push_back(p + "." + axis + ".layer" + std::to_string(i) + ".bias");
        }
    }
    names.push_back(p + ".query_embed.weight");
    names.push_back(p + ".reference_points.weight");
    names.push_back(p + ".output_layer_norm.weight");
    names.push_back(p + ".output_layer_norm.bias");
    return names;
}

std::vector<std::string> instructsam_glue_tensor_names() {
    return {
        "instructsam.mask_queries",
        "instructsam.mask_hidden_fcs.0.0.weight",
        "instructsam.mask_hidden_fcs.0.0.bias",
        "instructsam.mask_hidden_fcs.0.2.weight",
        "instructsam.mask_hidden_fcs.0.2.bias",
        "instructsam.text_hidden_fcs.0.0.weight",
        "instructsam.text_hidden_fcs.0.0.bias",
        "instructsam.text_hidden_fcs.0.2.weight",
        "instructsam.text_hidden_fcs.0.2.bias",
    };
}

}  // namespace

InstructsamDecoder::InstructsamDecoder(const GgufModel & model) : model_(model) {}

size_t InstructsamDecoder::validate_all_tensors_present() const {
    size_t probed = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        for (const auto & name : per_layer_tensor_names(layer)) {
            if (model_.find_tensor(name) == nullptr) {
                throw std::runtime_error(
                    "InstructsamDecoder: missing per-layer tensor: " + name);
            }
            ++probed;
        }
    }
    for (const auto & name : global_decoder_tensor_names()) {
        if (model_.find_tensor(name) == nullptr) {
            throw std::runtime_error(
                "InstructsamDecoder: missing global tensor: " + name);
        }
        ++probed;
    }
    for (const auto & name : instructsam_glue_tensor_names()) {
        if (model_.find_tensor(name) == nullptr) {
            throw std::runtime_error(
                "InstructsamDecoder: missing glue tensor: " + name);
        }
        ++probed;
    }
    return probed;
}

DecoderOutput InstructsamDecoder::run(
    const std::vector<float> & queries,
    const std::vector<int64_t> & queries_shape,
    const std::vector<float> & text_memory,
    const std::vector<int64_t> & text_memory_shape,
    const std::vector<float> & vision_memory,
    const std::vector<int64_t> & vision_memory_shape,
    const std::vector<float> & vision_pos,
    const std::vector<int64_t> & vision_pos_shape,
    const std::vector<float> & text_mask,
    const std::vector<int64_t> & text_mask_shape,
    const std::vector<float> & query_pos_ext
) const {
    // ── Shape guards ────────────────────────────────────────────────────
    if (queries_shape.size() != 2 || queries_shape[0] != kNumQueries || queries_shape[1] != kModelDim) {
        throw std::runtime_error("InstructsamDecoder.run: queries must be [kNumQueries=10, 256]");
    }
    if (text_memory_shape.size() != 2 || text_memory_shape[1] != kModelDim) {
        throw std::runtime_error("InstructsamDecoder.run: text_memory must be [text_seq, 256]");
    }
    if (vision_memory_shape.size() != 2 || vision_memory_shape[1] != kModelDim) {
        throw std::runtime_error("InstructsamDecoder.run: vision_memory must be [hw, 256]");
    }
    if (vision_pos_shape != vision_memory_shape) {
        throw std::runtime_error("InstructsamDecoder.run: vision_pos shape must match vision_memory");
    }
    const int32_t text_seq = static_cast<int32_t>(text_memory_shape[0]);
    const int32_t hw = static_cast<int32_t>(vision_memory_shape[0]);
    if (text_mask_shape.size() != 1 || text_mask_shape[0] != text_seq) {
        throw std::runtime_error("InstructsamDecoder.run: text_mask must be [text_seq]");
    }

    // Convert text_mask to f16 attention-mask format [text_seq, kNumQueries]
    std::vector<ggml_fp16_t> text_mask_f16(static_cast<size_t>(text_seq * kNumQueries));
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int32_t tq = 0; tq < kNumQueries; ++tq) {
        for (int32_t sk = 0; sk < text_seq; ++sk) {
            const float m = text_mask[static_cast<size_t>(sk)] > 0.5f ? neg_inf : 0.0f;
            text_mask_f16[static_cast<size_t>(tq * text_seq + sk)] = ggml_fp32_to_fp16(m);
        }
    }

    ggml_backend_t backend = model_.backend();
    ggml_backend_t cpu_backend = nullptr;
    if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        cpu_backend = model_.cpu_backend();
    }

    // Working buffer for the queries — updated layer-by-layer
    std::vector<float> current_hidden = queries;
    // Query positional embedding — if caller supplied one, use it (real layer-0
    // pos from ref_point_head(sinusoidal(reference_points))). Otherwise zeros
    // (structural smoke only, not numerically meaningful).
    std::vector<float> query_pos_buf;
    if (query_pos_ext.empty()) {
        query_pos_buf.assign(static_cast<size_t>(kNumQueries * kModelDim), 0.0f);
    } else {
        if (query_pos_ext.size() != static_cast<size_t>(kNumQueries * kModelDim)) {
            throw std::runtime_error("InstructsamDecoder.run: query_pos must be [10, 256]");
        }
        query_pos_buf = query_pos_ext;
    }

    DecoderOutput out;
    out.num_layers = kLayers;
    out.num_queries = kNumQueries;
    out.hidden_dim = kModelDim;
    out.hs.resize(static_cast<size_t>(kLayers));
    out.reference_boxes.resize(static_cast<size_t>(kLayers));
    out.presence_logits.resize(static_cast<size_t>(kLayers));

    for (int layer = 0; layer < kLayers; ++layer) {
        const size_t graph_size = 65536;
        const size_t ctx_size = ggml_tensor_overhead() * graph_size +
                                ggml_graph_overhead_custom(graph_size, false);
        std::vector<uint8_t> ctx_buf(ctx_size);
        ggml_init_params params { ctx_size, ctx_buf.data(), true };
        ggml_context * ctx = ggml_init(params);
        if (ctx == nullptr) throw std::runtime_error("ggml_init failed");

        ggml_backend_t backends[2] = { backend, cpu_backend };
        const int n_backends = cpu_backend != nullptr ? 2 : 1;
        ggml_backend_sched_t sched = ggml_backend_sched_new(
            backends, nullptr, n_backends, graph_size, false, true);
        if (sched == nullptr) {
            ggml_free(ctx);
            throw std::runtime_error("sched_new failed");
        }
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

        // Input tensors
        ggml_tensor * hidden_t     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, kNumQueries);
        ggml_tensor * query_pos_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, kNumQueries);
        ggml_tensor * text_mem_t   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, text_seq);
        ggml_tensor * text_mask_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, text_seq, kNumQueries);
        ggml_tensor * vision_mem_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, hw);
        ggml_tensor * vision_pos_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kModelDim, hw);
        for (ggml_tensor * t : {hidden_t, query_pos_t, text_mem_t, vision_mem_t, vision_pos_t}) {
            ggml_backend_sched_set_tensor_backend(sched, t, backend);
        }
        ggml_backend_sched_set_tensor_backend(sched, text_mask_t, backend);

        ggml_tensor * layer_out = build_layer(
            ctx, model_, layer,
            hidden_t, query_pos_t,
            text_mem_t, text_seq, text_mask_t,
            vision_mem_t, vision_pos_t, hw);

        ggml_tensor * capture = ggml_cont(ctx, layer_out);
        ggml_build_forward_expand(gf, capture);
        ggml_backend_sched_alloc_graph(sched, gf);

        ggml_backend_tensor_set(hidden_t, current_hidden.data(), 0,
            current_hidden.size() * sizeof(float));
        ggml_backend_tensor_set(query_pos_t, query_pos_buf.data(), 0,
            query_pos_buf.size() * sizeof(float));
        ggml_backend_tensor_set(text_mem_t, text_memory.data(), 0,
            text_memory.size() * sizeof(float));
        ggml_backend_tensor_set(text_mask_t, text_mask_f16.data(), 0,
            text_mask_f16.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(vision_mem_t, vision_memory.data(), 0,
            vision_memory.size() * sizeof(float));
        ggml_backend_tensor_set(vision_pos_t, vision_pos.data(), 0,
            vision_pos.size() * sizeof(float));

        const ggml_status status = ggml_backend_sched_graph_compute(sched, gf);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_backend_sched_free(sched);
            ggml_free(ctx);
            throw std::runtime_error("decoder layer graph compute failed");
        }

        out.hs[static_cast<size_t>(layer)].resize(
            static_cast<size_t>(kNumQueries * kModelDim));
        ggml_backend_tensor_get(capture, out.hs[static_cast<size_t>(layer)].data(), 0,
            out.hs[static_cast<size_t>(layer)].size() * sizeof(float));
        current_hidden = out.hs[static_cast<size_t>(layer)];

        // Placeholder for reference_boxes + presence_logits — not yet
        // wired up. Step 2e will implement box refinement via bbox_embed
        // + reference_points update between layers.
        out.reference_boxes[static_cast<size_t>(layer)].assign(
            static_cast<size_t>(kNumQueries * 4), 0.0f);
        out.presence_logits[static_cast<size_t>(layer)].assign(1, 0.0f);

        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }

    return out;
}

}  // namespace sam3
