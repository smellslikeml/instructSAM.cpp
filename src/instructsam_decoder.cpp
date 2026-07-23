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

// CPU-side helpers for per-layer refinement. Box_head + ref_point_head
// evaluations run on 10 queries × 256 dim, so the CPU cost is tiny (fits
// well inside the layer-graph overhead). Keeping them CPU-side avoids
// building a second ggml graph per layer boundary.

std::vector<float> cpu_linear(
    const std::vector<float> & x,
    int64_t in_dim,
    int64_t out_dim,
    int64_t batch,
    const std::vector<float> & w,   // [out_dim, in_dim] row-major
    const std::vector<float> & b    // [out_dim]
) {
    std::vector<float> out(static_cast<size_t>(batch * out_dim));
    for (int64_t i = 0; i < batch; ++i) {
        for (int64_t o = 0; o < out_dim; ++o) {
            float s = b[static_cast<size_t>(o)];
            for (int64_t k = 0; k < in_dim; ++k) {
                s += w[static_cast<size_t>(o * in_dim + k)] *
                     x[static_cast<size_t>(i * in_dim + k)];
            }
            out[static_cast<size_t>(i * out_dim + o)] = s;
        }
    }
    return out;
}

void cpu_relu(std::vector<float> & v) {
    for (float & x : v) if (x < 0.0f) x = 0.0f;
}

// Layer norm over the trailing dim (LayerNorm on [batch, dim]).
std::vector<float> cpu_layer_norm(
    const std::vector<float> & x,
    int64_t batch,
    int64_t dim,
    const std::vector<float> & w,
    const std::vector<float> & b
) {
    std::vector<float> out(x.size());
    for (int64_t i = 0; i < batch; ++i) {
        double mean = 0.0, var = 0.0;
        for (int64_t d = 0; d < dim; ++d) mean += x[static_cast<size_t>(i * dim + d)];
        mean /= dim;
        for (int64_t d = 0; d < dim; ++d) {
            const double diff = x[static_cast<size_t>(i * dim + d)] - mean;
            var += diff * diff;
        }
        var /= dim;
        const double inv_std = 1.0 / std::sqrt(var + kLayerNormEps);
        for (int64_t d = 0; d < dim; ++d) {
            const double normed = (x[static_cast<size_t>(i * dim + d)] - mean) * inv_std;
            out[static_cast<size_t>(i * dim + d)] = static_cast<float>(
                normed * w[static_cast<size_t>(d)] + b[static_cast<size_t>(d)]);
        }
    }
    return out;
}

float cpu_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Matches sam3cpp's inverse_sigmoid_vec logic (clip then log(x/(1-x))).
constexpr float kInvSigmoidEps = 1e-3f;
float cpu_inverse_sigmoid(float x) {
    const float c = std::min(1.0f - kInvSigmoidEps, std::max(kInvSigmoidEps, x));
    return std::log(c / (1.0f - c));
}

// Compute per-layer query_pos from reference_boxes [num_queries, 4].
// Mirrors dump_reference_binaries.py's implementation: sinusoidal
// encoding (128 features per coord, order y,x,w,h concat) then
// ref_point_head 2-layer MLP (512 → 256 → 256, ReLU between).
std::vector<float> cpu_compute_query_pos(
    const std::vector<float> & ref_boxes,   // [num_queries, 4] sigmoid space
    int64_t num_queries,
    const std::vector<float> & rph1_w,      // [256, 512]
    const std::vector<float> & rph1_b,      // [256]
    const std::vector<float> & rph2_w,      // [256, 256]
    const std::vector<float> & rph2_b       // [256]
) {
    const int64_t half = kModelDim / 2;   // 128
    const float scale = 2.0f * static_cast<float>(M_PI);

    std::vector<float> dim_t(static_cast<size_t>(half));
    for (int64_t i = 0; i < half; ++i) {
        const float exponent = 2.0f * std::floor(static_cast<float>(i) / 2.0f) /
                               static_cast<float>(half);
        dim_t[static_cast<size_t>(i)] = std::pow(10000.0f, exponent);
    }

    // Compute sinusoidal features per coord: [num_queries, 4, 128]
    // Concat order y, x, w, h (per InstructSAM's encode_boxes).
    std::vector<float> query_sine(static_cast<size_t>(num_queries * 4 * half));
    const int coord_order[4] = {1, 0, 2, 3};  // y first, then x, w, h
    for (int64_t q = 0; q < num_queries; ++q) {
        for (int slot = 0; slot < 4; ++slot) {
            const int coord = coord_order[slot];
            const float p = ref_boxes[static_cast<size_t>(q * 4 + coord)] * scale;
            for (int64_t i = 0; i < half; ++i) {
                const float v = p / dim_t[static_cast<size_t>(i)];
                const float trig = (i % 2 == 0) ? std::sin(v) : std::cos(v);
                query_sine[static_cast<size_t>(q * 4 * half + slot * half + i)] = trig;
            }
        }
    }

    // ref_point_head: linear1 → ReLU → linear2
    auto h1 = cpu_linear(query_sine, 4 * half, kModelDim, num_queries, rph1_w, rph1_b);
    cpu_relu(h1);
    return cpu_linear(h1, kModelDim, kModelDim, num_queries, rph2_w, rph2_b);
}

// Update reference_boxes: sigmoid(inverse_sigmoid(prev) + delta)
std::vector<float> cpu_update_reference_boxes(
    const std::vector<float> & prev_boxes,   // [num_queries, 4]
    const std::vector<float> & delta_boxes,  // [num_queries, 4]
    int64_t num_queries
) {
    std::vector<float> out(static_cast<size_t>(num_queries * 4));
    for (size_t i = 0; i < prev_boxes.size(); ++i) {
        out[i] = cpu_sigmoid(cpu_inverse_sigmoid(prev_boxes[i]) + delta_boxes[i]);
    }
    return out;
}

// InstructSAM box relative position bias — mathematically identical to
// sam3cpp's build_box_rpb_bias, but with the presence-token slot removed
// (output shape is [heads, num_queries, hw] instead of [heads, num_queries+1, hw]).
//
// The RPB matrix is an additive bias on attention scores in the vision
// cross-attention block. For each (query, head, spatial-position h*w) it
// encodes how far that spatial position is from the query's predicted
// box edges, log-scaled and passed through a small 2-layer MLP. Verified
// against InstructSAM's _get_rpb_matrix (modeling_sam3.py:1644-1689) —
// identical arithmetic modulo the presence-token offset.
//
// Reference boxes are in (cx, cy, w, h) format, sigmoid-space (0..1).
std::vector<float> build_instructsam_box_rpb_bias(
    const std::vector<float> & reference_boxes,
    int64_t num_queries,
    int64_t feat_h,
    int64_t feat_w,
    const std::vector<float> & mlp_x_0_w,   // box_rpb_embed_x.layer1.weight [256, 2]
    const std::vector<float> & mlp_x_0_b,   // .layer1.bias   [256]
    const std::vector<float> & mlp_x_1_w,   // .layer2.weight [heads, 256]
    const std::vector<float> & mlp_x_1_b,   // .layer2.bias   [heads]
    const std::vector<float> & mlp_y_0_w,
    const std::vector<float> & mlp_y_0_b,
    const std::vector<float> & mlp_y_1_w,
    const std::vector<float> & mlp_y_1_b
) {
    std::vector<float> coords_h(static_cast<size_t>(feat_h));
    std::vector<float> coords_w(static_cast<size_t>(feat_w));
    for (int64_t i = 0; i < feat_h; ++i) {
        coords_h[static_cast<size_t>(i)] = static_cast<float>(i) / static_cast<float>(feat_h);
    }
    for (int64_t i = 0; i < feat_w; ++i) {
        coords_w[static_cast<size_t>(i)] = static_cast<float>(i) / static_cast<float>(feat_w);
    }

    auto mlp2 = [](const std::vector<float> & in,
                   const std::vector<float> & w0, const std::vector<float> & b0,
                   const std::vector<float> & w1, const std::vector<float> & b1,
                   int64_t out0, int64_t out1) {
        std::vector<float> h(static_cast<size_t>(out0));
        for (int64_t o = 0; o < out0; ++o) {
            float sum = b0[static_cast<size_t>(o)];
            for (size_t i = 0; i < in.size(); ++i) {
                sum += w0[static_cast<size_t>(o * static_cast<int64_t>(in.size()) +
                                             static_cast<int64_t>(i))] * in[i];
            }
            h[static_cast<size_t>(o)] = std::max(0.0f, sum);   // ReLU
        }
        std::vector<float> out(static_cast<size_t>(out1));
        for (int64_t o = 0; o < out1; ++o) {
            float sum = b1[static_cast<size_t>(o)];
            for (int64_t i = 0; i < out0; ++i) {
                sum += w1[static_cast<size_t>(o * out0 + i)] * h[static_cast<size_t>(i)];
            }
            out[static_cast<size_t>(o)] = sum;
        }
        return out;
    };

    std::vector<float> out(static_cast<size_t>(kHeads * num_queries * feat_h * feat_w), 0.0f);
    for (int64_t q = 0; q < num_queries; ++q) {
        const float cx = reference_boxes[static_cast<size_t>(q * 4 + 0)];
        const float cy = reference_boxes[static_cast<size_t>(q * 4 + 1)];
        const float bw = reference_boxes[static_cast<size_t>(q * 4 + 2)];
        const float bh = reference_boxes[static_cast<size_t>(q * 4 + 3)];
        const float x0 = cx - 0.5f * bw;
        const float x1 = cx + 0.5f * bw;
        const float y0 = cy - 0.5f * bh;
        const float y1 = cy + 0.5f * bh;

        std::vector<std::vector<float>> dx(static_cast<size_t>(feat_w));
        std::vector<std::vector<float>> dy(static_cast<size_t>(feat_h));
        for (int64_t x = 0; x < feat_w; ++x) {
            std::vector<float> in = {
                coords_w[static_cast<size_t>(x)] - x0,
                coords_w[static_cast<size_t>(x)] - x1,
            };
            for (float & v : in) {
                v *= 8.0f;
                v = std::copysign(std::log2(std::fabs(v) + 1.0f) / std::log2(8.0f), v);
            }
            dx[static_cast<size_t>(x)] = mlp2(in, mlp_x_0_w, mlp_x_0_b, mlp_x_1_w, mlp_x_1_b, kModelDim, kHeads);
        }
        for (int64_t y = 0; y < feat_h; ++y) {
            std::vector<float> in = {
                coords_h[static_cast<size_t>(y)] - y0,
                coords_h[static_cast<size_t>(y)] - y1,
            };
            for (float & v : in) {
                v *= 8.0f;
                v = std::copysign(std::log2(std::fabs(v) + 1.0f) / std::log2(8.0f), v);
            }
            dy[static_cast<size_t>(y)] = mlp2(in, mlp_y_0_w, mlp_y_0_b, mlp_y_1_w, mlp_y_1_b, kModelDim, kHeads);
        }

        for (int64_t h = 0; h < kHeads; ++h) {
            for (int64_t y = 0; y < feat_h; ++y) {
                for (int64_t x = 0; x < feat_w; ++x) {
                    const int64_t src = y * feat_w + x;
                    const size_t idx = static_cast<size_t>(
                        ((h * num_queries + q) * (feat_h * feat_w)) + src);
                    out[idx] = dy[static_cast<size_t>(y)][static_cast<size_t>(h)] +
                               dx[static_cast<size_t>(x)][static_cast<size_t>(h)];
                }
            }
        }
    }
    return out;
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
    int32_t hw,
    ggml_tensor * vision_mask      // [hw, kNumQueries, kHeads, 1] f32 RPB bias (may be nullptr)
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
        nq, hw, vision_mask);
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
    const std::vector<float> & query_pos_ext,
    const std::vector<float> & initial_reference_boxes
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

    // Load per-layer refinement weights ONCE before the layer loop. These
    // drive the CPU-side computation between layers that produces the
    // next layer's query_pos and vision box_rpb bias.
    //
    // The GGUF stores large weight matrices as F16 (to save space) and
    // small tensors (biases, norms) as F32. This accessor handles both
    // by reading the raw bytes and converting to F32 in-CPU.
    const std::string dp = "transformer.decoder";
    auto get_f32 = [&](const std::string & name, size_t n) {
        ggml_tensor * t = require_tensor(model_, name);
        std::vector<float> v(n);
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> buf(n);
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < n; ++i) v[i] = ggml_fp16_to_fp32(buf[i]);
        } else {
            throw std::runtime_error("get_f32: unsupported dtype for " + name);
        }
        return v;
    };
    // Refinement enabled only when caller supplied initial_reference_boxes.
    // Without it, query_pos falls back to caller's ext (or zeros) and
    // vision_bias stays empty — structural smoke only.
    const bool refine = !initial_reference_boxes.empty();
    if (refine && initial_reference_boxes.size() != static_cast<size_t>(kNumQueries * 4)) {
        throw std::runtime_error("InstructsamDecoder.run: initial_reference_boxes must be [10, 4]");
    }

    std::vector<float> bh1w, bh1b, bh2w, bh2b, bh3w, bh3b;
    std::vector<float> rph1w, rph1b, rph2w, rph2b;
    std::vector<float> oln_w, oln_b;
    std::vector<float> rpx0w, rpx0b, rpx1w, rpx1b, rpy0w, rpy0b, rpy1w, rpy1b;
    if (refine) {
        bh1w = get_f32(dp + ".box_head.layer1.weight", kModelDim * kModelDim);
        bh1b = get_f32(dp + ".box_head.layer1.bias",   kModelDim);
        bh2w = get_f32(dp + ".box_head.layer2.weight", kModelDim * kModelDim);
        bh2b = get_f32(dp + ".box_head.layer2.bias",   kModelDim);
        bh3w = get_f32(dp + ".box_head.layer3.weight", 4 * kModelDim);
        bh3b = get_f32(dp + ".box_head.layer3.bias",   4);

        rph1w = get_f32(dp + ".ref_point_head.layer1.weight", kModelDim * (2 * kModelDim));
        rph1b = get_f32(dp + ".ref_point_head.layer1.bias",   kModelDim);
        rph2w = get_f32(dp + ".ref_point_head.layer2.weight", kModelDim * kModelDim);
        rph2b = get_f32(dp + ".ref_point_head.layer2.bias",   kModelDim);

        oln_w = get_f32(dp + ".output_layer_norm.weight", kModelDim);
        oln_b = get_f32(dp + ".output_layer_norm.bias",   kModelDim);

        rpx0w = get_f32(dp + ".box_rpb_embed_x.layer1.weight", kModelDim * 2);
        rpx0b = get_f32(dp + ".box_rpb_embed_x.layer1.bias",   kModelDim);
        rpx1w = get_f32(dp + ".box_rpb_embed_x.layer2.weight", kHeads * kModelDim);
        rpx1b = get_f32(dp + ".box_rpb_embed_x.layer2.bias",   kHeads);
        rpy0w = get_f32(dp + ".box_rpb_embed_y.layer1.weight", kModelDim * 2);
        rpy0b = get_f32(dp + ".box_rpb_embed_y.layer1.bias",   kModelDim);
        rpy1w = get_f32(dp + ".box_rpb_embed_y.layer2.weight", kHeads * kModelDim);
        rpy1b = get_f32(dp + ".box_rpb_embed_y.layer2.bias",   kHeads);
    }

    // Query positional embedding buffer, updated each layer from
    // current_ref_boxes when refinement is enabled. If caller passed
    // query_pos_ext, use it for layer 0 (must match what ref_point_head
    // would produce from initial_reference_points); refinement then
    // overwrites for layers 1-5. If refinement is off and no ext, zeros.
    std::vector<float> query_pos_buf(static_cast<size_t>(kNumQueries * kModelDim), 0.0f);
    if (!query_pos_ext.empty()) {
        if (query_pos_ext.size() != static_cast<size_t>(kNumQueries * kModelDim)) {
            throw std::runtime_error("InstructsamDecoder.run: query_pos must be [10, 256]");
        }
        query_pos_buf = query_pos_ext;
    }

    // Feature grid is 72×72 for InstructSAM (matches 5184 vision tokens).
    constexpr int32_t kFeatH = 72;
    constexpr int32_t kFeatW = 72;

    // Current reference boxes — updated per layer via box_head + inverse_sigmoid.
    std::vector<float> current_ref_boxes = initial_reference_boxes;

    // Compute layer-0 box_rpb bias if refinement enabled. When disabled,
    // vision_bias stays empty and the vision cross-attn runs without RPB.
    std::vector<float> vision_bias;
    if (refine) {
        vision_bias = build_instructsam_box_rpb_bias(
            current_ref_boxes, kNumQueries, kFeatH, kFeatW,
            rpx0w, rpx0b, rpx1w, rpx1b,
            rpy0w, rpy0b, rpy1w, rpy1b);
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
        ggml_tensor * vision_mask_t = nullptr;
        if (!vision_bias.empty()) {
            // ggml layout is column-major; sam3cpp stores mask as
            // [hw, num_queries, heads, 1] with row-major linear buffer
            // matching build_instructsam_box_rpb_bias's [heads][queries][hw] order.
            vision_mask_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hw, kNumQueries, kHeads, 1);
            ggml_backend_sched_set_tensor_backend(sched, vision_mask_t, backend);
        }
        for (ggml_tensor * t : {hidden_t, query_pos_t, text_mem_t, vision_mem_t, vision_pos_t}) {
            ggml_backend_sched_set_tensor_backend(sched, t, backend);
        }
        ggml_backend_sched_set_tensor_backend(sched, text_mask_t, backend);

        ggml_tensor * layer_out = build_layer(
            ctx, model_, layer,
            hidden_t, query_pos_t,
            text_mem_t, text_seq, text_mask_t,
            vision_mem_t, vision_pos_t, hw,
            vision_mask_t);

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
        if (vision_mask_t != nullptr) {
            ggml_backend_tensor_set(vision_mask_t, vision_bias.data(), 0,
                vision_bias.size() * sizeof(float));
        }

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

        // Per-layer box refinement (CPU): output_layer_norm → box_head →
        // sigmoid(inverse_sigmoid(prev) + delta). Then rebuild query_pos
        // and box_rpb bias from the updated reference_boxes for the next
        // layer's forward pass.
        if (refine) {
            const auto normed = cpu_layer_norm(current_hidden, kNumQueries, kModelDim, oln_w, oln_b);
            auto h = cpu_linear(normed, kModelDim, kModelDim, kNumQueries, bh1w, bh1b);
            cpu_relu(h);
            auto h2 = cpu_linear(h, kModelDim, kModelDim, kNumQueries, bh2w, bh2b);
            cpu_relu(h2);
            const auto delta = cpu_linear(h2, kModelDim, 4, kNumQueries, bh3w, bh3b);
            current_ref_boxes = cpu_update_reference_boxes(current_ref_boxes, delta, kNumQueries);
            out.reference_boxes[static_cast<size_t>(layer)] = current_ref_boxes;

            if (layer + 1 < kLayers) {
                query_pos_buf = cpu_compute_query_pos(
                    current_ref_boxes, kNumQueries, rph1w, rph1b, rph2w, rph2b);
                vision_bias = build_instructsam_box_rpb_bias(
                    current_ref_boxes, kNumQueries, kFeatH, kFeatW,
                    rpx0w, rpx0b, rpx1w, rpx1b,
                    rpy0w, rpy0b, rpy1w, rpy1b);
            }
        } else {
            out.reference_boxes[static_cast<size_t>(layer)].assign(
                static_cast<size_t>(kNumQueries * 4), 0.0f);
        }
        out.presence_logits[static_cast<size_t>(layer)].assign(1, 0.0f);

        ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }

    return out;
}

}  // namespace sam3
