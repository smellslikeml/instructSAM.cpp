#include "sam3/instructsam_vision_encoder.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

constexpr int32_t kHiddenSize   = 1024;
constexpr int32_t kNumHeads     = 16;
constexpr int32_t kHeadDim      = kHiddenSize / kNumHeads;  // 64
constexpr int32_t kNumLayers    = 32;
constexpr int32_t kIntermediate = 4736;
constexpr int32_t kPatchSize    = 14;
constexpr int32_t kImageSize    = 1008;
constexpr int32_t kGridSize     = kImageSize / kPatchSize;  // 72
constexpr int32_t kNumPatches   = kGridSize * kGridSize;    // 5184
constexpr int32_t kWindowSize   = 24;
constexpr float   kLayerNormEps = 1e-6f;
constexpr int32_t kPretrainGrid = 24;   // 336px / 14 patch = 24; stored pos_embed is [1, 576, 1024]
constexpr float   kRopeTheta    = 10000.0f;
constexpr int32_t kMlpDim       = kIntermediate;  // 4736

const std::string kBase = "backbone.vision_backbone.trunk";

// Layers using global attention (window_size=0). All others are windowed 24×24.
bool is_global_layer(int layer) {
    return layer == 7 || layer == 15 || layer == 23 || layer == 31;
}

ggml_tensor * require_tensor(const GgufModel & model, const std::string & name) {
    ggml_tensor * t = model.find_weight(name);
    if (t == nullptr) throw std::runtime_error("vision_encoder: missing " + name);
    return t;
}

ggml_tensor * ensure_f32(ggml_context * ctx, ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) return t;
    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

ggml_tensor * layer_norm(
    ggml_context * ctx, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b
) {
    ggml_tensor * y = ggml_norm(ctx, ensure_f32(ctx, x), kLayerNormEps);
    // Use ggml_repeat for explicit broadcast — matches sam3cpp vision_trunk
    // convention. Some ggml backends don't do implicit broadcast on mul/add
    // when weight is a smaller-dim tensor.
    y = ggml_mul(ctx, y, ggml_repeat(ctx, ensure_f32(ctx, w), y));
    y = ggml_add(ctx, y, ggml_repeat(ctx, ensure_f32(ctx, b), y));
    return y;
}

std::string layer_prefix(int layer) {
    return kBase + ".layers." + std::to_string(layer);
}

// Load position_embeddings from GGUF (stored as [1, 576, 1024] in PyTorch
// = ne=[1024, 576, 1] in ggml column-major) and TILE it to the inference
// grid size [1, 5184, 1024]. Matches Sam3ViTEmbeddings._tile_position_embeddings.
//
// Algorithm (from PyTorch):
//   pretrain = 24 (sqrt(576))
//   reshape stored [1, 576, C] to [1, pretrain, pretrain, C] = [1, 24, 24, 1024]
//   permute → [1, C, 24, 24]
//   tile → [1, C, 24*4, 24*4] = [1, 1024, 96, 96]  (repeat_h = 72/24 + 1 = 4)
//   slice → [1, C, 72, 72]
//   permute + reshape → [1, 72*72, C]
//
// Result has hidden fastest (c inner), spatial layout (h, w) with h*72+w order.
std::vector<float> load_and_tile_pos_embed(const GgufModel & model) {
    ggml_tensor * pe = require_tensor(model, kBase + ".embeddings.position_embeddings");
    const int64_t stored_seq = kPretrainGrid * kPretrainGrid;  // 576
    const size_t n_stored = static_cast<size_t>(stored_seq * kHiddenSize);
    std::vector<float> stored(n_stored);
    if (pe->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(pe, stored.data(), 0, n_stored * sizeof(float));
    } else if (pe->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(n_stored);
        ggml_backend_tensor_get(pe, buf.data(), 0, n_stored * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n_stored; ++i) stored[i] = ggml_fp16_to_fp32(buf[i]);
    } else {
        throw std::runtime_error("pos_embed: unsupported dtype");
    }
    // stored is [576, 1024] in row-major (c fastest). Semantic: [24, 24, 1024].
    // Tile to [72, 72, 1024]: for each (h, w) in inference grid, look up
    // (h % 24, w % 24) in the pretrained grid.
    std::vector<float> tiled(static_cast<size_t>(kGridSize * kGridSize * kHiddenSize));
    for (int64_t h = 0; h < kGridSize; ++h) {
        for (int64_t w = 0; w < kGridSize; ++w) {
            const int64_t ph = h % kPretrainGrid;
            const int64_t pw = w % kPretrainGrid;
            const size_t src_off = static_cast<size_t>(
                (ph * kPretrainGrid + pw) * kHiddenSize);
            const size_t dst_off = static_cast<size_t>(
                (h * kGridSize + w) * kHiddenSize);
            std::memcpy(tiled.data() + dst_off, stored.data() + src_off,
                        kHiddenSize * sizeof(float));
        }
    }
    return tiled;
}

std::vector<std::string> per_layer_tensor_names(int layer) {
    const std::string p = layer_prefix(layer);
    std::vector<std::string> names;
    for (const std::string qkvo : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        names.push_back(p + ".attention." + qkvo + ".weight");
        names.push_back(p + ".attention." + qkvo + ".bias");
    }
    names.push_back(p + ".layer_norm1.weight");
    names.push_back(p + ".layer_norm1.bias");
    names.push_back(p + ".layer_norm2.weight");
    names.push_back(p + ".layer_norm2.bias");
    for (const std::string fc : {"fc1", "fc2"}) {
        names.push_back(p + ".mlp." + fc + ".weight");
        names.push_back(p + ".mlp." + fc + ".bias");
    }
    return names;
}

// ── 2D axial RoPE table (per-window, so 24×24 = 576 positions) ─────────
// Matches Sam3ViTRotaryEmbedding: freqs = 1/(theta^(arange(0,dim,4)/dim)) [16],
// concat freqs_x + freqs_y along feature dim → [576, 32], then
// repeat_interleave 2 → [576, 64]. cos/sin.
struct RopeTable {
    int64_t seq_len;                // 576 for window 24×24
    int64_t head_dim;               // 64
    std::vector<float> cos_tab;     // [seq_len, head_dim]
    std::vector<float> sin_tab;     // [seq_len, head_dim]
};

RopeTable build_rope_table_2d(int64_t grid_x, int64_t grid_y, float scale) {
    RopeTable t;
    t.head_dim = kHeadDim;
    t.seq_len  = grid_x * grid_y;
    // freqs [dim/4 = 16]
    const int64_t nfreqs = kHeadDim / 4;
    std::vector<float> freqs(static_cast<size_t>(nfreqs));
    for (int64_t i = 0; i < nfreqs; ++i) {
        // arange(0, dim, 4) → [0, 4, 8, ..., 60], divide by dim=64
        freqs[static_cast<size_t>(i)] = 1.0f / std::pow(kRopeTheta,
            static_cast<float>(4 * i) / static_cast<float>(kHeadDim));
    }
    t.cos_tab.assign(static_cast<size_t>(t.seq_len * t.head_dim), 0.0f);
    t.sin_tab.assign(static_cast<size_t>(t.seq_len * t.head_dim), 0.0f);

    for (int64_t seq = 0; seq < t.seq_len; ++seq) {
        const int64_t x = (seq % grid_x);
        const int64_t y = (seq / grid_x);
        const float xp = static_cast<float>(x) * scale;
        const float yp = static_cast<float>(y) * scale;
        // Match PyTorch: inv_freq = cat([outer(x_pos, freqs), outer(y_pos, freqs)])
        //                inv_freq = repeat_interleave(2, dim=-1)
        // So dim k in [0, 32): freq index k/2 with x_pos.
        //    dim k in [32, 64): freq index (k-32)/2 with y_pos.
        for (int64_t k = 0; k < kHeadDim / 2; ++k) {
            const float angle = xp * freqs[static_cast<size_t>(k / 2)];
            t.cos_tab[static_cast<size_t>(seq * kHeadDim + k)] = std::cos(angle);
            t.sin_tab[static_cast<size_t>(seq * kHeadDim + k)] = std::sin(angle);
        }
        for (int64_t k = kHeadDim / 2; k < kHeadDim; ++k) {
            const float angle = yp * freqs[static_cast<size_t>((k - kHeadDim / 2) / 2)];
            t.cos_tab[static_cast<size_t>(seq * kHeadDim + k)] = std::cos(angle);
            t.sin_tab[static_cast<size_t>(seq * kHeadDim + k)] = std::sin(angle);
        }
    }
    return t;
}

// Apply 2D axial RoPE to a Q or K tensor of shape [B, num_heads, seq, head_dim],
// stored row-major. rotate_pairwise: adjacent pairs (2k, 2k+1) rotate.
//
//   out[..., 2k]   = q[..., 2k]   * cos[..., 2k]   - q[..., 2k+1] * sin[..., 2k]
//   out[..., 2k+1] = q[..., 2k+1] * cos[..., 2k+1] + q[..., 2k]   * sin[..., 2k+1]
void apply_rope_2d_cpu(
    std::vector<float> & qk,   // [B, num_heads, seq, head_dim]
    int64_t B, int64_t H, int64_t S, int64_t D,
    const RopeTable & tab
) {
    if (D != tab.head_dim || S != tab.seq_len) {
        throw std::runtime_error("apply_rope_2d_cpu: shape mismatch");
    }
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t s = 0; s < S; ++s) {
                float * row = qk.data() + ((b * H + h) * S + s) * D;
                const float * c = tab.cos_tab.data() + s * D;
                const float * si = tab.sin_tab.data() + s * D;
                for (int64_t k = 0; k < D; k += 2) {
                    const float x0 = row[k];
                    const float x1 = row[k + 1];
                    row[k]     = x0 * c[k]     - x1 * si[k];
                    row[k + 1] = x1 * c[k + 1] + x0 * si[k + 1];
                }
            }
        }
    }
}

// Window partition: input [C, seq=W*H, 1] (with hidden fastest, spatial in
// row-major (h, w) order). Rearranges to [C, win*win, num_windows*B].
// For W=H=72, win=24: produces 9 windows per batch.
std::vector<float> cpu_window_partition(
    const std::vector<float> & spatial,   // [C, W*H, 1] flat, iterating (h, w, c) c fastest
    int64_t W, int64_t H, int64_t win
) {
    const int64_t nWx = W / win;
    const int64_t nWy = H / win;
    const int64_t num_windows = nWx * nWy;
    std::vector<float> out(spatial.size());
    for (int64_t wy = 0; wy < nWy; ++wy) {
        for (int64_t wx = 0; wx < nWx; ++wx) {
            const int64_t window_idx = wy * nWx + wx;
            for (int64_t iy = 0; iy < win; ++iy) {
                for (int64_t ix = 0; ix < win; ++ix) {
                    const int64_t src_h = wy * win + iy;
                    const int64_t src_w = wx * win + ix;
                    const int64_t src_seq = src_h * W + src_w;
                    const int64_t dst_seq = iy * win + ix;
                    // C is fastest — copy C values contiguously
                    const size_t src_off = static_cast<size_t>(src_seq) * kHiddenSize;
                    const size_t dst_off = static_cast<size_t>(
                        window_idx * win * win + dst_seq) * kHiddenSize;
                    std::memcpy(out.data() + dst_off, spatial.data() + src_off,
                                kHiddenSize * sizeof(float));
                }
            }
        }
    }
    (void)num_windows;
    return out;
}

// Inverse of window_partition.
std::vector<float> cpu_window_unpartition(
    const std::vector<float> & windowed,   // [C, win*win, num_windows] flat
    int64_t W, int64_t H, int64_t win
) {
    const int64_t nWx = W / win;
    const int64_t nWy = H / win;
    std::vector<float> out(windowed.size());
    for (int64_t wy = 0; wy < nWy; ++wy) {
        for (int64_t wx = 0; wx < nWx; ++wx) {
            const int64_t window_idx = wy * nWx + wx;
            for (int64_t iy = 0; iy < win; ++iy) {
                for (int64_t ix = 0; ix < win; ++ix) {
                    const int64_t dst_h = wy * win + iy;
                    const int64_t dst_w = wx * win + ix;
                    const int64_t dst_seq = dst_h * W + dst_w;
                    const int64_t src_seq = iy * win + ix;
                    const size_t src_off = static_cast<size_t>(
                        window_idx * win * win + src_seq) * kHiddenSize;
                    const size_t dst_off = static_cast<size_t>(dst_seq) * kHiddenSize;
                    std::memcpy(out.data() + dst_off, windowed.data() + src_off,
                                kHiddenSize * sizeof(float));
                }
            }
        }
    }
    return out;
}

std::vector<std::string> trunk_top_tensor_names() {
    return {
        kBase + ".embeddings.patch_embeddings.projection.weight",
        kBase + ".embeddings.position_embeddings",
        kBase + ".layer_norm.weight",
        kBase + ".layer_norm.bias",
    };
}

}  // namespace

InstructsamVisionEncoder::InstructsamVisionEncoder(const GgufModel & model) : model_(model) {}

size_t InstructsamVisionEncoder::validate_all_tensors_present() const {
    size_t probed = 0;
    for (const auto & name : trunk_top_tensor_names()) {
        if (model_.find_tensor(name) == nullptr) {
            throw std::runtime_error("vision_encoder: missing trunk-top tensor: " + name);
        }
        ++probed;
    }
    for (int layer = 0; layer < kNumLayers; ++layer) {
        for (const auto & name : per_layer_tensor_names(layer)) {
            if (model_.find_tensor(name) == nullptr) {
                throw std::runtime_error("vision_encoder: missing per-layer tensor: " + name);
            }
            ++probed;
        }
    }
    return probed;
}

std::vector<float> InstructsamVisionEncoder::run_patch_embed_only(
    const std::vector<float> & pixel_values,
    const std::vector<int64_t> & pixel_shape
) const {
    if (pixel_shape.size() != 3 || pixel_shape[0] != 3 ||
        pixel_shape[1] != kImageSize || pixel_shape[2] != kImageSize) {
        throw std::runtime_error("pixel_values must be [3, 1008, 1008]");
    }
    ggml_backend_t backend = model_.backend();
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_t backends[2] = { backend, nullptr };
    int n_backends = 1;
    if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        cpu_backend = model_.cpu_backend();
        if (cpu_backend != nullptr) { backends[1] = cpu_backend; n_backends = 2; }
    }
    const size_t graph_size = 4096;
    const size_t ctx_size = ggml_tensor_overhead() * graph_size +
                            ggml_graph_overhead_custom(graph_size, false);
    std::vector<uint8_t> ctx_buf(ctx_size);
    ggml_context * ctx = ggml_init({ctx_size, ctx_buf.data(), true});
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, graph_size, false, true);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_tensor * pv_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kImageSize, kImageSize, 3, 1);
    ggml_backend_sched_set_tensor_backend(sched, pv_t, backend);
    // No weight permute: PyTorch [OutC, InC, KH, KW] → GGUF ne=[KW, KH, InC, OutC]
    // exactly matches ggml_conv_2d's expected [KW, KH, IC, OC] layout.
    ggml_tensor * proj_w = ensure_f32(ctx, require_tensor(model_,
        kBase + ".embeddings.patch_embeddings.projection.weight"));
    ggml_tensor * conv = ggml_conv_2d(ctx, proj_w, pv_t,
        kPatchSize, kPatchSize, 0, 0, 1, 1);
    // conv ne = [W=72, H=72, C=1024, 1]. Match PyTorch flatten(2).transpose(1,2)
    // = [N, H*W, C]: permute so C is fastest, then W, then H → matches
    // PyTorch memory layout after the transpose.
    // ggml_permute uses `result.ne[axis_i] = a.ne[i]` — so `permute(a, 1, 2, 0, 3)`
    // takes conv output ne=[W, H, C, 1] and produces new ne=[C, W, H, 1] with
    // C fastest, W middle, H outer. Materialized memory order: (h, w, c) with
    // c innermost — exactly matches PyTorch's flatten(2).transpose(1,2) layout.
    ggml_tensor * out = ggml_cont(ctx, ggml_permute(ctx, conv, 1, 2, 0, 3));
    out = ggml_reshape_3d(ctx, out, kHiddenSize, kNumPatches, 1);
    ggml_tensor * cap = ggml_cont(ctx, out);
    ggml_build_forward_expand(gf, cap);
    ggml_backend_sched_alloc_graph(sched, gf);
    ggml_backend_tensor_set(pv_t, pixel_values.data(), 0, pixel_values.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_free(sched); ggml_free(ctx);
        throw std::runtime_error("patch_embed compute failed");
    }
    std::vector<float> result(static_cast<size_t>(ggml_nelements(cap)));
    ggml_backend_tensor_get(cap, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return result;
}

std::vector<float> InstructsamVisionEncoder::run_prenorm(
    const std::vector<float> & pixel_values,
    const std::vector<int64_t> & pixel_shape
) const {
    if (pixel_shape.size() != 3 ||
        pixel_shape[0] != 3 ||
        pixel_shape[1] != kImageSize ||
        pixel_shape[2] != kImageSize) {
        throw std::runtime_error("vision_encoder: pixel_values must be [3, 1008, 1008]");
    }

    ggml_backend_t backend = model_.backend();
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_t backends[2] = { backend, nullptr };
    int n_backends = 1;
    if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        cpu_backend = model_.cpu_backend();
        if (cpu_backend != nullptr) { backends[1] = cpu_backend; n_backends = 2; }
    }

    const size_t graph_size = 8192;
    const size_t ctx_size = ggml_tensor_overhead() * graph_size +
                            ggml_graph_overhead_custom(graph_size, false);
    std::vector<uint8_t> ctx_buf(ctx_size);
    ggml_context * ctx = ggml_init({ctx_size, ctx_buf.data(), true});
    if (!ctx) throw std::runtime_error("ggml_init failed");
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, n_backends, graph_size, false, true);
    if (!sched) { ggml_free(ctx); throw std::runtime_error("sched_new failed"); }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, graph_size, false);

    // pixel_values in ggml column-major: ne = [W, H, C, N] = [1008, 1008, 3, 1]
    ggml_tensor * pv_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
        kImageSize, kImageSize, 3, 1);
    ggml_backend_sched_set_tensor_backend(sched, pv_t, backend);

    // Patch embed: Conv2d(3→1024, k=14, s=14, no bias)
    // PyTorch weight [Out=1024, In=3, KH=14, KW=14] → GGUF ne [14, 14, 3, 1024]
    // = exactly what ggml_conv_2d expects [KW, KH, InC, OutC].
    ggml_tensor * proj_w = ensure_f32(ctx, require_tensor(model_,
        kBase + ".embeddings.patch_embeddings.projection.weight"));
    ggml_tensor * conv = ggml_conv_2d(ctx, proj_w, pv_t,
        kPatchSize, kPatchSize,     // stride
        0, 0,                        // padding
        1, 1);                       // dilation
    // conv now has ne = [72, 72, 1024, 1]

    // Reshape spatial [W=72, H=72, C=1024, N=1] to token layout
    // [C=1024, seq=5184, 1, 1] — matches Sam3ViTEmbeddings.forward's
    // flatten(2).transpose(1,2): [B, C, H, W] → [B, C, H*W] → [B, H*W, C].
    ggml_tensor * tokens = ggml_reshape_3d(ctx, conv, kHiddenSize, kNumPatches, 1);
    // Now tokens ne = [1024, 5184, 1] which in row-major is [seq, hidden] with
    // dim 0 = hidden (fastest). BUT ggml views are column-major-first, so this
    // shape means "hidden is fastest, then seq". PyTorch semantically wants
    // [seq_len, hidden] where hidden is trailing/fastest. So this IS correct.
    // Wait — need to be more careful. conv output has spatial layout
    // [W, H, C, 1]. Flatten(2) in PyTorch reduces [B, C, H, W] to
    // [B, C, H*W]. That's C outer, H*W inner. Then transpose(1,2) swaps to
    // [B, H*W, C]. So end result is [seq, C] with C fastest.
    // For ggml (column-major), [seq, C] as PyTorch view corresponds to
    // ne=[C, seq] where C is fastest. That matches ggml_reshape_3d(conv, C, seq, 1).
    // BUT — conv output layout in ggml: ne=[W, H, C, 1], stride=[1, W, W*H, ...].
    // W is fastest. To get PyTorch [B, C, H*W] layout, we need to permute
    // dim 0 (W) with dim 2 (C) so C becomes fastest. Then H*W folds.
    //
    // Actually the sam3cpp vision_trunk.cpp faces the exact same problem
    // and handles it. Let me use the same pattern: transpose spatial dims
    // together to get channel-fastest layout.
    //
    // For now: use ggml_cont(ggml_permute(...)) to bring C to fastest,
    // then reshape to [C, HW].
    // Corrected permute: ggml_permute uses `result.ne[axis_i] = a.ne[i]`.
    // permute(conv, 1, 2, 0, 3) with conv ne=[W, H, C, 1] gives result
    // ne=[C, W, H, 1] with C fastest, W middle, H outer — matches PyTorch's
    // flatten(2).transpose(1, 2) memory layout.
    tokens = ggml_cont(ctx, ggml_permute(ctx, conv, 1, 2, 0, 3));
    tokens = ggml_reshape_3d(ctx, tokens, kHiddenSize, kNumPatches, 1);

    // Add position embeddings — tiled from [1, 576, 1024] (24×24 pretrain
    // grid) to [1, 5184, 1024] (72×72 inference grid). Tiling is CPU-side.
    const std::vector<float> tiled_pos = load_and_tile_pos_embed(model_);
    ggml_tensor * pos_embed_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
        kHiddenSize, kNumPatches, 1);
    ggml_backend_sched_set_tensor_backend(sched, pos_embed_t, backend);
    tokens = ggml_add(ctx, tokens, pos_embed_t);

    // Pre-trunk LayerNorm
    tokens = layer_norm(ctx, tokens,
        require_tensor(model_, kBase + ".layer_norm.weight"),
        require_tensor(model_, kBase + ".layer_norm.bias"));

    // Reshape to spatial [W, H, C, 1] for windowed attention downstream —
    // but for validation, keep as [C, seq, 1] since PyTorch reference stores
    // this intermediate in the same format.
    ggml_tensor * out = ggml_cont(ctx, tokens);
    ggml_build_forward_expand(gf, out);
    ggml_backend_sched_alloc_graph(sched, gf);

    ggml_backend_tensor_set(pv_t, pixel_values.data(), 0, pixel_values.size() * sizeof(float));
    ggml_backend_tensor_set(pos_embed_t, tiled_pos.data(), 0, tiled_pos.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_sched_free(sched); ggml_free(ctx);
        throw std::runtime_error("vision_encoder prenorm compute failed");
    }

    std::vector<float> result(static_cast<size_t>(ggml_nelements(out)));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return result;
}

std::vector<float> InstructsamVisionEncoder::run_all_layers(
    const std::vector<float> & pixel_values,
    const std::vector<int64_t> & pixel_shape
) const {
    // Patch embed + pos embed + pre-trunk LN → [72*72, 1024]
    auto hidden = run_prenorm(pixel_values, pixel_shape);
    for (int layer = 0; layer < kNumLayers; ++layer) {
        hidden = run_layer(layer, hidden);
    }
    return hidden;
}

std::vector<float> InstructsamVisionEncoder::run_layer(
    int layer_idx, const std::vector<float> & hidden_in
) const {
    if (hidden_in.size() != static_cast<size_t>(kNumPatches * kHiddenSize)) {
        throw std::runtime_error("run_layer: expected [72*72, 1024] input");
    }
    const std::string p = layer_prefix(layer_idx);
    const bool global = is_global_layer(layer_idx);

    auto get_f32 = [&](const std::string & name, size_t n) {
        ggml_tensor * t = require_tensor(model_, name);
        std::vector<float> v(n);
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
        } else if (t->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> buf(n);
            ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(ggml_fp16_t));
            for (size_t i = 0; i < n; ++i) v[i] = ggml_fp16_to_fp32(buf[i]);
        } else throw std::runtime_error("run_layer: unsupported dtype");
        return v;
    };

    const auto ln1_w = get_f32(p + ".layer_norm1.weight", kHiddenSize);
    const auto ln1_b = get_f32(p + ".layer_norm1.bias",   kHiddenSize);
    const auto ln2_w = get_f32(p + ".layer_norm2.weight", kHiddenSize);
    const auto ln2_b = get_f32(p + ".layer_norm2.bias",   kHiddenSize);

    const std::vector<float> residual1 = hidden_in;

    auto cpu_ln = [&](const std::vector<float> & x, int64_t N, int64_t D,
                      const std::vector<float> & w, const std::vector<float> & b) {
        std::vector<float> out(x.size());
        for (int64_t i = 0; i < N; ++i) {
            double mean = 0.0, var = 0.0;
            for (int64_t d = 0; d < D; ++d) mean += x[static_cast<size_t>(i * D + d)];
            mean /= D;
            for (int64_t d = 0; d < D; ++d) {
                const double diff = x[static_cast<size_t>(i * D + d)] - mean;
                var += diff * diff;
            }
            var /= D;
            const double inv_std = 1.0 / std::sqrt(var + kLayerNormEps);
            for (int64_t d = 0; d < D; ++d) {
                const double normed = (x[static_cast<size_t>(i * D + d)] - mean) * inv_std;
                out[static_cast<size_t>(i * D + d)] = static_cast<float>(
                    normed * w[static_cast<size_t>(d)] + b[static_cast<size_t>(d)]);
            }
        }
        return out;
    };
    const std::vector<float> ln1_out = cpu_ln(hidden_in, kNumPatches, kHiddenSize, ln1_w, ln1_b);

    const int64_t win = global ? kGridSize : kWindowSize;
    const int64_t nW = global ? 1 : (kGridSize / kWindowSize);
    const int64_t num_windows = nW * nW;
    const int64_t seq_per_win = win * win;
    std::vector<float> windowed = global
        ? ln1_out
        : cpu_window_partition(ln1_out, kGridSize, kGridSize, win);

    const auto qw = get_f32(p + ".attention.q_proj.weight", kHiddenSize * kHiddenSize);
    const auto qb = get_f32(p + ".attention.q_proj.bias",   kHiddenSize);
    const auto kw = get_f32(p + ".attention.k_proj.weight", kHiddenSize * kHiddenSize);
    const auto kb = get_f32(p + ".attention.k_proj.bias",   kHiddenSize);
    const auto vw = get_f32(p + ".attention.v_proj.weight", kHiddenSize * kHiddenSize);
    const auto vb = get_f32(p + ".attention.v_proj.bias",   kHiddenSize);
    const auto ow = get_f32(p + ".attention.o_proj.weight", kHiddenSize * kHiddenSize);
    const auto ob = get_f32(p + ".attention.o_proj.bias",   kHiddenSize);

    auto cpu_linear = [](const std::vector<float> & x, int64_t N,
                         int64_t D_in, int64_t D_out,
                         const std::vector<float> & w,
                         const std::vector<float> & b) {
        std::vector<float> y(static_cast<size_t>(N * D_out));
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t o = 0; o < D_out; ++o) {
                float s = b[static_cast<size_t>(o)];
                for (int64_t k = 0; k < D_in; ++k) {
                    s += w[static_cast<size_t>(o * D_in + k)] *
                         x[static_cast<size_t>(n * D_in + k)];
                }
                y[static_cast<size_t>(n * D_out + o)] = s;
            }
        }
        return y;
    };

    const int64_t total_tokens = num_windows * seq_per_win;
    auto Q = cpu_linear(windowed, total_tokens, kHiddenSize, kHiddenSize, qw, qb);
    auto K = cpu_linear(windowed, total_tokens, kHiddenSize, kHiddenSize, kw, kb);
    auto V = cpu_linear(windowed, total_tokens, kHiddenSize, kHiddenSize, vw, vb);

    // Transpose to [num_windows, num_heads, seq_per_win, head_dim] for RoPE + attention
    auto to_heads = [&](const std::vector<float> & flat) {
        std::vector<float> out(flat.size());
        for (int64_t b = 0; b < num_windows; ++b) {
            for (int64_t h = 0; h < kNumHeads; ++h) {
                for (int64_t s = 0; s < seq_per_win; ++s) {
                    for (int64_t d = 0; d < kHeadDim; ++d) {
                        const size_t src = ((b * seq_per_win + s) * kNumHeads + h) * kHeadDim + d;
                        const size_t dst = ((b * kNumHeads + h) * seq_per_win + s) * kHeadDim + d;
                        out[dst] = flat[src];
                    }
                }
            }
        }
        return out;
    };
    Q = to_heads(Q); K = to_heads(K); V = to_heads(V);

    const float rope_scale = global ? (static_cast<float>(kWindowSize) / kGridSize) : 1.0f;
    const RopeTable rope = build_rope_table_2d(win, win, rope_scale);
    apply_rope_2d_cpu(Q, num_windows, kNumHeads, seq_per_win, kHeadDim, rope);
    apply_rope_2d_cpu(K, num_windows, kNumHeads, seq_per_win, kHeadDim, rope);

    // Scaled dot-product attention CPU-side (small compute — batched over windows/heads).
    const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    std::vector<float> attn(static_cast<size_t>(total_tokens * kHiddenSize));
    for (int64_t b = 0; b < num_windows; ++b) {
        for (int64_t h = 0; h < kNumHeads; ++h) {
            const float * qh = Q.data() + ((b * kNumHeads + h) * seq_per_win) * kHeadDim;
            const float * kh = K.data() + ((b * kNumHeads + h) * seq_per_win) * kHeadDim;
            const float * vh = V.data() + ((b * kNumHeads + h) * seq_per_win) * kHeadDim;
            std::vector<float> S(static_cast<size_t>(seq_per_win * seq_per_win));
            for (int64_t i = 0; i < seq_per_win; ++i) {
                for (int64_t j = 0; j < seq_per_win; ++j) {
                    float s = 0.0f;
                    for (int64_t d = 0; d < kHeadDim; ++d) {
                        s += qh[i * kHeadDim + d] * kh[j * kHeadDim + d];
                    }
                    S[static_cast<size_t>(i * seq_per_win + j)] = s * scale;
                }
            }
            for (int64_t i = 0; i < seq_per_win; ++i) {
                float * row = S.data() + i * seq_per_win;
                float m = row[0];
                for (int64_t j = 1; j < seq_per_win; ++j) if (row[j] > m) m = row[j];
                float sum = 0.0f;
                for (int64_t j = 0; j < seq_per_win; ++j) { row[j] = std::exp(row[j] - m); sum += row[j]; }
                for (int64_t j = 0; j < seq_per_win; ++j) row[j] /= sum;
            }
            for (int64_t i = 0; i < seq_per_win; ++i) {
                for (int64_t d = 0; d < kHeadDim; ++d) {
                    float y = 0.0f;
                    for (int64_t j = 0; j < seq_per_win; ++j) {
                        y += S[static_cast<size_t>(i * seq_per_win + j)] * vh[j * kHeadDim + d];
                    }
                    const size_t out_off = ((b * seq_per_win + i) * kNumHeads + h) * kHeadDim + d;
                    attn[out_off] = y;
                }
            }
        }
    }

    auto attn_proj = cpu_linear(attn, total_tokens, kHiddenSize, kHiddenSize, ow, ob);

    std::vector<float> attn_unwin = global
        ? std::move(attn_proj)
        : cpu_window_unpartition(attn_proj, kGridSize, kGridSize, win);

    for (size_t i = 0; i < attn_unwin.size(); ++i) attn_unwin[i] += residual1[i];
    const std::vector<float> residual2 = attn_unwin;

    const std::vector<float> ln2_out = cpu_ln(attn_unwin, kNumPatches, kHiddenSize, ln2_w, ln2_b);

    const auto fc1_w = get_f32(p + ".mlp.fc1.weight", kMlpDim * kHiddenSize);
    const auto fc1_b = get_f32(p + ".mlp.fc1.bias",   kMlpDim);
    const auto fc2_w = get_f32(p + ".mlp.fc2.weight", kHiddenSize * kMlpDim);
    const auto fc2_b = get_f32(p + ".mlp.fc2.bias",   kHiddenSize);

    auto mlp_mid = cpu_linear(ln2_out, kNumPatches, kHiddenSize, kMlpDim, fc1_w, fc1_b);
    for (float & x : mlp_mid) {
        x = 0.5f * x * (1.0f + std::erf(x / std::sqrt(2.0f)));
    }
    auto mlp_out = cpu_linear(mlp_mid, kNumPatches, kMlpDim, kHiddenSize, fc2_w, fc2_b);

    for (size_t i = 0; i < mlp_out.size(); ++i) mlp_out[i] += residual2[i];
    return mlp_out;
}

}  // namespace sam3
