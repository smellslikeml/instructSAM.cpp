// instructsam — Day 9d
//
// End-to-end runnable CLI: image + text query → mask PNGs.
//
// What's fully native ggml/CPU as of Day 9d:
//   - image preprocess (stb_image)
//   - SAM3 vision encoder (32-layer ViT + FPN neck)
//   - mmproj (Qwen3-VL vision → LM embed space, via libmtmd)
//   - tokenizer + chat template (via llama.cpp vocab-only)
//   - Qwen3 LM fork forward with image-conditioned prefix
//   - mask_hidden_fcs bridge, DETR encoder+decoder, PCA, FPN, mask_tail
//
// What still uses PyTorch-captured helpers (documented follow-up):
//   - text_hidden_fcs port (would produce text_features + prompt_features
//     from phrase-token embeddings — for now, we load the per-object
//     captures from ref_dir/binaries_obj*/enc_text_features.f32 etc.)
//   - init_ref_boxes and query_pos_L0 come from PyTorch's box_head init;
//     for now loaded from ref_dir. This is a fixed learned init not per-image.
//   - free-form phrase generation from an open-ended query — currently we
//     accept an explicit phrase list.

#include "sam3/gguf_model.h"
#include "sam3/instructsam_decoder.h"
#include "sam3/instructsam_detr_encoder.h"
#include "sam3/instructsam_lm_bridge.h"
#include "sam3/instructsam_lm_forward.h"
#include "sam3/instructsam_mask_tail.h"
#include "sam3/instructsam_mmproj.h"
#include "sam3/instructsam_pixel_decoder.h"
#include "sam3/instructsam_preprocess.h"
#include "sam3/instructsam_prompt_cross_attn.h"
#include "sam3/instructsam_vision_encoder.h"
#include "sam3/instructsam_tokenizer.h"
#include "sam3/instructsam_vision_encoder.h"

#include "ggml.h"
#include "ggml-backend.h"

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

// stb_image_write for output PNGs — implementation lives in this TU.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// stb_image itself is already linked from sam3cpp (via pipeline.cpp) and
// libmtmd.a, so we can't include stb_image.h here without a double
// definition of stbi_*. Forward-declare just the two symbols we need.
extern "C" {
unsigned char * stbi_load(const char *, int *, int *, int *, int);
void            stbi_image_free(void *);
}

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_bin_f32(const std::string & path, std::vector<int64_t> & shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4]; f.read(magic, 4);
    if (std::string(magic, 4) != "BIN1") throw std::runtime_error("bad magic in " + path);
    int32_t ndim = 0; f.read(reinterpret_cast<char *>(&ndim), 4);
    shape.assign(static_cast<size_t>(ndim), 0);
    size_t total = 1;
    for (int i = 0; i < ndim; ++i) {
        int64_t d = 0; f.read(reinterpret_cast<char *>(&d), 8);
        shape[static_cast<size_t>(i)] = d; total *= static_cast<size_t>(d);
    }
    std::vector<float> data(total);
    f.read(reinterpret_cast<char *>(data.data()), total * sizeof(float));
    return data;
}

void write_bin_f32(const std::string & path, const std::vector<float> & data,
                   const std::vector<int64_t> & shape) {
    // Create parent directory tree if it doesn't exist — otherwise
    // std::ofstream silently fails to open the file and the mask blob
    // vanishes with no error printed.
    const auto slash = path.find_last_of('/');
    if (slash != std::string::npos) {
        std::filesystem::create_directories(path.substr(0, slash));
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("write_bin_f32: cannot open " + path +
                                      " (check --out-dir permissions)");
    f.write("BIN1", 4);
    int32_t ndim = static_cast<int32_t>(shape.size());
    f.write(reinterpret_cast<const char *>(&ndim), 4);
    for (int64_t d : shape) f.write(reinterpret_cast<const char *>(&d), 8);
    f.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
    if (!f) throw std::runtime_error("write_bin_f32: write failed for " + path);
}

std::vector<float> get_f32(const sam3::GgufModel & model, const std::string & name, size_t n) {
    ggml_tensor * t = model.find_weight(name);
    if (!t) throw std::runtime_error("get_f32 missing " + name);
    std::vector<float> v(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) v[i] = ggml_fp16_to_fp32(buf[i]);
    } else throw std::runtime_error("get_f32 unsupported dtype");
    return v;
}

std::vector<float> cpu_layer_norm(
    const std::vector<float> & x, int64_t N, int64_t D,
    const std::vector<float> & w, const std::vector<float> & b, float eps = 1e-5f
) {
    std::vector<float> out(x.size());
    for (int64_t i = 0; i < N; ++i) {
        double mean = 0.0, var = 0.0;
        for (int64_t d = 0; d < D; ++d) mean += x[i * D + d];
        mean /= D;
        for (int64_t d = 0; d < D; ++d) {
            const double diff = x[i * D + d] - mean;
            var += diff * diff;
        }
        var /= D;
        const double inv_std = 1.0 / std::sqrt(var + eps);
        for (int64_t d = 0; d < D; ++d) {
            out[i * D + d] = static_cast<float>((x[i * D + d] - mean) * inv_std
                                                 * w[d] + b[d]);
        }
    }
    return out;
}

// y[k, o] = sum_i x[k, i] * W[o, i] + b[o]  — matches PyTorch nn.Linear.
// x [N, in], W [out, in], b [out] → out [N, out].
std::vector<float> cpu_linear(
    const std::vector<float> & x, int64_t N, int64_t D_in,
    const std::vector<float> & W, const std::vector<float> & b, int64_t D_out
) {
    std::vector<float> out(N * D_out);
    for (int64_t k = 0; k < N; ++k) {
        for (int64_t o = 0; o < D_out; ++o) {
            double s = b[o];
            const float * xr = x.data() + k * D_in;
            const float * wr = W.data() + o * D_in;
            for (int64_t i = 0; i < D_in; ++i) s += xr[i] * wr[i];
            out[k * D_out + o] = static_cast<float>(s);
        }
    }
    return out;
}

// InstructSAM's Sam3DotProductScoring: classification head that scores
// each of the 10 decoder queries against the pooled text prompt.
//
//   text_features:    [T, H]  from text_hidden_fcs
//   text_mask:        [T]     1.0 = valid, 0.0 = pad
//   decoder_queries:  [Q, H]  last-layer DETR decoder output (post output_layer_norm)
//
// Returns [Q] pre-sigmoid pred_logits.
struct DotProdScoringWeights {
    // text_mlp: Linear(H→intermediate) + ReLU + Linear(intermediate→H)
    std::vector<float> tm_l1_w, tm_l1_b;    // [intermediate, H], [intermediate]
    std::vector<float> tm_l2_w, tm_l2_b;    // [H, intermediate], [H]
    std::vector<float> tm_out_ln_w, tm_out_ln_b;   // [H], [H]
    std::vector<float> text_proj_w, text_proj_b;   // [H, H], [H]
    std::vector<float> query_proj_w, query_proj_b; // [H, H], [H]
    int64_t H = 256;
    int64_t intermediate = 2048;
};

DotProdScoringWeights load_scoring_weights(const sam3::GgufModel & grounding) {
    DotProdScoringWeights w;
    w.tm_l1_w      = get_f32(grounding, "dot_prod_scoring.text_mlp.layer1.weight", 2048 * 256);
    w.tm_l1_b      = get_f32(grounding, "dot_prod_scoring.text_mlp.layer1.bias",   2048);
    w.tm_l2_w      = get_f32(grounding, "dot_prod_scoring.text_mlp.layer2.weight", 256 * 2048);
    w.tm_l2_b      = get_f32(grounding, "dot_prod_scoring.text_mlp.layer2.bias",   256);
    w.tm_out_ln_w  = get_f32(grounding, "dot_prod_scoring.text_mlp_out_norm.weight", 256);
    w.tm_out_ln_b  = get_f32(grounding, "dot_prod_scoring.text_mlp_out_norm.bias",   256);
    w.text_proj_w  = get_f32(grounding, "dot_prod_scoring.text_proj.weight", 256 * 256);
    w.text_proj_b  = get_f32(grounding, "dot_prod_scoring.text_proj.bias",   256);
    w.query_proj_w = get_f32(grounding, "dot_prod_scoring.query_proj.weight", 256 * 256);
    w.query_proj_b = get_f32(grounding, "dot_prod_scoring.query_proj.bias",   256);
    return w;
}

std::vector<float> dot_prod_scoring(
    const DotProdScoringWeights & w,
    const std::vector<float> & text_features,   // [T, H]
    const std::vector<float> & text_mask,       // [T]
    const std::vector<float> & decoder_queries  // [Q, H]
) {
    const int64_t T = static_cast<int64_t>(text_mask.size());
    const int64_t H = w.H;
    const int64_t Q = static_cast<int64_t>(decoder_queries.size() / H);

    // text_features = text_mlp_out_norm(text_features + text_mlp(text_features))
    const auto tm1 = cpu_linear(text_features, T, H, w.tm_l1_w, w.tm_l1_b, w.intermediate);
    std::vector<float> tm1_relu(tm1.size());
    for (size_t i = 0; i < tm1.size(); ++i) tm1_relu[i] = std::max(0.0f, tm1[i]);
    const auto tm2 = cpu_linear(tm1_relu, T, w.intermediate, w.tm_l2_w, w.tm_l2_b, H);
    std::vector<float> tf_res(text_features.size());
    for (size_t i = 0; i < tf_res.size(); ++i) tf_res[i] = text_features[i] + tm2[i];
    const auto tf_norm = cpu_layer_norm(tf_res, T, H, w.tm_out_ln_w, w.tm_out_ln_b);

    // mean-pool over valid text tokens
    std::vector<float> pooled(H, 0.0f);
    double n_valid = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        const float m = text_mask[t];
        if (m == 0.0f) continue;
        n_valid += m;
        for (int64_t d = 0; d < H; ++d) pooled[d] += m * tf_norm[t * H + d];
    }
    if (n_valid < 1.0) n_valid = 1.0;
    for (int64_t d = 0; d < H; ++d) pooled[d] = static_cast<float>(pooled[d] / n_valid);

    // proj_text = text_proj(pooled); shape [H]
    const auto proj_text = cpu_linear(pooled, 1, H, w.text_proj_w, w.text_proj_b, H);
    // proj_queries = query_proj(decoder_queries); shape [Q, H]
    const auto proj_queries = cpu_linear(decoder_queries, Q, H, w.query_proj_w, w.query_proj_b, H);

    // scores[q] = (proj_queries[q] · proj_text) * scale, clamped to ±12.
    const float scale = 1.0f / std::sqrt(static_cast<float>(H));
    std::vector<float> scores(Q);
    for (int64_t q = 0; q < Q; ++q) {
        double s = 0.0;
        for (int64_t d = 0; d < H; ++d) s += proj_queries[q * H + d] * proj_text[d];
        s *= scale;
        if (s >  12.0) s =  12.0;
        if (s < -12.0) s = -12.0;
        scores[q] = static_cast<float>(s);
    }
    return scores;
}

// ── Mask upsampling + PNG output ────────────────────────────────────────

// Sigmoid + bilinear resize a [288, 288] mask logit tensor to the input
// image size, then threshold at 0.5. Returns 8-bit binary mask [H*W].
inline std::vector<uint8_t> mask_logits_to_binary(
    const std::vector<float> & logits_288, int Hout, int Wout
) {
    constexpr int Msrc = 288;
    // Sigmoid to [0,1] on 288×288
    std::vector<float> p(Msrc * Msrc);
    for (int i = 0; i < Msrc * Msrc; ++i) p[i] = 1.0f / (1.0f + std::exp(-logits_288[i]));
    // Bilinear resize [Msrc,Msrc] → [Hout, Wout]. Uses PIL-convention centering.
    std::vector<uint8_t> out(Hout * Wout);
    const double sy = static_cast<double>(Msrc) / Hout;
    const double sx = static_cast<double>(Msrc) / Wout;
    for (int y = 0; y < Hout; ++y) {
        const double cy = (y + 0.5) * sy - 0.5;
        const int y0 = std::max(0, std::min(Msrc - 1, static_cast<int>(std::floor(cy))));
        const int y1 = std::min(Msrc - 1, y0 + 1);
        const double fy = cy - y0;
        for (int x = 0; x < Wout; ++x) {
            const double cx = (x + 0.5) * sx - 0.5;
            const int x0 = std::max(0, std::min(Msrc - 1, static_cast<int>(std::floor(cx))));
            const int x1 = std::min(Msrc - 1, x0 + 1);
            const double fx = cx - x0;
            const double a = p[y0 * Msrc + x0], b = p[y0 * Msrc + x1];
            const double c = p[y1 * Msrc + x0], d = p[y1 * Msrc + x1];
            const double v = a * (1-fy) * (1-fx) + b * (1-fy) * fx
                           + c * fy     * (1-fx) + d * fy     * fx;
            out[y * Wout + x] = (v > 0.5) ? 255 : 0;
        }
    }
    return out;
}

// Palette for overlay panel — 12 distinct colors, same order as the viz
// script so overlay-panel visuals stay comparable across paths.
constexpr uint8_t kPalette[12][3] = {
    {255,  80,  80}, { 80, 200,  80}, { 80, 140, 255}, {255, 200,  60},
    {200,  80, 200}, { 80, 220, 220}, {255, 140,  80}, {140,  80, 220},
    {120, 220,  80}, {220, 120, 120}, { 80, 100, 200}, {200, 200,  80},
};

// Compose an [orig | tinted-overlay] side-by-side RGB image and write it
// as PNG. rgb is [H*W*3] uint8. masks_bin is a list of [H*W] uint8 (0/255).
inline void write_overlay_panel(
    const std::string & path,
    const uint8_t * rgb, int W, int H,
    const std::vector<std::vector<uint8_t>> & masks_bin,
    const std::vector<std::string> & labels
) {
    const int panelW = W * 2;
    std::vector<uint8_t> panel(panelW * H * 3);
    // Left half: original
    for (int y = 0; y < H; ++y) {
        std::memcpy(panel.data() + y * panelW * 3, rgb + y * W * 3, W * 3);
    }
    // Right half: tinted overlay, alpha=0.5
    constexpr float kAlpha = 0.5f;
    for (int y = 0; y < H; ++y) {
        const uint8_t * src = rgb + y * W * 3;
        uint8_t * dst = panel.data() + (y * panelW + W) * 3;
        std::memcpy(dst, src, W * 3);
        for (size_t k = 0; k < masks_bin.size(); ++k) {
            const uint8_t * m = masks_bin[k].data() + y * W;
            const uint8_t (&col)[3] = kPalette[k % 12];
            for (int x = 0; x < W; ++x) {
                if (m[x]) {
                    dst[x*3+0] = static_cast<uint8_t>((1.0f - kAlpha) * dst[x*3+0] + kAlpha * col[0]);
                    dst[x*3+1] = static_cast<uint8_t>((1.0f - kAlpha) * dst[x*3+1] + kAlpha * col[1]);
                    dst[x*3+2] = static_cast<uint8_t>((1.0f - kAlpha) * dst[x*3+2] + kAlpha * col[2]);
                }
            }
        }
    }
    // Draw a legend as color-tagged text at top-left of the right half.
    // Text is a 4-pixel-tall dumb bitmap font — we just draw solid color
    // swatches; labels are printed to stdout so the user sees which color
    // maps to which phrase. (Real font rendering isn't worth the deps.)
    for (size_t k = 0; k < labels.size() && k < 12; ++k) {
        const int y0 = 6 + static_cast<int>(k) * 12;
        const int y1 = y0 + 8;
        if (y1 >= H) break;
        const uint8_t (&col)[3] = kPalette[k % 12];
        for (int y = y0; y < y1; ++y) {
            for (int x = 6; x < 22; ++x) {
                uint8_t * px = panel.data() + (y * panelW + (W + x)) * 3;
                px[0] = col[0]; px[1] = col[1]; px[2] = col[2];
            }
        }
    }
    stbi_write_png(path.c_str(), panelW, H, 3, panel.data(), panelW * 3);
}

// ── Detr-side constants that only depend on the grounding GGUF + image
// spatial shape (not on the per-image LM output). Previously loaded from
// per-object ref captures; now computed natively. ────────────────────────

// Sam3SinePositionEmbedding.forward for a mask-free HxW grid, hidden_dim=256.
// Returns [H*W, hidden_dim] flat (row-major over spatial), matching the
// enc_vision_pos_flat.f32 captures.
std::vector<float> compute_vision_pos_2d_sincos(
    int64_t H, int64_t W, int64_t hidden_dim,
    float temperature = 10000.0f, float scale = 2.0f * static_cast<float>(M_PI),
    float eps = 1e-6f
) {
    const int64_t num_pos_feats = hidden_dim / 2;
    std::vector<float> dim_t(static_cast<size_t>(num_pos_feats));
    for (int64_t i = 0; i < num_pos_feats; ++i) {
        const float exponent = 2.0f * std::floor(static_cast<float>(i) / 2.0f) /
                                static_cast<float>(num_pos_feats);
        dim_t[static_cast<size_t>(i)] = std::pow(temperature, exponent);
    }
    // y_embed / x_embed come from cumsum on unmasked positions → normalized
    // by row/col total → scaled to 2π.
    //   y_embed[i, j] = (i + 1) / H * 2π
    //   x_embed[i, j] = (j + 1) / W * 2π
    // pos_y and pos_x each land at [H*W, num_pos_feats] via sin/cos interleave;
    // final flat layout is [H*W, 2*num_pos_feats] with pos_y first (see the
    // PyTorch cat((pos_y, pos_x), dim=3) order).
    std::vector<float> out(static_cast<size_t>(H * W * hidden_dim));
    for (int64_t i = 0; i < H; ++i) {
        const float y_pos = static_cast<float>(i + 1) / (static_cast<float>(H) + eps) * scale;
        for (int64_t j = 0; j < W; ++j) {
            const float x_pos = static_cast<float>(j + 1) / (static_cast<float>(W) + eps) * scale;
            float * row = out.data() + (i * W + j) * hidden_dim;
            for (int64_t k = 0; k < num_pos_feats; ++k) {
                const float vy = y_pos / dim_t[static_cast<size_t>(k)];
                const float vx = x_pos / dim_t[static_cast<size_t>(k)];
                row[k]                       = (k % 2 == 0) ? std::sin(vy) : std::cos(vy);
                row[num_pos_feats + k]       = (k % 2 == 0) ? std::sin(vx) : std::cos(vx);
            }
        }
    }
    return out;
}

// initial_reference_boxes: sigmoid(transformer.decoder.reference_points.weight).
// The tensor is stored as [4, 10] col-major in GGUF; we read as row-major so
// the [num_queries=10, 4] layout comes out as its transpose. Undo it.
std::vector<float> compute_initial_reference_boxes(const sam3::GgufModel & grounding) {
    // PyTorch stores nn.Embedding.weight as [num_embeddings=10, embedding_dim=4],
    // flat layout [q0f0..q0f3, q1f0..q1f3, ...]. GGUF reverses the printed shape
    // to [4, 10] but the flat data is unchanged, so we can read it as [10, 4]
    // directly.
    const auto raw = get_f32(grounding, "transformer.decoder.reference_points.weight", 4 * 10);
    std::vector<float> refs(10 * 4);
    for (int i = 0; i < 40; ++i) refs[i] = 1.0f / (1.0f + std::exp(-raw[i]));
    return refs;
}

// query_pos_layer_0: sinusoidal encoding of initial_reference_boxes (y,x,w,h
// ordering, 128 feats per coord) → ref_point_head 2-layer MLP (512→256→256).
std::vector<float> compute_query_pos_layer_0(
    const sam3::GgufModel & grounding, const std::vector<float> & ref_boxes
) {
    constexpr int64_t Nq = 10;
    constexpr int64_t H = 256;
    constexpr int64_t half = H / 2;  // 128
    const float scale = 2.0f * static_cast<float>(M_PI);

    std::vector<float> dim_t(static_cast<size_t>(half));
    for (int64_t i = 0; i < half; ++i) {
        const float exponent = 2.0f * std::floor(static_cast<float>(i) / 2.0f) /
                                static_cast<float>(half);
        dim_t[static_cast<size_t>(i)] = std::pow(10000.0f, exponent);
    }
    // Sinusoidal encoding, y,x,w,h concat order — mirrors decoder helper.
    std::vector<float> qs(static_cast<size_t>(Nq * 4 * half));
    const int order[4] = {1, 0, 2, 3};
    for (int64_t q = 0; q < Nq; ++q) {
        for (int slot = 0; slot < 4; ++slot) {
            const float p = ref_boxes[q * 4 + order[slot]] * scale;
            for (int64_t i = 0; i < half; ++i) {
                const float v = p / dim_t[static_cast<size_t>(i)];
                qs[q * 4 * half + slot * half + i] = (i % 2 == 0) ? std::sin(v) : std::cos(v);
            }
        }
    }
    // ref_point_head: linear (512→256) + ReLU + linear (256→256)
    const auto w1 = get_f32(grounding, "transformer.decoder.ref_point_head.layer1.weight", 512 * 256);
    const auto b1 = get_f32(grounding, "transformer.decoder.ref_point_head.layer1.bias",   256);
    const auto w2 = get_f32(grounding, "transformer.decoder.ref_point_head.layer2.weight", 256 * 256);
    const auto b2 = get_f32(grounding, "transformer.decoder.ref_point_head.layer2.bias",   256);

    // Linear + ReLU (layer 1)
    std::vector<float> h1(Nq * 256);
    for (int64_t q = 0; q < Nq; ++q) {
        for (int64_t o = 0; o < 256; ++o) {
            float s = b1[o];
            for (int64_t k = 0; k < 512; ++k) s += w1[o * 512 + k] * qs[q * 512 + k];
            h1[q * 256 + o] = std::max(0.0f, s);
        }
    }
    // Linear (layer 2, no activation)
    std::vector<float> out(Nq * 256);
    for (int64_t q = 0; q < Nq; ++q) {
        for (int64_t o = 0; o < 256; ++o) {
            float s = b2[o];
            for (int64_t k = 0; k < 256; ++k) s += w2[o * 256 + k] * h1[q * 256 + k];
            out[q * 256 + o] = s;
        }
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string image_path = "/home/thorax/Downloads/warehouse_rgb.jpg";
    std::string query = "Please segment the objects in the image.";
    std::string grounding_gguf = "/tmp/pathA_gguf/instructsam-grounding-f16.gguf";
    std::string lm_gguf = "/tmp/claude-1001/-home-thorax/e5355e82-8c80-4141-8828-424676e4ee8f/scratchpad/instructsam-as-qwen3vl/instructsam-lm-f16.gguf";
    std::string mmproj = "/tmp/claude-1001/-home-thorax/e5355e82-8c80-4141-8828-424676e4ee8f/scratchpad/instructsam-as-qwen3vl/instructsam-mmproj-f16.gguf";
    std::string ref_dir = "/tmp/pathA_reference/warehouse_rgb";
    std::string out_dir = "/tmp/pathA_reference/warehouse_rgb";
    std::string mask_queries_path = "/tmp/pathA_reference/instructsam_mask_queries.f32";
    bool use_cached_vision = true;
    int32_t max_new_tokens = 128;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--image" && i+1 < argc) image_path = argv[++i];
        else if (a == "--query" && i+1 < argc) query = argv[++i];
        else if (a == "--grounding" && i+1 < argc) grounding_gguf = argv[++i];
        else if (a == "--lm" && i+1 < argc) lm_gguf = argv[++i];
        else if (a == "--mmproj" && i+1 < argc) mmproj = argv[++i];
        else if (a == "--ref-dir" && i+1 < argc) ref_dir = argv[++i];
        else if (a == "--out-dir" && i+1 < argc) out_dir = argv[++i];
        else if (a == "--mask-queries" && i+1 < argc) mask_queries_path = argv[++i];
        else if (a == "--use-cached-vision") use_cached_vision = true;
        else if (a == "--run-vision") use_cached_vision = false;
        else if (a == "--max-new-tokens" && i+1 < argc) max_new_tokens = std::atoi(argv[++i]);
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "instructsam — natural-language image segmentation\n"
                "\n"
                "  --image PATH             input JPEG/PNG (default: warehouse_rgb.jpg)\n"
                "  --query STRING           natural-language request (default: \"Please\n"
                "                           segment the objects in the image.\"). When\n"
                "                           the LM interprets the query as segmentation\n"
                "                           it emits <|object_ref_start|>PHRASE<|object_ref_end|>\n"
                "                           markers around each target; we extract those\n"
                "                           phrases and produce a binary mask per phrase.\n"
                "                           Non-segmentation queries just print the LM's\n"
                "                           text response.\n"
                "  --grounding PATH         sam3cpp grounding GGUF\n"
                "  --lm PATH                Qwen3-VL LM GGUF (F16 or Q4_K_M)\n"
                "  --mmproj PATH            mmproj GGUF (vision→LM embedding)\n"
                "  --mask-queries PATH      instructsam_mask_queries.f32 sidecar\n"
                "                           ([10, 2048] BIN1 blob — extracted from the\n"
                "                           InstructSAM checkpoint via\n"
                "                           tools/extract_mask_queries.py)\n"
                "  --run-vision             recompute vision natively (~1 min on 12-core\n"
                "                           CPU). Default is --use-cached-vision which\n"
                "                           only works for the warehouse_rgb.jpg reference.\n"
                "  --out-dir PATH           output directory (mkdir'd if missing)\n"
                "  --max-new-tokens INT     LM AR decode budget (default 128)\n"
                "\n"
                "Outputs (in --out-dir):\n"
                "  pred_masks.f32           raw [N, 10, 288, 288] mask logits\n"
                "  mask_<i>_<phrase>.png    per-phrase binary mask, upsampled to input size\n"
                "  overlay.png              input | tinted overlay panel with legend\n";
            return 0;
        }
    }

    // Phrases are discovered by the LM (see stage 3 below), not passed in.
    std::vector<std::string> phrases;
    std::vector<std::vector<int32_t>> phrase_tokens;  // LM-emitted subtokens per phrase

    std::cout << "=== instructsam ===\n"
              << "  image     : " << image_path << "\n"
              << "  query     : " << query << "\n\n";

    // ── Load all models ──────────────────────────────────────────────────
    std::cout << "=== stage 0: load models ===\n" << std::flush;
    auto tload = std::chrono::steady_clock::now();
    sam3::GgufModel grounding, lm;
    if (!grounding.load(grounding_gguf)) { std::cerr << "grounding load failed\n"; return 2; }
    if (!lm.load(lm_gguf, false, {}, true)) { std::cerr << "lm load failed\n"; return 3; }
    sam3::InstructsamLmForward       lm_fwd(lm);
    sam3::InstructsamMaskHiddenFcs   mask_bridge(grounding);
    sam3::InstructsamTextHiddenFcs   text_bridge(grounding);
    sam3::InstructsamDetrEncoder     enc(grounding);
    sam3::InstructsamDecoder         dec(grounding);
    sam3::InstructsamPromptCrossAttn pca(grounding);
    sam3::InstructsamPixelDecoder    fpn(grounding);
    sam3::InstructsamMaskTail        tail(grounding);
    auto tokz = sam3::InstructsamTokenizer::load(lm_gguf);
    // (InstructsamMmproj — our thin wrapper around libmtmd — is no longer
    // needed: stage 3 goes through mtmd_helper_eval_chunks directly to get
    // correct M-RoPE positions for image tokens.)
    const auto oln_w = get_f32(grounding, "transformer.decoder.output_layer_norm.weight", 256);
    const auto oln_b = get_f32(grounding, "transformer.decoder.output_layer_norm.bias",   256);
    std::cout << "  ✓ all models loaded ("
              << std::chrono::duration_cast<std::chrono::seconds>(
                     std::chrono::steady_clock::now() - tload).count() << "s)\n";

    // ── Preprocess image (SAM3 path: 1008x1008, [-1,1]) ─────────────────
    std::cout << "\n=== stage 1: preprocess image ===\n" << std::flush;
    const auto pv = sam3::InstructsamPreprocess::load_image_pixel_values(image_path);
    std::cout << "  ✓ pixel_values [3, 1008, 1008] loaded\n";

    // ── Vision: SAM3 encoder trunk + FPN neck ────────────────────────────
    std::vector<float> bb0, bb1, bb2;
    if (use_cached_vision) {
        std::cout << "\n=== stage 2: vision (using cached backbone_features) ===\n";
        std::vector<int64_t> s0, s1, sflat;
        bb0 = read_bin_f32(ref_dir + "/binaries_obj0/md_fpn_bb0.f32", s0);
        bb1 = read_bin_f32(ref_dir + "/binaries_obj0/md_fpn_bb1.f32", s1);
        const auto bb2_hwc = read_bin_f32(ref_dir + "/binaries_obj0/enc_vision_features_flat.f32", sflat);
        // Reshape bb2 to [256, 72, 72] channels-first for FPN input
        bb2.assign(256 * 72 * 72, 0.0f);
        for (int t = 0; t < 5184; ++t)
            for (int c = 0; c < 256; ++c) bb2[c * 5184 + t] = bb2_hwc[t * 256 + c];
    } else {
        std::cout << "\n=== stage 2: vision — full SAM3 encoder (32 layers + FPN neck) ===\n" << std::flush;
        sam3::InstructsamVisionEncoder vis(grounding);
        auto tvis = std::chrono::steady_clock::now();
        const auto trunk = vis.run_all_layers(pv, {3, 1008, 1008});
        std::cout << "  ✓ trunk (32 layers) in "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - tvis).count() << "s\n" << std::flush;
        auto tneck = std::chrono::steady_clock::now();
        const auto fpn = vis.run_neck(trunk);
        std::cout << "  ✓ FPN neck (3 levels) in "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - tneck).count() << "s\n";
        bb0 = std::move(fpn.bb0);
        bb1 = std::move(fpn.bb1);
        bb2 = std::move(fpn.bb2);
    }

    // ── LM stage: prefill via mtmd (correct M-RoPE) + AR decode via llama ──
    //
    // We used to hand-roll prefill with 1-D sequential positions through our
    // native InstructsamLmForward. That was wrong for Qwen3-VL, which uses
    // M-RoPE — image tokens need 3-D (t,h,w) coordinates matching the mmproj
    // spatial grid. `mtmd_helper_get_n_pos` docs it plainly: "normally,
    // n_pos is equal to n_tokens, but for M-RoPE it is different". Symptom
    // was the LM routing to describe/OCR mode instead of emitting
    // <|object_ref_start|> markers, despite correctly identifying objects.
    //
    // libmtmd + llama's mtmd_helper_eval_chunks bakes in the correct M-RoPE
    // layout. We keep lm_fwd for its embed_for_token utility only.
    std::vector<std::vector<float>> seg_outs;
    std::string gen_text;
    {
        std::cout << "\n=== stage 3: LM (mtmd prefill + llama AR decode) ===\n" << std::flush;
        const auto & sp = tokz->specials();

        // --- Load LM model + create context with embeddings=true so we can
        //     read per-token hidden states for the seg-output injection.
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model * lm_model = llama_model_load_from_file(lm_gguf.c_str(), mp);
        if (!lm_model) throw std::runtime_error("llama_model_load_from_file failed: " + lm_gguf);

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = 4096;
        cp.n_batch         = 1024;
        cp.n_ubatch        = 1024;
        cp.n_seq_max       = 2;                          // seq 0 for AR, seq 1 for per-phrase seg injection
        cp.embeddings      = true;                       // enable hidden-state readout
        cp.pooling_type    = LLAMA_POOLING_TYPE_NONE;    // per-token, not pooled
        cp.no_perf         = true;
        cp.n_threads       = 12;
        cp.n_threads_batch = 12;
        llama_context * lctx = llama_init_from_model(lm_model, cp);
        if (!lctx) throw std::runtime_error("llama_init_from_model failed");

        const int32_t H = llama_model_n_embd(lm_model);
        const int32_t V = llama_vocab_n_tokens(llama_model_get_vocab(lm_model));

        // --- Load mtmd (vision projector) tied to lm_model.
        mtmd_context_params mprm = mtmd_context_params_default();
        mprm.use_gpu     = false;
        mprm.print_timings = false;
        mprm.n_threads   = 12;
        mprm.warmup      = false;
        mtmd_context * mctx = mtmd_init_from_file(mmproj.c_str(), lm_model, mprm);
        if (!mctx) throw std::runtime_error("mtmd_init_from_file failed: " + mmproj);

        // --- Build the prompt with the media marker mtmd wants (default
        //     "<__media__>"). Match the InstructSAM chat template — no
        //     system message.
        const std::string prompt =
            "<|im_start|>user\n" + std::string(mtmd_default_marker()) + query +
            "<|im_end|>\n<|im_start|>assistant\n";
        mtmd_input_text txt;
        txt.text          = prompt.c_str();
        txt.text_len      = static_cast<size_t>(prompt.size());
        txt.add_special   = false;
        txt.parse_special = true;

        // --- Load bitmap, tokenize, prefill.
        auto bmw = mtmd_helper_bitmap_init_from_file(mctx, image_path.c_str(), false);
        if (!bmw.bitmap) throw std::runtime_error("bitmap init failed: " + image_path);
        const mtmd_bitmap * bitmap_arr[] = { bmw.bitmap };

        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        int32_t rc = mtmd_tokenize(mctx, chunks, &txt, bitmap_arr, 1);
        if (rc != 0) throw std::runtime_error("mtmd_tokenize rc=" + std::to_string(rc));

        auto tpref = std::chrono::steady_clock::now();
        llama_pos n_past = 0;
        rc = mtmd_helper_eval_chunks(mctx, lctx, chunks, /*n_past=*/0, /*seq_id=*/0,
                                     cp.n_batch, /*logits_last=*/true, &n_past);
        if (rc != 0) throw std::runtime_error("mtmd_helper_eval_chunks rc=" + std::to_string(rc));
        const llama_pos n_past_prefill = n_past;
        std::cout << "  ✓ mtmd prefill (n_past=" << n_past_prefill << ") in "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - tpref).count() << "s\n";

        // --- AR greedy decode via llama_decode.
        std::vector<int32_t> gen_tokens;
        llama_batch bat = llama_batch_init(/*n_tokens=*/1, /*embd=*/0, /*n_seq_max=*/1);
        auto tgen = std::chrono::steady_clock::now();
        for (int step = 0; step < max_new_tokens; ++step) {
            const float * logits = llama_get_logits_ith(lctx, -1);
            if (!logits) throw std::runtime_error("logits null at step " + std::to_string(step));
            int32_t best = 0; float bv = logits[0];
            for (int32_t v = 1; v < V; ++v)
                if (logits[v] > bv) { bv = logits[v]; best = v; }
            gen_tokens.push_back(best);
            if (best == sp.im_end || best == sp.eos) break;

            bat.n_tokens          = 1;
            bat.token[0]          = best;
            bat.pos[0]            = n_past++;
            bat.n_seq_id[0]       = 1;
            bat.seq_id[0][0]      = 0;
            bat.logits[0]         = 1;
            if (llama_decode(lctx, bat) != 0)
                throw std::runtime_error("llama_decode failed at step " + std::to_string(step));
        }
        std::cout << "  ✓ AR decode " << gen_tokens.size() << " tokens in "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - tgen).count() << "s\n";
        llama_batch_free(bat);
        gen_text = tokz->detokenize(gen_tokens);

        // --- Extract phrases between object_ref_start / object_ref_end.
        // Also record each phrase's ref_end position (in the seq 0 KV) so
        // we can branch and inject there for per-phrase seg extraction.
        std::vector<llama_pos> phrase_ref_end_pos;
        for (size_t i = 0; i < gen_tokens.size(); ++i) {
            if (gen_tokens[i] == sp.object_ref_start) {
                size_t j = i + 1;
                while (j < gen_tokens.size() && gen_tokens[j] != sp.object_ref_end) ++j;
                if (j < gen_tokens.size()) {
                    std::vector<int32_t> mid(gen_tokens.begin() + i + 1,
                                             gen_tokens.begin() + j);
                    phrases.push_back(tokz->detokenize(mid));
                    phrase_tokens.push_back(mid);
                    // gen_tokens[j] = ref_end lives at KV position n_past_prefill + j
                    phrase_ref_end_pos.push_back(n_past_prefill + static_cast<llama_pos>(j));
                }
                i = j;
            }
        }
        std::cout << "  extracted " << phrases.size() << " phrase(s)";
        if (!phrases.empty()) {
            std::cout << ": ";
            for (auto & p : phrases) std::cout << "\"" << p << "\" ";
        }
        std::cout << "\n";

        if (phrases.empty()) {
            std::cout << "\n=== LM response ===\n" << gen_text << "\n\n"
                         "(no <|object_ref_start|> markers in output — the LM did not\n"
                         "interpret this query as a segmentation task. No masks written.)\n";
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bmw.bitmap);
            mtmd_free(mctx);
            llama_free(lctx);
            llama_model_free(lm_model);
            return 0;
        }

        // --- Per-phrase seg-output extraction via KV branching + embd batch.
        //
        // Strategy: seq 0 already holds prefill + all AR-generated tokens.
        // For each phrase, branch seq 0 → seq 1 (llama_memory_seq_cp only
        // supports full-buffer copies), then trim seq 1 back to just past
        // the phrase's ref_end. Push a 12-embd batch on seq 1:
        //   [mask_start, mask_queries×10, mask_end]
        // starting at position (ref_end_pos + 1). Read the 10 mask_query
        // hidden states via llama_get_embeddings_ith — these are the
        // seg_output_embeddings that feed mask_hidden_fcs → DETR.
        std::vector<int64_t> mq_s;
        const auto mask_queries = read_bin_f32(mask_queries_path, mq_s);
        if (mask_queries.size() != static_cast<size_t>(10 * H)) {
            throw std::runtime_error("mask_queries size mismatch: got " +
                std::to_string(mask_queries.size()) + " expected " + std::to_string(10 * H));
        }
        const auto mask_start_embed = lm_fwd.embed_for_token(sp.mask_start);
        const auto mask_end_embed   = lm_fwd.embed_for_token(sp.mask_end);

        llama_memory_t mem = llama_get_memory(lctx);
        for (size_t pi = 0; pi < phrases.size(); ++pi) {
            auto tphr = std::chrono::steady_clock::now();
            const llama_pos ref_end_pos = phrase_ref_end_pos[pi];

            // Branch seq 0 → seq 1 (full copy), then trim tail past ref_end.
            llama_memory_seq_rm(mem, /*seq_id=*/1, /*p0=*/0, /*p1=*/-1);
            llama_memory_seq_cp(mem, /*src=*/0, /*dst=*/1, /*p0=*/0, /*p1=*/-1);
            llama_memory_seq_rm(mem, /*seq_id=*/1, /*p0=*/ref_end_pos + 1, /*p1=*/-1);

            // 12-slot embd batch: [mask_start, mask_queries×10, mask_end].
            const int32_t n_inject = 12;
            llama_batch pb = llama_batch_init(n_inject, H, 1);
            std::memcpy(pb.embd + 0 * H, mask_start_embed.data(), H * sizeof(float));
            for (int j = 0; j < 10; ++j) {
                std::memcpy(pb.embd + (1 + j) * H, mask_queries.data() + j * H,
                            H * sizeof(float));
            }
            std::memcpy(pb.embd + 11 * H, mask_end_embed.data(), H * sizeof(float));

            for (int i = 0; i < n_inject; ++i) {
                pb.pos[i]       = ref_end_pos + 1 + i;
                pb.n_seq_id[i]  = 1;
                pb.seq_id[i][0] = 1;
                pb.logits[i]    = (i >= 1 && i <= 10) ? 1 : 0;   // mask_queries only
            }
            pb.n_tokens = n_inject;

            if (llama_decode(lctx, pb) != 0)
                throw std::runtime_error("llama_decode (seg inject) failed for phrase " +
                                          std::to_string(pi));

            std::vector<float> seg_out(10 * H);
            for (int j = 0; j < 10; ++j) {
                const float * hs = llama_get_embeddings_ith(lctx, j);
                if (!hs) throw std::runtime_error("seg embed null slot " +
                                                   std::to_string(j) + " phrase " +
                                                   std::to_string(pi));
                std::memcpy(seg_out.data() + j * H, hs, H * sizeof(float));
            }
            seg_outs.push_back(std::move(seg_out));
            llama_batch_free(pb);

            std::cout << "  ✓ seg_output for \"" << phrases[pi] << "\" ("
                      << std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - tphr).count() << "ms)\n";
        }

        // Cleanup mtmd/llama — the rest of the pipeline uses lm_fwd (embed
        // table only) and downstream native tensors.
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bmw.bitmap);
        mtmd_free(mctx);
        llama_free(lctx);
        llama_model_free(lm_model);
    }

    // ── DETR chain per object ──────────────────────────────────────────
    // text_features / prompt_features / text_mask are computed natively via
    // text_hidden_fcs from phrase tokens (see below). The remaining ref-loads
    // (vision_pos, init_refs, qpos_L0) are per-vision-shape/per-decoder
    // constants that don't depend on the LM output — they'll come from the
    // grounding GGUF once wired directly.
    constexpr int64_t kPhraseH = 2048;
    constexpr int64_t kPhraseMax = 32;
    const auto pad_embed = lm_fwd.embed_for_token(0);  // padding = token id 0

    // Compute per-image detr constants once (all objects share them).
    const auto vision_pos = compute_vision_pos_2d_sincos(72, 72, 256);
    const auto init_refs  = compute_initial_reference_boxes(grounding);
    const auto qpos_L0    = compute_query_pos_layer_0(grounding, init_refs);
    const std::vector<int64_t> vps = {5184, 256};
    const std::vector<int64_t> rps = {10, 4};
    const std::vector<int64_t> qps = {10, 256};
    std::cout << "\n  detr constants computed natively:\n"
              << "    vision_pos [5184, 256] first=[" << vision_pos[0] << ", "
                  << vision_pos[1] << ", " << vision_pos[2] << "]\n"
              << "    init_refs  [10, 4]   first=[" << init_refs[0] << ", "
                  << init_refs[1] << ", " << init_refs[2] << "]\n"
              << "    qpos_L0    [10, 256] first=[" << qpos_L0[0] << ", "
                  << qpos_L0[1] << ", " << qpos_L0[2] << "]\n";

    // Scoring head — one call per phrase gives us 10 real cls_scores so we
    // can pick each phrase's best mask slot properly instead of falling back
    // to peak-mask-sigmoid.
    const auto dp_w = load_scoring_weights(grounding);

    std::cout << "\n=== stage 4: DETR chain per object ===\n";
    std::vector<std::vector<float>> per_obj_masks;
    std::vector<std::vector<float>> per_obj_cls_scores;   // sigmoid-space, [N][10]
    for (size_t pi = 0; pi < phrases.size(); ++pi) {
        const std::string odir = ref_dir + "/binaries_obj" + std::to_string(pi);
        (void)odir;  // no per-obj ref files needed anymore

        // Native: build padded phrase embed tensor, project via text_hidden_fcs.
        // Phrase tokens came directly from the LM's own AR output, so BPE
        // matches the standalone-context tokens InstructSAM emits between
        // <|object_ref_start|> and <|object_ref_end|>.
        std::vector<int32_t> phrase_ids = {151646};  // <|object_ref_start|>
        for (int32_t t : phrase_tokens[pi]) phrase_ids.push_back(t);
        phrase_ids.push_back(151647);  // <|object_ref_end|>
        const int64_t n_valid = static_cast<int64_t>(phrase_ids.size());
        std::vector<float> phrase_padded(kPhraseMax * kPhraseH, 0.0f);
        for (int64_t i = 0; i < n_valid; ++i) {
            const auto e = lm_fwd.embed_for_token(phrase_ids[static_cast<size_t>(i)]);
            std::memcpy(phrase_padded.data() + i * kPhraseH, e.data(), kPhraseH * sizeof(float));
        }
        for (int64_t i = n_valid; i < kPhraseMax; ++i) {
            std::memcpy(phrase_padded.data() + i * kPhraseH, pad_embed.data(),
                        kPhraseH * sizeof(float));
        }
        const auto txt_out = text_bridge.run(phrase_padded, {kPhraseMax, kPhraseH});
        const auto & text_features = txt_out.data;          // [32, 256]
        const auto & prompt_feats  = txt_out.data;          // identical (see obj-diff check)
        std::vector<float> text_mask(kPhraseMax, 0.0f);
        for (int64_t i = 0; i < n_valid; ++i) text_mask[static_cast<size_t>(i)] = 1.0f;
        std::vector<int64_t> tfs = {kPhraseMax, 256};
        std::vector<int64_t> pfs = {kPhraseMax, 256};
        std::vector<int64_t> tms = {kPhraseMax};

        // 1) mask_hidden_fcs bridge on our LM-computed seg_output
        std::vector<int64_t> seg_shape = {10, 2048};
        const auto queries_out = mask_bridge.run(seg_outs[pi], seg_shape);

        // 2) DETR encoder — feed vision features shape [5184, 256]
        std::vector<float> vision_features(5184 * 256);
        for (int c = 0; c < 256; ++c)
            for (int t = 0; t < 5184; ++t)
                vision_features[t * 256 + c] = bb2[c * 5184 + t];
        std::vector<int64_t> vfs = {5184, 256};

        const auto enc_out = enc.run(vision_features, vfs, vision_pos, vps,
                                     text_features, tfs, text_mask, tms);

        // 3) DETR decoder
        const auto dec_out = dec.run(queries_out.data, {10, 256}, text_features, tfs,
            enc_out.last_hidden_state, {enc_out.vision_seq, enc_out.hidden_dim},
            vision_pos, vps, text_mask, tms, qpos_L0, init_refs);
        const auto decoder_queries = cpu_layer_norm(dec_out.hs.back(), 10, 256, oln_w, oln_b);

        // 4) prompt_cross_attn
        const auto post_encoder = pca.run(
            enc_out.last_hidden_state, {enc_out.vision_seq, enc_out.hidden_dim},
            prompt_feats, pfs, text_mask, tms);
        std::vector<float> bb2_local(256 * 72 * 72);
        for (int tk = 0; tk < 5184; ++tk)
            for (int c = 0; c < 256; ++c) bb2_local[c * 5184 + tk] = post_encoder[tk * 256 + c];

        // 5) FPN + mask_tail
        const auto pixel_embed = fpn.run(bb0, {256, 288, 288}, bb1, {256, 144, 144},
                                         bb2_local, {256, 72, 72});
        const auto out = tail.run(pixel_embed, {256, 288, 288}, decoder_queries, {10, 256});

        // Real cls_score via Sam3DotProductScoring on the last-layer
        // decoder queries + phrase text_features.
        const auto pred_logits = dot_prod_scoring(dp_w, text_features, text_mask, decoder_queries);
        std::vector<float> cls_score(10);
        for (int s = 0; s < 10; ++s) cls_score[s] = 1.0f / (1.0f + std::exp(-pred_logits[s]));
        per_obj_cls_scores.push_back(cls_score);

        per_obj_masks.push_back(out.pred_masks);
        std::cout << "  ✓ obj " << pi << " (\"" << phrases[pi] << "\"): pred_masks ["
                  << (out.pred_masks.size() / (288*288)) << ", 288, 288]"
                  << "  cls_score top=[";
        int top = 0; float tv = cls_score[0];
        for (int s = 1; s < 10; ++s) if (cls_score[s] > tv) { tv = cls_score[s]; top = s; }
        std::cout << "slot " << top << " → " << tv << "]\n";
    }

    // ── Write raw f32 blob + per-object PNGs + overlay panel ────────────
    std::vector<float> all_masks;
    for (auto & m : per_obj_masks) all_masks.insert(all_masks.end(), m.begin(), m.end());
    const int64_t N = static_cast<int64_t>(phrases.size());
    write_bin_f32(out_dir + "/pred_masks.f32", all_masks, {N, 10, 288, 288});
    std::cout << "\n  ✓ wrote " << out_dir << "/pred_masks.f32  [" << N << ", 10, 288, 288]\n";

    // Load the input image for output-size masks + overlay panel.
    int in_w = 0, in_h = 0, in_comp = 0;
    unsigned char * in_rgb = stbi_load(image_path.c_str(), &in_w, &in_h, &in_comp, 3);
    if (!in_rgb) {
        std::cerr << "  ✗ stbi_load failed for " << image_path << " — skipping PNG output\n";
        std::cout << "\n✓ CLI complete\n";
        return 0;
    }

    // Pick each phrase's best mask slot by argmax over cls_score (sigmoid
    // of pred_logits from Sam3DotProductScoring, computed above in stage 4).
    std::vector<std::vector<uint8_t>> per_obj_binary;
    per_obj_binary.reserve(phrases.size());
    for (size_t pi = 0; pi < phrases.size(); ++pi) {
        const float * m10 = per_obj_masks[pi].data();
        const auto & cls = per_obj_cls_scores[pi];
        int best_slot = 0; float best_conf = cls[0];
        for (int s = 1; s < 10; ++s)
            if (cls[s] > best_conf) { best_conf = cls[s]; best_slot = s; }

        std::vector<float> slot_logits(m10 + best_slot * 288 * 288,
                                       m10 + (best_slot + 1) * 288 * 288);
        auto binary = mask_logits_to_binary(slot_logits, in_h, in_w);

        // File-safe label — replace spaces/slashes so "pallet jack" → "pallet_jack"
        std::string label_safe = phrases[pi];
        for (char & c : label_safe) if (c == ' ' || c == '/' || c == '\\') c = '_';
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s/mask_%02zu_%s.png",
                      out_dir.c_str(), pi, label_safe.c_str());
        stbi_write_png(buf, in_w, in_h, 1, binary.data(), in_w);
        std::cout << "  ✓ wrote " << buf << "  (slot " << best_slot
                  << ", cls_score " << best_conf << ")\n";
        per_obj_binary.push_back(std::move(binary));
    }

    // Overlay panel: [orig | tinted composite].
    const std::string overlay_path = out_dir + "/overlay.png";
    write_overlay_panel(overlay_path, in_rgb, in_w, in_h, per_obj_binary, phrases);
    std::cout << "  ✓ wrote " << overlay_path << "  ("
              << (in_w * 2) << "×" << in_h << ")\n";
    std::cout << "  legend (color → phrase):\n";
    for (size_t k = 0; k < phrases.size() && k < 12; ++k) {
        std::cout << "    ["
                  << (int)kPalette[k % 12][0] << "," << (int)kPalette[k % 12][1]
                  << "," << (int)kPalette[k % 12][2] << "] → " << phrases[k] << "\n";
    }
    stbi_image_free(in_rgb);

    std::cout << "\n✓ CLI complete\n";
    return 0;
}
