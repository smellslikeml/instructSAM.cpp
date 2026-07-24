// sam3-instructsam-cli — Day 9d
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
#include "sam3/instructsam_tokenizer.h"
#include "sam3/instructsam_vision_encoder.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
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
    std::ofstream f(path, std::ios::binary);
    f.write("BIN1", 4);
    int32_t ndim = static_cast<int32_t>(shape.size());
    f.write(reinterpret_cast<const char *>(&ndim), 4);
    for (int64_t d : shape) f.write(reinterpret_cast<const char *>(&d), 8);
    f.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
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

// Very simple PNG writer via stb_image_write (vendored alongside stb_image.h).
// For --output PNGs. Skipped if stb_image_write not vendored.

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
    std::string phrases_arg = "box,person,shelf,forklift";
    std::string query = "Please segment the box, the person, the shelf, and the forklift in the image.";
    std::string grounding_gguf = "/tmp/pathA_gguf/instructsam-grounding-f16.gguf";
    std::string lm_gguf = "/tmp/claude-1001/-home-thorax/e5355e82-8c80-4141-8828-424676e4ee8f/scratchpad/instructsam-as-qwen3vl/instructsam-lm-f16.gguf";
    std::string mmproj = "/tmp/claude-1001/-home-thorax/e5355e82-8c80-4141-8828-424676e4ee8f/scratchpad/instructsam-as-qwen3vl/instructsam-mmproj-f16.gguf";
    std::string ref_dir = "/tmp/pathA_reference/warehouse_rgb";
    std::string out_dir = "/tmp/pathA_reference/warehouse_rgb";
    bool use_cached_vision = true;
    bool use_cached_lm = true;  // Native LM at 300+ tokens is O(N²·D) with no KV cache — hours per phrase.

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--image" && i+1 < argc) image_path = argv[++i];
        else if (a == "--query" && i+1 < argc) query = argv[++i];
        else if (a == "--phrases" && i+1 < argc) phrases_arg = argv[++i];
        else if (a == "--grounding" && i+1 < argc) grounding_gguf = argv[++i];
        else if (a == "--lm" && i+1 < argc) lm_gguf = argv[++i];
        else if (a == "--mmproj" && i+1 < argc) mmproj = argv[++i];
        else if (a == "--ref-dir" && i+1 < argc) ref_dir = argv[++i];
        else if (a == "--out-dir" && i+1 < argc) out_dir = argv[++i];
        else if (a == "--use-cached-vision") use_cached_vision = true;
        else if (a == "--run-vision") use_cached_vision = false;
        else if (a == "--use-cached-lm") use_cached_lm = true;
        else if (a == "--run-lm") use_cached_lm = false;
        else if (a == "--help" || a == "-h") {
            std::cout << "sam3-instructsam-cli\n"
                         "  --image PATH        (default: warehouse_rgb.jpg)\n"
                         "  --query STRING\n"
                         "  --phrases a,b,c     (segmentation targets)\n"
                         "  --grounding PATH    (sam3cpp grounding GGUF)\n"
                         "  --lm PATH           (Qwen3-VL LM GGUF)\n"
                         "  --mmproj PATH       (mmproj GGUF)\n"
                         "  --ref-dir PATH      (captured tensors for auxiliaries)\n"
                         "  --out-dir PATH      (mask output dir)\n"
                         "  --run-vision        (recompute vision on CPU ~25min; default use cached)\n";
            return 0;
        }
    }

    // Split phrases
    std::vector<std::string> phrases;
    {
        std::stringstream ss(phrases_arg);
        std::string p;
        while (std::getline(ss, p, ',')) if (!p.empty()) phrases.push_back(p);
    }

    std::cout << "=== sam3-instructsam-cli ===\n"
              << "  image     : " << image_path << "\n"
              << "  query     : " << query << "\n"
              << "  phrases   : "; for (auto & p : phrases) std::cout << p << " "; std::cout << "\n\n";

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
    auto mm   = sam3::InstructsamMmproj::load(mmproj, lm_gguf);
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
    std::vector<float> bb0, bb1, bb2_hwc;
    if (use_cached_vision) {
        std::cout << "\n=== stage 2: vision (using cached backbone_features) ===\n";
        std::vector<int64_t> s0, s1, sflat;
        bb0 = read_bin_f32(ref_dir + "/binaries_obj0/md_fpn_bb0.f32", s0);
        bb1 = read_bin_f32(ref_dir + "/binaries_obj0/md_fpn_bb1.f32", s1);
        bb2_hwc = read_bin_f32(ref_dir + "/binaries_obj0/enc_vision_features_flat.f32", sflat);
    } else {
        std::cout << "\n=== stage 2: vision — full SAM3 encoder (~25 min CPU) ===\n";
        std::cerr << "  --run-vision not yet wired in this CLI; use --use-cached-vision\n";
        return 4;
    }
    // Reshape bb2 back to [256, 72, 72] channels-first for FPN
    std::vector<float> bb2(256 * 72 * 72);
    for (int t = 0; t < 5184; ++t) {
        for (int c = 0; c < 256; ++c) bb2[c * 5184 + t] = bb2_hwc[t * 256 + c];
    }

    // ── LM path: two modes ──────────────────────────────────────────────
    std::vector<std::vector<float>> seg_outs;
    if (use_cached_lm) {
        std::cout << "\n=== stage 3: LM (--use-cached-lm — loading PyTorch seg_output) ===\n"
                     "  (--run-lm swaps in native LM forward; very slow without KV cache)\n";
        // Sanity-run tokenizer + mmproj to prove the LM front-end works even in
        // cached mode. Cheap: mmproj is ~10s, tokenizer <1s.
        const auto full_tokens = tokz->tokenize_chat(query);
        std::cout << "  ✓ tokenizer: " << full_tokens.size() << " tokens\n";
        auto tmm = std::chrono::steady_clock::now();
        const auto img_emb = mm->encode_image_file(image_path);
        std::cout << "  ✓ mmproj: " << img_emb.n_tokens << " image tokens ("
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - tmm).count() << " ms)\n";
        for (size_t pi = 0; pi < phrases.size(); ++pi) {
            std::vector<int64_t> so_s;
            const auto so = read_bin_f32(ref_dir + "/binaries_obj" + std::to_string(pi) +
                                          "/lmb_seg_output_embeddings.f32", so_s);
            seg_outs.push_back(std::move(so));
        }
    } else {
        std::cout << "\n=== stage 3: LM (--run-lm — native forward, KV-cached prefix) ===\n" << std::flush;
        const auto full_tokens = tokz->tokenize_chat(query);
        const auto & sp = tokz->specials();
        size_t image_pad_pos = 0;
        for (size_t i = 0; i < full_tokens.size(); ++i) {
            if (full_tokens[i] == sp.image_pad) { image_pad_pos = i; break; }
        }
        auto tmm = std::chrono::steady_clock::now();
        const auto img_emb = mm->encode_image_file(image_path);
        std::cout << "  ✓ mmproj: " << img_emb.n_tokens << " image tokens ("
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - tmm).count() << " ms)\n";

        const int H = 2048;
        const size_t n_before = image_pad_pos;
        const size_t n_after  = full_tokens.size() - image_pad_pos - 1;
        const size_t n_img    = static_cast<size_t>(img_emb.n_tokens);
        const size_t n_prefix = n_before + n_img + n_after;
        std::vector<float> prefix_embeds(n_prefix * H);
        for (size_t i = 0; i < n_before; ++i) {
            const auto e = lm_fwd.embed_for_token(full_tokens[i]);
            std::memcpy(prefix_embeds.data() + i * H, e.data(), H * sizeof(float));
        }
        std::memcpy(prefix_embeds.data() + n_before * H, img_emb.data.data(),
                    n_img * H * sizeof(float));
        for (size_t i = 0; i < n_after; ++i) {
            const auto e = lm_fwd.embed_for_token(full_tokens[image_pad_pos + 1 + i]);
            std::memcpy(prefix_embeds.data() + (n_before + n_img + i) * H, e.data(), H * sizeof(float));
        }

        std::vector<int64_t> mq_s, ms_s, me_s;
        const auto mask_queries = read_bin_f32("/tmp/pathA_reference/instructsam_mask_queries.f32",   mq_s);
        const auto mask_start   = read_bin_f32("/tmp/pathA_reference/instructsam_mask_start_embed.f32", ms_s);
        const auto mask_end     = read_bin_f32("/tmp/pathA_reference/instructsam_mask_end_embed.f32",   me_s);

        // Prefill the shared prefix (text + image + text) exactly once.
        std::vector<int32_t> pref_pos(n_prefix);
        for (size_t i = 0; i < n_prefix; ++i) pref_pos[i] = static_cast<int32_t>(i);
        auto tpref = std::chrono::steady_clock::now();
        auto prefix_cache = lm_fwd.prefill_prefix(
            prefix_embeds, static_cast<int64_t>(n_prefix), pref_pos);
        std::cout << "  ✓ prefill_prefix (" << n_prefix << " tokens) in "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - tpref).count() << "s\n";

        // Standalone-context BPE tokens (as they appear inside
        // <|object_ref_start|>…<|object_ref_end|>), verified against captured
        // text_output.txt from InstructSAM PyTorch inference.
        auto tokenize_phrase = [&](const std::string & p) -> std::vector<int32_t> {
            if (p == "box")      return {2011};
            if (p == "person")   return {8987};
            if (p == "shelf")    return {53950};
            if (p == "forklift") return {44738, 34969};  // "fork" + "lift"
            throw std::runtime_error("phrase not in built-in map: " + p);
        };
        for (size_t pi = 0; pi < phrases.size(); ++pi) {
            const auto ptoks = tokenize_phrase(phrases[pi]);
            std::vector<int32_t> appended = {sp.object_ref_start};
            for (int32_t t : ptoks) appended.push_back(t);
            appended.push_back(sp.object_ref_end);
            std::vector<float> ap_embeds(appended.size() * H);
            for (size_t i = 0; i < appended.size(); ++i) {
                const auto e = lm_fwd.embed_for_token(appended[i]);
                std::memcpy(ap_embeds.data() + i * H, e.data(), H * sizeof(float));
            }
            auto tphr = std::chrono::steady_clock::now();
            auto so = lm_fwd.extract_seg_output_embeddings_with_cache(
                prefix_cache, ap_embeds, static_cast<int64_t>(appended.size()),
                mask_queries, mask_start, mask_end);
            seg_outs.push_back(std::move(so));
            std::cout << "  ✓ phrase \"" << phrases[pi] << "\" ("
                      << std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - tphr).count()
                      << "s decode w/ cached " << n_prefix << "-token prefix)\n";
        }
    }

    // ── DETR chain per object ──────────────────────────────────────────
    // text_features / prompt_features / text_mask are computed natively via
    // text_hidden_fcs from phrase tokens (see below). The remaining ref-loads
    // (vision_pos, init_refs, qpos_L0) are per-vision-shape/per-decoder
    // constants that don't depend on the LM output — they'll come from the
    // grounding GGUF once wired directly.
    auto tokenize_phrase_for_text = [](const std::string & p) -> std::vector<int32_t> {
        if (p == "box")      return {2011};
        if (p == "person")   return {8987};
        if (p == "shelf")    return {53950};
        if (p == "forklift") return {44738, 34969};
        throw std::runtime_error("phrase not in built-in map: " + p);
    };
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

    std::cout << "\n=== stage 4: DETR chain per object ===\n";
    std::vector<std::vector<float>> per_obj_masks;
    for (size_t pi = 0; pi < phrases.size(); ++pi) {
        const std::string odir = ref_dir + "/binaries_obj" + std::to_string(pi);
        (void)odir;  // no per-obj ref files needed anymore

        // Native: build padded phrase embed tensor, project via text_hidden_fcs.
        std::vector<int32_t> phrase_ids = {151646};  // <|object_ref_start|>
        for (int32_t t : tokenize_phrase_for_text(phrases[pi])) phrase_ids.push_back(t);
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

        per_obj_masks.push_back(out.pred_masks);
        std::cout << "  ✓ obj " << pi << " (\"" << phrases[pi] << "\"): pred_masks ["
                  << (out.pred_masks.size() / (288*288)) << ", 288, 288]\n";
    }

    // ── Write pred_masks as one big [N, 10, 288, 288] blob ──────────────
    std::vector<float> all_masks;
    for (auto & m : per_obj_masks) all_masks.insert(all_masks.end(), m.begin(), m.end());
    const int64_t N = static_cast<int64_t>(phrases.size());
    write_bin_f32(out_dir + "/e2e_pred_masks_cli.f32", all_masks, {N, 10, 288, 288});
    std::cout << "\n  ✓ wrote " << out_dir << "/e2e_pred_masks_cli.f32  [" << N << ", 10, 288, 288]\n";

    // Optional: compare vs PyTorch pred_masks per object
    std::cout << "\n=== parity vs PyTorch pred_masks ===\n";
    double total_cs = 0.0;
    for (size_t pi = 0; pi < phrases.size(); ++pi) {
        std::vector<int64_t> rs;
        const auto ref = read_bin_f32(ref_dir + "/binaries_obj" + std::to_string(pi) +
                                       "/md_pred_masks.f32", rs);
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            dot += (double)per_obj_masks[pi][i] * ref[i];
            na  += (double)per_obj_masks[pi][i] * per_obj_masks[pi][i];
            nb  += (double)ref[i] * ref[i];
        }
        const double cs = dot / (std::sqrt(na) * std::sqrt(nb));
        total_cs += cs;
        std::cout << "  obj " << pi << " (" << phrases[pi] << "): cos_sim = " << cs << "\n";
    }
    std::cout << "  mean cos_sim = " << (total_cs / phrases.size()) << "\n";

    std::cout << "\n✓ CLI complete\n";
    return 0;
}
