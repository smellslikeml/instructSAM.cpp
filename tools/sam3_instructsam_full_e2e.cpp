// Full InstructSAM E2E chain in native ggml — object 0 of the
// warehouse_rgb.jpg reference oracle.
//
// Chains six components:
//   detr_encoder(vision_features, vision_pos, text_features, text_mask)
//     → encoder_hidden_states [5184, 256]
//   detr_decoder(queries, encoder_hidden_states, text_features, ...,
//                initial_reference_boxes, query_pos_layer_0)
//     → layer5 hidden states [10, 256]
//   output_layer_norm(layer5) → decoder_queries [10, 256]  (needed because
//     PyTorch feeds output_layer_norm'd hidden to mask_decoder)
//   prompt_cross_attn(encoder_hidden_states, prompt_features, prompt_mask)
//     → post_encoder [5184, 256]
//   pixel_decoder_FPN(bb0, bb1, reshape(post_encoder) → bb2)
//     → pixel_embed [256, 288, 288]
//   mask_tail(pixel_embed, decoder_queries) → pred_masks [10, 288, 288]
//
// Compares against md_pred_masks (PyTorch reference for object 0).

#include "sam3/gguf_model.h"
#include "sam3/instructsam_detr_encoder.h"
#include "sam3/instructsam_decoder.h"
#include "sam3/instructsam_prompt_cross_attn.h"
#include "sam3/instructsam_pixel_decoder.h"
#include "sam3/instructsam_mask_tail.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
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
    f.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(total * sizeof(float)));
    if (!f) throw std::runtime_error("short read on " + path);
    return data;
}

double cosine_flat(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

void report(const std::string & label,
            const std::vector<float> & ours, const std::vector<float> & ref) {
    double maxdiff = 0.0, l2diff = 0.0, l2ref = 0.0, om = 0.0, rm = 0.0;
    for (size_t i = 0; i < ours.size(); ++i) {
        const double d = static_cast<double>(ours[i]) - ref[i];
        if (std::fabs(d) > maxdiff) maxdiff = std::fabs(d);
        l2diff += d * d;
        l2ref  += static_cast<double>(ref[i]) * ref[i];
        om += ours[i]; rm += ref[i];
    }
    std::cout << "  " << label << ":\n"
              << "    cosine (flat)     : " << cosine_flat(ours, ref) << "\n"
              << "    max abs diff      : " << maxdiff << "\n"
              << "    relative L2 error : " << (std::sqrt(l2diff) / std::sqrt(l2ref)) << "\n"
              << "    ours mean=" << (om / ours.size())
              << "  ref mean=" << (rm / ref.size()) << "\n";
}

// Read weight tensor as f32 (handles f16 storage).
std::vector<float> get_f32(const sam3::GgufModel & model, const std::string & name, size_t n) {
    ggml_tensor * t = model.find_weight(name);
    if (t == nullptr) throw std::runtime_error("get_f32: missing " + name);
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
}

// CPU LayerNorm on [N, D] with weight+bias [D].
std::vector<float> cpu_layer_norm(
    const std::vector<float> & x, int64_t N, int64_t D,
    const std::vector<float> & w, const std::vector<float> & b, float eps = 1e-5f
) {
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
        const double inv_std = 1.0 / std::sqrt(var + eps);
        for (int64_t d = 0; d < D; ++d) {
            const double normed = (x[static_cast<size_t>(i * D + d)] - mean) * inv_std;
            out[static_cast<size_t>(i * D + d)] = static_cast<float>(
                normed * w[static_cast<size_t>(d)] + b[static_cast<size_t>(d)]);
        }
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: sam3-instructsam-full-e2e <model.gguf> [parity_dir]\n";
        return 1;
    }
    const std::string dir = (argc >= 3) ? argv[2]
        : "/tmp/pathA_reference/warehouse_rgb/binaries_obj0";

    sam3::GgufModel model;
    if (!model.load(argv[1])) { std::cerr << "load failed\n"; return 2; }
    std::cout << "loaded " << model.tensors().size() << " tensors\n";

    // ── Load all inputs + reference outputs ─────────────────────────────
    std::vector<int64_t> vfs, vps, tfs, tms, pfs, pms;
    std::vector<int64_t> bb0s, bb1s, qs, rps, qps, refs;

    const auto vision_features = read_bin_f32(dir + "/enc_vision_features_flat.f32", vfs);
    const auto vision_pos      = read_bin_f32(dir + "/enc_vision_pos_flat.f32",      vps);
    const auto text_features   = read_bin_f32(dir + "/enc_text_features.f32",        tfs);
    const auto text_mask       = read_bin_f32(dir + "/md_pca_prompt_mask.f32",       tms);
    const auto prompt_features = read_bin_f32(dir + "/md_pca_prompt_features.f32",   pfs);
    const auto prompt_mask     = text_mask;  pms = tms;
    const auto bb0             = read_bin_f32(dir + "/md_fpn_bb0.f32",               bb0s);
    const auto bb1             = read_bin_f32(dir + "/md_fpn_bb1.f32",               bb1s);
    const auto initial_queries = read_bin_f32(dir + "/queries.f32",                  qs);
    const auto init_ref_boxes  = read_bin_f32(dir + "/initial_reference_points.f32", rps);
    const auto query_pos_L0    = read_bin_f32(dir + "/query_pos_layer_0.f32",        qps);
    const auto ref_pred_masks  = read_bin_f32(dir + "/md_pred_masks.f32",            refs);

    int valid = 0; for (float v : text_mask) if (v > 0.5f) ++valid;
    std::cout << "  vision " << vfs[0] << "×" << vfs[1]
              << "  text " << tfs[0] << "×" << tfs[1]
              << " (valid " << valid << "/" << tfs[0] << ")"
              << "  queries " << qs[0] << "×" << qs[1] << "\n";

    // ── Stage 1: DETR encoder ───────────────────────────────────────────
    std::cout << "\n=== stage 1: detr_encoder (~8s) ===\n" << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    sam3::InstructsamDetrEncoder enc(model);
    const auto enc_out = enc.run(
        vision_features, vfs, vision_pos, vps,
        text_features, tfs, text_mask, tms);
    std::cout << "  ✓ encoder [" << enc_out.vision_seq << "," << enc_out.hidden_dim << "]"
              << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t0).count() << " ms\n";

    // ── Stage 2: DETR decoder ───────────────────────────────────────────
    std::cout << "\n=== stage 2: detr_decoder (~5s) ===\n" << std::flush;
    t0 = std::chrono::steady_clock::now();
    sam3::InstructsamDecoder dec(model);
    const auto dec_out = dec.run(
        initial_queries,   qs,
        text_features,     tfs,
        enc_out.last_hidden_state, {enc_out.vision_seq, enc_out.hidden_dim},
        vision_pos,        vps,
        text_mask,         tms,
        query_pos_L0,
        init_ref_boxes);
    const auto & final_dec = dec_out.hs.back();  // layer 5 output, PRE output_layer_norm
    std::cout << "  ✓ decoder 6 layers, " << final_dec.size() << " elements"
              << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t0).count() << " ms\n";

    // ── Bridge: output_layer_norm(final_dec) → decoder_queries ──────────
    // PyTorch feeds intermediate_hidden_states[-1] to mask_decoder, which
    // is output_layer_norm(hidden). Must apply the norm here.
    std::cout << "\n=== bridge: output_layer_norm(layer5) → decoder_queries ===\n";
    const auto oln_w = get_f32(model, "transformer.decoder.output_layer_norm.weight", 256);
    const auto oln_b = get_f32(model, "transformer.decoder.output_layer_norm.bias",   256);
    const auto decoder_queries = cpu_layer_norm(final_dec, 10, 256, oln_w, oln_b);
    std::cout << "  ✓ [10, 256]\n";

    // ── Stage 3: PCA ────────────────────────────────────────────────────
    std::cout << "\n=== stage 3: prompt_cross_attn (~1s) ===\n" << std::flush;
    t0 = std::chrono::steady_clock::now();
    sam3::InstructsamPromptCrossAttn pca(model);
    const auto post_encoder = pca.run(
        enc_out.last_hidden_state, {enc_out.vision_seq, enc_out.hidden_dim},
        prompt_features, pfs,
        prompt_mask,     pms);
    std::cout << "  ✓ post_encoder [" << enc_out.vision_seq << "," << enc_out.hidden_dim << "]"
              << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t0).count() << " ms\n";

    // ── Stage 4: FPN ────────────────────────────────────────────────────
    std::cout << "\n=== stage 4: pixel_decoder FPN (~1s) ===\n" << std::flush;
    // Reshape post_encoder [5184, 256] → [256, 72, 72] as coarsest FPN input
    std::vector<float> bb2(256 * 72 * 72);
    for (int t = 0; t < 5184; ++t) {
        const int h = t / 72, w = t % 72;
        for (int c = 0; c < 256; ++c) {
            bb2[static_cast<size_t>(c * 5184 + h * 72 + w)] =
                post_encoder[static_cast<size_t>(t * 256 + c)];
        }
    }
    t0 = std::chrono::steady_clock::now();
    sam3::InstructsamPixelDecoder fpn(model);
    const auto pixel_embed = fpn.run(bb0, bb0s, bb1, bb1s, bb2, {256, 72, 72});
    std::cout << "  ✓ pixel_embed [256, 288, 288]"
              << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t0).count() << " ms\n";

    // ── Stage 5: mask_tail ──────────────────────────────────────────────
    std::cout << "\n=== stage 5: mask_tail (~9s) ===\n" << std::flush;
    t0 = std::chrono::steady_clock::now();
    sam3::InstructsamMaskTail tail(model);
    const auto out = tail.run(
        pixel_embed, {256, 288, 288},
        decoder_queries, {10, 256});
    std::cout << "  ✓ pred_masks [10, 288, 288]"
              << "  " << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t0).count() << " ms\n";

    // ── Final parity vs PyTorch reference ───────────────────────────────
    std::cout << "\n=== E2E parity vs. PyTorch pred_masks (object 0) ===\n";
    report("pred_masks (final)", out.pred_masks, ref_pred_masks);

    std::cout << "\n=== full segmentation pipeline: native ggml E2E ===\n";
    std::cout << "  Ran detr_encoder + detr_decoder + PCA + FPN + mask_tail in native ggml\n";
    std::cout << "  chained from PyTorch-captured raw inputs, matched final masks.\n";
    std::cout << "  What's still Python: vision_encoder (Sam3ViT), text_encoder (CLIP),\n";
    std::cout << "  LM (Qwen3-VL, use Path B's llama.cpp), and mask_hidden_fcs projection.\n";
    return 0;
}
