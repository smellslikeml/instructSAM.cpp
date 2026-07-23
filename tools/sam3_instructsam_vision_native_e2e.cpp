// Vision-native E2E: takes pixel_values, runs our own vision encoder
// (trunk + FPN neck) instead of using PyTorch-captured backbone_features.
// Then loops 4 objects through the existing chain (DETR encoder + decoder +
// PCA + FPN + mask_tail). Everything downstream of pixel_values runs in
// native ggml (except LM outputs which are still precomputed by PyTorch).
//
// Runtime notes on CPU:
//   Vision trunk (32 layers)   ~20 min
//   FPN neck                   ~5 min
//   Per-object chain (×4)      ~66s total
//   ------------------------------------
//   Total per image            ~26 min
//
// Use --use-cached-vision to skip the trunk+neck compute and read backbone
// features from disk (equivalent to batch_e2e), which cuts runtime to ~66s.

#include "sam3/gguf_model.h"
#include "sam3/instructsam_vision_encoder.h"
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

void write_bin_f32(const std::string & path, const std::vector<float> & data,
                   const std::vector<int64_t> & shape) {
    std::ofstream f(path, std::ios::binary);
    f.write("BIN1", 4);
    int32_t ndim = static_cast<int32_t>(shape.size());
    f.write(reinterpret_cast<const char *>(&ndim), 4);
    for (int64_t d : shape) f.write(reinterpret_cast<const char *>(&d), 8);
    f.write(reinterpret_cast<const char *>(data.data()),
        static_cast<std::streamsize>(data.size() * sizeof(float)));
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

std::vector<float> get_f32(const sam3::GgufModel & model, const std::string & name, size_t n) {
    ggml_tensor * t = model.find_weight(name);
    if (t == nullptr) throw std::runtime_error("get_f32 missing " + name);
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
        std::cerr << "usage: sam3-instructsam-vision-native-e2e <model.gguf>\n"
                     "  [vision_dir=/tmp/pathA_reference/warehouse_rgb/binaries_vision]\n"
                     "  [ref_dir=/tmp/pathA_reference/warehouse_rgb]\n"
                     "  [--use-cached-vision]  (skip trunk+neck, load bb0/1/2 from ref)\n"
                     "  [--out=path.f32]       (default: e2e_pred_masks_vision_native.f32)\n";
        return 1;
    }
    const std::string vdir = (argc >= 3 && argv[2][0] != '-') ? argv[2]
        : "/tmp/pathA_reference/warehouse_rgb/binaries_vision";
    const std::string ref_root = (argc >= 4 && argv[3][0] != '-') ? argv[3]
        : "/tmp/pathA_reference/warehouse_rgb";
    bool use_cached_vision = false;
    std::string out_path = "/tmp/pathA_reference/warehouse_rgb/e2e_pred_masks_vision_native.f32";
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--use-cached-vision") use_cached_vision = true;
        else if (arg.rfind("--out=", 0) == 0) out_path = arg.substr(6);
    }

    sam3::GgufModel model;
    if (!model.load(argv[1])) { std::cerr << "load failed\n"; return 2; }
    std::cout << "loaded " << model.tensors().size() << " tensors\n";

    sam3::InstructsamVisionEncoder    vis(model);
    sam3::InstructsamDetrEncoder      enc(model);
    sam3::InstructsamDecoder          dec(model);
    sam3::InstructsamPromptCrossAttn  pca(model);
    sam3::InstructsamPixelDecoder     fpn(model);
    sam3::InstructsamMaskTail         tail(model);

    const auto oln_w = get_f32(model, "transformer.decoder.output_layer_norm.weight", 256);
    const auto oln_b = get_f32(model, "transformer.decoder.output_layer_norm.bias",   256);

    // ── Vision: produce bb0/bb1/bb2 ONCE (image-level, shared across objects) ─
    std::vector<float> bb0, bb1, bb2;
    if (use_cached_vision) {
        std::cout << "\n=== vision (CACHED, loading backbone_features from " << ref_root << "/binaries_obj0/) ===\n";
        std::vector<int64_t> s0, s1, sflat;
        bb0 = read_bin_f32(ref_root + "/binaries_obj0/md_fpn_bb0.f32", s0);
        bb1 = read_bin_f32(ref_root + "/binaries_obj0/md_fpn_bb1.f32", s1);
        // md_fpn_bb2.f32 is post-PCA-encoder reshape (for FPN's coarsest input),
        // NOT the raw FPN bb2 output. For detr_encoder we need the raw FPN bb2.
        // Load enc_vision_features_flat [5184, 256] and reshape to [256, 72, 72].
        const auto vfeat = read_bin_f32(ref_root + "/binaries_obj0/enc_vision_features_flat.f32", sflat);
        bb2.assign(256 * 72 * 72, 0.0f);
        for (int t = 0; t < 5184; ++t) {
            const int h = t / 72, w = t % 72;
            for (int c = 0; c < 256; ++c) {
                bb2[c * 5184 + h * 72 + w] = vfeat[t * 256 + c];
            }
        }
    } else {
        std::cout << "\n=== vision (NATIVE ggml, trunk 32 layers + FPN neck, ~25 min CPU) ===\n" << std::flush;
        std::vector<int64_t> pvs;
        const auto pixel_values = read_bin_f32(vdir + "/pixel_values.f32", pvs);
        std::cout << "  pixel_values [" << pvs[0] << "," << pvs[1] << "," << pvs[2] << "]  running trunk..." << std::flush;
        auto t0 = std::chrono::steady_clock::now();
        const auto trunk = vis.run_all_layers(pixel_values, pvs);
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "  " << std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count() << "s\n" << std::flush;
        std::cout << "  running FPN neck..." << std::flush;
        const auto neck = vis.run_neck(trunk);
        auto t2 = std::chrono::steady_clock::now();
        std::cout << "  " << std::chrono::duration_cast<std::chrono::seconds>(t2-t1).count() << "s\n";
        bb0 = neck.bb0; bb1 = neck.bb1; bb2 = neck.bb2;
    }

    // Aggregated output over 4 objects
    std::vector<float> all_masks(4 * 10 * 288 * 288);
    std::vector<float> all_ref(4 * 10 * 288 * 288);

    auto session_start = std::chrono::steady_clock::now();
    for (int obj = 0; obj < 4; ++obj) {
        const std::string odir = ref_root + "/binaries_obj" + std::to_string(obj);
        std::cout << "\n=== object " << obj << " ===\n" << std::flush;

        std::vector<int64_t> vps, tfs, tms, pfs, pms, qs, rps, qps, refs;
        std::vector<int64_t> _u;
        const auto vision_pos      = read_bin_f32(odir + "/enc_vision_pos_flat.f32",      vps);
        const auto text_features   = read_bin_f32(odir + "/enc_text_features.f32",        tfs);
        const auto text_mask       = read_bin_f32(odir + "/md_pca_prompt_mask.f32",       tms);
        const auto prompt_features = read_bin_f32(odir + "/md_pca_prompt_features.f32",   pfs);
        const auto prompt_mask     = text_mask;  pms = tms;
        const auto initial_queries = read_bin_f32(odir + "/queries.f32",                  qs);
        const auto init_ref_boxes  = read_bin_f32(odir + "/initial_reference_points.f32", rps);
        const auto query_pos_L0    = read_bin_f32(odir + "/query_pos_layer_0.f32",        qps);
        const auto ref_pred_masks  = read_bin_f32(odir + "/md_pred_masks.f32",            refs);

        // vision_features for detr_encoder: reshape bb2 [256, 72, 72] → [5184, 256]
        // by flatten(1).transpose(0, 1). Same as dump_reference_binaries.py did.
        std::vector<float> vision_features(5184 * 256);
        for (int c = 0; c < 256; ++c) {
            for (int p = 0; p < 5184; ++p) {
                vision_features[p * 256 + c] = bb2[c * 5184 + p];
            }
        }
        std::vector<int64_t> vfs = {5184, 256};

        auto t0 = std::chrono::steady_clock::now();
        const auto enc_out = enc.run(vision_features, vfs, vision_pos, vps,
                                     text_features, tfs, text_mask, tms);
        const auto dec_out = dec.run(initial_queries, qs, text_features, tfs,
            enc_out.last_hidden_state, {enc_out.vision_seq, enc_out.hidden_dim},
            vision_pos, vps, text_mask, tms, query_pos_L0, init_ref_boxes);
        const auto decoder_queries = cpu_layer_norm(dec_out.hs.back(), 10, 256, oln_w, oln_b);
        const auto post_encoder = pca.run(
            enc_out.last_hidden_state, {enc_out.vision_seq, enc_out.hidden_dim},
            prompt_features, pfs, prompt_mask, pms);

        // Rebuild bb2_local for FPN: reshape (post_encoder [5184, 256]) → [256, 72, 72]
        std::vector<float> bb2_local(256 * 72 * 72);
        for (int t = 0; t < 5184; ++t) {
            const int h = t / 72, w = t % 72;
            for (int c = 0; c < 256; ++c) {
                bb2_local[c * 5184 + h * 72 + w] = post_encoder[t * 256 + c];
            }
        }
        const auto pixel_embed = fpn.run(
            bb0, {256, 288, 288},
            bb1, {256, 144, 144},
            bb2_local, {256, 72, 72});
        const auto out = tail.run(pixel_embed, {256, 288, 288}, decoder_queries, {10, 256});
        auto t1 = std::chrono::steady_clock::now();

        const size_t obj_off = static_cast<size_t>(obj) * 10 * 288 * 288;
        std::memcpy(all_masks.data() + obj_off, out.pred_masks.data(),
                    out.pred_masks.size() * sizeof(float));
        std::memcpy(all_ref.data()   + obj_off, ref_pred_masks.data(),
                    ref_pred_masks.size() * sizeof(float));

        double maxdiff = 0.0, l2diff = 0.0, l2ref = 0.0;
        for (size_t i = 0; i < out.pred_masks.size(); ++i) {
            const double d = static_cast<double>(out.pred_masks[i]) - ref_pred_masks[i];
            if (std::fabs(d) > maxdiff) maxdiff = std::fabs(d);
            l2diff += d * d;
            l2ref  += static_cast<double>(ref_pred_masks[i]) * ref_pred_masks[i];
        }
        std::cout << "  chain " << std::chrono::duration_cast<std::chrono::seconds>(t1-t0).count()
                  << "s  parity: cos=" << cosine_flat(out.pred_masks, ref_pred_masks)
                  << " max_diff=" << maxdiff
                  << " rel_L2=" << (std::sqrt(l2diff) / std::sqrt(l2ref)) << "\n";
    }

    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - session_start).count();

    // Aggregate parity
    double amaxd = 0.0, al2d = 0.0, al2r = 0.0;
    for (size_t i = 0; i < all_masks.size(); ++i) {
        const double d = static_cast<double>(all_masks[i]) - all_ref[i];
        if (std::fabs(d) > amaxd) amaxd = std::fabs(d);
        al2d += d * d; al2r += static_cast<double>(all_ref[i]) * all_ref[i];
    }
    std::cout << "\n=== aggregated (chain-only time: " << total_ms/1000.0 << "s) ===\n"
              << "  cos_sim  : " << cosine_flat(all_masks, all_ref) << "\n"
              << "  max_diff : " << amaxd << "\n"
              << "  rel_L2   : " << (std::sqrt(al2d) / std::sqrt(al2r)) << "\n";

    write_bin_f32(out_path, all_masks, {4, 10, 288, 288});
    std::cout << "\n  wrote pred_masks → " << out_path << "\n";
    return 0;
}
