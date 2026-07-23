// Batched E2E for all 4 objects of warehouse_rgb.jpg → pred_masks.f32
// (aggregated shape [4, 10, 288, 288]). Also compares against PyTorch
// reference and writes the output for visualization by a Python script.

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
        std::cerr << "usage: sam3-instructsam-batch-e2e <model.gguf> [ref_dir] [out.f32]\n";
        return 1;
    }
    const std::string ref_root = (argc >= 3) ? argv[2] : "/tmp/pathA_reference/warehouse_rgb";
    const std::string out_path = (argc >= 4) ? argv[3] : "/tmp/pathA_reference/warehouse_rgb/e2e_pred_masks_ggml.f32";

    sam3::GgufModel model;
    if (!model.load(argv[1])) { std::cerr << "load failed\n"; return 2; }
    std::cout << "loaded " << model.tensors().size() << " tensors\n";

    sam3::InstructsamDetrEncoder     enc(model);
    sam3::InstructsamDecoder         dec(model);
    sam3::InstructsamPromptCrossAttn pca(model);
    sam3::InstructsamPixelDecoder    fpn(model);
    sam3::InstructsamMaskTail        tail(model);

    const auto oln_w = get_f32(model, "transformer.decoder.output_layer_norm.weight", 256);
    const auto oln_b = get_f32(model, "transformer.decoder.output_layer_norm.bias",   256);

    // Aggregated final output [4, 10, 288, 288]
    std::vector<float> all_masks(4 * 10 * 288 * 288);
    std::vector<float> all_ref(4 * 10 * 288 * 288);

    auto session_start = std::chrono::steady_clock::now();

    for (int obj = 0; obj < 4; ++obj) {
        const std::string dir = ref_root + "/binaries_obj" + std::to_string(obj);
        std::cout << "\n=== object " << obj << " ===\n" << std::flush;

        std::vector<int64_t> vfs, vps, tfs, tms, pfs, pms, bb0s, bb1s, qs, rps, qps, refs;
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
        std::cout << "  prompt valid " << valid << "/" << tfs[0] << "\n";

        auto t0 = std::chrono::steady_clock::now();
        const auto enc_out = enc.run(vision_features, vfs, vision_pos, vps,
                                     text_features, tfs, text_mask, tms);
        std::cout << "  detr_encoder      " << std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count() << " ms\n";

        t0 = std::chrono::steady_clock::now();
        const auto dec_out = dec.run(initial_queries, qs, text_features, tfs,
            enc_out.last_hidden_state, {enc_out.vision_seq, enc_out.hidden_dim},
            vision_pos, vps, text_mask, tms, query_pos_L0, init_ref_boxes);
        std::cout << "  detr_decoder      " << std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count() << " ms\n";

        const auto decoder_queries = cpu_layer_norm(dec_out.hs.back(), 10, 256, oln_w, oln_b);

        t0 = std::chrono::steady_clock::now();
        const auto post_encoder = pca.run(
            enc_out.last_hidden_state, {enc_out.vision_seq, enc_out.hidden_dim},
            prompt_features, pfs, prompt_mask, pms);
        std::cout << "  prompt_cross_attn " << std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count() << " ms\n";

        std::vector<float> bb2(256 * 72 * 72);
        for (int t = 0; t < 5184; ++t) {
            const int h = t / 72, w = t % 72;
            for (int c = 0; c < 256; ++c) {
                bb2[static_cast<size_t>(c * 5184 + h * 72 + w)] =
                    post_encoder[static_cast<size_t>(t * 256 + c)];
            }
        }
        t0 = std::chrono::steady_clock::now();
        const auto pixel_embed = fpn.run(bb0, bb0s, bb1, bb1s, bb2, {256, 72, 72});
        std::cout << "  pixel_decoder FPN " << std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count() << " ms\n";

        t0 = std::chrono::steady_clock::now();
        const auto out = tail.run(pixel_embed, {256, 288, 288}, decoder_queries, {10, 256});
        std::cout << "  mask_tail         " << std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count() << " ms\n";

        // Copy this object's masks into the aggregated buffer
        const size_t obj_off = static_cast<size_t>(obj) * 10 * 288 * 288;
        std::memcpy(all_masks.data() + obj_off, out.pred_masks.data(),
                    out.pred_masks.size() * sizeof(float));
        std::memcpy(all_ref.data()   + obj_off, ref_pred_masks.data(),
                    ref_pred_masks.size() * sizeof(float));

        // per-object parity
        double maxdiff = 0.0, l2diff = 0.0, l2ref = 0.0;
        for (size_t i = 0; i < out.pred_masks.size(); ++i) {
            const double d = static_cast<double>(out.pred_masks[i]) - ref_pred_masks[i];
            if (std::fabs(d) > maxdiff) maxdiff = std::fabs(d);
            l2diff += d * d;
            l2ref  += static_cast<double>(ref_pred_masks[i]) * ref_pred_masks[i];
        }
        std::cout << "  parity: cos=" << cosine_flat(out.pred_masks, ref_pred_masks)
                  << " max_diff=" << maxdiff
                  << " rel_L2=" << (std::sqrt(l2diff) / std::sqrt(l2ref)) << "\n";
    }

    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - session_start).count();
    std::cout << "\n=== aggregated E2E ==="
              << "\n  total runtime: " << total_ms / 1000.0 << " s (" << (total_ms / 4) << " ms/obj)\n";

    // Aggregate parity
    double amaxdiff = 0.0, al2diff = 0.0, al2ref = 0.0;
    for (size_t i = 0; i < all_masks.size(); ++i) {
        const double d = static_cast<double>(all_masks[i]) - all_ref[i];
        if (std::fabs(d) > amaxdiff) amaxdiff = std::fabs(d);
        al2diff += d * d;
        al2ref  += static_cast<double>(all_ref[i]) * all_ref[i];
    }
    std::cout << "  4-object aggregate parity vs. PyTorch pred_masks:\n"
              << "    cosine (flat)     : " << cosine_flat(all_masks, all_ref) << "\n"
              << "    max abs diff      : " << amaxdiff << "\n"
              << "    relative L2 error : " << (std::sqrt(al2diff) / std::sqrt(al2ref)) << "\n";

    // Write ggml output for Python-side visualization
    write_bin_f32(out_path, all_masks, {4, 10, 288, 288});
    std::cout << "\n  wrote ggml pred_masks → " << out_path << "\n";
    std::cout << "  visualize: python3 tools/visualize_e2e_masks.py " << out_path << "\n";
    return 0;
}
