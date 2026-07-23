// Scaffold test for InstructsamDecoder — proves every tensor the
// double-cross-attn decoder graph will need is addressable via sam3cpp's
// GgufModel + our sidecar. Runs before any ggml graph is built.
//
// Exhaustive enumeration: 6 layers × ~30 tensors/layer (self_attn +
// text_cross_attn + vision_cross_attn + MLP + norms) + global decoder
// tensors (box_head, ref_point_head, box_rpb_embed_x/y, query_embed,
// reference_points, output_layer_norm) + InstructSAM glue tensors.
//
// A pass here means the C++ decoder graph implementation can proceed
// without discovering "oh, tensor X isn't actually addressable" mid-way
// through — every require_tensor() call in the eventual graph is
// pre-validated.

#include "sam3/gguf_model.h"
#include "sam3/instructsam_decoder.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Reads a raw fp32 tensor dump written by tools/dump_reference_binaries.py.
// Header: "BIN1" + int32 ndim + int64[ndim] shape + payload.
std::vector<float> read_bin_f32(const std::string & path, std::vector<int64_t> & shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "BIN1") {
        throw std::runtime_error("bad magic in " + path);
    }
    int32_t ndim = 0;
    f.read(reinterpret_cast<char *>(&ndim), 4);
    shape.assign(static_cast<size_t>(ndim), 0);
    size_t total = 1;
    for (int i = 0; i < ndim; ++i) {
        int64_t d = 0;
        f.read(reinterpret_cast<char *>(&d), 8);
        shape[static_cast<size_t>(i)] = d;
        total *= static_cast<size_t>(d);
    }
    std::vector<float> data(total);
    f.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(total * sizeof(float)));
    if (!f) throw std::runtime_error("short read on " + path);
    return data;
}

// Per-query cosine similarity over [nq, dim].
double mean_cosine(const std::vector<float> & a, const std::vector<float> & b, int nq, int dim) {
    double sum = 0.0;
    int counted = 0;
    for (int q = 0; q < nq; ++q) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int d = 0; d < dim; ++d) {
            const float av = a[static_cast<size_t>(q * dim + d)];
            const float bv = b[static_cast<size_t>(q * dim + d)];
            dot += static_cast<double>(av) * bv;
            na  += static_cast<double>(av) * av;
            nb  += static_cast<double>(bv) * bv;
        }
        if (na > 0 && nb > 0) {
            sum += dot / (std::sqrt(na) * std::sqrt(nb));
            ++counted;
        }
    }
    return counted > 0 ? sum / counted : 0.0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: sam3-instructsam-decoder-test <model.gguf>\n";
        return 1;
    }

    sam3::GgufModel model;
    if (!model.load(argv[1])) {
        std::cerr << "failed to load model: " << argv[1] << "\n";
        return 2;
    }
    std::cout << "loaded " << model.tensors().size() << " tensors\n";

    sam3::InstructsamDecoder decoder(model);

    try {
        const size_t probed = decoder.validate_all_tensors_present();
        std::cout << "\n✓ InstructsamDecoder: all " << probed
                  << " required tensors resolvable via sam3cpp find_tensor()\n";
        // Per-layer: self_attn q/k/v/o (4 × [w+b] = 8) + self_attn_norm (2)
        //          + text_cross_attn (8) + text_cross_norm (2)
        //          + vision_cross_attn (8) + vision_cross_norm (2)
        //          + mlp fc1+fc2 (4) + mlp_norm (2) = 36 per layer × 6 = 216
        // Global decoder: box_head (6) + ref_point_head (4)
        //          + box_rpb_embed_x/y (8) + query_embed (1)
        //          + reference_points (1) + output_layer_norm (2) = 22
        // InstructSAM glue: mask_queries + mask_hidden_fcs (4) + text_hidden_fcs (4) = 9
        constexpr size_t kPerLayer = 36;
        constexpr size_t kNumLayers = 6;
        constexpr size_t kGlobalDecoder = 22;
        constexpr size_t kGlue = 9;
        std::cout << "\n  breakdown:\n";
        std::cout << "    per-layer:              "
                  << (kPerLayer * kNumLayers) << " (" << kNumLayers
                  << " layers × " << kPerLayer << ")\n";
        std::cout << "    global decoder tensors: " << kGlobalDecoder << "\n";
        std::cout << "    InstructSAM glue:       " << kGlue << "\n";
        std::cout << "    total (expected):       "
                  << (kPerLayer * kNumLayers + kGlobalDecoder + kGlue) << "\n";
    } catch (const std::exception & e) {
        std::cerr << "✗ " << e.what() << "\n";
        return 3;
    }

    // ── Graph-smoke: exercise InstructsamDecoder::run() with dummy inputs
    //
    // Does NOT check numerical correctness — just proves the ggml graph
    // (self_attn + text_cross_attn + vision_cross_attn + MLP × 6 layers)
    // builds cleanly and computes without asserting or NaN-ing on synthetic
    // inputs. Real numerical parity vs. the PyTorch reference oracle lands
    // in step 2e.
    //
    // Modes:
    //   default:    dummy-input smoke (stability check only)
    //   --parity:   load reference oracle binaries + compare per-layer output
    //   --no-run:   skip the graph exercise (only validates tensor presence)
    bool run_graph = true;
    bool parity_mode = false;
    std::string parity_dir;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--no-run") run_graph = false;
        else if (arg == "--parity") {
            parity_mode = true;
            if (i + 1 < argc) parity_dir = argv[++i];
            else parity_dir = "/tmp/pathA_reference/warehouse_rgb/binaries_obj0";
        }
    }

    if (parity_mode) {
        std::cout << "\n=== parity: reference oracle ==\n";
        std::cout << "  binaries: " << parity_dir << "\n";
        std::vector<int64_t> qs, tms, vms, vps, exp0s;
        const auto queries       = read_bin_f32(parity_dir + "/queries.f32", qs);
        const auto text_memory   = read_bin_f32(parity_dir + "/text_memory.f32", tms);
        const auto vision_memory = read_bin_f32(parity_dir + "/vision_memory.f32", vms);
        const auto vision_pos    = read_bin_f32(parity_dir + "/vision_pos.f32", vps);
        const auto expected0     = read_bin_f32(parity_dir + "/expected_layer_0.f32", exp0s);
        std::cout << "  queries       shape=[" << qs[0]  << "," << qs[1]  << "]\n";
        std::cout << "  text_memory   shape=[" << tms[0] << "," << tms[1] << "]\n";
        std::cout << "  vision_memory shape=[" << vms[0] << "," << vms[1] << "]\n";
        std::cout << "  vision_pos    shape=[" << vps[0] << "," << vps[1] << "]\n";
        std::cout << "  expected_L0   shape=[" << exp0s[0] << "," << exp0s[1] << "]\n";

        // no text mask needed for this reference — text_features have no
        // padding at this stage (all 32 tokens are real)
        std::vector<float> text_mask(static_cast<size_t>(tms[0]), 0.0f);

        // query_pos from ref_point_head(sinusoidal(sigmoid(reference_points.weight)))
        // — pre-computed in dump_reference_binaries.py for layer 0.
        std::vector<int64_t> qps;
        std::vector<float> query_pos;
        try {
            query_pos = read_bin_f32(parity_dir + "/query_pos_layer_0.f32", qps);
            std::cout << "  query_pos_L0  shape=[" << qps[0] << "," << qps[1] << "]\n";
        } catch (const std::exception & e) {
            std::cout << "  (no query_pos file — falling back to zeros: " << e.what() << ")\n";
        }

        try {
            const sam3::DecoderOutput out = decoder.run(
                queries,       qs,
                text_memory,   tms,
                vision_memory, vms,
                vision_pos,    vps,
                text_mask,     {tms[0]},
                query_pos);

            const int nq  = out.num_queries;
            const int dim = out.hidden_dim;
            for (int l = 0; l < out.num_layers; ++l) {
                const auto & hs = out.hs[static_cast<size_t>(l)];
                double sum = 0.0, sumsq = 0.0, absmax = 0.0;
                for (float x : hs) {
                    sum += x; sumsq += static_cast<double>(x) * x;
                    if (std::fabs(x) > absmax) absmax = std::fabs(x);
                }
                const double mean = sum / hs.size();
                const double var  = sumsq / hs.size() - mean * mean;
                std::cout << "  L" << l << " ours:  mean=" << mean
                          << " std=" << std::sqrt(std::max(0.0, var))
                          << " absmax=" << absmax << "\n";
            }

            // Reference comparison — layer 0 only for now. Other layers
            // require box refinement between layers (deferred to step 2e
            // continuation).
            const auto & hs0 = out.hs[0];
            double sum = 0.0, sumsq = 0.0, absmax = 0.0;
            for (float x : expected0) {
                sum += x; sumsq += static_cast<double>(x) * x;
                if (std::fabs(x) > absmax) absmax = std::fabs(x);
            }
            const double emean = sum / expected0.size();
            const double evar  = sumsq / expected0.size() - emean * emean;
            std::cout << "  L0 ref:  mean=" << emean
                      << " std=" << std::sqrt(std::max(0.0, evar))
                      << " absmax=" << absmax << "\n";

            const double cos_sim = mean_cosine(hs0, expected0, nq, dim);
            double maxdiff = 0.0, l2diff = 0.0, l2ref = 0.0;
            for (size_t i = 0; i < hs0.size(); ++i) {
                const double d = static_cast<double>(hs0[i]) - expected0[i];
                if (std::fabs(d) > maxdiff) maxdiff = std::fabs(d);
                l2diff += d * d;
                l2ref  += static_cast<double>(expected0[i]) * expected0[i];
            }
            const double rel_l2 = std::sqrt(l2diff) / std::sqrt(l2ref);
            std::cout << "\n  layer-0 parity (ours vs. reference):\n";
            std::cout << "    mean cosine similarity  : " << cos_sim << "\n";
            std::cout << "    max absolute diff       : " << maxdiff << "\n";
            std::cout << "    relative L2 error       : " << rel_l2 << "\n";
        } catch (const std::exception & e) {
            std::cerr << "  ✗ parity threw: " << e.what() << "\n";
            return 6;
        }
        std::cout << "\n=== step 2e — numerical parity dashboard ===\n";
        std::cout << "  cos_sim=1, max_diff=0, rel_l2=0 => byte-identical\n";
        std::cout << "  next: wire query_pos + box_head + box_rpb, iterate on cos_sim\n";
        return 0;
    }

    if (run_graph) {
        std::cout << "\n=== graph-smoke: run() with dummy inputs ===\n";
        // Shapes match what the caller will feed in step 3+:
        //   queries       : [10, 256]   — mask_hidden_fcs output
        //   text_memory   : [4, 256]    — text_hidden_fcs output on phrase tokens
        //   vision_memory : [5184, 256] — detr_encoder last_hidden_state (72×72)
        constexpr int32_t kNQ  = 10;
        constexpr int32_t kDim = 256;
        constexpr int32_t kTxt = 4;
        constexpr int32_t kHW  = 5184;

        auto uniform = [](size_t n, float scale) {
            std::vector<float> v(n);
            for (size_t i = 0; i < n; ++i) {
                v[i] = scale * (static_cast<float>(i % 97) / 97.0f - 0.5f);
            }
            return v;
        };

        const std::vector<float> queries       = uniform(kNQ * kDim, 0.02f);
        const std::vector<float> text_memory   = uniform(kTxt * kDim, 0.02f);
        const std::vector<float> vision_memory = uniform(kHW * kDim, 0.02f);
        const std::vector<float> vision_pos    = uniform(kHW * kDim, 0.01f);
        const std::vector<float> text_mask(kTxt, 0.0f);  // no-mask

        try {
            const sam3::DecoderOutput out = decoder.run(
                queries,       {kNQ, kDim},
                text_memory,   {kTxt, kDim},
                vision_memory, {kHW, kDim},
                vision_pos,    {kHW, kDim},
                text_mask,     {kTxt});

            std::cout << "  ✓ run() returned num_layers=" << out.num_layers
                      << " num_queries=" << out.num_queries
                      << " hidden_dim=" << out.hidden_dim << "\n";

            for (int l = 0; l < out.num_layers; ++l) {
                const auto & hs = out.hs[static_cast<size_t>(l)];
                float sum = 0.0f, absmax = 0.0f;
                int nan_count = 0;
                for (float x : hs) {
                    if (std::isnan(x) || std::isinf(x)) { ++nan_count; continue; }
                    sum += x;
                    if (std::fabs(x) > absmax) absmax = std::fabs(x);
                }
                std::cout << "    layer " << l << ": mean=" << (sum / hs.size())
                          << " absmax=" << absmax
                          << " NaN/Inf=" << nan_count << "/" << hs.size() << "\n";
                if (nan_count > 0) {
                    std::cerr << "    ✗ layer " << l << " produced NaN/Inf — graph unstable\n";
                    return 4;
                }
            }
            std::cout << "  ✓ 6-layer forward pass ran to completion, no NaN/Inf\n";
        } catch (const std::exception & e) {
            std::cerr << "  ✗ run() threw: " << e.what() << "\n";
            return 5;
        }
    }

    std::cout << "\n=== phase-2-step-2d graph-fork pass ===\n";
    std::cout << "  next (step 2e): numerical parity vs reference oracle\n";
    std::cout << "  reference: docs/instructsam/reference/warehouse_rgb/manifest.json\n";
    return 0;
}
