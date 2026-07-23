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
#include <iostream>
#include <stdexcept>
#include <vector>

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
    // Skip if --no-run is passed (for CI runs without the GGUF hot in cache).
    bool run_graph = true;
    if (argc >= 3 && std::string(argv[2]) == "--no-run") {
        run_graph = false;
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
