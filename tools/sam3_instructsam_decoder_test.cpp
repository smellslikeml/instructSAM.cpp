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

#include <iostream>
#include <stdexcept>

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

    std::cout << "\n=== phase-2-step-2 skeleton pass ===\n";
    std::cout << "  next: implement double-cross-attn ggml graph per layer\n";
    std::cout << "  reference oracle for validation: docs/instructsam/reference/warehouse_rgb/manifest.json\n";
    return 0;
}
