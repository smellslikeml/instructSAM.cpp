// InstructSAM GGUF sanity check for sam3cpp.
//
// Loads our emitted GGUF via sam3cpp's GgufModel::load(), then looks up
// a set of canonical sam3cpp-namespace tensor names via find_tensor()
// (which uses the alias-resolving path — sidecar mapping short_name to
// canonical). Reports which resolve + their shapes.
//
// Success proves:
//   1. sam3cpp's GGUFReader reads our converter's output
//   2. sam3cpp's load_tensor_aliases() consumes our .tensor_map.json
//   3. sam3cpp's require_tensor() calls (the C++ decoder graph code)
//      will find tensors under the sam3cpp-namespace canonical names
//
// This is Path A phase-2 step-1: convert-side ↔ runtime-side handshake.
// Any failure identifies a schema-level bug before the decoder-graph
// C++ work starts.

#include "sam3/gguf_model.h"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: sam3-instructsam-sanity <model.gguf>\n";
        return 1;
    }

    sam3::GgufModel model;
    if (!model.load(argv[1])) {
        std::cerr << "failed to load model: " << argv[1] << "\n";
        return 2;
    }

    std::cout << "loaded " << model.tensors().size() << " tensors from "
              << argv[1] << "\n\n";

    // Canonical names sam3cpp's runtime code would look up. These are
    // the names our .tensor_map.json sidecar aliases to short GGUF names.
    // A subset covering each component + the InstructSAM-specific glue.
    const std::vector<std::string> canonical_probes = {
        // Vision encoder — SAM3 ViT backbone
        "backbone.vision_backbone.trunk.embeddings.patch_embeddings.projection.weight",
        "backbone.vision_backbone.trunk.embeddings.position_embeddings",
        "backbone.vision_backbone.trunk.layer_norm.weight",
        // Vision encoder — FPN neck
        "backbone.vision_backbone.convs.fpn_layers.0.proj1.weight",
        "backbone.vision_backbone.convs.fpn_layers.0.proj2.weight",
        // Encoder fusion (transformer.encoder) — InstructSAM naming is
        // q_proj/k_proj/v_proj/o_proj (matches decoder); note the encoder
        // ALSO has cross_attn per layer (deformable-DETR shape with text
        // cross-attention).
        "transformer.encoder.layers.0.self_attn.q_proj.weight",
        "transformer.encoder.layers.5.self_attn.o_proj.weight",
        "transformer.encoder.layers.0.cross_attn.q_proj.weight",
        // Decoder — self_attn (present in both sam3cpp + InstructSAM)
        "transformer.decoder.layers.0.self_attn.q_proj.weight",
        "transformer.decoder.layers.0.self_attn.o_proj.weight",
        "transformer.decoder.layers.5.self_attn.q_proj.weight",
        // Decoder — InstructSAM-specific DOUBLE cross-attn (text + vision separate)
        "transformer.decoder.layers.0.text_cross_attn.q_proj.weight",
        "transformer.decoder.layers.0.vision_cross_attn.q_proj.weight",
        "transformer.decoder.layers.5.text_cross_attn.q_proj.weight",
        "transformer.decoder.layers.5.vision_cross_attn.q_proj.weight",
        // Decoder — MLP + box head
        "transformer.decoder.layers.0.mlp.fc1.weight",
        "transformer.decoder.box_head.layer1.weight",
        "transformer.decoder.box_head.layer3.weight",
        // Mask decoder (segmentation head)
        "segmentation_head.instance_projection.weight",
        "segmentation_head.mask_embedder.layers.0.weight",
        "segmentation_head.pixel_decoder.conv_layers.0.weight",
        // Grounding + geometry
        "dot_prod_scoring.query_proj.weight",
        "geometry_encoder.boxes_direct_project.weight",
        "text_projection.weight",
        // InstructSAM custom glue
        "instructsam.mask_queries",
        "instructsam.mask_hidden_fcs.0.0.weight",
        "instructsam.mask_hidden_fcs.0.2.weight",
        "instructsam.text_hidden_fcs.0.0.weight",
        "instructsam.text_hidden_fcs.0.2.weight",
    };

    size_t found = 0;
    size_t missing = 0;
    std::vector<std::string> missing_names;
    for (const auto & name : canonical_probes) {
        const auto * info = model.find_tensor(name);
        if (info == nullptr) {
            std::cout << "  ✗ " << name << "  MISSING\n";
            missing_names.push_back(name);
            ++missing;
            continue;
        }
        std::cout << "  ✓ " << name << "  [";
        for (size_t j = 0; j < info->shape.size(); ++j) {
            if (j) std::cout << ", ";
            std::cout << info->shape[j];
        }
        std::cout << "]  " << info->type << "\n";
        ++found;
    }

    std::cout << "\n=== summary ===\n";
    std::cout << "  probed: " << canonical_probes.size() << "\n";
    std::cout << "  found:  " << found << "\n";
    std::cout << "  missing: " << missing << "\n";

    return missing == 0 ? 0 : 3;
}
