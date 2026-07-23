#include "sam3/instructsam_decoder.h"

#include <stdexcept>
#include <string>

namespace sam3 {

namespace {

// InstructSAM's decoder is 6 layers @ 256-dim, matching sam3cpp's stock
// decoder. See detr_decoder.layers.N in InstructSAM-2B checkpoint.
constexpr int32_t kLayers = 6;
constexpr int32_t kBoxHeadLayers = 3;
constexpr int32_t kRefPointHeadLayers = 2;
constexpr int32_t kBoxRpbLayers = 2;

// Enumerate per-layer tensor names. Names use InstructSAM's canonical
// convention (q_proj/k_proj/v_proj/o_proj + layer_norm suffixes) as
// emitted by tools/convert_instructsam_to_gguf.py under the
// "transformer.decoder.*" prefix in the sidecar map.
std::vector<std::string> per_layer_tensor_names(int layer) {
    const std::string p = "transformer.decoder.layers." + std::to_string(layer);
    std::vector<std::string> names;

    // Self-attention: Q K V O + biases
    for (const std::string qkvo : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        names.push_back(p + ".self_attn." + qkvo + ".weight");
        names.push_back(p + ".self_attn." + qkvo + ".bias");
    }
    names.push_back(p + ".self_attn_layer_norm.weight");
    names.push_back(p + ".self_attn_layer_norm.bias");

    // Text cross-attention (InstructSAM-specific — separate from vision)
    for (const std::string qkvo : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        names.push_back(p + ".text_cross_attn." + qkvo + ".weight");
        names.push_back(p + ".text_cross_attn." + qkvo + ".bias");
    }
    names.push_back(p + ".text_cross_attn_layer_norm.weight");
    names.push_back(p + ".text_cross_attn_layer_norm.bias");

    // Vision cross-attention (InstructSAM-specific — separate from text)
    for (const std::string qkvo : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        names.push_back(p + ".vision_cross_attn." + qkvo + ".weight");
        names.push_back(p + ".vision_cross_attn." + qkvo + ".bias");
    }
    names.push_back(p + ".vision_cross_attn_layer_norm.weight");
    names.push_back(p + ".vision_cross_attn_layer_norm.bias");

    // MLP: fc1 (256 → 2048) + fc2 (2048 → 256) + norm
    for (const std::string fc : {"fc1", "fc2"}) {
        names.push_back(p + ".mlp." + fc + ".weight");
        names.push_back(p + ".mlp." + fc + ".bias");
    }
    names.push_back(p + ".mlp_layer_norm.weight");
    names.push_back(p + ".mlp_layer_norm.bias");

    return names;
}

// Non-per-layer decoder tensors: box head, box rpb (relative position bias),
// reference points, query embed, output norm.
std::vector<std::string> global_decoder_tensor_names() {
    std::vector<std::string> names;
    const std::string p = "transformer.decoder";

    // Box refinement head: 3-layer MLP (256 → 256 → 256 → 4)
    for (int i = 1; i <= kBoxHeadLayers; ++i) {
        names.push_back(p + ".box_head.layer" + std::to_string(i) + ".weight");
        names.push_back(p + ".box_head.layer" + std::to_string(i) + ".bias");
    }
    // Reference-point head: 2-layer MLP
    for (int i = 1; i <= kRefPointHeadLayers; ++i) {
        names.push_back(p + ".ref_point_head.layer" + std::to_string(i) + ".weight");
        names.push_back(p + ".ref_point_head.layer" + std::to_string(i) + ".bias");
    }
    // Box relative position bias embeddings (X and Y): 2-layer MLPs
    for (const std::string axis : {"box_rpb_embed_x", "box_rpb_embed_y"}) {
        for (int i = 1; i <= kBoxRpbLayers; ++i) {
            names.push_back(p + "." + axis + ".layer" + std::to_string(i) + ".weight");
            names.push_back(p + "." + axis + ".layer" + std::to_string(i) + ".bias");
        }
    }
    // Learnable query embeddings + reference points
    names.push_back(p + ".query_embed.weight");
    names.push_back(p + ".reference_points.weight");
    // Output layer norm
    names.push_back(p + ".output_layer_norm.weight");
    names.push_back(p + ".output_layer_norm.bias");

    return names;
}

// InstructSAM-specific glue tensors that live outside the decoder graph
// but are addressable via the same GGUF. Consumed by the caller's
// hidden-state projection pipeline, not by the ggml decoder graph.
std::vector<std::string> instructsam_glue_tensor_names() {
    return {
        "instructsam.mask_queries",
        "instructsam.mask_hidden_fcs.0.0.weight",
        "instructsam.mask_hidden_fcs.0.0.bias",
        "instructsam.mask_hidden_fcs.0.2.weight",
        "instructsam.mask_hidden_fcs.0.2.bias",
        "instructsam.text_hidden_fcs.0.0.weight",
        "instructsam.text_hidden_fcs.0.0.bias",
        "instructsam.text_hidden_fcs.0.2.weight",
        "instructsam.text_hidden_fcs.0.2.bias",
    };
}

}  // namespace

InstructsamDecoder::InstructsamDecoder(const GgufModel & model) : model_(model) {}

size_t InstructsamDecoder::validate_all_tensors_present() const {
    size_t probed = 0;

    // Per-layer tensors × 6 layers
    for (int layer = 0; layer < kLayers; ++layer) {
        for (const auto & name : per_layer_tensor_names(layer)) {
            if (model_.find_tensor(name) == nullptr) {
                throw std::runtime_error(
                    "InstructsamDecoder: missing per-layer tensor: " + name);
            }
            ++probed;
        }
    }

    // Global decoder tensors
    for (const auto & name : global_decoder_tensor_names()) {
        if (model_.find_tensor(name) == nullptr) {
            throw std::runtime_error(
                "InstructsamDecoder: missing global tensor: " + name);
        }
        ++probed;
    }

    // InstructSAM glue tensors (verified addressable but not consumed
    // by the decoder graph itself; caller uses them for LM hidden state
    // projection before feeding queries into the decoder).
    for (const auto & name : instructsam_glue_tensor_names()) {
        if (model_.find_tensor(name) == nullptr) {
            throw std::runtime_error(
                "InstructsamDecoder: missing glue tensor: " + name);
        }
        ++probed;
    }

    return probed;
}

}  // namespace sam3
