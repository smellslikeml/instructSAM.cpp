#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward-decl mtmd types so this header stays clean of libmtmd includes.
struct mtmd_context;

namespace sam3 {

// Wraps llama.cpp's mtmd (multimodal) to turn an image file into LM-space
// input embeddings for Qwen3-VL. The mmproj GGUF is a full 24-layer ViT
// (image_size=768, patch=16, projection_dim=2048) that lives outside our
// SAM3 vision encoder — this path feeds the LM's semantic reasoning, while
// SAM3's own 32-layer encoder feeds the segmentation head.
class InstructsamMmproj {
public:
    struct ImageEmbeddings {
        std::vector<float> data;   // [n_tokens, hidden = 2048], row-major
        int32_t n_tokens = 0;      // Number of image "tokens" mtmd emits
        int32_t hidden  = 0;       // Should be 2048 for InstructSAM's Qwen3-VL-2B
    };

    static std::unique_ptr<InstructsamMmproj> load(const std::string & mmproj_gguf_path,
                                                   const std::string & lm_gguf_path);
    ~InstructsamMmproj();

    // Preprocess + encode a JPEG/PNG file, return the LM-space image
    // embeddings. Throws on failure. n_tokens depends on image size (256
    // for 768² with spatial_merge=2 and 16² patches → 48²/2² = 576, but
    // Qwen3-VL packs 2x2 into one → 144 tokens per 768² tile).
    ImageEmbeddings encode_image_file(const std::string & image_path) const;

private:
    InstructsamMmproj() = default;
    mtmd_context * ctx_ = nullptr;
    // We need an llama model handle for mtmd_init_from_file — but only for
    // model config lookup; owned separately.
    void * model_ = nullptr;  // llama_model *, kept opaque to avoid llama.h leak
};

}  // namespace sam3
