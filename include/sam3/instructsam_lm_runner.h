#pragma once

#include <memory>
#include <string>
#include <vector>

namespace sam3 {

// Output of one full LM autoregressive pass on (image, query).
// Feeds directly into our lm_bridge (mask_hidden_fcs, text_hidden_fcs)
// to produce decoder queries + text_features for the grounding chain.
struct LmOutput {
    int num_objects = 0;

    // seg_output_embeddings[obj] shape [10, n_embd=2048]
    // = LM hidden states at the 10 mask_queries positions per object.
    // Feed to InstructsamMaskHiddenFcs → decoder queries [10, 256].
    std::vector<std::vector<float>> seg_output_embeddings;

    // phrase_embeddings[obj] shape [<=32, n_embd=2048]
    // = LM input embedding table looked up at phrase token IDs
    // (tokens between <|object_ref_start|> and <|object_ref_end|>).
    // Feed to InstructsamTextHiddenFcs → text_features [32, 256].
    std::vector<std::vector<float>> phrase_embeddings;

    // phrase_masks[obj] shape [32] — 1.0 for valid tokens, 0.0 padding.
    std::vector<std::vector<float>> phrase_masks;

    // Human-readable text (with [SEG] insertions per InstructSAM's post-process).
    std::string generated_text;
};

// LM runner for InstructSAM — wraps llama.cpp Qwen3-VL with the
// InstructSAM-specific mask_queries embedding injection at <|object_ref_end|>.
//
// See docs/instructsam/LM_INTEGRATION_PLAN.md for design + llama.cpp API surface.
//
// Uses:
//   - Path B's instructsam-lm-f16.gguf (Qwen3-VL text)
//   - Path B's instructsam-mmproj-f16.gguf (vision projector)
//   - Our sam3cpp-fork GGUF (for the InstructSAM-specific mask_queries
//     tensor, which was filtered out of Path B's LM GGUF)
//
// This class is a SKELETON — see the plan doc for the ~5 days of
// implementation work. Class exists so downstream code can compile
// against the interface while the impl is built out.
class InstructsamLmRunner {
public:
    InstructsamLmRunner(
        const std::string & lm_gguf_path,
        const std::string & mmproj_path,
        const std::string & grounding_gguf,
        int max_generated_tokens = 200);
    ~InstructsamLmRunner();

    // Run mtmd-style inference on image + query text.
    // NOT YET IMPLEMENTED — currently throws std::runtime_error.
    // See LM_INTEGRATION_PLAN.md for the concrete llama.cpp call sequence.
    LmOutput run(const std::string & image_path, const std::string & query);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sam3
