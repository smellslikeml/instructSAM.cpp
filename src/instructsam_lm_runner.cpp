#include "sam3/instructsam_lm_runner.h"

#include <stdexcept>
#include <string>

namespace sam3 {

// Skeleton — no llama.cpp linkage yet. See LM_INTEGRATION_PLAN.md for
// the concrete API-surface investigation and 5-day roadmap. Implementing
// this requires linking libllama + libmtmd from the llama.cpp build.
//
// This file exists so downstream code (CLI orchestrator, etc.) can
// compile against InstructsamLmRunner's interface while the impl grows.

struct InstructsamLmRunner::Impl {
    std::string lm_gguf_path;
    std::string mmproj_path;
    std::string grounding_gguf;
    int max_generated_tokens;
};

InstructsamLmRunner::InstructsamLmRunner(
    const std::string & lm_gguf_path,
    const std::string & mmproj_path,
    const std::string & grounding_gguf,
    int max_generated_tokens
) : impl_(std::make_unique<Impl>()) {
    impl_->lm_gguf_path = lm_gguf_path;
    impl_->mmproj_path = mmproj_path;
    impl_->grounding_gguf = grounding_gguf;
    impl_->max_generated_tokens = max_generated_tokens;
}

InstructsamLmRunner::~InstructsamLmRunner() = default;

LmOutput InstructsamLmRunner::run(const std::string & /*image_path*/,
                                  const std::string & /*query*/) {
    throw std::runtime_error(
        "InstructsamLmRunner::run() is a skeleton — impl deferred to "
        "Piece 3 Day 1-5 (see docs/instructsam/LM_INTEGRATION_PLAN.md). "
        "Requires linking libllama + libmtmd from llama.cpp.");
}

}  // namespace sam3
