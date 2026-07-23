#include "sam3/instructsam_lm_forward.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

constexpr int32_t kNumLayers    = 28;
constexpr int32_t kHiddenSize   = 2048;
constexpr int32_t kNumHeads     = 16;
constexpr int32_t kNumKvHeads   = 8;
constexpr int32_t kHeadDim      = 128;
constexpr int32_t kIntermediate = 6144;
constexpr int32_t kVocabSize    = 151936;
// constexpr float   kRmsNormEps   = 1e-6f;
// constexpr float   kRopeTheta    = 5000000.0f;  // Qwen3 default; verify per-config

ggml_tensor * require_tensor(const GgufModel & model, const std::string & name) {
    ggml_tensor * t = model.find_weight(name);
    if (t == nullptr) throw std::runtime_error("lm_forward: missing " + name);
    return t;
}

std::vector<std::string> per_layer_tensor_names(int layer) {
    const std::string p = "blk." + std::to_string(layer);
    return {
        p + ".attn_norm.weight",
        p + ".attn_q.weight", p + ".attn_q.bias",
        p + ".attn_k.weight", p + ".attn_k.bias",
        p + ".attn_v.weight", p + ".attn_v.bias",
        p + ".attn_output.weight",
        p + ".attn_q_norm.weight",
        p + ".attn_k_norm.weight",
        p + ".ffn_norm.weight",
        p + ".ffn_gate.weight",
        p + ".ffn_up.weight",
        p + ".ffn_down.weight",
    };
}

std::vector<std::string> top_level_tensor_names() {
    return {
        "token_embd.weight",
        "output_norm.weight",
    };
}

}  // namespace

InstructsamLmForward::InstructsamLmForward(const GgufModel & model) : model_(model) {}

size_t InstructsamLmForward::validate_all_tensors_present() const {
    size_t probed = 0;
    for (const auto & name : top_level_tensor_names()) {
        // Some tensors (like biases for attn_q) may not exist in Qwen3
        // depending on config. Skip biases from must-have list if not found —
        // but tensor names above are all required in Qwen3 as of the
        // reference checkpoint. Throw on genuinely missing.
        if (model_.find_tensor(name) == nullptr) {
            throw std::runtime_error("lm_forward: missing top-level tensor: " + name);
        }
        ++probed;
    }
    for (int layer = 0; layer < kNumLayers; ++layer) {
        for (const auto & name : per_layer_tensor_names(layer)) {
            // Bias tensors may or may not exist depending on GGUF conversion
            // choices for Qwen3-VL (some converters skip biases if 0.0).
            // Attempt the lookup; if it's a .bias, tolerate absent.
            const bool is_bias = name.size() >= 5 && name.compare(name.size() - 5, 5, ".bias") == 0;
            if (model_.find_tensor(name) == nullptr) {
                if (is_bias) continue;
                throw std::runtime_error("lm_forward: missing per-layer tensor: " + name);
            }
            ++probed;
        }
    }
    return probed;
}

std::vector<float> InstructsamLmForward::embed_for_token(int32_t token_id) const {
    if (token_id < 0 || token_id >= kVocabSize) {
        throw std::runtime_error("lm_forward: token_id out of range");
    }
    ggml_tensor * t = require_tensor(model_, "token_embd.weight");
    // token_embd.weight stored as [hidden=2048, vocab=151936] in ggml (col-major).
    // Row `token_id` is at offset token_id * hidden * sizeof(dtype).
    std::vector<float> out(kHiddenSize);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(),
            token_id * kHiddenSize * sizeof(float),
            kHiddenSize * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(kHiddenSize);
        ggml_backend_tensor_get(t, buf.data(),
            token_id * kHiddenSize * sizeof(ggml_fp16_t),
            kHiddenSize * sizeof(ggml_fp16_t));
        for (int i = 0; i < kHiddenSize; ++i) out[i] = ggml_fp16_to_fp32(buf[i]);
    } else {
        throw std::runtime_error("lm_forward: token_embd unsupported dtype");
    }
    return out;
}

}  // namespace sam3
