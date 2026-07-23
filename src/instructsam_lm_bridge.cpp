#include "sam3/instructsam_lm_bridge.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

constexpr int64_t kLmDim  = 2048;
constexpr int64_t kOutDim = 256;

ggml_tensor * require_tensor(const GgufModel & model, const std::string & name) {
    ggml_tensor * t = model.find_weight(name);
    if (t == nullptr) throw std::runtime_error("lm_bridge: missing " + name);
    return t;
}

std::vector<float> get_f32(const GgufModel & model, const std::string & name, size_t n) {
    ggml_tensor * t = require_tensor(model, name);
    std::vector<float> v(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> buf(n);
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) v[i] = ggml_fp16_to_fp32(buf[i]);
    } else {
        throw std::runtime_error("lm_bridge: unsupported dtype for " + name);
    }
    return v;
}

// Fused CPU Linear + optional ReLU: y = ReLU(x @ w.T + b) or y = x @ w.T + b
// x: [N, in], w: [out, in], b: [out] → y: [N, out]
std::vector<float> cpu_linear(
    const std::vector<float> & x, int64_t N, int64_t in_dim, int64_t out_dim,
    const std::vector<float> & w, const std::vector<float> & b, bool with_relu
) {
    std::vector<float> y(static_cast<size_t>(N * out_dim));
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t o = 0; o < out_dim; ++o) {
            float s = b[static_cast<size_t>(o)];
            for (int64_t k = 0; k < in_dim; ++k) {
                s += w[static_cast<size_t>(o * in_dim + k)] *
                     x[static_cast<size_t>(n * in_dim + k)];
            }
            y[static_cast<size_t>(n * out_dim + o)] = with_relu ? std::max(0.0f, s) : s;
        }
    }
    return y;
}

LmBridgeOutput run_bridge(
    const GgufModel & model,
    const std::string & prefix,     // "instructsam.mask_hidden_fcs" or ".text_hidden_fcs"
    const std::vector<float> & x,
    const std::vector<int64_t> & shape
) {
    if (shape.size() < 2 || shape.back() != kLmDim) {
        throw std::runtime_error("lm_bridge: input must end with dim=2048");
    }
    int64_t N = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) N *= shape[i];

    // Sequential MLP: [0]=Linear(2048→2048), [1]=ReLU, [2]=Linear(2048→256), [3]=Dropout(0.0 no-op)
    const auto w0 = get_f32(model, prefix + ".0.0.weight", kLmDim * kLmDim);
    const auto b0 = get_f32(model, prefix + ".0.0.bias",   kLmDim);
    const auto w2 = get_f32(model, prefix + ".0.2.weight", kOutDim * kLmDim);
    const auto b2 = get_f32(model, prefix + ".0.2.bias",   kOutDim);

    const auto h1 = cpu_linear(x,  N, kLmDim,  kLmDim,  w0, b0, /*relu=*/true);
    auto       h2 = cpu_linear(h1, N, kLmDim,  kOutDim, w2, b2, /*relu=*/false);

    LmBridgeOutput out;
    out.batch   = (shape.size() >= 3) ? shape[0] : 1;
    out.seq     = (shape.size() >= 3) ? shape[1] : shape[0];
    out.out_dim = kOutDim;
    out.data    = std::move(h2);
    return out;
}

}  // namespace

InstructsamMaskHiddenFcs::InstructsamMaskHiddenFcs(const GgufModel & model) : model_(model) {}
LmBridgeOutput InstructsamMaskHiddenFcs::run(
    const std::vector<float> & seg_output_embeddings,
    const std::vector<int64_t> & shape
) const {
    return run_bridge(model_, "instructsam.mask_hidden_fcs", seg_output_embeddings, shape);
}

InstructsamTextHiddenFcs::InstructsamTextHiddenFcs(const GgufModel & model) : model_(model) {}
LmBridgeOutput InstructsamTextHiddenFcs::run(
    const std::vector<float> & phrase_embeddings,
    const std::vector<int64_t> & shape
) const {
    return run_bridge(model_, "instructsam.text_hidden_fcs", phrase_embeddings, shape);
}

}  // namespace sam3
