#pragma once

#include "sam3/gguf_model.h"

#include <cstdint>
#include <vector>

namespace sam3 {

// FPN-style pixel decoder for InstructSAM.
//
// Given the 3 backbone features (coarsest already replaced with
// post-prompt-cross-attn encoder features by the caller), iterate
// coarse → fine: upsample nearest-2x, add skip connection, conv3x3
// (same padding), GroupNorm(8), ReLU.
//
// InstructSAM checkpoints ship 3 conv_layers + 3 norms but only layers
// 0 and 1 execute with our 3-input FPN (matches PyTorch runtime).
// layer 2 stays unused — its weights are dead in this config.
//
// Weights consumed:
//   segmentation_head.pixel_decoder.conv_layers.0.{weight,bias}   256→256, 3×3
//   segmentation_head.pixel_decoder.norms.0.{weight,bias}         GroupNorm(8) affine
//   segmentation_head.pixel_decoder.conv_layers.1.{weight,bias}
//   segmentation_head.pixel_decoder.norms.1.{weight,bias}
class InstructsamPixelDecoder {
public:
    explicit InstructsamPixelDecoder(const GgufModel & model);

    // bb0: finest [C, H0, W0]  — typically [256, 288, 288]
    // bb1: mid    [C, H1, W1]  — typically [256, 144, 144]
    // bb2: coarsest [C, H2, W2] — typically [256, 72, 72]
    //      Caller pre-computes this = reshape(post-PCA encoder [5184,256]) → [256,72,72]
    // Returns pixel_embed [C, H0, W0].
    std::vector<float> run(
        const std::vector<float> & bb0, const std::vector<int64_t> & bb0_shape,
        const std::vector<float> & bb1, const std::vector<int64_t> & bb1_shape,
        const std::vector<float> & bb2, const std::vector<int64_t> & bb2_shape
    ) const;

private:
    const GgufModel & model_;
};

}  // namespace sam3
