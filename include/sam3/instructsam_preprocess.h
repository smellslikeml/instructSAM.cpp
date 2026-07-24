#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sam3 {

// SAM3 image preprocessor — matches Sam3ImageProcessorFast defaults from
// processor_config.json:
//   - do_convert_rgb: true
//   - resize to 1008x1008, bilinear (resample=2)
//   - rescale by 1/255
//   - normalize: (x - 0.5) / 0.5  → [-1, 1]
//   - channels-first layout: [3, 1008, 1008]
//
// Returns pixel_values flat vector [3 * 1008 * 1008] f32, ready to feed
// InstructsamVisionEncoder::run_all_layers(pv, {3, 1008, 1008}).
class InstructsamPreprocess {
public:
    // Load + preprocess an image file (JPEG/PNG/etc via stb_image).
    // Throws on load failure.
    static std::vector<float> load_image_pixel_values(
        const std::string & image_path,
        int32_t target_size = 1008
    );
};

}  // namespace sam3
