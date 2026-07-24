#include "sam3/instructsam_preprocess.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

// Bilinear resample from (src_h, src_w) → (dst_h, dst_w) on a single channel.
// PyTorch-Pillow BILINEAR without antialiasing (resample=2 in PIL) uses the
// same "pixel center at integer coord" convention we implement here.
void bilinear_resize_channel(
    const std::vector<float> & src, int64_t src_h, int64_t src_w,
    std::vector<float> & dst, int64_t dst_h, int64_t dst_w
) {
    // Ratios: how many src pixels per dst pixel
    const double sy = static_cast<double>(src_h) / dst_h;
    const double sx = static_cast<double>(src_w) / dst_w;
    for (int64_t dy = 0; dy < dst_h; ++dy) {
        // PIL convention: cy = (dy + 0.5) * sy - 0.5
        const double cy = (dy + 0.5) * sy - 0.5;
        const int64_t iy0 = static_cast<int64_t>(std::floor(cy));
        const int64_t iy1 = iy0 + 1;
        const double fy = cy - iy0;
        const int64_t y0 = std::max<int64_t>(0, std::min(src_h - 1, iy0));
        const int64_t y1 = std::max<int64_t>(0, std::min(src_h - 1, iy1));

        for (int64_t dx = 0; dx < dst_w; ++dx) {
            const double cx = (dx + 0.5) * sx - 0.5;
            const int64_t ix0 = static_cast<int64_t>(std::floor(cx));
            const int64_t ix1 = ix0 + 1;
            const double fx = cx - ix0;
            const int64_t x0 = std::max<int64_t>(0, std::min(src_w - 1, ix0));
            const int64_t x1 = std::max<int64_t>(0, std::min(src_w - 1, ix1));

            const double a = src[static_cast<size_t>(y0 * src_w + x0)];
            const double b = src[static_cast<size_t>(y0 * src_w + x1)];
            const double c = src[static_cast<size_t>(y1 * src_w + x0)];
            const double d = src[static_cast<size_t>(y1 * src_w + x1)];
            const double v = a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx
                           + c * fy       * (1 - fx) + d * fy       * fx;
            dst[static_cast<size_t>(dy * dst_w + dx)] = static_cast<float>(v);
        }
    }
}

}  // namespace

std::vector<float> InstructsamPreprocess::load_image_pixel_values(
    const std::string & image_path, int32_t target_size
) {
    int w, h, comp;
    unsigned char * data = stbi_load(image_path.c_str(), &w, &h, &comp, 3);  // force RGB
    if (data == nullptr) {
        throw std::runtime_error("stbi_load failed: " + image_path +
                                 " (" + stbi_failure_reason() + ")");
    }

    // Split into per-channel float [0, 1] normalized [src_h, src_w] tensors
    std::vector<float> src_r(static_cast<size_t>(h * w));
    std::vector<float> src_g(static_cast<size_t>(h * w));
    std::vector<float> src_b(static_cast<size_t>(h * w));
    const float scale = 1.0f / 255.0f;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const unsigned char * px = data + (y * w + x) * 3;
            const size_t idx = static_cast<size_t>(y * w + x);
            src_r[idx] = px[0] * scale;
            src_g[idx] = px[1] * scale;
            src_b[idx] = px[2] * scale;
        }
    }
    stbi_image_free(data);

    // Bilinear resize each channel to target_size × target_size
    std::vector<float> dst_r(static_cast<size_t>(target_size * target_size));
    std::vector<float> dst_g(static_cast<size_t>(target_size * target_size));
    std::vector<float> dst_b(static_cast<size_t>(target_size * target_size));
    bilinear_resize_channel(src_r, h, w, dst_r, target_size, target_size);
    bilinear_resize_channel(src_g, h, w, dst_g, target_size, target_size);
    bilinear_resize_channel(src_b, h, w, dst_b, target_size, target_size);

    // Normalize (x - 0.5) / 0.5 = 2x - 1 → [-1, 1], and pack channels-first.
    // Output layout [3, target_size, target_size] in memory-order (c outer, w inner).
    const size_t plane = static_cast<size_t>(target_size) * target_size;
    std::vector<float> pv(3 * plane);
    for (size_t i = 0; i < plane; ++i) pv[0 * plane + i] = 2.0f * dst_r[i] - 1.0f;
    for (size_t i = 0; i < plane; ++i) pv[1 * plane + i] = 2.0f * dst_g[i] - 1.0f;
    for (size_t i = 0; i < plane; ++i) pv[2 * plane + i] = 2.0f * dst_b[i] - 1.0f;
    return pv;
}

}  // namespace sam3
