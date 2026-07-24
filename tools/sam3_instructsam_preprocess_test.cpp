// Test image preprocess parity: load warehouse_rgb.jpg via our C++
// preprocessor, compare vs captured pixel_values.f32 from PyTorch.

#include "sam3/instructsam_preprocess.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_bin_f32(const std::string & path, std::vector<int64_t> & shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4]; f.read(magic, 4);
    if (std::string(magic, 4) != "BIN1") throw std::runtime_error("bad magic");
    int32_t ndim = 0; f.read(reinterpret_cast<char *>(&ndim), 4);
    shape.assign(static_cast<size_t>(ndim), 0);
    size_t total = 1;
    for (int i = 0; i < ndim; ++i) {
        int64_t d = 0; f.read(reinterpret_cast<char *>(&d), 8);
        shape[static_cast<size_t>(i)] = d; total *= static_cast<size_t>(d);
    }
    std::vector<float> data(total);
    f.read(reinterpret_cast<char *>(data.data()), total * sizeof(float));
    return data;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string image_path = (argc >= 2) ? argv[1] : "/home/thorax/Downloads/warehouse_rgb.jpg";
    const std::string ref_path = (argc >= 3) ? argv[2]
        : "/tmp/pathA_reference/warehouse_rgb/binaries_vision/pixel_values.f32";

    std::cout << "loading + preprocessing " << image_path << "\n" << std::flush;
    const auto pv = sam3::InstructsamPreprocess::load_image_pixel_values(image_path);
    std::cout << "  ours: [3, 1008, 1008] = " << pv.size() << " floats\n";
    float mn = pv[0], mx = pv[0]; double sum = 0.0;
    for (float v : pv) { if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
    std::cout << "  range: min=" << mn << " max=" << mx << " mean=" << (sum / pv.size()) << "\n";
    std::cout << "  first 6: " << pv[0] << " " << pv[1] << " " << pv[2] << " " << pv[3] << " " << pv[4] << " " << pv[5] << "\n";

    std::vector<int64_t> ref_shape;
    const auto ref = read_bin_f32(ref_path, ref_shape);
    std::cout << "\n  ref pixel_values (from PyTorch capture): " << ref.size() << " floats\n";
    std::cout << "  ref first 6: " << ref[0] << " " << ref[1] << " " << ref[2] << " " << ref[3] << " " << ref[4] << " " << ref[5] << "\n";

    if (ref.size() != pv.size()) { std::cerr << "  ✗ size mismatch\n"; return 3; }

    double maxdiff = 0.0, l2diff = 0.0, l2ref = 0.0, dot = 0.0, l2a = 0.0, l2b = 0.0;
    for (size_t i = 0; i < pv.size(); ++i) {
        const double d = pv[i] - ref[i];
        if (std::fabs(d) > maxdiff) maxdiff = std::fabs(d);
        l2diff += d * d; l2ref += static_cast<double>(ref[i]) * ref[i];
        dot += static_cast<double>(pv[i]) * ref[i];
        l2a += static_cast<double>(pv[i]) * pv[i]; l2b += static_cast<double>(ref[i]) * ref[i];
    }
    const double cos = dot / (std::sqrt(l2a) * std::sqrt(l2b));
    std::cout << "\n=== preprocess parity vs PyTorch reference ===\n"
              << "  cos_sim  : " << cos << "\n"
              << "  max_diff : " << maxdiff << "\n"
              << "  rel_L2   : " << (std::sqrt(l2diff) / std::sqrt(l2ref)) << "\n";
    if (cos > 0.999) std::cout << "  ✓ preprocess ~= PyTorch (bilinear resize + normalize match)\n";
    else std::cout << "  ✗ significant divergence — check resample algorithm\n";
    return 0;
}
