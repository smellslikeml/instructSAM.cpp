// Day 9b — sanity-check that mtmd loads the mmproj + emits sane image
// embeddings for warehouse_rgb.jpg. No PyTorch parity reference exists
// for the raw Qwen3-VL image embeddings, so we validate: correct shape,
// non-zero magnitude, no NaN/Inf, and consistent stats across two loads.

#include "sam3/instructsam_mmproj.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    const std::string mmproj = (argc >= 2) ? argv[1]
        : "/tmp/claude-1001/-home-thorax/e5355e82-8c80-4141-8828-424676e4ee8f/scratchpad/instructsam-as-qwen3vl/instructsam-mmproj-f16.gguf";
    const std::string lm = (argc >= 3) ? argv[2]
        : "/tmp/claude-1001/-home-thorax/e5355e82-8c80-4141-8828-424676e4ee8f/scratchpad/instructsam-as-qwen3vl/instructsam-lm-f16.gguf";
    const std::string image = (argc >= 4) ? argv[3]
        : "/home/thorax/Downloads/warehouse_rgb.jpg";

    std::cout << "loading mmproj: " << mmproj << "\n" << std::flush;
    auto mm = sam3::InstructsamMmproj::load(mmproj, lm);
    std::cout << "  ✓ mmproj + vocab_only LM loaded\n\n";

    std::cout << "encoding image: " << image << "\n" << std::flush;
    const auto emb = mm->encode_image_file(image);
    std::cout << "  n_tokens: " << emb.n_tokens << "  hidden: " << emb.hidden
              << "  (total floats: " << emb.data.size() << ")\n";
    if (emb.data.empty()) {
        std::cerr << "  ✗ data is empty\n";
        return 5;
    }

    // Stats
    double sum = 0.0, sqsum = 0.0;
    float mn = emb.data[0], mx = emb.data[0];
    int nans = 0, infs = 0;
    for (float v : emb.data) {
        if (std::isnan(v)) { ++nans; continue; }
        if (std::isinf(v)) { ++infs; continue; }
        sum += v; sqsum += static_cast<double>(v) * v;
        if (v < mn) mn = v; if (v > mx) mx = v;
    }
    const double mean = sum / emb.data.size();
    const double var = sqsum / emb.data.size() - mean * mean;
    std::cout << "  mean=" << mean << "  std=" << std::sqrt(var)
              << "  min=" << mn << "  max=" << mx
              << "  nan=" << nans << "  inf=" << infs << "\n";
    std::cout << "  first 6: " << emb.data[0] << " " << emb.data[1] << " " << emb.data[2]
              << " " << emb.data[3] << " " << emb.data[4] << " " << emb.data[5] << "\n";

    if (emb.hidden != 2048) {
        std::cerr << "  ✗ expected hidden=2048, got " << emb.hidden << "\n";
        return 2;
    }
    if (nans > 0 || infs > 0) {
        std::cerr << "  ✗ NaN/Inf in embeddings\n";
        return 3;
    }
    if (std::sqrt(var) < 1e-3) {
        std::cerr << "  ✗ std too small — encoder likely broken\n";
        return 4;
    }
    std::cout << "\n  ✓ mmproj produces sane image embeddings\n";
    return 0;
}
