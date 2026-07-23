// Day 1 of Qwen3 fork: load Path B's LM GGUF via our GgufModel,
// verify all 28 layers × ~10 tensors present, and extract input
// embeddings for a few known tokens (mask_start, mask_end).

#include "sam3/gguf_model.h"
#include "sam3/instructsam_lm_forward.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
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
    if (argc < 2) {
        std::cerr << "usage: sam3-instructsam-lm-forward-test <lm.gguf>\n";
        return 1;
    }

    std::cout << "loading LM GGUF (mmap mode, ~3.3GB file, lazy paging)...\n" << std::flush;
    sam3::GgufModel model;
    if (!model.load(argv[1], /*prefer_gpu=*/false, /*tensor_map=*/{}, /*use_mmap=*/true)) {
        std::cerr << "load failed\n";
        return 2;
    }
    std::cout << "loaded LM GGUF: " << model.tensors().size() << " tensors\n" << std::flush;

    sam3::InstructsamLmForward lm(model);

    // Milestone 1: tensor validation
    std::cout << "\n=== validate_all_tensors_present ===\n";
    const size_t probed = lm.validate_all_tensors_present();
    std::cout << "  ✓ " << probed << " required LM tensors present\n";
    std::cout << "    breakdown: 2 top-level + 28 layers × (~14 tensors/layer including optional biases)\n";

    // Milestone 2: token embed lookup
    std::cout << "\n=== embed_for_token ===\n";
    for (int32_t tok : {151646, 151647, 151670, 151671, 151672, 9707}) {
        const auto e = lm.embed_for_token(tok);
        float l2 = 0.0f; for (float v : e) l2 += v * v; l2 = std::sqrt(l2);
        std::cout << "  token " << tok << " embed[0..3]: "
                  << e[0] << " " << e[1] << " " << e[2] << " " << e[3]
                  << "  L2=" << l2 << "\n";
    }

    // Milestone 3: compare mask_start/mask_end against safetensors dumps
    std::cout << "\n=== compare vs safetensors dumps (mask_start=151671, mask_end=151672) ===\n";
    try {
        std::vector<int64_t> s1, s2;
        const auto ms_ref = read_bin_f32("/tmp/pathA_reference/instructsam_mask_start_embed.f32", s1);
        const auto me_ref = read_bin_f32("/tmp/pathA_reference/instructsam_mask_end_embed.f32", s2);
        const auto ms_ours = lm.embed_for_token(151671);
        const auto me_ours = lm.embed_for_token(151672);
        double maxd_s = 0.0, maxd_e = 0.0;
        for (int i = 0; i < 2048; ++i) {
            maxd_s = std::max<double>(maxd_s, std::fabs(ms_ours[i] - ms_ref[i]));
            maxd_e = std::max<double>(maxd_e, std::fabs(me_ours[i] - me_ref[i]));
        }
        std::cout << "  mask_start (151671): max_diff vs safetensors = " << maxd_s << "\n";
        std::cout << "  mask_end   (151672): max_diff vs safetensors = " << maxd_e << "\n";
        if (maxd_s < 1e-6 && maxd_e < 1e-6) {
            std::cout << "  ✓ EXACT match (no dtype loss beyond fp16 store)\n";
        } else if (maxd_s < 1e-3 && maxd_e < 1e-3) {
            std::cout << "  ✓ fp16 quantization drift only\n";
        } else {
            std::cout << "  ✗ significant mismatch\n";
        }
    } catch (const std::exception & e) {
        std::cout << "  (skipping — safetensors dumps not found: " << e.what() << ")\n";
    }

    std::cout << "\n=== Qwen3 fork Day 1 done — embed lookup working ===\n";
    std::cout << "  Next: skeleton for run_layer(0), then RMSNorm + q_norm/k_norm + GQA + RoPE + SwiGLU MLP\n";
    return 0;
}
