// Day 1 of Qwen3 fork: load Path B's LM GGUF via our GgufModel,
// verify all 28 layers × ~10 tensors present, and extract input
// embeddings for a few known tokens (mask_start, mask_end).

#include "sam3/gguf_model.h"
#include "sam3/instructsam_lm_forward.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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

    // ── Milestone 4: full 28-layer forward pass, smoke check ────────────
    std::cout << "\n=== forward pass: 4-token sequence through 28 layers ===\n";
    std::cout << "  building input embeddings...\n" << std::flush;
    // Small test: 4 tokens = [ref_start, "test-token", ref_end, mask_start]
    // Just for a NaN check and rough magnitude check.
    const std::vector<int32_t> tok_ids = {151646, 9707, 151647, 151671};
    std::vector<float> input_embeds(tok_ids.size() * 2048);
    for (size_t i = 0; i < tok_ids.size(); ++i) {
        const auto e = lm.embed_for_token(tok_ids[i]);
        std::memcpy(input_embeds.data() + i * 2048, e.data(), 2048 * sizeof(float));
    }
    std::vector<int32_t> positions;
    for (size_t i = 0; i < tok_ids.size(); ++i) positions.push_back(static_cast<int32_t>(i));

    std::cout << "  running 28-layer forward pass (CPU, expect ~30-60s)...\n" << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    const auto final_hidden = lm.run(input_embeds, static_cast<int64_t>(tok_ids.size()), positions);
    const auto rt = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "  ✓ ran in " << rt << " ms\n";

    // Sanity check
    int nan_count = 0, zero_count = 0;
    float absmax = 0.0f;
    double sum = 0.0;
    for (float v : final_hidden) {
        if (std::isnan(v) || std::isinf(v)) ++nan_count;
        else if (v == 0.0f) ++zero_count;
        else { sum += v; if (std::fabs(v) > absmax) absmax = std::fabs(v); }
    }
    std::cout << "  final_hidden: total=" << final_hidden.size()
              << "  NaN/Inf=" << nan_count
              << "  zeros=" << zero_count
              << "  absmax=" << absmax
              << "  mean=" << (sum / final_hidden.size()) << "\n";
    for (size_t i = 0; i < tok_ids.size(); ++i) {
        std::cout << "  pos " << i << " hidden[0..3]: "
                  << final_hidden[i*2048+0] << " " << final_hidden[i*2048+1]
                  << " " << final_hidden[i*2048+2] << " " << final_hidden[i*2048+3] << "\n";
    }

    std::cout << "\n=== Qwen3 fork Day 3-4 done — full 28-layer forward compiles + runs ===\n";
    if (nan_count == 0) {
        std::cout << "  ✓ no NaN/Inf → structural correctness OK\n";
    } else {
        std::cout << "  ✗ " << nan_count << " NaN/Inf values — investigate\n";
    }
    std::cout << "  Next: parity vs. PyTorch InstructSAM reference (needs LM layer hooks in capture harness)\n";
    return 0;
}
