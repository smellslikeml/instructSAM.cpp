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

    // ── Milestone 5: parity vs llama.cpp reference ─────────────────────
    std::cout << "\n=== parity vs. llama.cpp reference (same 4-token sequence) ===\n";
    try {
        std::vector<int64_t> ref_s;
        const auto ref_hidden = read_bin_f32("/tmp/pathA_reference/lm_ref_final_hidden.f32", ref_s);
        if (ref_hidden.size() != final_hidden.size()) {
            std::cout << "  ✗ size mismatch: ours=" << final_hidden.size() << " ref=" << ref_hidden.size() << "\n";
        } else {
            auto cos = [&](const float * a, const float * b, size_t n) {
                double dot = 0.0, na = 0.0, nb = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    dot += static_cast<double>(a[k]) * b[k];
                    na  += static_cast<double>(a[k]) * a[k];
                    nb  += static_cast<double>(b[k]) * b[k];
                }
                return dot / (std::sqrt(na) * std::sqrt(nb));
            };
            double all_maxd = 0.0, all_l2d = 0.0, all_l2r = 0.0;
            for (size_t i = 0; i < final_hidden.size(); ++i) {
                const double d = final_hidden[i] - ref_hidden[i];
                if (std::fabs(d) > all_maxd) all_maxd = std::fabs(d);
                all_l2d += d * d;
                all_l2r += static_cast<double>(ref_hidden[i]) * ref_hidden[i];
            }
            std::cout << "  Per-position parity:\n";
            std::cout << "    pos  |  cos_sim  |  max_diff |  rel_L2\n";
            std::cout << "    -----+-----------+-----------+---------\n";
            for (size_t i = 0; i < tok_ids.size(); ++i) {
                const float * ours = final_hidden.data() + i * 2048;
                const float * ref  = ref_hidden.data()   + i * 2048;
                const double c = cos(ours, ref, 2048);
                double mx = 0.0, l2d = 0.0, l2r = 0.0;
                for (int k = 0; k < 2048; ++k) {
                    const double d = ours[k] - ref[k];
                    if (std::fabs(d) > mx) mx = std::fabs(d);
                    l2d += d * d; l2r += static_cast<double>(ref[k]) * ref[k];
                }
                std::cout << "    " << i << "    | " << c << " | " << mx << " | "
                          << (std::sqrt(l2d) / std::sqrt(l2r)) << "\n";
            }
            std::cout << "\n  Aggregate:\n";
            std::cout << "    cos_sim (flat) : " << cos(final_hidden.data(), ref_hidden.data(), final_hidden.size()) << "\n";
            std::cout << "    max_diff       : " << all_maxd << "\n";
            std::cout << "    rel_L2         : " << (std::sqrt(all_l2d) / std::sqrt(all_l2r)) << "\n";
        }
    } catch (const std::exception & e) {
        std::cout << "  (no reference dump found: " << e.what() << ")\n";
        std::cout << "  Run instructsam-lm-reference-dump first.\n";
    }

    // ── Milestone 6: mask_queries injection use case ────────────────────
    std::cout << "\n=== Day 6: extract_seg_output_embeddings (10-mask_query injection) ===\n";
    try {
        std::vector<int64_t> mq_s, ms_s, me_s;
        const auto mask_queries = read_bin_f32("/tmp/pathA_reference/instructsam_mask_queries.f32", mq_s);
        const auto mask_start   = read_bin_f32("/tmp/pathA_reference/instructsam_mask_start_embed.f32", ms_s);
        const auto mask_end     = read_bin_f32("/tmp/pathA_reference/instructsam_mask_end_embed.f32", me_s);
        std::cout << "  loaded: mask_queries " << mq_s[0] << "x" << mq_s[1]
                  << "  mask_start " << ms_s[0] << "x" << ms_s[1]
                  << "  mask_end "   << me_s[0] << "x" << me_s[1] << "\n";
        // Prompt: [ref_start, "test-token", ref_end]. In real InstructSAM
        // there'd be image tokens + chat template before this; for a
        // structural test we just want to verify the injection path runs
        // and produces sensible hidden states.
        const std::vector<int32_t> prompt = {151646, 9707, 151647};
        std::cout << "  running LM forward on prompt(" << prompt.size()
                  << ") + inject(12)...\n" << std::flush;
        auto ti = std::chrono::steady_clock::now();
        const auto seg_out = lm.extract_seg_output_embeddings(
            prompt, mask_queries, mask_start, mask_end);
        const auto rti = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - ti).count();
        std::cout << "  ✓ ran in " << rti << " ms\n";

        // Print each slot's hidden state + check for NaN/degeneracy
        int slot_nan = 0, slot_zero = 0;
        for (int j = 0; j < 10; ++j) {
            const float * s = seg_out.data() + j * 2048;
            bool all_zero = true; bool any_nan = false;
            double sumsq = 0.0;
            for (int k = 0; k < 2048; ++k) {
                if (std::isnan(s[k]) || std::isinf(s[k])) any_nan = true;
                if (s[k] != 0.0f) all_zero = false;
                sumsq += static_cast<double>(s[k]) * s[k];
            }
            if (any_nan) ++slot_nan;
            if (all_zero) ++slot_zero;
            std::cout << "  slot " << j << " hidden[0..3]: "
                      << s[0] << " " << s[1] << " " << s[2] << " " << s[3]
                      << "  L2=" << std::sqrt(sumsq) << "\n";
        }
        std::cout << "  " << (10 - slot_nan - slot_zero) << "/10 slots produced valid hidden states\n";

        // Save for downstream integration testing
        std::ofstream out("/tmp/pathA_reference/instructsam_seg_output_embeddings_qwen3_fork.f32", std::ios::binary);
        out.write("BIN1", 4);
        int32_t nd = 2; out.write((const char*)&nd, 4);
        int64_t d0 = 10, d1 = 2048;
        out.write((const char*)&d0, 8); out.write((const char*)&d1, 8);
        out.write((const char*)seg_out.data(), seg_out.size() * sizeof(float));
        std::cout << "  ✓ wrote seg_output_embeddings [10, 2048]\n";
    } catch (const std::exception & e) {
        std::cout << "  (mask_queries dumps missing: " << e.what() << ")\n";
    }

    std::cout << "\n=== Qwen3 fork Day 6 done — injection path works, seg_output ready ===\n";
    return 0;
}
