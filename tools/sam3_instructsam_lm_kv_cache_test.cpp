// KV cache validation: run extract_seg_output_embeddings on a modest
// prefix via both paths (no-cache single forward, prefill+cache decode),
// verify cos_sim ≈ 1.0, and time both.
//
// Kept small (~40 token prefix, 15-token delta) so it finishes in
// minutes on CPU rather than hours.

#include "sam3/gguf_model.h"
#include "sam3/instructsam_lm_forward.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string lm_gguf = (argc >= 2) ? argv[1]
        : "/tmp/claude-1001/-home-thorax/e5355e82-8c80-4141-8828-424676e4ee8f/scratchpad/instructsam-as-qwen3vl/instructsam-lm-f16.gguf";

    std::cout << "loading LM (mmap) from " << lm_gguf << "\n" << std::flush;
    sam3::GgufModel lm;
    if (!lm.load(lm_gguf, /*prefer_gpu=*/false, /*tensor_map=*/{}, /*use_mmap=*/true)) {
        std::cerr << "load failed\n"; return 2;
    }
    sam3::InstructsamLmForward lm_fwd(lm);

    // Constants
    std::vector<int64_t> mq_s, ms_s, me_s;
    const auto mask_queries = read_bin_f32("/tmp/pathA_reference/instructsam_mask_queries.f32",   mq_s);
    const auto mask_start   = read_bin_f32("/tmp/pathA_reference/instructsam_mask_start_embed.f32", ms_s);
    const auto mask_end     = read_bin_f32("/tmp/pathA_reference/instructsam_mask_end_embed.f32",   me_s);
    std::cout << "  ✓ loaded mask_queries/start/end\n\n";

    // Build a modest 40-token synthetic prefix (all token 100 for simplicity)
    // + 3 appended-prefix tokens. Keeps the test tractable (~40+3+12 = 55 total
    // seq for the no-cache path, prefill 40 + decode 15 for cache path).
    const int H = 2048;
    const int64_t n_prefix = 40;
    const int64_t n_appended_prefix = 3;  // e.g. object_ref_start, phrase, object_ref_end
    const int64_t total = n_prefix + n_appended_prefix + 12;

    std::vector<float> prefix_embeds(n_prefix * H);
    for (int64_t i = 0; i < n_prefix; ++i) {
        const auto e = lm_fwd.embed_for_token(100 + static_cast<int32_t>(i));  // tokens 100..139
        std::memcpy(prefix_embeds.data() + i * H, e.data(), H * sizeof(float));
    }
    std::vector<float> appended_prefix_embeds(n_appended_prefix * H);
    // Use tokens 151646 (object_ref_start), 8912 (arbitrary phrase token), 151647 (object_ref_end)
    const int32_t ap_toks[3] = {151646, 8912, 151647};
    for (int64_t i = 0; i < n_appended_prefix; ++i) {
        const auto e = lm_fwd.embed_for_token(ap_toks[i]);
        std::memcpy(appended_prefix_embeds.data() + i * H, e.data(), H * sizeof(float));
    }

    // ── Path A: no-cache — one full forward on total = 55 tokens ────────
    std::cout << "=== path A: no-cache (single full forward on " << total << " tokens) ===\n" << std::flush;
    auto tA = std::chrono::steady_clock::now();
    std::vector<float> combined_prefix(n_prefix * H + n_appended_prefix * H);
    std::memcpy(combined_prefix.data(), prefix_embeds.data(), n_prefix * H * sizeof(float));
    std::memcpy(combined_prefix.data() + n_prefix * H, appended_prefix_embeds.data(),
                n_appended_prefix * H * sizeof(float));
    const auto seg_A = lm_fwd.extract_seg_output_embeddings_from_prefix(
        combined_prefix, n_prefix + n_appended_prefix, mask_queries, mask_start, mask_end);
    const auto ms_A = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - tA).count();
    std::cout << "  ✓ done in " << ms_A << " ms\n\n";

    // ── Path B: prefill prefix + decode appended+injection ──────────────
    std::cout << "=== path B: prefill (" << n_prefix << ") + decode ("
              << (n_appended_prefix + 12) << ") ===\n" << std::flush;
    auto tB0 = std::chrono::steady_clock::now();
    std::vector<int32_t> pref_pos(n_prefix);
    for (int64_t i = 0; i < n_prefix; ++i) pref_pos[i] = static_cast<int32_t>(i);
    auto cache = lm_fwd.prefill_prefix(prefix_embeds, n_prefix, pref_pos);
    const auto ms_prefill = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - tB0).count();

    auto tB1 = std::chrono::steady_clock::now();
    const auto seg_B = lm_fwd.extract_seg_output_embeddings_with_cache(
        cache, appended_prefix_embeds, n_appended_prefix, mask_queries, mask_start, mask_end);
    const auto ms_decode = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - tB1).count();
    std::cout << "  ✓ prefill " << ms_prefill << " ms + decode " << ms_decode << " ms = "
              << (ms_prefill + ms_decode) << " ms\n\n";

    // ── Compare ──────────────────────────────────────────────────────────
    const double cs = cosine(seg_A, seg_B);
    double maxdiff = 0.0;
    for (size_t i = 0; i < seg_A.size(); ++i) {
        const double d = std::fabs(seg_A[i] - seg_B[i]);
        if (d > maxdiff) maxdiff = d;
    }
    std::cout << "=== parity ===\n";
    std::cout << "  cos_sim(A, B) : " << cs << "\n";
    std::cout << "  max_diff      : " << maxdiff << "\n";
    std::cout << "  seg_A[0..5]   : ";
    for (int i = 0; i < 6; ++i) std::cout << seg_A[i] << " "; std::cout << "\n";
    std::cout << "  seg_B[0..5]   : ";
    for (int i = 0; i < 6; ++i) std::cout << seg_B[i] << " "; std::cout << "\n";

    // ── Second phrase decode (proves cache reuse) ────────────────────────
    std::cout << "\n=== second phrase decode (cache reuse) ===\n";
    std::vector<float> ap2(n_appended_prefix * H);
    const int32_t ap2_toks[3] = {151646, 8913, 151647};  // different phrase
    for (int64_t i = 0; i < n_appended_prefix; ++i) {
        const auto e = lm_fwd.embed_for_token(ap2_toks[i]);
        std::memcpy(ap2.data() + i * H, e.data(), H * sizeof(float));
    }
    auto tB2 = std::chrono::steady_clock::now();
    const auto seg_B2 = lm_fwd.extract_seg_output_embeddings_with_cache(
        cache, ap2, n_appended_prefix, mask_queries, mask_start, mask_end);
    const auto ms_decode2 = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - tB2).count();
    std::cout << "  ✓ decode phrase 2 in " << ms_decode2
              << " ms (cache reused, no re-prefill)\n";
    std::cout << "  seg_B2[0..3]  : " << seg_B2[0] << " " << seg_B2[1] << " "
              << seg_B2[2] << " " << seg_B2[3] << "\n";

    // Verdict
    std::cout << "\n=== verdict ===\n";
    if (cs > 0.9999 && maxdiff < 5e-3) {
        std::cout << "  ✓ KV cache produces numerically identical output to no-cache path\n";
        std::cout << "  speedup on decode = " << (double)ms_A / ms_decode << "x\n";
        return 0;
    } else {
        std::cerr << "  ✗ divergence: cos_sim=" << cs << " max_diff=" << maxdiff << "\n";
        return 3;
    }
}
