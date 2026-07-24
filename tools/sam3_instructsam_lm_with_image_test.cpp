// Day 9c — LM forward with real image context via mmproj + tokenizer.
//
// Chain: tokenize prompt → get image embeds → splice text+image embed prefix
//        → append phrase tokens (<|object_ref_start|>PHRASE<|object_ref_end|>)
//        → extract_seg_output_embeddings_from_prefix
//        → compare vs PyTorch InstructSAM's captured per-object seg_output_embeddings
//
// This is the acid test for whether image conditioning fixes the 0.42
// cos_sim we saw in full_native_e2e (which used only 3-token prompt).

#include "sam3/gguf_model.h"
#include "sam3/instructsam_lm_forward.h"
#include "sam3/instructsam_mmproj.h"
#include "sam3/instructsam_tokenizer.h"

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
    if (std::string(magic, 4) != "BIN1") throw std::runtime_error("bad magic in " + path);
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
    const std::string mmproj = (argc >= 3) ? argv[2]
        : "/tmp/claude-1001/-home-thorax/e5355e82-8c80-4141-8828-424676e4ee8f/scratchpad/instructsam-as-qwen3vl/instructsam-mmproj-f16.gguf";
    const std::string image_path = (argc >= 4) ? argv[3]
        : "/home/thorax/Downloads/warehouse_rgb.jpg";
    const std::string ref_dir = (argc >= 5) ? argv[4]
        : "/tmp/pathA_reference/warehouse_rgb";
    const std::string query = "Please segment the box, the person, the shelf, and the forklift in the image.";
    const std::vector<std::string> phrases = {"box", "person", "shelf", "forklift"};

    // ── Load components ──────────────────────────────────────────────────
    std::cout << "=== stage 0: loading models ===\n" << std::flush;
    auto t0 = std::chrono::steady_clock::now();

    sam3::GgufModel lm_model;
    if (!lm_model.load(lm_gguf, /*prefer_gpu=*/false, /*tensor_map=*/{}, /*use_mmap=*/true)) {
        std::cerr << "LM GGUF load failed\n"; return 2;
    }
    sam3::InstructsamLmForward lm_fwd(lm_model);

    auto tok = sam3::InstructsamTokenizer::load(lm_gguf);
    auto mm  = sam3::InstructsamMmproj::load(mmproj, lm_gguf);

    std::cout << "  ✓ loaded LM, tokenizer, mmproj (" << std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t0).count() << "s)\n\n";

    // ── Tokenize prompt ──────────────────────────────────────────────────
    std::cout << "=== stage 1: tokenize chat prompt ===\n" << std::flush;
    const auto full_tokens = tok->tokenize_chat(query);
    std::cout << "  " << full_tokens.size() << " tokens total\n";

    // Find image_pad token position (there's exactly one in our template)
    const auto & sp = tok->specials();
    size_t image_pad_pos = 0; bool found = false;
    for (size_t i = 0; i < full_tokens.size(); ++i) {
        if (full_tokens[i] == sp.image_pad) { image_pad_pos = i; found = true; break; }
    }
    if (!found) { std::cerr << "no image_pad token\n"; return 3; }
    std::cout << "  <|image_pad|> at position " << image_pad_pos << "\n";

    // ── Get image embeddings via mmproj ──────────────────────────────────
    std::cout << "\n=== stage 2: mmproj — encoding image ===\n" << std::flush;
    t0 = std::chrono::steady_clock::now();
    const auto img_emb = mm->encode_image_file(image_path);
    std::cout << "  ✓ " << img_emb.n_tokens << " image tokens × " << img_emb.hidden
              << " hidden ("
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::steady_clock::now() - t0).count() << " ms)\n";

    // ── Build prefix embed sequence ──────────────────────────────────────
    // [tokens[0..image_pad_pos)] + [image embeds] + [tokens[image_pad_pos+1..end)]
    std::cout << "\n=== stage 3: build prefix embeds ===\n" << std::flush;
    const int H = img_emb.hidden;
    const size_t n_before  = image_pad_pos;
    const size_t n_after   = full_tokens.size() - image_pad_pos - 1;
    const size_t n_img     = static_cast<size_t>(img_emb.n_tokens);
    const size_t n_prefix  = n_before + n_img + n_after;

    std::vector<float> prefix_embeds(n_prefix * H);
    // (a) text before image_pad
    for (size_t i = 0; i < n_before; ++i) {
        const auto e = lm_fwd.embed_for_token(full_tokens[i]);
        std::memcpy(prefix_embeds.data() + i * H, e.data(), H * sizeof(float));
    }
    // (b) image embeds
    std::memcpy(prefix_embeds.data() + n_before * H, img_emb.data.data(), n_img * H * sizeof(float));
    // (c) text after image_pad
    for (size_t i = 0; i < n_after; ++i) {
        const auto e = lm_fwd.embed_for_token(full_tokens[image_pad_pos + 1 + i]);
        std::memcpy(prefix_embeds.data() + (n_before + n_img + i) * H, e.data(), H * sizeof(float));
    }
    std::cout << "  prefix: " << n_before << " text + " << n_img << " image + "
              << n_after << " text = " << n_prefix << " tokens\n";

    // ── Load mask injection constants ────────────────────────────────────
    std::vector<int64_t> mq_s, ms_s, me_s;
    const auto mask_queries = read_bin_f32("/tmp/pathA_reference/instructsam_mask_queries.f32",   mq_s);
    const auto mask_start   = read_bin_f32("/tmp/pathA_reference/instructsam_mask_start_embed.f32", ms_s);
    const auto mask_end     = read_bin_f32("/tmp/pathA_reference/instructsam_mask_end_embed.f32",   me_s);

    // ── Per-phrase: append tokens, run LM, extract seg_output ────────────
    std::cout << "\n=== stage 4: per-phrase LM forward + seg_output ===\n";
    std::vector<double> cos_scores;
    for (size_t p_idx = 0; p_idx < phrases.size(); ++p_idx) {
        const std::string & phrase = phrases[p_idx];
        // Tokenize the phrase (BPE, no special) then wrap with ref_start/ref_end.
        // We tokenize " box" (with leading space) since that's how it appears mid-text.
        // Actually per BPE, phrase tokens depend on preceding text. Simple approach:
        // tokenize as "<|object_ref_start|> PHRASE <|object_ref_end|>" then take
        // all tokens.
        const std::string wrap = std::string("<|object_ref_start|>") + phrase +
                                 std::string("<|object_ref_end|>");
        // Use tokenizer's raw tokenize by treating it as a chat-less string.
        // We don't have a public raw tokenize method, so hack via the chat wrapper
        // is awkward. Simpler: manually build the appended token stream.
        // The impl: <|object_ref_start|> + tokenize(phrase) + <|object_ref_end|>
        // Get phrase-only tokens via re-using tokenize_chat pattern is hard;
        // instead, call llama_tokenize directly via a lightweight path.
        // Workaround: append tokens by inspecting tokenize_chat("query" + wrap)
        // and diffing. Simpler for now: hard-code tokens for the 4 known phrases
        // using known Qwen vocab (box=3745, person=1697, shelf=27645, forklift=369 10561).

        // For correctness across arbitrary phrases we'd need a raw tokenize()
        // API on our wrapper. To keep this test focused, use a small map for
        // our fixed test set (matches captured text_output.txt).
        std::vector<int32_t> phrase_tokens;
        if      (phrase == "box")      phrase_tokens = {3745};
        else if (phrase == "person")   phrase_tokens = {1697};
        else if (phrase == "shelf")    phrase_tokens = {27645};
        else if (phrase == "forklift") phrase_tokens = {369, 10561};
        else { std::cerr << "unknown phrase " << phrase << "\n"; return 5; }

        // Build appended sequence: [<|object_ref_start|>, phrase_tokens..., <|object_ref_end|>]
        std::vector<int32_t> appended;
        appended.push_back(sp.object_ref_start);
        for (int32_t t : phrase_tokens) appended.push_back(t);
        appended.push_back(sp.object_ref_end);

        // Extend prefix_embeds with appended tokens' embeddings
        std::vector<float> ext(prefix_embeds);
        const size_t old_n = n_prefix;
        ext.resize((old_n + appended.size()) * H);
        for (size_t i = 0; i < appended.size(); ++i) {
            const auto e = lm_fwd.embed_for_token(appended[i]);
            std::memcpy(ext.data() + (old_n + i) * H, e.data(), H * sizeof(float));
        }
        const int64_t total_prefix = static_cast<int64_t>(old_n + appended.size());

        std::cout << "  phrase \"" << phrase << "\" (" << phrase_tokens.size()
                  << " subtoken(s)) — running LM forward on "
                  << total_prefix << "+12 injected = "
                  << (total_prefix + 12) << " token seq...\n" << std::flush;
        auto tphr = std::chrono::steady_clock::now();
        const auto seg_out = lm_fwd.extract_seg_output_embeddings_from_prefix(
            ext, total_prefix, mask_queries, mask_start, mask_end);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - tphr).count();

        // Compare vs captured obj{p_idx}/lmb_seg_output_embeddings.f32
        const std::string ref_path = ref_dir + "/binaries_obj" + std::to_string(p_idx) +
                                     "/lmb_seg_output_embeddings.f32";
        std::vector<int64_t> ref_s;
        const auto seg_out_ref = read_bin_f32(ref_path, ref_s);
        const double cs = cosine(seg_out, seg_out_ref);
        cos_scores.push_back(cs);
        std::cout << "    ✓ obj" << p_idx << " (\"" << phrase << "\") cos_sim vs PyTorch = "
                  << cs << "   (" << ms << " ms)\n";
    }

    std::cout << "\n=== summary ===\n";
    double sum = 0.0;
    for (size_t i = 0; i < cos_scores.size(); ++i) {
        std::cout << "  obj" << i << " (\"" << phrases[i] << "\") : cos_sim = "
                  << cos_scores[i] << "\n";
        sum += cos_scores[i];
    }
    const double mean = sum / cos_scores.size();
    std::cout << "  mean cos_sim = " << mean << "\n";
    if (mean > 0.9) {
        std::cout << "\n  ✓ image conditioning yields high parity — LM path is functional\n";
    } else if (mean > 0.7) {
        std::cout << "\n  ~ moderate parity — some divergence, may be phrase-token quantization\n";
    } else {
        std::cout << "\n  ✗ still-low parity — check image_pad substitution / order / dtypes\n";
    }
    return 0;
}
