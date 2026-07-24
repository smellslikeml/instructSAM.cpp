// Quick AR-generation sanity test: prefill on a tiny synthetic prompt,
// greedy-decode 5 tokens, print the ids + pieces. Confirms
// prefill_with_last_hidden + decode_step + logits_for_hidden all wire up
// correctly without waiting on the 4-min 301-token image prefill of the
// full CLI.

#include "sam3/gguf_model.h"
#include "sam3/instructsam_lm_forward.h"
#include "sam3/instructsam_tokenizer.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const std::string lm_gguf = (argc >= 2) ? argv[1]
        : "/tmp/claude-1001/-home-thorax/e5355e82-8c80-4141-8828-424676e4ee8f/scratchpad/instructsam-as-qwen3vl/instructsam-lm-f16.gguf";
    const int32_t max_new = (argc >= 3) ? std::atoi(argv[2]) : 5;

    std::cout << "loading LM (mmap) + tokenizer\n" << std::flush;
    sam3::GgufModel lm;
    if (!lm.load(lm_gguf, false, {}, true)) { std::cerr << "load failed\n"; return 2; }
    sam3::InstructsamLmForward lm_fwd(lm);
    auto tokz = sam3::InstructsamTokenizer::load(lm_gguf);

    // Very short synthetic prompt: "The capital of France is"  (5 tokens).
    const std::string p = "The capital of France is";
    std::vector<int32_t> ptoks;
    {
        // Reuse chat-tokenize path but the raw text tokenize would be cleaner.
        // Since our wrapper only exposes tokenize_chat, use its output ids for
        // "user\n<QUERY>" prefix instead. Even simpler: just embed_for_token
        // over a manually chosen small ID sequence.
        (void)p;
        // Use ids 785 = "The", 6722 = " capital", 315 = " of", 9625 = " France", 374 = " is"
        // (verified against Qwen BPE — these are common tokens).
        ptoks = {785, 6722, 315, 9625, 374};
    }
    const int64_t n_prefix = static_cast<int64_t>(ptoks.size());
    std::cout << "  prompt tokens: ";
    for (auto t : ptoks) std::cout << t << " "; std::cout << "\n";

    const int H = 2048;
    std::vector<float> prefix_embeds(n_prefix * H);
    std::vector<int32_t> pref_pos(n_prefix);
    for (int64_t i = 0; i < n_prefix; ++i) {
        const auto e = lm_fwd.embed_for_token(ptoks[i]);
        std::memcpy(prefix_embeds.data() + i * H, e.data(), H * sizeof(float));
        pref_pos[i] = static_cast<int32_t>(i);
    }

    auto t0 = std::chrono::steady_clock::now();
    auto pfr = lm_fwd.prefill_with_last_hidden(prefix_embeds, n_prefix, pref_pos);
    std::cout << "  ✓ prefill in " << std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t0).count() << "s\n";

    auto cache = std::move(pfr.cache);
    auto cur = std::move(pfr.last_hidden);
    std::vector<int32_t> gen;
    for (int step = 0; step < max_new; ++step) {
        auto ts = std::chrono::steady_clock::now();
        const auto logits = lm_fwd.logits_for_hidden(cur);
        int32_t best = 0; float bv = logits[0];
        for (int32_t v = 1; v < static_cast<int32_t>(logits.size()); ++v)
            if (logits[v] > bv) { bv = logits[v]; best = v; }
        gen.push_back(best);
        const auto piece = tokz->detokenize({best});
        std::cout << "  step " << step << ": token=" << best << " piece=\""
                  << piece << "\" logit=" << bv << " ("
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - ts).count() << "s)\n" << std::flush;
        const auto emb = lm_fwd.embed_for_token(best);
        cur = lm_fwd.decode_step(cache, emb, static_cast<int32_t>(n_prefix + step));
    }
    std::cout << "\n  full generation: \"" << tokz->detokenize(gen) << "\"\n";
    return 0;
}
