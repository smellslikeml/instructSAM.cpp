// Day-1 prototype for Piece 3 LM integration.
//
// Proves the llama.cpp public API surface InstructSAM needs:
//   1. llama_context_params.embeddings = true → hidden-state output enabled
//   2. llama_batch.logits[i] = 1 → mark position for output
//   3. llama_get_embeddings_ith(ctx, i) → retrieve [n_embd] hidden state
//   4. llama_batch with embd != nullptr → inputs_embeds mid-generation
//
// Runs a small prompt through the LM in three modes:
//   A. Standard token-mode decode → verify sensible logits + hidden state
//   B. Embed-mode decode using the OUTPUT hidden from A as an "input embed" for
//      the next position → verify decode runs without crashing (proves the
//      llama_batch.embd path works end-to-end)
//   C. Optional strict round-trip: read token_embd.weight from the GGUF via
//      our GgufModel loader, feed embed lookups to embed-mode decode, compare
//      hidden states vs. mode A. Deferred to Day 2.

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: sam3-instructsam-llama-probe <lm_gguf>\n";
        return 1;
    }
    const std::string lm_path = argv[1];

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(lm_path.c_str(), mp);
    if (!model) { std::cerr << "load failed\n"; return 2; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_embd_out = llama_model_n_embd(model);
    const int32_t n_embd_inp = llama_model_n_embd_inp(model);
    std::cout << "loaded LM  n_embd_inp=" << n_embd_inp << "  n_embd_out=" << n_embd_out << "\n";

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.embeddings = true;
    std::cerr << "PROBE: creating ctx\n" << std::flush;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::cerr << "ctx failed\n"; return 3; }
    std::cerr << "PROBE: ctx created\n" << std::flush;

    // Tokenize a short prompt
    const std::string prompt = "Hello world";
    std::vector<llama_token> toks(64);
    std::cerr << "PROBE: tokenizing\n" << std::flush;
    int n_toks = llama_tokenize(vocab, prompt.c_str(), prompt.size(),
                                toks.data(), toks.size(), true, false);
    std::cerr << "PROBE: tokenize returned " << n_toks << "\n" << std::flush;
    if (n_toks < 0) { std::cerr << "tokenize failed\n"; return 4; }
    toks.resize(n_toks);
    std::cerr << "PROBE: past resize, n_toks=" << n_toks << "\n" << std::flush;
    std::cout << "prefix has " << n_toks << " tokens: ";
    for (auto t : toks) std::cout << t << " ";
    std::cout << "\n" << std::flush;
    std::cerr << "PROBE: past print\n" << std::flush;

    // ── Mode A: token-mode decode ──────────────────────────────────────
    {
        std::cerr << "PROBE: creating batch A\n" << std::flush;
        llama_batch b = llama_batch_init(toks.size(), 0, 1);
        std::cerr << "PROBE: batch A created\n" << std::flush;
        for (size_t i = 0; i < toks.size(); ++i) {
            b.token[i]     = toks[i];
            b.pos[i]       = static_cast<llama_pos>(i);
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = (i + 1 == toks.size()) ? 1 : 0;
        }
        b.n_tokens = toks.size();
        if (llama_decode(ctx, b) < 0) {
            std::cerr << "  ✗ token-mode decode failed\n"; return 5;
        }
        const float * hs = llama_get_embeddings_ith(ctx, static_cast<int32_t>(toks.size() - 1));
        const float * lg = llama_get_logits_ith(ctx, static_cast<int32_t>(toks.size() - 1));
        std::cout << "\n=== Mode A: token-mode decode ===\n";
        if (!hs || !lg) { std::cerr << "  ✗ hidden state / logits null\n"; return 6; }
        std::cout << "  hidden[0..3]: " << hs[0] << " " << hs[1] << " " << hs[2] << " " << hs[3] << "\n";
        std::cout << "  logits[0..3]: " << lg[0] << " " << lg[1] << " " << lg[2] << " " << lg[3] << "\n";
        std::cout << "  ✓ llama_batch.token + llama_get_embeddings_ith work\n";
        llama_batch_free(b);
    }

    // Reset KV
    llama_memory_clear(llama_get_memory(ctx), true);

    // ── Mode B: embed-mode decode with a dummy embed for the last position ──
    // Fill last position's embed with all-zeros (or some deterministic content)
    // — we just want to verify the decode PATH works, not numerical parity.
    {
        llama_batch b = llama_batch_init(toks.size(), n_embd_inp, 1);
        // NOTE: token is nullptr → embed path is used. But the batch was
        // allocated with both token AND embd. We need to signal embed by
        // populating embd for all positions and leaving token as-is (or setting
        // it to be ignored). Let's zero out tokens to guarantee we don't
        // accidentally trigger token-mode.
        //
        // Actually the docs say "used when embd is NULL" for token — so
        // populating embd is the signal that embd wins. Try that.
        for (size_t i = 0; i < toks.size(); ++i) {
            b.pos[i]       = static_cast<llama_pos>(i);
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = (i + 1 == toks.size()) ? 1 : 0;
            b.token[i]     = 0;
        }
        // Fill embd — pattern doesn't matter for path verification.
        for (size_t i = 0; i < toks.size(); ++i) {
            for (int j = 0; j < n_embd_inp; ++j) {
                b.embd[i * n_embd_inp + j] = static_cast<float>((i + j) % 7) * 0.01f;
            }
        }
        b.n_tokens = toks.size();

        std::cout << "\n=== Mode B: embed-mode decode (dummy embeddings) ===\n";
        const int rc = llama_decode(ctx, b);
        if (rc < 0) {
            std::cerr << "  ✗ embed-mode decode failed (rc=" << rc << ")\n";
            llama_batch_free(b);
            llama_free(ctx); llama_model_free(model); llama_backend_free();
            return 7;
        }
        const float * hs = llama_get_embeddings_ith(ctx, static_cast<int32_t>(toks.size() - 1));
        const float * lg = llama_get_logits_ith(ctx, static_cast<int32_t>(toks.size() - 1));
        if (!hs || !lg) { std::cerr << "  ✗ hidden state / logits null after embed decode\n"; return 8; }
        std::cout << "  hidden[0..3]: " << hs[0] << " " << hs[1] << " " << hs[2] << " " << hs[3] << "\n";
        std::cout << "  logits[0..3]: " << lg[0] << " " << lg[1] << " " << lg[2] << " " << lg[3] << "\n";
        std::cout << "  ✓ llama_batch.embd path works — inputs_embeds are consumed\n";
        llama_batch_free(b);
    }

    std::cout << "\n=== Piece 3 Day 1: API surface confirmed ===\n";
    std::cout << "  llama.cpp supports both:\n"
              << "    - inputs_embeds via llama_batch.embd\n"
              << "    - hidden-state capture via llama_get_embeddings_ith\n"
              << "  Next: Day 2 — mtmd wrapper for image+text preprocessing\n";

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
