// Day-1 probe for Piece 3 LM integration — v2, uses common_init_from_params.
#include "common.h"
#include "llama.h"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: instructsam-llama-probe <lm_gguf>\n";
        return 1;
    }

    common_params params;
    params.model.path = argv[1];
    params.n_ctx = 512;
    params.n_batch = 512;
    params.embedding = true;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    params.warmup = false;
    params.n_parallel = 1;
    params.kv_unified = true;

    common_init();

    auto llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (!model || !ctx) { std::cerr << "load failed\n"; return 2; }
    const int32_t n_embd = llama_model_n_embd(model);
    std::cout << "loaded LM, n_embd=" << n_embd << "\n" << std::flush;

    std::vector<llama_token> toks = common_tokenize(ctx, "Hello world", true, true);
    std::cout << "prefix has " << toks.size() << " tokens\n" << std::flush;

    // Mode A: token-mode decode
    {
        llama_batch b = llama_batch_init(toks.size(), 0, 1);
        for (size_t i = 0; i < toks.size(); ++i) {
            b.token[i] = toks[i]; b.pos[i] = i;
            b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        b.n_tokens = toks.size();
        std::cout << "\n=== Mode A: decode via token IDs ===\n" << std::flush;
        int rc = llama_decode(ctx, b);
        if (rc < 0) { std::cerr << "  x decode rc=" << rc << "\n"; return 5; }
        const float * hs = llama_get_embeddings_ith(ctx, toks.size() - 1);
        if (!hs) { std::cerr << "  x get_embeddings null\n"; return 6; }
        std::cout << "  hidden[0..3]: " << hs[0] << " " << hs[1] << " " << hs[2] << " " << hs[3] << "\n";
        std::cout << "  OK: token + get_embeddings_ith work\n";
        llama_batch_free(b);
    }

    llama_memory_clear(llama_get_memory(ctx), true);

    // Mode B: embed-mode decode with dummy embds
    {
        llama_batch b = llama_batch_init(toks.size(), n_embd, 1);
        for (size_t i = 0; i < toks.size(); ++i) {
            b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
            for (int j = 0; j < n_embd; ++j) {
                b.embd[i * n_embd + j] = static_cast<float>((i + j) % 7) * 0.01f;
            }
        }
        b.n_tokens = toks.size();
        std::cout << "\n=== Mode B: decode via embeddings ===\n" << std::flush;
        int rc = llama_decode(ctx, b);
        if (rc < 0) { std::cerr << "  x embed decode rc=" << rc << "\n"; return 7; }
        const float * hs = llama_get_embeddings_ith(ctx, toks.size() - 1);
        if (!hs) { std::cerr << "  x get_embeddings null\n"; return 8; }
        std::cout << "  hidden[0..3]: " << hs[0] << " " << hs[1] << " " << hs[2] << " " << hs[3] << "\n";
        std::cout << "  OK: llama_batch.embd path works\n";
        llama_batch_free(b);
    }

    std::cout << "\n=== Piece 3 Day 1: API SURFACE CONFIRMED ===\n";
    return 0;
}
