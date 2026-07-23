// Reference dump: run our test 4-token sequence [ref_start, "test-token",
// ref_end, mask_start] through Path B's llama.cpp Qwen3-VL LM in token
// mode with embeddings=true. Save the final hidden state at each position
// to a .f32 binary for cross-validation against our C++ Qwen3 fork
// (sam3-instructsam-lm-forward-test).

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::cerr << "usage: instructsam-lm-reference-dump <lm.gguf> <out.f32>\n";
        return 1;
    }

    common_params params;
    params.model.path = argv[1];
    params.n_ctx = 512;
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_outputs_max = 512;
    params.embedding = true;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    params.warmup = false;
    params.n_parallel = 1;
    params.kv_unified = true;

    common_init();
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { std::cerr << "load failed\n"; return 2; }
    const int32_t n_embd = llama_model_n_embd(model);

    // Same 4 tokens our forward-test uses
    const std::vector<llama_token> toks = {151646, 9707, 151647, 151671};

    llama_batch b = llama_batch_init(toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i]; b.pos[i] = (llama_pos)i;
        b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = toks.size();
    if (llama_decode(ctx, b) < 0) { std::cerr << "decode failed\n"; return 3; }

    // Dump hidden state for each of the 4 positions
    std::vector<float> all_hidden(toks.size() * n_embd);
    for (size_t i = 0; i < toks.size(); ++i) {
        const float * hs = llama_get_embeddings_ith(ctx, static_cast<int32_t>(i));
        if (!hs) { std::cerr << "get_embeddings_ith null at " << i << "\n"; return 4; }
        std::memcpy(all_hidden.data() + i * n_embd, hs, n_embd * sizeof(float));
        std::cout << "pos " << i << " hidden[0..3]: "
                  << hs[0] << " " << hs[1] << " " << hs[2] << " " << hs[3] << "\n";
    }

    std::ofstream out(argv[2], std::ios::binary);
    out.write("BIN1", 4);
    int32_t ndim = 2; out.write((const char*)&ndim, 4);
    int64_t d0 = toks.size(), d1 = n_embd;
    out.write((const char*)&d0, 8); out.write((const char*)&d1, 8);
    out.write((const char*)all_hidden.data(), all_hidden.size() * sizeof(float));
    std::cout << "\nwrote " << all_hidden.size() << " floats = [" << d0 << ", " << d1 << "] to " << argv[2] << "\n";

    llama_batch_free(b);
    return 0;
}
