// Day-2 probe v2: inject mask_queries ONE-AT-A-TIME to work around
// llama.cpp's per-decode n_outputs limitation.
#include "common.h"
#include "llama.h"

#include <cstdio>
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
    if (argc < 5) {
        std::cerr << "usage: instructsam-inject-probe <lm.gguf> <mask_queries.f32> "
                     "<mask_start_embed.f32> <mask_end_embed.f32>\n";
        return 1;
    }

    std::vector<int64_t> mq_s, ms_s, me_s;
    const auto mq = read_bin_f32(argv[2], mq_s);
    const auto ms = read_bin_f32(argv[3], ms_s);
    const auto me = read_bin_f32(argv[4], me_s);
    std::cout << "mask_queries " << mq_s[0] << "x" << mq_s[1]
              << "  mask_start " << ms_s[0] << "x" << ms_s[1]
              << "  mask_end "   << me_s[0] << "x" << me_s[1] << "\n";

    common_params params;
    params.model.path = argv[1];
    params.n_ctx = 2048;
    params.n_batch = 2048;
    params.n_ubatch = 2048;
    params.n_outputs_max = 512;
    params.embedding = true;
    params.pooling_type = LLAMA_POOLING_TYPE_NONE;
    params.warmup = false;
    params.n_parallel = 1;
    params.kv_unified = true;
    // NOTE: flash_attn=off + one-at-a-time embed decode returns all zeros
    // (KV cache doesn't accumulate across separate decode calls when
    // switching between token-mode and embed-mode). Keep it enabled.

    common_init();
    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { std::cerr << "load failed\n"; return 2; }
    const int32_t n_embd = llama_model_n_embd(model);
    std::cout << "loaded LM, n_embd=" << n_embd << "\n";

    const std::string prompt = "The following is a segmentation task.";
    std::vector<llama_token> toks = common_tokenize(ctx, prompt, true, true);
    std::cout << "prompt has " << toks.size() << " tokens\n";

    // Decode prompt
    {
        llama_batch b = llama_batch_init(toks.size(), 0, 1);
        for (size_t i = 0; i < toks.size(); ++i) {
            b.token[i] = toks[i]; b.pos[i] = (llama_pos)i;
            b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        b.n_tokens = toks.size();
        if (llama_decode(ctx, b) < 0) { std::cerr << "prompt decode failed\n"; return 4; }
        llama_batch_free(b);
    }
    int32_t n_past = (int32_t)toks.size();

    // Feed ref_end normally
    const llama_token ref_end = 151647;
    {
        llama_batch b = llama_batch_init(1, 0, 1);
        b.token[0] = ref_end; b.pos[0] = n_past;
        b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
        b.n_tokens = 1;
        if (llama_decode(ctx, b) < 0) { std::cerr << "ref_end decode failed\n"; return 5; }
        llama_batch_free(b);
    }
    n_past += 1;
    std::cout << "\n=== injecting mask_queries ONE-BY-ONE (single-position decodes) ===\n";

    // Helper: decode a single embed at a given position, return hidden state
    auto decode_single_embed = [&](const float * embd_ptr) -> std::vector<float> {
        llama_batch eb = llama_batch_init(1, n_embd, 1);
        eb.pos[0] = n_past;
        eb.n_seq_id[0] = 1; eb.seq_id[0][0] = 0;
        eb.logits[0] = 1;
        std::memcpy(eb.embd, embd_ptr, n_embd * sizeof(float));
        eb.n_tokens = 1;
        int rc = llama_decode(ctx, eb);
        std::vector<float> out;
        if (rc == 0) {
            const float * hs = llama_get_embeddings_ith(ctx, 0);
            if (hs) {
                out.resize(n_embd);
                std::memcpy(out.data(), hs, n_embd * sizeof(float));
            }
        }
        llama_batch_free(eb);
        return out;
    };

    // Inject mask_start
    {
        auto hs = decode_single_embed(ms.data());
        if (hs.empty()) { std::cerr << "  x mask_start hs empty\n"; return 6; }
        std::cout << "  mask_start hidden[0..3]: " << hs[0] << " " << hs[1] << " " << hs[2] << " " << hs[3] << "\n";
        n_past += 1;
    }

    // Inject 10 mask_queries and capture hidden state per slot
    std::vector<float> seg_out(10 * n_embd);
    for (int j = 0; j < 10; ++j) {
        auto hs = decode_single_embed(mq.data() + j * n_embd);
        if (hs.empty()) { std::cerr << "  x slot " << j << " hs empty\n"; return 7; }
        std::memcpy(seg_out.data() + j * n_embd, hs.data(), n_embd * sizeof(float));
        std::cout << "  slot " << j << " hidden[0..3]: "
                  << hs[0] << " " << hs[1] << " " << hs[2] << " " << hs[3] << "\n";
        n_past += 1;
    }

    // Inject mask_end
    {
        auto hs = decode_single_embed(me.data());
        std::cout << "  mask_end hidden[0..3]: " << hs[0] << " " << hs[1] << " " << hs[2] << " " << hs[3] << "\n";
        n_past += 1;
    }

    // Save
    std::ofstream out("/tmp/pathA_reference/instructsam_seg_output_embeddings_probe.f32", std::ios::binary);
    out.write("BIN1", 4);
    int32_t nd = 2; out.write((const char*)&nd, 4);
    int64_t d0 = 10, d1 = n_embd;
    out.write((const char*)&d0, 8); out.write((const char*)&d1, 8);
    out.write((const char*)seg_out.data(), seg_out.size() * sizeof(float));
    std::cout << "\n  ✓ wrote 10 x " << n_embd << " seg_output_embeddings to disk\n";
    std::cout << "\n=== Piece 3 Day 2 (v2): one-at-a-time injection works ===\n";
    return 0;
}
