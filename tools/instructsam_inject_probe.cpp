// Day-2 probe: mask_queries embedding injection.
//
// Loads Path B LM. Tokenizes a text prompt containing <|object_ref_end|>.
// Runs generation up to the ref_end token. Then INJECTS the mask_queries
// embedding sequence (mask_start + 10 mask_queries + mask_end) and captures
// the 10 hidden states = seg_output_embeddings.
//
// This is the CORE MECHANISM InstructSAM's PyTorch inference does via
// prepare_inputs_for_generation. If this works, Day 3 (mtmd image wrapping)
// is just plumbing.

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
    if (std::string(magic, 4) != "BIN1") throw std::runtime_error("bad magic in " + path);
    int32_t ndim = 0; f.read(reinterpret_cast<char *>(&ndim), 4);
    shape.assign(static_cast<size_t>(ndim), 0);
    size_t total = 1;
    for (int i = 0; i < ndim; ++i) {
        int64_t d = 0; f.read(reinterpret_cast<char *>(&d), 8);
        shape[static_cast<size_t>(i)] = d; total *= static_cast<size_t>(d);
    }
    std::vector<float> data(total);
    f.read(reinterpret_cast<char *>(data.data()),
        static_cast<std::streamsize>(total * sizeof(float)));
    return data;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "usage: instructsam-inject-probe <lm.gguf> <mask_queries.f32> "
                     "<mask_start_embed.f32> <mask_end_embed.f32>\n";
        return 1;
    }
    const std::string lm_path = argv[1];
    const std::string mq_path = argv[2];
    const std::string ms_path = argv[3];
    const std::string me_path = argv[4];

    // ── Load embed artifacts ────────────────────────────────────────────
    std::vector<int64_t> mq_s, ms_s, me_s;
    const auto mq = read_bin_f32(mq_path, mq_s);   // [10, 2048]
    const auto ms = read_bin_f32(ms_path, ms_s);   // [1, 2048]
    const auto me = read_bin_f32(me_path, me_s);   // [1, 2048]
    std::cout << "mask_queries " << mq_s[0] << "x" << mq_s[1]
              << "  mask_start " << ms_s[0] << "x" << ms_s[1]
              << "  mask_end "   << me_s[0] << "x" << me_s[1] << "\n";

    // ── Load LM via common_init_from_params ────────────────────────────
    common_params params;
    params.model.path = lm_path;
    params.n_ctx = 2048;
    params.n_batch = 2048;
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
    if (n_embd != mq_s[1] || n_embd != ms_s[1]) {
        std::cerr << "n_embd mismatch: model=" << n_embd
                  << " mask_queries=" << mq_s[1] << " mask_start=" << ms_s[1] << "\n";
        return 3;
    }
    std::cout << "loaded LM, n_embd=" << n_embd << "\n";

    // ── Build a short text prompt (no image needed for the probe) ──────
    // Just seed the LM with any context, then feed ref_end + injection.
    const std::string prompt = "The following is a segmentation task.";
    std::vector<llama_token> toks = common_tokenize(ctx, prompt, true, true);
    std::cout << "prompt has " << toks.size() << " tokens\n";

    // Decode the prompt
    llama_batch b = llama_batch_init(2048, 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i]; b.pos[i] = (llama_pos)i;
        b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    b.n_tokens = toks.size();
    if (llama_decode(ctx, b) < 0) { std::cerr << "prompt decode failed\n"; return 4; }
    int32_t n_past = (int32_t)toks.size();

    // Feed ref_end token normally
    const llama_token ref_end = 151647;
    b.n_tokens = 1;
    b.token[0] = ref_end; b.pos[0] = n_past;
    b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
    if (llama_decode(ctx, b) < 0) { std::cerr << "ref_end decode failed\n"; return 5; }
    n_past += 1;
    std::cout << "\n=== detected <|object_ref_end|>, injecting mask_queries ===\n";

    llama_batch_free(b);

    // ── Injection: build embed batch [mask_start, 10 mask_queries, mask_end] ─
    const int inject_n = 12;
    llama_batch eb = llama_batch_init(inject_n, n_embd, 1);
    for (int i = 0; i < inject_n; ++i) {
        eb.pos[i] = n_past + i;
        eb.n_seq_id[i] = 1; eb.seq_id[i][0] = 0;
        // Request output at all 12 positions; we care about positions 1..10
        // (the 10 mask_queries). Others are output too — harmless.
        eb.logits[i] = 1;
    }
    // Fill embeddings
    std::memcpy(eb.embd + 0 * n_embd, ms.data(), n_embd * sizeof(float));
    for (int j = 0; j < 10; ++j) {
        std::memcpy(eb.embd + (1 + j) * n_embd, mq.data() + j * n_embd, n_embd * sizeof(float));
    }
    std::memcpy(eb.embd + 11 * n_embd, me.data(), n_embd * sizeof(float));
    eb.n_tokens = inject_n;

    if (llama_decode(ctx, eb) < 0) { std::cerr << "  x embed injection decode failed\n"; return 6; }
    std::cout << "  ✓ embed injection decode succeeded (" << inject_n << " embedded tokens)\n";

    // ── Capture 10 seg_output_embeddings ────────────────────────────────
    std::cout << "\n=== seg_output_embeddings (hidden state at each mask_query position) ===\n";
    std::vector<float> seg_out(10 * n_embd);
    for (int j = 0; j < 10; ++j) {
        const float * hs = llama_get_embeddings_ith(ctx, 1 + j);
        if (!hs) { std::cerr << "  x hidden state null at " << j << "\n"; return 7; }
        std::memcpy(seg_out.data() + j * n_embd, hs, n_embd * sizeof(float));
        if (j < 3 || j == 9) {
            std::cout << "  slot " << j << " hidden[0..3]: "
                      << hs[0] << " " << hs[1] << " " << hs[2] << " " << hs[3] << "\n";
        } else if (j == 3) {
            std::cout << "  ... (slots 3..8) ...\n";
        }
    }
    // Save to disk
    std::ofstream out("/tmp/pathA_reference/instructsam_seg_output_embeddings_probe.f32", std::ios::binary);
    out.write("BIN1", 4);
    int32_t nd = 2; out.write((const char*)&nd, 4);
    int64_t d0 = 10, d1 = n_embd;
    out.write((const char*)&d0, 8); out.write((const char*)&d1, 8);
    out.write((const char*)seg_out.data(), seg_out.size() * sizeof(float));
    std::cout << "\n  ✓ wrote 10 * " << n_embd << " seg_output_embeddings to disk\n";

    llama_batch_free(eb);
    std::cout << "\n=== Piece 3 Day 2: mask_queries INJECTION MECHANISM WORKS ===\n";
    std::cout << "  Next: Day 3 = wrap with mtmd (image + text preprocessing)\n"
              << "         Day 4 = phrase extraction + text embed lookup\n"
              << "         Day 5 = integrate with vision-native E2E\n";
    return 0;
}
