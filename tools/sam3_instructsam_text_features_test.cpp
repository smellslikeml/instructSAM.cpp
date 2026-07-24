// text_features from phrase tokens — validates the "phrase tokens →
// LM embed lookup → pad to 32 → text_hidden_fcs → [32,256]" pipeline
// against per-object captures.
//
// n_valid = 3 for single-subtoken phrases (ref_start + phrase + ref_end),
// or 3 + extra for multi-subtoken (e.g. forklift = 2 subtokens → 4 valid).

#include "sam3/gguf_model.h"
#include "sam3/instructsam_lm_bridge.h"
#include "sam3/instructsam_lm_forward.h"

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

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string grounding_gguf = (argc >= 2) ? argv[1] : "/tmp/pathA_gguf/instructsam-grounding-f16.gguf";
    const std::string lm_gguf = (argc >= 3) ? argv[2]
        : "/tmp/claude-1001/-home-thorax/e5355e82-8c80-4141-8828-424676e4ee8f/scratchpad/instructsam-as-qwen3vl/instructsam-lm-f16.gguf";
    const std::string ref_dir = (argc >= 4) ? argv[3] : "/tmp/pathA_reference/warehouse_rgb";

    std::cout << "loading grounding + LM (both mmap)\n" << std::flush;
    sam3::GgufModel grounding, lm;
    if (!grounding.load(grounding_gguf)) { std::cerr << "grounding load failed\n"; return 2; }
    if (!lm.load(lm_gguf, false, {}, true)) { std::cerr << "lm load failed\n"; return 3; }
    sam3::InstructsamLmForward     lm_fwd(lm);
    sam3::InstructsamTextHiddenFcs text_bridge(grounding);
    std::cout << "  ✓ loaded\n\n";

    // Special-token IDs (from Day 9a — <|object_ref_start|>=151646, <|object_ref_end|>=151647).
    // Phrase subtokens follow the same map as CLI/lm-with-image test.
    // BPE subtokens as they appear inside <|object_ref_start|>...<|object_ref_end|>
    // (no leading space — different from mid-sentence tokenization). Verified
    // by re-tokenizing the captured LM output text_output.txt.
    struct Phrase { std::string name; std::vector<int32_t> tokens; };
    const std::vector<Phrase> phrases = {
        {"box",      {2011}},
        {"person",   {8987}},
        {"shelf",    {53950}},
        {"forklift", {44738, 34969}},  // "fork" + "lift"
    };

    const int64_t H = 2048;
    const int64_t OUT = 256;
    const int64_t MAX_LEN = 32;

    std::cout << "=== per-phrase text_features validation ===\n";
    double sum_cs = 0.0;
    int n = 0;
    for (size_t pi = 0; pi < phrases.size(); ++pi) {
        const auto & p = phrases[pi];
        std::vector<int32_t> ids;
        ids.push_back(151646);  // <|object_ref_start|>
        for (int32_t t : p.tokens) ids.push_back(t);
        ids.push_back(151647);  // <|object_ref_end|>
        const int64_t n_valid = static_cast<int64_t>(ids.size());

        // Build [32, 2048] padded embed tensor.
        // Padding = token id 0 (per InstructSAM PyTorch: phrase_ids initialized
        // via torch.zeros((n, max_len), dtype=input_ids.dtype) — utils.py:106).
        std::vector<float> padded(MAX_LEN * H, 0.0f);
        const auto pad_embed = lm_fwd.embed_for_token(0);
        for (int64_t i = 0; i < n_valid; ++i) {
            const auto e = lm_fwd.embed_for_token(ids[static_cast<size_t>(i)]);
            std::memcpy(padded.data() + i * H, e.data(), H * sizeof(float));
        }
        for (int64_t i = n_valid; i < MAX_LEN; ++i) {
            std::memcpy(padded.data() + i * H, pad_embed.data(), H * sizeof(float));
        }

        // Project via text_hidden_fcs
        const auto out = text_bridge.run(padded, {MAX_LEN, H});
        // out.data is [32, 256]

        // Compare vs captured obj{pi}/enc_text_features.f32
        std::vector<int64_t> ref_s;
        const auto ref = read_bin_f32(ref_dir + "/binaries_obj" + std::to_string(pi) + "/enc_text_features.f32", ref_s);
        // Ref includes the padding-projected values too. Compare full tensor.
        const double cs_full = cosine(out.data, ref);

        // Also just the valid rows (first n_valid × 256)
        std::vector<float> ours_valid(out.data.begin(), out.data.begin() + n_valid * OUT);
        std::vector<float> ref_valid(ref.begin(),        ref.begin()        + n_valid * OUT);
        const double cs_valid = cosine(ours_valid, ref_valid);

        std::cout << "  obj" << pi << " (\"" << p.name << "\", n_valid=" << n_valid << "): "
                  << "cos_sim full=[32,256]=" << cs_full
                  << "  valid-rows=" << cs_valid << "\n";
        // Debug: print my padding row (index n_valid) vs ref padding row
        if (pi == 0) {
            std::cout << "    my  pad row: [" << out.data[n_valid * OUT + 0] << ", "
                      << out.data[n_valid * OUT + 1] << ", "
                      << out.data[n_valid * OUT + 2] << ", "
                      << out.data[n_valid * OUT + 3] << "]\n";
            std::cout << "    ref pad row: [" << ref[n_valid * OUT + 0] << ", "
                      << ref[n_valid * OUT + 1] << ", "
                      << ref[n_valid * OUT + 2] << ", "
                      << ref[n_valid * OUT + 3] << "]\n";
        }
        sum_cs += cs_full;
        ++n;
    }
    std::cout << "\n  mean cos_sim = " << (sum_cs / n) << "\n";

    // ── Also validate text_mask reconstruction: 1s for n_valid positions ─
    std::cout << "\n=== text_mask reconstruction ===\n";
    for (size_t pi = 0; pi < phrases.size(); ++pi) {
        std::vector<int64_t> ms;
        const auto ref_mask = read_bin_f32(ref_dir + "/binaries_obj" + std::to_string(pi) + "/md_pca_prompt_mask.f32", ms);
        const int64_t n_valid_ref = static_cast<int64_t>(ref_mask[0] + ref_mask[1] + ref_mask[2] + ref_mask[3] +
                                                         ref_mask[4] + ref_mask[5] + ref_mask[6] + ref_mask[7]);
        const int64_t ours_n_valid = 2 + static_cast<int64_t>(phrases[pi].tokens.size());
        std::cout << "  obj" << pi << " (\"" << phrases[pi].name << "\"): ours n_valid=" << ours_n_valid
                  << ", ref sum=" << (int)std::round(ref_mask[0] + ref_mask[1] + ref_mask[2] + ref_mask[3] +
                                                     ref_mask[4] + ref_mask[5] + ref_mask[6] + ref_mask[7])
                  << " (mask matches if equal)\n";
        (void)n_valid_ref;
    }
    return 0;
}
