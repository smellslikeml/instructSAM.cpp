// Day 9a — validate Qwen3-VL tokenizer + chat template against the
// HuggingFace tokenizers reference captured to prompt_tokens.i32.

#include "sam3/instructsam_tokenizer.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<int32_t> read_tokens_i32(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4]; f.read(magic, 4);
    if (std::string(magic, 4) != "TOK1") throw std::runtime_error("bad magic");
    int32_t n = 0; f.read(reinterpret_cast<char *>(&n), 4);
    std::vector<int32_t> ids(static_cast<size_t>(n));
    f.read(reinterpret_cast<char *>(ids.data()), n * sizeof(int32_t));
    return ids;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string gguf = (argc >= 2) ? argv[1]
        : "/tmp/claude-1001/-home-thorax/e5355e82-8c80-4141-8828-424676e4ee8f/scratchpad/instructsam-as-qwen3vl/instructsam-lm-f16.gguf";
    const std::string ref_path = (argc >= 3) ? argv[2]
        : "/tmp/pathA_reference/warehouse_rgb/prompt_tokens.i32";
    const std::string query = (argc >= 4) ? argv[3]
        : "Please segment the box, the person, the shelf, and the forklift in the image.";

    std::cout << "loading tokenizer (vocab_only) from " << gguf << "\n" << std::flush;
    auto tok = sam3::InstructsamTokenizer::load(gguf);
    std::cout << "  vocab_size: " << tok->vocab_size() << "\n";

    const auto & sp = tok->specials();
    std::cout << "  specials: im_start=" << sp.im_start
              << " im_end=" << sp.im_end
              << " vision_start=" << sp.vision_start
              << " image_pad=" << sp.image_pad
              << " vision_end=" << sp.vision_end
              << " object_ref_start=" << sp.object_ref_start
              << " object_ref_end=" << sp.object_ref_end
              << " mask_start=" << sp.mask_start
              << " mask_end=" << sp.mask_end
              << " seg=" << sp.seg
              << " eos=" << sp.eos << "\n";

    std::cout << "\ntokenizing query: \"" << query << "\"\n";
    const auto ours = tok->tokenize_chat(query);
    std::cout << "  ours (" << ours.size() << " tokens): ";
    for (auto id : ours) std::cout << id << " ";
    std::cout << "\n";

    std::cout << "\nloading HF reference from " << ref_path << "\n";
    const auto ref = read_tokens_i32(ref_path);
    std::cout << "  ref  (" << ref.size() << " tokens): ";
    for (auto id : ref) std::cout << id << " ";
    std::cout << "\n";

    if (ours.size() != ref.size()) {
        std::cerr << "\n  ✗ token COUNT mismatch: ours=" << ours.size()
                  << " ref=" << ref.size() << "\n";
        return 2;
    }
    int mismatches = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ours[i] != ref[i]) {
            if (mismatches < 10) {
                std::cerr << "  ✗ [" << i << "] ours=" << ours[i] << " ref=" << ref[i] << "\n";
            }
            ++mismatches;
        }
    }
    if (mismatches != 0) {
        std::cerr << "\n  ✗ " << mismatches << " total token mismatches\n";
        return 3;
    }
    std::cout << "\n  ✓ token IDs match HuggingFace tokenizers exactly\n";

    // Sanity: detokenize the specials round-trip
    std::cout << "\nround-trip detokenize: \n  ";
    std::cout << "'" << tok->detokenize({sp.im_start, sp.im_end, sp.vision_start,
                                          sp.image_pad, sp.vision_end,
                                          sp.object_ref_start, sp.object_ref_end,
                                          sp.mask_start, sp.seg, sp.mask_end}) << "'\n";
    return 0;
}
