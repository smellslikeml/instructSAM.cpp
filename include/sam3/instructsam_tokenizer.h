#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Forward-decls to avoid dragging llama.h into every translation unit that
// pulls this header — the impl includes llama.h.
struct llama_model;
struct llama_vocab;

namespace sam3 {

// Qwen3-VL tokenizer + chat-template wrapper.
//
// Backed by llama.cpp's tokenizer (BPE, vocab_only load — no weights, so
// this is cheap: <1s to init from a 3.3GB GGUF). Applies a hand-rolled
// InstructSAM chat prompt (system + vision-tagged user + assistant) since
// llama_chat_apply_template's built-in "chatml" doesn't inject the
// <|vision_start|><|image_pad|><|vision_end|> block InstructSAM expects.
//
// Special-token IDs are cached at load time so downstream generation-loop
// code (Day 9c) doesn't have to re-tokenize marker strings.
class InstructsamTokenizer {
public:
    // Load vocab from a Qwen3-VL GGUF (LM half, not mmproj). Throws on
    // failure. Safe to call multiple times per process; first call also
    // initializes the llama.cpp backend.
    static std::unique_ptr<InstructsamTokenizer> load(const std::string & gguf_path);
    ~InstructsamTokenizer();

    // Apply InstructSAM chat template to a user query, then tokenize.
    // Produces: <|im_start|>system\n...\n<|im_end|>\n
    //           <|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>QUERY<|im_end|>\n
    //           <|im_start|>assistant\n
    std::vector<int32_t> tokenize_chat(const std::string & user_query) const;

    // Detokenize a run of tokens back to a UTF-8 string (special tokens rendered).
    std::string detokenize(const std::vector<int32_t> & tokens) const;

    // BPE-tokenize a raw text string with no chat template applied and no
    // BOS/EOS added. Used by the CLI to tokenize per-phrase strings like
    // "pallet jack" the same way they appear inside <|object_ref_start|>…
    // <|object_ref_end|> — no leading space, no special token parsing.
    std::vector<int32_t> tokenize_raw(const std::string & text) const;

    // Special-token IDs cached at load time.
    struct SpecialTokens {
        int32_t im_start;          // <|im_start|>
        int32_t im_end;            // <|im_end|>
        int32_t vision_start;      // <|vision_start|>
        int32_t vision_end;        // <|vision_end|>
        int32_t image_pad;         // <|image_pad|>
        int32_t object_ref_start;  // <|object_ref_start|>  (151646)
        int32_t object_ref_end;    // <|object_ref_end|>    (151647)
        int32_t mask_start;        // <|mask_start|>
        int32_t mask_end;          // <|mask_end|>
        int32_t seg;               // "[SEG]"
        int32_t eos;               // <|endoftext|>
    };
    const SpecialTokens & specials() const { return specials_; }
    int32_t vocab_size() const;

private:
    InstructsamTokenizer() = default;
    llama_model * model_ = nullptr;
    const llama_vocab * vocab_ = nullptr;  // borrowed from model_
    SpecialTokens specials_{};
};

}  // namespace sam3
