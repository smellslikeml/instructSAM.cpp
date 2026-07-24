#include "sam3/instructsam_tokenizer.h"

#include "llama.h"

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace sam3 {

namespace {

std::once_flag g_backend_once;
void ensure_backend() {
    std::call_once(g_backend_once, []() { llama_backend_init(); });
}

int32_t token_id_or_die(const llama_vocab * vocab, const std::string & piece) {
    // BPE tokenize the exact literal — for special tokens this returns
    // a single-token result; for anything else it may split.
    std::vector<llama_token> buf(16, 0);
    const int32_t n = llama_tokenize(vocab, piece.c_str(), static_cast<int32_t>(piece.size()),
                                     buf.data(), static_cast<int32_t>(buf.size()),
                                     /*add_special=*/false, /*parse_special=*/true);
    if (n != 1) {
        throw std::runtime_error("expected single-token piece for '" + piece +
                                 "', got " + std::to_string(n));
    }
    return static_cast<int32_t>(buf[0]);
}

}  // namespace

std::unique_ptr<InstructsamTokenizer> InstructsamTokenizer::load(const std::string & gguf_path) {
    ensure_backend();

    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    llama_model * m = llama_model_load_from_file(gguf_path.c_str(), mp);
    if (!m) throw std::runtime_error("llama_model_load_from_file (vocab_only) failed: " + gguf_path);

    auto t = std::unique_ptr<InstructsamTokenizer>(new InstructsamTokenizer());
    t->model_ = m;
    t->vocab_ = llama_model_get_vocab(m);
    t->specials_.im_start         = token_id_or_die(t->vocab_, "<|im_start|>");
    t->specials_.im_end           = token_id_or_die(t->vocab_, "<|im_end|>");
    t->specials_.vision_start     = token_id_or_die(t->vocab_, "<|vision_start|>");
    t->specials_.vision_end       = token_id_or_die(t->vocab_, "<|vision_end|>");
    t->specials_.image_pad        = token_id_or_die(t->vocab_, "<|image_pad|>");
    t->specials_.object_ref_start = token_id_or_die(t->vocab_, "<|object_ref_start|>");
    t->specials_.object_ref_end   = token_id_or_die(t->vocab_, "<|object_ref_end|>");
    t->specials_.mask_start       = token_id_or_die(t->vocab_, "<|mask_start|>");
    t->specials_.mask_end         = token_id_or_die(t->vocab_, "<|mask_end|>");
    t->specials_.seg              = token_id_or_die(t->vocab_, "[SEG]");
    t->specials_.eos              = token_id_or_die(t->vocab_, "<|endoftext|>");
    return t;
}

InstructsamTokenizer::~InstructsamTokenizer() {
    if (model_) llama_model_free(model_);
}

int32_t InstructsamTokenizer::vocab_size() const {
    return llama_vocab_n_tokens(vocab_);
}

std::vector<int32_t> InstructsamTokenizer::tokenize_chat(const std::string & user_query) const {
    // InstructSAM's Qwen3-VL front-end renders WITHOUT a system message
    // (verified against transformers apply_chat_template on the shipped
    // tokenizer). Adding "You are a helpful assistant" makes the LM
    // route to describe/OCR mode instead of emitting <|object_ref_start|>
    // segmentation markers, because that system-turn was never in
    // training data.
    const std::string prompt =
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>" + user_query +
        "<|im_end|>\n"
        "<|im_start|>assistant\n";

    std::vector<llama_token> buf(prompt.size() + 16, 0);
    int32_t n = llama_tokenize(vocab_, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                               buf.data(), static_cast<int32_t>(buf.size()),
                               /*add_special=*/false, /*parse_special=*/true);
    if (n < 0) {
        // buffer too small — llama_tokenize returns -needed_size
        buf.assign(static_cast<size_t>(-n), 0);
        n = llama_tokenize(vocab_, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                           buf.data(), static_cast<int32_t>(buf.size()),
                           /*add_special=*/false, /*parse_special=*/true);
    }
    if (n < 0) throw std::runtime_error("llama_tokenize failed");
    std::vector<int32_t> out(buf.begin(), buf.begin() + n);
    return out;
}

std::string InstructsamTokenizer::detokenize(const std::vector<int32_t> & tokens) const {
    std::vector<llama_token> ll(tokens.begin(), tokens.end());
    std::string s; s.resize(tokens.size() * 8 + 32);
    int32_t n = llama_detokenize(vocab_, ll.data(), static_cast<int32_t>(ll.size()),
                                 s.data(), static_cast<int32_t>(s.size()),
                                 /*remove_special=*/false, /*unparse_special=*/true);
    if (n < 0) {
        s.resize(static_cast<size_t>(-n));
        n = llama_detokenize(vocab_, ll.data(), static_cast<int32_t>(ll.size()),
                             s.data(), static_cast<int32_t>(s.size()),
                             false, true);
    }
    if (n < 0) throw std::runtime_error("llama_detokenize failed");
    s.resize(static_cast<size_t>(n));
    return s;
}

std::vector<int32_t> InstructsamTokenizer::tokenize_raw(const std::string & text) const {
    std::vector<llama_token> buf(text.size() + 16, 0);
    int32_t n = llama_tokenize(vocab_, text.c_str(), static_cast<int32_t>(text.size()),
                               buf.data(), static_cast<int32_t>(buf.size()),
                               /*add_special=*/false, /*parse_special=*/false);
    if (n < 0) {
        buf.assign(static_cast<size_t>(-n), 0);
        n = llama_tokenize(vocab_, text.c_str(), static_cast<int32_t>(text.size()),
                           buf.data(), static_cast<int32_t>(buf.size()),
                           false, false);
    }
    if (n < 0) throw std::runtime_error("llama_tokenize failed");
    return std::vector<int32_t>(buf.begin(), buf.begin() + n);
}

}  // namespace sam3
