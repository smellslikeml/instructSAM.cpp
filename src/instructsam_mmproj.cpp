#include "sam3/instructsam_mmproj.h"

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <atomic>
#include <cstdio>
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
}  // namespace

std::unique_ptr<InstructsamMmproj> InstructsamMmproj::load(
    const std::string & mmproj_gguf_path, const std::string & lm_gguf_path
) {
    ensure_backend();

    // mtmd needs a text_model handle to look up embedding dim + special tokens.
    // Use vocab_only load — no weights, cheap.
    auto mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    llama_model * lm = llama_model_load_from_file(lm_gguf_path.c_str(), mp);
    if (!lm) throw std::runtime_error("mmproj: failed loading LM handle from " + lm_gguf_path);

    auto cp = mtmd_context_params_default();
    cp.use_gpu = false;
    cp.print_timings = false;
    cp.n_threads = 4;
    cp.warmup = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    mtmd_context * ctx = mtmd_init_from_file(mmproj_gguf_path.c_str(), lm, cp);
    if (!ctx) {
        llama_model_free(lm);
        throw std::runtime_error("mtmd_init_from_file failed for " + mmproj_gguf_path);
    }

    if (!mtmd_support_vision(ctx)) {
        mtmd_free(ctx); llama_model_free(lm);
        throw std::runtime_error("mmproj does not support vision");
    }

    auto out = std::unique_ptr<InstructsamMmproj>(new InstructsamMmproj());
    out->ctx_ = ctx;
    out->model_ = static_cast<void *>(lm);
    return out;
}

InstructsamMmproj::~InstructsamMmproj() {
    if (ctx_) mtmd_free(ctx_);
    if (model_) llama_model_free(static_cast<llama_model *>(model_));
}

InstructsamMmproj::ImageEmbeddings InstructsamMmproj::encode_image_file(
    const std::string & image_path
) const {
    // Load the JPEG/PNG via mtmd's helper (uses stb_image internally).
    auto wrap = mtmd_helper_bitmap_init_from_file(ctx_, image_path.c_str(),
                                                  /*placeholder=*/false);
    if (!wrap.bitmap) throw std::runtime_error("mtmd_helper_bitmap_init_from_file failed: " + image_path);

    // Build a prompt containing only the media marker so mtmd tokenizes
    // "just this one image" — we drop the resulting text chunks and keep
    // the image chunk's embeddings.
    const char * marker = mtmd_default_marker();
    const std::string prompt = marker;
    mtmd_input_text inp{};
    inp.text = prompt.c_str();
    inp.text_len = prompt.size();
    inp.add_special = false;
    inp.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bmps[1] = { wrap.bitmap };
    const int32_t rc = mtmd_tokenize(ctx_, chunks, &inp, bmps, 1);
    if (rc != 0) {
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(wrap.bitmap);
        throw std::runtime_error("mtmd_tokenize returned " + std::to_string(rc));
    }

    // Find the image chunk. There may be surrounding text chunks (start/end
    // of image markers) — we want the middle one whose type is not text.
    ImageEmbeddings out;
    const size_t n_chunks = mtmd_input_chunks_size(chunks);
    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
        const mtmd_input_chunk_type type = mtmd_input_chunk_get_type(ch);
        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) continue;

        const int32_t enc_rc = mtmd_encode_chunk(ctx_, ch);
        if (enc_rc != 0) {
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(wrap.bitmap);
            throw std::runtime_error("mtmd_encode_chunk returned " + std::to_string(enc_rc));
        }

        // vocab_only load doesn't populate n_embd_inp; fall back through
        // n_embd and finally a hardcoded 2048 for InstructSAM's Qwen3-VL-2B.
        int32_t hidden = llama_model_n_embd_inp(static_cast<llama_model *>(model_));
        if (hidden <= 0) hidden = llama_model_n_embd(static_cast<llama_model *>(model_));
        if (hidden <= 0) hidden = 2048;
        const int32_t n_tok = static_cast<int32_t>(mtmd_input_chunk_get_n_tokens(ch));
        const float * embd = mtmd_get_output_embd(ctx_);
        if (!embd) {
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(wrap.bitmap);
            throw std::runtime_error("mtmd_get_output_embd returned null");
        }

        out.data.assign(embd, embd + static_cast<size_t>(n_tok) * hidden);
        out.n_tokens = n_tok;
        out.hidden = hidden;
        break;
    }

    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(wrap.bitmap);

    if (out.n_tokens == 0) {
        throw std::runtime_error("mmproj: no image chunk found in tokenize output");
    }
    return out;
}

}  // namespace sam3
