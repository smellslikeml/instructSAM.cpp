# Piece 3 — LM integration via llama.cpp Qwen3-VL

**Correction to PATH-B-RECIPE.md**: llama.cpp *does* support mid-generation
inputs_embeds and hidden-state capture out of the box. The recipe assumed
"weeks of ggml C++ engineering" would be needed to patch llama.cpp. It
turns out both mechanisms are already in the standard C API. This piece
becomes ~3-5 days of orchestration instead of weeks of C++ patching.

## API surface — what we use

From `llama.cpp/include/llama.h`:

**Inputs**:
- `llama_batch.embd` (line 258): pass embeddings instead of token IDs.
  Signal: set `batch.token[i] = 0` and populate `batch.embd + i*n_embd` with the
  embedding vector for that position. Line 940: "If embd != 0, llama_batch.embd
  will be allocated with size of n_tokens * embd * sizeof(float)."
- `llama_batch_init(n_tokens, embd, n_seq_max)` (line 944): allocate a batch
  with embedding storage.

**Output capture**:
- `llama_context_params.embeddings = true` (line 387): enable hidden-state
  output alongside logits.
- `llama_set_embeddings(ctx, true)` (line 991): toggle dynamically at runtime.
- `llama_batch.logits[i] != 0` (line 262): mark positions for which we want
  logits/embeddings output.
- `llama_get_embeddings_ith(ctx, i)` (line 1041): retrieve hidden state
  `[n_embd]` for the i-th token in the last decoded batch.

**Model introspection**:
- `llama_model_n_embd_inp(model)` / `llama_model_n_embd_out(model)`: input
  (embedding table) vs. output (hidden state) dims. For Qwen3-VL text these
  are both 2048.

## InstructSAM's exotic generation flow — mapped to llama.cpp calls

InstructSAM's `prepare_inputs_for_generation` (models/instructsam.py:663-681)
detects when LM emits `<|object_ref_end|>` (id 151647) and injects a special
sequence via `inputs_embeds`:

```
<|object_ref_end|> embed → <|mask_start|> embed → mask_queries[10, 2048] → <|mask_end|> embed
```

The 10 hidden states at the mask_queries positions become `seg_output_embeddings`
which our lm_bridge (already ported) projects to decoder queries.

### Mapped to llama.cpp calls

Pseudo-code (skipping error handling + kv-cache management details):

```cpp
// 1. Standard mtmd pipeline — encode image + text, generate tokens
llama_model_params  mp = {...};
llama_context_params cp = llama_context_default_params();
cp.embeddings = true;   // ← enable hidden-state extraction
auto model = llama_model_load_from_file(lm_gguf_path, mp);
auto ctx   = llama_init_from_model(model, cp);

// mtmd handles vision + text encoding to produce initial batch
mtmd_helper_eval_chunks(mtmd_ctx, image, prompt);  // fills the KV cache

// 2. Standard token-by-token greedy generation
llama_batch batch = llama_batch_init(1, 0, 1);  // token-mode batch, 1 token
llama_token tok = mtmd_helper_get_next_token(...);
std::vector<llama_token> generated_ids;

while (!done) {
    batch.token[0] = tok;
    batch.pos[0]   = pos++;
    batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;  // want logits at this position
    batch.n_tokens = 1;

    llama_decode(ctx, batch);
    tok = sample_greedy(llama_get_logits_ith(ctx, 0));
    generated_ids.push_back(tok);

    // 3. When LM emits <|object_ref_end|>, inject mask_queries
    if (tok == 151647 /* <|object_ref_end|> */) {
        // Load mask_queries [10, 2048] from OUR sam3cpp GGUF
        auto mask_queries = load_mask_queries();  // via GgufModel

        // Get input embed table via a mini-decode of the special tokens
        //   OR access the embedding table directly (llama.cpp exposes model_get_tensor).
        // For simplicity: run a "prefix" decode that gets us the embeddings
        // for <|mask_start|> and <|mask_end|> by tokenizing them and
        // capturing pre-attention embed via the below API path.

        // Build embedding batch: 12 positions total
        //   pos 0:  <|mask_start|> embed
        //   pos 1..10: mask_queries[0..9]
        //   pos 11: <|mask_end|> embed
        int32_t n_inject = 12;
        int32_t n_embd = llama_model_n_embd_inp(model);
        llama_batch ebatch = llama_batch_init(n_inject, n_embd, 1);
        ebatch.n_tokens = n_inject;
        // set ebatch.token to nullptr means embd is used
        for (int i = 0; i < n_inject; ++i) {
            ebatch.pos[i]      = pos + i;
            ebatch.n_seq_id[i] = 1;
            ebatch.seq_id[i][0] = 0;
            ebatch.logits[i]   = (i >= 1 && i <= 10) ? 1 : 0;  // capture 10 mask_queries hidden states
            std::memcpy(ebatch.embd + i * n_embd, ...);         // fill each embedding
        }

        llama_decode(ctx, ebatch);

        // Extract seg_output_embeddings [10, 2048]
        std::vector<float> seg_out_embeds(10 * n_embd);
        for (int i = 0; i < 10; ++i) {
            std::memcpy(seg_out_embeds.data() + i * n_embd,
                        llama_get_embeddings_ith(ctx, 1 + i),  // positions 1..10 in ebatch
                        n_embd * sizeof(float));
        }
        // Append to the batch's accumulated seg_output_embeddings for this object
        all_seg_out_embeds.push_back(seg_out_embeds);

        pos += n_inject;
        llama_batch_free(ebatch);

        // Resume greedy generation from the next position
        // (last mask_end embed's logits give us the next token distribution)
        tok = sample_greedy(llama_get_logits_ith(ctx, n_inject - 1));
        generated_ids.push_back(tok);
    }
}

// 4. Extract phrase_ids from generated_ids (between ref_start 151646 and ref_end 151647)
auto phrases = extract_phrases(generated_ids, 151646, 151647);

// 5. Text embed lookup for phrase_ids — same API pattern:
//    Build a batch with the phrase_ids, one decode, extract embeddings at those positions.
//    This gives us phrase_embeddings [num_phrases, ≤32, n_embd].
```

### Getting the input embedding table

For steps 3 (need embeddings for `<|mask_start|>`, `<|mask_end|>`) and 5
(text_hidden_fcs input), we need access to the LM's input embedding table.
Three viable approaches:

1. **Direct tensor access via `llama_model_get_tensor(model, "token_embd.weight")`**.
   Standard tensor name in GGUF. Simplest.

2. **Mini-decode + `llama_get_embeddings_ith` on token positions**. Uses
   only the public API but requires an extra decode pass.

3. **Access mask_queries + special tokens FROM OUR SAM3CPP GGUF**. Our
   converter already stores `instructsam.mask_queries` (2 KB). Special
   token embeddings can be extracted from the LM's embed table via (1).

Recommend approach 1 for embedding table + our GGUF's `mask_queries`.

## Fallback: mtmd-cli patch approach

If direct API access has issues (e.g. embeddings mode disables sampling
in some way), an alternative is to fork `llama-mtmd-cli` and add the
InstructSAM-specific hook. The recipe is:

- Add `--emit-instructsam-seg` flag
- In the generation loop, check `tok == 151647`
- Do the embedding injection via the same API calls above (they're
  called from mtmd-cli source too — just wrapped in the same binary)
- Serialize seg_output_embeddings + phrase_ids to stdout / a file

This is ~200 LOC of mtmd-cli modification, and doesn't require deep
llama.cpp internals — just uses the public API.

## Concrete C++ interface

New class `InstructsamLmRunner` in sam3cpp-fork:

```cpp
namespace sam3 {

struct LmOutput {
    // Per-object outputs from LM autoregressive generation
    int num_objects = 0;             // number of <|object_ref_end|> events detected
    // seg_output_embeddings[obj] shape [10, n_embd=2048]
    std::vector<std::vector<float>> seg_output_embeddings;
    // phrase_ids[obj] shape [<=32, n_embd=2048] — LM embed lookups for
    // the tokens between <|object_ref_start|> and <|object_ref_end|>
    std::vector<std::vector<float>> phrase_embeddings;
    // phrase_masks[obj] shape [<=32] — 1.0 for valid tokens, 0.0 padding
    std::vector<std::vector<float>> phrase_masks;
    // Generated text (with [SEG] insertions per InstructSAM's post-process)
    std::string generated_text;
};

class InstructsamLmRunner {
public:
    // lm_gguf_path: Path B output, `instructsam-lm-f16.gguf`
    // mmproj_path:  Path B output, `instructsam-mmproj-f16.gguf`
    // grounding_gguf: our sam3cpp-fork GGUF (for mask_queries + special
    //   glue tensors)
    InstructsamLmRunner(
        const std::string & lm_gguf_path,
        const std::string & mmproj_path,
        const std::string & grounding_gguf,
        int max_generated_tokens = 200);
    ~InstructsamLmRunner();

    // Run mtmd-style inference on image + query. Returns per-object
    // seg_output_embeddings + phrase_embeddings + phrase_masks that
    // feed directly into our existing sam3cpp chain via lm_bridge
    // (mask_hidden_fcs and text_hidden_fcs).
    LmOutput run(const std::string & image_path, const std::string & query);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sam3
```

The `LmOutput` maps directly to what `sam3_instructsam_batch_e2e.cpp`
loads today from disk (obj-i's queries.f32 comes from
`mask_hidden_fcs(seg_output_embeddings)`).

## Session plan (~3-5 days total for Piece 3)

### Day 1 — Prototype: verify inputs_embeds + hidden-state capture

- Write a minimal 100-LOC test that: loads Path B's LM GGUF via
  llama.cpp, does ONE token generation with token IDs (verify), does ONE
  token generation with `llama_batch.embd` (verify same output), captures
  hidden state at a position via `llama_get_embeddings_ith` (verify shape
  n_embd=2048).
- **Deliverable**: proof that the API surface works as documented.

### Day 2 — Adapter around mtmd for image+text preprocessing

- Depend on `libllama` + `libmtmd`. Wrap the existing mtmd-cli logic
  (image loading, tokenization, chat template) in a callable function.
- **Deliverable**: `preprocess_image_prompt(image_path, query)` returns
  the initial KV-cache-loaded state ready for token generation.

### Day 3 — Generation loop with `<|object_ref_end|>` detection + injection

- Detect ref_end tokens during generation.
- On detection: build embed batch with mask_queries loaded from our
  grounding GGUF.
- Capture seg_output_embeddings.
- **Deliverable**: `LmOutput.seg_output_embeddings` filled for
  warehouse_rgb.jpg, cross-checked against PyTorch reference.

### Day 4 — Phrase extraction + text embedding lookup

- Walk generated tokens, extract phrase_ids between ref_start/ref_end.
- Get LM embed table via `llama_model_get_tensor(model, "token_embd.weight")`.
- Look up phrase_id embeddings, build phrase_embeddings +
  phrase_masks (pad to 32).
- **Deliverable**: full `LmOutput` populated, matches PyTorch
  `phrase_embedding` at cos-sim > 0.99.

### Day 5 — Integrate with vision-native E2E

- Modify `sam3-instructsam-vision-native-e2e` to accept `LmOutput`
  instead of loading from disk.
- Add a new tool `sam3-instructsam-full-native-e2e` that combines:
  vision encoder (native) + LM runner (via llama.cpp) + our ggml
  detection + segmentation chain.
- Compare masks vs. PyTorch reference.
- **Deliverable**: single-binary image + text → masks with no
  PyTorch dependency at runtime.

## After Piece 3

Piece 5 (convert-script extensions) becomes: ensure our sam3cpp-fork
GGUF has the `instructsam.mask_queries` tensor (it does — already
verified in the sanity check tool). Add a script that documents which
GGUFs are needed for the runtime (LM + mmproj + grounding).

Piece 6 (CLI orchestrator) becomes trivial once Piece 3 lands: image
loading via stb_image → preprocess → `InstructsamVisionEncoder.run_all_layers` +
`run_neck` → `InstructsamLmRunner.run` → chain everything via
`sam3-instructsam-full-native-e2e` logic.

**Estimated Milestone 3 total after this correction: ~1.5 weeks**
(was 2-4 weeks based on incorrect llama.cpp API assumption).
