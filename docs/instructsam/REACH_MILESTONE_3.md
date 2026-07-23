# Reach Milestone 3: fully-standalone InstructSAM in native ggml

**Target**: `instructsam-cli /path/to/image.jpg "segment the box, person, ..."`
runs entirely in ggml (via llama.cpp for LM + our sam3cpp fork for
grounding/segmentation) and produces per-object mask PNGs.

**Current state** (feat/instructsam-decoder, 19 commits ahead of upstream):
- All 5 detection/segmentation components ported to ggml (0.9987 cos-sim
  vs PyTorch on warehouse_rgb.jpg batched E2E)
- `tools/convert_instructsam_to_gguf.py` converts the grounding/segmentation
  half only (1315 tensors, 969 MB)
- `sam3-instructsam-batch-e2e` runs the full ggml chain from **captured
  PyTorch inputs** — not raw image + text

## Deliverable shape

Branch will contain:
- `tools/convert_instructsam_to_gguf.py` — extends current script to also
  emit LM (Qwen3-VL text), Qwen3-VL vision, mask_hidden_fcs/text_hidden_fcs
  glue, text_projection. Single unified GGUF for the whole model.
- `tools/instructsam-cli` (C++ binary) — takes image + text query,
  produces per-object mask PNGs. Internally chains:
  1. Load image + preprocess to two `pixel_values` tensors (Qwen3-VL
     for LM, SAM3-sized 1008×1008 for segmentation)
  2. LM half: Qwen3-VL vision (via llama.cpp mtmd) + Qwen3-VL text
     (llama.cpp), with a custom hook to detect `[SEG]` positions +
     capture hidden states + return generated token IDs
  3. Extract seg_output_embeddings (LM hidden at [SEG]) + phrase_ids
     (LM token IDs between object_ref_start/end)
  4. Run `mask_hidden_fcs(seg_output_embeddings)` → decoder queries
  5. Run text_hidden_fcs(LM_embed_table[phrase_ids]) → text_features
  6. Run SAM3 vision encoder → backbone_features + fpn_position_encoding
  7. Run our ggml chain: detr_encoder → detr_decoder →
     output_layer_norm → PCA + FPN + mask_tail → pred_masks
  8. Sigmoid + threshold + upsample → binary masks → save PNGs

## Ordered phases

Order chosen to (a) close small unknowns early, (b) leave the biggest
piece (vision encoder) for parallel/dedicated work while other pieces
consolidate, (c) test each phase against captured reference tensors so
integration bugs surface immediately.

### Phase A: LM-bridge glue (~2 days)

**Piece 4 — mask_hidden_fcs C++ port**
- 2-layer MLP with ReLU: 2048 → 2048 → 256
- Weights already in GGUF as `instructsam.mask_hidden_fcs.0.{0,2}.{weight,bias}`
- Validation: input `seg_output_embeddings.pt` [4, 10, 2048], output
  should match `mask_hidden_fcs_0.pt` [4, 10, 256] to bf16→fp16 noise
- **Landed this session** — see `src/instructsam_lm_bridge.cpp`

**Piece 4b — text_hidden_fcs C++ port**
- Same structure as mask_hidden_fcs, different weights
  (`instructsam.text_hidden_fcs.0.{0,2}.{weight,bias}`)
- Validation: extend reference capture to grab LM embedding lookups for
  phrase_ids; output should match `text_hidden_fcs_0.pt` [4, 32, 256]
- Deferred: need to extend `pathA_reference_capture.py` to save
  `phrase_ids` + `phrase_embedding_pre_projection` so this can be
  validated end-to-end. ~30 min extension + one re-capture run.

### Phase B: SAM3 vision encoder (~1-2 weeks — critical path)

**Piece 1 — InstructsamVisionEncoder**
- 32-layer ViT + 3-level FPN neck. Structurally similar to sam3cpp's
  stock trunk/neck but tensor naming differs (separate q/k/v vs fused
  qkv, `.attention.*` not `.attn.*`, explicit `position_embeddings` +
  `layer_norm` at trunk top, `.embeddings.patch_embeddings.projection`).
- Reference oracle already captures:
  - `vision_encoder_backbone::last_hidden_state` [1, 5184, 1024]
  - `mask_decoder_in::backbone_features_{0,1,2}` — the FPN outputs
    (post-neck) as backbone_features [1, 256, 288/144/72, 288/144/72]
- Suggested sub-phases:
  1. Skeleton + tensor enumeration + patch embed + layer 0 parity (1 day)
  2. Layers 1-31 (~2 days once layer 0 works — same structure)
  3. FPN neck: 3-level output via proj1/proj2/scale_layers (~2 days)
  4. Numerical debugging (fp16 drift, RoPE indexing, windowed attn) (~2-5 days)
- Critical path item — everything else waits on this for true E2E

### Phase C: convert script extensions (~1-2 days)

**Piece 5 — extend converter for LM + vision**
- LM (Qwen3-VL text): use llama.cpp's existing conversion path (Path B
  has already done this — model at
  `/tmp/claude-1001/.../scratchpad/instructsam-as-qwen3vl/instructsam-lm-f16.gguf`)
- Qwen3-VL vision encoder: also from Path B
  (`instructsam-mmproj-f16.gguf`)
- text_projection [256, 1024]: currently in our GGUF but only in the
  grounding namespace — verify LM path uses it via a lookup, or embed
  it in the LM-bridge convert step
- Suggested: keep three separate .gguf files (grounding, LM, mmproj) at
  first to avoid touching llama.cpp's conversion scripts; CLI loads all
  three. Merge into single file as a later polish.

### Phase D: CLI orchestrator (~2-3 days)

**Piece 6 — instructsam-cli**
- Image loading: stb_image or Pillow-via-C-binding (single-image, JPEG/PNG
  input). Two preprocess paths: Qwen3-VL-sized for LM (~336×336 patches
  ×3) + SAM3-sized (1008×1008) for segmentation.
- LM invocation: llama.cpp API for Qwen3-VL. Need custom callback to
  detect `<|object_ref_end|>` (151647) in generated tokens and
  capture the LM's hidden state at that position (10 slots' worth of
  hidden states — `seg_output_embeddings`).
- Phrase ID extraction: walk generated tokens, extract sequences
  between `<|object_ref_start|>` (151646) and `<|object_ref_end|>`
  (151647), pad to max_len=32 per phrase.
- Text embedding lookup: hit LM's input embedding table (needs
  `llama_model_get_embed_table` or equivalent) for each phrase_id
  → `text_hidden_fcs` → text_features.
- Vision encoding: run SAM3 vision (Piece 1) on SAM3-sized pixel_values.
- Chain the existing ggml components (detr_encoder → detr_decoder →
  output_layer_norm bridge → PCA + FPN + mask_tail).
- Batch loop over N objects (N = number of phrases extracted from LM).
- Mask post-process: sigmoid → threshold → upsample to original image
  size → save PNGs (or overlay via a companion Python script).

## Risk profile

| Risk | Likelihood | Mitigation |
|---|---|---|
| SAM3 vision encoder numerical bugs (RoPE, windowed attn) | High | Reference-oracle per-layer validation from day 1 (same pattern that got detr_encoder right first try) |
| llama.cpp API doesn't expose hidden-state capture at `[SEG]` | Medium | Fall back to a small llama.cpp patch (add a callback hook) or run LM in a subprocess and pipe hidden states out |
| InstructSAM's Qwen3-VL runtime differs from stock (mask_queries embedding injection mid-generation) | Medium | Same handling can be done in the CLI layer above llama.cpp (inject mask_queries embed at the right generation step) |
| Text embedding lookup table not directly exposed by llama.cpp | Low | The embed table is a normal weight tensor; direct file read from the LM GGUF is fine |
| Single-image assumption breaks for batch inference | N/A | Milestone 3 target is single-image CLI; batch is a follow-up |

## Total estimate

| Phase | Duration | Blocking |
|---|---|---|
| A: LM-bridge | 2 days | nothing |
| B: Vision encoder | 1-2 weeks | critical path — nothing else can produce true E2E |
| C: Convert script | 1-2 days | can start in parallel with B |
| D: CLI orchestrator | 2-3 days | needs A + B + C |

**Realistic total: 2-4 weeks** (was 3-5 before Piece 2 collapsed).

If SAM3 vision encoder work parallelizes to a dedicated session/agent
while another does Piece 4/4b/5, calendar time drops closer to 2 weeks.

## Update — Piece 1 (SAM3 vision) landed + Piece 3 rescoped

**Piece 1 (SAM3 vision encoder) — COMPLETE** (2026-07-23):
- 32-layer ViT trunk + FPN neck all in ggml
- 5-milestone parity harness — all layers cos-sim 0.99999+
- Vision-native E2E (pixel_values → pred_masks) proven — cos-sim 0.9986 vs. PyTorch

**Piece 3 (LM integration) — RESCOPED from "weeks" to ~3-5 days**:
The PATH-B-RECIPE originally concluded that InstructSAM's embedding-injection
mechanism would need "several weeks of ggml C++ engineering" to add to
llama.cpp. Turns out llama.cpp's public C API already supports both
`llama_batch.embd` (mid-generation inputs_embeds) and
`llama_get_embeddings_ith` (hidden-state capture). See
`docs/instructsam/LM_INTEGRATION_PLAN.md` for the API surface + day-by-day
plan.

**Revised Milestone 3 estimate: ~1.5 weeks** (was 2-4).

## What's in the branch RIGHT NOW that's usable

Anyone can already:
- Convert InstructSAM's grounding/segmentation weights to GGUF:
  `python3 tools/convert_instructsam_to_gguf.py`
- Run the ggml chain against captured PyTorch inputs:
  `./build/sam3-instructsam-batch-e2e /tmp/pathA_gguf/instructsam-grounding-f16.gguf`
- See it visualized:
  `python3 tools/visualize_e2e_masks.py`
- Verify per-component correctness with the individual test tools
  (decoder-test, mask-tail-test, pixel-decoder-test, seg-head-e2e,
  detr-encoder-test, prompt-cross-attn is inside seg-head-e2e).

What's NOT yet in the branch:
- Anything image → tensor (needs Piece 1)
- Anything text → tensor (needs Piece 4b + Piece 6 LM glue)
- Anything token → mask end-to-end from a single command
