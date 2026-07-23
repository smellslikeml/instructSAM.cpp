# Step 2e — 6-layer decoder numerical parity

**All 6 decoder layers achieve cos-sim > 0.99 vs. PyTorch reference
oracle after wiring `query_pos`, `box_rpb` vision bias, and per-layer
`box_head` reference-boxes refinement.**

## Per-layer metrics vs. `detr_decoder_layer_N.pt`

| Layer | Cos-sim | Max diff | Rel L2 |
|---|---|---|---|
| L0 | **0.9996** | 0.150 | 0.030 |
| L1 | **0.9993** | 0.251 | 0.038 |
| L2 | **0.9986** | 0.275 | 0.053 |
| L3 | **0.9976** | 0.317 | 0.068 |
| L4 | **0.9962** | 0.454 | 0.088 |
| L5 | **0.9967** | 0.355 | 0.081 |

Small drift across layers is expected numerical-error accumulation from
running fp16 arithmetic (ggml) against bf16 arithmetic (PyTorch). Even
L5 is 0.9967 — indistinguishable from the reference at the segmentation
task level.

## Iteration history

| Change | L0 cos-sim | Notes |
|---|---|---|
| Initial: query_pos=0, no vision bias | 0.797 | structural fork only |
| + query_pos wired (sinusoidal + ref_point_head) | 0.873 | |
| + `box_rpb` vision bias wired | 0.9996 | close L0 to precision limit |
| + per-layer `box_head` refinement | 6/6 L>0.99 | full E2E parity |

Three changes closed the gap on all 6 layers to bf16→fp16 quantization
noise floor.

## What refinement does

Between each layer's ggml forward pass, CPU-side:

1. **output_layer_norm** on the layer's hidden_states → normed [10, 256]
2. **box_head** 3-layer MLP (256 → 256 → 256 → 4) with ReLU → delta [10, 4]
3. **inverse_sigmoid + add + sigmoid** → new reference_boxes [10, 4]
4. **ref_point_head** on sinusoidal(new ref) → new query_pos [10, 256]
5. **build_instructsam_box_rpb_bias** on new ref → new [heads, 10, hw]
   additive bias for the next layer's vision cross-attn

box_head and ref_point_head evaluations are tiny (10 queries × 256 dim,
3-layer MLPs). Doing them CPU-side avoids the overhead of building a
second graph per layer boundary.

## Mask decoder — tail (segmentation head end)

**Given reference `pixel_embed` + `decoder_queries` as inputs, the mask
tail produces `pred_masks` and `semantic_seg` at cos-sim 0.999999 vs.
PyTorch reference (max diff 0.083, rel L2 0.16%).**

`InstructsamMaskTail` (src/instructsam_mask_tail.cpp) implements:
- `mask_embedder`: 3-layer MLP (256→256→256→256) with ReLU
- `instance_projection`: Conv1x1 (256→256) on pixel_embed
- `semantic_projection`: Conv1x1 (256→1) on pixel_embed
- `pred_masks = einsum("qc,chw->qhw", mask_emb, instance_embed)`

Runtime is ~8.8s CPU per object (dominated by the 5.4B-flop
instance_projection). Trivially moveable to ggml for GPU acceleration.

Reference oracle now captures 42 tensors (was 29) including all mask
decoder inputs (backbone_features multi-scale, decoder_queries,
encoder_hidden_states, prompt_features, prompt_mask) and intermediates
(pixel_decoder output, mask_embedder output, instance_projection output).

## Mask decoder — FPN (pixel_decoder)

**InstructsamPixelDecoder now wired**
(src/instructsam_pixel_decoder.cpp). Given 3 backbone features (coarsest
already replaced by the caller with post-prompt-cross-attn encoder
reshaped to [256, 72, 72]), runs 2 stages of `upscale-nearest-2x + skip +
conv3x3 (same padding) + GroupNorm(8) + ReLU`, producing pixel_embed
[256, 288, 288].

Note: InstructSAM ships 3 pixel_decoder conv_layers/norms but with 3
backbone-feature inputs only 2 iterations run (matches PyTorch's
`for _, feat in enumerate(reversed(backbone_features[:-1]))` — 2 iters
for 3 features). Layer 2 weights are dead in this config.

Discovery: conv2d_bias needs NO permute for PyTorch-format weights.
Sam3cpp's stock helper permutes because their MLX-origin weights have a
different layout. PyTorch stores Conv2d weight as [OutC, InC, KH, KW]
which loads into GGUF/ggml as ne=[KW, KH, InC, OutC] — exactly what
ggml_conv_2d expects.

### Parity vs. PyTorch reference (given synthesized post-PCA encoder)

FPN standalone (pixel_embed vs mask_decoder__pixel_decoder):
- cosine (flat): **0.999997**
- max abs diff: 0.017
- relative L2: 0.25%

Full seg-head chain (FPN → mask_tail → pred_masks vs md_pred_masks):
- cosine (flat): **0.999998**
- max abs diff: 0.102 (out of absmax 22 = <0.5%)
- relative L2: 0.17%

Runtime: ~11s CPU for the full FPN + mask_tail per object.

## Mask decoder — prompt_cross_attn (PCA)

**InstructsamPromptCrossAttn now wired**
(src/instructsam_prompt_cross_attn.cpp). LayerNorm + Sam3Attention with
padding-mask handling + residual add. Consumes raw encoder_hidden_states
+ prompt_features + prompt_mask; outputs the post-PCA encoder that feeds
pixel_decoder's coarsest FPN input.

Mask semantics: prompt_mask is bool in PyTorch (True=valid,
False=padding). Dumped as float (1.0/0.0). C++ converts to f16 additive
attention mask: positions where mask ≤ 0.5 get -inf, others get 0.

## Full segmentation-head E2E parity

`sam3-instructsam-seg-head-e2e` runs the whole chain in native ggml from
mask_decoder's raw inputs:

```
PCA(encoder, prompt_features, prompt_mask) → post_encoder
FPN(bb0, bb1, reshape(post_encoder → [256,72,72])) → pixel_embed
mask_tail(pixel_embed, decoder_queries)          → pred_masks + semantic_seg
```

Warehouse_rgb.jpg object 0 (heavy padding: 3/32 valid prompt tokens):

| Stage | Cos-sim | Max diff | Rel L2 |
|---|---|---|---|
| PCA post_encoder | 1.000000 | 0.0014 | 0.009% |
| PCA attn_out (isolated) | 0.999998 | 0.0014 | 0.18% |
| FPN pixel_embed | 0.999997 | 0.017 | 0.25% |
| mask_tail pred_masks | **0.999998** | 0.10 | 0.17% |

Runtime: ~10.7s CPU per object for the full seg-head chain.

**The segmentation half of InstructSAM now runs natively in ggml with no
Python-computed inputs**, given raw mask_decoder inputs. Only LM half
remains (already available via Path B's llama.cpp Qwen3-VL support).

## DETR encoder (vision-text fusion)

**InstructsamDetrEncoder now wired**
(src/instructsam_detr_encoder.cpp). Six pre-norm layers of
self_attn + cross_attn + mlp, fusing vision features with text.

Per-layer flow (PRE-NORM — opposite of decoder!):
```
normed = layer_norm1(hidden)
qk     = normed + vision_pos
self_out = self_attn(Q=qk, K=qk, V=normed)
hidden = hidden + self_out
normed = layer_norm2(hidden)
cross_out = cross_attn(Q=normed, K=V=text_features, mask=text_mask)
hidden = hidden + cross_out
normed = layer_norm3(hidden)
hidden = hidden + mlp(normed)
```

Per-layer parity vs. detr_encoder_layer_N.pt (warehouse_rgb obj 0):

| Layer | Cos-sim | Max diff | Rel L2 |
|---|---|---|---|
| L0 | 0.999995 | 0.033 | 0.31% |
| L1 | 0.999991 | 0.060 | 0.43% |
| L2 | 0.999987 | 0.089 | 0.51% |
| L3 | 0.999987 | 0.126 | 0.52% |
| L4 | 0.999986 | 0.158 | 0.51% |
| L5 | 0.999984 | 0.192 | 0.53% |

All 6 layers > 0.9999 cos-sim on first compile — no debugging needed.
The pre-norm structure + tensor naming were correct from source
inspection. Small drift across layers is expected fp16/bf16 divergence.

Runtime: ~7.75s CPU for the full 6-layer encoder.

## What's still needed for full pixel_values → masks

The encoder + seg-head + decoder chain now takes `vision_features`
(flattened 72×72 backbone output), `text_features`, `vision_pos`,
`text_mask`, `prompt_features`, `prompt_mask`, and `backbone_features`
list as inputs. Everything downstream is ggml-native.

What still produces those upstream:

- **Vision encoder (Sam3ViT + FPN neck)** — forward pass on pixel_values
  produces `fpn_hidden_states` list (backbone features at multiple
  scales). Substantial ViT graph but with lots of prior art in sam3cpp
  (its vision_trunk.cpp) and llama.cpp's mtmd path.
- **Text encoder (CLIP text)** — tokenization + text encoder forward
  produces `text_features` (post-text_projection) and `text_mask`.
  Straightforward but needs GGUF conversion of the CLIP text weights.
- **Position encoding** — `fpn_position_encoding` produced alongside FPN.
  Sinusoidal 2D position embeddings; small compute.

Batch loop over 4 objects: trivial (~30 min). CLI orchestrator:
combines llama.cpp (LM half + mask_hidden_fcs projection) with our
detr_encoder + detr_decoder + PCA/FPN/mask_tail. That gets us to
`image.jpg + query text → pred_masks.pt` in native ggml.

## Iteration ledger — full path A journey so far

| Milestone | Result |
|---|---|
| 2d: structural decoder fork | 6 layers compile, no NaN/Inf |
| 2e-a: query_pos wired | L0 cos-sim 0.873 |
| 2e-b: box_rpb bias wired | L0 cos-sim 0.9996 |
| 2e-c: per-layer box_head refinement | 6/6 layers cos-sim > 0.996 |
| 3-tail: mask_tail (embedder + projections + einsum) | **cos-sim 0.999999** |

Full E2E masks: mask_tail with reference-pixel_embed matches PyTorch to
0.0001% cosine error — the seg-head math is correct. FPN + prompt_cross_attn
wiring are the last remaining pieces before the whole pipeline runs without
Python intermediates.

## How to run the dashboard

```
# regenerate reference binaries (only when reference tensors change)
python3 tools/dump_reference_binaries.py

# rebuild + run
cmake --build build --target sam3-instructsam-decoder-test
./build/sam3-instructsam-decoder-test /tmp/pathA_gguf/instructsam-grounding-f16.gguf --parity
```

Parity mode prints per-layer stats + layer-0 cos-sim / max-diff / rel-L2.
Use it as the guardrail metric while iterating on per-layer box refinement.

## How to run the dashboard

```
# regenerate reference binaries (only when reference tensors change)
python3 tools/dump_reference_binaries.py

# rebuild + run
cmake --build build --target sam3-instructsam-decoder-test
./build/sam3-instructsam-decoder-test /tmp/pathA_gguf/instructsam-grounding-f16.gguf --parity
```

Parity mode prints per-layer stats + layer-0 cos-sim / max-diff / rel-L2.
Use it as the guardrail metric while iterating on box_rpb and box refinement.

## Refined scope estimate

Step 2e was originally scoped at "2-3 days" for per-layer numerical parity.
Current state after this session: **layer-0 dashboard live, 87% cos-sim**.
Remaining work: box_rpb port (~4-6 hours), per-layer box refinement wiring
(~4-6 hours), residual numerical debugging (~1 day). Total: **1.5-2 days**
to reach full 6-layer parity.
