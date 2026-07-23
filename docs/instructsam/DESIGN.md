# InstructSAM support in sam3cpp — design notes

Status: **feature branch WIP** — Python weight converter done + 100% mapping
coverage validated; C++ decoder graph work pending.

## Motivation

[InstructSAM](https://github.com/DCDmllm/InstructSAM) is a
Qwen3-VL-based instruction-driven segmentation model that reuses Meta
SAM3's `grounding_model` architecture as its mask-generation half, then
feeds LM `[SEG]` token hidden states into it as decoder queries via
custom projection glue. Full-ggml runtime is currently blocked; the
model runs only in PyTorch.

sam3cpp gives us the ggml infrastructure for SAM3's grounding_model
components. llama.cpp gives us Qwen3-VL for the LM half. This branch
extends sam3cpp so InstructSAM's specific decoder variant loads and
runs correctly, closing the gap between "we have both primitives" and
"we can run InstructSAM in ggml."

## What's already done on this branch

### `tools/convert_instructsam_to_gguf.py`

Namespace remap converter. Reads InstructSAM's `model.safetensors` and
maps its tensor names to sam3cpp's expected schema:

| InstructSAM prefix | sam3cpp prefix |
|---|---|
| `model.grounding_model.model.vision_encoder.backbone.*` | `backbone.vision_backbone.trunk.*` |
| `model.grounding_model.model.vision_encoder.neck.*` | `backbone.vision_backbone.convs.*` |
| `model.grounding_model.model.detr_encoder.*` | `transformer.encoder.*` |
| `model.grounding_model.model.detr_decoder.*` | `transformer.decoder.*` |
| `model.grounding_model.model.mask_decoder.*` | `segmentation_head.*` |
| `model.grounding_model.model.dot_product_scoring.*` | `dot_prod_scoring.*` |
| `model.grounding_model.model.geometry_encoder.*` | `geometry_encoder.*` |
| `model.grounding_model.model.text_projection.*` | `text_projection.*` |
| `model.mask_hidden_fcs.*` | `instructsam.mask_hidden_fcs.*` |
| `model.text_hidden_fcs.*` | `instructsam.text_hidden_fcs.*` |
| `model.mask_queries` | `instructsam.mask_queries` |

Qwen3-VL weights (`model.language_model.*`, `model.visual.*`) are
skipped — handled by llama.cpp's native Qwen3-VL support.

**Dry-run verification on the InstructSAM-2B checkpoint (2026-07-23):**
1704 input tensors → 1079 mapped, 625 skipped (Qwen3-VL), 0 unmapped.

GGUF emission itself is stubbed pending the C++ work below; without the
decoder-graph modifications, an emitted GGUF would "load" but produce
incorrect masks.

## What's NOT done — the C++ engineering

### Decoder graph: double-cross-attention per layer

sam3cpp's `src/decoder.cpp` implements a single-cross-attention
decoder layer:

```
per layer:
  self_attn(queries)
  cross_attn(queries, memory)   # memory = pre-fused text+vision from encoder_fusion
  mlp
```

InstructSAM's `detr_decoder` uses **two separate cross-attention blocks**
per layer, keeping text and vision streams unfused:

```
per layer:
  self_attn(queries)
  text_cross_attn(queries, text_memory)
  vision_cross_attn(queries, vision_memory)
  mlp
```

The graph modification needed:

- Duplicate the existing `cross_attn` block in `src/decoder.cpp` (around
  lines 459-466) into `text_cross_attn` and `vision_cross_attn`, both
  reading their own tensor prefixes (which the converter already emits
  under `transformer.decoder.layers.N.text_cross_attn.*` and
  `.vision_cross_attn.*`)
- Extend `Decoder::run()` signature to accept TWO memory tensors
  (`text_memory` and `vision_memory`) instead of one
- Skip sam3cpp's `encoder_fusion` stage entirely for InstructSAM
  (InstructSAM doesn't fuse; text and vision stay separate)

**Complexity estimate: 1-2 days of focused C++ work.** All ops
(linear, layer norm, attention) are already supported by sam3cpp's
ggml graph builder; the work is duplicating an existing pattern.

### Custom glue tensor loading

- `instructsam.mask_queries` — shape `(10, 2048)`, learnable slot
  embeddings at LM hidden dim
- `instructsam.mask_hidden_fcs.0.*` — MLP `(2048 → 2048 → 256)` that
  projects `[SEG]` token hidden states down to decoder query dim
- `instructsam.text_hidden_fcs.0.*` — MLP `(2048 → 2048 → 256)` that
  projects text embeddings for text_cross_attn

These need registration in the runtime as first-class tensors that the
inference orchestration reads. Straightforward — standard linear layers,
no exotic ops. **Complexity: ~1 day.**

### `[SEG]` token routing from Qwen3-VL

The runtime orchestration piece. Options:

1. **IPC boundary** — llama.cpp process emits `[SEG]` token positions
   + their 2048-dim hidden states; a small wire format (npy or MsgPack)
   ships them to a sam3cpp process that consumes them as decoder query
   inputs.
2. **Unified runtime** — wrap both llama.cpp and sam3cpp inference in
   a single binary that shares ggml contexts. Faster, no serialization
   cost, but larger blast radius.

MVP: option 1. Optimization: option 2 later.

**Complexity for option 1: ~2-3 days** (message format, wire protocol,
sam3cpp CLI arg to accept external query tensors).

### Total remaining engineering estimate

~5-8 focused engineering days for the C++ work, plus 1-2 days for
end-to-end numerical parity validation against the PyTorch reference.
**Realistic overall: 1.5-2 weeks** to a working "run InstructSAM in
ggml on CPU" MVP.

## Validation strategy

1. **Tensor-shape validation** (per-tensor): converter emits GGUF, then
   a small C++ tool loads each tensor and asserts shape matches
   sam3cpp's runtime expectations.
2. **Layer-level numerical parity**: dump PyTorch reference outputs
   at each decoder-layer boundary (using InstructSAM's own inference
   harness), compare against sam3cpp's per-layer outputs. Threshold:
   max relative error < 1e-3 (fp16 tolerance).
3. **End-to-end IoU parity**: run the fixed `infer.py` (from
   [DCDmllm/InstructSAM#4](https://github.com/DCDmllm/InstructSAM/pull/4))
   as the reference, sam3cpp's runtime as candidate, compare masks on
   the same test image. Threshold: IoU > 0.95 per-object.

## References

- Upstream InstructSAM: https://github.com/DCDmllm/InstructSAM
- Our CLI-enumeration fix PR: https://github.com/DCDmllm/InstructSAM/pull/4
- SAM3 paper: (Meta FAIR)
- sam3cpp: https://github.com/ropoctl/sam3cpp
- Qwen3-VL in llama.cpp: `tools/mtmd/models/qwen3vl.cpp`
