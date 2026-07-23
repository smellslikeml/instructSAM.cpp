# InstructSAM reference-tensor captures

Ground-truth intermediate tensors from PyTorch `InstructSAM-2B` inference,
used as the numerical parity oracle when validating any ggml-native
re-implementation (sam3cpp fork per Path A).

## Regenerating

The `.pt` tensor payloads are large (~120 MB per image) and NOT committed
to git. Regenerate them with:

```bash
/tmp/InstructSAM/.venv/bin/python3 tools/pathA_reference_capture.py \
    --image /path/to/image.jpg \
    --out-dir /tmp/pathA_reference/
```

Each run produces `<out-dir>/<image-name>/` containing:
- `manifest.json` — tensor names, shapes, dtypes, fp16 SHA-256 fingerprints (committed here for provenance)
- `text_output.txt` — the LM's generated segmentation output
- `*.pt` — per-tensor torch payloads (~120 MB total, NOT committed)

The manifest's SHA-256 fingerprints let anyone re-generate the tensors
locally and verify byte-identical outputs.

## Committed references

- **`warehouse_rgb/`** — `~/Downloads/warehouse_rgb.jpg` with query
  `"Please segment the box, the person, the shelf, and the forklift in the image."`
  Captured 2026-07-23 on InstructSAM-2B (snapshot `238da683...`),
  transformers `5.0.0.dev0`, torch `2.12.0+cpu`.

## Key tensor interfaces (for the C++ port)

From reading the manifest + verifying via SHA-256 matches, the
architectural connections are:

- `text_hidden_fcs_0` output (`[4, 32, 256]`) IS `detr_encoder::text_features` input
  → **interface point** where external LM-derived text embeddings enter the mask pipeline
- `seg_output_embeddings` (`[4, 10, 2048]`) is the raw LM hidden state at
  mask_queries positions (after generation with embedding injection)
  → **interface point** where external LM-derived query embeddings enter
- `mask_hidden_fcs_0` output (`[4, 10, 256]`) = `seg_output_embeddings` projected via
  the mask_hidden_fcs 2-layer MLP (2048→2048→256)
  → **first submodule** any ggml port must implement correctly
- `vision_encoder::last_hidden_state` (`[1, 5184, 1024]`) IS the vision
  feature input to `detr_encoder`
  → **interface point** where vision features flow in (either recomputed via
  ggml vision_encoder OR loaded from external source)
- `detr_encoder_layer_5` == `detr_encoder::last_hidden_state`
  → layer 5 IS the final encoder output; no separate final norm
- `mask_decoder::pred_masks` == final `pred_masks`
  → mask_decoder output IS the endpoint; no post-processing between

## How to use for C++ validation

Layer-by-layer check pattern:

```cpp
// After ggml runs, e.g., detr_decoder layer 0:
auto out = /* extract sam3cpp's tensor */;
auto fp16 = cast_to_fp16(out);
std::string sha = sha256_hex(fp16.data(), fp16.bytes());
// Compare against manifest.tensors["detr_decoder_layer_0"].sha256_fp16
// e.g., expect "e0bfb1c4f14364c3"
```

Any mismatch pinpoints which submodule diverged — no need to backtrack
through end-to-end mask differences.

## Sample-set expansion

Additional images can be captured by rerunning with different `--image`.
Recommended targets to add over time:
- Different scene types (indoor, outdoor, dense/sparse objects) — tests
  vision-encoder generalization
- Single-object query — tests the N_OBJECTS=1 path where the buggy CLI
  enumeration (upstream Issue #3) originally hid
- 10+ object query — tests the max_seg_nums slot allocation edge
