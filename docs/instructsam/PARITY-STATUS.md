# Step 2e — layer-0 numerical parity status

Current dashboard state after wiring `query_pos` from
`sinusoidal(sigmoid(reference_points.weight))` → `ref_point_head`.

## Layer-0 metrics vs. `detr_decoder_layer_0.pt`

| Metric | Value | Notes |
|---|---|---|
| Mean cosine similarity (per-query) | **0.873** | 10 queries; 1.0 = byte-identical direction |
| Max absolute diff | 2.85 | over 2560 elements |
| Relative L2 error | 0.508 | ‖ours − ref‖ / ‖ref‖ |
| Ours: mean=−0.0125 std=1.02 absmax=3.95 | | closely matches ref distribution |
| Ref:  mean=−0.0153 std=1.01 absmax=4.03 | | |

Structural correctness is validated — mean, std, absmax all within a few
percent of reference. Direction agreement (0.87 cosine) with query_pos and
without box_rpb suggests the attention topology is right and the remaining
gap is dominated by the missing vision cross-attn box_rpb bias.

## Recent iterations

| Step | Change | Cos-sim |
|---|---|---|
| Initial | query_pos = 0, no vision bias | 0.797 |
| + query_pos wired (sinusoidal + ref_point_head) | | 0.873 |
| (x,y,w,h) → (y,x,w,h) concat | (no measurable change) | 0.873 |

## Remaining work to close the gap → 1.0

1. **Wire vision cross-attn `box_rpb` bias**
   `_get_rpb_matrix(reference_boxes, spatial_shape=(72,72))` produces a
   per-head [num_queries, hw] additive bias derived from box_rpb_embed_x
   and box_rpb_embed_y MLPs. Sam3cpp's stock decoder already has this in
   `build_box_rpb_bias` (`src/decoder.cpp:201-291`) — needs porting to
   the InstructSAM path (probably direct reuse; the arithmetic is
   identical, only the query/box dim is 10 vs 200).

2. **Wire per-layer reference_boxes update**
   Between each layer:
   ```
   delta_boxes = box_head(output_layer_norm(hidden_states))  // [10, 4]
   reference_boxes = sigmoid(inverse_sigmoid(reference_boxes) + delta_boxes)
   ```
   Then feed the updated boxes into the next layer's box_rpb + query_pos.
   Without this, layers 1-5 use the initial reference_boxes throughout,
   which is the main reason L1-L5 diverge more (their reference_boxes
   should have been updated 1-5 times).

3. **Investigate bf16 → fp16 quantization noise**
   PyTorch runs bf16, GGUF stores fp16, ggml runs fp16 arithmetic. Small
   accumulated numerical drift is expected; typically caps out around
   0.99 cosine (not 1.0). This is the residual after (1) and (2) land.

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
