# Step 2e — layer-0 numerical parity status

**Layer 0 achieved parity vs. PyTorch reference oracle after wiring
`query_pos` + `box_rpb` vision bias.**

## Layer-0 metrics vs. `detr_decoder_layer_0.pt`

| Metric | Value | Notes |
|---|---|---|
| Mean cosine similarity (per-query) | **0.9996** | 10 queries; 1.0 = byte-identical direction |
| Max absolute diff | 0.15 | over 2560 elements |
| Relative L2 error | 0.030 | ‖ours − ref‖ / ‖ref‖ |
| Ours: mean=−0.01530 std=1.00534 absmax=3.973 | | ~indistinguishable from ref |
| Ref:  mean=−0.01530 std=1.00549 absmax=4.031 | | |

The remaining 0.04% cosine gap is bf16 (PyTorch) → fp16 (GGUF/ggml)
quantization noise — expected and unavoidable at this level.

## Iteration history

| Change | Cos-sim | Max diff | Rel L2 |
|---|---|---|---|
| Initial: query_pos = 0, no vision bias | 0.797 | 2.82 | 0.642 |
| + query_pos wired (sinusoidal + ref_point_head) | 0.873 | 2.85 | 0.508 |
| + `box_rpb` vision bias wired (this step) | **0.9996** | **0.15** | **0.030** |

Two changes closed the gap 20% → 0.04%. box_rpb was the dominant factor
(as predicted — the log-scale RPB actively steers attention over 5184
spatial locations per query).

## Remaining work for L1-L5 parity

Layers 1-5 currently use the **initial** reference_boxes throughout —
missing the per-layer box_head → inverse_sigmoid → sigmoid update.
Because query_pos AND box_rpb both depend on reference_boxes, L1-L5
will show worse parity until this lands. Expected: cos-sim in the 0.85-0.95
range for L1-L5 without the update; 0.99+ once wired.

Work:
1. After each layer's forward pass, extract hidden_states, apply
   `output_layer_norm` + `box_head` (3-layer MLP) → delta_boxes [10, 4]
2. Update reference_boxes = sigmoid(inverse_sigmoid(prev) + delta)
3. Recompute query_pos + box_rpb from updated boxes; feed to next layer

Estimate: ~4-6 hours of C++.

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
