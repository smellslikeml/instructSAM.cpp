# Path A step 2b — sam3cpp/InstructSAM decoder aliasing analysis

Investigation output from attempting phase-2 step 2 (fork decoder.cpp for
double-cross-attn). Key finding: **sam3cpp's stock decoder ALREADY
implements the three-attention-block-per-layer pattern InstructSAM uses**
(`self_attn` + `ca_text` + `cross_attn`). What differs is (a) tensor
naming, and (b) presence-token machinery that InstructSAM omits.

Downstream implication: the C++ engineering scope for InstructSAM support
is much smaller than a decoder rewrite — mostly a sidecar-aliasing scheme
plus one surgical fork to skip presence-token operations.

## Layer structure — same on both sides

Sam3cpp's stock decoder (`src/decoder.cpp:430-472`) per-layer flow:

```
self_kv_in     = concat(presence_token, output_queries)
self_q_in      = concat(zero_presence_pos, query_pos)
self_qk        = self_kv_in + self_q_in

self_attn(Q=self_qk, K=self_qk, V=self_kv_in)      → post-norm → tgt
text_ca_attn(Q=tgt+self_q_in, K=V=prompt, mask)    → post-norm → tgt
image_ca_attn(Q=tgt+self_q_in, K=memory+pos, V=memory, mask=box_rpb)  → post-norm → tgt
FFN(linear1 → relu → linear2)                       → post-norm → tgt
```

InstructSAM's decoder (`instructsam/models/sam3/modeling_sam3.py:1509-1585`)
per-layer flow:

```
self_attn(Q=hidden+query_pos, K=hidden+query_pos, V=hidden)  → post-norm
text_cross_attn(Q=hidden+query_pos, K=V=text_features, mask) → post-norm
vision_cross_attn(Q=hidden+query_pos, K=vision+vision_pos, V=vision, mask)  → post-norm
mlp(fc1 → activation → fc2)                                  → post-norm
```

**Byte-identical structure.** No graph-code rewrite needed for the core
attention path.

## Tensor-name correspondence

| Sam3cpp expects | InstructSAM has |
|---|---|
| `.self_attn.query_proj.*` | `.self_attn.q_proj.*` |
| `.self_attn.key_proj.*` | `.self_attn.k_proj.*` |
| `.self_attn.value_proj.*` | `.self_attn.v_proj.*` |
| `.self_attn.out_proj.*` | `.self_attn.o_proj.*` |
| `.norm2.*` (post-self-attn) | `.self_attn_layer_norm.*` |
| `.ca_text.*` | `.text_cross_attn.*` |
| `.catext_norm.*` | `.text_cross_attn_layer_norm.*` |
| `.cross_attn.*` (vision) | `.vision_cross_attn.*` |
| `.norm1.*` (post-vision-attn) | `.vision_cross_attn_layer_norm.*` |
| `.linear1.*` (MLP fc1) | `.mlp.fc1.*` |
| `.linear2.*` (MLP fc2) | `.mlp.fc2.*` |
| `.norm3.*` (post-MLP) | `.mlp_layer_norm.*` |
| `boxRPB_embed_x.layers.0/1.*` | `box_rpb_embed_x.layer1/2.*` (1-indexed) |
| `boxRPB_embed_y.layers.0/1.*` | `box_rpb_embed_y.layer1/2.*` (1-indexed) |
| `bbox_embed.layers.0/1/2.*` | `box_head.layer1/2/3.*` (1-indexed) |
| `ref_point_head.layers.0/1.*` | `ref_point_head.layer1/2.*` (1-indexed) |
| `transformer.decoder.norm.*` (output norm) | `.output_layer_norm.*` |

## Tensors sam3cpp needs that InstructSAM lacks

These are the presence-token mechanism sam3cpp uses for Meta's SAM3.
InstructSAM uses its own `mask_queries` mechanism instead and has no
presence prediction:

| sam3cpp requires | InstructSAM equivalent |
|---|---|
| `transformer.decoder.presence_token.weight` | — (absent) |
| `transformer.decoder.presence_token_out_norm.*` | — (absent) |
| `transformer.decoder.presence_token_head.layers.N.*` | — (absent) |

To run the stock decoder against InstructSAM weights, these operations
would need to be surgically skipped or bypassed. Options:

1. **Fork** `sam3::Decoder::run()` into `sam3::InstructsamDecoder::run()`
   that omits the presence-token block. Copies ~530 LOC with ~10 lines
   removed.
2. **Guard the presence-token operations** in stock `Decoder::run()`
   behind `if (model.metadata().instructsam_variant)` conditionals.
   Smaller diff but couples stock code to the variant.

Fork is cleaner (option 1).

## Sidecar aliasing scheme — the naming problem

Sam3cpp's `load_tensor_aliases()` parses key:value pairs from a JSON
sidecar and populates `aliases[original_name] = short_name`. A tensor
is looked up via `find_tensor(canonical_name)` →
`resolve_tensor_name(canonical_name)` → returns the short_name →
tensor_index_ lookup by that short_name.

To make BOTH sam3cpp's canonical names AND InstructSAM's canonical
names resolve to the same tensor requires the sidecar to contain
aliases for both. Sam3cpp's parser handles this correctly only if:

- Each `"key": "value"` pair in the JSON has the KEY equal to a
  real tensor name present in the GGUF, AND
- Multiple pairs may share the same VALUE (both `X → short` and
  `Y → short` is fine — sam3cpp reads all matches via regex, no
  dedup)

Two implementation options:

**Option A (recommended)**: **duplicate the tensor data** in the GGUF
under a second `t_<hash>` name, and add a second sidecar entry mapping
that second short_name to sam3cpp's canonical name. Cost: ~50-100 MB
extra for the decoder tensors that need dual-aliasing (out of 948 MB
total). Runtime: both names resolve, both find real tensors.

**Option B**: extend sam3cpp's `load_tensor_aliases()` to accept a
richer sidecar format (list of pairs, or a value that's a list of
alternate names). More invasive change to sam3cpp's C++ but doesn't
duplicate tensor data.

For our scope, Option A is the least-friction path.

## Actual remaining C++ work — refined estimate

Given the above:

1. **Converter update**: emit dual-aliasing GGUF (each decoder tensor
   twice, sidecar has both InstructSAM + sam3cpp aliases). Python
   changes only; small. ~4-6 hours.
2. **Fork `Decoder` → `InstructsamDecoder`**: copy stock decoder.cpp
   `Decoder::run()` and remove presence-token machinery. Sam3cpp's
   `Decoder::run()` is 200 LOC of ggml graph; ~20 LOC removed. Compile
   + smoke test that the graph builds and doesn't require presence
   tensors. ~1 day.
3. **Numerical parity**: run the InstructsamDecoder against the
   reference oracle's captured decoder layer inputs
   (`detr_encoder::last_hidden_state`, `detr_encoder::text_features`,
   etc.) and verify per-layer output SHA-256 matches
   `detr_decoder_layer_N` in `manifest.json`. Debug shape/dtype/order
   issues. ~2-3 days.
4. **External-input CLI**: wrap InstructsamDecoder in a CLI that reads
   external inputs (mask_hidden_fcs output, text_hidden_fcs output,
   vision features) from disk and dumps final masks. ~1 day.

Total: **~4-5 days engineering** — meaningfully less than the earlier
2.5-week estimate, because the layer graph doesn't need to be rewritten
(only presence-token bits removed) and the naming is bulk mechanical
aliasing.

## Reference for future implementer

See the corresponding tensors already validated as addressable in
`tools/sam3_instructsam_sanity.cpp` (26/28 → 29/29 after naming fix)
and `tools/sam3_instructsam_decoder_test.cpp` (247/247 needed by
InstructsamDecoder skeleton).

The reference-oracle SHA-256 fingerprints for per-layer parity checks
are in `docs/instructsam/reference/warehouse_rgb/manifest.json`.
Expected `detr_decoder_layer_0` fp16 SHA-256: `e0bfb1c4f14364c3`
(shape `[4, 10, 256]`).
