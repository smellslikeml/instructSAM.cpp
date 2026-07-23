# Piece 3 (fork approach) — Qwen3 forward pass in sam3cpp

Session kickoff of Option 4 from PIECE3_DAY2_STATUS.md: fork Qwen3-VL
text-model forward pass into sam3cpp instead of debugging llama.cpp's
embed-mode decode corner case.

## Session deliverables

- `include/sam3/instructsam_lm_forward.h` — skeleton class with all
  Qwen3-VL text-model architecture constants documented:
    28 layers, hidden=2048, heads=16, kv_heads=8 (GQA), head_dim=128,
    intermediate=6144, rms_norm_eps=1e-06, SwiGLU MLP, per-head
    q_norm/k_norm, 1D RoPE, no biases (Qwen3 native)
- `src/instructsam_lm_forward.cpp` — impl of:
    validate_all_tensors_present() — checks 2 top-level + 28 × 11 = 310
      required tensors present
    embed_for_token(token_id) — reads a row from token_embd.weight
      (handles F16 → F32 conversion)
- `tools/sam3_instructsam_lm_forward_test.cpp` — test harness that
  loads Path B's LM GGUF, validates tensor presence, extracts
  embeddings for known tokens (mask_start=151671, mask_end=151672,
  ref_start=151646, ref_end=151647, [SEG]=151670), compares to the
  safetensors dumps for exact match verification.

## Tensor structure (verified via Python probe)

llama.cpp GGUF naming convention (differs from HF safetensors):

| Path B GGUF name | HF safetensors name | Shape (GGUF ne) | Dtype |
|---|---|---|---|
| `token_embd.weight` | `model.language_model.embed_tokens.weight` | [2048, 151936] | F16 |
| `output_norm.weight` | `model.language_model.norm.weight` | [2048] | F32 |
| `blk.N.attn_norm.weight` | `input_layernorm.weight` | [2048] | F32 |
| `blk.N.attn_q.weight` | `self_attn.q_proj.weight` | [2048, 2048] | F16 |
| `blk.N.attn_k.weight` | `self_attn.k_proj.weight` | [2048, 1024] | F16 |
| `blk.N.attn_v.weight` | `self_attn.v_proj.weight` | [2048, 1024] | F16 |
| `blk.N.attn_output.weight` | `self_attn.o_proj.weight` | [2048, 2048] | F16 |
| `blk.N.attn_q_norm.weight` | `self_attn.q_norm.weight` | [128] | F32 |
| `blk.N.attn_k_norm.weight` | `self_attn.k_norm.weight` | [128] | F32 |
| `blk.N.ffn_norm.weight` | `post_attention_layernorm.weight` | [2048] | F32 |
| `blk.N.ffn_gate.weight` | `mlp.gate_proj.weight` | [2048, 6144] | F16 |
| `blk.N.ffn_up.weight` | `mlp.up_proj.weight` | [2048, 6144] | F16 |
| `blk.N.ffn_down.weight` | `mlp.down_proj.weight` | [6144, 2048] | F16 |

No bias tensors (Qwen3 uses `attention_bias: False`).
Total: 2 top-level + 28 × 11 = 310 tensors. Matches Python enumeration.

## Session blocker: OOM loading LM GGUF via our GgufModel

Our `sam3::GgufModel` calls `ggml_backend_alloc_ctx_tensors` which
eagerly copies all tensor data into a single backend buffer. For the
3.3GB LM GGUF, this requires ~3.3GB of continuous backend memory.

On the current dev box (16GB RAM, ~800MB free, swap saturated), the
process gets SIGKILL'd during load. llama.cpp's own load path avoids
this by using mmap — the GGUF file is mapped as-is and tensors are
accessed lazily via pointers into the mapped region. Only the pages
touched during actual computation get faulted in.

## Path forward — three options

### Option A: Add mmap support to our GgufModel (recommended)

Extend `GgufModel::load` to accept an mmap mode. When on, use
`ggml_backend_cpu_buffer_from_ptr` on the mmap'd region instead of
`ggml_backend_alloc_ctx_tensors`. This mirrors llama.cpp's approach.

Effort: ~1 day. Benefits all future large-model loads. Also lets us
load the LM alongside our grounding GGUF without doubling memory.

### Option B: Write InstructsamLmForward with its own file reader

Skip GgufModel for this component. Use gguf.h directly to open the
file, read specific tensors on-demand into temporary buffers, close.
Slower per-access but no memory overhead.

Effort: ~4-6 hours. Isolated to InstructsamLmForward but doesn't
solve the underlying mmap need for future large models.

### Option C: Kill firefox etc. and try again (not really)

Not sustainable. The 3.3GB alloc will still fight for RAM every time
the tool runs. mmap is the right long-term solution.

**Recommendation**: Option A next session. ~1 day. After that, the
Qwen3 layer forward implementation begins (RMSNorm + q_norm/k_norm +
GQA + 1D RoPE + SwiGLU MLP — same CPU-first pattern as vision layer 0).

## Full Qwen3 fork estimate (revised after this session)

Days 1-5 originally scoped for a llama.cpp mtmd wrapper (blocked on
their embed decode bug). Re-scoped for our own Qwen3 forward:

| Day | Deliverable |
|---|---|
| 1 (done) | Skeleton class + tensor validation + embed lookup |
| 2 | Add mmap to GgufModel + test full load runs |
| 3 | CPU helpers: RMSNorm, SiLU, GQA-mha, 1D RoPE, q_norm/k_norm |
| 4 | Layer 0 forward + parity vs PyTorch reference |
| 5-7 | Layers 1-27 loop, full 28-layer parity vs PyTorch dump |
| 8 | Embed injection semantics: build sequence with tokens and
      injected mask_queries, extract 10 hidden states at those positions |
| 9 | Integrate with vision-native E2E: image + text query → pred_masks |

**Total: ~9 days focused engineering.** Same order-of-magnitude as
the vision encoder port (which took ~1 week of sessions). No new
architectural unknowns.

## Path A journey — 32 commits

All numerical components proven in native ggml at cos-sim > 0.99. LM
integration path chosen (fork over llama.cpp patch), skeleton in place,
concrete week-long plan documented.
