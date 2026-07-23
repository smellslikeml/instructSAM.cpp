# Piece 3 Day 2 — status after slot 2-9 investigation

## Confirmed working

- **Injection MECHANISM**: `llama_batch.embd`-mode decode with 12
  positions (mask_start + 10 × mask_queries + mask_end) succeeds. The
  batched decode returns without error.
- **Hidden-state capture API**: `llama_get_embeddings_ith` returns
  non-null pointers, `llama_get_embeddings` gives a raw buffer.
- **Slots 0 and 1** consistently produce real, sensible hidden state
  values (magnitudes -3 to +3, dense non-zero patterns).

## Confirmed broken

- **Slots 2-3**: raw buffer contains all zeros (uninitialized memory)
- **Slots 4-9**: raw buffer contains NaN
- **Values are non-deterministic across runs** — slots 0-1 give
  different values each execution → we're reading buffer memory that
  llama.cpp writes SOMETIMES but not always

## Things tried, all didn't fix

| Change | Result |
|---|---|
| `n_ubatch = 2048` (no ubatch splitting) | Same pattern |
| `n_outputs_max = 512` explicit | Same pattern |
| Confirmed logits[i]=1 for all 12 positions | Same pattern |
| `llama_get_embeddings` raw buffer (bypass output_ids) | Same values |
| One-at-a-time embed decode (each 1-position batch) | ALL zeros (KV cache disconnect) |
| Flash attention off + one-at-a-time | ALL zeros |

## Working hypothesis

llama.cpp's embed-mode decode with `output_all=true` (which is what
`embeddings=true` requires) has an internal limit on the number of
outputs it will produce per decode. The n_outputs_max setting doesn't
override this — some other internal reservation is capping. The first
2 positions of the embed batch get real hidden states; positions 2+
hit unallocated memory.

The graph re-reservation shown in sched_reserve says
`worst-case n_outputs = 1`. Even setting `n_outputs_max = 512`
doesn't propagate to the graph reserve. This is probably the root
cause — the graph is reserved for at most 1 output per decode, so
requesting more outputs writes past the reserved region.

## Correction: n_outputs_max was already being used

Investigation of the llama.cpp source (line 621, llama-context.cpp):
```cpp
const uint32_t n_outputs_pp = std::min(n_tokens, cparams.n_outputs_max);
// reserve pp graph:
auto * gf = graph_reserve(n_tokens, n_seqs, n_outputs_pp, ...);
```

The PP (prompt processing) graph IS already reserved with
`n_outputs_pp = min(n_tokens, n_outputs_max) = min(2048, 512) = 512`.
So the "sched_reserve honors n_outputs_max" fix I was planning to make
would be **a no-op** — it already does.

The `n_outputs = 1` in the log is just the WORST-CASE printed line
(line 608, `const int n_outputs = n_seqs;` for the log message), not
the actual reservation.

The output BUFFER (`output_reserve(n_outputs_all)`, line 1827) is
sized based on actual batch outputs at decode time = 12. Fine.

So neither the graph reserve nor the buffer reserve is the bug.

## Real bug is elsewhere — deeper in graph construction

The bug must be in **how the model's build_graph constructs the output
tensor `t_embd` when a multi-position embed batch is decoded**. Some
positions get valid hidden states, others get uninitialized memory /
NaN. Values are non-deterministic across runs of the same input.

Diagnosis paths (all substantive C++ reading):
- Look at how build_graph selects rows from full hidden_states for the
  output tensor (typically ggml_get_rows with output_ids). Are all
  output_ids populated correctly for embed-mode multi-position?
- Check if there's a hidden per-decode limit somewhere (e.g. a
  ggml_backend buffer that only fits 1-2 output rows for a decoder
  model in "generation mode" vs. embedding mode).
- Check if flash_attn interacts badly with output-per-position when
  n_outputs > 1 (flash_attn was on in all my tests).

## Path forward

**Options** for next session:

1. **Patch llama.cpp** to make sched_reserve honor `n_outputs_max`
   properly (or to reserve for `n_batch` outputs when embeddings mode
   is on). ~1 day investigation + potentially upstreamable.

2. **Decode ref_end and each mask_query token as separate 1-batch
   decodes**, feeding the KV cache incrementally. The one-at-a-time
   result showed zeros, but likely because the batches after the first
   were treating each as isolated (fresh KV cache). Investigate why
   KV state didn't carry over — should be automatic when re-using
   the same context.

3. **Fork a simpler path**: use `llama-mtmd-debug` or similar to
   dump per-layer intermediates, then compute the mask_queries hidden
   state manually via CPU forward pass through the LM's last few
   layers. Slow but correct. Fits with our pattern of CPU-first
   validation.

4. **Bypass llama.cpp entirely** for this bit: implement a
   minimal Qwen3-VL LM forward pass with embed injection in our
   sam3cpp fork (like we did for the vision encoder). Adds 1-2 weeks
   but no llama.cpp integration issues. Feasible because the LM
   architecture (Qwen3 text transformer) has fewer moving parts than
   the vision path we already conquered.

## Session-scoped conclusion

The API SURFACE for InstructSAM's inputs_embeds mechanism is CONFIRMED
to exist and be callable. The DAY 2 GOAL (get all 10 mask_queries hidden
states via the injection) is only PARTIALLY MET (2/10) due to what
appears to be an internal llama.cpp output-buffer sizing issue that
doesn't surface via error return — silent failure into uninitialized
memory.

Two paths forward, both realistic:
  - Investigate n_outputs sched_reserve behavior in llama.cpp (~1 day)
  - Or fork a CPU-side Qwen3 LM forward pass (~1-2 weeks, but full
    control over the embed-injection semantics)

For an actual E2E demo, path 2 is more predictable. For a proper
llama.cpp integration (production quality), path 1 is right.

**Milestone 3 total estimate now: 1.5-2 weeks** (was 1 week before
this Day 2 investigation surfaced the output-buffer issue). Not a
scientific unknown — just more C++ engineering than the initial
optimistic estimate.

## Update: patch attempt in follow-up session confirmed n_outputs_max
## was already being honored — real bug is deeper (see Correction section).

## Practical recommendation (revised)

Given:
- Days spent investigating llama.cpp's embed-mode decode without
  finding the fix
- llama.cpp is actively maintained; version-drift risk on any patch
- The mechanism DOES work for the FIRST TWO positions of an embed
  batch — proving the API is real, just buggy for larger batches

**Recommendation shifted toward Option 4** (fork a minimal Qwen3 text
forward pass into our sam3cpp fork, use direct control of the embed
injection). Reasons:
1. Same pattern we used successfully for SAM3 vision encoder port
2. No dependence on llama.cpp version stability for a niche feature
3. Full control over embed injection semantics = easy to match
   InstructSAM's PyTorch behavior exactly
4. Uses code patterns we already have (get_f32 accessor, cpu_linear,
   cpu_layer_norm, mha primitives) — most of the plumbing already
   exists

Estimated 1.5-2 weeks for a Qwen3 text-only forward pass port that
supports our specific injection needs. Comparable to what llama.cpp
patching would take + gives us a robust deliverable.

An alternative for anyone continuing the llama.cpp path: file an
issue against ggerganov/llama.cpp with our repro (`instructsam-inject-probe`).
The maintainers likely know which corner of embed-mode decode is
implicated. Attaching the small probe binary + expected 10-row output
is a concrete bug report.
