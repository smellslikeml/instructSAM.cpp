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
