# Piece 3 Day 1 — status

## What worked

- **llama.cpp API surface confirmed exists** (documented in
  LM_INTEGRATION_PLAN.md):
  - `llama_batch.embd` is a real field in the public struct
  - `llama_get_embeddings_ith(ctx, i)` returns hidden states
  - `llama_context_params.embeddings = true` enables the capture path
  - `llama_model_n_embd_inp` returns 8192 (vision) vs
    `llama_model_n_embd` = 2048 (text) — use `n_embd` for text-side
    embed injection.

- **Path B llama.cpp Qwen3-VL runs** on warehouse_rgb.jpg (verified
  via llama-mtmd-cli — 22s, correctly emits box/person/shelf/forklift
  phrases between `<|object_ref_start|>...<|object_ref_end|>`).

- **Static-library linkage** works up to `llama_batch_init`:
  - `sam3-instructsam-llama-probe` links against pre-built
    libllama.a, libggml*.a with `-Wl,--whole-archive` on libggml-cpu.a
    (needed — backend init code gets dropped by GC otherwise).
  - Loads model, creates context with `embeddings=true`, tokenizes
    "Hello world" → 2 tokens.
  - `llama_batch_init(2, 0, 1)` succeeds.

## What blocks Day-2 progress

**`llama_decode(ctx, batch)` segfaults** inside libllama after the
"init: embeddings required but some input tokens were not marked as
outputs -> overriding" log line. No visible progress past that point.

Root cause hypotheses (untested):

1. **ABI/linkage subtleties from mixing our CMake with llama.cpp's
   pre-built .a files.** Llama.cpp's build sets many compile flags
   (`-DGGML_USE_CPU_HBM`, `-DNDEBUG`, various instruction-set gates)
   that our sam3cpp-fork target doesn't set. If some struct is
   compiled differently in the two units, decode may access an
   inconsistent field layout.

2. **Missing backend registration** — even with `--whole-archive` on
   libggml-cpu.a, additional backend adapters may need forced-linking.

3. **Embeddings mode requires specific batch/context setup** that
   we're missing — e.g. `llama_context_params.pooling_type =
   LLAMA_POOLING_TYPE_NONE` explicitly.

## Path forward (options for next session)

### Option A: Proper CMake integration via `add_subdirectory`

Add llama.cpp as a sub-project in sam3cpp-fork's CMake. Inherit all
their compile flags automatically. This is the cleanest fix for the
ABI-subtlety hypothesis.

Effort: 2-4 hours + validating the probe passes.

### Option B: Build the Piece-3 tool INSIDE llama.cpp's tree

Add an `add_executable(sam3-instructsam-cli ...)` to
`llama.cpp/tools/mtmd/CMakeLists.txt`. Follow the same target
pattern as `llama-mtmd-cli`. Link to our sam3cpp static lib from
inside llama.cpp's build.

Trade-off: our sam3cpp code is inside llama.cpp's build tree
(reverse of the natural direction), or we accept two-repo build.

Effort: 4-6 hours + integration testing.

### Option C: Fork llama-mtmd-cli in llama.cpp/tools/mtmd/

Copy `mtmd-cli.cpp` to `instructsam-cli.cpp` in llama.cpp's tools/mtmd,
add the InstructSAM-specific hook (detect ref_end, inject
mask_queries, capture hidden states). Uses proven code as a base.
When done, it's a self-contained binary in the llama.cpp build.

Trade-off: patches to llama.cpp are external to our sam3cpp-fork
branch — harder to co-develop.

Effort: 6-8 hours + integration.

### Recommendation

**Option A**: cleanest boundary. Keeps our sam3cpp-fork the source of
truth for InstructSAM's ggml chain, uses llama.cpp as a proper
dependency.

## Session Piece-3 deliverables shipped

- `docs/instructsam/LM_INTEGRATION_PLAN.md` — 5-day plan, corrects the
  earlier "weeks of C++ patching" conclusion in PATH-B-RECIPE.md
- `docs/instructsam/PIECE3_DAY1_STATUS.md` — this file
- `include/sam3/instructsam_lm_runner.h` — interface skeleton (compiles clean)
- `src/instructsam_lm_runner.cpp` — impl skeleton, `run()` throws with pointer
  to the plan doc
- `tools/sam3_instructsam_llama_probe.cpp` — Day-1 API-verification probe
  (partial — segfaults inside `llama_decode`)
- CMakeLists.txt — auto-detects llama.cpp at `../llama.cpp` and builds
  Piece-3 targets if found (with `-Wl,--whole-archive` on ggml-cpu.a)

## Overall Milestone 3 status after Piece 3 kickoff

All numerical components validated end-to-end in ggml (26 commits).
Only LM orchestration + CLI plumbing between here and the standalone
`instructsam-cli` binary. Piece-3 Day-1 confirmed the API surface
exists; a decode-time issue needs CMake integration cleanup to unblock.
Best estimate to Milestone 3 completion: **1 week** (was 1.5 pre-Piece-3
kickoff; the API surface confirmation eliminated the "will llama.cpp
even support this" unknown, at the cost of surfacing a linkage cleanup
task).
