# Path B — GGUF conversion of InstructSAM's Qwen3-VL half via llama.cpp

**Status (2026-07-23):** Conversion + build + basic multimodal inference proven
end-to-end. `[SEG]` token emission from the converted LM diverges from the
PyTorch reference (see "Open issue" below) — investigation deferred.

Path B produces GGUF artifacts for InstructSAM's Qwen3-VL LM + vision projector,
using llama.cpp's existing Qwen3-VL support with a small tensor-filter patch to
skip the grounding_model half (which Path A handles separately via a
sam3cpp fork).

## Reproduction

One command from a clean checkout of this branch:

```bash
bash tools/pathb_convert_lm_mmproj.sh
```

Prerequisites:
- HF-cached InstructSAM-2B checkpoint (~8 GB) at `~/.cache/huggingface/hub/models--CircleRadon--InstructSAM-2B/`
- Python 3.10+, cmake, gcc/g++ toolchain
- ~15 GB free disk (llama.cpp source + build + venv + GGUF outputs)
- ~5 GB free RAM for build; ~4 GB peak RSS for inference

Outputs to `/tmp/pathb-work/gguf-out/`:
- `instructsam-lm-f16.gguf` (3.3 GB) — Qwen3-VL text model
- `instructsam-mmproj-f16.gguf` (782 MB) — vision projector

## What the recipe does

1. **Symlinks InstructSAM's checkpoint into a fixed-arch working dir** — no
   weight duplication. Rewrites `config.json` to declare
   `Qwen3VLForConditionalGeneration` (llama.cpp doesn't know
   `InstructSAMForConditionalGeneration`); strips 16 InstructSAM-only config
   fields the converter ignores. Rewrites `tokenizer_config.json`'s
   `extra_special_tokens` from list to empty dict (actual tokens live in
   `tokenizer.json` and stay intact).
2. **Clones llama.cpp shallow** and applies
   `docs/instructsam/llamacpp-qwen3vl-instructsam-filter.patch` — 10-line diff
   to `conversion/qwen3vl.py` that skips InstructSAM's grounding_model tensors
   in the Qwen3-VL text-model converter.
3. **Sets up a venv** with llama.cpp's pinned `transformers==4.57.6` (the
   converter is version-sensitive; the system env's `transformers>=5` breaks
   the tokenizer loading path).
4. **Runs converter twice** — once for the LM, once for `--mmproj` (vision).
5. **Builds `llama-mtmd-cli`** CPU-only (~2-3 min on 12 cores). No CUDA, no
   Metal, no BLAS — just the minimal target for smoke testing.

## Validation done

Smoke test on `~/Downloads/warehouse_rgb.jpg`:

```bash
llama-mtmd-cli --model instructsam-lm-f16.gguf --mmproj instructsam-mmproj-f16.gguf \
  --image warehouse_rgb.jpg --prompt "Please describe this image in one sentence." \
  -n 30 --ctx-size 2048 --temp 0 --threads 8
# → "three people walking through a warehouse with cardboard boxes."
```

On the segmentation prompt (`Please segment the box, the person, the shelf,
and the forklift in the image.`) the model correctly emits the four
`<|object_ref_start|>NAME<|object_ref_end|>` object spans with names matching
the request. So the LM knows the segmentation task and identifies the right
objects.

## Open issue: `[SEG]` token emission divergence

**Symptom:** Under greedy sampling (`--temp 0`), our converted model emits
`<|mask_start|>` (id 151671) but then `<|object_ref_end|>` (id 151647)
instead of `[SEG]` (id 151670). Repeats the wrong pattern until the token
budget is exhausted.

**Contrast with PyTorch reference:** Same weights, same prompt, run via
InstructSAM's own `mm_infer_segmentation` in Python correctly emits
`<|object_ref_start|>NAME<|object_ref_end|><|mask_start|>[SEG]<|mask_end|>`
for each requested object.

**Attempted fixes that didn't help:**
- `--logit-bias 151670+10` — no change in output (may not be reaching mtmd
  sampler; needs verification)

**Not yet investigated (probable-cause ranking):**
1. Chat template — llama.cpp's Qwen3-VL template may prepend a system message
   or use special token wrappings InstructSAM's inference does not
2. Sampling defaults (repetition penalty, top-k, min-p) diverging from
   `transformers.generate()` defaults
3. `[SEG]` token being flagged "special" and suppressed from sampling by
   default in llama.cpp
4. bf16 → f16 dequantization drift affecting near-tied logits at these
   positions

**Path forward for next investigation:** Compare llama.cpp vs PyTorch
token-by-token — same prompt, same weights, dump top-10 logits at the
position immediately after `<|mask_start|>` from both runtimes. Whichever
factor moves them out of alignment is the fix target. Then either patch
llama.cpp or use `llama-cpp-python` with a custom logits processor.

## Downstream (blocked on the `[SEG]` fix)

Once `[SEG]` tokens are emitted correctly:

1. Patch `llama-mtmd-cli` (or fork it as `llama-mtmd-seg-extract`) to enable
   `ctx_params.embeddings = true` and dump hidden states via
   `llama_get_embeddings_ith()` at positions where the sampled token id is
   `151670`.
2. Write a Python wrapper that loads InstructSAM's `grounding_model` submodule
   (from the original PyTorch checkpoint) and feeds the captured hidden states
   as decoder queries via `mask_hidden_fcs`.
3. Compare produced masks against the pure-PyTorch reference (from
   `instructsam_pr_validation.py` — earlier smoke that ran successfully on the
   same warehouse image).

Estimated engineering: ~1 day once `[SEG]` divergence is diagnosed.
