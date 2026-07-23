# Path B — GGUF conversion of InstructSAM's Qwen3-VL half via llama.cpp

**Status (2026-07-23):** Conversion + build + basic multimodal inference proven
end-to-end. `[SEG]` token emission from the converted LM diverges from the
PyTorch reference (see "Open issue" below) — investigation deferred.

**Update (later):** The "Open issue" section below concludes that InstructSAM's
embedding-injection mechanism requires "several weeks of ggml C++ engineering"
to add to llama.cpp. **That conclusion was wrong** — llama.cpp's public API
already supports both mid-generation inputs_embeds (via `llama_batch.embd`)
and hidden-state capture at specific positions (via `llama_get_embeddings_ith`).
See `docs/instructsam/LM_INTEGRATION_PLAN.md` for the correct API surface and
the ~3-5 day integration plan.

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

## Open issue: `[SEG]` is not a generated token — investigation resolved with a surprise

**Initial symptom:** Under greedy sampling (`--temp 0`), our converted model
emits `<|mask_start|>` (id 151671) followed by `<|object_ref_end|>` (id
151647) instead of `[SEG]` (id 151670).

**Sampler / bias check:** `--logit-bias 151670+100` DOES change output (model
generates `[SEG][SEG][SEG]...` under boosted bias), so mtmd-cli's sampler
correctly consumes bias flags. `-100` bias on `<|object_ref_end|>` produces
degenerate `box_column_column...` output. This proves: (a) all listed
special tokens are sampled unrestricted (`[SEG]` is `CONTROL` type in the
GGUF vocab but so are `<|mask_start|>` and `<|object_ref_end|>` — llama.cpp
does emit `CONTROL` tokens without special flagging), (b) the model's raw
logits genuinely rank `<|object_ref_end|>` above `[SEG]` at that position.

**Actual root cause (from reading InstructSAM's PyTorch inference):**

`[SEG]` is **not a generated token in InstructSAM's real pipeline.** Read
`instructsam/__init__.py:51`:

```python
outputs = outputs.replace("<|object_ref_end|>", "<|object_ref_end|><|mask_start|>[SEG]<|mask_end|>")
```

The `[SEG]` string is inserted into the DISPLAY output via `.replace()`
post-hoc. The actual mask generation happens inside `model.inference()`
via a completely different mechanism (`instructsam/models/instructsam.py:663-681`):

1. During autoregressive generation, when the LM emits `<|object_ref_end|>`,
   InstructSAM's `prepare_inputs_for_generation` override intercepts the
   next step.
2. Instead of `input_ids`, it passes `inputs_embeds` = concat of:
   - `<|object_ref_end|>` embedding
   - `<|mask_start|>` embedding
   - `mask_queries` tensor (learnable `[10, 2048]` — the actual "query slots")
   - `<|mask_end|>` embedding
3. LM forward runs on those embeddings; hidden states at the
   `mask_queries` positions become the mask decoder queries.
4. Autoregressive generation continues after the injected sequence.

**What this means for llama.cpp's runtime:**

llama.cpp's CLI + standard C API don't support **generation via
`inputs_embeds` mid-sequence**. The generation loop is `input_ids`-based
end-to-end. Supporting InstructSAM's embedding-injection pattern requires:

- Detecting `<|object_ref_end|>` emission during generation
- Loading the separate `mask_queries [10, 2048]` tensor (not currently
  in our GGUF — filtered out with the other InstructSAM glue)
- Running forward pass on injected embeddings (llama.cpp's ggml graph
  is set up for token-id inputs; embedding inputs need new API surface)
- Extracting hidden states at 10 specific positions per object
- Continuing generation after the injected block

Realistically **several weeks of ggml C++ engineering**. Not a "small
variant CLI patch."

## Path B, honestly reassessed

**Original pitch** (days-of-engineering): quantize LM via llama.cpp,
capture hidden states at `[SEG]` positions, feed PyTorch mask decoder.

**Actual state**: assumed `[SEG]` was a real emitted token; it isn't.
The mask-decoder query is the LM's hidden state at INJECTED-embedding
positions, and injecting embeddings requires runtime API llama.cpp
doesn't expose today.

**What Path B artifacts still validate:**
- LM+visual weights convert to GGUF cleanly (proven — files are 3.3 GB + 782 MB)
- llama.cpp can load + run the converted model on real images (proven —
  produces sensible image descriptions)
- llama.cpp correctly reproduces the LM's token-level generation up to
  `<|object_ref_end|>` (proven — 4-object structure emitted correctly)

**What Path B artifacts do NOT enable:**
- End-to-end mask generation from our GGUFs (blocked on embedding-injection
  runtime, weeks of C++ work)

## Revised roadmap

**For E2E mask validation on `warehouse_rgb.jpg`:** use pure-PyTorch
InstructSAM as the reference oracle. Already-validated smoke exists — see
`instructsam_pr_validation.py` from 2026-07-22 which produced clean
person/box/shelf/forklift masks on the same image.

**For a deployable ggml-native InstructSAM runtime:** Path A (sam3cpp
fork) is now the ONLY viable ggml path — it's building a custom runtime
anyway, so the embedding-injection mechanism can be built in as part of
that C++ work. This adds ~1 week to Path A's estimate (was 1.5-2 weeks;
now 2.5-3 weeks) but doesn't fundamentally change its architecture.

**For text-only Qwen3-VL usage** (image description, VQA without masks):
Path B's GGUFs are directly usable via `llama-mtmd-cli` today. Not the
segmentation endpoint, but a real capability from the same weights.
