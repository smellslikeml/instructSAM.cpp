#!/usr/bin/env bash
# Path B: convert InstructSAM's LM+visual halves to GGUF via llama.cpp's
# native Qwen3-VL converter, and build llama-mtmd-cli for smoke testing.
#
# Path A (sam3cpp fork) handles the grounding_model half — see
# docs/instructsam/DESIGN.md. Path B (this script) handles the LM half
# and provides a runtime for capturing [SEG] token hidden states that
# feed into the Path A decoder.
#
# Pre-req: HF-cached InstructSAM-2B checkpoint at
#   ~/.cache/huggingface/hub/models--CircleRadon--InstructSAM-2B/
#
# Outputs (in $OUT_DIR):
#   instructsam-lm-f16.gguf      (~3.3 GB — Qwen3-VL text model)
#   instructsam-mmproj-f16.gguf  (~782 MB — vision projector)
#   llama-mtmd-cli binary        (for smoke inference)

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────
: "${WORK_DIR:=/tmp/pathb-work}"
: "${OUT_DIR:=${WORK_DIR}/gguf-out}"
: "${LLAMACPP_REPO:=https://github.com/ggml-org/llama.cpp.git}"
: "${INSTRUCTSAM_HF_CACHE:=$HOME/.cache/huggingface/hub/models--CircleRadon--InstructSAM-2B/snapshots}"

# Directory containing THIS script (so we can find the patch file)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PATCH_FILE="${SCRIPT_DIR}/../docs/instructsam/llamacpp-qwen3vl-instructsam-filter.patch"

mkdir -p "$WORK_DIR" "$OUT_DIR"

# ── Step 1: locate the HF-cached InstructSAM checkpoint ───────────────
if [ ! -d "$INSTRUCTSAM_HF_CACHE" ]; then
    echo "ERROR: InstructSAM cache not found at $INSTRUCTSAM_HF_CACHE" >&2
    echo "First: python -c 'from huggingface_hub import snapshot_download; snapshot_download(\"CircleRadon/InstructSAM-2B\")'" >&2
    exit 1
fi
INSTRUCTSAM_SNAP=$(ls -d "$INSTRUCTSAM_HF_CACHE"/*/ | head -1)
INSTRUCTSAM_SNAP="${INSTRUCTSAM_SNAP%/}"
echo "→ InstructSAM snapshot: $INSTRUCTSAM_SNAP"

# ── Step 2: build a fixed-arch checkpoint dir ─────────────────────────
# InstructSAM's config declares `InstructSAMForConditionalGeneration` which
# llama.cpp doesn't know. We present it as vanilla `Qwen3VLForConditionalGeneration`
# via a config swap; the tensor names already match Qwen3-VL conventions.
FIXED_DIR="${WORK_DIR}/instructsam-as-qwen3vl"
rm -rf "$FIXED_DIR"; mkdir -p "$FIXED_DIR"

# Symlink weights + tokenizer files (avoid 8 GB duplication)
ln -sf "$INSTRUCTSAM_SNAP/model.safetensors" "$FIXED_DIR/model.safetensors"
for f in tokenizer.json chat_template.jinja generation_config.json processor_config.json; do
    [ -f "$INSTRUCTSAM_SNAP/$f" ] && ln -sf "$INSTRUCTSAM_SNAP/$f" "$FIXED_DIR/$f"
done

# Rewrite config.json — swap architectures + strip InstructSAM-only fields
python3 - <<PYEOF
import json, sys
from pathlib import Path
cfg = json.loads(Path("$INSTRUCTSAM_SNAP/config.json").read_text())
cfg['architectures'] = ['Qwen3VLForConditionalGeneration']
for k in ('bce_loss_weight', 'cls_loss_weight', 'dice_loss_weight',
          'loss_sample_points', 'mask_decoder_model', 'max_seg_nums',
          'mm_mask_decoder', 'out_dim',
          'ref_end_token_index', 'ref_start_token_index',
          'region_token_index', 'seg_decoder', 'seg_encoder',
          'seg_end_token_index', 'seg_start_token_index', 'seg_token_index'):
    cfg.pop(k, None)
Path("$FIXED_DIR/config.json").write_text(json.dumps(cfg, indent=2))
PYEOF

# Rewrite tokenizer_config.json — extra_special_tokens is a list in InstructSAM's
# checkpoint but transformers expects a dict; empty-dict is a valid override since
# the actual token entries live in tokenizer.json.
python3 - <<PYEOF
import json
from pathlib import Path
tok = json.loads(Path("$INSTRUCTSAM_SNAP/tokenizer_config.json").read_text())
tok['extra_special_tokens'] = {}
Path("$FIXED_DIR/tokenizer_config.json").write_text(json.dumps(tok, indent=2))
PYEOF
echo "→ fixed-arch dir: $FIXED_DIR"

# ── Step 3: clone llama.cpp + apply the InstructSAM tensor-filter patch ──
LLAMACPP_DIR="${WORK_DIR}/llama.cpp"
if [ ! -d "$LLAMACPP_DIR" ]; then
    git clone --depth=1 "$LLAMACPP_REPO" "$LLAMACPP_DIR"
fi
cd "$LLAMACPP_DIR"

# Apply the patch (idempotent — skip if already applied)
if ! grep -q "LOCAL PATCH: InstructSAM wraps Qwen3-VL" conversion/qwen3vl.py; then
    echo "→ applying InstructSAM tensor-filter patch"
    git apply "$PATCH_FILE"
else
    echo "→ patch already applied, skipping"
fi

# ── Step 4: set up conversion venv (llama.cpp pins transformers==4.57.6) ──
VENV="${WORK_DIR}/convert-venv"
if [ ! -d "$VENV" ]; then
    python3 -m venv --system-site-packages "$VENV"
    "$VENV/bin/pip" install --quiet -r requirements/requirements-convert_hf_to_gguf.txt
fi

# ── Step 5: convert LM half ───────────────────────────────────────────
LM_GGUF="${OUT_DIR}/instructsam-lm-f16.gguf"
if [ ! -f "$LM_GGUF" ]; then
    echo "→ converting LM half → $LM_GGUF"
    "$VENV/bin/python" convert_hf_to_gguf.py --outfile "$LM_GGUF" --outtype f16 "$FIXED_DIR"
else
    echo "→ LM GGUF exists, skipping"
fi

# ── Step 6: convert vision mmproj ─────────────────────────────────────
MMPROJ_GGUF="${OUT_DIR}/instructsam-mmproj-f16.gguf"
if [ ! -f "$MMPROJ_GGUF" ]; then
    echo "→ converting mmproj → $MMPROJ_GGUF"
    "$VENV/bin/python" convert_hf_to_gguf.py --outfile "$MMPROJ_GGUF" --outtype f16 --mmproj "$FIXED_DIR"
else
    echo "→ mmproj GGUF exists, skipping"
fi

# ── Step 7: build llama-mtmd-cli (CPU-only, minimal) ──────────────────
if [ ! -f "${LLAMACPP_DIR}/build/bin/llama-mtmd-cli" ]; then
    echo "→ building llama-mtmd-cli"
    cmake -S . -B build \
        -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DGGML_HIP=OFF \
        -DGGML_BLAS=OFF -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build build --target llama-mtmd-cli --parallel "$(nproc)"
else
    echo "→ llama-mtmd-cli exists, skipping build"
fi

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "=== Path B artifacts ready ==="
ls -lh "$LM_GGUF" "$MMPROJ_GGUF"
echo "  binary: ${LLAMACPP_DIR}/build/bin/llama-mtmd-cli"
echo ""
echo "Smoke test:"
echo "  ${LLAMACPP_DIR}/build/bin/llama-mtmd-cli \\"
echo "    --model $LM_GGUF \\"
echo "    --mmproj $MMPROJ_GGUF \\"
echo "    --image path/to/image.jpg \\"
echo "    --prompt 'Please describe this image in one sentence.' \\"
echo "    -n 30 --ctx-size 2048 --temp 0 --threads 8"
