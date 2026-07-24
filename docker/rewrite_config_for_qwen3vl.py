#!/usr/bin/env python3
"""Rewrite an InstructSAM HF checkpoint's config.json + tokenizer_config.json
so llama.cpp's Qwen3-VL converter accepts it as a vanilla
Qwen3VLForConditionalGeneration checkpoint.

Mirrors tools/pathb_convert_lm_mmproj.sh steps 2 + 3.

Usage:
    rewrite_config_for_qwen3vl.py <src_dir> <dst_dir>
"""
import json
import sys
from pathlib import Path

STRIP_KEYS = (
    "bce_loss_weight", "cls_loss_weight", "dice_loss_weight",
    "loss_sample_points", "mask_decoder_model", "max_seg_nums",
    "mm_mask_decoder", "out_dim",
    "ref_end_token_index", "ref_start_token_index",
    "region_token_index", "seg_decoder", "seg_encoder",
    "seg_end_token_index", "seg_start_token_index", "seg_token_index",
)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: rewrite_config_for_qwen3vl.py <src_dir> <dst_dir>",
              file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    cfg["architectures"] = ["Qwen3VLForConditionalGeneration"]
    for k in STRIP_KEYS:
        cfg.pop(k, None)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    tok = json.loads((src / "tokenizer_config.json").read_text())
    # InstructSAM ships extra_special_tokens as a list; transformers expects
    # a dict (the actual token entries live in tokenizer.json anyway).
    tok["extra_special_tokens"] = {}
    (dst / "tokenizer_config.json").write_text(json.dumps(tok, indent=2))
    print(f"rewrote config.json + tokenizer_config.json in {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
