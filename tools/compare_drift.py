#!/usr/bin/env python3
"""Compare DETR-chain intermediate tensors between our CLI's dumps and
the PyTorch reference oracle, to isolate the first stage where drift enters.

Usage:
    # 1. Run the CLI with dumps enabled:
    SAM3_CLI_DUMP_INTERMEDIATES=/tmp/instructsam_dumps ./build/instructsam \\
        --image /path/to/warehouse_rgb.jpg \\
        --query "Please segment 'box', 'person', 'shelf', 'forklift' in this image." \\
        --run-vision \\
        --out-dir /tmp/x --lm /path/to/instructsam-lm-Q4_K_M.gguf

    # 2. Compare against the reference oracle:
    python3 tools/compare_drift.py \\
        --dumps /tmp/instructsam_dumps \\
        --ref   /tmp/pathA_reference/warehouse_rgb \\
        --phrases "box,person,shelf,forklift"

Output: per-phrase, per-stage table of cos_sim, max_abs_diff, rel_L2 vs
reference. The first stage where cos_sim drops materially below ~0.995
is the drift entry point — that's where the numerical investigation
focuses.

The stage list below maps our dump names to the reference oracle's file
names in <ref>/binaries_obj<pi>/. Stages are ordered by pipeline
dependency so drift accumulation is legible top-to-bottom.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np


# (our-dump-name, ref-oracle-name).  Add pairs here as new intermediates
# get dumped; leave pairs whose ref file doesn't exist — the comparison
# skips them with a "no ref" note.
STAGE_PAIRS = [
    # LM-side and inputs to DETR (per-phrase)
    ("lmb_seg_output_embeddings.f32", "lmb_seg_output_embeddings.f32"),
    ("lmb_mask_hidden_fcs_out.f32",   "lmb_mask_hidden_fcs_out.f32"),
    ("enc_text_features.f32",         "enc_text_features.f32"),
    ("enc_vision_features_flat.f32",  "enc_vision_features_flat.f32"),
    # DETR encoder
    ("detr_encoder_last_hidden.f32",  "md_pca_encoder_in.f32"),  # same tensor, different name in ref
    # DETR decoder — per-layer
    ("expected_layer_0.f32",          "expected_layer_0.f32"),
    ("expected_layer_1.f32",          "expected_layer_1.f32"),
    ("expected_layer_2.f32",          "expected_layer_2.f32"),
    ("expected_layer_3.f32",          "expected_layer_3.f32"),
    ("expected_layer_4.f32",          "expected_layer_4.f32"),
    ("expected_layer_5.f32",          "expected_layer_5.f32"),
    # Post-decoder
    ("md_decoder_queries.f32",        "md_decoder_queries.f32"),
    # Mask decoder side
    ("md_pixel_embed.f32",            "md_pixel_embed.f32"),
    ("md_pred_masks.f32",             "md_pred_masks.f32"),
]

# Per-image (not per-phrase) tensors — compared once at the top.
IMAGE_STAGE_PAIRS = [
    ("md_fpn_bb0.f32", "binaries_obj0/md_fpn_bb0.f32"),
    ("md_fpn_bb1.f32", "binaries_obj0/md_fpn_bb1.f32"),
    ("md_fpn_bb2.f32", "binaries_obj0/md_fpn_bb2.f32"),
]


def read_bin1(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    with path.open("rb") as f:
        if f.read(4) != b"BIN1":
            return None
        ndim = struct.unpack("<i", f.read(4))[0]
        shape = [struct.unpack("<q", f.read(8))[0] for _ in range(ndim)]
        n = 1
        for d in shape:
            n *= d
        return np.frombuffer(f.read(n * 4), dtype=np.float32).reshape(shape)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def compare_pair(ours_path: Path, ref_path: Path) -> tuple[str, str, str] | None:
    o = read_bin1(ours_path)
    r = read_bin1(ref_path)
    if o is None:
        return None  # our dump missing — skip silently, may not be produced yet
    if r is None:
        return ("no ref", "-", "-")
    if o.shape != r.shape:
        return (f"shape mismatch: ours={list(o.shape)} ref={list(r.shape)}", "-", "-")
    cs = cos_sim(o, r)
    mad = float(np.abs(o - r).max())
    rel = float(np.linalg.norm(o - r) / (np.linalg.norm(r) + 1e-9))
    return (f"{cs:.6f}", f"{mad:.4g}", f"{rel:.4g}")


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"  {'stage':<40} {'cos_sim':<10} {'max_diff':<12} {'rel_L2':<10}")
    print(f"  {'-'*40} {'-'*10} {'-'*12} {'-'*10}")


def print_row(name: str, result: tuple[str, str, str]) -> None:
    if result is None:
        return
    cs, mad, rel = result
    tag = ""
    try:
        cs_val = float(cs)
        if cs_val < 0.99:
            tag = "  ← drift"
        elif cs_val < 0.999:
            tag = "  ← minor drift"
    except ValueError:
        tag = "  ← " + cs
    print(f"  {name:<40} {cs:<10} {mad:<12} {rel:<10}{tag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dumps",   type=Path, required=True,
                    help="Directory the CLI wrote to (SAM3_CLI_DUMP_INTERMEDIATES)")
    ap.add_argument("--ref",     type=Path, required=True,
                    help="Reference oracle root (e.g. /tmp/pathA_reference/warehouse_rgb)")
    ap.add_argument("--phrases", type=str, required=True,
                    help="Comma-separated phrase list, in the order the CLI emitted them")
    args = ap.parse_args()

    phrases = [p.strip() for p in args.phrases.split(",") if p.strip()]

    print_header("Per-image (backbone / FPN features)")
    for ours_name, ref_rel in IMAGE_STAGE_PAIRS:
        result = compare_pair(args.dumps / ours_name, args.ref / ref_rel)
        print_row(ours_name, result)

    for pi, phrase in enumerate(phrases):
        print_header(f"obj {pi}: '{phrase}'")
        our_obj_dir = args.dumps / f"obj{pi}"
        ref_obj_dir = args.ref / f"binaries_obj{pi}"
        for ours_name, ref_name in STAGE_PAIRS:
            result = compare_pair(our_obj_dir / ours_name, ref_obj_dir / ref_name)
            print_row(ours_name, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
