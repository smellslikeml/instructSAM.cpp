#!/usr/bin/env python3
"""Weight-level round-trip validation for the InstructSAM GGUF.

For each canonical tensor in the sam3cpp namespace, verifies that:
  1. It resolves to a real tensor in the emitted GGUF via the sidecar
  2. Its shape matches the source InstructSAM safetensors
  3. Its content matches after dtype normalization (fp16 SHA-256)

This is weight-level correctness (converter validation), separate from
runtime-level correctness (which is what the reference-tensor manifest
covers).

Success criteria:
  - 100% of mapped tensors resolvable via sidecar
  - 100% shape-match
  - 100% fp16-content-match (bit-identical after cast normalization)

Any failure identifies exactly which tensor + why, before any C++ work
starts.

Usage:
    python3 validate_instructsam_gguf.py \\
        --gguf /tmp/pathA_gguf/instructsam-grounding-f16.gguf \\
        --sidecar /tmp/pathA_gguf/instructsam-grounding-f16.gguf.tensor_map.json \\
        --source ~/.cache/huggingface/hub/models--CircleRadon--InstructSAM-2B/.../model.safetensors
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

try:
    from gguf import GGUFReader
except ImportError:
    print("gguf module needed", file=sys.stderr); sys.exit(1)


# Duplicate the converter's remap table (kept here to avoid import cycle
# and because validation should be self-contained)
INSTRUCTSAM_TO_SAM3CPP = [
    ("model.grounding_model.model.vision_encoder.neck",
     "backbone.vision_backbone.convs"),
    ("model.grounding_model.model.vision_encoder.backbone",
     "backbone.vision_backbone.trunk"),
    ("model.grounding_model.model.detr_encoder", "transformer.encoder"),
    ("model.grounding_model.model.detr_decoder", "transformer.decoder"),
    ("model.grounding_model.model.mask_decoder", "segmentation_head"),
    ("model.grounding_model.model.dot_product_scoring", "dot_prod_scoring"),
    ("model.grounding_model.model.geometry_encoder", "geometry_encoder"),
    ("model.grounding_model.model.text_projection", "text_projection"),
    ("model.mask_hidden_fcs", "instructsam.mask_hidden_fcs"),
    ("model.text_hidden_fcs", "instructsam.text_hidden_fcs"),
    ("model.mask_queries",    "instructsam.mask_queries"),
]


def remap(name: str) -> str | None:
    for src, dst in INSTRUCTSAM_TO_SAM3CPP:
        if name == src: return dst
        if name.startswith(src + "."): return dst + name[len(src):]
    return None


def sha256_fp16(t: torch.Tensor | np.ndarray) -> str:
    if isinstance(t, torch.Tensor):
        arr = t.to(torch.float16).cpu().contiguous().numpy()
    else:
        arr = t.astype(np.float16, copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path)
    ap.add_argument("--source", type=Path, required=True,
                    help="InstructSAM's original model.safetensors")
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    if args.sidecar is None:
        args.sidecar = args.gguf.with_suffix(args.gguf.suffix + ".tensor_map.json")

    # ── load sidecar ───────────────────────────────────────────────────
    tensor_map = json.loads(args.sidecar.read_text())
    canonical_to_short = {v: k for k, v in tensor_map.items()}
    print(f"sidecar: {len(tensor_map)} entries")

    # ── open GGUF ──────────────────────────────────────────────────────
    reader = GGUFReader(str(args.gguf))
    gguf_tensors = {t.name: t for t in reader.tensors}
    print(f"gguf:    {len(gguf_tensors)} tensors")

    # ── open source safetensors ────────────────────────────────────────
    src_handle = safe_open(str(args.source), framework="pt")
    src_names = list(src_handle.keys())
    # Build mapping from canonical name back to source name
    canonical_to_source = {}
    for sn in src_names:
        dst = remap(sn)
        if dst is not None:
            canonical_to_source[dst] = sn
    print(f"source:  {len(canonical_to_source)} tensors mappable to canonical")

    # ── per-tensor validation ──────────────────────────────────────────
    results = {"pass": [], "shape_mismatch": [], "content_mismatch": [],
               "missing_in_gguf": [], "missing_in_source": []}

    for canonical, src_name in sorted(canonical_to_source.items()):
        short = canonical_to_short.get(canonical)
        if short is None:
            results["missing_in_gguf"].append(canonical)
            continue
        gt = gguf_tensors.get(short)
        if gt is None:
            results["missing_in_gguf"].append(f"{canonical} → {short} (not in gguf)")
            continue

        # Source tensor (torch, may be bf16)
        src_t = src_handle.get_tensor(src_name)
        expected_shape = tuple(src_t.shape)
        # GGUF tensor shape is stored reversed vs pytorch (col-major); normalize
        gt_shape = tuple(reversed(gt.shape.tolist()))

        if gt_shape != expected_shape:
            results["shape_mismatch"].append(
                f"{canonical}: source={expected_shape}, gguf={gt_shape}"
            )
            continue

        # Content: hash both at fp16 for a consistent comparison
        # (converter cast bf16 → fp16/f32 based on target_qtype)
        src_sha = sha256_fp16(src_t)

        # Read gguf data — it's already fp16 or fp32 in the file
        gguf_arr = gt.data
        # gt.data comes back as int8/etc. for quantized; for F16/F32 it's the raw dtype
        # Reshape to expected shape (GGUF stores flat)
        gguf_arr = np.asarray(gguf_arr).reshape(expected_shape)
        gguf_sha = sha256_fp16(gguf_arr)

        if src_sha == gguf_sha:
            results["pass"].append(canonical)
        else:
            results["content_mismatch"].append(
                f"{canonical}: src_fp16={src_sha}, gguf_fp16={gguf_sha}"
            )

    src_handle.__exit__(None, None, None)

    # ── report ─────────────────────────────────────────────────────────
    total = sum(len(v) for v in results.values())
    print(f"\n=== validation ({total} canonical tensors) ===")
    for k, v in results.items():
        icon = "✓" if k == "pass" else "✗"
        print(f"  {icon} {k}: {len(v)}")

    for cat in ("shape_mismatch", "content_mismatch", "missing_in_gguf"):
        if results[cat]:
            print(f"\n  {cat}[:20]:")
            for item in results[cat][:20]:
                print(f"    {item}")

    if args.report:
        args.report.write_text(json.dumps(
            {k: v[:100] for k, v in results.items()}, indent=2, sort_keys=True))
        print(f"\n  report: {args.report}")

    return 0 if len(results["pass"]) == total else 2


if __name__ == "__main__":
    sys.exit(main())
