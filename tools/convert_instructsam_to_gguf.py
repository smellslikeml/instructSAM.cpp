#!/usr/bin/env python3
"""InstructSAM → sam3cpp GGUF converter (first-pass, decoder-graph WIP).

InstructSAM (CircleRadon/InstructSAM-2B) reuses Meta SAM3's grounding_model
architecture verbatim as the mask-generation half, then feeds Qwen3-VL LM
`[SEG]` token hidden states in as decoder queries via custom glue
(`mask_hidden_fcs`, `text_hidden_fcs`, `mask_queries`).

The grounding_model components map 1:1 to sam3cpp's expected tensor
namespace after prefix rewriting. This converter handles that remap and
emits a GGUF that sam3cpp's decoder CAN load — but it will NOT run
correctly until sam3cpp's decoder graph is extended to support
InstructSAM's double-cross-attention-per-layer variant (text_cross_attn
and vision_cross_attn as SEPARATE blocks; sam3cpp currently fuses text
and vision upstream and runs a single cross_attn per decoder layer).

See DESIGN.md for the decoder-graph modifications needed on the C++ side.

This tool is scoped to the InstructSAM `grounding_model` submodule only.
The Qwen3-VL LM half (`model.language_model.*`, `model.visual.*`) is
handled separately via llama.cpp's native Qwen3-VL support; the
custom glue tensors (`mask_hidden_fcs`, `text_hidden_fcs`,
`mask_queries`) are emitted as extras that a wrapping runtime pipes
between the two halves.

Usage:
    python3 tools/convert_instructsam_to_gguf.py \\
        --input  ~/.cache/huggingface/hub/models--CircleRadon--InstructSAM-2B/snapshots/<sha>/model.safetensors \\
        --output models/instructsam-decoder-f16.gguf \\
        [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors import safe_open


# ────────────────────────────────────────────────────────────────────────
# Namespace remap: InstructSAM prefix  →  sam3cpp expected prefix
#
# InstructSAM reuses Meta SAM3's grounding_model architecture verbatim
# for its mask-generation half, so component-level correspondence is 1:1.
# sam3cpp identifies components via `infer_stage()` prefix matching
# (tools/convert_mlx_sam3_to_gguf.py L201-220).
# ────────────────────────────────────────────────────────────────────────

INSTRUCTSAM_TO_SAM3CPP: list[tuple[str, str]] = [
    # Grounding model submodules — all live under model.grounding_model.model.*
    #
    # Longer-prefix entries must come BEFORE their parent prefixes so
    # remap_name() matches specifically. `vision_encoder.neck` (FPN) has
    # to precede `vision_encoder.backbone` in the mapping order.
    ("model.grounding_model.model.vision_encoder.neck",
     "backbone.vision_backbone.convs"),   # sam3cpp bucket: `vision_convs`
    ("model.grounding_model.model.vision_encoder.backbone",
     "backbone.vision_backbone.trunk"),
    ("model.grounding_model.model.detr_encoder",
     "transformer.encoder"),
    ("model.grounding_model.model.detr_decoder",
     "transformer.decoder"),
    ("model.grounding_model.model.mask_decoder",
     "segmentation_head"),
    ("model.grounding_model.model.dot_product_scoring",
     "dot_prod_scoring"),
    ("model.grounding_model.model.geometry_encoder",
     "geometry_encoder"),
    ("model.grounding_model.model.text_projection",
     "text_projection"),
    # Custom InstructSAM glue — no sam3cpp equivalent; emit under
    # `instructsam.*` so a wrapping runtime can look them up.
    ("model.mask_hidden_fcs", "instructsam.mask_hidden_fcs"),
    ("model.text_hidden_fcs", "instructsam.text_hidden_fcs"),
    ("model.mask_queries",    "instructsam.mask_queries"),
]

# Prefixes to skip entirely — handled by llama.cpp's own Qwen3-VL path.
SKIP_PREFIXES = (
    "model.language_model.",
    "model.visual.",
)


def remap_name(instructsam_name: str) -> str | None:
    """Rewrite an InstructSAM tensor name to sam3cpp's expected schema.

    Returns None if the tensor should be skipped (handled elsewhere) or
    doesn't match any known component (which means our mapping table is
    incomplete for this checkpoint).
    """
    if any(instructsam_name.startswith(p) for p in SKIP_PREFIXES):
        return None
    for src_prefix, dst_prefix in INSTRUCTSAM_TO_SAM3CPP:
        if instructsam_name == src_prefix:
            return dst_prefix
        if instructsam_name.startswith(src_prefix + "."):
            return dst_prefix + instructsam_name[len(src_prefix):]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="Path to InstructSAM's model.safetensors")
    ap.add_argument("--output", type=Path,
                    help="Path to output GGUF (skipped in --dry-run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Enumerate name mapping + report unmapped tensors; "
                         "don't emit GGUF. Use to validate mapping coverage.")
    ap.add_argument("--report", type=Path,
                    help="Write mapping-report JSON to this path")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"✗ input not found: {args.input}", file=sys.stderr)
        return 1

    print(f"reading {args.input}")
    with safe_open(str(args.input), framework="pt") as f:
        all_names = list(f.keys())

    # Classify each tensor
    mapped: list[tuple[str, str, tuple[int, ...]]] = []      # (src_name, dst_name, shape)
    skipped: list[str] = []                                   # Qwen3-VL — handled by llama.cpp
    unmapped: list[str] = []                                  # ← the surface we care about

    for name in all_names:
        if any(name.startswith(p) for p in SKIP_PREFIXES):
            skipped.append(name)
            continue
        remapped = remap_name(name)
        if remapped is None:
            unmapped.append(name)
            continue
        with safe_open(str(args.input), framework="pt") as g:
            shape = tuple(g.get_slice(name).get_shape())
        mapped.append((name, remapped, shape))

    # Bucket mapped tensors by destination top-level component
    by_bucket: dict[str, int] = defaultdict(int)
    for _, dst, _ in mapped:
        top = dst.split(".", 1)[0]
        by_bucket[top] += 1

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n=== summary ===")
    print(f"  total tensors:  {len(all_names)}")
    print(f"  mapped:         {len(mapped)}   → will emit under sam3cpp schema")
    print(f"  skipped:        {len(skipped)}  → Qwen3-VL half (llama.cpp handles)")
    print(f"  unmapped:       {len(unmapped)} → ← IMPORTANT: these are gaps in the mapping")

    print(f"\n=== mapped by destination component ===")
    for bucket, count in sorted(by_bucket.items(), key=lambda x: -x[1]):
        print(f"  {count:5d}  {bucket}")

    if unmapped:
        print(f"\n=== ⚠ unmapped tensors (first 20) ===")
        for n in unmapped[:20]:
            print(f"  {n}")
        if len(unmapped) > 20:
            print(f"  … and {len(unmapped) - 20} more")

    # ── Verify shape sanity on a few key mapped tensors ────────────────
    # Sam3cpp's decoder loads tensors under names like
    # `transformer.decoder.layers.0.self_attn.query_proj.weight`.
    # Confirm we're producing that name for InstructSAM's equivalent.
    print(f"\n=== spot-check mapping on decoder tensors ===")
    layer0_examples = [(s, d, sh) for s, d, sh in mapped
                       if "detr_decoder.layers.0." in s][:6]
    for src, dst, shape in layer0_examples:
        print(f"  {src[-70:]:70s}\n     → {dst[-70:]:70s}  {shape}")

    # ── Report ─────────────────────────────────────────────────────────
    if args.report:
        report = {
            "input": str(args.input),
            "totals": {
                "input_tensors": len(all_names),
                "mapped": len(mapped),
                "skipped_qwen3vl": len(skipped),
                "unmapped": len(unmapped),
            },
            "mapped_by_component": dict(by_bucket),
            "unmapped_tensor_names": unmapped,
            "mapping_table": INSTRUCTSAM_TO_SAM3CPP,
        }
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nreport: {args.report}")

    if args.dry_run:
        print("\n(dry-run mode — GGUF not emitted)")
        return 0 if not unmapped else 2

    # ── GGUF emission ──────────────────────────────────────────────────
    # NOT implemented in this first pass. The mapping is the interesting
    # part; the actual GGUFWriter invocation is straightforward
    # (see convert_mlx_sam3_to_gguf.py for reference). Deferred until:
    #   1. Mapping-coverage is 100% (no unmapped tensors)
    #   2. sam3cpp's decoder graph supports double-cross-attn (see DESIGN.md)
    #
    # Emitting GGUF without (2) produces a file sam3cpp will "load" but
    # can't run correctly — worse than not emitting at all.
    print(f"\n⚠ GGUF emission not implemented in this first pass.")
    print(f"  Blocking on: DESIGN.md's decoder-graph modifications.")
    print(f"  Use --dry-run to validate mapping coverage in the meantime.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
