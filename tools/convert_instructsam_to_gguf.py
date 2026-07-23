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
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors import safe_open

# sam3cpp's schema lives in its bundled llama.cpp gguf-py checkout.
# Also common: system-installed gguf. Try both.
try:
    from gguf import GGMLQuantizationType, GGUFWriter
except ImportError:
    print("✗ gguf module not available; install with `pip install gguf`", file=sys.stderr)
    sys.exit(1)


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


# ────────────────────────────────────────────────────────────────────────
# Dual-aliasing — InstructSAM canonical ↔ sam3cpp stock-decoder canonical
#
# Sam3cpp's stock Decoder::run() uses SAM3-native tensor names
# (query_proj / ca_text / norm2 / etc.). InstructSAM's checkpoint uses
# transformers-style names (q_proj / text_cross_attn / self_attn_layer_norm
# / etc.). Structurally the tensors are identical.
#
# For sam3cpp's stock decoder to load InstructSAM tensors without a code
# fork, each dual-named tensor must be reachable under BOTH names. We
# implement this by emitting the tensor data twice in the GGUF (once per
# short_name), with both short_names present in the sidecar. Runtime cost:
# ~50-100 MB extra for decoder tensors. See docs/instructsam/DECODER-ALIASING.md.
#
# Non-decoder tensors (vision_encoder, mask_decoder, geometry_encoder,
# grounding, text_projection, InstructSAM glue) use identical names in
# both conventions and don't need aliasing.
# ────────────────────────────────────────────────────────────────────────

# Per-layer decoder tensor renames (InstructSAM sub-path → sam3cpp sub-path)
PER_LAYER_DECODER_RENAMES = {
    # Attention Q/K/V/O projections
    "self_attn.q_proj":         "self_attn.query_proj",
    "self_attn.k_proj":         "self_attn.key_proj",
    "self_attn.v_proj":         "self_attn.value_proj",
    "self_attn.o_proj":         "self_attn.out_proj",
    "text_cross_attn.q_proj":   "ca_text.query_proj",
    "text_cross_attn.k_proj":   "ca_text.key_proj",
    "text_cross_attn.v_proj":   "ca_text.value_proj",
    "text_cross_attn.o_proj":   "ca_text.out_proj",
    "vision_cross_attn.q_proj": "cross_attn.query_proj",
    "vision_cross_attn.k_proj": "cross_attn.key_proj",
    "vision_cross_attn.v_proj": "cross_attn.value_proj",
    "vision_cross_attn.o_proj": "cross_attn.out_proj",
    # Layer norms (post-norm placement matches; just different names)
    "self_attn_layer_norm":         "norm2",
    "text_cross_attn_layer_norm":   "catext_norm",
    "vision_cross_attn_layer_norm": "norm1",
    "mlp_layer_norm":               "norm3",
    # MLP linears (InstructSAM: mlp.fc1/fc2; sam3cpp: linear1/linear2)
    "mlp.fc1": "linear1",
    "mlp.fc2": "linear2",
}

# Non-per-layer decoder tensor renames — 1-indexed vs 0-indexed .layers.N,
# and top-level names differ.
GLOBAL_DECODER_RENAMES = {
    "transformer.decoder.output_layer_norm":     "transformer.decoder.norm",
    "transformer.decoder.box_head.layer1":       "transformer.decoder.bbox_embed.layers.0",
    "transformer.decoder.box_head.layer2":       "transformer.decoder.bbox_embed.layers.1",
    "transformer.decoder.box_head.layer3":       "transformer.decoder.bbox_embed.layers.2",
    "transformer.decoder.ref_point_head.layer1": "transformer.decoder.ref_point_head.layers.0",
    "transformer.decoder.ref_point_head.layer2": "transformer.decoder.ref_point_head.layers.1",
    "transformer.decoder.box_rpb_embed_x.layer1": "transformer.decoder.boxRPB_embed_x.layers.0",
    "transformer.decoder.box_rpb_embed_x.layer2": "transformer.decoder.boxRPB_embed_x.layers.1",
    "transformer.decoder.box_rpb_embed_y.layer1": "transformer.decoder.boxRPB_embed_y.layers.0",
    "transformer.decoder.box_rpb_embed_y.layer2": "transformer.decoder.boxRPB_embed_y.layers.1",
}


def sam3cpp_alias_for(instructsam_canonical: str) -> str | None:
    """Return sam3cpp's stock canonical name for a tensor, or None if
    the InstructSAM name is already what sam3cpp expects (no aliasing
    needed)."""
    # Per-layer decoder tensor: `transformer.decoder.layers.<N>.<sub>.<attr>`
    if instructsam_canonical.startswith("transformer.decoder.layers."):
        after = instructsam_canonical[len("transformer.decoder.layers."):]
        try:
            layer_idx, rest = after.split(".", 1)
        except ValueError:
            return None
        for src, dst in PER_LAYER_DECODER_RENAMES.items():
            if rest == src or rest.startswith(src + "."):
                new_rest = dst + rest[len(src):]
                return f"transformer.decoder.layers.{layer_idx}.{new_rest}"
        return None
    # Global decoder tensor
    for src, dst in GLOBAL_DECODER_RENAMES.items():
        if instructsam_canonical == src:
            return dst
        if instructsam_canonical.startswith(src + "."):
            return dst + instructsam_canonical[len(src):]
    return None


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
    # Schema follows sam3cpp's convention (convert_mlx_sam3_to_gguf.py):
    #   - short tensor names in the GGUF: t_<sha1(canonical_name)[:12]>
    #   - .tensor_map.json sidecar maps short_name → canonical_name
    #   - sam3cpp's runtime looks up by canonical name, resolves via alias
    #
    # Quantization: F16 for large weight matrices; F32 for sensitive tensors
    # (norm / bias / small vectors) to avoid dequant drift accumulating.
    if not args.output:
        print("✗ --output required (unless --dry-run)", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(args.output), "sam3")
    writer.add_name("instructsam-2b")
    writer.add_string("sam3.source_repo", "CircleRadon/InstructSAM-2B")
    writer.add_string("sam3.variant", "instructsam")
    writer.add_string("sam3.task", "image-segmentation")
    writer.add_string("general.quantized_by",
                      "sam3cpp-fork/tools/convert_instructsam_to_gguf.py")
    # SAM3 vision parameters (matches Meta's SAM3 image encoder)
    writer.add_int32("sam3.image_size", 1008)
    writer.add_int32("sam3.patch_size", 14)
    # InstructSAM decoder variant signal — runtime switches on this
    writer.add_bool("sam3.instructsam.double_cross_attn", True)
    writer.add_bool("sam3.instructsam.skip_encoder_fusion", False)  # keep for now; instructsam DOES use encoder
    # Note: text_layers is 0 because InstructSAM feeds LM embeddings externally
    # (no built-in text encoder in the grounding_model)
    writer.add_int32("sam3.text_layers", 0)

    # Determine per-tensor dtype
    def target_qtype(canonical_name: str, tensor: np.ndarray) -> GGMLQuantizationType:
        """F32 for sensitive tensors; F16 for the rest."""
        lower = canonical_name.lower()
        # Norms, biases, small position embeddings — keep at F32 to avoid
        # cumulative dequant error at ε-sensitive positions
        if lower.endswith(".bias"):
            return GGMLQuantizationType.F32
        if "norm" in lower or "layernorm" in lower:
            return GGMLQuantizationType.F32
        # Small vectors (embeddings for special tokens etc.) — F32
        if tensor.ndim <= 1 or tensor.size < 4096:
            return GGMLQuantizationType.F32
        return GGMLQuantizationType.F16

    def short_name(canonical: str) -> str:
        return "t_" + hashlib.sha1(canonical.encode()).hexdigest()[:12]

    # Deduplicate — sha1 collisions on 12 hex = 48 bits, extremely improbable
    # for our N<2000 tensors but we detect if it ever happens
    seen_shortnames: dict[str, str] = {}
    tensor_map: dict[str, str] = {}
    dtype_counts: dict[str, int] = defaultdict(int)
    total_out_bytes = 0

    print(f"\n=== writing GGUF: {args.output} ===")
    # Use torch backend for safetensors — numpy can't handle bfloat16 which
    # InstructSAM's checkpoint uses for most weights.
    import torch as _t
    with safe_open(str(args.input), framework="pt") as handle:
        for src_name, canonical_name, shape in mapped:
            t = handle.get_tensor(src_name)  # torch.Tensor, possibly bfloat16

            # Choose output dtype based on canonical name
            fake_np = np.zeros(t.shape, dtype=np.float32)  # shape-only for qtype decision
            qtype = target_qtype(canonical_name, fake_np)

            if qtype == GGMLQuantizationType.F16:
                stored = t.to(_t.float16).contiguous().numpy()
            else:
                stored = t.to(_t.float32).contiguous().numpy()

            sn = short_name(canonical_name)
            if sn in seen_shortnames:
                raise RuntimeError(
                    f"sha1 collision: {sn} maps to both "
                    f"{seen_shortnames[sn]!r} and {canonical_name!r}"
                )
            seen_shortnames[sn] = canonical_name
            tensor_map[sn] = canonical_name

            if qtype == GGMLQuantizationType.F16:
                writer.add_tensor(sn, stored)
            else:
                writer.add_tensor(sn, stored, raw_dtype=qtype)

            dtype_counts[qtype.name] += 1
            total_out_bytes += stored.nbytes

            # Dual-alias emission: if this tensor has a sam3cpp-canonical
            # equivalent (decoder-layer or decoder-global tensor with
            # renaming), emit the tensor data a SECOND time under a fresh
            # short_name so both names resolve at runtime. Sidecar entry
            # for the second short → sam3cpp_canonical.
            sam3cpp_alias = sam3cpp_alias_for(canonical_name)
            if sam3cpp_alias is not None:
                # Fresh short_name distinct from `sn` — hash the sam3cpp
                # canonical string so the two shorts differ
                alias_sn = short_name(sam3cpp_alias)
                if alias_sn == sn:
                    # sha1 collision on the aliased name — extremely
                    # unlikely at 48-bit prefix but detect anyway
                    raise RuntimeError(
                        f"alias sha1 collision: {sn} == alias for "
                        f"{sam3cpp_alias!r}"
                    )
                if alias_sn in seen_shortnames:
                    raise RuntimeError(
                        f"alias short_name conflict: {alias_sn} already "
                        f"seen for {seen_shortnames[alias_sn]!r}"
                    )
                seen_shortnames[alias_sn] = sam3cpp_alias
                tensor_map[alias_sn] = sam3cpp_alias
                # Write the same tensor data under the alias short_name.
                # sam3cpp's runtime finds it via find_tensor(sam3cpp_alias)
                # → resolve_tensor_name → alias_sn → tensor_index_.
                if qtype == GGMLQuantizationType.F16:
                    writer.add_tensor(alias_sn, stored)
                else:
                    writer.add_tensor(alias_sn, stored, raw_dtype=qtype)
                dtype_counts[qtype.name + "_alias"] += 1
                total_out_bytes += stored.nbytes

    writer.add_uint64("sam3.tensor_count", len(tensor_map))
    print(f"  writing {len(tensor_map)} tensors to disk...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Sidecar map — canonical_name → short_name (and also short→canonical
    # per sam3cpp's convention of populating both directions)
    sidecar = args.output.with_suffix(args.output.suffix + ".tensor_map.json")
    sidecar.write_text(json.dumps(tensor_map, indent=2, sort_keys=True))
    print(f"  writing sidecar: {sidecar}")

    print(f"\n=== emission complete ===")
    print(f"  gguf:    {args.output}  ({total_out_bytes / 1024 / 1024:.1f} MB)")
    print(f"  sidecar: {sidecar}  ({sidecar.stat().st_size} bytes)")
    print(f"  dtypes:  {dict(dtype_counts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
