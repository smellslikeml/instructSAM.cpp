#!/usr/bin/env python3
"""Path A reference-tensor capture harness.

Runs InstructSAM's PyTorch inference against a test image with forward
hooks registered on every mask-generation-relevant submodule of
`grounding_model`, plus the custom InstructSAM glue tensors (mask_queries,
mask_hidden_fcs output, text_hidden_fcs output, seg_output_embeddings).

Each captured tensor is written to disk with a SHA-256 fingerprint and
recorded in a JSON manifest. These become the ground-truth oracle for
validating a future ggml-native InstructSAM runtime (sam3cpp fork per
Path A) at every layer boundary.

Layer-by-layer numerical parity against this reference is how we know a
port is correct — instead of only observing end-to-end mask output
differences and backtracking through weeks of code.

Usage:
    /tmp/InstructSAM/.venv/bin/python3 pathA_reference_capture.py \\
        --image /path/to/image.jpg \\
        --query "Please segment the box, the person, ... in the image." \\
        --out-dir /path/to/reference/  # gets image-name subdirectory

Output layout:
    <out_dir>/<image-name>/
        manifest.json              # tensor metadata + SHA256s + shapes
        vision_encoder_backbone.pt
        vision_encoder_neck.pt
        detr_encoder.pt
        detr_decoder_layer_0.pt   # ... through layer_5
        mask_decoder.pt
        mask_queries.pt           # the learnable [10, 2048] tensor
        mask_hidden_fcs_out.pt    # (n_seg_positions, 256) projected queries
        text_hidden_fcs_out.pt    # (n_phrases, 256) projected text
        seg_output_embeddings.pt  # (n_seg_positions, 2048) raw LM hs at mask_queries positions
        pred_masks.pt             # final (n_objects, n_slots, H, W) mask logits
        cls_score.pt              # final (n_objects, n_slots) confidence scores
        text_output.txt           # what the LM generated
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/tmp/InstructSAM")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_bytes(t: torch.Tensor) -> str:
    """Content fingerprint — cast to fp16 first to normalize dtype drift."""
    arr = t.detach().to(torch.float16).cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()[:16]


def save_tensor(t: torch.Tensor | None, path: Path, name: str,
                manifest: dict) -> None:
    if t is None:
        manifest[name] = {"skipped": "None"}
        return
    t = t.detach().cpu()
    torch.save(t, path)
    manifest[name] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sha256_fp16": sha256_bytes(t),
        "path": path.name,
    }


class HookRegistry:
    """Register forward hooks on named modules; store the FIRST output
    tensor at each site (later invocations overwritten only after a reset,
    which we don't do — first-call captures the initial-shape forward)."""

    def __init__(self) -> None:
        self.captures: dict[str, torch.Tensor] = {}
        self._handles = []

    def hook(self, module: torch.nn.Module, name: str) -> None:
        def _fn(_mod, _inp, out):
            # Skip if already captured (e.g. per-layer decoders called
            # once per generation step; the first-step activations are
            # what we want as the reference).
            if name in self.captures:
                return
            if isinstance(out, torch.Tensor):
                self.captures[name] = out
            elif isinstance(out, (tuple, list)) and len(out) and isinstance(out[0], torch.Tensor):
                self.captures[name] = out[0]
            elif isinstance(out, dict):
                # e.g. mask_decoder returns {'pred_masks': ..., 'pred_logits': ...}
                for k, v in out.items():
                    if isinstance(v, torch.Tensor):
                        self.captures[f"{name}::{k}"] = v
        self._handles.append(module.register_forward_hook(_fn))

    def remove_all(self) -> None:
        for h in self._handles:
            h.remove()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path,
                    default=Path("/home/thorax/Downloads/warehouse_rgb.jpg"))
    ap.add_argument("--query", type=str,
                    default="Please segment the box, the person, the shelf, "
                            "and the forklift in the image.")
    ap.add_argument("--model", type=str, default="CircleRadon/InstructSAM-2B")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/tmp/pathA_reference"))
    args = ap.parse_args()

    if not args.image.exists():
        log(f"✗ image not found: {args.image}")
        return 1

    out_dir = args.out_dir / args.image.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "image": str(args.image),
        "query": args.query,
        "model": args.model,
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tensors": {},
    }

    log(f"loading {args.model} on CPU (device_map='cpu' to avoid meta-tensor offload)")
    from instructsam.models import load_pretrained_model  # noqa: E402
    from instructsam import mm_infer_segmentation  # noqa: E402
    t0 = time.time()
    tokenizer, model, processor = load_pretrained_model(
        args.model, None,
        attn_implementation="sdpa",
        device_map={"": "cpu"},
    )
    model.eval()
    log(f"✓ loaded in {time.time()-t0:.1f}s")

    # ── Register hooks on grounding_model submodules ────────────────────
    hooks = HookRegistry()
    gm = model.model.grounding_model.model

    # Vision path
    hooks.hook(gm.vision_encoder.backbone, "vision_encoder_backbone")
    hooks.hook(gm.vision_encoder.neck, "vision_encoder_neck")
    hooks.hook(gm.vision_encoder, "vision_encoder")

    # DETR encoder (fusion stage)
    hooks.hook(gm.detr_encoder, "detr_encoder")
    for i, layer in enumerate(gm.detr_encoder.layers):
        hooks.hook(layer, f"detr_encoder_layer_{i}")

    # DETR decoder — per-layer + final
    hooks.hook(gm.detr_decoder, "detr_decoder")
    for i, layer in enumerate(gm.detr_decoder.layers):
        hooks.hook(layer, f"detr_decoder_layer_{i}")

    # Mask decoder — final output
    hooks.hook(gm.mask_decoder, "mask_decoder")

    # Geometry encoder + dot-product scoring
    hooks.hook(gm.geometry_encoder, "geometry_encoder")
    hooks.hook(gm.dot_product_scoring, "dot_product_scoring")

    # InstructSAM-specific glue
    hooks.hook(model.model.mask_hidden_fcs[0], "mask_hidden_fcs_0")
    hooks.hook(model.model.text_hidden_fcs[0], "text_hidden_fcs_0")

    log(f"✓ registered {len(hooks._handles)} forward hooks")

    # ── Run inference ────────────────────────────────────────────────────
    log(f"running mm_infer_segmentation on {args.image.name} (CPU, expect several min)")
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(args.image)},
            {"type": "text", "text": args.query},
        ],
    }]
    t0 = time.time()
    with torch.no_grad():
        text_output, masks, cls_score = mm_infer_segmentation(
            str(args.image), processor, conversation, model, tokenizer,
        )
    log(f"✓ inference in {time.time()-t0:.1f}s")

    # ── Save results ─────────────────────────────────────────────────────
    (out_dir / "text_output.txt").write_text(str(text_output))

    # Custom InstructSAM tensors — captured from model state, not hooks
    save_tensor(model.model.mask_queries, out_dir / "mask_queries.pt",
                "mask_queries", manifest["tensors"])

    # seg_output_embeddings (list of tensors captured during generation)
    seg_embeds = getattr(model, "seg_output_embeddings", None)
    if seg_embeds is not None and len(seg_embeds) > 0:
        cat = torch.cat(seg_embeds, dim=0)
        save_tensor(cat, out_dir / "seg_output_embeddings.pt",
                    "seg_output_embeddings", manifest["tensors"])
    else:
        manifest["tensors"]["seg_output_embeddings"] = {"skipped": "empty"}
        log("⚠ seg_output_embeddings empty — no [SEG] positions detected")

    # Final outputs
    save_tensor(masks, out_dir / "pred_masks.pt", "pred_masks",
                manifest["tensors"])
    save_tensor(cls_score, out_dir / "cls_score.pt", "cls_score",
                manifest["tensors"])

    # Hook captures
    for name, tensor in hooks.captures.items():
        safe = name.replace("::", "__").replace("/", "_")
        save_tensor(tensor, out_dir / f"{safe}.pt", name,
                    manifest["tensors"])

    hooks.remove_all()

    # ── Summary ──────────────────────────────────────────────────────────
    n_tensors = sum(1 for v in manifest["tensors"].values() if "shape" in v)
    total_bytes = 0
    for f in out_dir.glob("*.pt"):
        total_bytes += f.stat().st_size
    manifest["totals"] = {
        "tensors_captured": n_tensors,
        "bytes_on_disk": total_bytes,
        "text_output_chars": len(str(text_output)),
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    log(f"✓ captured {n_tensors} tensors, {total_bytes/1024/1024:.1f} MB")
    log(f"  output: {out_dir}")

    # Print a summary of the most important shapes
    print(f"\n=== key reference tensors ===")
    important = ["vision_encoder_backbone", "vision_encoder_neck",
                 "detr_encoder", "detr_decoder", "mask_decoder::pred_masks",
                 "mask_hidden_fcs_0", "text_hidden_fcs_0",
                 "seg_output_embeddings", "mask_queries",
                 "pred_masks", "cls_score"]
    for k in important:
        v = manifest["tensors"].get(k)
        if v and "shape" in v:
            print(f"  {k:35s}  shape={v['shape']}  sha256={v['sha256_fp16']}")
        elif v:
            print(f"  {k:35s}  {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
