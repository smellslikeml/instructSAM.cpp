"""Overlay sam3-instructsam-cli's pred_masks on an image.

Reads e2e_pred_masks_cli.f32 (or e2e_pred_masks_ggml.f32) [N, 10, 288, 288]
and picks the highest-confidence mask slot per object.

Two modes:
- Reference mode (warehouse_rgb, backward compat): if `cls_score.pt` is
  found in the ref dir, use it to pick per-object slots and render an
  additional PyTorch-reference PNG alongside.
- Standalone mode (any image): pick each object's best slot by max
  mask activation, skip the PyTorch comparison.

Usage:
    # Standalone on your own image + your own CLI output:
    python3 visualize_e2e_masks.py \\
        --image  /path/to/your.jpg \\
        --masks  /path/to/e2e_pred_masks_cli.f32 \\
        --phrases "car,person,tree" \\
        --out    /tmp/my_run/overlay.png

    # Warehouse backward-compat (no args):
    python3 visualize_e2e_masks.py
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WAREHOUSE_REF_DIR = Path("/tmp/pathA_reference/warehouse_rgb")
WAREHOUSE_IMAGE   = Path("/home/thorax/Downloads/warehouse_rgb.jpg")
WAREHOUSE_MASKS   = WAREHOUSE_REF_DIR / "e2e_pred_masks_ggml.f32"
WAREHOUSE_LABELS  = ["box", "person", "shelf", "forklift"]


def read_bin(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"BIN1", f"bad magic in {path}"
        ndim = struct.unpack("<i", f.read(4))[0]
        shape = [struct.unpack("<q", f.read(8))[0] for _ in range(ndim)]
        n = 1
        for d in shape:
            n *= d
        data = np.frombuffer(f.read(n * 4), dtype=np.float32)
        return data.reshape(shape)


PALETTE = [
    (255,  80,  80), ( 80, 200,  80), ( 80, 140, 255), (255, 200,  60),
    (200,  80, 200), ( 80, 220, 220), (255, 140,  80), (140,  80, 220),
    (120, 220,  80), (220, 120, 120), ( 80, 100, 200), (200, 200,  80),
]


def overlay_masks(rgb_np, masks_hw_per_object, labels, alpha=0.45):
    """rgb_np [H,W,3] uint8; masks_hw_per_object list of [H,W] float 0..1."""
    out = rgb_np.astype(np.float32).copy()
    for i, mask in enumerate(masks_hw_per_object):
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        m = mask[..., None]
        out = (1.0 - alpha * m) * out + (alpha * m) * color
    out = np.clip(out, 0, 255).astype(np.uint8)
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    for i, label in enumerate(labels):
        color = PALETTE[i % len(PALETTE)]
        y = 15 + 22 * i
        draw.rectangle([10, y - 4, 30, y + 12], fill=color)
        draw.text((36, y - 2), label, fill=(255, 255, 255))
    return img


def render(masks_all, image_path, labels, out_path, title, cls_score=None):
    """masks_all [N, 10, 288, 288] logits, labels of length N, image at image_path.

    If cls_score [N, 10] is provided, pick per-object slots by max cls_score.
    Otherwise pick by max sigmoid activation (peak spatial confidence).
    """
    orig = Image.open(image_path).convert("RGB")
    W, H = orig.size
    N = masks_all.shape[0]
    if len(labels) < N:
        labels = list(labels) + [f"obj{i}" for i in range(len(labels), N)]

    picks = []
    picked_masks = []
    for obj in range(N):
        if cls_score is not None:
            top = int(np.argmax(cls_score[obj]))
            conf = float(cls_score[obj, top])
        else:
            # Fallback: pick slot with highest peak mask activation.
            probs = 1.0 / (1.0 + np.exp(-masks_all[obj].astype(np.float64)))
            slot_peaks = probs.max(axis=(1, 2))
            top = int(np.argmax(slot_peaks))
            conf = float(slot_peaks[top])
        picks.append((obj, top, conf))
        m_logits = masks_all[obj, top]
        m_prob   = 1.0 / (1.0 + np.exp(-m_logits.astype(np.float64)))
        m_pil    = Image.fromarray((m_prob * 255).astype(np.uint8), mode="L").resize(
            (W, H), Image.BILINEAR)
        m_np     = np.asarray(m_pil).astype(np.float32) / 255.0
        picked_masks.append((m_np > 0.5).astype(np.float32))

    label_texts = [f"{labels[o]} (slot {t}, conf {c:.2f})" for o, t, c in picks]
    img = overlay_masks(np.asarray(orig), picked_masks, label_texts)

    banner = Image.new("RGB", (img.width, 24), (30, 30, 30))
    dj = ImageDraw.Draw(banner)
    dj.text((10, 4), title, fill=(255, 255, 255))
    combined = Image.new("RGB", (img.width, img.height + 24))
    combined.paste(banner, (0, 0))
    combined.paste(img, (0, 24))
    combined.save(out_path)
    print(f"  wrote {out_path} ({img.width}×{img.height})")


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image",   type=Path, default=None,
                    help="input image (default: warehouse_rgb.jpg)")
    ap.add_argument("--masks",   type=Path, default=None,
                    help="pred_masks.f32 shape [N, 10, 288, 288] "
                         "(default: warehouse ref)")
    ap.add_argument("--phrases", type=str,  default=None,
                    help="comma-separated labels, one per object")
    ap.add_argument("--out",     type=Path, default=None,
                    help="output PNG path (default: next to --masks)")
    # Backward-compat positional (first non-flag arg = masks path):
    ap.add_argument("positional_masks", nargs="?", default=None)
    return ap.parse_args(argv)


def main(argv) -> int:
    args = parse_args(argv)

    masks_path = args.masks or (Path(args.positional_masks) if args.positional_masks
                                else WAREHOUSE_MASKS)
    image_path = args.image or WAREHOUSE_IMAGE
    print(f"loading pred_masks from {masks_path}")
    masks = read_bin(masks_path)
    N = masks.shape[0]
    print(f"  shape={list(masks.shape)}  mean={masks.mean():.3f}  absmax={np.abs(masks).max():.3f}")

    if args.phrases:
        labels = [p.strip() for p in args.phrases.split(",") if p.strip()]
    elif image_path == WAREHOUSE_IMAGE:
        labels = WAREHOUSE_LABELS[:N]
    else:
        labels = [f"obj{i}" for i in range(N)]

    # Look for reference cls_score.pt only if we're in warehouse mode.
    cls_score = None
    ref_masks = None
    if image_path == WAREHOUSE_IMAGE and WAREHOUSE_REF_DIR.exists():
        try:
            import torch
            cls_score = torch.load(WAREHOUSE_REF_DIR / "cls_score.pt",
                                    map_location="cpu", weights_only=True).float().numpy()
            ref_masks = torch.load(WAREHOUSE_REF_DIR / "mask_decoder__pred_masks.pt",
                                    map_location="cpu", weights_only=True).float().numpy()
            print(f"  loaded cls_score {list(cls_score.shape)} + ref masks")
        except Exception as e:
            print(f"  (warehouse refs not loadable: {e})")

    ours_out = args.out or masks_path.with_suffix(".png")
    render(masks, image_path, labels, ours_out, "sam3-instructsam-cli output",
           cls_score=cls_score)
    if ref_masks is not None:
        ref_out = ours_out.with_name(ours_out.stem + ".pytorch_ref.png")
        render(ref_masks, image_path, labels, ref_out, "PyTorch reference",
               cls_score=cls_score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
