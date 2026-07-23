"""Visualize the ggml E2E chain's pred_masks on the warehouse_rgb.jpg image.

Reads /tmp/pathA_reference/warehouse_rgb/e2e_pred_masks_ggml.f32 (shape
[4, 10, 288, 288]) plus the reference pred_masks (same shape) + cls_score
[4, 10], and produces two side-by-side PNGs — one with our masks, one
with PyTorch's — overlaid on the original image.

Uses cls_score to pick the highest-confidence slot per object.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

REF_DIR = Path("/tmp/pathA_reference/warehouse_rgb")
IMAGE = Path("/home/thorax/Downloads/warehouse_rgb.jpg")
DEFAULT_OURS = REF_DIR / "e2e_pred_masks_ggml.f32"
OUT_OURS = Path("/tmp/pathA_reference/warehouse_rgb/e2e_masks_ggml.png")
OUT_REF  = Path("/tmp/pathA_reference/warehouse_rgb/e2e_masks_pytorch.png")


def read_bin(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"BIN1"
        ndim = struct.unpack("<i", f.read(4))[0]
        shape = [struct.unpack("<q", f.read(8))[0] for _ in range(ndim)]
        n = 1
        for d in shape:
            n *= d
        data = np.frombuffer(f.read(n * 4), dtype=np.float32)
        return data.reshape(shape)


PALETTE = [
    (255,  80,  80),   # red     — object 0
    ( 80, 200,  80),   # green   — object 1
    ( 80, 140, 255),   # blue    — object 2
    (255, 200,  60),   # yellow  — object 3
]


def overlay_masks(rgb_np, masks_hw_per_object, labels, alpha=0.45):
    """rgb_np: [H, W, 3] uint8.  masks_hw_per_object: list of [H, W] float 0..1."""
    out = rgb_np.astype(np.float32).copy()
    for i, mask in enumerate(masks_hw_per_object):
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        # mask has same H, W as image
        m = mask[..., None]  # [H, W, 1]
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


def render(masks_all, cls_score, out_path, title):
    # masks_all: [4, 10, 288, 288]  logits.
    # cls_score: [4, 10]  confidences.
    orig = Image.open(IMAGE).convert("RGB")
    W, H = orig.size

    # For each object, pick top-1 slot by cls_score, upsample sigmoid(mask)
    # to full image size, threshold at 0.5.
    picks = []
    labels_used = ["box", "person", "shelf", "forklift"]
    picked_masks = []
    for obj in range(4):
        top = int(np.argmax(cls_score[obj]))
        picks.append((obj, top, float(cls_score[obj, top])))
        m_logits = masks_all[obj, top]                            # [288, 288]
        m_prob   = 1.0 / (1.0 + np.exp(-m_logits.astype(np.float64)))  # sigmoid
        m_pil    = Image.fromarray((m_prob * 255).astype(np.uint8), mode="L").resize(
            (W, H), Image.BILINEAR)
        m_np     = np.asarray(m_pil).astype(np.float32) / 255.0
        # soft mask (threshold at 0.5 for hard binary, or blend the prob)
        m_bin    = (m_np > 0.5).astype(np.float32)
        picked_masks.append(m_bin)

    label_texts = [f"{labels_used[o]} (slot {t}, conf {c:.2f})" for o, t, c in picks]
    img = overlay_masks(np.asarray(orig), picked_masks, label_texts)

    # Add title
    banner = Image.new("RGB", (img.width, 24), (30, 30, 30))
    dj = ImageDraw.Draw(banner)
    dj.text((10, 4), title, fill=(255, 255, 255))
    combined = Image.new("RGB", (img.width, img.height + 24))
    combined.paste(banner, (0, 0))
    combined.paste(img, (0, 24))

    combined.save(out_path)
    print(f"  wrote {out_path} ({img.width}×{img.height})")


def main() -> int:
    ours_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OURS
    print(f"loading ggml pred_masks from {ours_path}")
    ours = read_bin(ours_path)                                     # [4, 10, 288, 288]

    print(f"loading PyTorch reference pred_masks + cls_score")
    ref  = torch.load(REF_DIR / "mask_decoder__pred_masks.pt",
                      map_location="cpu", weights_only=True).float().numpy()
    cls  = torch.load(REF_DIR / "cls_score.pt",
                      map_location="cpu", weights_only=True).float().numpy()  # [4, 10]

    print(f"ours  shape={list(ours.shape)}  mean={ours.mean():.3f}  absmax={np.abs(ours).max():.3f}")
    print(f"ref   shape={list(ref.shape)}   mean={ref.mean():.3f}  absmax={np.abs(ref).max():.3f}")
    print(f"cls   shape={list(cls.shape)}   max per object: {cls.max(axis=1).round(2).tolist()}")

    render(ours, cls, OUT_OURS, "InstructSAM segmentation (native ggml E2E chain)")
    render(ref,  cls, OUT_REF,  "InstructSAM segmentation (PyTorch reference)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
