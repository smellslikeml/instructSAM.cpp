"""Dump SAM3 vision-encoder intermediates for per-layer parity testing.

Reads the reference oracle's per-layer ViT captures + pixel_values input
and writes them to /tmp/pathA_reference/warehouse_rgb/binaries_vision/.

Object-independent (SAM3 vision has batch_size=1 — same for all objects).
"""
import os
import struct
import sys
import torch

REF_DIR = "/tmp/pathA_reference/warehouse_rgb"
OUT_DIR = "/tmp/pathA_reference/warehouse_rgb/binaries_vision"


def dump_fp32(path, t):
    t = t.detach().float().contiguous().cpu()
    with open(path, "wb") as f:
        f.write(b"BIN1")
        f.write(struct.pack("<i", t.ndim))
        for d in t.shape: f.write(struct.pack("<q", d))
        f.write(t.numpy().tobytes())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # pixel_values input to backbone: [1, 3, 1008, 1008]
    pv_path = os.path.join(REF_DIR, "vision_backbone_in__pixel_values.pt")
    if not os.path.exists(pv_path):
        print(f"MISSING {pv_path} — re-run pathA_reference_capture.py with vision hooks", file=sys.stderr)
        return 1
    pv = torch.load(pv_path, map_location="cpu", weights_only=True).float()
    # Squeeze batch dim to match C++ [3, 1008, 1008]
    if pv.dim() == 4 and pv.shape[0] == 1:
        pv = pv[0]
    dump_fp32(os.path.join(OUT_DIR, "pixel_values.f32"), pv)
    print(f"  pixel_values          {list(pv.shape)}")

    # patch_embed output: [1, 5184, 1024]
    for pt_name, out_name in [
        ("vision_backbone__patch_embed.pt",     "patch_embed"),
        ("vision_backbone__embeddings.pt",      "embeddings"),
        ("vision_backbone__pre_layer_norm.pt",  "pre_layer_norm"),
    ]:
        p = os.path.join(REF_DIR, pt_name)
        if not os.path.exists(p):
            print(f"  (missing {pt_name})")
            continue
        t = torch.load(p, map_location="cpu", weights_only=True).float()
        # Squeeze batch if [1, ...]. Preserve rest.
        if t.dim() >= 3 and t.shape[0] == 1:
            t = t[0]
        dump_fp32(os.path.join(OUT_DIR, out_name + ".f32"), t)
        print(f"  {out_name:22s} {list(t.shape)}")

    # Per-layer ViT outputs
    for i in range(32):
        p = os.path.join(REF_DIR, f"vision_backbone__layer_{i:02d}.pt")
        if not os.path.exists(p):
            print(f"  (missing layer_{i:02d})")
            continue
        t = torch.load(p, map_location="cpu", weights_only=True).float()
        if t.dim() >= 3 and t.shape[0] == 1:
            t = t[0]
        dump_fp32(os.path.join(OUT_DIR, f"layer_{i:02d}.f32"), t)
        if i == 0 or i == 31:
            print(f"  layer_{i:02d}               {list(t.shape)}")

    print(f"\ndone. binaries at {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
