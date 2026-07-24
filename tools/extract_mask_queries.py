#!/usr/bin/env python3
"""Extract the mask_queries [10, 2048] tensor from the InstructSAM
checkpoint and write it as a BIN1 sidecar (`instructsam_mask_queries.f32`)
that the C++ CLI can mmap.

Usage:
    python3 tools/extract_mask_queries.py \\
        --input  /path/to/model.safetensors \\
        --output /path/to/instructsam_mask_queries.f32

mask_queries is a learned nn.Parameter that InstructSAM injects into
the LM at each phrase's `<|object_ref_end|>` position (see
sam3_instructsam_cli.cpp:721 for the injection recipe). It's needed
alongside the LM/mmproj/grounding GGUFs and doesn't fit any of them
naturally, so it ships as a small .f32 sidecar.
"""
import argparse
import struct
from pathlib import Path

import safetensors.torch as st

CANONICAL_KEY = "model.mask_queries"
EXPECTED_SHAPE = (10, 2048)


def write_bin1(path: Path, arr) -> None:
    """Match the CLI's read_bin_f32 magic: 'BIN1' + i32 ndim + i64*ndim shape + f32 data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = arr.detach().to("cpu").float().contiguous().numpy()
    with path.open("wb") as f:
        f.write(b"BIN1")
        f.write(struct.pack("<i", data.ndim))
        for d in data.shape:
            f.write(struct.pack("<q", d))
        f.write(data.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input",  required=True, type=Path,
                    help="model.safetensors from CircleRadon/InstructSAM-2B")
    ap.add_argument("--output", required=True, type=Path,
                    help="output BIN1 blob path")
    args = ap.parse_args()

    sd = st.load_file(str(args.input))
    if CANONICAL_KEY not in sd:
        # Some checkpoints use a slightly different prefix; try to find it.
        candidates = [k for k in sd if k.endswith("mask_queries")]
        if len(candidates) != 1:
            raise SystemExit(
                f"cannot locate mask_queries in {args.input} "
                f"(searched for '{CANONICAL_KEY}', got candidates: {candidates})")
        key = candidates[0]
        print(f"note: '{CANONICAL_KEY}' not found; using '{key}' instead")
    else:
        key = CANONICAL_KEY

    tensor = sd[key]
    if tuple(tensor.shape) != EXPECTED_SHAPE:
        raise SystemExit(
            f"unexpected shape {tuple(tensor.shape)} for {key}; "
            f"expected {EXPECTED_SHAPE}")

    write_bin1(args.output, tensor)
    print(f"wrote {args.output}  shape={tuple(tensor.shape)}  "
          f"({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
