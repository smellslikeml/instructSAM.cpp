"""Dump PyTorch reference oracle .pt tensors to raw fp32 binary files.

C++ InstructsamDecoder test harness reads these in via std::ifstream::read()
for numerical parity comparison. We dump object 0 of each 4-object batch
(single-object test); batch parity comes later.

Also dumps the initial reference_boxes (before layer 0) — the InstructSAM
decoder captures reference_boxes after each layer, so reference_boxes[0]
is AFTER layer 0. Initial (before layer 0) reference_boxes come from
sigmoid(transformer.decoder.reference_points.weight); we compute that
separately here for query_pos wiring in step 2e.
"""

import os
import struct
import sys

import torch

REF_DIR = "/tmp/pathA_reference/warehouse_rgb"
OUT_DIR = "/tmp/pathA_reference/warehouse_rgb/binaries_obj0"

# (filename, [batch selector])
DUMPS = [
    ("mask_hidden_fcs_0.pt",                     "queries"),
    ("detr_encoder__text_features.pt",           "text_memory"),
    ("detr_encoder__last_hidden_state.pt",       "vision_memory"),
    ("detr_encoder__pos_embeds_flattened.pt",    "vision_pos"),
    ("detr_decoder_layer_0.pt",                  "expected_layer_0"),
    ("detr_decoder_layer_1.pt",                  "expected_layer_1"),
    ("detr_decoder_layer_2.pt",                  "expected_layer_2"),
    ("detr_decoder_layer_3.pt",                  "expected_layer_3"),
    ("detr_decoder_layer_4.pt",                  "expected_layer_4"),
    ("detr_decoder_layer_5.pt",                  "expected_layer_5"),
    ("detr_decoder__reference_boxes.pt",         "reference_boxes_all"),
    # Mask decoder inputs + intermediates (for step 3+ mask-tail work)
    ("mask_decoder_in__decoder_queries.pt",      "md_decoder_queries"),
    ("mask_decoder__pixel_decoder.pt",           "md_pixel_embed"),
    ("mask_decoder__mask_embedder.pt",           "md_mask_embeddings"),
    ("mask_decoder__instance_projection.pt",     "md_instance_embeds"),
    ("mask_decoder__pred_masks.pt",              "md_pred_masks"),
    ("mask_decoder__semantic_projection.pt",     "md_semantic_seg"),
]


def dump_fp32(path: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().float().contiguous().cpu()
    with open(path, "wb") as f:
        # Header: 4 bytes magic + 4 bytes ndim + ndim*8 bytes shape (int64)
        f.write(b"BIN1")
        f.write(struct.pack("<i", tensor.ndim))
        for d in tensor.shape:
            f.write(struct.pack("<q", d))
        f.write(tensor.numpy().tobytes())


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    for filename, label in DUMPS:
        src = os.path.join(REF_DIR, filename)
        if not os.path.exists(src):
            print(f"MISSING: {src}", file=sys.stderr)
            return 1
        t = torch.load(src, map_location="cpu", weights_only=True)
        # For reference_boxes_all we keep all 6 layers × 4 objects × 10 × 4.
        # For everything else we take object 0.
        if label == "reference_boxes_all":
            out_t = t  # [6, 4, 10, 4]
        else:
            out_t = t[0]  # peel batch dim
        out_path = os.path.join(OUT_DIR, label + ".f32")
        dump_fp32(out_path, out_t)
        print(f"  {label:24s} {list(out_t.shape)}  -> {out_path}")

    # Extract reference_points + ref_point_head from raw safetensors +
    # compute layer-0 query_pos (what the InstructsamDecoder graph needs
    # to augment Q in attention blocks). Bypasses transformers loader
    # (which has version-drift issues) — reads weights directly.
    from safetensors import safe_open

    ckpt_path = "/home/thorax/.cache/huggingface/hub/models--CircleRadon--InstructSAM-2B/snapshots/238da6833641dbc63264db46a03899be3e7c9465/model.safetensors"
    if os.path.exists(ckpt_path):
        with safe_open(ckpt_path, framework="pt") as f:
            base = "model.grounding_model.model.detr_decoder"
            rp_w   = f.get_tensor(f"{base}.reference_points.weight").float()      # [10, 4]
            rph_1w = f.get_tensor(f"{base}.ref_point_head.layer1.weight").float() # [256, 512]
            rph_1b = f.get_tensor(f"{base}.ref_point_head.layer1.bias").float()   # [256]
            rph_2w = f.get_tensor(f"{base}.ref_point_head.layer2.weight").float() # [256, 256]
            rph_2b = f.get_tensor(f"{base}.ref_point_head.layer2.bias").float()   # [256]

        init_rp = torch.sigmoid(rp_w)  # [10, 4]
        dump_fp32(os.path.join(OUT_DIR, "initial_reference_points.f32"), init_rp)
        print(f"  initial_reference_points {list(init_rp.shape)}")

        # Sinusoidal encoding — matches sam3cpp's gen_sineembed_for_position:
        # for each of the 4 coords, produce 128 sin/cos features; reordering
        # applies to match the (y, x, w, h) concat convention used in the MLX
        # reference. But InstructSAM's actual code (models/sam3/modeling_sam3.py)
        # concatenates (x, y, w, h) with alternating sin/cos. Try the
        # transformers-standard convention first.
        HIDDEN = 256
        HALF   = HIDDEN // 2   # 128 features per coord
        dim_t = torch.arange(HALF, dtype=torch.float32)
        dim_t = 10000.0 ** (2.0 * (dim_t // 2) / HALF)      # [128]

        # reference_points: [10, 4] in (x, y, w, h) format per Sam3SinePositionEmbedding.encode_boxes
        # Concat order MUST be (pos_y, pos_x, pos_w, pos_h) — this is
        # InstructSAM's convention (verified against modeling_sam3.py:906).
        scale = 2.0 * torch.pi  # Sam3SinePositionEmbedding uses scale=2π (normalize=False path still uses this scale in encode_boxes)
        def sc(coord_idx):
            pos = init_rp[:, coord_idx] * scale         # [10]
            div = pos.unsqueeze(-1) / dim_t              # [10, 128]
            return torch.stack((div[..., 0::2].sin(), div[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_x = sc(0); pos_y = sc(1); pos_w = sc(2); pos_h = sc(3)
        query_sine = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)   # [10, 512]

        # ref_point_head MLP: linear1 → ReLU → linear2
        h  = torch.nn.functional.linear(query_sine, rph_1w, rph_1b).clamp(min=0)
        qp = torch.nn.functional.linear(h,          rph_2w, rph_2b)   # [10, 256]

        dump_fp32(os.path.join(OUT_DIR, "query_pos_layer_0.f32"), qp)
        print(f"  query_pos_layer_0        {list(qp.shape)} (from sinusoidal+ref_point_head)")
    else:
        print(f"  (skipping ref_point_head: {ckpt_path} not found)")

    print(f"\ndone. dumps at {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
