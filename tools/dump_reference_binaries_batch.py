"""Dump ALL 4 objects' inputs for the full E2E chain.

Writes to /tmp/pathA_reference/warehouse_rgb/binaries_obj{0,1,2,3}/, with
each subdirectory containing the same per-object inputs the single-object
dump produces. Shared tensors (initial_reference_points, query_pos) are
duplicated across dirs so the C++ E2E code path can just parameterize
its loading dir.
"""
import os
import struct
import sys
import torch
from safetensors import safe_open

REF_DIR = "/tmp/pathA_reference/warehouse_rgb"
BASE_OUT = "/tmp/pathA_reference/warehouse_rgb"

# Per-object tensors: name maps to (input_file, output_name).
# Shape: batch tensor is [4, ...], we index [obj_idx].
PER_OBJECT = [
    ("mask_hidden_fcs_0.pt",                        "queries"),
    ("detr_encoder__text_features.pt",              "enc_text_features"),
    ("detr_encoder__pos_embeds_flattened.pt",       "enc_vision_pos_flat"),
    ("mask_decoder_in__backbone_features_0.pt",     "md_fpn_bb0"),
    ("mask_decoder_in__backbone_features_1.pt",     "md_fpn_bb1"),
    ("mask_decoder_in__backbone_features_2.pt",     "_bb2_precomputed_flatten_transpose"),
    ("mask_decoder_in__encoder_hidden_states.pt",   "md_pca_encoder_in"),
    ("mask_decoder_in__prompt_features.pt",         "md_pca_prompt_features"),
    ("mask_decoder_in__prompt_mask.pt",             "md_pca_prompt_mask"),
    ("mask_decoder__pred_masks.pt",                 "md_pred_masks"),
]


def dump_fp32(path: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().float().contiguous().cpu()
    with open(path, "wb") as f:
        f.write(b"BIN1")
        f.write(struct.pack("<i", tensor.ndim))
        for d in tensor.shape:
            f.write(struct.pack("<q", d))
        f.write(tensor.numpy().tobytes())


def main() -> int:
    # Load shared weights once (initial_reference_points + query_pos)
    ckpt = "/home/thorax/.cache/huggingface/hub/models--CircleRadon--InstructSAM-2B/snapshots/238da6833641dbc63264db46a03899be3e7c9465/model.safetensors"
    with safe_open(ckpt, framework="pt") as f:
        base = "model.grounding_model.model.detr_decoder"
        rp_w   = f.get_tensor(f"{base}.reference_points.weight").float()
        rph_1w = f.get_tensor(f"{base}.ref_point_head.layer1.weight").float()
        rph_1b = f.get_tensor(f"{base}.ref_point_head.layer1.bias").float()
        rph_2w = f.get_tensor(f"{base}.ref_point_head.layer2.weight").float()
        rph_2b = f.get_tensor(f"{base}.ref_point_head.layer2.bias").float()

    init_rp = torch.sigmoid(rp_w)  # [10, 4]
    HIDDEN = 256
    HALF   = HIDDEN // 2
    dim_t = torch.arange(HALF, dtype=torch.float32)
    dim_t = 10000.0 ** (2.0 * (dim_t // 2) / HALF)
    scale = 2.0 * torch.pi
    def sc(coord_idx):
        pos = init_rp[:, coord_idx] * scale
        div = pos.unsqueeze(-1) / dim_t
        return torch.stack((div[..., 0::2].sin(), div[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_x = sc(0); pos_y = sc(1); pos_w = sc(2); pos_h = sc(3)
    query_sine = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=-1)
    h = torch.nn.functional.linear(query_sine, rph_1w, rph_1b).clamp(min=0)
    qp = torch.nn.functional.linear(h, rph_2w, rph_2b)

    # Load per-object batched tensors once
    loaded = {}
    for filename, _ in PER_OBJECT:
        p = os.path.join(REF_DIR, filename)
        loaded[filename] = torch.load(p, map_location="cpu", weights_only=True)

    # PCA output (for synthesizing bb2 and post-encoder if needed)
    pca_out_all  = torch.load(os.path.join(REF_DIR, "mask_decoder__prompt_cross_attn.pt"), map_location="cpu", weights_only=True)

    for obj_idx in range(4):
        out_dir = os.path.join(BASE_OUT, f"binaries_obj{obj_idx}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== object {obj_idx} → {out_dir} ===")

        for filename, out_name in PER_OBJECT:
            t_all = loaded[filename]
            if out_name.startswith("_"):
                continue  # handled below (bb2 synthesis)
            t = t_all[obj_idx]
            out_path = os.path.join(out_dir, out_name + ".f32")
            dump_fp32(out_path, t)
            print(f"  {out_name:32s} {list(t.shape)}")

        # Shared: initial_reference_points, query_pos_layer_0
        dump_fp32(os.path.join(out_dir, "initial_reference_points.f32"), init_rp)
        dump_fp32(os.path.join(out_dir, "query_pos_layer_0.f32"), qp)

        # Synthesized: encoder vision_features_flat = bb2 flatten+transpose
        bb2_full = loaded["mask_decoder_in__backbone_features_2.pt"][obj_idx]  # [256, 72, 72]
        vision_flat = bb2_full.flatten(1).transpose(0, 1).contiguous().float()  # [5184, 256]
        dump_fp32(os.path.join(out_dir, "enc_vision_features_flat.f32"), vision_flat)
        print(f"  enc_vision_features_flat (synth) {list(vision_flat.shape)}")

    print(f"\ndone. per-object dumps under {BASE_OUT}/binaries_obj{{0..3}}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
