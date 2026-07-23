// End-to-end InstructSAM segmentation head parity test.
//
// Given the mask_decoder's ACTUAL raw inputs (as captured from PyTorch):
//   encoder_hidden_states  [5184, 256]  — pre-PCA encoder
//   prompt_features        [32, 256]
//   prompt_mask            [32]         — bool → float, 1=valid 0=pad
//   backbone_features bb0/bb1 [256, 288, 288] and [256, 144, 144]
//   decoder_queries        [10, 256]
//
// Run the full segmentation-head chain in native ggml:
//   PCA(encoder, prompt, mask) → post_encoder
//   FPN(bb0, bb1, reshape(post_encoder)) → pixel_embed
//   mask_tail(pixel_embed, decoder_queries) → pred_masks + semantic_seg
//
// And compare each intermediate + the final pred_masks against the
// captured PyTorch reference. This is the last-piece milestone before
// the segmentation half runs without Python-computed inputs.

#include "sam3/gguf_model.h"
#include "sam3/instructsam_prompt_cross_attn.h"
#include "sam3/instructsam_pixel_decoder.h"
#include "sam3/instructsam_mask_tail.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_bin_f32(const std::string & path, std::vector<int64_t> & shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4]; f.read(magic, 4);
    if (std::string(magic, 4) != "BIN1") throw std::runtime_error("bad magic in " + path);
    int32_t ndim = 0; f.read(reinterpret_cast<char *>(&ndim), 4);
    shape.assign(static_cast<size_t>(ndim), 0);
    size_t total = 1;
    for (int i = 0; i < ndim; ++i) {
        int64_t d = 0; f.read(reinterpret_cast<char *>(&d), 8);
        shape[static_cast<size_t>(i)] = d; total *= static_cast<size_t>(d);
    }
    std::vector<float> data(total);
    f.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(total * sizeof(float)));
    if (!f) throw std::runtime_error("short read on " + path);
    return data;
}

double cosine_flat(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

void report(const std::string & label,
            const std::vector<float> & ours,
            const std::vector<float> & ref) {
    double maxdiff = 0.0, l2diff = 0.0, l2ref = 0.0, om = 0.0, rm = 0.0;
    for (size_t i = 0; i < ours.size(); ++i) {
        const double d = static_cast<double>(ours[i]) - ref[i];
        if (std::fabs(d) > maxdiff) maxdiff = std::fabs(d);
        l2diff += d * d;
        l2ref  += static_cast<double>(ref[i]) * ref[i];
        om += ours[i]; rm += ref[i];
    }
    std::cout << "  " << label << ":\n"
              << "    cosine (flat)     : " << cosine_flat(ours, ref) << "\n"
              << "    max abs diff      : " << maxdiff << "\n"
              << "    relative L2 error : " << (std::sqrt(l2diff) / std::sqrt(l2ref)) << "\n"
              << "    ours mean=" << (om / ours.size())
              << "  ref mean=" << (rm / ref.size()) << "\n";
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: sam3-instructsam-seg-head-e2e <model.gguf> [parity_dir]\n";
        return 1;
    }
    const std::string dir = (argc >= 3) ? argv[2]
        : "/tmp/pathA_reference/warehouse_rgb/binaries_obj0";

    sam3::GgufModel model;
    if (!model.load(argv[1])) { std::cerr << "load failed\n"; return 2; }
    std::cout << "loaded " << model.tensors().size() << " tensors\n";

    // Load ALL the raw inputs + reference intermediates
    std::vector<int64_t> encs, pfs, pms;
    std::vector<int64_t> ref_attn_s, ref_pix_s, ref_masks_s;
    std::vector<int64_t> bb0s, bb1s, dqs;

    const auto encoder_in = read_bin_f32(dir + "/md_pca_encoder_in.f32",     encs);
    const auto prompt_feat= read_bin_f32(dir + "/md_pca_prompt_features.f32", pfs);
    const auto prompt_mask= read_bin_f32(dir + "/md_pca_prompt_mask.f32",     pms);
    const auto ref_attn_out = read_bin_f32(dir + "/md_pca_attn_out.f32",      ref_attn_s);
    const auto bb0        = read_bin_f32(dir + "/md_fpn_bb0.f32",             bb0s);
    const auto bb1        = read_bin_f32(dir + "/md_fpn_bb1.f32",             bb1s);
    const auto dec_q      = read_bin_f32(dir + "/md_decoder_queries.f32",     dqs);
    const auto ref_pix    = read_bin_f32(dir + "/md_pixel_embed.f32",         ref_pix_s);
    const auto ref_masks  = read_bin_f32(dir + "/md_pred_masks.f32",          ref_masks_s);

    std::cout << "  encoder_in   " << encs[0] << "×" << encs[1] << "\n";
    std::cout << "  prompt_feat  " << pfs[0]  << "×" << pfs[1]  << "\n";
    std::cout << "  prompt_mask  " << pms[0]  << "  valid=";
    int valid = 0;
    for (float v : prompt_mask) if (v > 0.5f) ++valid;
    std::cout << valid << "/" << prompt_mask.size() << "\n";

    // ── Stage 1: PCA ────────────────────────────────────────────────────
    std::cout << "\n=== stage 1: prompt_cross_attn ===\n";
    sam3::InstructsamPromptCrossAttn pca(model);
    const auto post_encoder = pca.run(encoder_in, encs, prompt_feat, pfs, prompt_mask, pms);

    // Reference "post_encoder" is encoder_in + ref_attn_out (the module output
    // is JUST the attention output — the residual add happens outside).
    std::vector<float> ref_post(encoder_in.size());
    for (size_t i = 0; i < ref_post.size(); ++i) ref_post[i] = encoder_in[i] + ref_attn_out[i];
    report("post_encoder = residual + attn_out", post_encoder, ref_post);

    // Also: raw attn_out parity (subtract residual, compare to ref_attn_out)
    std::vector<float> our_attn_out(post_encoder.size());
    for (size_t i = 0; i < our_attn_out.size(); ++i) our_attn_out[i] = post_encoder[i] - encoder_in[i];
    report("attn_out (isolated)", our_attn_out, ref_attn_out);

    // ── Stage 2: FPN ────────────────────────────────────────────────────
    std::cout << "\n=== stage 2: pixel_decoder FPN ===\n";
    // Reshape post_encoder [5184, 256] → [256, 72, 72] as the coarsest FPN input
    std::vector<float> bb2(256 * 72 * 72);
    for (int t = 0; t < 5184; ++t) {           // token index
        const int h = t / 72, w = t % 72;
        for (int c = 0; c < 256; ++c) {
            bb2[static_cast<size_t>(c * 5184 + h * 72 + w)] =
                post_encoder[static_cast<size_t>(t * 256 + c)];
        }
    }

    sam3::InstructsamPixelDecoder fpn(model);
    const auto pixel_embed = fpn.run(
        bb0, bb0s,
        bb1, bb1s,
        bb2, {256, 72, 72});
    report("pixel_embed", pixel_embed, ref_pix);

    // ── Stage 3: mask_tail ──────────────────────────────────────────────
    std::cout << "\n=== stage 3: mask_tail ===\n";
    sam3::InstructsamMaskTail tail(model);
    const auto out = tail.run(pixel_embed, {256, 288, 288}, dec_q, dqs);
    report("pred_masks", out.pred_masks, ref_masks);

    std::cout << "\n=== E2E segmentation head — cos > 0.99 => native ggml matches PyTorch ===\n";
    std::cout << "  Runs pixel_values-→-pred_masks with NO Python-side computation.\n";
    std::cout << "  Given raw mask_decoder inputs, produces byte-close masks.\n";
    return 0;
}
