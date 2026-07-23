// InstructSAM pixel decoder (FPN) parity test — standalone + chained.
//
// Standalone: given bb0 [256,288,288] + bb1 [256,144,144] + bb2 [256,72,72]
// (bb2 is the encoder-hidden-states-replaced coarsest, pre-computed by the
// dump script), run 2-stage FPN and compare pixel_embed against reference
// mask_decoder__pixel_decoder (md_pixel_embed.f32).
//
// Chained: FPN → mask_tail (embedder + projections + einsum) and compare
// final pred_masks against md_pred_masks.

#include "sam3/gguf_model.h"
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

void report_parity(const std::string & label,
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
    om /= ours.size(); rm /= ref.size();
    std::cout << "  " << label << ":\n"
              << "    cosine (flat)     : " << cosine_flat(ours, ref) << "\n"
              << "    max abs diff      : " << maxdiff << "\n"
              << "    relative L2 error : " << (std::sqrt(l2diff) / std::sqrt(l2ref)) << "\n"
              << "    ours mean=" << om << "  ref mean=" << rm << "\n";
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: sam3-instructsam-pixel-decoder-test <model.gguf> [parity_dir]\n";
        return 1;
    }
    const std::string dir = (argc >= 3) ? argv[2]
        : "/tmp/pathA_reference/warehouse_rgb/binaries_obj0";

    sam3::GgufModel model;
    if (!model.load(argv[1])) { std::cerr << "load failed\n"; return 2; }
    std::cout << "loaded " << model.tensors().size() << " tensors\n";

    std::vector<int64_t> bb0s, bb1s, bb2s, pes, dqs, pms;
    const auto bb0 = read_bin_f32(dir + "/md_fpn_bb0.f32", bb0s);
    const auto bb1 = read_bin_f32(dir + "/md_fpn_bb1.f32", bb1s);
    const auto bb2 = read_bin_f32(dir + "/md_fpn_bb2.f32", bb2s);
    const auto ref_pix = read_bin_f32(dir + "/md_pixel_embed.f32", pes);
    const auto dec_q   = read_bin_f32(dir + "/md_decoder_queries.f32", dqs);
    const auto ref_pm  = read_bin_f32(dir + "/md_pred_masks.f32", pms);
    std::cout << "  bb0 " << bb0s[0] << "×" << bb0s[1] << "×" << bb0s[2]
              << ", bb1 " << bb1s[0] << "×" << bb1s[1] << "×" << bb1s[2]
              << ", bb2 " << bb2s[0] << "×" << bb2s[1] << "×" << bb2s[2] << "\n";

    // ── Standalone: FPN parity vs md_pixel_embed ─────────────────────────
    std::cout << "\n=== pixel_decoder FPN standalone ===\n";
    sam3::InstructsamPixelDecoder fpn(model);
    const auto pixel_embed = fpn.run(bb0, bb0s, bb1, bb1s, bb2, bb2s);
    if (pixel_embed.size() != ref_pix.size()) {
        std::cerr << "  ✗ size mismatch: ours=" << pixel_embed.size()
                  << " ref=" << ref_pix.size() << "\n";
        return 3;
    }
    report_parity("pixel_embed", pixel_embed, ref_pix);

    // ── Chained: FPN → mask_tail parity vs md_pred_masks ─────────────────
    std::cout << "\n=== full seg-head chain: FPN → mask_tail ===\n";
    sam3::InstructsamMaskTail tail(model);
    const auto out = tail.run(pixel_embed, {256, 288, 288}, dec_q, dqs);
    if (out.pred_masks.size() != ref_pm.size()) {
        std::cerr << "  ✗ pred_masks size mismatch\n";
        return 4;
    }
    report_parity("pred_masks", out.pred_masks, ref_pm);

    std::cout << "\n=== FPN + mask_tail: E2E seg-head (post-PCA encoder given) ===\n";
    std::cout << "  cos > 0.99 => end-to-end mask generation numerically correct\n";
    std::cout << "  next: wire prompt_cross_attn to eliminate the last Python-side dep\n";
    return 0;
}
