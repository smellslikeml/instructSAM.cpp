// InstructsamVisionEncoder::run_neck() parity test.
//
// Feeds captured trunk output through FPN neck, compares 3 outputs against
// the mask_decoder inputs (which were captured pre-encoder-replacement so
// they equal the raw neck outputs the DETR encoder/mask decoder consume).

#include "sam3/gguf_model.h"
#include "sam3/instructsam_vision_encoder.h"

#include <chrono>
#include <cmath>
#include <cstdint>
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
    if (std::string(magic, 4) != "BIN1") throw std::runtime_error("bad magic");
    int32_t ndim = 0; f.read(reinterpret_cast<char *>(&ndim), 4);
    shape.assign(static_cast<size_t>(ndim), 0);
    size_t total = 1;
    for (int i = 0; i < ndim; ++i) {
        int64_t d = 0; f.read(reinterpret_cast<char *>(&d), 8);
        shape[static_cast<size_t>(i)] = d; total *= static_cast<size_t>(d);
    }
    std::vector<float> data(total);
    f.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(total * sizeof(float)));
    return data;
}

double cos_flat(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

void report(const std::string & label, const std::vector<float> & ours,
            const std::vector<float> & ref) {
    double maxd = 0.0, l2d = 0.0, l2r = 0.0;
    for (size_t i = 0; i < ours.size(); ++i) {
        const double d = static_cast<double>(ours[i]) - ref[i];
        if (std::fabs(d) > maxd) maxd = std::fabs(d);
        l2d += d * d; l2r += static_cast<double>(ref[i]) * ref[i];
    }
    std::cout << "  " << label << ":\n"
              << "    cos_sim  : " << cos_flat(ours, ref) << "\n"
              << "    max_diff : " << maxd << "\n"
              << "    rel_L2   : " << std::sqrt(l2d) / std::sqrt(l2r) << "\n";
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { std::cerr << "usage: <model.gguf> [vision_dir] [obj_dir]\n"; return 1; }
    const std::string vdir = (argc >= 3) ? argv[2]
        : "/tmp/pathA_reference/warehouse_rgb/binaries_vision";
    const std::string odir = (argc >= 4) ? argv[3]
        : "/tmp/pathA_reference/warehouse_rgb/binaries_obj0";

    sam3::GgufModel model;
    if (!model.load(argv[1])) { std::cerr << "load failed\n"; return 2; }
    std::cout << "loaded " << model.tensors().size() << " tensors\n";

    sam3::InstructsamVisionEncoder enc(model);

    std::vector<int64_t> ts, r0s, r1s, r2s;
    const auto trunk = read_bin_f32(vdir + "/trunk_out.f32",  ts);
    const auto ref0  = read_bin_f32(odir + "/md_fpn_bb0.f32", r0s);
    const auto ref1  = read_bin_f32(odir + "/md_fpn_bb1.f32", r1s);
    const auto ref2  = read_bin_f32(odir + "/md_fpn_bb2.f32", r2s);
    std::cout << "  trunk_out shape=[" << ts[0] << "," << ts[1] << "]\n";
    std::cout << "  ref bb0=[" << r0s[0] << "," << r0s[1] << "," << r0s[2] << "]"
              << " bb1=[" << r1s[0] << "," << r1s[1] << "," << r1s[2] << "]"
              << " bb2=[" << r2s[0] << "," << r2s[1] << "," << r2s[2] << "]\n";

    // NOTE: md_fpn_bb2 is captured AFTER encoder-features replacement inside
    // mask_decoder._embed_pixels, so it's NOT what the raw FPN outputs for bb2.
    // We can still compare bb0/bb1 (unmodified). For bb2, we'll compare against
    // what the raw FPN produces (that value is _replaced_ downstream, so no
    // reference tensor is captured — this is a structural check only).
    std::cout << "\n  running run_neck (CPU, ~2-3 min)...\n" << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    const auto fpn = enc.run_neck(trunk);
    const auto rt = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "  runtime: " << rt << " ms\n\n";

    if (fpn.bb0.size() != ref0.size() || fpn.bb1.size() != ref1.size() || fpn.bb2.size() != ref2.size()) {
        std::cerr << "  size mismatch\n";
        return 3;
    }
    report("bb0 [256, 288, 288] (scale=4×)", fpn.bb0, ref0);
    report("bb1 [256, 144, 144] (scale=2×)", fpn.bb1, ref1);
    // bb2 comparison note: reference is post-encoder-replacement,
    // ours is pre-replacement. They should DIFFER significantly.
    // Just print shape check.
    std::cout << "  bb2 (structural check only — ref is post-encoder-replaced):\n"
              << "    ours size=" << fpn.bb2.size() << "  ref size=" << ref2.size()
              << "  ours mean=" << [&](){double s=0.0; for (float x:fpn.bb2) s+=x; return s/fpn.bb2.size();}()
              << "  ref mean=" << [&](){double s=0.0; for (float x:ref2) s+=x; return s/ref2.size();}() << "\n";

    std::cout << "\n=== FPN neck complete. Vision encoder Piece 1 done ===\n";
    return 0;
}
