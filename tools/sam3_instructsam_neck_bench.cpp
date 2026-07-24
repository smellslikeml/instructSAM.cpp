// Fast neck-only benchmark: loads trunk output from a cached f32 dump and
// runs run_neck, so we can iterate on FPN perf without waiting on the
// 4-minute trunk each rebuild.

#include "sam3/gguf_model.h"
#include "sam3/instructsam_vision_encoder.h"

#include <chrono>
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
    if (std::string(magic, 4) != "BIN1") throw std::runtime_error("bad magic");
    int32_t ndim = 0; f.read(reinterpret_cast<char *>(&ndim), 4);
    shape.assign(static_cast<size_t>(ndim), 0);
    size_t total = 1;
    for (int i = 0; i < ndim; ++i) {
        int64_t d = 0; f.read(reinterpret_cast<char *>(&d), 8);
        shape[static_cast<size_t>(i)] = d; total *= static_cast<size_t>(d);
    }
    std::vector<float> data(total);
    f.read(reinterpret_cast<char *>(data.data()), total * sizeof(float));
    return data;
}

double cos_sim(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string grounding = (argc >= 2) ? argv[1]
        : "/tmp/pathA_gguf/instructsam-grounding-f16.gguf";
    const std::string ref_dir = (argc >= 3) ? argv[2]
        : "/tmp/pathA_reference/warehouse_rgb";

    std::cout << "loading grounding gguf\n" << std::flush;
    sam3::GgufModel model;
    if (!model.load(grounding)) { std::cerr << "load failed\n"; return 2; }
    sam3::InstructsamVisionEncoder vis(model);
    std::cout << "  ✓ loaded\n\n";

    // Read PyTorch trunk output. It's stored as [5184, 1024] hidden-fastest.
    std::cout << "loading captured trunk output\n" << std::flush;
    std::vector<int64_t> ts;
    const auto trunk_pt = read_bin_f32(
        ref_dir + "/vision_backbone__last_hidden_state.f32", ts);
    if (trunk_pt.size() != static_cast<size_t>(5184) * 1024) {
        std::cerr << "unexpected trunk size " << trunk_pt.size() << "\n"; return 3;
    }
    std::cout << "  ✓ trunk [5184, 1024]\n\n";

    std::cout << "running neck\n" << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    const auto fpn = vis.run_neck(trunk_pt);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "  ✓ neck complete in " << ms << " ms\n\n";
    std::cout << "  bb0 size: " << fpn.bb0.size() << " (expect " << (256*288*288) << ")\n";
    std::cout << "  bb1 size: " << fpn.bb1.size() << " (expect " << (256*144*144) << ")\n";
    std::cout << "  bb2 size: " << fpn.bb2.size() << " (expect " << (256*72*72) << ")\n\n";

    // Parity vs cached FPN outputs
    std::vector<int64_t> s0, s1, sflat;
    const auto bb0_ref = read_bin_f32(ref_dir + "/binaries_obj0/md_fpn_bb0.f32", s0);
    const auto bb1_ref = read_bin_f32(ref_dir + "/binaries_obj0/md_fpn_bb1.f32", s1);
    const auto bb2_ref_hwc = read_bin_f32(ref_dir + "/binaries_obj0/enc_vision_features_flat.f32", sflat);
    std::vector<float> bb2_ref(256*72*72);
    for (int t = 0; t < 5184; ++t)
        for (int c = 0; c < 256; ++c) bb2_ref[c * 5184 + t] = bb2_ref_hwc[t * 256 + c];

    std::cout << "=== parity vs cached FPN outputs ===\n";
    std::cout << "  bb0 cos_sim = " << cos_sim(fpn.bb0, bb0_ref) << "\n";
    std::cout << "  bb1 cos_sim = " << cos_sim(fpn.bb1, bb1_ref) << "\n";
    std::cout << "  bb2 cos_sim = " << cos_sim(fpn.bb2, bb2_ref) << "\n";
    return 0;
}
