// InstructsamVisionEncoder parity test — first milestone: skeleton +
// patch embed + pos embed + pre-trunk layer_norm.
//
// Feeds captured pixel_values through run_prenorm() and compares
// against the captured pre-layer-norm intermediate.

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

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) { std::cerr << "usage: <model.gguf> [parity_dir]\n"; return 1; }
    const std::string dir = (argc >= 3) ? argv[2]
        : "/tmp/pathA_reference/warehouse_rgb/binaries_vision";

    sam3::GgufModel model;
    if (!model.load(argv[1])) { std::cerr << "load failed\n"; return 2; }
    std::cout << "loaded " << model.tensors().size() << " tensors\n";

    // ── Milestone 1: tensor enumeration ─────────────────────────────────
    sam3::InstructsamVisionEncoder enc(model);
    std::cout << "\n=== milestone 1: tensor validation ===\n";
    const size_t probed = enc.validate_all_tensors_present();
    std::cout << "  ✓ " << probed << " vision encoder tensors present\n"
              << "    breakdown: 4 trunk-top + 32 × " << ((probed - 4) / 32)
              << " per-layer = " << probed << "\n";

    // ── Milestone 2a: patch embed alone parity ─────────────────────────
    std::cout << "\n=== milestone 2a: patch_embed alone parity ===\n";
    {
        std::vector<int64_t> pvs, pes;
        const auto pv  = read_bin_f32(dir + "/pixel_values.f32", pvs);
        const auto ref = read_bin_f32(dir + "/patch_embed.f32",  pes);
        std::cout << "  ref shape=[" << pes[0] << "," << pes[1] << "]\n";
        const auto ours = enc.run_patch_embed_only(pv, pvs);
        std::cout << "  ours size=" << ours.size() << " ref size=" << ref.size() << "\n";
        double maxd = 0.0, l2d = 0.0, l2r = 0.0;
        for (size_t i = 0; i < ours.size(); ++i) {
            const double d = static_cast<double>(ours[i]) - ref[i];
            if (std::fabs(d) > maxd) maxd = std::fabs(d);
            l2d += d * d; l2r += static_cast<double>(ref[i]) * ref[i];
        }
        std::cout << "    cos_sim  : " << cos_flat(ours, ref) << "\n"
                  << "    max_diff : " << maxd << "\n"
                  << "    rel_L2   : " << std::sqrt(l2d) / std::sqrt(l2r) << "\n"
                  << "    ours[0..4]  : " << ours[0] << " " << ours[1] << " " << ours[2] << " " << ours[3] << "\n"
                  << "    ref[0..4]   : " << ref[0]  << " " << ref[1]  << " " << ref[2]  << " " << ref[3]  << "\n"
                  << "    ours[1024]  : " << ours[1024] << "  (would be seq=1,c=0 if [C,seq] layout)\n"
                  << "    ref[1024]   : " << ref[1024]  << "  (is seq=1,c=0 for [seq,C] layout)\n"
                  << "    ours[1]     : " << ours[1]    << "  (would be c=1,seq=0 if [C,seq] layout)\n"
                  << "    check: does ours[k*1024] equal ref[k] for k=1..3?\n"
                  << "      ours[1024]=" << ours[1024] << "  ref[1]=" << ref[1] << "\n"
                  << "      ours[2048]=" << ours[2048] << "  ref[2]=" << ref[2] << "\n"
                  << "      ours[3072]=" << ours[3072] << "  ref[3]=" << ref[3] << "\n";
    }

    // ── Milestone 2: patch embed + pos embed + pre-trunk LN parity ─────
    std::cout << "\n=== milestone 2: pre-norm parity ===\n";
    std::vector<int64_t> pvs, refs;
    std::vector<float> pixel_values;
    std::vector<float> ref_prenorm;
    try {
        pixel_values = read_bin_f32(dir + "/pixel_values.f32", pvs);
        ref_prenorm  = read_bin_f32(dir + "/pre_layer_norm.f32", refs);
    } catch (const std::exception & e) {
        std::cerr << "  ✗ (need to run tools/dump_vision_binaries.py first): " << e.what() << "\n";
        return 3;
    }
    std::cout << "  pixel_values  shape=[";
    for (size_t i = 0; i < pvs.size(); ++i) std::cout << (i?",":"") << pvs[i];
    std::cout << "]\n  ref_prenorm   shape=[";
    for (size_t i = 0; i < refs.size(); ++i) std::cout << (i?",":"") << refs[i];
    std::cout << "]\n";

    const auto ours = enc.run_prenorm(pixel_values, pvs);
    std::cout << "  ours size = " << ours.size() << " (expected " << ref_prenorm.size() << ")\n";
    if (ours.size() != ref_prenorm.size()) {
        std::cerr << "  ✗ size mismatch\n";
        return 4;
    }

    double maxdiff = 0.0, l2diff = 0.0, l2ref = 0.0;
    for (size_t i = 0; i < ours.size(); ++i) {
        const double d = static_cast<double>(ours[i]) - ref_prenorm[i];
        if (std::fabs(d) > maxdiff) maxdiff = std::fabs(d);
        l2diff += d * d;
        l2ref  += static_cast<double>(ref_prenorm[i]) * ref_prenorm[i];
    }
    std::cout << "\n  pre-trunk-LN parity:\n"
              << "    cosine (flat)     : " << cos_flat(ours, ref_prenorm) << "\n"
              << "    max abs diff      : " << maxdiff << "\n"
              << "    relative L2 error : " << (std::sqrt(l2diff) / std::sqrt(l2ref)) << "\n";

    // ── Milestone 3: layer 0 forward parity ─────────────────────────────
    std::cout << "\n=== milestone 3: layer 0 forward parity ===\n";
    {
        std::vector<int64_t> l0s;
        std::vector<float> ref_l0;
        try {
            ref_l0 = read_bin_f32(dir + "/layer_00.f32", l0s);
        } catch (const std::exception & e) {
            std::cerr << "  ✗ layer_00.f32 missing: " << e.what() << "\n";
            return 5;
        }
        std::cout << "  ref layer_00 shape=[";
        for (size_t i = 0; i < l0s.size(); ++i) std::cout << (i?",":"") << l0s[i];
        std::cout << "]\n" << std::flush;

        std::cout << "  running layer 0 (CPU, ~30-60s for the 5.4B-flop MLP)...\n" << std::flush;
        const auto t0 = std::chrono::steady_clock::now();
        const auto layer0_out = enc.run_layer(0, ours);
        const auto t1 = std::chrono::steady_clock::now();
        std::cout << "  runtime: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";

        if (layer0_out.size() != ref_l0.size()) {
            std::cerr << "  ✗ size mismatch: ours=" << layer0_out.size() << " ref=" << ref_l0.size() << "\n";
            return 6;
        }
        double maxd = 0.0, l2d = 0.0, l2r = 0.0;
        for (size_t i = 0; i < layer0_out.size(); ++i) {
            const double d = static_cast<double>(layer0_out[i]) - ref_l0[i];
            if (std::fabs(d) > maxd) maxd = std::fabs(d);
            l2d += d * d; l2r += static_cast<double>(ref_l0[i]) * ref_l0[i];
        }
        std::cout << "\n  layer 0 parity:\n"
                  << "    cos_sim  : " << cos_flat(layer0_out, ref_l0) << "\n"
                  << "    max_diff : " << maxd << "\n"
                  << "    rel_L2   : " << std::sqrt(l2d) / std::sqrt(l2r) << "\n"
                  << "    ours[0..3]: " << layer0_out[0] << " " << layer0_out[1] << " " << layer0_out[2] << " " << layer0_out[3] << "\n"
                  << "    ref[0..3] : " << ref_l0[0]     << " " << ref_l0[1]     << " " << ref_l0[2]     << " " << ref_l0[3]     << "\n";
    }

    std::cout << "\n=== next milestone: layers 1-31 (mostly a loop) ===\n";
    return 0;
}
