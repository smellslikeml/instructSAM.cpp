// Per-layer parity test for InstructsamDetrEncoder.
//
// Loads captured DETR encoder inputs (vision_features flattened,
// vision_pos, text_features, text_mask) and runs 6 encoder layers,
// comparing each layer's output against detr_encoder_layer_N.pt.

#include "sam3/gguf_model.h"
#include "sam3/instructsam_detr_encoder.h"

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

double cosine_per_token(const std::vector<float> & a, const std::vector<float> & b, int N, int D) {
    double sum = 0.0; int counted = 0;
    for (int q = 0; q < N; ++q) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int d = 0; d < D; ++d) {
            const float av = a[static_cast<size_t>(q * D + d)];
            const float bv = b[static_cast<size_t>(q * D + d)];
            dot += static_cast<double>(av) * bv;
            na  += static_cast<double>(av) * av;
            nb  += static_cast<double>(bv) * bv;
        }
        if (na > 0 && nb > 0) { sum += dot / (std::sqrt(na) * std::sqrt(nb)); ++counted; }
    }
    return counted > 0 ? sum / counted : 0.0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cerr << "usage: sam3-instructsam-detr-encoder-test <model.gguf> [parity_dir]\n";
        return 1;
    }
    const std::string dir = (argc >= 3) ? argv[2]
        : "/tmp/pathA_reference/warehouse_rgb/binaries_obj0";

    sam3::GgufModel model;
    if (!model.load(argv[1])) { std::cerr << "load failed\n"; return 2; }
    std::cout << "loaded " << model.tensors().size() << " tensors\n";

    std::vector<int64_t> vs, vps, ts, ms;
    const auto vision   = read_bin_f32(dir + "/enc_vision_features_flat.f32", vs);
    const auto vpos     = read_bin_f32(dir + "/enc_vision_pos_flat.f32",      vps);
    const auto text     = read_bin_f32(dir + "/enc_text_features.f32",        ts);
    const auto tmask    = read_bin_f32(dir + "/md_pca_prompt_mask.f32",       ms);
    std::cout << "  vision " << vs[0] << "×" << vs[1]
              << "  vpos " << vps[0] << "×" << vps[1]
              << "  text " << ts[0] << "×" << ts[1]
              << "  tmask " << ms[0]
              << " valid=";
    int valid = 0; for (float v : tmask) if (v > 0.5f) ++valid;
    std::cout << valid << "/" << tmask.size() << "\n";

    std::cout << "\nrunning 6-layer encoder (CPU, ~1-2 min)...\n" << std::flush;
    sam3::InstructsamDetrEncoder enc(model);
    const auto out = enc.run(vision, vs, vpos, vps, text, ts, tmask, ms);

    std::cout << "\n  per-layer parity vs. detr_encoder_layer_N (obj 0):\n";
    std::cout << "    L  |  cos_sim  |  max_diff  |  rel_L2\n";
    std::cout << "    ---+-----------+------------+---------\n";
    int good = 0;
    for (int l = 0; l < out.num_layers; ++l) {
        std::vector<int64_t> es;
        std::vector<float> exp;
        try {
            exp = read_bin_f32(dir + "/enc_expected_layer_" + std::to_string(l) + ".f32", es);
        } catch (...) {
            std::cout << "    " << l << "  |  (no reference)\n";
            continue;
        }
        const auto & hs = out.hs[static_cast<size_t>(l)];
        const double cos = cosine_per_token(hs, exp, static_cast<int>(out.vision_seq), static_cast<int>(out.hidden_dim));
        double maxdiff = 0.0, l2diff = 0.0, l2ref = 0.0;
        for (size_t i = 0; i < hs.size(); ++i) {
            const double d = static_cast<double>(hs[i]) - exp[i];
            if (std::fabs(d) > maxdiff) maxdiff = std::fabs(d);
            l2diff += d * d;
            l2ref  += static_cast<double>(exp[i]) * exp[i];
        }
        const double rel_l2 = std::sqrt(l2diff) / std::sqrt(l2ref);
        if (cos > 0.99) ++good;
        std::cout << "    L" << l << " | " << cos << " | " << maxdiff << " | " << rel_l2 << "\n";
    }
    std::cout << "\n  " << good << "/6 layers achieve cos_sim > 0.99\n";
    return good == 6 ? 0 : 3;
}
