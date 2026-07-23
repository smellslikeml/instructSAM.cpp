// Parity test for mask_hidden_fcs (LM → decoder queries bridge).
//
// Input:  lmb_seg_output_embeddings [10, 2048] (obj 0 slice)
// Output: lmb_mask_hidden_fcs_out   [10, 256]  (reference)
// Compare our C++ output against the reference.

#include "sam3/gguf_model.h"
#include "sam3/instructsam_lm_bridge.h"

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
    if (argc < 2) { std::cerr << "usage: <model.gguf>\n"; return 1; }
    const std::string dir = (argc >= 3) ? argv[2]
        : "/tmp/pathA_reference/warehouse_rgb/binaries_obj0";

    sam3::GgufModel model;
    if (!model.load(argv[1])) { std::cerr << "load failed\n"; return 2; }
    std::cout << "loaded " << model.tensors().size() << " tensors\n";

    std::vector<int64_t> is, os;
    const auto input  = read_bin_f32(dir + "/lmb_seg_output_embeddings.f32", is);
    const auto ref    = read_bin_f32(dir + "/lmb_mask_hidden_fcs_out.f32",   os);
    std::cout << "  input  " << is[0] << "×" << is[1] << "\n";
    std::cout << "  ref    " << os[0] << "×" << os[1] << "\n";

    sam3::InstructsamMaskHiddenFcs bridge(model);
    const auto out = bridge.run(input, is);

    if (out.data.size() != ref.size()) {
        std::cerr << "size mismatch: ours=" << out.data.size() << " ref=" << ref.size() << "\n";
        return 3;
    }
    double maxdiff = 0.0, l2diff = 0.0, l2ref = 0.0;
    for (size_t i = 0; i < out.data.size(); ++i) {
        const double d = static_cast<double>(out.data[i]) - ref[i];
        if (std::fabs(d) > maxdiff) maxdiff = std::fabs(d);
        l2diff += d * d;
        l2ref  += static_cast<double>(ref[i]) * ref[i];
    }
    std::cout << "\n  mask_hidden_fcs parity:\n"
              << "    cosine (flat)     : " << cos_flat(out.data, ref) << "\n"
              << "    max abs diff      : " << maxdiff << "\n"
              << "    relative L2 error : " << (std::sqrt(l2diff) / std::sqrt(l2ref)) << "\n";
    return 0;
}
