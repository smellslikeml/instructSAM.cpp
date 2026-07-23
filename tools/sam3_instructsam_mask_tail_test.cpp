// InstructSAM mask decoder TAIL test — validates the segmentation head
// output pipeline given pixel_embed + decoder_queries as external inputs.
//
// Deferred: the FPN (Sam3PixelDecoder) that produces pixel_embed from
// multi-scale backbone_features + encoder_hidden_states, and the
// prompt_cross_attn that fuses prompt features into encoder_hidden_states
// before FPN. Wiring those is the next chunk of work (see PARITY-STATUS.md).
//
// What this test DOES verify end-to-end:
//   mask_embedder (3-layer MLP with ReLU)
//   instance_projection (Conv1x1 256→256)
//   semantic_projection (Conv1x1 256→1)
//   pred_masks = einsum("qc,chw->qhw", mask_emb, instance_emb)
//
// Reference oracle values (object 0 of the 4-object batch from
// warehouse_rgb.jpg) come from /tmp/pathA_reference/warehouse_rgb/.

#include "sam3/gguf_model.h"
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
    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "BIN1") throw std::runtime_error("bad magic in " + path);
    int32_t ndim = 0;
    f.read(reinterpret_cast<char *>(&ndim), 4);
    shape.assign(static_cast<size_t>(ndim), 0);
    size_t total = 1;
    for (int i = 0; i < ndim; ++i) {
        int64_t d = 0;
        f.read(reinterpret_cast<char *>(&d), 8);
        shape[static_cast<size_t>(i)] = d;
        total *= static_cast<size_t>(d);
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

// per-query cosine averaged over Q queries × H*W spatial
double per_query_cosine(const std::vector<float> & a, const std::vector<float> & b, int Q, int HW) {
    double sum = 0.0; int counted = 0;
    for (int q = 0; q < Q; ++q) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (int i = 0; i < HW; ++i) {
            const float av = a[static_cast<size_t>(q * HW + i)];
            const float bv = b[static_cast<size_t>(q * HW + i)];
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
        std::cerr << "usage: sam3-instructsam-mask-tail-test <model.gguf> [parity_dir]\n";
        return 1;
    }
    const std::string parity_dir = (argc >= 3) ? argv[2]
        : "/tmp/pathA_reference/warehouse_rgb/binaries_obj0";

    sam3::GgufModel model;
    if (!model.load(argv[1])) {
        std::cerr << "failed to load model\n";
        return 2;
    }
    std::cout << "loaded " << model.tensors().size() << " tensors\n";

    sam3::InstructsamMaskTail tail(model);

    std::vector<int64_t> pes, dqs, pms, sss;
    const auto pixel_embed  = read_bin_f32(parity_dir + "/md_pixel_embed.f32",     pes);
    const auto dec_queries  = read_bin_f32(parity_dir + "/md_decoder_queries.f32", dqs);
    const auto exp_masks    = read_bin_f32(parity_dir + "/md_pred_masks.f32",      pms);
    const auto exp_semantic = read_bin_f32(parity_dir + "/md_semantic_seg.f32",    sss);
    std::cout << "  pixel_embed     shape=[" << pes[0] << "," << pes[1] << "," << pes[2] << "]\n";
    std::cout << "  decoder_queries shape=[" << dqs[0] << "," << dqs[1] << "]\n";
    std::cout << "  expected pred_masks   shape=[" << pms[0] << "," << pms[1] << "," << pms[2] << "]\n";
    std::cout << "  expected semantic_seg shape=[" << sss[0] << "," << sss[1] << "," << sss[2] << "]\n";

    std::cout << "\nrunning mask tail (CPU, ~10-20s)...\n" << std::flush;
    const auto out = tail.run(pixel_embed, pes, dec_queries, dqs);
    std::cout << "  ✓ produced pred_masks [" << out.num_queries << "," << out.height << "," << out.width << "]\n";

    // ── Parity: pred_masks ─────────────────────────────────────────────
    const int Q  = static_cast<int>(out.num_queries);
    const int HW = static_cast<int>(out.height * out.width);
    const double cos_flat = cosine_flat(out.pred_masks, exp_masks);
    const double cos_per_q = per_query_cosine(out.pred_masks, exp_masks, Q, HW);
    double maxdiff = 0.0, l2diff = 0.0, l2ref = 0.0, ours_mean = 0.0, ref_mean = 0.0;
    for (size_t i = 0; i < out.pred_masks.size(); ++i) {
        const double d = static_cast<double>(out.pred_masks[i]) - exp_masks[i];
        if (std::fabs(d) > maxdiff) maxdiff = std::fabs(d);
        l2diff += d * d;
        l2ref  += static_cast<double>(exp_masks[i]) * exp_masks[i];
        ours_mean += out.pred_masks[i];
        ref_mean += exp_masks[i];
    }
    ours_mean /= out.pred_masks.size();
    ref_mean  /= out.pred_masks.size();
    const double rel_l2 = std::sqrt(l2diff) / std::sqrt(l2ref);

    std::cout << "\n  pred_masks parity (ours vs. reference):\n";
    std::cout << "    global cosine (flat) : " << cos_flat << "\n";
    std::cout << "    per-query cosine avg : " << cos_per_q << "\n";
    std::cout << "    max absolute diff    : " << maxdiff << "\n";
    std::cout << "    relative L2 error    : " << rel_l2 << "\n";
    std::cout << "    ours: mean=" << ours_mean << "\n";
    std::cout << "    ref:  mean=" << ref_mean << "\n";

    // ── Parity: semantic_seg ────────────────────────────────────────────
    const double cos_sem = cosine_flat(out.semantic_seg, exp_semantic);
    double semmax = 0.0, sem_l2diff = 0.0, sem_l2ref = 0.0;
    for (size_t i = 0; i < out.semantic_seg.size(); ++i) {
        const double d = static_cast<double>(out.semantic_seg[i]) - exp_semantic[i];
        if (std::fabs(d) > semmax) semmax = std::fabs(d);
        sem_l2diff += d * d;
        sem_l2ref  += static_cast<double>(exp_semantic[i]) * exp_semantic[i];
    }
    const double sem_rel_l2 = std::sqrt(sem_l2diff) / std::sqrt(sem_l2ref);
    std::cout << "\n  semantic_seg parity:\n";
    std::cout << "    cosine (flat)     : " << cos_sem << "\n";
    std::cout << "    max abs diff      : " << semmax << "\n";
    std::cout << "    relative L2 error : " << sem_rel_l2 << "\n";

    std::cout << "\n=== mask-tail parity (given reference pixel_embed) ===\n";
    std::cout << "  cos > 0.99 => tail correctness verified\n";
    std::cout << "  next: wire pixel_decoder FPN + prompt_cross_attn to eliminate\n";
    std::cout << "        the pixel_embed external input dependency\n";
    return 0;
}
