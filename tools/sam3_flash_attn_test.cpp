// Standalone test for detail::flash_attn. Runs a small (batch=1, heads=2,
// seq=8, head_dim=4) attention and compares against a scalar reference.

#include "sam3/detail/cpu_attn.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

// Scalar reference: same layout (batch, n_head, seq, head_dim).
std::vector<float> scalar_attn(
    const std::vector<float> & Q, const std::vector<float> & K, const std::vector<float> & V,
    int64_t B, int64_t H, int64_t S, int64_t D, float scale
) {
    std::vector<float> out(B * H * S * D, 0.0f);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            const float * q = Q.data() + ((b * H + h) * S) * D;
            const float * k = K.data() + ((b * H + h) * S) * D;
            const float * v = V.data() + ((b * H + h) * S) * D;
            std::vector<float> Score(S * S);
            for (int64_t i = 0; i < S; ++i) {
                for (int64_t j = 0; j < S; ++j) {
                    float s = 0.0f;
                    for (int64_t d = 0; d < D; ++d) s += q[i*D+d] * k[j*D+d];
                    Score[i * S + j] = s * scale;
                }
            }
            for (int64_t i = 0; i < S; ++i) {
                float m = Score[i*S];
                for (int64_t j = 1; j < S; ++j) if (Score[i*S+j] > m) m = Score[i*S+j];
                float sum = 0.0f;
                for (int64_t j = 0; j < S; ++j) { Score[i*S+j] = std::exp(Score[i*S+j] - m); sum += Score[i*S+j]; }
                for (int64_t j = 0; j < S; ++j) Score[i*S+j] /= sum;
            }
            for (int64_t i = 0; i < S; ++i) {
                for (int64_t d = 0; d < D; ++d) {
                    float y = 0.0f;
                    for (int64_t j = 0; j < S; ++j) y += Score[i*S+j] * v[j*D+d];
                    out[((b * H + h) * S + i) * D + d] = y;
                }
            }
        }
    }
    return out;
}

}  // namespace

int main() {
    const int64_t B = 1, H = 2, S = 64, D = 64;
    const int64_t N = B * H * S * D;

    std::vector<float> Q(N), K(N), V(N);
    for (int64_t i = 0; i < N; ++i) {
        Q[i] = std::sin(i * 0.1f);
        K[i] = std::cos(i * 0.11f);
        V[i] = 0.3f * i - 0.5f;
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::cout << "running scalar reference\n" << std::flush;
    const auto ref = scalar_attn(Q, K, V, B, H, S, D, scale);
    std::cout << "  first 4: " << ref[0] << " " << ref[1] << " " << ref[2] << " " << ref[3] << "\n\n";

    std::cout << "running flash_attn\n" << std::flush;
    const auto fa = sam3::detail::flash_attn(Q, K, V, B, H, H, S, S, D, scale);
    std::cout << "  first 4: " << fa[0] << " " << fa[1] << " " << fa[2] << " " << fa[3] << "\n\n";

    double dot=0, na=0, nb=0, maxdiff=0;
    for (int64_t i = 0; i < N; ++i) {
        dot += ref[i] * fa[i]; na += ref[i]*ref[i]; nb += fa[i]*fa[i];
        double d = std::fabs(ref[i] - fa[i]); if (d > maxdiff) maxdiff = d;
    }
    const double cs = dot / std::sqrt(na * nb);
    std::cout << "cos_sim=" << cs << "  max_diff=" << maxdiff << "\n";
    // F16 K/V rounding produces ~0.03% relative error on values with
    // large magnitudes. cos_sim is the real bar.
    if (cs > 0.9999) { std::cout << "  ✓ parity (F16-level rounding)\n"; return 0; }
    std::cerr << "  ✗ divergence\n"; return 1;
}
