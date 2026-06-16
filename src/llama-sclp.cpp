#include "llama-sclp.h"
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <map>
#include <set>
#include <cfloat>
#include <thread>

// SCLP Bit Layouts:
// SCLP8:  idx(4b) | sign(1b) | mant_top3(3b)  - 1 byte per weight, palette size 16
// SCLP4:  idx(2b) | sign(1b) | mant_top1(1b)  - 0.5 bytes per weight, palette size 4
// SCLP6:  idx(3b) | sign(1b) | mant_top2(2b)  - 0.75 bytes per weight, palette size 8
// SCLP4M: idx(3b) | sign(1b)                  - 0.5 bytes per weight + 16 B/block codebook
//   (8 free BF16 magnitudes per 256-weight block, Lloyd k-means in linear space —
//    no exponent/mantissa grid; decode = bf16(cb[idx] | sign<<15))

struct sclp_expert_encoded {
    uint8_t palette[16];
    uint8_t palette_size;
    std::vector<uint16_t> scales;
    std::vector<uint8_t> block_palettes; // SCLP4: 4 bytes per block (per-block palette)
    std::vector<uint8_t> ws_stream;
    std::vector<uint32_t> sc_indices;
    std::vector<uint16_t> sc_values;
};

// Simple xorshift32 PRNG — deterministic, matches reference Python seed=42.
struct xorshift32 {
    uint32_t state;
    explicit xorshift32(uint32_t seed) : state(seed) {
        if (state == 0) state = 1;
    }
    uint32_t next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
    float uniform() {
        return (float)next() / 4294967296.0f;
    }
};

static uint16_t float_to_bf16(float f) {
    uint32_t i;
    std::memcpy(&i, &f, 4);
    uint32_t lsb = (i >> 16) & 1;
    uint32_t bias = 0x7FFF + lsb;
    return (uint16_t)((i + bias) >> 16);
}

static float bf16_to_float(uint16_t w) {
    uint32_t i = (uint32_t)w << 16;
    float f;
    std::memcpy(&f, &i, 4);
    return f;
}

// Stochastic rounding of BF16 mantissa LSBs.
static void stochastic_mantissa_round(uint16_t * w, int64_t n, int drop_bits) {
    const uint16_t drop_mask  = (uint16_t)((1u << drop_bits) - 1);
    const uint16_t round_unit = (uint16_t)(1u << drop_bits);
    for (int64_t i = 0; i < n; i++) {
        uint16_t v = w[i];
        uint16_t discarded = v & drop_mask;
        if (discarded == 0) { w[i] = v; continue; }
        uint32_t rng = (uint32_t)((uint64_t)i * 2654435761ULL + 0x9E3779B9u);
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        uint16_t threshold = (uint16_t)(rng & drop_mask);
        v &= (uint16_t)~drop_mask;
        if (threshold < discarded) v = (uint16_t)(v + round_unit);
        w[i] = v;
    }
}

// Soft exponent clipping
static void soft_exponent_clip(
    const uint16_t * input,
    uint16_t * output,
    int64_t n,
    uint8_t threshold
) {
    if (threshold == 0 || input == nullptr || output == nullptr) {
        if (input != output) std::memcpy(output, input, n * sizeof(uint16_t));
        return;
    }
    for (int64_t i = 0; i < n; i++) {
        uint16_t w = input[i];
        uint8_t exp = (w >> 7) & 0xFF;
        uint8_t sign = (w >> 15) & 1;
        uint8_t mant = w & 0x7F;

        if (exp > threshold + 1) {
            exp = threshold;
        } else if (exp == threshold + 1) {
            uint32_t rng = (uint32_t)((uint64_t)i * 2654435761ULL + 42);
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            if (rng & 0x80000000) {
                exp = threshold;
            }
        }

        output[i] = ((uint16_t)sign << 15) | ((uint16_t)exp << 7) | mant;
    }
}

static void kmeans_palette(const float counts[256], uint8_t * palette, int k) {
    std::vector<uint8_t> unique;
    std::vector<float>   weights;
    for (int i = 0; i < 256; i++) {
        if (counts[i] > 0.0f) {
            unique.push_back((uint8_t)i);
            weights.push_back(counts[i]);
        }
    }
    int u = (int)unique.size();
    if (u <= k) {
        for (int i = 0; i < u; i++) palette[i] = unique[i];
        for (int i = u; i < k; i++) palette[i] = 0;
        return;
    }

    float total_w = 0.0f;
    for (int i = 0; i < u; i++) total_w += weights[i];

    xorshift32 rng(42);
    std::vector<float> centers(k);

    float r = rng.uniform() * total_w;
    float cum = 0.0f;
    int first = 0;
    for (int i = 0; i < u; i++) {
        cum += weights[i];
        if (cum >= r) { first = i; break; }
    }
    centers[0] = (float)unique[first];

    for (int c = 1; c < k; c++) {
        std::vector<float> d2(u);
        float d2_sum = 0.0f;
        for (int i = 0; i < u; i++) {
            float min_d = FLT_MAX;
            for (int j = 0; j < c; j++) {
                float d = (float)unique[i] - centers[j];
                d = d * d;
                if (d < min_d) min_d = d;
            }
            d2[i] = min_d * weights[i];
            d2_sum += d2[i];
        }
        if (d2_sum <= 0.0f) {
            centers[c] = (float)unique[c % u];
            continue;
        }
        r = rng.uniform() * d2_sum;
        cum = 0.0f;
        int pick = u - 1;
        for (int i = 0; i < u; i++) {
            cum += d2[i];
            if (cum >= r) { pick = i; break; }
        }
        centers[c] = (float)unique[pick];
    }

    const int max_iter = 20;
    for (int iter = 0; iter < max_iter; iter++) {
        std::vector<int> labels(u);
        for (int i = 0; i < u; i++) {
            float min_d = std::abs((float)unique[i] - centers[0]);
            int best = 0;
            for (int j = 1; j < k; j++) {
                float d = std::abs((float)unique[i] - centers[j]);
                if (d < min_d) { min_d = d; best = j; }
            }
            labels[i] = best;
        }

        bool converged = true;
        std::vector<float> new_centers(k);
        for (int j = 0; j < k; j++) {
            float num = 0.0f, den = 0.0f;
            for (int i = 0; i < u; i++) {
                if (labels[i] == j) {
                    float w = weights[i];
                    num += (float)unique[i] * w;
                    den += w;
                }
            }
            if (den > 0.0f) {
                new_centers[j] = num / den;
            } else {
                std::set<uint8_t> represented;
                for (int i = 0; i < u; i++) {
                    for (int j2 = 0; j2 < k; j2++) {
                        if (j2 != j && labels[i] == j2) {
                            represented.insert(unique[i]);
                        }
                    }
                }
                int best_u = -1;
                float best_cnt = 0.0f;
                for (int i = 0; i < u; i++) {
                    if (represented.find(unique[i]) == represented.end() && weights[i] > best_cnt) {
                        best_cnt = weights[i];
                        best_u = i;
                    }
                }
                if (best_u >= 0) new_centers[j] = (float)unique[best_u];
                else new_centers[j] = centers[j];
            }
            if (std::abs(new_centers[j] - centers[j]) > 1e-6f) converged = false;
        }
        centers = new_centers;
        if (converged) break;
    }

    std::set<uint8_t> seen;
    std::vector<uint8_t> snapped;
    for (int j = 0; j < k; j++) {
        int best = 0;
        float min_d = std::abs((float)unique[0] - centers[j]);
        for (int i = 1; i < u; i++) {
            float d = std::abs((float)unique[i] - centers[j]);
            if (d < min_d) { min_d = d; best = i; }
        }
        uint8_t v = unique[best];
        if (seen.find(v) == seen.end()) {
            seen.insert(v);
            snapped.push_back(v);
        }
    }

    if ((int)snapped.size() < k) {
        std::vector<std::pair<uint32_t, uint8_t>> freq;
        for (int i = 0; i < u; i++) {
            freq.push_back({(uint32_t)weights[i], unique[i]});
        }
        std::sort(freq.rbegin(), freq.rend());
        for (auto & p : freq) {
            if (seen.find(p.second) == seen.end()) {
                seen.insert(p.second);
                snapped.push_back(p.second);
                if ((int)snapped.size() == k) break;
            }
        }
    }

    for (int i = 0; i < k; i++) {
        palette[i] = (i < (int)snapped.size()) ? snapped[i] : 0;
    }
}

// SCLP4M: 1-D Lloyd k-means over |w| for one block → 8 BF16 magnitudes.
// Deterministic: quantile init on the sorted magnitudes, 15 Lloyd iterations.
// Centroids are free BF16 values (not constrained to observed weights).
static void kmeans_magnitude_codebook(const float * w, int64_t n, uint16_t cb[8]) {
    std::vector<float> mags(n);
    for (int64_t i = 0; i < n; i++) mags[i] = std::fabs(w[i]);
    std::sort(mags.begin(), mags.end());

    float c[8];
    for (int j = 0; j < 8; j++) {
        int64_t q = (2 * j + 1) * n / 16;
        if (q >= n) q = n - 1;
        c[j] = mags[q];
    }

    for (int iter = 0; iter < 15; iter++) {
        // Centers stay sorted (means of ordered segments), so assignment is by
        // midpoint boundaries over the sorted magnitude array.
        double sum[8] = {0}; int64_t cnt[8] = {0};
        int j = 0;
        for (int64_t i = 0; i < n; i++) {
            while (j < 7 && mags[i] > 0.5f * (c[j] + c[j + 1])) j++;
            sum[j] += mags[i];
            cnt[j]++;
        }
        bool converged = true;
        for (int k2 = 0; k2 < 8; k2++) {
            if (cnt[k2] > 0) {
                float nc = (float)(sum[k2] / cnt[k2]);
                if (std::fabs(nc - c[k2]) > 1e-8f) converged = false;
                c[k2] = nc;
            }
        }
        if (converged) break;
    }

    for (int j = 0; j < 8; j++) {
        cb[j] = (uint16_t)(float_to_bf16(c[j]) & 0x7FFF); // magnitude: sign bit clear
    }
}

static sclp_expert_encoded encode_sclp_expert(
    ggml_type type,
    const float * data,
    int64_t n,
    uint32_t expert_offset,
    const float * imatrix,
    int64_t K,
    uint8_t  clip_threshold,
    float    sidecar_imatrix_budget
) {
    sclp_expert_encoded enc;

    int sm_bits;
    int p_max;
    if (type == GGML_TYPE_SCLP8) {
        sm_bits = 4; p_max = 16;
    } else if (type == GGML_TYPE_SCLP4) {
        sm_bits = 2; p_max = 4;
    } else if (type == GGML_TYPE_SCLP5) {
        sm_bits = 3; p_max = 4;  // idx2 | sign1 | mant2, 4-entry per-block palette
    } else if (type == GGML_TYPE_SCLP4M) {
        sm_bits = 1; p_max = 0;  // idx3 | sign1, per-block 8-entry magnitude codebook
    } else {
        sm_bits = 3; p_max = 8;
    }
    const bool per_block_palette  = (type == GGML_TYPE_SCLP4 || type == GGML_TYPE_SCLP5);
    const bool per_block_codebook = (type == GGML_TYPE_SCLP4M);
    const int  mant_bits = sm_bits - 1;                  // SCLP4:1, SCLP5:2, SCLP6:2, SCLP8:3
    const int  mant_shift = 7 - mant_bits;               // bit position of kept mantissa top bits

    const int qk = qk_for_type(type);
    int64_t n_blocks = (n + qk - 1) / qk;
    std::vector<float> scaled_data(n);

    if (per_block_palette || per_block_codebook) {
        // SCLP4/SCLP5/SCLP4M: per-block table modes — no PBS normalization
        for (int64_t i = 0; i < n; i++) scaled_data[i] = data[i];
    } else {
        enc.scales.resize(n_blocks);
        static const bool no_pbs = (getenv("SCLP_NO_PBS") != nullptr);
        for (int64_t b = 0; b < n_blocks; b++) {
            int64_t b_start = b * qk;
            int64_t b_end = std::min(b_start + qk, n);
            if (no_pbs) {
                enc.scales[b] = float_to_bf16(1.0f);
                for (int64_t i = b_start; i < b_end; i++) {
                    scaled_data[i] = data[i];
                }
            } else {
                float max_val = 0.0f;
                for (int64_t i = b_start; i < b_end; i++) {
                    max_val = std::max(max_val, std::abs(data[i]));
                }
                enc.scales[b] = float_to_bf16(max_val);
                float s = bf16_to_float(enc.scales[b]);
                float inv_s = (s > 0.0f) ? 1.0f / s : 0.0f;
                for (int64_t i = b_start; i < b_end; i++) {
                    scaled_data[i] = data[i] * inv_s;
                }
            }
        }
    }

    std::vector<uint16_t> bf16_weights(n);
    for (int64_t i = 0; i < n; i++) {
        bf16_weights[i] = float_to_bf16(scaled_data[i]);
    }

    if (clip_threshold > 0) {
        std::vector<uint16_t> clipped(n);
        soft_exponent_clip(bf16_weights.data(), clipped.data(), n, clip_threshold);
        bf16_weights.swap(clipped);
    }

    if (!per_block_codebook) {  // SCLP4M has no mantissa grid — codebook search below
        // Stochastic rounding now defaults OFF (opt-in via SCLP_STOCHASTIC_ROUND=1).
        // The joint (idx, mantissa) min-error search below picks the stored mantissa
        // optimally on every path, so SR no longer affects any stored code — its only
        // residual effect is perturbing the exponent histogram (palette k-means) and
        // sidecar classification, which is pure noise. A/B on Llama-3-8B SCLP6
        // (wikitext, 80 chunks): SR off 10.041 ± 0.200 vs SR on 10.088 ± 0.203, and
        // the SR-off blob is smaller (less sidecar). See review item #3.
        static const char * env = std::getenv("SCLP_STOCHASTIC_ROUND");
        const bool enabled = (env && env[0] == '1');
        if (enabled) {
            int drop_bits = mant_shift;  // drop bits below the kept mantissa top bits
            stochastic_mantissa_round(bf16_weights.data(), n, drop_bits);
        }
    }

    std::vector<uint8_t> indices(n);
    std::vector<uint8_t> sm_nibbles(n);
    bool has_imatrix = (imatrix != nullptr) && (sidecar_imatrix_budget > 0.0f) && (K > 0);
    std::vector<float> priority(n, 0.0f);

    if (per_block_codebook) {
        // SCLP4M: each QK_SCLP4-weight block gets its own 8-entry magnitude codebook
        // (free BF16 values from linear-space Lloyd k-means). No mandatory sidecar tier:
        // the codebook adapts to outliers, so rescue is purely discretionary
        // (importance × squared-error ranking below).
        enc.palette_size = 0;
        enc.block_palettes.resize(n_blocks * 16);

        for (int64_t b = 0; b < n_blocks; b++) {
            int64_t b_start = b * qk;
            int64_t b_end = std::min(b_start + qk, n);

            uint16_t cb[8];
            kmeans_magnitude_codebook(data + b_start, b_end - b_start, cb);
            std::memcpy(enc.block_palettes.data() + b * 16, cb, 16);

            float cbf[8];
            for (int j = 0; j < 8; j++) cbf[j] = bf16_to_float(cb[j]);

            for (int64_t i = b_start; i < b_end; i++) {
                float m = std::fabs(data[i]);
                int best = 0;
                float best_err = std::fabs(cbf[0] - m);
                for (int j = 1; j < 8; j++) {
                    float err = std::fabs(cbf[j] - m);
                    if (err < best_err) { best_err = err; best = j; }
                }
                indices[i]    = (uint8_t)best;
                sm_nibbles[i] = (uint8_t)(std::signbit(data[i]) ? 1 : 0);
                if (has_imatrix) {
                    priority[i] = imatrix[(uint64_t)i % K] * best_err * best_err;
                }
            }
        }
    } else if (per_block_palette) {
        // Per-block palette: each QK_SCLP4-weight block gets its own 4-entry k-means palette.
        // SCLP4 keeps 1 mantissa bit; SCLP5 keeps 2 (mant_bits / mant_shift).
        const int mant_levels = 1 << mant_bits;
        const uint8_t mant_mask = (uint8_t)(mant_levels - 1);
        enc.palette_size = 0;
        enc.block_palettes.resize(n_blocks * 4);

        for (int64_t b = 0; b < n_blocks; b++) {
            int64_t b_start = b * qk;
            int64_t b_end = std::min(b_start + qk, n);

            float bcounts[256] = {};
            for (int64_t i = b_start; i < b_end; i++) {
                uint8_t exp = (bf16_weights[i] >> 7) & 0xFF;
                bcounts[exp] += 1.0f;
            }

            uint8_t bpal[4];
            kmeans_palette(bcounts, bpal, 4);
            for (int j = 0; j < 4; j++) enc.block_palettes[b * 4 + j] = bpal[j];

            uint8_t bexp_to_idx[256];
            uint8_t bexp_distance[256];
            for (int e2 = 0; e2 < 256; e2++) {
                int best = 0, mind = 1000;
                for (int j = 0; j < 4; j++) {
                    int d = std::abs((int)e2 - (int)bpal[j]);
                    if (d < mind) { mind = d; best = j; }
                }
                bexp_to_idx[e2] = (uint8_t)best;
                bexp_distance[e2] = (uint8_t)mind;
            }

            bool bin_pal[256] = {};
            for (int j = 0; j < 4; j++) bin_pal[bpal[j]] = true;

            for (int64_t i = b_start; i < b_end; i++) {
                uint16_t w = bf16_weights[i];
                uint8_t exp = (w >> 7) & 0xFF;
                uint8_t sign = (w >> 15) & 1;

                // Pick the (palette idx, mantissa bit) pair that minimizes
                // reconstruction error against the original value. Beats nearest-
                // exponent + top-mantissa-bit: a neighbouring palette exponent often
                // reconstructs closer (e.g. 1.8*2^e -> 1.0*2^(e+1) beats 1.5*2^e).
                // Pure encoder change — wire format and decoders are unchanged.
                float orig = data[i];
                float best_err = FLT_MAX;
                uint8_t best_idx = bexp_to_idx[exp], best_mant = (uint8_t)((w >> mant_shift) & mant_mask);
                for (int j = 0; j < 4; j++) {
                    for (int m = 0; m < mant_levels; m++) {
                        uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)bpal[j] << 7) | ((uint16_t)m << mant_shift);
                        float err = std::fabs(bf16_to_float(bits) - orig);
                        if (err < best_err) { best_err = err; best_idx = (uint8_t)j; best_mant = (uint8_t)m; }
                    }
                }
                indices[i] = best_idx;
                sm_nibbles[i] = (sign << mant_bits) | best_mant;

                if (!bin_pal[exp] && bexp_distance[exp] > 1) {
                    enc.sc_indices.push_back(expert_offset + (uint32_t)i);
                    enc.sc_values.push_back(float_to_bf16(data[i]));
                } else if (has_imatrix) {
                    // Rank by importance × squared reconstruction error (not exponent
                    // distance). best_err is already computed against original data[i]
                    // and includes mantissa-grid error, which the sidecar does fix.
                    // Weights with best_err == 0 naturally get priority 0 and are skipped.
                    priority[i] = imatrix[(uint64_t)i % K] * best_err * best_err;
                }
            }
        }
    } else {
        // Global palette path for SCLP6/SCLP8
        // Weight each exponent bin by sum(|w|) rather than raw count so that rare
        // high-magnitude exponents pull palette centroids toward them.  Near-zero
        // exponents (e.g. exp≈0, |w|≈0) can dominate the count histogram and waste
        // palette slots on values that contribute negligibly to reconstruction error.
        // When SCLP_IMATRIX_PALETTE=1 the imatrix signal is folded in on top.
        float wcounts[256] = {0.0f};
        static const char * env_imp = std::getenv("SCLP_IMATRIX_PALETTE");
        const bool weight_with_imatrix = (env_imp && env_imp[0] == '1') && (imatrix != nullptr) && (K > 0);
        for (int64_t i = 0; i < n; i++) {
            uint8_t exp = (bf16_weights[i] >> 7) & 0xFF;
            float mag = std::fabs(scaled_data[i]);  // magnitude in encoded domain (after PBS)
            float w = weight_with_imatrix ? imatrix[i % K] * mag : mag;
            wcounts[exp] += w;
        }

        kmeans_palette(wcounts, enc.palette, p_max);
        enc.palette_size = p_max;

        uint8_t exp_to_idx[256];
        uint8_t exp_distance[256];
        for (int e2 = 0; e2 < 256; e2++) {
            int best_idx = 0;
            int min_diff = 1000;
            for (int i = 0; i < enc.palette_size; i++) {
                int diff = std::abs((int)e2 - (int)enc.palette[i]);
                if (diff < min_diff) {
                    min_diff = diff;
                    best_idx = i;
                }
            }
            exp_to_idx[e2] = (uint8_t)best_idx;
            exp_distance[e2] = (uint8_t)min_diff;
        }

        bool in_palette[256] = {false};
        for (int i = 0; i < enc.palette_size; i++) in_palette[enc.palette[i]] = true;

        // Joint (palette-idx, mantissa) min-error search for SCLP6/SCLP8 global palette.
        // For each weight we try every (palette entry j, mantissa level m) pair, reconstruct
        // the BF16 value scaled back to the original domain, and pick the pair minimising
        // |reconstruction - original|. This mirrors the per-block-palette path above and
        // beats nearest-exponent + copied-mantissa because a neighbouring palette exponent
        // often reconstructs closer (e.g. 1.8·2^e → 1.0·2^(e+1) beats 1.5·2^e).
        // SCLP8: ≤16 palette entries × 8 mantissa levels = ≤128 candidates/weight.
        // SCLP6: ≤8  palette entries × 4 mantissa levels = ≤32  candidates/weight.
        // Pure encoder change — wire format and decoders are unchanged.
        const int mant_levels = 1 << mant_bits;
        const uint8_t mant_mask = (uint8_t)(mant_levels - 1);

        for (int64_t i = 0; i < n; i++) {
            uint16_t w = bf16_weights[i];
            uint8_t exp  = (w >> 7) & 0xFF;
            uint8_t sign = (w >> 15) & 1;

            // PBS scale for this block — decoder multiplies reconstructed BF16 by s_b,
            // so we compare scale×bf16(bits) against the original unscaled data[i].
            float s_b  = bf16_to_float(enc.scales[(size_t)(i / qk)]);
            float orig = data[i];

            float   best_err  = FLT_MAX;
            uint8_t best_idx  = exp_to_idx[exp];
            uint8_t best_mant = (uint8_t)((w >> mant_shift) & mant_mask);

            for (int j = 0; j < enc.palette_size; j++) {
                for (int m = 0; m < mant_levels; m++) {
                    uint16_t bits = ((uint16_t)sign << 15)
                                  | ((uint16_t)enc.palette[j] << 7)
                                  | ((uint16_t)m << mant_shift);
                    float recon = bf16_to_float(bits) * s_b;
                    float err   = std::fabs(recon - orig);
                    if (err < best_err) {
                        best_err  = err;
                        best_idx  = (uint8_t)j;
                        best_mant = (uint8_t)m;
                    }
                }
            }

            indices[i]    = best_idx;
            sm_nibbles[i] = (sign << mant_bits) | best_mant;

            if (!in_palette[exp] && exp_distance[exp] > 1) {
                enc.sc_indices.push_back(expert_offset + (uint32_t)i);
                enc.sc_values.push_back(float_to_bf16(data[i]));
            } else if (has_imatrix) {
                // Use best_err (already against original data[i]) for the discretionary
                // sidecar priority — avoids a second reconstruction pass.
                priority[i] = imatrix[(uint64_t)i % K] * best_err * best_err;
            }
        }
    }

    if (has_imatrix) {
        int n_extra = (int)((float)n * sidecar_imatrix_budget);
        if (n_extra > 0) {
            std::vector<std::pair<float, uint32_t>> cand;
            cand.reserve(n);
            for (int64_t i = 0; i < n; i++) {
                if (priority[i] > 0.0f) {
                    cand.push_back({priority[i], (uint32_t)(expert_offset + i)});
                }
            }
            if ((int)cand.size() > n_extra) {
                std::partial_sort(cand.begin(), cand.begin() + n_extra, cand.end(),
                    [](const std::pair<float, uint32_t> & a, const std::pair<float, uint32_t> & b) {
                        return a.first > b.first;
                    });
                cand.resize(n_extra);
            }
            for (auto & c : cand) {
                uint32_t idx = c.second - expert_offset;
                enc.sc_indices.push_back(c.second);
                enc.sc_values.push_back(float_to_bf16(data[idx]));
            }
        }
    }

    // Sort sidecar by index. Mandatory entries are already in index order but the
    // discretionary imatrix entries were appended by priority. A sorted sidecar lets the
    // fused-GEMV sidecar correction find each row's entries as one contiguous range
    // (gidx = row*K + col), enabling a warp-per-row, atomic-free correction. The two-pass
    // fixup and CPU decode scatter by index, so order is irrelevant to them.
    if (enc.sc_indices.size() > 1) {
        std::vector<uint32_t> order(enc.sc_indices.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = (uint32_t)i;
        std::sort(order.begin(), order.end(),
            [&](uint32_t a, uint32_t b) { return enc.sc_indices[a] < enc.sc_indices[b]; });
        std::vector<uint32_t> si(enc.sc_indices.size());
        std::vector<uint16_t> sv(enc.sc_values.size());
        for (size_t i = 0; i < order.size(); i++) { si[i] = enc.sc_indices[order[i]]; sv[i] = enc.sc_values[order[i]]; }
        enc.sc_indices.swap(si);
        enc.sc_values.swap(sv);
    }

    if (type == GGML_TYPE_SCLP8) {
        enc.ws_stream.resize(n);
        for (int64_t i = 0; i < n; i++) {
            enc.ws_stream[i] = (indices[i] << 4) | (sm_nibbles[i] & 0xF);
        }
    } else if (type == GGML_TYPE_SCLP4) {
        enc.ws_stream.resize((n + 1) / 2);
        for (int64_t i = 0; i < n; i += 2) {
            uint8_t b = (indices[i] << 6) | (sm_nibbles[i] << 4);
            if (i + 1 < n) {
                b |= (indices[i+1] << 2) | sm_nibbles[i+1];
            }
            enc.ws_stream[i/2] = b;
        }
    } else if (type == GGML_TYPE_SCLP4M) {
        // nibble = idx(3:1) | sign(0); high nibble = even weight
        enc.ws_stream.resize((n + 1) / 2);
        for (int64_t i = 0; i < n; i += 2) {
            uint8_t b = (uint8_t)(((indices[i] << 1) | sm_nibbles[i]) << 4);
            if (i + 1 < n) {
                b |= (uint8_t)((indices[i+1] << 1) | sm_nibbles[i+1]);
            }
            enc.ws_stream[i/2] = b;
        }
    } else if (type == GGML_TYPE_SCLP5) {
        // 8 weights -> 5 bytes: eight 5-bit codes (idx2|sign1|mant2) MSB-first in 40 bits.
        enc.ws_stream.resize((n + 7) / 8 * 5);
        for (int64_t i = 0; i < n; i += 8) {
            uint64_t val = 0;
            for (int j = 0; j < 8; j++) {
                uint8_t idx = (i + j < n) ? indices[i + j] : 0;
                uint8_t sm  = (i + j < n) ? sm_nibbles[i + j] : 0;  // 3-bit: sign(2)|mant(1:0)
                uint8_t code = (uint8_t)(((idx & 0x3) << 3) | (sm & 0x7));
                val |= (uint64_t)code << (5 * (7 - j));
            }
            int64_t out = (i / 8) * 5;
            enc.ws_stream[out + 0] = (uint8_t)((val >> 32) & 0xFF);
            enc.ws_stream[out + 1] = (uint8_t)((val >> 24) & 0xFF);
            enc.ws_stream[out + 2] = (uint8_t)((val >> 16) & 0xFF);
            enc.ws_stream[out + 3] = (uint8_t)((val >> 8)  & 0xFF);
            enc.ws_stream[out + 4] = (uint8_t)( val        & 0xFF);
        }
    } else {
        enc.ws_stream.resize((n + 3) / 4 * 3);
        for (int64_t i = 0; i < n; i += 4) {
            uint32_t val = 0;
            for (int j = 0; j < 4; j++) {
                uint8_t idx = (i + j < n) ? indices[i + j] : 0;
                uint8_t sm  = (i + j < n) ? sm_nibbles[i + j] : 0;
                val |= (uint32_t)((idx << 3) | sm) << (6 * (3 - j));
            }
            int64_t out_idx = (i / 4) * 3;
            enc.ws_stream[out_idx + 0] = (val >> 16) & 0xFF;
            enc.ws_stream[out_idx + 1] = (val >> 8)  & 0xFF;
            enc.ws_stream[out_idx + 2] = val & 0xFF;
        }
    }

    return enc;
}

size_t llama_tensor_quantize_sclp(
    ggml_type type,
    const float * f32_data,
    void * new_data,
    int64_t nelements,
    int64_t n_experts,
    int64_t K,
    const float * imatrix,
    uint8_t  clip_threshold,
    float    sidecar_imatrix_budget
) {
    int64_t weights_per_expert = nelements / n_experts;
    std::vector<sclp_expert_encoded> experts(n_experts);

    int n_threads = std::min((int)n_experts, (int)std::thread::hardware_concurrency());
    if (n_threads <= 1 || n_experts <= 1) {
        for (int64_t e = 0; e < n_experts; e++) {
            const float * expert_imatrix = imatrix ? imatrix + e * K : nullptr;
            experts[e] = encode_sclp_expert(
                type, f32_data + e * weights_per_expert, weights_per_expert,
                (uint32_t)(e * weights_per_expert), expert_imatrix, K,
                clip_threshold, sidecar_imatrix_budget);
        }
    } else {
        std::vector<std::thread> threads(n_threads);
        auto work = [&](int tid) {
            for (int64_t e = tid; e < n_experts; e += n_threads) {
                const float * expert_imatrix = imatrix ? imatrix + e * K : nullptr;
                experts[e] = encode_sclp_expert(
                    type, f32_data + e * weights_per_expert, weights_per_expert,
                    (uint32_t)(e * weights_per_expert), expert_imatrix, K,
                    clip_threshold, sidecar_imatrix_budget);
            }
        };
        for (int t = 0; t < n_threads; t++) threads[t] = std::thread(work, t);
        for (auto & t : threads) t.join();
    }

    uint8_t * out = (uint8_t *)new_data;
    uint32_t nw = (uint32_t)nelements;
    uint32_t ne = (uint32_t)n_experts;

    std::memcpy(out, &nw, 4);
    std::memcpy(out + 4, &ne, 4);
    uint8_t * p = out + 8;

    for (uint32_t e = 0; e < ne; e++) {
        *p++ = experts[e].palette_size;
        std::memcpy(p, experts[e].palette, experts[e].palette_size);
        p += experts[e].palette_size;
    }

    if (type == GGML_TYPE_SCLP4 || type == GGML_TYPE_SCLP5 || type == GGML_TYPE_SCLP4M) {
        // Per-block tables: 4 B/block palettes (SCLP4/5) or 16 B/block codebooks (SCLP4M)
        for (uint32_t e = 0; e < ne; e++) {
            std::memcpy(p, experts[e].block_palettes.data(), experts[e].block_palettes.size());
            p += experts[e].block_palettes.size();
        }
    } else {
        for (uint32_t e = 0; e < ne; e++) {
            std::memcpy(p, experts[e].scales.data(), experts[e].scales.size() * 2);
            p += experts[e].scales.size() * 2;
        }
    }

    for (uint32_t e = 0; e < ne; e++) {
        std::memcpy(p, experts[e].ws_stream.data(), experts[e].ws_stream.size());
        p += experts[e].ws_stream.size();
    }

    std::vector<uint32_t> all_sc_indices;
    std::vector<uint16_t> all_sc_values;
    for (uint32_t e = 0; e < ne; e++) {
        all_sc_indices.insert(all_sc_indices.end(), experts[e].sc_indices.begin(), experts[e].sc_indices.end());
        all_sc_values.insert(all_sc_values.end(), experts[e].sc_values.begin(), experts[e].sc_values.end());
    }

    // Sidecar v2 (see sclp_bridge_common.cuh): [u32 count|bit31][u32 K]
    // [u32 row_offsets[n_rows+1]][u16 cols][u16 vals]. Entries arrive sorted by
    // global index (sorted per expert, experts in ascending offset order), which
    // is exactly (row, col) order.
    uint32_t sc_count = (uint32_t)all_sc_indices.size();
    uint32_t count_word = 0x80000000u | sc_count;
    std::memcpy(p, &count_word, 4);
    p += 4;
    if (sc_count > 0) {
        GGML_ASSERT(K > 0 && K <= 65536 && nelements % K == 0 &&
                    "SCLP sidecar v2 requires row length K in (0, 65536] dividing nelements");
        const uint32_t n_rows = (uint32_t)(nelements / K);
        uint32_t Ku = (uint32_t)K;
        std::memcpy(p, &Ku, 4);
        p += 4;

        std::vector<uint32_t> row_offsets(n_rows + 1, 0);
        for (uint32_t i = 0; i < sc_count; i++) {
            row_offsets[all_sc_indices[i] / Ku + 1]++;
        }
        for (uint32_t r = 0; r < n_rows; r++) {
            row_offsets[r + 1] += row_offsets[r];
        }
        std::memcpy(p, row_offsets.data(), (size_t)(n_rows + 1) * 4);
        p += (size_t)(n_rows + 1) * 4;

        std::vector<uint16_t> cols(sc_count);
        for (uint32_t i = 0; i < sc_count; i++) {
            cols[i] = (uint16_t)(all_sc_indices[i] % Ku);
        }
        std::memcpy(p, cols.data(), (size_t)sc_count * 2);
        p += (size_t)sc_count * 2;
        std::memcpy(p, all_sc_values.data(), (size_t)sc_count * 2);
        p += (size_t)sc_count * 2;
    }

    return (size_t)(p - out);
}

void llama_tensor_dequantize_sclp(
    ggml_type type,
    const void * data,
    float * f32_data,
    int64_t nelements
) {
    const uint8_t * p = (const uint8_t *)data;
    uint32_t nw, ne;
    std::memcpy(&nw, p, 4);
    std::memcpy(&ne, p + 4, 4);
    p += 8;

    std::vector<uint8_t> palette_sizes(ne);
    std::vector<const uint8_t*> palettes(ne);
    for (uint32_t e = 0; e < ne; e++) {
        palette_sizes[e] = *p++;
        palettes[e] = p;
        p += palette_sizes[e];
    }

    const int qk = qk_for_type(type);
    int64_t wpe = nw / ne;
    int64_t bpe = (wpe + qk - 1) / qk; // blocks per expert

    const uint8_t * block_palettes_base = nullptr;
    const uint16_t * scales = nullptr;
    if (type == GGML_TYPE_SCLP4 || type == GGML_TYPE_SCLP5) {
        // Per-block palette: 4 bytes per block
        block_palettes_base = p;
        p += (uint64_t)ne * bpe * 4;
    } else if (type == GGML_TYPE_SCLP4M) {
        // Per-block magnitude codebook: 16 bytes (8 × BF16) per block
        block_palettes_base = p;
        p += (uint64_t)ne * bpe * 16;
    } else {
        // PBS: 2-byte BF16 scale per block
        scales = (const uint16_t *)p;
        p += (uint64_t)ne * bpe * 2;
    }

    const uint8_t * ws = p;
    int64_t weights_per_expert = wpe;

    for (int64_t i = 0; i < nelements; i++) {
        int64_t e = i / weights_per_expert;
        int64_t local_idx = i % weights_per_expert;

        uint8_t exp, smn;
        float scale = 1.0f;

        if (type == GGML_TYPE_SCLP4M) {
            // Direct codebook decode — no exponent/mantissa assembly, no scale.
            int64_t block_idx = e * bpe + (local_idx / qk);
            const uint8_t * cb = block_palettes_base + block_idx * 16;
            int64_t expert_ws_bytes = (weights_per_expert + 1) / 2;
            const uint8_t * ws_e = ws + (uint64_t)e * expert_ws_bytes;
            uint8_t b = ws_e[local_idx / 2];
            uint8_t nibble = (local_idx % 2 == 0) ? (b >> 4) : (b & 0xF);
            uint16_t mag;
            std::memcpy(&mag, cb + (nibble >> 1) * 2, 2);
            uint16_t bits = (uint16_t)(mag | ((nibble & 1) << 15));
            f32_data[i] = bf16_to_float(bits);
            continue;
        }

        if (type == GGML_TYPE_SCLP4) {
            int64_t block_idx = e * bpe + (local_idx / qk);
            const uint8_t * bpal = block_palettes_base + block_idx * 4;
            uint8_t b = ws[i/2];
            uint8_t nibble = (i % 2 == 0) ? (b >> 4) : (b & 0xF);
            exp = bpal[nibble >> 2];
            smn = nibble & 0x3;
        } else if (type == GGML_TYPE_SCLP5) {
            int64_t block_idx = e * bpe + (local_idx / qk);
            const uint8_t * bpal = block_palettes_base + block_idx * 4;
            int64_t expert_ws_bytes = (weights_per_expert + 7) / 8 * 5;
            int64_t group = local_idx / 8;
            int64_t sub   = local_idx % 8;
            const uint8_t * g = ws + (uint64_t)e * expert_ws_bytes + group * 5;
            uint64_t val = ((uint64_t)g[0] << 32) | ((uint64_t)g[1] << 24) |
                           ((uint64_t)g[2] << 16) | ((uint64_t)g[3] << 8) | (uint64_t)g[4];
            uint8_t code = (uint8_t)((val >> (5 * (7 - sub))) & 0x1F);
            exp = bpal[code >> 3];
            smn = code & 0x7;
        } else if (type == GGML_TYPE_SCLP8) {
            int64_t scale_idx = e * bpe + (local_idx / qk);
            scale = bf16_to_float(scales[scale_idx]);
            uint8_t b = ws[i];
            exp = palettes[e][b >> 4];
            smn = b & 0x0F;
        } else { // SCLP6
            int64_t scale_idx = e * bpe + (local_idx / qk);
            scale = bf16_to_float(scales[scale_idx]);
            int64_t group = local_idx / 4;
            int64_t sub = local_idx % 4;
            const uint8_t * expert_ws = ws + (uint64_t)e * ((weights_per_expert + 3) / 4 * 3);
            const uint8_t * group_ptr = expert_ws + group * 3;
            uint32_t val = (group_ptr[0] << 16) | (group_ptr[1] << 8) | group_ptr[2];
            uint8_t sixbits = (val >> (6 * (3 - sub))) & 0x3F;
            exp = palettes[e][sixbits >> 3];
            smn = sixbits & 0x7;
        }

        int sm_bits = (type == GGML_TYPE_SCLP8 ? 4 : type == GGML_TYPE_SCLP4 ? 2 : 3);
        uint8_t sign = (smn >> (sm_bits - 1)) & 1;
        uint8_t mant = smn & ((1 << (sm_bits - 1)) - 1);

        int shift = (type == GGML_TYPE_SCLP8 ? 4 : type == GGML_TYPE_SCLP4 ? 6 : 5);  // SCLP5/SCLP6 = 5
        uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp << 7) | ((uint16_t)mant << shift);
        f32_data[i] = bf16_to_float(bits) * scale;
    }

    // Sidecar fixup
    int64_t ws_bytes_per_expert =
        (type == GGML_TYPE_SCLP8) ? weights_per_expert :
        (type == GGML_TYPE_SCLP4 || type == GGML_TYPE_SCLP4M) ? (weights_per_expert + 1) / 2 :
        (type == GGML_TYPE_SCLP5) ? (weights_per_expert + 7) / 8 * 5 :
                                    (weights_per_expert + 3) / 4 * 3;
    // Sidecar v2: [u32 count|bit31][u32 K][u32 row_offsets[n_rows+1]][u16 cols][u16 vals]
    p = ws + (uint64_t)ne * ws_bytes_per_expert;
    uint32_t count_word;
    std::memcpy(&count_word, p, 4);
    const uint32_t sc_count = count_word & 0x7FFFFFFFu;
    if (!(count_word & 0x80000000u) || sc_count == 0) {
        return; // empty, or pre-v2 blob (sidecar skipped — regenerate the GGUF)
    }
    uint32_t Ku;
    std::memcpy(&Ku, p + 4, 4);
    const uint32_t n_rows = (uint32_t)(nw / Ku);
    const uint8_t * offs = p + 8;
    const uint8_t * cols = offs + (uint64_t)(n_rows + 1) * 4;
    const uint8_t * vals = cols + (uint64_t)sc_count * 2;
    for (uint32_t r = 0; r < n_rows; r++) {
        uint32_t lo, hi;
        std::memcpy(&lo, offs + (uint64_t)r * 4,     4);
        std::memcpy(&hi, offs + (uint64_t)r * 4 + 4, 4);
        for (uint32_t i = lo; i < hi && i < sc_count; i++) {
            uint16_t c, v;
            std::memcpy(&c, cols + (uint64_t)i * 2, 2);
            std::memcpy(&v, vals + (uint64_t)i * 2, 2);
            uint64_t gidx = (uint64_t)r * Ku + c;
            if (gidx < (uint64_t)nelements) {
                f32_data[gidx] = bf16_to_float(v);
            }
        }
    }
}
