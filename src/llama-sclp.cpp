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
// SCLP8: idx(4b) | sign(1b) | mant_top3(3b)   - 1 byte per weight, palette size 16
// SCLP4: idx(2b) | sign(1b) | mant_top1(1b)   - 0.5 bytes per weight, palette size 4
// SCLP6: idx(3b) | sign(1b) | mant_top2(2b)   - 0.75 bytes per weight, palette size 8

struct sclp_expert_encoded {
    uint8_t palette[16];
    uint8_t palette_size;
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

// Stochastic rounding of BF16 mantissa LSBs. Replaces deterministic truncation
// (which produces systematic per-weight bias in the same direction for every
// positive weight) with unbiased noise. The expected value of each rounded
// weight equals the original, so per-token errors no longer compound across
// autoregressive generation steps.
//
// drop_bits = number of mantissa LSBs that will be discarded by the encoder:
//   SCLP4 keeps 1 mantissa bit -> drop_bits = 6
//   SCLP6 keeps 2 mantissa bits -> drop_bits = 5
//   SCLP8 keeps 3 mantissa bits -> drop_bits = 4
//
// If rounding overflows the kept mantissa region the carry naturally propagates
// into the exponent (BF16 bit layout: sign(1) exp(8) mant(7)). The downstream
// palette/sidecar steps then see the rounded exponent, which is the correct
// behavior — keeps expected value exact.
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

// Soft exponent clipping, matching Python soft_exponent_clip().
//   exponent > threshold+1  → hard-clip to threshold
//   exponent == threshold+1 → 50% survive (flat stochastic)
//   threshold == 0          → passthrough (no clipping)
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
            // Per-weight deterministic PRNG: position-seeded xorshift, 50% flat.
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

// 1-D weighted k-means over BF16 exponent values (0-255).
// Matches Python _kmeans_palette(), but accepts arbitrary float weights so
// callers can mix raw frequency with imatrix-importance per exponent bucket.
static void kmeans_palette(const float counts[256], uint8_t * palette, int k) {
    // Collect unique exponents and their weights
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

    // k-means++ initialization with deterministic seed 42
    xorshift32 rng(42);
    std::vector<float> centers(k);

    // First centroid: weighted random choice
    float r = rng.uniform() * total_w;
    float cum = 0.0f;
    int first = 0;
    for (int i = 0; i < u; i++) {
        cum += weights[i];
        if (cum >= r) { first = i; break; }
    }
    centers[0] = (float)unique[first];

    // Remaining centroids
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

    // EM iterations (max 20)
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
                // Dead cluster: reinitialize to most-frequent unrepresented exponent
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

    // Snap each centroid to the nearest observed exponent value
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

    // Fill remaining slots by frequency
    if ((int)snapped.size() < k) {
        std::vector<std::pair<uint32_t, uint8_t>> freq;
        for (int i = 0; i < u; i++) {
            freq.push_back({weights[i], unique[i]});
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

    int p_bits, sm_bits;
    int p_max;
    if (type == GGML_TYPE_SCLP8) {
        p_bits = 4; sm_bits = 4; p_max = 16;
    } else if (type == GGML_TYPE_SCLP4) {
        p_bits = 2; sm_bits = 2; p_max = 4;
    } else {
        p_bits = 3; sm_bits = 3; p_max = 8;
    }

    // 1. Convert F32 -> BF16
    std::vector<uint16_t> bf16_weights(n);
    for (int64_t i = 0; i < n; i++) {
        bf16_weights[i] = float_to_bf16(data[i]);
    }

    // 2. Soft exponent clipping
    if (clip_threshold > 0) {
        std::vector<uint16_t> clipped(n);
        soft_exponent_clip(bf16_weights.data(), clipped.data(), n, clip_threshold);
        bf16_weights.swap(clipped);
    }

    // 2b. Stochastic mantissa rounding (default on; set SCLP_STOCHASTIC_ROUND=0
    // to disable for A/B comparison). Eliminates per-tensor bias that accumulates
    // across autoregressive steps and triggers mode collapse on sensitive tensors
    // (e.g., ffn_down_exps in MoE IT models).
    {
        static const char * env = std::getenv("SCLP_STOCHASTIC_ROUND");
        const bool enabled = !(env && env[0] == '0');
        if (enabled) {
            int drop_bits = (type == GGML_TYPE_SCLP4) ? 6
                          : (type == GGML_TYPE_SCLP6) ? 5
                          : 4; // SCLP8
            stochastic_mantissa_round(bf16_weights.data(), n, drop_bits);
        }
    }

    // 3. Collect exponent frequencies (and imatrix-weighted counts if available).
    // Imatrix-weighted: each weight contributes its column's imatrix importance
    // instead of 1.0. Causes k-means palette centers to cluster around exponent
    // bands that the actual activations care about, rather than just numeric
    // frequency. Stochastic rounding (step 2b above) makes this safe to use —
    // the previous attempt at imatrix-weighted palette regressed PPL 5× because
    // deterministic truncation converted reduced quant-error into systematic
    // bias; with stoch round the residual is zero-mean.
    uint32_t counts[256] = {0};
    float    wcounts[256] = {0.0f};
    // Imatrix-weighted palette is opt-in via SCLP_IMATRIX_PALETTE=1. Default is
    // raw-frequency k-means because a naive imatrix-weighted palette regressed
    // PPL 5× in earlier experiments — the palette centers move toward bands
    // that activate often but lose coverage of rare exponents that the sidecar
    // would otherwise rescue. Better to keep frequency-based here and let the
    // sidecar do imatrix-aware outlier promotion (see step 6).
    static const char * env_imp = std::getenv("SCLP_IMATRIX_PALETTE");
    const bool weight_with_imatrix = (env_imp && env_imp[0] == '1') && (imatrix != nullptr) && (K > 0);
    for (int64_t i = 0; i < n; i++) {
        uint8_t exp = (bf16_weights[i] >> 7) & 0xFF;
        counts[exp]++;
        wcounts[exp] += weight_with_imatrix ? imatrix[i % K] : 1.0f;
    }

    // 4. Select palette — all SCLP variants use k-means (formerly SCLP8 used
    // frequency + "all out-of-palette goes to sidecar", but the MoE GEMV kernel
    // omits sidecar correction for speed, which made distant-exponent weights
    // decode with catastrophic scale error on residual-feeder tensors. K-means
    // + "distance > 1" sidecar (below) keeps every weight within at most a 2×
    // scale error from its true value, making sidecar omission safe).
    kmeans_palette(wcounts, enc.palette, p_max);
    enc.palette_size = p_max;

    // 5. Build exponent-to-palette mapping and distances
    uint8_t exp_to_idx[256];
    uint8_t exp_distance[256];
    for (int e = 0; e < 256; e++) {
        int best_idx = 0;
        int min_diff = 1000;
        for (int i = 0; i < enc.palette_size; i++) {
            int diff = std::abs((int)e - (int)enc.palette[i]);
            if (diff < min_diff) {
                min_diff = diff;
                best_idx = i;
            }
        }
        exp_to_idx[e] = (uint8_t)best_idx;
        exp_distance[e] = (uint8_t)min_diff;
    }

    // 6. Encode weights and identify sidecar outliers
    std::vector<uint8_t> indices(n);
    std::vector<uint8_t> sm_nibbles(n);

    bool in_palette[256] = {false};
    for (int i = 0; i < enc.palette_size; i++) in_palette[enc.palette[i]] = true;

    bool has_imatrix = (imatrix != nullptr) && (sidecar_imatrix_budget > 0.0f) && (K > 0);
    std::vector<float> priority(n, 0.0f);

    for (int64_t i = 0; i < n; i++) {
        uint16_t w = bf16_weights[i];
        uint8_t exp = (w >> 7) & 0xFF;
        indices[i] = exp_to_idx[exp];

        uint8_t sign = (w >> 15) & 1;
        uint8_t mant;
        if (type == GGML_TYPE_SCLP8)       mant = (w >> 4) & 0x7;
        else if (type == GGML_TYPE_SCLP4) mant = (w >> 6) & 0x1;
        else                              mant = (w >> 5) & 0x3;

        sm_nibbles[i] = (sign << (sm_bits - 1)) | mant;

        if (!in_palette[exp]) {
            {
                // All SCLP types: only sidecar if distance > 1.
                // Distance=1 means exponent off by 1 — BF16 scale factor of 2x,
                // negligible error that's cheaper to keep than sidecar.
                if (exp_distance[exp] > 1) {
                    enc.sc_indices.push_back(expert_offset + (uint32_t)i);
                    enc.sc_values.push_back(w);
                } else if (has_imatrix) {
                    priority[i] = imatrix[(uint64_t)i % K] * (float)exp_distance[exp];
                }
            }
        } else if (has_imatrix && exp_distance[exp] > 0) {
            // In palette but distance > 0: candidate for imatrix sidecar
            priority[i] = imatrix[(uint64_t)i % K] * (float)exp_distance[exp];
        }
    }

    // 7. Discretionary imatrix sidecar
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
                enc.sc_values.push_back(bf16_weights[idx]);
            }
        }
    }

    // 8. Pack WS stream
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

    // Pack into final blob
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

    for (uint32_t e = 0; e < ne; e++) {
        std::memcpy(p, experts[e].ws_stream.data(), experts[e].ws_stream.size());
        p += experts[e].ws_stream.size();
    }

    // Merge sidecars
    std::vector<uint32_t> all_sc_indices;
    std::vector<uint16_t> all_sc_values;
    for (uint32_t e = 0; e < ne; e++) {
        all_sc_indices.insert(all_sc_indices.end(), experts[e].sc_indices.begin(), experts[e].sc_indices.end());
        all_sc_values.insert(all_sc_values.end(), experts[e].sc_values.begin(), experts[e].sc_values.end());
    }

    uint32_t sc_count = (uint32_t)all_sc_indices.size();
    std::memcpy(p, &sc_count, 4);
    p += 4;
    if (sc_count > 0) {
        std::memcpy(p, all_sc_indices.data(), sc_count * 4);
        p += sc_count * 4;
        std::memcpy(p, all_sc_values.data(), sc_count * 2);
        p += sc_count * 2;
    }

    return (size_t)(p - out);
}
