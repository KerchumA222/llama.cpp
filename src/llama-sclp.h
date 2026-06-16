#pragma once

#include "ggml.h"
#include <vector>
#include <cstdint>

// Quantize a whole tensor (possibly with multiple experts) to SCLP format.
//
// nelements   = total weight count (ne[0] * ne[1] * ne[2])
// n_experts   = ne[2] (1 for dense, >1 for MoE)
// K           = ne[0] (column / input dimension) — needed for imatrix stride
// imatrix     = per-column importance, [n_experts][K] flattened (or nullptr)
// clip_threshold: 0 = no soft clipping, 125 = recommended default for BF16.
// sidecar_imatrix_budget: fraction of weights to add to sidecar via imatrix
//   ranking (0.0 = disabled, 0.01 = recommended default when imatrix is present).
// Returns the total size of the quantized blob.
size_t llama_tensor_quantize_sclp(
    ggml_type type,
    const float * f32_data,
    void * new_data,
    int64_t nelements,
    int64_t n_experts,
    int64_t K,
    const float * imatrix,
    uint8_t  clip_threshold = 0,
    float    sidecar_imatrix_budget = 0.01f
);
