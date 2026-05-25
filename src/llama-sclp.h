#pragma once

#include "ggml.h"
#include <vector>
#include <cstdint>

#define QK_SCLP 32
#define QK_SCLP4 256

static inline int qk_for_type(ggml_type type) {
    return (type == GGML_TYPE_SCLP4 || type == GGML_TYPE_SCLP5) ? QK_SCLP4 : QK_SCLP;
}

// Quantize a whole tensor (possibly with multiple experts) to SCLP format.
//
// nelements   = total weight count (ne[0] * ne[1] * ne[2])
// n_experts   = ne[2] (1 for dense, >1 for MoE)
// K           = ne[0] (column / input dimension) — needed for imatrix stride
// imatrix     = per-column importance, [n_experts][K] flattened (or nullptr)
// clip_threshold: 0 = no soft clipping, 125 = recommended default for BF16.
// sidecar_imatrix_budget: fraction of weights to add to sidecar via imatrix
//   ranking (0.0 = disabled, 0.01 = recommended default when imatrix is present).
//
// Layout (SCLP6/SCLP8): [header][scales][ws_stream][sidecar]
//   header: [uint32 num_weights][uint32 n_experts][per-expert: palette_size, palette]
//   scales: [uint16 scale_bf16] x (num_weights / QK_SCLP)
//   ws_stream: packed indices + SM bits
//   sidecar: [uint32 count][uint32 indices][uint16 bf16_bits]
//
// Layout (SCLP4, per-block palette): [header][block_palettes][ws_stream][sidecar]
//   header: [uint32 num_weights][uint32 n_experts][per-expert: palette_size=0]
//   block_palettes: [4 x uint8] x (num_weights / QK_SCLP4)
//   ws_stream: packed indices + SM bits
//   sidecar: [uint32 count][uint32 indices][uint16 bf16_bits]
//
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

// Dequantize a SCLP blob back to F32.
// Mostly for testing and CPU fallback.
void llama_tensor_dequantize_sclp(
    ggml_type type,
    const void * data,
    float * f32_data,
    int64_t nelements
);
