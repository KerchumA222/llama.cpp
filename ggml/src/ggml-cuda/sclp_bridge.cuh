#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// SCLP decode bridge for llama.cpp HIP backend.
//
// Wire format (SCLP blob stored in VRAM):
//   [uint32 num_weights][uint8 palette_size][palette (palette_size bytes)]
//   [packed_indices (ceil(num_weights/2) bytes)][sm_stream (num_weights bytes)]
//   [uint32 sidecar_count]
//   [uint32 × sidecar_count sidecar_indices]
//   [uint16 × sidecar_count sidecar_values]
//   [zero padding to fill num_weights*2 bytes total]
//
// Weights whose exponent falls outside the top-16 palette are stored verbatim in
// the sidecar section and restored exactly by sclp_fixup_sidecar_kernel.
//
// Both kernels read all header/count fields on-device — no D2H memcpy needed.
// Safe during HIP stream capture (GGML_HIP_GRAPHS=ON).

__global__ void sclp_decode_blob_kernel(
    const uint8_t* __restrict__ blob,
    uint16_t*      __restrict__ output,
    uint32_t                    num_weights
) {
    __shared__ uint8_t palette_size_s;
    __shared__ uint8_t s_palette[16];

    if (threadIdx.x == 0) {
        palette_size_s = blob[4];
    }
    __syncthreads();

    if (threadIdx.x < palette_size_s) {
        s_palette[threadIdx.x] = blob[5 + threadIdx.x];
    }
    __syncthreads();

    const uint8_t* packed = blob + 5 + palette_size_s;
    const uint8_t* sm     = packed + ((num_weights + 1) / 2);

    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t base_idx = pair_idx << 1;

    if (base_idx >= num_weights) return;

    uint8_t packed_byte = packed[pair_idx];
    uint8_t exp0 = s_palette[packed_byte >> 4];
    uint8_t exp1 = s_palette[packed_byte & 0x0F];

    uint8_t sm0 = sm[base_idx];
    output[base_idx] = ((uint16_t)(sm0 >> 7) << 15) | ((uint16_t)exp0 << 7) | (sm0 & 0x7F);

    if (base_idx + 1 < num_weights) {
        uint8_t sm1 = sm[base_idx + 1];
        output[base_idx + 1] = ((uint16_t)(sm1 >> 7) << 15) | ((uint16_t)exp1 << 7) | (sm1 & 0x7F);
    }
}

// Restore outlier weights written verbatim in the sidecar section.
// Must run after sclp_decode_blob_kernel on the same output buffer.
// Uses a grid-stride loop so a fixed small grid handles any sidecar count.
__global__ void sclp_fixup_sidecar_kernel(
    const uint8_t* __restrict__ blob,
    uint16_t*      __restrict__ output,
    uint32_t                    num_weights
) {
    __shared__ uint8_t  palette_size_s;
    __shared__ uint32_t sidecar_count_s;

    if (threadIdx.x == 0) {
        palette_size_s = blob[4];
    }
    __syncthreads();

    const uint8_t* packed       = blob + 5 + palette_size_s;
    const uint8_t* sm           = packed + ((num_weights + 1) / 2);
    const uint8_t* sidecar_base = sm + num_weights;

    if (threadIdx.x == 0) {
        uint32_t sc;
        __builtin_memcpy(&sc, sidecar_base, sizeof(uint32_t));
        sidecar_count_s = sc;
    }
    __syncthreads();

    if (sidecar_count_s == 0) return;

    // Sidecar layout: [uint32 × count indices][uint16 × count values]
    const uint8_t* idx_base = sidecar_base + 4;
    const uint8_t* val_base = sidecar_base + 4 + (uint64_t)sidecar_count_s * sizeof(uint32_t);

    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < sidecar_count_s;
         i += stride) {
        uint32_t idx;
        uint16_t val;
        __builtin_memcpy(&idx, idx_base + (uint64_t)i * sizeof(uint32_t), sizeof(uint32_t));
        __builtin_memcpy(&val, val_base + (uint64_t)i * sizeof(uint16_t), sizeof(uint16_t));
        output[idx] = val;
    }
}

// Decode an SCLP blob (device pointer) into a flat BF16 uint16_t buffer.
// num_weights must equal ggml_nelements(src0).
// No host-side device reads; safe during HIP stream capture.
inline void llama_sclp_dispatch(
    const void* sclp_data,
    uint16_t*   output,
    uint32_t    num_weights,
    hipStream_t stream
) {
    const uint8_t* data = (const uint8_t*)sclp_data;
    dim3 block(256);

    // Main decode: one thread per pair of weights
    uint32_t pairs = (num_weights + 1) / 2;
    dim3 decode_grid((pairs + 255) / 256);
    sclp_decode_blob_kernel<<<decode_grid, block, 0, stream>>>(data, output, num_weights);

    // Sidecar fixup: fixed 4-block grid with stride loop handles any sidecar count.
    // Threads where i >= sidecar_count return after a single shared-memory read.
    sclp_fixup_sidecar_kernel<<<4, block, 0, stream>>>(data, output, num_weights);
}
