#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

// SCLP decode bridge for llama.cpp HIP backend.
//
// Wire format (SCLP blob stored in VRAM):
//   [uint32 num_weights][uint8 palette_size][palette (palette_size bytes)]
//   [packed_indices (ceil(num_weights/2) bytes)][sm_stream (num_weights bytes)]
//   [zero padding to fill num_weights*2 bytes total]
//
// The kernel reads the header entirely on-device so no D2H memcpy is needed.
// This makes it compatible with HIP stream capture (graph mode).

__global__ void sclp_decode_blob_kernel(
    const uint8_t* __restrict__ blob,
    uint16_t*      __restrict__ output,
    uint32_t                    num_weights
) {
    // Block 0 thread 0 loads palette_size; all threads in the block share it.
    __shared__ uint8_t palette_size_s;
    __shared__ uint8_t s_palette[16];

    if (threadIdx.x == 0) {
        // palette_size is byte 4 of the blob (after the uint32 num_weights)
        palette_size_s = blob[4];
    }
    __syncthreads();

    if (threadIdx.x < palette_size_s) {
        s_palette[threadIdx.x] = blob[5 + threadIdx.x];
    }
    __syncthreads();

    // Pointers into the blob — identical for every thread in this block
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

// Decode an SCLP blob (device pointer) into a flat BF16 uint16_t buffer.
// num_weights must equal ggml_nelements(src0) — the caller owns this.
// No host-side device reads; safe during HIP stream capture.
inline void llama_sclp_dispatch(
    const void* sclp_data,
    uint16_t*   output,
    uint32_t    num_weights,
    hipStream_t stream
) {
    const uint8_t* data = (const uint8_t*)sclp_data;
    uint32_t threads_needed = (num_weights + 1) / 2;
    dim3 block(256);
    dim3 grid((threads_needed + 255) / 256);
    sclp_decode_blob_kernel<<<grid, block, 0, stream>>>(data, output, num_weights);
}
