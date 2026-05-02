#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
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

// Fused decode-GEMV: decode SCLP weights on-the-fly while accumulating the
// dot product.  Eliminates the intermediate BF16 weight buffer entirely.
// x must be BF16 [K], y is written as F32 [N].
// Does NOT apply sidecar fixup — see sclp_fused_gemv_kernel docs.
//
// Optimization: vectorized loads — each thread loads 8 sm bytes (uint64_t)
// and 4 packed bytes (uint32_t) per iteration, processing 8 weights at once.
// This reduces load instructions 8× vs the scalar byte-load baseline.
// __launch_bounds__(512, 4): max 512 threads/block, 4 blocks/CU target
// gfx1100: 48 CUs × 4 blocks = 192 blocks, each 512 threads = 16 warps → good occupancy
__launch_bounds__(512, 4)
__global__ void sclp_fused_gemv_kernel(
    const uint8_t*        __restrict__ blob,
    const __hip_bfloat16* __restrict__ x,
    float*                __restrict__ y,
    uint32_t N,
    uint32_t K
) {
    __shared__ uint8_t s_palette_size;
    __shared__ uint8_t s_palette[16];

    if (threadIdx.x == 0) {
        s_palette_size = blob[4];
    }
    __syncthreads();

    if (threadIdx.x < (uint32_t)s_palette_size) {
        s_palette[threadIdx.x] = blob[5 + threadIdx.x];
    }
    __syncthreads();

    const uint8_t* packed = blob + 5 + s_palette_size;
    const uint8_t* sm     = packed + ((uint64_t)(N * K + 1) / 2);

    const int warps_per_block = blockDim.x / 32;
    const int warp_id  = threadIdx.x / 32;
    const int lane     = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    if (row >= N) return;

    // Two accumulators for ILP — lane processes two interleaved chunks.
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint64_t row_base = (uint64_t)row * K;

    // --- Vectorized loop: process 16 weights per iteration (2×8), dual accumulator ---
    // Stride = 32*16 weights per round so each lane handles two independent 8-weight chunks
    const uint32_t K16 = (K / 16) * 16;

    for (uint32_t k16 = lane * 16; k16 < K16; k16 += 32 * 16) {
        // Chunk A: weights [k16 .. k16+7]
        {
            uint64_t w_base = row_base + k16;
            uint32_t p4; __builtin_memcpy(&p4, packed + (w_base >> 1), sizeof(uint32_t));
            uint64_t sm8; __builtin_memcpy(&sm8, sm + w_base, sizeof(uint64_t));
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint8_t pb    = (uint8_t)(p4  >> ((j >> 1) * 8));
                uint8_t p_idx = (j & 1) ? (pb & 0x0F) : (pb >> 4);
                uint8_t sm_val = (uint8_t)(sm8 >> (j * 8));
                uint16_t bits = ((uint16_t)(sm_val >> 7) << 15)
                              | ((uint16_t)s_palette[p_idx] << 7)
                              | (sm_val & 0x7F);
                acc0 += __bfloat162float(*(__hip_bfloat16*)&bits)
                      * __bfloat162float(x[k16 + j]);
            }
        }
        // Chunk B: weights [k16+8 .. k16+15]
        {
            uint64_t w_base = row_base + k16 + 8;
            uint32_t p4; __builtin_memcpy(&p4, packed + (w_base >> 1), sizeof(uint32_t));
            uint64_t sm8; __builtin_memcpy(&sm8, sm + w_base, sizeof(uint64_t));
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint8_t pb    = (uint8_t)(p4  >> ((j >> 1) * 8));
                uint8_t p_idx = (j & 1) ? (pb & 0x0F) : (pb >> 4);
                uint8_t sm_val = (uint8_t)(sm8 >> (j * 8));
                uint16_t bits = ((uint16_t)(sm_val >> 7) << 15)
                              | ((uint16_t)s_palette[p_idx] << 7)
                              | (sm_val & 0x7F);
                acc1 += __bfloat162float(*(__hip_bfloat16*)&bits)
                      * __bfloat162float(x[k16 + 8 + j]);
            }
        }
    }

    float acc = acc0 + acc1;

    // --- Scalar tail: handle remaining weights (k in [K16, K)) ---
    for (uint32_t k = K16 + lane; k < K; k += 32) {
        uint64_t w_idx = row_base + k;

        uint8_t packed_byte = packed[w_idx >> 1];
        uint8_t p_idx  = (w_idx & 1) ? (packed_byte & 0x0F) : (packed_byte >> 4);
        uint8_t sm_val = sm[w_idx];

        uint16_t bits = ((uint16_t)(sm_val >> 7) << 15)
                      | ((uint16_t)s_palette[p_idx] << 7)
                      | (sm_val & 0x7F);

        acc += __bfloat162float(*(__hip_bfloat16*)&bits)
             * __bfloat162float(x[k]);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }

    if (lane == 0) {
        y[row] = acc;
    }
}

// Helper: convert F32 activations to BF16 in-place into a pool buffer.
__global__ void f32_to_bf16_kernel(
    const float*    __restrict__ src,
    __hip_bfloat16* __restrict__ dst,
    uint32_t n
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

// Launch fused GEMV for the M=1 inference case.
// src_f32: F32 activation vector [K] (ggml standard for activations)
// blob:    SCLP weight blob [N×K] device pointer
// dst_f32: F32 output [N]
inline void llama_sclp_fused_gemv(
    const void*   blob_ptr,
    const float*  src_f32,
    float*        dst_f32,
    uint32_t      N,
    uint32_t      K,
    void*         tmp_bf16,    // caller-allocated [K] __hip_bfloat16 scratch
    hipStream_t   stream
) {
    // Convert F32 activations → BF16 (the fused kernel expects BF16 input).
    dim3 cvt_block(256);
    dim3 cvt_grid((K + 255) / 256);
    f32_to_bf16_kernel<<<cvt_grid, cvt_block, 0, stream>>>(
        src_f32, (__hip_bfloat16*)tmp_bf16, K);

    // Fused decode-GEMV: 16 warps per block → 16 output rows per block.
    constexpr int WARPS_PER_BLOCK = 16;
    dim3 gemv_block(WARPS_PER_BLOCK * 32);
    dim3 gemv_grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sclp_fused_gemv_kernel<<<gemv_grid, gemv_block, 0, stream>>>(
        (const uint8_t*)blob_ptr,
        (const __hip_bfloat16*)tmp_bf16,
        dst_f32, N, K);
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
