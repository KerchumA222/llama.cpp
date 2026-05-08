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

// Sidecar correction kernel for fused GEMV (M=1).
// After the main GEMV, sidecar weights were approximated with the nearest palette exponent.
// This kernel reads each sidecar entry, computes the delta vs the palette approximation,
// and atomically adds (delta * x[k]) to the output y[n].
// Grid: fixed small grid with stride loop (sidecar is tiny, ~0.01-0.03% of weights).
__global__ void sclp_sidecar_correct_gemv_kernel(
    const uint8_t*        __restrict__ blob,
    const __hip_bfloat16* __restrict__ x,   // BF16 activation [K]
    float*                             y,   // F32 output [N], updated atomically
    uint32_t N, uint32_t K
) {
    __shared__ uint8_t  s_palette_size;
    __shared__ uint32_t s_sidecar_count;
    __shared__ uint8_t  s_palette[16];

    if (threadIdx.x == 0) s_palette_size = blob[4];
    __syncthreads();
    if (threadIdx.x < (uint32_t)s_palette_size) s_palette[threadIdx.x] = blob[5 + threadIdx.x];
    __syncthreads();

    const uint8_t* packed       = blob + 5 + s_palette_size;
    const uint8_t* sm           = packed + ((uint64_t)(N * K + 1) / 2);
    const uint8_t* sidecar_base = sm + (uint64_t)N * K;

    if (threadIdx.x == 0) {
        uint32_t sc;
        __builtin_memcpy(&sc, sidecar_base, sizeof(uint32_t));
        s_sidecar_count = sc;
    }
    __syncthreads();

    if (s_sidecar_count == 0) return;

    const uint8_t* idx_base = sidecar_base + 4;
    const uint8_t* val_base = idx_base + (uint64_t)s_sidecar_count * sizeof(uint32_t);

    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < s_sidecar_count; i += stride) {
        uint32_t global_idx;
        uint16_t correct_bits;
        __builtin_memcpy(&global_idx, idx_base + (uint64_t)i * sizeof(uint32_t), sizeof(uint32_t));
        __builtin_memcpy(&correct_bits, val_base + (uint64_t)i * sizeof(uint16_t), sizeof(uint16_t));

        uint32_t n = global_idx / K;
        uint32_t k = global_idx % K;

        // Reconstruct the palette-approximated weight (what GEMV used).
        uint64_t w_idx = (uint64_t)n * K + k;
        uint8_t pb     = packed[w_idx >> 1];
        uint8_t p_idx  = (w_idx & 1) ? (pb & 0x0F) : (pb >> 4);
        uint8_t sm_val = sm[w_idx];
        uint16_t approx_bits = ((uint16_t)(sm_val >> 7) << 15)
                             | ((uint16_t)s_palette[p_idx] << 7)
                             | (sm_val & 0x7F);

        float w_correct = __bfloat162float(*(__hip_bfloat16*)&correct_bits);
        float w_approx  = __bfloat162float(*(__hip_bfloat16*)&approx_bits);
        float delta     = (w_correct - w_approx) * __bfloat162float(x[k]);

        atomicAdd(&y[n], delta);
    }
}

// Sidecar correction kernel for fused GEMM (M>1).
// Same as above, but applies correction for all M activation rows simultaneously.
__global__ void sclp_sidecar_correct_gemm_kernel(
    const uint8_t*        __restrict__ blob,
    const __hip_bfloat16* __restrict__ X,  // BF16 activations [M × K]
    float*                             Y,  // F32 output [M × N], updated atomically
    uint32_t N, uint32_t K, uint32_t M
) {
    __shared__ uint8_t  s_palette_size;
    __shared__ uint32_t s_sidecar_count;
    __shared__ uint8_t  s_palette[16];

    if (threadIdx.x == 0) s_palette_size = blob[4];
    __syncthreads();
    if (threadIdx.x < (uint32_t)s_palette_size) s_palette[threadIdx.x] = blob[5 + threadIdx.x];
    __syncthreads();

    const uint8_t* packed       = blob + 5 + s_palette_size;
    const uint8_t* sm           = packed + ((uint64_t)(N * K + 1) / 2);
    const uint8_t* sidecar_base = sm + (uint64_t)N * K;

    if (threadIdx.x == 0) {
        uint32_t sc;
        __builtin_memcpy(&sc, sidecar_base, sizeof(uint32_t));
        s_sidecar_count = sc;
    }
    __syncthreads();

    if (s_sidecar_count == 0) return;

    const uint8_t* idx_base = sidecar_base + 4;
    const uint8_t* val_base = idx_base + (uint64_t)s_sidecar_count * sizeof(uint32_t);

    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < s_sidecar_count; i += stride) {
        uint32_t global_idx;
        uint16_t correct_bits;
        __builtin_memcpy(&global_idx, idx_base + (uint64_t)i * sizeof(uint32_t), sizeof(uint32_t));
        __builtin_memcpy(&correct_bits, val_base + (uint64_t)i * sizeof(uint16_t), sizeof(uint16_t));

        uint32_t n = global_idx / K;
        uint32_t k = global_idx % K;

        uint64_t w_idx = (uint64_t)n * K + k;
        uint8_t pb     = packed[w_idx >> 1];
        uint8_t p_idx  = (w_idx & 1) ? (pb & 0x0F) : (pb >> 4);
        uint8_t sm_val = sm[w_idx];
        uint16_t approx_bits = ((uint16_t)(sm_val >> 7) << 15)
                             | ((uint16_t)s_palette[p_idx] << 7)
                             | (sm_val & 0x7F);

        float w_correct = __bfloat162float(*(__hip_bfloat16*)&correct_bits);
        float w_approx  = __bfloat162float(*(__hip_bfloat16*)&approx_bits);
        float w_delta   = w_correct - w_approx;

        for (uint32_t m = 0; m < M; m++) {
            float xval = __bfloat162float(X[(uint64_t)m * K + k]);
            atomicAdd(&Y[(uint64_t)m * N + n], w_delta * xval);
        }
    }
}

// Helper: convert F32 activations to BF16 into a pool buffer.
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

// Fused decode-GEMM: decode SCLP weights on-the-fly while accumulating the
// matrix product for M > 1 (prefill).  Eliminates the intermediate BF16
// weight buffer entirely, reading each weight once instead of twice.
//
// Each warp (32 threads) handles one weight row n; each block covers
// GEMM_WARPS_PER_BLOCK weight rows and GEMM_TILE_M activation rows.
// Weights are decoded once and multiplied against all TILE_M activation rows.
//
// Grid: (ceil(N/WARPS), ceil(M/TILE_M))
// Sidecar fixup is intentionally skipped — see §11 of experimental_results.md:
// sidecar weights are overwhelmingly near-zero outliers whose palette
// approximation acts as mild regularization (PPL neutral or slightly better).
constexpr int GEMM_TILE_M        = 4;
constexpr int GEMM_WARPS_PER_BLOCK = 8;

__launch_bounds__(GEMM_WARPS_PER_BLOCK * 32, 4)
__global__ void sclp_fused_gemm_kernel(
    const uint8_t*        __restrict__ blob,
    const __hip_bfloat16* __restrict__ X,   // [M × K] activations, row-major
    float*                __restrict__ Y,   // [M × N] output, row-major
    uint32_t N, uint32_t K, uint32_t M
) {
    __shared__ uint8_t s_palette_size;
    __shared__ uint8_t s_palette[16];

    if (threadIdx.x == 0) s_palette_size = blob[4];
    __syncthreads();
    if (threadIdx.x < (uint32_t)s_palette_size) s_palette[threadIdx.x] = blob[5 + threadIdx.x];
    __syncthreads();

    const uint8_t* packed = blob + 5 + s_palette_size;
    const uint8_t* sm     = packed + ((uint64_t)(N * K + 1) / 2);

    const int warp_id      = threadIdx.x / 32;
    const int lane         = threadIdx.x & 31;
    const uint32_t n       = (uint32_t)blockIdx.x * GEMM_WARPS_PER_BLOCK + warp_id;
    const uint32_t m_start = (uint32_t)blockIdx.y * GEMM_TILE_M;

    if (n >= N || m_start >= M) return;
    const int m_count = (int)min((uint32_t)GEMM_TILE_M, M - m_start);

    // Scalar accumulators — no array, no register spill.
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    const uint64_t row_base = (uint64_t)n * K;
    const uint32_t K8 = (K / 8) * 8;

    // Row pointers computed once outside the k loop.
    const __hip_bfloat16* x0 = X + (uint64_t)(m_start + 0) * K;
    const __hip_bfloat16* x1 = (m_count > 1) ? X + (uint64_t)(m_start + 1) * K : x0;
    const __hip_bfloat16* x2 = (m_count > 2) ? X + (uint64_t)(m_start + 2) * K : x0;
    const __hip_bfloat16* x3 = (m_count > 3) ? X + (uint64_t)(m_start + 3) * K : x0;

    // Vectorized loop: 8 weights per lane per iteration.
    // Inline decode (no wf[] array) to avoid scratch-memory spill.
    for (uint32_t k8 = (uint32_t)lane * 8; k8 < K8; k8 += 32 * 8) {
        uint64_t w_base = row_base + k8;
        uint32_t p4; __builtin_memcpy(&p4, packed + (w_base >> 1), sizeof(uint32_t));
        uint64_t sm8; __builtin_memcpy(&sm8, sm + w_base, sizeof(uint64_t));

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint8_t pb     = (uint8_t)(p4 >> ((j >> 1) * 8));
            uint8_t p_idx  = (j & 1) ? (pb & 0x0F) : (pb >> 4);
            uint8_t sm_val = (uint8_t)(sm8 >> (j * 8));
            uint16_t bits  = ((uint16_t)(sm_val >> 7) << 15)
                           | ((uint16_t)s_palette[p_idx] << 7)
                           | (sm_val & 0x7F);
            float w = __bfloat162float(*(__hip_bfloat16*)&bits);
                          a0 += w * __bfloat162float(x0[k8 + j]);
            if (m_count > 1) a1 += w * __bfloat162float(x1[k8 + j]);
            if (m_count > 2) a2 += w * __bfloat162float(x2[k8 + j]);
            if (m_count > 3) a3 += w * __bfloat162float(x3[k8 + j]);
        }
    }

    // Scalar tail for K not divisible by 8.
    for (uint32_t k = K8 + (uint32_t)lane; k < K; k += 32) {
        uint64_t w_idx = row_base + k;
        uint8_t pb     = packed[w_idx >> 1];
        uint8_t p_idx  = (w_idx & 1) ? (pb & 0x0F) : (pb >> 4);
        uint8_t sm_val = sm[w_idx];
        uint16_t bits  = ((uint16_t)(sm_val >> 7) << 15)
                       | ((uint16_t)s_palette[p_idx] << 7)
                       | (sm_val & 0x7F);
        float w = __bfloat162float(*(__hip_bfloat16*)&bits);
                      a0 += w * __bfloat162float(x0[k]);
        if (m_count > 1) a1 += w * __bfloat162float(x1[k]);
        if (m_count > 2) a2 += w * __bfloat162float(x2[k]);
        if (m_count > 3) a3 += w * __bfloat162float(x3[k]);
    }

    // Warp reduction: each accumulator is a scalar register — __shfl_down is safe.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        a0 += __shfl_down(a0, offset);
        a1 += __shfl_down(a1, offset);
        a2 += __shfl_down(a2, offset);
        a3 += __shfl_down(a3, offset);
    }
    if (lane == 0) {
        if (m_count > 0) Y[(uint64_t)(m_start + 0) * N + n] = a0;
        if (m_count > 1) Y[(uint64_t)(m_start + 1) * N + n] = a1;
        if (m_count > 2) Y[(uint64_t)(m_start + 2) * N + n] = a2;
        if (m_count > 3) Y[(uint64_t)(m_start + 3) * N + n] = a3;
    }
}

// Launch fused GEMM for M > 1 (prefill) inference case.
// src_f32: F32 activations [M × K], dst_f32: F32 output [M × N]
// tmp_bf16: caller-allocated [M × K] __hip_bfloat16 scratch
inline void llama_sclp_fused_gemm(
    const void*   blob_ptr,
    const float*  src_f32,
    float*        dst_f32,
    uint32_t N, uint32_t K, uint32_t M,
    void*         tmp_bf16,
    hipStream_t   stream
) {
    // Convert F32 activations → BF16.
    uint32_t mk = M * K;
    dim3 cvt_block(256);
    dim3 cvt_grid((mk + 255) / 256);
    f32_to_bf16_kernel<<<cvt_grid, cvt_block, 0, stream>>>(
        src_f32, (__hip_bfloat16*)tmp_bf16, mk);

    constexpr int WARPS   = GEMM_WARPS_PER_BLOCK;
    constexpr int TILE_M  = GEMM_TILE_M;
    dim3 block(WARPS * 32);
    dim3 grid(
        (N + WARPS  - 1) / WARPS,
        (M + TILE_M - 1) / TILE_M
    );
    sclp_fused_gemm_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)blob_ptr,
        (const __hip_bfloat16*)tmp_bf16,
        dst_f32, N, K, M);

    // Sidecar correction for all M token rows.
    sclp_sidecar_correct_gemm_kernel<<<4, 256, 0, stream>>>(
        (const uint8_t*)blob_ptr,
        (const __hip_bfloat16*)tmp_bf16,
        dst_f32, N, K, M);
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

    // Sidecar correction: atomically fix the small fraction of outlier weights
    // that were approximated by the nearest palette exponent in the GEMV above.
    sclp_sidecar_correct_gemv_kernel<<<4, 256, 0, stream>>>(
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
