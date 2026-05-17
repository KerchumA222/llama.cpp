#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#ifndef SCLP_MEMSET_STUB
#define SCLP_MEMSET_STUB 0
#endif

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

// Each thread decodes 8 weights from 8 ws bytes → 8 uint16_t outputs.
// ws_stream: one byte per weight — idx(7:4) | smn(3:0)
//   idx = palette index (4 bits), smn = sign(3) | mantissa_top3(2:0)
// Single uint64_t load covers 8 consecutive weights, improving cache locality
// vs separate packed+SM arrays that were ceil(N*K/2) bytes apart.
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

    const uint8_t* ws = blob + 5 + palette_size_s;  // ws_stream: N bytes

    // Each thread handles 8 weights via a single uint64_t load
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t base_idx   = thread_idx * 8;

    if (base_idx >= num_weights) return;

    uint64_t ws8 = 0;
    uint32_t remaining = num_weights - base_idx;
    __builtin_memcpy(&ws8, ws + base_idx, min(8u, remaining));

    uint16_t out[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint8_t b   = (uint8_t)(ws8 >> (i * 8));
        uint8_t exp = s_palette[b >> 4];
        uint8_t smn = b & 0x0F;
        out[i] = ((uint16_t)(smn >> 3) << 15) | ((uint16_t)exp << 7) | ((uint16_t)(smn & 0x7) << 4);
    }

    // Write outputs; use 128-bit store when all 8 fit
    if (remaining >= 8) {
        uint4 v;
        __builtin_memcpy(&v, out, sizeof(uint4));
        *reinterpret_cast<uint4*>(output + base_idx) = v;
    } else {
        for (uint32_t i = 0; i < remaining; ++i)
            output[base_idx + i] = out[i];
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

    const uint8_t* ws           = blob + 5 + palette_size_s;
    const uint8_t* sidecar_base = ws + num_weights;  // ws_stream is N bytes (1 per weight)

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

// Fused decode-GEMV with inline sidecar correction.
// Eliminates the intermediate BF16 weight buffer and the separate sidecar kernel.
//
// Each warp handles one output row n (16 warps/block → 16 rows/block).
// After the main dot-product reduction, all threads in the block collaborate
// to scan the sidecar and atomically correct any outlier weights whose row
// falls within [block_n_start, block_n_end). This avoids a separate kernel
// launch while keeping all corrections in the same GPU invocation.
//
// Phase 1 (sync 1): load palette_size.
// Phase 2 (sync 2): load palette entries + sidecar_count in parallel.
// Phase 3 (main GEMV, conditional on row < N).
// Phase 4 (sync 3): wait for all y[n] writes in this block.
// Phase 5 (sidecar scan): stride loop, skip entries outside our row range.
// Accepts F32 activations directly — no separate f32_to_bf16_kernel needed.
// Saves one kernel launch + one BF16 scratch buffer per layer (224 launches/token).
__launch_bounds__(1024, 2)
__global__ void sclp_fused_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ x,   // F32 activations [K]
    float*         __restrict__ y,
    uint32_t N,
    uint32_t K
) {
    __shared__ uint8_t s_palette_size;
    __shared__ uint8_t s_palette[16];

    if (threadIdx.x == 0) s_palette_size = blob[4];
    __syncthreads();

    if (threadIdx.x < (uint32_t)s_palette_size)
        s_palette[threadIdx.x] = blob[5 + threadIdx.x];
    __syncthreads();

    const uint8_t* ws = blob + 5 + s_palette_size;  // ws_stream: N*K bytes, 1 per weight
    const int warps_per_block = blockDim.x / 32;
    const int warp_id  = threadIdx.x / 32;
    const int lane     = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    if (row < N) {
        float acc0 = 0.0f, acc1 = 0.0f;
        const uint64_t row_base = (uint64_t)row * K;
        const uint32_t K16 = (K / 16) * 16;

        for (uint32_t k16 = lane * 16; k16 < K16; k16 += 32 * 16) {
            {
                // 8 weights at row_base+k16: one uint64_t covers all 8 ws bytes
                uint64_t ws8; __builtin_memcpy(&ws8, ws + row_base + k16, sizeof(uint64_t));
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    uint8_t b     = (uint8_t)(ws8 >> (j * 8));
                    uint8_t smn   = b & 0x0F;
                    uint16_t bits = ((uint16_t)(smn >> 3) << 15)
                                  | ((uint16_t)s_palette[b >> 4] << 7)
                                  | ((uint16_t)(smn & 0x7) << 4);
                    acc0 += __bfloat162float(*(__hip_bfloat16*)&bits) * x[k16 + j];
                }
            }
            {
                uint64_t ws8; __builtin_memcpy(&ws8, ws + row_base + k16 + 8, sizeof(uint64_t));
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    uint8_t b     = (uint8_t)(ws8 >> (j * 8));
                    uint8_t smn   = b & 0x0F;
                    uint16_t bits = ((uint16_t)(smn >> 3) << 15)
                                  | ((uint16_t)s_palette[b >> 4] << 7)
                                  | ((uint16_t)(smn & 0x7) << 4);
                    acc1 += __bfloat162float(*(__hip_bfloat16*)&bits) * x[k16 + 8 + j];
                }
            }
        }
        float acc = acc0 + acc1;
        for (uint32_t k = K16 + lane; k < K; k += 32) {
            uint8_t b     = ws[row_base + k];
            uint8_t smn   = b & 0x0F;
            uint16_t bits = ((uint16_t)(smn >> 3) << 15)
                          | ((uint16_t)s_palette[b >> 4] << 7)
                          | ((uint16_t)(smn & 0x7) << 4);
            acc += __bfloat162float(*(__hip_bfloat16*)&bits) * x[k];
        }
        for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
        if (lane == 0) y[row] = acc;
    }
    // Sidecar correction is intentionally omitted for M=1 decode: the per-element error
    // from the palette approximation (~0.02% of weights) is tolerable for single-token
    // generation against a correctly-prefilled KV cache. A block-scoped scan would force
    // every block to read all sidecar entries (256× more reads than a separate 4-block
    // kernel), causing ~37% throughput regression.
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

    const uint8_t* ws           = blob + 5 + s_palette_size;
    const uint8_t* sidecar_base = ws + (uint64_t)N * K;  // ws_stream is N*K bytes

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

        uint8_t b     = ws[(uint64_t)n * K + k];
        uint8_t smn   = b & 0x0F;
        uint16_t approx_bits = ((uint16_t)(smn >> 3) << 15)
                             | ((uint16_t)s_palette[b >> 4] << 7)
                             | ((uint16_t)(smn & 0x7) << 4);

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
constexpr int GEMM_TILE_M        = 16;
constexpr int GEMM_WARPS_PER_BLOCK = 8;

template <int TILE_M>
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

    const uint8_t* ws = blob + 5 + s_palette_size;  // ws_stream: N*K bytes, 1 per weight

    const int warp_id      = threadIdx.x / 32;
    const int lane         = threadIdx.x & 31;
    const uint32_t n       = (uint32_t)blockIdx.x * GEMM_WARPS_PER_BLOCK + warp_id;
    const uint32_t m_start = (uint32_t)blockIdx.y * TILE_M;

    if (n >= N || m_start >= M) return;
    const int m_count = (int)min((uint32_t)TILE_M, M - m_start);

    // Compile-time-sized accumulator array — all VGPRs, no spill.
    float acc[TILE_M];
    #pragma unroll TILE_M
    for (int i = 0; i < TILE_M; i++) acc[i] = 0.0f;

    const uint64_t row_base = (uint64_t)n * K;
    const uint32_t K8 = (K / 8) * 8;

    // Vectorized loop: 8 weights per lane per iteration.
    // Single uint64_t covers 8 ws bytes (idx+smn co-located per byte).
    for (uint32_t k8 = (uint32_t)lane * 8; k8 < K8; k8 += 32 * 8) {
        uint64_t ws8; __builtin_memcpy(&ws8, ws + row_base + k8, sizeof(uint64_t));

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint8_t b     = (uint8_t)(ws8 >> (j * 8));
            uint8_t smn   = b & 0x0F;
            uint16_t bits = ((uint16_t)(smn >> 3) << 15)
                          | ((uint16_t)s_palette[b >> 4] << 7)
                          | ((uint16_t)(smn & 0x7) << 4);
            float w = __bfloat162float(*(__hip_bfloat16*)&bits);
            #pragma unroll TILE_M
            for (int mi = 0; mi < TILE_M; mi++) {
                if (mi < m_count) acc[mi] += w * __bfloat162float(X[(uint64_t)(m_start + mi) * K + k8 + j]);
            }
        }
    }

    // Scalar tail for K not divisible by 8.
    for (uint32_t k = K8 + (uint32_t)lane; k < K; k += 32) {
        uint8_t b     = ws[row_base + k];
        uint8_t smn   = b & 0x0F;
        uint16_t bits = ((uint16_t)(smn >> 3) << 15)
                      | ((uint16_t)s_palette[b >> 4] << 7)
                      | ((uint16_t)(smn & 0x7) << 4);
        float w = __bfloat162float(*(__hip_bfloat16*)&bits);
        #pragma unroll TILE_M
        for (int mi = 0; mi < TILE_M; mi++) {
            if (mi < m_count) acc[mi] += w * __bfloat162float(X[(uint64_t)(m_start + mi) * K + k]);
        }
    }

    // Warp reduction.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll TILE_M
        for (int mi = 0; mi < TILE_M; mi++) acc[mi] += __shfl_down(acc[mi], offset);
    }
    if (lane == 0) {
        #pragma unroll TILE_M
        for (int mi = 0; mi < TILE_M; mi++) {
            if (mi < m_count) Y[(uint64_t)(m_start + mi) * N + n] = acc[mi];
        }
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
    sclp_fused_gemm_kernel<TILE_M><<<grid, block, 0, stream>>>(
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
// Accepts F32 activations directly — no separate conversion kernel or scratch buffer.
inline void llama_sclp_fused_gemv(
    const void*   blob_ptr,
    const float*  src_f32,
    float*        dst_f32,
    uint32_t      N,
    uint32_t      K,
    hipStream_t   stream
) {
    constexpr int WARPS_PER_BLOCK = 32;
    dim3 gemv_block(WARPS_PER_BLOCK * 32);
    dim3 gemv_grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sclp_fused_gemv_kernel<<<gemv_grid, gemv_block, 0, stream>>>(
        (const uint8_t*)blob_ptr,
        src_f32,
        dst_f32, N, K);
}

// Fused decode-GEMV for MoE MUL_MAT_ID with SCLP expert weights.
// src0 blob: [K, N, n_experts] stacked experts encoded as one SCLP blob.
// src1: F32 [K, n_active] routed activations (one column per active expert).
// ids:  int32 [n_active] expert routing indices (which expert weight to use).
// dst:  F32 [N, n_active] output.
// Sidecar correction is omitted (same rationale as sclp_fused_gemv_kernel).
__launch_bounds__(1024, 2)
__global__ void sclp_fused_moe_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ src1,   // [K × n_active × n_batches]
    const int32_t* __restrict__ ids,    // [n_active × n_batches]
    float*         __restrict__ dst,    // [N × n_active × n_batches]
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t src1_ne1
) {
    __shared__ uint8_t s_palette_size;
    __shared__ uint8_t s_palette[16];

    if (threadIdx.x == 0) s_palette_size = blob[4];
    __syncthreads();
    if (threadIdx.x < (uint32_t)s_palette_size) s_palette[threadIdx.x] = blob[5 + threadIdx.x];
    __syncthreads();

    const uint8_t* ws = blob + 5 + s_palette_size;

    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;

    // blockIdx.x = row group, blockIdx.y = flat index (i_active + i_batch * n_active)
    const uint32_t row    = (uint32_t)blockIdx.x * warps_per_block + warp_id;
    const uint32_t flat   = blockIdx.y;  // flat (i_active, i_batch) index

    if (row >= N) return;

    const uint32_t i_active = flat % n_active;
    const uint32_t i_batch  = flat / n_active;

    const int32_t e = ids[flat];
    // ws index for weight (k, row, e): k + row*K + e*N*K
    const uint64_t ws_base = (uint64_t)e * N * K + (uint64_t)row * K;
    const float* x = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;

    float acc0 = 0.0f, acc1 = 0.0f;
    const uint32_t K16 = (K / 16) * 16;

    for (uint32_t k16 = lane * 16; k16 < K16; k16 += 32 * 16) {
        {
            uint64_t ws8; __builtin_memcpy(&ws8, ws + ws_base + k16, sizeof(uint64_t));
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint8_t b     = (uint8_t)(ws8 >> (j * 8));
                uint8_t smn   = b & 0x0F;
                uint16_t bits = ((uint16_t)(smn >> 3) << 15)
                              | ((uint16_t)s_palette[b >> 4] << 7)
                              | ((uint16_t)(smn & 0x7) << 4);
                acc0 += __bfloat162float(*(__hip_bfloat16*)&bits) * x[k16 + j];
            }
        }
        {
            uint64_t ws8; __builtin_memcpy(&ws8, ws + ws_base + k16 + 8, sizeof(uint64_t));
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint8_t b     = (uint8_t)(ws8 >> (j * 8));
                uint8_t smn   = b & 0x0F;
                uint16_t bits = ((uint16_t)(smn >> 3) << 15)
                              | ((uint16_t)s_palette[b >> 4] << 7)
                              | ((uint16_t)(smn & 0x7) << 4);
                acc1 += __bfloat162float(*(__hip_bfloat16*)&bits) * x[k16 + 8 + j];
            }
        }
    }
    float acc = acc0 + acc1;
    for (uint32_t k = K16 + lane; k < K; k += 32) {
        uint8_t b     = ws[ws_base + k];
        uint8_t smn   = b & 0x0F;
        uint16_t bits = ((uint16_t)(smn >> 3) << 15)
                      | ((uint16_t)s_palette[b >> 4] << 7)
                      | ((uint16_t)(smn & 0x7) << 4);
        acc += __bfloat162float(*(__hip_bfloat16*)&bits) * x[k];
    }
    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) dst[row + (uint64_t)flat * N] = acc;
}

// Launch fused MoE GEMV for SCLP expert weights (MUL_MAT_ID, any n_batches).
// Processes each (row, i_active, i_batch) output element independently.
// src1: F32 [K × n_active × n_batches], ids: int32 [n_active × n_batches],
// dst: F32 [N × n_active × n_batches].
inline void llama_sclp_fused_moe_gemv(
    const void*    blob_ptr,
    const float*   src1,       // F32 [K × n_active × n_batches]
    const int32_t* ids,        // int32 [n_active × n_batches]
    float*         dst,        // F32 [N × n_active × n_batches]
    uint32_t       N,          // output rows per expert
    uint32_t       K,          // input dim
    uint32_t       n_active,   // active experts per token
    uint32_t       n_batches,  // number of token batches
    uint32_t       src1_ne1,   // number of columns in src1 per batch (1 or n_active)
    hipStream_t    stream
) {
    constexpr int WARPS_PER_BLOCK = 32;
    dim3 block(WARPS_PER_BLOCK * 32);
    // blockIdx.y = i_active + i_batch * n_active  (flat expert-batch index)
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, n_active * n_batches);
    sclp_fused_moe_gemv_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)blob_ptr,
        src1, ids, dst, N, K, n_active, src1_ne1);
}

// Fused decode-WMMA kernel for M > 1 prefill using RDNA3 wave32 WMMA instructions.
// Eliminates the intermediate BF16 weight buffer entirely — weights are decoded
// from the SCLP blob on-the-fly and fed directly into the WMMA A fragment.
//
// RDNA3 WMMA fragment layout (DATA_LAYOUT_I_MAJOR_MIRRORED, wave32):
//   A/B: 16 BF16 per thread, all from row (threadIdx.x % 16), columns 0..15.
//        Both lanes t and t+16 hold the same row (mirrored).
//   C/D:  8 F32 per thread. Thread t holds D[t%16][2*l + t/16] for l=0..7.
//         → lanes 0..15 get even columns, lanes 16..31 get odd columns.
//
// Block: 4 warps (2 N-tiles × 2 M-tiles of 16×16 each → 32N × 32M per block).
// LDS: 16×33 BF16 transposed X tile (512 B, +1 column for bank-conflict avoidance).
// Grid: (ceil(N/32), ceil(M/32)).
//
// Sidecar correction is intentionally omitted — see GEMV comment for rationale.
// The separate sclp_sidecar_correct_gemm_kernel handles it after this kernel returns.
constexpr int WMMA_WARPS_N = 2;
constexpr int WMMA_WARPS_M = 2;

__launch_bounds__((WMMA_WARPS_N * WMMA_WARPS_M) * 32, 4)
__global__ void sclp_fused_wmma_kernel(
    const uint8_t*        __restrict__ blob,
    const __hip_bfloat16* __restrict__ X,   // [M × K] activations, row-major
    float*                __restrict__ Y,   // [M × N] output, row-major
    uint32_t N, uint32_t K, uint32_t M
) {
    constexpr int WMMA_TILE = 16;
    constexpr int BLOCK_N   = WMMA_WARPS_N * WMMA_TILE;  // 32
    constexpr int BLOCK_M   = WMMA_WARPS_M * WMMA_TILE;  // 32

    __shared__ uint8_t s_palette_size;
    __shared__ uint8_t s_palette[16];
    // Transposed activation tile: s_XT[k_local][m_local] = X[block_m+m_local][k16+k_local]
    // +1 column padding avoids LDS bank conflicts on the 32-bank RDNA3 layout.
    __shared__ __hip_bfloat16 s_XT[WMMA_TILE][BLOCK_M + 1];

    if (threadIdx.x == 0) s_palette_size = blob[4];
    __syncthreads();
    if (threadIdx.x < (uint32_t)s_palette_size) s_palette[threadIdx.x] = blob[5 + threadIdx.x];
    __syncthreads();

    const uint8_t* packed = blob + 5 + s_palette_size;
    const uint8_t* sm     = packed + ((uint64_t)(N * K + 1) / 2);

    const int warp_id  = threadIdx.x / 32;
    const int lane     = threadIdx.x & 31;
    const int frag_row = lane % WMMA_TILE;   // row index within WMMA fragment (0..15)

    // 2×2 warp tile: warp_n selects N-tile, warp_m selects M-tile.
    const int warp_n = warp_id % WMMA_WARPS_N;
    const int warp_m = warp_id / WMMA_WARPS_N;

    const uint32_t n_base   = ((uint32_t)blockIdx.x * WMMA_WARPS_N + warp_n) * WMMA_TILE;
    const uint32_t block_m  = (uint32_t)blockIdx.y * BLOCK_M;
    const uint32_t m_base   = block_m + warp_m * WMMA_TILE;  // global M base for this warp

    // WMMA fragment types for RDNA3 bf16 input, f32 accumulator.
    using bf16x16_t = __attribute__((ext_vector_type(16))) __bf16;
    using floatx8_t = __attribute__((ext_vector_type(8)))  float;

    floatx8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (uint32_t k16 = 0; k16 < K; k16 += WMMA_TILE) {
        // ── Load X tile cooperatively into LDS ───────────────────────────────
        // Threads are mapped as: k_local = li % WMMA_TILE, m_local = li / WMMA_TILE
        // so each group of WMMA_TILE=16 consecutive threads loads one X row (coalesced).
        for (int li = threadIdx.x; li < WMMA_TILE * BLOCK_M; li += blockDim.x) {
            const int k_local = li % WMMA_TILE;
            const int m_local = li / WMMA_TILE;
            const uint32_t mg = block_m + m_local;
            const uint32_t kg = k16    + k_local;
            s_XT[k_local][m_local] = (mg < M && kg < K)
                ? X[(uint64_t)mg * K + kg] : (__hip_bfloat16)0.f;
        }
        __syncthreads();

        // ── Decode A fragment from SCLP ──────────────────────────────────────
        // Thread frag_row holds row (n_base+frag_row) of the weight matrix,
        // columns k16..k16+15 — 16 consecutive SCLP-decoded BF16 values.
        bf16x16_t a_frag;
        const uint32_t n_row = n_base + frag_row;
        #pragma unroll
        for (int j = 0; j < WMMA_TILE; j++) {
            const uint32_t kg = k16 + j;
            if (n_row < N && kg < K) {
                uint64_t w_idx = (uint64_t)n_row * K + kg;
                uint8_t pb     = packed[w_idx >> 1];
                uint8_t p_idx  = (w_idx & 1) ? (pb & 0x0F) : (pb >> 4);
                uint8_t sm_val = sm[w_idx];
                uint16_t bits  = ((uint16_t)(sm_val >> 7) << 15)
                               | ((uint16_t)s_palette[p_idx] << 7)
                               | (sm_val & 0x7F);
                a_frag[j] = *(const __bf16*)&bits;
            } else {
                a_frag[j] = (__bf16)0.f;
            }
        }

        // ── Load B fragment from LDS ──────────────────────────────────────────
        // Thread frag_row holds row frag_row of the B tile:
        //   B[frag_row][j] = s_XT[frag_row][m_warp_local+j] = X[m_base+j][k16+frag_row]
        bf16x16_t b_frag;
        const int m_warp_local = warp_m * WMMA_TILE;
        #pragma unroll
        for (int j = 0; j < WMMA_TILE; j++) {
            b_frag[j] = s_XT[frag_row][m_warp_local + j];
        }

        // D[n_rel][m_rel] = sum_k A[n_rel][k] * B[k][m_rel]
        //                 = sum_k W[n_base+n_rel][k16+k] * X[m_base+m_rel][k16+k]
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc);

        __syncthreads();
    }

    // ── Write output to Y[M×N] ───────────────────────────────────────────────
    // RDNA3 C layout: thread t holds D[t%16][2*l + t/16] for l=0..7.
    // D[n_rel][m_rel] → Y[(m_base+m_rel)*N + (n_base+n_rel)]
    const uint32_t n_out  = n_base + frag_row;
    if (n_out < N) {
        const int col_lo = lane / WMMA_TILE;  // 0 for lanes 0..15, 1 for lanes 16..31
        #pragma unroll
        for (int l = 0; l < 8; l++) {
            const uint32_t m_out = m_base + 2 * l + col_lo;
            if (m_out < M) {
                Y[(uint64_t)m_out * N + n_out] = acc[l];
            }
        }
    }
}

// Launch fused WMMA GEMM for medium-to-large M (M > WMMA_THRESHOLD).
// Uses RDNA3 wave32 WMMA instructions to avoid an intermediate BF16 buffer.
// Sidecar correction is applied afterward via sclp_sidecar_correct_gemm_kernel.
inline void llama_sclp_fused_wmma(
    const void*   blob_ptr,
    const float*  src_f32,
    float*        dst_f32,
    uint32_t N, uint32_t K, uint32_t M,
    void*         tmp_bf16,
    hipStream_t   stream
) {
    constexpr int TILE = 16;
    constexpr int BN   = WMMA_WARPS_N * TILE;  // 32
    constexpr int BM   = WMMA_WARPS_M * TILE;  // 32

    // Convert F32 → BF16 activations.
    uint32_t mk = M * K;
    f32_to_bf16_kernel<<<(mk + 255) / 256, 256, 0, stream>>>(
        src_f32, (__hip_bfloat16*)tmp_bf16, mk);

    dim3 block(WMMA_WARPS_N * WMMA_WARPS_M * 32);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sclp_fused_wmma_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)blob_ptr,
        (const __hip_bfloat16*)tmp_bf16,
        dst_f32, N, K, M);

    // Sidecar correction (same kernel used by scalar fused GEMM path).
    sclp_sidecar_correct_gemm_kernel<<<4, 256, 0, stream>>>(
        (const uint8_t*)blob_ptr,
        (const __hip_bfloat16*)tmp_bf16,
        dst_f32, N, K, M);
}

// ============================================================
// SCLP4 kernels — 4 bits/weight, palette ≤4, 2 weights/byte
// Nibble layout: bits[3:2]=palette_idx, bit[1]=sign, bit[0]=mantissa_top1
// BF16: (sign<<15) | (palette[idx]<<7) | (mant_top1<<6)
// New header: [uint32 num_weights][uint32 n_experts][per-expert: uint8 palette_size, palette]
// ws_stream: per-expert nibble sections concatenated; sidecar follows total nibble bytes
// ============================================================

__global__ void sclp4_decode_blob_kernel(
    const uint8_t* __restrict__ blob,
    uint16_t*      __restrict__ output,
    uint32_t                    num_weights
) {
    // Parse header: n_experts and all palettes (thread 0 only)
    __shared__ uint32_t s_n_experts;
    __shared__ uint32_t s_ws_start;
    __shared__ uint8_t  s_palette_sizes[128];
    __shared__ uint8_t  s_palettes[128][4];

    if (threadIdx.x == 0) {
        uint32_t ne;
        __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        s_n_experts = ne;
        const uint8_t* p = blob + 8;
        for (uint32_t e = 0; e < ne && e < 128; e++) {
            s_palette_sizes[e] = p[0];
            for (int i = 0; i < (int)s_palette_sizes[e]; i++) s_palettes[e][i] = p[1 + i];
            p += 1 + p[0];
        }
        s_ws_start = (uint32_t)(p - blob);
    }
    __syncthreads();

    const uint8_t* ws = blob + s_ws_start;
    uint32_t n_experts = s_n_experts;
    uint32_t expert_nw = num_weights / n_experts;
    uint32_t expert_nibble_bytes = (expert_nw + 1) / 2;

    // Each thread handles 16 weights (8 bytes of nibble data)
    // Global weight index: thread processes weights [base_idx .. base_idx+15]
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t base_idx   = thread_idx * 16;

    if (base_idx >= num_weights) return;

    uint32_t e = base_idx / expert_nw;
    uint32_t local_base = base_idx - e * expert_nw;
    if (e >= n_experts) return;

    uint32_t remaining_expert = expert_nw - local_base;
    uint32_t n_this = min(16u, remaining_expert);

    uint64_t ws8 = 0;
    __builtin_memcpy(&ws8, ws + (uint64_t)e * expert_nibble_bytes + local_base / 2,
                     min(8u, (n_this + 1) / 2));

    uint16_t out[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint8_t byte   = (uint8_t)(ws8 >> ((i / 2) * 8));
        uint8_t nibble = (i % 2 == 0) ? (byte >> 4) : (byte & 0xF);
        uint8_t pidx   = nibble >> 2;
        uint8_t smn    = nibble & 0x3;
        uint8_t exp    = (pidx < s_palette_sizes[e]) ? s_palettes[e][pidx] : 0;
        uint8_t sign   = (smn >> 1) & 1;
        uint8_t mant   = smn & 1;
        out[i] = ((uint16_t)sign << 15) | ((uint16_t)exp << 7) | ((uint16_t)mant << 6);
    }

    // Vectorized stores: 2× uint4 (8 uint16 each) when full; scalar fallback otherwise
    if (n_this == 16) {
        uint4 v0, v1;
        __builtin_memcpy(&v0, out,     sizeof(uint4));
        __builtin_memcpy(&v1, out + 8, sizeof(uint4));
        *reinterpret_cast<uint4*>(output + base_idx)     = v0;
        *reinterpret_cast<uint4*>(output + base_idx + 8) = v1;
    } else {
        for (uint32_t i = 0; i < n_this; ++i) {
            output[base_idx + i] = out[i];
        }
    }
}

__global__ void sclp4_fixup_sidecar_kernel(
    const uint8_t* __restrict__ blob,
    uint16_t*      __restrict__ output,
    uint32_t                    num_weights
) {
    __shared__ uint32_t s_ws_start;
    __shared__ uint32_t s_n_experts;
    __shared__ uint32_t sidecar_count_s;

    if (threadIdx.x == 0) {
        uint32_t ne;
        __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        s_n_experts = ne;
        const uint8_t* p = blob + 8;
        for (uint32_t e = 0; e < ne; e++) { p += 1 + p[0]; }
        s_ws_start = (uint32_t)(p - blob);
    }
    __syncthreads();

    const uint8_t* ws    = blob + s_ws_start;
    uint32_t expert_nw   = num_weights / s_n_experts;
    uint64_t total_nibble_bytes = (uint64_t)s_n_experts * ((expert_nw + 1) / 2);
    const uint8_t* sidecar_base = ws + total_nibble_bytes;

    if (threadIdx.x == 0) {
        uint32_t sc;
        __builtin_memcpy(&sc, sidecar_base, sizeof(uint32_t));
        sidecar_count_s = sc;
    }
    __syncthreads();

    if (sidecar_count_s == 0) return;

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

// Fused decode-GEMV for SCLP4, M=1.
// Each warp handles one output row; reads nibble pairs from ws.
// For MoE use, the blob encodes a single expert (caller passes per-expert slice).
// For dense use (n_experts=1), row indexing is unchanged.
__launch_bounds__(1024, 2)
__global__ void sclp4_fused_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    uint32_t N,
    uint32_t K
) {
    // Parse header on device
    __shared__ uint8_t  s_palette_size;
    __shared__ uint8_t  s_palette[4];
    __shared__ uint32_t s_ws_start;

    if (threadIdx.x == 0) {
        uint32_t n_experts;
        __builtin_memcpy(&n_experts, blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        // For GEMV we use the single (first/only) expert palette that maps to this N×K block.
        // The dispatch code slices the ws_stream per expert, so n_experts=1 in practice here.
        s_palette_size = p[0];
        for (int i = 0; i < (int)s_palette_size; i++) s_palette[i] = p[1 + i];
        p += 1 + p[0];
        s_ws_start = (uint32_t)(p - blob);
    }
    __syncthreads();

    const uint8_t* ws = blob + s_ws_start;  // packed nibbles

    const int warps_per_block = blockDim.x / 32;
    const int warp_id  = threadIdx.x / 32;
    const int lane     = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    if (row >= N) return;

    float acc = 0.0f;
    const uint64_t row_nibble_base = (uint64_t)row * K;

    for (uint32_t k = lane; k < K; k += 32) {
        uint64_t nib_idx = row_nibble_base + k;
        uint8_t byte   = ws[nib_idx / 2];
        uint8_t nibble = (nib_idx % 2 == 0) ? (byte >> 4) : (byte & 0xF);
        uint8_t pidx   = nibble >> 2;
        uint8_t smn    = nibble & 0x3;
        uint8_t exp    = (pidx < s_palette_size) ? s_palette[pidx] : 0;
        uint8_t sign   = (smn >> 1) & 1;
        uint8_t mant   = smn & 1;
        uint16_t bits  = ((uint16_t)sign << 15) | ((uint16_t)exp << 7) | ((uint16_t)mant << 6);
        acc += __bfloat162float(*(__hip_bfloat16*)&bits) * x[k];
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) y[row] = acc;
}

inline void llama_sclp4_dispatch(
    const void* sclp_data,
    uint16_t*   output,
    uint32_t    num_weights,
    hipStream_t stream
) {
    const uint8_t* data = (const uint8_t*)sclp_data;
    dim3 block(256);

#if SCLP_MEMSET_STUB
    (void)data; (void)block;
    hipMemsetAsync(output, 0, (size_t)num_weights * sizeof(uint16_t), stream);
#else
    // Each thread handles 16 weights (8 nibble bytes)
    uint32_t groups = (num_weights + 15) / 16;
    dim3 decode_grid((groups + 255) / 256);
    sclp4_decode_blob_kernel<<<decode_grid, block, 0, stream>>>(data, output, num_weights);

    sclp4_fixup_sidecar_kernel<<<4, block, 0, stream>>>(data, output, num_weights);
#endif
}

inline void llama_sclp4_fused_gemv(
    const void*   blob_ptr,
    const float*  src_f32,
    float*        dst_f32,
    uint32_t      N,
    uint32_t      K,
    hipStream_t   stream
) {
    constexpr int WARPS_PER_BLOCK = 32;
    dim3 gemv_block(WARPS_PER_BLOCK * 32);
    dim3 gemv_grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sclp4_fused_gemv_kernel<<<gemv_grid, gemv_block, 0, stream>>>(
        (const uint8_t*)blob_ptr,
        src_f32,
        dst_f32, N, K);
}

// Fused decode-GEMV for SCLP4 MoE, single-token generation (n_batches=1).
// Mirrors sclp6_fused_moe_gemv_kernel: one block per (output_row_tile, active_expert_slot);
// each warp computes one output row of the routed expert e=ids[flat]. Reads only the
// routed expert's ws bytes — never materializes a full BF16 expert buffer.
// Sidecar is skipped (per SCLP convention for fused GEMV — ~0.9% of weights would be
// scattered, and a block-scoped scan empirically regressed throughput substantially).
__global__ void sclp4_fused_moe_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ src1,     // [K × src1_ne1 × n_batches]
    const int32_t* __restrict__ ids,      // [n_active × n_batches]
    float*         __restrict__ dst,      // [N × n_active × n_batches]
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t src1_ne1
) {
    // Shared memory layout:
    //   [0..3]   s_ws_offset (uint32) — byte offset of this expert's ws within blob
    //   [4..67]  s_lut[16]   (float, 64 bytes) — decoded float for each (pidx, smn) pair
    //   [68..]   s_x[K]      (float, K*4 bytes) — activation vector
    extern __shared__ char smem[];
    uint32_t* s_ws_offset = (uint32_t*)smem;
    float*    s_lut       = (float*)(smem + 4);    // 16 floats = 64 bytes
    float*    s_x         = (float*)(smem + 68);   // K floats

    const uint32_t flat     = blockIdx.y;
    const int32_t  e        = ids[flat];
    const uint32_t i_active = flat % n_active;
    const uint32_t i_batch  = flat / n_active;

    // Thread 0: parse blob header, locate this expert's ws, build 4×4 decode LUT.
    if (threadIdx.x == 0) {
        uint32_t total_nw, n_experts;
        __builtin_memcpy(&total_nw,  blob,     sizeof(uint32_t));
        __builtin_memcpy(&n_experts, blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        uint8_t pal[4];
        uint8_t pal_size = 0;
        for (uint32_t ei = 0; ei < n_experts; ei++) {
            if (ei == (uint32_t)e) {
                pal_size = p[0];
                for (int i = 0; i < (int)p[0]; i++) pal[i] = p[1 + i];
            }
            p += 1 + p[0];
        }
        uint32_t ws_start  = (uint32_t)(p - blob);
        uint32_t expert_nw = total_nw / n_experts;
        uint32_t expert_nibble_bytes = (expert_nw + 1) / 2;
        *s_ws_offset = ws_start + (uint32_t)e * expert_nibble_bytes;

        // 4×4 LUT: s_lut[pidx*4 + smn] = decoded float
        // SCLP4 nibble: pidx in bits[3:2], smn in bits[1:0]; smn = sign(1)|mant_top1(0)
        for (int pidx = 0; pidx < 4; pidx++) {
            uint8_t exp = (pidx < (int)pal_size) ? pal[pidx] : 0;
            for (int smn = 0; smn < 4; smn++) {
                uint8_t sign = (smn >> 1) & 1;
                uint8_t mant = smn & 1;
                uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp << 7) | ((uint16_t)mant << 6);
                s_lut[pidx * 4 + smn] = __bfloat162float(*(__hip_bfloat16*)&bits);
            }
        }
    }

    // Cooperatively load activation vector into shared memory.
    const float* x = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) {
        s_x[k] = x[k];
    }
    __syncthreads();

    const uint8_t* ws = blob + *s_ws_offset;
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    if (row >= N) return;

    float acc = 0.0f;
    // Each row has K nibbles → (K+1)/2 bytes. We iterate one byte per lane per step
    // (2 weights per byte). For typical K=2816 that's 1408 bytes/row, 44 iterations/warp.
    const uint32_t n_bytes_row = (K + 1) / 2;
    const uint64_t row_byte_base = (uint64_t)row * n_bytes_row;

    for (uint32_t b = lane; b < n_bytes_row; b += 32) {
        uint8_t byte = ws[row_byte_base + b];
        uint8_t n0   = byte >> 4;       // weight at column 2*b
        uint8_t n1   = byte & 0xF;      // weight at column 2*b + 1
        uint32_t k0  = b * 2;
        acc += s_lut[n0] * s_x[k0];
        if (k0 + 1 < K) {
            acc += s_lut[n1] * s_x[k0 + 1];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) dst[row + (uint64_t)flat * N] = acc;
}

inline void llama_sclp4_fused_moe_gemv(
    const void*    blob_ptr,
    const float*   src1,
    const int32_t* ids,
    float*         dst,
    uint32_t       N,
    uint32_t       K,
    uint32_t       n_active,
    uint32_t       n_batches,
    uint32_t       src1_ne1,
    hipStream_t    stream
) {
    constexpr int WARPS_PER_BLOCK = 16;
    dim3 block(WARPS_PER_BLOCK * 32);
    // Dynamic shared memory: 4 (ws_offset) + 64 (lut) + K*4 (activations)
    size_t smem_bytes = 68 + (size_t)K * sizeof(float);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, n_active * n_batches);
    sclp4_fused_moe_gemv_kernel<<<grid, block, smem_bytes, stream>>>(
        (const uint8_t*)blob_ptr,
        src1, ids, dst, N, K, n_active, src1_ne1);
}

// ============================================================
// SCLP6 kernels — 6 bits/weight, palette ≤8, 4 weights per 3 bytes
// sixbit layout: bits[5:3]=palette_idx, bit[2]=sign, bits[1:0]=mantissa_top2
// BF16: (sign<<15) | (palette[idx]<<7) | (mant_top2<<5)
// New header: [uint32 num_weights][uint32 n_experts][per-expert: uint8 palette_size, palette]
// ws_stream: per-expert 3-byte-group sections concatenated; sidecar follows total_groups*3 bytes
// ============================================================

__global__ void sclp6_decode_blob_kernel(
    const uint8_t* __restrict__ blob,
    uint16_t*      __restrict__ output,
    uint32_t                    num_weights
) {
    // Parse header on device (thread 0 only)
    __shared__ uint32_t s_n_experts;
    __shared__ uint32_t s_ws_start;
    __shared__ uint8_t  s_palette_sizes[128];
    __shared__ uint8_t  s_palettes[128][8];

    if (threadIdx.x == 0) {
        uint32_t ne;
        __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        s_n_experts = ne;
        const uint8_t* p = blob + 8;
        for (uint32_t e = 0; e < ne && e < 128; e++) {
            s_palette_sizes[e] = p[0];
            for (int i = 0; i < (int)s_palette_sizes[e]; i++) s_palettes[e][i] = p[1 + i];
            p += 1 + p[0];
        }
        s_ws_start = (uint32_t)(p - blob);
    }
    __syncthreads();

    const uint8_t* ws = blob + s_ws_start;
    uint32_t n_experts   = s_n_experts;
    uint32_t expert_nw   = num_weights / n_experts;
    uint32_t expert_groups = (expert_nw + 3) / 4;
    uint32_t total_groups  = expert_groups * n_experts;

    // Each thread handles one 3-byte group (4 weights)
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_groups) return;

    uint32_t e      = gid / expert_groups;
    uint32_t g_local = gid % expert_groups;

    const uint8_t b0 = ws[gid * 3 + 0];
    const uint8_t b1 = ws[gid * 3 + 1];
    const uint8_t b2 = ws[gid * 3 + 2];

    uint8_t sixbits[4];
    sixbits[0] = b0 >> 2;
    sixbits[1] = ((b0 & 0x3) << 4) | (b1 >> 4);
    sixbits[2] = ((b1 & 0xF) << 2) | (b2 >> 6);
    sixbits[3] = b2 & 0x3F;

    uint32_t base_idx = e * expert_nw + g_local * 4;
    uint32_t remaining = expert_nw - g_local * 4;
    uint32_t n_this = remaining < 4 ? remaining : 4;

    uint16_t out[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint8_t pidx = sixbits[i] >> 3;
        uint8_t smn  = sixbits[i] & 0x7;
        uint8_t exp  = (pidx < s_palette_sizes[e]) ? s_palettes[e][pidx] : 0;
        uint8_t sign = (smn >> 2) & 1;
        uint8_t mant = smn & 0x3;
        out[i] = ((uint16_t)sign << 15) | ((uint16_t)exp << 7) | ((uint16_t)mant << 5);
    }

    if (n_this == 4) {
        uint64_t v;
        __builtin_memcpy(&v, out, sizeof(uint64_t));
        *reinterpret_cast<uint64_t*>(output + base_idx) = v;
    } else {
        for (uint32_t i = 0; i < n_this; ++i) output[base_idx + i] = out[i];
    }
}

__global__ void sclp6_fixup_sidecar_kernel(
    const uint8_t* __restrict__ blob,
    uint16_t*      __restrict__ output,
    uint32_t                    num_weights
) {
    __shared__ uint32_t s_ws_start;
    __shared__ uint32_t s_n_experts;
    __shared__ uint32_t sidecar_count_s;

    if (threadIdx.x == 0) {
        uint32_t ne;
        __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        s_n_experts = ne;
        const uint8_t* p = blob + 8;
        for (uint32_t e = 0; e < ne; e++) { p += 1 + p[0]; }
        s_ws_start = (uint32_t)(p - blob);
    }
    __syncthreads();

    const uint8_t* ws    = blob + s_ws_start;
    uint32_t expert_nw   = num_weights / s_n_experts;
    uint64_t expert_groups = ((uint64_t)expert_nw + 3) / 4;
    uint64_t total_groups  = expert_groups * s_n_experts;
    const uint8_t* sidecar_base = ws + total_groups * 3;

    if (threadIdx.x == 0) {
        uint32_t sc;
        __builtin_memcpy(&sc, sidecar_base, sizeof(uint32_t));
        sidecar_count_s = sc;
    }
    __syncthreads();

    if (sidecar_count_s == 0) return;

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

// Fused decode-GEMV for SCLP6, M=1.
// Each warp handles one output row; reads 3-byte groups producing 4 weights each.
// Supports per-expert palettes: thread 0 walks the header to find ws_start and
// selects the palette for the expert that owns this N×K block.
// For dense (n_experts=1), behavior is identical to the old single-palette version.
__launch_bounds__(1024, 2)
__global__ void sclp6_fused_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    uint32_t N,
    uint32_t K,
    uint32_t expert_idx   // 0 for dense tensors
) {
    __shared__ uint8_t  s_palette_size;
    __shared__ uint8_t  s_palette[8];
    __shared__ uint32_t s_ws_offset;  // byte offset into blob for this expert's ws data

    if (threadIdx.x == 0) {
        uint32_t n_experts;
        __builtin_memcpy(&n_experts, blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        for (uint32_t e = 0; e < n_experts; e++) {
            if (e == expert_idx) {
                s_palette_size = p[0];
                for (int i = 0; i < (int)p[0]; i++) s_palette[i] = p[1 + i];
            }
            p += 1 + p[0];
        }
        uint32_t ws_start = (uint32_t)(p - blob);
        uint32_t total_nw; __builtin_memcpy(&total_nw, blob, sizeof(uint32_t));
        uint32_t expert_nw     = total_nw / n_experts;
        uint32_t expert_groups = (expert_nw + 3) / 4;
        s_ws_offset = ws_start + expert_idx * expert_groups * 3;
    }
    __syncthreads();

    const uint8_t* ws = blob + s_ws_offset;

    const int warps_per_block = blockDim.x / 32;
    const int warp_id  = threadIdx.x / 32;
    const int lane     = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    if (row >= N) return;

    float acc = 0.0f;
    const uint64_t row_group_base = (uint64_t)row * ((K + 3) / 4);
    const uint32_t n_groups_row = (K + 3) / 4;

    for (uint32_t g = lane; g < n_groups_row; g += 32) {
        uint64_t byte_off = (row_group_base + g) * 3;
        const uint8_t b0 = ws[byte_off + 0];
        const uint8_t b1 = ws[byte_off + 1];
        const uint8_t b2 = ws[byte_off + 2];

        uint8_t sixbits[4];
        sixbits[0] = b0 >> 2;
        sixbits[1] = ((b0 & 0x3) << 4) | (b1 >> 4);
        sixbits[2] = ((b1 & 0xF) << 2) | (b2 >> 6);
        sixbits[3] = b2 & 0x3F;

        uint32_t base_k = g * 4;
        uint32_t n_k = (base_k + 4 <= K) ? 4 : (K - base_k);
        for (uint32_t j = 0; j < n_k; j++) {
            uint8_t pidx  = sixbits[j] >> 3;
            uint8_t smn   = sixbits[j] & 0x7;
            uint8_t exp   = (pidx < s_palette_size) ? s_palette[pidx] : 0;
            uint8_t sign  = (smn >> 2) & 1;
            uint8_t mant  = smn & 0x3;
            uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp << 7) | ((uint16_t)mant << 5);
            acc += __bfloat162float(*(__hip_bfloat16*)&bits) * x[base_k + j];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) y[row] = acc;
}

inline void llama_sclp6_dispatch(
    const void* sclp_data,
    uint16_t*   output,
    uint32_t    num_weights,
    uint32_t    n_experts,   // passed from tensor ne[2]; avoids reading device pointer on host
    hipStream_t stream
) {
    const uint8_t* data = (const uint8_t*)sclp_data;
    dim3 block(256);

    uint32_t expert_nw = num_weights / n_experts;
    uint32_t expert_groups = (expert_nw + 3) / 4;
    uint32_t total_groups  = expert_groups * n_experts;

#if SCLP_MEMSET_STUB
    (void)data; (void)block; (void)expert_nw; (void)expert_groups; (void)total_groups;
    hipMemsetAsync(output, 0, (size_t)num_weights * sizeof(uint16_t), stream);
#else
    dim3 decode_grid((total_groups + 255) / 256);
    sclp6_decode_blob_kernel<<<decode_grid, block, 0, stream>>>(data, output, num_weights);

    sclp6_fixup_sidecar_kernel<<<4, block, 0, stream>>>(data, output, num_weights);
#endif
}

inline void llama_sclp6_fused_gemv(
    const void*   blob_ptr,
    const float*  src_f32,
    float*        dst_f32,
    uint32_t      N,
    uint32_t      K,
    uint32_t      expert_idx,  // 0 for dense tensors
    hipStream_t   stream
) {
    constexpr int WARPS_PER_BLOCK = 32;
    dim3 gemv_block(WARPS_PER_BLOCK * 32);
    dim3 gemv_grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sclp6_fused_gemv_kernel<<<gemv_grid, gemv_block, 0, stream>>>(
        (const uint8_t*)blob_ptr,
        src_f32,
        dst_f32, N, K, expert_idx);
}

// Fused decode-GEMV for MoE MUL_MAT_ID with SCLP6 expert weights.
// Each block handles one (row_group, flat) pair where flat = i_active + i_batch*n_active.
// Thread 0 walks the SCLP6 header to find expert e's palette and ws offset.
// No intermediate BF16 buffer — avoids the 1 GB allocation for 128 experts.
__launch_bounds__(1024, 2)
// Optimized fused MoE GEMV for SCLP6, single-token generation (n_batches==1).
// Key optimizations vs v1:
//   1. Activation vector (K floats) loaded cooperatively into shared memory once per block,
//      eliminating 32x redundant global reads (one per warp).
//   2. Decoded float lookup table (s_lut[64]) built from palette in shared memory —
//      replaces per-weight BF16 bit-twiddling with a single smem table lookup.
__global__ void sclp6_fused_moe_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ src1,     // [K × src1_ne1 × n_batches]
    const int32_t* __restrict__ ids,      // [n_active × n_batches]
    float*         __restrict__ dst,      // [N × n_active × n_batches]
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t src1_ne1
) {
    // Shared memory layout:
    //   [0..3]   s_ws_offset  (uint32)
    //   [4..255] s_lut[64]    (float, 256 bytes) — decoded float for each (pidx,smn) pair
    //   [256..]  s_x[K]       (float, K*4 bytes) — activation vector
    extern __shared__ char smem[];
    uint32_t* s_ws_offset = (uint32_t*)smem;
    float*    s_lut       = (float*)(smem + 4);    // 64 floats = 256 bytes
    float*    s_x         = (float*)(smem + 260);  // K floats

    const uint32_t flat    = blockIdx.y;
    const int32_t  e       = ids[flat];
    const uint32_t i_active = flat % n_active;
    const uint32_t i_batch  = flat / n_active;

    // Thread 0: parse blob header, build LUT.
    if (threadIdx.x == 0) {
        uint32_t total_nw, n_experts;
        __builtin_memcpy(&total_nw,   blob,     sizeof(uint32_t));
        __builtin_memcpy(&n_experts,  blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        uint8_t pal[8];
        uint8_t pal_size = 0;
        for (uint32_t ei = 0; ei < n_experts; ei++) {
            if (ei == (uint32_t)e) {
                pal_size = p[0];
                for (int i = 0; i < (int)p[0]; i++) pal[i] = p[1 + i];
            }
            p += 1 + p[0];
        }
        uint32_t ws_start     = (uint32_t)(p - blob);
        uint32_t expert_nw    = total_nw / n_experts;
        uint32_t expert_groups = (expert_nw + 3) / 4;
        *s_ws_offset = ws_start + (uint32_t)e * expert_groups * 3;

        // Build 8×8 decode LUT: s_lut[pidx * 8 + smn] = float value
        for (int pidx = 0; pidx < 8; pidx++) {
            uint8_t exp = (pidx < (int)pal_size) ? pal[pidx] : 0;
            for (int smn = 0; smn < 8; smn++) {
                uint8_t sign = (smn >> 2) & 1;
                uint8_t mant = smn & 0x3;
                uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp << 7) | ((uint16_t)mant << 5);
                s_lut[pidx * 8 + smn] = __bfloat162float(*(__hip_bfloat16*)&bits);
            }
        }
    }

    // Cooperatively load activation vector into shared memory.
    const float* x = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) {
        s_x[k] = x[k];
    }
    __syncthreads();

    const uint8_t* ws = blob + *s_ws_offset;
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    if (row >= N) return;

    float acc = 0.0f;
    const uint64_t row_group_base = (uint64_t)row * ((K + 3) / 4);
    const uint32_t n_groups_row   = (K + 3) / 4;

    for (uint32_t g = lane; g < n_groups_row; g += 32) {
        uint64_t byte_off = (row_group_base + g) * 3;
        const uint8_t b0 = ws[byte_off + 0];
        const uint8_t b1 = ws[byte_off + 1];
        const uint8_t b2 = ws[byte_off + 2];

        uint8_t sixbits[4];
        sixbits[0] = b0 >> 2;
        sixbits[1] = ((b0 & 0x3) << 4) | (b1 >> 4);
        sixbits[2] = ((b1 & 0xF) << 2) | (b2 >> 6);
        sixbits[3] = b2 & 0x3F;

        uint32_t base_k = g * 4;
        uint32_t n_k = (base_k + 4 <= K) ? 4 : (K - base_k);
        for (uint32_t j = 0; j < n_k; j++) {
            acc += s_lut[sixbits[j]] * s_x[base_k + j];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) dst[row + (uint64_t)flat * N] = acc;
}

inline void llama_sclp6_fused_moe_gemv(
    const void*    blob_ptr,
    const float*   src1,
    const int32_t* ids,
    float*         dst,
    uint32_t       N,
    uint32_t       K,
    uint32_t       n_active,
    uint32_t       n_batches,
    uint32_t       src1_ne1,
    hipStream_t    stream
) {
    constexpr int WARPS_PER_BLOCK = 16;
    dim3 block(WARPS_PER_BLOCK * 32);
    // Dynamic shared memory: 4 (ws_offset) + 256 (lut) + K*4 (activations)
    size_t smem_bytes = 260 + (size_t)K * sizeof(float);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, n_active * n_batches);
    sclp6_fused_moe_gemv_kernel<<<grid, block, smem_bytes, stream>>>(
        (const uint8_t*)blob_ptr,
        src1, ids, dst, N, K, n_active, src1_ne1);
}

// ============================================================
// SCLP8 (original) decode dispatch
// ============================================================

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

#if SCLP_MEMSET_STUB
    (void)data; (void)block;
    hipMemsetAsync(output, 0, (size_t)num_weights * sizeof(uint16_t), stream);
#else
    // Main decode: one thread per 8 weights (single uint64_t load from ws_stream)
    uint32_t groups = (num_weights + 7) / 8;
    dim3 decode_grid((groups + 255) / 256);
    sclp_decode_blob_kernel<<<decode_grid, block, 0, stream>>>(data, output, num_weights);

    // Sidecar fixup: fixed 4-block grid with stride loop handles any sidecar count.
    // Threads where i >= sidecar_count return after a single shared-memory read.
    sclp_fixup_sidecar_kernel<<<4, block, 0, stream>>>(data, output, num_weights);
#endif
}
