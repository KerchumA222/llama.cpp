#include "sclp_bridge_common.cuh"

__global__ void sclp_decode_blob_kernel(
    const uint8_t* __restrict__ blob,
    uint16_t*      __restrict__ output,
    uint32_t                    num_weights
) {
    __shared__ uint32_t s_n_experts;
    __shared__ uint32_t s_scales_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint8_t  s_palette_sizes[256];
    __shared__ uint8_t  s_palettes[256][16];
    __shared__ uint32_t s_expert_nw;

    if (threadIdx.x == 0) {
        uint32_t ne;
        __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        s_n_experts = ne;
        const uint8_t* p = blob + 8;
        for (uint32_t e = 0; e < ne && e < 256; e++) {
            s_palette_sizes[e] = p[0];
            for (int i = 0; i < (int)s_palette_sizes[e]; i++) s_palettes[e][i] = p[1 + i];
            p += 1 + p[0];
        }
        s_scales_start = (uint32_t)(p - blob);
        s_ws_start = s_scales_start + (num_weights / QK_SCLP) * sizeof(uint16_t);
        s_expert_nw = num_weights / ne;
    }
    __syncthreads();

    const uint8_t* scales = blob + s_scales_start;
    const uint8_t* ws     = blob + s_ws_start;

    // Each thread handles 8 weights via a single uint64_t load
    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t base_idx   = thread_idx * 8;

    if (base_idx >= num_weights) return;

    const uint32_t expert_nw = s_expert_nw;
    const uint32_t e         = base_idx / expert_nw;
    const uint32_t local_idx = base_idx % expert_nw;

    if (e >= s_n_experts) return;

    uint64_t ws8 = 0;
    uint32_t remaining = expert_nw - local_idx;
    __builtin_memcpy(&ws8, ws + (uint64_t)e * expert_nw + local_idx, min(8u, remaining));

    // Per-block scale (one scale per 32 weights, shared by 4 threads)
    const uint16_t* s_ptr = (const uint16_t*)(scales + (uint64_t)e * (expert_nw / QK_SCLP) * sizeof(uint16_t));
    float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(s_ptr + (local_idx / QK_SCLP)));

    uint16_t out[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint8_t b   = (uint8_t)(ws8 >> (i * 8));
        uint8_t exp = s_palettes[e][b >> 4];
        uint8_t smn = b & 0x0F;
        uint16_t bits = ((uint16_t)(smn >> 3) << 15) | ((uint16_t)exp << 7) | ((uint16_t)(smn & 0x7) << 4);
        float w = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&bits)) * scale;
        out[i] = float_to_bf16_dev(w);
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
    __shared__ uint32_t s_ws_start;

    if (threadIdx.x == 0) {
        uint32_t ne;
        __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        for (uint32_t ei = 0; ei < ne; ei++) p += 1 + p[0];
        uint32_t scales_start = (uint32_t)(p - blob);
        s_ws_start = scales_start + (num_weights / QK_SCLP) * sizeof(uint16_t);
    }
    __syncthreads();

    const uint8_t* ws           = blob + s_ws_start;
    const uint8_t* sidecar_base = ws + num_weights;  // ws_stream is num_weights bytes (1 per weight)

    // Sidecar v2: CSR-style per-row ranges; one thread per row.
    const sclp_sidecar_view sc = sclp_sidecar_parse(sidecar_base, num_weights);
    if (sc.count == 0) return;

    const uint32_t n_rows = num_weights / sc.K;
    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t r = blockIdx.x * blockDim.x + threadIdx.x; r < n_rows; r += stride) {
        uint32_t lo, hi;
        sclp_sidecar_row_range(sc, r, &lo, &hi);
        for (uint32_t i = lo; i < hi; i++) {
            output[(uint64_t)r * sc.K + sclp_sidecar_col(sc, i)] = sclp_sidecar_val(sc, i);
        }
    }
}

// K-tiled fused decode-GEMV for SCLP8 with folded sidecar correction.
// LUT (256 floats = 1024 bytes) stays in smem permanently; x is K-tiled.
// Sidecar reads from global x (not s_x which only holds the current tile).
// smem = 1024 (LUT) + 16384 (TILE_K×4) = 17408 → 3 blocks/CU on RDNA3.
__launch_bounds__(512, 3)
__global__ void sclp_fused_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    uint32_t N,
    uint32_t K
) {
    constexpr uint32_t TILE_K = 4096;
    extern __shared__ char smem[];
    float* s_lut = (float*)smem;
    float* s_x   = (float*)(smem + 1024);

    __shared__ uint32_t s_scales_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint32_t s_sc_base;
    if (threadIdx.x < 16) {
        uint32_t ne;
        __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        uint8_t pal_size = p[0];
        uint8_t pidx     = (uint8_t)threadIdx.x;
        uint8_t exp_     = (pidx < pal_size) ? p[1 + pidx] : 0;
        for (int smn = 0; smn < 16; smn++) {
            uint8_t sign = (smn >> 3) & 1;
            uint8_t mant = smn & 0x7;
            uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp_ << 7) | ((uint16_t)mant << 4);
            s_lut[(int)pidx * 16 + smn] = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&bits));
        }
        if (threadIdx.x == 0) {
            const uint8_t* p_p = blob + 8;
            for (uint32_t ei = 0; ei < ne; ei++) p_p += 1 + p_p[0];
            s_scales_start = (uint32_t)(p_p - blob);
            s_ws_start = s_scales_start + ((uint64_t)N * K / QK_SCLP) * sizeof(uint16_t);
            s_sc_base  = s_ws_start + (uint32_t)((uint64_t)N * K);
        }
    }
    __syncthreads();

    const uint16_t* scales = (const uint16_t*)(blob + s_scales_start);
    const uint8_t*  ws     = blob + s_ws_start;

    const int warps_per_block = blockDim.x / 32;
    const int warp_id  = threadIdx.x / 32;
    const int lane     = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    const bool valid = (row < N);

    float acc = 0.0f;
    const uint64_t row_base = valid ? (uint64_t)row * K : 0;

    for (uint32_t k_tile = 0; k_tile < K; k_tile += TILE_K) {
        const uint32_t tile_end = min(k_tile + TILE_K, K);
        const uint32_t tile_len = tile_end - k_tile;

        for (uint32_t k = threadIdx.x; k < tile_len; k += blockDim.x) {
            s_x[k] = x[k_tile + k];
        }
        __syncthreads();

        if (valid) {
            const uint32_t K8_start = ((k_tile + 7) / 8) * 8;
            const uint32_t K8_end   = (tile_end / 8) * 8;

            for (uint32_t k8 = K8_start + lane * 8; k8 < K8_end; k8 += 32 * 8) {
                uint64_t ws8; __builtin_memcpy(&ws8, ws + row_base + k8, sizeof(uint64_t));
                float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + ((row_base + k8) >> 5)));
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    acc += (s_lut[(uint8_t)(ws8 >> (j * 8))] * scale) * s_x[k8 + j - k_tile];
                }
            }
            for (uint32_t k = k_tile + lane; k < K8_start && k < tile_end; k += 32) {
                float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + ((row_base + k) >> 5)));
                acc += (s_lut[ws[row_base + k]] * scale) * s_x[k - k_tile];
            }
            for (uint32_t k = K8_end + lane; k < tile_end; k += 32) {
                float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + ((row_base + k) >> 5)));
                acc += (s_lut[ws[row_base + k]] * scale) * s_x[k - k_tile];
            }
        }
        __syncthreads();
    }

    if (!valid) return;

    // Folded sidecar v2: O(1) row-range lookup, reads from global x (not s_x).
    const sclp_sidecar_view sc = sclp_sidecar_parse(blob + s_sc_base, (uint32_t)((uint64_t)N * K));
    if (sc.count > 0) {
        uint32_t lo = 0, hi = 0;
        if (lane == 0) sclp_sidecar_row_range(sc, row, &lo, &hi);
        lo = __shfl(lo, 0); hi = __shfl(hi, 0);
        for (uint32_t si = lo + lane; si < hi; si += 32) {
            uint32_t col  = sclp_sidecar_col(sc, si);
            uint64_t gidx = row_base + col;
            float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (gidx >> 5)));
            float approx = s_lut[ws[gidx]] * scale;
            uint16_t tb = sclp_sidecar_val(sc, si);
            float trueval = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&tb));
            acc += (trueval - approx) * x[col];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) y[row] = acc;
}

// Sidecar correction kernel for fused GEMM (M>1).
// Same as above, but applies correction for all M activation rows simultaneously.
__global__ void sclp_sidecar_correct_gemm_kernel(
    const uint8_t*        __restrict__ blob,
    const __hip_bfloat16* __restrict__ X,  // BF16 activations [M × K]
    float*                             Y,  // F32 output [M × N], updated atomically
    uint32_t N, uint32_t K, uint32_t M
) {
    __shared__ uint32_t s_scales_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint8_t  s_palette[16];

    if (threadIdx.x == 0) {
        uint32_t ne;
        __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        // Sidecar kernel is currently only used for Dense (n_experts=1) SCLP8.
        // We parse expert 0's palette.
        for (int i = 0; i < (int)p[0]; i++) s_palette[i] = p[1 + i];
        for (uint32_t ei = 0; ei < ne; ei++) p += 1 + p[0];
        s_scales_start = (uint32_t)(p - blob);
        s_ws_start = s_scales_start + ((uint64_t)N * K / QK_SCLP) * sizeof(uint16_t);
    }
    __syncthreads();

    uint32_t pal0, pal1, pal2, pal3;
    __builtin_memcpy(&pal0, s_palette + 0,  sizeof(uint32_t));
    __builtin_memcpy(&pal1, s_palette + 4,  sizeof(uint32_t));
    __builtin_memcpy(&pal2, s_palette + 8,  sizeof(uint32_t));
    __builtin_memcpy(&pal3, s_palette + 12, sizeof(uint32_t));

    const uint8_t* ws = blob + s_ws_start;
    const uint8_t* sidecar_base = ws + (uint64_t)N * K;  // ws_stream is N*K bytes

    // Sidecar v2: one thread per weight row, direct range lookup.
    const sclp_sidecar_view sc = sclp_sidecar_parse(sidecar_base, (uint32_t)((uint64_t)N * K));
    if (sc.count == 0) return;

    const uint16_t* scales = (const uint16_t*)(blob + s_scales_start);

    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t n = blockIdx.x * blockDim.x + threadIdx.x; n < N; n += stride) {
        uint32_t lo, hi;
        sclp_sidecar_row_range(sc, n, &lo, &hi);
        for (uint32_t i = lo; i < hi; i++) {
            uint32_t k = sclp_sidecar_col(sc, i);
            uint16_t correct_bits = sclp_sidecar_val(sc, i);
            uint64_t global_idx = (uint64_t)n * K + k;

            float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + global_idx / QK_SCLP));

            uint8_t b     = ws[global_idx];
            uint8_t smn   = b & 0x0F;
            uint16_t approx_bits = ((uint16_t)(smn >> 3) << 15)
                                 | ((uint16_t)sclp8_palette_pick(pal0, pal1, pal2, pal3, b >> 4) << 7)
                                 | ((uint16_t)(smn & 0x7) << 4);

            float w_correct = __bfloat162float(*(__hip_bfloat16*)&correct_bits);
            float w_approx  = __bfloat162float(*(__hip_bfloat16*)&approx_bits) * scale;
            float w_delta   = w_correct - w_approx;

            for (uint32_t m = 0; m < M; m++) {
                float xval = __bfloat162float(X[(uint64_t)m * K + k]);
                atomicAdd(&Y[(uint64_t)m * N + n], w_delta * xval);
            }
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
    __shared__ uint32_t s_scales_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint8_t  s_palette[16];

    if (threadIdx.x < 16) {
        uint32_t ne;
        __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        uint8_t pal_size = p[0];
        uint8_t pidx     = (uint8_t)threadIdx.x;
        s_palette[pidx]  = (pidx < pal_size) ? p[1 + pidx] : 0;
        if (threadIdx.x == 0) {
            const uint8_t* p_ws = blob + 8;
            for (uint32_t ei = 0; ei < ne; ei++) p_ws += 1 + p_ws[0];
            s_scales_start = (uint32_t)(p_ws - blob);
            s_ws_start = s_scales_start + ((uint64_t)N * K / QK_SCLP) * sizeof(uint16_t);
        }
    }
    __syncthreads();

    uint32_t pal0, pal1, pal2, pal3;
    __builtin_memcpy(&pal0, s_palette + 0,  sizeof(uint32_t));
    __builtin_memcpy(&pal1, s_palette + 4,  sizeof(uint32_t));
    __builtin_memcpy(&pal2, s_palette + 8,  sizeof(uint32_t));
    __builtin_memcpy(&pal3, s_palette + 12, sizeof(uint32_t));

    const uint16_t* scales = (const uint16_t*)(blob + s_scales_start);
    const uint8_t* ws = blob + s_ws_start;  // ws_stream: N*K bytes, 1 per weight

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
        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (row_base + k8) / QK_SCLP));

        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint8_t b     = (uint8_t)(ws8 >> (j * 8));
            uint8_t smn   = b & 0x0F;
            uint16_t bits = ((uint16_t)(smn >> 3) << 15)
                          | ((uint16_t)sclp8_palette_pick(pal0, pal1, pal2, pal3, b >> 4) << 7)
                          | ((uint16_t)(smn & 0x7) << 4);
            float w = __bfloat162float(*(__hip_bfloat16*)&bits) * scale;
            #pragma unroll TILE_M
            for (int mi = 0; mi < TILE_M; mi++) {
                if (mi < m_count) acc[mi] += w * __bfloat162float(X[(uint64_t)(m_start + mi) * K + k8 + j]);
            }
        }
    }

    // Scalar tail for K not divisible by 8.
    for (uint32_t k = K8 + (uint32_t)lane; k < K; k += 32) {
        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (row_base + k) / QK_SCLP));
        uint8_t b     = ws[row_base + k];
        uint8_t smn   = b & 0x0F;
        uint16_t bits = ((uint16_t)(smn >> 3) << 15)
                      | ((uint16_t)sclp8_palette_pick(pal0, pal1, pal2, pal3, b >> 4) << 7)
                      | ((uint16_t)(smn & 0x7) << 4);
        float w = __bfloat162float(*(__hip_bfloat16*)&bits) * scale;
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
void llama_sclp_fused_gemm(
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
    sclp_sidecar_correct_gemm_kernel<<<256, 256, 0, stream>>>(
        (const uint8_t*)blob_ptr,
        (const __hip_bfloat16*)tmp_bf16,
        dst_f32, N, K, M);
}

// Launch fused GEMV for the M=1 inference case.
// Accepts F32 activations directly — no separate conversion kernel or scratch buffer.
void llama_sclp_fused_gemv(
    const void*   blob_ptr,
    const float*  src_f32,
    float*        dst_f32,
    uint32_t      N,
    uint32_t      K,
    hipStream_t   stream
) {
    constexpr int WARPS_PER_BLOCK = 16;
    dim3 gemv_block(WARPS_PER_BLOCK * 32);
    dim3 gemv_grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    constexpr uint32_t TILE_K = 4096;
    size_t smem_bytes = 1024 + (size_t)min(K, TILE_K) * sizeof(float);
    SCLP_TG_TIME_BEGIN(sclp_tg::CAT_SCLP_DENSE);
    sclp_fused_gemv_kernel<<<gemv_grid, gemv_block, smem_bytes, stream>>>(
        (const uint8_t*)blob_ptr,
        src_f32,
        dst_f32, N, K);
    SCLP_TG_TIME_END();
}

// Fused decode-GEMV for MoE MUL_MAT_ID with SCLP expert weights.
// src0 blob: [K, N, n_experts] stacked experts encoded as one SCLP blob.
// src1: F32 [K, n_active] routed activations (one column per active expert).
// ids:  int32 [n_active] expert routing indices (which expert weight to use).
// dst:  F32 [N, n_active] output.
// Sidecar correction is omitted (same rationale as sclp_fused_gemv_kernel).
__launch_bounds__(1024, 2)
// K-tiled fused MoE GEMV for SCLP8 with smem x broadcast + folded sidecar.
__launch_bounds__(512, 2)
__global__ void sclp_fused_moe_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ src1,
    const int32_t* __restrict__ ids,
    float*         __restrict__ dst,
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t src1_ne1
) {
    constexpr uint32_t TILE_K = 4096;
    extern __shared__ char smem[];
    __shared__ uint32_t s_scales_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint32_t s_sc_base;
    __shared__ uint32_t s_total_nw;
    __shared__ uint8_t  s_palette[16];
    float* s_x = (float*)smem;

    const uint32_t flat = blockIdx.y;
    const int32_t  e    = ids[flat];
    const uint32_t i_active = flat % n_active;
    const uint32_t i_batch  = flat / n_active;

    if (threadIdx.x == 0) {
        uint32_t ne, total_nw;
        __builtin_memcpy(&total_nw, blob,     sizeof(uint32_t));
        __builtin_memcpy(&ne,       blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        for (uint32_t ei = 0; ei < ne; ei++) {
            if ((int32_t)ei == e) {
                for (int i = 0; i < (int)p[0]; i++) s_palette[i] = p[1 + i];
            }
            p += 1 + p[0];
        }
        s_scales_start = (uint32_t)(p - blob);
        s_ws_start = s_scales_start + (total_nw / QK_SCLP) * sizeof(uint16_t);
        s_sc_base  = s_ws_start + total_nw;
        s_total_nw = total_nw;
    }
    __syncthreads();

    uint32_t pal0, pal1, pal2, pal3;
    __builtin_memcpy(&pal0, s_palette + 0,  sizeof(uint32_t));
    __builtin_memcpy(&pal1, s_palette + 4,  sizeof(uint32_t));
    __builtin_memcpy(&pal2, s_palette + 8,  sizeof(uint32_t));
    __builtin_memcpy(&pal3, s_palette + 12, sizeof(uint32_t));

    const uint16_t* scales = (const uint16_t*)(blob + s_scales_start);
    const uint8_t*  ws     = blob + s_ws_start;

    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    const bool valid = (row < N);

    const float* x = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;
    const uint64_t ws_base = valid ? (uint64_t)e * N * K + (uint64_t)row * K : 0;

    float acc = 0.0f;

    for (uint32_t k_tile = 0; k_tile < K; k_tile += TILE_K) {
        const uint32_t tile_end = min(k_tile + TILE_K, K);
        const uint32_t tile_len = tile_end - k_tile;

        for (uint32_t k = threadIdx.x; k < tile_len; k += blockDim.x) {
            s_x[k] = x[k_tile + k];
        }
        __syncthreads();

        if (valid) {
            const uint32_t K8_start = ((k_tile + 7) / 8) * 8;
            const uint32_t K8_end   = (tile_end / 8) * 8;

            for (uint32_t k8 = K8_start + lane * 8; k8 < K8_end; k8 += 32 * 8) {
                uint64_t ws8; __builtin_memcpy(&ws8, ws + ws_base + k8, sizeof(uint64_t));
                float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + ((ws_base + k8) >> 5)));
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    uint8_t b   = (uint8_t)(ws8 >> (j * 8));
                    uint8_t smn = b & 0x0F;
                    uint16_t bits = ((uint16_t)(smn >> 3) << 15)
                                  | ((uint16_t)sclp8_palette_pick(pal0, pal1, pal2, pal3, b >> 4) << 7)
                                  | ((uint16_t)(smn & 0x7) << 4);
                    acc += (__bfloat162float(*(__hip_bfloat16*)&bits) * scale) * s_x[k8 + j - k_tile];
                }
            }
            for (uint32_t k = k_tile + lane; k < K8_start && k < tile_end; k += 32) {
                float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + ((ws_base + k) >> 5)));
                uint8_t b = ws[ws_base + k]; uint8_t smn = b & 0x0F;
                uint16_t bits = ((uint16_t)(smn >> 3) << 15) | ((uint16_t)sclp8_palette_pick(pal0, pal1, pal2, pal3, b >> 4) << 7) | ((uint16_t)(smn & 0x7) << 4);
                acc += (__bfloat162float(*(__hip_bfloat16*)&bits) * scale) * s_x[k - k_tile];
            }
            for (uint32_t k = K8_end + lane; k < tile_end; k += 32) {
                float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + ((ws_base + k) >> 5)));
                uint8_t b = ws[ws_base + k]; uint8_t smn = b & 0x0F;
                uint16_t bits = ((uint16_t)(smn >> 3) << 15) | ((uint16_t)sclp8_palette_pick(pal0, pal1, pal2, pal3, b >> 4) << 7) | ((uint16_t)(smn & 0x7) << 4);
                acc += (__bfloat162float(*(__hip_bfloat16*)&bits) * scale) * s_x[k - k_tile];
            }
        }
        __syncthreads();
    }

    if (!valid) return;

    // Folded sidecar v2: global row = e*N + row, O(1) range lookup.
    const sclp_sidecar_view sc = sclp_sidecar_parse(blob + s_sc_base, s_total_nw);
    if (sc.count > 0) {
        const uint32_t grow = (uint32_t)e * N + row;
        uint32_t lo = 0, hi = 0;
        if (lane == 0) sclp_sidecar_row_range(sc, grow, &lo, &hi);
        lo = __shfl(lo, 0); hi = __shfl(hi, 0);
        for (uint32_t si = lo + lane; si < hi; si += 32) {
            uint32_t col  = sclp_sidecar_col(sc, si);
            uint64_t gidx = ws_base + col;
            float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (gidx >> 5)));
            uint8_t b = ws[gidx]; uint8_t smn = b & 0x0F;
            uint16_t ab = ((uint16_t)(smn >> 3) << 15) | ((uint16_t)sclp8_palette_pick(pal0, pal1, pal2, pal3, b >> 4) << 7) | ((uint16_t)(smn & 0x7) << 4);
            float approx = __bfloat162float(*(__hip_bfloat16*)&ab) * scale;
            uint16_t tb = sclp_sidecar_val(sc, si);
            float trueval = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&tb));
            acc += (trueval - approx) * x[col];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) dst[row + (uint64_t)flat * N] = acc;
}

void llama_sclp_fused_moe_gemv(
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
    constexpr uint32_t TILE_K = 4096;
    dim3 block(WARPS_PER_BLOCK * 32);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, n_active * n_batches);
    size_t smem_bytes = (size_t)min(K, TILE_K) * sizeof(float);
    SCLP_TG_TIME_BEGIN(sclp_tg::CAT_SCLP_MOE);
    sclp_fused_moe_gemv_kernel<<<grid, block, smem_bytes, stream>>>(
        (const uint8_t*)blob_ptr,
        src1, ids, dst, N, K, n_active, src1_ne1);
    SCLP_TG_TIME_END();
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

    __shared__ uint32_t s_scales_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint8_t  s_palette[16];
    // Transposed activation tile: s_XT[k_local][m_local] = X[block_m+m_local][k16+k_local]
    // +1 column padding avoids LDS bank conflicts on the 32-bank RDNA3 layout.
    __shared__ __hip_bfloat16 s_XT[WMMA_TILE][BLOCK_M + 1];

    if (threadIdx.x < 16) {
        uint32_t ne;
        __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        uint8_t pal_size = p[0];
        uint8_t pidx     = (uint8_t)threadIdx.x;
        s_palette[pidx]  = (pidx < pal_size) ? p[1 + pidx] : 0;
        if (threadIdx.x == 0) {
            const uint8_t* p_ws = blob + 8;
            for (uint32_t ei = 0; ei < ne; ei++) p_ws += 1 + p_ws[0];
            s_scales_start = (uint32_t)(p_ws - blob);
            s_ws_start = s_scales_start + ((uint64_t)N * K / QK_SCLP) * sizeof(uint16_t);
        }
    }
    __syncthreads();

    uint32_t pal0, pal1, pal2, pal3;
    __builtin_memcpy(&pal0, s_palette + 0,  sizeof(uint32_t));
    __builtin_memcpy(&pal1, s_palette + 4,  sizeof(uint32_t));
    __builtin_memcpy(&pal2, s_palette + 8,  sizeof(uint32_t));
    __builtin_memcpy(&pal3, s_palette + 12, sizeof(uint32_t));

    const uint16_t* scales = (const uint16_t*)(blob + s_scales_start);
    const uint8_t* ws = blob + s_ws_start;  // ws_stream: N*K bytes, 1 per weight

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
                float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + w_idx / QK_SCLP));
                uint8_t b      = ws[w_idx];
                uint8_t p_idx  = b >> 4;
                uint8_t smn    = b & 0x0F;
                uint16_t bits  = ((uint16_t)(smn >> 3) << 15)
                               | ((uint16_t)sclp8_palette_pick(pal0, pal1, pal2, pal3, p_idx) << 7)
                               | ((uint16_t)(smn & 0x7) << 4);
                float w = __bfloat162float(*(__hip_bfloat16*)&bits) * scale;
                a_frag[j] = __float2bfloat16(w);
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
void llama_sclp_fused_wmma(
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
    sclp_sidecar_correct_gemm_kernel<<<256, 256, 0, stream>>>(
        (const uint8_t*)blob_ptr,
        (const __hip_bfloat16*)tmp_bf16,
        dst_f32, N, K, M);
}


// ============================================================
// SCLP8 (original) decode dispatch
// ============================================================

// Decode an SCLP blob (device pointer) into a flat BF16 uint16_t buffer.
// num_weights must equal ggml_nelements(src0).
// No host-side device reads; safe during HIP stream capture.
void llama_sclp_dispatch(
    const void* sclp_data,
    uint16_t*   output,
    uint32_t    num_weights,
    hipStream_t stream
) {
    const uint8_t* data = (const uint8_t*)sclp_data;
    dim3 block(256);

    // Main decode: one thread per 8 weights (single uint64_t load from ws_stream)
    uint32_t groups = (num_weights + 7) / 8;
    dim3 decode_grid((groups + 255) / 256);
    sclp_decode_blob_kernel<<<decode_grid, block, 0, stream>>>(data, output, num_weights);

    // Sidecar fixup: fixed 4-block grid with stride loop handles any sidecar count.
    // Threads where i >= sidecar_count return after a single shared-memory read.
    sclp_fixup_sidecar_kernel<<<256, block, 0, stream>>>(data, output, num_weights);
}
