#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <atomic>

// SCLP_TIME_TG=1: aggregate fused-GEMV timings by category, print summary on exit.
// Each launcher wraps its kernel launch with SCLP_TG_TIME_BEGIN/END(cat). When the
// env var is unset the macros expand to nothing — zero overhead in release runs.
namespace sclp_tg {
    enum Cat : int {
        CAT_SCLP_DENSE = 0, CAT_SCLP_MOE,
        CAT_SCLP4_DENSE,    CAT_SCLP4_MOE,
        CAT_SCLP6_DENSE,    CAT_SCLP6_MOE,
        CAT_COUNT
    };
    inline const char* cat_name(int c) {
        switch (c) {
            case CAT_SCLP_DENSE: return "SCLP8.dense";
            case CAT_SCLP_MOE:   return "SCLP8.moe  ";
            case CAT_SCLP4_DENSE:return "SCLP4.dense";
            case CAT_SCLP4_MOE:  return "SCLP4.moe  ";
            case CAT_SCLP6_DENSE:return "SCLP6.dense";
            case CAT_SCLP6_MOE:  return "SCLP6.moe  ";
            default: return "?";
        }
    }
    struct Agg { std::atomic<uint64_t> ns_total{0}; std::atomic<uint64_t> calls{0}; };
    inline Agg& agg(int c) { static Agg a[CAT_COUNT]; return a[c]; }

    inline bool enabled() {
        static const bool v = (getenv("SCLP_TIME_TG") != nullptr);
        return v;
    }
    inline void dump() {
        bool any = false;
        for (int c = 0; c < CAT_COUNT; c++) if (agg(c).calls.load()) { any = true; break; }
        if (!any) return;
        fprintf(stderr, "\n[SCLP_TIME_TG] fused-GEMV totals (synchronous, includes hipStreamSynchronize):\n");
        fprintf(stderr, "  %-12s %10s %12s %12s\n", "category", "calls", "total_ms", "avg_us");
        for (int c = 0; c < CAT_COUNT; c++) {
            uint64_t n = agg(c).calls.load();
            uint64_t ns = agg(c).ns_total.load();
            if (n == 0) continue;
            fprintf(stderr, "  %-12s %10llu %12.3f %12.2f\n",
                cat_name(c), (unsigned long long)n,
                ns / 1.0e6, (ns / 1000.0) / n);
        }
    }
    inline int register_atexit() { atexit(&dump); return 0; }
}
#define SCLP_TG_INIT() static int _sclp_tg_atexit = sclp_tg::register_atexit()
#define SCLP_TG_TIME_BEGIN(cat) \
    hipEvent_t _tg_ev_a = nullptr, _tg_ev_b = nullptr; \
    const int  _tg_cat = (cat); \
    const bool _tg_on  = sclp_tg::enabled(); \
    if (_tg_on) { \
        (void)hipEventCreate(&_tg_ev_a); (void)hipEventCreate(&_tg_ev_b); \
        (void)hipEventRecord(_tg_ev_a, stream); \
    }
#define SCLP_TG_TIME_END() \
    if (_tg_on) { \
        (void)hipEventRecord(_tg_ev_b, stream); \
        (void)hipStreamSynchronize(stream); \
        float _tg_ms = 0.0f; \
        (void)hipEventElapsedTime(&_tg_ms, _tg_ev_a, _tg_ev_b); \
        sclp_tg::agg(_tg_cat).ns_total.fetch_add((uint64_t)(_tg_ms * 1.0e6), std::memory_order_relaxed); \
        sclp_tg::agg(_tg_cat).calls.fetch_add(1, std::memory_order_relaxed); \
        (void)hipEventDestroy(_tg_ev_a); (void)hipEventDestroy(_tg_ev_b); \
    }
inline int _sclp_tg_atexit_register = sclp_tg::register_atexit();

#define QK_SCLP 32

static __device__ __forceinline__ uint16_t float_to_bf16_dev(float f) {
    union { float f; uint32_t u; } x;
    x.f = f;
    return (uint16_t)(x.u >> 16);
}

// SCLP decode bridge for llama.cpp HIP backend.
//
// Wire format (SCLP blob stored in VRAM, all types share the same header):
//   [uint32 num_weights][uint32 n_experts]
//   [per-expert: uint8 palette_size, uint8 × palette_size palette] ...
//   [scales (num_weights / QK_SCLP * 2 bytes): fp16 scales]
//   [ws_stream (num_weights bytes): palette_idx(7:4) | smn(3:0)]
//   [uint32 sidecar_count]
//   [uint32 × sidecar_count sidecar_indices]
//   [uint16 × sidecar_count sidecar_values]
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
    __shared__ uint32_t sidecar_count_s;

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
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorized path: 4 entries per thread (32 bytes per thread: 16 index + 8 value + 8 output_write)
    uint32_t i = tid * 4;
    for (; i + 3 < sidecar_count_s; i += stride * 4) {
        uint4 idx4;
        uint64_t val4_64;
        __builtin_memcpy(&idx4,    idx_base + (uint64_t)i * sizeof(uint32_t), sizeof(uint4));
        __builtin_memcpy(&val4_64, val_base + (uint64_t)i * sizeof(uint16_t), sizeof(uint64_t));
        output[idx4.x] = (uint16_t)(val4_64 & 0xFFFF);
        output[idx4.y] = (uint16_t)((val4_64 >> 16) & 0xFFFF);
        output[idx4.z] = (uint16_t)((val4_64 >> 32) & 0xFFFF);
        output[idx4.w] = (uint16_t)(val4_64 >> 48);
    }

    // Scalar tail
    for (uint32_t si = i; si < sidecar_count_s; si++) {
        uint32_t idx;
        uint16_t val;
        __builtin_memcpy(&idx, idx_base + (uint64_t)si * sizeof(uint32_t), sizeof(uint32_t));
        __builtin_memcpy(&val, val_base + (uint64_t)si * sizeof(uint16_t), sizeof(uint16_t));
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
// Shared memory layout:
//   [0..1023]   s_lut[256] (float, 1024 bytes) — decoded float for each (pidx, smn) pair
//                                                 indexed as s_lut[byte] where byte = pidx<<4 | smn
//   [1024..]    s_x[K]     (float, K*4 bytes) — activation vector
__launch_bounds__(512, 2)
__global__ void sclp_fused_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    uint32_t N,
    uint32_t K
) {
    extern __shared__ char smem[];
    float* s_lut = (float*)smem;          // 256 floats = 1024 bytes
    float* s_x   = (float*)(smem + 1024); // K floats

    // Threads 0..15 build the LUT collaboratively (one palette entry each).
    // For GEMV, we assume n_experts=1 since it's the dense kernel.
    __shared__ uint32_t s_scales_start;
    __shared__ uint32_t s_ws_start;
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
        }
    }

    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) s_x[k] = x[k];
    __syncthreads();

    const uint16_t* scales = (const uint16_t*)(blob + s_scales_start);
    const uint8_t*  ws     = blob + s_ws_start;

    const int warps_per_block = blockDim.x / 32;
    const int warp_id  = threadIdx.x / 32;
    const int lane     = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    if (row >= N) return;

    float acc = 0.0f;
    const uint64_t row_base = (uint64_t)row * K;
    const uint32_t K8 = (K / 8) * 8;

    // 8 weights per uint64 load.
    for (uint32_t k8 = lane * 8; k8 < K8; k8 += 32 * 8) {
        uint64_t ws8; __builtin_memcpy(&ws8, ws + row_base + k8, sizeof(uint64_t));
        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (row_base + k8) / QK_SCLP));
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            acc += (s_lut[(uint8_t)(ws8 >> (j * 8))] * scale) * s_x[k8 + j];
        }
    }
    for (uint32_t k = K8 + lane; k < K; k += 32) {
        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (row_base + k) / QK_SCLP));
        acc += (s_lut[ws[row_base + k]] * scale) * s_x[k];
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) y[row] = acc;
    // Sidecar omitted per SCLP convention (~0.02% of weights; block-scoped scan
    // empirically regresses throughput ~37%).
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
    __shared__ uint32_t s_sidecar_count;
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

    const uint8_t* ws = blob + s_ws_start;
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
    const uint16_t* scales = (const uint16_t*)(blob + s_scales_start);

    uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < s_sidecar_count; i += stride) {
        uint32_t global_idx;
        uint16_t correct_bits;
        __builtin_memcpy(&global_idx, idx_base + (uint64_t)i * sizeof(uint32_t), sizeof(uint32_t));
        __builtin_memcpy(&correct_bits, val_base + (uint64_t)i * sizeof(uint16_t), sizeof(uint16_t));

        uint32_t n = global_idx / K;
        uint32_t k = global_idx % K;

        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (uint64_t)global_idx / QK_SCLP));

        uint8_t b     = ws[(uint64_t)n * K + k];
        uint8_t smn   = b & 0x0F;
        uint16_t approx_bits = ((uint16_t)(smn >> 3) << 15)
                             | ((uint16_t)s_palette[b >> 4] << 7)
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
                          | ((uint16_t)s_palette[b >> 4] << 7)
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
                      | ((uint16_t)s_palette[b >> 4] << 7)
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
    sclp_sidecar_correct_gemm_kernel<<<256, 256, 0, stream>>>(
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
    constexpr int WARPS_PER_BLOCK = 16;
    dim3 gemv_block(WARPS_PER_BLOCK * 32);
    dim3 gemv_grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    size_t smem_bytes = 1024 + (size_t)K * sizeof(float);
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
__global__ void sclp_fused_moe_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ src1,   // [K × n_active × n_batches]
    const int32_t* __restrict__ ids,    // [n_active × n_batches]
    float*         __restrict__ dst,    // [N × n_active × n_batches]
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t src1_ne1
) {
    __shared__ uint32_t s_scales_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint8_t  s_palette[16];

    // blockIdx.x = row group, blockIdx.y = flat index (i_active + i_batch * n_active)
    const uint32_t flat = blockIdx.y;
    const int32_t  e    = ids[flat];

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
    }
    __syncthreads();

    const uint16_t* scales = (const uint16_t*)(blob + s_scales_start);
    const uint8_t*  ws     = blob + s_ws_start;

    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;

    // blockIdx.x = row group, blockIdx.y = flat index (i_active + i_batch * n_active)
    const uint32_t row    = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    if (row >= N) return;

    const uint32_t i_active = flat % n_active;
    const uint32_t i_batch  = flat / n_active;

    // ws index for weight (k, row, e): k + row*K + e*N*K
    const uint64_t ws_base = (uint64_t)e * N * K + (uint64_t)row * K;
    const float* x = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;

    float acc0 = 0.0f, acc1 = 0.0f;
    const uint32_t K16 = (K / 16) * 16;

    for (uint32_t k16 = lane * 16; k16 < K16; k16 += 32 * 16) {
        float scale0 = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (ws_base + k16) / QK_SCLP));
        float scale1 = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (ws_base + k16 + 8) / QK_SCLP));
        {
            uint64_t ws8; __builtin_memcpy(&ws8, ws + ws_base + k16, sizeof(uint64_t));
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint8_t b     = (uint8_t)(ws8 >> (j * 8));
                uint8_t smn   = b & 0x0F;
                uint16_t bits = ((uint16_t)(smn >> 3) << 15)
                              | ((uint16_t)s_palette[b >> 4] << 7)
                              | ((uint16_t)(smn & 0x7) << 4);
                acc0 += (__bfloat162float(*(__hip_bfloat16*)&bits) * scale0) * x[k16 + j];
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
                acc1 += (__bfloat162float(*(__hip_bfloat16*)&bits) * scale1) * x[k16 + 8 + j];
            }
        }
    }
    float acc = acc0 + acc1;
    for (uint32_t k = K16 + lane; k < K; k += 32) {
        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (ws_base + k) / QK_SCLP));
        uint8_t b     = ws[ws_base + k];
        uint8_t smn   = b & 0x0F;
        uint16_t bits = ((uint16_t)(smn >> 3) << 15)
                      | ((uint16_t)s_palette[b >> 4] << 7)
                      | ((uint16_t)(smn & 0x7) << 4);
        acc += (__bfloat162float(*(__hip_bfloat16*)&bits) * scale) * x[k];
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
    SCLP_TG_TIME_BEGIN(sclp_tg::CAT_SCLP_MOE);
    sclp_fused_moe_gemv_kernel<<<grid, block, 0, stream>>>(
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
                               | ((uint16_t)s_palette[p_idx] << 7)
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
    sclp_sidecar_correct_gemm_kernel<<<256, 256, 0, stream>>>(
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
    __shared__ uint32_t s_n_experts;
    __shared__ uint32_t s_scales_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint8_t  s_palette_sizes[256];
    __shared__ uint8_t  s_palettes[256][4];
    __shared__ uint32_t s_expert_nw;
    __shared__ uint32_t s_expert_nibble_bytes;

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
        uint32_t enw = num_weights / ne;
        s_expert_nw = enw;
        s_expert_nibble_bytes = (enw + 1) / 2;
    }
    __syncthreads();

    const uint8_t* scales = blob + s_scales_start;
    const uint8_t* ws     = blob + s_ws_start;
    uint32_t expert_nw = s_expert_nw;

    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr uint32_t W_PER_THREAD = 32;
    const uint32_t base_idx   = thread_idx * W_PER_THREAD;

    if (base_idx >= num_weights) return;

    const uint32_t e          = base_idx / expert_nw;
    const uint32_t local_base = base_idx - e * expert_nw;
    if (e >= s_n_experts) return;

    const uint8_t pal0 = s_palettes[e][0];
    const uint8_t pal1 = s_palettes[e][1];
    const uint8_t pal2 = s_palettes[e][2];
    const uint8_t pal3 = s_palettes[e][3];
    const uint8_t pal_size = s_palette_sizes[e];

    const uint8_t* ws_p = ws + (uint64_t)e * s_expert_nibble_bytes + local_base / 2;
    const uint16_t* s_ptr = (const uint16_t*)(scales + (uint64_t)e * (expert_nw / QK_SCLP) * sizeof(uint16_t));
    uint32_t remaining_expert = expert_nw - local_base;
    uint32_t n_this = min(W_PER_THREAD, remaining_expert);

    uint16_t out[32];
    if (n_this == 32) {
        // Handle 32 weights (one QK_SCLP block) using 2x 64-bit vectorized loads
        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(s_ptr + (local_base / QK_SCLP)));
        for (int q = 0; q < 2; q++) {
            uint64_t ws8;
            __builtin_memcpy(&ws8, ws_p + q * 8, 8);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint8_t byte = (uint8_t)(ws8 >> (i * 8));
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    uint8_t nibble = (j == 0) ? (byte >> 4) : (byte & 0xF);
                    uint8_t pidx = nibble >> 2;
                    uint8_t smn  = nibble & 0x3;
                    uint8_t exp_ = 0;
                    if (pidx < pal_size) {
                        switch(pidx) {
                            case 0: exp_ = pal0; break;
                            case 1: exp_ = pal1; break;
                            case 2: exp_ = pal2; break;
                            case 3: exp_ = pal3; break;
                        }
                    }
                    uint8_t sign = (smn >> 1) & 1;
                    uint8_t mant = smn & 1;
                    uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp_ << 7) | ((uint16_t)mant << 6);
                    float w = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&bits)) * scale;
                    out[q * 16 + i * 2 + j] = float_to_bf16_dev(w);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint4 v;
            __builtin_memcpy(&v, out + i * 8, sizeof(uint4));
            __builtin_memcpy(output + base_idx + i * 8, &v, sizeof(uint4));
        }
    } else {
        for (uint32_t i = 0; i < n_this; ++i) {
            float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(s_ptr + ((local_base + i) / QK_SCLP)));
            uint8_t byte = ws_p[i / 2];
            uint8_t nibble = (i % 2 == 0) ? (byte >> 4) : (byte & 0xF);
            uint8_t pidx = nibble >> 2;
            uint8_t smn  = nibble & 0x3;
            uint8_t exp_ = 0;
            if (pidx < pal_size) {
                switch(pidx) {
                    case 0: exp_ = pal0; break;
                    case 1: exp_ = pal1; break;
                    case 2: exp_ = pal2; break;
                    case 3: exp_ = pal3; break;
                }
            }
            uint8_t sign = (smn >> 1) & 1;
            uint8_t mant = smn & 1;
            uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp_ << 7) | ((uint16_t)mant << 6);
            float w = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&bits)) * scale;
            output[base_idx + i] = float_to_bf16_dev(w);
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
        uint32_t scales_start = (uint32_t)(p - blob);
        s_ws_start = scales_start + (num_weights / QK_SCLP) * sizeof(uint16_t);
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
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t i = tid * 4;
    for (; i + 3 < sidecar_count_s; i += stride * 4) {
        uint4 idx4;
        uint64_t val4_64;
        __builtin_memcpy(&idx4,    idx_base + (uint64_t)i * sizeof(uint32_t), sizeof(uint4));
        __builtin_memcpy(&val4_64, val_base + (uint64_t)i * sizeof(uint16_t), sizeof(uint64_t));
        output[idx4.x] = (uint16_t)(val4_64 & 0xFFFF);
        output[idx4.y] = (uint16_t)((val4_64 >> 16) & 0xFFFF);
        output[idx4.z] = (uint16_t)((val4_64 >> 32) & 0xFFFF);
        output[idx4.w] = (uint16_t)(val4_64 >> 48);
    }

    for (uint32_t si = i; si < sidecar_count_s; si++) {
        uint32_t idx;
        uint16_t val;
        __builtin_memcpy(&idx, idx_base + (uint64_t)si * sizeof(uint32_t), sizeof(uint32_t));
        __builtin_memcpy(&val, val_base + (uint64_t)si * sizeof(uint16_t), sizeof(uint16_t));
        output[idx] = val;
    }
}

// Fused decode-GEMV for SCLP4, M=1 (dense).
// Mirrors the MoE GEMV structure: cooperative LDS load of activation vector + 4×4 decode LUT.
// Previous per-warp-reads-x version left ~17× HBM headroom: 32 warps/block all re-read
// the same x[k] from global memory, costing ~K*32*4 redundant bytes per block.
// Shared memory layout:
//   [0..63]   s_lut[16]  (float, 64 bytes) — decoded float for each (pidx, smn) pair
//   [64..]    s_x[K]     (float, K*4 bytes) — activation vector
__launch_bounds__(512, 2)
__global__ void sclp4_fused_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    uint32_t N,
    uint32_t K
) {
    extern __shared__ char smem[];
    __shared__ uint32_t s_scales_start;
    __shared__ uint32_t s_ws_start;
    float* s_lut = (float*)smem;          // 16 floats = 64 bytes
    float* s_x   = (float*)(smem + 64);   // K floats

    // Thread 0: parse header (n_experts=1 for dense), build 4×4 decode LUT.
    if (threadIdx.x == 0) {
        uint32_t ne; __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        uint8_t pal_size = p[0];
        uint8_t pal[4];
        for (int i = 0; i < (int)pal_size; i++) pal[i] = p[1 + i];
        for (int pidx = 0; pidx < 4; pidx++) {
            uint8_t exp_ = (pidx < (int)pal_size) ? pal[pidx] : 0;
            for (int smn = 0; smn < 4; smn++) {
                uint8_t sign = (smn >> 1) & 1;
                uint8_t mant = smn & 1;
                uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp_ << 7) | ((uint16_t)mant << 6);
                s_lut[pidx * 4 + smn] = __bfloat162float(*(__hip_bfloat16*)&bits);
            }
        }
        for (uint32_t ei = 0; ei < ne; ei++) p += 1 + p[0];
        s_scales_start = (uint32_t)(p - blob);
        s_ws_start = s_scales_start + (((uint64_t)N * K) / QK_SCLP) * sizeof(uint16_t);
    }

    // Cooperative LDS load of x[K]. One global read per element across the whole block,
    // not per warp — saves (warps_per_block - 1) × K × 4 bytes per block of redundant traffic.
    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) {
        s_x[k] = x[k];
    }
    __syncthreads();

    const uint16_t* scales = (const uint16_t*)(blob + s_scales_start);
    const uint8_t*  ws     = blob + s_ws_start;

    const int warps_per_block = blockDim.x / 32;
    const int warp_id  = threadIdx.x / 32;
    const int lane     = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    if (row >= N) return;

    float acc = 0.0f;
    const uint32_t n_bytes_row    = (K + 1) / 2;
    const uint64_t row_byte_base  = (uint64_t)row * n_bytes_row;
    const uint64_t row_weight_base = (uint64_t)row * K;

    for (uint32_t b = lane; b < n_bytes_row; b += 32) {
        uint8_t byte = ws[row_byte_base + b];
        uint32_t k0  = b * 2;
        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (row_weight_base + k0) / QK_SCLP));
        acc += (s_lut[byte >> 4] * scale) * s_x[k0];
        if (k0 + 1 < K) {
            acc += (s_lut[byte & 0xF] * scale) * s_x[k0 + 1];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) y[row] = acc;
}

inline void llama_sclp4_dispatch(
    const void* sclp_data,
    uint16_t*   output,
    uint32_t    num_weights,
    hipStream_t stream,
    bool        apply_sidecar = true
) {
    const uint8_t* data = (const uint8_t*)sclp_data;
    dim3 block(256);

    // Each thread handles 32 weights (16 nibble bytes).
    uint32_t groups = (num_weights + 31) / 32;
    dim3 decode_grid((groups + 255) / 256);

    static const bool time_decode = getenv("SCLP_TIME_DECODE") != nullptr;
    hipEvent_t ev_a = nullptr, ev_b = nullptr, ev_c = nullptr;
    if (time_decode) {
        (void)hipEventCreate(&ev_a);
        (void)hipEventCreate(&ev_b);
        (void)hipEventCreate(&ev_c);
        (void)hipEventRecord(ev_a, stream);
    }

    sclp4_decode_blob_kernel<<<decode_grid, block, 0, stream>>>(data, output, num_weights);

    if (time_decode) (void)hipEventRecord(ev_b, stream);

    // Increase grid size to 1024 blocks to saturate all 96 CUs on gfx1100.
    if (apply_sidecar) {
        sclp4_fixup_sidecar_kernel<<<1024, block, 0, stream>>>(data, output, num_weights);
    }

    if (time_decode) {
        (void)hipEventRecord(ev_c, stream);
        (void)hipStreamSynchronize(stream);
        float ms_dec = 0.0f, ms_sc = 0.0f;
        (void)hipEventElapsedTime(&ms_dec, ev_a, ev_b);
        (void)hipEventElapsedTime(&ms_sc,  ev_b, ev_c);
        static int dec_call = 0;
        if (dec_call < 10) {
            fprintf(stderr, "[DEC] call=%d nw=%u  decode=%.3fms  sidecar_fixup=%.3fms\n",
                    dec_call, num_weights, ms_dec, ms_sc);
            dec_call++;
        }
        (void)hipEventDestroy(ev_a);
        (void)hipEventDestroy(ev_b);
        (void)hipEventDestroy(ev_c);
    }
}

inline void llama_sclp4_fused_gemv(
    const void*   blob_ptr,
    const float*  src_f32,
    float*        dst_f32,
    uint32_t      N,
    uint32_t      K,
    hipStream_t   stream
) {
    // 16 warps/block matches the MoE variant. Dynamic shared mem: 64 (LUT) + K*4 (activations).
    // Max K for RDNA3 LDS (64 KB): ~16000. Llama-3 ffn K=14336 fits with headroom.
    constexpr int WARPS_PER_BLOCK = 16;
    dim3 gemv_block(WARPS_PER_BLOCK * 32);
    dim3 gemv_grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    size_t smem_bytes = 64 + (size_t)K * sizeof(float);
    SCLP_TG_TIME_BEGIN(sclp_tg::CAT_SCLP4_DENSE);
    sclp4_fused_gemv_kernel<<<gemv_grid, gemv_block, smem_bytes, stream>>>(
        (const uint8_t*)blob_ptr,
        src_f32,
        dst_f32, N, K);
    SCLP_TG_TIME_END();
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
    //   [0..63]  s_lut[16]   (float, 64 bytes) — decoded float for each (pidx, smn) pair
    //   [64..]   s_x[K]      (float, K*4 bytes) — activation vector
    extern __shared__ char smem[];
    __shared__ uint32_t s_scales_offset;
    __shared__ uint32_t s_ws_offset;
    float*    s_lut       = (float*)smem;          // 16 floats = 64 bytes
    float*    s_x         = (float*)(smem + 64);   // K floats

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
        uint32_t expert_nw = total_nw / n_experts;
        uint32_t scales_start = (uint32_t)(p - blob);
        s_scales_offset = scales_start + (uint32_t)e * (expert_nw / QK_SCLP) * sizeof(uint16_t);

        uint32_t ws_start  = scales_start + (total_nw / QK_SCLP) * sizeof(uint16_t);
        uint32_t expert_nibble_bytes = (expert_nw + 1) / 2;
        s_ws_offset = ws_start + (uint32_t)e * expert_nibble_bytes;

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

    const uint16_t* scales = (const uint16_t*)(blob + s_scales_offset);
    const uint8_t* ws = blob + s_ws_offset;
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
    const uint64_t row_weight_base = (uint64_t)row * K;

    for (uint32_t b = lane; b < n_bytes_row; b += 32) {
        uint8_t byte = ws[row_byte_base + b];
        uint32_t k0  = b * 2;
        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (row_weight_base + k0) / QK_SCLP));
        acc += (s_lut[byte >> 4] * scale) * s_x[k0];
        if (k0 + 1 < K) {
            acc += (s_lut[byte & 0xF] * scale) * s_x[k0 + 1];
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
    // Dynamic shared memory: 64 (lut) + K*4 (activations)
    size_t smem_bytes = 64 + (size_t)K * sizeof(float);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, n_active * n_batches);
    SCLP_TG_TIME_BEGIN(sclp_tg::CAT_SCLP4_MOE);
    sclp4_fused_moe_gemv_kernel<<<grid, block, smem_bytes, stream>>>(
        (const uint8_t*)blob_ptr,
        src1, ids, dst, N, K, n_active, src1_ne1);
    SCLP_TG_TIME_END();
}

// ============================================================
// SCLP4 fused MoE WMMA prefill (M > 1)
// ============================================================
//
// Goal: eliminate the two-pass decode → rocBLAS path for SCLP4 MoE prefill.
// The current path materializes the full BF16 expert tensor (~6 GB on Gemma4)
// and feeds rocBLAS, which then issues many small per-expert GEMMs.
//
// Strategy:
//   1. Counting-sort kernel bins flat_slot indices (i_active + i_batch*n_active)
//      by their expert id. Produces:
//        perm[total_slots]              — flat_slots sorted by expert
//        expert_offsets[n_experts+1]    — exclusive prefix sum
//   2. Each block handles one (n_tile=16 × m_tile=16) output tile mapped to
//      a single (expert_id, intra-expert m_group). All 16 m positions in the
//      tile share an expert ⇒ single A fragment per K iteration is valid.
//   3. Activations are gathered through perm; outputs are scattered through perm.
//
// Status: WIP (May 2026), default off. Behind these env vars in ggml-cuda.cu:
//   SCLP_FUSED_MOE_WMMA=1       use fused kernel; baseline two-pass otherwise
//   SCLP_FUSED_MOE_WMMA_DIFF=1  run BOTH; compare dst values; print stats
//   SCLP_FUSED_MOE_SCALAR=1     use the scalar reference kernel (slower)
//   SCLP_FUSED_MOE_NO_SIDECAR=1 skip the sidecar correction kernel
//
// Current status (May 2026):
//   1. Route-sort / slot coverage is validated:
//      - sentinel test shows 0 unwritten cells
//      - probe modes confirm mg/i_batch/expert resolution are correct
//   2. Sidecar is not the source of the remaining drift:
//      - SCLP_FUSED_MOE_SIDECAR_MODE=0/1 produces essentially identical
//        prefill drift metrics on ids ne=[8,512]
//      - [SCLP_SPLIT] pre_vs_ref and post_vs_pre are very close
//   3. Nondeterminism is not the source:
//      - [SCLP_REPEAT] fused_run2_vs_run1 remains tiny (>1e-4 = 0)
//   4. Remaining delta is in core GEMM numerics/order:
//      - [SCLP_CORE] fused_pre_vs_ref_pre closely matches [SCLP_DIFF]
//        on the same call/shape, so sidecar and coverage are ruled out.
//
// Practical impact observed in current runs (Gemma4 MIXED sidecar1%, pp512):
//   - call-level drift on large prefill shapes is small but systematic
//     (order ~1e-3 mean abs, ~1e-1 max)
//   - end-to-end PPL remains stable at ~9657.355 for the tested config.
//
// Working hypothesis:
//   fused scalar accumulation order differs from the two-pass path's effective
//   math path (decode + backend GEMM), producing deterministic numeric drift.
//
// Next triage target:
//   add a two-pass-core repeat baseline (same inputs/path twice) to quantify the
//   reference noise floor, then decide whether tighter numeric matching is worth
//   the performance/complexity cost.

__global__ void sclp_moe_route_sort_kernel(
    const int32_t* __restrict__ ids,            // [n_active × n_batches]
    int32_t*       __restrict__ perm,           // [n_experts * TILE + total_slots] out — bins padded to TILE
    int32_t*       __restrict__ expert_offsets, // [n_experts + 1] out, padded prefix sum
    uint32_t                    total_slots,
    uint32_t                    n_active,
    uint32_t                    ids_s1,
    uint32_t                    n_experts,
    uint32_t                    tile            // TILE size each bin is padded up to
) {
    // Counting sort with per-expert bins padded to multiples of TILE so the WMMA
    // GEMM kernel's m-tile boundaries align to single experts. Empty padding
    // entries are filled with -1 sentinel; the GEMM kernel zeros their contributions.
    extern __shared__ int32_t s_smem[];
    int32_t* s_count  = s_smem;                       // [n_experts + 1]
    int32_t* s_cursor = s_smem + (n_experts + 1);     // [n_experts + 1] write cursor copy

    for (uint32_t e = threadIdx.x; e <= n_experts; e += blockDim.x) {
        s_count[e]  = 0;
        s_cursor[e] = 0;
    }
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < total_slots; i += blockDim.x) {
        const uint32_t i_active = i % n_active;
        const uint32_t i_batch  = i / n_active;
        int32_t e = ids[(uint64_t)i_batch * ids_s1 + i_active];
        if (e >= 0 && (uint32_t)e < n_experts) atomicAdd(&s_count[e], 1);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int32_t acc = 0;
        for (uint32_t e = 0; e < n_experts; e++) {
            expert_offsets[e] = acc;
            s_cursor[e]       = acc;
            int32_t c = s_count[e];
            int32_t padded = (c + (int32_t)tile - 1) & ~((int32_t)tile - 1);
            acc += padded;
        }
        expert_offsets[n_experts] = acc;
        s_cursor[n_experts] = acc;
    }
    __syncthreads();

    // Pre-fill the padded perm region with -1 (sentinel).
    int32_t total_padded = expert_offsets[n_experts];
    for (int32_t i = (int32_t)threadIdx.x; i < total_padded; i += (int32_t)blockDim.x) {
        perm[i] = -1;
    }
    __syncthreads();

    // Scatter real slot indices into their expert's bin head.
    for (uint32_t i = threadIdx.x; i < total_slots; i += blockDim.x) {
        const uint32_t i_active = i % n_active;
        const uint32_t i_batch  = i / n_active;
        int32_t e = ids[(uint64_t)i_batch * ids_s1 + i_active];
        if (e >= 0 && (uint32_t)e < n_experts) {
            int32_t pos = atomicAdd(&s_cursor[e], 1);
            perm[pos] = (int32_t)i;
        }
    }
}

// Fused decode-WMMA kernel for SCLP4 MoE prefill.
// Each block computes a 16(N) × 16(M) output tile for a single expert.
// Grid layout: blockIdx.x = n_tile (output row group), blockIdx.y = global m_tile.
// The m_tile index is mapped to (expert, intra-expert offset) via expert_offsets.
//
// Single-warp-per-block first version: 32 lanes, one 16×16 WMMA tile per block.
// Future: 2×2 warp tile like sclp_fused_wmma_kernel for 4× throughput per block.
__launch_bounds__(32, 8)
__global__ void sclp4_fused_moe_wmma_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ src1,             // [K × src1_ne1 × n_batches]
    const int32_t* __restrict__ perm,             // [total_slots] sorted by expert
    const int32_t* __restrict__ expert_offsets,   // [n_experts+1] prefix sums
    const int32_t* __restrict__ ids,              // [n_active × n_batches]
    float*         __restrict__ dst,              // [N × n_active × n_batches]
    uint32_t N, uint32_t K,
    uint32_t n_active, uint32_t src1_ne1, uint32_t n_experts
) {
    constexpr int TILE = 16;

    // Locate this block's expert via binary search on expert_offsets.
    const uint32_t m_tile = blockIdx.y;
    const uint32_t m_base = m_tile * TILE;
    int32_t e = -1;
    // Linear search is fine for n_experts ≤ 128. Could replace with binary.
    for (uint32_t ei = 0; ei < n_experts; ei++) {
        if ((uint32_t)expert_offsets[ei] <= m_base && m_base < (uint32_t)expert_offsets[ei + 1]) {
            e = (int32_t)ei;
            break;
        }
    }
    if (e < 0) return;  // m_base past end of sorted list

    // Parse blob header on-device to find this expert's ws section + palette.
    __shared__ uint8_t s_palette[4];
    __shared__ uint8_t s_palette_size;
    __shared__ uint32_t s_scales_offset;
    __shared__ uint32_t s_ws_offset;
    __shared__ int32_t  s_m_global[TILE];

    if (threadIdx.x == 0) {
        uint32_t total_nw, ne;
        __builtin_memcpy(&total_nw, blob,     sizeof(uint32_t));
        __builtin_memcpy(&ne,       blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        uint8_t pal[4] = {0,0,0,0};
        uint8_t psz = 0;
        for (uint32_t ei = 0; ei < ne; ei++) {
            if (ei == (uint32_t)e) {
                psz = p[0];
                for (int i = 0; i < (int)p[0]; i++) pal[i] = p[1 + i];
            }
            p += 1 + p[0];
        }
        s_palette_size = psz;
        for (int i = 0; i < 4; i++) s_palette[i] = pal[i];
        uint32_t expert_nw = total_nw / ne;
        uint32_t scales_start = (uint32_t)(p - blob);
        s_scales_offset = scales_start + (uint32_t)e * (expert_nw / QK_SCLP) * sizeof(uint16_t);
        uint32_t ws_start = scales_start + (total_nw / QK_SCLP) * sizeof(uint16_t);
        uint32_t expert_nibble_bytes = (expert_nw + 1) / 2;
        s_ws_offset = ws_start + (uint32_t)e * expert_nibble_bytes;
    }

    // Each lane loads one m index from perm. Padding-aware: slots past this
    // expert's bin end carry the -1 sentinel set by the routing-sort kernel.
    if (threadIdx.x < TILE) {
        uint32_t m_g = m_base + threadIdx.x;
        int32_t pe = perm[m_g];   // -1 for padded entries
        s_m_global[threadIdx.x] = pe;
    }
    __syncthreads();

    const uint16_t* scales = (const uint16_t*)(blob + s_scales_offset);
    const uint8_t* ws = blob + s_ws_offset;
    const uint32_t n_bytes_row = (K + 1) / 2;

    const int lane     = threadIdx.x;
    const int frag_row = lane % TILE;       // 0..15 (lanes 0..15 and 16..31 mirror)
    const uint32_t n_base = (uint32_t)blockIdx.x * TILE;
    const uint32_t n_row  = n_base + (uint32_t)frag_row;

    using bf16x16_t = __attribute__((ext_vector_type(16))) __bf16;
    using floatx8_t = __attribute__((ext_vector_type(8)))  float;
    floatx8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (uint32_t k16 = 0; k16 < K; k16 += TILE) {
        // ── A fragment: 16 BF16 weights from row n_row, columns k16..k16+15.
        bf16x16_t a_frag;
        #pragma unroll
        for (int j = 0; j < TILE; j++) {
            const uint32_t kg = k16 + j;
            if (n_row < N && kg < K) {
                uint64_t w_idx = (uint64_t)n_row * K + kg;
                float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + w_idx / QK_SCLP));
                uint8_t byte  = ws[w_idx >> 1];
                uint8_t nib   = (w_idx & 1) ? (byte & 0xF) : (byte >> 4);
                uint8_t pidx  = nib >> 2;
                uint8_t smn   = nib & 0x3;
                uint8_t exp_  = (pidx < s_palette_size) ? s_palette[pidx] : 0;
                uint8_t sign  = (smn >> 1) & 1;
                uint8_t mant  = smn & 1;
                uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp_ << 7) | ((uint16_t)mant << 6);
                float w = __bfloat162float(*(__hip_bfloat16*)&bits) * scale;
                a_frag[j] = __float2bfloat16(w);
            } else {
                a_frag[j] = (__bf16)0.f;
            }
        }

        // ── B fragment: B[frag_row][j] = X[m_base+j][k16+frag_row].
        //   X is row-major [n_active × n_batches × K] indexed by perm[m].
        bf16x16_t b_frag;
        const uint32_t kg = k16 + (uint32_t)frag_row;
        #pragma unroll
        for (int j = 0; j < TILE; j++) {
            int32_t mg = s_m_global[j];
            if (mg >= 0 && kg < K) {
                uint32_t i_active = (uint32_t)mg % n_active;
                uint32_t i_batch  = (uint32_t)mg / n_active;
                const float* x_row = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;
                __hip_bfloat16 v = (__hip_bfloat16)x_row[kg];
                b_frag[j] = *(const __bf16*)&v;
            } else {
                __hip_bfloat16 z = (__hip_bfloat16)0.f;
                b_frag[j] = *(const __bf16*)&z;
            }
        }

        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc);
    }

    // ── Write output. RDNA3 C layout: thread t holds D[t%16][2*l + t/16] for l=0..7.
    if (n_row < N) {
        const int col_lo = lane / TILE;  // 0 for lanes 0..15, 1 for lanes 16..31
        #pragma unroll
        for (int l = 0; l < 8; l++) {
            const uint32_t m_local = (uint32_t)(2 * l + col_lo);
            if (m_local < TILE) {
                int32_t mg = s_m_global[m_local];
                if (mg >= 0) {
                    dst[(uint64_t)mg * N + n_row] = acc[l];
                }
            }
        }
    }
    (void)ids;
}

// Scalar reference variant of sclp4_fused_moe_wmma_kernel. Uses the IDENTICAL
// routing/indexing/gather logic but replaces the WMMA inner product with a per-lane
// scalar accumulator. If this produces correct PPL while the WMMA variant doesn't,
// the bug is in WMMA fragment layout. If both are wrong, the bug is in routing/gather.
__launch_bounds__(32, 4)
__global__ void sclp4_fused_moe_scalar_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ src1,
    const int32_t* __restrict__ perm,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ ids,
    float*         __restrict__ dst,
    uint32_t N, uint32_t K,
    uint32_t n_active, uint32_t src1_ne1, uint32_t n_experts, uint32_t math_mode
) {
    constexpr int TILE = 16;
    const uint32_t m_tile = blockIdx.y;
    const uint32_t m_base = m_tile * TILE;

    int32_t e = -1;
    for (uint32_t ei = 0; ei < n_experts; ei++) {
        if ((uint32_t)expert_offsets[ei] <= m_base && m_base < (uint32_t)expert_offsets[ei + 1]) {
            e = (int32_t)ei;
            break;
        }
    }
    if (e < 0) return;

    __shared__ uint8_t  s_palette[4];
    __shared__ uint8_t  s_palette_size;
    __shared__ uint32_t s_scales_offset;
    __shared__ uint32_t s_ws_offset;
    __shared__ int32_t  s_m_global[TILE];

    if (threadIdx.x == 0) {
        uint32_t total_nw, ne;
        __builtin_memcpy(&total_nw, blob,     sizeof(uint32_t));
        __builtin_memcpy(&ne,       blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        uint8_t pal[4] = {0,0,0,0};
        uint8_t psz = 0;
        for (uint32_t ei = 0; ei < ne; ei++) {
            if (ei == (uint32_t)e) {
                psz = p[0];
                for (int i = 0; i < (int)p[0]; i++) pal[i] = p[1 + i];
            }
            p += 1 + p[0];
        }
        s_palette_size = psz;
        for (int i = 0; i < 4; i++) s_palette[i] = pal[i];
        uint32_t expert_nw = total_nw / ne;
        uint32_t scales_start = (uint32_t)(p - blob);
        s_scales_offset = scales_start + (uint32_t)e * (expert_nw / QK_SCLP) * sizeof(uint16_t);
        uint32_t ws_start = scales_start + (total_nw / QK_SCLP) * sizeof(uint16_t);
        uint32_t expert_nibble_bytes = (expert_nw + 1) / 2;
        s_ws_offset = ws_start + (uint32_t)e * expert_nibble_bytes;
    }
    if (threadIdx.x < TILE) {
        uint32_t m_g = m_base + threadIdx.x;
        s_m_global[threadIdx.x] = perm[m_g];
    }
    __syncthreads();

    const uint16_t* scales = (const uint16_t*)(blob + s_scales_offset);
    const uint8_t* ws = blob + s_ws_offset;
    const int lane = threadIdx.x;

    // Each lane computes 8 output cells using the SAME lane→(n_row, m_local) mapping
    // as the WMMA variant: n_row = n_base + lane%16, m_local ∈ {2l+col_lo : l=0..7}.
    const int frag_row = lane % TILE;
    const int col_lo   = lane / TILE;
    const uint32_t n_base = (uint32_t)blockIdx.x * TILE;
    const uint32_t n_row  = n_base + (uint32_t)frag_row;
    if (n_row >= N) return;

    // Decode this lane's row of weights into a __bf16 buffer (one row of K weights).
    // Then dot it with each of the 8 assigned activations.
    // For correctness we just compute scalar dot products on the fly.
    float acc[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    float acc_c[8] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f}; // Kahan compensation
    int32_t mgs[8];
    bool    valid[8];
    for (int l = 0; l < 8; l++) {
        uint32_t m_local = (uint32_t)(2 * l + col_lo);
        mgs[l]   = s_m_global[m_local];
        valid[l] = (mgs[l] >= 0);
    }

    // Simplest possible loop: no caches, recompute row pointer each access. This
    // rules out any caching/aliasing in x_rows[] as a cause of the prefill-shape bug.
    for (uint32_t k = 0; k < K; k++) {
        uint64_t w_idx = (uint64_t)n_row * K + k;
        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + w_idx / QK_SCLP));
        uint8_t byte  = ws[w_idx >> 1];
        uint8_t nib   = (w_idx & 1) ? (byte & 0xF) : (byte >> 4);
        uint8_t pidx  = nib >> 2;
        uint8_t smn   = nib & 0x3;
        uint8_t exp_  = (pidx < s_palette_size) ? s_palette[pidx] : 0;
        uint8_t sign  = (smn >> 1) & 1;
        uint8_t mant  = smn & 1;
        uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp_ << 7) | ((uint16_t)mant << 6);
        float w = __bfloat162float(*(__hip_bfloat16*)&bits) * scale;

        for (int l = 0; l < 8; l++) {
            if (!valid[l]) continue;
            uint32_t i_active = (uint32_t)mgs[l] % n_active;
            uint32_t i_batch  = (uint32_t)mgs[l] / n_active;
            const float* x_row = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;
            float xf = x_row[k];
            if (math_mode & 2u) {
                // Optional: force activation to BF16 to test sensitivity to input precision.
                __hip_bfloat16 xbf = (__hip_bfloat16)xf;
                xf = __bfloat162float(xbf);
            }
            const float prod = w * xf;
            if (math_mode & 1u) {
                // Kahan-compensated accumulation for numerical-order triage.
                float y = prod - acc_c[l];
                float t = acc[l] + y;
                acc_c[l] = (t - acc[l]) - y;
                acc[l] = t;
            } else {
                acc[l] += prod;
            }
        }
    }

    // SCLP_FUSED_MOE_PROBE: replace acc with diagnostic values to verify indexing.
    // mode=1: write i_batch  → expect dst[mg*N+n] == mg/n_active
    // mode=2: write i_active → expect dst[mg*N+n] == mg%n_active
    // mode=3: write mgs[l] (the perm-resolved slot index) — expect dst[mg*N+n] == mg
    // mode=4: write x_row[0] (activation src1[i_batch*ne11*K + (i_active%ne11)*K + 0])
    // mode=5: write x_row[K-1] (last activation in the row)
#ifdef SCLP_PROBE_IBATCH
    for (int l = 0; l < 8; l++) {
        if (!valid[l]) continue;
        uint32_t i_active = (uint32_t)mgs[l] % n_active;
        uint32_t i_batch  = (uint32_t)mgs[l] / n_active;
#if SCLP_PROBE_IBATCH == 1
        acc[l] = (float)i_batch;
#elif SCLP_PROBE_IBATCH == 2
        acc[l] = (float)i_active;
#elif SCLP_PROBE_IBATCH == 3
        acc[l] = (float)mgs[l];
#elif SCLP_PROBE_IBATCH == 4
        const float* xrp = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;
        acc[l] = xrp[0];
#elif SCLP_PROBE_IBATCH == 5
        const float* xrp = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;
        acc[l] = xrp[K - 1];
#elif SCLP_PROBE_IBATCH == 6
        acc[l] = (float)e;
#elif SCLP_PROBE_IBATCH == 7
        // Read first nibble of expert e's weight slab and decode it. Should be consistent
        // across all cells of the same expert; varies across experts.
        uint8_t b0 = ws[0];
        acc[l] = (float)(b0 >> 4);
#elif SCLP_PROBE_IBATCH == 8
        // Honest dot product with manual ids lookup: w_e is from ids[mg], not block-resolved e.
        acc[l] = (float)ids[mgs[l]];
#endif
        (void)i_active; (void)i_batch;
    }
#endif

    for (int l = 0; l < 8; l++) {
        if (valid[l]) dst[(uint64_t)mgs[l] * N + n_row] = acc[l];
    }
}

// Sidecar correction for SCLP4 MoE fused output.
// For each sidecar weight (n_e, k) in expert e, computes delta = correct - approx
// and atomically adds delta * X[mg, k] to Y[mg, n_e] for every routed slot mg
// whose ids[mg] == e. Without this, the fused kernel undershoots magnitude by
// ~5-15% (the sidecar weights are systematically larger than the palette
// approximation, since they were placed in sidecar precisely because of large
// exponent-distance to the palette).
__global__ void sclp4_moe_sidecar_correct_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ src1,
    const int32_t* __restrict__ perm,
    const int32_t* __restrict__ expert_offsets,
    float*         __restrict__ dst,
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t src1_ne1, uint32_t n_experts,
    uint32_t sidecar_mode
) {
    __shared__ uint32_t s_total_nw;
    __shared__ uint32_t s_n_experts;
    __shared__ uint32_t s_scales_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint32_t s_sidecar_count;
    // Per-expert palette (cached for first 4 experts in shared mem if needed).
    if (threadIdx.x == 0) {
        uint32_t total_nw, ne;
        __builtin_memcpy(&total_nw, blob,     sizeof(uint32_t));
        __builtin_memcpy(&ne,       blob + 4, sizeof(uint32_t));
        s_total_nw = total_nw;
        s_n_experts = ne;
        const uint8_t* p = blob + 8;
        for (uint32_t e = 0; e < ne; e++) p += 1 + p[0];
        uint32_t scales_start = (uint32_t)(p - blob);
        s_scales_start = scales_start;
        uint32_t ws_start = scales_start + (total_nw / QK_SCLP) * sizeof(uint16_t);
        s_ws_start = ws_start;
        uint32_t expert_nw = total_nw / ne;
        uint32_t expert_nibble_bytes = (expert_nw + 1) / 2;
        uint64_t total_ws_bytes = (uint64_t)ne * expert_nibble_bytes;
        uint32_t sc;
        __builtin_memcpy(&sc, blob + ws_start + total_ws_bytes, sizeof(uint32_t));
        s_sidecar_count = sc;
    }
    __syncthreads();

    if (s_sidecar_count == 0) return;

    const uint32_t expert_nw = s_total_nw / s_n_experts;
    const uint32_t expert_nibble_bytes = (expert_nw + 1) / 2;
    const uint64_t total_ws_bytes = (uint64_t)s_n_experts * expert_nibble_bytes;
    const uint8_t* sidecar_base = blob + s_ws_start + total_ws_bytes;
    const uint8_t* idx_base = sidecar_base + 4;
    const uint8_t* val_base = idx_base + (uint64_t)s_sidecar_count * sizeof(uint32_t);
    const uint16_t* scales = (const uint16_t*)(blob + s_scales_start);

    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < s_sidecar_count; i += stride) {
        uint32_t global_idx;
        uint16_t correct_bits;
        __builtin_memcpy(&global_idx, idx_base + (uint64_t)i * sizeof(uint32_t), sizeof(uint32_t));
        __builtin_memcpy(&correct_bits, val_base + (uint64_t)i * sizeof(uint16_t), sizeof(uint16_t));

        uint32_t e        = global_idx / expert_nw;
        uint32_t local    = global_idx - e * expert_nw;
        uint32_t n_e      = local / K;
        uint32_t k        = local % K;

        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + global_idx / QK_SCLP));

        // Re-parse this expert's palette and the approximate nibble value at (n_e, k).
        const uint8_t* p = blob + 8;
        for (uint32_t ei = 0; ei < e; ei++) p += 1 + p[0];
        uint8_t pal_size = p[0];
        // SCLP4 palette is <=4 bytes; read inline.
        uint8_t pal[4] = {0,0,0,0};
        for (int j = 0; j < (int)pal_size && j < 4; j++) pal[j] = p[1 + j];

        const uint8_t* ws_e = blob + s_ws_start + (uint64_t)e * expert_nibble_bytes;
        uint64_t w_idx = (uint64_t)n_e * K + k;
        uint8_t byte = ws_e[w_idx >> 1];
        uint8_t nib  = (w_idx & 1) ? (byte & 0xF) : (byte >> 4);
        uint8_t pidx = nib >> 2;
        uint8_t smn  = nib & 0x3;
        uint8_t exp_a = (pidx < pal_size) ? pal[pidx] : 0;
        uint8_t sign_a = (smn >> 1) & 1;
        uint8_t mant_a = smn & 1;
        uint16_t approx_bits = ((uint16_t)sign_a << 15) | ((uint16_t)exp_a << 7) | ((uint16_t)mant_a << 6);

        float w_correct = __bfloat162float(*(__hip_bfloat16*)&correct_bits);
        float w_approx  = __bfloat162float(*(__hip_bfloat16*)&approx_bits) * scale;
        float w_delta   = w_correct - w_approx;
        if (w_delta == 0.f) continue;

        // For every routed slot mg whose ids[mg] == e (i.e., whose perm position is
        // in [offsets[e], offsets[e+1])), apply correction.
        int32_t bin_start = expert_offsets[e];
        int32_t bin_end   = expert_offsets[e + 1];
        for (int32_t p_pos = bin_start; p_pos < bin_end; p_pos++) {
            int32_t mg = perm[p_pos];
            if (mg < 0) continue;
            uint32_t i_active = (uint32_t)mg % n_active;
            uint32_t i_batch  = (uint32_t)mg / n_active;
            const float* x_row = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;
            float xval = x_row[k];
            if (sidecar_mode & 1u) {
                __hip_bfloat16 xbf = (__hip_bfloat16)xval;
                xval = __bfloat162float(xbf);
            }
            atomicAdd(&dst[(uint64_t)mg * N + n_e], w_delta * xval);
        }
    }
}

// Blocked sidecar correction. One block per fused-output tile (n_tile, m_tile).
// Each block:
//   1. Resolves its expert e from m_base (same as fused).
//   2. Cooperatively scans the sidecar list, filtering for (expert==e AND
//      n_e in [n_base, n_base+TILE)). Stores matches in LDS.
//   3. For each (m_local, n_e) cell it owns, scans LDS matches and accumulates
//      the correction. Non-atomic dst[mg, n_e] += correction.
// Since each (mg, n_e) cell maps to exactly one (n_tile, m_tile), no atomic.
// Replaces sclp4_moe_sidecar_correct_kernel's per-sidecar atomicAdd loop, which
// was the ~92% bottleneck of fused MoE prefill time.
__launch_bounds__(128, 1)
__global__ void sclp4_moe_sidecar_correct_blocked_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ src1,
    const int32_t* __restrict__ perm,
    const int32_t* __restrict__ expert_offsets,
    float*         __restrict__ dst,
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t src1_ne1, uint32_t n_experts
) {
    constexpr int TILE = 16;
    constexpr int MAX_SC_PER_BLOCK = 4096;

    const uint32_t m_tile = blockIdx.y;
    const uint32_t m_base = m_tile * TILE;
    int32_t e = -1;
    for (uint32_t ei = 0; ei < n_experts; ei++) {
        if ((uint32_t)expert_offsets[ei] <= m_base && m_base < (uint32_t)expert_offsets[ei + 1]) {
            e = (int32_t)ei;
            break;
        }
    }
    if (e < 0) return;

    __shared__ uint32_t s_total_nw;
    __shared__ uint32_t s_ne;
    __shared__ uint32_t s_scales_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint8_t  s_palette[4];
    __shared__ uint8_t  s_palette_size;
    __shared__ uint32_t s_ws_offset_e;
    __shared__ int32_t  s_m_global[TILE];
    __shared__ uint32_t s_sc_n_e[MAX_SC_PER_BLOCK];
    __shared__ uint32_t s_sc_k[MAX_SC_PER_BLOCK];
    __shared__ float    s_sc_delta[MAX_SC_PER_BLOCK];
    __shared__ uint32_t s_sc_count;
    __shared__ uint32_t s_sc_range_begin;
    __shared__ uint32_t s_sc_range_end;

    if (threadIdx.x == 0) {
        uint32_t total_nw, ne;
        __builtin_memcpy(&total_nw, blob,     sizeof(uint32_t));
        __builtin_memcpy(&ne,       blob + 4, sizeof(uint32_t));
        s_total_nw = total_nw;
        s_ne = ne;
        const uint8_t* p = blob + 8;
        for (uint32_t ei = 0; ei < ne; ei++) {
            if (ei == (uint32_t)e) {
                s_palette_size = p[0];
                for (int i = 0; i < (int)p[0] && i < 4; i++) s_palette[i] = p[1 + i];
            }
            p += 1 + p[0];
        }
        uint32_t scales_start = (uint32_t)(p - blob);
        s_scales_start = scales_start;
        uint32_t ws_start = scales_start + (total_nw / QK_SCLP) * sizeof(uint16_t);
        s_ws_start = ws_start;
        uint32_t expert_nw = total_nw / ne;
        uint32_t expert_nibble_bytes = (expert_nw + 1) / 2;
        s_ws_offset_e = ws_start + (uint32_t)e * expert_nibble_bytes;

        // Inline binary search for this expert's sidecar range.
        uint64_t total_ws_bytes = (uint64_t)ne * expert_nibble_bytes;
        const uint8_t* sc_base = blob + ws_start + total_ws_bytes;
        uint32_t sc_total;
        __builtin_memcpy(&sc_total, sc_base, sizeof(uint32_t));
        const uint8_t* sc_idx = sc_base + 4;

        uint32_t target_lo = (uint32_t)e * expert_nw;
        uint32_t target_hi = ((uint32_t)e + 1) * expert_nw;

        // lower_bound for target_lo
        uint32_t lo = 0, hi = sc_total;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            uint32_t val;
            __builtin_memcpy(&val, sc_idx + (uint64_t)mid * 4, 4);
            if (val < target_lo) lo = mid + 1; else hi = mid;
        }
        s_sc_range_begin = lo;

        // lower_bound for target_hi
        lo = s_sc_range_begin; hi = sc_total;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            uint32_t val;
            __builtin_memcpy(&val, sc_idx + (uint64_t)mid * 4, 4);
            if (val < target_hi) lo = mid + 1; else hi = mid;
        }
        s_sc_range_end = lo;
        s_sc_count = 0;
    }
    if (threadIdx.x < TILE) {
        uint32_t m_g = m_base + threadIdx.x;
        s_m_global[threadIdx.x] = perm[m_g];
    }
    __syncthreads();

    const uint32_t sc_begin = s_sc_range_begin;
    const uint32_t sc_end   = s_sc_range_end;
    if (sc_begin >= sc_end) return;

    const uint32_t expert_nw = s_total_nw / s_ne;
    const uint32_t expert_nibble_bytes = (expert_nw + 1) / 2;
    const uint64_t total_ws_bytes = (uint64_t)s_ne * expert_nibble_bytes;
    const uint8_t* sidecar_base = blob + s_ws_start + total_ws_bytes;
    uint32_t sc_total;
    __builtin_memcpy(&sc_total, sidecar_base, sizeof(uint32_t));
    const uint8_t* sc_idx_base = sidecar_base + 4;
    const uint8_t* sc_val_base = sc_idx_base + (uint64_t)sc_total * sizeof(uint32_t);
    const uint8_t* ws_e = blob + s_ws_offset_e;
    const uint16_t* scales = (const uint16_t*)(blob + s_scales_start);

    const uint32_t n_base = (uint32_t)blockIdx.x * TILE;
    const uint32_t n_end  = min(n_base + (uint32_t)TILE, N);

    // Only scan this expert's sidecar range (not the full sidecar list).
    const uint32_t range_size = sc_end - sc_begin;
    for (uint32_t ri = threadIdx.x; ri < range_size; ri += blockDim.x) {
        uint32_t i = sc_begin + ri;
        uint32_t global_idx;
        __builtin_memcpy(&global_idx, sc_idx_base + (uint64_t)i * sizeof(uint32_t), sizeof(uint32_t));

        uint32_t local = global_idx - (uint32_t)e * expert_nw;
        uint32_t n_e = local / K;
        if (n_e < n_base || n_e >= n_end) continue;

        uint32_t k = local % K;
        uint16_t correct_bits;
        __builtin_memcpy(&correct_bits, sc_val_base + (uint64_t)i * sizeof(uint16_t), sizeof(uint16_t));

        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + global_idx / QK_SCLP));

        uint64_t w_idx = (uint64_t)n_e * K + k;
        uint8_t byte = ws_e[w_idx >> 1];
        uint8_t nib  = (w_idx & 1) ? (byte & 0xF) : (byte >> 4);
        uint8_t pidx = nib >> 2;
        uint8_t smn  = nib & 0x3;
        uint8_t exp_a = (pidx < s_palette_size) ? s_palette[pidx] : 0;
        uint8_t sign_a = (smn >> 1) & 1;
        uint8_t mant_a = smn & 1;
        uint16_t approx_bits = ((uint16_t)sign_a << 15) | ((uint16_t)exp_a << 7) | ((uint16_t)mant_a << 6);

        float w_correct = __bfloat162float(*(__hip_bfloat16*)&correct_bits);
        float w_approx  = __bfloat162float(*(__hip_bfloat16*)&approx_bits) * scale;
        float w_delta   = w_correct - w_approx;
        if (w_delta == 0.f) continue;

        uint32_t slot = atomicAdd(&s_sc_count, 1);
        if (slot >= MAX_SC_PER_BLOCK) continue;
        s_sc_n_e[slot] = n_e;
        s_sc_k[slot]   = k;
        s_sc_delta[slot] = w_delta;
    }
    __syncthreads();

    const uint32_t sc_count = min(s_sc_count, (uint32_t)MAX_SC_PER_BLOCK);
    if (sc_count == 0) return;

    for (uint32_t cell_idx = threadIdx.x; cell_idx < (uint32_t)TILE * (uint32_t)TILE; cell_idx += blockDim.x) {
        uint32_t n_local = cell_idx / TILE;
        uint32_t m_local = cell_idx % TILE;
        uint32_t n_e = n_base + n_local;
        if (n_e >= N) continue;
        int32_t mg = s_m_global[m_local];
        if (mg < 0) continue;

        uint32_t i_active = (uint32_t)mg % n_active;
        uint32_t i_batch  = (uint32_t)mg / n_active;
        const float* x_row = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;

        float correction = 0.f;
        for (uint32_t s = 0; s < sc_count; s++) {
            if (s_sc_n_e[s] != n_e) continue;
            correction += s_sc_delta[s] * x_row[s_sc_k[s]];
        }
        if (correction != 0.f) {
            dst[(uint64_t)mg * N + n_e] += correction;
        }
    }
}

// Launcher for fused SCLP4 MoE WMMA prefill. Requires routing-sort scratch
// allocated by caller (perm + expert_offsets in the CUDA pool).
inline void llama_sclp4_fused_moe_wmma(
    const void*    blob_ptr,
    const float*   src1,
    const int32_t* ids,
    float*         dst,
    float*         dst_pre_sidecar,          // optional [N * n_active * n_batches], may be nullptr
    int32_t*       perm_scratch,            // [n_active × n_batches]
    int32_t*       expert_offsets_scratch,  // [n_experts + 1]
    uint32_t       N,
    uint32_t       K,
    uint32_t       n_active,
    uint32_t       n_batches,
    uint32_t       ids_s1,
    uint32_t       src1_ne1,
    uint32_t       n_experts,
    uint32_t       scalar_math_mode,
    uint32_t       sidecar_mode,
    hipStream_t    stream
) {
    const uint32_t total_slots = n_active * n_batches;
    constexpr int TILE = 16;

    // Step 1: routing sort with TILE-padded bins. perm_scratch must be sized at
    // least total_slots + n_experts * TILE (worst-case padding waste).
    sclp_moe_route_sort_kernel<<<1, 256, 2 * (n_experts + 1) * sizeof(int32_t), stream>>>(
        ids, perm_scratch, expert_offsets_scratch, total_slots, n_active, ids_s1, n_experts, (uint32_t)TILE);

    // Step 2: fused WMMA. Grid m-extent is total padded length / TILE.
    // We over-launch to (total_slots + n_experts * TILE + TILE - 1)/TILE — a fixed
    // upper bound — and let blocks that fall past expert_offsets[n_experts] early-return.
    const uint32_t m_tile_count = (total_slots + n_experts * TILE + TILE - 1) / TILE;
    dim3 block(32);
    dim3 grid((N + TILE - 1) / TILE, m_tile_count);

    // SCLP_FUSED_MOE_SCALAR=1 uses scalar dot product kernel (slow, for correctness debug).
    static const bool use_scalar = getenv("SCLP_FUSED_MOE_SCALAR") != nullptr;
    if (use_scalar) {
        sclp4_fused_moe_scalar_kernel<<<grid, block, 0, stream>>>(
            (const uint8_t*)blob_ptr,
            src1, perm_scratch, expert_offsets_scratch, ids, dst,
            N, K, n_active, src1_ne1, n_experts, scalar_math_mode);
    } else {
        sclp4_fused_moe_wmma_kernel<<<grid, block, 0, stream>>>(
            (const uint8_t*)blob_ptr,
            src1, perm_scratch, expert_offsets_scratch, ids, dst,
            N, K, n_active, src1_ne1, n_experts);
    }

    if (dst_pre_sidecar != nullptr) {
        const size_t out_elems = (size_t)N * (size_t)n_active * (size_t)n_batches;
        hipMemcpyAsync(dst_pre_sidecar, dst, out_elems * sizeof(float), hipMemcpyDeviceToDevice, stream);
    }

    // Apply sidecar corrections: ~1-2% of weights need exact-value restoration to
    // match the two-pass path's PPL.
    static const bool skip_sidecar    = getenv("SCLP_FUSED_MOE_NO_SIDECAR") != nullptr;
    static const bool legacy_sidecar  = getenv("SCLP_FUSED_MOE_SIDECAR_LEGACY") != nullptr;
    if (!skip_sidecar) {
        if (legacy_sidecar) {
            // Original per-sidecar-weight kernel with atomicAdd over routed slots.
            // Kept for A/B comparison; the blocked kernel below is the default.
            sclp4_moe_sidecar_correct_kernel<<<32, 256, 0, stream>>>(
                (const uint8_t*)blob_ptr,
                src1, perm_scratch, expert_offsets_scratch, dst,
                N, K, n_active, src1_ne1, n_experts, sidecar_mode);
        } else {
            // Blocked sidecar with inline per-expert binary search.
            constexpr int SC_TILE = 16;
            const uint32_t sc_m_tile_count = (total_slots + n_experts * SC_TILE + SC_TILE - 1) / SC_TILE;
            dim3 sc_block(128);
            dim3 sc_grid((N + SC_TILE - 1) / SC_TILE, sc_m_tile_count);
            sclp4_moe_sidecar_correct_blocked_kernel<<<sc_grid, sc_block, 0, stream>>>(
                (const uint8_t*)blob_ptr,
                src1, perm_scratch, expert_offsets_scratch, dst,
                N, K, n_active, src1_ne1, n_experts);
        }
    }
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
    __shared__ uint32_t s_scales_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint8_t  s_palette_sizes[256];
    __shared__ uint8_t  s_palettes[256][8];
    __shared__ uint32_t s_expert_groups;

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
        uint32_t enw = num_weights / ne;
        s_expert_groups = (enw + 3) / 4;
    }
    __syncthreads();

    const uint16_t* scales = (const uint16_t*)(blob + s_scales_start);
    const uint8_t* ws = blob + s_ws_start;
    uint32_t n_experts     = s_n_experts;
    uint32_t expert_groups = s_expert_groups;
    uint32_t expert_nw     = num_weights / n_experts;

    // Each thread now handles SUPER_GROUPS=8 groups = 32 weights = 24 bytes input,
    // 64 bytes (4× uint4) output. Reduces total threads 8x vs original.
    constexpr uint32_t SUPER_GROUPS = 8;
    const uint32_t super_gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t first_gid = super_gid * SUPER_GROUPS;
    if (first_gid >= (uint32_t)(expert_groups * n_experts)) return;

    // Each super-group stays inside one expert (we don't cross expert boundaries
    // within a thread). Compute expert and intra-expert group offset.
    const uint32_t e         = first_gid / expert_groups;
    const uint32_t g0_local  = first_gid - e * expert_groups;
    if (e >= n_experts) return;

    // Load palette into thread-private regs (4 of 8 entries used per nibble pidx; we
    // load up to 8 to be safe).
    uint8_t pal0 = s_palettes[e][0];
    uint8_t pal1 = s_palettes[e][1];
    uint8_t pal2 = s_palettes[e][2];
    uint8_t pal3 = s_palettes[e][3];
    uint8_t pal4 = s_palettes[e][4];
    uint8_t pal5 = s_palettes[e][5];
    uint8_t pal6 = s_palettes[e][6];
    uint8_t pal7 = s_palettes[e][7];
    uint8_t pal_size = s_palette_sizes[e];

    uint16_t out16[32];

    #pragma unroll
    for (uint32_t sub = 0; sub < SUPER_GROUPS; ++sub) {
        uint32_t g_local = g0_local + sub;
        if (g_local >= expert_groups) break;
        uint32_t gid = e * expert_groups + g_local;

        const uint8_t b0 = ws[gid * 3 + 0];
        const uint8_t b1 = ws[gid * 3 + 1];
        const uint8_t b2 = ws[gid * 3 + 2];

        uint8_t sixbits[4];
        sixbits[0] = b0 >> 2;
        sixbits[1] = ((b0 & 0x3) << 4) | (b1 >> 4);
        sixbits[2] = ((b1 & 0xF) << 2) | (b2 >> 6);
        sixbits[3] = b2 & 0x3F;

        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (uint64_t)gid * 4 / QK_SCLP));

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t pidx = sixbits[i] >> 3;
            uint8_t smn  = sixbits[i] & 0x7;
            uint8_t exp_ = 0;
            if (pidx < pal_size) {
                switch (pidx) {
                    case 0: exp_ = pal0; break;
                    case 1: exp_ = pal1; break;
                    case 2: exp_ = pal2; break;
                    case 3: exp_ = pal3; break;
                    case 4: exp_ = pal4; break;
                    case 5: exp_ = pal5; break;
                    case 6: exp_ = pal6; break;
                    case 7: exp_ = pal7; break;
                }
            }
            uint8_t sign = (smn >> 2) & 1;
            uint8_t mant = smn & 0x3;
            uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp_ << 7) | ((uint16_t)mant << 5);
            float w = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&bits)) * scale;
            out16[sub * 4 + i] = float_to_bf16_dev(w);
        }
    }

    // Fast path: super-group fully inside expert with all 16 weights valid.
    uint32_t base_idx = e * expert_nw + g0_local * 4;
    uint32_t remaining_expert = expert_nw - g0_local * 4;
    uint32_t n_groups_this   = min((uint32_t)SUPER_GROUPS, expert_groups - g0_local);
    uint32_t n_weights_this  = min(n_groups_this * 4, remaining_expert);

    if (n_weights_this == 32) {
        uint4 v0, v1, v2, v3;
        __builtin_memcpy(&v0, out16,      sizeof(uint4));
        __builtin_memcpy(&v1, out16 + 8,  sizeof(uint4));
        __builtin_memcpy(&v2, out16 + 16, sizeof(uint4));
        __builtin_memcpy(&v3, out16 + 24, sizeof(uint4));
        *reinterpret_cast<uint4*>(output + base_idx)      = v0;
        *reinterpret_cast<uint4*>(output + base_idx + 8)  = v1;
        *reinterpret_cast<uint4*>(output + base_idx + 16) = v2;
        *reinterpret_cast<uint4*>(output + base_idx + 24) = v3;
    } else {
        for (uint32_t i = 0; i < n_weights_this; ++i) output[base_idx + i] = out16[i];
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
        uint32_t scales_start = (uint32_t)(p - blob);
        s_ws_start = scales_start + (num_weights / QK_SCLP) * sizeof(uint16_t);
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
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorized path: 4 entries per thread (32 bytes per thread: 16 index + 8 value + 8 output_write)
    uint32_t i = tid * 4;
    for (; i + 3 < sidecar_count_s; i += stride * 4) {
        uint4 idx4;
        uint64_t val4_64;
        __builtin_memcpy(&idx4,    idx_base + (uint64_t)i * sizeof(uint32_t), sizeof(uint4));
        __builtin_memcpy(&val4_64, val_base + (uint64_t)i * sizeof(uint16_t), sizeof(uint64_t));
        output[idx4.x] = (uint16_t)(val4_64 & 0xFFFF);
        output[idx4.y] = (uint16_t)((val4_64 >> 16) & 0xFFFF);
        output[idx4.z] = (uint16_t)((val4_64 >> 32) & 0xFFFF);
        output[idx4.w] = (uint16_t)(val4_64 >> 48);
    }

    // Scalar tail
    for (uint32_t si = i; si < sidecar_count_s; si++) {
        uint32_t idx;
        uint16_t val;
        __builtin_memcpy(&idx, idx_base + (uint64_t)si * sizeof(uint32_t), sizeof(uint32_t));
        __builtin_memcpy(&val, val_base + (uint64_t)si * sizeof(uint16_t), sizeof(uint16_t));
        output[idx] = val;
    }
}

// Fused decode-GEMV for SCLP6, M=1.
// Each warp handles one output row; reads 3-byte groups producing 4 weights each.
// Supports per-expert palettes: thread 0 walks the header to find ws_start and
// selects the palette for the expert that owns this N×K block.
// For dense (n_experts=1), behavior is identical to the old single-palette version.
// Shared memory layout:
//   [0..3]      s_ws_offset (uint32) — byte offset of this expert's ws within blob
//   [4..259]    s_lut[64]   (float, 256 bytes) — decoded float for each 6-bit weight
//   [260..]     s_x[K]      (float, K*4 bytes) — activation vector
__launch_bounds__(512, 2)
__global__ void sclp6_fused_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    uint32_t N,
    uint32_t K,
    uint32_t expert_idx   // 0 for dense tensors
) {
    extern __shared__ char smem[];
    uint32_t* s_scales_offset = (uint32_t*)smem;
    uint32_t* s_ws_offset     = (uint32_t*)(smem + 4);
    float*    s_lut           = (float*)(smem + 8);
    float*    s_x             = (float*)(smem + 264);

    // Thread 0: walk header, locate ws_offset, build the 64-entry LUT in one pass.
    if (threadIdx.x == 0) {
        uint32_t n_experts;
        __builtin_memcpy(&n_experts, blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        uint8_t pal[8];
        uint8_t pal_size = 0;
        for (uint32_t e = 0; e < n_experts; e++) {
            if (e == expert_idx) {
                pal_size = p[0];
                for (int i = 0; i < (int)p[0]; i++) pal[i] = p[1 + i];
            }
            p += 1 + p[0];
        }
        uint32_t scales_start = (uint32_t)(p - blob);
        uint32_t total_nw; __builtin_memcpy(&total_nw, blob, sizeof(uint32_t));
        uint32_t expert_nw     = total_nw / n_experts;
        *s_scales_offset = scales_start + expert_idx * (expert_nw / QK_SCLP) * sizeof(uint16_t);

        uint32_t ws_start = scales_start + (total_nw / QK_SCLP) * sizeof(uint16_t);
        uint32_t expert_groups = (expert_nw + 3) / 4;
        *s_ws_offset = ws_start + expert_idx * expert_groups * 3;

        for (int pidx = 0; pidx < 8; pidx++) {
            uint8_t exp_ = (pidx < (int)pal_size) ? pal[pidx] : 0;
            for (int smn = 0; smn < 8; smn++) {
                uint8_t sign = (smn >> 2) & 1;
                uint8_t mant = smn & 0x3;
                uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp_ << 7) | ((uint16_t)mant << 5);
                s_lut[pidx * 8 + smn] = __bfloat162float(*(__hip_bfloat16*)&bits);
            }
        }
    }

    for (uint32_t k = threadIdx.x; k < K; k += blockDim.x) s_x[k] = x[k];
    __syncthreads();

    const uint16_t* scales = (const uint16_t*)(blob + *s_scales_offset);
    const uint8_t* ws = blob + *s_ws_offset;
    const int warps_per_block = blockDim.x / 32;
    const int warp_id  = threadIdx.x / 32;
    const int lane     = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    if (row >= N) return;

    float acc = 0.0f;
    const uint64_t row_group_base = (uint64_t)row * ((K + 3) / 4);
    const uint32_t n_groups_row   = (K + 3) / 4;
    const uint64_t row_weight_base = (uint64_t)row * K;

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
        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (row_weight_base + base_k) / QK_SCLP));
        for (uint32_t j = 0; j < n_k; j++) {
            acc += (s_lut[sixbits[j]] * scale) * s_x[base_k + j];
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

    // Thread now handles 8 groups (32 weights). Divide total_groups by 8.
    uint32_t super_groups = (total_groups + 7) / 8;
    dim3 decode_grid((super_groups + 255) / 256);

    static const bool time_sclp6 = getenv("SCLP6_TIME_DECODE") != nullptr;
    hipEvent_t s6a, s6b, s6c;
    if (time_sclp6) { hipEventCreate(&s6a); hipEventCreate(&s6b); hipEventCreate(&s6c); hipEventRecord(s6a, stream); }

    sclp6_decode_blob_kernel<<<decode_grid, block, 0, stream>>>(data, output, num_weights);

    if (time_sclp6) hipEventRecord(s6b, stream);

    sclp6_fixup_sidecar_kernel<<<256, block, 0, stream>>>(data, output, num_weights);

    if (time_sclp6) {
        hipEventRecord(s6c, stream);
        hipStreamSynchronize(stream);
        float ms_dec, ms_sc;
        hipEventElapsedTime(&ms_dec, s6a, s6b);
        hipEventElapsedTime(&ms_sc,  s6b, s6c);
        static int s6_call = 0;
        if (s6_call < 5) {
            fprintf(stderr, "[S6] call=%d nw=%u decode=%.3fms sidecar=%.3fms\n",
                    s6_call, num_weights, ms_dec, ms_sc);
            s6_call++;
        }
        hipEventDestroy(s6a); hipEventDestroy(s6b); hipEventDestroy(s6c);
    }
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
    constexpr int WARPS_PER_BLOCK = 16;
    dim3 gemv_block(WARPS_PER_BLOCK * 32);
    dim3 gemv_grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    size_t smem_bytes = 264 + (size_t)K * sizeof(float);
    SCLP_TG_TIME_BEGIN(sclp_tg::CAT_SCLP6_DENSE);
    sclp6_fused_gemv_kernel<<<gemv_grid, gemv_block, smem_bytes, stream>>>(
        (const uint8_t*)blob_ptr,
        src_f32,
        dst_f32, N, K, expert_idx);
    SCLP_TG_TIME_END();
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
    //   [0..3]   s_scales_offset
    //   [4..7]   s_ws_offset
    //   [8..263] s_lut[64]    (float, 256 bytes) — decoded float for each (pidx,smn) pair
    //   [264..]  s_x[K]       (float, K*4 bytes) — activation vector
    extern __shared__ char smem[];
    uint32_t* s_scales_offset = (uint32_t*)smem;
    uint32_t* s_ws_offset     = (uint32_t*)(smem + 4);
    float*    s_lut           = (float*)(smem + 8);    // 64 floats = 256 bytes
    float*    s_x             = (float*)(smem + 264);  // K floats

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
        uint32_t expert_nw    = total_nw / n_experts;
        uint32_t scales_start = (uint32_t)(p - blob);
        *s_scales_offset = scales_start + (uint32_t)e * (expert_nw / QK_SCLP) * sizeof(uint16_t);

        uint32_t ws_start      = scales_start + (total_nw / QK_SCLP) * sizeof(uint16_t);
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

    const uint16_t* scales = (const uint16_t*)(blob + *s_scales_offset);
    const uint8_t* ws = blob + *s_ws_offset;
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    if (row >= N) return;

    float acc = 0.0f;
    const uint64_t row_group_base = (uint64_t)row * ((K + 3) / 4);
    const uint32_t n_groups_row   = (K + 3) / 4;
    const uint64_t row_weight_base = (uint64_t)row * K;

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
        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (row_weight_base + base_k) / QK_SCLP));
        for (uint32_t j = 0; j < n_k; j++) {
            acc += (s_lut[sixbits[j]] * scale) * s_x[base_k + j];
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
    // Dynamic shared memory: 8 (offsets) + 256 (lut) + K*4 (activations)
    size_t smem_bytes = 264 + (size_t)K * sizeof(float);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, n_active * n_batches);
    SCLP_TG_TIME_BEGIN(sclp_tg::CAT_SCLP6_MOE);
    sclp6_fused_moe_gemv_kernel<<<grid, block, smem_bytes, stream>>>(
        (const uint8_t*)blob_ptr,
        src1, ids, dst, N, K, n_active, src1_ne1);
    SCLP_TG_TIME_END();
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

    // Main decode: one thread per 8 weights (single uint64_t load from ws_stream)
    uint32_t groups = (num_weights + 7) / 8;
    dim3 decode_grid((groups + 255) / 256);
    sclp_decode_blob_kernel<<<decode_grid, block, 0, stream>>>(data, output, num_weights);

    // Sidecar fixup: fixed 4-block grid with stride loop handles any sidecar count.
    // Threads where i >= sidecar_count return after a single shared-memory read.
    sclp_fixup_sidecar_kernel<<<256, block, 0, stream>>>(data, output, num_weights);
}
