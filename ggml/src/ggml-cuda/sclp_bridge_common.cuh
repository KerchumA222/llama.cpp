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
#define QK_SCLP4 256

static __device__ __forceinline__ uint16_t float_to_bf16_dev(float f) {
    union { float f; uint32_t u; } x;
    x.f = f;
    return (uint16_t)(x.u >> 16);
}

static __device__ __forceinline__ uint8_t sclp4_palette_pick(uint32_t packed_palette, uint8_t pidx) {
    return (uint8_t)(packed_palette >> ((uint32_t)pidx << 3));
}

static __device__ __forceinline__ float sclp5_decode_code(uint32_t packed_palette, uint8_t code) {
    uint8_t idx  = code >> 3;
    uint8_t sign = (code >> 2) & 1;
    uint8_t mant = code & 0x3;
    uint16_t bits = ((uint16_t)sign << 15)
                  | ((uint16_t)sclp4_palette_pick(packed_palette, idx) << 7)
                  | ((uint16_t)mant << 5);
    return __bfloat162float(*(__hip_bfloat16*)&bits);
}

static __device__ __forceinline__ float sclp5_decode_weight_at(
    const uint8_t* __restrict__ ws,
    uint64_t w_idx,
    uint32_t packed_palette
) {
    const uint8_t* g = ws + ((w_idx >> 3) * 5);
    uint64_t v = ((uint64_t)g[0] << 32) | ((uint64_t)g[1] << 24) |
                 ((uint64_t)g[2] << 16) | ((uint64_t)g[3] << 8)  | (uint64_t)g[4];
    uint8_t code = (uint8_t)((v >> (5 * (7 - (w_idx & 7)))) & 0x1F);
    return sclp5_decode_code(packed_palette, code);
}

static __device__ __forceinline__ uint8_t sclp8_palette_pick(
    uint32_t pal0,
    uint32_t pal1,
    uint32_t pal2,
    uint32_t pal3,
    uint8_t  pidx
) {
    switch (pidx >> 2) {
        case 0: return (uint8_t)(pal0 >> ((uint32_t)(pidx & 3) << 3));
        case 1: return (uint8_t)(pal1 >> ((uint32_t)(pidx & 3) << 3));
        case 2: return (uint8_t)(pal2 >> ((uint32_t)(pidx & 3) << 3));
        default:return (uint8_t)(pal3 >> ((uint32_t)(pidx & 3) << 3));
    }
}

static __device__ __forceinline__ float sclp6_decode_code(
    const uint8_t* __restrict__ pal,
    uint8_t code
) {
    uint8_t exp  = pal[code >> 3];
    uint8_t smn  = code & 0x7;
    uint16_t bits = ((uint16_t)(smn >> 2) << 15)
                  | ((uint16_t)exp << 7)
                  | ((uint16_t)(smn & 0x3) << 5);
    return __bfloat162float(*(__hip_bfloat16*)&bits);
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

__global__ void sclp_moe_route_sort_kernel(
    const int32_t* __restrict__ ids,
    int32_t*       __restrict__ perm,
    int32_t*       __restrict__ expert_offsets,
    uint32_t total_slots, uint32_t n_active,
    uint32_t ids_s1, uint32_t n_experts, uint32_t tile
);
