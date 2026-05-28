#include "sclp_bridge_common.cuh"


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
//   [4..11]     s_pal[8]    (uint8_t, 8 bytes)  — palette exponents
//   [12..]      s_x[K]      (float, K*4 bytes) — activation vector
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
    uint32_t* s_sc_base       = (uint32_t*)(smem + 8);
    uint32_t* s_sc_count      = (uint32_t*)(smem + 12);
    uint8_t*  s_pal           = (uint8_t*)(smem + 16);
    float*    s_x             = (float*)(smem + 24);

    // Thread 0: walk header, locate ws_offset, copy the 8-byte palette into smem.
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

        for (int i = 0; i < 8; i++) {
            s_pal[i] = (i < (int)pal_size) ? pal[i] : 0;
        }
        uint32_t ws_total = n_experts * expert_groups * 3;
        *s_sc_base = ws_start + ws_total;
        __builtin_memcpy(s_sc_count, blob + *s_sc_base, sizeof(uint32_t));
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

        uint32_t base_k = g * 4;
        uint32_t n_k = (base_k + 4 <= K) ? 4 : (K - base_k);
        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + ((row_weight_base + base_k) >> 5)));
        acc += (sclp6_decode_code(s_pal, b0 >> 2) * scale) * s_x[base_k + 0];
        if (n_k > 1) acc += (sclp6_decode_code(s_pal, ((b0 & 0x3) << 4) | (b1 >> 4)) * scale) * s_x[base_k + 1];
        if (n_k > 2) acc += (sclp6_decode_code(s_pal, ((b1 & 0xF) << 2) | (b2 >> 6)) * scale) * s_x[base_k + 2];
        if (n_k > 3) acc += (sclp6_decode_code(s_pal, b2 & 0x3F) * scale) * s_x[base_k + 3];
    }

    // Fold sidecar (dense only; sidecar indices are global gidx = row*K + col, sorted).
    uint32_t sc_count = (expert_idx == 0) ? *s_sc_count : 0;
    if (sc_count > 0) {
        uint32_t sc_base = *s_sc_base;
        const uint32_t* sc_idx = (const uint32_t*)(blob + sc_base + 4);
        const uint16_t* sc_val = (const uint16_t*)(blob + sc_base + 4 + (uint64_t)sc_count * 4);
        uint32_t lo = 0, hi = 0;
        if (lane == 0) {
            uint32_t t0 = (uint32_t)row_weight_base, t1 = t0 + K, a = 0, b = sc_count;
            while (a < b) { uint32_t m = (a + b) >> 1; if (sc_idx[m] < t0) a = m + 1; else b = m; }
            lo = a; b = sc_count;
            while (a < b) { uint32_t m = (a + b) >> 1; if (sc_idx[m] < t1) a = m + 1; else b = m; }
            hi = a;
        }
        lo = __shfl(lo, 0); hi = __shfl(hi, 0);
        for (uint32_t e = lo + lane; e < hi; e += 32) {
            uint32_t gidx = sc_idx[e];
            uint32_t col  = gidx - (uint32_t)row_weight_base;
            uint64_t byte_off = (uint64_t)(gidx / 4) * 3;
            uint8_t b0 = ws[byte_off + 0], b1 = ws[byte_off + 1], b2 = ws[byte_off + 2];
            uint8_t six;
            switch (gidx & 3) {
                case 0:  six = b0 >> 2; break;
                case 1:  six = ((b0 & 0x3) << 4) | (b1 >> 4); break;
                case 2:  six = ((b1 & 0xF) << 2) | (b2 >> 6); break;
                default: six = b2 & 0x3F; break;
            }
            float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (gidx >> 5)));
            float approx = sclp6_decode_code(s_pal, six) * scale;
            uint16_t tb = sc_val[e];
            float trueval = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&tb));
            acc += (trueval - approx) * s_x[col];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) y[row] = acc;
}

void llama_sclp6_dispatch(
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

void llama_sclp6_fused_gemv(
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
    size_t smem_bytes = 24 + (size_t)K * sizeof(float);
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
//   2. Palette copied into shared memory and decoded directly with bit ops —
//      removes the 64-entry shared-memory LUT from the hot loop.
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
    //   [8..15]  s_pal[8]     (uint8_t, 8 bytes) — palette exponents
    //   [16..]   s_x[K]       (float, K*4 bytes) — activation vector
    extern __shared__ char smem[];
    uint32_t* s_scales_offset = (uint32_t*)smem;
    uint32_t* s_ws_offset     = (uint32_t*)(smem + 4);
    uint8_t*  s_pal           = (uint8_t*)(smem + 8);
    float*    s_x             = (float*)(smem + 16);

    const uint32_t flat    = blockIdx.y;
    const int32_t  e       = ids[flat];
    const uint32_t i_active = flat % n_active;
    const uint32_t i_batch  = flat / n_active;

    // Thread 0: parse blob header, copy the palette into smem.
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

        for (int i = 0; i < 8; i++) {
            s_pal[i] = (i < (int)pal_size) ? pal[i] : 0;
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

        uint32_t base_k = g * 4;
        uint32_t n_k = (base_k + 4 <= K) ? 4 : (K - base_k);
        float scale = __bfloat162float(*reinterpret_cast<const __hip_bfloat16*>(scales + (row_weight_base + base_k) / QK_SCLP));
        acc += (sclp6_decode_code(s_pal, b0 >> 2) * scale) * s_x[base_k + 0];
        if (n_k > 1) acc += (sclp6_decode_code(s_pal, ((b0 & 0x3) << 4) | (b1 >> 4)) * scale) * s_x[base_k + 1];
        if (n_k > 2) acc += (sclp6_decode_code(s_pal, ((b1 & 0xF) << 2) | (b2 >> 6)) * scale) * s_x[base_k + 2];
        if (n_k > 3) acc += (sclp6_decode_code(s_pal, b2 & 0x3F) * scale) * s_x[base_k + 3];
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) dst[row + (uint64_t)flat * N] = acc;
}

void llama_sclp6_fused_moe_gemv(
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
    // Dynamic shared memory: 8 (offsets) + 8 (palette) + K*4 (activations)
    size_t smem_bytes = 16 + (size_t)K * sizeof(float);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, n_active * n_batches);
    SCLP_TG_TIME_BEGIN(sclp_tg::CAT_SCLP6_MOE);
    sclp6_fused_moe_gemv_kernel<<<grid, block, smem_bytes, stream>>>(
        (const uint8_t*)blob_ptr,
        src1, ids, dst, N, K, n_active, src1_ne1);
    SCLP_TG_TIME_END();
}

