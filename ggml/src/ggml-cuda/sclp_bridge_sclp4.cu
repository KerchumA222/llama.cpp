#include "sclp_bridge_common.cuh"

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
    __shared__ uint32_t s_bpal_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint32_t s_expert_nw;
    __shared__ uint32_t s_expert_nibble_bytes;
    __shared__ uint32_t s_blocks_per_expert;

    if (threadIdx.x == 0) {
        uint32_t ne;
        __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        s_n_experts = ne;
        const uint8_t* p = blob + 8;
        for (uint32_t e = 0; e < ne; e++) { p += 1 + p[0]; }
        s_bpal_start = (uint32_t)(p - blob);
        uint32_t enw = num_weights / ne;
        s_expert_nw = enw;
        s_blocks_per_expert = enw / QK_SCLP4;
        s_ws_start = s_bpal_start + (num_weights / QK_SCLP4) * 4;
        s_expert_nibble_bytes = (enw + 1) / 2;
    }
    __syncthreads();

    const uint8_t* bpal_base = blob + s_bpal_start;
    const uint8_t* ws        = blob + s_ws_start;
    uint32_t expert_nw = s_expert_nw;

    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr uint32_t W_PER_THREAD = 32;
    const uint32_t base_idx   = thread_idx * W_PER_THREAD;

    if (base_idx >= num_weights) return;

    const uint32_t e          = base_idx / expert_nw;
    const uint32_t local_base = base_idx - e * expert_nw;
    if (e >= s_n_experts) return;

    // Per-block palette: read 4 palette bytes for this block
    uint32_t block_idx = e * s_blocks_per_expert + (local_base / QK_SCLP4);
    const uint8_t* bp = bpal_base + block_idx * 4;
    const uint8_t bp0 = bp[0], bp1 = bp[1], bp2 = bp[2], bp3 = bp[3];

    const uint8_t* ws_p = ws + (uint64_t)e * s_expert_nibble_bytes + local_base / 2;
    uint32_t remaining_expert = expert_nw - local_base;
    uint32_t n_this = min(W_PER_THREAD, remaining_expert);

    uint16_t out[32];
    if (n_this == 32) {
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
                    uint8_t exp_;
                    switch(pidx) {
                        case 0: exp_ = bp0; break;
                        case 1: exp_ = bp1; break;
                        case 2: exp_ = bp2; break;
                        default: exp_ = bp3; break;
                    }
                    uint8_t sign = (smn >> 1) & 1;
                    uint8_t mant = smn & 1;
                    uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp_ << 7) | ((uint16_t)mant << 6);
                    out[q * 16 + i * 2 + j] = *reinterpret_cast<uint16_t*>(&bits);
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
            uint8_t byte = ws_p[i / 2];
            uint8_t nibble = (i % 2 == 0) ? (byte >> 4) : (byte & 0xF);
            uint8_t pidx = nibble >> 2;
            uint8_t smn  = nibble & 0x3;
            uint8_t exp_;
            switch(pidx) {
                case 0: exp_ = bp0; break;
                case 1: exp_ = bp1; break;
                case 2: exp_ = bp2; break;
                default: exp_ = bp3; break;
            }
            uint8_t sign = (smn >> 1) & 1;
            uint8_t mant = smn & 1;
            uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp_ << 7) | ((uint16_t)mant << 6);
            output[base_idx + i] = bits;
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
        uint32_t bpal_start = (uint32_t)(p - blob);
        s_ws_start = bpal_start + (num_weights / QK_SCLP4) * 4;
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
// K-tiled: processes activation vector in TILE_K chunks to reduce shared memory
// from K*4 to TILE_K*4 bytes, increasing occupancy (blocks per CU) for large K.
// For K=14336: smem drops from 56 KB (1 block/CU) to 16 KB (4 blocks/CU).
__launch_bounds__(512, 4)
__global__ void sclp4_fused_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    uint32_t N,
    uint32_t K
) {
    constexpr uint32_t TILE_K = 4096;
    extern __shared__ char smem[];
    __shared__ uint32_t s_bpal_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint32_t s_sc_base;
    __shared__ uint32_t s_sc_count;
    float* s_x = (float*)smem;

    if (threadIdx.x == 0) {
        uint32_t ne; __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        for (uint32_t ei = 0; ei < ne; ei++) p += 1 + p[0];
        s_bpal_start = (uint32_t)(p - blob);
        uint64_t nw = (uint64_t)N * K;
        s_ws_start = s_bpal_start + (uint32_t)((nw / QK_SCLP4) * 4);
        uint32_t enw = (uint32_t)(nw / ne);
        uint64_t ws_total = (uint64_t)ne * ((enw + 1) / 2);
        s_sc_base = s_ws_start + (uint32_t)ws_total;
        __builtin_memcpy(&s_sc_count, blob + s_sc_base, sizeof(uint32_t));
    }
    __syncthreads();

    const uint8_t*  bpal_base = blob + s_bpal_start;
    const uint8_t*  ws        = blob + s_ws_start;

    const int warps_per_block = blockDim.x / 32;
    const int warp_id  = threadIdx.x / 32;
    const int lane     = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    const bool valid = (row < N);

    float acc = 0.0f;
    const uint32_t n_bytes_row     = valid ? (K + 1) / 2 : 0;
    const uint64_t row_byte_base   = valid ? (uint64_t)row * n_bytes_row : 0;
    const uint64_t row_weight_base = valid ? (uint64_t)row * K : 0;

    for (uint32_t k_tile = 0; k_tile < K; k_tile += TILE_K) {
        const uint32_t tile_end = min(k_tile + TILE_K, K);
        const uint32_t tile_len = tile_end - k_tile;

        for (uint32_t k = threadIdx.x; k < tile_len; k += blockDim.x) {
            s_x[k] = x[k_tile + k];
        }
        __syncthreads();

        if (valid) {
            uint32_t cached_block_idx = UINT32_MAX;
            uint32_t pal = 0;
            const uint32_t b_start = k_tile / 2;
            const uint32_t b_end   = (tile_end + 1) / 2;

            for (uint32_t b = b_start + lane; b < b_end; b += 32) {
                uint8_t byte = ws[row_byte_base + b];
                uint32_t k0  = b * 2;
                uint32_t block_idx = (uint32_t)((row_weight_base + k0) >> 8);
                if (block_idx != cached_block_idx) {
                    cached_block_idx = block_idx;
                    __builtin_memcpy(&pal, bpal_base + block_idx * 4, sizeof(pal));
                }

                uint8_t pidx_hi = byte >> 6;
                uint8_t smn_hi  = (byte >> 4) & 0x3;
                uint8_t exp_hi  = sclp4_palette_pick(pal, pidx_hi);
                uint16_t bits_hi = ((uint16_t)((smn_hi >> 1) & 1) << 15)
                                 | ((uint16_t)exp_hi << 7)
                                 | ((uint16_t)(smn_hi & 1) << 6);
                float w_hi = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&bits_hi));
                acc += w_hi * s_x[k0 - k_tile];

                if (k0 + 1 < tile_end) {
                    uint8_t pidx_lo = (byte >> 2) & 0x3;
                    uint8_t smn_lo  = byte & 0x3;
                    uint8_t exp_lo  = sclp4_palette_pick(pal, pidx_lo);
                    uint16_t bits_lo = ((uint16_t)((smn_lo >> 1) & 1) << 15)
                                     | ((uint16_t)exp_lo << 7)
                                     | ((uint16_t)(smn_lo & 1) << 6);
                    float w_lo = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&bits_lo));
                    acc += w_lo * s_x[k0 + 1 - k_tile];
                }
            }
        }
        __syncthreads();
    }

    if (!valid) return;

    uint32_t sc_count = s_sc_count;
    if (sc_count > 0) {
        const uint32_t* sc_idx = (const uint32_t*)(blob + s_sc_base + 4);
        const uint16_t* sc_val = (const uint16_t*)(blob + s_sc_base + 4 + (uint64_t)sc_count * 4);
        uint32_t lo = 0, hi = 0;
        if (lane == 0) {
            uint32_t t0 = (uint32_t)row_weight_base, t1 = t0 + K, a = 0, b2 = sc_count;
            while (a < b2) { uint32_t m = (a + b2) >> 1; if (sc_idx[m] < t0) a = m + 1; else b2 = m; }
            lo = a; b2 = sc_count;
            while (a < b2) { uint32_t m = (a + b2) >> 1; if (sc_idx[m] < t1) a = m + 1; else b2 = m; }
            hi = a;
        }
        lo = __shfl(lo, 0); hi = __shfl(hi, 0);
        for (uint32_t e = lo + lane; e < hi; e += 32) {
            uint32_t gidx = sc_idx[e];
            uint32_t col  = gidx - (uint32_t)row_weight_base;
            uint8_t byte  = ws[(uint64_t)gidx >> 1];
            uint8_t nib   = (gidx & 1) ? (byte & 0xF) : (byte >> 4);
            uint32_t pal;
            __builtin_memcpy(&pal, bpal_base + ((uint64_t)gidx >> 8) * 4, sizeof(pal));
            uint16_t ab = ((uint16_t)((nib >> 1) & 1) << 15) | ((uint16_t)sclp4_palette_pick(pal, nib >> 2) << 7) | ((uint16_t)(nib & 1) << 6);
            float approx = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&ab));
            uint16_t tb = sc_val[e];
            float trueval = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&tb));
            acc += (trueval - approx) * x[col];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) y[row] = acc;
}

void llama_sclp4_dispatch(
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


void llama_sclp4_fused_gemv(
    const void*   blob_ptr,
    const float*  src_f32,
    float*        dst_f32,
    uint32_t      N,
    uint32_t      K,
    hipStream_t   stream
) {
    constexpr int WARPS_PER_BLOCK = 16;
    constexpr uint32_t TILE_K = 4096;
    dim3 gemv_block(WARPS_PER_BLOCK * 32);
    dim3 gemv_grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    size_t smem_bytes = (size_t)min(K, TILE_K) * sizeof(float);
    SCLP_TG_TIME_BEGIN(sclp_tg::CAT_SCLP4_DENSE);
    sclp4_fused_gemv_kernel<<<gemv_grid, gemv_block, smem_bytes, stream>>>(
        (const uint8_t*)blob_ptr,
        src_f32,
        dst_f32, N, K);
    SCLP_TG_TIME_END();
}

// Fused decode-GEMV for SCLP4 MoE, single-token generation (n_batches=1).
// K-tiled fused decode-GEMV for SCLP4 MoE, with folded sidecar correction.
// One block per (output_row_tile, active_expert_slot); one warp per output row.
// K-tiling reduces smem from K*4 to TILE_K*4 bytes for higher occupancy.
// Sidecar correction folded in via sorted-index binary search (no atomics).
__launch_bounds__(512, 4)
__global__ void sclp4_fused_moe_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ src1,
    const int32_t* __restrict__ ids,
    float*         __restrict__ dst,
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t src1_ne1
) {
    constexpr uint32_t TILE_K = 4096;
    extern __shared__ char smem[];
    __shared__ uint32_t s_bpal_offset;
    __shared__ uint32_t s_ws_offset;
    __shared__ uint32_t s_sc_base;
    __shared__ uint32_t s_sc_count;
    __shared__ uint32_t s_expert_nw;
    float*    s_x = (float*)smem;

    const uint32_t flat     = blockIdx.y;
    const int32_t  e        = ids[flat];
    const uint32_t i_active = flat % n_active;
    const uint32_t i_batch  = flat / n_active;

    if (threadIdx.x == 0) {
        uint32_t total_nw, n_experts;
        __builtin_memcpy(&total_nw,  blob,     sizeof(uint32_t));
        __builtin_memcpy(&n_experts, blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        for (uint32_t ei = 0; ei < n_experts; ei++) { p += 1 + p[0]; }
        uint32_t expert_nw = total_nw / n_experts;
        s_expert_nw = expert_nw;
        uint32_t bpal_start = (uint32_t)(p - blob);
        uint32_t bpe = expert_nw / QK_SCLP4;
        s_bpal_offset = bpal_start + (uint32_t)e * bpe * 4;

        uint32_t ws_start = bpal_start + (total_nw / QK_SCLP4) * 4;
        uint32_t expert_nibble_bytes = (expert_nw + 1) / 2;
        s_ws_offset = ws_start + (uint32_t)e * expert_nibble_bytes;

        uint64_t ws_total = (uint64_t)n_experts * expert_nibble_bytes;
        uint32_t sc_base = ws_start + (uint32_t)ws_total;
        s_sc_base = sc_base;
        __builtin_memcpy(&s_sc_count, blob + sc_base, sizeof(uint32_t));
    }
    __syncthreads();

    const float* x = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;

    const uint8_t* bpal_base = blob + s_bpal_offset;
    const uint8_t* ws = blob + s_ws_offset;
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    const bool valid = (row < N);

    float acc = 0.0f;
    const uint32_t n_bytes_row     = valid ? (K + 1) / 2 : 0;
    const uint64_t row_byte_base   = valid ? (uint64_t)row * n_bytes_row : 0;
    const uint64_t row_weight_base = valid ? (uint64_t)row * K : 0;

    for (uint32_t k_tile = 0; k_tile < K; k_tile += TILE_K) {
        const uint32_t tile_end = min(k_tile + TILE_K, K);
        const uint32_t tile_len = tile_end - k_tile;

        for (uint32_t k = threadIdx.x; k < tile_len; k += blockDim.x) {
            s_x[k] = x[k_tile + k];
        }
        __syncthreads();

        if (valid) {
            uint32_t cached_block_idx = UINT32_MAX;
            uint32_t pal = 0;
            const uint32_t b_start = k_tile / 2;
            const uint32_t b_end   = (tile_end + 1) / 2;

            for (uint32_t b = b_start + lane; b < b_end; b += 32) {
                uint8_t byte = ws[row_byte_base + b];
                uint32_t k0  = b * 2;
                uint32_t block_idx = (uint32_t)((row_weight_base + k0) >> 8);
                if (block_idx != cached_block_idx) {
                    cached_block_idx = block_idx;
                    __builtin_memcpy(&pal, bpal_base + block_idx * 4, sizeof(pal));
                }

                uint8_t pidx_hi = byte >> 6;
                uint8_t smn_hi  = (byte >> 4) & 0x3;
                uint8_t exp_hi  = sclp4_palette_pick(pal, pidx_hi);
                uint16_t bits_hi = ((uint16_t)((smn_hi >> 1) & 1) << 15)
                                 | ((uint16_t)exp_hi << 7)
                                 | ((uint16_t)(smn_hi & 1) << 6);
                acc += __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&bits_hi)) * s_x[k0 - k_tile];

                if (k0 + 1 < tile_end) {
                    uint8_t pidx_lo = (byte >> 2) & 0x3;
                    uint8_t smn_lo  = byte & 0x3;
                    uint8_t exp_lo  = sclp4_palette_pick(pal, pidx_lo);
                    uint16_t bits_lo = ((uint16_t)((smn_lo >> 1) & 1) << 15)
                                     | ((uint16_t)exp_lo << 7)
                                     | ((uint16_t)(smn_lo & 1) << 6);
                    acc += __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&bits_lo)) * s_x[k0 + 1 - k_tile];
                }
            }
        }
        __syncthreads();
    }

    if (!valid) return;

    // Folded sidecar correction: sidecar indices are global (expert_offset + row*K + col).
    uint32_t sc_count = s_sc_count;
    if (sc_count > 0) {
        const uint32_t* sc_idx = (const uint32_t*)(blob + s_sc_base + 4);
        const uint16_t* sc_val = (const uint16_t*)(blob + s_sc_base + 4 + (uint64_t)sc_count * 4);
        uint32_t expert_off = (uint32_t)e * s_expert_nw;
        uint32_t lo = 0, hi = 0;
        if (lane == 0) {
            uint32_t t0 = expert_off + (uint32_t)row_weight_base;
            uint32_t t1 = t0 + K;
            uint32_t a = 0, b2 = sc_count;
            while (a < b2) { uint32_t m = (a + b2) >> 1; if (sc_idx[m] < t0) a = m + 1; else b2 = m; }
            lo = a; b2 = sc_count;
            while (a < b2) { uint32_t m = (a + b2) >> 1; if (sc_idx[m] < t1) a = m + 1; else b2 = m; }
            hi = a;
        }
        lo = __shfl(lo, 0); hi = __shfl(hi, 0);
        for (uint32_t si = lo + lane; si < hi; si += 32) {
            uint32_t gidx = sc_idx[si];
            uint32_t local_idx = gidx - expert_off;
            uint32_t col = local_idx - (uint32_t)row_weight_base;
            uint8_t byte = ws[(uint64_t)local_idx >> 1];
            uint8_t nib  = (local_idx & 1) ? (byte & 0xF) : (byte >> 4);
            uint32_t pal_sc;
            __builtin_memcpy(&pal_sc, bpal_base + ((uint64_t)local_idx >> 8) * 4, sizeof(pal_sc));
            uint16_t ab = ((uint16_t)((nib >> 1) & 1) << 15) | ((uint16_t)sclp4_palette_pick(pal_sc, nib >> 2) << 7) | ((uint16_t)(nib & 1) << 6);
            float approx = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&ab));
            uint16_t tb = sc_val[si];
            float trueval = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&tb));
            acc += (trueval - approx) * x[col];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) dst[row + (uint64_t)flat * N] = acc;
}

void llama_sclp4_fused_moe_gemv(
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
    size_t smem_bytes = (size_t)min(K, TILE_K) * sizeof(float);
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

    __shared__ uint32_t s_bpal_offset;
    __shared__ uint32_t s_ws_offset;
    __shared__ int32_t  s_m_global[TILE];

    if (threadIdx.x == 0) {
        uint32_t total_nw, ne;
        __builtin_memcpy(&total_nw, blob,     sizeof(uint32_t));
        __builtin_memcpy(&ne,       blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        for (uint32_t ei = 0; ei < ne; ei++) { p += 1 + p[0]; }
        uint32_t expert_nw = total_nw / ne;
        uint32_t bpal_start = (uint32_t)(p - blob);
        uint32_t bpe = expert_nw / QK_SCLP4;
        s_bpal_offset = bpal_start + (uint32_t)e * bpe * 4;
        uint32_t ws_start = bpal_start + (total_nw / QK_SCLP4) * 4;
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

    const uint8_t* bpal_base = blob + s_bpal_offset;
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
        bf16x16_t a_frag;
        #pragma unroll
        for (int j = 0; j < TILE; j++) {
            const uint32_t kg = k16 + j;
            if (n_row < N && kg < K) {
                uint64_t w_idx = (uint64_t)n_row * K + kg;
                uint32_t block_idx = w_idx / QK_SCLP4;
                const uint8_t* bp = bpal_base + block_idx * 4;
                uint8_t byte  = ws[w_idx >> 1];
                uint8_t nib   = (w_idx & 1) ? (byte & 0xF) : (byte >> 4);
                uint8_t pidx  = nib >> 2;
                uint8_t smn   = nib & 0x3;
                uint8_t exp_  = bp[pidx];
                uint8_t sign  = (smn >> 1) & 1;
                uint8_t mant  = smn & 1;
                uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp_ << 7) | ((uint16_t)mant << 6);
                a_frag[j] = *(__bf16*)&bits;
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

    __shared__ uint32_t s_bpal_offset;
    __shared__ uint32_t s_ws_offset;
    __shared__ int32_t  s_m_global[TILE];

    if (threadIdx.x == 0) {
        uint32_t total_nw, ne;
        __builtin_memcpy(&total_nw, blob,     sizeof(uint32_t));
        __builtin_memcpy(&ne,       blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        for (uint32_t ei = 0; ei < ne; ei++) { p += 1 + p[0]; }
        uint32_t expert_nw = total_nw / ne;
        uint32_t bpal_start = (uint32_t)(p - blob);
        uint32_t bpe = expert_nw / QK_SCLP4;
        s_bpal_offset = bpal_start + (uint32_t)e * bpe * 4;
        uint32_t ws_start = bpal_start + (total_nw / QK_SCLP4) * 4;
        uint32_t expert_nibble_bytes = (expert_nw + 1) / 2;
        s_ws_offset = ws_start + (uint32_t)e * expert_nibble_bytes;
    }
    if (threadIdx.x < TILE) {
        uint32_t m_g = m_base + threadIdx.x;
        s_m_global[threadIdx.x] = perm[m_g];
    }
    __syncthreads();

    const uint8_t* bpal_base = blob + s_bpal_offset;
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
        uint32_t block_idx = w_idx / QK_SCLP4;
        const uint8_t* bp = bpal_base + block_idx * 4;
        uint8_t byte  = ws[w_idx >> 1];
        uint8_t nib   = (w_idx & 1) ? (byte & 0xF) : (byte >> 4);
        uint8_t pidx  = nib >> 2;
        uint8_t smn   = nib & 0x3;
        uint8_t exp_  = bp[pidx];
        uint8_t sign  = (smn >> 1) & 1;
        uint8_t mant  = smn & 1;
        uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp_ << 7) | ((uint16_t)mant << 6);
        float w = __bfloat162float(*(__hip_bfloat16*)&bits);

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
    __shared__ uint32_t s_bpal_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint32_t s_sidecar_count;
    if (threadIdx.x == 0) {
        uint32_t total_nw, ne;
        __builtin_memcpy(&total_nw, blob,     sizeof(uint32_t));
        __builtin_memcpy(&ne,       blob + 4, sizeof(uint32_t));
        s_total_nw = total_nw;
        s_n_experts = ne;
        const uint8_t* p = blob + 8;
        for (uint32_t e = 0; e < ne; e++) p += 1 + p[0];
        uint32_t bpal_start = (uint32_t)(p - blob);
        s_bpal_start = bpal_start;
        uint32_t ws_start = bpal_start + (total_nw / QK_SCLP4) * 4;
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
    const uint8_t* bpal_base = blob + s_bpal_start;

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

        const uint8_t* ws_e = blob + s_ws_start + (uint64_t)e * expert_nibble_bytes;
        uint64_t w_idx = (uint64_t)n_e * K + k;
        uint32_t block_idx = e * (expert_nw / QK_SCLP4) + (uint32_t)(w_idx / QK_SCLP4);
        const uint8_t* bp = bpal_base + block_idx * 4;
        uint8_t byte = ws_e[w_idx >> 1];
        uint8_t nib  = (w_idx & 1) ? (byte & 0xF) : (byte >> 4);
        uint8_t pidx = nib >> 2;
        uint8_t smn  = nib & 0x3;
        uint8_t exp_a = bp[pidx];
        uint8_t sign_a = (smn >> 1) & 1;
        uint8_t mant_a = smn & 1;
        uint16_t approx_bits = ((uint16_t)sign_a << 15) | ((uint16_t)exp_a << 7) | ((uint16_t)mant_a << 6);

        float w_correct = __bfloat162float(*(__hip_bfloat16*)&correct_bits);
        float w_approx  = __bfloat162float(*(__hip_bfloat16*)&approx_bits);
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
    __shared__ uint32_t s_bpal_start;
    __shared__ uint32_t s_ws_start;
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
        for (uint32_t ei = 0; ei < ne; ei++) { p += 1 + p[0]; }
        uint32_t bpal_start = (uint32_t)(p - blob);
        s_bpal_start = bpal_start;
        uint32_t ws_start = bpal_start + (total_nw / QK_SCLP4) * 4;
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

    if (s_sc_range_begin >= s_sc_range_end) return;

    const uint32_t expert_nw = s_total_nw / s_ne;
    const uint32_t expert_nibble_bytes = (expert_nw + 1) / 2;
    const uint64_t total_ws_bytes = (uint64_t)s_ne * expert_nibble_bytes;
    const uint8_t* sidecar_base = blob + s_ws_start + total_ws_bytes;
    uint32_t sc_total;
    __builtin_memcpy(&sc_total, sidecar_base, sizeof(uint32_t));
    const uint8_t* sc_idx_base = sidecar_base + 4;
    const uint8_t* sc_val_base = sc_idx_base + (uint64_t)sc_total * sizeof(uint32_t);
    const uint8_t* ws_e = blob + s_ws_offset_e;
    const uint8_t* bpal_base = blob + s_bpal_start;

    const uint32_t n_base = (uint32_t)blockIdx.x * TILE;
    const uint32_t n_end  = min(n_base + (uint32_t)TILE, N);

    if (threadIdx.x == 0) {
        const uint32_t tile_lo = (uint32_t)e * expert_nw + n_base * K;
        const uint32_t tile_hi = (uint32_t)e * expert_nw + n_end * K;
        uint32_t lo = s_sc_range_begin, hi = s_sc_range_end;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            uint32_t val;
            __builtin_memcpy(&val, sc_idx_base + (uint64_t)mid * sizeof(uint32_t), sizeof(uint32_t));
            if (val < tile_lo) lo = mid + 1; else hi = mid;
        }
        s_sc_range_begin = lo;
        hi = s_sc_range_end;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            uint32_t val;
            __builtin_memcpy(&val, sc_idx_base + (uint64_t)mid * sizeof(uint32_t), sizeof(uint32_t));
            if (val < tile_hi) lo = mid + 1; else hi = mid;
        }
        s_sc_range_end = lo;
    }
    __syncthreads();

    const uint32_t sc_begin = s_sc_range_begin;
    const uint32_t sc_end   = s_sc_range_end;
    if (sc_begin >= sc_end) return;

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

        uint64_t w_idx = (uint64_t)n_e * K + k;
        uint32_t block_idx = (uint32_t)e * (expert_nw / QK_SCLP4) + (uint32_t)(w_idx / QK_SCLP4);
        const uint8_t* bp = bpal_base + block_idx * 4;
        uint8_t byte = ws_e[w_idx >> 1];
        uint8_t nib  = (w_idx & 1) ? (byte & 0xF) : (byte >> 4);
        uint8_t pidx = nib >> 2;
        uint8_t smn  = nib & 0x3;
        uint8_t exp_a = bp[pidx];
        uint8_t sign_a = (smn >> 1) & 1;
        uint8_t mant_a = smn & 1;
        uint16_t approx_bits = ((uint16_t)sign_a << 15) | ((uint16_t)exp_a << 7) | ((uint16_t)mant_a << 6);

        float w_correct = __bfloat162float(*(__hip_bfloat16*)&correct_bits);
        float w_approx  = __bfloat162float(*(__hip_bfloat16*)&approx_bits);
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
void llama_sclp4_fused_moe_wmma(
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
