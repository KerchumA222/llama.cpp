#include "sclp_bridge_common.cuh"

// ===================== SCLP5 (5-bit: idx2|sign1|mant2) =====================
// Per-block palette identical to SCLP4 (4 exponent entries per QK_SCLP4=256 block,
// 4 bytes/block, palette_size=0 header). ws_stream packs 8 weights into 5 bytes:
// 8 five-bit codes MSB-first into a 40-bit big-endian field. Per weight:
// code = idx(4:3) | sign(2) | mant(1:0); bits = sign<<15 | exp<<7 | mant<<5.
__global__ void sclp5_decode_blob_kernel(
    const uint8_t* __restrict__ blob,
    uint16_t*      __restrict__ output,
    uint32_t                    num_weights
) {
    __shared__ uint32_t s_n_experts;
    __shared__ uint32_t s_bpal_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint32_t s_expert_nw;
    __shared__ uint32_t s_expert_ws_bytes;
    __shared__ uint32_t s_blocks_per_expert;
    __shared__ uint32_t s_groups_per_expert;

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
        s_groups_per_expert = (enw + 7) / 8;
        s_ws_start = s_bpal_start + (num_weights / QK_SCLP4) * 4;
        s_expert_ws_bytes = ((enw + 7) / 8) * 5;
    }
    __syncthreads();

    const uint8_t* bpal_base = blob + s_bpal_start;
    const uint8_t* ws        = blob + s_ws_start;
    const uint32_t enw       = s_expert_nw;
    const uint32_t gpe       = s_groups_per_expert;
    const uint32_t total_groups = gpe * s_n_experts;

    const uint32_t stride = gridDim.x * blockDim.x;
    for (uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x; gid < total_groups; gid += stride) {
        uint32_t e          = gid / gpe;
        uint32_t g_local    = gid % gpe;
        uint32_t local_base = g_local * 8;
        uint32_t base_idx   = e * enw + local_base;

        uint32_t block_idx = e * s_blocks_per_expert + (local_base >> 8);
        uint32_t pal;
        __builtin_memcpy(&pal, bpal_base + (uint64_t)block_idx * 4, sizeof(pal));

        const uint8_t* g = ws + (uint64_t)e * s_expert_ws_bytes + (uint64_t)g_local * 5;
        uint64_t v = ((uint64_t)g[0] << 32) | ((uint64_t)g[1] << 24) |
                     ((uint64_t)g[2] << 16) | ((uint64_t)g[3] << 8)  | (uint64_t)g[4];

        uint32_t n_this = min(8u, enw - local_base);
        for (uint32_t i = 0; i < n_this; i++) {
            uint8_t code = (uint8_t)((v >> (5 * (7 - i))) & 0x1F);
            uint8_t idx  = code >> 3;
            uint8_t sign = (code >> 2) & 1;
            uint8_t mant = code & 0x3;
            uint8_t exp_ = sclp4_palette_pick(pal, idx);
            uint16_t bits = ((uint16_t)sign << 15) | ((uint16_t)exp_ << 7) | ((uint16_t)mant << 5);
            output[base_idx + i] = bits;
        }
    }
}

__global__ void sclp5_fixup_sidecar_kernel(
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

    const uint8_t* ws  = blob + s_ws_start;
    uint32_t expert_nw = num_weights / s_n_experts;
    uint64_t total_ws_bytes = (uint64_t)s_n_experts * (((expert_nw + 7) / 8) * 5);
    const uint8_t* sidecar_base = ws + total_ws_bytes;

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
    for (uint32_t si = tid; si < sidecar_count_s; si += stride) {
        uint32_t idx; uint16_t val;
        __builtin_memcpy(&idx, idx_base + (uint64_t)si * sizeof(uint32_t), sizeof(uint32_t));
        __builtin_memcpy(&val, val_base + (uint64_t)si * sizeof(uint16_t), sizeof(uint16_t));
        output[idx] = val;
    }
}

void llama_sclp5_dispatch(
    const void* sclp_data,
    uint16_t*   output,
    uint32_t    num_weights,
    hipStream_t stream,
    bool        apply_sidecar
) {
    const uint8_t* data = (const uint8_t*)sclp_data;
    dim3 block(256);
    uint32_t groups = (num_weights + 7) / 8;
    dim3 decode_grid((groups + 255) / 256);
    sclp5_decode_blob_kernel<<<decode_grid, block, 0, stream>>>(data, output, num_weights);
    if (apply_sidecar) {
        sclp5_fixup_sidecar_kernel<<<1024, block, 0, stream>>>(data, output, num_weights);
    }
}

// Fused decode-GEMV for SCLP5, M=1 (single-token gen). One warp per output row;
// K-tiled fused decode-GEMV for SCLP5, M=1, with folded sidecar correction.
// Sidecar reads from global x (not s_x) since s_x only holds the current tile.
__launch_bounds__(512, 4)
__global__ void sclp5_fused_gemv_kernel(
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
        uint64_t ws_total = (uint64_t)ne * (((enw + 7) / 8) * 5);
        s_sc_base = s_ws_start + (uint32_t)ws_total;
        __builtin_memcpy(&s_sc_count, blob + s_sc_base, sizeof(uint32_t));
    }
    __syncthreads();

    const uint8_t* bpal_base = blob + s_bpal_start;
    const uint8_t* ws        = blob + s_ws_start;

    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    const bool valid = (row < N);

    float acc = 0.0f;
    const uint32_t groups_per_row  = valid ? (K + 7) / 8 : 0;
    const uint64_t row_group_base  = valid ? (uint64_t)row * groups_per_row : 0;
    const uint64_t row_weight_base = valid ? (uint64_t)row * K : 0;

    for (uint32_t k_tile = 0; k_tile < K; k_tile += TILE_K) {
        const uint32_t tile_end = min(k_tile + TILE_K, K);
        const uint32_t tile_len = tile_end - k_tile;

        for (uint32_t k = threadIdx.x; k < tile_len; k += blockDim.x) {
            s_x[k] = x[k_tile + k];
        }
        __syncthreads();

        if (valid) {
            const uint32_t g_start = k_tile / 8;
            const uint32_t g_end   = (tile_end + 7) / 8;

            for (uint32_t g = g_start + lane; g < g_end; g += 32) {
                const uint8_t* gp = ws + (row_group_base + g) * 5;
                uint64_t v = ((uint64_t)gp[0] << 32) | ((uint64_t)gp[1] << 24) |
                             ((uint64_t)gp[2] << 16) | ((uint64_t)gp[3] << 8) | (uint64_t)gp[4];
                uint32_t k0 = g * 8;
                uint32_t block_idx = (uint32_t)((row_weight_base + k0) >> 8);
                uint32_t pal;
                __builtin_memcpy(&pal, bpal_base + (uint64_t)block_idx * 4, sizeof(pal));
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    uint32_t k = k0 + j;
                    if (k < k_tile || k >= tile_end) continue;
                    uint8_t code = (uint8_t)((v >> (5 * (7 - j))) & 0x1F);
                    acc += sclp5_decode_code(pal, code) * s_x[k - k_tile];
                }
            }
        }
        __syncthreads();
    }

    if (!valid) return;

    // Folded sidecar: reads from global x (not s_x which only holds last tile).
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
        for (uint32_t si = lo + lane; si < hi; si += 32) {
            uint32_t gidx = sc_idx[si];
            uint32_t col  = gidx - (uint32_t)row_weight_base;
            uint32_t g    = gidx >> 3;
            uint32_t sub  = gidx & 7;
            const uint8_t* gp = ws + (uint64_t)g * 5;
            uint64_t v = ((uint64_t)gp[0] << 32) | ((uint64_t)gp[1] << 24) |
                         ((uint64_t)gp[2] << 16) | ((uint64_t)gp[3] << 8) | (uint64_t)gp[4];
            uint8_t code = (uint8_t)((v >> (5 * (7 - sub))) & 0x1F);
            uint32_t pal_sc;
            __builtin_memcpy(&pal_sc, bpal_base + ((uint64_t)gidx >> 8) * 4, sizeof(pal_sc));
            float approx = sclp5_decode_code(pal_sc, code);
            uint16_t tb = sc_val[si];
            float trueval = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&tb));
            acc += (trueval - approx) * x[col];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) y[row] = acc;
}

void llama_sclp5_fused_gemv(
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
    sclp5_fused_gemv_kernel<<<gemv_grid, gemv_block, smem_bytes, stream>>>(
        (const uint8_t*)blob_ptr, src_f32, dst_f32, N, K);
}

// K-tiled fused MoE GEMV for SCLP5, with folded sidecar correction.
__launch_bounds__(512, 4)
__global__ void sclp5_fused_moe_gemv_kernel(
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
    float* s_x = (float*)smem;

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
        uint32_t expert_ws_bytes = ((expert_nw + 7) / 8) * 5;
        s_ws_offset = ws_start + (uint32_t)e * expert_ws_bytes;

        uint64_t ws_total = (uint64_t)n_experts * expert_ws_bytes;
        uint32_t sc_base = ws_start + (uint32_t)ws_total;
        s_sc_base = sc_base;
        __builtin_memcpy(&s_sc_count, blob + sc_base, sizeof(uint32_t));
    }
    __syncthreads();

    const float* x = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;

    const uint8_t* bpal_base = blob + s_bpal_offset;
    const uint8_t* ws        = blob + s_ws_offset;
    const int warps_per_block = blockDim.x / 32;
    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x & 31;
    const uint32_t row = (uint32_t)blockIdx.x * warps_per_block + warp_id;

    const bool valid = (row < N);

    float acc = 0.0f;
    const uint32_t groups_per_row  = valid ? (K + 7) / 8 : 0;
    const uint64_t row_group_base  = valid ? (uint64_t)row * groups_per_row : 0;
    const uint64_t row_weight_base = valid ? (uint64_t)row * K : 0;

    for (uint32_t k_tile = 0; k_tile < K; k_tile += TILE_K) {
        const uint32_t tile_end = min(k_tile + TILE_K, K);
        const uint32_t tile_len = tile_end - k_tile;

        for (uint32_t k = threadIdx.x; k < tile_len; k += blockDim.x) {
            s_x[k] = x[k_tile + k];
        }
        __syncthreads();

        if (valid) {
            const uint32_t g_start = k_tile / 8;
            const uint32_t g_end   = (tile_end + 7) / 8;
            uint32_t cached_block_idx = UINT32_MAX;
            uint32_t pal = 0;

            for (uint32_t g = g_start + lane; g < g_end; g += 32) {
                const uint8_t* gp = ws + (row_group_base + g) * 5;
                uint64_t v = ((uint64_t)gp[0] << 32) | ((uint64_t)gp[1] << 24) |
                             ((uint64_t)gp[2] << 16) | ((uint64_t)gp[3] << 8) | (uint64_t)gp[4];
                uint32_t k0 = g * 8;
                uint32_t block_idx = (uint32_t)((row_weight_base + k0) >> 8);
                if (block_idx != cached_block_idx) {
                    cached_block_idx = block_idx;
                    __builtin_memcpy(&pal, bpal_base + (uint64_t)block_idx * 4, sizeof(pal));
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    uint32_t k = k0 + j;
                    if (k < k_tile || k >= tile_end) continue;
                    uint8_t code = (uint8_t)((v >> (5 * (7 - j))) & 0x1F);
                    acc += sclp5_decode_code(pal, code) * s_x[k - k_tile];
                }
            }
        }
        __syncthreads();
    }

    if (!valid) return;

    // Folded sidecar: indices are global (expert_offset + row*K + col).
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
            uint32_t g   = local_idx >> 3;
            uint32_t sub = local_idx & 7;
            const uint8_t* gp = ws + (uint64_t)g * 5;
            uint64_t v = ((uint64_t)gp[0] << 32) | ((uint64_t)gp[1] << 24) |
                         ((uint64_t)gp[2] << 16) | ((uint64_t)gp[3] << 8) | (uint64_t)gp[4];
            uint8_t code = (uint8_t)((v >> (5 * (7 - sub))) & 0x1F);
            uint32_t pal_sc;
            __builtin_memcpy(&pal_sc, bpal_base + ((uint64_t)local_idx >> 8) * 4, sizeof(pal_sc));
            float approx = sclp5_decode_code(pal_sc, code);
            uint16_t tb = sc_val[si];
            float trueval = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&tb));
            acc += (trueval - approx) * x[col];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) dst[row + (uint64_t)flat * N] = acc;
}

void llama_sclp5_fused_moe_gemv(
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
    sclp5_fused_moe_gemv_kernel<<<grid, block, smem_bytes, stream>>>(
        (const uint8_t*)blob_ptr,
        src1, ids, dst, N, K, n_active, src1_ne1);
}

// Fused decode-WMMA kernel for SCLP5 MoE prefill (M > 1).
__launch_bounds__(32, 8)
__global__ void sclp5_fused_moe_wmma_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ src1,
    const int32_t* __restrict__ perm,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ ids,
    float*         __restrict__ dst,
    uint32_t N, uint32_t K,
    uint32_t n_active, uint32_t src1_ne1, uint32_t n_experts
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
    __shared__ __hip_bfloat16 s_XT[TILE][TILE + 1];

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
        uint32_t expert_ws_bytes = ((expert_nw + 7) / 8) * 5;
        s_ws_offset = ws_start + (uint32_t)e * expert_ws_bytes;
    }

    if (threadIdx.x < TILE) {
        uint32_t m_g = m_base + threadIdx.x;
        s_m_global[threadIdx.x] = perm[m_g];
    }
    __syncthreads();

    const uint8_t* bpal_base = blob + s_bpal_offset;
    const uint8_t* ws = blob + s_ws_offset;
    const int lane     = threadIdx.x;
    const int frag_row = lane % TILE;
    const uint32_t n_base = (uint32_t)blockIdx.x * TILE;
    const uint32_t n_row  = n_base + (uint32_t)frag_row;
    const bool valid = n_row < N;

    using bf16x16_t = __attribute__((ext_vector_type(16))) __bf16;
    using floatx8_t  = __attribute__((ext_vector_type(8))) float;
    floatx8_t acc = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};

    for (uint32_t k16 = 0; k16 < K; k16 += TILE) {
        bf16x16_t a_frag;
        const uint64_t row_weight_base = (uint64_t)n_row * K;
        uint64_t group_v[2] = {0, 0};
        uint32_t group_pal[2] = {0, 0};
        bool group_fast[2] = {false, false};
        if (valid) {
            #pragma unroll
            for (int gg = 0; gg < 2; gg++) {
                const uint32_t kg = k16 + (uint32_t)gg * 8;
                const uint64_t w_idx = row_weight_base + kg;
                if (kg + 7 < K && (w_idx & 255u) <= 248u) {
                    const uint8_t* gp = ws + (w_idx >> 3) * 5;
                    group_v[gg] = ((uint64_t)gp[0] << 32) | ((uint64_t)gp[1] << 24) |
                                  ((uint64_t)gp[2] << 16) | ((uint64_t)gp[3] << 8)  | (uint64_t)gp[4];
                    __builtin_memcpy(&group_pal[gg], bpal_base + ((w_idx >> 8) * 4), sizeof(uint32_t));
                    group_fast[gg] = true;
                }
            }
        }
        #pragma unroll
        for (int j = 0; j < TILE; j++) {
            const uint32_t kg = k16 + j;
            if (valid && kg < K) {
                uint64_t w_idx = row_weight_base + kg;
                const int gg = j >> 3;
                float w;
                if (group_fast[gg]) {
                    uint8_t code = (uint8_t)((group_v[gg] >> (5 * (7 - (j & 7)))) & 0x1F);
                    w = sclp5_decode_code(group_pal[gg], code);
                } else {
                    uint32_t pal;
                    __builtin_memcpy(&pal, bpal_base + ((w_idx >> 8) * 4), sizeof(pal));
                    w = sclp5_decode_weight_at(ws, w_idx, pal);
                }
                a_frag[j] = __float2bfloat16(w);
            } else {
                a_frag[j] = (__bf16)0.f;
            }
        }

        {
            const uint32_t k_local = frag_row;
            const uint32_t kg = k16 + k_local;
            #pragma unroll
            for (int j = 0; j < TILE; j += (32 / TILE)) {
                const uint32_t m_local = lane / TILE + j;
                if (m_local < TILE) {
                    int32_t mg = s_m_global[m_local];
                    if (mg >= 0 && kg < K) {
                        uint32_t i_active = (uint32_t)mg % n_active;
                        uint32_t i_batch  = (uint32_t)mg / n_active;
                        const float* x_row = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;
                        s_XT[k_local][m_local] = (__hip_bfloat16)x_row[kg];
                    } else {
                        s_XT[k_local][m_local] = (__hip_bfloat16)0.f;
                    }
                }
            }
        }
        __syncthreads();

        bf16x16_t b_frag;
        #pragma unroll
        for (int j = 0; j < TILE; j++) {
            b_frag[j] = *(const __bf16*)&s_XT[frag_row][j];
        }
        __syncthreads();

        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, acc);
    }

    if (valid) {
        const int col_lo = lane / TILE;
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


// Blocked sidecar correction for SCLP5 MoE fused output.
// Mirrors the SCLP4 blocked sidecar kernel but decodes SCLP5's 5-byte/8-weight
// ws stream on the fly. Each block owns one (n_tile, m_tile) output tile and
// accumulates corrections for the routed slots that map into that expert.
__launch_bounds__(128, 1)
__global__ void sclp5_moe_sidecar_correct_blocked_kernel(
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

    __shared__ uint32_t s_expert_nw;
    __shared__ uint32_t s_bpal_offset_e;
    __shared__ uint32_t s_ws_offset_e;
    __shared__ uint32_t s_sc_base_offset;
    __shared__ uint32_t s_sc_total;
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
        const uint8_t* p = blob + 8;
        for (uint32_t ei = 0; ei < ne; ei++) { p += 1 + p[0]; }
        uint32_t expert_nw = total_nw / ne;
        s_expert_nw = expert_nw;
        uint32_t bpal_start = (uint32_t)(p - blob);
        uint32_t bpe = expert_nw / QK_SCLP4;
        s_bpal_offset_e = bpal_start + (uint32_t)e * bpe * 4;
        uint32_t ws_start = bpal_start + (total_nw / QK_SCLP4) * 4;
        uint32_t expert_ws_bytes = ((expert_nw + 7) / 8) * 5;
        s_ws_offset_e = ws_start + (uint32_t)e * expert_ws_bytes;

        uint64_t total_ws_bytes = (uint64_t)ne * expert_ws_bytes;
        uint32_t sc_base_off = ws_start + (uint32_t)total_ws_bytes;
        s_sc_base_offset = sc_base_off;
        const uint8_t* sc_base = blob + sc_base_off;
        uint32_t sc_total;
        __builtin_memcpy(&sc_total, sc_base, sizeof(uint32_t));
        s_sc_total = sc_total;
        const uint8_t* sc_idx = sc_base + 4;

        uint32_t target_lo = (uint32_t)e * expert_nw;
        uint32_t target_hi = ((uint32_t)e + 1) * expert_nw;

        uint32_t lo = 0, hi = sc_total;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo) / 2;
            uint32_t val;
            __builtin_memcpy(&val, sc_idx + (uint64_t)mid * 4, 4);
            if (val < target_lo) lo = mid + 1; else hi = mid;
        }
        s_sc_range_begin = lo;

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

    const uint32_t expert_nw = s_expert_nw;
    const uint32_t sc_total = s_sc_total;
    const uint8_t* sidecar_base = blob + s_sc_base_offset;
    const uint8_t* sc_idx_base = sidecar_base + 4;
    const uint8_t* sc_val_base = sc_idx_base + (uint64_t)sc_total * sizeof(uint32_t);
    const uint8_t* ws_e = blob + s_ws_offset_e;
    const uint8_t* bpal_e = blob + s_bpal_offset_e;

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
        uint32_t block_idx = (uint32_t)(w_idx >> 8);
        uint32_t pal;
        __builtin_memcpy(&pal, bpal_e + (uint64_t)block_idx * 4, sizeof(pal));
        float w_approx = sclp5_decode_weight_at(ws_e, w_idx, pal);
        float w_correct = __bfloat162float(*(__hip_bfloat16*)&correct_bits);
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

void llama_sclp5_fused_moe_wmma(
    const void*    blob_ptr,
    const float*   src1,
    const int32_t* ids,
    float*         dst,
    float*         dst_pre_sidecar,
    int32_t*       perm_scratch,
    int32_t*       expert_offsets_scratch,
    uint32_t       N,
    uint32_t       K,
    uint32_t       n_active,
    uint32_t       n_batches,
    uint32_t       ids_s1,
    uint32_t       src1_ne1,
    uint32_t       n_experts,
    hipStream_t    stream
) {
    const uint32_t total_slots = n_active * n_batches;
    constexpr int TILE = 16;

    sclp_moe_route_sort_kernel<<<1, 256, 2 * (n_experts + 1) * sizeof(int32_t), stream>>>(
        ids, perm_scratch, expert_offsets_scratch, total_slots, n_active, ids_s1, n_experts, (uint32_t)TILE);

    const uint32_t m_tile_count = (total_slots + n_experts * TILE + TILE - 1) / TILE;
    dim3 block(32);
    dim3 grid((N + TILE - 1) / TILE, m_tile_count);
    sclp5_fused_moe_wmma_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t*)blob_ptr,
        src1, perm_scratch, expert_offsets_scratch, ids, dst,
        N, K, n_active, src1_ne1, n_experts);

    if (dst_pre_sidecar != nullptr) {
        const size_t out_elems = (size_t)N * (size_t)n_active * (size_t)n_batches;
        hipMemcpyAsync(dst_pre_sidecar, dst, out_elems * sizeof(float), hipMemcpyDeviceToDevice, stream);
    }

    sclp5_moe_sidecar_correct_blocked_kernel<<<dim3((N + TILE - 1) / TILE, m_tile_count), 128, 0, stream>>>(
        (const uint8_t*)blob_ptr,
        src1, perm_scratch, expert_offsets_scratch, dst,
        N, K, n_active, src1_ne1, n_experts);
}
