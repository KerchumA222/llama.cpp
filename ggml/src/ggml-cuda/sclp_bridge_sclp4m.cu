#include "sclp_bridge_common.cuh"

// ============================================================
// SCLP4M kernels — 4 bits/weight, per-block 8-entry BF16 magnitude codebook.
// Nibble layout: bits[3:1]=codebook idx (0..7), bit[0]=sign.
// Decode (no exponent assembly, no mantissa, no per-block scale):
//   idx  = nibble >> 1;  sign = nibble & 1;
//   mag  = cb[idx]      (BF16 bits, sign already clear — magnitude only);
//   bits = mag | (sign << 15);  w = bf16(bits).
// Header: [u32 num_weights][u32 n_experts][per-expert: u8 palette_size=0]
// Block codebooks: 16 bytes (8 × BF16) per QK_SCLP4 (=256) block, experts contiguous,
//   block_idx = e * (expert_nw/256) + local/256.
// ws_stream: per-expert nibble sections, ceil(expert_nw/2) bytes each; high nibble = even.
// Sidecar v2 follows the ws_stream (see sclp_bridge_common.cuh).
// ============================================================

// Decode one weight from a 16-byte codebook + nibble.
static __device__ __forceinline__ uint16_t sclp4m_decode_bits(const uint8_t* cb, uint8_t nibble) {
    uint16_t mag;
    __builtin_memcpy(&mag, cb + (uint32_t)(nibble >> 1) * 2, sizeof(uint16_t));
    return (uint16_t)(mag | ((uint16_t)(nibble & 1) << 15));
}

static __device__ __forceinline__ float sclp4m_decode_f32(const uint8_t* cb, uint8_t nibble) {
    uint16_t bits = sclp4m_decode_bits(cb, nibble);
    return __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&bits));
}

__global__ void sclp4m_decode_blob_kernel(
    const uint8_t* __restrict__ blob,
    uint16_t*      __restrict__ output,
    uint32_t                    num_weights
) {
    __shared__ uint32_t s_n_experts;
    __shared__ uint32_t s_cb_start;
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
        s_cb_start = (uint32_t)(p - blob);
        uint32_t enw = num_weights / ne;
        s_expert_nw = enw;
        s_blocks_per_expert = enw / QK_SCLP4;
        s_ws_start = s_cb_start + (num_weights / QK_SCLP4) * 16;
        s_expert_nibble_bytes = (enw + 1) / 2;
    }
    __syncthreads();

    const uint8_t* cb_base = blob + s_cb_start;
    const uint8_t* ws      = blob + s_ws_start;
    uint32_t expert_nw = s_expert_nw;

    const uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr uint32_t W_PER_THREAD = 32;
    const uint32_t base_idx = thread_idx * W_PER_THREAD;

    if (base_idx >= num_weights) return;

    const uint32_t e          = base_idx / expert_nw;
    const uint32_t local_base = base_idx - e * expert_nw;
    if (e >= s_n_experts) return;

    const uint8_t* ws_p = ws + (uint64_t)e * s_expert_nibble_bytes + local_base / 2;
    uint32_t remaining_expert = expert_nw - local_base;
    uint32_t n_this = min(W_PER_THREAD, remaining_expert);

    uint16_t out[32];
    if (n_this == 32) {
        // 32 weights = 16 nibble bytes. They may straddle a 256-weight block
        // boundary, so refresh the codebook pointer when the block index changes.
        uint32_t cached_block = UINT32_MAX;
        const uint8_t* cb = cb_base;
        for (int q = 0; q < 2; q++) {
            uint64_t ws8;
            __builtin_memcpy(&ws8, ws_p + q * 8, 8);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint8_t byte = (uint8_t)(ws8 >> (i * 8));
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    uint32_t w = base_idx + q * 16 + i * 2 + j;
                    uint32_t block_idx = e * s_blocks_per_expert + ((local_base + q * 16 + i * 2 + j) / QK_SCLP4);
                    if (block_idx != cached_block) {
                        cached_block = block_idx;
                        cb = cb_base + (uint64_t)block_idx * 16;
                    }
                    uint8_t nibble = (j == 0) ? (byte >> 4) : (byte & 0xF);
                    out[q * 16 + i * 2 + j] = sclp4m_decode_bits(cb, nibble);
                    (void)w;
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
            uint32_t block_idx = e * s_blocks_per_expert + ((local_base + i) / QK_SCLP4);
            const uint8_t* cb = cb_base + (uint64_t)block_idx * 16;
            output[base_idx + i] = sclp4m_decode_bits(cb, nibble);
        }
    }
}

__global__ void sclp4m_fixup_sidecar_kernel(
    const uint8_t* __restrict__ blob,
    uint16_t*      __restrict__ output,
    uint32_t                    num_weights
) {
    __shared__ uint32_t s_ws_start;
    __shared__ uint32_t s_n_experts;

    if (threadIdx.x == 0) {
        uint32_t ne;
        __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        s_n_experts = ne;
        const uint8_t* p = blob + 8;
        for (uint32_t e = 0; e < ne; e++) { p += 1 + p[0]; }
        uint32_t cb_start = (uint32_t)(p - blob);
        s_ws_start = cb_start + (num_weights / QK_SCLP4) * 16;
    }
    __syncthreads();

    const uint8_t* ws    = blob + s_ws_start;
    uint32_t expert_nw   = num_weights / s_n_experts;
    uint64_t total_nibble_bytes = (uint64_t)s_n_experts * ((expert_nw + 1) / 2);
    const uint8_t* sidecar_base = ws + total_nibble_bytes;

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

// Fused decode-GEMV for SCLP4M, M=1 (dense). K-tiled, warp-per-row, folded v2 sidecar.
__launch_bounds__(512, 4)
__global__ void sclp4m_fused_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ x,
    float*         __restrict__ y,
    uint32_t N,
    uint32_t K
) {
    constexpr uint32_t TILE_K = 4096;
    extern __shared__ char smem[];
    __shared__ uint32_t s_cb_start;
    __shared__ uint32_t s_ws_start;
    __shared__ uint32_t s_sc_base;
    float* s_x = (float*)smem;

    if (threadIdx.x == 0) {
        uint32_t ne; __builtin_memcpy(&ne, blob + 4, sizeof(uint32_t));
        const uint8_t* p = blob + 8;
        for (uint32_t ei = 0; ei < ne; ei++) p += 1 + p[0];
        s_cb_start = (uint32_t)(p - blob);
        uint64_t nw = (uint64_t)N * K;
        s_ws_start = s_cb_start + (uint32_t)((nw / QK_SCLP4) * 16);
        uint32_t enw = (uint32_t)(nw / ne);
        uint64_t ws_total = (uint64_t)ne * ((enw + 1) / 2);
        s_sc_base = s_ws_start + (uint32_t)ws_total;
    }
    __syncthreads();

    const uint8_t* cb_base = blob + s_cb_start;
    const uint8_t* ws      = blob + s_ws_start;

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
            const uint8_t* cb = cb_base;
            const uint32_t b_start = k_tile / 2;
            const uint32_t b_end   = (tile_end + 1) / 2;

            for (uint32_t b = b_start + lane; b < b_end; b += 32) {
                uint8_t byte = ws[row_byte_base + b];
                uint32_t k0  = b * 2;
                uint32_t block_idx = (uint32_t)((row_weight_base + k0) >> 8);
                if (block_idx != cached_block_idx) {
                    cached_block_idx = block_idx;
                    cb = cb_base + (uint64_t)block_idx * 16;
                }

                acc += sclp4m_decode_f32(cb, byte >> 4) * s_x[k0 - k_tile];

                if (k0 + 1 < tile_end) {
                    acc += sclp4m_decode_f32(cb, byte & 0xF) * s_x[k0 + 1 - k_tile];
                }
            }
        }
        __syncthreads();
    }

    if (!valid) return;

    // Folded sidecar v2: O(1) row-range lookup.
    const sclp_sidecar_view sc = sclp_sidecar_parse(blob + s_sc_base, (uint32_t)((uint64_t)N * K));
    if (sc.count > 0) {
        uint32_t lo = 0, hi = 0;
        if (lane == 0) sclp_sidecar_row_range(sc, row, &lo, &hi);
        lo = __shfl(lo, 0); hi = __shfl(hi, 0);
        for (uint32_t e = lo + lane; e < hi; e += 32) {
            uint32_t col  = sclp_sidecar_col(sc, e);
            uint64_t gidx = row_weight_base + col;
            uint8_t byte  = ws[gidx >> 1];
            uint8_t nib   = (gidx & 1) ? (byte & 0xF) : (byte >> 4);
            const uint8_t* cb = cb_base + (uint64_t)(gidx >> 8) * 16;
            float approx  = sclp4m_decode_f32(cb, nib);
            uint16_t tb   = sclp_sidecar_val(sc, e);
            float trueval = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&tb));
            acc += (trueval - approx) * x[col];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) y[row] = acc;
}

// Fused decode-GEMV for SCLP4M MoE, single-token generation (n_batches=1).
// One block per (output_row_tile, active_expert_slot); one warp per output row.
__launch_bounds__(512, 4)
__global__ void sclp4m_fused_moe_gemv_kernel(
    const uint8_t* __restrict__ blob,
    const float*   __restrict__ src1,
    const int32_t* __restrict__ ids,
    float*         __restrict__ dst,
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t src1_ne1
) {
    constexpr uint32_t TILE_K = 4096;
    extern __shared__ char smem[];
    __shared__ uint32_t s_cb_offset;
    __shared__ uint32_t s_ws_offset;
    __shared__ uint32_t s_sc_base;
    __shared__ uint32_t s_total_nw;
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
        uint32_t cb_start = (uint32_t)(p - blob);
        uint32_t bpe = expert_nw / QK_SCLP4;
        s_cb_offset = cb_start + (uint32_t)e * bpe * 16;

        uint32_t ws_start = cb_start + (total_nw / QK_SCLP4) * 16;
        uint32_t expert_nibble_bytes = (expert_nw + 1) / 2;
        s_ws_offset = ws_start + (uint32_t)e * expert_nibble_bytes;

        uint64_t ws_total = (uint64_t)n_experts * expert_nibble_bytes;
        s_sc_base = ws_start + (uint32_t)ws_total;
        s_total_nw = total_nw;
    }
    __syncthreads();

    const float* x = src1 + (uint64_t)(i_batch * src1_ne1 + (i_active % src1_ne1)) * K;

    const uint8_t* cb_base = blob + s_cb_offset;
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
            const uint8_t* cb = cb_base;
            const uint32_t b_start = k_tile / 2;
            const uint32_t b_end   = (tile_end + 1) / 2;

            for (uint32_t b = b_start + lane; b < b_end; b += 32) {
                uint8_t byte = ws[row_byte_base + b];
                uint32_t k0  = b * 2;
                uint32_t block_idx = (uint32_t)((row_weight_base + k0) >> 8);
                if (block_idx != cached_block_idx) {
                    cached_block_idx = block_idx;
                    cb = cb_base + (uint64_t)block_idx * 16;
                }

                acc += sclp4m_decode_f32(cb, byte >> 4) * s_x[k0 - k_tile];

                if (k0 + 1 < tile_end) {
                    acc += sclp4m_decode_f32(cb, byte & 0xF) * s_x[k0 + 1 - k_tile];
                }
            }
        }
        __syncthreads();
    }

    if (!valid) return;

    // Folded sidecar v2: global row = e * N + row, O(1) range lookup.
    const sclp_sidecar_view sc = sclp_sidecar_parse(blob + s_sc_base, s_total_nw);
    if (sc.count > 0) {
        const uint32_t grow = (uint32_t)e * N + row;
        uint32_t lo = 0, hi = 0;
        if (lane == 0) sclp_sidecar_row_range(sc, grow, &lo, &hi);
        lo = __shfl(lo, 0); hi = __shfl(hi, 0);
        for (uint32_t si = lo + lane; si < hi; si += 32) {
            uint32_t col = sclp_sidecar_col(sc, si);
            uint32_t local_idx = (uint32_t)row_weight_base + col;
            uint8_t byte = ws[(uint64_t)local_idx >> 1];
            uint8_t nib  = (local_idx & 1) ? (byte & 0xF) : (byte >> 4);
            const uint8_t* cb = cb_base + (uint64_t)((uint64_t)local_idx >> 8) * 16;
            float approx = sclp4m_decode_f32(cb, nib);
            uint16_t tb = sclp_sidecar_val(sc, si);
            float trueval = __bfloat162float(*reinterpret_cast<__hip_bfloat16*>(&tb));
            acc += (trueval - approx) * x[col];
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) acc += __shfl_down(acc, offset);
    if (lane == 0) dst[row + (uint64_t)flat * N] = acc;
}

// ------------------------------------------------------------
// Launchers
// ------------------------------------------------------------

void llama_sclp4m_dispatch(
    const void* sclp_data,
    uint16_t*   output,
    uint32_t    num_weights,
    hipStream_t stream,
    bool        apply_sidecar = true
) {
    const uint8_t* data = (const uint8_t*)sclp_data;
    dim3 block(256);

    uint32_t groups = (num_weights + 31) / 32;
    dim3 decode_grid((groups + 255) / 256);

    sclp4m_decode_blob_kernel<<<decode_grid, block, 0, stream>>>(data, output, num_weights);

    if (apply_sidecar) {
        sclp4m_fixup_sidecar_kernel<<<1024, block, 0, stream>>>(data, output, num_weights);
    }
}

void llama_sclp4m_fused_gemv(
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
    sclp4m_fused_gemv_kernel<<<gemv_grid, gemv_block, smem_bytes, stream>>>(
        (const uint8_t*)blob_ptr, src_f32, dst_f32, N, K);
    SCLP_TG_TIME_END();
}

void llama_sclp4m_fused_moe_gemv(
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
    sclp4m_fused_moe_gemv_kernel<<<grid, block, smem_bytes, stream>>>(
        (const uint8_t*)blob_ptr, src1, ids, dst, N, K, n_active, src1_ne1);
    SCLP_TG_TIME_END();
}
