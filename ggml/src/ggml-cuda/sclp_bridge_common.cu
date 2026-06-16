#include "sclp_bridge_common.cuh"

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
