#pragma once
#include "sclp_bridge_common.cuh"

constexpr int SCLP_GEMM_TILE_M = 16;

// SCLP8
void llama_sclp_dispatch(
    const void* sclp_data, uint16_t* output,
    uint32_t num_weights, hipStream_t stream);

void llama_sclp_fused_gemv(
    const void* blob_ptr, const float* src_f32, float* dst_f32,
    uint32_t N, uint32_t K, hipStream_t stream);

void llama_sclp_fused_gemm(
    const void* blob_ptr, const float* src_f32, float* dst_f32,
    uint32_t N, uint32_t K, uint32_t M, void* tmp_bf16, hipStream_t stream);

void llama_sclp_fused_wmma(
    const void* blob_ptr, const float* src_f32, float* dst_f32,
    uint32_t N, uint32_t K, uint32_t M, void* tmp_bf16, hipStream_t stream);

void llama_sclp_fused_moe_gemv(
    const void* blob_ptr, const float* src1, const int32_t* ids, float* dst,
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t n_batches,
    uint32_t src1_ne1, hipStream_t stream);

// SCLP4
void llama_sclp4_dispatch(
    const void* sclp_data, uint16_t* output,
    uint32_t num_weights, hipStream_t stream, bool apply_sidecar = true);

void llama_sclp4_fused_gemv(
    const void* blob_ptr, const float* src_f32, float* dst_f32,
    uint32_t N, uint32_t K, hipStream_t stream);

void llama_sclp4_fused_moe_gemv(
    const void* blob_ptr, const float* src1, const int32_t* ids, float* dst,
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t n_batches,
    uint32_t src1_ne1, hipStream_t stream);

void llama_sclp4_fused_moe_wmma(
    const void* blob_ptr, const float* src1, const int32_t* ids,
    float* dst, float* dst_pre_sidecar,
    int32_t* perm_scratch, int32_t* expert_offsets_scratch,
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t n_batches,
    uint32_t ids_s1, uint32_t src1_ne1, uint32_t n_experts,
    uint32_t scalar_math_mode, uint32_t sidecar_mode, hipStream_t stream);

// SCLP5
void llama_sclp5_dispatch(
    const void* sclp_data, uint16_t* output,
    uint32_t num_weights, hipStream_t stream, bool apply_sidecar = true);

void llama_sclp5_fused_gemv(
    const void* blob_ptr, const float* src_f32, float* dst_f32,
    uint32_t N, uint32_t K, hipStream_t stream);

void llama_sclp5_fused_moe_gemv(
    const void* blob_ptr, const float* src1, const int32_t* ids, float* dst,
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t n_batches,
    uint32_t src1_ne1, hipStream_t stream);

void llama_sclp5_fused_moe_wmma(
    const void* blob_ptr, const float* src1, const int32_t* ids,
    float* dst, float* dst_pre_sidecar,
    int32_t* perm_scratch, int32_t* expert_offsets_scratch,
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t n_batches,
    uint32_t ids_s1, uint32_t src1_ne1, uint32_t n_experts,
    hipStream_t stream);

// SCLP6
void llama_sclp6_dispatch(
    const void* sclp_data, uint16_t* output,
    uint32_t num_weights, uint32_t n_experts, hipStream_t stream);

void llama_sclp6_fused_gemv(
    const void* blob_ptr, const float* src_f32, float* dst_f32,
    uint32_t N, uint32_t K, uint32_t expert_idx, hipStream_t stream);

void llama_sclp6_fused_moe_gemv(
    const void* blob_ptr, const float* src1, const int32_t* ids, float* dst,
    uint32_t N, uint32_t K, uint32_t n_active, uint32_t n_batches,
    uint32_t src1_ne1, hipStream_t stream);
