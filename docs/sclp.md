# SCLP — Soft Clipping Lossless-First Compression

SCLP is a family of weight compression formats for BF16 neural network weights. It encodes each weight as a palette index (exponent) + sign + truncated mantissa, achieving 4–8 bits per weight with lossless sidecar recovery for outliers.

## Quick Start

### Quantize a model

```bash
# Single-type: all linear projections at SCLP6 (6 bits/weight)
llama-quantize \
    --tensor-type '^token_embd\.weight$=BF16' \
    --tensor-type '^output\.weight$=BF16' \
    input-bf16.gguf output.gguf SCLP6 8

# Mixed precision (recommended for Gemma4 / MoE):
#   SCLP6 for attention + ffn_down, SCLP4 for gate/up
llama-quantize \
    --imatrix imatrix.dat \
    --tensor-type '^token_embd\.weight$=BF16' \
    --tensor-type '^output\.weight$=BF16' \
    --tensor-type '^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$=SCLP6' \
    --tensor-type '^blk\.[0-9]+\.ffn_down(_exps)?\.weight$=SCLP6' \
    input-bf16.gguf output.gguf SCLP4 8
```

> **Important:** The input GGUF must contain BF16 weights. F16 bits have a different exponent/mantissa layout (1-5-10 vs 1-8-7) and will produce garbage.

> **Flag order:** `--tensor-type` and `--imatrix` must come **before** the positional arguments. Arguments after the input file path are silently ignored.

### Run inference

```bash
llama-completion -m output.gguf -ngl 99 -n 200 -no-cnv --repeat-penalty 1.3 \
    -p "The capital of France is"
```

SCLP types are GPU-only (`MUL_MAT`). Embeddings and output projections must remain at a native type (BF16, Q6_K, etc.) since they use `GET_ROWS` on the CPU.

## Format Variants

| Type | Bits/weight | Palette entries | Mantissa bits | Use case |
|------|-------------|-----------------|---------------|----------|
| SCLP8 | 8 | 16 (global) | 3 | Highest quality, ~2x compression |
| SCLP6 | 6 | 8 (global) | 2 | Attention weights, error-sensitive layers |
| SCLP5 | 5 | 4 (per-block) | 2 | Intermediate (pareto-dominated by SCLP4+imatrix) |
| SCLP4 | 4 | 4 (per-block) | 1 | Bulk MLP weights, best bits-per-byte |

### Bit layout

Each weight is encoded as a fixed-width code:

- **SCLP8**: `idx(4) | sign(1) | mant(3)` — 1 byte
- **SCLP6**: `idx(3) | sign(1) | mant(2)` — 6 bits, packed 4 weights per 3 bytes
- **SCLP5**: `idx(2) | sign(1) | mant(2)` — 5 bits, packed 8 weights per 5 bytes
- **SCLP4**: `idx(2) | sign(1) | mant(1)` — 4 bits, packed 2 weights per byte

### Sidecar

Weights whose exponent falls outside the palette are stored verbatim (full BF16) in a sidecar section. Typically 0.01–3% of weights depending on the palette size. The sidecar is lossless — these weights are restored exactly.

### imatrix-aware sidecar

When an importance matrix is provided (`--imatrix`), the encoder promotes additional high-activation-importance weights to sidecar storage. This targets the highest-impact quantization errors. Use `--sidecar-imatrix-budget 0.01` (1%, the default) for the best size/quality tradeoff.

## Benchmarks

### Llama-3-8B (RX 7900 XTX, dense)

| Config | Size | tg128 (t/s) | pp512 (t/s) |
|--------|------|-------------|-------------|
| BF16 | 14.97 GB | ~52 | ~12,000 |
| Q8_0 | 7.95 GB | ~52 | ~3,430 |
| SCLP8 | 7.92 GB | 43 | ~2,650 |
| SCLP6 | 6.71 GB | 38 | ~2,780 |
| SCLP5 | 5.72 GB | 40 | ~2,650 |
| SCLP4 | 4.86 GB | 34 | ~2,640 |

Token generation benefits from reduced memory bandwidth (fewer bits per weight read). Prefill uses a two-pass decode (SCLP blob → BF16 → rocBLAS GEMM), which adds overhead vs formats that go directly through rocBLAS.

### Gemma4-26B-A4B-IT (RX 7900 XTX, MoE)

OOD perplexity on held-out agentic traces (opus-trace, 50 chunks):

| Config | Size | tg (t/s) | OOD PPL |
|--------|------|----------|---------|
| MIXED (SCLP6 attn+down, SCLP4 gate/up, imatrix) | 14.9 GiB | 55 | 132.6 |
| SCLP6+Q4_K hybrid | 15.8 GiB | 50 | 290.4 |
| SCLP6attn (SCLP6 attn, SCLP4 rest) | 14.6 GiB | — | 40.0 |
| Pure SCLP4 | 14.3 GiB | — | 97.1 |

## Architecture

### GPU kernels (`ggml/src/ggml-cuda/sclp_bridge_*.cu`)

Each SCLP type has its own compilation unit for optimal register allocation:

- **Two-pass decode**: `sclp*_decode_blob_kernel` parses the blob header on-device, decodes to BF16. `sclp_fixup_sidecar_kernel` scatter-writes sidecar values. Used for prefill (M > 1).
- **Fused GEMV**: `sclp*_fused_gemv_kernel` decodes weights inline and accumulates the dot product in a single pass. One warp per output row, K-tiled for occupancy. Used for token generation (M = 1).
- **Fused MoE GEMV**: `sclp*_fused_moe_gemv_kernel` decodes only the routed experts' weights, avoiding the 16x waste of decoding all experts.
- **Folded sidecar**: The encoder sorts sidecar entries by weight index. Each fused GEMV row binary-searches its contiguous sidecar range and applies corrections inline — no atomics, no second kernel.

### C++ encoder (`src/llama-sclp.cpp`)

- Per-expert parallel encoding (one thread per expert)
- K-means palette selection (20 iterations, k-means++ init)
- imatrix-aware sidecar promotion
- Sidecar indices sorted for fused GEMV compatibility

### CPU decode (`ggml/src/ggml-cpu/ggml-cpu.c`)

Single-threaded fallback that decodes SCLP blobs to BF16, then delegates to BF16 `mul_mat`. Supports all four types including per-block palette (SCLP4/5) and per-block scaling (SCLP6/8).

### Compact GGUF storage

Each SCLP tensor is stored at its actual compressed size (not padded to `ggml_nbytes`). The loader infers `disk_size` from consecutive tensor offsets. This saves significant disk space — Llama-3-8B SCLP8 is 7.9 GB vs 15.0 GB BF16.

## Mixed Precision Recommendations

| Workload | Config | Rationale |
|----------|--------|-----------|
| Chat / agentic (decode-bound) | SCLP6 attn + SCLP4 gate/up | Best PPL per byte; attention errors compound through softmax |
| Long-context / RAG (prefill-bound) | SCLP6 attn + Q4_K gate/up | Q4_K goes directly through rocBLAS, +60% prefill |
| 16 GB VRAM target | Q6_K embeds + SCLP6 attn + SCLP4 rest | 14.6 GiB, leaves room for KV cache |

## Known Limitations

- **GPU-only for MUL_MAT.** CPU fallback works but is single-threaded and slow.
- **ROCm/HIP only.** CUDA support requires porting the HIP kernels (straightforward but not yet done).
- **BF16 input required.** F16 GGUFs must be converted to BF16 first.
- **Two-pass prefill.** Prefill lags behind native integer quants that go directly through rocBLAS/cuBLAS.
- **Sorted sidecar required.** Old GGUFs with unsorted sidecars cause catastrophic slowdown in fused GEMV (binary search returns bogus ranges). Regenerate after any format change.
