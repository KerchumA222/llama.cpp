# SCLP4 Performance Investigation Handoff

## Goal

Improve SCLP4 quality and throughput to match or exceed Q4_K_M on Llama-3-8B (the reference model for apples-to-apples comparison). Currently SCLP4 is ~10× worse on PPL and ~2× slower on prefill.

## Current Baselines (Llama-3-8B, RX 7900 XTX, `-ngl 99`)

| Config | Size | tg128 (t/s) | pp512 (t/s) | PPL (wiki-2) |
|---|---|---|---|---|
| Q4_K_M | 4.69 GB | ~52 | ~3,430 | **9.99** |
| SCLP4 (per-block palette) | 4.86 GB | 26.4 | 2,445 | **~102** |
| BF16 | 14.97 GB | ~52 | ~12,000 | 10.59 |

Numbers verified 2026-05-26 after the prefill stall fix (see below).

## What Just Got Fixed

The SCLP4 M>1 prefill was hanging entirely — root cause was the **last tensor in GGUF** having its `disk_size` computed as `ggml_nbytes(tensor)` instead of from the file boundary. For compact SCLP4 blobs, `ggml_nbytes` is smaller than the actual blob (misses sidecar data), so the sidecar fixup kernel read garbage → infinite loop → GPU hang. Fix is in `src/llama-model-loader.h` (committed `sclp-turboquant` branch). This also fixed the llama-bench stall.

## Two Gaps to Close

### 1. Quality Gap (~10× PPL)

**Root cause**: SCLP4 nibble layout is `idx(2b) | sign(1b) | mant_top1(1b)`. With a 4-entry per-block exponent palette and 1 mantissa bit, each weight can only represent `{1.0, 1.5} × 2^exp` — 8 exponentially-spaced values per block (vs Q4_K's 16 linearly-spaced values per 32-weight block). For near-Gaussian weight distributions, linear spacing wins decisively.

**Possible improvements** (in rough priority order):

A. **Rebalance nibble to `idx(1b) | sign(1b) | mant_top2(2b)`** — k=2 block palette (2 exponents per 256-weight block) but 2 mantissa bits → `{1.0, 1.25, 1.5, 1.75} × 2^exp`. This is 8 levels like before but more evenly spaced within each exponent. Format-breaking: touches encoder (`src/llama-sclp.cpp`), decode kernel (`sclp_bridge.cuh:934`), fused GEMV (`sclp_bridge.cuh:1107`), MoE GEMV (`sclp_bridge.cuh:1551`), MoE WMMA (`sclp_bridge.cuh:1782`), CPU decode (`ggml/src/ggml-cpu/ggml-cpu.c`).

B. **Increase imatrix sidecar budget** — current default is 1%. On Gemma4 MIXED, going from 0% → 1% sidecar gave 15× PPL improvement. SCLP4 on Llama-3-8B may benefit from higher budgets (2-5%), especially for attention tensors.

C. **Per-tensor type policy** — attention at SCLP6, gate/up at SCLP4 (the "MIXED" config). Already documented in CLAUDE.md. On Gemma4 this achieves OOD PPL 132.6 at 14.9 GiB (vs Q4_K-based hybrid at 290.4 / 15.8 GiB). But on Llama-3-8B, pure SCLP4 PPL ~102 vs Q4_K 9.99 — MIXED helps but doesn't close the gap.

D. **Non-uniform mantissa allocation** — give high-importance weights (per imatrix) more mantissa bits by promoting them to sidecar. Already partially done via imatrix-sidecar; the question is whether the budget curve has more room.

### 2. Throughput Gap (tg: 26 vs 52, pp: 2445 vs 3430)

**tg (token generation, M=1)**: SCLP4 fused GEMV (`sclp4_fused_gemv_kernel`, `sclp_bridge.cuh:1107`) does warp-per-row decode + dot product. 26.4 t/s is 51% of Q4_K's 52 t/s. Possible bottlenecks:
- Shared memory broadcast of `x` vector (K floats per block)
- Per-byte nibble unpacking + block palette LUT lookup
- Sidecar binary search per row (sorted sidecar, folded in)
- The fused GEMV was only recently re-enabled with folded sidecar; may have low-hanging optimization fruit

**pp (prefill, M>1)**: SCLP4 uses **two-pass decode** — decode entire weight matrix to BF16, then call rocBLAS GEMM. This has two costs:
1. Full weight matrix read + write (decode pass)
2. BF16 rocBLAS GEMM (which has a known Tensile bug on gfx1100 for large-M bf16 — see "rocBLAS issue" below)

Q4_K goes directly through rocBLAS INT8 path — no decode pass needed. Closing this gap would require either:
- A fused decode+GEMM kernel (attempted, abandoned due to F32 accumulation order divergence — see CLAUDE.md "Fused MoE prefill")
- Routing two-pass through F16 or F32 GEMM to avoid the broken BF16 Tensile kernel (correctness workaround, not a speed win)

### rocBLAS BF16 GEMM Bug (gfx1100, ROCm 7.2.2/7.2.3)

The two-pass prefill decodes to BF16 then calls rocBLAS for a BF16 matmul. On this box, rocBLAS's Tensile library has a logic/code-object mismatch for `Cijk_Alik_Bljk_BBS_BH_MT128x128` (the large-M bf16 kernel for gfx1100). The `.dat` file selects a solution that isn't compiled into the `.co`. rocBLAS sometimes falls back to `.hsaco`, sometimes hangs. This affects ALL large-M bf16 GEMM, not just SCLP.

The prefill stall fix (disk_size) resolved the immediate hang, but this rocBLAS bug may still cause degraded prefill perf or intermittent failures on large-M shapes. Upgrading rocBLAS or routing through F32/F16 are the workarounds.

## Key Files

| File | What's there |
|---|---|
| `ggml/src/ggml-cuda/sclp_bridge.cuh` | All GPU kernels: decode (L934), sidecar fixup (L1038), fused GEMV (L1107), fused MoE GEMV (L1551), fused MoE WMMA (L1782) |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Dispatch logic: dense mul_mat (L2700), mul_mat_id MoE (L2911), alloc_size (L855), supports_op |
| `src/llama-sclp.cpp` | C++ encoder (llama-quantize integration), k-means palette, imatrix sidecar |
| `src/llama-sclp.h` | Encoder API |
| `ggml/src/ggml-cpu/ggml-cpu.c` | CPU decode fallback |
| `src/llama-model-loader.h` | disk_size computation (the just-fixed code, L50-60) |

## Repro Commands

```bash
cd /home/ajkerchum/llama.cpp

# Build (ROCm, gfx1100)
cmake -B build2 -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
cmake --build build2 --config Release -j$(nproc)

# Token generation speed
build2/bin/llama-completion \
    -m /home/ajkerchum/poc/models/llama3/Meta-Llama-3-8B-SCLP4-sorted.gguf \
    -ngl 99 -n 100 -no-cnv --repeat-penalty 1.3 \
    -p "The capital of France is"

# llama-bench (now works)
build2/bin/llama-bench \
    -m /home/ajkerchum/poc/models/llama3/Meta-Llama-3-8B-SCLP4-sorted.gguf \
    -ngl 99 -n 128 -p 512 -r 1

# PPL (use -fa off if it hangs — rocBLAS bf16 bug workaround)
build2/bin/llama-perplexity \
    -m /home/ajkerchum/poc/models/llama3/Meta-Llama-3-8B-SCLP4-sorted.gguf \
    -ngl 99 -c 512 -b 512 --chunks 20 -fa off

# Q4_K baseline for comparison
build2/bin/llama-bench \
    -m /home/ajkerchum/poc/models/llama3/Meta-Llama-3-8B-Q4KM.gguf \
    -ngl 99 -n 128 -p 512 -r 1

# Regenerate SCLP4 GGUF (if format changes)
build2/bin/llama-quantize \
    --tensor-type '^token_embd\.weight$=BF16' \
    --tensor-type '^output\.weight$=BF16' \
    /home/ajkerchum/poc/models/llama3/Meta-Llama-3-8B.fp16.gguf \
    /home/ajkerchum/poc/models/llama3/Meta-Llama-3-8B-SCLP4-new.gguf SCLP4 8
```

## Important Caveats

1. **Always smoke-test with 200+ token generation** before trusting PPL — PPL can miss mode collapse.
2. **BF16 source required** — F16 bits fed to the SCLP encoder produce garbage (different exponent/mantissa layout).
3. **`-fa off` for perplexity** — flash-attn + logits_all + SCLP4 can hang (rocBLAS bf16 bug). Real inference (no logits_all) is fine.
4. **Don't `kill -9` GPU processes** — wedges WSL2/ROCm GPU; use `timeout` wrapper. Recovery needs `wsl --shutdown`.
5. **Old SCLP4 GGUFs are incompatible** with sorted-sidecar fused GEMV — binary search on unsorted sidecar → 0.16 t/s. Always regenerate after format changes.
6. **CMake cache traps** — `-D` flags persist across rebuilds. If a debug toggle won't turn off, delete `CMakeCache.txt`.

## Throughput Follow-Up

Benchmarks rerun on 2026-05-26 with the current build on RX 7900 XTX after a low-risk kernel cleanup pass.

| Config | tg128 (t/s) before | tg128 (t/s) after | pp512 (t/s) before | pp512 (t/s) after | Note |
|---|---:|---:|---:|---:|---|
| SCLP4 | 26.50 | 34.01 | 2647.49 | 2641.46 | Packed palette + cached block palette in fused GEMV / sidecar path |
| SCLP5 | 29.00 | 40.02 | 2654.01 | 2645.11 | Same palette cache pattern applied cleanly |
| SCLP6 | 33.73 | 37.75 | 2783.03 | 2783.03 | Replaced the 64-entry shared LUT with direct palette decode in the hot loop |
| SCLP8 | 42.75 | 42.84 | 2690.12 | 2648.51 | Palette-cache rewrite was effectively flat on tg and noisy on pp |

Takeaway: the repeated-palette read pattern was a real win in SCLP4/SCLP5, but SCLP6 needed a different cleanup. Removing the shared LUT from the hot loop recovered most of the tg gap; SCLP8 still did not show a reliable throughput improvement from the same approach.

## SCLP5 MoE Branch

Added a dedicated SCLP5 `mul_mat_id` MoE path so SCLP5 no longer falls through to the generic non-MoE dispatcher behavior.

- Single-token generation (`n_batches == 1`) now uses the fused SCLP5 MoE GEMV path.
- Prefill now uses a fused SCLP5 MoE WMMA path by default, including blocked sidecar correction, so it no longer needs the full BF16 expert buffer.
- Set `SCLP5_FUSED_MOE_WMMA=0` to force the old two-pass decode + recursive GEMM path for bisecting.
- Build validation passed after wiring the branch into `ggml/src/ggml-cuda/ggml-cuda.cu`.
- I did not run a runtime benchmark for SCLP5 MoE because the local MoE model on disk is SCLP4 MoE (`Qwen3.6-SCLP6attn-SCLP4moe.gguf`), not SCLP5 MoE.

Practical note: the 21 GiB Qwen3.6 MoE file fits within the 24 GiB card budget, but it exercises the SCLP4 MoE kernels, so it is not a valid benchmark for the new SCLP5 branch.
