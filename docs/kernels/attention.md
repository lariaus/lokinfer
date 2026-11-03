# Attention kernels

Kernel docs: [convolution](convolution.md) · [gemm](gemm.md) · [elementwise](elementwise.md) ·
[reductions](reductions.md) · **attention**

Every scaled-dot-product attention in lokinfer, CPU and GPU: full (non-causal) attention for Kokoro's ALBERT,
causal prefill for GPT-2, and decode / chunked attention over persistent KV caches with the cache writes. All f32,
row-major. The QKV and output projections around them are GEMMs ([gemm.md](gemm.md)); the add + bias + LayerNorm
after them is in [reductions.md](reductions.md).

Devices: **Mac** = M2 Max (30-core GPU), **iPhone** = iPhone 18 Pro Max ([baselines.md](../baselines.md)).

Sources: `kernels/cpu/kokoro_misc.cpp` (prefill / full attention), `kernels/cpu/decode.cpp` (KV caches),
`kernels/gpu/kokoro_tok.metal` (all GPU attention), `kernels/gpu/kokoro_lc.cpp` and `kernels/gpu/transformer_gpu.cpp`
(dispatch). Python defs: `python/lokinfer/ops/{kokoro,kokoro_gpu,transformer}.py`. Models: `models/gpt2.py`,
`models/kokoro/model.py`.

## 1. Overview

Common input: the **fused QKV projection** `qkv [T, 3D]` (`q | k | v`, D = heads × head_dim), so no split / transpose
kernels exist. On the CPU the QKV GEMM has the bias fused, so `qkv` arrives biased; on the GPU the QKV GEMM is a
plain MPS GEMM and the attention kernels add the q / k / v biases on read (`bias [3D]` operand). Scale is
1/√head_dim. Output `[T, D]`, heads interleaved like the input.

| registered name | device | kernel | inputs | mask | used by | constraints / selection |
|---|---|---|---|---|---|---|
| `attention_f32_cpu` | CPU | `attention_f32` | separate q, k, v `[T, D]` | none | kernel tests only | – |
| `attention_qkv_f32_cpu` | CPU | `attention_qkv_f32` | qkv (biased) | none | Kokoro ALBERT (CPU build) | heads on the pool when T²·hd·heads ≥ 2²¹ |
| `attention_qkv_causal_f32_cpu` | CPU | same, causal | qkv | causal | GPT-2 CPU `cpu="basic"`, or `cattn=None` | same |
| `attention_qkv_causal_blk_f32_cpu` | CPU | `attention_qkv_causal_blk_f32` | qkv | causal, key-trimmed | GPT-2 CPU prefill (default `cattn=(128, 0)`) | attrs `qb` (query block, clamped to 1..256; Python default 64, GPT-2 passes 128), `mode` 0 = pool, 1 = sequential |
| `attention_qkv_f32_gpu`, `attention_qkv_causal_f32_gpu` | GPU | `lk_attention_qkv_flash` | qkv + bias | none / causal | GPT-2 GPU `attn="flash"`; kernel tests | head_dim 64; one 128-thread threadgroup per (32-query block, head); Python def also checks T ≤ 1024 |
| `attention_qkv_sg_f32_gpu`, `attention_qkv_sg_causal_f32_gpu` | GPU | `lk_attention_qkv_sg{2,4,8}` | qkv + bias | none / causal | Kokoro ALBERT GPU (non-causal), GPT-2 GPU prefill (default `attn="sg"`, causal), `build_generate` prefill at pos 0 | head_dim 64; `nsg` ∈ {2, 4, 8} simdgroups (8 queries each) per threadgroup, default 4 |
| `kv_write_f32_cpu` | CPU | memcpy | caches `[P, D]` (inout), qkv `[1, 3D]`, pos `i64[1]` | – | GPT-2 CPU `build_decode` | pos < P |
| `attention_decode_f32_cpu` | CPU | NEON | qkv `[1, 3D]`, caches, pos | rows 0..pos | GPT-2 CPU `build_decode` | heads on the pool when (pos+1)·D ≥ 16 384 |
| `kv_write_rows_f32_cpu` | CPU | memcpy | caches (inout), qkv `[T, 3D]`, pos | – | GPT-2 CPU `build_generate` | pos + T ≤ P |
| `attention_cache_f32_cpu` | CPU | 2 sgemm + vDSP softmax per head | qkv `[T, 3D]`, caches, pos | query i: rows 0..pos+i | GPT-2 CPU `build_generate` | heads on the pool when T·(pos+T)·hd·heads ≥ 2²¹ |
| `kv_write_f32_gpu` | GPU | `lk_kv_write` | caches (inout), qkv, pos, bias | – | GPT-2 GPU `build_decode(attn="old")` | adds k / v bias |
| `attention_decode_f32_gpu` | GPU | `lk_attention_decode` | qkv, caches, pos, bias | rows 0..pos | GPT-2 GPU `build_decode(attn="old")` | head_dim 64, cap ≤ 1024 |
| `attention_decode_kvw_f32_gpu` | GPU | `lk_attention_decode_kvw` | qkv, caches (inout), pos, bias | rows 0..pos | GPT-2 GPU `build_decode` (default `attn="kvw"`), `build_generate` T = 1 | head_dim 64, cap ≤ 1024; fuses the cache write |
| `kv_write_rows_f32_gpu` | GPU | `lk_kv_write_rows` | caches (inout), qkv `[T, 3D]`, pos, bias | – | `build_generate` T > 1 | adds k / v bias |
| `attention_cache_f32_gpu` | GPU | `lk_attention_cache_flash` | qkv, caches, pos, bias | query i: rows 0..pos+i | `build_generate` T > 1 with pos > 0, and `kernels="old"` | head_dim 64 |
| `attention_gen_f32_gpu` | GPU | dispatcher | qkv, caches (inout), pos, bias | causal over cache | GPT-2 GPU `build_generate` (default `kernels="auto"`) | per call: T = 1 → `attention_decode_kvw`; T > 1 → `kv_write_rows`, then `attention_qkv_sg_causal` (nsg 4) if pos = 0 else `attention_cache` |

`lk_attention_qkv` (one 128-thread threadgroup per (query, head), all scores in threadgroup memory, T ≤ 1024) was
removed from `kokoro_tok.metal` (2026-11-02; nothing dispatched it): `attention_qkv_f32_gpu` runs the flash kernel.

## 2. How each works

### CPU full / causal attention (`kokoro_misc.cpp`)

`attention_qkv_f32` (and `attention_f32` on separate q, k, v): per head, `S = scale · Q_h K_hᵀ` (`detail::sgemm` with
`CblasTrans` on the strided qkv view, ld = 3D), a vectorized row softmax (`vDSP_maxv`, `vDSP_vsadd`, `vvexpf`,
`vDSP_sve`, `vDSP_vsmul`; causal rows only over their prefix and zero-filled after it), then `O_h = S V_h`. Heads are
independent: one per pool task with single-threaded BLAS each (`parallelForSt`) once T²·hd·heads ≥ 2²¹, else
sequential.

`attention_qkv_causal_blk_f32` (GPT-2 prefill): work items (head, query block of `qb` rows), largest (last) blocks
first. Each item computes scores only against keys `0 .. i1` (the block's causal extent: half the T×T work), a NEON
softmax (`exp4`, two accumulators) that returns 1/sum instead of scaling the row, the `P V` GEMM, then the 1/sum
scale applied to the 64-wide output rows. Mode 0 runs items on the pool (single-threaded BLAS each), mode 1
sequentially.

### CPU decode and chunked attention (`decode.cpp`)

- `kv_write` / `kv_write_rows`: copy k and v rows of `qkv` (biases already added by the CPU QKV GEMM) into cache rows
  `pos .. pos+T-1`, in place (`inout` operands); the position is a runtime `i64[1]` input and is range-checked (the
  kernel sets `ctx.error` if it is out of the cache).
- `attention_decode`: one query, plain NEON, one head per pool task (sequential while (pos+1)·D < 16 384): scores
  `K_h q_h` with 4 accumulators of 4 lanes (hd % 16 = 0), max, NEON `exp4` with the sum, then `out_h = Σ p_r V_h[r]`
  with 16 vector accumulators for hd = 64.
- `attention_cache`: T queries at positions `pos0 + i`; per head one sgemm for `Q Kᵀ` over the `n = pos0 + T` visible
  cache rows, vDSP softmax over each row's visible prefix (zero after it), one sgemm for `P V`.

### GPU flash attention, scalar (`lk_attention_qkv_flash`, `lk_attention_cache_flash`)

One threadgroup (128 threads) per (32-query block, head). Each 32-key K/V tile is loaded into threadgroup memory once
(biases added; rows padded to 65 floats) and reused by the 32 queries. 4 threads per query split the 64 dims (16
each), partial dots reduced with two `simd_shuffle_xor`; online softmax per query (running max `m`, sum `l`,
accumulator rescaled by `exp(m - m_new)`). Causal: the block stops at its last query's key, the diagonal tile is
masked. `lk_attention_cache_flash` is the same scheme reading K/V from the cache (`kmax = pos0 + min(T, block end)`,
mask `j ≤ pos0 + i`); only q gets its bias on read since the cache rows are already biased.

### GPU flash attention on simdgroup matrices (`lk_attention_qkv_sg{NSG}`)

One threadgroup of NSG simdgroups per (8·NSG-query block, head); simdgroup s owns query rows 8s..8s+7 and keeps its
Q fragments (8 × `simdgroup_float8x8`, bias and `scale · log2 e` applied) in registers. Per 32-key tile, K and V go
through threadgroup memory as float4 loads (shared by the simdgroups, row stride 68). `S = Q Kᵀ` with 4 independent
accumulators (8 MMAs each, K loaded transposed), online softmax in the exp2 domain directly on the fragment elements
(each lane holds 2 adjacent columns of one row; a row lives on lanes {x, x^1, x^8, x^9}, so row max is two
`simd_shuffle_xor`), `O = diag(corr) O + P V` with 8 independent accumulators; row sums stay per lane until the end.
Biases: k's is dropped (q·b_k is constant over keys, softmax-invariant), v's is added to the output (weights sum
to 1). Causal grids run the longest (last) query blocks first; a simdgroup skips tiles past its own rows but still
helps load them. Every loop is fully unrolled (`clang loop unroll(full)`); results are f32 (Apple GPUs have no TF32)
up to summation order.

### GPU decode attention (`lk_attention_decode`, `lk_attention_decode_kvw`)

- `lk_attention_decode` (+ separate `lk_kv_write`): one 128-thread threadgroup per head, scores for all cache rows in
  threadgroup memory (cap ≤ 1024), max / sum by SIMD reductions, then `Σ p_j v_j` split over (dim, key half).
- `lk_attention_decode_kvw` (fused): one 1024-thread threadgroup (32 simdgroups) per head. Lane = 2 dims (float2,
  coalesced 256-byte row reads); simdgroup sg takes keys sg, sg+32, … with 4 keys per iteration (`UN = 4`); dots by
  `simd_sum`. This step's own k / v come from the qkv row (+ bias) and are written to cache row `pos` by simdgroup 0
  of the same threadgroup, so the write and the read need no barrier between kernels. Pass 1: scores to threadgroup
  memory + per-simdgroup max; pass 2: `p = exp2(s - max)` and `p·v` partial sums per simdgroup, combined through
  threadgroup memory.

### Persistent KV caches and `build_generate`

`lk.persistent_buffer(f32, [1024, 768], device)` per layer for K and V (GPT-2: 12 layers × 2): model-lifetime state,
never freed or planned into the arena, written in place by `inout` kernels ([python-api.md](../python-api.md),
[compiler.md](../compiler.md)). `build_decode` runs one token per call (`kv_write` + `attention_decode`, or the fused
`attention_decode_kvw`). `build_generate` compiles one graph with symbolic T so prefill and decode share the caches;
on the GPU `attention_gen_f32_gpu` prepares its four delegates and picks per call: T = 1 → the fused decode kernel;
T > 1 → `kv_write_rows`, then, if the chunk starts at position 0 (pos is a graph input, readable at encode time), the
simdgroup-matrix kernel over qkv itself (the chunk only attends to its own rows), else the cache-tile flash kernel.
The CPU `build_generate` path always uses `kv_write_rows` + `attention_cache`.

### Kokoro ALBERT

12 layers sharing one set of weights, 12 heads × 64, non-causal, T = tokens + 2 (≤ 512). CPU build: fused QKV GEMM with
bias (`gemm_bias_f32_cpu_accelerate`) → `attention_qkv_f32_cpu`. GPU build (`albert_gpu`): fused QKV MPS GEMM →
`attention_qkv_sg_f32_gpu` (nsg 4) with the biases folded into the reads; the CPU TextEncoder overlaps it through async
commit ([kokoro.md](../models/kokoro.md)).

## 3. Measured performance

Attention timings in the sources are mostly e2e model deltas; kernel-only numbers are marked as such. Per-kernel
times in any model: `tools/prof-lokinfer` ([profiling.md](../profiling.md)).

| what | device | case | result | source |
|---|---|---|---|---|
| CPU softmax vectorized + heads on the pool (Kokoro attention, kernel total) | Mac | Kokoro CPU | 122 → 12 ms | [todo.md](../todo.md) M3 |
| CPU causal blocked attention (e2e) | Mac / iPhone | GPT-2 prefill T = 512 | 74.6 → 70.3 ms / 87.5 → 79.0 ms (tuning-log row: iPhone 77.4 → 75.0) | [gpt2.md](../models/gpt2.md) |
| CPU GPT-2 attention (kernel, before) | iPhone | prefill T = 128 | 1.6 ms | [gpt2.md](../models/gpt2.md) |
| CPU old per-head `cblas_sgemv` + vDSP decode attention (kernel, 12 layers) | iPhone | P = 512 | 0.96 ms, single-threaded → replaced by NEON per head on the pool | `decode.cpp` |
| CPU fused decode step (K-split GEMVs + parallel NEON attention, e2e) | Mac / iPhone | GPT-2 decode | 10.9 → 7.9 ms / 6.68 → 5.46 ms | [gpt2.md](../models/gpt2.md) |
| GPU scalar flash vs one-query-per-threadgroup (e2e) | Mac | GPT-2 prefill T = 512 | 23.5 → 20.9 ms | [todo.md](../todo.md) M4 |
| GPU simdgroup-matrix flash (kernel) | Mac | GPT-2 prefill T = 512 | 6.4 → 1.36 ms | [gpt2.md](../models/gpt2.md), [todo.md](../todo.md) M6 |
| same (e2e) | Mac / iPhone | GPT-2 T = 128 / 512 | 6.1 → 5.4 / 19.9 → 14.9 ms; 11.8 → 11.46 / 43.3 → 39.2 ms | [gpt2.md](../models/gpt2.md) |
| same for Kokoro ALBERT | Mac | rand510 | ~10 ms → below the profile's top 12 | [todo.md](../todo.md) M6 |
| GPU fused kv-write + 1024-thread decode attention (kernel, per step) | Mac | GPT-2 decode | 1.29 → 0.26 ms | [gpt2.md](../models/gpt2.md) |
| same (e2e decode step) | Mac / iPhone | GPT-2 decode | 3.55 → 2.70 ms / 6.9 → 6.85 ms | [gpt2.md](../models/gpt2.md) |
| `lk_attention_decode_kvw` UN = 4 vs 8 (per step) | Mac | GPT-2 decode | 0.258 vs 0.288 ms | `kokoro_tok.metal`, [gpt2.md](../models/gpt2.md) |
| `attention_gen` + `gemm_auto` (e2e decode step) | Mac | `build_generate`, P = 512 | 6.0 → 2.05 ms | [gpt2.md](../models/gpt2.md) |
| GPT-2 GPU dashboard (e2e, all kernels) | Mac / iPhone | T = 128 / 512 / decode | 4.99 / 15.1 / 2.01 ms; 11.45 / 39.2 / 5.57 ms | [gpt2.md](../models/gpt2.md) |

Not measured: CPU ALBERT attention as a separate kernel number after the pool change, `nsg` 2 / 8 vs 4, the scalar
cache-tile kernel vs the simdgroup kernel on later prefill chunks.

## 4. Tried and rejected

- **`lk_attention_qkv`** (one threadgroup per (query, head)): re-read all K/V per query, memory-bound; replaced by the
  32-query flash kernel (GPT-2 T = 512 e2e 23.5 → 20.9 ms) ([todo.md](../todo.md) M4, `kokoro_tok.metal`).
- **Scalar flash as the default**: replaced by the simdgroup-matrix kernel for GPT-2 and ALBERT (6.4 → 1.36 ms at
  T = 512); still selectable with `build(attn="flash")` ([gpt2.md](../models/gpt2.md)).
- In the simdgroup kernel: **register prefetch of the next K/V tile** and **K/V straight from device memory** (no
  threadgroup memory) — both slower; **without full unrolling** the fragment arrays spill (3× slower)
  ([gpt2.md](../models/gpt2.md), [profiling.md](../profiling.md)).
- **Separate `kv_write` + 128-thread decode attention** on the GPU: replaced by the fused 1024-thread kernel (1.29 →
  0.26 ms per step); **8-deep key unrolling** in it was slower (0.288 vs 0.258 ms) ([gpt2.md](../models/gpt2.md)).
  The old path stays selectable (`build_decode(attn="old")`).
- **MPS + scalar cache attention for every T** in `build_generate` (`kernels="old"`): decode step 6.0 ms vs 2.05 ms
  with the per-call dispatch ([gpt2.md](../models/gpt2.md)).
- **CPU**: the per-head `cblas_sgemv` + vDSP decode loop (single-threaded, 0.96 ms for 12 layers at P = 512 on the
  iPhone) → NEON per head on the pool (`decode.cpp`); the full T×T causal kernel → the key-trimmed blocked kernel
  ([gpt2.md](../models/gpt2.md)).

## 5. Open ideas

From [todo.md](../todo.md) (M6) and [research-kernels.md](../impl-notes/research-kernels.md) §2.5; none is built.
- `build_generate` on the GPU: later prefill chunks (pos > 0) still use the scalar cache-tile attention, not the
  simdgroup-matrix kernel.
- Mac decode attention uses one threadgroup per head (12 threadgroups on the GPU); split the keys across threadgroups
  (~0.2 ms/step noted as the target).
- Re-check whether the old scalar decode kernels were only missing the full-unroll fix before calling them slower.
- iPhone GPU: f32 Metal 4 tensor ops (todo M2, Kokoro GPU agent's list); the research note expects TensorOps-based
  flash attention to win on the iPhone's GPU (not measured).
- Known limit (not a todo item): the GPU decode kernels (`lk_attention_decode`, `lk_attention_decode_kvw`) hold the
  scores in a 1024-float threadgroup array, so cache capacity is ≤ 1024 (GPT-2's context). The research note's
  reference design for long caches is MLX's `sdpa_vector` with a 2-pass split over keys (research-kernels.md §2.5).
