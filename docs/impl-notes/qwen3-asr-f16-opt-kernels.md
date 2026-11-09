# Qwen3-ASR f16 optimized kernels — the op contract (Q5)

The "slightly optimized" variant of Qwen3-ASR phase 1 (`qwen3-asr-port-plan.md` decision 5, Q5): the same model
with the obvious fusions and a reasonable kernel per device — **not** profiling-driven tuning. Every op exists twice
with the same signature: `<name>_f16_cpu` (`kernels/cpu/f16_fused.cpp`, ops in `python/lokinfer/ops/f16.py`) and
`<name>_f16_gpu` (`kernels/gpu/f16_fused.metal` + `kernels/gpu/f16_gpu.cpp`, ops in `python/lokinfer/ops/f16_gpu.py`).
Tests: `tests/kernels/test_f16_fused_{cpu,gpu}.py` against **the composition of the basic f16 kernels** (the oracle,
`qwen3-asr-f16-kernels.md`) and the float64 torch composition, same tolerance style as the basic tests, real shapes
(M = 1 for decode, 44 / 401 / 2048 for prefill chunks, the encoder's 15,600 rows) + odd sizes; on Mac and iPhone.

Rules: f16 storage, f32 accumulation and epilogue math, one rounding at the output. Weights may be pre-packed at
build time (a `*_pack` helper in Python or a `prepare` step: the layout is the kernel's choice); the contract fixes
only the math. Every GEMM-like op must be fast for **M = 1** (decode: a bandwidth-bound GEMV, multithreaded K- or
N-split on the CPU) and for large M.

| op | signature → output | semantics |
|---|---|---|
| `gemm_opt` | `(x f16[M, K], w f16[K, N]) → f16[M, N]` | `x @ w` |
| `gemm_bias_opt` | `(x, w, b f16[N]) → f16[M, N]` | `x @ w + b` |
| `gemm_bias_gelu_opt` | `(x, w, b) → f16[M, N]` | `gelu_erf(x @ w + b)` (encoder fc1, projector 1) |
| `gemm_add_opt` | `(x, w, res f16[M, N]) → f16[M, N]` | `res + x @ w` (LM o_proj / down_proj + residual) |
| `gemm_bias_add_opt` | `(x, w, b, res) → f16[M, N]` | `res + (x @ w + b)` (encoder out_proj + residual) |
| `gemm_bias_add_clamp_opt` | `(x, w, b, res, lim) → f16[M, N]` | `clamp(res + (x @ w + b), ±lim)` (encoder fc2 + residual + the f16 clamp) |
| `gemm_swiglu_opt` | `(x f16[M, K], w f16[K, 2N]) → f16[M, N]` | `y = x @ w`; `silu(y[:, :N]) · y[:, N:]` (w = [gate \| up] concatenated) |
| `rmsnorm_gemm_opt` | `(x f16[M, K], g f16[K], w f16[K, N], eps) → (y f16[M, N], ws f16[M, K])` | `y = rmsnorm(x, g) @ w` (the basic `rmsnorm` roundings, then `gemm_opt`); `ws` is a workspace (the normalized rows when M > 1 or on the CPU; unspecified otherwise) — added 2026-11-05 (perf): the GPU decode step runs it as one dispatch (`lk_f16o_rms_gemv`, each threadgroup normalizes the row with `lk_f16_rmsnorm`'s exact sum order), bit-identical to the composition |
| `rmsnorm_gemm_swiglu_opt` | `(x, g, w f16[K, 2N], eps) → (y f16[M, N], ws)` | `gemm_swiglu_opt(rmsnorm(x, g), w)`, same workspace rule |
| `qknorm_rope_kv_opt` | `(qkv f16[T, 2048 + 2·1024], qg f16[128], kg f16[128], pos i64[1], kc f16[P, 1024] inout, vc f16[P, 1024] inout, eps, theta) → f16[T, 2048]` | q = RoPE(rmsnorm_heads(qkv[:, :2048], qg)) is returned; k = RoPE(rmsnorm_heads(qkv[:, 2048:3072], kg)) and v = qkv[:, 3072:] are written to `kc[pos:pos+T]`, `vc[pos:pos+T]` (the basic `rmsnorm_heads` / `rope` / `kv_write` semantics, incl. rounding k to f16 before the cache, as the basic composition does) |
| `attention_gqa_opt` | `(q f16[T, 2048], kc, vc f16[P, 1024], pos i64[1], q_heads, kv_heads) → f16[T, 2048]` | the basic `attention_gqa` semantics; flash-style (tiles of keys, online f32 softmax); T = 1 must be fast too (key-split decode) |
| `attention_window_qkv_opt` | `(qkv f16[S, 3·1024], heads, window) → f16[S, 1024]` | the basic `attention_window` on q = qkv[:, :1024], k = qkv[:, 1024:2048], v = qkv[:, 2048:] |
| `audio_cnn_opt` | the basic `audio_cnn` signature (weights may be pre-packed) | same math (conv layers rounded to f16 between layers), conv as im2col + the fast GEMM with the bias + GELU epilogue, chunk by chunk |
| `lm_head_last_opt` | `(x f16[T, D], table f16[V, D]) → f32[1, V]` | the basic `lm_head_last`; a fast bandwidth-bound GEMV over the 151,936 × 2048 table |

Basic ops reused unchanged by the optimized graph: `embedding`, `scatter_audio`, `rmsnorm` (the per-layer norms on
the residual stream), `layernorm`, `add_pos_prefix`, `gelu_erf` (projector output path if not fused). The graph
(`models/qwen3_asr/model.py`, variant `"opt"`) is written by the coordinator against this table.

Per-device kernel choices (decision 5): GPU — simdgroup-matrix (`simdgroup_half8x8`, f32 accumulators) or MPS f16
for the large GEMMs, a `gemv4`-style f16 GEMV for M = 1, simdgroup flash attention with f32 softmax. Mac CPU — NEON
`FMLAL` (f16 × f16 → f32) register-blocked tiles over the thread pool (Accelerate has no f16 GEMM), K-split GEMV for
M = 1 (GPT-2's scheme). iPhone CPU — the same NEON path, plus an SME2 `FMOPA` (f16 → f32) path in `sme2_gemm` where
it wins (runtime-checked like the f32 SME2 kernels). Side measurement for the CPU agent: `BFMMLA` (bf16) vs `FMLAL`
(f16) GEMM tile throughput on the Mac, recorded in `docs/models/qwen3-asr.md` (no change to the f16 decision).
