# GPT-2 124M on lokinfer (prefill)

HF `GPT2LMHeadModel` "openai-community/gpt2" (pinned revision), f32. lokinfer runs the **prefill forward**: token ids
`i64[T]` (B = 1, T symbolic up to 1024) → logits `f32[T, 50257]` (`last_only=True`: the last row only). The
tokenizer is outside the graph. Plan: `impl-notes/research-models.md` §5.

Status (2026-11-01): **prefill, decode and generation built and tested on CPU and GPU** (`models/gpt2.py`,
`tests/models/test_gpt2.py`). The decode step uses persistent KV caches (`lk.persistent_buffer`): greedy generation
matches HF `generate(do_sample=False)` token for token.

## Decode

`M.build_decode(sd, device)`: inputs token id `i64[1]` and its position `i64[1]` → logits `f32[1, 50257]`. Per layer
two persistent caches `f32[1024, 768]` (K, V; model-owned state): `kv_write` stores this step's K/V row at `pos`
in place (`inout`), `attention_decode` attends over rows `0..pos`. `M.greedy_generate(step, prompt, n)` feeds the
prompt one token at a time, then the argmax.

GPU decode (**built**, 2026-11-01): `attention_decode_kvw_f32_gpu` fuses the cache write into the decode attention
(one 1024-thread threadgroup per head: lane = 2 dims, 32 simdgroups split the keys, dots by `simd_sum`; this step's
own k/v come from the QKV row and are written to cache row `pos` by the same threadgroup). The GEMVs are
`gemv4_f32_gpu` (`kokoro_tok.metal` `lk_gemv4_c{2,4,8,16}`: each 256-thread threadgroup computes `cw` float4 output
columns over `256 / cw` K slices, 4-deep unrolled loads, fused bias + gelu for `c_fc`); `cw = 0` picks the
platform's tuned value at run time (macOS 4, iOS 8). The LM head stays MPS (`gemm_nt`, ~385 GB/s on the M2 Max).
Old paths stay selectable: `build_decode(..., gemm="mps", attn="old")`.
Key-split decode attention (2026-11-02, `attn="split"`, default): `attention_decode_split_f32_gpu` =
`lk_attention_decode_split{4,8,16,32}[f]` over a grid (heads, splits), each split writing a partial (max, sum, p.v)
that the last split of the head to finish merges in the same dispatch (device atomic counter, reset by it; `single=0`
= a second `lk_attention_decode_combine` dispatch, same speed). On the M2 Max it only pays for long contexts, so the
auto policy runs the one-threadgroup-per-head `kvw` kernel below 768 cached keys and 4 splits x 16 simdgroups above
(e2e step: P=16 kvw 1.87 vs split 1.92 ms, P=512 2.02 vs 2.02, P=1000 2.23-2.29 vs 2.17-2.25; 8x8, 4x8, 8x16, 16x4
never better). The per-kernel profile overstated the gain: its ~22 µs per attention includes ~10-14 µs of fixed
per-dispatch cost (P=16 reads 14 µs), the real data cost at P=512 is ~0.15 ms per step. iPhone: always `kvw`
(split not measured there yet).

CPU decode (**built**, 2026-11-01): `cpu_split=0` (default) is the fused step: GEMVs through `gemm_accel_f32_cpu`
(auto blocking = a **K split**: contiguous row ranges of W per pool task, single-threaded Accelerate each, partial
vectors reduced by column chunks with the bias / gelu epilogue; 3 tasks per pool thread on the Mac, 2 on the iPhone),
`add_bias_layernorm2_f32_cpu` for residual + bias + next LN. `attention_decode_f32_cpu` is plain NEON, one head per
pool task (the old per-head `cblas_sgemv` + vDSP loop was single-threaded). Mac decode 10.9 → 7.9 ms (column-block
GEMVs streamed 128-byte row segments); iPhone a wash (6.2 vs 6.0 ms; the iPhone GEMVs stream ~80-100 GB/s either
way). `cpu_split=None` = the old unfused step.

## Generation (chunked prefill + decode, one model)

`M.build_generate(sd, device)`: inputs ids `i64[T]` (T new tokens, symbolic) and their start position `i64[1]` →
logits of the last new token `f32[1, 50257]`. One compiled model, so prefill and decode share its persistent caches:
`M.generate(model, prompt, n)` prefills the whole prompt in one call, then runs one call per new token. Kernels:
`kv_write_rows` writes the T new K/V rows in place; `attention_cache` (CPU: per-head GEMMs + causal softmax; GPU: the
flash scheme over 32-row cache tiles) lets query i attend to cache rows `0 .. pos + i`; `arange_offset` builds the
position ids. Greedy output matches HF `generate` token for token (CPU and GPU).

GPU kernels (**built**, 2026-11-01; `kernels="auto"`, default): T is symbolic, so the choice between the decode and
the prefill kernels is made **per call from the runtime T** by three dispatching ops (`transformer_gpu.cpp`, each
prepares the registered kernels it delegates to): `gemm_auto_f32_gpu` / `gemm_auto_bias_gelu_f32_gpu` (M = 1 →
`gemv4` with the fused bias + gelu; M > 1 → MPS, + `bias_gelu` in place) and `attention_gen_f32_gpu` (T = 1 → the
fused kv-write decode attention `attention_decode_kvw`; T > 1 → `kv_write_rows`, then the simdgroup-matrix flash
attention over qkv when the chunk starts at pos 0 (pos is a graph input, read at encode time), else the cache-tile
flash `attention_cache`). Mac GPU decode step at P = 512: 6.0 → 2.05 ms (= `build_decode`'s 2.07 ms); prefill
unchanged (MPS either way). `kernels="old"` keeps MPS + the scalar cache attention for every T. Tests:
`test_gpt2_chunked_generate_matches_hf`, `test_gpt2_generate_two_chunk_prefill` (the pos > 0 chunk path).

2026-11-02 (`kernels="auto"`, default; the previous auto is `kernels="auto1"`): `attention_gen2_f32_gpu` runs
**later prefill chunks (pos > 0) on the simdgroup-matrix flash kernel reading K/V from the cache**
(`lk_attention_cache_sg{2,4,8}`: the `lk_attention_qkv_sg` body shared through `attn_sg_body`, keys/values from the
cache rows with a position offset, causal mask `j <= pos + i`) and T = 1 on the decode attention above; the two
N = 768 projections use **split-K for T > 1** like `build()` (`gemm_auto_sk_f32_gpu`: M = 1 → gemv4 into plane 0,
else `gemm_splitk` with its MPS fallback; `build_generate(sk=(2, 4))`, `(0, 0)` = MPS). Mac GPU, medians:

| case | auto1 | auto | of which split-K |
|---|---|---|---|
| chunk T=128 at pos 384 | 10.1-10.4 | 6.6 | 6.85-7.25 without |
| chunk T=16 at pos 496 | 7.7 | 3.75 | 3.95 without |
| prefill T=128 at pos 0 | 5.7 | 5.07 | 6.1 without |
| decode at 512 | 2.05 | 2.05 | |

Tests: `test_gpt2_generate_multi_chunk_prefill_gpu` (chunks 40, 70, 1, 117, 1, 71 vs HF after each call),
`tests/kernels/test_transformer_fused.py` (`attention_cache_sg`, `attention_decode_split`, `gemm_auto_sk`). iPhone:
not validated yet (the phone was unavailable).

## Using it

```python
from models import gpt2 as M
sd = M.weights(M.torch_reference())
model = M.build(sd)                 # or device="gpu", last_only=True
logits = model(ids)                 # ids i64[T]
```

## Graph

Pre-LN blocks ×12: LN → fused QKV GEMM (HF `Conv1D` weights are already `[in, out]`) → causal attention reading
q/k/v from the fused projection → projection GEMM + residual → LN → GEMM + gelu_tanh → GEMM + residual; final LN;
the **tied LM head** is a transposed-B GEMM against `wte` itself (`gemm_nt_*`: Accelerate `CblasTrans` on the CPU,
an MPS transposed descriptor on the GPU), so no transposed 154 MB copy.
- CPU (`cpu="fused"`, default, 2026-11-01): `gemm_accel_f32_cpu` (`kernels/cpu/gemm.cpp`: Accelerate with the bias /
  gelu_tanh epilogue fused per block; gelu = x / (1 + exp(-2u)) on a NEON exp, rel. error ~1e-7) with a per-platform
  auto blocking (`autoSplit`): **iPhone: 4 row blocks, each a single-threaded Accelerate stream on a pool task**
  (Accelerate's own threading underuses the A19's SME at M = 128-512: T = 128 26.7 → 23.0 ms; 2 blocks 23.7, 3 blocks
  25.8); Mac: one multithreaded call (it spreads over both P-clusters' AMX; a single stream is 1.7x slower).
  `add_bias_layernorm2_f32_cpu` = residual + bias + next LayerNorm in one pass (CPU twin of the GPU kernel).
  `attention_qkv_causal_blk_f32_cpu` (`kokoro_misc.cpp`): work items (head, 128-query block) on the pool, keys trimmed
  to the block's causal extent (half the T x T work), NEON softmax with 1/sum applied to the 64-wide output rows
  (iPhone T = 512 87.5 → 79.0 ms, Mac 74.6 → 70.3 ms). `cpu="basic"` = the old unfused graph
  (`attention_qkv_causal_f32_cpu`, separate gelu / add / LN).
- GPU: MPS GEMMs, except the two N = 768 output projections at small T on the Mac (**split-K**, 2026-11-01:
  `gemm_splitk_f32_gpu` = steel 32x32 tiles, grid z = K slice, partials `f32[S, M, N]`; the following
  `add_bias_layernorm2_sk_f32_gpu` sums the S partials while it loads them, so the reduction costs no extra pass;
  `build(sk=(2, 4))`: attn proj K = 768 in 2, MLP proj K = 3072 in 4). At T = 128 MPS left 24 output tiles of 64x64
  on 30 cores (4-4.5 TF/s); split-K: the 24 projections 2.0 → 1.55 ms, e2e 5.45 → 4.95 ms. The kernel runs MPS into
  plane 0 (and the LN reads one plane) once the 64x64 tiles fill the GPU (>= 48 tiles: T = 512 split-K steel was
  slower than MPS) and always on the iPhone (6 GPU cores: T = 128 11.43 vs 11.89 ms with split-K); `attention_qkv_sg_causal_f32_gpu` (flash attention on simdgroup matrices, `kokoro_tok.metal`
  `lk_attention_qkv_sg{2,4,8}`, default 4 simdgroups = 32 queries per threadgroup: Q fragments in registers, K/V
  32-key tiles in threadgroup memory, online softmax on the MMA fragments; k's bias dropped (softmax-invariant), v's
  added to the output; `build(..., attn="flash")` selects the older scalar flash kernel), `bias_gelu_f32_gpu`, and
  `add_bias_layernorm2_f32_gpu`, which outputs both the new residual stream and its LayerNorm (the next sub-block's
  input). The embedding gather runs on the CPU (wte is kept once per device: a CPU copy for the gather, a GPU copy for
  the LM head).

## Correctness

vs HF at T ∈ {1, 7, 128, 511, 1024}: all-position logits within 2e-4 relative (of the max), > 99 % identical argmax;
`last_only` matches HF's last row. CPU and GPU.

## Performance

`bench/models/bench_m6_models.py gpt2` (lokinfer) and `bench/models/bench_other_baselines.py` (baselines).
Profiles: `tools/prof-lokinfer run gpt2 -c "prefill T=512"` and `tools/prof-lokinfer run gpt2_decode` (one decode step).

GPU tuning log (2026-11-01, medians in ms; best reference in brackets):

| change | Mac T=128 | Mac T=512 | Mac decode | iPhone T=128 | iPhone T=512 | iPhone decode |
|---|---|---|---|---|---|---|
| before | 6.1 | 19.9 | 3.55 | 11.8 | 43.3 | 6.9 |
| simdgroup-matrix flash attention (attention 6.4 → 1.36 ms at T=512 on the Mac) | 5.4 | 14.9 | | 11.46 | 39.2 | |
| fused kv-write + 1024-thread decode attention (1.29 → 0.26 ms per step) | | | 2.70 | | | 6.85 |
| gemv4 decode GEMVs (MPS 1.34 → 1.17 ms Mac) | | | 2.05 | | | 5.60 |
| references | [MLX 5.65] | [compile 18.7] | [compile 3.21] | [MLX 12.75] | [41.8] | [6.0] |
| split-K N = 768 projections on the Mac (dashboard, 2026-11-01) | 4.99 | 15.1 | 2.01 | 11.45 | 39.2 | 5.57 |
| 2026-11-02 dashboard (build()/build_decode unchanged except decode auto policy; shared Mac, bimodal 15.3 / 17.3 ms at T=512 across runs) | 5.0-5.5 | 15.3-17.6 (prof 15.3-15.7) | 2.00 | needs iPhone | needs iPhone | needs iPhone |

CPU tuning log (2026-11-01, medians in ms; iPhone 18 Pro Max thermal nominal, A/B rounds interleaved; best reference
ExecuTorch XNNPACK on the iPhone, torch.compile on the Mac):

| change | Mac T=128 | Mac T=512 | Mac decode | iPhone T=128 | iPhone T=512 | iPhone decode |
|---|---|---|---|---|---|---|
| before (unfused, multithreaded Accelerate) | 34.2 | 74.6 | 10.9 | 27.4 | 85.3 | 6.68 |
| fused epilogues + add/bias/LN pass | 28.5 | | | 26.7 | 80.1 | |
| iPhone: 4 row blocks x single-threaded Accelerate | | | | 23.0 | 77.4 | |
| causal blocked attention (qb 128) | | 70.3 | | | 75.0 | |
| fused decode step: K-split GEMVs + parallel NEON attention | | | 7.9 | | | 5.46 |
| dashboard | 26.3 | 69.5 | 7.92 | 23.16 | 75.06 | 5.46 |
| references | [compile 45.0] | [compile 105.4] | [eager 19.8] | [XNNPACK 26.8] | [XNNPACK 164.1] | [XNNPACK 8.69] |

Profile at T = 128 on the iPhone before (prof-lokinfer): GEMMs 20.5 ms (1.06 TF/s), gelu 2.9, attention 1.6, LM head
1.6 (154 MB at ~100 GB/s, a floor), LN + add 0.7. A single SME stream reaches ~1.5-1.9 TF/s on these shapes; the
four row-block streams do not add SME throughput (one unit per cluster) but overlap the epilogues and the per-call
overheads.

Re-checked 2026-11-02 (Mac GPU; whether the old "slower than MPS" verdicts only lacked the unroll fix): the steel
GEMM's loops are already unrolled (MLX `STEEL_PRAGMA_UNROLL`) and it still loses to MPS on the full graph
(`gemm="steel"` T=128 6.2-6.7 vs 4.9-5.8 ms, T=512 18.8-19.0 vs 17.2-17.5); split-K forced on at T=512 (64x64
17.8, 32x32 19.0, sk (2, 2) 18.4 vs MPS 17.2-17.5) loses, so the 48-tile threshold stays; the LM head as
`lk_gemv_rows` ties MPS (decode 2.03 vs 2.03 ms: both stream the 154 MB at ~380 GB/s), MPS kept. The scalar decode
kernels `lk_gemv` / `lk_attention_decode` are superseded by `gemv4` / `attention_decode_kvw` (both unrolled).

Measured and dropped: register prefetch of the next K/V tile and K/V straight from device memory (no threadgroup
memory) in the attention (both slower); the loops must be fully unrolled (`clang loop unroll(full)`), else the
fragment arrays spill (3x slower); 8-deep key unrolling in the decode attention (0.288 vs 0.258 ms).
CPU, measured and dropped (iPhone, 2026-11-01): multithreaded Accelerate + a pool epilogue (26.7 vs 23.0 ms);
column blocks (2: 24.3, 6: 25.4, 12: 24.6, 24: 26.5 ms); one single-threaded stream (28.1); a pipelined GEMM (one
thread streams column blocks, the others apply the finished blocks' epilogue: 30.0 / 26.5 ms); a helper thread at
QOS_CLASS_UTILITY to reach the E-cluster's SME (no gain: the GEMM split did not run faster); prefetching the next
GEMM's weights from the other cores while one thread runs the GEMM (26.8 vs 23.0 ms); vvtanhf instead of the NEON
exp gelu (same speed). Decode: column-block GEMVs vs K split on the iPhone (6.2 vs 6.0 ms, kept K split for the Mac).
