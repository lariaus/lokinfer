# Qwen3-ASR f16 basic kernels — the op contract (Q2)

The f16 basic kernel set for Qwen3-ASR phase 1 (`qwen3-asr-port-plan.md` Q2). Every op exists twice with **the
same signature and semantics**: `<name>_f16_cpu` (`kernels/cpu/f16_basic.cpp`, ops in `python/lokinfer/ops/f16.py`)
and `<name>_f16_gpu` (`kernels/gpu/f16_basic.metal` + `kernels/gpu/f16_gpu.cpp`, ops in
`python/lokinfer/ops/f16_gpu.py`), so a graph switches device by suffix. Tests: `tests/kernels/test_f16_basic_cpu.py`,
`tests/kernels/test_f16_basic_gpu.py`.

Basic = simple, readable, obviously correct (the oracle for the fused kernels later); not tuned. Common rules:
- Storage f16 (IEEE half) unless stated; **all arithmetic and accumulation in f32**, one rounding to f16 at the
  output (more accurate than the torch f16 reference, which rounds between ops — fine: the tests bound the error
  against a float64 composition).
- Row-major, contiguous. Weight layouts are lokinfer's (`fold_weights` in Q3 converts HF's); GEMM weights are
  `[K, N]` like the existing `gemm_f32_*` ops.
- Shape checks in the type rule (`check(...)`), data-dependent errors through `ctx.error` (static strings).
- Attention never materializes a `[T, P]` score matrix: per (head, query row), a streaming (online) f32 softmax.
- Tests: each op through a one-op graph vs the float64 torch composition (inputs drawn as f16, upcast), real model
  shapes (below) + odd sizes + length 1 + the chunk / window boundaries; tolerance stated per op (f16 output rounding
  2⁻¹¹ relative, plus ≈ 2⁻¹¹·√K·(typical |term|) for reductions). They also run on the iPhone with
  `pytest --device=iphone`.

| op (`_f16_cpu` / `_f16_gpu`) | signature → output | semantics |
|---|---|---|
| `audio_cnn` | `(features f32[n, 128, 100], w1 f16[480, 3, 3, 1], b1 f16[480], w2 f16[480, 3, 3, 480], b2 f16[480], w3 f16[480, 3, 3, 480], b3 f16[480], w_out f16[7680, 1024]) → f16[n, 13, 1024]` | The encoder's conv front end, **one 1 s chunk at a time** (a small per-chunk scratch; never all chunks' intermediates: n reaches 1200). Per chunk c: `x = features[c]ᵀ` as NHWC `[H = 100 time, W = 128 freq, C = 1]` (cast to f16); three times `conv2d 3×3, stride 2, pad 1, + bias, GELU(erf)` → `[50, 64, 480]` → `[25, 32, 480]` → `[13, 16, 480]`; flatten each time step as `(freq, channel)` → `[13, 7680]`; `@ w_out` (no bias) → `[13, 1024]`. Conv weights `[Co, kt, kf, Ci]` (kt along time = H, kf along freq = W). Intermediates rounded to f16 between layers (like the reference). |
| `add_pos_prefix` | `(x f16[n, 13, 1024], pos f16[13, 1024], rows: Dim) → f16[rows, 1024]` | `y = (x + pos[None]).reshape(13·n, 1024)[:rows]` — positions restart every chunk; `rows` = N_a (a symbol from `dim_read`, `rows ≤ 13·n`, else `ctx.error`). |
| `gemm` | `(x f16[M, K], w f16[K, N]) → f16[M, N]` | `x @ w` |
| `gemm_bias` | `(x f16[M, K], w f16[K, N], b f16[N]) → f16[M, N]` | `x @ w + b` |
| `lm_head_last` | `(x f16[T, D], table f16[V, D]) → f32[1, V]` | `x[T-1] @ tableᵀ` in f32 (the tied embedding as the LM head; output **f32**) |
| `gelu_erf` | `(x f16[R, C]) → f16[R, C]` | `0.5·x·(1 + erf(x/√2))` |
| `layernorm` | `(x f16[R, C], g f16[C], b f16[C], eps) → f16[R, C]` | `(x − mean) / sqrt(var + eps) · g + b`, all f32, one rounding |
| `rmsnorm` | `(x f16[R, C], g f16[C], eps) → f16[R, C]` | HF Qwen3RMSNorm: `h = f16(x · rsqrt(mean(x²) + eps))` (f32 math, round), then `y = g · h` (the product of two f16 values, computed in f32, rounded) |
| `rmsnorm_heads` | `(x f16[R, H·128], g f16[128], eps) → f16[R, H·128]` | `rmsnorm` over each 128-wide head (QK-norm) |
| `rope` | `(x f16[T, H·128], pos i64[1], theta) → f16[T, H·128]` | rotate_half RoPE per 128-wide head at positions `pos + i`: `inv_freq[j] = theta^(−2j/128)` (j < 64), `angle = (pos + i)·inv_freq[j]` in f32 (f64 for the angle product is fine), `y[j] = x[j]·cos − x[j+64]·sin`, `y[j+64] = x[j+64]·cos + x[j]·sin` |
| `add` | `(x, y f16[R, C]) → f16[R, C]` | `x + y` |
| `add_clamp` | `(x, y f16[R, C], lim) → f16[R, C]` | `clamp(x + y, −lim, lim)` (the encoder's f16 clamp, lim = 64504) |
| `silu_mul` | `(g, u f16[R, C]) → f16[R, C]` | `g·sigmoid(g)·u` |
| `embedding` | `(ids i64[T], table f16[V, D]) → f16[T, D]` | gather rows; an id outside `[0, V)` → `ctx.error` |
| `scatter_audio` | `(x f16[T, D], ids i64[T], audio f16[A, D], pad_id) → f16[T, D]` | copy of x where the j-th row with `ids == pad_id` (in order) becomes `audio[j]`; more pad ids than A rows → `ctx.error`; `pad_id` = 151676 |
| `attention_window` | `(q, k, v f16[S, 1024], heads, window) → f16[S, 1024]` | encoder self-attention: 16 heads × 64, scale 1/8, **non-causal within windows**: row i attends to rows of its window `[⌊i/window⌋·window, min(S, (⌊i/window⌋ + 1)·window))`, window = 104 |
| `kv_write` | `(kc f16[cap, 1024] inout, vc f16[cap, 1024] inout, k f16[T, 1024], v f16[T, 1024], pos i64[1])` | `kc[pos:pos+T] = k`, `vc[pos:pos+T] = v`; `pos + T > cap` → `ctx.error` (the `inout` pattern of `kv_write_rows_f32_*`) |
| `attention_gqa` | `(q f16[T, 2048], kc, vc f16[cap, 1024], pos i64[1], q_heads, kv_heads) → f16[T, 2048]` | causal GQA over the cache: query row i (absolute position `pos + i`) of q-head h attends to cache rows `0 .. pos + i` of kv-head `h // (q_heads / kv_heads)`; head dim 128, scale 1/√128; 16 q heads, 8 kv heads. Prefill (T ≤ 2048) and decode (T = 1) alike; any `cap` (25,600 on the Mac). |

Real shapes for tests: encoder S = N_a up to 15,600 (windows of 104; last window partial), LM T = 1, 44, 401, 2048,
cache contexts up to ~20k; GEMMs `[M, 1024]·[1024, 3072|1024|4096]`, `[M, 4096]·[4096, 1024]`, projector
`[1024, 1024]`, `[1024, 2048]`, LM `[M, 2048]·[2048, 2048|1024|6144]`, `[M, 6144]·[6144, 2048]`, conv_out
`[13, 7680]·[7680, 1024]`; `lm_head_last` V = 151936, D = 2048.
