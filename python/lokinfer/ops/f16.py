"""f16 CPU ops (Qwen3-ASR phase 1; kernels/cpu/f16_basic.cpp). One function per kernel.

The op contract (same signatures and semantics as `ops.f16_gpu`) is docs/impl-notes/qwen3-asr-f16-kernels.md:
f16 storage, f32 math and accumulation, one rounding at the output. Used as `lk.ops.f16.<name>`; the kernel of
`<name>` is `<name>_f16_cpu`."""
from ..dims import as_dim
from ..dtypes import f16, f32, i64
from ..types import TensorType
from ._def import check, kernel_op
from .kokoro import _float, _int

_MAX_HD = 256  # attention head dim limit of the CPU kernels (per-row f32 accumulator on the stack)


def _f16(*xs, rank=None):
    for x in xs:
        check(isinstance(x.type, TensorType) and x.dtype is f16, f"expected f16 tensor operands, got {x.type}")
        if rank is not None:
            check(x.rank == rank, f"expected rank-{rank} operands, got {x.type}")


def _shape(x, dims, what):
    check(x.rank == len(dims) and all(as_dim(a) == as_dim(b) for a, b in zip(x.shape, dims)),
          f"{what} must have shape {list(dims)}, got {x.type}")


def _pos(pos):
    check(isinstance(pos.type, TensorType) and pos.dtype is i64 and pos.rank == 1 and pos.shape[0] == 1,
          "pos must be i64[1]")


def _const(d, what):
    d = as_dim(d)
    check(d.is_const, f"{what} must be static")
    return d.value


# --- encoder front end ----------------------------------------------------------------------------------------


@kernel_op("audio_cnn_f16_cpu")
def audio_cnn(features, w1, b1, w2, b2, w3, b3, w_out):
    """Conv front end, one 1 s chunk at a time: features f32[n, 128, 100] -> f16[n, 13, 1024] (three conv 3×3 s2 p1 +
    bias + GELU(erf), NHWC with H = time; flatten (freq, channel); @ w_out f16[7680, 1024]). Conv weights
    f16[Co, kt, kf, Ci]."""
    check(isinstance(features.type, TensorType) and features.dtype is f32 and features.rank == 3,
          "features must be f32[n, 128, 100]")
    check(features.shape[1] == 128 and features.shape[2] == 100, "features must be f32[n, 128, 100]")
    _f16(w1, b1, w2, b2, w3, b3, w_out)
    _shape(w1, [480, 3, 3, 1], "w1")
    _shape(w2, [480, 3, 3, 480], "w2")
    _shape(w3, [480, 3, 3, 480], "w3")
    for b in (b1, b2, b3):
        _shape(b, [480], "conv bias")
    _shape(w_out, [7680, 1024], "w_out")
    return TensorType(f16, [features.shape[0], 13, 1024])


@kernel_op("add_pos_prefix_f16_cpu")
def _add_pos_prefix(x, pos, *, rows):
    _f16(x, rank=3)
    _f16(pos, rank=2)
    check(pos.shape[0] == x.shape[1] and pos.shape[1] == x.shape[2], "pos must be f16[13, C] for x f16[n, 13, C]")
    r, n, p = as_dim(rows), as_dim(x.shape[0]), as_dim(x.shape[1])
    check(not (r.is_const and n.is_const and p.is_const) or r.value <= n.value * p.value, "rows must be <= 13 * n")
    return TensorType(f16, [rows, x.shape[2]])


def add_pos_prefix(x, pos, rows):
    """y = (x + pos[None]).reshape(13·n, C)[:rows]: x f16[n, 13, C], pos f16[13, C] (restarting every chunk);
    `rows` (N_a, usually from `dim_read`) <= 13·n, checked at run time."""
    return _add_pos_prefix(x, pos, rows=as_dim(rows))


# --- GEMMs ------------------------------------------------------------------------------------------------------


def _gemm_rule(x, w, *bias):
    _f16(x, w, rank=2)
    check(x.shape[1] == w.shape[0], f"x [M, K] and w [K, N]: inner dims differ: {x.shape}, {w.shape}")
    if bias:
        _f16(bias[0], rank=1)
        check(bias[0].shape[0] == w.shape[1], "bias must be f16[N]")
    return TensorType(f16, [x.shape[0], w.shape[1]])


@kernel_op("gemm_f16_cpu")
def gemm(x, w):
    """x f16[M, K] @ w f16[K, N] -> f16[M, N] (f32 accumulation)."""
    return _gemm_rule(x, w)


@kernel_op("gemm_bias_f16_cpu")
def gemm_bias(x, w, b):
    """x f16[M, K] @ w f16[K, N] + b f16[N] -> f16[M, N]."""
    return _gemm_rule(x, w, b)


@kernel_op("lm_head_last_f16_cpu")
def lm_head_last(x, table):
    """Tied LM head on the last row: x[T-1] @ table f16[V, D]ᵀ -> f32[1, V]."""
    _f16(x, table, rank=2)
    check(x.shape[1] == table.shape[1], "x [T, D] and table [V, D] must share D")
    return TensorType(f32, [1, table.shape[0]])


# --- elementwise / normalization ----------------------------------------------------------------------------------


def _same2(x, y):
    _f16(x, y, rank=2)
    check(x.shape == y.shape, f"operands must have the same shape: {x.shape}, {y.shape}")
    return TensorType(f16, x.shape)


@kernel_op("gelu_erf_f16_cpu")
def gelu_erf(x):
    """0.5·x·(1 + erf(x/√2)) over f16[R, C]."""
    _f16(x, rank=2)
    return TensorType(f16, x.shape)


@kernel_op("add_f16_cpu")
def add(x, y):
    """x + y over f16[R, C]."""
    return _same2(x, y)


@kernel_op("add_clamp_f16_cpu")
def _add_clamp(x, y, *, lim):
    return _same2(x, y)


def add_clamp(x, y, lim=64504.0):
    """clamp(x + y, -lim, lim) over f16[R, C] (the encoder's f16 clamp)."""
    return _add_clamp(x, y, lim=_float(lim, "lim"))


@kernel_op("silu_mul_f16_cpu")
def silu_mul(g, u):
    """g·sigmoid(g)·u over f16[R, C] (SwiGLU)."""
    return _same2(g, u)


@kernel_op("layernorm_f16_cpu")
def _layernorm(x, g, b, *, eps):
    _f16(x, rank=2)
    _f16(g, b, rank=1)
    check(g.shape[0] == x.shape[1] and b.shape[0] == x.shape[1], "g / b must be f16[C]")
    return TensorType(f16, x.shape)


def layernorm(x, g, b, eps=1e-5):
    """(x - mean) / sqrt(var + eps) · g + b over the rows of f16[R, C] (f32 statistics, one rounding)."""
    return _layernorm(x, g, b, eps=_float(eps, "eps"))


@kernel_op("rmsnorm_f16_cpu")
def _rmsnorm(x, g, *, eps):
    _f16(x, rank=2)
    _f16(g, rank=1)
    check(g.shape[0] == x.shape[1], "g must be f16[C]")
    return TensorType(f16, x.shape)


def rmsnorm(x, g, eps=1e-6):
    """HF Qwen3RMSNorm over the rows of f16[R, C]: h = f16(x·rsqrt(mean(x²) + eps)), y = f16(g·h)."""
    return _rmsnorm(x, g, eps=_float(eps, "eps"))


@kernel_op("rmsnorm_heads_f16_cpu")
def _rmsnorm_heads(x, g, *, eps):
    _f16(x, rank=2)
    _f16(g, rank=1)
    hd, c = _const(g.shape[0], "head width"), as_dim(x.shape[1])
    check(not c.is_const or c.value % hd == 0, "C must be a multiple of the head width (len(g))")
    return TensorType(f16, x.shape)


def rmsnorm_heads(x, g, eps=1e-6):
    """rmsnorm over each len(g)-wide head (QK-norm; 128) of f16[R, H·128]."""
    return _rmsnorm_heads(x, g, eps=_float(eps, "eps"))


@kernel_op("rope_f16_cpu")
def _rope(x, pos, *, theta):
    _f16(x, rank=2)
    _pos(pos)
    c = as_dim(x.shape[1])
    check(not c.is_const or c.value % 128 == 0, "x must be f16[T, H·128]")
    return TensorType(f16, x.shape)


def rope(x, pos, theta=1e6):
    """rotate_half RoPE per 128-wide head of f16[T, H·128] at positions pos + i (pos i64[1])."""
    return _rope(x, pos, theta=_float(theta, "theta"))


# --- gathers ------------------------------------------------------------------------------------------------------


@kernel_op("embedding_f16_cpu")
def embedding(ids, table):
    """table[ids]: ids i64[T], table f16[V, D] -> f16[T, D]; an id outside [0, V) is a run-time error."""
    check(isinstance(ids.type, TensorType) and ids.dtype is i64 and ids.rank == 1, "ids must be i64[T]")
    _f16(table, rank=2)
    return TensorType(f16, [ids.shape[0], table.shape[1]])


@kernel_op("scatter_audio_f16_cpu")
def _scatter_audio(x, ids, audio, *, pad_id):
    _f16(x, audio, rank=2)
    check(isinstance(ids.type, TensorType) and ids.dtype is i64 and ids.rank == 1, "ids must be i64[T]")
    check(ids.shape[0] == x.shape[0], "ids must have one entry per row of x")
    check(audio.shape[1] == x.shape[1], "audio must be f16[A, D]")
    return TensorType(f16, x.shape)


def scatter_audio(x, ids, audio, pad_id=151676):
    """Copy of x f16[T, D] where the j-th row with ids == pad_id becomes audio[j] (more pads than A rows: error)."""
    return _scatter_audio(x, ids, audio, pad_id=_int(pad_id, "pad_id"))


# --- attention / KV cache -------------------------------------------------------------------------------------------


def _head_dim(d, heads):
    d = _const(d, "the model width")
    check(heads > 0 and d % heads == 0, "width must be divisible by heads")
    hd = d // heads
    check(hd % 8 == 0 and hd <= _MAX_HD, f"head dim must be a multiple of 8 and <= {_MAX_HD}")
    return hd


@kernel_op("attention_window_f16_cpu")
def _attention_window(q, k, v, *, heads, window):
    _f16(q, k, v, rank=2)
    check(q.shape == k.shape and q.shape == v.shape, "q, k, v must be f16[S, D]")
    _head_dim(q.shape[1], heads)
    check(window > 0, "window must be positive")
    return TensorType(f16, q.shape)


def attention_window(q, k, v, heads=16, window=104):
    """Encoder self-attention, non-causal within blocks of `window` rows (scale 1/√hd): f16[S, D] -> f16[S, D]."""
    return _attention_window(q, k, v, heads=_int(heads, "heads"), window=_int(window, "window"))


@kernel_op("kv_write_f16_cpu", inout=[0, 1])
def kv_write(kc, vc, k, v, pos):
    """kc[pos:pos+T] = k, vc[pos:pos+T] = v (caches f16[cap, D], written in place); pos + T > cap is an error."""
    _f16(kc, vc, k, v, rank=2)
    check(kc.shape == vc.shape, "caches must both be f16[cap, D]")
    check(k.shape == v.shape and k.shape[1] == kc.shape[1], "k / v must be f16[T, D] with the caches' D")
    _pos(pos)
    return ()


@kernel_op("attention_gqa_f16_cpu")
def _attention_gqa(q, kc, vc, pos, *, kv_heads, q_heads):
    _f16(q, kc, vc, rank=2)
    check(kc.shape == vc.shape, "caches must both be f16[cap, Dk]")
    _pos(pos)
    check(kv_heads > 0 and q_heads % kv_heads == 0, "q_heads must be a multiple of kv_heads")
    hd = _head_dim(q.shape[1], q_heads)
    check(as_dim(kc.shape[1]) == as_dim(kv_heads * hd), "caches must be f16[cap, kv_heads·hd]")
    return TensorType(f16, q.shape)


def attention_gqa(q, kc, vc, pos, q_heads=16, kv_heads=8):
    """Causal GQA over the cache: query row i of q f16[T, q_heads·hd] (position pos + i) attends to cache rows
    0 .. pos + i of its kv-head (the caches must already hold them); scale 1/√hd -> f16[T, q_heads·hd]."""
    return _attention_gqa(q, kc, vc, pos, kv_heads=_int(kv_heads, "kv_heads"), q_heads=_int(q_heads, "q_heads"))


# --- optimized ops (Q5: docs/impl-notes/qwen3-asr-f16-opt-kernels.md; kernels/cpu/f16_fused.cpp) ---------------------
# Same math as the composition of the basic ops above (f32 accumulation and epilogue, one rounding at the output);
# no packed weight layouts: every op takes the basic layouts ([K, N] GEMM weights, the basic audio_cnn weights).


@kernel_op("gemm_opt_f16_cpu")
def gemm_opt(x, w):
    """x f16[M, K] @ w f16[K, N] -> f16[M, N] (fast for M = 1 and large M)."""
    return _gemm_rule(x, w)


def _rms_gemm_rule(x, g, w, swiglu):
    _f16(g, rank=1)
    _gemm_rule(x, w)
    check(g.shape[0] == x.shape[1], "g must be f16[K]")
    n = w.shape[1]
    if swiglu:
        n2 = _const(n, "2N")
        check(n2 % 2 == 0, "w must be f16[K, 2N] ([gate | up])")
        n = n2 // 2
    return TensorType(f16, [x.shape[0], n]), TensorType(f16, list(x.shape))


@kernel_op("rmsnorm_gemm_opt_f16_cpu")
def _rmsnorm_gemm_opt(x, g, w, *, eps):
    return _rms_gemm_rule(x, g, w, False)


def rmsnorm_gemm_opt(x, g, w, eps=1e-6):
    """rmsnorm(x, g) @ w -> (y f16[M, N], ws f16[M, K]). ws is a workspace (the normalized rows on the CPU / for M > 1;
    unspecified otherwise): use y only."""
    return _rmsnorm_gemm_opt(x, g, w, eps=_float(eps, "eps"))


@kernel_op("rmsnorm_gemm_swiglu_opt_f16_cpu")
def _rmsnorm_gemm_swiglu_opt(x, g, w, *, eps):
    return _rms_gemm_rule(x, g, w, True)


def rmsnorm_gemm_swiglu_opt(x, g, w, eps=1e-6):
    """gemm_swiglu_opt(rmsnorm(x, g), w) -> (y f16[M, N], ws f16[M, K] workspace)."""
    return _rmsnorm_gemm_swiglu_opt(x, g, w, eps=_float(eps, "eps"))


@kernel_op("gemm_bias_opt_f16_cpu")
def gemm_bias_opt(x, w, b):
    """x @ w + b."""
    return _gemm_rule(x, w, b)


@kernel_op("gemm_bias_gelu_opt_f16_cpu")
def gemm_bias_gelu_opt(x, w, b):
    """gelu_erf(x @ w + b) (encoder fc1, projector 1)."""
    return _gemm_rule(x, w, b)


def _res(t, res):
    _f16(res, rank=2)
    check(res.shape == t.shape, f"res must be f16{list(t.shape)}, got {res.type}")
    return t


@kernel_op("gemm_add_opt_f16_cpu")
def gemm_add_opt(x, w, res):
    """res + x @ w (LM o_proj / down_proj + residual)."""
    return _res(_gemm_rule(x, w), res)


@kernel_op("gemm_bias_add_opt_f16_cpu")
def gemm_bias_add_opt(x, w, b, res):
    """res + (x @ w + b) (encoder out_proj + residual)."""
    return _res(_gemm_rule(x, w, b), res)


@kernel_op("gemm_bias_add_clamp_opt_f16_cpu")
def _gemm_bias_add_clamp_opt(x, w, b, res, *, lim):
    return _res(_gemm_rule(x, w, b), res)


def gemm_bias_add_clamp_opt(x, w, b, res, lim=64504.0):
    """clamp(res + (x @ w + b), -lim, lim) (encoder fc2 + residual + the f16 clamp)."""
    return _gemm_bias_add_clamp_opt(x, w, b, res, lim=_float(lim, "lim"))


@kernel_op("gemm_swiglu_opt_f16_cpu")
def gemm_swiglu_opt(x, w):
    """y = x f16[M, K] @ w f16[K, 2N] (w = [gate | up]); silu(y[:, :N])·y[:, N:] -> f16[M, N]."""
    _f16(x, w, rank=2)
    check(x.shape[1] == w.shape[0], f"x [M, K] and w [K, 2N]: inner dims differ: {x.shape}, {w.shape}")
    n2 = _const(w.shape[1], "w columns")
    check(n2 % 2 == 0, "w must be f16[K, 2N] ([gate | up])")
    return TensorType(f16, [x.shape[0], n2 // 2])


@kernel_op("qknorm_rope_kv_opt_f16_cpu", inout=[4, 5])
def _qknorm_rope_kv_opt(qkv, qg, kg, pos, kc, vc, *, eps, theta):
    _f16(qkv, kc, vc, rank=2)
    _f16(qg, kg, rank=1)
    _pos(pos)
    hd = _const(qg.shape[0], "head width")
    check(as_dim(kg.shape[0]) == as_dim(hd) and hd % 2 == 0 and hd <= _MAX_HD, "qg / kg must be f16[hd], hd even")
    check(kc.shape == vc.shape, "caches must both be f16[P, Dk]")
    dk, w = _const(kc.shape[1], "the cache width"), _const(qkv.shape[1], "the qkv width")
    dq = w - 2 * dk
    check(dq > 0 and dq % hd == 0 and dk % hd == 0, "qkv must be f16[T, Dq + 2·Dk] with whole heads")
    return TensorType(f16, [qkv.shape[0], dq])


def qknorm_rope_kv_opt(qkv, qg, kg, pos, kc, vc, eps=1e-6, theta=1e6):
    """QK-norm + RoPE + KV write: qkv f16[T, Dq + 2·Dk] (q | k | v). Returns q = rope(rmsnorm_heads(q, qg)) f16[T, Dq];
    writes k = rope(rmsnorm_heads(k, kg)) to kc[pos:pos+T] and v to vc[pos:pos+T] (caches f16[P, Dk], in place; the
    basic composition's roundings kept); pos + T > P is a run-time error."""
    return _qknorm_rope_kv_opt(qkv, qg, kg, pos, kc, vc, eps=_float(eps, "eps"), theta=_float(theta, "theta"))


@kernel_op("attention_gqa_opt_f16_cpu")
def _attention_gqa_opt(q, kc, vc, pos, *, kv_heads, q_heads):
    return _gqa_rule(q, kc, vc, pos, kv_heads, q_heads)


def _gqa_rule(q, kc, vc, pos, kv_heads, q_heads):
    _f16(q, kc, vc, rank=2)
    check(kc.shape == vc.shape, "caches must both be f16[cap, Dk]")
    _pos(pos)
    check(kv_heads > 0 and q_heads % kv_heads == 0, "q_heads must be a multiple of kv_heads")
    hd = _head_dim(q.shape[1], q_heads)
    check(as_dim(kc.shape[1]) == as_dim(kv_heads * hd), "caches must be f16[cap, kv_heads·hd]")
    check(q_heads // kv_heads <= 128, "at most 128 q heads per kv head")
    return TensorType(f16, q.shape)


def attention_gqa_opt(q, kc, vc, pos, q_heads=16, kv_heads=8):
    """attention_gqa (causal GQA over the cache), flash-style; key-split for decode."""
    return _attention_gqa_opt(q, kc, vc, pos, kv_heads=_int(kv_heads, "kv_heads"), q_heads=_int(q_heads, "q_heads"))


@kernel_op("attention_window_qkv_opt_f16_cpu")
def _attention_window_qkv_opt(qkv, *, heads, window):
    _f16(qkv, rank=2)
    w = _const(qkv.shape[1], "the qkv width")
    check(w % 3 == 0, "qkv must be f16[S, 3·D]")
    _head_dim(w // 3, heads)
    check(window > 0, "window must be positive")
    return TensorType(f16, [qkv.shape[0], w // 3])


def attention_window_qkv_opt(qkv, heads=16, window=104):
    """attention_window on q = qkv[:, :D], k = qkv[:, D:2D], v = qkv[:, 2D:] -> f16[S, D]."""
    return _attention_window_qkv_opt(qkv, heads=_int(heads, "heads"), window=_int(window, "window"))


@kernel_op("audio_cnn_opt_f16_cpu")
def audio_cnn_opt(features, w1, b1, w2, b2, w3, b3, w_out):
    """audio_cnn (same signature and weights), conv layers as im2col + the optimized GEMM with the bias + GELU
    epilogue."""
    return audio_cnn.__wrapped__(features, w1, b1, w2, b2, w3, b3, w_out)


@kernel_op("lm_head_last_opt_f16_cpu")
def lm_head_last_opt(x, table):
    """x[T-1] @ table f16[V, D]ᵀ -> f32[1, V] (bandwidth-bound GEMV over the table)."""
    return lm_head_last.__wrapped__(x, table)
