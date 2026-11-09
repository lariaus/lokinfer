"""f16 GPU ops (Qwen3-ASR phase 1; kernels/gpu/f16_basic.metal + f16_gpu.cpp). One function per kernel.

The op contract (signatures, semantics; the same on the CPU, `ops/f16.py`) is docs/impl-notes/qwen3-asr-f16-kernels.md.
Storage f16, float math inside the kernels, one rounding at the output."""
from ..dims import as_dim
from ..dtypes import f16, f32, i64
from ..types import TensorType
from ._def import check, kernel_op


def _f16(*xs, rank=None):
    for x in xs:
        check(isinstance(x.type, TensorType) and x.dtype is f16, "expected f16 tensor operands")
        if rank is not None:
            check(x.rank == rank, f"expected rank-{rank} operands, got {x.type}")


def _pos(pos):
    check(isinstance(pos.type, TensorType) and pos.dtype is i64 and pos.rank == 1 and pos.shape[0] == 1,
          "pos must be i64[1]")


def _same(a, b, what):
    check(len(a.shape) == len(b.shape) and all(x == y for x, y in zip(a.shape, b.shape)), f"{what} shape mismatch")


def _static(d, what):
    d = as_dim(d)
    check(d.is_const, f"{what} must be static")
    return d.value


def _int(v, name):
    check(isinstance(v, int) and not isinstance(v, bool), f"{name} must be an int")
    return int(v)


def _float(v, name):
    check(isinstance(v, (int, float)) and not isinstance(v, bool), f"{name} must be a number")
    return float(v)


def _conv_out(n):
    return (n - 1) // 2 + 1  # 3x3, stride 2, pad 1


# --- audio encoder front end --------------------------------------------------------------------------------------


@kernel_op("audio_cnn_f16_gpu")
def audio_cnn_f16_gpu(features, w1, b1, w2, b2, w3, b3, w_out):
    """The conv front end, one chunk at a time: features f32[n, F, Tm] (real: [n, 128, 100]); per chunk x =
    features[c]^T as NHWC [Tm, F, 1] (cast to f16), three conv2d 3x3 stride 2 pad 1 + bias + GELU(erf) (weights
    f16[Co, 3, 3, Ci], H = time, W = freq; f16 between layers) -> [H3, W3, Co], flattened per time step as
    (freq, channel) and @ w_out f16[W3 * Co, D] -> f16[n, H3, D] (real: [n, 13, 1024])."""
    check(isinstance(features.type, TensorType) and features.dtype is f32 and features.rank == 3,
          "features must be f32[n, F, T]")
    _f16(w1, w2, w3, rank=4)
    _f16(b1, b2, b3, rank=1)
    _f16(w_out, rank=2)
    f, tm = _static(features.shape[1], "features F"), _static(features.shape[2], "features T")
    co = _static(w1.shape[0], "Co")
    check([as_dim(d) for d in w1.shape] == [as_dim(x) for x in (co, 3, 3, 1)], "w1 must be f16[Co, 3, 3, 1]")
    for w, b in ((w2, b2), (w3, b3)):
        check([as_dim(d) for d in w.shape] == [as_dim(x) for x in (co, 3, 3, co)], "w2 / w3 must be f16[Co, 3, 3, Co]")
    for b in (b1, b2, b3):
        check(as_dim(b.shape[0]) == as_dim(co), "conv biases must be f16[Co]")
    h3, w3_ = _conv_out(_conv_out(_conv_out(tm))), _conv_out(_conv_out(_conv_out(f)))
    check(as_dim(w_out.shape[0]) == as_dim(w3_ * co), f"w_out must be f16[{w3_ * co}, D]")
    return TensorType(f16, [features.shape[0], h3, w_out.shape[1]])


@kernel_op("add_pos_prefix_f16_gpu")
def _add_pos_prefix(x, pos, *, rows):
    _f16(x, rank=3)
    _f16(pos, rank=2)
    check(as_dim(pos.shape[0]) == as_dim(x.shape[1]) and as_dim(pos.shape[1]) == as_dim(x.shape[2]),
          "pos must be f16[P, D] for x f16[n, P, D]")
    r, n = as_dim(rows), as_dim(x.shape[0])
    if r.is_const and n.is_const:
        check(0 <= r.value <= n.value * _static(x.shape[1], "P"), "rows must be <= n * P")
    return TensorType(f16, [rows, x.shape[2]])


def add_pos_prefix_f16_gpu(x, pos, rows):
    """y = (x + pos[None]).reshape(n * P, D)[:rows] for x f16[n, P, D], pos f16[P, D] (positions restart every chunk);
    rows: a Dim (e.g. N_a from dim_read) or int, rows <= n * P (checked at run time)."""
    return _add_pos_prefix(x, pos, rows=as_dim(rows))


# --- GEMM ---------------------------------------------------------------------------------------------------------


def _gemm_rule(x, w, *bias):
    _f16(x, w, rank=2)
    check(as_dim(x.shape[1]) == as_dim(w.shape[0]), f"x [M, K] and w [K, N]: inner dims differ: {x.shape}, {w.shape}")
    if bias:
        _f16(bias[0], rank=1)
        check(as_dim(bias[0].shape[0]) == as_dim(w.shape[1]), "bias must be f16[N]")
    return TensorType(f16, [x.shape[0], w.shape[1]])


@kernel_op("gemm_f16_gpu")
def gemm_f16_gpu(x, w):
    """x f16[M, K] @ w f16[K, N] -> f16[M, N] (float accumulation)."""
    return _gemm_rule(x, w)


@kernel_op("gemm_bias_f16_gpu")
def gemm_bias_f16_gpu(x, w, b):
    """x f16[M, K] @ w f16[K, N] + b f16[N] -> f16[M, N]."""
    return _gemm_rule(x, w, b)


@kernel_op("lm_head_last_f16_gpu")
def lm_head_last_f16_gpu(x, table):
    """x[T-1] @ table^T in f32 for x f16[T, D], table f16[V, D] (the tied embedding) -> f32[1, V]."""
    _f16(x, table, rank=2)
    check(as_dim(x.shape[1]) == as_dim(table.shape[1]), "x f16[T, D] and table f16[V, D] must share D")
    return TensorType(f32, [1, table.shape[0]])


# --- elementwise / norms ------------------------------------------------------------------------------------------


@kernel_op("gelu_erf_f16_gpu")
def gelu_erf_f16_gpu(x):
    """0.5 x (1 + erf(x / sqrt 2)) over f16[R, C]."""
    _f16(x, rank=2)
    return TensorType(f16, x.shape)


@kernel_op("layernorm_f16_gpu")
def _layernorm(x, g, b, *, eps):
    _f16(x, rank=2)
    _f16(g, b, rank=1)
    check(as_dim(g.shape[0]) == as_dim(x.shape[1]) and as_dim(b.shape[0]) == as_dim(x.shape[1]), "g, b must be f16[C]")
    return TensorType(f16, x.shape)


def layernorm_f16_gpu(x, g, b, eps=1e-5):
    """(x - mean) / sqrt(var + eps) * g + b over the rows of f16[R, C], f32 statistics, one rounding."""
    return _layernorm(x, g, b, eps=_float(eps, "eps"))


@kernel_op("rmsnorm_f16_gpu")
def _rmsnorm(x, g, *, eps):
    _f16(x, rank=2)
    _f16(g, rank=1)
    check(as_dim(g.shape[0]) == as_dim(x.shape[1]), "g must be f16[C]")
    return TensorType(f16, x.shape)


def rmsnorm_f16_gpu(x, g, eps=1e-6):
    """HF Qwen3RMSNorm over the rows of f16[R, C]: h = f16(x * rsqrt(mean(x^2) + eps)), y = f16(g * h)."""
    return _rmsnorm(x, g, eps=_float(eps, "eps"))


@kernel_op("rmsnorm_heads_f16_gpu")
def _rmsnorm_heads(x, g, *, eps):
    _f16(x, rank=2)
    _f16(g, rank=1)
    check(as_dim(g.shape[0]) == as_dim(128), "g must be f16[128]")
    check(_static(x.shape[1], "C") % 128 == 0, "x must be f16[R, H * 128]")
    return TensorType(f16, x.shape)


def rmsnorm_heads_f16_gpu(x, g, eps=1e-6):
    """rmsnorm_f16_gpu over each 128-wide head of f16[R, H * 128] (QK-norm)."""
    return _rmsnorm_heads(x, g, eps=_float(eps, "eps"))


@kernel_op("rope_f16_gpu")
def _rope(x, pos, *, theta):
    _f16(x, rank=2)
    _pos(pos)
    check(_static(x.shape[1], "C") % 128 == 0, "x must be f16[T, H * 128]")
    return TensorType(f16, x.shape)


def rope_f16_gpu(x, pos, theta=1e6):
    """rotate_half RoPE per 128-wide head of x f16[T, H * 128] at positions pos[0] + i: inv_freq[j] = theta^(-2j/128),
    angle = (pos + i) * inv_freq[j] rounded to f32, cos / sin evaluated accurately."""
    return _rope(x, pos, theta=_float(theta, "theta"))


def _binary(x, y):
    _f16(x, y, rank=2)
    _same(x, y, "operand")
    return TensorType(f16, x.shape)


@kernel_op("add_f16_gpu")
def add_f16_gpu(x, y):
    """x + y over f16[R, C]."""
    return _binary(x, y)


@kernel_op("add_clamp_f16_gpu")
def _add_clamp(x, y, *, lim):
    return _binary(x, y)


def add_clamp_f16_gpu(x, y, lim=64504.0):
    """clamp(x + y, -lim, lim) over f16[R, C] (the encoder's f16 clamp)."""
    return _add_clamp(x, y, lim=_float(lim, "lim"))


@kernel_op("silu_mul_f16_gpu")
def silu_mul_f16_gpu(g, u):
    """g * sigmoid(g) * u over f16[R, C] (SwiGLU)."""
    return _binary(g, u)


# --- gathers ------------------------------------------------------------------------------------------------------


@kernel_op("embedding_f16_gpu")
def embedding_f16_gpu(ids, table):
    """table[ids] for ids i64[T], table f16[V, D] -> f16[T, D]; an id outside [0, V) is a run-time error."""
    check(isinstance(ids.type, TensorType) and ids.dtype is i64 and ids.rank == 1, "ids must be i64[T]")
    _f16(table, rank=2)
    return TensorType(f16, [ids.shape[0], table.shape[1]])


@kernel_op("scatter_audio_f16_gpu")
def _scatter_audio(x, ids, audio, *, pad_id):
    _f16(x, audio, rank=2)
    check(isinstance(ids.type, TensorType) and ids.dtype is i64 and ids.rank == 1, "ids must be i64[T]")
    check(as_dim(ids.shape[0]) == as_dim(x.shape[0]), "ids must be i64[T] for x f16[T, D]")
    check(as_dim(audio.shape[1]) == as_dim(x.shape[1]), "audio must be f16[A, D]")
    return TensorType(f16, x.shape)


def scatter_audio_f16_gpu(x, ids, audio, pad_id=151676):
    """Copy of x f16[T, D] where the j-th row whose id is pad_id (in order) becomes audio[j] (audio f16[A, D]); more
    pad ids than A is a run-time error."""
    return _scatter_audio(x, ids, audio, pad_id=_int(pad_id, "pad_id"))


# --- attention ----------------------------------------------------------------------------------------------------


def _head_dim(c, heads):
    c = _static(c, "C")
    check(heads >= 1 and c % heads == 0, "C must be divisible by heads")
    hd = c // heads
    check(hd % 32 == 0 and hd <= 128, "head_dim must be 32, 64, 96 or 128")
    return hd


@kernel_op("attention_window_f16_gpu")
def _attention_window(q, k, v, *, heads, window):
    _f16(q, k, v, rank=2)
    _same(q, k, "k")
    _same(q, v, "v")
    _head_dim(q.shape[1], heads)
    check(window >= 1, "window must be >= 1")
    return TensorType(f16, q.shape)


def attention_window_f16_gpu(q, k, v, heads=16, window=104):
    """Encoder self-attention over q, k, v f16[S, C] (heads x C / heads, scale 1/sqrt(head_dim)), non-causal within
    windows: row i attends to rows [floor(i / window) * window, min(S, (floor(i / window) + 1) * window))."""
    return _attention_window(q, k, v, heads=_int(heads, "heads"), window=_int(window, "window"))


@kernel_op("kv_write_f16_gpu", inout=[0, 1])
def kv_write_f16_gpu(kc, vc, k, v, pos):
    """kc[pos:pos+T] = k, vc[pos:pos+T] = v (caches f16[cap, D], written in place; k, v f16[T, D]); pos + T > cap
    is a run-time error."""
    _f16(kc, vc, k, v, rank=2)
    _same(kc, vc, "vc")
    _same(k, v, "v")
    check(as_dim(k.shape[1]) == as_dim(kc.shape[1]), "k must be f16[T, D] for caches f16[cap, D]")
    _pos(pos)
    return ()


@kernel_op("attention_gqa_f16_gpu")
def _attention_gqa(q, kc, vc, pos, *, kv_heads, q_heads):
    _f16(q, kc, vc, rank=2)
    _same(kc, vc, "vc")
    _pos(pos)
    check(kv_heads >= 1 and q_heads % kv_heads == 0, "q_heads must be a multiple of kv_heads")
    hd = _head_dim(q.shape[1], q_heads)
    check(as_dim(kc.shape[1]) == as_dim(kv_heads * hd), "caches must be f16[cap, kv_heads * head_dim]")
    return TensorType(f16, q.shape)


def attention_gqa_f16_gpu(q, kc, vc, pos, q_heads=16, kv_heads=8):
    """Causal GQA over the caches: query row i of q f16[T, q_heads * hd] (position pos + i) of head h attends to cache
    rows 0 .. pos + i of kv-head h // (q_heads / kv_heads) (kc, vc f16[cap, kv_heads * hd]), scale 1/sqrt(hd).
    Prefill and decode (T = 1) alike; pos + T > cap is a run-time error."""
    return _attention_gqa(q, kc, vc, pos, kv_heads=_int(kv_heads, "kv_heads"), q_heads=_int(q_heads, "q_heads"))


# --- Q5 optimized ops (kernels/gpu/f16_fused.metal; contract docs/impl-notes/qwen3-asr-f16-opt-kernels.md) ----------
# Same math as the basic composition they replace, one rounding at the output (float accumulators and epilogues).
# GEMM-like ops: M = 1 runs a bandwidth-bound GEMV, M > 1 MLX's steel GEMM (half in, float simdgroup-matrix
# accumulators) with the epilogue fused. Weights are the plain [K, N] layout (no packing).


def _res_rule(x, w, res, n):
    _f16(res, rank=2)
    check(as_dim(res.shape[0]) == as_dim(x.shape[0]) and as_dim(res.shape[1]) == as_dim(n), "res must be f16[M, N]")


@kernel_op("gemm_opt_f16_gpu")
def gemm_opt_f16_gpu(x, w):
    """x f16[M, K] @ w f16[K, N] -> f16[M, N]."""
    return _gemm_rule(x, w)


@kernel_op("gemm_bias_opt_f16_gpu")
def gemm_bias_opt_f16_gpu(x, w, b):
    """x @ w + b (b f16[N])."""
    return _gemm_rule(x, w, b)


@kernel_op("gemm_bias_gelu_opt_f16_gpu")
def gemm_bias_gelu_opt_f16_gpu(x, w, b):
    """gelu_erf(x @ w + b) (encoder fc1, projector 1)."""
    return _gemm_rule(x, w, b)


@kernel_op("gemm_add_opt_f16_gpu")
def gemm_add_opt_f16_gpu(x, w, res):
    """res + x @ w (res f16[M, N]; LM o_proj / down_proj + residual)."""
    t = _gemm_rule(x, w)
    _res_rule(x, w, res, w.shape[1])
    return t


@kernel_op("gemm_bias_add_opt_f16_gpu")
def gemm_bias_add_opt_f16_gpu(x, w, b, res):
    """res + (x @ w + b) (encoder out_proj + residual)."""
    t = _gemm_rule(x, w, b)
    _res_rule(x, w, res, w.shape[1])
    return t


@kernel_op("gemm_bias_add_clamp_opt_f16_gpu")
def _gemm_bias_add_clamp_opt(x, w, b, res, *, lim):
    t = _gemm_rule(x, w, b)
    _res_rule(x, w, res, w.shape[1])
    return t


def gemm_bias_add_clamp_opt_f16_gpu(x, w, b, res, lim=64504.0):
    """clamp(res + (x @ w + b), -lim, lim) (encoder fc2 + residual + the f16 clamp)."""
    return _gemm_bias_add_clamp_opt(x, w, b, res, lim=_float(lim, "lim"))


def _rms_gemm_rule(x, g, w, swiglu):
    _f16(g, rank=1)
    _gemm_rule(x, w)
    check(g.shape[0] == x.shape[1], "g must be f16[K]")
    n = w.shape[1]
    if swiglu:
        n2 = _static(n, "2N")
        check(n2 % 2 == 0, "w must be f16[K, 2N] ([gate | up])")
        n = n2 // 2
    return TensorType(f16, [x.shape[0], n]), TensorType(f16, list(x.shape))


@kernel_op("rmsnorm_gemm_opt_f16_gpu")
def _rmsnorm_gemm_opt(x, g, w, *, eps):
    return _rms_gemm_rule(x, g, w, False)


def rmsnorm_gemm_opt_f16_gpu(x, g, w, eps=1e-6):
    """rmsnorm_f16_gpu(x, g) @ w -> (y f16[M, N], ws f16[M, K]). M = 1 (decode): one fused dispatch, ws not written;
    else ws holds the normalized rows. ws is a workspace: use y only."""
    return _rmsnorm_gemm_opt(x, g, w, eps=_float(eps, "eps"))


@kernel_op("rmsnorm_gemm_swiglu_opt_f16_gpu")
def _rmsnorm_gemm_swiglu_opt(x, g, w, *, eps):
    return _rms_gemm_rule(x, g, w, True)


def rmsnorm_gemm_swiglu_opt_f16_gpu(x, g, w, eps=1e-6):
    """gemm_swiglu_opt_f16_gpu(rmsnorm_f16_gpu(x, g), w) -> (y f16[M, N], ws f16[M, K] workspace)."""
    return _rmsnorm_gemm_swiglu_opt(x, g, w, eps=_float(eps, "eps"))


@kernel_op("gemm_swiglu_opt_f16_gpu")
def gemm_swiglu_opt_f16_gpu(x, w):
    """y = x f16[M, K] @ w f16[K, 2N] (w = [gate | up]); silu(y[:, :N]) * y[:, N:] -> f16[M, N]."""
    _gemm_rule(x, w)
    n2 = _static(w.shape[1], "2N")
    check(n2 % 2 == 0, "w must be f16[K, 2N] ([gate | up])")
    return TensorType(f16, [x.shape[0], n2 // 2])


@kernel_op("lm_head_last_opt_f16_gpu")
def lm_head_last_opt_f16_gpu(x, table):
    """x[T-1] @ table^T in f32 for x f16[T, D], table f16[V, D] -> f32[1, V] (bandwidth-bound GEMV over the table)."""
    _f16(x, table, rank=2)
    check(as_dim(x.shape[1]) == as_dim(table.shape[1]), "x f16[T, D] and table f16[V, D] must share D")
    return TensorType(f32, [1, table.shape[0]])


@kernel_op("qknorm_rope_kv_opt_f16_gpu", inout=[4, 5])
def _qknorm_rope_kv_opt(qkv, qg, kg, pos, kc, vc, *, eps, theta):
    _f16(qkv, kc, vc, rank=2)
    _f16(qg, kg, rank=1)
    _pos(pos)
    _same(kc, vc, "vc")
    check(as_dim(qg.shape[0]) == as_dim(128) and as_dim(kg.shape[0]) == as_dim(128), "qg, kg must be f16[128]")
    c, w = _static(qkv.shape[1], "qkv width"), _static(kc.shape[1], "cache width")
    check(w % 128 == 0 and c > 2 * w and (c - 2 * w) % 128 == 0,
          "qkv must be f16[T, Hq * 128 + 2 W] for caches f16[P, W] (W = Hkv * 128)")
    return TensorType(f16, [qkv.shape[0], c - 2 * w])


def qknorm_rope_kv_opt_f16_gpu(qkv, qg, kg, pos, kc, vc, eps=1e-6, theta=1e6):
    """qkv f16[T, Cq + 2 W] (q | k | v; Cq = Hq * 128, caches kc, vc f16[P, W], W = Hkv * 128, written in place):
    returns q = rope(rmsnorm_heads(qkv[:, :Cq], qg)) f16[T, Cq]; writes kc[pos:pos+T] = rope(rmsnorm_heads(k, kg))
    and vc[pos:pos+T] = v. The basic composition's roundings (normalized value, RoPE input, k before the cache);
    pos + T > P is a run-time error."""
    return _qknorm_rope_kv_opt(qkv, qg, kg, pos, kc, vc, eps=_float(eps, "eps"), theta=_float(theta, "theta"))


def _opt_head_dim(c, heads):
    hd = _head_dim(c, heads)
    check(hd in (64, 128), "head_dim must be 64 or 128")
    return hd


@kernel_op("attention_gqa_opt_f16_gpu")
def _attention_gqa_opt(q, kc, vc, pos, *, kv_heads, q_heads):
    _f16(q, kc, vc, rank=2)
    _same(kc, vc, "vc")
    _pos(pos)
    check(kv_heads >= 1 and q_heads % kv_heads == 0, "q_heads must be a multiple of kv_heads")
    hd = _opt_head_dim(q.shape[1], q_heads)
    check(as_dim(kc.shape[1]) == as_dim(kv_heads * hd), "caches must be f16[P, kv_heads * head_dim]")
    return TensorType(f16, q.shape)


def attention_gqa_opt_f16_gpu(q, kc, vc, pos, q_heads=16, kv_heads=8):
    """attention_gqa_f16_gpu's semantics (causal GQA over the caches, head_dim 64 or 128): simdgroup-matrix flash
    attention for T > 1, key-split decode for T = 1; pos + T > P is a run-time error."""
    return _attention_gqa_opt(q, kc, vc, pos, kv_heads=_int(kv_heads, "kv_heads"), q_heads=_int(q_heads, "q_heads"))


@kernel_op("attention_window_qkv_opt_f16_gpu")
def _attention_window_qkv_opt(qkv, *, heads, window):
    _f16(qkv, rank=2)
    c3 = _static(qkv.shape[1], "qkv width")
    check(c3 % 3 == 0, "qkv must be f16[S, 3 C]")
    _opt_head_dim(c3 // 3, heads)
    check(window >= 1, "window must be >= 1")
    return TensorType(f16, [qkv.shape[0], c3 // 3])


def attention_window_qkv_opt_f16_gpu(qkv, heads=16, window=104):
    """attention_window_f16_gpu on q = qkv[:, :C], k = qkv[:, C:2C], v = qkv[:, 2C:] (qkv f16[S, 3C], head_dim 64 or
    128) -> f16[S, C]: simdgroup-matrix flash attention, one threadgroup per window."""
    return _attention_window_qkv_opt(qkv, heads=_int(heads, "heads"), window=_int(window, "window"))


@kernel_op("audio_cnn_opt_f16_gpu")
def audio_cnn_opt_f16_gpu(features, w1, b1, w2, b2, w3, b3, w_out):
    """audio_cnn_f16_gpu's signature and math (Co % 4 == 0): conv1 direct, conv2 / conv3 as im2col + the steel GEMM
    with the bias + GELU epilogue, conv_out as one GEMM, a few chunks per batch (bounded scratch)."""
    t = audio_cnn_f16_gpu.__wrapped__(features, w1, b1, w2, b2, w3, b3, w_out)
    check(_static(w1.shape[0], "Co") % 4 == 0, "Co must be a multiple of 4")
    return t
