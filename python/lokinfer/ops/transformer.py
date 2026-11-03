"""Transformer kernels for GPT-2 (M6; research-models.md §5): causal attention over a fused QKV projection, the tied
LM head (GEMM with transposed B: reads wte [V, D] directly, no transposed copy), GPU LayerNorm / add+LayerNorm."""
from ..dims import as_dim
from ..dtypes import f32
from ..types import TensorType
from ._def import check, kernel_op
from .kokoro import _f32, _float, _int
from .kokoro_gpu import _same


def _attn_rule(qkv, *, heads):
    _f32(qkv, rank=2)
    d3 = as_dim(qkv.shape[1])
    check(d3.is_const and d3.value % (3 * heads) == 0, "qkv must be f32[T, 3D] with D divisible by heads")
    return TensorType(f32, [qkv.shape[0], d3.value // 3])


@kernel_op("attention_qkv_f32_cpu")
def _attn_cpu(qkv, *, heads):
    return _attn_rule(qkv, heads=heads)


@kernel_op("attention_qkv_causal_f32_cpu")
def _attn_causal_cpu(qkv, *, heads):
    return _attn_rule(qkv, heads=heads)


def attention_qkv_f32_cpu(qkv, heads, causal=False):
    """SDPA over a fused QKV projection qkv f32[T, 3D] (q | k | v, biases added) -> f32[T, D]; causal = GPT-2 mask."""
    return (_attn_causal_cpu if causal else _attn_cpu)(qkv, heads=_int(heads, "heads"))


def _gemm_nt_rule(a, b):
    _f32(a, b, rank=2)
    check(a.shape[1] == b.shape[1], f"a [M, K] and b [N, K] must share K: {a.shape}, {b.shape}")
    return TensorType(f32, [a.shape[0], b.shape[0]])


@kernel_op("gemm_nt_f32_cpu_accelerate")
def gemm_nt_f32_cpu_accelerate(a, b):
    """a f32[M, K] @ b f32[N, K]^T -> f32[M, N] (Accelerate)."""
    return _gemm_nt_rule(a, b)


@kernel_op("gemm_nt_f32_gpu_mps")
def gemm_nt_f32_gpu_mps(a, b):
    """GPU a f32[M, K] @ b f32[N, K]^T -> f32[M, N] (MPS, B read through a transposed view)."""
    return _gemm_nt_rule(a, b)


@kernel_op("add_bias_layernorm2_f32_gpu")
def _add_ln2(a, bias, r, gamma, beta, *, eps):
    _f32(a, r, rank=2)
    _f32(bias, gamma, beta, rank=1)
    _same(a, r, "residual")
    c = as_dim(a.shape[1])
    check(c.is_const and c.value <= 2048, "C must be static and <= 2048")
    t = TensorType(f32, a.shape)
    return t, t


def add_bias_layernorm2_f32_gpu(a, bias, r, gamma, beta, eps=1e-5):
    """(s, LN(s)) with s = a + bias + r over rows of f32[R, C] (pre-LN residual stream + next sub-block input)."""
    return _add_ln2(a, bias, r, gamma, beta, eps=_float(eps, "eps"))


@kernel_op("layernorm_f32_gpu")
def _ln_gpu(x, gamma, beta, *, eps):
    _f32(x, rank=2)
    _f32(gamma, beta, rank=1)
    c = as_dim(x.shape[1])
    check(c.is_const and c.value <= 2048, "C must be static and <= 2048")
    return TensorType(f32, x.shape)


def layernorm_f32_gpu(x, gamma, beta, eps=1e-5):
    """GPU F.layer_norm over the last dim of f32[R, C]."""
    return _ln_gpu(x, gamma, beta, eps=_float(eps, "eps"))


# --- decode with a persistent KV cache (kernels/cpu/decode.cpp, kokoro_tok.metal) -----------------------------


def _kv_rule(kc, vc, qkv, pos, *bias):
    _f32(kc, vc, qkv, rank=2)
    check(kc.shape[0] == vc.shape[0] and kc.shape[1] == vc.shape[1], "caches must be f32[P, D]")
    check(qkv.shape[0] == 1 and qkv.shape[1] == 3 * kc.shape[1], "qkv must be f32[1, 3D]")
    from ..dtypes import i64

    check(pos.dtype is i64 and pos.rank == 1 and pos.shape[0] == 1, "pos must be i64[1]")
    if bias:
        _f32(bias[0], rank=1)
        check(bias[0].shape[0] == qkv.shape[1], "bias must be f32[3D]")
    return ()


@kernel_op("kv_write_f32_cpu", inout=[0, 1])
def kv_write_f32_cpu(kcache, vcache, qkv, pos):
    """kcache[pos] = k, vcache[pos] = v from this step's fused QKV row qkv f32[1, 3D] (biases added); writes the
    persistent caches in place. pos: i64[1]."""
    return _kv_rule(kcache, vcache, qkv, pos)


@kernel_op("kv_write_f32_gpu", inout=[0, 1])
def kv_write_f32_gpu(kcache, vcache, qkv, pos, bias):
    """GPU kv_write; qkv is the bias-free GEMM output, `bias` f32[3D] is added on write."""
    return _kv_rule(kcache, vcache, qkv, pos, bias)


def _decode_rule(qkv, kc, vc, pos, *, heads):
    _kv_rule(kc, vc, qkv, pos)
    return TensorType(f32, [1, kc.shape[1]])


@kernel_op("attention_decode_f32_cpu")
def _attention_decode_cpu(qkv, kcache, vcache, pos, *, heads):
    return _decode_rule(qkv, kcache, vcache, pos, heads=heads)


@kernel_op("attention_decode_f32_gpu")
def _attention_decode_gpu(qkv, kcache, vcache, pos, bias, *, heads):
    _f32(bias, rank=1)
    return _decode_rule(qkv, kcache, vcache, pos, heads=heads)


def attention_decode_f32_cpu(qkv, kcache, vcache, pos, heads):
    """One query (qkv's q part, f32[1, 3D]) against cache rows 0..pos -> f32[1, D]."""
    return _attention_decode_cpu(qkv, kcache, vcache, pos, heads=_int(heads, "heads"))


def attention_decode_f32_gpu(qkv, kcache, vcache, pos, bias, heads):
    """GPU attention_decode (head_dim 64, cache capacity <= 1024); q bias added on read."""
    return _attention_decode_gpu(qkv, kcache, vcache, pos, bias, heads=_int(heads, "heads"))


# --- steel GEMM (kernels/gpu/gemm_steel.metal) -------------------------------------------------------------------


def _gemm_rule(a, b, *rest, nt=False):
    _f32(a, b, rank=2)
    if nt:
        check(a.shape[1] == b.shape[1], f"a [M, K] and b [N, K] must share K: {a.shape}, {b.shape}")
        n = b.shape[0]
    else:
        check(a.shape[1] == b.shape[0], f"a [M, K] and b [K, N]: inner dims differ: {a.shape}, {b.shape}")
        n = b.shape[1]
    t = TensorType(f32, [a.shape[0], n])
    if rest:
        _f32(rest[0], rank=1)
        check(rest[0].shape[0] == n, "bias must be f32[N]")
    if len(rest) > 1:
        _f32(rest[1], rank=2)
        _same(rest[1], t, "residual")
    return t


_gemm_steel = kernel_op("gemm_steel_f32_gpu")(lambda a, b: _gemm_rule(a, b))
_gemm_steel_bias = kernel_op("gemm_steel_bias_f32_gpu")(lambda a, b, bias: _gemm_rule(a, b, bias))
_gemm_steel_bias_gelu = kernel_op("gemm_steel_bias_gelu_f32_gpu")(lambda a, b, bias: _gemm_rule(a, b, bias))
_gemm_steel_bias_relu = kernel_op("gemm_steel_bias_relu_f32_gpu")(lambda a, b, bias: _gemm_rule(a, b, bias))
_gemm_steel_bias_res = kernel_op("gemm_steel_bias_res_f32_gpu")(lambda a, b, bias, r: _gemm_rule(a, b, bias, r))
gemm_nt_steel_f32_gpu = kernel_op("gemm_nt_steel_f32_gpu")(lambda a, b: _gemm_rule(a, b, nt=True))
gemm_nt_steel_f32_gpu.__doc__ = "GPU a f32[M, K] @ b f32[N, K]^T -> f32[M, N] (steel, transposed-B loader)."


def gemm_steel_f32_gpu(a, b, bias=None, act="none", residual=None):
    """GPU act(a f32[M, K] @ b f32[K, N] + bias) [+ residual] on steel tiles; act "none", "gelu" (tanh approx) or
    "relu". Supported combos: plain, +bias, +bias+gelu, +bias+relu, +bias+residual."""
    if bias is None:
        check(act == "none" and residual is None, "act/residual need a bias")
        return _gemm_steel(a, b)
    if residual is not None:
        check(act == "none", "bias+gelu+residual is not a kernel")
        return _gemm_steel_bias_res(a, b, bias, residual)
    if act == "gelu":
        return _gemm_steel_bias_gelu(a, b, bias)
    if act == "relu":
        return _gemm_steel_bias_relu(a, b, bias)
    check(act == "none", "act must be none or gelu")
    return _gemm_steel_bias(a, b, bias)



# --- decode GEMVs (B = 1, bandwidth-bound; kokoro_tok.metal) -----------------------------------------------------


def _gemv_rule(x, w, *bias):
    _f32(x, w, rank=2)
    check(x.shape[0] == 1 and x.shape[1] == w.shape[0], f"x must be f32[1, K] and w f32[K, N]: {x.shape}, {w.shape}")
    if bias:
        _f32(bias[0], rank=1)
        check(bias[0].shape[0] == w.shape[1], "bias must be f32[N]")
    return TensorType(f32, [1, w.shape[1]])


_gemv = kernel_op("gemv_f32_gpu")(lambda x, w: _gemv_rule(x, w))
_gemv_bias = kernel_op("gemv_bias_f32_gpu")(lambda x, w, b: _gemv_rule(x, w, b))
_gemv_bias_gelu = kernel_op("gemv_bias_gelu_f32_gpu")(lambda x, w, b: _gemv_rule(x, w, b))


def gemv_f32_gpu(x, w, bias=None, act="none"):
    """GPU x f32[1, K] @ w f32[K, N] (+ bias) (gelu) -> f32[1, N]: bandwidth-oriented decode GEMV."""
    if bias is None:
        check(act == "none", "act needs a bias")
        return _gemv(x, w)
    return _gemv_bias_gelu(x, w, bias) if act == "gelu" else _gemv_bias(x, w, bias)


@kernel_op("gemv_rows_f32_gpu")
def gemv_rows_f32_gpu(x, w):
    """GPU x f32[1, K] . rows of w f32[N, K] -> f32[1, N] (tied LM head over wte)."""
    _f32(x, w, rank=2)
    check(x.shape[0] == 1 and x.shape[1] == w.shape[1], "x must be f32[1, K] and w f32[N, K]")
    return TensorType(f32, [1, w.shape[0]])


# --- chunked prefill / decode (T new tokens at pos0 .. pos0+T-1; one model shares its caches) --------------------


def _rows_rule(kc, vc, qkv, pos, *bias):
    _f32(kc, vc, qkv, rank=2)
    check(kc.shape[1] == vc.shape[1] and kc.shape[0] == vc.shape[0], "caches must be f32[P, D]")
    check(qkv.shape[1] == 3 * kc.shape[1], "qkv must be f32[T, 3D]")
    from ..dtypes import i64

    check(pos.dtype is i64 and pos.rank == 1 and pos.shape[0] == 1, "pos must be i64[1] (start position)")
    if bias:
        _f32(bias[0], rank=1)
        check(bias[0].shape[0] == qkv.shape[1], "bias must be f32[3D]")
    return ()


@kernel_op("kv_write_rows_f32_cpu", inout=[0, 1])
def kv_write_rows_f32_cpu(kcache, vcache, qkv, pos):
    """kcache[pos + i] = k_i, vcache[pos + i] = v_i for the T rows of qkv f32[T, 3D] (biases added); in place."""
    return _rows_rule(kcache, vcache, qkv, pos)


@kernel_op("kv_write_rows_f32_gpu", inout=[0, 1])
def kv_write_rows_f32_gpu(kcache, vcache, qkv, pos, bias):
    """GPU kv_write_rows; `bias` f32[3D] is added on write (the GPU QKV GEMM has none)."""
    return _rows_rule(kcache, vcache, qkv, pos, bias)


def _cache_attn_rule(qkv, kc, vc, pos):
    _rows_rule(kc, vc, qkv, pos)
    return TensorType(f32, [qkv.shape[0], kc.shape[1]])


@kernel_op("attention_cache_f32_cpu")
def _attention_cache_cpu(qkv, kcache, vcache, pos, *, heads):
    return _cache_attn_rule(qkv, kcache, vcache, pos)


@kernel_op("attention_cache_f32_gpu")
def _attention_cache_gpu(qkv, kcache, vcache, pos, bias, *, heads):
    _f32(bias, rank=1)
    return _cache_attn_rule(qkv, kcache, vcache, pos)


def attention_cache_f32_cpu(qkv, kcache, vcache, pos, heads):
    """Causal attention of the T queries in qkv f32[T, 3D] (positions pos .. pos+T-1) over cache rows 0 .. pos+i."""
    return _attention_cache_cpu(qkv, kcache, vcache, pos, heads=_int(heads, "heads"))


def attention_cache_f32_gpu(qkv, kcache, vcache, pos, bias, heads):
    """GPU attention_cache (head_dim 64; q bias added on read)."""
    return _attention_cache_gpu(qkv, kcache, vcache, pos, bias, heads=_int(heads, "heads"))



@kernel_op("arange_offset_i64_cpu")
def _arange_offset(pos, *, length):
    from ..dtypes import i64

    check(pos.dtype is i64 and pos.rank == 1 and pos.shape[0] == 1, "pos must be i64[1]")
    return TensorType(i64, [length])


def arange_offset_i64_cpu(pos, length):
    """i64[length] = pos[0] + [0, 1, ..., length-1] (position ids of a chunk of new tokens)."""
    return _arange_offset(pos, length=as_dim(length))


# --- simdgroup-matrix flash attention (kernels/gpu/kokoro_tok.metal lk_attention_qkv_sg) -------------------------


def _attn_sg_rule(qkv, bias, *, heads, nsg):
    check(nsg in (2, 4, 8), "nsg must be 2, 4 or 8")
    t = _attn_rule(qkv, heads=heads)
    _f32(bias, rank=1)
    check(as_dim(bias.shape[0]) == as_dim(qkv.shape[1]), "bias must be f32[3D]")
    check(as_dim(qkv.shape[1]).value == 3 * 64 * heads, "head_dim must be 64")
    return t


_attention_qkv_sg = kernel_op("attention_qkv_sg_f32_gpu")(
    lambda qkv, bias, *, heads, nsg: _attn_sg_rule(qkv, bias, heads=heads, nsg=nsg))
_attention_qkv_sg_causal = kernel_op("attention_qkv_sg_causal_f32_gpu")(
    lambda qkv, bias, *, heads, nsg: _attn_sg_rule(qkv, bias, heads=heads, nsg=nsg))


def attention_qkv_sg_f32_gpu(qkv, bias, heads, causal=False, nsg=4):
    """attention_qkv_f32_gpu on simdgroup matrices: qkv f32[T, 3D] (before bias) + bias f32[3D] -> f32[T, D],
    head_dim 64, scale 1/8; causal = GPT-2's mask. nsg: simdgroups (8 query rows each) per threadgroup (2, 4, 8)."""
    return (_attention_qkv_sg_causal if causal else _attention_qkv_sg)(qkv, bias, heads=_int(heads, "heads"),
                                                                       nsg=_int(nsg, "nsg"))


@kernel_op("attention_decode_kvw_f32_gpu", inout=[1, 2])
def _attention_decode_kvw_gpu(qkv, kcache, vcache, pos, bias, *, heads):
    _f32(bias, rank=1)
    return _decode_rule(qkv, kcache, vcache, pos, heads=heads)


def attention_decode_kvw_f32_gpu(qkv, kcache, vcache, pos, bias, heads):
    """kv_write_f32_gpu + attention_decode_f32_gpu in one kernel: writes this step's k/v (+ bias) to cache row pos
    and attends the query over rows 0..pos (head_dim 64, cache capacity <= 1024; one 1024-thread threadgroup per
    head)."""
    return _attention_decode_kvw_gpu(qkv, kcache, vcache, pos, bias, heads=_int(heads, "heads"))


@kernel_op("attention_decode_split_f32_gpu", inout=[1, 2])
def _attention_decode_split_gpu(qkv, kcache, vcache, pos, bias, *, heads, nsg, single, splits):
    _f32(bias, rank=1)
    check(nsg in (0, 4, 8, 16, 32), "nsg must be 0 (auto), 4, 8, 16 or 32")
    check(0 <= splits <= 32 and heads <= 16, "splits must be 0 (auto) .. 32, heads <= 16")
    return _decode_rule(qkv, kcache, vcache, pos, heads=heads)


def attention_decode_split_f32_gpu(qkv, kcache, vcache, pos, bias, heads, splits=0, nsg=0, single=1):
    """attention_decode_kvw_f32_gpu with the keys split over `splits` threadgroups per head (nsg simdgroups each) and
    a combine: single=1 -> in the same dispatch (the last split of each head to finish merges, device atomic
    counter), 0 -> a second dispatch. 0 = per-platform default for splits / nsg."""
    return _attention_decode_split_gpu(qkv, kcache, vcache, pos, bias, heads=_int(heads, "heads"),
                                       nsg=_int(nsg, "nsg"), single=_int(single, "single"),
                                       splits=_int(splits, "splits"))


@kernel_op("attention_cache_sg_f32_gpu")
def _attention_cache_sg_gpu(qkv, kcache, vcache, pos, bias, *, heads, nsg):
    _f32(bias, rank=1)
    check(nsg in (2, 4, 8), "nsg must be 2, 4 or 8")
    check(as_dim(qkv.shape[1]).value == 3 * 64 * heads, "head_dim must be 64")
    return _cache_attn_rule(qkv, kcache, vcache, pos)


def attention_cache_sg_f32_gpu(qkv, kcache, vcache, pos, bias, heads, nsg=4):
    """attention_cache_f32_gpu on simdgroup matrices (lk_attention_cache_sg): the T queries of qkv (positions
    pos..pos+T-1, q bias added) attend causally to cache rows 0..pos+i, which must already hold the chunk's k/v."""
    return _attention_cache_sg_gpu(qkv, kcache, vcache, pos, bias, heads=_int(heads, "heads"), nsg=_int(nsg, "nsg"))


def _gemv4_rule(x, w, *rest, cw):
    check(cw in (0, 2, 4, 8, 16), "cw must be 0 (auto), 2, 4, 8 or 16")
    t = _gemv_rule(x, w, *rest[:1])
    n = as_dim(w.shape[1])
    check(n.is_const and n.value % 4 == 0, "N must be static and a multiple of 4")
    if len(rest) > 1:
        _f32(rest[1], rank=2)
        _same(rest[1], t, "residual")
    return t


_gemv4 = kernel_op("gemv4_f32_gpu")(lambda x, w, *, cw: _gemv4_rule(x, w, cw=cw))
_gemv4_bias = kernel_op("gemv4_bias_f32_gpu")(lambda x, w, b, *, cw: _gemv4_rule(x, w, b, cw=cw))
_gemv4_bias_gelu = kernel_op("gemv4_bias_gelu_f32_gpu")(lambda x, w, b, *, cw: _gemv4_rule(x, w, b, cw=cw))
_gemv4_bias_res = kernel_op("gemv4_bias_res_f32_gpu")(lambda x, w, b, r, *, cw: _gemv4_rule(x, w, b, r, cw=cw))


def gemv4_f32_gpu(x, w, bias=None, act="none", residual=None, cw=0):
    """GPU decode GEMV x f32[1, K] @ w f32[K, N] (N % 4 == 0) with a fused epilogue: + bias, gelu_tanh, + residual
    (combos: plain, bias, bias+gelu, bias+residual). Each 256-thread threadgroup computes cw float4 columns (cw in
    2, 4, 8, 16) with 256 / cw K slices; cw = 0 picks the platform's tuned default at run time (macOS 4, iOS 8)."""
    cw = _int(cw, "cw")
    if bias is None:
        check(act == "none" and residual is None, "act/residual need a bias")
        return _gemv4(x, w, cw=cw)
    if residual is not None:
        check(act == "none", "bias+gelu+residual is not a kernel")
        return _gemv4_bias_res(x, w, bias, residual, cw=cw)
    return _gemv4_bias_gelu(x, w, bias, cw=cw) if act == "gelu" else _gemv4_bias(x, w, bias, cw=cw)


# --- CPU prefill fusions (kernels/cpu/gemm.cpp, decode.cpp) ------------------------------------------------------

_ACTS = {"none": 0, "gelu": 1, "gelu_vv": 2, "relu": 3}


@kernel_op("gemm_bias_act_f32_cpu_accelerate")
def _gemm_bias_act_cpu(a, b, bias, *, act, split):
    return _gemm_rule(a, b, bias)


@kernel_op("gemm_split_f32_cpu_accelerate")
def _gemm_split_cpu(a, b, *, split):
    return _gemm_rule(a, b)


def gemm_accel_f32_cpu(a, b, bias=None, act="none", split=0):
    """act(a f32[M, K] @ b f32[K, N] + bias) on Accelerate with the epilogue fused per block. act: "none", "gelu"
    (tanh approximation, NEON exp), "gelu_vv" (same via vvtanhf), "relu". split: 0 = auto (per platform and shape,
    gemm.cpp autoSplit); 6000 = one multithreaded Accelerate call (epilogue on the pool when heavy); > 0 = that many
    column blocks, < 0 = that many row blocks, each a single-threaded Accelerate call + epilogue on a pool task;
    5000 + t = small-M K split over t tasks (partials reduced)."""
    check(act in _ACTS, f"act must be one of {list(_ACTS)}")
    if bias is None:
        check(act == "none", "act needs a bias")
        return _gemm_split_cpu(a, b, split=_int(split, "split"))
    return _gemm_bias_act_cpu(a, b, bias, act=_ACTS[act], split=_int(split, "split"))


@kernel_op("add_bias_layernorm2_f32_cpu")
def _add_ln2_cpu(a, bias, r, gamma, beta, *, eps):
    _f32(a, r, rank=2)
    _f32(bias, gamma, beta, rank=1)
    _same(a, r, "residual")
    t = TensorType(f32, a.shape)
    return t, t


def add_bias_layernorm2_f32_cpu(a, bias, r, gamma, beta, eps=1e-5):
    """CPU (s, LN(s)) with s = a + bias + r over rows of f32[R, C]."""
    return _add_ln2_cpu(a, bias, r, gamma, beta, eps=_float(eps, "eps"))


@kernel_op("attention_qkv_causal_blk_f32_cpu")
def _attn_causal_blk_cpu(qkv, *, heads, mode, qb):
    return _attn_rule(qkv, heads=heads)


def attention_qkv_causal_blk_f32_cpu(qkv, heads, qb=64, mode=0):
    """Causal attention_qkv_f32_cpu over (head, qb-row query block) work items with the keys trimmed to each block's
    causal extent; NEON softmax. mode 0: items on the pool, 1: sequential (one BLAS stream)."""
    return _attn_causal_blk_cpu(qkv, heads=_int(heads, "heads"), mode=_int(mode, "mode"), qb=_int(qb, "qb"))


# --- split-K GEMM (gemm_steel.metal lk_gemm_splitk) + fused partial-sum consumer -----------------------------------


def _splitk_rule(a, b, *, splits, tile):
    t = _gemm_rule(a, b)
    k = as_dim(a.shape[1])
    check(splits >= 1 and k.is_const and k.value % (16 * splits) == 0, "K must be static and divisible by 16*splits")
    check(tile in (32, 64), "tile must be 32 or 64")
    return TensorType(f32, [splits, t.shape[0], t.shape[1]])


_gemm_splitk_gpu = kernel_op("gemm_splitk_f32_gpu")(_splitk_rule)


def gemm_splitk_f32_gpu(a, b, splits, tile=64):
    """GPU split-K partials: f32[S, M, N] with part s = a[:, s*K/S:(s+1)*K/S] @ b[s*K/S:(s+1)*K/S, :] (steel tiles of
    tile x tile); consumed by add_bias_layernorm2_sk_f32_gpu, which sums them."""
    return _gemm_splitk_gpu(a, b, splits=_int(splits, "splits"), tile=_int(tile, "tile"))


@kernel_op("add_bias_layernorm2_sk_f32_gpu")
def _add_ln2_sk(parts, bias, r, gamma, beta, *, eps):
    _f32(parts, rank=3)
    _f32(r, rank=2)
    _f32(bias, gamma, beta, rank=1)
    check(parts.shape[1] == r.shape[0] and parts.shape[2] == r.shape[1], "parts must be f32[S, R, C] over r f32[R, C]")
    c = as_dim(r.shape[1])
    check(c.is_const and c.value <= 2048, "C must be static and <= 2048")
    t = TensorType(f32, r.shape)
    return t, t


def add_bias_layernorm2_sk_f32_gpu(parts, bias, r, gamma, beta, eps=1e-5):
    """add_bias_layernorm2_f32_gpu with a = sum of split-K partials parts f32[S, R, C]."""
    return _add_ln2_sk(parts, bias, r, gamma, beta, eps=_float(eps, "eps"))


# --- runtime-dispatched generation ops (one graph for prefill and decode; transformer_gpu.cpp) ----------------------

gemm_auto_f32_gpu = kernel_op("gemm_auto_f32_gpu")(lambda x, w: _gemm_rule(x, w))
gemm_auto_f32_gpu.__doc__ = "GPU x f32[M, K] @ w f32[K, N]: M = 1 -> gemv4 (decode), else MPS (chunked prefill)."
gemm_auto_bias_gelu_f32_gpu = kernel_op("gemm_auto_bias_gelu_f32_gpu")(lambda x, w, b: _gemm_rule(x, w, b))
gemm_auto_bias_gelu_f32_gpu.__doc__ = ("GPU gelu_tanh(x @ w + b): M = 1 -> gemv4 with the fused epilogue, else MPS + "
                                       "bias_gelu.")


@kernel_op("attention_gen_f32_gpu", inout=[1, 2])
def _attention_gen_gpu(qkv, kcache, vcache, pos, bias, *, heads):
    _f32(bias, rank=1)
    return _cache_attn_rule(qkv, kcache, vcache, pos)


def attention_gen_f32_gpu(qkv, kcache, vcache, pos, bias, heads):
    """Writes the T new k/v rows (+ bias) to the caches at pos.. and attends causally over rows 0..pos+i: T = 1 ->
    attention_decode_kvw (one fused kernel), T > 1 -> kv_write_rows + attention_cache (flash over cache tiles)."""
    return _attention_gen_gpu(qkv, kcache, vcache, pos, bias, heads=_int(heads, "heads"))


@kernel_op("attention_gen2_f32_gpu", inout=[1, 2])
def _attention_gen2_gpu(qkv, kcache, vcache, pos, bias, *, heads):
    _f32(bias, rank=1)
    return _cache_attn_rule(qkv, kcache, vcache, pos)


def attention_gen2_f32_gpu(qkv, kcache, vcache, pos, bias, heads):
    """attention_gen_f32_gpu with the newer kernels: T = 1 -> attention_decode_split (keys split across threadgroups
    + combine), T > 1 -> kv_write_rows + simdgroup-matrix flash attention (over qkv at pos 0, else over the cache:
    attention_cache_sg)."""
    return _attention_gen2_gpu(qkv, kcache, vcache, pos, bias, heads=_int(heads, "heads"))


@kernel_op("gemm_auto_sk_f32_gpu")
def _gemm_auto_sk_gpu(a, b, *, splits, tile):
    return _splitk_rule(a, b, splits=splits, tile=tile)


def gemm_auto_sk_f32_gpu(a, b, splits, tile=32):
    """gemm_splitk_f32_gpu for one graph serving T = 1 and T > 1 (build_generate): M = 1 -> gemv4 into plane 0, else
    split-K partials (MPS into plane 0 for big M); consumed by add_bias_layernorm2_sk_f32_gpu."""
    return _gemm_auto_sk_gpu(a, b, splits=_int(splits, "splits"), tile=_int(tile, "tile"))
