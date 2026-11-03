"""Kokoro-82M frame-rate GPU kernels, channel-last `[L, C]` (M4; kernels/gpu/kokoro_lc.metal).

Weights for `conv1d_lc*` are `[Co, K, Ci]` (torch's `[Co, Ci, K]` permuted offline). ConvTranspose1d is
`gemm_f32_gpu_mps(x, W[Ci, K*Co])` followed by `convt_gather_lc_f32_gpu`.
"""
from ..dims import as_dim, ceildiv
from ..dtypes import f32
from ..types import TensorType
from ._def import check, kernel_op
from .kokoro import STAT_CHUNK, _f32, _float, _ge, _int


def _same(a, b, what):
    check(len(a.shape) == len(b.shape) and all(x == y for x, y in zip(a.shape, b.shape)), f"{what} shape mismatch")


# --- conv -------------------------------------------------------------------------------------------------


def _conv_lc_rule(x, w, b, *, dilation, padding, stride):
    _f32(x, w, rank=None)
    check(x.rank == 2 and w.rank == 3, "x must be f32[L, Ci], w f32[Co, K, Ci]")
    _f32(b, rank=1)
    check(w.shape[2] == x.shape[1], f"w must be [Co, K, Ci]: x {x.shape}, w {w.shape}")
    check(b.shape[0] == w.shape[0], "bias must be f32[Co]")
    ci, k = as_dim(x.shape[1]), as_dim(w.shape[1])
    check(ci.is_const and ci.value % 16 == 0, "Ci must be a static multiple of 16 (steel loader)")
    check(k.is_const and k.value <= 16, "kernel size must be static and <= 16 (small-filter loader)")
    check(stride == 1 and dilation >= 1 and padding >= 0, "stride 1 only (strided convs stay on the CPU for now)")
    lo = (x.shape[0] + 2 * padding - dilation * (k.value - 1) - 1) // stride + 1
    check(_ge(lo, 1), "empty conv output")
    return TensorType(f32, [lo, w.shape[0]])


@kernel_op("conv1d_lc_f32_gpu")
def _conv1d_lc(x, w, b, *, dilation, padding, stride):
    return _conv_lc_rule(x, w, b, dilation=dilation, padding=padding, stride=stride)


@kernel_op("conv1d_lc_res_f32_gpu")
def _conv1d_lc_res(x, w, b, res, *, dilation, oscale, padding, stride):
    t = _conv_lc_rule(x, w, b, dilation=dilation, padding=padding, stride=stride)
    _f32(res, rank=2)
    _same(res, t, "residual")
    return t


def conv1d_lc_f32_gpu(x, w, b, residual=None, out_scale=1.0, padding=0, dilation=1):
    """(conv1d(x, w) + b [+ residual]) * out_scale, channel-last: x f32[L, Ci], w f32[Co, K, Ci] -> f32[Lo, Co]."""
    attrs = dict(padding=_int(padding, "padding"), dilation=_int(dilation, "dilation"), stride=1)
    if residual is None:
        check(out_scale == 1.0, "out_scale needs a residual")
        return _conv1d_lc(x, w, b, **attrs)
    return _conv1d_lc_res(x, w, b, residual, oscale=_float(out_scale, "out_scale"), **attrs)


# --- activations ------------------------------------------------------------------------------------------


@kernel_op("snake_affine_lc_f32_gpu")
def _snake_affine_lc(x, affine, alpha):
    _f32(x, affine, rank=2)
    _f32(alpha, rank=1)
    check(affine.shape[0] == 3 and affine.shape[1] == x.shape[1] and alpha.shape[0] == x.shape[1],
          "affine must be f32[3, C] and alpha f32[C]")
    return TensorType(f32, x.shape)


@kernel_op("leaky_affine_lc_f32_gpu")
def _leaky_affine_lc(x, affine, *, slope):
    _f32(x, affine, rank=2)
    check(affine.shape[0] == 3 and affine.shape[1] == x.shape[1], "affine must be f32[3, C]")
    return TensorType(f32, x.shape)


@kernel_op("leaky_lc_f32_gpu")
def _leaky_lc(x, *, slope):
    _f32(x, rank=2)
    return TensorType(f32, x.shape)


def snake_affine_lc_f32_gpu(x, affine, alpha):
    """snake((x - mean) * scale + beta, alpha) per channel of x f32[L, C]; affine = f32[3, C]."""
    return _snake_affine_lc(x, affine, alpha)


def leaky_lc_f32_gpu(x, slope, affine=None):
    """leaky_relu([(x - mean) * scale + beta], slope) per channel of x f32[L, C]."""
    if affine is None:
        return _leaky_lc(x, slope=_float(slope, "slope"))
    return _leaky_affine_lc(x, affine, slope=_float(slope, "slope"))


# --- AdaIN ------------------------------------------------------------------------------------------------


@kernel_op("channel_stats_lc_f32_gpu")
def channel_stats_lc_f32_gpu(x):
    """Per-(128-row chunk, channel) (mean, M2) of x f32[L, C] -> f32[ceil(L / 128), C, 2]."""
    _f32(x, rank=2)
    return TensorType(f32, [ceildiv(x.shape[0], STAT_CHUNK), x.shape[1], 2])


@kernel_op("adain_affine_lc_f32_gpu")
def _adain_affine_lc(stats, gb, *, eps, length):
    _f32(stats, rank=3)
    _f32(gb, rank=1)
    c = stats.shape[1]
    g2 = as_dim(gb.shape[0])
    check(g2.is_const and g2.value % 2 == 0 and _ge(c, g2.value // 2), "gb must be f32[2 Creal] with Creal <= C")
    check(stats.shape[0] == ceildiv(length, STAT_CHUNK), "stats chunks don't match length")
    return TensorType(f32, [3, c])


def adain_affine_lc_f32_gpu(stats, gb, length, eps=1e-5):
    """AdaIN affine f32[3, C] = (mean, (1 + g) / std, b) from channel_stats_lc over `length` rows. gb may cover
    only the first Creal = len(gb) / 2 channels: the rest are zero padding and get affine (0, 0, 0)."""
    return _adain_affine_lc(stats, gb, eps=_float(eps, "eps"), length=as_dim(length))


# --- ConvTranspose1d, misc --------------------------------------------------------------------------------


def _convt_rule(y, b, *, kernel, padding, reflect_left, stride, lin_out):
    _f32(y, rank=2)
    _f32(b, rank=1)
    co = b.shape[0]
    check(y.shape[1] == kernel * co, "y must be f32[Lin, K * Co]")
    check(stride >= 1 and padding >= 0 and reflect_left in (0, 1), "invalid convT attrs")
    lout = (y.shape[0] - 1) * stride - 2 * padding + kernel + reflect_left
    check(_ge(lout, 2 if reflect_left else 1), "empty convT output")
    return TensorType(f32, [lout, co])


@kernel_op("convt_gather_lc_f32_gpu")
def _convt_gather(y, b, *, kernel, padding, reflect_left, stride):
    return _convt_rule(y, b, kernel=kernel, padding=padding, reflect_left=reflect_left, stride=stride, lin_out=None)


@kernel_op("convt_gather_add_lc_f32_gpu")
def _convt_gather_add(y, b, add, *, kernel, padding, reflect_left, stride):
    t = _convt_rule(y, b, kernel=kernel, padding=padding, reflect_left=reflect_left, stride=stride, lin_out=None)
    _f32(add, rank=2)
    _same(add, t, "add")
    return t


def convt_gather_lc_f32_gpu(y, b, kernel, stride, padding, add=None, reflect_left=False):
    """ConvTranspose1d (output_padding 0) overlap-add of y = x @ W[Ci, K*Co] (f32[Lin, K*Co]) + b
    [+ add] -> f32[Lout, Co]; reflect_left prepends F.pad(..., (1, 0), mode="reflect")."""
    attrs = dict(kernel=_int(kernel, "kernel"), stride=_int(stride, "stride"), padding=_int(padding, "padding"),
                 reflect_left=1 if reflect_left else 0)
    if add is None:
        return _convt_gather(y, b, **attrs)
    return _convt_gather_add(y, b, add, **attrs)


@kernel_op("add3_scale_f32_gpu")
def _add3_scale_gpu(a, b, c, *, scale):
    _f32(a, b, c)
    _same(a, b, "add3")
    _same(a, c, "add3")
    return TensorType(f32, a.shape)


def add3_scale_f32_gpu(a, b, c, scale):
    """(a + b + c) * scale, any shape."""
    return _add3_scale_gpu(a, b, c, scale=_float(scale, "scale"))


@kernel_op("lc_to_cf_f32_gpu")
def _lc_to_cf(x, *, channels):
    _f32(x, rank=2)
    check(_ge(x.shape[1], channels), "channels exceeds the padded width")
    return TensorType(f32, [channels, x.shape[0]])


def lc_to_cf_f32_gpu(x, channels):
    """First `channels` columns of x f32[L, Cpad] as channel-first f32[channels, L]."""
    return _lc_to_cf(x, channels=_int(channels, "channels"))


@kernel_op("istft_logits_lc_f32_gpu")
def _istft_lc(x, *, hop, n_fft):
    _f32(x, rank=2)
    check(n_fft >= 2 and n_fft % 2 == 0 and hop >= 1, "n_fft must be even, hop >= 1")
    check(_ge(x.shape[1], n_fft + 2), "x needs n_fft + 2 columns")
    check(_ge(x.shape[0], 2), "need at least 2 frames")
    return TensorType(f32, [(x.shape[0] - 1) * hop])


def istft_logits_lc_f32_gpu(x, n_fft=20, hop=5):
    """istft_logits_f32_cpu straight from channel-last logits x f32[L, Cpad] (columns 0..n_fft+1 used) on the
    GPU -> f32[hop*(L-1)]."""
    return _istft_lc(x, n_fft=_int(n_fft, "n_fft"), hop=_int(hop, "hop"))


# --- decoder helpers --------------------------------------------------------------------------------------


@kernel_op("pad_cols_lc_f32_gpu")
def _pad_cols(a, *, cout):
    _f32(a, rank=2)
    check(_ge(cout, 1), "cout")
    return TensorType(f32, [a.shape[0], cout])


@kernel_op("concat_pad_lc_f32_gpu")
def _concat_pad(a, b, *, cout):
    _f32(a, b, rank=2)
    check(a.shape[0] == b.shape[0], "row counts differ")
    return TensorType(f32, [a.shape[0], cout])


def concat_pad_lc_f32_gpu(a, b=None, cout=None):
    """[a | b | 0 ...] per row: channel concat of f32[L, Ca] and f32[L, Cb], zero-padded to `cout` channels
    (default: the next multiple of 16, for the steel conv loader)."""
    ca = as_dim(a.shape[1])
    cb = as_dim(b.shape[1]) if b is not None else as_dim(0)
    check(ca.is_const and cb.is_const, "channel counts must be static")
    total = ca.value + cb.value
    cout = -(-total // 16) * 16 if cout is None else _int(cout, "cout")
    check(cout >= total, "cout smaller than the concatenation")
    return _pad_cols(a, cout=cout) if b is None else _concat_pad(a, b, cout=cout)


@kernel_op("dwconvt_lc_f32_gpu")
def _dwconvt(x, w, b, *, padding, stride):
    _f32(x, w, rank=2)
    _f32(b, rank=1)
    check(w.shape[0] == x.shape[1] and b.shape[0] == x.shape[1], "w must be f32[C, K], b f32[C]")
    k = as_dim(w.shape[1])
    check(k.is_const, "kernel size must be static")
    return TensorType(f32, [(x.shape[0] - 1) * stride - 2 * padding + k.value + (stride - 1), x.shape[1]])


def dwconvt_lc_f32_gpu(x, w, b, stride=2, padding=1):
    """Depthwise ConvTranspose1d with output_padding = stride - 1 (AdainResBlk1d's pool), channel-last."""
    return _dwconvt(x, w, b, padding=_int(padding, "padding"), stride=_int(stride, "stride"))


@kernel_op("upsample2_lc_f32_gpu")
def upsample2_lc_f32_gpu(x):
    """Nearest x2 along L: f32[L, C] -> f32[2L, C]."""
    _f32(x, rank=2)
    return TensorType(f32, [2 * x.shape[0], x.shape[1]])


# --- token rate (ALBERT on the GPU; kernels/gpu/kokoro_tok.metal) -------------------------------------------


def _attention_qkv_rule(qkv, bias, *, heads):
    _f32(qkv, rank=2)
    _f32(bias, rank=1)
    d3 = as_dim(qkv.shape[1])
    check(d3.is_const and d3.value == 3 * 64 * heads, "qkv must be f32[T, 3D] with head_dim = 64")
    check(bias.shape[0] == qkv.shape[1], "bias must be f32[3D]")
    return TensorType(f32, [qkv.shape[0], d3.value // 3])


_attention_qkv = kernel_op("attention_qkv_f32_gpu")(
    lambda qkv, bias, *, heads: _attention_qkv_rule(qkv, bias, heads=heads))
_attention_qkv_causal = kernel_op("attention_qkv_causal_f32_gpu")(
    lambda qkv, bias, *, heads: _attention_qkv_rule(qkv, bias, heads=heads))


def attention_qkv_f32_gpu(qkv, bias, heads, causal=False):
    """SDPA over a fused QKV projection: qkv f32[T, 3D] (q | k | v, before bias) + bias f32[3D] -> f32[T, D];
    scale 1/sqrt(D/heads); causal = GPT-2's mask."""
    return (_attention_qkv_causal if causal else _attention_qkv)(qkv, bias, heads=_int(heads, "heads"))


@kernel_op("bias_gelu_f32_gpu")
def bias_gelu_f32_gpu(x, bias):
    """gelu_tanh(x + bias[col]) for x f32[R, C]."""
    _f32(x, rank=2)
    _f32(bias, rank=1)
    check(bias.shape[0] == x.shape[1], "bias must be f32[C]")
    return TensorType(f32, x.shape)


@kernel_op("add_bias_layernorm_f32_gpu")
def _add_bias_ln(a, bias, r, gamma, beta, *, eps):
    _f32(a, r, rank=2)
    _f32(bias, gamma, beta, rank=1)
    _same(a, r, "residual")
    c = as_dim(a.shape[1])
    check(c.is_const and c.value <= 2048, "C must be static and <= 2048")
    check(all(v.shape[0] == a.shape[1] for v in (bias, gamma, beta)), "bias/gamma/beta must be f32[C]")
    return TensorType(f32, a.shape)


def add_bias_layernorm_f32_gpu(a, bias, r, gamma, beta, eps=1e-12):
    """LayerNorm(a + bias + r) * gamma + beta over the last dim of f32[R, C]."""
    return _add_bias_ln(a, bias, r, gamma, beta, eps=_float(eps, "eps"))
