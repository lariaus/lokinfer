"""Kokoro-82M CPU kernels (docs/impl-notes/kokoro-port-plan.md, "Basic CPU kernel set").

Layouts: token-rate tensors are row-major `[T, C]`, frame-rate tensors channel-first `[C, L]`.
Float attrs are coerced to float and int attrs to int by the public wrappers (the IR attribute type
decides whether a value lands in the kernel's `ints` or `floats`)."""
import os

from ..dims import as_dim, ceildiv
from ..dtypes import f32, i64
from ..types import ScalarType, TensorType
from ._def import check, kernel_op


def _f32(*xs, rank=None):
    for x in xs:
        check(isinstance(x.type, TensorType) and x.dtype is f32, "expected f32 tensor operands")
        if rank is not None:
            check(x.rank == rank, f"expected rank-{rank} operands, got {x.type}")


def _float(v, name):
    check(isinstance(v, (int, float)) and not isinstance(v, bool), f"{name} must be a number")
    return float(v)


def _int(v, name):
    check(isinstance(v, int) and not isinstance(v, bool), f"{name} must be an int")
    return int(v)


def _ge(d, n) -> bool:
    """True unless `d` is a constant below n (symbolic dims are checked at run time by the caller)."""
    d = as_dim(d)
    return not d.is_const or d.value >= n


# --- token-rate ---------------------------------------------------------------------------------


@kernel_op("embedding_f32_cpu")
def embedding_f32_cpu(table, ids):
    """out[t] = table[ids[t]]: table f32[V, D], ids i64[T] -> f32[T, D]."""
    _f32(table, rank=2)
    check(isinstance(ids.type, TensorType) and ids.dtype is i64 and ids.rank == 1, "ids must be i64[T]")
    return TensorType(f32, [ids.shape[0], table.shape[1]])


@kernel_op("layernorm_f32_cpu")
def _layernorm(x, gamma, beta, *, eps):
    _f32(x, rank=2)
    _f32(gamma, beta, rank=1)
    check(gamma.shape[0] == x.shape[1] and beta.shape[0] == x.shape[1], "gamma/beta must be f32[C]")
    return TensorType(f32, x.shape)


def layernorm_f32_cpu(x, gamma, beta, eps=1e-5):
    """F.layer_norm(x, (C,), gamma, beta, eps) over x f32[R, C]."""
    return _layernorm(x, gamma, beta, eps=_float(eps, "eps"))


@kernel_op("adalayernorm_f32_cpu")
def _adalayernorm(x, gb, *, eps):
    _f32(x, rank=2)
    _f32(gb, rank=1)
    check(gb.shape[0] == x.shape[1] * 2, "gb must be f32[2C]")
    return TensorType(f32, x.shape)


def adalayernorm_f32_cpu(x, gb, eps=1e-5):
    """(1 + gb[:C]) * LN_noaffine(x) + gb[C:] over x f32[R, C]."""
    return _adalayernorm(x, gb, eps=_float(eps, "eps"))


@kernel_op("attention_f32_cpu")
def _attention(q, k, v, *, heads):
    _f32(q, k, v, rank=2)
    check(q.shape == k.shape and q.shape == v.shape, "q, k, v must have the same shape")
    d = as_dim(q.shape[1])
    check(not d.is_const or d.value % heads == 0, "channels must be divisible by heads")
    return TensorType(f32, q.shape)


def attention_f32_cpu(q, k, v, heads):
    """Non-causal SDPA per head, no mask, scale 1/sqrt(D/heads); q/k/v f32[T, D]."""
    return _attention(q, k, v, heads=_int(heads, "heads"))


@kernel_op("lstm_bidir_f32_cpu")
def lstm_bidir_f32_cpu(x, w_ih, w_hh, b):
    """1-layer bidirectional nn.LSTM (gates i,f,g,o). x f32[T, I], w_ih f32[8H, I] (rows 0..4H forward),
    w_hh f32[8H, H], b f32[8H] (= b_ih + b_hh) -> f32[T, 2H] = [h_fwd, h_bwd]."""
    _f32(x, w_ih, w_hh, rank=2)
    _f32(b, rank=1)
    h = w_hh.shape[1]
    check(w_ih.shape[1] == x.shape[1], "w_ih must be [8H, I]")
    check(w_ih.shape[0] == h * 8 and w_hh.shape[0] == h * 8 and b.shape[0] == h * 8, "expected 8H gate rows")
    return TensorType(f32, [x.shape[0], h * 2])


@kernel_op("lstm_bidir2_f32_cpu")
def lstm_bidir2_f32_cpu(x1, w_ih1, w_hh1, b1, x2, w_ih2, w_hh2, b2):
    """Two independent lstm_bidir_f32_cpu in one kernel: the 4 direction recurrences run concurrently (one core each).
    Same results as two separate calls. -> (f32[T1, 2H1], f32[T2, 2H2])."""
    return (lstm_bidir_f32_cpu.__wrapped__(x1, w_ih1, w_hh1, b1), lstm_bidir_f32_cpu.__wrapped__(x2, w_ih2, w_hh2, b2))


@kernel_op("lstm_bidir_t_f32_cpu")
def lstm_bidir_t_f32_cpu(x, w_ih, w_hh_t, b):
    """lstm_bidir_f32_cpu with w_hh_t = w_hh^T f32[H, 8H] (columns 0..4H forward): the layout of the SME2 step GEMV
    (the 4H gates accumulate in ZA, one multi-vector FMLA per 64 gates and hidden unit)."""
    _f32(x, w_ih, w_hh_t, rank=2)
    _f32(b, rank=1)
    h = w_hh_t.shape[0]
    check(w_ih.shape[1] == x.shape[1], "w_ih must be [8H, I]")
    check(w_ih.shape[0] == h * 8 and w_hh_t.shape[1] == h * 8 and b.shape[0] == h * 8, "expected 8H gate rows")
    return TensorType(f32, [x.shape[0], h * 2])


@kernel_op("durations_f32_cpu")
def durations_f32_cpu(x, speed):
    """dur = clamp(round(sigmoid(x).sum(-1) / speed), min=1) (half to even), total = sum(dur).
    x f32[T, N], speed f32 scalar -> (i64[T], i64[1])."""
    _f32(x, rank=2)
    check(isinstance(speed.type, ScalarType) and speed.dtype is f32, "speed must be an f32 scalar")
    return TensorType(i64, [x.shape[0]]), TensorType(i64, [1])


@kernel_op("repeat_interleave_f32_cpu")
def _repeat_interleave(x, dur, *, frames):
    _f32(x, rank=2)
    check(isinstance(dur.type, TensorType) and dur.dtype is i64 and dur.rank == 1, "dur must be i64[T]")
    check(dur.shape[0] == x.shape[0], "dur must have one entry per row of x")
    return TensorType(f32, [frames, x.shape[1]])


def repeat_interleave_f32_cpu(x, dur, frames):
    """x.repeat_interleave(dur, dim=0) -> f32[frames, C]; `frames` (usually from `dim_read`) must equal sum(dur)."""
    return _repeat_interleave(x, dur, frames=as_dim(frames))


@kernel_op("broadcast_rows_f32_cpu")
def _broadcast_rows(v, *, rows):
    _f32(v, rank=1)
    return TensorType(f32, [rows, v.shape[0]])


def broadcast_rows_f32_cpu(v, rows):
    """v.expand(rows, C): f32[C] -> f32[rows, C]."""
    return _broadcast_rows(v, rows=as_dim(rows))


@kernel_op("concat_rows_f32_cpu")
def concat_rows_f32_cpu(*xs):
    """torch.cat(xs, dim=0) for 2..8 f32[R_i, C]."""
    check(2 <= len(xs) <= 8, "concat takes 2..8 operands")
    _f32(*xs, rank=2)
    check(all(x.shape[1] == xs[0].shape[1] for x in xs), "column counts differ")
    r = as_dim(0)
    for x in xs:
        r = r + x.shape[0]
    return TensorType(f32, [r, xs[0].shape[1]])


@kernel_op("concat_cols_f32_cpu")
def concat_cols_f32_cpu(*xs):
    """torch.cat(xs, dim=1) for 2..8 f32[R, C_i]."""
    check(2 <= len(xs) <= 8, "concat takes 2..8 operands")
    _f32(*xs, rank=2)
    check(all(x.shape[0] == xs[0].shape[0] for x in xs), "row counts differ")
    c = as_dim(0)
    for x in xs:
        c = c + x.shape[1]
    return TensorType(f32, [xs[0].shape[0], c])


# --- elementwise --------------------------------------------------------------------------------


@kernel_op("gelu_tanh_f32_cpu")
def gelu_tanh_f32_cpu(x):
    """F.gelu(x, approximate='tanh'), any shape."""
    _f32(x)
    return TensorType(f32, x.shape)


@kernel_op("leaky_relu_f32_cpu")
def _leaky_relu(x, *, slope):
    _f32(x)
    return TensorType(f32, x.shape)


def leaky_relu_f32_cpu(x, slope):
    """F.leaky_relu(x, slope), any shape."""
    return _leaky_relu(x, slope=_float(slope, "slope"))


@kernel_op("scale_f32_cpu")
def _scale(x, *, scale):
    _f32(x)
    return TensorType(f32, x.shape)


def scale_f32_cpu(x, scale):
    """x * scale, any shape."""
    return _scale(x, scale=_float(scale, "scale"))


@kernel_op("add3_scale_f32_cpu")
def _add3_scale(a, b, c, *, scale):
    _f32(a, b, c)
    check(a.shape == b.shape and a.shape == c.shape, f"shape mismatch: {a.shape}, {b.shape}, {c.shape}")
    return TensorType(f32, a.shape)


def add3_scale_f32_cpu(a, b, c, scale):
    """(a + b + c) * scale, any shape."""
    return _add3_scale(a, b, c, scale=_float(scale, "scale"))


# --- frame-rate ([C, L]) ------------------------------------------------------------------------


def _conv_out(l, k, stride, padding, dilation):
    return (l + 2 * padding - dilation * (k - 1) - 1) // stride + 1


def _conv_rule(x, w, *, dilation, padding, stride):
    _f32(x, rank=2)
    _f32(w, rank=3)
    check(w.shape[1] == x.shape[0], f"w must be [Co, Ci, K]: x {x.shape}, w {w.shape}")
    check(stride >= 1 and dilation >= 1 and padding >= 0, "invalid stride/padding/dilation")
    k = as_dim(w.shape[2])
    check(k.is_const, "kernel size must be static")
    lo = _conv_out(x.shape[1], k.value, stride, padding, dilation)
    check(_ge(lo, 1), "empty conv output")
    return TensorType(f32, [w.shape[0], lo])


@kernel_op("conv1d_f32_cpu")
def _conv1d(x, w, b, *, dilation, padding, stride):
    t = _conv_rule(x, w, dilation=dilation, padding=padding, stride=stride)
    _f32(b, rank=1)
    check(b.shape[0] == w.shape[0], "bias must be f32[Co]")
    return t


@kernel_op("conv1d_nobias_f32_cpu")
def _conv1d_nobias(x, w, *, dilation, padding, stride):
    return _conv_rule(x, w, dilation=dilation, padding=padding, stride=stride)


def conv1d_f32_cpu(x, w, b, stride=1, padding=0, dilation=1):
    """F.conv1d: x f32[Ci, L], w f32[Co, Ci, K], b f32[Co] -> f32[Co, Lo]."""
    return _conv1d(x, w, b, stride=_int(stride, "stride"), padding=_int(padding, "padding"),
                   dilation=_int(dilation, "dilation"))


def conv1d_nobias_f32_cpu(x, w, stride=1, padding=0, dilation=1):
    """F.conv1d without bias."""
    return _conv1d_nobias(x, w, stride=_int(stride, "stride"), padding=_int(padding, "padding"),
                          dilation=_int(dilation, "dilation"))


def _convt_out(l, k, stride, padding, output_padding):
    return (l - 1) * stride - 2 * padding + k + output_padding


@kernel_op("conv_transpose1d_f32_cpu")
def _conv_transpose1d(x, w, b, *, output_padding, padding, stride):
    _f32(x, rank=2)
    _f32(w, rank=3)
    _f32(b, rank=1)
    check(w.shape[0] == x.shape[0], f"w must be [Ci, Co, K]: x {x.shape}, w {w.shape}")
    check(b.shape[0] == w.shape[1], "bias must be f32[Co]")
    k = as_dim(w.shape[2])
    check(k.is_const, "kernel size must be static")
    check(stride >= 1 and padding >= 0 and 0 <= output_padding < stride, "invalid stride/padding/output_padding")
    lo = _convt_out(x.shape[1], k.value, stride, padding, output_padding)
    check(_ge(lo, 1), "empty conv_transpose output")
    return TensorType(f32, [w.shape[1], lo])


def conv_transpose1d_f32_cpu(x, w, b, stride=1, padding=0, output_padding=0):
    """F.conv_transpose1d: x f32[Ci, L], w f32[Ci, Co, K], b f32[Co] -> f32[Co, (L-1)s - 2p + K + op]."""
    return _conv_transpose1d(x, w, b, stride=_int(stride, "stride"), padding=_int(padding, "padding"),
                             output_padding=_int(output_padding, "output_padding"))


def convt_poly_weights_cpu(w, stride: int, padding: int):
    """torch ConvTranspose1d weight f32[Ci, Co, K] (K = taps·s, 2p = (taps-1)·s) -> the polyphase panels of
    conv_transpose1d_poly_f32_cpu: f32[P, Ci·taps, 32], rows (r, co) of the phases with q_r = (r + p) // s = 0, then
    q_r = 1, each group in 32-row panels; entry [row, c·taps + tt] = w[c, co, (r + p) % s + (taps-1-tt)·s]."""
    import numpy as np

    ci, co, k = w.shape
    s, p = int(stride), int(padding)
    taps = k // s
    assert taps * s == k and 2 * p == (taps - 1) * s and 0 <= p < s, "needs K = taps*s and 2p = (taps-1)*s"
    panels = []
    for rs in (range(0, s - p), range(s - p, s)):
        rows = []
        for r in rs:
            rp = (r + p) % s
            # [co, c, tt] = w[c, co, rp + (taps-1-tt) s]
            idx = [rp + (taps - 1 - tt) * s for tt in range(taps)]
            rows.append(np.transpose(w[:, :, idx], (1, 0, 2)).reshape(co, ci * taps))
        if not rows:
            continue
        g = np.concatenate(rows, 0)  # [rows_q, Ci*taps]
        npan = -(-g.shape[0] // 32)
        pad = np.zeros((npan * 32, ci * taps), np.float32)
        pad[: g.shape[0]] = g
        panels.append(pad.reshape(npan, 32, ci * taps).transpose(0, 2, 1))
    return np.ascontiguousarray(np.concatenate(panels, 0).astype(np.float32))


@kernel_op("conv_transpose1d_poly_f32_cpu")
def _conv_transpose1d_poly(x, wp, b, res, *, pad_left, padding, stride, taps, slope):
    _f32(x, res, rank=2)
    _f32(wp, rank=3)
    _f32(b, rank=1)
    check(wp.shape[1] == x.shape[0] * taps and wp.shape[2] == 32, "wp must come from convt_poly_weights_cpu")
    check(2 * padding == (taps - 1) * stride and 0 <= padding < stride, "needs 2p = (taps-1)*s")
    check(pad_left in (0, 1), "pad_left must be 0 or 1")
    lo = x.shape[1] * stride + pad_left
    check(res.shape[0] == b.shape[0] and as_dim(res.shape[1]) == as_dim(lo), "residual must be f32[Co, L*s + pad_left]")
    return TensorType(f32, [b.shape[0], lo])


@kernel_op("conv_transpose1d_poly_mean3_f32_cpu")
def _conv_transpose1d_poly_mean3(x, x2, x3, wp, b, res, *, in_scale, pad_left, padding, slope, stride, taps):
    _f32(x2, x3, rank=2)
    check(all(as_dim(a) == as_dim(c) for y in (x2, x3) for a, c in zip(y.shape, x.shape)), "x2 / x3 must match x")
    return _conv_transpose1d_poly.__wrapped__(x, wp, b, res, pad_left=pad_left, padding=padding, stride=stride,
                                              taps=taps, slope=slope)


def conv_transpose1d_poly_f32_cpu(x, wp, b, res, stride, padding, taps, slope, pad_left=0, mean3=None):
    """reflect_pad_left(conv_transpose1d(leaky_relu(x, slope), w, b, stride, padding), pad_left) + res, polyphase
    (wp = convt_poly_weights_cpu(w, stride, padding), K = taps * stride): x f32[Ci, L] -> f32[Co, L*s + pad_left]."""
    at = dict(pad_left=_int(pad_left, "pad_left"), padding=_int(padding, "padding"), stride=_int(stride, "stride"),
              taps=_int(taps, "taps"), slope=_float(slope, "slope"))
    if mean3 is not None:  # (x2, x3, scale): the input is ((x + x2) + x3) * scale (add3_scale_f32_cpu fused)
        x2, x3, sc = mean3
        return _conv_transpose1d_poly_mean3(x, x2, x3, wp, b, res, in_scale=_float(sc, "in_scale"), **at)
    return _conv_transpose1d_poly(x, wp, b, res, **at)


@kernel_op("conv_transpose1d_dw_f32_cpu")
def _conv_transpose1d_dw(x, w, b, *, output_padding, padding, stride):
    _f32(x, rank=2)
    _f32(w, rank=3)
    _f32(b, rank=1)
    check(w.shape[0] == x.shape[0] and w.shape[1] == 1, "w must be [C, 1, K]")
    check(b.shape[0] == x.shape[0], "bias must be f32[C]")
    k = as_dim(w.shape[2])
    check(k.is_const, "kernel size must be static")
    check(stride >= 1 and padding >= 0 and 0 <= output_padding < stride, "invalid stride/padding/output_padding")
    lo = _convt_out(x.shape[1], k.value, stride, padding, output_padding)
    check(_ge(lo, 1), "empty conv_transpose output")
    return TensorType(f32, [x.shape[0], lo])


def conv_transpose1d_dw_f32_cpu(x, w, b, stride=1, padding=0, output_padding=0):
    """Depthwise F.conv_transpose1d(groups=C): x f32[C, L], w f32[C, 1, K], b f32[C]."""
    return _conv_transpose1d_dw(x, w, b, stride=_int(stride, "stride"), padding=_int(padding, "padding"),
                                output_padding=_int(output_padding, "output_padding"))


@kernel_op("adain1d_f32_cpu")
def _adain1d(x, gb, *, eps):
    _f32(x, rank=2)
    _f32(gb, rank=1)
    check(gb.shape[0] == x.shape[0] * 2, "gb must be f32[2C]")
    return TensorType(f32, x.shape)


def adain1d_f32_cpu(x, gb, eps=1e-5):
    """(1 + gb[:C, None]) * instance_norm(x) + gb[C:, None] (biased var) over x f32[C, L]."""
    return _adain1d(x, gb, eps=_float(eps, "eps"))


@kernel_op("snake_f32_cpu")
def snake_f32_cpu(x, alpha):
    """x + (1/alpha) * sin(alpha * x)^2, alpha f32[C] per channel of x f32[C, L]."""
    _f32(x, rank=2)
    _f32(alpha, rank=1)
    check(alpha.shape[0] == x.shape[0], "alpha must be f32[C]")
    return TensorType(f32, x.shape)


@kernel_op("upsample_nearest1d_f32_cpu")
def _upsample_nearest1d(x, *, scale):
    _f32(x, rank=2)
    check(scale >= 1, "scale must be >= 1")
    return TensorType(f32, [x.shape[0], x.shape[1] * scale])


def upsample_nearest1d_f32_cpu(x, scale):
    """F.interpolate(mode='nearest', scale_factor=scale) (integer scale): f32[C, L] -> f32[C, L*scale]."""
    return _upsample_nearest1d(x, scale=_int(scale, "scale"))


@kernel_op("pad_reflect1d_f32_cpu")
def _pad_reflect1d(x, *, left, right):
    _f32(x, rank=2)
    check(left >= 0 and right >= 0, "negative padding")
    check(_ge(x.shape[1], max(left, right) + 1), "reflect padding must be < L")
    return TensorType(f32, [x.shape[0], x.shape[1] + (left + right)])


def pad_reflect1d_f32_cpu(x, left, right):
    """F.pad(x, (left, right), mode='reflect') over f32[C, L] (left, right < L)."""
    return _pad_reflect1d(x, left=_int(left, "left"), right=_int(right, "right"))


@kernel_op("harmonic_source_f32_cpu")
def harmonic_source_f32_cpu(f0, noise, w, b):
    """Kokoro SourceModuleHnNSF (SineGen x300, 9 harmonics, Linear 9->1, tanh) in torch's op order.
    f0 f32[n] (80 Hz curve), noise f32[300n, 9] (the randn_like draw), w f32[9], b f32[1] -> f32[300n]."""
    _f32(f0, w, b, rank=1)
    _f32(noise, rank=2)
    s = f0.shape[0] * 300
    check(noise.shape[0] == s and noise.shape[1] == 9, f"noise must be f32[300n, 9], got {noise.shape}")
    check(w.shape[0] == 9 and b.shape[0] == 1, "w must be f32[9], b f32[1]")
    return TensorType(f32, [s])


@kernel_op("harmonic_source_rng_f32_cpu", nohoist=True)
def harmonic_source_rng_f32_cpu(f0, w, b, seed):
    """Same as harmonic_source_f32_cpu, with the noise generated in-kernel from `seed` (an i64 scalar input),
    like torch's randn_like after manual_seed (Philox4x32-10 + Box-Muller; deterministic per seed)."""
    _f32(f0, w, b, rank=1)
    check(isinstance(seed.type, ScalarType) and seed.dtype is i64, "seed must be an i64 scalar")
    check(w.shape[0] == 9 and b.shape[0] == 1, "w must be f32[9], b f32[1]")
    return TensorType(f32, [f0.shape[0] * 300])


@kernel_op("stft_magphase_f32_cpu")
def _stft_magphase(x, *, hop, n_fft):
    _f32(x, rank=1)
    check(n_fft >= 2 and n_fft % 2 == 0 and hop >= 1, "n_fft must be even, hop >= 1")
    check(_ge(x.shape[0], n_fft // 2 + 1), "signal must be longer than n_fft/2 (reflect pad)")
    return TensorType(f32, [n_fft + 2, x.shape[0] // hop + 1])


def stft_magphase_f32_cpu(x, n_fft=20, hop=5):
    """torch.stft(center, reflect, periodic hann(n_fft), onesided) -> cat(abs, angle): f32[S] -> f32[n_fft+2, S//hop+1]."""
    return _stft_magphase(x, n_fft=_int(n_fft, "n_fft"), hop=_int(hop, "hop"))


@kernel_op("istft_logits_f32_cpu")
def _istft_logits(x, *, hop, n_fft):
    _f32(x, rank=2)
    check(n_fft >= 2 and n_fft % 2 == 0 and hop >= 1, "n_fft must be even, hop >= 1")
    check(n_fft <= 126, "n_fft must be <= 126 (the kernel keeps n_fft/2 + 1 <= 64 bins on the stack)")
    check(x.shape[0] == n_fft + 2, f"x must be f32[n_fft+2, L], got {x.shape}")
    check(_ge(x.shape[1], 2), "need at least 2 frames")
    return TensorType(f32, [(x.shape[1] - 1) * hop])


def istft_logits_f32_cpu(x, n_fft=20, hop=5):
    """torch.istft(exp(x[:K]) * e^{i sin(x[K:])}, periodic hann, center), K = n_fft/2+1:
    f32[n_fft+2, L] -> f32[hop*(L-1)]."""
    return _istft_logits(x, n_fft=_int(n_fft, "n_fft"), hop=_int(hop, "hop"))


# --- fused frame-rate conv (M3, kernels/cpu/kokoro_conv_fused.cpp) ------------------------------

STAT_CHUNK = 128  # samples per (mean, M2) stats partial (kernels' kStatChunk); conv tiles are multiples of it
COL_BUDGET = 3 << 20  # im2col block bytes per task (3 MiB; M3 tile sweep on the M2 Max: best tile ≈ 3 MB block in 6 of 7 shapes)


def conv_tile(ci: int, k: int) -> int:
    """Default time tile for a fused conv: the largest multiple of STAT_CHUNK whose im2col block
    [Ci*K, tile] fits COL_BUDGET, clamped to [128, 768] (sweep: 768 best or within 3 % of best for every Kokoro
    shape; 1024-2048 never helps, even at k3 where the block is small)."""
    t = COL_BUDGET // (4 * ci * k) // STAT_CHUNK * STAT_CHUNK
    return max(STAT_CHUNK, min(768, t))


def _stats_type(c, l):
    return TensorType(f32, [c, ceildiv(l, STAT_CHUNK), 2])


def _fused_rule(x, w, b, affine, extra, *, dilation, padding, tile):
    t = _conv_rule(x, w, dilation=dilation, padding=padding, stride=1)
    _f32(b, *extra, rank=1)
    _f32(affine, rank=2)
    check(b.shape[0] == w.shape[0], "bias must be f32[Co]")
    check(affine.shape[0] == 3 and affine.shape[1] == x.shape[0], "affine must be f32[3, Ci]")
    check(tile >= STAT_CHUNK and tile % STAT_CHUNK == 0, f"tile must be a positive multiple of {STAT_CHUNK}")
    return t, _stats_type(t.shape[0], t.shape[1])


@kernel_op("conv1d_snake_f32_cpu")
def _conv1d_snake(x, w, b, affine, alpha, *, dilation, padding, tile):
    check(alpha.shape[0] == x.shape[0], "alpha must be f32[Ci]")
    return _fused_rule(x, w, b, affine, [alpha], dilation=dilation, padding=padding, tile=tile)


@kernel_op("conv1d_snake_res_f32_cpu")
def _conv1d_snake_res(x, w, b, affine, alpha, res, *, dilation, padding, tile):
    check(alpha.shape[0] == x.shape[0], "alpha must be f32[Ci]")
    out = _fused_rule(x, w, b, affine, [alpha], dilation=dilation, padding=padding, tile=tile)
    _f32(res, rank=2)
    check(res.shape[0] == out[0].shape[0] and res.shape[1] == out[0].shape[1], "residual must match the output")
    return out


@kernel_op("conv1d_leaky_f32_cpu")
def _conv1d_leaky(x, w, b, affine, *, dilation, padding, slope, tile):
    return _fused_rule(x, w, b, affine, [], dilation=dilation, padding=padding, tile=tile)


@kernel_op("conv1d_leaky_res_f32_cpu")
def _conv1d_leaky_res(x, w, b, affine, res, *, dilation, oscale, padding, slope, tile):
    out = _fused_rule(x, w, b, affine, [], dilation=dilation, padding=padding, tile=tile)
    _f32(res, rank=2)
    check(res.shape[0] == out[0].shape[0] and res.shape[1] == out[0].shape[1], "residual must match the output")
    return out


@kernel_op("conv1d_affine_f32_cpu")
def _conv1d_affine(x, w, b, affine, *, dilation, padding, tile):
    return _fused_rule(x, w, b, affine, [], dilation=dilation, padding=padding, tile=tile)


def _fused_attrs(x, w, padding, dilation, tile):
    if tile is None:
        ci, k = as_dim(x.shape[0]), as_dim(w.shape[2])
        tile = conv_tile(int(ci.value), int(k.value)) if ci.is_const and k.is_const else 512
    return dict(padding=_int(padding, "padding"), dilation=_int(dilation, "dilation"), tile=_int(tile, "tile"))


def conv1d_snake_f32_cpu(x, w, b, affine, alpha, residual=None, padding=0, dilation=1, tile=None):
    """conv1d(snake((x - mean) * scale + beta, alpha), w) + b [+ residual] -> (out f32[Co, Lo], stats f32[Co, T, 2]).
    affine = f32[3, Ci] rows (mean, scale, beta), e.g. from adain_affine_f32_cpu."""
    if residual is None:
        return _conv1d_snake(x, w, b, affine, alpha, **_fused_attrs(x, w, padding, dilation, tile))
    return _conv1d_snake_res(x, w, b, affine, alpha, residual, **_fused_attrs(x, w, padding, dilation, tile))


def conv1d_leaky_f32_cpu(x, w, b, affine, slope=0.1, residual=None, out_scale=1.0, padding=0, dilation=1,
                         tile=None):
    """(conv1d(leaky_relu((x - mean) * scale + beta, slope), w) + b [+ residual]) * out_scale -> (out, stats)."""
    attrs = _fused_attrs(x, w, padding, dilation, tile)
    if residual is None:
        check(out_scale == 1.0, "out_scale needs a residual")
        return _conv1d_leaky(x, w, b, affine, slope=_float(slope, "slope"), **attrs)
    return _conv1d_leaky_res(x, w, b, affine, residual, slope=_float(slope, "slope"),
                             oscale=_float(out_scale, "out_scale"), **attrs)


def conv1d_affine_f32_cpu(x, w, b, affine, padding=0, dilation=1, tile=None):
    """conv1d((x - mean) * scale + beta, w) + b -> (out, stats)."""
    return _conv1d_affine(x, w, b, affine, **_fused_attrs(x, w, padding, dilation, tile))


# --- Winograd snake conv (CPU; F(4,3) for K = 3, nested F(4,4) for K = 4..12, as ops/winograd.py) ------------

def wino_weights_cpu(w):
    """torch conv weight f32[Co, Ci, K] -> (U f32[A, ceil(Co/32), Ci*G, 32], tr f32[A*A + 4A]): ops.winograd's U
    repacked for the CPU kernel (reduction index c*G + g, 32 output channels innermost; zero channels past Co)."""
    import numpy as np

    from .winograd import variant, wino_weights
    co, ci, k = w.shape
    u, tr = wino_weights(w)
    a, (_, g) = u.shape[0], variant(k)
    u = u.reshape(a, g, ci, co).transpose(0, 2, 1, 3).reshape(a, ci * g, co)
    cb = -(-co // 32)
    up = np.zeros((a, ci * g, cb * 32), np.float32)
    up[:, :, :co] = u
    return np.ascontiguousarray(up.reshape(a, ci * g, cb, 32).transpose(0, 2, 1, 3)), tr


def wino_tile(ci: int) -> int:
    """Time tile of the Winograd conv: 512 up to Ci = 256, shrinking for wider inputs (iPhone sweep 2026-11-01 with the
    P/E-split pipeline: 512 1-3 % faster than 1024; Mac (no SME2) 512 ~ 1024). The kernel switches to one tile in
    two parallel phases when the sequence has fewer than 2 tiles per pool thread."""
    return max(STAT_CHUNK, min(512, (1 << 17) // ci // STAT_CHUNK * STAT_CHUNK))


def _wino_rule(x, u, tr, b, affine, extra, *, dilation, groups, kernel, padding, tile):
    _f32(x, affine, rank=2)
    _f32(tr, b, *extra, rank=1)
    _f32(u, rank=4)
    a = 6 if kernel == 3 else 7
    check(u.shape[0] == a and u.shape[2] == x.shape[0] * groups and u.shape[3] == 32, "u must come from wino_weights_cpu")
    check(u.shape[1] == ceildiv(b.shape[0], 32), "u must come from wino_weights_cpu (Co blocks)")
    check(groups == _wino_groups(kernel), "groups don't match the kernel size")
    check(tr.shape[0] == a * (a + 4), "bad transform table")
    check(affine.shape[0] == 3 and affine.shape[1] == x.shape[0], "affine must be f32[3, Ci]")
    for e in extra:
        check(e.shape[0] == x.shape[0], "alpha must be f32[Ci]")
    check(tile >= STAT_CHUNK and tile % STAT_CHUNK == 0, f"tile must be a positive multiple of {STAT_CHUNK}")
    lo = x.shape[1] + 2 * padding - dilation * (kernel - 1)
    check(_ge(lo, 1), "empty conv output")
    co = b.shape[0]
    return TensorType(f32, [co, lo]), _stats_type(co, lo)


def _wino_res(out, res):
    _f32(res, rank=2)
    check(res.shape[0] == out[0].shape[0] and res.shape[1] == out[0].shape[1], "residual must match the output")
    return out


def _wino_groups(kernel):
    return 1 if kernel == 3 else max(2, -(-kernel // 4))


@kernel_op("conv1d_snake_wino_f32_cpu")
def _conv1d_snake_wino(x, u, tr, b, affine, alpha, **at):
    return _wino_rule(x, u, tr, b, affine, [alpha], **at)


@kernel_op("conv1d_snake_wino_res_f32_cpu")
def _conv1d_snake_wino_res(x, u, tr, b, affine, alpha, res, **at):
    return _wino_res(_wino_rule(x, u, tr, b, affine, [alpha], **at), res)


@kernel_op("conv1d_leaky_wino_f32_cpu")
def _conv1d_leaky_wino(x, u, tr, b, affine, *, slope, **at):
    return _wino_rule(x, u, tr, b, affine, [], **at)


@kernel_op("conv1d_leaky_wino_res_f32_cpu")
def _conv1d_leaky_wino_res(x, u, tr, b, affine, res, *, oscale, slope, **at):
    return _wino_res(_wino_rule(x, u, tr, b, affine, [], **at), res)


@kernel_op("conv1d_leaky_wino_mean3_f32_cpu")
def _conv1d_leaky_wino_mean3(x, x2, x3, u, tr, b, affine, *, in_scale, slope, **at):
    _f32(x2, x3, rank=2)
    check(all(as_dim(a) == as_dim(c) for y in (x2, x3) for a, c in zip(y.shape, x.shape)), "x2 / x3 must match x")
    return _wino_rule(x, u, tr, b, affine, [], **at)


def conv1d_leaky_wino_mean3_f32_cpu(x, x2, x3, u, tr, b, affine, kernel, in_scale, slope, padding=0, dilation=1,
                                    tile=None):
    """conv1d_wino_f32_cpu(act="leaky") on ((x + x2) + x3) * in_scale (add3_scale_f32_cpu fused into the input window:
    the generator's resblock mean feeding conv_post) -> (out, stats)."""
    kernel = _int(kernel, "kernel")
    if tile is None:
        ci = as_dim(x.shape[0])
        tile = wino_tile(int(ci.value)) if ci.is_const else 1024
    tile = int(tile) + (int(os.environ.get("LKX_WINO_MODE", "0")) << 20)  # EXPERIMENT A/B knob (temporary)
    return _conv1d_leaky_wino_mean3(x, x2, x3, u, tr, b, affine, in_scale=_float(in_scale, "in_scale"),
                                    slope=_float(slope, "slope"), dilation=_int(dilation, "dilation"),
                                    groups=_wino_groups(kernel), kernel=kernel, padding=_int(padding, "padding"),
                                    tile=_int(tile, "tile"))


@kernel_op("conv1d_affine_wino_f32_cpu")
def _conv1d_affine_wino(x, u, tr, b, affine, **at):
    return _wino_rule(x, u, tr, b, affine, [], **at)


def conv1d_wino_f32_cpu(x, u, tr, b, affine, kernel, act="snake", alpha=None, slope=0.1, residual=None, out_scale=1.0,
                        padding=0, dilation=1, tile=None):
    """The fused conv1d_{snake,leaky,affine}_f32_cpu (act = "snake" with alpha, "leaky" with slope [and out_scale],
    "none") as a Winograd conv: (u, tr) = wino_weights_cpu(w), kernel = K; same result, ~half the multiplies
    (F(4,3) for K = 3, nested F(4,4) for K = 4..12). -> (out f32[Co, Lo], stats)."""
    kernel = _int(kernel, "kernel")
    if tile is None:
        ci = as_dim(x.shape[0])
        tile = wino_tile(int(ci.value)) if ci.is_const else 1024
    tile = int(tile) + (int(os.environ.get("LKX_WINO_MODE", "0")) << 20)  # EXPERIMENT A/B knob (temporary)
    at = dict(dilation=_int(dilation, "dilation"), groups=_wino_groups(kernel), kernel=kernel,
              padding=_int(padding, "padding"), tile=_int(tile, "tile"))
    if act == "snake":
        check(alpha is not None and out_scale == 1.0, "snake needs alpha (and no out_scale)")
        if residual is None:
            return _conv1d_snake_wino(x, u, tr, b, affine, alpha, **at)
        return _conv1d_snake_wino_res(x, u, tr, b, affine, alpha, residual, **at)
    if act == "leaky":
        if residual is None:
            check(out_scale == 1.0, "out_scale needs a residual")
            return _conv1d_leaky_wino(x, u, tr, b, affine, slope=_float(slope, "slope"), **at)
        return _conv1d_leaky_wino_res(x, u, tr, b, affine, residual, oscale=_float(out_scale, "out_scale"),
                                      slope=_float(slope, "slope"), **at)
    check(act == "none" and residual is None, "act must be snake, leaky or none (none: no residual)")
    return _conv1d_affine_wino(x, u, tr, b, affine, **at)


def conv1d_snake_wino_f32_cpu(x, u, tr, b, affine, alpha, kernel, residual=None, padding=0, dilation=1, tile=None):
    """conv1d_wino_f32_cpu with act = "snake"."""
    return conv1d_wino_f32_cpu(x, u, tr, b, affine, kernel, alpha=alpha, residual=residual, padding=padding,
                               dilation=dilation, tile=tile)


@kernel_op("channel_stats_f32_cpu")
def channel_stats_f32_cpu(x):
    """Per-(channel, 128-sample chunk) (mean, M2) of x f32[C, L] -> f32[C, ceil(L / 128), 2]."""
    _f32(x, rank=2)
    return _stats_type(x.shape[0], x.shape[1])


@kernel_op("adain_affine_f32_cpu")
def _adain_affine(stats, gb, *, eps, length):
    _f32(stats, rank=3)
    _f32(gb, rank=1)
    c = stats.shape[0]
    check(gb.shape[0] == 2 * c, "gb must be f32[2C]")
    check(stats.shape[1] == ceildiv(length, STAT_CHUNK), "stats chunks don't match length")
    return TensorType(f32, [3, c])


def adain_affine_f32_cpu(stats, gb, length, eps=1e-5):
    """AdaIN from chunk stats over `length` samples -> affine f32[3, C] = (mean, (1 + g) / std, b)."""
    return _adain_affine(stats, gb, eps=_float(eps, "eps"), length=as_dim(length))
