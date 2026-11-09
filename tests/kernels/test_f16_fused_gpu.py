"""f16 optimized GPU kernels (Qwen3-ASR Q5, docs/impl-notes/qwen3-asr-f16-opt-kernels.md) vs (a) the composition of
the basic f16 GPU kernels (the oracle, docs/impl-notes/qwen3-asr-f16-kernels.md) and (b) the float64 torch
composition of the contract's semantics. Each case is one compiled graph computing the optimized op and its oracle on
the same inputs (one upload with `--device=iphone`, where the oracle runs on the phone too). Real model shapes (M = 1,
44, 401, 2048; encoder S up to 15,600; caches up to 25,600 rows at pos ~20,000) + odd sizes + length 1. Very long
cases are `@pytest.mark.slow` (`--run-slow`).

Tolerances (per element; U = 2^-11, SUB = 2^-25, the float terms are test_f16_basic_gpu's, see its docstring):
- vs float64: the optimized op rounds once, so its bound is the basic one-rounding bound U (1 + 2^-10) |ref| + SUB +
  the op's float32 term (GEMM 2^-22 sqrt(K) rss(x_k w_k); attention 2^-24 4 (8 + sqrt(keys)) max|v|; LM head
  2^-22 sqrt(D) rss + 2^-23 |ref| (float output, no rounding); qknorm_rope_kv: the basic composition's roundings
  are part of the semantics, so the reference reproduces them (r16) and the bound adds one f16 ulp of the normalized
  value h, which float vs float64 statistics may round the other way: 2^-9 (|y_j| + |y_j+64|) on the RoPE output;
  audio_cnn: one conv-layer ulp flip, 2^-9 rms(ref)).
- the oracle vs float64: the same terms plus the oracle's own intermediate roundings, each <= U |intermediate| times
  the downstream op's Lipschitz bound: gelu (|gelu'| <= 1.13) of a rounded x @ w + b; the residual add of a rounded
  x @ w (+ b) (1); clamp (1); silu_mul of rounded gate and up (|d silu(g) u / dg| <= 1.1 |u|, |silu(g)|).
- optimized vs oracle: the sum of the two bounds above (triangle inequality through the float64 value). The two are
  also checked against float64 separately, so a wrong oracle cannot hide a wrong optimized kernel.
Measured worst ratios err / tol are printed by `-s` runs (`LOKINFER_F16O_REPORT=1`).

M > 1 GEMMs run on one of two kernel families (kernels/gpu/f16_gpu.cpp): steel (simdgroup matrices; the M2 Max) or
Metal 4 tensor ops (f16_tensor.metal; GPUs with neural accelerators, the iPhone 18 Pro Max). The GEMM-family tests
(`gemm_path`) run each case on the device's own choice and, on the Mac (in-process), again with the tensor-op kernels
forced (LOKINFER_F16_GEMM=tensor, read when the model is loaded), so both families are tested on every device that
can run them.

Speedups vs the basic composition: docs/models/qwen3-asr.md, "Q5 GPU kernels" (measured with
`bench/kernels/bench_f16_fused_gpu.py`)."""
import os

import numpy as np
import pytest
import torch

import lokinfer as lk
from lokinfer.ops import f16_gpu as F
from test_f16_basic_gpu import _cnn_ref, _cnn_weights, _rope_ref, _window_ref, h16, r16, rng, t64

U = 2.0 ** -11
SUB = 2.0 ** -25
REPORT = bool(os.environ.get("LOKINFER_F16O_REPORT"))


def C(a):
    return lk.from_numpy(np.ascontiguousarray(a))


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    if isinstance(x, np.ndarray):
        return x
    return x.numpy()


def check(got, ref, tol, what):
    """|got - ref| <= tol elementwise (tol already includes every term)."""
    got = np.asarray(_np(got), np.float64)
    ref = np.asarray(_np(ref), np.float64)
    assert got.shape == ref.shape, (what, got.shape, ref.shape)
    assert np.isfinite(got).all(), f"{what}: non-finite output"
    tol = np.broadcast_to(np.asarray(tol, np.float64), got.shape)
    err = np.abs(got - ref)
    ratio = float((err / tol).max()) if err.size else 0.0
    if REPORT:
        print(f"  {what}: max err/tol {ratio:.3f}")
    bad = err > tol
    if bad.any():
        i = np.unravel_index(np.argmax(err - tol), err.shape)
        raise AssertionError(f"{what}: {bad.sum()} / {bad.size} beyond tolerance; worst at {i}: got {got[i]!r} "
                             f"ref {ref[i]!r} err {err[i]:.3g} tol {tol[i]:.3g}")


def round_tol(ref):
    """The one-rounding bound of an f16 output."""
    return U * (1 + 2.0 ** -10) * np.abs(np.asarray(_np(ref), np.float64)) + SUB


def triple(opt, orc, ref, tol_opt, tol_orc, what):
    """opt vs float64 (tol_opt), oracle vs float64 (tol_orc), opt vs oracle (their sum)."""
    check(opt, ref, tol_opt, what + " opt vs f64")
    check(orc, ref, tol_orc, what + " oracle vs f64")
    check(opt, orc, np.asarray(tol_opt) + np.asarray(tol_orc), what + " opt vs oracle")


def _type(a):
    return lk.TensorType({np.float16: lk.f16, np.float32: lk.f32, np.int64: lk.i64}[a.dtype.type], list(a.shape))


def pair(fn, *arrays):
    """Compiles fn(*gpu_inputs) -> tuple of GPU tensors as one graph (f16 / f32 inputs moved to the GPU, i64 kept on
    the CPU: the kernels read pos at encode time) and runs it once on the arrays. Returns numpy outputs."""

    def wrapped(*xs):
        gx = [x if a.dtype == np.int64 else lk.ops.to_gpu(x) for x, a in zip(xs, arrays)]
        outs = fn(*gx)
        return tuple(lk.ops.to_cpu(o) for o in outs)

    model = lk.compile(wrapped, input_types=[_type(a) for a in arrays])
    return [o.numpy() for o in model(*[C(a) for a in arrays])]


def P(p):
    return np.array([p], np.int64)


@pytest.fixture(params=["auto", "tensor"])
def gemm_path(request, monkeypatch, device_name):
    """The M > 1 GEMM kernels: the device's own choice, or (Mac in-process only: the env is read by this process's
    runtime) the tensor-op kernels forced."""
    if request.param == "tensor":
        if device_name != "mac":
            pytest.skip("forcing the GEMM kernels needs the in-process runtime (the phone runs tensor ops anyway)")
        monkeypatch.setenv("LOKINFER_F16_GEMM", "tensor")
    return request.param


# --- GEMMs ----------------------------------------------------------------------------------------------------------

GEMM_SHAPES = [
    (1, 2048, 4096), (44, 2048, 4096), (401, 2048, 2048), (1, 6144, 2048), (44, 6144, 2048),  # LM qkv / o / down
    (1, 1024, 3072), (300, 1024, 3072), (37, 4096, 1024), (2, 1024, 2048), (13, 7680, 1024),  # encoder, proj, conv_out
    (1, 1, 1), (1, 15, 1), (3, 17, 5), (2, 7, 9), (33, 31, 65), (65, 129, 67), (64, 64, 64), (100, 7, 300),
    (1, 33, 7), (1, 64, 6),  # odd sizes, tile edges, the GEMV's scalar column path (N % 4 != 0)
    (130, 1031, 260), (100, 640, 2050), (520, 1056, 1030),  # tensor-op tiles 64x64 / 64x128 / 128x128, K tails
]
GEMM_SLOW = [(2048, 2048, 4096), (2048, 6144, 2048), (15_600, 1024, 3072), (15_600, 4096, 1024)]


def _gemm_data(m, k, n, seed):
    r = rng(seed)
    x, w = h16(r, m, k), h16(r, k, n, scale=1 / np.sqrt(k))
    xd, wd = t64(x), t64(w)
    y = xd @ wd
    fterm = (2.0 ** -22 * np.sqrt(k) * torch.sqrt((xd * xd) @ (wd * wd))).numpy()
    return r, x, w, y, fterm


def _gemm_cases(fast, slow):
    return [pytest.param(*s) for s in fast] + [pytest.param(*s, marks=pytest.mark.slow) for s in slow]


@pytest.mark.parametrize("m,k,n", _gemm_cases(GEMM_SHAPES, GEMM_SLOW))
def test_gemm_opt(m, k, n, gemm_path):
    _, x, w, y, ft = _gemm_data(m, k, n, m * 7 + k + n)
    opt, orc = pair(lambda x_, w_: (F.gemm_opt_f16_gpu(x_, w_), F.gemm_f16_gpu(x_, w_)), x, w)
    tol = round_tol(y) + ft
    triple(opt, orc, y, tol, tol, "gemm_opt")


EPI_SHAPES = [(1, 2048, 2048), (44, 1024, 4096), (401, 4096, 1024), (1, 4096, 1024), (300, 1024, 1024),
              (1, 1, 1), (3, 17, 5), (33, 31, 65), (1, 33, 7)]
EPI_SLOW = [(2048, 1024, 4096), (15_600, 4096, 1024)]


@pytest.mark.parametrize("m,k,n", _gemm_cases(EPI_SHAPES, EPI_SLOW))
def test_gemm_bias_opt(m, k, n, gemm_path):
    r, x, w, y, ft = _gemm_data(m, k, n, m + 3 * k + n)
    b = h16(r, n)
    ref = y + t64(b)
    opt, orc = pair(lambda x_, w_, b_: (F.gemm_bias_opt_f16_gpu(x_, w_, b_), F.gemm_bias_f16_gpu(x_, w_, b_)), x, w, b)
    tol = round_tol(ref) + ft + 2.0 ** -23 * ref.abs().numpy()
    triple(opt, orc, ref, tol, tol, "gemm_bias_opt")


@pytest.mark.parametrize("m,k,n", _gemm_cases(EPI_SHAPES, EPI_SLOW))
def test_gemm_bias_gelu_opt(m, k, n, gemm_path):
    r, x, w, y, ft = _gemm_data(m, k, n, m + 5 * k + n)
    b = h16(r, n)
    z = y + t64(b)
    ref = torch.nn.functional.gelu(z)  # exact erf
    opt, orc = pair(lambda x_, w_, b_: (F.gemm_bias_gelu_opt_f16_gpu(x_, w_, b_),
                                        F.gelu_erf_f16_gpu(F.gemm_bias_f16_gpu(x_, w_, b_))), x, w, b)
    zt = ft + 2.0 ** -23 * z.abs().numpy()  # float error of z (GEMM + bias)
    tol_opt = round_tol(ref) + 1.13 * zt + 2.0 ** -20 * ref.abs().numpy()
    tol_orc = tol_opt + 1.13 * round_tol(z)  # the oracle rounds z before the GELU
    triple(opt, orc, ref, tol_opt, tol_orc, "gemm_bias_gelu_opt")


@pytest.mark.parametrize("m,k,n", _gemm_cases(EPI_SHAPES, EPI_SLOW))
def test_gemm_add_opt(m, k, n, gemm_path):
    r, x, w, y, ft = _gemm_data(m, k, n, m + 7 * k + n)
    res = h16(r, m, n, scale=2)
    ref = t64(res) + y
    opt, orc = pair(lambda x_, w_, r_: (F.gemm_add_opt_f16_gpu(x_, w_, r_), F.add_f16_gpu(r_, F.gemm_f16_gpu(x_, w_))),
                    x, w, res)
    tol_opt = round_tol(ref) + ft + 2.0 ** -23 * ref.abs().numpy()
    triple(opt, orc, ref, tol_opt, tol_opt + round_tol(y), "gemm_add_opt")


@pytest.mark.parametrize("m,k,n", _gemm_cases(EPI_SHAPES, EPI_SLOW))
def test_gemm_bias_add_opt(m, k, n, gemm_path):
    r, x, w, y, ft = _gemm_data(m, k, n, m + 11 * k + n)
    b, res = h16(r, n), h16(r, m, n, scale=2)
    z = y + t64(b)
    ref = t64(res) + z
    opt, orc = pair(lambda x_, w_, b_, r_: (F.gemm_bias_add_opt_f16_gpu(x_, w_, b_, r_),
                                            F.add_f16_gpu(r_, F.gemm_bias_f16_gpu(x_, w_, b_))), x, w, b, res)
    tol_opt = round_tol(ref) + ft + 2.0 ** -23 * (z.abs() + ref.abs()).numpy()
    triple(opt, orc, ref, tol_opt, tol_opt + round_tol(z), "gemm_bias_add_opt")


@pytest.mark.parametrize("m,k,n", _gemm_cases(EPI_SHAPES, EPI_SLOW))
def test_gemm_bias_add_clamp_opt(m, k, n, gemm_path):
    """Residuals near the f16 limit: sums beyond +-64504 are clamped (to f16(64504) = 64512, as torch's clamp)."""
    r, x, w, y, ft = _gemm_data(m, k, n, m + 13 * k + n)
    b = h16(r, n)
    res = np.clip(64000 * r.standard_normal((m, n)), -65000, 65000).astype(np.float16)
    z = y + t64(b)
    ref = torch.clamp(t64(res) + z, -64504, 64504)
    opt, orc = pair(lambda x_, w_, b_, r_: (F.gemm_bias_add_clamp_opt_f16_gpu(x_, w_, b_, r_, 64504.0),
                                            F.add_clamp_f16_gpu(r_, F.gemm_bias_f16_gpu(x_, w_, b_), lim=64504.0)),
                    x, w, b, res)
    tol_opt = round_tol(ref) + ft + 2.0 ** -23 * (z.abs() + (t64(res) + z).abs()).numpy()
    triple(opt, orc, ref, tol_opt, tol_opt + round_tol(z), "gemm_bias_add_clamp_opt")
    assert np.abs(opt.astype(np.float64)).max() <= 64512


SWIGLU_SHAPES = [(1, 2048, 6144), (44, 2048, 6144), (401, 2048, 6144), (1, 1, 1), (3, 17, 5), (33, 31, 65),
                 (1, 7, 3), (70, 64, 96), (100, 1056, 2050), (300, 576, 1030)]
SWIGLU_SLOW = [(2048, 2048, 6144)]


@pytest.mark.parametrize("m,k,n", _gemm_cases(SWIGLU_SHAPES, SWIGLU_SLOW))
def test_gemm_swiglu_opt(m, k, n, gemm_path):
    """w = [gate | up] f16[K, 2N]; the oracle runs two GEMMs (the halves as separate inputs) + silu_mul."""
    r = rng(m + 17 * k + n)
    x, wg, wu = h16(r, m, k, scale=2), h16(r, k, n, scale=1 / np.sqrt(k)), h16(r, k, n, scale=1 / np.sqrt(k))
    w = np.ascontiguousarray(np.concatenate([wg, wu], 1))
    xd = t64(x)
    g, u = xd @ t64(wg), xd @ t64(wu)
    ref = torch.nn.functional.silu(g) * u
    fg = (2.0 ** -22 * np.sqrt(k) * torch.sqrt((xd * xd) @ (t64(wg) ** 2))).numpy()
    fu = (2.0 ** -22 * np.sqrt(k) * torch.sqrt((xd * xd) @ (t64(wu) ** 2))).numpy()
    sg, au, ag = torch.nn.functional.silu(g).abs().numpy(), u.abs().numpy(), g.abs().numpy()
    opt, orc = pair(lambda x_, w_, wg_, wu_: (F.gemm_swiglu_opt_f16_gpu(x_, w_),
                                              F.silu_mul_f16_gpu(F.gemm_f16_gpu(x_, wg_), F.gemm_f16_gpu(x_, wu_))),
                    x, w, wg, wu)
    tol_opt = round_tol(ref) + 1.1 * au * fg + sg * fu + 2.0 ** -20 * ref.abs().numpy()
    tol_orc = tol_opt + 1.1 * au * round_tol(g) + sg * round_tol(u) + 1.1 * round_tol(g) * round_tol(u)
    triple(opt, orc, ref, tol_opt, tol_orc, "gemm_swiglu_opt")
    assert opt.shape == (m, n)


# --- LM head --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("t,d,v", [(1, 2048, 151_936), (44, 2048, 4099), (3, 64, 1000), (1, 33, 7), (2, 1, 1),
                                    (1, 8, 5)])
def test_lm_head_last_opt(t, d, v):
    r = rng(t + d + v)
    x, table = h16(r, t, d), h16(r, v, d, scale=0.05)
    opt, orc = pair(lambda x_, t_: (F.lm_head_last_opt_f16_gpu(x_, t_), F.lm_head_last_f16_gpu(x_, t_)), x, table)
    assert opt.dtype == np.float32 and opt.shape == (1, v)
    xl = t64(x[-1])
    ref, tol = np.empty((1, v)), np.empty((1, v))
    for i in range(0, v, 16384):  # float64 in blocks (the full table in float64 would be 2.5 GB)
        tb = t64(table[i:i + 16384])
        ref[0, i:i + 16384] = (tb @ xl).numpy()
        tol[0, i:i + 16384] = (2.0 ** -22 * np.sqrt(d) * torch.sqrt((tb * tb) @ (xl * xl))).numpy()
    tol = tol + 2.0 ** -23 * np.abs(ref) + 1e-30
    triple(opt, orc, ref, tol, tol, "lm_head_last_opt")


# --- QK-norm + RoPE + KV write --------------------------------------------------------------------------------------


def _qkr_ref(x, g, pos, eps=1e-6, theta=1e6):
    """rope(r16(g * r16(rmsnorm(x)))) per 128-wide head, float64; tolerance (one rounding + rope's float term + one
    ulp flip of the normalized value)."""
    t, c = x.shape
    xd = t64(x).reshape(t, -1, 128)
    hh = r16(xd * torch.rsqrt((xd * xd).mean(-1, keepdim=True) + eps))
    y = r16(t64(g) * hh).reshape(t, c)
    ref, extra = _rope_ref(y.numpy().astype(np.float16), pos, theta)
    ya = y.abs().reshape(t, -1, 2, 64)
    flip = 2.0 ** -9 * (ya[:, :, 0] + ya[:, :, 1]).repeat(1, 1, 2).reshape(t, c).numpy()
    return ref, round_tol(ref) + extra + flip


def _qkr_graph(t, qh, kvh, cap, pos):
    """One graph: qknorm_rope_kv_opt into persistent caches, the oracle (rmsnorm_heads, rope, kv_write) into another
    pair; returns q (both) and the written cache rows pos .. pos + T - 1 (both, read back by a row gather)."""
    W = kvh * 128

    def fn(qkv, qpart, kpart, vpart, qg, kg, p):
        kc_o, vc_o, kc_r, vc_r = (lk.persistent_buffer(lk.f16, [cap, W], device="gpu") for _ in range(4))
        q_o = F.qknorm_rope_kv_opt_f16_gpu(qkv, qg, kg, p, kc_o, vc_o, 1e-6, 1e6)
        q_r = F.rope_f16_gpu(F.rmsnorm_heads_f16_gpu(qpart, qg, eps=1e-6), p, theta=1e6)
        k_r = F.rope_f16_gpu(F.rmsnorm_heads_f16_gpu(kpart, kg, eps=1e-6), p, theta=1e6)
        F.kv_write_f16_gpu(kc_r, vc_r, k_r, vpart, p)
        rows = lk.ops.to_gpu(lk.ops.arange_offset_i64_cpu(p, t))
        return (q_o, q_r) + tuple(F.embedding_f16_gpu(rows, c) for c in (kc_o, kc_r, vc_o, vc_r))

    return fn


@pytest.mark.parametrize("t,qh,kvh,cap,pos", [
    (1, 16, 8, 64, 0), (44, 16, 8, 64, 0), (401, 16, 8, 2048, 1000), (2048, 16, 8, 2048, 0),
    (1, 16, 8, 25_600, 20_000), (64, 16, 8, 20_064, 19_999), (1, 16, 8, 25_600, 25_599),  # long contexts, last row
    (3, 2, 1, 10, 7), (5, 1, 1, 5, 0),  # odd head counts; T fills the cache exactly
])
def test_qknorm_rope_kv_opt(t, qh, kvh, cap, pos):
    r = rng(t + qh + cap + pos)
    qkv = h16(r, t, (qh + 2 * kvh) * 128, scale=3)
    qkv[0, :128] *= np.float16(0.01)  # a head of a very different scale
    qg, kg = h16(r, 128, scale=0.5, mean=1), h16(r, 128, scale=0.5, mean=1)
    qp, kp, vp = (np.ascontiguousarray(a) for a in np.split(qkv, [qh * 128, (qh + kvh) * 128], 1))
    q_o, q_r, k_o, k_r, v_o, v_r = pair(_qkr_graph(t, qh, kvh, cap, pos), qkv, qp, kp, vp, qg, kg, P(pos))
    ref, tol = _qkr_ref(qp, qg, pos)
    triple(q_o, q_r, ref, tol, tol, "qknorm_rope_kv_opt q")
    ref, tol = _qkr_ref(kp, kg, pos)
    triple(k_o, k_r, ref, tol, tol, "qknorm_rope_kv_opt k (cache rows)")
    np.testing.assert_array_equal(v_o.view(np.uint16), vp.view(np.uint16))
    np.testing.assert_array_equal(v_r.view(np.uint16), vp.view(np.uint16))


def test_qknorm_rope_kv_opt_rejects_overflow():
    cap = 16

    def fn(qkv, qg, kg, p):
        kc, vc = (lk.persistent_buffer(lk.f16, [cap, 1024], device="gpu") for _ in range(2))
        return lk.ops.to_cpu(F.qknorm_rope_kv_opt_f16_gpu(lk.ops.to_gpu(qkv), lk.ops.to_gpu(qg), lk.ops.to_gpu(kg), p,
                                                          kc, vc))

    r = rng(3)
    qkv, g = h16(r, 4, 4096), h16(r, 128)
    model = lk.compile(fn, input_types=[_type(qkv), _type(g), _type(g), lk.TensorType(lk.i64, [1])])
    model(C(qkv), C(g), C(g), C(P(12)))  # ends exactly at the cache end: fine
    for p in (13, -1):
        with pytest.raises(RuntimeError, match="exceeds the cache"):
            model(C(qkv), C(g), C(g), C(P(p)))


# --- attention ------------------------------------------------------------------------------------------------------


def _attn_tol(ref, v, keys):
    return round_tol(ref) + 2.0 ** -24 * 4 * (8 + np.sqrt(keys)) * float(np.abs(v.astype(np.float64)).max())


def _gqa_ref(q, kc, vc, pos, q_heads, kv_heads):
    """float64 causal GQA, query rows in blocks of 256 (bounded memory at T = 2048, 22k keys)."""
    t, cq = q.shape
    hd = cq // q_heads
    keys = pos + t
    kd = t64(kc[:keys]).reshape(keys, kv_heads, hd)
    vd = t64(vc[:keys]).reshape(keys, kv_heads, hd)
    qd = t64(q).reshape(t, q_heads, hd)
    out = torch.empty(t, q_heads, hd, dtype=torch.float64)
    for i0 in range(0, t, 256):
        i1 = min(t, i0 + 256)
        mask = torch.arange(keys)[None, :] > (pos + torch.arange(i0, i1))[:, None]
        for h in range(q_heads):
            g = h // (q_heads // kv_heads)
            sc = (qd[i0:i1, h] @ kd[:, g].T) / np.sqrt(hd)
            sc[mask] = -torch.inf
            out[i0:i1, h] = torch.softmax(sc, -1) @ vd[:, g]
    return out.reshape(t, cq)


def _gqa_data(cap, pos, t, q_heads, kv_heads, hd, seed):
    r = rng(seed)
    q = h16(r, t, q_heads * hd, scale=2)
    kc = np.full((cap, kv_heads * hd), np.nan, np.float16)  # rows past pos + T: never read (NaN would show)
    vc = np.full((cap, kv_heads * hd), np.nan, np.float16)
    kc[:pos + t], vc[:pos + t] = h16(r, pos + t, kv_heads * hd, scale=2), h16(r, pos + t, kv_heads * hd)
    return q, kc, vc


GQA_CASES = [
    (64, 0, 1, 16, 8, 128), (64, 0, 44, 16, 8, 128), (300, 7, 33, 16, 8, 128), (64, 63, 1, 16, 8, 128),
    (2048, 0, 2048, 16, 8, 128), (4196, 100, 2048, 16, 8, 128), (2048, 1000, 401, 16, 8, 128),  # prefill chunks
    (25_600, 20_000, 1, 16, 8, 128), (25_600, 20_000, 64, 16, 8, 128), (25_600, 25_599, 1, 16, 8, 128),  # long
    (25_600, 25_536, 64, 16, 8, 128),  # the chunk ends at the cache's last row
    (1000, 700, 1, 16, 8, 128), (300, 255, 1, 16, 8, 128), (300, 256, 1, 16, 8, 128),  # split boundaries (256 keys)
    (40, 5, 3, 4, 1, 64), (40, 0, 9, 2, 2, 128), (17, 16, 1, 6, 3, 128), (33, 32, 1, 3, 1, 64),  # odd: G 4 / 1 / 2 / 3
    (100, 50, 1, 4, 1, 128), (40, 30, 2, 8, 8, 64), (12, 0, 12, 2, 1, 64),
]
GQA_SLOW = [(22_048, 20_000, 2048, 16, 8, 128), (25_600, 13_570, 2048, 16, 8, 128)]


@pytest.mark.parametrize("cap,pos,t,q_heads,kv_heads,hd",
                         [pytest.param(*c) for c in GQA_CASES] + [pytest.param(*c, marks=pytest.mark.slow)
                                                                  for c in GQA_SLOW])
def test_attention_gqa_opt(cap, pos, t, q_heads, kv_heads, hd):
    q, kc, vc = _gqa_data(cap, pos, t, q_heads, kv_heads, hd, cap + pos + t + hd)
    if t * (pos + t) <= 2 ** 24:
        opt, orc = pair(lambda q_, k_, v_, p_: (F.attention_gqa_opt_f16_gpu(q_, k_, v_, p_, q_heads, kv_heads),
                                                F.attention_gqa_f16_gpu(q_, k_, v_, p_, q_heads, kv_heads)),
                        q, kc, vc, P(pos))
    else:
        # The basic kernel is one dispatch of > 1 s here, which intermittently comes back as zeros (a GPU watchdog
        # kill the runtime does not report): the oracle runs on chunks of 256 query rows (rows i0.. at pos + i0, the
        # same semantics).
        (opt,) = pair(lambda q_, k_, v_, p_: (F.attention_gqa_opt_f16_gpu(q_, k_, v_, p_, q_heads, kv_heads),),
                      q, kc, vc, P(pos))
        orc = np.concatenate([pair(lambda q_, k_, v_, p_: (F.attention_gqa_f16_gpu(q_, k_, v_, p_, q_heads, kv_heads),),
                                   np.ascontiguousarray(q[i0:i0 + 256]), kc, vc, P(pos + i0))[0]
                              for i0 in range(0, t, 256)])
    ref = _gqa_ref(q, kc, vc, pos, q_heads, kv_heads)
    tol = _attn_tol(ref, vc[:pos + t], pos + t)
    triple(opt, orc, ref, tol, tol, "attention_gqa_opt")


def test_attention_gqa_opt_rejects_overflow():
    q, kc, vc = _gqa_data(16, 0, 4, 16, 8, 128, 1)
    model = lk.compile(lambda q_, k_, v_, p_: lk.ops.to_cpu(F.attention_gqa_opt_f16_gpu(
        lk.ops.to_gpu(q_), lk.ops.to_gpu(k_), lk.ops.to_gpu(v_), p_)),
        input_types=[lk.TensorType(lk.f16, [lk.Sym("T"), 2048]), _type(kc), _type(vc), lk.TensorType(lk.i64, [1])])
    model(C(q), C(kc), C(vc), C(P(12)))
    for pos in (13, -1):
        with pytest.raises(RuntimeError, match="exceeds the cache"):
            model(C(q), C(kc), C(vc), C(P(pos)))
    model(C(q[:1]), C(kc), C(vc), C(P(15)))  # the decode path: the last row
    for pos in (16, -1):
        with pytest.raises(RuntimeError, match="exceeds the cache"):
            model(C(q[:1]), C(kc), C(vc), C(P(pos)))


WINDOW_CASES = [
    (1, 16, 1024, 104), (5, 16, 1024, 104), (104, 16, 1024, 104), (105, 16, 1024, 104), (207, 16, 1024, 104),
    (208, 16, 1024, 104), (209, 16, 1024, 104), (1000, 16, 1024, 104), (15_600, 16, 1024, 104),  # real N_a 1200 s
    (9, 2, 128, 1), (33, 2, 128, 7), (20, 4, 512, 8), (300, 2, 128, 250), (50, 1, 64, 50),  # hd 128, windows 1..250
]


@pytest.mark.parametrize("s,heads,c,window", WINDOW_CASES)
def test_attention_window_qkv_opt(s, heads, c, window):
    r = rng(s + heads + window)
    q, k, v = h16(r, s, c, scale=2), h16(r, s, c, scale=2), h16(r, s, c)
    qkv = np.ascontiguousarray(np.concatenate([q, k, v], 1))
    opt, orc = pair(lambda a_, q_, k_, v_: (F.attention_window_qkv_opt_f16_gpu(a_, heads, window),
                                            F.attention_window_f16_gpu(q_, k_, v_, heads=heads, window=window)),
                    qkv, q, k, v)
    ref = _window_ref(q, k, v, heads, window)
    tol = _attn_tol(ref, v, min(window, s))
    triple(opt, orc, ref, tol, tol, "attention_window_qkv_opt")


def test_prefill_then_decode_mutable_cache():
    """The generate graph's pattern on the opt ops with a caller-owned mutable cache (decision 12): kv f16[2, P, W]
    (k and v caches are views of it) written by qknorm_rope_kv_opt, read by attention_gqa_opt; prefill chunks then
    decode steps through one compiled graph (T symbolic), each step vs the basic composition on the same caches."""
    from lokinfer.graph import current_graph
    from lokinfer.ops._def import _as_value

    cap, qh, kvh = 96, 16, 8
    W = kvh * 128
    T = lk.Sym("T")

    def fn(qkv, qg, kg, pos, kv):
        g = current_graph()
        kc = g.view(_as_value(g, kv), lk.MutableType(lk.f16, [cap, W]), 0)
        vc = g.view(_as_value(g, kv), lk.MutableType(lk.f16, [cap, W]), cap * W)
        q = F.qknorm_rope_kv_opt_f16_gpu(lk.ops.to_gpu(qkv), lk.ops.to_gpu(qg), lk.ops.to_gpu(kg), pos, kc, vc)
        return (lk.ops.to_cpu(F.attention_gqa_opt_f16_gpu(q, kc, vc, pos, qh, kvh)),
                lk.ops.to_cpu(F.attention_gqa_f16_gpu(q, kc, vc, pos, qh, kvh)), lk.ops.to_cpu(q))

    model = lk.compile(fn, input_types=[lk.TensorType(lk.f16, [T, (qh + 2 * kvh) * 128]),
                                        lk.TensorType(lk.f16, [128]), lk.TensorType(lk.f16, [128]),
                                        lk.TensorType(lk.i64, [1]), lk.MutableType(lk.f16, [2, cap, W])])
    r = rng(3)
    qg, kg = h16(r, 128, scale=0.5, mean=1), h16(r, 128, scale=0.5, mean=1)
    kv = np.zeros((2, cap, W), np.float16)
    kv_t = lk.from_numpy(kv)  # the caller's buffer: written in place by every call
    pos = 0
    for t in (40, 33, 1, 1, 1):
        qkv = h16(r, t, (qh + 2 * kvh) * 128, scale=3)
        opt, orc, q = (o.numpy() for o in model(C(qkv), C(qg), C(kg), C(P(pos)), kv_t))
        kvn = kv_t.numpy()
        ref = _gqa_ref(q, kvn[0], kvn[1], pos, qh, kvh)
        tol = _attn_tol(ref, kvn[1][:pos + t], pos + t)
        triple(opt, orc, ref, tol, tol, f"prefill/decode at {pos}")
        kref, ktol = _qkr_ref(np.ascontiguousarray(qkv[:, qh * 128:(qh + kvh) * 128]), kg, pos)
        check(kvn[0][pos:pos + t], kref, ktol, "cache k rows")
        np.testing.assert_array_equal(kvn[1][pos:pos + t].view(np.uint16), qkv[:, (qh + kvh) * 128:].view(np.uint16))
        pos += t


# --- audio_cnn ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,co,d,f,tm,chunks", [
    (1, 480, 1024, 128, 100, None), (3, 480, 1024, 128, 100, None),  # real shapes: [n, 128, 100] -> [n, 13, 1024]
    (9, 480, 1024, 128, 100, [0, 3, 4, 7, 8]),  # several batches (8 per batch on the Mac, 4 on the iPhone) + a tail
    pytest.param(300, 480, 1024, 128, 100, [0, 1, 150, 299], marks=pytest.mark.slow),  # iPhone capacity
    (2, 8, 7, 9, 5, None), (1, 4, 3, 1, 1, None), (4, 32, 40, 17, 12, None), (5, 12, 9, 6, 30, None),  # odd sizes
])
def test_audio_cnn_opt(n, co, d, f, tm, chunks, gemm_path):
    ws, bs, w_out = _cnn_weights(co, d, f, tm, n + co + f)
    features = (0.8 * rng(n + tm).standard_normal((n, f, tm))).astype(np.float32)
    wts = (ws[0], bs[0], ws[1], bs[1], ws[2], bs[2], w_out)
    if chunks is None:
        chunks = list(range(n))
        opt, orc = pair(lambda *a: (F.audio_cnn_opt_f16_gpu(*a), F.audio_cnn_f16_gpu(*a)), features, *wts)
    else:
        # Chunks are independent: the oracle runs on the checked chunks only (the basic kernel over all 300 chunks
        # after the opt op is a command buffer long enough to come back as zeros on the iPhone, the GPU watchdog).
        (opt,) = pair(lambda *a: (F.audio_cnn_opt_f16_gpu(*a),), features, *wts)
        (orc,) = pair(lambda *a: (F.audio_cnn_f16_gpu(*a),), np.ascontiguousarray(features[chunks]), *wts)
        opt = opt[chunks]
    ref, extra = _cnn_ref(features, ws, bs, w_out, chunks)
    rms = float(torch.sqrt((ref * ref).mean()))
    tol = round_tol(ref) + extra + 2.0 ** -9 * rms
    triple(opt, orc, ref, tol, tol, "audio_cnn_opt")


def test_audio_cnn_opt_repeated_calls():
    """The per-op scratch is reused across calls and batches: calls with different n give independent results."""
    co, d, f, tm = 16, 24, 20, 14
    ws, bs, w_out = _cnn_weights(co, d, f, tm, 7)
    consts = [lk.from_numpy(a, device="gpu", copy=True) for a in (ws[0], bs[0], ws[1], bs[1], ws[2], bs[2], w_out)]
    model = lk.compile(lambda x: lk.ops.to_cpu(F.audio_cnn_opt_f16_gpu(lk.ops.to_gpu(x), *consts)),
                       input_types=[lk.TensorType(lk.f32, [lk.Sym("n"), f, tm])])
    r = rng(8)
    for n in (11, 2, 7):
        feats = (0.8 * r.standard_normal((n, f, tm))).astype(np.float32)
        ref, extra = _cnn_ref(feats, ws, bs, w_out, range(n))
        rms = float(torch.sqrt((ref * ref).mean()))
        check(model(C(feats)), ref, round_tol(ref) + extra + 2.0 ** -9 * rms, f"audio_cnn_opt n={n}")
