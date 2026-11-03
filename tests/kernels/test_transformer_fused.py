"""GPT-2 fused / dispatching kernels vs float64 numpy: the CPU Accelerate GEMM with fused epilogues in every blocking
mode (gemm_accel_f32_cpu, incl. the small-M K-split mode), the CPU add+bias+LayerNorm pair, the
causal blocked CPU attention, the GPU split-K partials + their LayerNorm consumer, and the GPU generate ops that pick
the decode or prefill kernels from the runtime M / T."""
import numpy as np
import pytest

import lokinfer as lk
from lokinfer.ops import transformer as X


def C(a):
    return lk.from_numpy(np.ascontiguousarray(a, np.float32))


def G(a):
    return lk.from_numpy(np.ascontiguousarray(a, np.float32), device="gpu", copy=True)


def gelu(v):
    return 0.5 * v * (1 + np.tanh(0.7978845608028654 * (v + 0.044715 * v ** 3)))


def ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    return (x - m) / np.sqrt(((x - m) ** 2).mean(-1, keepdims=True) + eps) * g + b


@pytest.mark.parametrize("m,k,n", [(1, 768, 2304), (3, 3072, 768), (37, 100, 65), (128, 768, 3072)])
@pytest.mark.parametrize("split", [0, 1, 2, -2, -4, 6000, 5012])
def test_gemm_accel_cpu(m, k, n, split):
    r = np.random.default_rng(m + k + n)
    a, b = r.standard_normal((m, k)).astype(np.float32), (r.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
    bias = r.standard_normal(n).astype(np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    tol = dict(atol=2e-4 * max(1, np.abs(ref).max()), rtol=1e-4)
    np.testing.assert_allclose(X.gemm_accel_f32_cpu(C(a), C(b), split=split).numpy(), ref, **tol)
    np.testing.assert_allclose(X.gemm_accel_f32_cpu(C(a), C(b), C(bias), split=split).numpy(), ref + bias, **tol)
    for act, f in (("gelu", gelu), ("gelu_vv", gelu), ("relu", lambda v: np.maximum(v, 0))):
        np.testing.assert_allclose(X.gemm_accel_f32_cpu(C(a), C(b), C(bias), act=act, split=split).numpy(),
                                   f(ref + bias), **tol)


@pytest.mark.parametrize("rows,c", [(1, 768), (7, 100), (300, 768)])
def test_add_bias_layernorm2_cpu(rows, c):
    r = np.random.default_rng(rows)
    o, res = r.standard_normal((rows, c)).astype(np.float32), r.standard_normal((rows, c)).astype(np.float32)
    bias, g, b = (r.standard_normal(c).astype(np.float32) for _ in range(3))
    s, out = X.add_bias_layernorm2_f32_cpu(C(o), C(bias), C(res), C(g), C(b))
    ref = o.astype(np.float64) + bias + res
    np.testing.assert_allclose(s.numpy(), ref, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(out.numpy(), ln(ref, g, b), atol=1e-4, rtol=1e-4)


def _attn_ref(qkv, heads):
    t, d3 = qkv.shape
    d = d3 // 3
    hd = d // heads
    q, k, v = (qkv[:, i * d:(i + 1) * d].astype(np.float64) for i in range(3))
    out = np.zeros((t, d))
    for h in range(heads):
        sl = slice(h * hd, (h + 1) * hd)
        s = q[:, sl] @ k[:, sl].T / np.sqrt(hd)
        s[np.triu_indices(t, 1)] = -np.inf
        p = np.exp(s - s.max(-1, keepdims=True))
        out[:, sl] = (p / p.sum(-1, keepdims=True)) @ v[:, sl]
    return out


@pytest.mark.parametrize("t", [1, 5, 64, 129, 300])
@pytest.mark.parametrize("qb,mode", [(64, 0), (32, 1), (128, 0)])
def test_attention_causal_blk_cpu(t, qb, mode):
    qkv = np.random.default_rng(t).standard_normal((t, 3 * 768)).astype(np.float32)
    got = X.attention_qkv_causal_blk_f32_cpu(C(qkv), 12, qb=qb, mode=mode).numpy()
    np.testing.assert_allclose(got, _attn_ref(qkv, 12), atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("m,k", [(128, 768), (77, 3072), (512, 3072), (1, 768)])
@pytest.mark.parametrize("splits,tile", [(2, 32), (4, 64), (1, 32)])
def test_gemm_splitk_ln2_gpu(m, k, splits, tile):
    n = 768
    r = np.random.default_rng(m + k)
    a, w = r.standard_normal((m, k)).astype(np.float32), (r.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
    res = r.standard_normal((m, n)).astype(np.float32)
    bias, g, b = (r.standard_normal(n).astype(np.float32) for _ in range(3))
    parts = X.gemm_splitk_f32_gpu(G(a), G(w), splits, tile=tile)
    s, out = X.add_bias_layernorm2_sk_f32_gpu(parts, G(bias), G(res), G(g), G(b))
    ref = a.astype(np.float64) @ w + bias + res
    np.testing.assert_allclose(s.numpy(), ref, atol=2e-4 * np.abs(ref).max(), rtol=1e-4)
    np.testing.assert_allclose(out.numpy(), ln(ref, g, b), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("m", [1, 2, 128])
def test_gemm_auto_gpu(m):
    k, n = 768, 3072
    r = np.random.default_rng(m)
    a, w = r.standard_normal((m, k)).astype(np.float32), (r.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
    bias = r.standard_normal(n).astype(np.float32)
    ref = a.astype(np.float64) @ w
    tol = dict(atol=2e-4 * max(1, np.abs(ref).max()), rtol=1e-4)
    np.testing.assert_allclose(X.gemm_auto_f32_gpu(G(a), G(w)).numpy(), ref, **tol)
    np.testing.assert_allclose(X.gemm_auto_bias_gelu_f32_gpu(G(a), G(w), G(bias)).numpy(), gelu(ref + bias), **tol)


def _cache_model(kind, heads=12, **kw):
    """Compiled graph: caches (persistent) <- hist rows at pos 0 (kv_write_rows), then the new chunk's attention at
    pos: kind "cache_sg" (kv_write_rows + attention_cache_sg) or "split" (attention_decode_split, writes its own row).
    Returns (out, k cache, v cache)."""
    D = 64 * heads
    P, T = lk.Sym("P"), (1 if kind == "split" else lk.Sym("T"))

    def fn(hist, zero, qkv, pos, bias):
        hist, qkv, bias = lk.ops.to_gpu(hist), lk.ops.to_gpu(qkv), lk.ops.to_gpu(bias)
        kc = lk.persistent_buffer(lk.f32, [1024, D], device="gpu")
        vc = lk.persistent_buffer(lk.f32, [1024, D], device="gpu")
        X.kv_write_rows_f32_gpu(kc, vc, hist, zero, bias)
        if kind == "split":
            out = X.attention_decode_split_f32_gpu(qkv, kc, vc, pos, bias, heads, **kw)
        else:
            X.kv_write_rows_f32_gpu(kc, vc, qkv, pos, bias)
            out = X.attention_cache_sg_f32_gpu(qkv, kc, vc, pos, bias, heads, **kw)
        return lk.ops.to_cpu(out)

    return lk.compile(fn, input_types=[lk.TensorType(lk.f32, [P, 3 * D]), lk.TensorType(lk.i64, [1]),
                                       lk.TensorType(lk.f32, [T, 3 * D]), lk.TensorType(lk.i64, [1]),
                                       lk.TensorType(lk.f32, [3 * D])])


def _cache_attn_ref(hist, qkv, bias, heads):
    """Query i of qkv (position len(hist) + i) over keys 0 .. len(hist) + i of [hist; qkv] (+ bias), float64."""
    d = qkv.shape[1] // 3
    full = np.concatenate([hist, qkv]).astype(np.float64) + bias
    q = qkv.astype(np.float64)[:, :d] + bias[:d]
    k, v = full[:, d:2 * d], full[:, 2 * d:]
    p0, t, hd = len(hist), len(qkv), d // heads
    out = np.zeros((t, d))
    for h in range(heads):
        sl = slice(h * hd, (h + 1) * hd)
        s = q[:, sl] @ k[:, sl].T / np.sqrt(hd)
        s[np.arange(p0 + t)[None, :] > p0 + np.arange(t)[:, None]] = -np.inf
        pr = np.exp(s - s.max(-1, keepdims=True))
        out[:, sl] = (pr / pr.sum(-1, keepdims=True)) @ v[:, sl]
    return out


def _cache_inputs(p0, t, seed):
    r = np.random.default_rng(seed)
    hist = r.standard_normal((max(p0, 1), 3 * 768)).astype(np.float32)[:p0]
    qkv = r.standard_normal((t, 3 * 768)).astype(np.float32)
    bias = r.standard_normal(3 * 768).astype(np.float32)
    return hist, qkv, bias


@pytest.mark.parametrize("p0,t", [(6, 4), (5, 1), (100, 33), (31, 64), (384, 128), (700, 300), (1, 1023)])
@pytest.mark.parametrize("nsg", [2, 4, 8])
def test_attention_cache_sg_gpu(p0, t, nsg):
    """Later prefill chunks (pos > 0) on the simdgroup-matrix flash kernel reading K/V from the cache."""
    hist, qkv, bias = _cache_inputs(p0, t, p0 * 7 + t)
    m = _cache_model("cache_sg", nsg=nsg)
    h_in = hist if p0 else np.zeros((1, 3 * 768), np.float32)  # P >= 1; written at pos 0, then overwritten
    got = m(C(h_in), lk.from_numpy(np.array([0], np.int64)), C(qkv), lk.from_numpy(np.array([p0], np.int64)),
            C(bias)).numpy()
    np.testing.assert_allclose(got, _cache_attn_ref(hist, qkv, bias, 12), atol=3e-5, rtol=1e-4)


@pytest.mark.parametrize("p0", [0, 1, 37, 512, 1023])
@pytest.mark.parametrize("splits,nsg", [(0, 0), (1, 32), (3, 4), (8, 8), (16, 16), (32, 8)])
@pytest.mark.parametrize("single", [0, 1])
def test_attention_decode_split_gpu(p0, splits, nsg, single):
    """Decode attention with the keys split over threadgroups + combine (fused write of the step's own k/v row)."""
    hist, qkv, bias = _cache_inputs(p0, 1, p0 + 11)
    m = _cache_model("split", splits=splits, nsg=nsg, single=single)
    h_in = hist if p0 else np.zeros((1, 3 * 768), np.float32)
    exp = _cache_attn_ref(hist, qkv, bias, 12)
    for _ in range(3):  # repeated calls: the single-dispatch counters must reset
        got = m(C(h_in), lk.from_numpy(np.array([0], np.int64)), C(qkv), lk.from_numpy(np.array([p0], np.int64)),
                C(bias)).numpy()
        np.testing.assert_allclose(got, exp, atol=3e-5, rtol=1e-4)


@pytest.mark.parametrize("m", [1, 2, 128, 512])
@pytest.mark.parametrize("k,splits", [(768, 2), (3072, 4)])
def test_gemm_auto_sk_ln2_gpu(m, k, splits):
    """build_generate's N = 768 projections: one graph, M symbolic (M = 1 -> gemv4 plane, else split-K / MPS)."""
    n = 768
    r = np.random.default_rng(m + k)
    a, w = r.standard_normal((m, k)).astype(np.float32), (r.standard_normal((k, n)) / np.sqrt(k)).astype(np.float32)
    res = r.standard_normal((m, n)).astype(np.float32)
    bias, g, b = (r.standard_normal(n).astype(np.float32) for _ in range(3))
    M_ = lk.Sym("M")

    def fn(a_, res_):
        a_, res_ = lk.ops.to_gpu(a_), lk.ops.to_gpu(res_)
        parts = X.gemm_auto_sk_f32_gpu(a_, G(w), splits)
        s, out = X.add_bias_layernorm2_sk_f32_gpu(parts, G(bias), res_, G(g), G(b))
        return lk.ops.to_cpu(s), lk.ops.to_cpu(out)

    model = lk.compile(fn, input_types=[lk.TensorType(lk.f32, [M_, k]), lk.TensorType(lk.f32, [M_, n])])
    s, out = model(C(a), C(res))
    ref = a.astype(np.float64) @ w + bias + res
    np.testing.assert_allclose(s.numpy(), ref, atol=2e-4 * np.abs(ref).max(), rtol=1e-4)
    np.testing.assert_allclose(out.numpy(), ln(ref, g, b), atol=1e-3, rtol=1e-3)
