"""Fused Winograd / Toom-Cook 1-D conv on the GPU, channel-last (kernels/gpu/winograd.metal,
docs/impl-notes/winograd-conv.md).

Same math as `conv1d_lc_f32_gpu` (stride 1, any dilation with padding % dilation == 0) with ~half the
multiplications: K = 3 uses F(4, 3) (6 points), K = 4..12 a nested F(4, 4) (7 points, ceil(K / 4) tap groups at
offsets 0, 4, 8 that share one transformed input). The weights are transformed on the host once
(`wino_weights`): U = G w, f32[A, n * Ci, Co], plus the transform table `tr` = [BT (A x A) | AT (4 x A)].
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..dims import as_dim, ceildiv
from ..dtypes import f32
from ..types import TensorType
from ._def import check, kernel_op
from .kokoro import _f32, _float, _ge, _int

TILE = 4  # outputs per Winograd tile (both variants)
_POINTS = {  # finite interpolation points (+ infinity); F(4,3): the classic set, F(4,4): -1/2 measured best
    3: [0, 1, -1, 2, -2],
    4: [0, 1, -1, 2, -2, Fraction(-1, 2)],
}
MAX_GROUPS = 3  # kernel's NMAX for the nested variant


def _vandermonde(points, n):
    rows = [[Fraction(p) ** j for j in range(n)] for p in points]
    rows.append([Fraction(0)] * (n - 1) + [Fraction(1)])  # the point at infinity: leading coefficient
    return rows


def _inverse(mat):
    """Exact Gauss-Jordan inverse over Fractions."""
    n = len(mat)
    a = [list(row) + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(mat)]
    for c in range(n):
        piv = next(r for r in range(c, n) if a[r][c] != 0)
        a[c], a[piv] = a[piv], a[c]
        inv = 1 / a[c][c]
        a[c] = [v * inv for v in a[c]]
        for r in range(n):
            if r != c and a[r][c] != 0:
                f = a[r][c]
                a[r] = [v - f * w for v, w in zip(a[r], a[c])]
    return [row[n:] for row in a]


def transforms(m: int, r: int, points=None):
    """Exact correlation F(m, r): y = AT ((G g) * (BT d)); BT rows scaled to max |.| = 1 (G absorbs the scale)."""
    pts = _POINTS[r] if points is None else points
    alpha = m + r - 1
    assert len(pts) == alpha - 1
    va, vb = _vandermonde(pts, m), _vandermonde(pts, r)
    c = _inverse(_vandermonde(pts, alpha))  # conv c = C ((Va a) * (Vb b))
    bt = [[c[j][i] for j in range(alpha)] for i in range(alpha)]  # C^T
    g = [list(row) for row in vb]
    at = [[va[j][i] for j in range(alpha)] for i in range(m)]  # Va^T
    for i in range(alpha):
        s = max(abs(v) for v in bt[i])
        bt[i] = [v / s for v in bt[i]]
        g[i] = [v * s for v in g[i]]
    f = lambda mat: np.array([[float(v) for v in row] for row in mat], dtype=np.float64)  # noqa: E731
    return f(at), f(g), f(bt)


def variant(k: int) -> tuple[int, int]:
    """(inner r, groups n) for kernel size k."""
    if k == 3:
        return 3, 1
    check(4 <= k <= TILE * MAX_GROUPS, f"winograd conv: kernel size {k} unsupported (3..12)")
    return TILE, max(2, -(-k // TILE))  # kernels exist for n = 2, 3 (K = 4 pads a zero group)


def wino_weights(w: np.ndarray, tile: int = TILE) -> tuple[np.ndarray, np.ndarray]:
    """torch conv weight f32[Co, Ci, K] -> (U f32[A, n * Ci, Co], tr f32[A * A + m * A]), computed in float64.
    tile = 2 (K = 3 only): F(2,3), 4 points: half the output rows per tile (steel-like register use), 0.67x the
    multiplications instead of F(4,3)'s 0.5x."""
    co, ci, k = w.shape
    r, n = variant(k)
    check(tile == TILE or (tile == 2 and k == 3), "tile 2 is only for K = 3")
    at, g, bt = transforms(tile, r, [0, 1, -1] if tile == 2 else None)
    alpha = tile + r - 1
    wg = np.zeros((n, r, ci, co))
    for t in range(k):
        wg[t // TILE, t % TILE] = np.asarray(w[:, :, t], dtype=np.float64).T
    u = np.einsum("xk,gkio->xgio", g, wg).reshape(alpha, n * ci, co)
    tr = np.concatenate([bt.reshape(-1), at.reshape(-1)])
    return np.ascontiguousarray(u, dtype=np.float32), tr.astype(np.float32)


BT = 32  # tiles per GEMM threadgroup (winograd.cpp kBT); 64 for tile 2: always 128 output rows per threadgroup
ROWS_TG = 128


def _bt(tile):
    return ROWS_TG // tile


def _alpha(kernel, tile=TILE):
    return 4 if tile == 2 else 6 if kernel == 3 else 7


def _lo(x, kernel, dilation, padding):
    lo = x.shape[0] + 2 * padding - dilation * (kernel - 1)
    check(_ge(lo, 1), "empty conv output")
    return lo


def _attrs_ok(x, *, dilation, groups, kernel, padding, tile):
    _f32(x, rank=2)
    ci = as_dim(x.shape[1])
    check(ci.is_const and ci.value % 16 == 0, "Ci must be a static multiple of 16")
    check((tile == TILE or (tile == 2 and kernel == 3)) and variant(kernel)[1] == groups,
          "groups / tile don't match the kernel size")
    check(dilation >= 1 and padding >= 0 and padding % dilation == 0, "padding must be a multiple of dilation")


def _rows(x, *, dilation, groups, kernel, padding, tile=TILE):
    """V rows: d * A * rmax, rmax = (tiles of residue class 0 rounded up to the threadgroup's) + groups - 1."""
    lo = _lo(x, kernel, dilation, padding)
    rmax = ceildiv(ceildiv(lo, dilation), ROWS_TG) * _bt(tile) + (groups - 1)
    return rmax * (dilation * _alpha(kernel, tile))


@kernel_op("wino_tf_lc_f32_gpu")
def _wino_tf(x, tr, *, dilation, groups, kernel, padding, tile):
    _attrs_ok(x, dilation=dilation, groups=groups, kernel=kernel, padding=padding, tile=tile)
    _f32(tr, rank=1)
    return TensorType(f32, [_rows(x, dilation=dilation, groups=groups, kernel=kernel, padding=padding, tile=tile),
                            x.shape[1]])


@kernel_op("wino_tf_snake_lc_f32_gpu")
def _wino_tf_snake(x, tr, affine, alpha, *, dilation, groups, kernel, padding, tile):
    _attrs_ok(x, dilation=dilation, groups=groups, kernel=kernel, padding=padding, tile=tile)
    _f32(tr, alpha, rank=1)
    _f32(affine, rank=2)
    check(affine.shape[0] == 3 and affine.shape[1] == x.shape[1] and alpha.shape[0] == x.shape[1],
          "affine must be f32[3, C] and alpha f32[C]")
    return TensorType(f32, [_rows(x, dilation=dilation, groups=groups, kernel=kernel, padding=padding, tile=tile),
                            x.shape[1]])


@kernel_op("wino_tf_leaky_lc_f32_gpu")
def _wino_tf_leaky(x, tr, affine, *, dilation, groups, kernel, padding, slope, tile):
    _attrs_ok(x, dilation=dilation, groups=groups, kernel=kernel, padding=padding, tile=tile)
    _f32(tr, rank=1)
    _f32(affine, rank=2)
    check(affine.shape[0] == 3 and affine.shape[1] == x.shape[1], "affine must be f32[3, C]")
    return TensorType(f32, [_rows(x, dilation=dilation, groups=groups, kernel=kernel, padding=padding, tile=tile),
                            x.shape[1]])


def _mm_rule(v, u, b, tr, x, *, dilation, groups, kernel, padding, tile):
    _attrs_ok(x, dilation=dilation, groups=groups, kernel=kernel, padding=padding, tile=tile)
    _f32(v, rank=2)
    _f32(u, rank=3)
    _f32(b, tr, rank=1)
    alpha, nci, co = (as_dim(s) for s in u.shape)
    check(alpha.is_const and alpha.value == _alpha(kernel, tile), "u must come from wino_weights for this kernel size")
    check(nci.is_const and nci.value == groups * as_dim(x.shape[1]).value, "u must be f32[A, groups * Ci, Co]")
    check(co.is_const and co.value % 64 == 0, "Co must be a static multiple of 64")
    check(b.shape[0] == u.shape[2], "bias must be f32[Co]")
    check(as_dim(tr.shape[0]).is_const and tr.shape[0] == alpha.value * (alpha.value + tile), "bad transform table")
    check(v.shape[1] == x.shape[1] and v.shape[0] == _rows(x, dilation=dilation, groups=groups, kernel=kernel,
                                                            padding=padding, tile=tile), "v must be wino_tf of x")
    return TensorType(f32, [_lo(x, kernel, dilation, padding), u.shape[2]])


@kernel_op("wino_mm_lc_f32_gpu")
def _wino_mm(v, u, b, tr, x, *, dilation, groups, kernel, padding, tile):
    return _mm_rule(v, u, b, tr, x, dilation=dilation, groups=groups, kernel=kernel, padding=padding, tile=tile)


@kernel_op("wino_mm_res_lc_f32_gpu")
def _wino_mm_res(v, u, b, tr, x, res, *, dilation, groups, kernel, oscale, padding, tile):
    t = _mm_rule(v, u, b, tr, x, dilation=dilation, groups=groups, kernel=kernel, padding=padding, tile=tile)
    _f32(res, rank=2)
    check(res.shape[0] == t.shape[0] and res.shape[1] == t.shape[1], "residual shape mismatch")
    return t


def _stats_type(x, out, *, dilation, kernel, padding):
    chunks = ceildiv(ceildiv(_lo(x, kernel, dilation, padding), dilation), ROWS_TG)
    return TensorType(f32, [chunks * dilation, out.shape[1], 2])


@kernel_op("wino_mm_stats_lc_f32_gpu")
def _wino_mm_stats(v, u, b, tr, x, *, dilation, groups, kernel, padding, tile):
    t = _mm_rule(v, u, b, tr, x, dilation=dilation, groups=groups, kernel=kernel, padding=padding, tile=tile)
    return t, _stats_type(x, t, dilation=dilation, kernel=kernel, padding=padding)


@kernel_op("wino_mm_res_stats_lc_f32_gpu")
def _wino_mm_res_stats(v, u, b, tr, x, res, *, dilation, groups, kernel, oscale, padding, tile):
    t = _wino_mm_res.__wrapped__(v, u, b, tr, x, res, dilation=dilation, groups=groups, kernel=kernel, oscale=oscale,
                                 padding=padding, tile=tile)
    return t, _stats_type(x, t, dilation=dilation, kernel=kernel, padding=padding)


@kernel_op("wino_adain_affine_lc_f32_gpu")
def _wino_affine(stats, gb, *, dilation, eps, length):
    _f32(stats, rank=3)
    _f32(gb, rank=1)
    c = stats.shape[1]
    g2 = as_dim(gb.shape[0])
    check(g2.is_const and g2.value % 2 == 0 and _ge(c, g2.value // 2), "gb must be f32[2 Creal] with Creal <= C")
    check(stats.shape[0] == ceildiv(ceildiv(length, dilation), ROWS_TG) * dilation, "stats chunks don't match length")
    return TensorType(f32, [3, c])


def wino_adain_affine_lc_f32_gpu(stats, gb, length, dilation=1, eps=1e-5):
    """AdaIN affine f32[3, C] = (mean, (1 + g) / std, b) like adain_affine_lc_f32_gpu, from the stats a Winograd conv
    produced in its epilogue (conv1d_wino_lc_f32_gpu(..., stats=True)) over `length` output rows."""
    return _wino_affine(stats, gb, dilation=_int(dilation, "dilation"), eps=_float(eps, "eps"), length=as_dim(length))


def conv1d_wino_lc_f32_gpu(x, u, b, tr, kernel, residual=None, out_scale=1.0, padding=0, dilation=1, affine=None,
                           alpha=None, slope=None, stats=False):
    """(conv1d(h, w) + b [+ residual]) * out_scale like conv1d_lc_f32_gpu, with (u, tr) = wino_weights(w),
    kernel = K, and h = x; or h = snake(AdaIN affine(x), alpha) when `affine` and `alpha` are given (the
    snake_affine_lc_f32_gpu activation fused into the input transform); or h = leaky_relu(affine(x), slope) when
    `affine` and `slope` are given (leaky_lc_f32_gpu with an affine). stats=True also returns the output's channel
    statistics (computed in the GEMM's epilogue) for wino_adain_affine_lc_f32_gpu: (y, stats)."""
    kernel = _int(kernel, "kernel")
    _, n = variant(kernel)
    tile = 2 if as_dim(u.shape[0]) == as_dim(4) else TILE  # wino_weights(w, tile=2): F(2,3)
    attrs = dict(padding=_int(padding, "padding"), dilation=_int(dilation, "dilation"), groups=n, kernel=kernel,
                 tile=tile)
    if affine is None:
        v = _wino_tf(x, tr, **attrs)
    elif slope is not None:
        check(alpha is None, "give either alpha (snake) or slope (leaky)")
        v = _wino_tf_leaky(x, tr, affine, slope=_float(slope, "slope"), **attrs)
    else:
        v = _wino_tf_snake(x, tr, affine, alpha, **attrs)
    if residual is None:
        check(out_scale == 1.0, "out_scale needs a residual")
        return (_wino_mm_stats if stats else _wino_mm)(v, u, b, tr, x, **attrs)
    return (_wino_mm_res_stats if stats else _wino_mm_res)(v, u, b, tr, x, residual,
                                                          oscale=_float(out_scale, "out_scale"), **attrs)


# --- 2-D: F(2x2, 3x3), stride 1, padding 1, NHWC (ResNet's 3x3 convs) -------------------------------------------


def wino2d_weights(w_khkwcico: np.ndarray):
    """Folded conv weight f32[3, 3, Ci, Co] -> (U f32[16, Ci, Co], bt f32[16] (F(2,3)'s BT, for the input transform),
    tr f32[320] = [BT (x) BT | AT (x) AT] for the GEMM), all in float64."""
    w = np.asarray(w_khkwcico, dtype=np.float64)
    check(w.ndim == 4 and w.shape[:2] == (3, 3), "wino2d_weights: weight must be [3, 3, Ci, Co]")
    at, g, bt = transforms(2, 3, [0, 1, -1])
    u = np.einsum("ai,bk,ikco->abco", g, g, w).reshape(16, w.shape[2], w.shape[3])
    tr = np.concatenate([np.kron(bt, bt).reshape(-1), np.kron(at, at).reshape(-1)])
    return (np.ascontiguousarray(u, dtype=np.float32), bt.reshape(-1).astype(np.float32), tr.astype(np.float32))


def _tiles2d(x):
    n, h, w = x.shape[0], x.shape[1], x.shape[2]
    return ceildiv(n * ceildiv(h, 2) * ceildiv(w, 2), BT) * BT


@kernel_op("wino_tf2d_nhwc_f32_gpu")
def _wino_tf2d(x, bt):
    _f32(x, rank=4)
    _f32(bt, rank=1)
    c = as_dim(x.shape[3])
    check(c.is_const and c.value % 16 == 0, "C must be a static multiple of 16")
    check(as_dim(bt.shape[0]).is_const and as_dim(bt.shape[0]).value == 16, "bt must be f32[16]")
    return TensorType(f32, [_tiles2d(x) * 16, x.shape[3]])


def _mm2_rule(v, u, b, tr, x):
    _f32(v, rank=2)
    _f32(u, rank=3)
    _f32(b, tr, rank=1)
    _f32(x, rank=4)
    a, ci, co = (as_dim(s) for s in u.shape)
    check(a.is_const and a.value == 16 and ci == as_dim(x.shape[3]), "u must be wino2d_weights' f32[16, Ci, Co]")
    check(co.is_const and co.value % 64 == 0, "Co must be a static multiple of 64")
    check(b.shape[0] == u.shape[2], "bias must be f32[Co]")
    check(as_dim(tr.shape[0]).is_const and as_dim(tr.shape[0]).value == 320, "tr must be f32[320]")
    check(v.shape[0] == _tiles2d(x) * 16 and v.shape[1] == x.shape[3], "v must be wino_tf2d of x")
    return TensorType(f32, [x.shape[0], x.shape[1], x.shape[2], u.shape[2]])


@kernel_op("wino_mm2d_nhwc_f32_gpu")
def _wino_mm2d(v, u, b, tr, x):
    return _mm2_rule(v, u, b, tr, x)


@kernel_op("wino_mm2d_nhwc_relu_f32_gpu")
def _wino_mm2d_relu(v, u, b, tr, x):
    return _mm2_rule(v, u, b, tr, x)


def _mm2_res_rule(v, u, b, tr, x, res):
    t = _mm2_rule(v, u, b, tr, x)
    _f32(res, rank=4)
    check(all(r == s for r, s in zip(res.shape, t.shape)), "residual shape mismatch")
    return t


@kernel_op("wino_mm2d_nhwc_res_f32_gpu")
def _wino_mm2d_res(v, u, b, tr, x, res):
    return _mm2_res_rule(v, u, b, tr, x, res)


@kernel_op("wino_mm2d_nhwc_res_relu_f32_gpu")
def _wino_mm2d_res_relu(v, u, b, tr, x, res):
    return _mm2_res_rule(v, u, b, tr, x, res)


def conv2d_wino_nhwc_f32_gpu(x, u, b, bt, tr, residual=None, relu=False):
    """[relu](conv2d_3x3_pad1(x, w) + b [+ residual]) with (u, bt, tr) = wino2d_weights(w); x f32[N, H, W, Ci]
    (Ci % 16 == 0), output f32[N, H, W, Co] (Co % 64 == 0). Same contract as conv2d_nhwc_f32_gpu(stride=1,
    padding=1)."""
    v = _wino_tf2d(x, bt)
    if residual is None:
        return (_wino_mm2d_relu if relu else _wino_mm2d)(v, u, b, tr, x)
    return (_wino_mm2d_res_relu if relu else _wino_mm2d_res)(v, u, b, tr, x, residual)

