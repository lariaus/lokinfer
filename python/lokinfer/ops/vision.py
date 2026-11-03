"""2-D vision kernels, NHWC (channels-last), f32 (M6: ResNet-18 / MobileNetV2; kernels/cpu/conv2d.cpp).

conv2d weights are `[KH, KW, Ci, Co]` (torch's `[Co, Ci, KH, KW]` permuted offline); the graph input arrives NCHW.
"""
from ..dims import as_dim
from ..dtypes import f32
from ..types import TensorType
from ._def import check, kernel_op
from .kokoro import _f32, _ge, _int


def _out(h, k, s, p):
    return (h + 2 * p - k) // s + 1


def _conv2d_rule(x, w, b, *, padding, stride):
    _f32(x, w, rank=4)
    _f32(b, rank=1)
    check(w.shape[2] == x.shape[3], f"w must be [KH, KW, Ci, Co]: x {x.shape}, w {w.shape}")
    check(b.shape[0] == w.shape[3], "bias must be f32[Co]")
    kh, kw = as_dim(w.shape[0]), as_dim(w.shape[1])
    check(kh.is_const and kw.is_const, "kernel size must be static")
    check(stride >= 1 and padding >= 0, "invalid stride/padding")
    ho, wo = _out(x.shape[1], kh.value, stride, padding), _out(x.shape[2], kw.value, stride, padding)
    check(_ge(ho, 1) and _ge(wo, 1), "empty conv output")
    return TensorType(f32, [x.shape[0], ho, wo, w.shape[3]])


_ACTS = ("none", "relu", "relu6", "leaky")  # leaky: LeakyReLU(0.2) (DCGAN D)


def _conv_ops(device, rule):
    """Registers conv2d_nhwc[_res][_relu|_relu6]_f32_<device> and returns {(res, act): op}."""
    table = {}
    for res in (False, True):
        for act in _ACTS:
            name = "conv2d_nhwc" + ("_res" if res else "") + ("" if act == "none" else "_" + act) + f"_f32_{device}"
            if (device, res, act) in _SKIP:
                continue
            if res:
                def r(x, w, b, residual, *, padding, stride):
                    t = rule(x, w, b, padding=padding, stride=stride)
                    _f32(residual, rank=4)
                    check(all(a == c for a, c in zip(residual.shape, t.shape)), "residual must match the output")
                    return t
            else:
                def r(x, w, b, *, padding, stride):
                    return rule(x, w, b, padding=padding, stride=stride)
            r.__name__ = r.__qualname__ = "_" + name
            table[(res, act)] = kernel_op(name)(r)
    return table


_SKIP = {(d, True, a) for d in ("cpu", "gpu") for a in ("relu6", "leaky")}  # not used by any model yet


def _conv(table, x, w, b, stride, padding, act, residual):
    check(act in _ACTS, f"act must be one of {_ACTS}")
    key = (residual is not None, act)
    check(key in table, f"no kernel for residual={key[0]} act={act}")
    attrs = dict(stride=_int(stride, "stride"), padding=_int(padding, "padding"))
    return table[key](x, w, b, residual, **attrs) if residual is not None else table[key](x, w, b, **attrs)


_CPU_CONV = _conv_ops("cpu", _conv2d_rule)


def conv2d_nhwc_f32_cpu(x, w, b, stride=1, padding=0, act="none", residual=None):
    """act(conv2d(x, w) + b [+ residual]): x f32[N, H, W, Ci], w f32[KH, KW, Ci, Co] -> f32[N, Ho, Wo, Co];
    act in ("none", "relu", "relu6", "leaky" = LeakyReLU(0.2))."""
    return _conv(_CPU_CONV, x, w, b, stride, padding, act, residual)


def _dw_rule(x, w, b, *, padding, stride, w_channels_last=True):
    _f32(x, rank=4)
    _f32(w, rank=3)
    _f32(b, rank=1)
    check(w.shape[2] == x.shape[3] and b.shape[0] == x.shape[3], "w must be f32[KH, KW, C], b f32[C]")
    kh, kw = as_dim(w.shape[0]), as_dim(w.shape[1])
    check(kh.is_const and kw.is_const, "kernel size must be static")
    ho, wo = _out(x.shape[1], kh.value, stride, padding), _out(x.shape[2], kw.value, stride, padding)
    check(_ge(ho, 1) and _ge(wo, 1), "empty conv output")
    return TensorType(f32, [x.shape[0], ho, wo, x.shape[3]])


def _dw_ops(device):
    table = {}
    for act in ("none", "relu6"):
        name = "dwconv2d_nhwc" + ("" if act == "none" else "_" + act) + f"_f32_{device}"

        def r(x, w, b, *, padding, stride):
            return _dw_rule(x, w, b, padding=padding, stride=stride)
        r.__name__ = r.__qualname__ = "_" + name
        table[act] = kernel_op(name)(r)
    return table


_CPU_DW = _dw_ops("cpu")


def dwconv2d_nhwc_f32_cpu(x, w, b, stride=1, padding=0, act="none"):
    """Depthwise conv (groups = C): x f32[N, H, W, C], w f32[KH, KW, C], b f32[C]; act "none" or "relu6"."""
    check(act in _CPU_DW, "act must be none or relu6")
    return _CPU_DW[act](x, w, b, stride=_int(stride, "stride"), padding=_int(padding, "padding"))


def _mbv2_rule(x, we, be, wd, bd, wp, bp, *, stride):
    _f32(x, rank=4)
    _f32(we, wp, rank=2)
    _f32(wd, rank=3)
    _f32(be, bd, bp, rank=1)
    ce = we.shape[1]
    check(we.shape[0] == x.shape[3], "we must be f32[Ci, Ce]")
    kh, kw = as_dim(wd.shape[0]), as_dim(wd.shape[1])
    check(kh.is_const and kh.value == 3 and kw.is_const and kw.value == 3 and wd.shape[2] == ce, "wd must be f32[3, 3, Ce]")
    check(be.shape[0] == ce and bd.shape[0] == ce and wp.shape[0] == ce and bp.shape[0] == wp.shape[1],
          "bias / projection shapes must match Ce / Co")
    check(stride in (1, 2), "stride must be 1 or 2")
    return TensorType(f32, [x.shape[0], _out(x.shape[1], 3, stride, 1), _out(x.shape[2], 3, stride, 1), wp.shape[1]])


@kernel_op("mbv2_block_nhwc_f32_cpu")
def _mbv2(x, we, be, wd, bd, wp, bp, *, stride):
    return _mbv2_rule(x, we, be, wd, bd, wp, bp, stride=stride)


@kernel_op("mbv2_block_nhwc_res_f32_cpu")
def _mbv2_res(x, we, be, wd, bd, wp, bp, *, stride):
    t = _mbv2_rule(x, we, be, wd, bd, wp, bp, stride=stride)
    check(stride == 1 and all(a == c for a, c in zip(x.shape, t.shape)), "residual block needs stride 1, Ci == Co")
    return t


def mbv2_block_nhwc_f32_cpu(x, we, be, wd, bd, wp, bp, stride=1, residual=False):
    """MobileNetV2 inverted residual in one kernel: relu6(x @ we + be) -> relu6(dw3x3(stride, pad 1) + bd) -> @ wp + bp
    [+ x]. x f32[N, H, W, Ci], we f32[Ci, Ce], wd f32[3, 3, Ce], wp f32[Ce, Co] (kernels/cpu/conv2d.cpp)."""
    op = _mbv2_res if residual else _mbv2
    return op(x, we, be, wd, bd, wp, bp, stride=_int(stride, "stride"))


@kernel_op("maxpool2d_nhwc_f32_cpu")
def _maxpool(x, *, kernel, padding, stride):
    _f32(x, rank=4)
    ho, wo = _out(x.shape[1], kernel, stride, padding), _out(x.shape[2], kernel, stride, padding)
    check(_ge(ho, 1) and _ge(wo, 1) and padding <= kernel // 2, "invalid maxpool")
    return TensorType(f32, [x.shape[0], ho, wo, x.shape[3]])


def maxpool2d_nhwc_f32_cpu(x, kernel=3, stride=2, padding=1):
    """F.max_pool2d (padding acts as -inf), NHWC."""
    return _maxpool(x, kernel=_int(kernel, "kernel"), stride=_int(stride, "stride"), padding=_int(padding, "padding"))


@kernel_op("global_avgpool_nhwc_f32_cpu")
def global_avgpool_nhwc_f32_cpu(x):
    """Mean over H, W: f32[N, H, W, C] -> f32[N, C]."""
    _f32(x, rank=4)
    return TensorType(f32, [x.shape[0], x.shape[3]])


@kernel_op("nchw_to_nhwc_f32_cpu")
def nchw_to_nhwc_f32_cpu(x):
    """Layout conversion f32[N, C, H, W] -> f32[N, H, W, C]."""
    _f32(x, rank=4)
    return TensorType(f32, [x.shape[0], x.shape[2], x.shape[3], x.shape[1]])


def _conv_smallci_rule(x, w, b, *, padding, stride):
    _f32(x, w, rank=4)
    _f32(b, rank=1)
    check(w.shape[2] == x.shape[1], f"w must be [KH, KW, Ci, Co] with x NCHW: x {x.shape}, w {w.shape}")
    check(b.shape[0] == w.shape[3], "bias must be f32[Co]")
    kh, kw, ci, co = (as_dim(d) for d in w.shape)
    check(kh.is_const and kw.is_const and ci.is_const and co.is_const, "weights must be static")
    check(ci.value * kh.value <= 64 and co.value % 16 == 0, "needs Ci * KH <= 64 and Co % 16 == 0")
    check(stride in (1, 2) and padding >= 0, "stride must be 1 or 2")
    ho, wo = _out(x.shape[2], kh.value, stride, padding), _out(x.shape[3], kw.value, stride, padding)
    return TensorType(f32, [x.shape[0], ho, wo, w.shape[3]])


_SMALLCI = {act: kernel_op(f"conv2d_smallci_nchw_{act}_f32_cpu")(
    (lambda x, w, b, *, padding, stride: _conv_smallci_rule(x, w, b, padding=padding, stride=stride)))
    for act in ("relu6", "leaky")}


def conv2d_smallci_nchw_f32_cpu(x, w, b, stride, padding, act):
    """act(conv2d(x) + b) for few input channels (network stems), x NCHW f32[N, Ci, H, W] -> NHWC f32[N, Ho, Wo, Co]:
    direct NEON (kernels/cpu/conv2d.cpp), no im2col and no NCHW -> NHWC pass. w [KH, KW, Ci, Co], Co % 16 == 0."""
    check(act in _SMALLCI, f"act must be one of {tuple(_SMALLCI)}")
    return _SMALLCI[act](x, w, b, stride=_int(stride, "stride"), padding=_int(padding, "padding"))


# --- CPU channel-planar padded layout ("cnhw", kernels/cpu/conv2d.cpp; the SME2 path, docs/kernels/convolution.md) ---
# A cnhw tensor is f32[C, N, H + 2 + extra, W + 2 + extra]: border 1 (plus `extra` rows / cols at the bottom / right),
# all zero; the image at rows / cols 1..H / 1..W.


def pack_conv_weights_cnhw(w):
    """numpy [KH, KW, Ci, Co] -> the SME2 A panels f32[ceil(Co / 32) * 32 * KH*KW*Ci] (kk = (ky*KW + kx)*Ci + c),
    packed offline (constants)."""
    import numpy as np

    kh, kw, ci, co = w.shape
    k = kh * kw * ci
    cop = (co + 31) // 32 * 32
    m = np.zeros((k, cop), np.float32)
    m[:, :co] = w.reshape(k, co)
    return np.ascontiguousarray(m.reshape(k, cop // 32, 32).transpose(1, 0, 2).reshape(-1))


def _conv_cnhw_rule(x, wp, b, *, extra, kh, kw, pad, pin, stride):
    _f32(wp, b, rank=1)
    check(stride in (1, 2), "stride must be 1 or 2")
    if stride == 1:
        _f32(x, rank=4)
        check(extra == 0 and 2 * pad == kh - 1 and kh == kw, "stride-1 cnhw convs keep the geometry (k = 2 pad + 1)")
        ci, n, hpo, wpo = x.shape
    else:
        _f32(x, rank=5)
        check(as_dim(x.shape[0]).is_const and as_dim(x.shape[0]).value == 4, "stride 2 takes the 4 phase planes")
        check(pin >= pad and (kh - 1 - pad + pin) // 2 <= 2 + extra, "phases too small for this kernel (extra)")
        _, ci, n, hpo, wpo = x.shape
    co = b.shape[0]
    ci_c, co_c = as_dim(ci), as_dim(co)
    if ci_c.is_const and co_c.is_const and as_dim(wp.shape[0]).is_const:
        check(as_dim(wp.shape[0]).value == (co_c.value + 31) // 32 * 32 * kh * kw * ci_c.value,
              "wp must be pack_conv_weights_cnhw(w)")
    return TensorType(f32, [co, n, hpo, wpo])


def _cnhw_ops():
    table = {}
    for res, act in ((False, "none"), (False, "relu"), (True, "relu")):
        name = "conv2d_cnhw" + ("_res" if res else "") + ("" if act == "none" else "_" + act) + "_f32_cpu"
        if res:
            def r(x, wp, b, residual, **at):
                t = _conv_cnhw_rule(x, wp, b, **at)
                _f32(residual, rank=4)
                check(all(a == c for a, c in zip(residual.shape, t.shape)), "residual must match the output")
                return t
        else:
            def r(x, wp, b, **at):
                return _conv_cnhw_rule(x, wp, b, **at)
        r.__name__ = r.__qualname__ = "_" + name
        table[(res, act)] = kernel_op(name)(r)
    return table


_CNHW_CONV = _cnhw_ops()


def conv2d_cnhw_f32_cpu(x, wp, b, kernel, stride=1, padding=0, act="none", residual=None, pin=1, extra=0):
    """act(conv2d(x) + b [+ residual]) in the cnhw layout. Stride 1: x f32[Ci, N, Hp, Wp] (k = 2·padding + 1, same
    geometry out). Stride 2: x = the 4 phase planes f32[4, Ci, N, Hpo, Wpo] in the output geometry (cnhw_to_phases /
    nchw_to_phases; `pin` = the source's border). wp = pack_conv_weights_cnhw(w). Output f32[Co, N, Hpo, Wpo]."""
    key = (residual is not None, act)
    check(key in _CNHW_CONV, f"no cnhw kernel for residual={key[0]} act={act}")
    at = dict(extra=_int(extra, "extra"), kh=_int(kernel, "kernel"), kw=_int(kernel, "kernel"),
              pad=_int(padding, "padding"), pin=_int(pin, "pin"), stride=_int(stride, "stride"))
    op = _CNHW_CONV[key]
    return op(x, wp, b, residual, **at) if residual is not None else op(x, wp, b, **at)


def pack_wino_weights_cnhw(w):
    """numpy [3, 3, Ci, Co] -> F(2x2, 3x3) Winograd weights for conv2d_wino_cnhw: U_p = (G w GT)_p (float64, then f32)
    as the SME2 A panels of each of the 16 points, f32[16 * ceil(Co / 32) * 32 * Ci] (offline constant)."""
    import numpy as np

    from .winograd import transforms

    check(w.ndim == 4 and w.shape[:2] == (3, 3), "pack_wino_weights_cnhw: weight must be [3, 3, Ci, Co]")
    _, _, ci, co = w.shape
    _, g, _ = transforms(2, 3, [0, 1, -1])
    u = np.einsum("ai,bk,ikco->abco", g, g, np.asarray(w, np.float64)).reshape(16, ci, co)
    cop = (co + 31) // 32 * 32
    m = np.zeros((16, ci, cop))
    m[:, :, :co] = u
    return np.ascontiguousarray(m.reshape(16, ci, cop // 32, 32).transpose(0, 2, 1, 3).reshape(-1), np.float32)


def _conv_wino_cnhw_rule(x, wp, u, b):
    _f32(x, rank=4)
    _f32(wp, u, b, rank=1)
    ci, n, hp, wp_ = x.shape
    co = b.shape[0]
    if as_dim(ci).is_const and as_dim(co).is_const and as_dim(u.shape[0]).is_const:
        check(as_dim(u.shape[0]).value == 16 * ((as_dim(co).value + 31) // 32 * 32) * as_dim(ci).value,
              "u must be pack_wino_weights_cnhw(w)")
    return TensorType(f32, [co, n, hp, wp_])


@kernel_op("conv2d_wino_cnhw_relu_f32_cpu")
def _conv_wino_cnhw_relu(x, wp, u, b):
    return _conv_wino_cnhw_rule(x, wp, u, b)


@kernel_op("conv2d_wino_cnhw_res_relu_f32_cpu")
def _conv_wino_cnhw_res_relu(x, wp, u, b, residual):
    t = _conv_wino_cnhw_rule(x, wp, u, b)
    _f32(residual, rank=4)
    check(all(a == c for a, c in zip(residual.shape, t.shape)), "residual must match the output")
    return t


def conv2d_wino_cnhw_f32_cpu(x, wp, u, b, residual=None):
    """relu(conv3x3(x) + b [+ residual]), stride 1, padding 1, in the cnhw layout (border 1, no extra): a Winograd
    F(2x2, 3x3) conv (kernels/cpu/conv2d.cpp, SME2 GEMMs on SME2 CPUs) when the batch has enough 2x2 tiles (>= 512, or
    >= 64 with >= 4 images), else the direct planar GEMM (chosen at run time: N is symbolic).
    wp = pack_conv_weights_cnhw(w), u = pack_wino_weights_cnhw(w)."""
    if residual is not None:
        return _conv_wino_cnhw_res_relu(x, wp, u, b, residual)
    return _conv_wino_cnhw_relu(x, wp, u, b)


@kernel_op("nchw_to_phases_f32_cpu")
def _nchw_phases(x, *, extra, kernel, pad):
    _f32(x, rank=4)
    n, c, h, w = x.shape
    ho, wo = _out(h, kernel, 2, pad), _out(w, kernel, 2, pad)
    return TensorType(f32, [4, c, n, as_dim(ho) + 2 + extra, as_dim(wo) + 2 + extra])


def nchw_to_phases_f32_cpu(x, kernel, padding, extra=0):
    """NCHW image -> the 4 stride-2 phase planes of its zero-padded (by `padding`) version, in the geometry of a
    kernel x kernel stride-2 conv's cnhw output with `extra` border: f32[4, C, N, Ho + 2 + extra, Wo + 2 + extra]."""
    return _nchw_phases(x, extra=_int(extra, "extra"), kernel=_int(kernel, "kernel"), pad=_int(padding, "padding"))


@kernel_op("cnhw_to_phases_f32_cpu")
def cnhw_to_phases_f32_cpu(x):
    """cnhw f32[C, N, H + 2, W + 2] -> phase planes f32[4, C, N, Ho + 2, Wo + 2] for a 3x3 pad-1 (or 1x1 pad-0)
    stride-2 conv (Ho = ceil(H / 2))."""
    _f32(x, rank=4)
    c, n, hp, wp = x.shape
    return TensorType(f32, [4, c, n, _out(as_dim(hp) - 2, 3, 2, 1) + 2, _out(as_dim(wp) - 2, 3, 2, 1) + 2])


@kernel_op("maxpool_cnhw_f32_cpu")
def _maxpool_cnhw(x, *, extra):
    _f32(x, rank=4)
    c, n, hp, wp = x.shape
    h, w = as_dim(hp) - 2 - extra, as_dim(wp) - 2 - extra
    return TensorType(f32, [c, n, _out(h, 3, 2, 1) + 2, _out(w, 3, 2, 1) + 2])


def maxpool_cnhw_f32_cpu(x, extra=0):
    """3x3 stride-2 pad-1 max pool, cnhw -> cnhw. Input must be >= 0 (post-ReLU): the zero border is the padding."""
    return _maxpool_cnhw(x, extra=_int(extra, "extra"))


@kernel_op("global_avgpool_cnhw_f32_cpu")
def global_avgpool_cnhw_f32_cpu(x):
    """Mean over the interior: cnhw f32[C, N, H + 2, W + 2] -> f32[N, C]."""
    _f32(x, rank=4)
    return TensorType(f32, [x.shape[1], x.shape[0]])


# --- GPU (kernels/gpu/vision.metal + the steel conv in kokoro_lc.metal) -----------------------------------


def _conv2d_gpu_rule(x, w, b, *, padding, stride):
    _f32(x, w, rank=4)
    _f32(b, rank=1)
    check(w.shape[3] == x.shape[3], f"w must be [Co, KH, KW, Ci]: x {x.shape}, w {w.shape}")
    check(b.shape[0] == w.shape[0], "bias must be f32[Co]")
    ci, kh, kw = as_dim(x.shape[3]), as_dim(w.shape[1]), as_dim(w.shape[2])
    check(ci.is_const and ci.value % 16 == 0, "Ci must be a static multiple of 16 (steel loader)")
    check(kh.is_const and kw.is_const and kh.value <= 16 and kw.value <= 16, "kernel must be static and <= 16")
    check(stride >= 1 and padding >= 0, "invalid stride/padding")
    ho, wo = _out(x.shape[1], kh.value, stride, padding), _out(x.shape[2], kw.value, stride, padding)
    check(_ge(ho, 1) and _ge(wo, 1), "empty conv output")
    return TensorType(f32, [x.shape[0], ho, wo, w.shape[0]])


_GPU_CONV = _conv_ops("gpu", _conv2d_gpu_rule)
_GPU_DW = _dw_ops("gpu")


def conv2d_nhwc_f32_gpu(x, w, b, stride=1, padding=0, act="none", residual=None):
    """GPU act(conv2d(x, w) + b [+ residual]): x f32[N, H, W, Ci] (Ci % 16 == 0), w f32[Co, KH, KW, Ci]."""
    return _conv(_GPU_CONV, x, w, b, stride, padding, act, residual)


def dwconv2d_nhwc_f32_gpu(x, w, b, stride=1, padding=0, act="none"):
    """GPU depthwise conv: x f32[N, H, W, C], w f32[KH, KW, C], b f32[C]; act "none" or "relu6"."""
    check(act in _GPU_DW, "act must be none or relu6")
    return _GPU_DW[act](x, w, b, stride=_int(stride, "stride"), padding=_int(padding, "padding"))


@kernel_op("nchw_to_nhwc_pad_f32_gpu")
def _to_nhwc_pad(x, *, channels):
    _f32(x, rank=4)
    check(_ge(channels, 1), "channels")
    return TensorType(f32, [x.shape[0], x.shape[2], x.shape[3], channels])


def nchw_to_nhwc_pad_f32_gpu(x, channels):
    """f32[N, C, H, W] -> f32[N, H, W, channels], extra channels zero."""
    return _to_nhwc_pad(x, channels=_int(channels, "channels"))


@kernel_op("maxpool2d_nhwc_f32_gpu")
def _maxpool_gpu(x, *, kernel, padding, stride):
    return _maxpool.__wrapped__(x, kernel=kernel, padding=padding, stride=stride)


def maxpool2d_nhwc_f32_gpu(x, kernel=3, stride=2, padding=1):
    """GPU F.max_pool2d (padding acts as -inf), NHWC."""
    return _maxpool_gpu(x, kernel=_int(kernel, "kernel"), stride=_int(stride, "stride"),
                        padding=_int(padding, "padding"))


@kernel_op("global_avgpool_nhwc_f32_gpu")
def global_avgpool_nhwc_f32_gpu(x):
    """GPU mean over H, W: f32[N, H, W, C] -> f32[N, C]."""
    _f32(x, rank=4)
    return TensorType(f32, [x.shape[0], x.shape[3]])



# --- DCGAN: ConvTranspose2d (GEMM + gather), layout, sigmoid --------------------------------------------------


def _convt2d_rule(y, b, *, kernel, padding, stride):
    _f32(y, rank=4)
    _f32(b, rank=1)
    check(y.shape[3] == kernel * kernel * b.shape[0], "y must be f32[B, H, W, K*K*Co]")
    check(stride >= 1 and padding >= 0, "invalid convT attrs")
    ho = (y.shape[1] - 1) * stride - 2 * padding + kernel
    wo = (y.shape[2] - 1) * stride - 2 * padding + kernel
    check(_ge(ho, 1) and _ge(wo, 1), "empty convT output")
    return TensorType(f32, [y.shape[0], ho, wo, b.shape[0]])


_CONVT = {}
for _dev in ("cpu", "gpu"):
    for _act in ("relu", "tanh"):
        _name = f"convt2d_gather_nhwc_{_act}_f32_{_dev}"

        def _r(y, b, *, kernel, padding, stride):
            return _convt2d_rule(y, b, kernel=kernel, padding=padding, stride=stride)
        _r.__name__ = _r.__qualname__ = "_" + _name
        _CONVT[(_dev, _act)] = kernel_op(_name)(_r)


def _convt2d(dev, y, b, kernel, stride, padding, act):
    check(act in ("relu", "tanh"), "act must be relu or tanh")
    return _CONVT[(dev, act)](y, b, kernel=_int(kernel, "kernel"), stride=_int(stride, "stride"),
                              padding=_int(padding, "padding"))


def convt2d_gather_nhwc_f32_cpu(y, b, kernel, stride, padding, act):
    """ConvTranspose2d overlap-add of y = x @ W[Ci, K*K*Co] (f32[B, H, W, K*K*Co]) + b, then act ("relu" | "tanh")
    -> f32[B, Ho, Wo, Co]."""
    return _convt2d("cpu", y, b, kernel, stride, padding, act)


def convt2d_gather_nhwc_f32_gpu(y, b, kernel, stride, padding, act):
    """GPU convt2d_gather_nhwc_f32_cpu."""
    return _convt2d("gpu", y, b, kernel, stride, padding, act)


def _to_nchw_rule(x):
    _f32(x, rank=4)
    return TensorType(f32, [x.shape[0], x.shape[3], x.shape[1], x.shape[2]])


@kernel_op("nhwc_to_nchw_f32_cpu")
def nhwc_to_nchw_f32_cpu(x):
    """Layout conversion f32[N, H, W, C] -> f32[N, C, H, W]."""
    return _to_nchw_rule(x)


@kernel_op("nhwc_to_nchw_f32_gpu")
def nhwc_to_nchw_f32_gpu(x):
    """GPU nhwc_to_nchw_f32_cpu."""
    return _to_nchw_rule(x)


@kernel_op("sigmoid_f32_cpu")
def sigmoid_f32_cpu(x):
    """1 / (1 + exp(-x)), any shape."""
    _f32(x)
    return TensorType(f32, x.shape)


@kernel_op("sigmoid_f32_gpu")
def sigmoid_f32_gpu(x):
    """GPU sigmoid, any shape."""
    _f32(x)
    return TensorType(f32, x.shape)



@kernel_op("im2col_nchw_f32_gpu")
def _im2col(x, *, kernel, kpad, padding, stride):
    _f32(x, rank=4)
    ho, wo = _out(x.shape[2], kernel, stride, padding), _out(x.shape[3], kernel, stride, padding)
    c = as_dim(x.shape[1])
    check(c.is_const and kpad >= kernel * kernel * c.value, "kpad must cover KH*KW*C")
    return TensorType(f32, [x.shape[0] * ho * wo, kpad])


def im2col_nchw_f32_gpu(x, kernel, stride, padding, kpad):
    """GPU im2col of an NCHW input f32[N, C, H, W] -> patches f32[N*Ho*Wo, kpad], column (y*K + x)*C + c, zero-padded
    (for small-C stems run as one GEMM)."""
    return _im2col(x, kernel=_int(kernel, "kernel"), kpad=_int(kpad, "kpad"), padding=_int(padding, "padding"),
                   stride=_int(stride, "stride"))



def _gemv1_rule(x, w, b):
    _f32(x, w, rank=2)
    _f32(b, rank=1)
    check(w.shape[0] == x.shape[1] and w.shape[1] == 1 and b.shape[0] == 1, "w must be f32[K, 1], b f32[1]")
    return TensorType(f32, [x.shape[0], 1])


_gemv1 = kernel_op("gemv1_bias_f32_gpu")(_gemv1_rule)
_gemv1_sigmoid = kernel_op("gemv1_bias_sigmoid_f32_gpu")(_gemv1_rule)


def gemv1_bias_f32_gpu(x, w, b, act="none"):
    """GPU x f32[R, K] @ w f32[K, 1] + b -> f32[R, 1] (one threadgroup per row); act "none" or "sigmoid"."""
    check(act in ("none", "sigmoid"), "act must be none or sigmoid")
    return (_gemv1_sigmoid if act == "sigmoid" else _gemv1)(x, w, b)
