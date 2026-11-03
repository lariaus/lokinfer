"""Kokoro-82M forward as a lokinfer graph (porting step 6: basic, one kernel per
primitive op; see docs/impl-notes/kokoro-port-plan.md).

Inputs:  input_ids i64[T] (= [0, *phoneme_ids, 0], exactly KModel's input_ids),
         ref_s f32[256], speed f32 (or dur i64[T] when built with
         forced_durations=True), noise f32[600*F, 9] (SineGen's randn; F = frames)
Output:  audio f32[600*F]

Layouts: token-rate part [T, C]; frame-rate conv part channel-first [C, L].
Weights: weight-norm folded (exactly as torch computes it), Linear weights
transposed to [in, out], LSTM biases merged, all done once in numpy.
"""
from __future__ import annotations

import os

import numpy as np

import lokinfer as lk
from lokinfer import ops
from lokinfer.graph import current_graph
from lokinfer.ops._def import _as_value
from lokinfer.ops.transformer import attention_qkv_sg_f32_gpu
from lokinfer.ops.winograd import wino_weights

SQRT1_2 = float(1.0 / np.sqrt(2.0))  # AdainResBlk1d divides by np.sqrt(2)


# ----------------------------------------------------------------------------- weights

def fold_weights(model) -> dict[str, np.ndarray]:
    """State dict with weight norm folded: w = g·v/‖v‖ computed by torch's own
    `_weight_norm` (the function the legacy weight_norm hook calls in forward),
    norm over every dim except 0 (for ConvTranspose1d that is per *input*
    channel). The model isn't modified or copied."""
    import torch

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    out: dict[str, np.ndarray] = {}
    for k, v in sd.items():
        if k.endswith(".weight_g"):
            prefix = k[: -len(".weight_g")]
            w = torch._weight_norm(sd[prefix + ".weight_v"], sd[k], 0)
            out[prefix + ".weight"] = w.numpy().astype(np.float32)
        elif k.endswith(".weight_v"):
            continue
        else:
            out[k] = v.numpy().astype(np.float32)
    return out


class W:
    """Lazily wraps numpy weights as lokinfer Tensors (captured as constants)."""

    def __init__(self, sd: dict[str, np.ndarray]):
        self.sd = sd
        self._t: dict[str, lk.Tensor] = {}
        self.banks: dict = {}  # style-bank GEMV outputs per (graph, style vector, root), see style_vec

    def t(self, key: str, fn=None) -> lk.Tensor:
        k = key if fn is None else f"{key}#{fn.__name__}"
        if k not in self._t:
            a = self.sd[key] if fn is None else fn(self.sd[key])
            self._t[k] = lk.from_numpy(np.ascontiguousarray(a, dtype=a.dtype))
        return self._t[k]

    def style_bank(self, root: str):
        """All `<root>.*.fc` style projections (AdaIN1d / AdaLayerNorm) as one [128, total] weight + bias, and each
        fc's element offset in the output (M3: one GEMV per style vector instead of ~60 tiny GEMMs)."""
        key = f"{root}#style_bank"
        if key not in self._t:
            names = sorted(k[: -len(".weight")] for k in self.sd if k.startswith(root + ".") and k.endswith(".fc.weight"))
            wt = np.concatenate([self.sd[n + ".weight"].T for n in names], axis=1)  # [128, total]
            b = np.concatenate([self.sd[n + ".bias"] for n in names])
            offs, o = {}, 0
            for n in names:
                offs[n] = (o, self.sd[n + ".bias"].shape[0])
                o += offs[n][1]
            self._t[key] = (lk.from_numpy(np.ascontiguousarray(wt, dtype=np.float32)),
                            lk.from_numpy(np.ascontiguousarray(b, dtype=np.float32)), offs)
        return self._t[key]

    def g(self, key: str, fn=None) -> lk.Tensor:
        """State-dict tensor (optionally transformed by fn) as a GPU constant, built once."""
        k = f"{key}#gpu" if fn is None else f"{key}#{fn.__name__}#gpu"
        if k not in self._t:
            a = self.sd[key] if fn is None else fn(self.sd[key])
            self._t[k] = lk.from_numpy(np.ascontiguousarray(a, dtype=np.float32), device="gpu", copy=True)
        return self._t[k]

    def const(self, key: str, make, device: str = "cpu") -> lk.Tensor:
        """A derived constant (not in the state dict), f32, built once by make() and cached under `key`."""
        if key not in self._t:
            self._t[key] = lk.from_numpy(np.ascontiguousarray(make(), dtype=np.float32), device=device, copy=True)
        return self._t[key]

    def linear(self, prefix: str):
        """(W^T [in, out], b [out])"""
        return self.t(prefix + ".weight", _transpose), self.t(prefix + ".bias")

    def lstm(self, prefix: str):
        sd = self.sd
        w_ih = np.concatenate([sd[prefix + ".weight_ih_l0"], sd[prefix + ".weight_ih_l0_reverse"]])
        w_hh = np.concatenate([sd[prefix + ".weight_hh_l0"], sd[prefix + ".weight_hh_l0_reverse"]])
        b = np.concatenate([sd[prefix + ".bias_ih_l0"] + sd[prefix + ".bias_hh_l0"],
                            sd[prefix + ".bias_ih_l0_reverse"] + sd[prefix + ".bias_hh_l0_reverse"]])
        key = prefix + "#lstm"
        if key not in self._t:
            self._t[key] = tuple(lk.from_numpy(np.ascontiguousarray(a)) for a in (w_ih, w_hh, b))
        return self._t[key]


def _transpose(a):
    return np.ascontiguousarray(a.T)


def _flat_alpha(a):  # Snake alpha (1, C, 1) -> (C,)
    return np.ascontiguousarray(a.reshape(-1))


# ----------------------------------------------------------------------------- building blocks

def view(x, shape, offset=0):
    """Zero-cost view (offset in elements) of a traced value or captured Tensor."""
    g = current_graph()
    x = _as_value(g, x)
    return g.view(x, lk.TensorType(x.dtype, shape), offset)


def linear(x, w: W, prefix: str):
    wt, b = w.linear(prefix)
    return ops.gemm_bias_f32_cpu_accelerate(x, wt, b)


def style_vec(s, w: W, prefix: str):
    """fc(style) for AdaIN / AdaLayerNorm: [128] -> [2C], as a zero-copy view into the style bank: one GEMV computes
    every fc of the same model root (decoder / predictor) for this style vector, on first use."""
    root = prefix.split(".")[0]
    wt, b, offs = w.style_bank(root)
    key = (id(current_graph()), s.name, root)  # one bank per traced graph and style vector
    if key not in w.banks:
        w.banks[key] = ops.reshape(ops.gemm_bias_f32_cpu_accelerate(ops.reshape(s, [1, 128]), wt, b), [b.shape[0]])
    off, n = offs[prefix]
    return view(w.banks[key], [n], off)


def conv1d(x, w: W, prefix: str, stride=1, padding=0, dilation=1, bias=True):
    if bias:
        return ops.conv1d_f32_cpu(x, w.t(prefix + ".weight"), w.t(prefix + ".bias"), stride=stride, padding=padding,
                                  dilation=dilation)
    return ops.conv1d_nobias_f32_cpu(x, w.t(prefix + ".weight"), stride=stride, padding=padding, dilation=dilation)


def adain(x, s, w: W, prefix: str):
    return ops.adain1d_f32_cpu(x, style_vec(s, w, prefix + ".fc"), eps=1e-5)


def adain_resblk(x, s, w: W, prefix: str, cin: int, cout: int, upsample: bool):
    """AdainResBlk1d (StyleTTS2), channel-first [C, L]."""
    r = adain(x, s, w, prefix + ".norm1")
    r = ops.leaky_relu_f32_cpu(r, slope=0.2)
    if upsample:
        r = ops.conv_transpose1d_dw_f32_cpu(r, w.t(prefix + ".pool.weight"), w.t(prefix + ".pool.bias"), stride=2,
                                            padding=1, output_padding=1)
    r = conv1d(r, w, prefix + ".conv1", padding=1)
    r = adain(r, s, w, prefix + ".norm2")
    r = ops.leaky_relu_f32_cpu(r, slope=0.2)
    r = conv1d(r, w, prefix + ".conv2", padding=1)
    sc = ops.upsample_nearest1d_f32_cpu(x, scale=2) if upsample else x
    if cin != cout:
        sc = conv1d(sc, w, prefix + ".conv1x1", bias=False)
    return ops.scale_f32_cpu(ops.add_f32_cpu(r, sc), scale=SQRT1_2)


# CPU fused convs (generator resblocks, decoder / F0-N AdainResBlk1d): Winograd (F(4,3) / nested F(4,4), the GPU path's
# decomposition) with the AdaIN + Snake / leaky prologue fused into the input transform; the GEMMs run on SME2 on SME2 CPUs (iPhone 18 Pro Max) /
# M4+ CPUs, on Accelerate elsewhere. LOKINFER_KOKORO_WINO_CPU=0 selects the direct (im2col + Accelerate) fused convs.
WINOGRAD_CPU = os.environ.get("LOKINFER_KOKORO_WINO_CPU", "1") != "0"
WINOGRAD_CPU_DEC = os.environ.get("LOKINFER_KOKORO_WINO_CPU_DEC", "2")  # K = 3 AdainResBlk1d: 0 off, 1 all, 2 decoder only


def _wino_cpu_u(a):
    return ops.wino_weights_cpu(a)[0]


def _wino_cpu_tr(a):
    return ops.wino_weights_cpu(a)[1]


def _conv3_cpu(x, af, w: W, conv: str, act: str, residual=None, out_scale=1.0):
    """AdainResBlk1d's K = 3 conv (padding 1) on act(affine(x)), act "leaky" (slope 0.2) or "none" -> (out, stats)."""
    if WINOGRAD_CPU and (WINOGRAD_CPU_DEC == "1" or (WINOGRAD_CPU_DEC == "2" and conv.startswith("decoder."))):
        return ops.conv1d_wino_f32_cpu(x, w.t(conv + ".weight", _wino_cpu_u), w.t(conv + ".weight", _wino_cpu_tr),
                                       w.t(conv + ".bias"), af, 3, act=act, slope=0.2, residual=residual,
                                       out_scale=out_scale, padding=1)
    if act == "none":
        return ops.conv1d_affine_f32_cpu(x, w.t(conv + ".weight"), w.t(conv + ".bias"), af, padding=1)
    return ops.conv1d_leaky_f32_cpu(x, w.t(conv + ".weight"), w.t(conv + ".bias"), af, slope=0.2, residual=residual,
                                    out_scale=out_scale, padding=1)


def adain_resblk_fused(x, s, w: W, prefix: str, cin: int, cout: int, upsample: bool):
    """AdainResBlk1d through the fused conv kernels (M3): stats -> adain_affine -> conv1d_leaky -> adain_affine ->
    conv1d_leaky(+shortcut, x1/sqrt2). The upsample block keeps the basic AdaIN/leaky/depthwise-convT front (the
    convT changes the length between the activation and conv1); its convs are still fused."""
    af = ops.adain_affine_f32_cpu(ops.channel_stats_f32_cpu(x), style_vec(s, w, prefix + ".norm1.fc"),
                                  length=x.shape[1])
    if upsample:
        r = ops.leaky_relu_f32_cpu(adain(x, s, w, prefix + ".norm1"), slope=0.2)
        r = ops.conv_transpose1d_dw_f32_cpu(r, w.t(prefix + ".pool.weight"), w.t(prefix + ".pool.bias"), stride=2,
                                            padding=1, output_padding=1)
        ident = w.const(f"identity_affine.{cin}", lambda: np.stack([np.zeros(cin), np.ones(cin), np.zeros(cin)]))
        h, st = _conv3_cpu(r, ident, w, prefix + ".conv1", "none")
    else:
        h, st = _conv3_cpu(x, af, w, prefix + ".conv1", "leaky")
    af = ops.adain_affine_f32_cpu(st, style_vec(s, w, prefix + ".norm2.fc"), length=h.shape[1])
    sc = ops.upsample_nearest1d_f32_cpu(x, scale=2) if upsample else x
    if cin != cout:
        sc = conv1d(sc, w, prefix + ".conv1x1", bias=False)
    out, _ = _conv3_cpu(h, af, w, prefix + ".conv2", "leaky", residual=sc, out_scale=SQRT1_2)
    return out


def _snake_conv_cpu(x, af, w: W, alpha_key: str, conv: str, k: int, d: int, residual=None):
    """conv(snake(AdaIN affine(x))) + bias [+ residual], channel-first, "same" padding, dilation d -> (out, stats)."""
    alpha, pad = w.t(alpha_key, _flat_alpha), d * (k - 1) // 2
    if WINOGRAD_CPU:
        return ops.conv1d_snake_wino_f32_cpu(x, w.t(conv + ".weight", _wino_cpu_u), w.t(conv + ".weight", _wino_cpu_tr),
                                             w.t(conv + ".bias"), af, alpha, k, residual=residual, padding=pad,
                                             dilation=d)
    return ops.conv1d_snake_f32_cpu(x, w.t(conv + ".weight"), w.t(conv + ".bias"), af, alpha, residual=residual,
                                    padding=pad, dilation=d)


def adain_resblock1_fused(x, s, w: W, prefix: str, k: int, x_stats=None):
    """AdaINResBlock1 through the fused conv kernels (M3): per iteration
    adain_affine -> conv1d_snake -> adain_affine -> conv1d_snake(+x); every conv emits the chunk stats the next
    AdaIN needs, so no standalone norm/snake/add pass touches the frame-rate tensor. `x_stats`: channel stats of x
    if already known (the three resblocks of a stage share their input). Returns (x, stats(x))."""
    n = x.shape[1]
    st = ops.channel_stats_f32_cpu(x) if x_stats is None else x_stats
    for j, d in enumerate((1, 3, 5)):
        af = ops.adain_affine_f32_cpu(st, style_vec(s, w, f"{prefix}.adain1.{j}.fc"), length=n)
        xt, st1 = _snake_conv_cpu(x, af, w, f"{prefix}.alpha1.{j}", f"{prefix}.convs1.{j}", k, d)
        af = ops.adain_affine_f32_cpu(st1, style_vec(s, w, f"{prefix}.adain2.{j}.fc"), length=n)
        x, st = _snake_conv_cpu(xt, af, w, f"{prefix}.alpha2.{j}", f"{prefix}.convs2.{j}", k, 1, residual=x)
    return x, st


def adain_resblock1(x, s, w: W, prefix: str, k: int):
    """AdaINResBlock1 (generator workhorse), dilations 1, 3, 5. Basic kernels (parity reference for the fused path)."""
    for j, d in enumerate((1, 3, 5)):
        xt = adain(x, s, w, f"{prefix}.adain1.{j}")
        xt = ops.snake_f32_cpu(xt, w.t(f"{prefix}.alpha1.{j}", _flat_alpha))
        xt = conv1d(xt, w, f"{prefix}.convs1.{j}", padding=d * (k - 1) // 2, dilation=d)
        xt = adain(xt, s, w, f"{prefix}.adain2.{j}")
        xt = ops.snake_f32_cpu(xt, w.t(f"{prefix}.alpha2.{j}", _flat_alpha))
        xt = conv1d(xt, w, f"{prefix}.convs2.{j}", padding=(k - 1) // 2)
        x = ops.add_f32_cpu(xt, x)
    return x


def _w_lc(a):  # conv weight [Co, Ci, K] -> [Co, K, Ci] (channel-last steel conv)
    return np.ascontiguousarray(a.transpose(0, 2, 1))


def _w_convt(a):  # convT weight [Ci, Co, K] -> [Ci, K*Co] (GEMM then overlap-add gather)
    ci, co, k = a.shape
    return np.ascontiguousarray(a.transpose(0, 2, 1).reshape(ci, k * co))


def _w_post_pad(a):  # conv_post [22, 128, 7] -> [64, 7, 128], zero rows 22..63 (steel tiles are 64 wide)
    out = np.zeros((64, a.shape[2], a.shape[1]), np.float32)
    out[: a.shape[0]] = a.transpose(0, 2, 1)
    return out


def _w_post_co64(a):  # conv_post [22, 128, 7] -> [64, 128, 7], zero output channels 22..63
    out = np.zeros((64,) + a.shape[1:], np.float32)
    out[: a.shape[0]] = a
    return out


def _wino_u_post(a):
    return wino_weights(_w_post_co64(a))[0]


def _wino_tr_post(a):
    return wino_weights(_w_post_co64(a))[1]


def _b_post_pad(a):
    out = np.zeros(64, np.float32)
    out[: a.shape[0]] = a
    return out


def _pad16(n: int) -> int:
    return -(-n // 16) * 16


def _w_lc_pad(a):  # [Co, Ci, K] -> [Co, K, Ci padded to a multiple of 16] (zero input channels)
    co, ci, k = a.shape
    out = np.zeros((co, k, _pad16(ci)), np.float32)
    out[:, :, :ci] = a.transpose(0, 2, 1)
    return out


def _dw_pad(a):  # depthwise convT weight [C, 1, K] -> [Cpad, K]
    out = np.zeros((_pad16(a.shape[0]), a.shape[2]), np.float32)
    out[: a.shape[0]] = a[:, 0, :]
    return out


def _vec_pad(a):
    out = np.zeros(_pad16(a.shape[0]), np.float32)
    out[: a.shape[0]] = a
    return out


def adain_resblk_gpu(x, s, w: W, prefix: str, cin: int, cout: int, upsample: bool):
    """AdainResBlk1d on the GPU, channel-last x f32[L, pad16(cin)] (padding channels are 0 and stay 0):
    leaky(AdaIN) -> [dw convT x2] -> conv1 -> leaky(AdaIN) -> conv2 (+ conv1x1 shortcut, x 1/sqrt2)."""
    af = ops.adain_affine_lc_f32_gpu(ops.channel_stats_lc_f32_gpu(x), style_vec_gpu(s, w, prefix + ".norm1.fc"),
                                     length=x.shape[0])
    if upsample:
        h = ops.leaky_lc_f32_gpu(x, 0.2, affine=af)
        h = ops.dwconvt_lc_f32_gpu(h, w.g(prefix + ".pool.weight", _dw_pad), w.g(prefix + ".pool.bias", _vec_pad))
        h = _leaky_conv3_gpu(h, None, w, prefix + ".conv1", cout)
    else:
        h = _leaky_conv3_gpu(x, af, w, prefix + ".conv1", cout)
    af = ops.adain_affine_lc_f32_gpu(ops.channel_stats_lc_f32_gpu(h), style_vec_gpu(s, w, prefix + ".norm2.fc"),
                                     length=h.shape[0])
    sc = ops.upsample2_lc_f32_gpu(x) if upsample else x
    if prefix + ".conv1x1.weight" in w.sd:  # learned 1x1 shortcut (cin != cout); identity otherwise (predictor F0/N)
        zero = w.const(f"zeros#{cout}#gpu", lambda: np.zeros(cout), device="gpu")
        sc = ops.conv1d_lc_f32_gpu(sc, w.g(prefix + ".conv1x1.weight", _w_lc_pad), zero)
    return _leaky_conv3_gpu(h, af, w, prefix + ".conv2", cout, residual=sc, out_scale=SQRT1_2)


def _w_pad16_ci(a):  # torch conv weight [Co, Ci, K] -> Ci zero-padded to a multiple of 16
    co, ci, k = a.shape
    out = np.zeros((co, _pad16(ci), k), np.float32)
    out[:, :ci] = a
    return out


def _wino_u_pad(a):
    return wino_weights(_w_pad16_ci(a))[0]


def _wino_tr_pad(a):
    return wino_weights(_w_pad16_ci(a))[1]


def _leaky_conv3_gpu(x, af, w: W, conv: str, cout: int, residual=None, out_scale=1.0):
    """conv_k3(leaky_relu(affine(x), 0.2)) (or conv_k3(x) when af is None) + bias [+ residual] * out_scale: Winograd
    F(4,3) with the activation fused into its input transform, or leaky + steel (A/B: LOKINFER_KOKORO_WINO_DEC=0)."""
    if WINOGRAD_DEC and cout % 64 == 0:
        return ops.conv1d_wino_lc_f32_gpu(x, w.g(conv + ".weight", _wino_u_pad), w.g(conv + ".bias"),
                                          w.g(conv + ".weight", _wino_tr_pad), 3, residual=residual,
                                          out_scale=out_scale, padding=1, affine=af,
                                          slope=None if af is None else 0.2)
    h = x if af is None else ops.leaky_lc_f32_gpu(x, 0.2, affine=af)
    return ops.conv1d_lc_f32_gpu(h, w.g(conv + ".weight", _w_lc_pad), w.g(conv + ".bias"), residual=residual,
                                 out_scale=out_scale, padding=1)


def decoder_gpu(asr, f0d, nd, s_d, w: W):
    """Kokoro Decoder (encode + 4 decode blocks) on the GPU. asr f32[512, F], f0d / nd f32[1, F] arrive
    channel-first on the CPU (asr_res, a 1x1 conv, runs there too); returns dec_x channel-last f32[2F, 512] on
    the GPU (the Generator's input layout)."""
    base = ops.transpose_f32_gpu(ops.to_gpu(ops.concat_rows_f32_cpu(asr, f0d, nd)))  # [F, 514]
    extra = ops.concat_rows_f32_cpu(conv1d(asr, w, "decoder.asr_res.0"), f0d, nd)  # [66, F]
    extra = ops.transpose_f32_gpu(ops.to_gpu(extra))  # [F, 66]
    x = adain_resblk_gpu(ops.concat_pad_lc_f32_gpu(base), s_d, w, "decoder.encode", 514, 1024, False)
    for i in range(4):
        x = adain_resblk_gpu(ops.concat_pad_lc_f32_gpu(x, extra), s_d, w, f"decoder.decode.{i}", 1090,
                             1024 if i < 3 else 512, i == 3)
    return x


def _proj_col(a):  # F0_proj / N_proj weight [1, 256, 1] -> [256, 1] (a one-column GEMV)
    return np.ascontiguousarray(a.reshape(-1, 1))


def f0n_gpu(sh, s_p, w: W):
    """Prosody predictor F0 / N curves on the GPU: sh f32[F, 512] (the shared LSTM's output, channel-last as it
    comes) -> (f0, n) f32[2F] still on the GPU. Three AdainResBlk1d per curve + a one-column projection. The GPU is
    otherwise idle at this point of the graph, so these kernels run while the CPU computes the TextEncoder — the
    caller must emit the to_cpu syncs only after the TextEncoder (a sync waits for the GPU). On the CPU they cost
    ~3.7 ms of the app59f critical path (Mac, 2026-11-01)."""
    x = ops.to_gpu(sh)
    out = []
    for name in ("F0", "N"):
        y = adain_resblk_gpu(x, s_p, w, f"predictor.{name}.0", 512, 512, False)
        y = adain_resblk_gpu(y, s_p, w, f"predictor.{name}.1", 512, 256, True)
        y = adain_resblk_gpu(y, s_p, w, f"predictor.{name}.2", 256, 256, False)
        y = ops.gemv1_bias_f32_gpu(y, w.g(f"predictor.{name}_proj.weight", _proj_col), w.g(f"predictor.{name}_proj.bias"))
        out.append(ops.reshape(y, [y.shape[0]]))
    return out


def style_vec_gpu(s, w: W, prefix: str):
    """style_vec as a view into the GPU copy of the style bank (one cpu_to_gpu sync per bank)."""
    root = prefix.split(".")[0]
    style_vec(s, w, prefix)  # make sure the CPU bank exists
    key = (id(current_graph()), s.name, root)
    gkey = key + ("gpu",)
    if gkey not in w.banks:
        w.banks[gkey] = ops.to_gpu(w.banks[key])
    off, n = w.style_bank(root)[2][prefix]
    return view(w.banks[gkey], [n], off)


# Generator resblock convs on the GPU: snake(AdaIN) fused into a Winograd conv's input transform (F(4,3) for K = 3,
# nested F(4,4) for K = 7 / 11; kernels/gpu/winograd.metal, docs/impl-notes/winograd-conv.md) — vs the
# snake + steel conv pair per conv: 0.91-1.41x on the M2 Max, 0.87-1.64x on the iPhone (losses only for K = 3 at
# word-length inputs). LOKINFER_KOKORO_WINO=0 selects the steel path (A/B).
WINOGRAD = os.environ.get("LOKINFER_KOKORO_WINO", "1") != "0"
WINOGRAD_DEC = os.environ.get("LOKINFER_KOKORO_WINO_DEC", "1") != "0"  # decoder / F0-N AdainResBlk1d convs (K = 3)


def _wino_u(a):
    return wino_weights(a)[0]


def _wino_tr(a):
    return wino_weights(a)[1]


def _snake_conv_gpu(x, af, w: W, alpha_key: str, conv: str, k: int, d: int, residual=None, stats=False):
    """conv(snake(AdaIN affine(x))) (+ bias [+ residual]), channel-last, "same" padding, dilation d. stats=True:
    returns (y, stats, sd): the output's channel stats, from the Winograd GEMM's epilogue (sd = d, for
    wino_adain_affine) or from channel_stats_lc (sd = 0)."""
    alpha = w.g(alpha_key, _flat_alpha)
    pad = d * (k - 1) // 2
    if WINOGRAD:
        # Epilogue stats: K = 3 / 11 (-8 % .. +3 % on the GEMM, saves the stats pass); K = 7's kernel (2 tap groups)
        # slows 8-19 % with them (M2 Max, 2026-11-01), so it keeps the separate stats pass.
        fused = stats and k != 7
        y = ops.conv1d_wino_lc_f32_gpu(x, w.g(conv + ".weight", _wino_u), w.g(conv + ".bias"),
                                       w.g(conv + ".weight", _wino_tr), k, residual=residual, padding=pad,
                                       dilation=d, affine=af, alpha=alpha, stats=fused)
        if fused:
            return y[0], y[1], d
    else:
        h = ops.snake_affine_lc_f32_gpu(x, af, alpha)
        y = ops.conv1d_lc_f32_gpu(h, w.g(conv + ".weight", _w_lc), w.g(conv + ".bias"), residual=residual,
                                  padding=pad, dilation=d)
    return (y, ops.channel_stats_lc_f32_gpu(y), 0) if stats else y


def _affine_gpu(stats, sd, gb, n):
    """AdaIN affine from (stats, sd) of _snake_conv_gpu (sd = 0: channel_stats_lc's chunks)."""
    if sd:
        return ops.wino_adain_affine_lc_f32_gpu(stats, gb, n, dilation=sd)
    return ops.adain_affine_lc_f32_gpu(stats, gb, length=n)


def adain_resblock1_gpu(x, s, w: W, prefix: str, k: int, x_stats=None):
    """AdaINResBlock1 on the GPU, channel-last x f32[L, C]: per iteration stats -> affine -> snake -> conv
    (+bias) -> stats -> affine -> snake -> conv (+bias +x). Returns x."""
    n = x.shape[0]
    st = ops.channel_stats_lc_f32_gpu(x) if x_stats is None else x_stats
    st_d = 0  # st's producer: 0 channel_stats_lc, else the dilation of the Winograd conv whose epilogue made it
    for j, d in enumerate((1, 3, 5)):
        af = _affine_gpu(st, st_d, style_vec_gpu(s, w, f"{prefix}.adain1.{j}.fc"), n)
        h, hst, hsd = _snake_conv_gpu(x, af, w, f"{prefix}.alpha1.{j}", f"{prefix}.convs1.{j}", k, d, stats=True)
        af = _affine_gpu(hst, hsd, style_vec_gpu(s, w, f"{prefix}.adain2.{j}.fc"), n)
        if j < 2:
            x, st, st_d = _snake_conv_gpu(h, af, w, f"{prefix}.alpha2.{j}", f"{prefix}.convs2.{j}", k, 1, residual=x,
                                          stats=True)
        else:
            x = _snake_conv_gpu(h, af, w, f"{prefix}.alpha2.{j}", f"{prefix}.convs2.{j}", k, 1, residual=x)
    return x


def generator_gpu(x, har, s_d, w: W):
    """Kokoro Generator (after the harmonic source + STFT) on the GPU, channel-last. x = dec_x f32[2F, 512] on the
    GPU; har f32[22, 120F+1] arrives channel-first on the CPU (the noise convs, Ci = 22 and strided, stay there).
    Returns conv_post's output channel-last f32[120F+1, 64] (columns 0..21 real) on the GPU; the caller runs the
    iSTFT on the GPU from it (istft_logits_lc_f32_gpu)."""
    G = "decoder.generator."
    for i, (k_up, u, k_noise) in enumerate(((20, 10, 7), (12, 6, 11))):
        xs = conv1d(har, w, G + f"noise_convs.{i}", **({"stride": 6, "padding": 3} if i == 0 else {}))
        xs = ops.transpose_f32_gpu(ops.to_gpu(xs))  # [L, C]
        xs = adain_resblock1_gpu(xs, s_d, w, f"{G}noise_res.{i}", k_noise)
        x = ops.leaky_lc_f32_gpu(x, 0.1)
        y = ops.gemm_f32_gpu_mps(x, w.g(f"{G}ups.{i}.weight", _w_convt))  # [Lin, K*Co]
        x = ops.convt_gather_lc_f32_gpu(y, w.g(f"{G}ups.{i}.bias"), kernel=k_up, stride=u, padding=(k_up - u) // 2,
                                        add=xs, reflect_left=(i == 1))
        xst = ops.channel_stats_lc_f32_gpu(x)  # shared by the three resblocks' first AdaIN
        rb = [adain_resblock1_gpu(x, s_d, w, f"{G}resblocks.{3 * i + j}", kk, xst) for j, kk in enumerate((3, 7, 11))]
        x = ops.add3_scale_f32_gpu(rb[0], rb[1], rb[2], 1.0 / 3.0)
    if WINOGRAD:  # leaky(0.01) fused into the Winograd conv's input transform (identity affine)
        c = 128  # generator output channels
        ident = w.const(f"wino_identity_affine#{c}#gpu", lambda: np.stack([np.zeros(c), np.ones(c), np.zeros(c)]),
                        device="gpu")
        post = ops.conv1d_wino_lc_f32_gpu(x, w.g(G + "conv_post.weight", _wino_u_post),
                                          w.g(G + "conv_post.bias", _b_post_pad),
                                          w.g(G + "conv_post.weight", _wino_tr_post), 7, padding=3, affine=ident,
                                          slope=0.01)  # [L, 64]
    else:
        x = ops.leaky_lc_f32_gpu(x, 0.01)
        post = ops.conv1d_lc_f32_gpu(x, w.g(G + "conv_post.weight", _w_post_pad),
                                     w.g(G + "conv_post.bias", _b_post_pad), padding=3)  # [L, 64]
    return post  # channel-last logits [120F+1, 64] (columns 0..21 real), still on the GPU


def albert(ids_padded, w: W, T):
    sd_pre = "bert.embeddings."
    e = ops.embedding_f32_cpu(w.t(sd_pre + "word_embeddings.weight"), ids_padded)
    e = ops.add_f32_cpu(e, view(w.t(sd_pre + "position_embeddings.weight"), [T, 128]))  # pos[0:T]
    e = ops.add_f32_cpu(e, ops.broadcast_rows_f32_cpu(_token_type_row(w), rows=T))
    e = ops.layernorm_f32_cpu(e, w.t(sd_pre + "LayerNorm.weight"), w.t(sd_pre + "LayerNorm.bias"), eps=1e-12)
    h = linear(e, w, "bert.encoder.embedding_hidden_mapping_in")
    L = "bert.encoder.albert_layer_groups.0.albert_layers.0."
    _split = os.environ.get("LKX_ALBERT_SPLIT")  # EXPERIMENT A/B knob (temporary)
    if _split is not None:
        from lokinfer.ops.transformer import gemm_accel_f32_cpu

        def gb(x, wt, b):
            return gemm_accel_f32_cpu(x, wt, b, split=int(_split))

        def lin(x, w, prefix):
            wt, b = w.linear(prefix)
            return gb(x, wt, b)
    else:
        gb, lin = ops.gemm_bias_f32_cpu_accelerate, linear
    for _ in range(12):  # 12 layers sharing one set of weights
        # One fused QKV GEMM [T, 768] x [768, 2304] instead of three: at T ~ 60 the three 768-wide GEMMs cost 3 x 117 us
        # on the Mac CPU vs ~112 us fused (Accelerate's small-M efficiency), app59f 2026-11-01.
        # Tried 2026-11-02: GPT-2's fused CPU ops (gemm_accel_f32_cpu bias + gelu epilogue, add_bias_layernorm2;
        # bit-identical): Mac ALBERT 8.85 -> 8.41 ms but iPhone app59f ALBERT GEMMs +1.4-2.8 ms (interleaved A/B, 3
        # rounds; every shape slower under gemm_accel's auto split), so not used.
        qkv = gb(h, w.const("albert#wqkv", lambda: _qkv_w(w.sd, L)),
                                               w.const("albert#bqkv", lambda: np.concatenate(
                                                   [w.sd[L + f"attention.{n}.bias"] for n in ("query", "key", "value")])))
        a = ops.attention_qkv_f32_cpu(qkv, heads=12)
        a = lin(a, w, L + "attention.dense")
        h = ops.layernorm_f32_cpu(ops.add_f32_cpu(h, a), w.t(L + "attention.LayerNorm.weight"),
                                  w.t(L + "attention.LayerNorm.bias"), eps=1e-12)
        f = ops.gelu_tanh_f32_cpu(lin(h, w, L + "ffn"))
        f = lin(f, w, L + "ffn_output")
        h = ops.layernorm_f32_cpu(ops.add_f32_cpu(f, h), w.t(L + "full_layer_layer_norm.weight"),
                                  w.t(L + "full_layer_layer_norm.bias"), eps=1e-12)
    return h


def _qkv_w(sd, L):  # fused QKV weight [768, 2304] = [Wq^T | Wk^T | Wv^T]
    return np.ascontiguousarray(np.concatenate([sd[L + f"attention.{n}.weight"].T for n in ("query", "key", "value")],
                                               axis=1))


def albert_gpu(ids_padded, w: W, T):
    """ALBERT (12 layers, shared weights) on the GPU: embeddings + LN on the CPU, then per layer one fused QKV MPS
    GEMM, attention (biases folded into the reads), dense GEMM + add+bias+LN, FFN GEMM + bias+gelu, FFN-out GEMM +
    add+bias+LN. Returns the CPU result f32[T, 768]."""
    sd_pre = "bert.embeddings."
    e = ops.embedding_f32_cpu(w.t(sd_pre + "word_embeddings.weight"), ids_padded)
    e = ops.add_f32_cpu(e, view(w.t(sd_pre + "position_embeddings.weight"), [T, 128]))
    e = ops.add_f32_cpu(e, ops.broadcast_rows_f32_cpu(_token_type_row(w), rows=T))
    e = ops.layernorm_f32_cpu(e, w.t(sd_pre + "LayerNorm.weight"), w.t(sd_pre + "LayerNorm.bias"), eps=1e-12)
    h = linear(e, w, "bert.encoder.embedding_hidden_mapping_in")  # [T, 768], small: CPU
    h = ops.to_gpu(h)
    L = "bert.encoder.albert_layer_groups.0.albert_layers.0."
    wqkv = w.const("albert#wqkv#gpu", lambda: _qkv_w(w.sd, L), device="gpu")
    bqkv = w.const("albert#bqkv#gpu", lambda: np.concatenate(
        [w.sd[L + f"attention.{n}.bias"] for n in ("query", "key", "value")]), device="gpu")
    for _ in range(12):
        a = attention_qkv_sg_f32_gpu(ops.gemm_f32_gpu_mps(h, wqkv), bqkv, heads=12)  # simdgroup-matrix flash
        a = ops.gemm_f32_gpu_mps(a, w.g(L + "attention.dense.weight", _transpose))
        h = ops.add_bias_layernorm_f32_gpu(a, w.g(L + "attention.dense.bias"), h, w.g(L + "attention.LayerNorm.weight"),
                                           w.g(L + "attention.LayerNorm.bias"))
        f = ops.bias_gelu_f32_gpu(ops.gemm_f32_gpu_mps(h, w.g(L + "ffn.weight", _transpose)), w.g(L + "ffn.bias"))
        f = ops.gemm_f32_gpu_mps(f, w.g(L + "ffn_output.weight", _transpose))
        h = ops.add_bias_layernorm_f32_gpu(f, w.g(L + "ffn_output.bias"), h, w.g(L + "full_layer_layer_norm.weight"),
                                           w.g(L + "full_layer_layer_norm.bias"))
    return ops.to_cpu(h)


def _token_type_row(w: W):
    key = "bert.embeddings.token_type_embeddings.weight#row0"
    if key not in w._t:
        w._t[key] = lk.from_numpy(np.ascontiguousarray(w.sd["bert.embeddings.token_type_embeddings.weight"][0]))
    return w._t[key]


# BiLSTMs with w_hh^T (lstm_bidir_t_f32_cpu: the step GEMV on SME2 where available): iPhone T = 59 0.51 -> 0.39 ms, Mac
# 0.74 -> 0.52 ms, but OFF by default (LOKINFER_KOKORO_LSTM_T=1 opts in): the reordered GEMV sums move app47m's audio SNR
# 37.8 -> 36.1 dB vs the f64 golden (test limit 37.5; word 32.5 -> 32.6), an ulp-level chaotic effect via F0 / phase.
LSTM_T = os.environ.get("LOKINFER_KOKORO_LSTM_T", "0") == "1"


def lstm(x, w: W, prefix: str):
    w_ih, w_hh, b = w.lstm(prefix)
    if LSTM_T:
        sd = w.sd
        w_hh_t = w.const(prefix + "#lstm_whh_t", lambda: np.concatenate(
            [sd[prefix + ".weight_hh_l0"], sd[prefix + ".weight_hh_l0_reverse"]]).T)
        return ops.lstm_bidir_t_f32_cpu(x, w_ih, w_hh_t, b)
    return ops.lstm_bidir_f32_cpu(x, w_ih, w_hh, b)


def text_encoder_cnn(ids_p, w: W):
    """TextEncoder embedding + 3 x (conv k5, LayerNorm, leaky 0.2) -> f32[T, 512] (the BiLSTM's input)."""
    t = ops.embedding_f32_cpu(w.t("text_encoder.embedding.weight"), ids_p)  # [T, 512]
    for i in range(3):
        tc = conv1d(ops.transpose_f32_cpu(t), w, f"text_encoder.cnn.{i}.0", padding=2)  # [512, T]
        t = ops.layernorm_f32_cpu(ops.transpose_f32_cpu(tc), w.t(f"text_encoder.cnn.{i}.1.gamma"),
                                  w.t(f"text_encoder.cnn.{i}.1.beta"), eps=1e-5)
        t = ops.leaky_relu_f32_cpu(t, slope=0.2)
    return t


# ----------------------------------------------------------------------------- the model

def build(sd: dict[str, np.ndarray], *, forced_durations: bool = False, explicit_noise: bool = False,
          outputs: tuple[str, ...] = ("audio",), fused: bool = True, gpu: bool = False):
    """Compiles the Kokoro forward.

    Shipped variant (default): inputs (input_ids, ref_s, speed, seed i64) — the SineGen noise is generated
    in-kernel from `seed`, like PyTorch's randn_like after manual_seed.
    Parity variant (explicit_noise=True): the 4th input is the reference noise f32[600*F, 9] instead, so the
    output can be compared with PyTorch sample by sample. forced_durations=True (tests only, needs
    explicit_noise) replaces `speed` with the reference durations i64[T].
    `outputs` selects named intermediates (stage tests). fused=False builds the generator from the basic kernels
    only (the M1 graph; parity reference for the M3 fused kernels). gpu=True runs the Generator (after the
    harmonic source + STFT) on the GPU, channel-last (M4 stage 1); the rest stays on the CPU, with explicit
    to_gpu / to_cpu boundaries.
    """
    if forced_durations and not explicit_noise:
        raise ValueError("forced_durations needs explicit_noise (F is bound by the noise input)")
    w = W(sd)
    T, F = lk.Sym("T"), lk.Sym("F")
    in_types = [lk.TensorType(lk.i64, [T]), lk.TensorType(lk.f32, [256]),
                lk.TensorType(lk.i64, [T]) if forced_durations else lk.ScalarType(lk.f32),
                lk.TensorType(lk.f32, [600 * F, 9]) if explicit_noise else lk.ScalarType(lk.i64)]

    resblk = adain_resblk_fused if fused else adain_resblk

    def fn(ids_p, ref_s, speed_or_dur, noise_or_seed):
        st: dict = {}
        s_d = view(ref_s, [128])  # ref_s[:128]
        s_p = view(ref_s, [128], offset=128)  # ref_s[128:]
        # --- PL-BERT + bert_encoder
        st["bert_dur"] = (albert_gpu if gpu else albert)(ids_p, w, T)
        st["d_en"] = d_en = linear(st["bert_dur"], w, "bert_encoder")
        # --- CPU build: the TextEncoder CNN first, so its BiLSTM runs concurrently with the first duration-encoder
        # BiLSTM (independent; lstm_bidir2: 4 recurrences on 4 cores instead of 2 + 2 in sequence)
        t_en = None
        pair = not gpu and not LSTM_T
        if pair:
            t = text_encoder_cnn(ids_p, w)
        # --- DurationEncoder [T, 640]
        sp_rows = ops.broadcast_rows_f32_cpu(s_p, rows=T)
        x = ops.concat_cols_f32_cpu(d_en, sp_rows)
        for i in range(3):
            if i == 0 and pair:
                x, t_en = ops.lstm_bidir2_f32_cpu(x, *w.lstm("predictor.text_encoder.lstms.0"), t,
                                                  *w.lstm("text_encoder.lstm"))
            else:
                x = lstm(x, w, f"predictor.text_encoder.lstms.{2 * i}")
            x = ops.adalayernorm_f32_cpu(x, style_vec(s_p, w, f"predictor.text_encoder.lstms.{2 * i + 1}.fc"),
                                         eps=1e-5)
            x = ops.concat_cols_f32_cpu(x, sp_rows)
        st["d"] = d = x
        # --- durations
        st["lstm_x"] = lx = lstm(d, w, "predictor.lstm")
        st["dur_logits"] = logits = linear(lx, w, "predictor.duration_proj.linear_layer")
        if forced_durations:
            dur = speed_or_dur
        else:
            dur, total = ops.durations_f32_cpu(logits, speed_or_dur)
            # Defines F at runtime; the runtime checks it against the noise input's 600*F.
            current_graph().dim_read(total, 0, "F")
        st["pred_dur"] = dur
        # --- alignment (gather) + F0/N
        st["en"] = en = ops.repeat_interleave_f32_cpu(d, dur, frames=F)  # [F, 640]
        st["shared"] = sh = lstm(en, w, "predictor.shared")  # [F, 512]
        if gpu:  # F0 / N on the (idle) GPU, overlapped with the CPU TextEncoder below; synced after it
            f0_gpu, n_gpu = f0n_gpu(sh, s_p, w)
        else:
            shc = ops.transpose_f32_cpu(sh)  # [512, F]
            curves = {}
            for name in ("F0", "N"):
                y = resblk(shc, s_p, w, f"predictor.{name}.0", 512, 512, False)
                y = resblk(y, s_p, w, f"predictor.{name}.1", 512, 256, True)
                y = resblk(y, s_p, w, f"predictor.{name}.2", 256, 256, False)
                y = conv1d(y, w, f"predictor.{name}_proj")  # [1, 2F]
                curves[name] = ops.reshape(y, [2 * F])
            f0, n_curve = curves["F0"], curves["N"]
        # --- TextEncoder -> asr
        if t_en is None:
            t_en = lstm(text_encoder_cnn(ids_p, w), w, "text_encoder.lstm")
        st["t_en"] = t_en  # [T, 512]
        if gpu:
            f0, n_curve = ops.reshape(ops.to_cpu(f0_gpu), [2 * F]), ops.reshape(ops.to_cpu(n_gpu), [2 * F])
        st["F0_pred"], st["N_pred"] = f0, n_curve
        asr = ops.transpose_f32_cpu(ops.repeat_interleave_f32_cpu(t_en, dur, frames=F))  # [512, F]
        st["asr"] = asr
        # --- Decoder
        f0d = conv1d(ops.reshape(f0, [1, 2 * F]), w, "decoder.F0_conv", stride=2, padding=1)  # [1, F]
        nd = conv1d(ops.reshape(n_curve, [1, 2 * F]), w, "decoder.N_conv", stride=2, padding=1)
        if gpu:
            x = decoder_gpu(asr, f0d, nd, s_d, w)  # [2F, 512] channel-last, on the GPU
            if "dec_x" in outputs:
                st["dec_x"] = ops.to_cpu(ops.lc_to_cf_f32_gpu(x, 512))
        else:
            x = ops.concat_rows_f32_cpu(asr, f0d, nd)  # [514, F]
            x = resblk(x, s_d, w, "decoder.encode", 514, 1024, False)
            asr_res = conv1d(asr, w, "decoder.asr_res.0")  # [64, F]
            for i in range(4):
                x = ops.concat_rows_f32_cpu(x, asr_res, f0d, nd)  # [1090, F]
                x = resblk(x, s_d, w, f"decoder.decode.{i}", 1090, 1024 if i < 3 else 512, i == 3)
            st["dec_x"] = x  # [512, 2F]
        # --- Generator
        G = "decoder.generator."
        lw = w.t(G + "m_source.l_linear.weight", _flat_w9)
        if not explicit_noise:  # shipped: noise generated in-kernel from the seed
            har_source = ops.harmonic_source_rng_f32_cpu(f0, lw, w.t(G + "m_source.l_linear.bias"), noise_or_seed)
        else:
            har_source = ops.harmonic_source_f32_cpu(f0, noise_or_seed, lw, w.t(G + "m_source.l_linear.bias"))
        st["har_source"] = har_source
        st["har"] = har = ops.stft_magphase_f32_cpu(har_source, n_fft=20, hop=5)  # [22, 120F+1]
        if gpu:
            post_lc = generator_gpu(x, har, s_d, w)
            if "post" in outputs:  # stage tests only: the channel-first copy on the CPU
                st["post"] = ops.to_cpu(ops.lc_to_cf_f32_gpu(post_lc, 22))
            # iSTFT on the GPU straight from the channel-last logits (was ~1.1 ms of CPU at the end of the graph)
            st["audio"] = ops.to_cpu(ops.istft_logits_lc_f32_gpu(post_lc, n_fft=20, hop=5))
            res = tuple(st[o] for o in outputs)
            return res[0] if len(res) == 1 else res
        for i, (k_up, u, k_noise) in enumerate(((20, 10, 7), (12, 6, 11))):
            if i == 0:
                xs = conv1d(har, w, G + "noise_convs.0", stride=6, padding=3)
            else:
                xs = conv1d(har, w, G + "noise_convs.1")
            if fused:
                xs, _ = adain_resblock1_fused(xs, s_d, w, f"{G}noise_res.{i}", k_noise)
            else:
                xs = adain_resblock1(xs, s_d, w, f"{G}noise_res.{i}", k_noise)
            if fused:  # polyphase convT with the leaky prologue (stage 1: + the stage-0 resblock mean) and the
                # (reflect pad +) x_source epilogue
                x = ops.conv_transpose1d_poly_f32_cpu(
                    x if i == 0 else rb[0], w.t(f"{G}ups.{i}.weight", _convt_poly0 if i == 0 else _convt_poly1),
                    w.t(f"{G}ups.{i}.bias"), xs, stride=u, padding=(k_up - u) // 2, taps=k_up // u, slope=0.1,
                    pad_left=1 if i == 1 else 0, mean3=None if i == 0 else (rb[1], rb[2], 1.0 / 3.0))
            else:
                x = ops.leaky_relu_f32_cpu(x, slope=0.1)
                x = ops.conv_transpose1d_f32_cpu(x, w.t(f"{G}ups.{i}.weight"), w.t(f"{G}ups.{i}.bias"), stride=u,
                                                 padding=(k_up - u) // 2, output_padding=0)
                if i == 1:
                    x = ops.pad_reflect1d_f32_cpu(x, left=1, right=0)
                x = ops.add_f32_cpu(x, xs)
            if fused:
                xst = ops.channel_stats_f32_cpu(x)  # shared by the three resblocks' first AdaIN
                rb = [adain_resblock1_fused(x, s_d, w, f"{G}resblocks.{3 * i + j}", kk, xst)[0]
                      for j, kk in enumerate((3, 7, 11))]
            else:
                rb = [adain_resblock1(x, s_d, w, f"{G}resblocks.{3 * i + j}", kk) for j, kk in enumerate((3, 7, 11))]
            if not fused:
                x = ops.add3_scale_f32_cpu(rb[0], rb[1], rb[2], scale=1.0 / 3.0)
        if fused:  # resblock mean + leaky prologue fused; Winograd (nested F(4,4), K = 7) or the direct fused conv
            ident = w.const("#identity_affine128", lambda: np.stack(
                [np.zeros(128), np.ones(128), np.zeros(128)]).astype(np.float32))
            if WINOGRAD_CPU:
                post, _ = ops.conv1d_leaky_wino_mean3_f32_cpu(
                    rb[0], rb[1], rb[2], w.t(G + "conv_post.weight", _wino_cpu_u),
                    w.t(G + "conv_post.weight", _wino_cpu_tr), w.t(G + "conv_post.bias"), ident, 7, in_scale=1.0 / 3.0,
                    slope=0.01, padding=3)
            else:
                x = ops.add3_scale_f32_cpu(rb[0], rb[1], rb[2], scale=1.0 / 3.0)
                post, _ = ops.conv1d_leaky_f32_cpu(x, w.t(G + "conv_post.weight"), w.t(G + "conv_post.bias"), ident,
                                                   slope=0.01, padding=3)
            st["post"] = post  # [22, 120F+1]
        else:
            x = ops.leaky_relu_f32_cpu(x, slope=0.01)
            st["post"] = post = conv1d(x, w, G + "conv_post", padding=3)  # [22, 120F+1]
        st["audio"] = ops.istft_logits_f32_cpu(post, n_fft=20, hop=5)  # [600F]
        res = tuple(st[o] for o in outputs)
        return res[0] if len(res) == 1 else res

    return lk.compile(fn, input_types=in_types)


def _convt_poly0(a):  # ups[0] ConvTranspose1d weight [Ci, Co, 20] (s 10, p 5) -> polyphase panels (CPU)
    return ops.convt_poly_weights_cpu(a, 10, 5)


def _convt_poly1(a):  # ups[1] [Ci, Co, 12] (s 6, p 3)
    return ops.convt_poly_weights_cpu(a, 6, 3)


def _flat_w9(a):  # l_linear.weight (1, 9) -> (9,)
    return np.ascontiguousarray(a.reshape(9))
