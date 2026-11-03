"""MobileNetV2 (ImageNet) on lokinfer (docs/impl-notes/research-models.md §3, docs/models/mobilenet_v2.md).

Reference: `torchvision.models.mobilenet_v2(weights=IMAGENET1K_V1)`. lokinfer graph: NHWC; BN folded into every
conv offline; each conv / depthwise conv is one kernel with a bias + ReLU6 epilogue; the linear bottleneck's 1x1
projection adds the block residual in its epilogue (no activation, as in the model). Input: normalized image
f32[N, 3, H, W] (NCHW), N / H / W symbolic. Output: logits f32[N, 1000]. The GPU graph pads channel counts to a
multiple of 16 (the steel loader's K step: only the 24-channel stage and the 3-channel input need it) with zero
weights, so padded channels stay exactly 0.
"""
from __future__ import annotations

import numpy as np

import lokinfer as lk
from lokinfer import ops
from lokinfer.ops.vision import mbv2_block_nhwc_f32_cpu

BN_EPS = 1e-5


def torch_reference():
    import torchvision

    return torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1).eval()


def _fold(conv, bn):
    w = conv.weight.detach().double().numpy()  # [Co, Ci/groups, KH, KW]
    scale = (bn.weight / (bn.running_var + BN_EPS).sqrt()).detach().double().numpy()
    b = (bn.bias - bn.running_mean * bn.weight / (bn.running_var + BN_EPS).sqrt()).detach().double().numpy()
    return w * scale[:, None, None, None], b


def fold_weights(model) -> list:
    """Model as a list of blocks: {"layers": [layer, ...], "res": bool}, layer = dict(kind "conv" | "dw",
    w (torch layout, BN folded, float64), b, stride, padding, act) — plus a final {"fc": (w [1280, 1000], b)}."""
    from torch import nn
    from torchvision.ops.misc import Conv2dNormActivation

    def cna(m):  # Conv2dNormActivation -> layer
        conv, bn = m[0], m[1]
        w, b = _fold(conv, bn)
        return dict(kind="dw" if conv.groups > 1 else "conv", w=w, b=b, stride=conv.stride[0], padding=conv.padding[0],
                    act="relu6")

    blocks = []
    for m in model.features:
        if isinstance(m, Conv2dNormActivation):
            blocks.append({"layers": [cna(m)], "res": False})
            continue
        layers = []
        seq = list(m.conv)
        i = 0
        while i < len(seq):
            if isinstance(seq[i], Conv2dNormActivation):
                layers.append(cna(seq[i]))
                i += 1
            else:  # projection: Conv2d + BatchNorm2d, linear
                assert isinstance(seq[i], nn.Conv2d) and isinstance(seq[i + 1], nn.BatchNorm2d)
                w, b = _fold(seq[i], seq[i + 1])
                layers.append(dict(kind="conv", w=w, b=b, stride=1, padding=0, act="none"))
                i += 2
        blocks.append({"layers": layers, "res": bool(m.use_res_connect)})
    fc = model.classifier[1]
    blocks.append({"fc": (np.ascontiguousarray(fc.weight.detach().numpy().T), fc.bias.detach().numpy().copy())})
    return blocks


def _pad16(n: int) -> int:
    return -(-n // 16) * 16


def build(blocks, device: str = "cpu", fused: bool = True, stem_direct: bool = True):
    """x f32[N, 3, H, W] -> logits f32[N, 1000]; device "cpu" or "gpu" (GPU end to end, weights included).
    CPU, fused=True: every expand -> dw -> project block is one mbv2_block_nhwc kernel (the expanded tensor stays in
    cache, kernels/cpu/conv2d.cpp)."""
    gpu = device == "gpu"
    N, H, W = lk.Sym("N"), lk.Sym("H"), lk.Sym("W")
    T = (lambda a: lk.from_numpy(np.ascontiguousarray(a, np.float32), device="gpu", copy=True)) if gpu else \
        (lambda a: lk.from_numpy(np.ascontiguousarray(a, np.float32)))
    conv_op = ops.conv2d_nhwc_f32_gpu if gpu else ops.conv2d_nhwc_f32_cpu
    dw_op = ops.dwconv2d_nhwc_f32_gpu if gpu else ops.dwconv2d_nhwc_f32_cpu

    # Prepare every layer's constants once (device layout, GPU channel padding).
    prepared = []
    for blk in blocks:
        if "fc" in blk:
            w, b = blk["fc"]  # input 1280 channels: no GPU padding needed
            prepared.append(("fc", T(w), T(b)))
            continue
        layers = []
        for L in blk["layers"]:
            w, b = L["w"], L["b"]
            if L["kind"] == "dw":  # [C, 1, KH, KW] -> [KH, KW, C]
                c = w.shape[0]
                cp = _pad16(c) if gpu else c
                wl = np.zeros((w.shape[2], w.shape[3], cp))
                wl[:, :, :c] = w[:, 0].transpose(1, 2, 0)
                bl = np.zeros(cp)
                bl[:c] = b
            else:
                co, ci = w.shape[0], w.shape[1]
                if gpu:  # [Co, Ci, KH, KW] -> [Co_pad, KH, KW, Ci_pad]
                    cop, cip = _pad16(co), _pad16(ci)
                    wl = np.zeros((cop, w.shape[2], w.shape[3], cip))
                    wl[:co, :, :, :ci] = w.transpose(0, 2, 3, 1)
                else:  # [KH, KW, Ci, Co]
                    cop = co
                    wl = w.transpose(2, 3, 1, 0)
                bl = np.zeros(cop)
                bl[:co] = b
            layers.append((L, T(wl), T(bl)))
        prepared.append(("block", layers, blk["res"]))

    def fn(x):
        items = prepared
        if gpu:
            x = ops.nchw_to_nhwc_pad_f32_gpu(ops.to_gpu(x), channels=16)
        elif stem_direct:  # 3x3 s2 3 -> 32 stem straight from NCHW (direct NEON, no im2col / layout pass)
            (L, w, b), = prepared[0][1]
            x = ops.conv2d_smallci_nchw_f32_cpu(x, w, b, stride=L["stride"], padding=L["padding"], act=L["act"])
            items = prepared[1:]
        else:
            x = ops.nchw_to_nhwc_f32_cpu(x)
        for item in items:
            if item[0] == "fc":
                x = ops.global_avgpool_nhwc_f32_gpu(x) if gpu else ops.global_avgpool_nhwc_f32_cpu(x)
                if gpu:
                    return ops.to_cpu(ops.gemm_bias_f32_gpu(x, item[1], item[2]))
                return ops.gemm_bias_f32_cpu_accelerate(x, item[1], item[2])
            _, layers, res = item
            inp = x
            if fused and not gpu and len(layers) == 3:
                (_, we, be), (Ld, wd, bd), (_, wp, bp) = layers
                ci, ce = we.shape[2], we.shape[3]
                x = mbv2_block_nhwc_f32_cpu(x, ops.reshape(we, [ci, ce]), be, wd, bd,
                                            ops.reshape(wp, [ce, wp.shape[3]]), bp, stride=Ld["stride"], residual=res)
                continue
            for i, (L, w, b) in enumerate(layers):
                last = i == len(layers) - 1
                if L["kind"] == "dw":
                    x = dw_op(x, w, b, stride=L["stride"], padding=L["padding"], act=L["act"])
                else:
                    x = conv_op(x, w, b, stride=L["stride"], padding=L["padding"], act=L["act"],
                                residual=inp if (res and last) else None)
        raise AssertionError("missing fc")

    return lk.compile(fn, input_types=[lk.TensorType(lk.f32, [N, 3, H, W])])
