"""DCGAN 64x64 Generator + Discriminator on lokinfer (docs/impl-notes/research-models.md §4, docs/models/dcgan.md).

Reference: the PyTorch DCGAN tutorial architecture (nz 100, ngf = ndf = 64, nc 3) with the CelebA weights of
`hussamalafandi/DCGAN_CelebA` (pinned revision; loaded strictly into our own copy of the tutorial classes — the
repo's dcgan.py is never executed). lokinfer graphs, NHWC inside, BN folded offline:
- G: z f32[B, 100, 1, 1] -> image f32[B, 3, 64, 64]. Layer 1 on a 1x1 input is a GEMM [B, 100] x [100, 4·4·512]
  (+bias, ReLU) viewed as [B, 4, 4, 512]; layers 2-5 (ConvTranspose2d k4 s2 p1) are one GEMM x @ W[Ci, 16·Co] plus an
  overlap-add gather with bias + ReLU / tanh.
- D: image f32[B, 3, 64, 64] -> probability f32[B]. conv2d k4 s2 p1 with a bias + LeakyReLU(0.2) epilogue (BN folded),
  the last 4x4 conv on a 4x4 map is a GEMV over the NHWC-flattened features, then sigmoid. On the GPU the first conv
  is im2col (straight from NCHW, K = 48) + a 1x1 steel conv, instead of padding 3 channels to 16 (K = 256).
B is symbolic. device "cpu" or "gpu" (GPU end to end, weights on the GPU).
"""
from __future__ import annotations

import numpy as np

import lokinfer as lk
from lokinfer import ops

HF_REPO = "hussamalafandi/DCGAN_CelebA"
HF_REV = "0c4e26b8498c0ccf1ff06f47e0810ff4aaf8e14a"
NZ, NGF, NDF, NC = 100, 64, 64, 3
BN_EPS = 1e-5


def torch_reference():
    """(G, D) tutorial modules with the pinned CelebA weights, eval mode."""
    import torch
    from huggingface_hub import hf_hub_download
    from torch import nn

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = nn.Sequential(
                nn.ConvTranspose2d(NZ, NGF * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(NGF * 8), nn.ReLU(True),
                nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(NGF * 4), nn.ReLU(True),
                nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(NGF * 2), nn.ReLU(True),
                nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False), nn.BatchNorm2d(NGF), nn.ReLU(True),
                nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False), nn.Tanh())

        def forward(self, x):
            return self.main(x)

    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv2d(NC, NDF, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(NDF * 2), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(NDF * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(NDF * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())

        def forward(self, x):
            return self.main(x).view(-1, 1).squeeze(1)

    g, d = Generator(), Discriminator()
    for m, f in ((g, "generator.pth"), (d, "discriminator.pth")):
        sd = torch.load(hf_hub_download(HF_REPO, f, revision=HF_REV), map_location="cpu", weights_only=True)
        m.load_state_dict(sd, strict=True)
        m.eval()
    return g, d


def _bn(bn):
    scale = (bn.weight / (bn.running_var + BN_EPS).sqrt()).detach().double().numpy()
    shift = (bn.bias - bn.running_mean * bn.weight / (bn.running_var + BN_EPS).sqrt()).detach().double().numpy()
    return scale, shift


def fold_generator(g) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(W, b)] per layer: layer 0 W [100, 16·512] (columns (h, w, co)), b [16·512]; layers 1-4 W [Ci, 16·Co]
    (columns (kh, kw, co)), b [Co]. BN folded per output channel."""
    m = list(g.main)
    convs = [m[0], m[3], m[6], m[9], m[12]]
    bns = [m[1], m[4], m[7], m[10], None]
    out = []
    for i, (cv, bn) in enumerate(zip(convs, bns)):
        w = cv.weight.detach().double().numpy()  # [Ci, Co, 4, 4]
        ci, co = w.shape[0], w.shape[1]
        scale, shift = _bn(bn) if bn is not None else (np.ones(co), np.zeros(co))
        w = w * scale[None, :, None, None]
        wm = np.ascontiguousarray(w.transpose(0, 2, 3, 1).reshape(ci, 16 * co), np.float32)  # (kh, kw, co)
        b = np.tile(shift, 16) if i == 0 else shift
        out.append((wm, b.astype(np.float32)))
    return out


def fold_discriminator(d) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(w [Co, Ci, 4, 4] float64, b [Co])] for the 4 strided convs (BN folded), then (w_last [8192, 1] in NHWC
    flatten order (h, w, ci), b_last [1])."""
    m = list(d.main)
    convs, bns = [m[0], m[2], m[5], m[8]], [None, m[3], m[6], m[9]]
    out = []
    for cv, bn in zip(convs, bns):
        w = cv.weight.detach().double().numpy()
        co = w.shape[0]
        scale, shift = _bn(bn) if bn is not None else (np.ones(co), np.zeros(co))
        out.append((w * scale[:, None, None, None], shift))
    wl = m[11].weight.detach().double().numpy()[0]  # [512, 4, 4]
    out.append((np.ascontiguousarray(wl.transpose(1, 2, 0).reshape(-1, 1), np.float32), np.zeros(1, np.float32)))
    return out


def build_generator(layers, device: str = "cpu"):
    gpu = device == "gpu"
    B = lk.Sym("B")
    T = (lambda a: lk.from_numpy(np.ascontiguousarray(a, np.float32), device="gpu", copy=True)) if gpu else \
        (lambda a: lk.from_numpy(np.ascontiguousarray(a, np.float32)))
    W = [(T(w), T(b)) for w, b in layers]
    gemm = ops.gemm_f32_gpu_mps if gpu else ops.gemm_f32_cpu_accelerate
    gather = ops.convt2d_gather_nhwc_f32_gpu if gpu else ops.convt2d_gather_nhwc_f32_cpu

    def fn(z):
        if gpu:
            z = ops.to_gpu(z)
        z = ops.reshape(z, [B, NZ])
        x = (ops.gemm_bias_relu_f32_gpu if gpu else ops.gemm_bias_relu_f32_cpu_accelerate)(z, W[0][0], W[0][1])
        x = ops.reshape(x, [B, 4, 4, NGF * 8])
        h = 4
        for i in range(1, 5):
            ci = x.shape[3]
            y = gemm(ops.reshape(x, [B * h * h, ci]), W[i][0])
            co = W[i][1].shape[0]
            y = ops.reshape(y, [B, h, h, 16 * co])
            x = gather(y, W[i][1], kernel=4, stride=2, padding=1, act="tanh" if i == 4 else "relu")
            h *= 2
        x = (ops.nhwc_to_nchw_f32_gpu if gpu else ops.nhwc_to_nchw_f32_cpu)(x)
        return ops.to_cpu(x) if gpu else x

    return lk.compile(fn, input_types=[lk.TensorType(lk.f32, [B, NZ, 1, 1])])


def build_discriminator(layers, device: str = "cpu", stem_direct: bool = False):
    gpu = device == "gpu"
    B = lk.Sym("B")
    T = (lambda a: lk.from_numpy(np.ascontiguousarray(a, np.float32), device="gpu", copy=True)) if gpu else \
        (lambda a: lk.from_numpy(np.ascontiguousarray(a, np.float32)))
    convs = []
    for i, (w, b) in enumerate(layers[:4]):
        if gpu and i == 0:  # stem as a 1x1 conv over im2col patches: [Co, 1, 1, (y, x, c) = 4*4*3 = 48]
            wl = w.transpose(0, 2, 3, 1).reshape(w.shape[0], 1, 1, -1)
        elif gpu:  # [Co, KH, KW, Ci(pad 16)]
            ci = w.shape[1]
            cip = -(-ci // 16) * 16
            wl = np.zeros((w.shape[0], 4, 4, cip))
            wl[..., :ci] = w.transpose(0, 2, 3, 1)
        else:  # [KH, KW, Ci, Co]
            wl = w.transpose(2, 3, 1, 0)
        convs.append((T(wl), T(b)))
    wl, bl = T(layers[4][0]), T(layers[4][1])
    conv = ops.conv2d_nhwc_f32_gpu if gpu else ops.conv2d_nhwc_f32_cpu

    def fn(x):
        if gpu:  # layer 1: im2col straight from NCHW (K = 48, no 3 -> 16 channel padding) + 1x1 steel conv w/ leaky
            p = ops.im2col_nchw_f32_gpu(ops.to_gpu(x), kernel=4, stride=2, padding=1, kpad=4 * 4 * NC)
            x = conv(ops.reshape(p, [B, 32, 32, 4 * 4 * NC]), *convs[0], stride=1, padding=0, act="leaky")
            rest = convs[1:]
        elif stem_direct:  # layer 1 (4x4 s2, 3 -> 64) from NCHW by direct NEON: off, iPhone N=1 0.275 vs 0.203 ms with
            # im2col + Accelerate (NEON on the E cores is far below the SME-backed sgemm here; MobileNetV2's stem gains)
            x = ops.conv2d_smallci_nchw_f32_cpu(x, *convs[0], stride=2, padding=1, act="leaky")
            rest = convs[1:]
        else:
            x, rest = ops.nchw_to_nhwc_f32_cpu(x), convs
        for w, b in rest:
            x = conv(x, w, b, stride=2, padding=1, act="leaky")
        x = ops.reshape(x, [B, 4 * 4 * NDF * 8])
        if gpu:  # one-column GEMV + sigmoid in one kernel (a tiled GEMM leaves the GPU idle at N = 1)
            p = ops.reshape(ops.gemv1_bias_f32_gpu(x, wl, bl, act="sigmoid"), [B])
        else:
            p = ops.reshape(ops.sigmoid_f32_cpu(ops.gemm_bias_f32_cpu_accelerate(x, wl, bl)), [B])
        return ops.to_cpu(p) if gpu else p

    return lk.compile(fn, input_types=[lk.TensorType(lk.f32, [B, NC, 64, 64])])
