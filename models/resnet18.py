"""ResNet-18 (ImageNet) on lokinfer (docs/impl-notes/research-models.md §2, docs/models/resnet18.md).

Reference: `torchvision.models.resnet18(weights=IMAGENET1K_V1)`. lokinfer graph: NHWC internally; every BatchNorm
is folded into the preceding conv offline (w' = w·γ/√(σ²+ε), b' = β − μ·γ/√(σ²+ε)), so each conv is one
`conv2d_nhwc` kernel with a bias + [residual] + [ReLU] epilogue. Input: the normalized image f32[N, 3, H, W]
(NCHW, like PyTorch; preprocessing stays outside the graph), N / H / W symbolic. Output: logits f32[N, 1000].
"""
from __future__ import annotations

import numpy as np

import lokinfer as lk
from lokinfer import ops

BN_EPS = 1e-5


def torch_reference():
    import torchvision

    return torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()


def fold_weights(model) -> dict[str, np.ndarray]:
    """{conv name: (w [KH, KW, Ci, Co], b [Co])} with BN folded; plus fc."""
    sd = {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}
    out: dict[str, np.ndarray] = {}

    def fold(conv: str, bn: str):
        w = sd[conv + ".weight"]  # [Co, Ci, KH, KW]
        scale = sd[bn + ".weight"] / np.sqrt(sd[bn + ".running_var"] + BN_EPS)
        out[conv + ".w"] = np.ascontiguousarray((w * scale[:, None, None, None]).transpose(2, 3, 1, 0), np.float32)
        out[conv + ".b"] = (sd[bn + ".bias"] - sd[bn + ".running_mean"] * scale).astype(np.float32)

    fold("conv1", "bn1")
    for layer in range(1, 5):
        for blk in range(2):
            p = f"layer{layer}.{blk}"
            fold(p + ".conv1", p + ".bn1")
            fold(p + ".conv2", p + ".bn2")
            if p + ".downsample.0.weight" in sd:
                fold(p + ".downsample.0", p + ".downsample.1")
    out["fc.w"] = np.ascontiguousarray(sd["fc.weight"].T, np.float32)  # [512, 1000]
    out["fc.b"] = sd["fc.bias"].astype(np.float32)
    return out


def build(weights: dict[str, np.ndarray]):
    """CPU graph: x f32[N, 3, H, W] -> logits f32[N, 1000]."""
    t = {k: lk.from_numpy(v) for k, v in weights.items()}
    N, H, W = lk.Sym("N"), lk.Sym("H"), lk.Sym("W")

    def conv(x, name, stride=1, padding=0, relu=True, residual=None):
        return ops.conv2d_nhwc_f32_cpu(x, t[name + ".w"], t[name + ".b"], stride=stride, padding=padding,
                                       act="relu" if relu else "none", residual=residual)

    def fn(x):
        x = ops.nchw_to_nhwc_f32_cpu(x)
        x = conv(x, "conv1", stride=2, padding=3)
        x = ops.maxpool2d_nhwc_f32_cpu(x, kernel=3, stride=2, padding=1)
        for layer in range(1, 5):
            for blk in range(2):
                p = f"layer{layer}.{blk}"
                stride = 2 if layer > 1 and blk == 0 else 1
                sc = conv(x, p + ".downsample.0", stride=stride, relu=False) if p + ".downsample.0.w" in t else x
                h = conv(x, p + ".conv1", stride=stride, padding=1)
                x = conv(h, p + ".conv2", padding=1, residual=sc)
        x = ops.global_avgpool_nhwc_f32_cpu(x)
        return ops.gemm_bias_f32_cpu_accelerate(x, t["fc.w"], t["fc.b"])

    return lk.compile(fn, input_types=[lk.TensorType(lk.f32, [N, 3, H, W])])


def build_cnhw(weights: dict[str, np.ndarray], wino_layers=(1, 2, 3, 4)):
    """CPU graph in the channel-planar padded layout (the iPhone SME2 path, opt-in until validated on the phone;
    docs/models/resnet18.md): every conv is one GEMM W^T · planes with the activations read in place (no im2col, no
    per-call packing) and the weights packed offline; stride 2 reads 2x2 phase planes. Off SME2 CPUs (the Mac) the
    convs gather B and call Accelerate (correct, not tuned). Same signature as build()."""
    t = {k: lk.from_numpy(v) for k, v in weights.items() if not k.endswith(".w") or k == "fc.w"}
    for k, v in weights.items():
        if k.endswith(".w") and k != "fc.w":
            t[k] = lk.from_numpy(ops.pack_conv_weights_cnhw(v))
    # Stride-1 3x3 convs of these layers as Winograd F(2x2, 3x3) (conv2d_wino_cnhw).
    wino = {k: lk.from_numpy(ops.pack_wino_weights_cnhw(v)) for k, v in weights.items()
            if k.startswith("layer") and int(k[5]) in wino_layers and k.endswith(".w") and v.shape[:2] == (3, 3)}
    N, H, W = lk.Sym("N"), lk.Sym("H"), lk.Sym("W")

    def conv(x, name, kernel, stride=1, padding=0, relu=True, residual=None, pin=1, extra=0):
        if stride == 1 and kernel == 3 and relu and name + ".w" in wino:
            return ops.conv2d_wino_cnhw_f32_cpu(x, t[name + ".w"], wino[name + ".w"], t[name + ".b"], residual=residual)
        return ops.conv2d_cnhw_f32_cpu(x, t[name + ".w"], t[name + ".b"], kernel, stride=stride, padding=padding,
                                       act="relu" if relu else "none", residual=residual, pin=pin, extra=extra)

    def fn(x):
        x = ops.nchw_to_phases_f32_cpu(x, kernel=7, padding=3, extra=1)
        x = conv(x, "conv1", 7, stride=2, padding=3, pin=3, extra=1)
        x = ops.maxpool_cnhw_f32_cpu(x, extra=1)
        for layer in range(1, 5):
            for blk in range(2):
                p = f"layer{layer}.{blk}"
                if layer > 1 and blk == 0:
                    ph = ops.cnhw_to_phases_f32_cpu(x)
                    sc = conv(ph, p + ".downsample.0", 1, stride=2, relu=False)
                    h = conv(ph, p + ".conv1", 3, stride=2, padding=1)
                else:
                    sc = x
                    h = conv(x, p + ".conv1", 3, padding=1)
                x = conv(h, p + ".conv2", 3, padding=1, residual=sc)
        x = ops.global_avgpool_cnhw_f32_cpu(x)
        return ops.gemm_bias_f32_cpu_accelerate(x, t["fc.w"], t["fc.b"])

    return lk.compile(fn, input_types=[lk.TensorType(lk.f32, [N, 3, H, W])])


def _gpu_layout(w_khkwcico: np.ndarray, ci_pad: int | None = None) -> np.ndarray:
    """[KH, KW, Ci, Co] -> [Co, KH, KW, Ci(pad)] for the steel conv."""
    w = w_khkwcico.transpose(3, 0, 1, 2)
    if ci_pad is not None and ci_pad > w.shape[3]:
        w = np.concatenate([w, np.zeros(w.shape[:3] + (ci_pad - w.shape[3],), np.float32)], axis=3)
    return np.ascontiguousarray(w, np.float32)


def build_gpu(weights: dict[str, np.ndarray]):
    """GPU graph end to end (weights on the GPU): x f32[N, 3, H, W] -> logits f32[N, 1000]. NHWC steel convs with
    bias / residual / ReLU epilogues; the 3-channel stem input is zero-padded to 16 channels."""
    t = {}
    for k, v in weights.items():
        if k == "conv1.w":  # stem as one GEMM over im2col patches: [7*7*3 = 147 -> 160, 64]
            v = np.concatenate([v.reshape(147, 64), np.zeros((13, 64), np.float32)])
        elif k.endswith(".w") and k != "fc.w":
            v = _gpu_layout(v)
        t[k] = lk.from_numpy(np.ascontiguousarray(v), device="gpu", copy=True)
    N, H, W = lk.Sym("N"), lk.Sym("H"), lk.Sym("W")

    # Stride-1 3x3 convs with Co >= 256 (layers 3-4): Winograd F(2x2, 3x3) with the 16 points split across
    # threadgroups on small grids (docs/impl-notes/winograd-conv.md): Mac GPU N=1 2.18 -> 1.43 ms, N=8 7.07 -> 6.6-6.9.
    # The 56x56 / 28x28 layers stay on the steel conv (Winograd <= 1.0x there).
    wino = {k: [lk.from_numpy(a, device="gpu", copy=True) for a in ops.wino2d_weights(v)]
            for k, v in weights.items() if k.endswith(".w") and v.ndim == 4 and v.shape[:2] == (3, 3) and v.shape[3] >= 256}

    def conv(x, name, stride=1, padding=0, relu=True, residual=None):
        if stride == 1 and padding == 1 and name + ".w" in wino:
            u, bt, tr = wino[name + ".w"]
            return ops.conv2d_wino_nhwc_f32_gpu(x, u, t[name + ".b"], bt, tr, residual=residual, relu=relu)
        return ops.conv2d_nhwc_f32_gpu(x, t[name + ".w"], t[name + ".b"], stride=stride, padding=padding,
                                       act="relu" if relu else "none", residual=residual)

    def fn(x):
        # Stem: im2col straight from the NCHW input + one steel GEMM (bias + ReLU epilogue): K = 147 (pad 160) instead
        # of a steel conv with channels padded to 16 (K = 784); the GEMM's [N*Ho*Wo, 64] output is already NHWC.
        p = ops.im2col_nchw_f32_gpu(ops.to_gpu(x), kernel=7, stride=2, padding=3, kpad=160)
        p = ops.gemm_steel_f32_gpu(p, t["conv1.w"], bias=t["conv1.b"], act="relu")
        Ho, Wo = (H + 6 - 7) // 2 + 1, (W + 6 - 7) // 2 + 1
        x = ops.reshape(p, [N, Ho, Wo, 64])
        x = ops.maxpool2d_nhwc_f32_gpu(x, kernel=3, stride=2, padding=1)
        for layer in range(1, 5):
            for blk in range(2):
                p = f"layer{layer}.{blk}"
                stride = 2 if layer > 1 and blk == 0 else 1
                sc = conv(x, p + ".downsample.0", stride=stride, relu=False) if p + ".downsample.0.w" in t else x
                h = conv(x, p + ".conv1", stride=stride, padding=1)
                x = conv(h, p + ".conv2", padding=1, residual=sc)
        x = ops.global_avgpool_nhwc_f32_gpu(x)
        return ops.to_cpu(ops.gemm_bias_f32_gpu(x, t["fc.w"], t["fc.b"]))

    return lk.compile(fn, input_types=[lk.TensorType(lk.f32, [N, 3, H, W])])
