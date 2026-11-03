"""MNIST MLP 784 -> 128 -> 64 -> 10 (docs/impl-notes/research-models.md §1).

Reference: a 6-line nn.Sequential matching the pretrained state dict
`Vecrist/mnist-pytorch-digit-classifiers @ 574952b0 MNIST_MLPmodel_weights.pth`.
lokinfer: PyTorch Linear weights are [out, in]; they're transposed to [in, out]
by a kernel on the model's device that the compiler hoists into init (runs once
at load). The GPU variant is GPU end to end, weights included.
"""
from __future__ import annotations

import numpy as np

import lokinfer as lk
from lokinfer import ops

HF_REPO = "Vecrist/mnist-pytorch-digit-classifiers"
HF_FILE = "MNIST_MLPmodel_weights.pth"
HF_REV = "574952b057f9e9858efdcddd1404a8c105cb5d7b"
MEAN, STD = 0.1307, 0.3081

# kernel set per variant: (gemm+bias+relu, gemm+bias)
VARIANTS = {
    "cpu": (ops.gemm_bias_relu_f32_cpu, ops.gemm_bias_f32_cpu),
    "cpu-accelerate": (ops.gemm_bias_relu_f32_cpu_accelerate, ops.gemm_bias_f32_cpu_accelerate),
    "gpu": (ops.gemm_bias_relu_f32_gpu, ops.gemm_bias_f32_gpu),
}


def torch_reference():
    import torch
    from huggingface_hub import hf_hub_download
    from torch import nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                        nn.Linear(64, 10))

        def forward(self, x):
            return self.layers(x)

    path = hf_hub_download(HF_REPO, HF_FILE, revision=HF_REV)
    m = MLP()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


def weights_from_torch(model) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}


def build(weights: dict[str, np.ndarray], variant: str = "cpu") -> lk.CompiledModel:
    """Input: normalized images [B, 1, 28, 28] f32 (B dynamic). Output: logits [B, 10]."""
    relu_layer, last_layer = VARIANTS[variant]
    gpu = variant == "gpu"
    device = "gpu" if gpu else "cpu"
    # Weights are resident on the device that uses them (explicit in the IR).
    t = {k: lk.from_numpy(np.ascontiguousarray(v), device=device) for k, v in weights.items()}
    transpose = ops.transpose_f32_gpu if gpu else ops.transpose_f32_cpu

    def weight(key):
        return transpose(t[key])  # kernel on a constant: hoisted into init, runs once at load

    def fn(x):
        h = ops.flatten(x)  # [B, 784], a free view
        h = relu_layer(h, weight("layers.1.weight"), t["layers.1.bias"])
        h = relu_layer(h, weight("layers.3.weight"), t["layers.3.bias"])
        return last_layer(h, weight("layers.5.weight"), t["layers.5.bias"])

    B = lk.Sym("B")
    return lk.compile(fn, input_types=[lk.TensorType(lk.f32, [B, 1, 28, 28])])
