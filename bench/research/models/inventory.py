"""Leaf-module op inventory (shapes at batch 1) + param counts + MAC counts for conv/linear."""
import collections, torch, torch.nn as nn, torchvision
from common import load_mlp, load_g, load_d


def inventory(name, m, x):
    rows, macs = [], [0]

    def hook(mod, inp, out):
        i = inp[0]
        o = out
        desc = type(mod).__name__
        if isinstance(mod, nn.Conv2d):
            desc += f"(cin={mod.in_channels},cout={mod.out_channels},k={mod.kernel_size[0]},s={mod.stride[0]},p={mod.padding[0]},g={mod.groups},bias={mod.bias is not None})"
            macs[0] += o.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        elif isinstance(mod, nn.ConvTranspose2d):
            desc += f"(cin={mod.in_channels},cout={mod.out_channels},k={mod.kernel_size[0]},s={mod.stride[0]},p={mod.padding[0]})"
            macs[0] += i.numel() * mod.out_channels * mod.kernel_size[0] * mod.kernel_size[1]
        elif isinstance(mod, nn.Linear):
            desc += f"({mod.in_features}->{mod.out_features})"
            macs[0] += o.numel() * mod.in_features
        rows.append((desc, tuple(i.shape), tuple(o.shape)))

    hs = [mod.register_forward_hook(hook) for mod in m.modules() if len(list(mod.children())) == 0]
    with torch.inference_mode():
        m(x)
    for h in hs:
        h.remove()
    print(f"\n===== {name}: params {sum(p.numel() for p in m.parameters()) / 1e6:.3f} M, MACs(b1) {macs[0] / 1e6:.2f} M")
    for d, i, o in rows:
        print(f"  {d:70s} {str(i):22s} -> {o}")
    c = collections.Counter(r[0].split("(")[0] for r in rows)
    print("  counts:", dict(c))


inventory("MNIST MLP", load_mlp(), torch.randn(1, 1, 28, 28))
inventory("DCGAN G", load_g(), torch.randn(1, 100, 1, 1))
inventory("DCGAN D", load_d(), torch.randn(1, 3, 64, 64))
inventory("ResNet-18", torchvision.models.resnet18().eval(), torch.randn(1, 3, 224, 224))
inventory("MobileNetV2", torchvision.models.mobilenet_v2().eval(), torch.randn(1, 3, 224, 224))
