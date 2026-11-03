"""Evaluate the Vecrist pretrained MLP on MNIST test, and time a fallback from-scratch training run."""
import time, torch, torch.nn as nn, torchvision
from common import DL, MLP, load_mlp

torch.manual_seed(0)
root = f"{DL}/mnist"
mean, std = 0.1307, 0.3081


def load(train):
    ds = torchvision.datasets.MNIST(root, train=train, download=True)
    x = ds.data.float().div(255).sub(mean).div(std).unsqueeze(1)
    return x, ds.targets


xte, yte = load(False)
xtr, ytr = load(True)


def acc(m, x, y):
    with torch.inference_mode():
        return (m(x).argmax(1) == y).float().mean().item()


m = load_mlp()
print(f"Vecrist pretrained (normalized input): test acc {acc(m, xte, yte):.4f}")
print(f"Vecrist pretrained (raw [0,1] input):  test acc {acc(m, xte * std + mean, yte):.4f}")

# fallback recipe: Adam lr 1e-3, bs 128, 5 epochs, CPU
m2 = MLP()
opt = torch.optim.Adam(m2.parameters(), 1e-3)
lossf = nn.CrossEntropyLoss()
t = time.time()
for ep in range(5):
    perm = torch.randperm(len(xtr))
    for i in range(0, len(xtr), 128):
        idx = perm[i:i + 128]
        opt.zero_grad()
        lossf(m2(xtr[idx]), ytr[idx]).backward()
        opt.step()
    print(f"epoch {ep + 1}: {time.time() - t:.1f}s  test acc {acc(m2.eval(), xte, yte):.4f}")
    m2.train()
