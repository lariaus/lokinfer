import os, statistics, subprocess, time, torch, torch.nn as nn

from pathlib import Path

# downloads: <work_dir>/dl ($LOKINFER_WORK_DIR, default .cache/work in the repo)
DL = os.path.join(os.environ.get("LOKINFER_WORK_DIR", str(Path(__file__).resolve().parents[3] / ".cache" / "work")), "dl")


class MLP(nn.Module):
    # key layout matches Vecrist/mnist-pytorch-digit-classifiers (layers.1/3/5)
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.layers(x)


class G(nn.Module):  # PyTorch DCGAN tutorial / examples/dcgan Generator
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, x):
        return self.main(x)


class D(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)


def load_mlp():
    m = MLP()
    m.load_state_dict(torch.load(f"{DL}/vec/mlp.pth", map_location="cpu", weights_only=True))
    return m.eval()


def load_g():
    m = G()
    m.load_state_dict(torch.load(f"{DL}/hus/generator.pth", map_location="cpu", weights_only=True))
    return m.eval()


def load_d():
    m = D()
    m.load_state_dict(torch.load(f"{DL}/hus/discriminator.pth", map_location="cpu", weights_only=True))
    return m.eval()


def others_cpu():
    """Total %CPU of all processes except this one (ps snapshot)."""
    me = os.getpid()
    out = subprocess.run(["ps", "-Ao", "pid=,pcpu="], capture_output=True, text=True).stdout
    tot = 0.0
    for line in out.splitlines():
        pid, pc = line.split()
        if int(pid) != me:
            tot += float(pc)
    return tot


def wait_quiet(limit=200.0):
    """Block until other processes use < limit %CPU for 3 consecutive 1s polls (other agents benchmark too)."""
    ok = 0
    waited = 0
    while ok < 3:
        if others_cpu() < limit:
            ok += 1
        else:
            ok = 0
            waited += 1
            if waited % 60 == 1:
                print(f"  [waiting for quiet machine, others at {others_cpu():.0f}% CPU]", flush=True)
        time.sleep(1)


def timeit(fn, dev, warmup=10, iters=50, rounds=5):
    """Median-of-rounds of mean per-iter time in ms. Returns (median, min, max)."""
    sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync()
        res = []
        while len(res) < rounds:
            wait_quiet()
            t = time.perf_counter()
            for _ in range(iters):
                fn()
            sync()
            dt = (time.perf_counter() - t) / iters * 1e3
            if others_cpu() < 200.0:  # discard rounds contaminated by other load
                res.append(dt)
    return statistics.median(res), min(res), max(res)


def auto_iters(fn, dev, budget_ms=400):
    """Pick an iteration count so each round takes ~budget_ms."""
    sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)
    with torch.inference_mode():
        for _ in range(3):
            fn()
        sync()
        t = time.perf_counter()
        fn()
        sync()
        dt = (time.perf_counter() - t) * 1e3
    return max(3, min(500, int(budget_ms / max(dt, 1e-3))))
