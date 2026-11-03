"""Apples-to-apples reference baselines on this Mac for the non-Kokoro port targets (research-models.md):
ResNet-18, MobileNetV2, DCGAN 64x64 G and D, GPT-2 124M. f32, eval mode, no grad, deterministic weights + inputs,
recorded into perf/results.json (docs/perf-dashboard.md).

Usage: .venv/bin/python bench/models/bench_other_baselines.py [--model resnet18|mobilenetv2|dcgan_g|dcgan_d|gpt2 ...]
       [--impl NAME ...] [--no-record] [--meta-only]
--model / --impl may repeat (default: everything). MLX runs in its own venv as a subprocess
(bench/models/runners/other_mlx.py); set mlx_python (lokinfer.local.json or $LOKINFER_MLX_PYTHON) to a Python with mlx + numpy.

Weights (same for every framework): torchvision ResNet18_Weights.IMAGENET1K_V1, MobileNet_V2_Weights.IMAGENET1K_V2
(research-models.md §3.1), HF hussamalafandi/DCGAN_CelebA @ 0c4e26b8 (generator.pth / discriminator.pth loaded
strictly into the pytorch/examples DCGAN classes), HF openai-community/gpt2 @ 607a30d7 (sdpa attention).
Inputs: np.random.default_rng(0) — conv nets standard_normal (already-normalized image / z), GPT-2 integers in
[0, 50257). Cases (X axis):
  ResNet-18 / MobileNetV2: "N=1 224x224", "N=8 224x224"   input [N,3,224,224] -> logits [N,1000]
  DCGAN G: "N=1", "N=64"   z [N,100,1,1] -> [N,3,64,64];   DCGAN D: "N=1", "N=64"   [N,3,64,64] -> [N]
  GPT-2: "prefill T=128", "prefill T=512": one forward over T ids, use_cache=False, last-position logits [1,50257];
         "decode P=512": one token at position 512 with a 512-token KV cache -> logits [1,50257] (cache built untimed;
         torch crops its DynamicCache back to 512 after each step, MLX concatenates without keeping)
Impls: torch eager (default CPU eager), torch eager tuned (ResNet-18 / DCGAN: NNPACK off + channels_last,
research-models.md §4.4; skipped for MobileNetV2, where it makes the depthwise convs 3-8x slower), torch.compile (CPU), torch MPS, torch.compile MPS, MLX (GPU; the faster of plain and mx.compile, notes say
which). Every output is verified against torch eager CPU (DCGAN D: on its pre-sigmoid logits, since the
sigmoid saturates to exactly 1.0 on these inputs): rel err = max|y - ref| / max|ref| must be <= 1e-3, else the
entry is NOT recorded. Timing: 3 warm-ups, then >= 10 runs and >= 2 s, median of per-call times; the GPU is synced
inside each timed call (torch.mps.synchronize / mx.eval). Inputs and outputs stay on the device (no copies timed).
"""
import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
import perfdb  # noqa: E402

sys.path.insert(0, str(ROOT / "python"))
from lokinfer import localconfig  # noqa: E402

MLX_PY = localconfig.get("mlx_python")  # lokinfer.local.json / $LOKINFER_MLX_PYTHON
RUNNER = Path(__file__).resolve().parent / "runners" / "other_mlx.py"
TOL = 1e-3
WARMUP, MIN_RUNS, MIN_S = 3, 10, 2.0
DCGAN_REPO, DCGAN_REV = "hussamalafandi/DCGAN_CelebA", "0c4e26b8498c0ccf1ff06f47e0810ff4aaf8e14a"
GPT2_REPO, GPT2_REV = "openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e"

MODELS = {  # key -> (dashboard name, cases, description)
    "resnet18": ("ResNet-18", ["N=1 224x224", "N=8 224x224"],
                 "torchvision resnet18 IMAGENET1K_V1, f32: normalized image [N,3,224,224] -> logits [N,1000]"),
    "mobilenetv2": ("MobileNetV2", ["N=1 224x224", "N=8 224x224"],
                    "torchvision mobilenet_v2 IMAGENET1K_V2, f32: normalized image [N,3,224,224] -> logits [N,1000]"),
    "dcgan_g": ("DCGAN G", ["N=1", "N=64"],
                "DCGAN 64x64 generator (HF hussamalafandi/DCGAN_CelebA), f32: z [N,100,1,1] -> image [N,3,64,64]"),
    "dcgan_d": ("DCGAN D", ["N=1", "N=64"],
                "DCGAN 64x64 discriminator (HF hussamalafandi/DCGAN_CelebA), f32: image [N,3,64,64] -> prob [N]"),
    "gpt2": ("GPT-2 124M", ["prefill T=128", "prefill T=512", "decode P=512"],
             "HF openai-community/gpt2 f32 (sdpa): prefill = one forward over T ids -> last-position logits; "
             "decode = 1 token with a 512-token KV cache -> logits"),
}


class G(nn.Module):  # pytorch/examples dcgan Generator (nz=100, ngf=64, nc=3)
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


class D(nn.Module):  # pytorch/examples dcgan Discriminator (ndf=64, nc=3)
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


def load_model(key):
    import torchvision
    if key == "resnet18":
        return torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval(), \
            "torchvision ResNet18_Weights.IMAGENET1K_V1"
    if key == "mobilenetv2":
        return torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2).eval(), \
            "torchvision MobileNet_V2_Weights.IMAGENET1K_V2"
    if key in ("dcgan_g", "dcgan_d"):
        from huggingface_hub import hf_hub_download
        m, f = (G(), "generator.pth") if key == "dcgan_g" else (D(), "discriminator.pth")
        m.load_state_dict(torch.load(hf_hub_download(DCGAN_REPO, f, revision=DCGAN_REV), map_location="cpu",
                                     weights_only=True))
        return m.eval(), f"HF {DCGAN_REPO}@{DCGAN_REV[:8]} {f}"
    from transformers import GPT2LMHeadModel
    m = GPT2LMHeadModel.from_pretrained(GPT2_REPO, revision=GPT2_REV, attn_implementation="sdpa",
                                        dtype=torch.float32).eval()
    return m, f"HF {GPT2_REPO}@{GPT2_REV[:8]} (sdpa)"


def make_inputs(key):
    """case name -> dict of numpy inputs (deterministic)."""
    cases = MODELS[key][1]
    out = {}
    for c in cases:
        r = np.random.default_rng(0)
        if key in ("resnet18", "mobilenetv2"):
            n = int(c.split()[0][2:])
            out[c] = {"x": r.standard_normal((n, 3, 224, 224)).astype(np.float32)}
        elif key == "dcgan_g":
            out[c] = {"x": r.standard_normal((int(c[2:]), 100, 1, 1)).astype(np.float32)}
        elif key == "dcgan_d":
            out[c] = {"x": r.standard_normal((int(c[2:]), 3, 64, 64)).astype(np.float32)}
        elif c.startswith("prefill"):
            out[c] = {"x": r.integers(0, 50257, (1, int(c.split("=")[1]))).astype(np.int64)}
        else:
            P = int(c.split("=")[1])
            ids = r.integers(0, 50257, (1, P + 1)).astype(np.int64)
            out[c] = {"ctx": ids[:, :P], "x": ids[:, P:]}
    return out


def sync_for(dev):
    return torch.mps.synchronize if dev == "mps" else (lambda: None)


def timeit(fn, sync):
    for _ in range(WARMUP):
        fn()
    sync()
    ts, t0 = [], time.perf_counter()
    while len(ts) < MIN_RUNS or (time.perf_counter() - t0 < MIN_S and len(ts) < 1000):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3, len(ts)


def others_cpu():
    me = os.getpid()
    out = subprocess.run(["ps", "-Ao", "pid=,pcpu="], capture_output=True, text=True).stdout
    return sum(float(pc) for pid, pc in (ln.split() for ln in out.splitlines()) if int(pid) != me)


def wait_quiet(limit=200.0, max_wait_s=120):
    """Other sessions benchmark too: wait (bounded) for < limit %CPU elsewhere; returns the load seen."""
    t0 = time.time()
    while (load := others_cpu()) >= limit and time.time() - t0 < max_wait_s:
        print(f"  [waiting for a quiet machine: others at {load:.0f}% CPU]", flush=True)
        time.sleep(5)
    return load


def rel_err(y, ref):
    y, ref = np.asarray(y, np.float64), np.asarray(ref, np.float64)
    assert y.shape == ref.shape, (y.shape, ref.shape)
    return float(np.abs(y - ref).max() / np.abs(ref).max())


def torch_runner(key, m, dev, inp):
    """Returns fn() -> output tensor for one case, on device dev; m is already on dev."""
    x = torch.from_numpy(inp["x"]).to(dev)
    if key != "gpt2":
        if getattr(m, "_cl", False):
            x = x.contiguous(memory_format=torch.channels_last)
        return lambda: m(x)
    if "ctx" not in inp:
        return lambda: m(input_ids=x, use_cache=False, logits_to_keep=1).logits[:, -1]
    from transformers import DynamicCache
    P = inp["ctx"].shape[1]
    cache = DynamicCache(config=m.config)
    base = m._orig_mod if hasattr(m, "_orig_mod") else m
    base(input_ids=torch.from_numpy(inp["ctx"]).to(dev), past_key_values=cache, use_cache=True)
    pos = torch.tensor([[P]], device=dev)

    def step():
        y = m(input_ids=x, past_key_values=cache, position_ids=pos, use_cache=True, logits_to_keep=1).logits[:, -1]
        cache.crop(-1)  # back to the P-token cache
        return y
    return step


def logits_variant(key, base):
    """DCGAN D saturates (sigmoid == 1.0 exactly on these inputs), so every framework is verified on the
    pre-sigmoid logits: the same model with its final Sigmoid replaced by Identity, run through the same impl."""
    if key != "dcgan_d":
        return None
    v = copy.deepcopy(base)
    v.main[12] = nn.Identity()
    return v


def prep(mod, impl, dev):
    mod = mod.to(dev)
    mod._cl = impl == "torch eager tuned"
    if mod._cl:
        mod = mod.to(memory_format=torch.channels_last)
    return torch.compile(mod) if "compile" in impl else mod


def run_torch_impl(key, impl, base, inputs, refs, record, state, wsrc):
    dev = "mps" if "MPS" in impl else "cpu"
    device = "Mac GPU" if dev == "mps" else "Mac CPU"
    ctx = torch.backends.nnpack.flags(enabled=impl != "torch eager tuned")
    m = prep(base, impl, dev)
    vbase = logits_variant(key, base)
    vm = prep(vbase, impl, dev) if vbase is not None else m
    name = MODELS[key][0]
    with ctx:
        for case, inp in inputs.items():
            try:
                fn = torch_runner(key, m, dev, inp)
                t_c = time.perf_counter()
                y = fn()
                sync_for(dev)()
                first_s = time.perf_counter() - t_c
                if vm is not m:
                    y = torch_runner(key, vm, dev, inp)()
                err = rel_err(y.float().cpu().numpy(), refs[case])
            except Exception as e:  # noqa: BLE001 — a framework that can't run a case is reported, not recorded
                print(f"{name:>11} {impl:>18} {case:>14}: FAILED ({type(e).__name__}: {str(e)[:200]})", flush=True)
                continue
            if err > TOL:
                print(f"{name:>11} {impl:>18} {case:>14}: NOT RECORDED, rel err {err:.2e} > {TOL}", flush=True)
                continue
            load = wait_quiet()
            ms, n = timeit(fn, sync_for(dev))
            print(f"{name:>11} {impl:>18} {case:>14}: {ms:9.3f} ms  ({n} runs, rel err {err:.1e}, first call "
                  f"{first_s:.1f} s, others {load:.0f}% CPU)", flush=True)
            if record:
                extra = {"torch eager tuned": "; NNPACK off + channels_last",
                         "torch.compile": "; torch.compile(model) default mode",
                         "torch.compile MPS": "; torch.compile(model) default mode"}.get(impl, "")
                perfdb.record(name, device, case, impl, ms, state=state,
                              notes=f"torch {torch.__version__}, {torch.get_num_threads()} threads{extra}; {wsrc}; "
                                    f"f32, inputs default_rng(0); verified vs torch eager CPU: rel err {err:.1e}; "
                                    f"median of {n} runs; other processes at {load:.0f}% CPU when it started")
    base._cl = False
    base.to("cpu")
    if key != "gpt2":
        base.to(memory_format=torch.contiguous_format)


def run_mlx(key, base, inputs, refs, record, state, wsrc):
    name, cases = MODELS[key][0], list(inputs)
    if not MLX_PY or not Path(MLX_PY).exists():
        print(f"MLX: no Python at {MLX_PY} (set mlx_python in lokinfer.local.json or $LOKINFER_MLX_PYTHON)")
        return
    with tempfile.TemporaryDirectory() as d:
        if key == "gpt2":
            from huggingface_hub import hf_hub_download
            wpath = hf_hub_download(GPT2_REPO, "model.safetensors", revision=GPT2_REV)
        else:
            wpath = f"{d}/w.npz"
            np.savez(wpath, **{k: v.detach().cpu().numpy() for k, v in base.state_dict().items()})
        arrs = {}
        kinds = []
        for i, c in enumerate(cases):
            arrs[f"case_{i}"] = inputs[c]["x"]
            if "ctx" in inputs[c]:
                arrs[f"ctx_{i}"] = inputs[c]["ctx"]
            kinds.append([c, "decode" if "ctx" in inputs[c] else ("prefill" if key == "gpt2" else "x")])
        np.savez(f"{d}/in.npz", **arrs)
        wait_quiet()
        out = subprocess.run([MLX_PY, "-I", str(RUNNER), key, wpath, f"{d}/in.npz", f"{d}/out.npz", json.dumps(kinds),
                              str(WARMUP), str(MIN_RUNS), str(MIN_S)], capture_output=True, text=True)
        if out.returncode:
            print(f"MLX {name}: runner failed\n{out.stderr[-3000:]}")
            return
        res = np.load(f"{d}/out.npz")
        res = {k: res[k] for k in res.files}
    ver = "?"
    for line in out.stdout.splitlines():
        r = json.loads(line)
        if "case" not in r:
            ver = r.get("mlx")
            continue
        i = cases.index(r["case"])
        errs = {mode: rel_err(res[f"{mode}_{i}"], refs[r["case"]]) for mode in ("eager", "compiled")}
        ok = [m for m in ("eager", "compiled") if errs[m] <= TOL and r["f32"]]
        if not ok:
            print(f"{name:>11} {'MLX':>18} {r['case']:>14}: NOT RECORDED, rel err {errs} (f32 {r['f32']})")
            continue
        best = min(ok, key=lambda m: r[f"{m}_ms"])
        ms = r[f"{best}_ms"]
        print(f"{name:>11} {'MLX':>18} {r['case']:>14}: {ms:9.3f} ms  ({best}; plain {r['eager_ms']:.3f} ms err "
              f"{errs['eager']:.1e}, mx.compile {r['compiled_ms']:.3f} ms err {errs['compiled']:.1e})", flush=True)
        if record:
            perfdb.record(name, "Mac GPU", r["case"], "MLX", ms, state=state,
                          notes=f"mlx {ver}, own exact transcription (bench/models/runners/other_mlx.py), "
                                f"{'mx.compile' if best == 'compiled' else 'plain (no mx.compile)'} was faster "
                                f"(plain {r['eager_ms']:.3f} ms, mx.compile {r['compiled_ms']:.3f} ms); {wsrc}; f32, "
                                f"inputs default_rng(0); verified vs torch eager CPU: rel err {errs[best]:.1e}; "
                                f"median of {r[f'{best}_runs']} runs")


def write_meta(keys):
    db = perfdb.load()
    models = db.setdefault("meta", perfdb.DEFAULT_META).setdefault("models", {})
    for k in keys:
        name, cases, desc = MODELS[k]
        models[name] = {"cases": cases, "description": desc}
    perfdb._save(db)


ALL_IMPLS = ["torch eager", "torch eager tuned", "torch.compile", "torch MPS", "torch.compile MPS", "MLX"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", choices=list(MODELS))
    ap.add_argument("--impl", action="append", choices=ALL_IMPLS)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--meta-only", action="store_true")
    a = ap.parse_args()
    keys, impls, record = a.model or list(MODELS), a.impl or ALL_IMPLS, not a.no_record
    if record:
        write_meta(keys)
    if a.meta_only:
        return
    state = perfdb.machine_state()
    print("machine:", state, "| torch", torch.__version__, "threads", torch.get_num_threads(), flush=True)
    if state["low_power_mode"]:
        print("WARNING: Low Power Mode is ON: baselines measured now are not valid (CLAUDE.md)")
    torch.manual_seed(0)
    with torch.inference_mode():
        for key in keys:
            base, wsrc = load_model(key)
            inputs = make_inputs(key)
            refs = {}
            vref = logits_variant(key, base) or base
            for c, inp in inputs.items():  # the reference: torch eager CPU (DCGAN D: its pre-sigmoid logits)
                refs[c] = torch_runner(key, vref, "cpu", inp)().float().numpy().copy()
            for impl in impls:
                # "tuned" (NNPACK off + channels_last) only helps the dense convs; MobileNetV2's depthwise convs get
                # 3-8x slower with it (measured 264 vs 41 ms at N=1), so it is not a tuned baseline there
                if impl == "torch eager tuned" and key in ("gpt2", "mobilenetv2"):
                    continue
                if impl == "MLX":
                    run_mlx(key, base, inputs, refs, record, state, wsrc)
                else:
                    run_torch_impl(key, impl, base, inputs, refs, record, state, wsrc)
            del base
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
