"""Qwen3-ASR Q5: the f16 optimized GPU ops (kernels/gpu/f16_fused.metal) vs the composition of the basic f16 GPU kernels
they replace, on real shapes. Each side is one compiled graph (inputs = activations and weights, caches as persistent
buffers where the op writes them) that runs the op R times back to back (R = 8 for short ops, fewer for long ones,
so a call is >= ~5 ms when possible: the GPU then runs at its sustained clock, as inside the model, and the runtime
overhead is amortized); the reported time is the call's time / R.

Mac: warm-up >= 1 s, then >= 30 runs and >= 2 s, median (run under /tmp/lokinfer-macbench.lock).
iPhone (`--iphone`, under /tmp/lokinfer-phone.lock, after scripts/ios_deploy.sh): the remote executor's RUN2 (inputs
stored on the phone as blobs once), 5 warm-up + N timed calls (N so that ~2 s are measured), median.

--cpu: the same cases with the CPU ops (`lokinfer.ops.f16`): the Q5 CPU kernel table (Mac / iPhone).
Usage: .venv/bin/python bench/kernels/bench_f16_fused_gpu.py [--cpu] [--iphone] [--only SUBSTRING] [--mps] [--opt-only]
  --mps: also time torch MPS f16 matmuls (MPSGraph) for the GEMM shapes (Mac only; the MPS-vs-own-kernel check).
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from bench_kokoro import machine_state, timeit  # noqa: E402

import lokinfer as lk  # noqa: E402
from lokinfer.ops import f16 as _f16_cpu  # noqa: E402
from lokinfer.ops import f16_gpu as _f16_gpu  # noqa: E402

DEV = "gpu"  # --cpu switches every op below to its CPU twin


class _Ops:
    """`F.<name>_f16_gpu` resolves to the CPU op `<name>` of lokinfer.ops.f16 when DEV is "cpu" (same signatures)."""

    def __getattr__(self, name):
        if DEV == "cpu":
            return getattr(_f16_cpu, name.removesuffix("_f16_gpu"))
        return getattr(_f16_gpu, name)


F = _Ops()

f16, i64 = lk.f16, lk.i64


def TT(dt, *shape):
    return lk.TensorType(dt, list(shape))


# Each case: (name, input specs [(dtype, shape, scale) or ("pos", value)], opt fn, basic fn). The fns take the GPU-side
# inputs (pos stays on the CPU) and return one GPU tensor.

def gemm_case(name, m, k, n, kind):
    sx, sw = 1.0, 1 / np.sqrt(k)
    specs = [(f16, (m, k), sx), (f16, (k, n), sw)]
    if kind == "plain":
        return name, specs, F.gemm_opt_f16_gpu, F.gemm_f16_gpu
    if kind == "add":
        specs.append((f16, (m, n), 1.0))
        return name, specs, F.gemm_add_opt_f16_gpu, lambda x, w, r: F.add_f16_gpu(r, F.gemm_f16_gpu(x, w))
    specs.append((f16, (n,), 0.1))
    if kind == "bias":
        return name, specs, F.gemm_bias_opt_f16_gpu, F.gemm_bias_f16_gpu
    if kind == "gelu":
        return name, specs, F.gemm_bias_gelu_opt_f16_gpu, lambda x, w, b: F.gelu_erf_f16_gpu(F.gemm_bias_f16_gpu(x, w, b))
    specs.append((f16, (m, n), 1.0))
    if kind == "bias_add":
        return name, specs, F.gemm_bias_add_opt_f16_gpu, lambda x, w, b, r: F.add_f16_gpu(r, F.gemm_bias_f16_gpu(x, w, b))
    if kind == "clamp":
        return (name, specs, lambda x, w, b, r: F.gemm_bias_add_clamp_opt_f16_gpu(x, w, b, r, 64504.0),
                lambda x, w, b, r: F.add_clamp_f16_gpu(r, F.gemm_bias_f16_gpu(x, w, b), lim=64504.0))
    raise ValueError(kind)


def swiglu_case(name, m, k, n):
    s = 1 / np.sqrt(k)
    specs = [(f16, (m, k), 1.0), (f16, (k, 2 * n), s), (f16, (k, n), s), (f16, (k, n), s)]
    return (name, specs, lambda x, w, wg, wu: F.gemm_swiglu_opt_f16_gpu(x, w),
            lambda x, w, wg, wu: F.silu_mul_f16_gpu(F.gemm_f16_gpu(x, wg), F.gemm_f16_gpu(x, wu)))


def qkr_case(name, t, pos):
    cap = pos + t

    def opt(qkv, qp, kp, vp, qg, kg, p):
        kc, vc = (lk.persistent_buffer(f16, [cap, 1024], device=DEV) for _ in range(2))
        return F.qknorm_rope_kv_opt_f16_gpu(qkv, qg, kg, p, kc, vc)

    def basic(qkv, qp, kp, vp, qg, kg, p):
        kc, vc = (lk.persistent_buffer(f16, [cap, 1024], device=DEV) for _ in range(2))
        k = F.rope_f16_gpu(F.rmsnorm_heads_f16_gpu(kp, kg), p)
        F.kv_write_f16_gpu(kc, vc, k, vp, p)
        return F.rope_f16_gpu(F.rmsnorm_heads_f16_gpu(qp, qg), p)

    specs = [(f16, (t, 4096), 2.0), (f16, (t, 2048), 2.0), (f16, (t, 1024), 2.0), (f16, (t, 1024), 1.0),
             (f16, (128,), 0.5), (f16, (128,), 0.5), ("pos", pos)]
    return name, specs, opt, basic


def gqa_case(name, t, pos):
    cap = pos + t
    specs = [(f16, (t, 2048), 1.0), (f16, (cap, 1024), 1.0), (f16, (cap, 1024), 1.0), ("pos", pos)]
    return name, specs, F.attention_gqa_opt_f16_gpu, F.attention_gqa_f16_gpu


def window_case(name, s):
    specs = [(f16, (s, 3072), 1.0), (f16, (s, 1024), 1.0), (f16, (s, 1024), 1.0), (f16, (s, 1024), 1.0)]
    return (name, specs, lambda qkv, q, k, v: F.attention_window_qkv_opt_f16_gpu(qkv),
            lambda qkv, q, k, v: F.attention_window_f16_gpu(q, k, v))


def lm_head_case(name):
    specs = [(f16, (1, 2048), 1.0), (f16, (151_936, 2048), 0.05)]
    return name, specs, F.lm_head_last_opt_f16_gpu, F.lm_head_last_f16_gpu


def cnn_case(name, n):
    co = 480
    specs = [(lk.f32, (n, 128, 100), 0.8), (f16, (co, 3, 3, 1), 0.5), (f16, (co,), 0.1),
             (f16, (co, 3, 3, co), 1.5 / np.sqrt(9 * co)), (f16, (co,), 0.1),
             (f16, (co, 3, 3, co), 1.5 / np.sqrt(9 * co)), (f16, (co,), 0.1), (f16, (16 * co, 1024), 1 / np.sqrt(7680))]
    return name, specs, F.audio_cnn_opt_f16_gpu, F.audio_cnn_f16_gpu


def cases(iphone: bool):
    out = []
    for m in (1, 44, 401, 2048):
        out.append(gemm_case(f"gemm qkv M={m} [2048x4096]", m, 2048, 4096, "plain"))
    for m in (1, 2048):
        out.append(gemm_case(f"gemm_add o M={m} [2048x2048]", m, 2048, 2048, "add"))
        out.append(swiglu_case(f"gemm_swiglu M={m} [2048x2x6144]", m, 2048, 6144))
        out.append(gemm_case(f"gemm_add down M={m} [6144x2048]", m, 6144, 2048, "add"))
    na = 3900  # encoder rows of the 300 s clip
    out.append(gemm_case(f"gemm_bias enc qkv M={na} [1024x3072]", na, 1024, 3072, "bias"))
    out.append(gemm_case(f"gemm_bias_add enc o M={na} [1024x1024]", na, 1024, 1024, "bias_add"))
    out.append(gemm_case(f"gemm_bias_gelu enc fc1 M={na} [1024x4096]", na, 1024, 4096, "gelu"))
    out.append(gemm_case(f"gemm_bias_add_clamp enc fc2 M={na} [4096x1024]", na, 4096, 1024, "clamp"))
    out.append(lm_head_case("lm_head_last V=151936 D=2048"))
    out.append(qkr_case("qknorm_rope_kv T=1 pos=1850", 1, 1850))
    out.append(qkr_case("qknorm_rope_kv T=2048 pos=0", 2048, 0))
    for pos in (52, 1850, 4900) + (() if iphone else (20_000,)):
        out.append(gqa_case(f"attention_gqa decode pos={pos}", 1, pos))
    out.append(gqa_case("attention_gqa prefill T=2048 pos=0", 2048, 0))
    out.append(gqa_case("attention_gqa prefill T=401 pos=3517", 401, 3517))
    if not iphone:
        out.append(gqa_case("attention_gqa prefill T=2048 pos=13570", 2048, 13_570))
    for s in (3900,) + (() if iphone else (15_600,)):
        out.append(window_case(f"attention_window S={s}", s))
    for n in (5, 30):
        out.append(cnn_case(f"audio_cnn n={n}", n))
    return out


def build(fn, specs, reps=1):
    types = [TT(i64, 1) if s[0] == "pos" else TT(s[0], *s[1]) for s in specs]

    def wrapped(*xs):
        if DEV == "cpu":
            return tuple(fn(*xs) for _ in range(reps))
        gx = [x if s[0] == "pos" else lk.ops.to_gpu(x) for x, s in zip(xs, specs)]
        outs = [fn(*gx) for _ in range(reps)]  # all on the GPU first: one segment
        return tuple(lk.ops.to_cpu(o) for o in outs)

    return lk.compile(wrapped, input_types=types)


def mac_arrays(specs, seed=0):
    r = np.random.default_rng(seed)
    return [np.array([s[1]], np.int64) if s[0] == "pos" else
            (s[2] * r.standard_normal(s[1])).astype(np.float16 if s[0] is f16 else np.float32) for s in specs]


def mac_inputs(specs, seed=0):
    return [lk.from_numpy(a) for a in mac_arrays(specs, seed)]


def iphone_inputs(dev, specs):
    """pos inline; f32 generated on the device; f16 (the device's generator has no f16) stored once as blobs (kept by
    the phone across sessions)."""
    from lokinfer.remote import Generated

    out = []
    for i, (s, a) in enumerate(zip(specs, mac_arrays(specs))):
        if s[0] == "pos":
            out.append(a)
        elif s[0] is f16:
            out.append(dev.put_tensor(a))
        else:
            out.append(Generated(s[1], seed=i + 1, dtype=np.float32, b=s[2]))
    return out


def connect_iphone():
    from lokinfer.remote import RemoteDevice

    udid = os.environ.get("LOKINFER_IPHONE")
    if not udid:
        out = subprocess.run(["xcrun", "devicectl", "list", "devices"], capture_output=True, text=True).stdout
        udid = next(m.group(0) for line in out.splitlines() if "physical" in line and "unavailable" not in line
                    for m in [re.search(r"[0-9A-F]{8}-[0-9A-F]{16}", line)] if m)
    return RemoteDevice.connect(f"{udid}.coredevice.local", timeout=900.0)


def time_iphone(dev, model, inputs):
    mid = dev.load(model)
    _, _, ts = dev.run2(mid, inputs, return_outputs=False, warmup=3, iters=3)
    per = float(np.median(ts)) / 1e9
    iters = int(min(200, max(10, 2.0 / max(per, 1e-6))))
    _, _, ts = dev.run2(mid, inputs, return_outputs=False, warmup=5, iters=iters)
    return float(np.median(ts)) / 1e6


def mps_gemms():
    import torch

    print("torch MPS f16 matmul (MPSGraph), x [M, K] @ w [K, N], same timing protocol:")
    for m, k, n in [(1, 2048, 4096), (1, 6144, 2048), (44, 2048, 4096), (401, 2048, 4096), (2048, 2048, 4096),
                    (2048, 2048, 12288), (3900, 1024, 4096), (3900, 4096, 1024)]:
        x = torch.randn(m, k, dtype=torch.float16, device="mps")
        w = (torch.randn(k, n, dtype=torch.float16, device="mps") / k ** 0.5)

        def run():
            torch.mm(x, w)
            torch.mps.synchronize()

        t = timeit(run)["median_ms"]
        print(f"  M={m} K={k} N={n}: {t:.3f} ms  ({2 * m * k * n / t / 1e9:.2f} TFLOP/s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iphone", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="the CPU ops instead of the GPU ones")
    ap.add_argument("--only", default="")
    ap.add_argument("--mps", action="store_true")
    ap.add_argument("--opt-only", action="store_true", help="time only the optimized op (kernel A/B runs)")
    a = ap.parse_args()
    global DEV
    DEV = "cpu" if a.cpu else "gpu"
    print(machine_state(), "device:", DEV, flush=True)
    dev = None
    for name, specs, opt, basic in cases(a.iphone):
        if a.only not in name:
            continue
        if a.iphone:  # a fresh session per case: its models (and their arenas) are freed with it
            if dev:
                dev.close()
            dev = connect_iphone()
        ins = iphone_inputs(dev, specs) if dev else mac_inputs(specs)

        def measure(fn):
            m = build(fn, specs)
            t1 = time_iphone(dev, m, ins) if dev else timeit(lambda: m(*ins), min_warm_s=0.3, min_run_s=0.5)["median_ms"]
            reps = int(min(8, max(1, np.ceil(5.0 / t1))))
            if reps == 1:
                return t1
            m = build(fn, specs, reps)
            t = time_iphone(dev, m, ins) if dev else timeit(lambda: m(*ins))["median_ms"]
            return t / reps

        to = measure(opt)
        tb = float("nan") if a.opt_only else measure(basic)
        st = ""
        if dev:  # the phone's state after the case (heat throttles; Low Power Mode throttles more)
            ds = dev.state()
            st = f"  [{ds.get('thermal')}{', LPM' if ds.get('low_power_mode') else ''}]"
        print(f"{name:48s} basic {tb:9.3f} ms  opt {to:9.3f} ms  {tb / to:6.2f}x{st}", flush=True)
    if a.mps and not dev:
        mps_gemms()


if __name__ == "__main__":
    main()
