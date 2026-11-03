"""lokinfer on the iPhone (remote executor in LokiHost; scripts/ios_deploy.sh), CPU and GPU graphs of every model,
recorded to the perf dashboard as "iPhone CPU" / "iPhone GPU" (impl "lokinfer" / "lokinfer GPU"; Kokoro's CPU graph is
"lokinfer fused"). Same cases and inputs as the Mac benches (bench_m6_models.py, bench_kokoro.py).
Usage: .venv/bin/python bench/models/bench_remote_models.py [kokoro resnet18 mobilenet_v2 dcgan gpt2 ...]
       [--udid UDID] [--no-record]"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
import perfdb  # noqa: E402
from models import dcgan, gpt2, mnist_mlp, mobilenet_v2, resnet18  # noqa: E402
from models.kokoro import model as K  # noqa: E402
from models.kokoro import reference as R  # noqa: E402

from lokinfer.remote import RemoteDevice  # noqa: E402

from lokinfer import localconfig  # noqa: E402


def img(n):
    return np.random.default_rng(0).standard_normal((n, 3, 224, 224)).astype(np.float32)


def suites():
    """key -> () -> (title, {case: [inputs]}, {impl: (compiled model, device)}, extra decode suite or None)"""
    def kokoro():
        w = K.fold_weights(R.load_model())
        cases = {}
        for name in perfdb.DEFAULT_META["models"]["Kokoro-82M"]["cases"]:  # the dashboard's Kokoro cases
            c = R.load_case(name)
            cases[name] = [np.concatenate([[0], c["ids"], [0]]).astype(np.int64), np.asarray(c["ref_s"], np.float32),
                           np.asarray(c["speed"], np.float32).reshape(()), np.array(0, np.int64)]
        return "Kokoro-82M", cases, {"lokinfer fused": (K.build(w), "iPhone CPU"),
                                     "lokinfer GPU": (K.build(w, gpu=True), "iPhone GPU")}, None

    def resnet():
        w = resnet18.fold_weights(resnet18.torch_reference())
        return "ResNet-18", {"N=1 224x224": [img(1)], "N=8 224x224": [img(8)]}, {
            "lokinfer": (resnet18.build(w), "iPhone CPU"),
            # planar padded layout + pre-packed SME2 weights + Winograd F(2x2, 3x3) (build_cnhw), iPhone 2026-11-02:
            # N=1 2.8 vs 4.3 ms, N=8 18.6 vs 25 — both recorded; the dashboard shows the best per case
            "lokinfer planar": (resnet18.build_cnhw(w), "iPhone CPU"),
            "lokinfer GPU": (resnet18.build_gpu(w), "iPhone GPU")}, None

    def mobilenet():
        b = mobilenet_v2.fold_weights(mobilenet_v2.torch_reference())
        return "MobileNetV2", {"N=1 224x224": [img(1)], "N=8 224x224": [img(8)]}, {
            "lokinfer": (mobilenet_v2.build(b), "iPhone CPU"),
            "lokinfer GPU": (mobilenet_v2.build(b, device="gpu"), "iPhone GPU")}, None

    def dcgan_g():
        g, _ = dcgan.torch_reference()
        L = dcgan.fold_generator(g)
        z = {f"N={n}": [np.random.default_rng(0).standard_normal((n, 100, 1, 1)).astype(np.float32)] for n in (1, 64)}
        return "DCGAN G", z, {"lokinfer": (dcgan.build_generator(L), "iPhone CPU"),
                              "lokinfer GPU": (dcgan.build_generator(L, device="gpu"), "iPhone GPU")}, None

    def dcgan_d():
        _, d = dcgan.torch_reference()
        L = dcgan.fold_discriminator(d)
        x = {f"N={n}": [np.tanh(np.random.default_rng(0).standard_normal((n, 3, 64, 64))).astype(np.float32)]
             for n in (1, 64)}
        return "DCGAN D", x, {"lokinfer": (dcgan.build_discriminator(L), "iPhone CPU"),
                              "lokinfer GPU": (dcgan.build_discriminator(L, device="gpu"), "iPhone GPU")}, None

    def gpt():
        sd = gpt2.weights(gpt2.torch_reference())
        ids = {f"prefill T={t}": [np.random.default_rng(0).integers(0, gpt2.VOCAB, t).astype(np.int64)]
               for t in (128, 512)}
        return "GPT-2 124M", ids, {
            "lokinfer": (gpt2.build(sd, last_only=True), "iPhone CPU"),
            "lokinfer GPU": (gpt2.build(sd, device="gpu", last_only=True), "iPhone GPU")}, {
            "lokinfer": (gpt2.build_decode(sd), "iPhone CPU"),
            "lokinfer GPU": (gpt2.build_decode(sd, device="gpu"), "iPhone GPU")}

    def mnist():
        w = mnist_mlp.weights_from_torch(mnist_mlp.torch_reference())
        xs = {f"B={b}": [np.random.default_rng(b).random((b, 1, 28, 28)).astype(np.float32)] for b in (1, 64, 1024)}
        return "MNIST MLP", xs, {"lokinfer": (mnist_mlp.build(w, "cpu"), "iPhone CPU"),
                                 "lokinfer accelerate": (mnist_mlp.build(w, "cpu-accelerate"), "iPhone CPU"),
                                 "lokinfer GPU": (mnist_mlp.build(w, "gpu"), "iPhone GPU")}, None

    return {"mnist": mnist, "kokoro": kokoro, "resnet18": resnet, "mobilenet_v2": mobilenet, "dcgan_g": dcgan_g, "dcgan_d": dcgan_d,
            "gpt2": gpt}


def median_ms(times_ns):
    t = sorted(x * 1e-6 for x in times_ns)
    return t[len(t) // 2]


def wait_nominal(dev, max_wait_s=60):
    """Short cool-down before each model, MLPerf-Mobile style (a break between tests, not between runs): wait until
    the phone is "nominal" again, at most `max_wait_s` (user decision 2026-11-01: "wait a bit, but not much")."""
    import time
    t0 = time.time()
    while dev.state().get("thermal") not in ("nominal", None) and time.time() - t0 < max_wait_s:
        time.sleep(10)


def timed_run(dev, mid, inputs):
    """Timing protocol (docs/perf-dashboard.md): ~1 s of warm-up, then >= 30 runs and
    >= ~2 s measured, median — so the GPU's clocks have ramped up (DVFS: a 1-kernel GPU call goes from ~290 us cold
    to < 50 us warm on the iPhone)."""
    start = dev.state()  # no cool-down wait (user decision 2026-11-01): thermal state is recorded, not waited on
    _, t = dev.run(mid, inputs, warmup=1, iters=3)
    est = max(median_ms(t), 0.01)
    warm = int(min(5000, max(3, 1000 / est)))
    iters = int(min(5000, max(30, 2000 / est)))
    _, times = dev.run(mid, inputs, warmup=warm, iters=iters)
    return median_ms(times), start, dev.state().get("thermal"), len(times)


def main():
    argv = sys.argv[1:]
    udid = argv[argv.index("--udid") + 1] if "--udid" in argv else localconfig.require("iphone_udid")
    want = [a for a in argv if not a.startswith("--") and a != udid]
    if "dcgan" in want:
        want += ["dcgan_g", "dcgan_d"]
    record = "--no-record" not in argv
    # --cpu-only / --gpu-only: one device per session, so a long sweep's second half doesn't run on a phone the
    # first half heated (2026-11-01: 10 Kokoro cases GPU-then-CPU left the CPU rows ~35-50 % slow at "fair").
    # --cases=a,b: only these case names (e.g. newly added dashboard cases)
    case_filter = next((set(a.split("=", 1)[1].split(",")) for a in argv if a.startswith("--cases=")), None)
    only = "iPhone CPU" if "--cpu-only" in argv else "iPhone GPU" if "--gpu-only" in argv else None
    dev = RemoteDevice.connect(f"{udid}.coredevice.local", timeout=900.0)  # long cases: >= 30 runs of ~1 s
    print("device:", dev.state(), flush=True)
    wait_nominal(dev, max_wait_s=300)  # one longer cool-down before the sweep (a hot start throttled 35/37 rows)
    print("start state:", dev.state(), flush=True)
    for key, make in suites().items():
        if want and key not in want:
            continue
        title, cases, variants, decode = make()
        # One capped break per model (not per case / run): longer before the heavy models, which heat the phone within
        # ~2 min; the references were all measured cool, so lokinfer is too (2026-11-01).
        wait_nominal(dev, max_wait_s=180 if key in ("kokoro", "gpt2") else 60)
        # GPU variants first: the long CPU runs (Kokoro rand510 is 1.3 s per call) heat the phone to "fair", which made
        # the GPU rows that followed ~25 % slow (real117 310 ms vs 242 ms cool, 2026-11-01); the CPU rows are much less
        # sensitive to the GPU runs before them.
        for impl, (model, device) in sorted(variants.items(), key=lambda kv: kv[1][1] != "iPhone GPU"):
            if only and device != only:
                continue
            mid = dev.load(model)
            for case, inputs in cases.items():
                if case_filter and case not in case_filter:
                    continue
                ms, st, end_thermal, n = timed_run(dev, mid, inputs)
                print(f"{title:>12} {case:>14} {device:>10} {impl:>14}: {ms:8.2f} ms  ({st.get('thermal')}, "
                      f"LPM {st.get('low_power_mode')})", flush=True)
                if record:
                    perfdb.record(title, device, case, impl, ms, state=st,
                                  notes=f"lokinfer via the remote executor; median of {n} runs after ~1 s warm-up; "
                                        f"thermal at start {st.get('thermal')}, at end {end_thermal}")
        if decode:
            toks = np.random.default_rng(0).integers(0, gpt2.VOCAB, 513).astype(np.int64)
            for impl, (model, device) in decode.items():
                if only and device != only:
                    continue
                mid = dev.load(model)
                for p in range(512):  # fill the persistent caches (model state lives on the phone)
                    dev.run(mid, [toks[p:p + 1], np.array([p], np.int64)])
                ms, st, end_thermal, n = timed_run(dev, mid, [toks[512:513], np.array([512], np.int64)])
                print(f"{title:>12} {'decode P=512':>14} {device:>10} {impl:>14}: {ms:8.2f} ms", flush=True)
                if record:
                    perfdb.record(title, device, "decode P=512", impl, ms, state=st,
                                  notes=f"lokinfer via the remote executor; one step at position 512 (cache filled); "
                                        f"median of {n} runs; thermal at start {st.get('thermal')}, at end {end_thermal}")
    dev.close()


if __name__ == "__main__":
    main()
