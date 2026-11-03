"""lokinfer forward timings for the M6 models (ResNet-18, MobileNetV2, DCGAN G / D, GPT-2 124M prefill) on CPU and
GPU, recorded to the perf dashboard. Baselines (torch eager / compile / MPS / MLX) come from
bench/models/bench_other_baselines.py with the same case names, inputs and weights.
Usage: .venv/bin/python bench/models/bench_m6_models.py [resnet18 mobilenet_v2 dcgan gpt2 ...] [--no-record]
       [--gpu-only | --cpu-only]"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import perfdb  # noqa: E402
from bench_kokoro import timeit  # noqa: E402
from models import dcgan, gpt2, mobilenet_v2, resnet18  # noqa: E402

import lokinfer as lk  # noqa: E402


def _img(n):
    return lk.from_numpy(np.random.default_rng(0).standard_normal((n, 3, 224, 224)).astype(np.float32))


def suites():
    """(key, title, {case: input}, {impl: (model, device)}, notes) per model, built lazily."""
    def resnet():
        w = resnet18.fold_weights(resnet18.torch_reference())
        return ("ResNet-18", {"N=1 224x224": _img(1), "N=8 224x224": _img(8)},
                {"lokinfer": (resnet18.build(w), "Mac CPU"), "lokinfer GPU": (resnet18.build_gpu(w), "Mac GPU")})

    def mobilenet():
        b = mobilenet_v2.fold_weights(mobilenet_v2.torch_reference())
        return ("MobileNetV2", {"N=1 224x224": _img(1), "N=8 224x224": _img(8)},
                {"lokinfer": (mobilenet_v2.build(b), "Mac CPU"),
                 "lokinfer GPU": (mobilenet_v2.build(b, device="gpu"), "Mac GPU")})

    def dcgan_g():
        g, _ = dcgan.torch_reference()
        L = dcgan.fold_generator(g)
        z = {f"N={n}": lk.from_numpy(np.random.default_rng(0).standard_normal((n, 100, 1, 1)).astype(np.float32))
             for n in (1, 64)}
        return ("DCGAN G", z, {"lokinfer": (dcgan.build_generator(L), "Mac CPU"),
                               "lokinfer GPU": (dcgan.build_generator(L, device="gpu"), "Mac GPU")})

    def dcgan_d():
        _, d = dcgan.torch_reference()
        L = dcgan.fold_discriminator(d)
        x = {f"N={n}": lk.from_numpy(np.tanh(np.random.default_rng(0).standard_normal((n, 3, 64, 64))).astype(np.float32))
             for n in (1, 64)}
        return ("DCGAN D", x, {"lokinfer": (dcgan.build_discriminator(L), "Mac CPU"),
                               "lokinfer GPU": (dcgan.build_discriminator(L, device="gpu"), "Mac GPU")})

    def gpt():
        sd = gpt2.weights(gpt2.torch_reference())
        ids = {f"prefill T={t}": lk.from_numpy(np.random.default_rng(0).integers(0, gpt2.VOCAB, t).astype(np.int64))
               for t in (128, 512)}
        # last_only: logits of the last position only, like the HF baseline (logits_to_keep=1)
        variants = {"lokinfer": (gpt2.build(sd, last_only=True), "Mac CPU"),
                    "lokinfer GPU": (gpt2.build(sd, device="gpu", last_only=True), "Mac GPU")}
        decode = {"lokinfer": (gpt2.build_decode(sd), "Mac CPU"),
                  "lokinfer GPU": (gpt2.build_decode(sd, device="gpu"), "Mac GPU")}
        return ("GPT-2 124M", ids, variants, decode)

    return {"resnet18": resnet, "mobilenet_v2": mobilenet, "dcgan": None, "dcgan_g": dcgan_g, "dcgan_d": dcgan_d,
            "gpt2": gpt}


def main():
    want = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "dcgan" in want:
        want += ["dcgan_g", "dcgan_d"]
    record = "--no-record" not in sys.argv
    only = "Mac GPU" if "--gpu-only" in sys.argv else "Mac CPU" if "--cpu-only" in sys.argv else None
    state = perfdb.machine_state()
    print("machine:", state)
    for key, make in suites().items():
        if make is None or (want and key not in want):
            continue
        title, cases, variants, *extra = make()
        if extra:  # GPT-2 decode: one step at position 512 after filling the caches with 512 tokens
            toks = np.random.default_rng(0).integers(0, gpt2.VOCAB, 513)
            for impl, (step, device) in extra[0].items():
                if only and device != only:
                    continue
                for p in range(512):
                    step(lk.from_numpy(toks[p:p + 1].astype(np.int64)), lk.from_numpy(np.array([p], np.int64)))
                tok, pos = lk.from_numpy(toks[512:513].astype(np.int64)), lk.from_numpy(np.array([512], np.int64))
                t = timeit(lambda: step(tok, pos), warmup=3, runs=25)["median_ms"]
                print(f"{title:>12} {'decode P=512':>14} {impl:>13}: {t:8.2f} ms", flush=True)
                if record:
                    perfdb.record(title, device, "decode P=512", impl, t, state=state,
                                  notes="one decode step at position 512 (512 cached tokens), persistent KV cache")
        for case, x in cases.items():
            for impl, (m, device) in variants.items():
                if only and device != only:
                    continue
                t = timeit(lambda m=m: m(x), warmup=3, runs=15)["median_ms"]
                print(f"{title:>12} {case:>14} {impl:>13}: {t:8.2f} ms", flush=True)
                if record:
                    perfdb.record(title, device, case, impl, t, state=state,
                                  notes="lokinfer, f32; inputs/weights as bench_other_baselines.py")


if __name__ == "__main__":
    main()
