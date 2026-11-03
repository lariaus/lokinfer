"""PyTorch eager f32 baselines (CPU + MPS) for lokinfer's port targets.

usage: python bench.py [model ...]   (models: mlp resnet18 mobilenetv2 dcgan_g dcgan_d gpt2)
Prints one line per (model, config, device): median / min / max ms per forward.
"""
import sys, json, torch
import torchvision
from common import load_mlp, load_g, load_d, timeit, auto_iters

torch.manual_seed(0)
DEVS = ["cpu", "mps"]
OUT = []


def report(model, cfg, dev, r):
    med, lo, hi = r
    OUT.append(dict(model=model, cfg=cfg, dev=dev, med=med, lo=lo, hi=hi))
    print(f"{model:12s} {cfg:28s} {dev:4s} median {med:9.3f} ms  (min {lo:.3f}, max {hi:.3f})", flush=True)


def bench_module(name, m, shapes, conv=True):
    for dev in DEVS:
        md = m.to(dev)
        for cfg, shape in shapes:
            x = torch.randn(*shape, device=dev)
            fn = lambda: md(x)
            report(name, cfg, dev, timeit(fn, dev, iters=auto_iters(fn, dev)))
    if conv:
        # best-effort tuned PyTorch CPU eager: NNPACK off (pathological on arm64 macOS) + channels_last
        mc = m.to("cpu").to(memory_format=torch.channels_last)
        with torch.backends.nnpack.flags(enabled=False):
            for cfg, shape in shapes:
                x = torch.randn(*shape).contiguous(memory_format=torch.channels_last)
                fn = lambda: mc(x)
                report(name, cfg, "cpu*", timeit(fn, "cpu", iters=auto_iters(fn, "cpu")))
        m.to(memory_format=torch.contiguous_format)


def run_mlp():
    bench_module("mnist_mlp", load_mlp(), [("b1", (1, 1, 28, 28)), ("b64", (64, 1, 28, 28)), ("b1024", (1024, 1, 28, 28))], conv=False)


def run_resnet18():
    m = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    bench_module("resnet18", m, [("b1 224", (1, 3, 224, 224)), ("b8 224", (8, 3, 224, 224))])


def run_mobilenetv2():
    m = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V2").eval()
    bench_module("mobilenetv2", m, [("b1 224", (1, 3, 224, 224)), ("b8 224", (8, 3, 224, 224))])


def run_dcgan_g():
    bench_module("dcgan_G", load_g(), [("b1 z100", (1, 100, 1, 1)), ("b64 z100", (64, 100, 1, 1))])


def run_dcgan_d():
    bench_module("dcgan_D", load_d(), [("b1 64x64", (1, 3, 64, 64)), ("b64 64x64", (64, 3, 64, 64))])


def run_gpt2():
    from transformers import GPT2LMHeadModel, DynamicCache
    base = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="sdpa").eval()
    for dev in DEVS:
        m = base.to(dev)
        # prefill: full forward, logits for all positions (HF default) and last-only
        for T in [1, 8, 32, 128, 512, 1024]:
            ids = torch.randint(0, 50257, (1, T), device=dev)
            fn = lambda: m(input_ids=ids, use_cache=False)
            report("gpt2", f"prefill T={T} all-logits", dev, timeit(fn, dev, warmup=5, iters=auto_iters(fn, dev), rounds=5))
            fn = lambda: m(input_ids=ids, use_cache=False, logits_to_keep=1)
            report("gpt2", f"prefill T={T} last-logit", dev, timeit(fn, dev, warmup=5, iters=auto_iters(fn, dev), rounds=5))
        ids = torch.randint(0, 50257, (4, 128), device=dev)
        fn = lambda: m(input_ids=ids, use_cache=False, logits_to_keep=1)
        report("gpt2", "prefill B=4 T=128 last-logit", dev, timeit(fn, dev, warmup=5, iters=auto_iters(fn, dev), rounds=5))
        # decode: one token with KV cache of fixed past length P (cache cropped back each step)
        for P in [16, 128, 512, 1023]:
            with torch.inference_mode():
                cache = DynamicCache(config=m.config)
                m(input_ids=torch.randint(0, 50257, (1, P), device=dev), past_key_values=cache, use_cache=True)
            tok = torch.randint(0, 50257, (1, 1), device=dev)
            pos = torch.tensor([[P]], device=dev)

            def step():
                m(input_ids=tok, past_key_values=cache, position_ids=pos, use_cache=True, logits_to_keep=1)
                cache.crop(-1)
            report("gpt2", f"decode 1 tok, past={P}", dev, timeit(step, dev, warmup=5, iters=auto_iters(step, dev), rounds=5))


def run_mv2_variants():
    """MobileNetV2 CPU: isolate nnpack vs channels_last effects."""
    m = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V2").eval()
    for B in [1, 8]:
        x = torch.randn(B, 3, 224, 224)
        with torch.backends.nnpack.flags(enabled=False):
            fn = lambda: m(x)
            report("mobilenetv2", f"b{B} 224 nnpack-off NCHW", "cpu", timeit(fn, "cpu", iters=auto_iters(fn, "cpu")))
        mc = m.to(memory_format=torch.channels_last)
        xc = x.contiguous(memory_format=torch.channels_last)
        fn = lambda: mc(xc)
        report("mobilenetv2", f"b{B} 224 nnpack-on CL", "cpu", timeit(fn, "cpu", iters=auto_iters(fn, "cpu")))
        m.to(memory_format=torch.contiguous_format)


if __name__ == "__main__":
    print("torch", torch.__version__, "threads", torch.get_num_threads(), flush=True)
    names = sys.argv[1:] or ["mlp", "resnet18", "mobilenetv2", "dcgan_g", "dcgan_d", "gpt2"]
    for n in names:
        globals()[f"run_{n}"]()
    with open(f"results_{'_'.join(names)}.json", "w") as f:
        json.dump(OUT, f, indent=1)

