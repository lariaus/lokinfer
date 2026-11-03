"""MLX (GPU) runner for bench_other_baselines.py: ResNet-18, MobileNetV2, DCGAN G/D, GPT-2 124M, f32.
Runs in a separate venv with mlx + numpy (not lokinfer's .venv):

    python -I other_mlx.py MODEL WEIGHTS INPUTS.npz OUT.npz CASES_JSON WARMUP MIN_RUNS MIN_SECONDS

MODEL: resnet18 | mobilenetv2 | dcgan_g | dcgan_d | gpt2. WEIGHTS: an .npz of the PyTorch state dict (conv nets)
or GPT-2's HF model.safetensors. INPUTS.npz holds case_<i> (the exact torch input, NCHW f32 or i64 ids); for the
GPT-2 decode case also ctx_<i> (the 512 cached ids). CASES_JSON: list of [case name, kind] (kind: "x", "prefill",
"decode"). Each model is a direct transcription of the PyTorch reference (same ops, unfused eval BatchNorm, NCHW
in / out with the NHWC transposes inside the timed call). Every case is timed twice: plain MLX (lazy graph, one
mx.eval per call) and mx.compile'd; both outputs (DCGAN D: its pre-sigmoid logits, from the same
code path minus the final sigmoid, because the output saturates to exactly 1.0) go to OUT.npz (eager_<i>, compiled_<i>) so the caller verifies
them against torch eager CPU. Prints one JSON line per case."""
import json
import math
import statistics
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

model, wpath, inpath, outpath, cases_json, warmup, min_runs, min_s = sys.argv[1:9]
warmup, min_runs, min_s = int(warmup), int(min_runs), float(min_s)
cases = json.loads(cases_json)


def load_torch_npz(path):
    """PyTorch state dict -> MLX arrays; conv weights [O,I,kh,kw] -> [O,kh,kw,I], convT [I,O,kh,kw] -> [O,kh,kw,I]."""
    d = np.load(path)
    return {k: d[k] for k in d.files}


def conv_w(w):
    return mx.array(np.ascontiguousarray(w.transpose(0, 2, 3, 1)))


def convT_w(w):
    return mx.array(np.ascontiguousarray(w.transpose(1, 2, 3, 0)))


def bn_params(sd, pre):
    return {k: mx.array(sd[f"{pre}.{k}"]) for k in ("weight", "bias", "running_mean", "running_var")}


def bn(x, p, eps=1e-5):  # == torch BatchNorm2d eval
    return (x - p["running_mean"]) * mx.rsqrt(p["running_var"] + eps) * p["weight"] + p["bias"]


relu = nn.relu


def relu6(x):
    return mx.minimum(mx.maximum(x, 0), 6)


_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # pads with -inf like torch


def build_resnet18(sd):
    W = {k: conv_w(v) for k, v in sd.items() if v.ndim == 4}
    B = {k.rsplit(".", 1)[0]: None for k in sd if k.endswith("running_mean")}
    B = {k: bn_params(sd, k) for k in B}
    fcw, fcb = mx.array(np.ascontiguousarray(sd["fc.weight"].T)), mx.array(sd["fc.bias"])

    def f(x):
        x = x.transpose(0, 2, 3, 1)
        x = relu(bn(mx.conv2d(x, W["conv1.weight"], stride=2, padding=3), B["bn1"]))
        x = _maxpool(x)
        for li, s0 in zip((1, 2, 3, 4), (1, 2, 2, 2)):
            for b in range(2):
                pre, s = f"layer{li}.{b}", (s0 if b == 0 else 1)
                h = relu(bn(mx.conv2d(x, W[pre + ".conv1.weight"], stride=s, padding=1), B[pre + ".bn1"]))
                h = bn(mx.conv2d(h, W[pre + ".conv2.weight"], stride=1, padding=1), B[pre + ".bn2"])
                if pre + ".downsample.0.weight" in W:
                    x = bn(mx.conv2d(x, W[pre + ".downsample.0.weight"], stride=s), B[pre + ".downsample.1"])
                x = relu(h + x)
        return x.mean(axis=(1, 2)) @ fcw + fcb
    return f


def build_mobilenetv2(sd):
    W = {k: conv_w(v) for k, v in sd.items() if v.ndim == 4}
    B = {k.rsplit(".", 1)[0]: None for k in sd if k.endswith("running_mean")}
    B = {k: bn_params(sd, k) for k in B}
    fcw, fcb = mx.array(np.ascontiguousarray(sd["classifier.1.weight"].T)), mx.array(sd["classifier.1.bias"])
    cfg = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def f(x):
        x = x.transpose(0, 2, 3, 1)
        x = relu6(bn(mx.conv2d(x, W["features.0.0.weight"], stride=2, padding=1), B["features.0.1"]))
        idx, cin = 1, 32
        for t, c, n, s in cfg:
            for i in range(n):
                stride, pre, h, j = (s if i == 0 else 1), f"features.{idx}.conv", x, 0
                if t != 1:
                    h = relu6(bn(mx.conv2d(h, W[f"{pre}.0.0.weight"]), B[f"{pre}.0.1"]))
                    j = 1
                hidden = h.shape[-1]
                h = relu6(bn(mx.conv2d(h, W[f"{pre}.{j}.0.weight"], stride=stride, padding=1, groups=hidden),
                             B[f"{pre}.{j}.1"]))
                h = bn(mx.conv2d(h, W[f"{pre}.{j + 1}.weight"]), B[f"{pre}.{j + 2}"])
                x = x + h if (stride == 1 and cin == c) else h
                cin, idx = c, idx + 1
        x = relu6(bn(mx.conv2d(x, W["features.18.0.weight"]), B["features.18.1"]))
        return x.mean(axis=(1, 2)) @ fcw + fcb
    return f


def build_dcgan_g(sd):
    W = {k: convT_w(v) for k, v in sd.items() if v.ndim == 4}
    B = {i: bn_params(sd, f"main.{i}") for i in (1, 4, 7, 10)}

    def f(z):
        x = z.transpose(0, 2, 3, 1)
        x = relu(bn(mx.conv_transpose2d(x, W["main.0.weight"], stride=1, padding=0), B[1]))
        for wi, bi in ((3, 4), (6, 7), (9, 10)):
            x = relu(bn(mx.conv_transpose2d(x, W[f"main.{wi}.weight"], stride=2, padding=1), B[bi]))
        x = mx.tanh(mx.conv_transpose2d(x, W["main.12.weight"], stride=2, padding=1))
        return x.transpose(0, 3, 1, 2)
    return f


def build_dcgan_d(sd, logits=False):
    W = {k: conv_w(v) for k, v in sd.items() if v.ndim == 4}
    B = {i: bn_params(sd, f"main.{i}") for i in (3, 6, 9)}

    def f(x):
        x = x.transpose(0, 2, 3, 1)
        x = nn.leaky_relu(mx.conv2d(x, W["main.0.weight"], stride=2, padding=1), 0.2)
        for wi, bi in ((2, 3), (5, 6), (8, 9)):
            x = nn.leaky_relu(bn(mx.conv2d(x, W[f"main.{wi}.weight"], stride=2, padding=1), B[bi]), 0.2)
        x = mx.conv2d(x, W["main.11.weight"])
        return (x if logits else mx.sigmoid(x)).reshape(-1)
    return f


def build_gpt2(path):
    """HF GPT2LMHeadModel (sdpa, eval, no dropout) transcribed; Conv1D weights are [in, out] (x @ W)."""
    P = mx.load(path)
    P = {k: v for k, v in P.items() if not k.endswith("attn.bias") or k.endswith("c_attn.bias")}
    wteT = mx.contiguous(P["wte.weight"].T)  # tied lm_head (same values; prepared at init like a linear weight)
    mx.eval(wteT)
    H, D, L = 12, 64, 12

    def ln(x, pre):
        return mx.fast.layer_norm(x, P[pre + ".weight"], P[pre + ".bias"], 1e-5)

    def gelu_new(x):
        return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3.0))))

    def run(ids, pos0, caches):
        Bn, T = ids.shape
        x = P["wte.weight"][ids] + P["wpe.weight"][mx.arange(pos0, pos0 + T)]
        new = []
        for l in range(L):
            pre = f"h.{l}"
            h = ln(x, pre + ".ln_1")
            qkv = h @ P[pre + ".attn.c_attn.weight"] + P[pre + ".attn.c_attn.bias"]
            q, k, v = (t.reshape(Bn, T, H, D).transpose(0, 2, 1, 3) for t in mx.split(qkv, 3, axis=-1))
            if caches is not None:
                k = mx.concatenate([caches[l][0], k], axis=2)
                v = mx.concatenate([caches[l][1], v], axis=2)
            new.append((k, v))
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(D),
                                                     mask="causal" if T > 1 else None)
            o = o.transpose(0, 2, 1, 3).reshape(Bn, T, H * D)
            x = x + (o @ P[pre + ".attn.c_proj.weight"] + P[pre + ".attn.c_proj.bias"])
            h = ln(x, pre + ".ln_2")
            h = gelu_new(h @ P[pre + ".mlp.c_fc.weight"] + P[pre + ".mlp.c_fc.bias"])
            x = x + (h @ P[pre + ".mlp.c_proj.weight"] + P[pre + ".mlp.c_proj.bias"])
        x = ln(x, "ln_f")[:, -1, :]  # logits_to_keep=1: last position only
        return x @ wteT, new
    return run


def bench(fn, args):
    for _ in range(warmup):
        mx.eval(fn(*args))
    ts, t0 = [], time.perf_counter()
    while len(ts) < min_runs or (time.perf_counter() - t0 < min_s and len(ts) < 1000):
        t = time.perf_counter()
        y = fn(*args)
        mx.eval(y)
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3, len(ts), y


inputs = np.load(inpath)
outs = {}
print(json.dumps({"mlx": mx.__version__, "device": str(mx.default_device())}), flush=True)
if model == "gpt2":
    run = build_gpt2(wpath)
else:
    sd = load_torch_npz(wpath)
    f = {"resnet18": build_resnet18, "mobilenetv2": build_mobilenetv2, "dcgan_g": build_dcgan_g,
         "dcgan_d": build_dcgan_d}[model](sd)
    # D saturates (sigmoid == 1.0 exactly on these inputs), so it is verified on its pre-sigmoid logits
    fv = build_dcgan_d(sd, logits=True) if model == "dcgan_d" else None
    all_f32 = all(v.dtype == np.float32 for k, v in sd.items() if not k.endswith("num_batches_tracked"))
for i, (name, kind) in enumerate(cases):
    x = mx.array(inputs[f"case_{i}"])
    if model == "gpt2":
        all_f32 = True
        if kind == "prefill":
            def fe(ids):
                return run(ids, 0, None)[0]
            args = (x,)
        else:  # decode: one token at position P with a P-token KV cache (cache built untimed)
            ctx = mx.array(inputs[f"ctx_{i}"])
            _, caches = run(ctx, 0, None)
            flat = [a for kv in caches for a in kv]
            mx.eval(flat)
            P0 = ctx.shape[1]

            def fe(tok, *flat):
                cs = [(flat[2 * l], flat[2 * l + 1]) for l in range(len(flat) // 2)]
                return run(tok, P0, cs)[0]
            args = (x, *flat)
    else:
        fe, args = f, (x,)
    res = {}
    for mode, fn in (("eager", fe), ("compiled", mx.compile(fe))):
        ms, n, y = bench(fn, args)
        res[mode] = (ms, n)
        if model == "dcgan_d":
            y = (fv if mode == "eager" else mx.compile(fv))(*args)
        outs[f"{mode}_{i}"] = np.array(y)
    print(json.dumps({"case": name, "eager_ms": res["eager"][0], "eager_runs": res["eager"][1],
                      "compiled_ms": res["compiled"][0], "compiled_runs": res["compiled"][1], "f32": all_f32}),
          flush=True)
np.savez(outpath, **outs)
