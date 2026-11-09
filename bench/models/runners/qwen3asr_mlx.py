"""MLX (mlx-audio) Qwen3-ASR-1.7B runner for bench_qwen3asr_baselines.py, f16. Runs in a separate venv with mlx +
mlx-audio (not lokinfer's .venv):

    python -I qwen3asr_mlx.py SNAPSHOT_DIR IN.npz JOBS_JSON

SNAPSHOT_DIR: <baselines>/models/qwen3-asr-mlx-f16 (preferred: already f16, mlx layout) or the pinned HF snapshot of
Qwen/Qwen3-ASR-1.7B-hf (the same weights as every other framework; converted per tensor). The HF snapshot's
transformers-native keys are mapped onto mlx-audio's `qwen3_asr` model (model.audio_tower.* -> audio_tower.*,
model.multi_modal_projector.linear_{1,2} -> audio_tower.proj{1,2}, model.language_model.* -> model.*, conv weights
OIHW -> OHWI) and every array is cast bf16 -> float16 (`astype(mx.float16)`); the encoder's sinusoid table (a plain
attribute, not a parameter) is cast too, so the encoder runs in f16 like torch's (whose buffer the model's dtype
casts). IN.npz holds, per clip c: feats_c (f32 [1, 128, 100 n], the processor's features), mask_c (i32 [1, 100 n]),
ids_c (i64 [T], the caller contract's prompt), ref_c (i64, the f32 CPU reference tokens), maxnew_c.
JOBS_JSON: list of [kind, clip, context]: kind in verify | enc | prefill | decode | e2e (bench_qwen3asr_baselines.py
docstring); context = C for decode (keys attended by the timed step). An e2e clip may be "a+b": the passes of a long
clip (the caller's split), transcribed one after another inside one timed call.

The case functions use mlx-audio's modules (AudioEncoder, TextModel, KVCache, the tied embedding's as_linear) with
lokinfer's prepared inputs; the splice is one concatenate (the pads are one contiguous run) instead of mlx-audio's
per-row Python loop, and the LM head runs on the last row only (as HF's generate does). Plain MLX (lazy graph, mx.eval
inside every timed call), no mx.compile.

Timing protocol (decision 11): warm-up >= 2 calls and >= 1 s (1 call when a call > 10 s), then >= 7 runs (3 when a
call > 10 s), stopping at >= 30 runs or >= 2 s; median. Prints one JSON line per job (verify: the generated tokens)."""
import glob
import json
import statistics
import sys
import time
from importlib.metadata import version

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_audio.stt.models.qwen3_asr.config import ModelConfig
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel

snap, in_path, jobs = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
AUDIO_PAD, EOS, CHUNK = 151676, (151643, 151645), 2048
F16 = mx.float16


def hf_to_mlx_f16(raw: dict) -> dict:
    """HF -hf keys -> mlx-audio keys, cast to f16 ONE TENSOR AT A TIME (the bf16 source is popped and the f16 copy
    evaluated before the next one), so the whole model is never held twice (memory rule: 4.08 GB, not 8)."""
    w = {}
    for k in list(raw):
        v = raw.pop(k)
        if k == "lm_head.weight":
            continue  # tied to the embedding
        if k.startswith("model.audio_tower."):
            k = "audio_tower." + k[len("model.audio_tower."):]
            if "conv2d" in k and k.endswith(".weight"):
                v = v.transpose(0, 2, 3, 1)
        elif k.startswith("model.multi_modal_projector.linear_"):
            k = "audio_tower.proj" + k[len("model.multi_modal_projector.linear_"):]
        elif k.startswith("model.language_model."):
            k = "model." + k[len("model.language_model."):]
        else:
            raise KeyError(k)
        w[k] = mx.contiguous(v.astype(F16))
        mx.eval(w[k])
        del v
    return w


def load(snap):
    """snap: either an f16 mlx-layout dir (<baselines>/models/qwen3-asr-mlx-f16, written by
    <baselines>/scripts/convert_qwen3asr_mlx_f16.py: loaded as is) or the HF snapshot (converted per tensor)."""
    cfg = ModelConfig.from_dict(json.load(open(f"{snap}/config.json")))
    model = Qwen3ASRModel(cfg)
    raw = {}
    for f in sorted(glob.glob(f"{snap}/*.safetensors")):
        raw.update(mx.load(f))
    src_dtypes = sorted({str(v.dtype) for v in raw.values()})
    w = raw if any(k.startswith("audio_tower.") for k in raw) else hf_to_mlx_f16(raw)
    assert all(v.dtype == F16 for v in w.values())
    model.load_weights(list(w.items()), strict=True)
    del raw, w
    pe = model.audio_tower.positional_embedding
    pe._positional_embedding = pe._positional_embedding.astype(F16)
    model.eval()
    params = tree_flatten(model.parameters())
    mx.eval([p for _, p in params], pe._positional_embedding)
    info = {"mlx": mx.__version__, "mlx_audio": version("mlx-audio"), "device": str(mx.default_device()),
            "n_params": int(sum(p.size for _, p in params)), "all_params_f16": all(p.dtype == F16 for _, p in params),
            "pos_emb": str(pe._positional_embedding.dtype), "src_dtypes": src_dtypes,
            "active_gb": round(mx.get_active_memory() / 1e9, 2)}
    return model, info


model, info = load(snap)
print(json.dumps({"info": info}), flush=True)
D = np.load(in_path)
enc, lm = model.audio_tower, model.model
emb_table = lm.embed_tokens


def inputs(c):
    ids = D[f"ids_{c}"]
    pads = np.flatnonzero(ids == AUDIO_PAD)
    assert pads.size and np.all(np.diff(pads) == 1), "audio pads must be one contiguous run"
    return (mx.array(D[f"feats_{c}"]).astype(F16), mx.array(D[f"mask_{c}"]), mx.array(ids.astype(np.int32)),
            int(pads[0]), int(pads.size))


def encoder(feats, mask):
    a = enc(feats, mask)
    mx.eval(a)
    return a


def embed_prompt(ids, a0, na, audio):
    e = emb_table(ids[None])
    return mx.concatenate([e[:, :a0], audio[None].astype(e.dtype), e[:, a0 + na:]], axis=1)


def prefill_embeds(e, cache):
    T, h = e.shape[1], None
    for s in range(0, T, CHUNK):
        h = lm(inputs_embeds=e[:, s:s + CHUNK], cache=cache)
        if s + CHUNK < T:
            mx.eval([c.keys for c in cache] + [c.values for c in cache])
    logits = emb_table.as_linear(h[:, -1:])[0, -1]
    mx.eval(logits)
    return logits


def prefill(ids, a0, na, audio):
    cache = model.make_cache()
    return prefill_embeds(embed_prompt(ids, a0, na, audio), cache), cache


def step(tok, cache):
    logits = emb_table.as_linear(lm(input_ids=mx.array([[tok]]), cache=cache))[0, -1]
    mx.eval(logits)
    return logits


def transcribe(c):
    feats, mask, ids, a0, na = inputs(c)
    logits, cache = prefill(ids, a0, na, encoder(feats, mask))
    toks, max_new = [], int(D[f"maxnew_{c}"])
    while True:
        tok = int(mx.argmax(logits).item())
        toks.append(tok)
        if tok in EOS or len(toks) >= max_new:
            return toks
        logits = step(tok, cache)


def timeit(fn, after=lambda: None):
    t = time.perf_counter()
    fn()
    first = time.perf_counter() - t
    after()
    slow = first > 10.0
    n_warm, warm_end = 1, time.perf_counter() + (0.0 if slow else 1.0)
    while n_warm < (1 if slow else 2) or time.perf_counter() < warm_end:
        fn()
        after()
        n_warm += 1
    ts, t0, min_runs = [], time.perf_counter(), (3 if slow else 7)
    while len(ts) < min_runs or not (len(ts) >= 30 or time.perf_counter() - t0 >= 2.0):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
        after()
    return {"median_ms": statistics.median(ts) * 1e3, "min_ms": min(ts) * 1e3, "runs": len(ts), "warmup": n_warm,
            "p10_ms": float(np.percentile(ts, 10) * 1e3), "p90_ms": float(np.percentile(ts, 90) * 1e3)}


for kind, c, C in jobs:
    if kind == "verify":
        print(json.dumps({"kind": kind, "clip": c, "tokens": transcribe(c)}), flush=True)
        continue
    if kind == "e2e":  # "a+b": a long clip's passes (the caller's split), transcribed one after another per call
        res = {"kind": kind, "clip": c}
        res.update(timeit(lambda: [transcribe(x) for x in c.split("+")]))
        print(json.dumps(res), flush=True)
        continue
    feats, mask, ids, a0, na = inputs(c)
    audio = encoder(feats, mask)
    res = {"kind": kind, "clip": c, "audio_dtype": str(audio.dtype)}
    if kind == "enc":
        res.update(timeit(lambda: encoder(feats, mask)))
    elif kind == "prefill":
        res.update(timeit(lambda: prefill(ids, a0, na, audio)))
    elif kind == "decode":
        ref = [int(t) for t in D[f"ref_{c}"]]
        T = int(ids.shape[0])
        n_tf = C - 1 - T
        tf = (ref * ((n_tf + 1) // len(ref) + 1))[: n_tf + 1]
        e = embed_prompt(ids, a0, na, audio)
        if n_tf:
            e = mx.concatenate([e, emb_table(mx.array([tf[:n_tf]]))], axis=1)
        cache = model.make_cache()
        prefill_embeds(e, cache)
        assert cache[0].offset == C - 1
        res.update(timeit(lambda: step(tf[n_tf], cache), lambda: [kc.trim(1) for kc in cache]))
        res["context"] = C
    audio = cache = e = None  # free the case's arrays (the decode cache) before the next case
    mx.clear_cache()
    print(json.dumps(res), flush=True)
