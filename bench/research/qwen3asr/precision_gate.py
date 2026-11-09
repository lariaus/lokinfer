"""Qwen3-ASR precision gate (plan Q0.4): is f16 safe for this model before any f16 kernel is written?

Runs the HF model on every gate clip (speech <= 300 s + edge clips) in several precisions, one model in memory at a
time (f64 is 16 GB), and records per clip:
  - encoder output (audio embeddings) and last-row prefill logits;
  - the greedy transcript (the caller contract's loop, chunked prefill);
  - max |activation| per layer (LM residual stream, SwiGLU product = down_proj input, down_proj output; encoder layer
    outputs) during the prefill, and any NaN / inf;
then, with the float64 models: errors vs G-true (bf16 checkpoint in f64) and vs G-f16w (f16-rounded weights in f64),
and whether f64 greedy decoding agrees with each precision's transcript (teacher forcing: one f64 forward over prompt
+ tokens; f64 greedy = that transcript iff the f64 argmax equals the token at every position).

    .venv/bin/python bench/research/qwen3asr/precision_gate.py run [config ...]   # configs below; default all
    .venv/bin/python bench/research/qwen3asr/precision_gate.py report             # markdown tables
Results: ~/.cache/lokinfer/qwen3asr/gate/<config>.npz + report.md
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from models.qwen3_asr import clips as C  # noqa: E402
from models.qwen3_asr import reference as R  # noqa: E402

OUT = Path.home() / ".cache" / "lokinfer" / "qwen3asr" / "gate"
SPEECH = ["s2", "s5", "s10", "s30", "s60", "s111", "s300"]
CLIPS = SPEECH + list(C.EDGE)
SHORT = ["s2", "s5", "e_short", "e_1s", "e_silence", "e_noise", "e_tone"]  # torch CPU f16 is a slow fallback
# config -> (dtype, device, weights, clips)
CONFIGS = {
    "f32_cpu": (torch.float32, "cpu", "true", CLIPS),
    "f16_mps": (torch.float16, "mps", "true", CLIPS),  # the app's path
    "bf16_mps": (torch.bfloat16, "mps", "true", CLIPS),
    "f16_cpu": (torch.float16, "cpu", "true", SHORT),
}
GOLDEN = {"f64_true": "true", "f64_f16w": "f16w"}


def _sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def _hooks(model, rec: dict):
    """Max |x| and non-finite counts per hooked tensor, accumulated in rec[name] = [max_abs, n_nonfinite]."""
    def note(name, t):
        t = t.detach()
        finite = torch.isfinite(t)
        m = float(t[finite].abs().max()) if finite.any() else float("inf")
        r = rec.setdefault(name, [0.0, 0])
        r[0] = max(r[0], m)
        r[1] += int((~finite).sum())

    hs = []
    lm = model.model.language_model
    for i, layer in enumerate(lm.layers):
        hs.append(layer.register_forward_hook(lambda m, i_, o, i=i: note(f"lm.{i:02d}.residual", o[0] if isinstance(o, tuple) else o)))
        def down_hook(m, i_, o, i=i):  # a forward hook must return None (a return value replaces the output)
            note(f"lm.{i:02d}.swiglu", i_[0])
            note(f"lm.{i:02d}.down_proj", o)

        hs.append(layer.mlp.down_proj.register_forward_hook(down_hook))
    enc = model.model.audio_tower
    for i, layer in enumerate(enc.layers):
        hs.append(layer.register_forward_hook(lambda m, i_, o, i=i: note(f"enc.{i:02d}.out", o[0] if isinstance(o, tuple) else o)))
    return hs


@torch.no_grad()
def prefill_logits(model, prep) -> np.ndarray:
    """Last-row logits after the whole prompt, prefilled in the contract's chunks (f32 numpy)."""
    from transformers import DynamicCache

    dev = next(model.parameters()).device
    emb = R.prompt_embeddings(model, prep)
    cache = DynamicCache(config=model.config.text_config)
    T = emb.shape[1]
    for s in range(0, T, R.PREFILL_CHUNK):
        e = min(T, s + R.PREFILL_CHUNK)
        out = model.model.language_model(inputs_embeds=emb[:, s:e], past_key_values=cache, use_cache=True,
                                         position_ids=torch.arange(s, e, device=dev)[None])
    return model.lm_head(out.last_hidden_state[:, -1:])[0, -1].float().cpu().numpy()


def run(name: str):
    dtype, dev, weights, clip_names = CONFIGS[name]
    print(f"== {name}: loading", flush=True)
    model = R.load_model(dtype, dev, weights)
    res = {}
    for c in clip_names:
        prep = R.prepare(C.load(c))
        rec: dict = {}
        hs = _hooks(model, rec)
        _sync(dev)
        t0 = time.perf_counter()
        audio = R.audio_embeddings(model, prep).float().cpu().numpy()
        logits = prefill_logits(model, prep)
        _sync(dev)
        t1 = time.perf_counter()
        for h in hs:
            h.remove()
        toks = R.manual_generate(model, prep)
        _sync(dev)
        t2 = time.perf_counter()
        res[c] = dict(audio=audio, logits=logits, tokens=np.array(toks, np.int64), acts=json.dumps(rec),
                      prefill_s=t1 - t0, generate_s=t2 - t1)
        nf = sum(v[1] for v in rec.values())
        print(f"  {c:9} T={prep.ids.shape[0]:5} tokens={len(toks):4} nonfinite={nf} prefill {t1 - t0:6.2f}s "
              f"generate {t2 - t1:7.2f}s  {R.decode(toks)[:70]!r}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{name}.npz", **{f"{c}/{k}": v for c, d in res.items() for k, v in d.items()})


def golden(name: str):
    """f64 CPU: audio embeddings + prefill logits per clip, and teacher-forced argmax agreement with every
    precision's transcript (configs already run)."""
    weights = GOLDEN[name]
    runs = {cfg: np.load(OUT / f"{cfg}.npz") for cfg in CONFIGS if (OUT / f"{cfg}.npz").exists()}
    print(f"== {name}: loading", flush=True)
    model = R.load_model(torch.float64, "cpu", weights)  # 2-byte weights, upcast per layer (R.load_model)
    res = {}
    for c in CLIPS:
        prep = R.prepare(C.load(c))
        t0 = time.perf_counter()
        audio = R.audio_embeddings(model, prep).cpu().numpy()
        logits = prefill_logits(model, prep)
        d = dict(audio=audio, logits=logits)
        seen = {}
        for cfg, z in runs.items():
            key = f"{c}/tokens"
            if key not in z.files:
                continue
            toks = z[key].tolist()
            if tuple(toks) not in seen:
                arg, margin, _ = R.teacher_forced(model, prep, toks)  # argmax + top-2 margin per position, chunked
                bad = [i for i, (a, t) in enumerate(zip(arg, toks)) if a != t]
                seen[tuple(toks)] = (len(bad), bad[0] if bad else -1, float(margin.min()))
            d[f"agree_{cfg}"] = np.array(seen[tuple(toks)])
        res[c] = d
        print(f"  {c:9} {time.perf_counter() - t0:7.1f}s  " +
              "  ".join(f"{k[6:]}: {int(v[0])} mismatches (first {int(v[1])}, min top-2 margin {v[2]:.3f})"
                        for k, v in d.items() if k.startswith("agree_")), flush=True)
    np.savez(OUT / f"{name}.npz", **{f"{c}/{k}": v for c, d in res.items() for k, v in d.items()})


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-30))


def report():
    z = {n: np.load(OUT / f"{n}.npz") for n in list(CONFIGS) + list(GOLDEN) if (OUT / f"{n}.npz").exists()}
    lines = ["| clip | config | audio rel err vs G-true | logits rel err vs G-true | logits vs G-f16w | f64 greedy agrees "
             "| tokens | max LM residual | max SwiGLU | max down_proj | max enc | non-finite |", "|" + "---|" * 12]
    worst = {}
    for c in CLIPS:
        for cfg in CONFIGS:
            if cfg not in z or f"{c}/tokens" not in z[cfg].files:
                continue
            r = z[cfg]
            acts = json.loads(str(r[f"{c}/acts"]))
            mx = lambda pre: max((v[0] for k, v in acts.items() if k.startswith(pre[0]) and k.endswith(pre[1])), default=0)
            nf = sum(v[1] for v in acts.values())
            res = mx(("lm.", "residual")); sw = mx(("lm.", "swiglu")); dp = mx(("lm.", "down_proj")); en = mx(("enc.", "out"))
            for k, v in (("residual", res), ("swiglu", sw), ("down_proj", dp), ("enc", en)):
                worst[(cfg, k)] = max(worst.get((cfg, k), 0), v)
            g = z.get("f64_true")
            gf = z.get("f64_f16w")
            ae = rel(r[f"{c}/audio"], g[f"{c}/audio"]) if g is not None else float("nan")
            le = rel(r[f"{c}/logits"], g[f"{c}/logits"]) if g is not None else float("nan")
            lf = rel(r[f"{c}/logits"], gf[f"{c}/logits"]) if gf is not None else float("nan")
            agree = g[f"{c}/agree_{cfg}"] if g is not None and f"{c}/agree_{cfg}" in g.files else None
            ag = "—" if agree is None else ("yes" if agree[0] == 0 else f"NO ({int(agree[0])}, first {int(agree[1])})")
            lines.append(f"| {c} | {cfg} | {ae:.2e} | {le:.2e} | {lf:.2e} | {ag} | {len(r[f'{c}/tokens'])} | {res:.0f} | "
                         f"{sw:.0f} | {dp:.0f} | {en:.1f} | {nf} |")
    lines += ["", "Worst max |activation| per config: " +
              ", ".join(f"{cfg} {k} {v:.0f}" for (cfg, k), v in sorted(worst.items()))]
    txt = "\n".join(lines)
    (OUT / "report.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    names = sys.argv[2:]
    if cmd == "run":
        for n in names or list(CONFIGS) + list(GOLDEN):
            (run if n in CONFIGS else golden)(n)
        report()
    else:
        report()
