"""Qwen3-ASR-1.7B on lokinfer, Mac CPU / Mac GPU, basic and fused (qwen3-asr-port-plan.md Q7, decision 11).

Cases are the dashboard's (bench/perfdb.py DEFAULT_META), on the golden inputs (`R.load_ref(<clip>)`, Q1):
  enc Ns       encoder (features -> audio embeddings [N_a, 2048]), clip sN
  prefill Ns   the whole prompt (T = N_a + 18) in <= 2048-id chunks at pos 0 into the caller's KV cache, last-row
               logits (audio embeddings precomputed, not timed)
  decode Ns    one decode step at the clip's final context C = T + n - 1 (n = golden tokens incl. eos): the cache
               holds the prompt + the golden tokens teacher-forced (built untimed; the step rewrites the same row)
  e2e Ns       encoder + prefill + the full greedy loop (`M.transcribe`, the caller contract), model time only
  e2e 30min    the caller splits s1800 (`R.split_long`, <= 20 min per pass) and transcribes each pass; Mac only
The same case kinds and timing protocol as bench_qwen3asr_baselines.py (warm-up >= 2 calls and >= 1 s, then >= 30
runs or >= 2 s, >= 7 runs; a call > 10 s: 1 warm-up, 3 runs; median).

Basic variant: timing only (no model-level correctness run on the basic kernels, user 2026-11-05; they are proven by
their op tests). Its e2e cases stop at 300 s (plan Q7: a 20 min pass on the naive kernels takes many minutes per call).

Usage (4 GB of weights: the big-model lock, then the Mac bench lock; check `sysctl vm.swapusage` first):
  scripts/lk-lock /tmp/lokinfer-bigmodel.lock scripts/lk-lock /tmp/lokinfer-macbench.lock \
      .venv/bin/python bench/models/bench_qwen3asr.py [--device cpu gpu] [--variant opt basic] [--case "enc 10s" ...]
      [--record]
--record writes the medians to perf/results.json ("lokinfer basic" / "lokinfer fused").
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "bench" / "models"))
sys.path.insert(0, str(ROOT / "python"))
import perfdb  # noqa: E402
from bench_qwen3asr_baselines import timeit  # noqa: E402
from models.qwen3_asr import clips  # noqa: E402
from models.qwen3_asr import model as M  # noqa: E402
from models.qwen3_asr import reference as R  # noqa: E402

MODEL = "Qwen3-ASR-1.7B"
DEVICE_NAME = {"cpu": "Mac CPU", "gpu": "Mac GPU"}
IMPL = {"basic": "lokinfer basic", "opt": "lokinfer fused"}
CASES = perfdb.DEFAULT_META["models"][MODEL]["cases"]
BASIC_E2E_MAX_S = 300
OUT = Path.home() / ".cache" / "lokinfer" / "qwen3asr" / "bench" / "lokinfer.jsonl"


def parse(case: str) -> tuple[str, str]:
    kind, n = case.split()
    return kind, ("s1800" if n == "30min" else "s" + n.rstrip("s"))


class Clip:
    """Golden inputs of a clip (features, n_audio, ids, golden tokens)."""

    def __init__(self, name: str):
        r = R.load_ref(name)
        self.name = name
        self.features = np.ascontiguousarray(r["inputs/features"])
        self.n_audio = int(r["inputs/n_audio"][0])
        self.ids = np.ascontiguousarray(r["inputs/ids"])
        self.seconds = float(r["inputs/seconds"][0])
        self.tokens = r["tokens"].tolist()
        self.max_new = int(128 + 8 * self.seconds)


def _rows(audio, s, e, n_audio):
    k0, k1 = min(max(s - M.AUDIO_START, 0), n_audio), min(max(e - M.AUDIO_START, 0), n_audio)
    return np.ascontiguousarray(audio[k0:k1]) if k1 > k0 else np.zeros((1, M.LM_D), np.float16)


def prefill(gen, seq, audio, n_audio, kv, upto=None):
    upto = len(seq) if upto is None else upto
    out = None
    for s in range(0, upto, R.PREFILL_CHUNK):
        e = min(upto, s + R.PREFILL_CHUNK)
        out = gen(np.ascontiguousarray(seq[s:e]), _rows(audio, s, e, n_audio), np.array([s], np.int64), kv).numpy()
    return out


class Lk:
    def __init__(self, device: str, variant: str):
        self.device, self.variant = device, variant
        sd = M.load_weights_opt() if variant == "opt" else M.load_weights()
        self.enc = M.build_encoder(sd, device, variant=variant)
        self.gen = M.build_generate(sd, device, variant=variant)
        self._kv = {}

    def kv(self, capacity: int):
        """The caller's cache, reused across calls and passes while it is big enough (a fresh 2.9 GB cache per call
        would time the allocation; the contract lets the caller reuse it). Grows in 1024-position steps."""
        cap = min(M.MAX_CAP["mac"], -(-capacity // 1024) * 1024)
        have = next(iter(self._kv), 0)
        if have < cap:
            self._kv.clear()
            gc.collect()
            self._kv[cap] = M.kv_cache(cap, self.device)
            have = cap
        return self._kv[have]

    def encode(self, c: Clip):
        return self.enc(c.features, np.array([c.n_audio], np.int64)).numpy()

    def transcribe(self, c: Clip) -> list[int]:
        return M.transcribe(self.enc, self.gen, c.features, c.n_audio, c.ids, c.max_new,
                            kv=self.kv(len(c.ids) + c.max_new), device=self.device)


def run_case(lk: Lk, case: str, clip_cache: dict) -> dict | None:
    kind, name = parse(case)
    if name == "s1800":
        if kind != "e2e":
            return None
        if "s1800" not in clip_cache:
            audio = clips.load("s1800")
            clip_cache["s1800"] = [R.prepare(a) for a in R.split_long(audio)]
        passes = clip_cache["s1800"]

        def e2e_long():
            return [lk.transcribe(_PrepClip(p)) for p in passes]
        res = timeit(e2e_long)
        res["passes"] = [int(p.ids.shape[0]) for p in passes]
        return res
    if name not in clip_cache:
        clip_cache[name] = Clip(name)
    c = clip_cache[name]
    if kind == "e2e" and lk.variant == "basic" and c.seconds > BASIC_E2E_MAX_S + 1:
        return None
    if kind == "enc":
        return timeit(lambda: lk.encode(c))
    audio = lk.encode(c)
    if kind == "prefill":
        kv = lk.kv(len(c.ids))
        return {**timeit(lambda: prefill(lk.gen, c.ids, audio, c.n_audio, kv)), "T": len(c.ids)}
    if kind == "decode":
        seq = np.concatenate([c.ids, np.array(c.tokens[:-1], np.int64)])
        kv = lk.kv(len(seq))
        prefill(lk.gen, seq, audio, c.n_audio, kv, upto=len(seq) - 1)
        tok, pos = seq[-1:].copy(), np.array([len(seq) - 1], np.int64)
        dummy = np.zeros((1, M.LM_D), np.float16)
        return {**timeit(lambda: lk.gen(tok, dummy, pos, kv).numpy()), "context": len(seq)}
    got = {}

    def e2e():
        got["toks"] = lk.transcribe(c)  # the golden check uses the timed calls' output (no extra multi-minute call)
    res = timeit(e2e)
    res["matches_golden"] = got["toks"] == c.tokens
    res["n_tokens"], res["golden_tokens"] = len(got["toks"]), len(c.tokens)
    if not res["matches_golden"]:  # where it left the golden path (a different length is different decode work)
        res["first_diff"] = next((i for i, (a, b) in enumerate(zip(got["toks"], c.tokens)) if a != b),
                                 min(len(got["toks"]), len(c.tokens)))
    return res


class _PrepClip:
    """A caller-split pass (R.Prepared) in Clip's shape (no golden tokens)."""

    def __init__(self, p):
        self.features = np.ascontiguousarray(p.features)
        self.n_audio, self.ids, self.max_new = int(p.n_audio), np.ascontiguousarray(p.ids), int(p.max_new)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", nargs="+", default=["cpu", "gpu"], choices=["cpu", "gpu"])
    ap.add_argument("--variant", nargs="+", default=["opt", "basic"], choices=list(M.VARIANTS))
    ap.add_argument("--case", nargs="+", default=CASES)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="A/B during development: 1 warm-up + 5 runs (3 when a call takes > 10 s); never recorded")
    a = ap.parse_args()
    if a.quick:
        assert not a.record, "--quick numbers are never recorded"
        import bench_qwen3asr_baselines
        bench_qwen3asr_baselines.QUICK = True
    bad = [c for c in a.case if c not in CASES]
    assert not bad, f"unknown cases {bad} (the dashboard's: {CASES})"
    state = perfdb.machine_state()
    print(f"machine: {state}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for variant in a.variant:
        for device in a.device:
            R.memory_guard(6.0, f"lokinfer {variant} {device}")
            print(f"== {IMPL[variant]} / {DEVICE_NAME[device]}: building", flush=True)
            t = time.perf_counter()
            lk = Lk(device, variant)
            print(f"   built in {time.perf_counter() - t:.1f} s", flush=True)
            clip_cache: dict = {}
            for case in a.case:
                res = run_case(lk, case, clip_cache)
                if res is None:
                    continue
                rec = {"impl": IMPL[variant], "device": DEVICE_NAME[device], "case": case, **res,
                       "state": state, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                print("   " + json.dumps(rec), flush=True)
                with OUT.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                if a.record:
                    perfdb.record(MODEL, DEVICE_NAME[device], case, IMPL[variant], res["median_ms"], state=state,
                                  notes=f"bench/models/bench_qwen3asr.py ({variant} graphs, f16): median of "
                                        f"{res['runs']} runs after {res['warmup']} warm-up"
                                        + ("" if res.get("matches_golden", True) else
                                           f"; transcript differs from the golden one from token {res['first_diff']} "
                                           f"({res['n_tokens']} vs {res['golden_tokens']} tokens)"))
            del lk, clip_cache
            gc.collect()


if __name__ == "__main__":
    main()
