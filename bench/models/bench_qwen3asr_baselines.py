"""Qwen3-ASR-1.7B reference baselines on this Mac, all in f16 (qwen3-asr-port-plan.md Q6 / Q7, decisions 4, 9, 11):
torch eager CPU, torch.compile CPU, torch MPS, torch.compile MPS, MLX (mlx-audio, Mac GPU), llama.cpp (CPU, Metal) and
MLX Swift (Mac GPU), the last two through their harnesses' Mac CLIs (docs/impl-notes/iphone-baselines.md, Qwen3-ASR).

Every impl is VERIFIED before it is timed: its greedy transcript of s10 and s30 must equal the f32 CPU reference's
(`R.manual_generate` on an f32 CPU model, cached in ~/.cache/lokinfer/qwen3asr/baselines/ref_<clip>.json). A
mismatch is reported with both texts and the impl is not timed. Longer clips (s60 ... s1200), whose f32 CPU greedy
run would take hours, take their reference tokens from the golden data (`REF_DIR/<clip>/inputs.npz`, key "tokens",
plan Q1: the precision gate's f32 CPU greedy run; s1200: the f16 MPS greedy run); `made_by` in the json says which.
They are only used to build the decode cases' caches (teacher forcing), never to verify.

Case kinds (the caller contract, docs/impl-notes/qwen3-asr-port-plan.md), on a prepared clip `R.prepare(clips.load(c))`:
  enc:<clip>          encoder + projector -> audio embeddings [N_a, 2048]
  prefill:<clip>      embed + splice + the whole prompt (T = N_a + 18) in <= 2048-id chunks at pos 0, fresh KV cache,
                      LM head on the last row (audio embeddings precomputed, not timed)
  decode:<clip>[@C]   one decode step attending over C keys (C - 1 cached + the new token); the cache holds the prompt
                      + the clip's reference tokens teacher-forced (built untimed, cropped back after each step).
                      Default C = T + n - 1 (n = reference tokens incl. eos): the clip's last greedy step.
  e2e:<clip>          encoder + prefill + the full greedy loop (`manual_generate`), model time only (features and
                      detokenization excluded, the loop's own overhead included)
The dashboard names work too: "enc 10s", "prefill 1200s", "decode 300s" (= decode:s300 at its default C), "e2e 5s",
"e2e 30min". "e2e 30min" (= e2e:s1800) is the caller's job on a 30 min clip: `R.split_long` cuts it into passes of
<= 20 min (pseudo-clips s1800p1 = 1196.8 s, s1800p2 = 604.1 s, each `R.prepare`d); torch and MLX time both passes in
one call, the CLIs (llama.cpp, MLX Swift) time each pass and the recorded value is the sum of the two medians.

Usage. Every run loads a full model (4 GB): run it under the big-model lock (one big model job at a time on the
Mac), and timing also under the Mac bench lock (order: bigmodel -> macbench, docs/benchmarking.md); check
`sysctl vm.swapusage` first (`R.memory_guard` also stops a case instead of swapping):
  scripts/lk-lock /tmp/lokinfer-bigmodel.lock .venv/bin/python bench/models/bench_qwen3asr_baselines.py --verify \
      [--impl NAME ...]
  scripts/lk-lock /tmp/lokinfer-bigmodel.lock scripts/lk-lock /tmp/lokinfer-macbench.lock \
      .venv/bin/python bench/models/bench_qwen3asr_baselines.py [--impl NAME ...] [--case "enc 10s" prefill:s10 ...] \
      [--record]
Impls run one after another in one process, each model freed before the next loads.
--impl may repeat; default: every torch impl. Names: "torch eager", "torch.compile", "torch MPS",
"torch.compile MPS", "MLX", "llama.cpp CPU", "llama.cpp Metal", "MLX Swift". `--case dashboard` = every dashboard case
(perfdb.DEFAULT_META). Results print as JSON lines and are appended to ~/.cache/lokinfer/qwen3asr/baselines/results.jsonl;
with --record, every timed dashboard case also goes to perf/results.json (perfdb.record; decode cases only at their
default C). The CLIs need their clip files: `.venv/bin/python <baselines>/scripts/prep_qwen3asr.py <clip> ...` (wav,
ref, f32 features in <baselines>/qwen3asr/data/; s1800p1 / s1800p2 for the 30 min case).

f16 forcing: torch `R.load_model(torch.float16, dev)` (the bf16 checkpoint loaded as bf16, cast to f16 per tensor; all
parameters and buffers f16; features cast to f16 by the model input path). MLX: the same HF snapshot's tensors cast
`astype(float16)` one at a time (<baselines>/models/qwen3-asr-mlx-f16, runners/qwen3asr_mlx.py); MLX Swift loads the same
file. llama.cpp: `convert_hf_to_gguf.py --outtype f16` (LM + mmproj), f16 KV cache (default). What each framework keeps
in f32 is in docs/baselines.md (Qwen3-ASR section).

Timing (decision 11): warm-up >= 2 calls and >= 1 s (1 call when a call takes > 10 s), then runs until >= 30 runs or
>= 2 s, at least 7 runs (3 when a call takes > 10 s); median. MPS synced inside every timed call.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "python"))
import perfdb  # noqa: E402
from models.qwen3_asr import clips  # noqa: E402
from models.qwen3_asr import reference as R  # noqa: E402


def _load_localconfig():
    """python/lokinfer/localconfig.py loaded by path: importing the `lokinfer` package would trigger the editable
    install's rebuild check (a cmake build) in a model benchmark that never uses the library."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("lokinfer_localconfig", ROOT / "python" / "lokinfer" / "localconfig.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


localconfig = _load_localconfig()

MODEL = "Qwen3-ASR-1.7B"
CACHE = Path.home() / ".cache" / "lokinfer" / "qwen3asr" / "baselines"
VERIFY_CLIPS = ("s10", "s30")
TORCH_IMPLS = {"torch eager": ("cpu", False), "torch.compile": ("cpu", True), "torch MPS": ("mps", False),
               "torch.compile MPS": ("mps", True)}
CLI_IMPLS = ("llama.cpp CPU", "llama.cpp Metal", "MLX Swift")
ALL_IMPLS = (*TORCH_IMPLS, "MLX", *CLI_IMPLS)
# dashboard (device, impl) of each harness impl
DASH_IMPL = {"torch eager": ("Mac CPU", "torch eager"), "torch.compile": ("Mac CPU", "torch.compile"),
             "torch MPS": ("Mac GPU", "torch MPS"), "torch.compile MPS": ("Mac GPU", "torch.compile MPS"),
             "MLX": ("Mac GPU", "MLX"), "llama.cpp CPU": ("Mac CPU", "llama.cpp"),
             "llama.cpp Metal": ("Mac GPU", "llama.cpp"), "MLX Swift": ("Mac GPU", "MLX Swift")}
DEFAULT_CASES = ("enc:s10", "prefill:s10", "decode:s10", "e2e:s10", "enc:s30", "prefill:s30", "decode:s30", "e2e:s30")
MLX_RUNNER = Path(__file__).resolve().parent / "runners" / "qwen3asr_mlx.py"
LLAMACPP_THREADS = 8  # the M2 Max's performance cores
SRC = f"HF {R.HF_ID}@{R.HF_REV[:8]}"


# ---------------------------------------------------------------- clips, cases

_PASS = re.compile(r"^(s\d+)p(\d+)$")


def passes(clip: str) -> list[str]:
    """The caller's passes of a clip: itself when it fits one pass, else <clip>p1, <clip>p2, ... (R.split_long)."""
    if clips.SPEECH.get(clip, 0) <= R.MAX_PASS_SECONDS:
        return [clip]
    return [f"{clip}p{i + 1}" for i in range(len(_split(clip)))]


_split_cache: dict[str, list[np.ndarray]] = {}


def _split(clip: str) -> list[np.ndarray]:
    if clip not in _split_cache:
        _split_cache[clip] = R.split_long(clips.load(clip))
    return _split_cache[clip]


def clip_audio(name: str) -> np.ndarray:
    """A clip's audio, or one pass of a long clip ("s1800p1")."""
    m = _PASS.match(name)
    return _split(m.group(1))[int(m.group(2)) - 1] if m else clips.load(name)


_DASH = re.compile(r"^(enc|prefill|decode|e2e) (\d+)(s|min)$")


def parse_case(case: str) -> tuple[str, str, int | None]:
    """"enc:s10" / "decode:s30@480" / dashboard names ("enc 10s", "e2e 30min") -> (kind, clip, C or None)."""
    m = _DASH.match(case.strip())
    if m:
        kind, n, unit = m.groups()
        return kind, f"s{int(n) * (60 if unit == 'min' else 1)}", None
    kind, rest = case.split(":", 1)
    clip, _, at = rest.partition("@")
    assert kind in ("enc", "prefill", "decode", "e2e"), case
    return kind, clip, (int(at) if at else None)


def dash_case(kind: str, clip: str, at: int | None) -> str | None:
    """The dashboard case name of a case (None: not a dashboard case, e.g. decode at a non-default C)."""
    if at is not None or not clip.startswith("s") or not clip[1:].isdigit():
        return None
    s = int(clip[1:])
    name = f"{kind} {s // 60}min" if s % 60 == 0 and s >= 1800 else f"{kind} {s}s"
    return name if name in perfdb.DEFAULT_META["models"][MODEL]["cases"] else None


def canon(case: str) -> str:
    kind, clip, at = parse_case(case)
    return f"{kind}:{clip}" + (f"@{at}" if at is not None else "")


# ---------------------------------------------------------------- reference transcripts

_prep_cache: dict[str, R.Prepared] = {}


def prep(name: str) -> R.Prepared:
    if name not in _prep_cache:
        _prep_cache[name] = R.prepare(clip_audio(name))
    return _prep_cache[name]


def ref(name: str) -> dict:
    """A clip's reference transcript: {"tokens", "text", "T", "made_by"} (computed once, cached). s2 ... s30: the f32
    CPU greedy run (`R.manual_generate`); clips with golden data (s60 ... s1200): its tokens; passes of a long clip
    (s1800p1, ...): none (e2e only, nothing to teacher-force or verify)."""
    path = CACHE / f"ref_{name}.json"
    if path.exists():
        return json.loads(path.read_text())
    p = prep(name)
    gold = Path(R.ref_path(name))
    if _PASS.match(name):
        d = {"clip": name, "tokens": [], "text": "", "T": int(p.ids.shape[0]),
             "made_by": "none: a pass of a long clip (e2e only)"}
    elif gold.exists():
        z = np.load(gold)  # only the inputs part (R.load_ref would also load every captured stage: GBs on s1200)
        assert np.array_equal(z["inputs/ids"], p.ids), f"{name}: golden ids differ from R.prepare's"
        toks = z["tokens"].tolist()
        how = "f16 MPS greedy" if p.seconds > 400 else "f32 CPU greedy (precision gate)"
        d = {"clip": name, "tokens": toks, "text": R.decode(toks), "T": int(p.ids.shape[0]),
             "made_by": f"golden data tokens ({gold.parent.name}/inputs.npz, plan Q1: {how}), {R.HF_ID}@{R.HF_REV[:8]}"}
    else:
        print(f"[ref] f32 CPU manual_generate on {name} (once)", flush=True)
        m = _ref_model()
        toks = R.manual_generate(m, p)
        d = {"clip": name, "tokens": toks, "text": R.decode(toks), "T": int(p.ids.shape[0]),
             "made_by": f"R.manual_generate, f32 CPU, torch {torch.__version__}, {R.HF_ID}@{R.HF_REV[:8]}"}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d) + "\n")
    return d


_REF_MODEL = None


def _ref_model():
    global _REF_MODEL
    if _REF_MODEL is None:
        _REF_MODEL = R.load_model(torch.float32, "cpu")
    return _REF_MODEL


def drop_ref_model():
    global _REF_MODEL
    _REF_MODEL = None


def decode_context(name: str, at: int | None = None) -> int:
    r = ref(name)
    return r["T"] + len(r["tokens"]) - 1 if at is None else at


def case_kv_gb(kind: str, clip: str, at: int | None) -> float:
    """Memory a case needs beyond the weights (KV cache of its longest context + 1 GB of activations)."""
    if kind == "enc":
        return 1.0
    if kind == "decode":
        return R.kv_gb(decode_context(clip, at)) + 1.0
    if kind == "prefill":
        return R.kv_gb(int(prep(clip).ids.shape[0])) + 1.0
    return max(R.kv_gb(int(prep(c).ids.shape[0]) + prep(c).max_new) for c in passes(clip)) + 1.0


# ---------------------------------------------------------------- timing

QUICK = False  # --quick (bench_qwen3asr*.py): 1 warm-up + 5 runs (3 when a call takes > 10 s), never recorded


def timeit(fn, sync=lambda: None, after=lambda: None):
    """Decision 11 protocol (or the quick A/B protocol when QUICK). `after` runs outside the timed region after every
    call (e.g. cache crop)."""
    t = time.perf_counter()
    fn()
    sync()
    first = time.perf_counter() - t
    after()
    if QUICK:  # the first call is the warm-up
        ts = []
        for _ in range(3 if first > 10.0 else 5):
            t = time.perf_counter()
            fn()
            sync()
            ts.append(time.perf_counter() - t)
            after()
        return {"median_ms": statistics.median(ts) * 1e3, "min_ms": min(ts) * 1e3, "runs": len(ts), "warmup": 1,
                "p10_ms": float(np.percentile(ts, 10) * 1e3), "p90_ms": float(np.percentile(ts, 90) * 1e3),
                "quick": True}
    slow = first > 10.0
    n_warm, warm_end = 1, time.perf_counter() + (0.0 if slow else 1.0)
    while n_warm < (1 if slow else 2) or time.perf_counter() < warm_end:
        fn()
        sync()
        after()
        n_warm += 1
    ts, t0, min_runs = [], time.perf_counter(), (3 if slow else 7)
    while len(ts) < min_runs or not (len(ts) >= 30 or time.perf_counter() - t0 >= 2.0):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
        after()
    return {"median_ms": statistics.median(ts) * 1e3, "min_ms": min(ts) * 1e3, "runs": len(ts), "warmup": n_warm,
            "p10_ms": float(np.percentile(ts, 10) * 1e3), "p90_ms": float(np.percentile(ts, 90) * 1e3)}


# ---------------------------------------------------------------- torch (CPU / MPS, eager / torch.compile)

class TorchQwen:
    """HF Qwen3ASRForConditionalGeneration in f16 on `device`; with `compiled`, the audio tower, the language model
    and the LM head forwards are torch.compile'd (dynamic=True) in place, so every case (and manual_generate)
    runs through the compiled modules."""

    def __init__(self, device: str, compiled: bool):
        self.device, self.compiled = device, compiled
        self.m = R.load_model(torch.float16, device)
        bad = [n for n, p in list(self.m.named_parameters()) + list(self.m.named_buffers())
               if p.is_floating_point() and p.dtype != torch.float16]
        self.f16_check = {"params_and_buffers_f16": not bad, "non_f16": bad[:5],
                          "attn_impl": self.m.config.text_config._attn_implementation}
        mm = self.m.model
        if compiled:
            mm.audio_tower.forward = torch.compile(mm.audio_tower.forward, dynamic=True)
            mm.multi_modal_projector.forward = torch.compile(mm.multi_modal_projector.forward, dynamic=True)
            mm.language_model.forward = torch.compile(mm.language_model.forward, dynamic=True)
            self.m.lm_head.forward = torch.compile(self.m.lm_head.forward, dynamic=True)
        self.lm = mm.language_model

    def sync(self):
        if self.device == "mps":
            torch.mps.synchronize()

    def _cache(self):
        from transformers import DynamicCache
        return DynamicCache(config=self.m.config.text_config)

    @torch.no_grad()
    def encoder(self, p):
        return R.audio_embeddings(self.m, p)

    @torch.no_grad()
    def _prefill_embeds(self, emb, cache, start=0):
        T, out = emb.shape[1], None
        for s in range(0, T, R.PREFILL_CHUNK):
            e = min(T, s + R.PREFILL_CHUNK)
            pos = torch.arange(start + s, start + e, device=self.device)[None]
            out = self.lm(inputs_embeds=emb[:, s:e], past_key_values=cache, use_cache=True, position_ids=pos)
        return self.m.lm_head(out.last_hidden_state[:, -1:])[0, -1].float()

    @torch.no_grad()
    def prefill(self, p, audio):
        return self._prefill_embeds(R.prompt_embeddings(self.m, p, audio), self._cache())

    @torch.no_grad()
    def decode_setup(self, name, p, audio, C):
        """Cache with C - 1 entries (prompt + teacher-forced reference tokens); returns (step, crop)."""
        r = ref(name)
        T, toks = r["T"], r["tokens"]
        n_tf = C - 1 - T
        assert 0 <= n_tf < len(toks) + 64, (C, T, len(toks))
        tf = (toks * ((n_tf + 1) // len(toks) + 1))[: n_tf + 1]  # past the reference length: repeat it (shape only)
        emb = R.prompt_embeddings(self.m, p, audio)
        if n_tf:
            emb = torch.cat([emb, self.lm.embed_tokens(torch.tensor([tf[:n_tf]], device=self.device))], dim=1)
        cache = self._cache()
        self._prefill_embeds(emb, cache)
        assert cache.get_seq_length() == C - 1
        tok = torch.tensor([[tf[n_tf]]], device=self.device)
        pos = torch.tensor([[C - 1]], device=self.device)

        @torch.no_grad()
        def step():
            e = self.lm.embed_tokens(tok)
            out = self.lm(inputs_embeds=e, past_key_values=cache, use_cache=True, position_ids=pos)
            return self.m.lm_head(out.last_hidden_state)[0, -1].float()
        return step, lambda: cache.crop(-1)  # drop the step's token: back to C - 1 entries

    def transcribe(self, p):
        return R.manual_generate(self.m, p)


def torch_notes(impl: str, q: TorchQwen) -> str:
    dev, compiled = TORCH_IMPLS[impl]
    thr = f", {torch.get_num_threads()} threads" if dev == "cpu" else ", MPS"
    return (f"torch {torch.__version__}, transformers {version('transformers')}{thr}; {SRC} "
            f"({q.f16_check['attn_impl']}); f16: R.load_model(torch.float16), every parameter and buffer f16 "
            f"(checked){' (CPU f16 = PyTorch generic f16 path)' if dev == 'cpu' else ''}"
            f"{'; torch.compile(dynamic=True) on audio tower, projector, LM, LM head' if compiled else ''}; "
            f"verified: s10 / s30 greedy tokens = the f32 CPU reference")


def run_torch(impl: str, cases: list[str], verify_only: bool, out: list, rec: bool):
    dev, compiled = TORCH_IMPLS[impl]
    print(f"== {impl}: loading f16 on {dev}{' (torch.compile)' if compiled else ''}", flush=True)
    R.memory_guard(4.5, f"{impl}: load")
    q = TorchQwen(dev, compiled)
    print(f"   f16 check: {q.f16_check}", flush=True)
    ok = verify(impl, lambda c: q.transcribe(prep(c)), out, extra=q.f16_check)
    if verify_only or not ok:
        return
    notes = torch_notes(impl, q)
    for case in cases:
        kind, clip, at = parse_case(case)
        try:
            R.memory_guard(case_kv_gb(kind, clip, at), f"{impl} {case}")
            if kind == "e2e":
                ps = [prep(c) for c in passes(clip)]
                res = timeit(lambda: [q.transcribe(p) for p in ps], q.sync)
                res["passes"] = len(ps)
                emit(out, impl, canon(case), res, ps[0], rec, notes)
                continue
            p = prep(clip)
            audio = q.encoder(p)
            q.sync()
            if kind == "enc":
                res = timeit(lambda: q.encoder(p), q.sync)
            elif kind == "prefill":
                res = timeit(lambda: q.prefill(p, audio), q.sync)
            else:
                C = decode_context(clip, at)
                step, crop = q.decode_setup(clip, p, audio, C)
                res = timeit(step, q.sync, crop)
                res["context"] = C
                del step, crop
            del audio
            emit(out, impl, canon(case), res, p, rec, notes)
        except (MemoryError, RuntimeError) as e:  # OOM / backend failure: nothing recorded for this case
            print(f"   {impl} {case}: FAILED ({type(e).__name__}: {str(e)[:300]}); not recorded", flush=True)
            out.append({"impl": impl, "case": canon(case), "error": f"{type(e).__name__}: {str(e)[:300]}"})
        gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()


# ---------------------------------------------------------------- MLX (separate venv, runners/qwen3asr_mlx.py)

def mlx_model_dir() -> Path:
    """The f16 mlx-layout copy (<baselines>/models/qwen3-asr-mlx-f16, scripts/convert_qwen3asr_mlx_f16.py) when it
    exists, else the pinned HF snapshot (the runner then converts tensor by tensor)."""
    b = localconfig.get("baselines_dir")
    if b and (Path(b) / "models" / "qwen3-asr-mlx-f16" / "model.safetensors").exists():
        return Path(b) / "models" / "qwen3-asr-mlx-f16"
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(R.HF_ID, revision=R.HF_REV))


def run_mlx(cases: list[str], verify_only: bool, out: list, rec: bool):
    py = localconfig.get("mlx_python")
    if not py or not Path(py).exists():
        print(f"MLX: no Python at {py} (set mlx_python in lokinfer.local.json or $LOKINFER_MLX_PYTHON)")
        return
    parsed = [parse_case(c) for c in cases]
    clip_names = sorted(set(VERIFY_CLIPS) | {pc for _, c, _ in parsed for pc in passes(c)})
    R.memory_guard(4.5 + max([case_kv_gb(*pc) for pc in parsed] or [0]), "MLX: load + longest case")
    with tempfile.TemporaryDirectory() as d:
        arrays = {}
        for c in clip_names:
            p = prep(c)
            r = ref(c)
            arrays[f"feats_{c}"] = p.hf["input_features"].numpy()  # [1, 128, 100 n] f32
            arrays[f"mask_{c}"] = p.hf["input_features_mask"].numpy().astype(np.int32)
            arrays[f"ids_{c}"] = p.ids
            arrays[f"ref_{c}"] = np.array(r["tokens"], np.int64)
            arrays[f"maxnew_{c}"] = np.array(p.max_new, np.int64)
        np.savez(f"{d}/in.npz", **arrays)
        jobs = [["verify", c, None] for c in VERIFY_CLIPS]
        if not verify_only:
            for kind, clip, at in parsed:
                if kind == "e2e":  # a long clip: its passes, transcribed one after another in one timed call
                    jobs.append([kind, "+".join(passes(clip)), None])
                else:
                    jobs.append([kind, clip, decode_context(clip, at) if kind == "decode" else None])
        cmd = [py, "-I", str(MLX_RUNNER), str(mlx_model_dir()), f"{d}/in.npz", json.dumps(jobs)]
        print("== MLX: " + " ".join(cmd[:3]), flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        verified, mlx_info = {}, {}
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                print("   " + line)
                continue
            d_ = json.loads(line)
            if "info" in d_:
                print(f"   {d_}", flush=True)
                mlx_info = d_["info"]
                continue
            if d_["kind"] == "verify":
                r = ref(d_["clip"])
                good = d_["tokens"] == r["tokens"]
                verified[d_["clip"]] = good
                rec_ = {"impl": "MLX", "verify": d_["clip"], "ok": good, "info": mlx_info}
                if not good:
                    rec_.update(text=R.decode(d_["tokens"]), ref_text=r["text"])
                print(f"   verify {d_['clip']}: {'OK' if good else 'MISMATCH'}", flush=True)
                out.append(rec_)
                if not all(verified.values()):
                    print("   MLX transcript mismatch: not timing", flush=True)
                    proc.kill()
                    break
            else:
                first = d_["clip"].split("+")[0]
                clip = re.sub(r"p\d+$", "", first) if "+" in d_["clip"] else d_["clip"]
                case = f"{d_['kind']}:{clip}" + (f"@{d_['context']}" if d_["kind"] == "decode" else "")
                if d_["kind"] == "decode" and d_["context"] == decode_context(clip):
                    case = f"decode:{clip}"
                if "+" in d_["clip"]:
                    d_["passes"] = d_["clip"].count("+") + 1
                notes = (f"mlx {mlx_info.get('mlx')}, mlx-audio {mlx_info.get('mlx_audio')} (qwen3_asr modules, plain "
                         f"MLX, no mx.compile); {SRC} cast astype(float16) per tensor, all parameters f16 (checked) + "
                         f"the encoder sinusoid table cast to f16; verified: s10 / s30 greedy tokens = the f32 CPU "
                         f"reference")
                emit(out, "MLX", case, d_, prep(first), rec, notes)
        proc.wait()
        if proc.returncode not in (0, -9):
            print(f"   MLX runner exited with {proc.returncode}: the cases after the last result were not timed")
            out.append({"impl": "MLX", "error": f"runner exit {proc.returncode}"})


# ---------------------------------------------------------------- llama.cpp / MLX Swift (the harnesses' Mac CLIs)

def baselines() -> Path:
    return Path(localconfig.require("baselines_dir"))


def cli_data() -> Path:
    return baselines() / "qwen3asr" / "data"


def run_cli(cmd: list[str], env: dict | None = None) -> tuple[list[dict], list[str]]:
    """Runs a harness CLI, echoing its output; returns (RESULT dicts, ERROR lines)."""
    import os
    print("   $ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                            env={**os.environ, **(env or {})})
    res, errs = [], []
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("RESULT "):
            r = json.loads(line[7:])
            res.append(r)
            print("   RESULT " + json.dumps({k: v for k, v in r.items() if k not in ("text", "tokens", "ref_text")}),
                  flush=True)
        elif line.startswith("ERROR"):
            errs.append(line)
            print("   " + line, flush=True)
        elif line.startswith("INFO") or line.startswith("MLX device"):
            print("   " + line[:400], flush=True)
    proc.wait()
    if proc.returncode not in (0, 2):
        errs.append(f"exit code {proc.returncode}")
    return res, errs


def need_files(clip_names, exts):
    missing = [f"{c}.{e}" for c in clip_names for e in exts if not (cli_data() / f"{c}.{e}").exists()]
    if missing:
        raise FileNotFoundError(f"missing {missing} in <baselines>/qwen3asr/data: run "
                                f".venv/bin/python <baselines>/scripts/prep_qwen3asr.py {' '.join(clip_names)}")


def cli_cmd(impl: str, clip_for_ctx: str | None) -> list:
    if impl.startswith("llama.cpp"):
        p = prep(clip_for_ctx)
        n_ctx = int(p.ids.shape[0]) + max(p.max_new, len(ref(clip_for_ctx)["tokens"])) + 64
        return [baselines() / "llamacpp-ios" / "llamacpp_asr_mac", cli_data(),
                "metal" if impl.endswith("Metal") else "cpu", LLAMACPP_THREADS, 2, 7, n_ctx]
    prod = baselines() / "qwen3asr-mlx-swift" / "build" / "dd" / "Build" / "Products" / "Release"
    return [prod / "MlxQwen3AsrMac", baselines() / "models" / "qwen3-asr-mlx-f16", cli_data(), 2, 7]


def cli_env(impl: str) -> dict:
    if impl == "MLX Swift":
        prod = baselines() / "qwen3asr-mlx-swift" / "build" / "dd" / "Build" / "Products" / "Release"
        return {"DYLD_FRAMEWORK_PATH": str(prod / "PackageFrameworks")}
    return {}


def cli_notes(impl: str) -> str:
    if impl == "MLX Swift":
        return ("mlx-audio-swift @927d4e30 (patched: sinusoid table cast to the activations' dtype), mlx-swift 0.33.0, "
                "Mac CLI MlxQwen3AsrMac (same Bench.swift as the iPhone app), default MLX memory limits; weights "
                "<baselines>/models/qwen3-asr-mlx-f16 (astype(float16) per tensor, 707/707 f16 checked); lokinfer's "
                "prepared features and prompt ids; verified: s10 / s30 greedy tokens = the f32 CPU reference")
    gpu = impl.endswith("Metal")
    return ("llama.cpp b12125 (7d516e25, GGML_BLAS=OFF), Mac CLI llamacpp_asr_mac (libllama + libmtmd, same core as "
            "the iPhone app); GGUF --outtype f16 LM + mmproj, f16 KV cache, ggml activations f32; "
            + ("Metal: every layer + the mmproj on the GPU" if gpu else
               f"CPU only: {LLAMACPP_THREADS} threads, no device / KV / op offload, mmproj on CPU")
            + "; n_batch 2048, n_ubatch 512, flash-attn auto; verified: s10 / s30 transcripts = the f32 CPU reference "
            "text")


def run_cli_impl(impl: str, cases: list[str], verify_only: bool, out: list, rec: bool):
    """llama.cpp (CPU / Metal) or MLX Swift through the harness's Mac CLI: verification first (one process), then one
    process per clip (llama.cpp: n_ctx sized for that clip; MLX Swift: one process for every case)."""
    print(f"== {impl}: Mac CLI", flush=True)
    parsed = [parse_case(c) for c in cases]
    exts = ("wav", "ref") if impl.startswith("llama.cpp") else ("f32", "ref")
    try:
        need_files(sorted(set(VERIFY_CLIPS) | {pc for _, c, _ in parsed for pc in passes(c)}), exts)
    except FileNotFoundError as e:
        print(f"   {impl}: {e}", flush=True)
        out.append({"impl": impl, "error": str(e)})
        return
    R.memory_guard(4.5 + R.kv_gb(1024), f"{impl}: verify")
    res, errs = run_cli(cli_cmd(impl, "s30") + [f"verify:{c}" for c in VERIFY_CLIPS], cli_env(impl))
    ok_all = len(res) == len(VERIFY_CLIPS) and all(r["ok"] for r in res) and not errs
    for r in res:
        rc = {"impl": impl, "verify": r["case"].split(":")[1], "ok": r["ok"], "s": r.get("seconds")}
        if not r["ok"]:
            rc.update(text=r.get("text") or R.decode(r.get("tokens", [])), ref_text=ref(rc["verify"])["text"])
        print(f"   verify {rc['verify']}: {'OK' if r['ok'] else 'MISMATCH'}", flush=True)
        out.append(rc)
    if not ok_all:
        print(f"   {impl}: verification failed ({errs}), not timing", flush=True)
        return
    if verify_only:
        return
    notes = cli_notes(impl)
    # jobs: (cli case, the harness case it belongs to); a long clip's e2e = its passes' e2e, summed
    jobs = []
    for case, (kind, clip, at) in zip(cases, parsed):
        if kind == "e2e":
            jobs += [(f"e2e:{pc}", canon(case)) for pc in passes(clip)]
        else:
            jobs.append((f"{kind}:{clip}" + (f"@{at}" if at is not None else ""), canon(case)))
    groups: dict[str, list] = {}
    for j in jobs:  # llama.cpp: one process per clip (n_ctx); MLX Swift: one process
        key = re.split("[:@]", j[0])[1] if impl.startswith("llama.cpp") else "all"
        groups.setdefault(key, []).append(j)
    got: dict[str, dict] = {}
    for key, js in groups.items():
        clips_ = {re.split("[:@]", c)[1] for c, _ in js}
        need = max(case_kv_gb(*parse_case(c)) for c, _ in js)
        try:
            R.memory_guard(4.5 + need, f"{impl} {key}")
        except MemoryError as e:
            print(f"   {impl} {key}: {e}; skipped", flush=True)
            out.append({"impl": impl, "group": key, "error": str(e)})
            continue
        ctx_clip = max(clips_, key=lambda c: int(prep(c).ids.shape[0]) + prep(c).max_new)
        res, errs = run_cli(cli_cmd(impl, ctx_clip) + [c for c, _ in js], cli_env(impl))
        if errs:
            out.append({"impl": impl, "group": key, "error": "; ".join(errs)[:500]})
        for r in res:
            got[r["case"]] = r
    # assemble per harness case
    for case in dict.fromkeys(h for _, h in jobs):
        parts = [got.get(c) for c, h in jobs if h == case]
        if not parts or any(p is None for p in parts):
            print(f"   {impl} {case}: no result (see above); not recorded", flush=True)
            continue
        kind, clip, at = parse_case(case)
        if len(parts) == 1:
            r = dict(parts[0])
        else:
            r = {"median_ms": sum(p["median_ms"] for p in parts), "min_ms": sum(p["min_ms"] for p in parts),
                 "runs": min(p["runs"] for p in parts), "warmup": min(p["warmup"] for p in parts),
                 "passes": len(parts), "pass_median_ms": [p["median_ms"] for p in parts],
                 "n_tokens": [p.get("n_tokens") for p in parts]}
            if "llama_prompt_tokens" in parts[0]:
                r["llama_prompt_tokens"] = [p["llama_prompt_tokens"] for p in parts]
        r.pop("state_before", None), r.pop("state_after", None), r.pop("case", None)
        if kind == "decode" and r.get("context") == decode_context(clip) and at is not None:
            case = f"decode:{clip}"
        emit(out, impl, case, r, prep(passes(clip)[0]), rec, notes)


# ---------------------------------------------------------------- shared

def verify(impl, tokens_fn, out, extra=None, text_fn=None) -> bool:
    """Transcripts of VERIFY_CLIPS vs the f32 CPU reference (tokens when the impl gives tokens, else text)."""
    ok_all = True
    for c in VERIFY_CLIPS:
        r = ref(c)
        t = time.perf_counter()
        if tokens_fn is not None:
            toks = tokens_fn(c)
            ok, text = toks == r["tokens"], R.decode(toks)
        else:
            text = text_fn(c)
            ok = text.strip() == r["text"].strip()
        rec = {"impl": impl, "verify": c, "ok": ok, "s": round(time.perf_counter() - t, 2), **(extra or {})}
        if not ok:
            rec.update(text=text, ref_text=r["text"])
        print(f"   verify {c}: {'OK' if ok else 'MISMATCH'} ({rec['s']} s)" + ("" if ok else f"\n     got: {text!r}"
              f"\n     ref: {r['text']!r}"), flush=True)
        out.append(rec)
        ok_all &= ok
    if not ok_all:
        print(f"   {impl}: transcript mismatch, not timing", flush=True)
    return ok_all


def case_notes(impl: str, case: str, res: dict, p: R.Prepared) -> str:
    kind, clip, _ = parse_case(case)
    s = f"T = {int(p.ids.shape[0])}, N_a = {p.n_audio}"
    if kind == "decode":
        s = f"one step at context C = {res['context']} (T = {int(p.ids.shape[0])} + the reference transcript)"
    if res.get("passes", 1) > 1:
        s = (f"{clips.SPEECH[clip]} s clip split by the caller (R.split_long) into {res['passes']} passes of <= 20 min"
             + (", each timed separately, value = the sum of the pass medians" if "pass_median_ms" in res else
                ", both in one timed call"))
    if impl.startswith("llama.cpp") and kind in ("enc", "e2e") and "llama_prompt_tokens" in res:
        s += (f"; llama.cpp's own audio preprocessing: prompt {res['llama_prompt_tokens']} tokens "
              "(pads the last partial 1 s chunk)")
    if kind == "e2e" and "transcript_ok" in res and res.get("passes", 1) == 1:
        s += (f"; its transcript {'=' if res['transcript_ok'] else '!='} the reference ({res.get('n_tokens')} tokens "
              f"vs {len(ref(clip)['tokens'])})")
    return s


def emit(out, impl, case, res, p, rec=False, notes=""):
    r = {"impl": impl, "case": case, "T": int(p.ids.shape[0]), "n_audio": p.n_audio, **res}
    print("   " + json.dumps(r), flush=True)
    out.append(r)
    if not rec:
        return
    dc = dash_case(*parse_case(case))
    if dc is None:
        return
    device, dimpl = DASH_IMPL[impl]
    full = f"{notes}; {case_notes(impl, case, res, p)}; median of {res['runs']} runs after {res['warmup']} warm-up calls"
    if perfdb.record(MODEL, device, dc, dimpl, res["median_ms"], state=perfdb.machine_state(), notes=full):
        print(f"   recorded {device} / {dc} / {dimpl}: {res['median_ms']:.1f} ms", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", action="append", choices=ALL_IMPLS)
    ap.add_argument("--case", nargs="+", default=list(DEFAULT_CASES))
    ap.add_argument("--verify", action="store_true", help="only verify the transcripts (no timing)")
    ap.add_argument("--record", action="store_true", help="record every timed dashboard case in perf/results.json")
    a = ap.parse_args()
    if a.case == ["dashboard"]:
        a.case = list(perfdb.DEFAULT_META["models"][MODEL]["cases"])
    impls = a.impl or list(TORCH_IMPLS)
    for c in VERIFY_CLIPS:
        ref(c)
    for c in a.case:
        for pc in passes(parse_case(c)[1]):
            ref(pc)
    drop_ref_model()
    gc.collect()
    out: list = []
    state = perfdb.machine_state()
    for impl in impls:
        try:
            if impl in TORCH_IMPLS:
                run_torch(impl, a.case, a.verify, out, a.record)
            elif impl == "MLX":
                run_mlx(a.case, a.verify, out, a.record)
            else:
                run_cli_impl(impl, a.case, a.verify, out, a.record)
        except MemoryError as e:
            print(f"== {impl}: stopped: {e}", flush=True)
            out.append({"impl": impl, "error": str(e)})
        gc.collect()  # the impl's model is freed before the next one loads (one big model at a time)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / "results.jsonl", "a") as f:
        for rec in out:
            f.write(json.dumps({**rec, "state": state, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "torch": torch.__version__}) + "\n")


if __name__ == "__main__":
    main()
