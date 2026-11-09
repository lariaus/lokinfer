"""Qwen3-ASR-1.7B reference: the caller contract and the HF model run stage by stage.

lokinfer ports only the neural model (docs/impl-notes/qwen3-asr-port-plan.md, "The caller contract"). This module is
the reference for everything around it (features, audio-token count, prompt ids, chunked prefill, greedy loop) and
runs the HF model (transformers' native `qwen3_asr`) in any dtype / device for golden data and the precision gate.

    from models.qwen3_asr import reference as R
    prep = R.prepare(audio_16k_f32)                 # features [n, 128, 100], n_audio, ids (+ the HF inputs)
    model = R.load_model(torch.float16, "mps")
    tokens = R.manual_generate(model, prep)         # == model.generate(...) greedy
    text = R.decode(tokens)

Does not import lokinfer.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

HF_ID = "Qwen/Qwen3-ASR-1.7B-hf"
HF_REV = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
SR = 16000
AUDIO_PAD = 151676
EOS = (151643, 151645)
PREFIX = [151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669]  # ... user\n<|audio_start|>
SUFFIX = [151670, 151645, 198, 151644, 77091, 198, 11528, 6364, 151704]  # <|audio_end|> ... language English<asr_text>
PREFILL_CHUNK = 2048  # the caller contract's chunk size (lokinfer's generate model takes T <= 2048 per call)
MAX_PASS_SECONDS = 1200  # qwen-asr 0.0.6 MAX_ASR_INPUT_SECONDS: one model pass is at most 20 min

_processor = None


# --- memory guard: big-model jobs stop instead of silently swapping -------------------------------------------------

def _mem_state() -> tuple[float, float]:
    """(available GB = free + inactive + speculative pages, swap used GB) on macOS."""
    import re
    import subprocess

    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", vm).group(1))
    pages = {k: int(v) for k, v in re.findall(r"Pages (free|inactive|speculative):\s+(\d+)", vm)}
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used = re.search(r"used = ([\d.]+)M", sw)
    return sum(pages.values()) * page / 1e9, (float(used.group(1)) / 1e3 if used else 0.0)


_swap0: float | None = None


def memory_guard(need_gb: float, what: str = ""):
    """Raises before a step that needs `need_gb` more memory when the Mac can't give it without swapping, or when
    swap grew by > 2 GB since the first check of this process (another job is pushing us into swap)."""
    global _swap0
    avail, swap = _mem_state()
    if _swap0 is None:
        _swap0 = swap
    if swap - _swap0 > 2.0:
        raise MemoryError(f"{what}: swap grew {swap - _swap0:.1f} GB during this job; stopping instead of swapping")
    if avail < need_gb + 2.0:  # 2 GB headroom for everything else
        raise MemoryError(f"{what}: needs ~{need_gb:.1f} GB, only {avail:.1f} GB available; stopping instead of swapping")


def kv_gb(tokens: int, dtype=torch.float16) -> float:
    """KV-cache size of the LM for `tokens` positions (28 layers x K, V x 1024)."""
    return tokens * 28 * 2 * 1024 * torch.tensor([], dtype=dtype).element_size() / 1e9


def processor():
    global _processor
    if _processor is None:
        from transformers import AutoProcessor

        _processor = AutoProcessor.from_pretrained(HF_ID, revision=HF_REV)
    return _processor


def load_model(dtype=torch.float32, device: str = "cpu", weights: str = "true", app_buffers: bool = False):
    """weights="true": the bf16 checkpoint (the real model). weights="f16w": the checkpoint rounded to f16 (isolates
    compute error from f16 weight rounding: the G-f16w golden).

    Memory: weights are STORED in their 2-byte form (bf16 for "true", f16 for "f16w"; 4.08 GB) and never held as a
    whole-model f32 / f64 copy. For dtype f16 / bf16 the model simply runs in that dtype. For f32 / f64 (CPU) each
    layer is upcast just before it runs and released after (`_streamed_upcast`): the peak is the 2-byte model + one
    layer in the compute dtype, instead of 8 GB (f32) or 16 GB (f64)."""
    from transformers import AutoModelForMultimodalLM

    if weights not in ("true", "f16w"):
        raise ValueError("weights must be 'true' or 'f16w'")
    m = AutoModelForMultimodalLM.from_pretrained(HF_ID, revision=HF_REV, dtype=torch.bfloat16).eval()
    # Buffers (RoPE inv_freq, the encoder's sinusoid table) are computed in f32 at init. `module.to(f16)` would round
    # them too: f16 inv_freq is off by up to 8.7e-3 relative (its small values are f16 subnormals), i.e. RoPE angles
    # off by up to 3 rad at position 17k — that is what the app's fp16 path does (app_buffers=True reproduces it), but
    # it is not "f16 weights / f16 compute": by default only PARAMETERS are rounded / cast, buffers stay exact (f32).
    exact = {} if app_buffers else {(mod, n): b.detach().clone() for mod in m.modules()
                                    for n, b in mod.named_buffers(recurse=False) if b.is_floating_point()}

    def restore(model):
        for (mod, n), b in exact.items():
            setattr(mod, n, b.to(next(model.parameters()).device))
        return model

    if weights == "f16w":
        m = restore(m.to(torch.float16))  # bf16 -> f16 rounding of the parameters, in place per tensor
    if dtype in (torch.float16, torch.bfloat16):
        return restore(m.to(device=device, dtype=dtype))
    if device != "cpu":
        raise ValueError("f32 / f64 reference runs are CPU only")
    return _streamed_upcast(m, dtype)


def _streamed_upcast(m, dtype):
    """Runs the model in `dtype` while its weights stay in their 2-byte storage: every leaf module with parameters
    gets a pre-hook that swaps in upcast copies of its parameters and a post-hook that restores the 2-byte ones.
    The tied embedding / LM head (311 M params, 2.5 GB in f64) is upcast per use like any other leaf. Activations
    flow in `dtype` (the inputs are cast by prompt_embeddings / the processor tensors)."""
    m._compute_dtype = dtype

    def pre(mod, args):
        mod._stored = {n: p.data for n, p in mod.named_parameters(recurse=False)}
        for n, p in mod.named_parameters(recurse=False):
            p.data = p.data.to(dtype)

    def post(mod, args, out):
        for n, p in mod.named_parameters(recurse=False):
            p.data = mod._stored[n]
        del mod._stored

    skip = {m.model.language_model.embed_tokens, m.lm_head}  # tied table: embed() / lm_logits() slice it instead
    for mod in m.modules():
        if mod not in skip and any(True for _ in mod.parameters(recurse=False)):
            mod.register_forward_pre_hook(pre)
            mod.register_forward_hook(post)
    # buffers (e.g. the encoder's sinusoid table, rotary inv_freq) are small: cast them once
    for mod in m.modules():
        for n, b in mod.named_buffers(recurse=False):
            if b.is_floating_point():
                setattr(mod, n, b.to(dtype))
    return m


def model_dtype(model):
    """The dtype the model computes in (the 2-byte storage dtype is not it for streamed f32 / f64 models)."""
    return getattr(model, "_compute_dtype", next(model.parameters()).dtype)


def n_audio_tokens(n_frames: int) -> int:
    """The processor's audio-token count for F mel frames (processing_qwen3_asr.py)."""
    def c(x):
        return (x - 1) // 2 + 1

    r = n_frames % 100
    return 13 * (n_frames // 100) + (c(c(c(r))) if r else 0)


@dataclass
class Prepared:
    features: np.ndarray  # f32[n, 128, 100]: the 1 s chunks (lokinfer encoder input)
    n_audio: int  # N_a
    ids: np.ndarray  # i64[T] = PREFIX + [AUDIO_PAD] * N_a + SUFFIX
    seconds: float  # audio length (max_new_tokens = 128 + 8 * seconds, the app's rule)
    hf: dict  # the processor's own outputs (torch tensors), for the HF model

    @property
    def max_new(self) -> int:
        return int(128 + 8 * self.seconds)


def prepare(audio: np.ndarray) -> Prepared:
    """The caller contract, steps 1-3, built by the HF processor and checked against our own formulas."""
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    inp = processor().apply_transcription_request(audio=audio, language="English")
    feats = inp["input_features"][0].numpy()  # [128, 100 n]
    n = feats.shape[1] // 100
    assert feats.shape == (128, 100 * n), feats.shape
    chunks = np.ascontiguousarray(feats.reshape(128, n, 100).transpose(1, 0, 2))  # [n, 128, 100]
    ids = inp["input_ids"][0].numpy().astype(np.int64)
    n_audio = int((ids == AUDIO_PAD).sum())
    n_frames = int(inp["input_features_mask"][0].sum())
    assert n_audio == n_audio_tokens(n_frames), (n_audio, n_frames)
    assert ids.tolist() == PREFIX + [AUDIO_PAD] * n_audio + SUFFIX, "unexpected prompt"
    return Prepared(chunks, n_audio, ids, len(audio) / SR, dict(inp))


def _hf_inputs(prep: Prepared, model):
    dev = next(model.parameters()).device
    feats = prep.hf["input_features"].to(dev, model_dtype(model))
    mask = prep.hf["input_features_mask"].to(dev)
    return feats, mask, torch.from_numpy(prep.ids)[None].to(dev)


def _chunk_for(model) -> int:
    """LM chunk length: 2048 (the contract) for 2-byte models; 512 for f32 / f64 CPU runs, whose attention may take
    torch's math path (scores [16, chunk, context] in f32 / f64: 512 x 20k x 16 x 8 B = 1.3 GB at most)."""
    return PREFILL_CHUNK if model_dtype(model) in (torch.float16, torch.bfloat16) else 512


@torch.no_grad()
def embed(model, ids: torch.Tensor) -> torch.Tensor:
    """Token embeddings in the compute dtype; gathers from the 2-byte table, never upcasts the whole table."""
    table = model.model.language_model.embed_tokens.weight
    return torch.nn.functional.embedding(ids, table).to(model_dtype(model))


@torch.no_grad()
def lm_logits(model, h: torch.Tensor, block: int = 16384) -> torch.Tensor:
    """f32 logits of hidden rows h [R, 2048] with the tied head; streamed models multiply by slices of the 2-byte
    table upcast one block at a time (block x 2048 x 8 B = 268 MB in f64)."""
    dt = model_dtype(model)
    w = model.lm_head.weight
    if w.dtype == dt:
        return model.lm_head(h).float()
    return torch.cat([h @ w[i:i + block].to(dt).t() for i in range(0, w.shape[0], block)], dim=-1).float()


@torch.no_grad()
def audio_embeddings(model, prep: Prepared) -> torch.Tensor:
    """Encoder + projector: [N_a, 2048] (the lokinfer encoder's output)."""
    feats, mask, _ = _hf_inputs(prep, model)
    return model.model.get_audio_features(feats, mask).pooler_output.reshape(-1, model.config.text_config.hidden_size)


@torch.no_grad()
def prompt_embeddings(model, prep: Prepared, audio: torch.Tensor | None = None) -> torch.Tensor:
    _, _, ids = _hf_inputs(prep, model)
    audio = audio_embeddings(model, prep) if audio is None else audio
    emb = embed(model, ids)
    mask = (ids == AUDIO_PAD).unsqueeze(-1).expand_as(emb)
    return emb.masked_scatter(mask, audio.to(emb.dtype))


@torch.no_grad()
def _lm_chunks(model, emb: torch.Tensor, cache, start: int, chunk: int):
    """Runs the LM over emb [1, L, D] in chunks continuing a KV cache at position `start`; yields (s, e, hidden)."""
    dev = emb.device
    L = emb.shape[1]
    for s in range(0, L, chunk):
        e = min(L, s + chunk)
        pos = torch.arange(start + s, start + e, device=dev)[None]
        out = model.model.language_model(inputs_embeds=emb[:, s:e], past_key_values=cache, use_cache=True,
                                         position_ids=pos)
        yield s, e, out.last_hidden_state[0]


@torch.no_grad()
def manual_generate(model, prep: Prepared, max_new: int | None = None, chunk: int | None = None,
                    return_logits: bool = False):
    """The caller contract's loop on the HF model: encoder, splice, prefill in chunks, greedy decode.
    Returns the generated ids (incl. the eos if reached); with return_logits, also the f32 logits of every step."""
    from transformers import DynamicCache

    dev = next(model.parameters()).device
    max_new = prep.max_new if max_new is None else max_new
    emb = prompt_embeddings(model, prep)
    cache = DynamicCache(config=model.config.text_config)
    T = emb.shape[1]
    for _, _, h in _lm_chunks(model, emb, cache, 0, chunk or _chunk_for(model)):
        pass
    logits = lm_logits(model, h[-1:])[0]
    toks, all_logits, pos = [], [], T
    while True:
        tok = int(logits.argmax())
        toks.append(tok)
        if return_logits:
            all_logits.append(logits.cpu().numpy())
        if tok in EOS or len(toks) >= max_new:
            break
        e = embed(model, torch.tensor([[tok]], device=dev))
        _, _, h = next(_lm_chunks(model, e, cache, pos, 1))
        logits = lm_logits(model, h[-1:])[0]
        pos += 1
    return (toks, np.stack(all_logits)) if return_logits else toks


@torch.no_grad()
def hf_generate(model, prep: Prepared, max_new: int | None = None):
    """HF's own `generate()` (greedy), for checking manual_generate (2-byte models)."""
    dev = next(model.parameters()).device
    dt = model_dtype(model)
    inp = {k: (v.to(dev, dt) if v.is_floating_point() else v.to(dev)) for k, v in prep.hf.items()}
    out = model.generate(**inp, max_new_tokens=prep.max_new if max_new is None else max_new, do_sample=False)
    return out[0, prep.ids.shape[0]:].tolist()


@torch.no_grad()
def teacher_forced(model, prep: Prepared, tokens: list[int], positions=None, on_hidden=None):
    """The logits that predict tokens[i] (after prompt + tokens[:i]), from a chunked forward over prompt + tokens
    with a KV cache (no autoregressive loop, never the whole sequence at once).

    Returns (argmax i64[n], top-2 margin f32[n]) over all n positions, and the full f32 logits [len(positions), V] at
    `positions` (indices into tokens) when given. `on_hidden(s, e, hidden)` sees every chunk's LM output (captures)."""
    from transformers import DynamicCache

    dev = next(model.parameters()).device
    emb = prompt_embeddings(model, prep)
    if len(tokens) > 1:
        emb = torch.cat([emb, embed(model, torch.tensor([tokens[:-1]], device=dev))], dim=1)
    T = prep.ids.shape[0]
    want = set(positions or [])
    arg, margin, full = np.zeros(len(tokens), np.int64), np.zeros(len(tokens), np.float32), {}
    cache = DynamicCache(config=model.config.text_config)
    for s, e, h in _lm_chunks(model, emb, cache, 0, _chunk_for(model)):
        if on_hidden is not None:
            on_hidden(s, e, h)
        lo, hi = max(s, T - 1), e  # rows predicting token i are T - 1 + i
        if lo >= hi:
            continue
        lg = lm_logits(model, h[lo - s:hi - s])
        top2 = lg.topk(2, dim=-1).values
        idx = np.arange(lo - (T - 1), hi - (T - 1))
        arg[idx] = lg.argmax(-1).cpu().numpy()
        margin[idx] = (top2[:, 0] - top2[:, 1]).cpu().numpy()
        for j, i in enumerate(idx):
            if i in want:
                full[int(i)] = lg[j].cpu().numpy()
    return arg, margin, (np.stack([full[i] for i in positions]) if positions else None)


def teacher_forced_logits(model, prep: Prepared, tokens: list[int], positions=None) -> np.ndarray:
    """Full f32 logits at `positions` (default: all) of the teacher-forced pass."""
    positions = list(range(len(tokens))) if positions is None else list(positions)
    return teacher_forced(model, prep, tokens, positions)[2]


def decode(tokens: list[int]) -> str:
    return processor().decode(torch.tensor([tokens]), return_format="transcription_only")[0]


def split_long(audio: np.ndarray, max_sec: float = MAX_PASS_SECONDS, search_sec: float = 5.0,
               window_ms: float = 100.0) -> list[np.ndarray]:
    """qwen-asr 0.0.6's `split_audio_into_chunks` (the caller's job for audio longer than one pass): near every
    max_sec boundary, within +-search_sec, find the 100 ms window with the lowest sum of |x| and cut at its quietest
    sample. Concatenating the pieces gives the input back exactly. (qwen-asr also zero-pads pieces shorter than its
    minimum input; our passes are never that short.)"""
    audio = np.asarray(audio, dtype=np.float32)
    total, max_n, exp = len(audio), int(max_sec * SR), int(search_sec * SR)
    win = max(4, int(window_ms / 1000.0 * SR))
    out, start = [], 0
    while total - start > max_n:
        cut = start + max_n
        left, right = max(start, cut - exp), min(total, cut + exp)
        if right - left <= win:
            boundary = cut
        else:
            seg = np.abs(audio[left:right])
            sums = np.convolve(seg, np.ones(win, dtype=np.float32), mode="valid")
            w = int(np.argmin(sums))
            boundary = left + w + int(np.argmin(seg[w:w + win]))
        boundary = min(max(boundary, start + 1), total)
        out.append(audio[start:boundary])
        start = boundary
    out.append(audio[start:])
    return out


# --- golden reference data (plan Q1) ------------------------------------------------------------------------------

REF_DIR = os.path.expanduser("~/.cache/lokinfer/qwen3asr/ref")
N_SAMPLED = 16  # decode positions whose full logits are stored (first, last, evenly spaced between)


def sampled_positions(n_tokens: int) -> list[int]:
    if n_tokens <= N_SAMPLED:
        return list(range(n_tokens))
    return sorted({round(i * (n_tokens - 1) / (N_SAMPLED - 1)) for i in range(N_SAMPLED)})


@torch.no_grad()
def capture(model, prep: Prepared, tokens: list[int]) -> dict:
    """One chunked teacher-forced pass over prompt + tokens with hooks: every reference stage of the plan (Q1).

    Encoder: conv2d1/2/3 outputs of the first and last chunk, conv_out output (before positions), encoder layers 0 /
    11 / 23, ln_post, projector output (= the lokinfer encoder's `audio`). LM (prompt rows only): layers 0 / 13 / 27,
    final norm. Logits (f32): the last prompt row (prefill) and the rows predicting tokens[i] at sampled_positions;
    plus the argmax / top-2 margin at every position. Memory: chunked forward, 2-byte weights (load_model)."""
    rec: dict[str, torch.Tensor] = {}
    enc = model.model.audio_tower
    lm = model.model.language_model
    T = prep.ids.shape[0]
    lm_parts: dict[str, list] = {}

    def keep(name, fn=lambda t: t):
        def h(m, i_, o):
            t = o[0] if isinstance(o, tuple) else o
            rec[name] = fn(t.detach()).float().cpu()
        return h

    def keep_lm(name):  # LM hooks fire once per chunk: keep the prompt rows of each chunk
        def h(m, i_, o):
            t = (o[0] if isinstance(o, tuple) else o).detach()[0]
            done = sum(x.shape[0] for x in lm_parts.get(name, []))
            if done < T:
                lm_parts.setdefault(name, []).append(t[: T - done].float().cpu())
        return h

    hs = [enc.conv2d1.register_forward_hook(keep("conv1", lambda t: t[[0, -1]])),
          enc.conv2d2.register_forward_hook(keep("conv2", lambda t: t[[0, -1]])),
          enc.conv2d3.register_forward_hook(keep("conv3", lambda t: t[[0, -1]])),
          enc.conv_out.register_forward_hook(keep("conv_out")),
          enc.ln_post.register_forward_hook(keep("ln_post")),
          lm.norm.register_forward_hook(keep_lm("lm_norm"))]
    for i in (0, 11, 23):
        hs.append(enc.layers[i].register_forward_hook(keep(f"enc{i:02d}")))
    for i in (0, 13, 27):
        hs.append(lm.layers[i].register_forward_hook(keep_lm(f"lm{i:02d}")))
    pos = sampled_positions(len(tokens))
    try:
        audio = audio_embeddings(model, prep)
        arg, margin, step = teacher_forced(model, prep, tokens, pos)
    finally:
        for x in hs:
            x.remove()
    out = {k: v.numpy() for k, v in rec.items()}
    out.update({k: torch.cat(v).numpy() for k, v in lm_parts.items()})
    out.update(audio=audio.float().cpu().numpy(), step_logits=step, step_positions=np.array(pos, np.int64),
               argmax=arg, margin=margin)
    # the prefill logits = the row predicting tokens[0] (position 0 is always sampled)
    out["prefill_logits"] = step[pos.index(0)]
    return out


def ref_path(case: str, part: str = "inputs") -> str:
    """REF_DIR/<case>/<part>.npz: `inputs` (features, n_audio, ids, seconds, tokens) and one file per variant
    (g_true, g_f16w, torch_f16), each written as soon as it is captured (a stopped capture loses at most one)."""
    return os.path.join(REF_DIR, case, f"{part}.npz")


def load_ref(case: str) -> dict:
    """All parts of a case merged: inputs/<k>, tokens, <variant>/<stage>."""
    out = {}
    d = os.path.join(REF_DIR, case)
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz"):
            z = np.load(os.path.join(d, f))
            out.update({k: z[k] for k in z.files})
    return out


def _capture_main(cases: list[str]):
    """G-true (f64, bf16 weights), G-f16w (f64, f16-rounded weights) and torch f16 MPS stages per case, one model in
    memory at a time. Tokens = the f32 CPU greedy transcript (the precision gate checked that f64 greedy agrees at
    every position; s1200 is beyond the gate: its tokens are f32 greedy, its golden is f32 CPU, stated in the npz).
    Writes REF_DIR/<case>.npz with keys <variant>/<stage> and inputs/*."""
    from models.qwen3_asr import clips as C

    os.makedirs(REF_DIR, exist_ok=True)
    preps = {c: prepare(C.load(c)) for c in cases}
    toks: dict[str, list[int]] = {}
    gate = os.path.expanduser("~/.cache/lokinfer/qwen3asr/gate/f32_cpu.npz")
    gz = np.load(gate) if os.path.exists(gate) else None
    for c in cases:  # resume: tokens saved by an earlier (stopped) capture
        if os.path.exists(ref_path(c)):
            toks[c] = np.load(ref_path(c))["tokens"].tolist()
    need = [c for c in cases if c not in toks and (gz is None or f"{c}/tokens" not in gz.files)]
    if need:
        # Cases beyond the gate (s1200): the transcript comes from the fast f16 MPS greedy run. The golden for them is
        # the f32 teacher-forced pass ON this transcript (lokinfer is compared on the same tokens), so the tokens'
        # origin does not matter; a streamed-f32 greedy run would re-upcast all weights for every one of ~4k tokens.
        memory_guard(4.5, "load f16 mps for greedy tokens")
        m = load_model(torch.float16, "mps")
        for c in need:
            memory_guard(kv_gb(preps[c].ids.shape[0] + preps[c].max_new) + 1.5, f"f16 greedy {c}")
            toks[c] = manual_generate(m, preps[c])
            print(f"  f16 mps greedy {c}: {len(toks[c])} tokens", flush=True)
        del m
        torch.mps.empty_cache()
    for c in cases:
        if c not in toks:
            toks.setdefault(c, gz[f"{c}/tokens"].tolist())
    for c, p in preps.items():
        os.makedirs(os.path.dirname(ref_path(c)), exist_ok=True)
        np.savez(ref_path(c), **{"inputs/features": p.features, "inputs/n_audio": np.array([p.n_audio], np.int64),
                                 "inputs/ids": p.ids, "inputs/seconds": np.array([p.seconds]),
                                 "tokens": np.array(toks[c], np.int64)})
    long_ = [c for c in cases if preps[c].seconds > 400]
    variants = [("g_true", torch.float64, "cpu", "true", [c for c in cases if c not in long_]),
                ("g_f16w", torch.float64, "cpu", "f16w", [c for c in cases if c not in long_]),
                ("g_true", torch.float32, "cpu", "true", long_),  # long passes: f32 golden (f64 would take hours)
                ("g_f16w", torch.float32, "cpu", "f16w", long_),
                ("torch_f16", torch.float16, "mps", "true", cases)]
    for name, dt, dev, w, cs in variants:
        cs = [c for c in cs if not os.path.exists(ref_path(c, name))]  # resume: skip what is already captured
        if not cs:
            continue
        print(f"== {name} ({str(dt)[6:]} {dev}, weights {w}): {cs}", flush=True)
        memory_guard(4.5, f"load {name}")  # 2-byte weights (+ one layer upcast)
        m = load_model(dt, dev, w)
        for c in cs:
            ctx = preps[c].ids.shape[0] + len(toks[c])
            memory_guard(kv_gb(ctx, dt) + 1.5, f"{name} {c} (context {ctx})")
            part = {f"{name}/{k}": v for k, v in capture(m, preps[c], toks[c]).items()}
            part[f"{name}/dtype"] = np.array(str(dt)[6:])
            np.savez(ref_path(c, name), **part)
            print(f"  {c}: wrote {ref_path(c, name)}", flush=True)
        del m
        if dev == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    import sys

    from models.qwen3_asr import clips as C

    _capture_main(sys.argv[1:] or [c for c in list(C.SPEECH) if c != "s1800"] + list(C.EDGE))
