"""Qwen3-ASR-1.7B on lokinfer vs the golden reference data (docs/impl-notes/qwen3-asr-port-plan.md Q4, decision 3).

Golden data: `python -m models.qwen3_asr.reference` (Q1) -> ~/.cache/lokinfer/qwen3asr/ref/<case>/{inputs, g_true,
g_f16w, torch_f16}.npz. Weights: the folded f16 checkpoint (`models.qwen3_asr.model.load_weights`, memory-mapped).

- precision, vs G-f16w (f64 compute on the f16-rounded weights) and against torch f16's own error on the same
  quantity (+ a small floor), on the encoder output and on the LM's final normed hidden state of every prompt row
  ("lm_norm"): RMS relative error <= 2x torch's (systematic precision: an f16 accumulation or a wrong rounding raises
  it everywhere) and max relative error <= 4x torch's (a localized gross error: a wrong chunk / row is orders of
  magnitude off). Why not the max alone (2026-11-06): f16 rounding noise amplified through 24 layers makes the single
  worst element a lottery: four equally precise CPU encoder variants (vector / scalar GELU x BNNS / sgemm) gave a s111
  worst row of 8.3e-3 ... 2.26e-2 vs torch's 9.5e-3, while their RMS stayed 1.8-2.4e-3 (torch 2.1e-3);
- logits (one row each: the prefill's last row, the long passes' sampled positions) are ill-conditioned as a precision
  measure: hidden states as accurate as torch's (layer 27: CPU 3.05e-3, GPU 3.15e-3, torch 2.80e-3) gave last-row
  logit RMS from 1.6e-3 to 6.4e-3 depending on the direction of their tiny errors. They are checked for what they
  decide: the argmax = G-f16w's (unless G-true's top-2 gap < 0.05), RMS <= 4x torch's per row (gross errors), and for
  the long passes the RMS pooled over the 16 positions <= 2x torch's pooled RMS;
- transcripts (full greedy loop, clips <= 111 s) = G-true's, except at a near-tie (G-true top-2 logit gap < 0.05);
- long passes (s300, s1200) teacher-forced: the golden tokens are fed in chunks ending at the sampled positions and
  the logits there are compared with G-f16w's (the logit rules above);
- no NaN / inf anywhere.
Devices: Mac CPU and Mac GPU here; `pytest --device=iphone` runs the same graphs on the phone.

These end-to-end tests run on the OPTIMIZED variant only (user, 2026-11-05): the basic f16 kernels are proven by their
independent per-op tests (tests/kernels/test_f16_basic_{cpu,gpu}.py) and are never run through the whole model here.
Until the optimized variant exists (plan Q5) the tests skip.
"""
import gc
import os

import numpy as np
import pytest

from models.qwen3_asr import model as M
from models.qwen3_asr import reference as R

REF = R.REF_DIR
pytestmark = pytest.mark.skipif(not os.path.isdir(REF) or not os.path.exists(os.path.join(M.WEIGHTS_DIR, "DONE")),
                                reason="Qwen3-ASR golden data / folded weights not present (see module docstring)")

# Tiers (tests/conftest.py): the default tier covers every code path (padding, a window boundary, a real transcript
# with decode, silence) in ~1-2 min per device; `slow` adds the other clips and the 300 s pass (milestone runs);
# `long` adds the 20 min pass (release runs; minutes to an hour on the basic kernels).
DEFAULT = ["s2", "s10", "e_short", "e_window1", "e_silence"]
SLOW = ["s5", "s30", "e_1s", "e_window", "e_noise", "e_tone", "s60", "s111"]
slow, long = pytest.mark.slow, pytest.mark.long
ENC_CASES = DEFAULT + [pytest.param(c, marks=slow) for c in SLOW + ["s300"]] + [pytest.param("s1200", marks=long)]
TRANSCRIPT_CASES = DEFAULT + [pytest.param(c, marks=slow) for c in SLOW]
LONG_CASES = [pytest.param("s300", marks=slow), pytest.param("s1200", marks=long)]
NEAR_TIE = 0.05
FLOOR = 2e-4  # absolute floor on the relative-error bounds (torch f16 can be lucky on a tiny case)


def _ref(case):
    path = os.path.join(REF, case)
    if not os.path.isdir(path):
        pytest.skip(f"no golden data for {case}")
    return R.load_ref(case)


RMS_X, MAX_X = 2.0, 4.0  # bounds vs torch f16's own error (module docstring)
LOGIT_RMS_X = 4.0  # per logits row (module docstring)


def _errs(a, b):
    """(max relative error, RMS relative error) of a vs the reference b."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert np.isfinite(a).all(), "non-finite values in the lokinfer output"
    d = a - b
    return (float(np.abs(d).max() / (np.abs(b).max() + 1e-30)),
            float(np.sqrt(np.mean(d * d)) / (np.sqrt(np.mean(b * b)) + 1e-30)))


def _within(ours, torch16):
    """ours / torch16: (max, rms) pairs. Failing bound names, empty when within both."""
    bad = []
    if ours[1] > RMS_X * torch16[1] + FLOOR:
        bad.append(f"rms {ours[1]:.2e} > {RMS_X:g}x torch {torch16[1]:.2e}")
    if ours[0] > MAX_X * torch16[0] + FLOOR:
        bad.append(f"max {ours[0]:.2e} > {MAX_X:g}x torch {torch16[0]:.2e}")
    return bad


class _Out:
    def __init__(self, a):
        self.a = a

    def numpy(self):
        return self.a


class _RemoteGen:
    """--device=iphone / serve: the generate graph run with RUN2 on a KV cache that is a device tensor (it stays on
    the device: the transparent remote path uploads a mutable host array and copies it back on every call, ~0.5 GB
    each way per call for a 5 min clip's cache)."""

    def __init__(self, rdev, gen):
        self.rdev, self.mid = rdev, rdev.load(gen)

    def __call__(self, ids, rows, pos, kv):
        outs, _, _ = self.rdev.run2(self.mid, [ids, rows, pos, kv])
        return [_Out(o) for o in outs]


def _root_conftest(config):
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conftest.py")
    for plugin in config.pluginmanager.get_plugins():
        if os.path.abspath(getattr(plugin, "__file__", "") or "") == root:
            return plugin
    raise RuntimeError("tests/conftest.py is not loaded")


@pytest.fixture(scope="module", params=["cpu", "gpu"])
def models(request):
    """(device, encoder, generate, kv(capacity)) built once per device. The KV cache is a mutable input, allocated per
    case by kv(capacity): a host array / GPU tensor locally, a device tensor on a remote device (freed after the test
    module)."""
    if request.param not in M.OPT_DEVICES:
        pytest.skip(f"the optimized {request.param} kernels are not in yet (plan Q5); basic kernels: op tests")
    sd = M.load_weights_opt()
    enc = M.build_encoder(sd, request.param, variant="opt")
    gen = M.build_generate(sd, request.param, variant="opt", debug_norm=True)  # (logits, every row's lm_norm)
    cf = _root_conftest(request.config)
    rdev = cf._state.get("device") if request.config.getoption("--device") in cf.REMOTE else None
    live = []

    def kv(capacity):
        if rdev is None:
            return M.kv_cache(capacity, request.param)
        for t in live:  # one cache at a time on the device (its memory budget)
            rdev.free_tensor(t)
        live.clear()
        live.append(rdev.alloc_tensor(np.float16, (M.LM_LAYERS, 2, capacity, M.LM_KVH * M.LM_HD)))
        return live[0]

    yield request.param, enc, (gen if rdev is None else _RemoteGen(rdev, gen)), kv
    for t in live:
        rdev.free_tensor(t)
    del enc, gen
    gc.collect()


def _inputs(r):
    return r["inputs/features"], int(r["inputs/n_audio"][0]), r["inputs/ids"], float(r["inputs/seconds"][0])


def _prefill(gen, audio, ids, n_audio, kv, upto=None):
    """Chunked prefill of ids[:upto] (the caller contract): (the last row's logits, every row's lm_norm)."""
    upto = len(ids) if upto is None else upto
    dummy = np.zeros((1, M.LM_D), np.float16)
    logits, normed = None, []
    for s in range(0, upto, 2048):
        e = min(upto, s + 2048)
        k0 = min(max(s - M.AUDIO_START, 0), n_audio)
        k1 = min(max(e - M.AUDIO_START, 0), n_audio)
        rows = audio[k0:k1] if k1 > k0 else dummy
        out = gen(np.ascontiguousarray(ids[s:e]), np.ascontiguousarray(rows), np.array([s], np.int64), kv)
        logits = out[0].numpy()
        normed.append(out[1].numpy())
    return logits[0], np.concatenate(normed)


class _LogitsOnly:
    """The debug_norm generate graph as the caller contract's graph (logits only), for M.transcribe."""

    def __init__(self, gen):
        self.gen = gen

    def __call__(self, *args):
        return self.gen(*args)[0]


def _logit_checks(ours, ref, torch16, margin, what):
    """The logit rules of the module docstring for one row; returns failure strings."""
    bad = []
    o, t = _errs(ours, ref), _errs(torch16, ref)
    if o[1] > LOGIT_RMS_X * t[1] + FLOOR:
        bad.append(f"{what}: logits rms {o[1]:.2e} > {LOGIT_RMS_X:g}x torch {t[1]:.2e}")
    if int(np.argmax(ours)) != int(np.argmax(ref)) and margin >= NEAR_TIE:
        bad.append(f"{what}: argmax {int(np.argmax(ours))} != {int(np.argmax(ref))} (G-true top-2 gap {margin:.3f})")
    return bad


@pytest.mark.parametrize("case", ENC_CASES)
def test_encoder_and_prefill(models, case):
    dev, enc, gen, kvf = models
    r = _ref(case)
    features, n_audio, ids, _ = _inputs(r)
    audio = enc(features, np.array([n_audio], np.int64)).numpy()
    bad = _within(_errs(audio, r["g_f16w/audio"]), _errs(r["torch_f16/audio"], r["g_f16w/audio"]))
    assert not bad, f"{dev} {case} encoder: {'; '.join(bad)}"
    logits, normed = _prefill(gen, audio, ids, n_audio, kvf(len(ids)))
    bad = _within(_errs(normed, r["g_f16w/lm_norm"]), _errs(r["torch_f16/lm_norm"], r["g_f16w/lm_norm"]))
    assert not bad, f"{dev} {case} LM final hidden states (lm_norm): {'; '.join(bad)}"
    bad = _logit_checks(logits, r["g_f16w/prefill_logits"], r["torch_f16/prefill_logits"],
                        float(r["g_true/margin"][0]), "prefill")
    assert not bad, f"{dev} {case}: {'; '.join(bad)}"


@pytest.mark.parametrize("case", TRANSCRIPT_CASES)
def test_transcript(models, case):
    """Full greedy loop (the caller contract) = G-true's transcript, except a first difference at a near-tie."""
    dev, enc, gen, kvf = models
    r = _ref(case)
    features, n_audio, ids, seconds = _inputs(r)
    max_new = int(128 + 8 * seconds)
    toks = M.transcribe(enc, _LogitsOnly(gen), features, n_audio, ids, max_new, kv=kvf(len(ids) + max_new),
                        device=dev)
    gold = r["tokens"].tolist()
    if toks == gold:
        return
    i = next((k for k, (a, b) in enumerate(zip(toks, gold)) if a != b), min(len(toks), len(gold)))
    margin = float(r["g_true/margin"][i]) if i < len(gold) else float("inf")
    assert margin < NEAR_TIE, (f"{dev} {case}: first difference at token {i} where G-true's top-2 gap is {margin:.3f}: "
                               f"{R.decode(toks)!r} vs {R.decode(gold)!r}")


@pytest.mark.parametrize("case", LONG_CASES)
def test_long_teacher_forced(models, case):
    """Decode at long contexts without a long greedy run: feed the golden tokens in chunks that end at the sampled
    positions; the last-row logits there predict the next golden token. Compared with G-f16w's step logits."""
    dev, enc, gen, kvf = models
    r = _ref(case)
    features, n_audio, ids, _ = _inputs(r)
    audio = enc(features, np.array([n_audio], np.int64)).numpy()
    tokens = r["tokens"].tolist()
    seq = np.concatenate([ids, np.array(tokens[:-1], np.int64)])
    T = len(ids)
    positions = r["g_f16w/step_positions"].tolist()
    ends = [T + p for p in positions]  # logits of row T-1+p predict tokens[p]: feed seq[:T+p]
    dummy = np.zeros((1, M.LM_D), np.float16)
    kv = kvf(len(seq))
    got, s = {}, 0
    for end in ends:
        while s < end:
            e = min(end, s + 2048)
            k0 = min(max(s - M.AUDIO_START, 0), n_audio)
            k1 = min(max(e - M.AUDIO_START, 0), n_audio)
            rows = audio[k0:k1] if k1 > k0 else dummy
            out = gen(np.ascontiguousarray(seq[s:e]), np.ascontiguousarray(rows), np.array([s], np.int64), kv)[0].numpy()
            s = e
        got[end] = out[0]
    bad, se, st, sr = [], 0.0, 0.0, 0.0
    for j, end in enumerate(ends):
        ref, t16, ours = (np.asarray(x, np.float64) for x in
                          (r["g_f16w/step_logits"][j], r["torch_f16/step_logits"][j], got[end]))
        bad += _logit_checks(ours, ref, t16, float(r["g_true/margin"][positions[j]]), f"position {positions[j]}")
        se, st, sr = se + ((ours - ref) ** 2).sum(), st + ((t16 - ref) ** 2).sum(), sr + (ref ** 2).sum()
    pooled, pooled_t = np.sqrt(se / sr), np.sqrt(st / sr)
    if pooled > RMS_X * pooled_t + FLOOR:
        bad.append(f"pooled logits rms {pooled:.2e} > {RMS_X:g}x torch {pooled_t:.2e}")
    assert not bad, f"{dev} {case}: {bad}"
