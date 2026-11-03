"""Kokoro-82M reference data (porting step 2).

Runs the exact math of english-practice's `kokoro_model.infer()` (KModel
forward on CPU, f32) stage by stage and saves every intermediate plus the
SineGen noise tensor, so lokinfer can be tested stage by stage and end to end.

    .venv/bin/python -m models.kokoro.reference            # all cases
    .venv/bin/python -m models.kokoro.reference real117    # one case

Output: ~/.cache/lokinfer/kokoro/ref/<case>.npz (see docs/impl-notes/kokoro-port-plan.md).
Does not import lokinfer (safe to run while the extension rebuilds).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HF_REPO = "hexgrad/Kokoro-82M"
HF_REV = "f3ff3571791e39611d31c381e3a41a3af07b4987"
REF_DIR = Path.home() / ".cache" / "lokinfer" / "kokoro" / "ref"

REAL_TEXTS = {
    "real28": ("Hello world, this is a test.", "af_heart", 1.0),
    "real117": (
        "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore, "
        "and the shells she sells are surely seashells.", "am_michael", 1.0),
    "real_speed125": ("Please speak a little faster, if you can.", "bf_emma", 1.25),
    "real_speed08": ("And now, slowly and carefully.", "af_heart", 0.8),
    # The real workload (2026-11-01): Kokoro sentences of the app's 1000-sentence TTS pack have T = 34..83 tokens
    # (p5 47, median 59, p95 71), F = 98..224 frames (2.5-5.6 s); the app speaks with two voices, af_heart (f) and
    # am_echo (m, ~10-15 % more frames). One real sentence at p5 / median / p95 in both voices, a one-word utterance
    # (short edge), and the median sentence in three more voices (af_nicole: the slowest of the 28 English voices,
    # +40 % frames; bf_emma: British G2P; am_fenrir: among the fastest). docs/models/kokoro.md "Representative shapes".
    "word": ("Thanks!", "af_heart", 1.0),
    **{f"app{n}{g}": (text, voice, 1.0)
       for n, text in ((47, "The thrift store clothes are thinner than they look."),
                       (59, "My mother and brother bathe the dog together on Sunday."),
                       (71, "Mathematics students follow a strict method when they prove theorems."))
       for g, voice in (("f", "af_heart"), ("m", "am_echo"))},
    "app59_nicole": ("My mother and brother bathe the dog together on Sunday.", "af_nicole", 1.0),
    "app59_emma": ("My mother and brother bathe the dog together on Sunday.", "bf_emma", 1.0),
    "app59_fenrir": ("My mother and brother bathe the dog together on Sunday.", "am_fenrir", 1.0),
    # Longer utterances (user, 2026-11-01): shadowing will move toward 5-10 s sentences, and occasional TTS of long
    # texts (20-60 s+; KPipeline chunks text into <= 510-token forwards, so a long text = several near-max forwards).
    # s6 / s9: 6 s / 8 s sentences in both app voices; s20: a ~20 s paragraph; s28: a real-text near-max chunk
    # (T = 487, 26 s) next to rand510's random ids.
    **{f"{name}{g}": (text, voice, 1.0)
       for name, text in (
           ("s6", "After work I usually take the long way home, through the park and past the old library."),
           ("s9", "When I moved to the city last spring, I didn't know anyone, so I joined a running club that meets "
                  "every Thursday evening by the river."))
       for g, voice in (("f", "af_heart"), ("m", "am_echo"))},
    **{f"s20{g}": ("I've been trying to practice my English every single day, even when I'm tired. In the morning I "
                   "listen to a short podcast while I make coffee, and I repeat the sentences I like out loud. At "
                   "lunch I read a few pages of a novel, and in the evening I write down three new words and use each "
                   "of them in a sentence about my day.", voice, 1.0)
       for g, voice in (("f", "af_heart"), ("m", "am_echo"))},
    "s28f": ("Last summer my brother and I drove all the way along the coast, from the little fishing town where our "
             "grandparents used to live to the big city in the south. We stopped whenever we saw something "
             "interesting: a market selling fresh peaches, a lighthouse on a cliff. In the evenings we found cheap "
             "hotels near the beach, ate grilled fish, and talked for hours about our childhood. By the end of the "
             "trip we had taken hundreds of photos, but the best part was spending time together again.",
             "af_heart", 1.0),
}
RANDOM_LENGTHS = {"rand1": 1, "rand14": 14, "rand128": 128, "rand256": 256, "rand510": 510}


def snapshot_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HF_REPO, revision=HF_REV, allow_patterns=["config.json", "kokoro-v1_0.pth",
                                                                           "voices/*.pt"]))


_model = None


def load_model():
    global _model
    if _model is None:
        from kokoro.model import KModel

        snap = snapshot_dir()
        _model = KModel(config=str(snap / "config.json"), model=str(snap / "kokoro-v1_0.pth")).to("cpu").eval()
    return _model


def voice_pack(voice: str) -> torch.Tensor:
    return torch.load(snapshot_dir() / "voices" / f"{voice}.pt", weights_only=True)


def text_to_ids(text: str, voice: str):
    """Kokoro's own G2P (misaki via KPipeline), exactly as KPipeline does."""
    from kokoro import KPipeline

    pipe = KPipeline(lang_code=voice[0], model=False)
    model = load_model()
    ps = "".join(r.phonemes for r in pipe(text))
    ids = [i for i in (model.vocab.get(p) for p in ps) if i is not None]
    return ids, ps


def run_case(ids: list[int], ref_s: np.ndarray, speed: float, seed: int = 0) -> dict[str, np.ndarray]:
    """Same steps as kokoro_model.infer (kokoro/model.py forward), capturing intermediates."""
    model = load_model()
    out: dict[str, np.ndarray] = {}
    captured_noise = []
    real_randn_like = torch.randn_like

    def capture_randn_like(t, *a, **k):
        r = real_randn_like(t, *a, **k)
        if t.dim() == 3 and t.shape[-1] == 9 and not captured_noise:  # SineGen noise [1, S, 9]
            captured_noise.append(r.clone())
        return r

    hooks = []

    def save(name, fn=lambda o: o):
        def hook(_m, _inp, o):
            out[name] = fn(o).detach().squeeze(0).numpy().copy()
        return hook

    gen = model.decoder.generator
    hooks.append(model.predictor.shared.register_forward_hook(save("shared", lambda o: o[0])))
    hooks.append(gen.register_forward_pre_hook(
        lambda _m, inp: out.__setitem__("dec_x", inp[0].detach().squeeze(0).numpy().copy())))
    hooks.append(gen.m_source.register_forward_hook(save("har_source", lambda o: o[0][..., 0])))
    hooks.append(gen.noise_convs[0].register_forward_pre_hook(
        lambda _m, inp: out.__setitem__("har", inp[0].detach().squeeze(0).numpy().copy())))
    hooks.append(gen.conv_post.register_forward_hook(save("post")))
    torch.randn_like = capture_randn_like
    try:
        torch.manual_seed(seed)
        with torch.no_grad():
            self = model
            input_ids = torch.LongTensor([[0, *ids, 0]])
            input_lengths = torch.LongTensor([input_ids.shape[-1]])
            text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
            text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
            bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
            d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
            ref = torch.from_numpy(np.ascontiguousarray(ref_s, dtype=np.float32)).reshape(1, 256)
            s = ref[:, 128:]
            d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = self.predictor.lstm(d)
            logits = self.predictor.duration_proj(x)
            duration = torch.sigmoid(logits).sum(axis=-1) / speed
            pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
            indices = torch.repeat_interleave(torch.arange(input_ids.shape[1]), pred_dur)
            pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]))
            pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
            pred_aln_trg = pred_aln_trg.unsqueeze(0)
            en = d.transpose(-1, -2) @ pred_aln_trg
            F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
            t_en = self.text_encoder(input_ids, input_lengths, text_mask)
            asr = t_en @ pred_aln_trg
            audio = self.decoder(asr, F0_pred, N_pred, ref[:, :128]).squeeze()
    finally:
        torch.randn_like = real_randn_like
        for h in hooks:
            h.remove()
    out.update({
        "ids": np.array(ids, dtype=np.int64),
        "ref_s": np.asarray(ref_s, dtype=np.float32).reshape(256),
        "speed": np.array(speed, dtype=np.float32),
        "bert_dur": bert_dur.squeeze(0).numpy(),
        "d_en": d_en.squeeze(0).transpose(0, 1).contiguous().numpy(),  # [T, 512]
        "d": d.squeeze(0).numpy(),  # [T, 640]
        "lstm_x": x.squeeze(0).numpy(),
        "dur_logits": logits.squeeze(0).numpy(),  # [T, 50]
        "pred_dur": pred_dur.reshape(-1).numpy(),
        "en": en.squeeze(0).transpose(0, 1).contiguous().numpy(),  # [F, 640]
        "F0_pred": F0_pred.squeeze(0).numpy(),
        "N_pred": N_pred.squeeze(0).numpy(),
        "t_en": t_en.squeeze(0).transpose(0, 1).contiguous().numpy(),  # [T, 512]
        "asr": asr.squeeze(0).numpy(),  # [512, F]
        "audio": audio.numpy(),
        "noise": captured_noise[0].squeeze(0).numpy(),  # [600F, 9]
    })
    return out


def build_cases(names=None) -> None:
    REF_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    model = load_model()
    valid_ids = sorted(i for i in model.vocab.values() if i != 0)
    jobs = {}
    for name, (text, voice, speed) in REAL_TEXTS.items():
        jobs[name] = ("text", text, voice, speed)
    for name, n in RANDOM_LENGTHS.items():
        jobs[name] = ("rand", n, "af_heart", 1.0)
    for name, job in jobs.items():
        if names and name not in names:
            continue
        kind, payload, voice, speed = job
        if kind == "text":
            ids, ps = text_to_ids(payload, voice)
        else:
            ids, ps = [int(i) for i in rng.choice(valid_ids, size=payload)], ""
        pack = voice_pack(voice)
        ref_s = pack[len(ids) - 1].numpy().reshape(256)  # KPipeline: pack[len(ps)-1]
        data = run_case(ids, ref_s, speed)
        np.savez_compressed(REF_DIR / f"{name}.npz", **data, voice=np.array(voice), phonemes=np.array(ps))
        F = int(data["pred_dur"].sum())
        print(f"{name}: T={len(ids) + 2} F={F} samples={data['audio'].shape[0]} -> {REF_DIR / (name + '.npz')}")


_model64 = None


def golden_audio(case: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """float64 golden of the same forward with the case's durations and noise
    forced (so F matches): the reference both torch f32 and lokinfer are judged
    against for the phase-sensitive harmonic source / iSTFT path
    (research-kokoro.md §2.5 #2). Returns audio and har_source in float64."""
    global _model64

    if _model64 is None:
        from kokoro.model import KModel

        snap = snapshot_dir()
        _model64 = KModel(config=str(snap / "config.json"), model=str(snap / "kokoro-v1_0.pth")).to("cpu").eval().double()
    self = _model64
    out = {}
    noise = torch.from_numpy(case["noise"].astype(np.float64))[None]
    real_randn_like = torch.randn_like
    used = []

    def forced_randn_like(t, *a, **k):
        if t.dim() == 3 and t.shape[-1] == 9 and not used:
            used.append(1)
            return noise.to(t.dtype)
        return real_randn_like(t, *a, **k)

    # .double() misses two f32 pieces of the phase path: SineGen builds its
    # harmonic multipliers with torch.FloatTensor, and TorchSTFT's window is a
    # plain attribute. Force both to f64 so the golden really is f64.
    gen = self.decoder.generator
    gen.stft.window = gen.stft.window.double()
    real_float_tensor = torch.FloatTensor
    torch.FloatTensor = torch.DoubleTensor
    real_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)  # the encoders' x_pad = torch.zeros(...)
    h = self.decoder.generator.m_source.register_forward_hook(
        lambda _m, _i, o: out.__setitem__("har_source", o[0][..., 0].detach().squeeze(0).numpy().copy()))
    torch.randn_like = forced_randn_like
    try:
        with torch.no_grad():
            ids = [0, *case["ids"].tolist(), 0]
            input_ids = torch.LongTensor([ids])
            input_lengths = torch.LongTensor([len(ids)])
            text_mask = torch.zeros((1, len(ids)), dtype=torch.bool)
            bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
            d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
            ref = torch.from_numpy(case["ref_s"].astype(np.float64)).reshape(1, 256)
            s = ref[:, 128:]
            d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            pred_dur = torch.from_numpy(case["pred_dur"].astype(np.int64))  # forced: same F as the case
            indices = torch.repeat_interleave(torch.arange(len(ids)), pred_dur)
            aln = torch.zeros((len(ids), indices.shape[0]), dtype=torch.float64)
            aln[indices, torch.arange(indices.shape[0])] = 1
            aln = aln.unsqueeze(0)
            en = d.transpose(-1, -2) @ aln
            F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
            t_en = self.text_encoder(input_ids, input_lengths, text_mask)
            asr = t_en @ aln
            out["audio"] = self.decoder(asr, F0_pred, N_pred, ref[:, :128]).squeeze().numpy()
    finally:
        torch.randn_like = real_randn_like
        torch.FloatTensor = real_float_tensor
        torch.set_default_dtype(real_default)
        h.remove()
    return out


def snr_db(x: np.ndarray, ref: np.ndarray) -> float:
    return float(10 * np.log10(np.sum(ref.astype(np.float64) ** 2) / max(np.sum((x - ref) ** 2), 1e-300)))


def load_case(name: str) -> dict[str, np.ndarray]:
    path = REF_DIR / f"{name}.npz"
    if not path.exists():
        build_cases([name])
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


if __name__ == "__main__":
    build_cases(sys.argv[1:] or None)
