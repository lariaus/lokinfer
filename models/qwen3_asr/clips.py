"""Speech clips for Qwen3-ASR tests and benchmarks: LibriSpeech test-clean (CC BY 4.0, public-domain audiobooks).

The audio is downloaded once into ~/.cache/lokinfer/qwen3asr/librispeech/ and never committed; the clip definitions
(speaker, ordered utterance ids, end sample) are committed in clips.json so every machine builds the same clips.

    from models.qwen3_asr import clips
    a = clips.load("s30")            # f32[16 kHz], ~30 s of one speaker reading
    clips.text("s30")                # LibriSpeech's own transcript (for information only)
    clips.edge("e_window")           # synthetic edge clips (padding, chunk / window boundaries, silence, ...)

    .venv/bin/python -m models.qwen3_asr.clips --build    # (re)build clips.json from the downloaded corpus
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
import urllib.request
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

SR = 16000
CACHE = Path.home() / ".cache" / "lokinfer" / "qwen3asr" / "librispeech"
URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
SHA256 = "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23"  # test-clean.tar.gz, 2026-11-05
TABLE = Path(__file__).with_name("clips.json")
# name -> nominal seconds. s1200 / s1800 are Mac-only cases (one 20 min pass; 30 min = two caller-split passes).
SPEECH = {"s2": 2, "s5": 5, "s10": 10, "s30": 30, "s60": 60, "s111": 111, "s300": 300, "s1200": 1200, "s1800": 1800}
EDGE = ("e_short", "e_1s", "e_window", "e_window1", "e_silence", "e_noise", "e_tone")


def _root() -> Path:
    return CACHE / "LibriSpeech" / "test-clean"


def download() -> Path:
    """Downloads and extracts test-clean once (346 MB). Records the archive's sha256 next to it."""
    if _root().is_dir():
        return _root()
    CACHE.mkdir(parents=True, exist_ok=True)
    tgz = CACHE / "test-clean.tar.gz"
    if not tgz.exists():
        urllib.request.urlretrieve(URL, tgz)
    h = hashlib.sha256(tgz.read_bytes()).hexdigest()
    if h != SHA256:
        raise RuntimeError(f"{tgz}: sha256 {h} != expected {SHA256} (delete it to re-download)")
    (CACHE / "test-clean.sha256").write_text(h + "\n")
    with tarfile.open(tgz) as t:
        t.extractall(CACHE, filter="data")
    return _root()


@lru_cache(maxsize=4096)
def _utt_audio(utt: str) -> np.ndarray:
    """One utterance as f32[16 kHz] (FLAC decoded by macOS afconvert, cached as WAV next to it)."""
    spk, chap, _ = utt.split("-")
    flac = _root() / spk / chap / f"{utt}.flac"
    wav_path = flac.with_suffix(".wav")
    if not wav_path.exists():
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(flac), str(wav_path)], check=True,
                       capture_output=True)
    with wave.open(str(wav_path)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def _transcripts() -> dict[str, str]:
    out = {}
    for f in _root().glob("*/*/*.trans.txt"):
        for line in f.read_text().splitlines():
            utt, text = line.split(" ", 1)
            out[utt] = text
    return out


def _quiet_cut(audio: np.ndarray, target: int, search: int = SR, win: int = SR // 10) -> int:
    """The quietest sample of the quietest 100 ms window within +-search of target (qwen-asr's split rule)."""
    lo, hi = max(1, target - search), min(len(audio), target + search)
    seg = np.abs(audio[lo:hi])
    w = int(np.argmin(np.convolve(seg, np.ones(win, np.float32), mode="valid")))
    return lo + w + int(np.argmin(seg[w:w + win]))


def build_table() -> dict:
    """Chooses, per clip, the speaker with the most consecutive audio (chapters in order, utterances in order),
    takes utterances until the nominal length, and cuts at a pause. Short clips use other speakers than long ones
    so the set covers several voices."""
    download()
    utts: dict[str, list[str]] = {}
    for spk_dir in sorted(_root().iterdir()):
        if spk_dir.is_dir():
            utts[spk_dir.name] = sorted(p.stem for p in spk_dir.glob("*/*.flac"))
    dur = {s: sum(len(_utt_audio(u)) for u in us) / SR for s, us in utts.items()}
    by_len = sorted(dur, key=dur.get, reverse=True)
    table, used = {}, set()
    for name, sec in sorted(SPEECH.items(), key=lambda kv: -kv[1]):
        # test-clean has ~8 min per speaker: clips longer than any speaker chain several speakers back to back
        # (still real read speech; s1200 / s1800 only). Shorter clips use one speaker, not reused across short clips.
        one = [s for s in by_len if dur[s] >= sec + 2 and (s not in used or sec >= 300)]
        speakers = one[:1] if one else by_len
        used.update(speakers[:1])
        chosen, n = [], 0
        for u in (u for s in speakers for u in utts[s]):
            chosen.append(u)
            n += len(_utt_audio(u))
            if n >= (sec + 1) * SR:
                break
        spk = speakers[0] if one else "+".join(sorted({u.split("-")[0] for u in chosen}))
        audio = np.concatenate([_utt_audio(u) for u in chosen])
        end = _quiet_cut(audio, sec * SR) if len(audio) > sec * SR + SR else len(audio)
        table[name] = {"speaker": spk, "utterances": chosen, "end": int(end), "seconds": round(end / SR, 3)}
    TABLE.write_text(json.dumps(dict(sorted(table.items(), key=lambda kv: SPEECH[kv[0]])), indent=1) + "\n")
    return table


@lru_cache(maxsize=None)
def _table() -> dict:
    return json.loads(TABLE.read_text())


def load(name: str) -> np.ndarray:
    """A speech clip (SPEECH) or an edge clip (EDGE) as f32[16 kHz]."""
    if name in EDGE:
        return edge(name)
    download()
    c = _table()[name]
    return np.concatenate([_utt_audio(u) for u in c["utterances"]])[: c["end"]].copy()


def text(name: str) -> str:
    """LibriSpeech's transcript of the utterances in the clip (the last one may be cut short by the pause cut)."""
    t = _transcripts()
    return " ".join(t[u] for u in _table()[name]["utterances"])


def edge(name: str) -> np.ndarray:
    """Synthetic edge clips (deterministic): padding path, chunk / window boundaries, silence, noise, a tone.
    The speech-based ones reuse the start of s10 so they are real speech."""
    rng = np.random.default_rng(0)
    if name == "e_short":  # 0.3 s < 0.5 s: the processor pads to 0.5 s (N_a = 7)
        return load("s10")[: int(0.3 * SR)].copy()
    if name == "e_1s":  # exactly one 1 s chunk (F = 100)
        return load("s10")[:SR].copy()
    if name == "e_window":  # 8.00 s: N_a = 104 = exactly one encoder attention window
        return load("s10")[: 8 * SR].copy()
    if name == "e_window1":  # 8.08 s: N_a = 105, a 1-row last window
        return load("s10")[: 8 * SR + int(0.08 * SR)].copy()
    if name == "e_silence":
        return np.zeros(SR, np.float32)
    if name == "e_noise":
        return (0.05 * rng.standard_normal(3 * SR)).astype(np.float32)
    if name == "e_tone":
        t = np.arange(2 * SR) / SR
        return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    raise KeyError(name)


if __name__ == "__main__":
    if "--build" in sys.argv:
        for k, v in build_table().items():
            print(f"{k:6} speaker {v['speaker']:5} {len(v['utterances']):3} utterances  {v['seconds']:8.2f} s")
    else:
        for k in SPEECH:
            print(k, f"{len(load(k)) / SR:.2f} s")
