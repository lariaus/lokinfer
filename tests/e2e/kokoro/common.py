"""Shared helpers of the Kokoro e2e project (paths, run dirs, inputs, WAV I/O, the similarity measure)."""
import json
import os
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = ROOT / "e2e-runs" / "kokoro"  # git-ignored
SR = 24000


def iphone_udid() -> str:
    """The iPhone of the ios_gpu backend: machine-local config (`iphone_udid` in lokinfer.local.json, or
    $LOKINFER_IPHONE; python/lokinfer/localconfig.py)."""
    from lokinfer import localconfig
    return localconfig.require("iphone_udid")

# backend key -> (label, audio subdir)
BACKENDS = {
    "python": "Python (english-practice kokoro_model.infer, PyTorch CPU)",
    "python_b": "Python, 2nd run with other seeds (baseline)",
    "mac_gpu": "lokinfer Mac GPU",
    "ios_gpu": "lokinfer iOS GPU",
}
SEED_BASE = {"python": 1_000_000, "python_b": 7_000_000, "mac_gpu": 1_000_000, "ios_gpu": 1_000_000}


def chunk_seed(backend: str, text_index: int, k: int) -> int:
    """The app seeds chunk k with seed + k (kokoro_tts.cpp's test seed); one seed per text here."""
    return SEED_BASE[backend] + 64 * text_index + k


def run_dir(name: str | None) -> Path:
    """`name` given: that run (created). None: the most recently modified run."""
    if name:
        d = RUNS / name
        d.mkdir(parents=True, exist_ok=True)
        return d
    runs = sorted((p for p in RUNS.glob("*") if p.is_dir()), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("no run yet: start with prepare.py --run NAME")
    return runs[-1]


def arg(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def load_inputs(run: Path):
    """inputs.json (texts + per-chunk graphemes/phonemes/ids) and the ref_s rows (inputs.npz, key '<id>_<k>')."""
    meta = json.loads((run / "inputs.json").read_text())
    npz = np.load(run / "inputs.npz")
    return meta, npz


def encode_pcm16(audio: np.ndarray) -> np.ndarray:
    """kokoro_frontend::encodeWav / soundfile PCM_16, as the app writes its WAVs: lrintf(x * 2^31) >> 16, clipped."""
    x = np.asarray(audio, np.float64) * 2.0**31
    x = np.clip(np.rint(np.nan_to_num(x, nan=0.0)), -(2.0**31), 2.0**31 - 1).astype(np.int64)
    return (x >> 16).astype(np.int16)


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(encode_pcm16(audio).tobytes())


def read_wav(path: Path) -> np.ndarray:
    """float32 in [-1, 1) like soundfile.read(dtype='float32')."""
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


# --- english-practice's own measure, copied from native-server/tests/kokoro_port/fixtures/compare_packs.py
def log_spectrogram(audio):
    n_fft, hop = 1024, 256
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop] * np.hanning(n_fft)
    return np.log1p(np.abs(np.fft.rfft(frames, axis=1)))


def spectral_similarity(a, b):
    n = min(len(a), len(b))
    sa, sb = log_spectrogram(a[:n]).ravel(), log_spectrogram(b[:n]).ravel()
    return float(np.dot(sa, sb) / (np.linalg.norm(sa) * np.linalg.norm(sb) + 1e-12))
# ---


def results_path(run: Path, backend: str) -> Path:
    return run / "results" / f"{backend}.json"


def load_results(run: Path, backend: str) -> dict:
    p = results_path(run, backend)
    return json.loads(p.read_text()) if p.exists() else {"backend": backend, "label": BACKENDS[backend], "texts": {}}


def save_results(run: Path, backend: str, res: dict) -> None:
    p = results_path(run, backend)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, p)
