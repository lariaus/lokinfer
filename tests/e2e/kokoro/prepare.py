"""Step 1: texts.json -> the model's per-chunk inputs, exactly as english-practice computes them.

    .venv/bin/python tests/e2e/kokoro/prepare.py --run NAME [--limit N]

english-practice's kokoro_frontend::Frontend::prepare (native-server/kokoro_frontend/lib/frontend.cpp, a bit-exact
C++ port of KPipeline, docs/kokoro-cpp-frontend.md there) does, per text:
  KPipeline.__call__ with lang_code 'a' (whatever the voice): split lines on \\n+, misaki G2P, en_tokenize chunks;
  skip chunks with empty phonemes; truncate phonemes > 510 (code points);
  inputIds = vocab ids of the phonemes, characters missing from the vocab dropped, WITHOUT the two 0 pads;
  refS = voicePack row len(phonemes) - 1 (length BEFORE the vocab filter);
  no chunk at all -> FrontendError (the app produces no audio).
Here the same thing runs through the Python KPipeline itself (model=False), which the C++ port is bit-exact with.
Writes <run>/inputs.json (texts + chunks) and <run>/inputs.npz (ref_s rows, key '<id>_<k>').
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402

from common import HERE, arg, run_dir  # noqa: E402


def snapshot_dir():
    """resolveKokoroSnapshotDir() of the app: the HF cache's refs/main snapshot."""
    from huggingface_hub.constants import HF_HUB_CACHE
    from pathlib import Path
    repo = Path(HF_HUB_CACHE) / "models--hexgrad--Kokoro-82M"
    return repo / "snapshots" / (repo / "refs" / "main").read_text().strip()


def main():
    argv = sys.argv[1:]
    run = run_dir(arg(argv, "--run"))
    limit = int(arg(argv, "--limit", 0)) or None
    corpus = json.loads((HERE / "texts.json").read_text())
    texts = corpus["texts"][:limit]

    from kokoro import KPipeline
    pipeline = KPipeline(lang_code="a", model=False)  # the app: lang_code 'a' for every voice
    snap = snapshot_dir()
    vocab = json.loads((snap / "config.json").read_text())["vocab"]
    packs = {}
    arrays = {}
    out = []
    t0 = time.time()
    for t in texts:
        voice = t["voice"]
        if voice not in packs:
            packs[voice] = pipeline.load_voice(voice).numpy().astype(np.float32)  # [510, 1, 256]
        pack = packs[voice]
        chunks = []
        for r in pipeline(t["text"]):  # model=False: yields (graphemes, phonemes) only, same chunking/truncation
            ps = r.phonemes
            if not ps:
                continue
            ids = [vocab[c] for c in ps if c in vocab]
            arrays[f"{t['id']}_{len(chunks)}"] = pack[len(ps) - 1, 0].copy()
            chunks.append({"graphemes": r.graphemes, "phonemes": ps, "ids": ids})
        out.append({**t, "chunks": chunks, "no_audio": not chunks})
    meta = {"snapshot": str(snap), "texts": out}
    (run / "inputs.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    np.savez(run / "inputs.npz", **arrays)
    n_chunks = sum(len(t["chunks"]) for t in out)
    multi = sum(len(t["chunks"]) > 1 for t in out)
    print(f"{run}: {len(out)} texts, {n_chunks} chunks ({multi} multi-chunk texts, "
          f"{sum(t['no_audio'] for t in out)} without audio), max T {max(len(c['ids']) + 2 for t in out for c in t['chunks'])}"
          f" ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
