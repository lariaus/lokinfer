"""Records the Qwen3-ASR iPhone baseline runs (MLX Swift, llama.cpp CPU / Metal) on the perf dashboard.

The iPhone harness apps (docs/impl-notes/iphone-baselines.md, "Reproducing (Qwen3-ASR)") print one
`RESULT {json}` line per case (`verify:<clip>`, `enc:<clip>`, `prefill:<clip>`, `decode:<clip>`, `e2e:<clip>`).
A log is recorded only if every `verify:` line in it says `"ok": true` (the transcripts equal the f32 CPU reference);
timing cases map to the dashboard names ("enc 10s", ...) as bench_qwen3asr_baselines.py does.

Usage: .venv/bin/python bench/models/record_qwen3asr_iphone_baselines.py [--max-word-diffs N] IMPL LOG [LOG ...]
       IMPL: "MLX Swift" | "llama.cpp CPU" | "llama.cpp Metal"
--max-word-diffs N (default 0 = exact): accept a `verify:` transcript that differs from the reference in at most N
words (word-level diff of the texts, or token-level against <clip>.ref for harnesses that report tokens only; printed
and written into each row's notes). For long clips only, where an implementation's
own input differs (llama.cpp's qwen3a front end adds up to 12 audio tokens per clip, iphone-baselines.md) and an exact
greedy transcript over ~750 words is not expected; short clips are verified exactly.
"""
import argparse
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "bench" / "models"))
sys.path.insert(0, str(ROOT / "python"))
import perfdb  # noqa: E402
from bench_qwen3asr_baselines import MODEL, dash_case, parse_case  # noqa: E402

DEVICE_IMPL = {"MLX Swift": ("iPhone GPU", "MLX Swift"), "llama.cpp CPU": ("iPhone CPU", "llama.cpp"),
               "llama.cpp Metal": ("iPhone GPU", "llama.cpp")}
NOTES = {"MLX Swift": "mlx-audio-swift (patched) + mlx-swift 0.33.0, f16 weights, the harness app MlxQwen3Asr on the "
                      "iPhone; lokinfer's prepared features and prompt ids",
         "llama.cpp CPU": "llama.cpp b12125 iOS framework (FP16 vector arithmetic, no BLAS), the harness app LlamaAsr, "
                          "CPU only (6 threads)",
         "llama.cpp Metal": "llama.cpp b12125 iOS framework, the harness app LlamaAsr, every layer + mmproj on Metal"}


def results(log: Path) -> list[dict]:
    out = []
    for line in log.read_text(errors="replace").splitlines():
        i = line.find("RESULT {")
        if i >= 0:
            out.append(json.loads(line[i + 7:]))
    return out


def ref_tokens(clip: str) -> list[int]:
    """The reference greedy tokens of a clip (`ref` line of <baselines>/qwen3asr/data/<clip>.ref)."""
    from lokinfer import localconfig
    path = Path(localconfig.require("baselines_dir")).expanduser() / "qwen3asr" / "data" / f"{clip}.ref"
    for line in path.read_text().splitlines():
        key, _, val = line.partition(" ")
        if key == "ref":
            return [int(x) for x in val.split()]
    raise ValueError(f"no ref tokens in {path}")


def word_diffs(r: dict) -> list[str]:
    """Differences from the reference: word level from the texts (llama.cpp), else token ids (MLX Swift reports
    tokens only)."""
    if "text" not in r and "tokens" in r:
        a, b = ref_tokens(r["case"].split(":", 1)[1]), r["tokens"]
        return [f"tokens {a[i1:i2]} -> {b[j1:j2]} at {i1}" for op, i1, i2, j1, j2
                in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op != "equal"]
    a, b = (r.get("ref_text") or "").split(), (r.get("text") or "").split()
    return [f"{' '.join(a[i1:i2]) or '-'} -> {' '.join(b[j1:j2]) or '-'}"
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op != "equal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-word-diffs", type=int, default=0)
    ap.add_argument("impl", choices=list(DEVICE_IMPL))
    ap.add_argument("logs", nargs="+", type=Path)
    a = ap.parse_args()
    impl, logs = a.impl, a.logs
    device, name = DEVICE_IMPL[impl]
    for log in logs:
        rs = results(log)
        verify = [r for r in rs if r["case"].startswith("verify:")]
        diffs = {r["case"]: ([] if r.get("ok") else word_diffs(r)) for r in verify}
        bad = [c for c, d in diffs.items() if not d and not next(r for r in verify if r["case"] == c).get("ok")]
        if not verify or bad or any(len(d) > a.max_word_diffs for d in diffs.values()):
            print(f"{log.name}: verification missing or failed ({[(r['case'], r.get('ok')) for r in verify]}, "
                  f"word diffs {diffs}); not recorded")
            continue
        verify_note = "; ".join(f"{c}: {len(d)} differences vs the reference ({', '.join(d)})"
                                for c, d in diffs.items() if d)
        if verify_note:
            print("   accepted:", verify_note)
        for r in rs:
            if r["case"].startswith("verify:"):
                continue
            case = dash_case(*parse_case(r["case"]))
            if case is None:
                print(f"   skip {r['case']} (not a dashboard case)")
                continue
            after = r.get("state_after") or {}
            perfdb.record(MODEL, device, case, name, r["median_ms"],
                          state={"thermal": after.get("thermal"), "low_power_mode": after.get("low_power_mode")},
                          notes=f"{NOTES[impl]}; median of {r.get('runs')} runs; log {log.name}"
                                + (f"; {verify_note}" if verify_note else ""))
            print(f"   {device} {name} {case}: {r['median_ms']:.2f} ms (thermal {after.get('thermal')})")


if __name__ == "__main__":
    main()
