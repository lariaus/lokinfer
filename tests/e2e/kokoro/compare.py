"""Step 3: the difference test.

    .venv/bin/python tests/e2e/kokoro/compare.py [--run NAME]

Measure: english-practice's own (native-server/tests/kokoro_port/fixtures/compare_packs.py), copied in common.py:
spectral_similarity = cosine of the log-magnitude spectrograms (n_fft 1024, hop 256) of the two 16-bit WAVs.
Kokoro's decoder draws random noise on every call, so two waveforms are never bit-identical, not even PyTorch vs
itself; durations are deterministic and must match exactly.

Per text: sample counts (whole text and per chunk) of Mac GPU / iOS GPU vs Python; similarities Mac<->Python,
iOS<->Python, Mac<->iOS, and the baseline Python<->Python-with-other-seeds (python_b, run on every text).

PASS (all must hold):
- every text produced audio on every backend, all finite;
- every duration (sample count) of Mac GPU and iOS GPU equals Python's exactly;
- for Mac<->Python and iOS<->Python: median similarity >= baseline median - TOL_MEDIAN and min similarity >=
  baseline min - TOL_MIN (the baseline is the Python<->Python distribution over the same texts).
Outliers (reported, not a FAIL by themselves): texts whose lokinfer<->Python similarity is below that text's own
Python<->Python similarity by more than TOL_TEXT, or below the baseline's 1st percentile.
Writes <run>/summary.json and <run>/rows.json (read by page.py).
"""
import json
import sys

import numpy as np

from common import BACKENDS, arg, load_inputs, load_results, read_wav, run_dir, spectral_similarity

TOL_MEDIAN = 0.001
TOL_MIN = 0.005
TOL_TEXT = 0.01


def stats(xs):
    xs = np.asarray([x for x in xs if x is not None], np.float64)
    if not xs.size:
        return None
    return {"n": int(xs.size), "min": float(xs.min()), "p1": float(np.percentile(xs, 1)),
            "p5": float(np.percentile(xs, 5)), "median": float(np.median(xs)), "mean": float(xs.mean()),
            "max": float(xs.max())}


def main():
    argv = sys.argv[1:]
    run = run_dir(arg(argv, "--run"))
    meta, _ = load_inputs(run)
    texts = meta["texts"]
    res = {b: load_results(run, b)["texts"] for b in BACKENDS}
    others = ("python_b", "mac_gpu", "ios_gpu")
    rows = []
    for i, t in enumerate(texts):
        row = {"i": i, "id": t["id"], "text": t["text"], "cls": t["cls"], "voice": t["voice"], "source": t["source"],
               "chunks": len(t["chunks"]), "tokens": [len(c["ids"]) + 2 for c in t["chunks"]],
               "phonemes": " | ".join(c["phonemes"] for c in t["chunks"]), "no_audio": t["no_audio"]}
        audio = {}
        for b in BACKENDS:
            r = res[b].get(t["id"])
            row[b] = r and {k: r[k] for k in ("samples", "chunk_samples", "ms", "wall_ms", "error", "nonfinite")
                            if k in r}
            if r and "samples" in r:
                audio[b] = read_wav(run / "audio" / b / f"{t['id']}.wav")
                assert audio[b].size == r["samples"], (b, t["id"])
        py = res["python"].get(t["id"], {})
        for b in others:
            r = res[b].get(t["id"], {})
            row[f"dur_{b}"] = (r.get("samples") == py.get("samples") and r.get("chunk_samples") == py.get("chunk_samples")
                               if "samples" in r and "samples" in py else None)
        pairs = {"sim_mac_py": ("mac_gpu", "python"), "sim_ios_py": ("ios_gpu", "python"),
                 "sim_mac_ios": ("mac_gpu", "ios_gpu"), "sim_base": ("python_b", "python")}
        for k, (a, b) in pairs.items():
            row[k] = spectral_similarity(audio[a], audio[b]) if a in audio and b in audio else None
        flags = []
        for b in ("mac_gpu", "ios_gpu"):
            if row[f"dur_{b}"] is False:
                flags.append(f"duration {b}")
            if b in BACKENDS and res[b].get(t["id"], {}).get("error"):
                flags.append(f"error {b}")
            if res[b].get(t["id"], {}).get("nonfinite"):
                flags.append(f"non-finite {b}")
        row["flags"] = flags
        rows.append(row)

    base = stats(r["sim_base"] for r in rows)
    summary = {"run": run.name, "texts": len(rows), "no_audio_texts": sum(r["no_audio"] for r in rows),
               "chunks": sum(r["chunks"] for r in rows),
               "tolerances": {"median": TOL_MEDIAN, "min": TOL_MIN, "per_text": TOL_TEXT},
               "baseline_python_vs_python": base, "backends": {}}
    verdict = True
    reasons = []
    for b, key in (("mac_gpu", "sim_mac_py"), ("ios_gpu", "sim_ios_py")):
        done = [r for r in rows if not r["no_audio"] and r[b] and "samples" in r[b]]
        expected = sum(not r["no_audio"] for r in rows)
        dur_ok = sum(r[f"dur_{b}"] is True for r in rows)
        s = stats(r[key] for r in rows)
        outl = []
        if base:
            for r in rows:
                if r[key] is None:
                    continue
                if (r["sim_base"] is not None and r[key] < r["sim_base"] - TOL_TEXT) or r[key] < base["p1"]:
                    outl.append({"id": r["id"], "voice": r["voice"], "cls": r["cls"], "sim": round(r[key], 5),
                                 "sim_base_same_text": r["sim_base"] and round(r["sim_base"], 5), "text": r["text"][:120]})
        info ={"texts_done": len(done), "texts_expected": expected, "durations_identical": dur_ok,
                "errors": [{"id": r["id"], "voice": r["voice"], "error": r[b]["error"]} for r in rows
                           if r[b] and "error" in r[b]],
                "nonfinite_texts": [r["id"] for r in rows if r[b] and r[b].get("nonfinite")],
                "duration_mismatches": [{"id": r["id"], "voice": r["voice"], "python": r["python"]["chunk_samples"],
                                         b: r[b]["chunk_samples"], "text": r["text"][:120]}
                                        for r in rows if r[f"dur_{b}"] is False],
                "similarity_vs_python": s, "outliers": outl}
        ok = (len(done) == expected and dur_ok == expected and not info["errors"] and not info["nonfinite_texts"]
              and s is not None and base is not None
              and s["median"] >= base["median"] - TOL_MEDIAN and s["min"] >= base["min"] - TOL_MIN)
        info["pass"] = bool(ok)
        if not ok:
            reasons.append(b)
        verdict &= bool(ok)
        summary["backends"][b] = info
    summary["mac_vs_ios"] = stats(r["sim_mac_ios"] for r in rows)
    summary["baseline_durations_identical"] = sum(r["dur_python_b"] is True for r in rows)
    # timing (setup excluded): per backend totals, real-time factor, medians per length class
    timing = {}
    for b in BACKENDS:
        rr = [(r, r[b]) for r in rows if r[b] and "ms" in r[b]]
        if not rr:
            continue
        ms = sum(x["ms"] for _, x in rr)
        audio_s = sum(x["samples"] for _, x in rr) / 24000
        per_cls = {}
        for cls in sorted({r["cls"] for r, _ in rr}):
            v = [x["ms"] for r, x in rr if r["cls"] == cls]
            per_cls[cls] = {"n": len(v), "median_ms": round(float(np.median(v)), 2)}
        timing[b] = {"texts": len(rr), "total_s": round(ms / 1e3, 2), "audio_s": round(audio_s, 1),
                     "rtf": round(ms / 1e3 / audio_s, 4), "median_ms_per_text": round(float(np.median([x["ms"] for _, x in rr])), 2),
                     "per_class": per_cls}
        if b == "ios_gpu" and all("wall_ms" in x for _, x in rr):
            timing[b]["total_wall_s_incl_transfer"] = round(sum(x["wall_ms"] for _, x in rr) / 1e3, 2)
    summary["timing"] = timing
    summary["pass"] = bool(verdict)
    summary["fail_reasons"] = reasons
    (run / "rows.json").write_text(json.dumps(rows, ensure_ascii=False))
    (run / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))

    print(f"run {run.name}: {len(rows)} texts, {summary['chunks']} chunks")
    if base:
        print(f"baseline Python<->Python (other seeds, {base['n']} texts): median {base['median']:.5f}  min {base['min']:.5f}"
              f"  p1 {base['p1']:.5f}; durations identical {summary['baseline_durations_identical']}/{len(rows)}")
    for b, info in summary["backends"].items():
        s = info["similarity_vs_python"]
        print(f"{BACKENDS[b]}: {info['texts_done']}/{info['texts_expected']} texts, durations identical "
              f"{info['durations_identical']}/{info['texts_expected']}, errors {len(info['errors'])}"
              + (f", similarity vs Python median {s['median']:.5f} min {s['min']:.5f}" if s else "")
              + f", outliers {len(info['outliers'])} -> {'PASS' if info['pass'] else 'FAIL'}")
        for o in info["outliers"]:
            print(f"   outlier {o['id']} {o['voice']} {o['cls']}: {o['sim']} (same text Py<->Py {o['sim_base_same_text']}) "
                  f"{o['text'][:70]!r}")
        for m in info["duration_mismatches"]:
            print(f"   duration mismatch {m['id']} {m['voice']}: python {m['python']} vs {m[b]}  {m['text'][:70]!r}")
        for e in info["errors"]:
            print(f"   error {e}")
    if summary["mac_vs_ios"]:
        print(f"Mac GPU<->iOS GPU: median {summary['mac_vs_ios']['median']:.5f} min {summary['mac_vs_ios']['min']:.5f}")
    for b, tm in timing.items():
        print(f"time {b:>9}: {tm['texts']} texts, {tm['total_s']:8.1f} s for {tm['audio_s']:.0f} s of audio "
              f"(RTF {tm['rtf']:.4f}), median {tm['median_ms_per_text']:.1f} ms/text")
    print("PASS" if verdict else f"FAIL ({', '.join(reasons)})")


if __name__ == "__main__":
    main()
