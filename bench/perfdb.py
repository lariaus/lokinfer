"""Performance results store for the dashboard (docs/perf-dashboard.md).

One JSON file, `perf/results.json`, holds the latest measurement per entry. An entry is keyed by
(section, model, device, case, impl); recording it again overwrites only that entry, so any bench can update
what it just ran without re-running everything else.

    from perfdb import record, machine_state
    record("Kokoro-82M", "Mac CPU", "rand510", "lokinfer fused", 1064.2, state=machine_state())

Values are milliseconds per call (E2E, one forward pass; lower is better) unless `unit` says otherwise.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "perf" / "results.json"

# Chart metadata: case order on the X axis and a one-line description per chart group.
DEFAULT_META = {
    "models": {
        "Kokoro-82M": {
            "cases": ['word', 'real28', 'app47f', 'app47m', 'app59f', 'app59m', 'app71f', 'app71m', 's6f', 's6m', 's9f', 's9m', 'real117', 's20f', 's28f', 'rand510'],
            "description": "ids + style + speed -> audio, one forward pass. appNN{f,m} = real sentences at the p5/median/p95 token length (T = 47/59/71, 3-5 s) of the app's TTS workload, in the app's two voices (f = af_heart, m = am_echo); s6/s9{f,m} = 6 s / 8 s sentences; s20f = a 20 s paragraph; s28f = a real-text near-max chunk (T = 487, 26 s); word = one-word edge (T = 9); real28 / real117 = older short / long cases; rand510 = max context with random ids",
        },
        "MNIST MLP": {
            "cases": ["B=1", "B=64", "B=1024"],
            "description": "784-256-10 MLP forward, batch B",
        },
    },
    "devices": ["Mac CPU", "Mac GPU", "iPhone CPU", "iPhone GPU"],
}


def machine_state() -> dict:
    """This Mac's power state: Low Power Mode, AC/battery, thermal warning (benchmarks must record it)."""
    pm = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout
    lpm = any(line.split()[:2] == ["lowpowermode", "1"] for line in pm.splitlines())
    warn = [line.strip() for line in therm.splitlines() if line.strip() and not line.startswith("Note:")]
    return {"low_power_mode": lpm, "power": "AC" if "AC Power" in batt else "battery",
            "thermal": "; ".join(warn) or "nominal"}


def load() -> dict:
    if DB.exists():
        return json.loads(DB.read_text())
    return {"meta": DEFAULT_META, "entries": []}


def _save(db: dict) -> None:
    DB.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=DB.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(db, f, indent=1)
        f.write("\n")
    os.replace(tmp, DB)  # atomic: the dashboard never reads a half-written file


_THROTTLED = ("fair", "serious", "critical")


def record(model: str, device: str, case: str, impl: str, value: float, *, section: str = "models",
           unit: str = "ms", state: dict | None = None, notes: str | None = None,
           measured_at: str | None = None) -> bool:
    """Insert or overwrite one entry. A measurement taken while the device was thermally throttled never replaces a
    *faster* one taken at "nominal" (a hot iPhone is 25-40 % slower; 2026-11-01 a hot sweep overwrote good rows). A
    throttled measurement that is faster than the nominal one does replace it: throttling only slows a run down, so it
    is still a valid (conservative) number, and keeping the old one would hide a real improvement. Returns whether the
    entry was written."""
    db = load()
    key = (section, model, device, case, impl)
    old = next((e for e in db["entries"]
                if (e["section"], e["model"], e["device"], e["case"], e["impl"]) == key), None)
    new_th = (state or {}).get("thermal")
    if (old and (old.get("state") or {}).get("thermal") == "nominal" and new_th in _THROTTLED
            and float(value) >= old["value"]):
        print(f"perfdb: kept the nominal entry for {model} / {device} / {case} / {impl} "
              f"(new measurement was at thermal {new_th})")
        return False
    entry = {"section": section, "model": model, "device": device, "case": case, "impl": impl,
             "value": round(float(value), 4), "unit": unit,
             "measured_at": measured_at or _dt.datetime.now().isoformat(timespec="seconds")}
    if state is not None:
        entry["state"] = state
    if notes:
        entry["notes"] = notes
    db["entries"] = [e for e in db["entries"]
                     if (e["section"], e["model"], e["device"], e["case"], e["impl"]) != key] + [entry]
    db.setdefault("meta", DEFAULT_META)
    _save(db)
    _warn_goal(db, model, device, case)
    return True


GOAL_MODELS = ("Kokoro-82M", "GPT-2 124M")  # docs/todo.md "Goals": lokinfer #1 in every configuration, never regress


def _warn_goal(db: dict, model: str, device: str, case: str) -> None:
    """Loud warning when a goal model's configuration is no longer won by lokinfer (tests/perf/test_dashboard_goals.py
    fails on it too)."""
    if model not in GOAL_MODELS:
        return
    rows = [e for e in db["entries"] if (e["model"], e["device"], e["case"]) == (model, device, case)
            and e["impl"] != "target"]
    lk = [e for e in rows if e["impl"].startswith("lokinfer")]
    base = [e for e in rows if not e["impl"].startswith("lokinfer")]
    if lk and base:
        best_lk, best_b = min(lk, key=lambda e: e["value"]), min(base, key=lambda e: e["value"])
        if best_lk["value"] > best_b["value"]:
            print(f"!!! GOAL REGRESSION: {model} / {device} / {case}: lokinfer {best_lk['value']:.3f} ms > "
                  f"{best_b['impl']} {best_b['value']:.3f} ms (docs/todo.md Goals: #1 everywhere)")


def remove(model: str, device: str | None = None, case: str | None = None, impl: str | None = None,
           section: str = "models") -> int:
    """Delete matching entries (None = any); returns how many were removed."""
    db = load()
    keep = [e for e in db["entries"] if not (
        e["section"] == section and e["model"] == model and device in (None, e["device"])
        and case in (None, e["case"]) and impl in (None, e["impl"]))]
    n = len(db["entries"]) - len(keep)
    db["entries"] = keep
    _save(db)
    return n
