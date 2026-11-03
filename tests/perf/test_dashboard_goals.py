"""User goal (docs/todo.md "Goals", 2026-11-02): lokinfer is #1 in EVERY recorded configuration (device × case) of
Kokoro-82M and GPT-2 124M — never regress it. Reads perf/results.json (no device needed): for each configuration
with a lokinfer number and a baseline, the best lokinfer variant must be at least as fast as the best baseline.
Other models are reported, not enforced (goal 2: within 5-10 %)."""
import collections
import json
from pathlib import Path

import pytest

DB = Path(__file__).resolve().parents[2] / "perf" / "results.json"
GOAL_MODELS = ("Kokoro-82M", "GPT-2 124M")


def configurations():
    entries = json.loads(DB.read_text())["entries"]
    by = collections.defaultdict(lambda: {"lk": None, "base": None})
    for e in entries:
        if e["impl"] == "target":
            continue
        k = (e["model"], e["device"], e["case"])
        side = "lk" if e["impl"].startswith("lokinfer") else "base"
        cur = by[k][side]
        if cur is None or e["value"] < cur[0]:
            by[k][side] = (e["value"], e["impl"])
    return {k: v for k, v in by.items() if v["lk"] and v["base"]}


@pytest.mark.skipif(not DB.exists(), reason="no perf/results.json")
@pytest.mark.parametrize("model", GOAL_MODELS)
def test_goal_model_is_first_everywhere(model):
    cfg = {k: v for k, v in configurations().items() if k[0] == model}
    assert cfg, f"no recorded {model} configurations"
    losses = [f"{d} / {c}: lokinfer {v['lk'][0]:.3f} ({v['lk'][1]}) vs {v['base'][1]} {v['base'][0]:.3f}"
              for (_, d, c), v in sorted(cfg.items()) if v["lk"][0] > v["base"][0]]
    assert not losses, f"{model} is not #1 in {len(losses)}/{len(cfg)} configurations:\n  " + "\n  ".join(losses)


def test_other_models_report():
    cfg = configurations()
    for model in sorted({k[0] for k in cfg} - set(GOAL_MODELS)):
        rows = [v for k, v in cfg.items() if k[0] == model]
        worst = min(v["base"][0] / v["lk"][0] for v in rows)
        print(f"{model}: #1 in {sum(v['lk'][0] <= v['base'][0] for v in rows)}/{len(rows)}, worst ratio {worst:.2f}x")
