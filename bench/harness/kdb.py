"""Kernel benchmark results DB (SQLite, `perf/kernels.sqlite`; docs/benchmarking.md "Kernel harness").

One row in `runs` per harness invocation (device, build id, power state, harness kind); one row in `results` per
(run, case). A case is a kernel at one concrete signature (`bench/harness/cases.py`). History is kept: every run
stays, so any two runs can be compared (`kbench compare`).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("LOKINFER_KBENCH_DB", ROOT / "perf" / "kernels.sqlite"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created TEXT NOT NULL,          -- local ISO time
  device TEXT NOT NULL,           -- "Mac CPU" | "Mac GPU" | "iPhone CPU" | "iPhone GPU"
  hardware TEXT,                  -- e.g. "Mac14,6" / "iPhone18,2"
  build_id TEXT,                  -- hash of the lokinfer binary that ran (Mac: _lokinfer .so + metallib; iPhone: LokiHost)
  source_rev TEXT,                -- commit the tree was on when the run started (read from .git, informative)
  harness TEXT NOT NULL,          -- "runtime" (kbench run: 1-kernel graphs) | "gbench" (lokinfer_bench_kernels)
  low_power_mode INTEGER,
  power TEXT,                     -- "AC" | "battery" | null (iPhone)
  thermal_start TEXT,
  thermal_end TEXT,
  label TEXT,
  settings TEXT                   -- JSON: protocol parameters
);
CREATE TABLE IF NOT EXISTS results (
  run_id INTEGER NOT NULL REFERENCES runs(id),
  case_key TEXT NOT NULL,         -- kernel|signature
  kernel TEXT NOT NULL,
  shape TEXT NOT NULL,            -- signature: operand types -> output types {attrs}
  models TEXT,                    -- JSON {model: instances per forward}
  repeats INTEGER,                -- kernel invocations per timed call
  n INTEGER,                      -- timed calls
  median_us REAL,                 -- per kernel invocation: GPU kernels = GPU time (timestamps), CPU = wall time
  p10_us REAL,
  p90_us REAL,
  wall_us REAL,                   -- per invocation, CPU wall time of the whole call / repeats
  gflop REAL,
  tflops REAL,
  thermal TEXT,                   -- at the end of this case
  error TEXT,
  measured_at TEXT,
  PRIMARY KEY (run_id, case_key)
);
CREATE INDEX IF NOT EXISTS results_kernel ON results(kernel, shape);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def new_run(con, *, device: str, hardware: str | None, build_id: str | None, source_rev: str | None, harness: str,
            low_power_mode, power, thermal_start, label: str | None, settings: dict) -> int:
    cur = con.execute(
        "INSERT INTO runs(created, device, hardware, build_id, source_rev, harness, low_power_mode, power, "
        "thermal_start, label, settings) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (now(), device, hardware, build_id, source_rev, harness, None if low_power_mode is None else int(low_power_mode),
         power, thermal_start, label, json.dumps(settings)))
    con.commit()
    return cur.lastrowid


def finish_run(con, run_id: int, thermal_end: str | None) -> None:
    con.execute("UPDATE runs SET thermal_end = ? WHERE id = ?", (thermal_end, run_id))
    con.commit()


def add_result(con, run_id: int, case: dict, **vals) -> None:
    con.execute(
        "INSERT OR REPLACE INTO results(run_id, case_key, kernel, shape, models, repeats, n, median_us, p10_us, p90_us,"
        " wall_us, gflop, tflops, thermal, error, measured_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (run_id, case["key"], case["kernel"], case["sig"], json.dumps(case.get("instances", {})), vals.get("repeats"),
         vals.get("n"), vals.get("median_us"), vals.get("p10_us"), vals.get("p90_us"), vals.get("wall_us"),
         vals.get("gflop"), vals.get("tflops"), vals.get("thermal"), vals.get("error"), now()))
    con.commit()


def runs(con, device: str | None = None) -> list[sqlite3.Row]:
    q = ("SELECT r.*, (SELECT COUNT(*) FROM results x WHERE x.run_id = r.id AND x.error IS NULL) AS ok, "
         "(SELECT COUNT(*) FROM results x WHERE x.run_id = r.id AND x.error IS NOT NULL) AS failed FROM runs r")
    args: tuple = ()
    if device:
        q += " WHERE r.device = ?"
        args = (device,)
    return con.execute(q + " ORDER BY r.id", args).fetchall()


def resolve_run(con, ref: str) -> int:
    """A run id, or "latest" / "latest:<device>" (device as in runs.device, case-insensitive, '-' for ' ')."""
    if ref.isdigit():
        return int(ref)
    if ref.startswith("latest"):
        dev = ref.split(":", 1)[1].replace("-", " ") if ":" in ref else None
        q = "SELECT id FROM runs" + (" WHERE lower(device) = lower(?)" if dev else "") + " ORDER BY id DESC LIMIT 1"
        row = con.execute(q, (dev,) if dev else ()).fetchone()
        if row:
            return row[0]
    raise SystemExit(f"no run matches '{ref}'")


def results(con, run_id: int) -> dict[str, sqlite3.Row]:
    return {r["case_key"]: r for r in con.execute("SELECT * FROM results WHERE run_id = ?", (run_id,))}
