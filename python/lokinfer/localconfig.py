"""Machine-local settings for the dev tooling (benchmarks, the iPhone harness, research scripts).

They live in `lokinfer.local.json` at the repo root, which is git-ignored (template: `lokinfer.local.example.json`).
Each key can be overridden by an environment variable. Nothing in the library itself depends on this file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # python/lokinfer/localconfig.py -> repo root
FILE = ROOT / "lokinfer.local.json"

# key -> (environment variable, default)
KEYS = {
    "iphone_udid": ("LOKINFER_IPHONE", None),
    "ios_team_id": ("LOKINFER_TEAM", None),
    "ios_bundle_id": ("LOKINFER_BUNDLE_ID", "com.example.lokinfer.testhost"),
    "baselines_dir": ("LOKINFER_BASELINES", None),
    "english_practice_dir": ("LOKINFER_ENGLISH_PRACTICE", None),
    "mlx_python": ("LOKINFER_MLX_PYTHON", None),
    "work_dir": ("LOKINFER_WORK_DIR", str(ROOT / ".cache" / "work")),
}


def _file() -> dict:
    try:
        return json.loads(FILE.read_text())
    except FileNotFoundError:
        return {}


def get(key: str) -> str | None:
    """The value of `key`: its environment variable, else lokinfer.local.json, else the default (may be None)."""
    env, default = KEYS[key]
    v = os.environ.get(env) or _file().get(key) or default
    if v and (key.endswith("_dir") or key == "mlx_python"):  # paths: ~ expanded, relative ones from the repo root
        v = str(ROOT / os.path.expanduser(v))
    return v


def require(key: str) -> str:
    """Like get(), but raises a clear error when the value isn't set."""
    v = get(key)
    if not v:
        env = KEYS[key][0]
        raise RuntimeError(f"{key} is not set: add it to {FILE.name} (see lokinfer.local.example.json) or set ${env}")
    return v


if __name__ == "__main__":  # scripts/lk-config KEY
    import sys

    try:
        print(require(sys.argv[1]))
    except (RuntimeError, KeyError) as e:
        sys.exit(f"lk-config: {e}")
