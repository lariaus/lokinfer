"""Self-contained HTML report of one run (report.html in its run directory): the viewer's app with the run and its
IR files embedded, so it opens from disk, from the viewer, or as an attachment, with no server."""
from __future__ import annotations

import json
from pathlib import Path

APP = Path(__file__).with_name("web") / "app.html"
_MARK = "/*EMBEDDED_DATA*/null"


def bundle(run_dir: Path) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    files = {}
    for p in (run.get("ir") or {}).get("passes", []):
        if p.get("file") and (run_dir / p["file"]).exists():
            files[p["file"]] = (run_dir / p["file"]).read_text()
    return {"run": run, "files": files}


def render(data: dict) -> str:
    app = APP.read_text()
    if _MARK not in app:
        raise RuntimeError("app.html lost its embedding marker")
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return app.replace(_MARK, "/*EMBEDDED_DATA*/" + payload, 1).replace(
        "<title>lokinfer profiles</title>",
        f"<title>{_html(data['run'].get('model'))} {_html(data['run'].get('device'))} profile</title>", 1)


def _html(s) -> str:
    return str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "report.html"
    out.write_text(render(bundle(run_dir)))
    return out
