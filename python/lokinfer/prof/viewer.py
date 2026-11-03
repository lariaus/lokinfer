"""mlogs-viewer: a localhost web viewer for profile runs (prof-lokinfer, prof-torch): browse runs, kernel tables, IR
per pass, timelines, memory plans, and compare two runs side by side.

    mlogs-viewer                 # serve the profile directory on http://127.0.0.1:8766/ and open it
    mlogs-viewer --no-open       # server only (idempotent: reuses a running one)
    mlogs-viewer --stop

Routes: /  (the app)   /api/runs   /api/run/<id>   /api/run/<id>/file/<relpath>   /api/compare?a=<id>&b=<id>
        /runs/<id>/report.html (and any other file of a run)
"""
from __future__ import annotations

import json
import mimetypes
import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import analysis
from .collect import runs_dir
from .report import APP

DEFAULT_PORT = 8766


def list_runs(base: Path) -> list[dict]:
    out = []
    if not base.exists():
        return out
    for d in sorted(base.iterdir(), reverse=True):
        f = d / "run.json"
        if not f.is_file():
            continue
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        s = r.get("summary", {})
        out.append({"id": d.name, "kind": r.get("kind"), "model": r.get("model"), "case": r.get("case"),
                    "device": r.get("device"), "created": r.get("created"),
                    "e2e_ms": (s.get("e2e_ms") or {}).get("median"), "kernel_calls": s.get("kernel_calls")})
    out.sort(key=lambda r: r.get("created") or "", reverse=True)
    return out


def _safe_child(base: Path, *parts: str) -> Path | None:
    p = base.joinpath(*parts).resolve()
    try:
        p.relative_to(base.resolve())
    except ValueError:
        return None
    return p


def make_handler(base: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            u = urllib.parse.urlparse(self.path)
            path = urllib.parse.unquote(u.path)
            q = urllib.parse.parse_qs(u.query)
            if path in ("/", "/index.html"):
                return self._send(200, APP.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/runs":
                return self._json(list_runs(base))
            if path == "/api/compare":
                a, b = q.get("a", [""])[0], q.get("b", [""])[0]
                fa, fb = _safe_child(base, a, "run.json"), _safe_child(base, b, "run.json")
                if not (fa and fb and fa.is_file() and fb.is_file()):
                    return self._json({"error": "unknown run"}, 404)
                return self._json(analysis.compare_runs(json.loads(fa.read_text()), json.loads(fb.read_text())))
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 3 and parts[:2] == ["api", "run"]:
                rid = parts[2]
                rel = ["run.json"] if len(parts) == 3 else (parts[4:] if len(parts) > 4 and parts[3] == "file" else None)
                if rel is None:
                    return self._json({"error": "bad path"}, 404)
                f = _safe_child(base, rid, *rel)
                if not f or not f.is_file():
                    return self._json({"error": "not found"}, 404)
                ctype = "application/json" if f.suffix == ".json" else "text/plain; charset=utf-8"
                return self._send(200, f.read_bytes(), ctype)
            if len(parts) >= 3 and parts[0] == "runs":
                f = _safe_child(base, *parts[1:])
                if not f or not f.is_file():
                    return self._json({"error": "not found"}, 404)
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                if ctype.startswith("text/"):
                    ctype += "; charset=utf-8"
                return self._send(200, f.read_bytes(), ctype)
            return self._json({"error": "not found"}, 404)

    return Handler


def serve(base: Path, port: int) -> None:
    base.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(base))
    print(f"mlogs-viewer: http://127.0.0.1:{port}/ serving {base}", flush=True)
    httpd.serve_forever()


def _alive(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs", timeout=1) as r:
            return r.status == 200
    except OSError:
        return False


def ensure_server(base: Path, port: int) -> None:
    """Starts the server in the background (log: <base>/viewer.log) unless one already answers on `port`."""
    if _alive(port):
        return
    base.mkdir(parents=True, exist_ok=True)
    log = open(base / "viewer.log", "ab")
    subprocess.Popen([sys.executable, "-m", "lokinfer.prof.cli", "mlogs-viewer", "--foreground", "--port", str(port),
                      "--dir", str(base)], stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(100):
        if _alive(port):
            return
        time.sleep(0.1)
    raise SystemExit(f"mlogs-viewer did not start; see {base / 'viewer.log'}")


def stop(port: int) -> None:
    out = subprocess.run(["pgrep", "-f", f"lokinfer.prof.cli mlogs-viewer --foreground --port {port}"],
                         capture_output=True, text=True).stdout.split()
    for pid in out:
        os.kill(int(pid), signal.SIGTERM)
    print(f"stopped {len(out)} server(s)")


def url_for(port: int, run_id: str | None = None, compare: tuple[str, str] | None = None) -> str:
    u = f"http://127.0.0.1:{port}/"
    if compare:
        return u + f"#a={urllib.parse.quote(compare[0])}&b={urllib.parse.quote(compare[1])}"
    if run_id:
        return u + f"runs/{urllib.parse.quote(run_id)}/report.html"
    return u


def main(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="mlogs-viewer", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=int(os.environ.get("MLOGS_PORT", DEFAULT_PORT)))
    ap.add_argument("--dir", default=None, help="profile runs directory (default: $LOKINFER_PROF_DIR or <repo>/prof-runs)")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    ap.add_argument("--foreground", action="store_true", help="serve in this process (blocks)")
    ap.add_argument("--stop", action="store_true", help="stop the background server")
    ap.add_argument("--run", help="open this run's report")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="open the side-by-side view of two runs")
    a = ap.parse_args(argv)
    base = Path(a.dir) if a.dir else runs_dir()
    if a.stop:
        return stop(a.port)
    if a.foreground:
        return serve(base, a.port)
    ensure_server(base, a.port)
    url = url_for(a.port, a.run, tuple(a.compare) if a.compare else None)
    print(url)
    if not a.no_open:
        subprocess.run(["open", url], check=False)
