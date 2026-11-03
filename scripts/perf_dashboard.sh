#!/bin/sh
# Serves the performance dashboard (perf/index.html + perf/results.json) on localhost and opens it.
# Idempotent: reuses the server if it is already up. See docs/perf-dashboard.md.
# Usage: scripts/perf_dashboard.sh [--no-open]   (PERF_PORT overrides the port, default 8765)
set -e
cd "$(dirname "$0")/.."
PORT=${PERF_PORT:-8765}
URL="http://127.0.0.1:$PORT/"
LOG=build/perf_dashboard.log

up() { curl -fsS -o /dev/null --max-time 2 "${URL}results.json" 2>/dev/null; }

if up; then
  echo "perf dashboard already running at $URL"
else
  mkdir -p build
  nohup .venv/bin/python -m http.server "$PORT" --bind 127.0.0.1 --directory perf >"$LOG" 2>&1 &
  i=0
  until up; do
    i=$((i + 1))
    if [ "$i" -ge 50 ]; then  # 5 s: the server failed to start (port taken by something else?)
      echo "perf dashboard failed to start on port $PORT; see $LOG" >&2
      exit 1
    fi
    sleep 0.1
  done
  echo "perf dashboard started at $URL (pid $!, log $LOG)"
fi

[ "$1" = "--no-open" ] || open "$URL"
