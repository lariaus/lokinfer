#!/bin/sh
# Links the profiling CLIs into ~/.local/bin (or $1): prof-lokinfer, prof-minfer (alias), mlogs-viewer, kbench, and
# prof-torch unless a different prof-torch already exists there (then use `prof-lokinfer torch ...` or tools/prof-torch).
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:-$HOME/.local/bin}
mkdir -p "$DEST"
for t in prof-lokinfer prof-minfer mlogs-viewer prof-torch kbench; do
  if [ -e "$DEST/$t" ] && [ "$(readlink "$DEST/$t")" != "$ROOT/tools/$t" ]; then
    echo "skip $t: $DEST/$t exists and is not ours"
    continue
  fi
  ln -sf "$ROOT/tools/$t" "$DEST/$t"
  echo "linked $DEST/$t"
done
