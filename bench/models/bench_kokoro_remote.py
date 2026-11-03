"""Kokoro-82M forward on a remote device (the iPhone test host via the remote executor; see
scripts/ios_deploy.sh). The model is compiled on the Mac, uploaded once (cached by sha256), and timed on the
device. Prints the device's thermal state / Low Power Mode when the host reports them, and compares against the
artificial iPhone targets (docs/impl-notes/m2-conv-proof.md).

Usage: .venv/bin/python bench/models/bench_kokoro_remote.py [--udid UDID] [--basic] [case ...]
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.kokoro import model as K  # noqa: E402
from models.kokoro import reference as R  # noqa: E402

from lokinfer.remote import RemoteDevice  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import perfdb  # noqa: E402  (dashboard store: perf/results.json, docs/perf-dashboard.md)

TARGETS_MS = {"real117": 930, "rand510": 2400}  # iPhone CPU targets (Mac bar x conv-throughput ratio 0.86)


def default_udid() -> str:
    out = subprocess.run(["xcrun", "devicectl", "list", "devices"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "physical" in line and "unavailable" not in line:
            for tok in line.split():
                if len(tok) == 25 and tok.count("-") == 1:
                    return tok
    raise SystemExit("no connected physical iPhone")


def main():
    argv = sys.argv[1:]
    udid = argv[argv.index("--udid") + 1] if "--udid" in argv else default_udid()
    cases = [a for a in argv if not a.startswith("--") and a != udid] or ["real28", "real117", "rand510"]
    fused = "--basic" not in argv
    model = K.build(K.fold_weights(R.load_model()), fused=fused)
    dev = RemoteDevice.connect(f"{udid}.coredevice.local")
    t0 = time.perf_counter()
    mid = dev.load(model)
    print(f"loaded ({'fused' if fused else 'basic'}) in {time.perf_counter() - t0:.1f} s; device {dev.state()}")
    for name in cases:
        c = R.load_case(name)
        ids = np.concatenate([[0], c["ids"], [0]]).astype(np.int64)
        inputs = [ids, np.asarray(c["ref_s"], np.float32), np.asarray(c["speed"], np.float32).reshape(()), np.array(0, np.int64)]
        _, times = dev.run(mid, inputs, warmup=1, iters=5)
        ms = sorted(t * 1e-6 for t in times)
        tgt = TARGETS_MS.get(name)
        verdict = "" if tgt is None else f"  target <= {tgt} ms: {'BEATEN' if ms[len(ms) // 2] <= tgt else 'not yet'}"
        st = dev.state()
        print(f"{name:>8} T={len(ids):<4} median {ms[len(ms) // 2]:7.1f} ms  min {ms[0]:7.1f} ms{verdict}  "
              f"(thermal {st.get('thermal', '?')}, LPM {st.get('low_power_mode', '?')})", flush=True)
        perfdb.record("Kokoro-82M", "iPhone CPU", name, "lokinfer fused" if fused else "lokinfer basic",
                      ms[len(ms) // 2], state=dict(st), notes="via remote executor")
    dev.close()


if __name__ == "__main__":
    main()
