"""Kokoro-82M speed-regression test (M3/M5 goal): real117 forward on CPU and GPU must not be slower than the stored
per-device baseline x margin (tests/perf/kokoro_speed.json). Opt-in: LOKINFER_PERF_TESTS=1 (it takes ~30 s and needs
an otherwise idle machine). With LOKINFER_IPHONE=<udid> the same check runs on the phone through the remote executor."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "models"))

pytestmark = pytest.mark.skipif(os.environ.get("LOKINFER_PERF_TESTS") != "1",
                                reason="speed-regression tests are opt-in: LOKINFER_PERF_TESTS=1")

CFG = json.loads((Path(__file__).with_name("kokoro_speed.json")).read_text())


@pytest.fixture(scope="module")
def setup():
    from models.kokoro import model as K
    from models.kokoro import reference as R

    w = K.fold_weights(R.load_model())
    c = R.load_case("real117")
    args = [np.concatenate([[0], c["ids"], [0]]).astype(np.int64), np.asarray(c["ref_s"], np.float32),
            np.asarray(c["speed"], np.float32).reshape(()), np.array(0, np.int64)]
    return K, w, args


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_kokoro_speed_mac(setup, device):
    from bench_kokoro import timeit

    K, w, args = setup
    model_id = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True).stdout.strip()
    base = CFG["devices"].get(model_id)
    if base is None:
        pytest.skip(f"no baseline for {model_id} in kokoro_speed.json")
    m = K.build(w, gpu=device == "gpu")
    ms = timeit(lambda: m(args[0], args[1], np.float32(args[2]), 0), warmup=2, runs=7)["median_ms"]
    limit = base[device] * CFG["margin"][device]
    print(f"{model_id} {device}: {ms:.1f} ms (baseline {base[device]}, limit {limit:.1f})")
    assert ms <= limit, f"Kokoro real117 {device} regressed: {ms:.1f} ms > {limit:.1f} ms (baseline {base[device]})"


@pytest.mark.skipif(not os.environ.get("LOKINFER_IPHONE"), reason="set LOKINFER_IPHONE=<udid>")
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_kokoro_speed_iphone(setup, device):
    from bench_remote_models import timed_run

    from lokinfer.remote import RemoteDevice

    K, w, args = setup
    dev = RemoteDevice.connect(f"{os.environ['LOKINFER_IPHONE']}.coredevice.local", timeout=900.0)
    base = CFG["devices"][dev.state()["device"]]
    ms, st, end_thermal, n = timed_run(dev, dev.load(K.build(w, gpu=device == "gpu")), args)
    dev.close()
    limit = base[device] * CFG["margin"][device]
    print(f"iPhone {device}: {ms:.1f} ms (baseline {base[device]}, limit {limit:.1f}; thermal {st.get('thermal')} -> "
          f"{end_thermal})")
    assert ms <= limit, f"Kokoro real117 iPhone {device} regressed: {ms:.1f} ms > {limit:.1f} ms"
