"""Kokoro-82M forward: lokinfer vs PyTorch (english-practice's exact infer path).

Times the model forward only (ids + style + speed -> audio), f32, on the
reference cases. Records Low Power Mode / thermal state with the results.
Usage: .venv/bin/python bench/models/bench_kokoro.py [case ...] [--no-mps] [--no-lokinfer]
(no case = every Kokoro case of the dashboard, perfdb.DEFAULT_META). The torch output length is checked against the
reference (600·F samples, length_check); a torch row more than 0.2 % off is not recorded.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.kokoro import model as K  # noqa: E402
from models.kokoro import reference as R  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import perfdb  # noqa: E402  (dashboard store: perf/results.json, docs/perf-dashboard.md)

# results key -> (dashboard device, dashboard impl)
PERF_IMPLS = {"torch_cpu": ("Mac CPU", "torch eager"), "torch_mps": ("Mac GPU", "torch MPS"),
              "lokinfer_cpu_basic": ("Mac CPU", "lokinfer basic"), "lokinfer_cpu_fused": ("Mac CPU", "lokinfer fused")}


def machine_state() -> dict:
    pm = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    low = any(l.split()[:2] == ["lowpowermode", "1"] for l in pm.splitlines())
    load = subprocess.run(["sysctl", "-n", "vm.loadavg"], capture_output=True, text=True).stdout.strip()
    return {"low_power_mode": low, "loadavg": load}


def timeit(fn, warmup=2, runs=7, min_warm_s=1.0, min_run_s=2.0):
    """Median timing with the reference apps' protocol: >= `warmup` calls and >= ~1 s of warm-up, then >= `runs` calls
    and >= ~2 s measured (lets the GPU / CPU clocks ramp up; short models get many runs, long ones keep `runs`)."""
    t_end = time.perf_counter() + min_warm_s
    n = 0
    while n < warmup or (time.perf_counter() < t_end and n < 5000):
        fn()
        n += 1
    ts = []
    t_end = time.perf_counter() + min_run_s
    while len(ts) < runs or (time.perf_counter() < t_end and len(ts) < 5000):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return {"median_ms": 1e3 * ts[len(ts) // 2], "p10_ms": 1e3 * ts[max(0, len(ts) // 10)],
            "p90_ms": 1e3 * ts[min(len(ts) - 1, (9 * len(ts)) // 10)]}


def length_check(n: int, ref_n: int, who: str) -> tuple[bool, str]:
    """The apples-to-apples work rule (docs/baselines.md §2): the output must be the reference length (600·F samples).
    A framework that predicts its own durations may land a frame off: within 0.2 % it is comparable and the difference
    goes into the entry's notes; beyond that it is not recorded."""
    if n == ref_n:
        return True, "output length matches the reference"
    return (abs(n - ref_n) <= 0.002 * ref_n,
            f"{who}'s own durations give F = {n // 600} vs reference {ref_n // 600} ({100 * (n - ref_n) / ref_n:+.2f} % work)")


TIMING_NOTE = "median of >= 7 runs and >= 2 s after >= 1 s of warm-up"


def torch_infer(model, c, device):
    """english-practice's kokoro_model.infer() math (KModel internals), unchanged."""
    ids_list = c["ids"].tolist()
    ref_bytes = c["ref_s"].tobytes()
    speed = float(c["speed"])

    def run():
        self = model
        with torch.no_grad():
            input_ids = torch.LongTensor([[0, *ids_list, 0]]).to(device)
            input_lengths = torch.LongTensor([input_ids.shape[-1]]).to(device)
            text_mask = torch.arange(input_lengths.max()).unsqueeze(0).expand(input_lengths.shape[0], -1).type_as(input_lengths)
            text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(device)
            bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int())
            d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
            ref_s = torch.frombuffer(bytearray(ref_bytes), dtype=torch.float32).reshape(1, 256).to(device)
            s = ref_s[:, 128:]
            d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = self.predictor.lstm(d)
            duration = self.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1) / speed
            pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
            indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=device), pred_dur)
            pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=device)
            pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
            pred_aln_trg = pred_aln_trg.unsqueeze(0).to(device)
            en = d.transpose(-1, -2) @ pred_aln_trg
            F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
            t_en = self.text_encoder(input_ids, input_lengths, text_mask)
            asr = t_en @ pred_aln_trg
            audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze().cpu()
        return audio

    return run


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cases = args or perfdb.DEFAULT_META["models"]["Kokoro-82M"]["cases"]
    use_mps = "--no-mps" not in sys.argv
    use_lk = "--no-lokinfer" not in sys.argv
    state = machine_state()
    perf_state = perfdb.machine_state()  # Low Power Mode / AC / thermal, stored with each dashboard entry
    print("machine:", state, perf_state)
    lks = []
    if use_lk:
        weights = K.fold_weights(R.load_model())
        lks = [("lokinfer_cpu_basic", K.build(weights, fused=False)),  # M1 basic-kernel graph
               ("lokinfer_cpu_fused", K.build(weights))]  # M3 fused generator (default)
    cpu_ref = R.load_model()
    mps_ref = None
    if use_mps:
        from kokoro.model import KModel

        snap = R.snapshot_dir()
        mps_ref = KModel(config=str(snap / "config.json"), model=str(snap / "kokoro-v1_0.pth")).to("mps").eval()
    rows = []
    for name in cases:
        c = R.load_case(name)
        T, F = len(c["ids"]) + 2, int(c["pred_dur"].sum())
        ids = np.concatenate([[0], c["ids"], [0]]).astype(np.int64)
        row = {"case": name, "T": T, "F": F}
        notes = {}
        for key, ref, dev in (("torch_cpu", cpu_ref, "cpu"), ("torch_mps", mps_ref, "mps")):
            if ref is None:
                continue
            run = torch_infer(ref, c, dev)
            ok, length = length_check(run().numel(), c["audio"].size, "torch " + dev.upper())
            if not ok:
                print(f"{key} {name}: NOT COMPARABLE ({length})")
                continue
            row[key] = timeit(run)
            notes[key] = (f"torch {torch.__version__} eager on {dev}, english-practice's kokoro_model.infer() math (own "
                          f"durations); same inputs as the reference case; {length}; {TIMING_NOTE}")
        for key, m in lks:
            row[key] = timeit(lambda m=m: m(ids, c["ref_s"], np.float32(c["speed"]), 0), warmup=1, runs=5)
        rows.append(row)
        for key, (device, impl) in PERF_IMPLS.items():
            if key in row:
                perfdb.record("Kokoro-82M", device, name, impl, row[key]["median_ms"], state=perf_state,
                              notes=notes.get(key))
        line = "  ".join(f"{k} {v['median_ms']:.0f} ms" for k, v in row.items() if isinstance(v, dict))
        print(f"{name:>14} T={T:<4} F={F:<5} {line}")
    out = Path(__file__).with_name("results_kokoro.json")
    old = json.loads(out.read_text()).get("rows", []) if out.exists() else []  # keep the cases not re-run
    rows = [r for r in old if r["case"] not in cases] + rows
    out.write_text(json.dumps({"machine": state, "rows": rows}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
