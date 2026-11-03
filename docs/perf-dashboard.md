# Performance dashboard

**[built]** The standard way lokinfer reports speed: one grouped bar chart per **(model, device)**. The Y axis is the
E2E average time per forward pass in milliseconds (lower is better), the X axis lists the setups (input cases or
shapes), and each implementation (torch eager, torch.compile, MLX, the lokinfer variants, ...) is a bar of its own
color. Artificial targets are drawn as a dashed grey marker over the group.

| piece | role |
|---|---|
| `perf/results.json` | the only data store: the latest measurement per entry (checked in, human-readable) |
| `bench/perfdb.py` | read/write API for the store (`record`, `remove`, `load`, `machine_state`) |
| `perf/index.html` | static page (Chart.js 4.4.1 + chartjs-plugin-datalabels 2.2.0 from CDNs, no build step) |
| `scripts/perf_dashboard.sh` | serves `perf/` on `http://127.0.0.1:8765/` and opens the browser |

What every non-lokinfer bar is and how it is measured: [baselines.md](baselines.md).

## Opening it

```sh
scripts/perf_dashboard.sh            # starts the server if it is not already up, then opens the page
scripts/perf_dashboard.sh --no-open  # server only
PERF_PORT=9000 scripts/perf_dashboard.sh
```

The server is `.venv/bin/python -m http.server --bind 127.0.0.1 --directory perf` (localhost only), started in the
background with `nohup`; its log is `build/perf_dashboard.log`. The script is idempotent: if
`http://127.0.0.1:8765/results.json` already answers, it reuses that server. Stop it with
`pkill -f "http.server 8765"`.

The page polls `results.json` every 3 s and redraws when it changes, so a bench run shows up without a reload.
Controls: the model menu (see [Navigation](#navigation)) and a log-scale toggle (useful for MNIST, where values
span 4 µs to 4 ms). The view choice is remembered per browser; the data never is. Under each chart, a summary line per case gives the
best lokinfer bar against the best baseline (`N× faster` / `N× slower`) and against the target when one exists.
Tooltips show the full value, the measurement time, the power-state caveat and the notes. Dark mode follows the
system.

## Navigation

One section per model, in order, each with its title and its four device cards (Mac CPU, Mac GPU, iPhone CPU,
iPhone GPU — always all four; a version without data shows an empty red "missing" card). **Click a model's title to
collapse / expand its section**; the choice is remembered per browser. The dot next to the title is the worst status
of its four cards. Cards sit two per row; a model with more than 8 cases (Kokoro: 16) gets one full-width card per row (user, 2026-11-01). `#/model/<name>` in the URL scrolls to that model. "Log scale" switches every chart's y axis.

## Recording from a bench

Each bench records what it just measured; recording an entry again overwrites only that entry, so nothing else
has to be re-run.

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the bench/ directory
import perfdb

state = perfdb.machine_state()  # {"low_power_mode": False, "power": "AC", "thermal": "nominal"}
perfdb.record("Kokoro-82M", "Mac CPU", "rand510", "lokinfer fused", median_ms, state=state,
              notes="optional free text shown in the tooltip")
perfdb.remove("Kokoro-82M", device="Mac CPU", impl="lokinfer basic")  # drop retired bars (None = any)
```

Wired in today: `bench/models/bench_kokoro.py` (torch eager, torch MPS, lokinfer basic, lokinfer fused),
`bench/models/bench_kokoro_remote.py` (lokinfer fused or basic on the iPhone CPU, with the device's own state)
`bench/models/bench_mnist.py` (all its columns) and `bench/models/bench_other_baselines.py` (reference
baselines for ResNet-18, MobileNetV2, DCGAN G/D, GPT-2 124M: torch CPU/MPS/compile + MLX via
`bench/models/runners/other_mlx.py`; `--model X` reruns one model). Their own `results_*.json` outputs are unchanged.
Numbers that come from elsewhere (baselines from `docs/impl-notes/`, targets) are recorded by hand with an
explicit `measured_at` and a `notes` saying where they come from.

## JSON schema

```json
{
 "meta": {
  "models": {
   "Kokoro-82M": {"cases": ["real28", "real117", "rand510"], "description": "..."},
   "MNIST MLP": {"cases": ["B=1", "B=64", "B=1024"], "description": "..."}
  },
  "devices": ["Mac CPU", "Mac GPU", "iPhone CPU", "iPhone GPU"]
 },
 "entries": [
  {"section": "models", "model": "Kokoro-82M", "device": "Mac CPU", "case": "rand510", "impl": "lokinfer fused",
   "value": 1064.0, "unit": "ms", "measured_at": "2026-11-01T14:30:00",
   "state": {"low_power_mode": false, "power": "AC", "thermal": "nominal"},
   "notes": "M3 fused generator + profile fixes"}
 ]
}
```

- **Key**: `(section, model, device, case, impl)`. One entry per key, latest value only: there is no history
  (re-measuring replaces the old number and its timestamp). Git history of the file is the history.
- `value` + `unit`: milliseconds per call by default (E2E, one forward pass, median over runs).
- `measured_at`: local ISO time of the measurement; `state` and `notes` are optional.
- `meta.<section>.<model>.cases` fixes the X-axis order (cases not listed sort after, naturally);
  `description` is the subtitle of that model's cards. `meta.devices` fixes the device order.
  `perfdb.py` writes `DEFAULT_META` into a new file; edit `meta` in the JSON (or `DEFAULT_META`) when adding a
  model or a case.
- Writes are atomic (temp file + rename), so the page never reads a half-written file.

## Naming conventions

- **section**: `models` (end-to-end model forwards). A later kernel section would use e.g. `kernels`, with
  the kernel family as `model`.
- **device**: exactly one of `Mac CPU`, `Mac GPU`, `iPhone CPU`, `iPhone GPU` (M2 Max Mac, iPhone 18 Pro Max).
  CPU and GPU of the same machine are separate charts: compare CPU to CPU and GPU to GPU.
- **case**: the model's reference input name (`real28`, `real117`, `rand510` for Kokoro) or a shape (`B=64`).
- **impl**: frameworks by their usual name (`torch eager`, `torch.compile`, `torch MPS`, `torch.compile MPS`,
  `MLX`, `CoreML`, `ONNX Runtime`). **Any name starting with `lokinfer` is ours** (`lokinfer basic`,
  `lokinfer fused`, `lokinfer GPU`, `lokinfer accelerate`, ...): ours are drawn in greens, after the baselines,
  and the summary compares the best `lokinfer*` bar against the best non-lokinfer bar.
- **`target`** is reserved for an artificial goal (e.g. the iPhone targets derived from the Mac bars in
  `docs/impl-notes/m2-conv-proof.md`): drawn as a dashed marker, never as a bar, never counted as a baseline.
  A chart with only a target still renders (iPhone GPU today).

## Power state ⚠

The project requires every benchmark result to record Low Power Mode and the thermal state, and baselines only
change with Low Power Mode off. Every entry measured by a bench therefore carries `state`
(`perfdb.machine_state()` on the Mac, `RemoteDevice.state()` on the iPhone). The page appends ⚠ to a bar's label
and flags it in the tooltip and summary when `state.low_power_mode` is true, `state.power` is `battery`, or
`state.thermal` is anything other than `nominal` / `unknown`. A ⚠ number is not a valid comparison: re-run it
with Low Power Mode off on AC power. (Today the MNIST numbers carry ⚠: they were measured in Low Power Mode.)


## Timing protocol (every implementation, every device)

Median of the measured runs after a warm-up of **≥ 3 calls and ≥ ~1 s**, measuring **≥ the bench's run count and
≥ ~2 s** — the reference apps' protocol, applied to lokinfer too (`bench_kokoro.timeit`, `bench_remote_models.timed_run`).
The time-based part matters for small models: GPU clocks ramp up under load (DVFS) — on the iPhone a 1-kernel GPU call
takes ~290 µs cold and < 50 µs warm — so a few-milliseconds measurement would time a cold GPU (fixed 2026-11-01;
earlier small-model rows used 3 + 11 calls). iPhone sweeps cool down once at the start (wait for thermal
"nominal", capped at 5 min) and briefly before each model (capped at 60 s) — MLPerf Mobile also breaks between tests,
never between runs (user decision 2026-11-01: "wait a bit, but not much"). Each entry records the thermal state at the
start and end of its case (⚠ if throttled), and `perfdb.record` never replaces an entry measured at "nominal" with a
throttled one: a sweep that started hot measured 35 of 37 rows throttled (25-40 % slower) and overwrote good rows.

**Sustained vs burst (Mac GPU, 2026-11-02).** A dashboard sweep keeps the GPU busy for minutes; mid-size Kokoro
cases then read ~20-25 % slower than in a short loop (s6f 87 vs 69 ms) while the thermal state stays "nominal".
Every implementation is measured the same way (a sweep), so comparisons stay fair; one-off app calls are closer to
the short-loop numbers.

## Apples-to-apples rule (baselines)

Every baseline must run the same model on the same inputs as lokinfer: Kokoro-82M v1.0 weights in f32, the exact
reference cases (`models/kokoro/reference.py`: ids incl. the `[0, …, 0]` padding, the voice row, speed), and the
same amount of work — the output length must equal the reference audio's (600·F samples, i.e. the same durations).
`bench/models/bench_kokoro_baselines.py` enforces it for torch.compile (CPU / MPS) and MLX (via
`bench/models/runners/kokoro_mlx.py` in a separate venv with mlx + mlx-audio; set `LOKINFER_MLX_PYTHON` and
`LOKINFER_MLX_MODEL`). A framework that computes its own durations may land a frame off: within 0.2 % the entry is
recorded with the exact F difference in its notes, beyond that it is not recorded. torch eager / MPS come from
`bench/models/bench_kokoro.py` (english-practice's exact `infer()` math on the same cases).

## iPhone reference baselines

**[built]** `ExecuTorch XNNPACK` (iPhone CPU, Mac CPU) and `MLX Swift` (iPhone GPU, Mac GPU) are reference bars
measured outside the repo: separate measurement apps in `<baselines>/`, signed with LokiHost's bundle
id, so they install over it. They follow the apples-to-apples rule above. Both run the full forward with the
reference durations **forced**, so the output is exactly 600·F samples, which is checked on every run, and both
use f32 weights, which are checked. lokinfer depends on neither framework. Versions, patches, the exact commands to
reproduce each number, and the fairness and fidelity notes (threads, memory entitlement, correlation with the
reference) are in [impl-notes/iphone-baselines.md](impl-notes/iphone-baselines.md). They are recorded with
`<baselines>/scripts/record_baselines.py`.
The same two bars exist for every other model (ResNet-18, MobileNetV2, DCGAN G/D, GPT-2 124M, MNIST MLP) on
"iPhone CPU" / "iPhone GPU". They use the Mac baselines' weights, inputs and cases, are verified on the phone
(rel err ≤ 1e-3 vs torch eager CPU), and run MLX with `MLX_ENABLE_TF32=0`. By default, MLX runs f32 matmuls as
TF32 on the A19 GPU. They are recorded with `<baselines>/other/record_other.py`; see the same impl note.

## Configuration counts (user, 2026-11-02)

A **configuration** is one model × device × case that has a lokinfer number and at least one baseline; lokinfer is
**#1** there when its best variant is at least as fast as the best baseline (targets excluded). Each model title shows
"lokinfer #1 in X / Y configurations" on the right; the bar under the page title shows the total over all models and
the same count per device (Mac CPU, Mac GPU, iPhone CPU, iPhone GPU). Colors: green = all, amber = some, red = none.

## Card headline and coverage

Each card (model × device) shows, under its title, **"Overall: lokinfer X% faster/slower than best (impl)"**: for
every baseline implementation, the geometric mean over the cases both have of (baseline time / lokinfer time), with
lokinfer's best variant per case; "best" is the baseline with the lowest ratio (the closest competitor). Per-case lines
below the chart give the same comparison case by case.

Coverage rule (user, 2026-11-01): every model has all four device cards — Mac CPU, Mac GPU, iPhone CPU, iPhone GPU —
each with lokinfer's numbers and measured baselines. lokinfer's iPhone numbers come from
`bench/models/bench_remote_models.py` (remote executor in LokiHost); the iPhone baselines from the apps described in
`impl-notes/iphone-baselines.md`.
