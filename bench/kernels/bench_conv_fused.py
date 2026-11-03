"""M3: one Kokoro AdaINResBlock1 iteration at rand510 stage-1/stage-0 size, fused vs the basic kernel chain.

basic: adain1d -> snake -> conv1d -> adain1d -> snake -> conv1d -> add   (7 kernels, every one a full pass)
fused: channel_stats -> adain_affine -> conv1d_snake -> adain_affine -> conv1d_snake_res   (2 heavy kernels)

Usage: .venv/bin/python bench/kernels/bench_conv_fused.py [--sweep]
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from bench_kokoro import machine_state  # noqa: E402

import lokinfer as lk  # noqa: E402
from lokinfer import ops  # noqa: E402

SHAPES = [(128, 132121, 3, 1), (128, 132121, 7, 3), (128, 132121, 11, 5), (256, 22020, 11, 5)]


def T(a):
    return lk.from_numpy(np.ascontiguousarray(a, dtype=np.float32), copy=True)


def bench(model, x, runs=7):
    model(x)
    best = 1e30
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


def main():
    print(machine_state())
    for c, l, k, d in SHAPES:
        r = np.random.default_rng(0)
        w1, w2 = (T(r.standard_normal((c, c, k)) / math.sqrt(c * k)) for _ in range(2))
        b1, b2 = T(r.standard_normal(c)), T(r.standard_normal(c))
        a1, a2 = (T(r.uniform(0.2, 3, c)) for _ in range(2))
        g1, g2 = T(r.standard_normal(2 * c) * 0.3), T(r.standard_normal(2 * c) * 0.3)
        p1, p2 = d * (k - 1) // 2, (k - 1) // 2

        def basic(x):
            h = ops.snake_f32_cpu(ops.adain1d_f32_cpu(x, g1), a1)
            h = ops.conv1d_f32_cpu(h, w1, b1, padding=p1, dilation=d)
            h = ops.snake_f32_cpu(ops.adain1d_f32_cpu(h, g2), a2)
            h = ops.conv1d_f32_cpu(h, w2, b2, padding=p2)
            return ops.add_f32_cpu(h, x)

        def fused(x):
            af = ops.adain_affine_f32_cpu(ops.channel_stats_f32_cpu(x), g1, length=l)
            h, st = ops.conv1d_snake_f32_cpu(x, w1, b1, af, a1, padding=p1, dilation=d)
            af = ops.adain_affine_f32_cpu(st, g2, length=l)
            h, _ = ops.conv1d_snake_f32_cpu(h, w2, b2, af, a2, residual=x, padding=p2)
            return h

        x = T(r.standard_normal((c, l)))
        ty = [lk.TensorType(lk.f32, [c, l])]
        mb, mf = lk.compile(basic, input_types=ty), lk.compile(fused, input_types=ty)
        err = np.abs(mb(x).numpy() - mf(x).numpy()).max()
        tb, tf = bench(mb, x), bench(mf, x)
        gfl = 2 * 2 * c * c * (k + 3 * 0) * l / 1e9  # conv1 k taps + conv2 k taps
        print(f"C={c} L={l} k{k} d{d}: basic {tb:7.2f} ms  fused {tf:7.2f} ms  ({tb / tf:.2f}x)  "
              f"fused conv TF {gfl / tf:.2f}  max|diff| {err:.2e}", flush=True)



def sweep():
    """Tile sweep of one fused conv (conv1d_snake) per shape: picks ops.COL_BUDGET."""
    print(machine_state())
    shapes = SHAPES + [(256, 22020, 3, 1), (256, 22020, 7, 3), (512, 1101, 3, 1)]
    for c, l, k, d in shapes:
        r = np.random.default_rng(0)
        w, b = T(r.standard_normal((c, c, k)) / math.sqrt(c * k)), T(r.standard_normal(c))
        a = T(r.uniform(0.2, 3, c))
        af = T(np.stack([np.zeros(c), np.ones(c), np.zeros(c)]))
        x = T(r.standard_normal((c, l)))
        row = []
        for tile in (128, 256, 384, 512, 768, 1024, 1536, 2048):
            m = lk.compile(lambda x, tile=tile: ops.conv1d_snake_f32_cpu(x, w, b, af, a, padding=d * (k - 1) // 2,
                                                                         dilation=d, tile=tile)[0],
                           input_types=[lk.TensorType(lk.f32, [c, l])])
            ms = bench(m, x)
            row.append(f"{tile}:{2 * c * c * k * l / ms / 1e9:.2f}")
        print(f"C={c} L={l} k{k} d{d} auto={ops.conv_tile(c, k)}  TF " + "  ".join(row), flush=True)


if __name__ == "__main__" and "--sweep" in sys.argv:
    sweep()
    sys.exit(0)
if __name__ == "__main__":
    main()
