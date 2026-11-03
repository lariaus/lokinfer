"""Explicit device boundaries (user decision 2026-11-01: devices are explicit in
the IR, written by the graph, never inferred by the compiler).

Memory is unified, so nothing is copied: `to_cpu(x)` makes CPU kernels wait for
the GPU work that produced `x`; `to_gpu(x)` marks a CPU-made value as used by GPU
kernels from here on. Using a value on the other device without one of these is
a compile-time error (the IR verifier, docs/verifier.md).
"""
from ..graph import current_graph
from ._def import OpError, _as_value


def _sync(x, kind: str):
    g = current_graph()
    if g is None:
        raise OpError(f"to_{kind.split('_')[-1]} is only available while tracing")
    return g.sync(_as_value(g, x), kind)


def to_cpu(x):
    return _sync(x, "gpu_to_cpu")


def to_gpu(x):
    return _sync(x, "cpu_to_gpu")
