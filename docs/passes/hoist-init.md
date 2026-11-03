# Pass: hoist-init

**[built]** Moves everything that doesn't depend on per-call inputs from `exec` into `init`, which runs once at
model load.
It is as aggressive as legality allows.

Two separate notions (`src/compiler/analysis.cpp`):
- **memory-hoistable**: constants, static non-output allocs (including the arena and persistent buffers),
  constant-offset views of those with a static result type (a view typed with a per-call symbol, e.g.
  `pos[0:T]`, stays in exec). Their memory can be set up once.
- **constant contents**: a constant, or a buffer whose every writer is a hoisted kernel.

A **kernel is hoisted** iff it isn't flagged `nohoist` (RNG kernels), all its operands have constant contents, and
all its outs are memory-hoistable buffers that no per-call kernel also writes (otherwise the init results would be
clobbered after the first call); it also needs outs, static result types and constant `dim` attrs (a `dim` attr
using a per-call symbol can't be evaluated at init). This is computed as an optimistic fixpoint that propagates demotions within a sweep.

Typical effect (MNIST, Kokoro weights). PyTorch `[out, in]` weights are transposed once at load, and per-call
execution only runs the GEMMs:
```
init {
  %w = const.internal @"w" : buffer<f32[128, 784]>
  %wt = buffer.alloc : buffer<f32[784, 128]>
  kernel "transpose_f32_cpu"(%w) outs(%wt)
  return %wt
}
exec(%x: ...) state(%wt_1: buffer<f32[784, 128]>) { kernel "gemm_bias_relu_f32_cpu"(%x, %wt_1, ...) ... }
```
Hoisted values used by exec become **state**. A `buffer.free` of a hoisted buffer moves to init if exec no longer
uses it (init-only temporaries). Otherwise it is dropped, and the buffer lives as long as the model.

Tests: `tests/lit/hoist_init/` (including a no-clobber case).
