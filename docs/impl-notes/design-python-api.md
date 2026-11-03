# Design: Python API

Status: **v1.1** (2026-11-01). Thin, declarative, IR-shaped **[user]**. **Built:** dtypes, dims, types, Tensor
(numpy/DLPack zero-copy on CPU, MPS import), `kernel_op`, tracing `compile()` (static/dynamic/input_types), eager
kernel calls, `reshape`/`flatten` views, `save`/`load`, `compile_ir`, `compile(..., pipeline=[...])`, explicit device
boundaries (`ops.to_cpu`/`ops.to_gpu`), `lk.persistent_buffer` + `inout` kernels, `lokinfer.remote`, `lokinfer.prof`.
Built 2026-11-02: `submit()/wait()`, `model(x, out=y)`, `internalize()`, `bake_init()`, torch MPS zero-copy both
ways (`Tensor.torch(device="mps")`). Sections are marked **[built]** or **[planned]**. **Planned:** `ir_text(stage=...)`, `testing.py` helpers,
`Tensor.randn` classmethod (today `lk.randn`), eager calls with scalar operands.

## Modules [built: as below; planned: `testing.py`]

```
lokinfer/
  _lokinfer          nanobind extension (imported as `_C`): compiler pipeline, runtime, Tensor storage
  dtypes.py          f32, f16, bf16, i8, ..., bool
  dims.py            Sym("N"), dim arithmetic via Python operators (+ - * // ceildiv mod min max)
  types.py           TensorType(dtype, shape), ScalarType
  tensor.py          Tensor (PyTorch-like UX), from_numpy/from_torch/from_dlpack, numpy()/torch()
  graph.py           Graph/Value tracing builder; emits textual IR (`source_ir`) that _C parses
  ops/               one Python function per kernel, grouped by family (elementwise.py, gemm.py, kokoro.py, ...);
                     `_def.py` holds `kernel_op`
  compiler.py        compile(), compile_ir(), load(), CompiledModel
  remote.py          remote-device executor client (iPhone test host)
  prof/              prof-lokinfer / prof-torch tooling (docs/profiling.md)
  testing.py         [planned] assert_close helpers
```

## Ops are defined in Python **[user]** [built]

```python
# lokinfer/ops/elementwise.py
from lokinfer.ops._def import kernel_op

@kernel_op("add_f32_cpu")
def add_f32_cpu(x, y):
    check(x.dtype == f32 and x.shape == y.shape)
    return TensorType(f32, x.shape)            # output type rule; nothing else needed

@kernel_op("gemm_bias_gelu_f16_gpu")           # options: nohoist=, inplace=[out, in], inout=[j]
def gemm_bias_gelu_f16_gpu(a, b, bias, *, trans_b=False):   # keyword args become kernel attrs
    M, K = a.shape
    N = b.shape[0] if trans_b else b.shape[1]
    return TensorType(f16, [M, N])
```
- `@kernel_op` makes the function, when called during tracing, (1) run the rule to get result type(s),
  (2) append `kernel "<name>"` to the current graph, and (3) return the result Value(s) **[user]**.
- Outside tracing (eager): build a one-op graph, compile it (cached per shape) and run it. This makes
  every kernel callable directly on Tensors, which is what correctness tests use **[user: test through
  graphs; compile/run overhead ≈ 0]**.
- Shape checks in the rule (`check`) raise in Python at build time. The C++ side never re-checks.

## Tensor **[user: PyTorch-like UX]** [built; planned: `Tensor.randn` classmethod]

```python
a = lk.randn(64, 128, dtype=lk.f32, device="gpu", seed=0)
b = lk.tensor([[1, 2], [3, 4]], dtype=lk.f16)              # cpu by default
t = lk.from_torch(torch_tensor)    # zero-copy (CPU and MPS) via DLPack
n = a.numpy()                      # zero-copy view (shared MTLBuffer contents) after sync
a.to("cpu"), a.shape, a.dtype, print(a), a[0, :4]
```

## Compile & run **[user: API from the plan]** [built; planned: `ir_text(stage=...)`]

```python
def my_model(a, b, c):
    mat = ops.gemm_f32_gpu(a, b)
    return ops.vector_agg_f32_gpu(mat, c)

model = lk.compile(my_model, inputs=[a, b, c])                  # static shapes from a/b/c
model = lk.compile(my_model, inputs=[a, b, c], dynamic=True)    # every input dim → fresh symbol
N = lk.Sym("N")
model = lk.compile(my_model, input_types=[lk.TensorType(lk.f32, [N, N]), ...])  # parametric
out = model(a, b, c)            # shape-checked, synchronous
fut = model.submit(a, b, c); out = fut.wait()   # [built] worker thread per model, calls in order
model(a, b, c, out=y)           # [built] outputs written into caller tensors (no copy when the model computes them)
model.ir_text(stage="final")    # [planned] IR after any pass; today ir_text() = after the pipeline, source_ir = traced
model.save("m.lkm"); m2 = lk.load("m.lkm")
```
- Concrete Tensors captured inside `fn` (weights) become `const.external` **[user]**. `save()` copies them into the
  file [built]; `model.internalize()` copies them into the model [built]; `model.bake_init()` stores init's results
  as constants [built].
- `lk.compile(..., pipeline=[...])` overrides the pass list for experiments [built].
- `model(x, out=y)` [built]: the runtime binds the output's alloc to the caller's memory (`Executable::run(...,
  callerOutputs)`), wrapping it for the GPU like inputs; outputs that are inputs/constants are copied into `y`.
- `submit()` is a Python-level worker thread, not a runtime-level asynchronous call (decision 2026-11-02, see
  `design-decisions.md`).

## Future (not v1) **[user]** [planned]

A higher-level layer (`lk.nn`) that picks the right kernel per device/dtype/shape, and possibly a
PyTorch-mimicking API.
