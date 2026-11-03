# The Python API

lokinfer's Python layer is a thin, declarative way to build graphs of **device-, dtype- and fusion-specific
kernels**. There is no automatic kernel selection (yet): you pick `ops.gemm_bias_relu_f32_cpu_accelerate` or
`ops.gemm_bias_relu_f32_gpu` explicitly.

Sections are marked **[built]** or **[planned]**.

Install for development: `uv pip install --no-build-isolation -e .`. C++ changes are rebuilt automatically on
`import lokinfer`.

## Tensors [built]

```python
import lokinfer as lk
a = lk.randn(64, 128, seed=0)                  # cpu, f32
g = lk.from_numpy(np_array, device="gpu")      # unified-memory MTLBuffer
t = lk.from_torch(torch_tensor)                # zero-copy for contiguous CPU and MPS tensors
a.numpy(), a.torch(), np.from_dlpack(a)        # zero-copy views (wait for pending GPU work first)
g.torch(device="mps")                          # GPU tensor -> MPS tensor sharing the MTLBuffer (DLPack kDLMetal)
a.shape, a.dtype, a.device, a.to("gpu"), print(a)
```
Contiguous CPU arrays are imported without copying. Strided arrays are copied, never silently aliased.

## Building and compiling a model [built]

```python
from lokinfer import ops

w = lk.randn(784, 128, seed=1)                 # concrete tensors captured in fn become constants
def fn(x):
    return ops.gemm_bias_relu_f32_cpu(x, w, b)

model = lk.compile(fn, inputs=[x])                                   # static shapes from examples
model = lk.compile(fn, inputs=[x], dynamic=True)                     # every input dim -> its own symbol
N = lk.Sym("N")
model = lk.compile(fn, input_types=[lk.TensorType(lk.f32, [N, 784])])  # parametric / shared dims
out = model(x)                                                       # shape-checked, synchronous
```
- Each `ops.*` function is one kernel. During tracing it checks its operands, appends one IR op and returns the
  result value(s). Called outside tracing (eager), it compiles a one-op graph (cached) and runs it.
- `ops.reshape` / `ops.flatten` are free views. `ops.check` errors are raised at build time, in Python.
- `model.source_ir` is the traced IR; `model.ir_text()` is the IR after the compiler pipeline.
- `lk.compile(..., pipeline=[...])` overrides the pass list; `lk.compile_ir(text, ...)` compiles textual IR directly.
- `model(...)` releases the GIL while the model runs.

## Defining a kernel op [built]

Ops are defined in Python only (`python/lokinfer/ops/*.py`):
```python
@kernel_op("gemm_bias_f32_cpu")
def gemm_bias_f32_cpu(a, b, bias):
    check(a.shape[1] == b.shape[0], "inner dims differ")
    return TensorType(f32, [a.shape[0], b.shape[1]])     # the output-type rule
```
Options: `nohoist=True` (never moved to init, e.g. RNG kernels), `inplace=[out, in]`, and `inout=[j, ...]` (the kernel
writes operand j in place and may return `()`; e.g. `kv_write_f32_cpu`).

## Persistent state (KV caches) [built]

`lk.persistent_buffer(dtype, shape, device)` inside a traced function creates model-lifetime state: its contents
persist across calls (uninitialized at first). `inout` kernels write it in place; other kernels read it like a tensor.
GPT-2's decode step keeps one K and one V cache per layer this way (`models/gpt2.py::build_decode`). The state belongs
to the compiled model: two compiled models don't share it.

## Mixed CPU and GPU graphs [built]

Devices are explicit in the graph (user decision 2026-11-01). Captured constants live on their tensor's device
(`lk.from_numpy(w, device="gpu")`). `ops.to_cpu(x)` / `ops.to_gpu(x)` mark a boundary; memory is unified, so
nothing is copied. A kernel using a value from the other device without a boundary is a compile-time error. A GPU
model is normally GPU end to end, with weight preparation (e.g. `ops.transpose_f32_gpu(w)`) hoisted into init.

Inputs and outputs are never copied: a GPU model wraps CPU tensors in place (cached per input), and outputs are
returned as views of the call's own storage (an output that *is* an input, a constant or model state is copied,
so callers never alias model memory).

## Saving, loading, remote devices [built]

```python
model.save("m.lkm")             # self-contained: captured weights are internalized (bake_init=True: see below)
m2 = lk.load("m.lkm")           # mmapped, zero-copy weights

from lokinfer.remote import RemoteDevice   # run on the iPhone test host (scripts/ios_deploy.sh)
dev = RemoteDevice.connect(f"{udid}.coredevice.local")
mid = dev.load(model)                          # cached on the device by sha256
outs, times_ns = dev.run(mid, [x], warmup=3, iters=20)
```

## Calls: outputs, async, export [built 2026-11-02]

```python
y = lk.empty(n, device="gpu")                  # or a contiguous numpy array, or a torch CPU / MPS tensor
model(x, out=y)                                # returns y; several outputs: out=(y1, None, y3), None = allocate
p = model.submit(x, out=y)                     # returns at once; p.wait() -> the result (or raises), p.done()
model.internalize()                            # captured tensors copied into the model (no aliasing any more)
stats = model.bake_init()                      # init results (e.g. transposed weights) become constants
model.save("m.lkm", bake_init=True)            # the file stores the transformed weights, loading does no init work
```
- **`out=`**: the model computes straight into the caller's memory, with no copy, when that output is computed by
  the model (its own output buffer, or a zero-offset view of it such as a reshape). A GPU model wraps CPU memory in
  place (cached per output, like inputs). Outputs the model doesn't compute (an input or a constant returned as
  is) are copied into `y`. dtype and shape must match exactly.
- **`submit()`**: the call runs on the model's own worker thread (one per model, so calls stay in order); the native
  call releases the GIL. Don't write the arguments (or `out`) until the call is done. A plain call after a submit
  waits for the earlier ones. The GPU part is not more asynchronous than in a plain call: the worker thread waits for
  the GPU like a plain call does.
- **`internalize()`**: copies every captured tensor (external constant) into the model and re-creates the
  executable; later writes to the tensors no longer reach the model. `save()` does the same on the fly.
- **`bake_init()`**: runs init once and stores its computed results as constants; init work nothing needs any more is
  removed (`docs/compiler.md` "Export").

**torch MPS, zero-copy both ways**: `lk.from_torch(mps_tensor)` wraps torch's MTLBuffer; `t.torch(device="mps")`
hands lokinfer's MTLBuffer to torch (DLPack device kDLMetal, which torch 2.14 imports). A GPU model can take MPS
inputs and write MPS outputs (`model(x_mps, out=y_mps)`) without a copy. The two frameworks use different command
queues: call `torch.mps.synchronize()` before lokinfer reads what torch wrote; lokinfer finishes its own GPU work
before handing a tensor to torch. Tests: `tests/python/test_call_api.py`.

## Planned

`ir_text(stage=...)`.
