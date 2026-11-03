# Design: IR, Types, Dims, Model Files

Status: **v1.1** (2026-11-01). Sections are marked **[built]** (implemented and tested) or **[planned]**.
Reviewed against the code on 2026-11-01; `plan-review-2026-11-01.md` lists the gaps. Decisions taken from `design-decisions.md` are marked **[user]**.
My own choices are marked **[derived]** and can be overturned.

## 1. Principles [built]

- The IR is **tiny**: a handful of op kinds. All the semantics live in *kernels*, which are opaque to the compiler.
- **No shape inference in the compiler** **[user]**. The graph builder (Python) writes every result type.
  The compiler only does *structural* verification: SSA, type equality where values flow, symbol definedness.
- **Contiguous row-major** buffers **[user, revisitable for perf]**. Zero-cost **views** cover
  reshape / squeeze / flatten / outer-dim slices **[user]**.
- **No exceptions** **[user]**. The parser/loader return `Status`/`Expected<T>`. The runtime hot path uses asserts.
- Every pass is **linear or n·log n** in op count. The full pipeline on a Kokoro-sized graph
  (~1–2k ops before fusion, ~100–300 after) should take well under 1 ms.

## 2. Dtypes [built]

`f32 f16 bf16 i8 u8 i16 i32 i64 bool` **[user: whatever the models need]**. Quantized formats
(e.g. int8 weights + f16 scales) are not new dtypes. They are separate buffers handed to dedicated kernels,
so the IR stays simple. A `q4`-style packed dtype is added only if a kernel really needs it.

## 3. Dims (symbolic shape expressions) [built]

A `Dim` is a hash-consed expression node owned by a `DimContext`:

```
Dim := Const(int64) | Sym(id) | Add(a,b) | Sub(a,b) | Mul(a,b)
     | FloorDiv(a,b) | CeilDiv(a,b) | Mod(a,b) | Min(a,b) | Max(a,b)
```
**[user]** Start with these and add more ops when a model needs them.

- **Canonicalization on construction** **[derived]**: constant folding, flattening of commutative
  `Add`/`Mul` chains, sorting operands by a stable key, `x*1`, `x+0`, `x floordiv 1`, and
  `(a*k) floordiv k → a`, and affine division `(a·X + b) // c` when `c | a` (so `(2F − 1)//2 + 1` is `F`). Equal expressions get the same pointer, so type equality
  (`[N, N]` vs `[N, M]`) is a pointer compare. No general algebraic solver. If two semantically equal
  forms don't canonicalize to the same node, the builder is expected to write the same form.
- **Symbols** are named (`N`, `T`, `F`, ...) and are defined in exactly one place:
  1. **graph input shapes**: at call time, a dim linear in one unbound symbol binds it (`N`, or `F` from
     `[600*F, 9]`). Later appearances are *checked* (this is how `[N, N]` square matrices work). A dim that can't
     be bound, or evaluating an unbound symbol, is an error.
  2. **`dim.read`**: a **data-dependent** symbol read from an i32/i64 buffer at runtime **[user]**.
     All types after that point may use it (Kokoro: `F = sum(pred_dur)`).
  3. **`mem.plan`**: memory-planner offsets for dynamic shapes (see `design-compiler-passes.md`).
- **Runtime evaluation** [built 2026-11-02: memoized, not compiled]: the runtime evaluates dims through
  `DimEvaluator`, a per-call memo indexed by node id, so each distinct (hash-consed) node is computed once per call;
  static dims are pre-evaluated at load. The planned alternative, a compiled **dim program** (three-address code
  over `int64 slots[]`, one sweep per symbol-binding point), was measured not to be worth it: on 1,000 tiny
  symbolic-shape kernels dim evaluation is ~4 % of the per-call time after memoization (`docs/compiler.md`).

## 4. Types [built]

```
f32 / i64 / ...                 scalar
tensor<f32[N, 2*N, 64]>         value-semantics tensor (pre-bufferization)
buffer<f32[N, 2*N, 64]>         memory reference (post-bufferization)
```
- No device in types **[derived]**. On Apple unified memory, every buffer of a model with any
  GPU kernel is a **shared-storage MTLBuffer** region (or plain aligned memory for CPU-only models), so CPU and GPU
  kernels can touch the same buffer. The kernel *name* says where it runs. Ordering comes from the explicit
  `sync` ops (written by the Python graph, checked by the verifier) and the runtime's GPU barriers. (If private storage measures faster for some GPU-only buffers, a memory-space attribute
  `buffer<f32[..], private>` can be added later.)
- A rank-0 tensor `tensor<f32[]>` is allowed (device-resident scalar), distinct from a host scalar `f32`.

## 5. Op set [built]

| Op | Form | Notes |
|---|---|---|
| `kernel "name"` | `%r.. = kernel "add_f32_cpu"(%a, %b) {attrs} : (types) -> (types)` | Tensor form. After bufferization: `kernel "name"(%a, %b) outs(%r)`. Results are outputs; a kernel may declare **in-place** pairs via the attr `inplace = [out_i, in_j, ...]`, so bufferization reuses the input buffer when it's dead afterwards, and **in/out** operands via `inout = [j, ...]` (the kernel writes operand j in place, e.g. a KV cache; it may then have no results). Structural attrs (`nohoist`, `inplace`, `inout`, `output`, `arena`, `device`, `persistent`, `gpu_only`) are kept out of the kernel's int ABI. |
| `const.inline` | `%c = const.inline dense<[1.0, 2.0]> : tensor<f32[2]>` / `const.inline 3 : i64` | Small literals (≤ 4 KiB) stored in the IR text/binary. |
| `const.internal` | `%w = const.internal @"bert.emb.weight" : tensor<f16[178, 128]>` | Data lives in the model file's constant section (mmapped, zero-copy). |
| `const.external` | `%w = const.external @"w0" : tensor<f32[784,128]>` | A pointer bound when the model is created (Python tensors captured during tracing). `internalize` copies external consts into the model for export **[user]**. |
| `view` | `%v = view %x[128*N] : buffer<f32[N*256]>` (offset in elements of the **result** dtype; source may be a `u8` arena) | Reinterpret contiguous memory with a new shape and element offset. Covers reshape, flatten, squeeze, outer-dim slicing and concat-in-place results. Becomes a buffer alias. |
| `buffer.alloc` / `buffer.free` | `%b = buffer.alloc : buffer<f32[N,64]>`, `buffer.free %b` | Introduced by bufferization, consumed by memory planning. The static arena is `buffer.alloc {arena = true} : buffer<u8[total]>` (there is no `mem.arena` op). `buffer.alloc {device = "gpu", persistent = true}` (from `lk.persistent_buffer`) is model-lifetime state: never freed or planned, hoisted into init. |
| `mem.plan` | see passes doc | Runtime memory planner for symbolic sizes: one arena + offset symbols (in elements of each buffer's dtype when `elem = [...]` is given, as plan-memory writes it; `gpu_only` marks the pool). |
| `dim.read` | `%F = dim.read %buf[0] : buffer<i64[1]> -> sym F` | Defines a data-dependent symbol. |
| `sync` | `%y = sync gpu_to_cpu %x` / `sync cpu_to_gpu` | A zero-cost alias resident on the target device, written by the Python graph (`ops.to_cpu` / `ops.to_gpu`), never inferred (user decision 2026-11-01). The runtime commits/waits at `gpu_to_cpu`. Makes mixed-device ordering visible in text and FileCheck-testable. |
| `return` | `return %y : tensor<f32[N]>` | |

That's it. Anything else is a kernel. **Copy, transpose, gather, concat (when not in-place), RNG and
argmax are all kernels.**

### Graph structure: init + exec [built]

```
model @kokoro {
  init -> (buffer<f16[512,768]>, buffer<u8[4194304]>) {
    %w   = const.internal @"dec.w" : buffer<f32[512,768]>
    %wh  = buffer.alloc : buffer<f16[512,768]>
    kernel "cast_f32_f16_cpu"(%w) outs(%wh)
    %ar  = buffer.alloc {arena = true} : buffer<u8[4194304]>
    return %wh, %ar
  }
  exec(%ids: buffer<i64[1, T]>, %ref: buffer<f32[1, 256]>, %speed: f32)
      state(%s0: ..., %s1: ...) -> buffer<f32[F*600]> {
    ...
  }
}
```
- Region result types follow `->` (MLIR style; printed always, optional in hand-written IR, checked against the
  `return` when present).
- `init` takes no inputs **[user]**. It runs once at model load. Its results become **state slots**
  that `exec` receives as `state(...)` arguments. This is how values move from init to exec: the runtime
  stores init's results in a slot table, and exec instructions reference slots directly (no copy).
- Before init hoisting, `init` is empty.
- No nested regions, no control flow. Python-side loops/ifs are unrolled at trace time. (Kokoro's LSTMs
  loop over time *inside* a kernel, so this never needs graph-level loops. GPT-2 generation is a host-side loop
  over calls; the KV caches are persistent buffers written in place by `inout` kernels [built].)

## 6. In-memory C++ representation [planned as written; built: a simpler form, see §10]

- `Model { DimContext dims; TypeTable types; ConstTable consts; Region init; Region exec; }`.
- `Region { std::vector<Op> ops; std::vector<ValueId> args, results; }`. The op order **is** the schedule.
  The compiler never reorders except when hoisting into init.
- Values are dense `uint32_t` ids into a value table `{TypeId type; OpRef def; }`. Types and Dims are
  interned. Attributes are a small tagged union (`i64, f64, string, i64[], f64[], dim`).
- Kernel names are interned strings resolved to the registry **once at load**, never per call.
- Everything is allocated in per-model arenas. Passes rebuild op vectors rather than mutating linked lists
  (simple and cache-friendly).

## 7. Textual format [built]

MLIR-like, line oriented, with a hand-written recursive-descent parser (~1–2k lines) and a printer that
round-trips exactly (`print(parse(print(m))) == print(m)`). Comments use `//`.
Lit tests use `lokinfer-opt --pass=bufferize %s | filecheck %s`.

## 8. Binary format (`.lkm`) [built: equivalent single-metadata-blob layout, see §10]

```
Header  { magic "LKM\0", version, flags, section table offset }
Section STRINGS   interned names
Section DIMS      dim expression nodes (topological)
Section TYPES
Section IR        ops of init/exec in a compact varint encoding of the in-memory structs
Section CONSTS    index: name -> (offset, size, dtype, shape)
Section DATA      raw constant bytes; every constant 16 KiB-aligned (Apple page size),
                  so the file can be mmapped and wrapped as MTLBuffers with no copy
```
- Loading = mmap + parse a few small sections (µs to low ms). Weights are never copied unless a
  hoisted init kernel transforms them.
- External constants can't be in a file: `export()` requires `internalize()` first (or does it automatically).
- The format is versioned. No compatibility promises until v1.

## 9. Python API surface (sketch; detailed in `design-python-api.md`) [built]

- `lokinfer.Tensor`: PyTorch-like UX (`shape`, `dtype`, `device`, printing, indexing to numpy),
  DLPack/buffer protocol interop **[user]**.
- `lokinfer.ops.<kernel_name>(...)`: one Python function per kernel, **defined in Python** **[user]**:
  its arg spec, attrs and output-type rule. During tracing it appends a `kernel` op and returns the result
  `Value`(s) (not the op) **[user]**.
- `lokinfer.compile(fn, inputs=[...], dynamic=False, input_types=None)` traces `fn` with symbolic
  Values, builds the IR, runs the pipeline and returns a callable `CompiledModel` **[user]**.
  Concrete tensors captured inside `fn` become `const.external`.
- `CompiledModel.__call__` checks input shapes against the signature (binding/checking symbols)
  **[user]**, runs synchronously by default, and exposes `submit()/wait()` **[user]** [built 2026-11-02], and
  `model(x, out=y)` (caller-provided outputs, no copy) [built].
- `save(path)`, `load(path)`, `source_ir` / `ir_text()`, `compile_ir(text)`, `internalize()`, `bake_init()` [built].

## 10. Known gaps (from the 2026-11-01 review; tracked in `docs/todo.md` M0)

Fixed in M0 [built]: affine division `(a·X + b) // c` when `c | a`; call-time binding of linear dims (`[600*F, 9]`);
structural attrs kept out of the kernel int ABI (`isStructuralAttr`); `view` bounds checks (verifier for static
views, runtime for dynamic ones) and `embedding` id checks (`ExecCtx::error`); unbound-symbol evaluation fails loudly.
Still open:
- In-memory representation is `std::string`/`std::vector` per op (not the arena/interned design of §6). The
  pipeline still meets < 1 ms (2026-11-02: dense tables in passes, allocation-free interning lookups); moving the
  224-byte `Op`s is now most of the passes' cost, the next lever if it ever matters.
- Binary format: single varint metadata blob + data section (not the multi-section layout of §8); equivalent in
  function.

## 11. Open questions (resolve during implementation) [planned]

- Do views need an element *stride* on the outermost dim (strided slice of rows) to avoid copies in
  Kokoro/GPT-2 (e.g. QKV split)? Possibly cheaper to give kernels a fused-split output instead.
  Decide from the kernel list.
- Constant dtype conversion at export (f32 → f16 weights) as an init-hoisted kernel vs. baked into the file.
  Probably both: a hoisted kernel during dev, and `bake_init()` at export to store transformed weights
  so load-time work is zero.
