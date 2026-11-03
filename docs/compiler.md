# The compiler

The lokinfer compiler is deliberately small. It does **no fusion and no shape inference**: kernels are already
fused by hand, and the Python graph builder writes every result type. It turns a graph of kernel calls on tensors
into a graph of kernel calls on buffers, with memory planned and load-time work moved out of the per-call path.
It must be fast: the pipeline runs every time a model is compiled. Sections are marked **[built]** or
**[planned]**.

Code: `include/lokinfer/ir/` (IR), `src/ir/` (dims, types, text/binary formats, verifier), `src/compiler/` (passes),
`tools/lokinfer_opt.cpp` (pass driver). Design notes: `docs/impl-notes/design-ir.md`, `design-compiler-passes.md`.

## The IR in one page [built]

A **model** has two regions:

```
model {
  init -> (buffer<f32[64, 64]>, buffer<u8[4096]>) { ... return %a, %b }       // runs once at load, no inputs
  exec(%x: buffer<f32[N, 64]>) state(%a2: ..., %b2: ...) -> buffer<f32[N, 64]> { ... return %out }   // per call
}
```

A region's **result types** follow `->` in its header, as in MLIR (one type bare, several in parentheses, omitted
when it returns nothing). The printer always writes them; the parser accepts IR without them and otherwise checks
them against the `return`.

`init`'s results reach `exec` as **state** values, matched by position. This is how weights transformed at load
time, and the static memory arena, become available to every call.

**Types.** Scalars (`f32`), value-semantics `tensor<f32[N, 2*N]>`, and memory `buffer<...>`. There's no device in
the type: on Apple unified memory every buffer is CPU-visible, and a kernel's name says where it runs.

**Dims** are symbolic integer expressions (`N`, `2*N + 1`, `(L - 1) // 2`, `ceildiv(N, 4)`, `min`, `max`, `%`).
They are hash-consed and canonicalized, so equal expressions are the same object and shape equality is a pointer
compare. Affine division simplifies (`(2*F - 1) // 2 + 1` is `F`), so conv output lengths prove equal to the
lengths they must join with. The parser rejects division by zero, integer overflow and excessive nesting. Symbols get their values in three ways:
- from input shapes at call time: a dim linear in one unbound symbol binds it (`[N, N]` binds `N` and checks the
  second dim; `[600*F, 9]` binds `F`); a dim that can't be bound is an error;
- from `dim.read` (a **data-dependent** dim read from a buffer at runtime, e.g. Kokoro's frame count `F`; if an
  input already bound `F`, the runtime checks that the two agree);
- from `mem.plan` (offsets planned at runtime).

**Ops:**

| op | purpose |
|---|---|
| `kernel "name"(...)` | call a kernel; tensor form has results, buffer form writes `outs(...)` |
| `const.inline` / `const.internal` / `const.external` | literal / stored in the model file / bound by pointer at load |
| `view %x[offset]` | zero-cost reshape or offset slice of contiguous memory (offset in result elements) |
| `buffer.alloc` / `buffer.free` | explicit memory (after bufferization). `buffer.alloc {persistent}` (written by the Python graph via `lk.persistent_buffer`) is model-lifetime state: never freed, never planned into the arena, hoisted into init so exec receives it as state |
| `mem.plan` | runtime-planned arena for dynamic shapes (offsets in elements with `elem`; `gpu_only` pool) |
| `dim.read` | define a data-dependent symbol |
| `%y = sync gpu_to_cpu %x` / `cpu_to_gpu` | explicit device boundary: a zero-cost alias resident on the target device (written by the Python graph, never inferred) |
| `return` | region results |

Everything else (copies, transposes, RNG, gathers) is a kernel.

Structural kernel attrs (never passed to the kernel): `nohoist`, `inplace = [out, in, ...]`, `inout = [j, ...]` (the
kernel writes operand j in place — a persistent buffer such as a KV cache; such a kernel may have no results;
the verifier makes j resident on the kernel's device, the runtime's GPU hazard tracking counts j as a write),
`output`, `arena`, `device`, `persistent`, `gpu_only` (bufferize: no CPU op touches the buffer). The runtime executes ops in program order, so an `inout` writer and a
later reader of the same buffer stay ordered.

## Formats [built]

The same model has three exactly equivalent forms (all round-trip tested):
- **in memory**: the C++ `Model`;
- **text**: MLIR-like (`printModel` / `parseModel`). Used by lit tests, and as the Python → C++ interchange;
- **binary `.lkm`**: a 64-byte header, varint metadata, then constant data with every blob 16 KiB-aligned. It is
  loaded with `mmap`, so weights are zero-copy on the CPU and wrapped as GPU buffers without copying. External
  constants must be internalized before export (`CompiledModel.save` does it). The loader treats files as
  untrusted: overflow-safe bounds, validated enums, ids and constant sizes; corrupt files return errors.

## Pipeline [built]

```
dce → bufferize → plan-memory → hoist-init
```

bufferize, plan-memory and hoist-init have their own doc in `docs/passes/`; `dce` (`src/compiler/dce.cpp`,
`tests/lit/dce/`) removes exec ops whose results are unused, keeping `dim.read` and `inout` / result-less kernels
(side effects). Verification is not a pass: the verifier (`docs/verifier.md`) checks
the IR when a model is loaded or built and again after every pass, and never modifies it.

**Speed** (2026-11-02, M2 Max, Release, the 1,000-kernel / ~3,800-op synthetic graph of
`tests/unit/test_compiler.cpp`): the whole pipeline with the verifier after every pass takes **0.92 ms** (was ~5 ms):
dce 0.01 ms, bufferize 0.23, plan-memory 0.26, hoist-init 0.13, plus 0.06-0.08 ms per verification. What it took:
dense per-value / per-type / per-dim tables instead of hash maps in every pass and in the verifier, hash-consing
lookups that allocate nothing on a hit (dims and types), passes that rewrite op lists in place or build them
once (an `Op` is 224 bytes; moving them is most of the remaining cost), and element-unit `mem.plan` offsets (no
`O // 4` dim per planned buffer). Measured with `lokinfer-opt --stats` (single run) or a loop over the pipeline
(median); the unit test only guards against blow-ups.

## Memory planning at run time [built]

`mem.plan` (plan-memory, `docs/passes/plan-memory.md`) runs the greedy planner per call. Three things keep that cheap:
- **Cache keyed by symbol values**: each `mem.plan` keeps its last 8 results, keyed by the values of the symbols
  its sizes depend on (plus those of the earlier plans it shares an arena with). A model called again with the same
  shapes evaluates no sizes and plans nothing (`Executable::memPlanStats()`: hits / misses).
- **One arena per pool, shared by all plans of a call**: the CPU-touched pool and the `gpu_only` pool (attr on
  `mem.plan`) each have one grow-only arena. A later plan (after a `dim.read`) places its buffers around the
  earlier plans' buffers that are still live (`planOffsets(..., fixed)`), instead of getting its own arena. An arena
  is only reallocated at its pool's first plan of a call (nothing in it is live then); a later plan that doesn't fit
  gets a temporary overflow storage for that call and raises the pool's size for the next one. Arena addresses are
  stable across calls (good for GPU replay).
- **Element offsets**: `mem.plan ... {elem = [4, 8]}` defines each offset symbol in elements of its buffer's dtype,
  so a view is `view %arena[M0_o3]`, with no division to evaluate. Without `elem` (hand-written IR) offsets are bytes.

**Dim evaluation [built 2026-11-02]: memoized per call, not a compiled program.** The runtime evaluates dims through
`DimEvaluator` (`lokinfer/ir/dim.h`): each distinct (hash-consed) dim node is computed at most once per call, so
`N`, `4*N`, ... shared by many kernels cost one load after the first use. Measured on 1,000 chained tiny CPU kernels
(M2 Max, Release; `N` vs a static shape): symbolic shapes cost ~20 ns per kernel over static ones (12 → 32 µs per
call), but after memoization dim evaluation is ~4 % of the per-call time; the rest is the per-call `view`
instructions into the `mem.plan` arena (static shapes have none: their views are hoisted into init) and their bounds
checks. A compiled three-address "dim program" would remove only those ~4 %, so it isn't built; on real models
(Kokoro ~50 ms, MobileNetV2 GPU ~1 ms with ~100 kernels) the whole symbolic overhead is < 0.3 %.

## Export: internalize and bake-init [built]

- `internalize(Model&, externals, copy)` (`src/compiler/internalize.cpp`) turns `const.external` (tensors captured
  while tracing, bound by pointer) into `const.internal`. `CompiledModel.save` uses it without copying (writing the
  file copies); `CompiledModel.internalize()` copies, so the model stops aliasing the captured tensors.
- `bake-init` (`rt::bakeInit`, `src/runtime/bake_init.cpp`; in the runtime library because it runs init) evaluates
  init once and stores every computed init result (a plain `buffer.alloc` written by init kernels, e.g. a transposed
  weight) as a `const.internal` resident on the device that wrote it; init work nothing needs any more is removed,
  including the original weights only that work read. Left alone, because baking them would change behavior or
  store garbage: static arenas (scratch), persistent buffers (state the calls write), constants, and any model whose
  init defines symbols. A baked model loads with no init work, and its file stores the transformed weights instead of
  the originals. Python: `model.bake_init()`, `model.save(path, bake_init=True)`.

## Tools [built]

```
build/lokinfer-opt --pass=bufferize,plan-memory file.lkir   # run passes, print IR
build/lokinfer-opt --pipeline file.lkir                     # full pipeline
build/lokinfer-opt --list-passes
build/lokinfer-opt --pipeline --stats file.lkir             # + per pass: time (pass + verifier), op counts; plan-memory:
                                                            #   arena bytes vs max-live lower bound vs no reuse (stderr)
build/lokinfer-opt --pipeline --no-verify file.lkir         # skip the verifier (debugging a verifier bug)
.venv/bin/lit tests/lit                                     # FileCheck tests of every pass + the verifier
```
From Python: `model.source_ir` (as traced) and `model.ir_text()` (after the pipeline).
