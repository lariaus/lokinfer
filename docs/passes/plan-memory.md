# Pass: plan-memory

**[built]** Replaces many allocations with one arena plus offsets.

**Candidates** are non-output allocs that are freed in the region. Buffers whose contents come only from constants
(weight transforms) are excluded: hoist-init moves them to init, and the arena must not reuse their memory.
Persistent buffers are never freed, so they are never candidates. Within each plan, `gpu_only` buffers (bufferize)
get an arena separate from CPU-touched ones, so a CPU kernel is never handed bytes the GPU may still read.

**Static sizes** (all dims constant). The greedy-by-size planner (TFLite's algorithm) sorts buffers by size,
largest first. Each buffer goes into the smallest gap, among buffers whose lifetimes overlap it, that fits; offsets
are 256-byte aligned. The result is a `buffer.alloc {arena = true} : buffer<u8[total]>` plus constant-offset views.
hoist-init then moves the arena into init, so per-call execution allocates nothing:
```
%arena = buffer.alloc {arena = true} : buffer<u8[8192]>
%a = view %arena[0] : buffer<f32[16, 64]>
%b = view %arena[1024] : buffer<f32[16, 64]>
%c = view %arena[0] : buffer<f32[16, 64]>      // reuses %a's slot
```

**Symbolic sizes.** Offsets can't be constants. The pass emits a `mem.plan` that the runtime executes per call:
it evaluates the sizes, runs the same greedy planner (microseconds), and binds the offset symbols. Offsets are in
elements of each buffer's dtype (`elem` lists the element sizes), so views use the symbol as is.
```
%arena = mem.plan sizes[512*T, 8] live[0, 7, 2, 8] align 256 -> M0_size, M0_o0, M0_o1 {elem = [4, 8]} : buffer<u8[M0_size]>
%e = view %arena[M0_o0] : buffer<f32[T, 128]>
```
Buffers whose size depends on a data-dependent symbol get their own `mem.plan`, placed right after that symbol's
`dim.read`. `gpu_only` buffers get separate plans (`{gpu_only = true}`), never sharing bytes with CPU-touched ones.

**At run time [built 2026-11-02]** (`docs/compiler.md` "Memory planning at run time"): each `mem.plan` caches its
last 8 results keyed by the values of the symbols it depends on (same shapes: no size evaluation, no planning), and
all plans of one pool share one grow-only arena per call: a later plan places its buffers around the earlier plans'
buffers still live (`planOffsets(..., fixed)`). The arena is only reallocated at the pool's first plan of a call; a
later plan that doesn't fit gets a temporary overflow storage for that call and grows the arena for the next.

**Statistics**: `lokinfer-opt --stats` prints, per static arena, its bytes vs the max-live lower bound vs the bytes
without reuse, and the buffer count of each `mem.plan` (`tests/lit/plan_memory/stats.lkir`).

The planner (`src/compiler/memplan.cpp`) is shared by compiler and runtime. A randomized property test checks that
no two buffers with overlapping lifetimes overlap in memory. Tests: `tests/lit/plan_memory/` (incl. `gpu_only_pool*.lkir`, `stats.lkir`), `tests/unit/test_compiler.cpp` (planner
property test; "mem.plan: repeated symbol values hit the plan cache", "mem.plan: the plans of a call share one arena").
