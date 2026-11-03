# Pass: bufferize

**[built]** Turns value-semantics tensors into buffers in destination-passing style.

**Before**
```
%out = kernel "add_f32_cpu"(%x, %y) : tensor<f32[N]>
return %out
```
**After**
```
%out = buffer.alloc {output = true} : buffer<f32[N]>
kernel "add_f32_cpu"(%x, %y) outs(%out)
return %out
```

Rules:
- Every tensor-typed value is retyped to the matching buffer type. Value ids don't change, so most ops need no rewriting.
- Each tensor result of a kernel gets a `buffer.alloc` right before the kernel and moves into `outs(...)`.
- **In-place.** If a kernel declares `inplace = [out_k, in_j, ...]` and operand `j`'s buffer dies at this kernel,
  output `k` reuses it. It must be an alloc root made in this region (not a view, not persistent), with the same
  type, dying at this kernel, not already reused, and no other operand of the kernel may alias it (the same value or
  a view of it). The reused buffer takes over the result's lifetime, so in-place chains keep reusing it.
- `view` becomes a buffer alias. Liveness is tracked per **alias root**, so a view keeps its source alive.
- `buffer.free` is inserted right after the last use of every alias set that isn't returned. Returned allocs get
  `{output = true}`. They are allocated fresh per call, never in the reusable arena.
- **GPU lifetimes.** A buffer a GPU kernel touched may still be in use by that dispatch until the next
  `sync gpu_to_cpu`. If a CPU op (kernel, `dim.read`, `return`) also touches it, its free is moved to that sync (or the
  end of the region). A buffer no CPU op touches gets `{gpu_only = true}` and is freed at its real last use;
  plan-memory keeps `gpu_only` buffers in their own arena (`docs/verifier.md`).
- `buffer.alloc {persistent}` (a KV cache written in place by `inout` kernels) is never freed and never reused
  in place.

Tests: `tests/lit/bufferize/*.lkir`. Code: `src/compiler/bufferize.cpp`.
