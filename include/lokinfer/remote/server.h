// Remote executor server (runs inside the iOS test host, or as the Mac
// `lokinfer-serve` CLI for local testing). See docs/impl-notes/research-infra.md §5.3.
//
// Frames (little-endian): 24-byte header
//   magic u32 "LOKI" | version u16 | type u16 | request_id u32 | flags u32 | payload_len u64
// then the payload. Responses echo request_id and type; flags = 0 ok,
// 1 error (payload = UTF-8 message).
//
// Messages:
//   HELLO       -> JSON {device, os, gpu, build}
//   HAS_BLOB    sha256hex                   -> u8 present
//   PUT_BLOB    sha256hex(64) + bytes       -> (verified, stored by hash)
//   LOAD_MODEL  sha256hex [i32 gpu_replay [u32 gpu_segments]] -> u32 model_id (mmapped .lkm); gpu_replay =
//               RuntimeOptions::gpuReplay (-1 env, 0, 1), gpu_segments = GpuContext::setMaxSegments (0 unchanged)
//   RUN         u32 model, u32 warmup, u32 iters, u32 n, n * tensor
//               -> u32 n, n * tensor, u32 iters, iters * f64 wall_ns
//               (error "input i is mutable: pass a device tensor" when exec arg i is a mutable input: RUN2 kind 3)
//   tensor := u32 dtype, u32 ndim, i64 shape[ndim], u64 nbytes, bytes
//   SHUTDOWN    -> stops the server
//   PROFILE     u32 model, u32 level, u32 flags (bit 0: serialize GPU kernels) -> (Executable::setProfile)
//   TAKE_PROFILE u32 model                -> UTF-8 JSON (Executable::takeProfileJson, docs/profiling.md)
//   GPU_INSPECT u8 archive, then '\n'-separated metallib function names (empty = all)
//               -> u64 json_len, JSON (rt::gpuInspectJson), then the binary archive's bytes when requested
//   MEMORY      (empty)                    -> UTF-8 JSON {phys_footprint, phys_footprint_peak, resident, resident_peak,
//               gpu_allocated, models, device_tensors, device_tensor_bytes} (task_vm_info +
//               MTLDevice.currentAllocatedSize, bytes; live device tensors); feature "memory_pressure" adds
//               available (iOS os_proc_available_memory, -1 on macOS), pressure ("normal" / "warn" / "critical",
//               dispatch memory-pressure source), pressure_events, pageins, page_size, weights, weights_resident,
//               weights_locked, system_free / _file_backed / _compressor / _wired (bytes)
//   RUN2        u32 model, u32 warmup, u32 iters, u32 flags (bit 0: return outputs), u32 n, n * input,
//               u32 c, c * comparison
//               -> u32 n_out, n_out * tensor (0 unless flag bit 0), u32 c, c * stats, u32 iters, iters * f64 wall_ns
//   input      := u8 kind; 0: tensor | 1: sha256hex(64) of a blob holding one encoded tensor (content-addressed test
//                 inputs / references, uploaded once with PUT_BLOB) | 2: generated := u32 dtype (f32/i32/i64),
//                 u32 ndim, i64 shape[ndim], u64 seed, u32 dist (0 normal(a, b), 1 uniform [a, b), 2 ints [a, b)),
//                 f64 a, f64 b (splitmix64; lokinfer.remote.generate reproduces it on the Mac) | 3: u32 device tensor
//                 handle (ALLOC_TENSOR), bound in place: no copy, its memory with its owner, so GPU models wrap it with
//                 no copy too. A mutable exec arg (`buffer<...> mutable`, written in place, e.g. a KV cache) must be
//                 kind 3: any other kind is an error ("input i is mutable: pass a device tensor"), never a run on a
//                 copy whose writes would be lost. Every warmup / timed iteration writes it.
//   comparison := u32 output, u8 kind (1: sha256hex(64) of a reference tensor blob | 3: u32 other output),
//                 f64 rtol, f64 atol
//   stats      := u32 output, u64 count, u64 nonfinite, u64 mismatches (|got - ref| > atol + rtol |ref|),
//                 i64 worst index, f64 max_abs_err, f64 max_abs_ref, f64 sum_sq_err, f64 sum_sq_ref
//   ALLOC_TENSOR u32 dtype, u32 ndim, i64 shape[ndim] -> u32 handle: a device tensor (server-owned, zeroed,
//               page-aligned; all live device tensors <= 4 GiB, <= 4096 of them). Device tensors belong to the server
//               session (they outlive a client connection) until FREE_TENSOR.
//   WRITE_TENSOR u32 handle, tensor (same dtype and shape) -> copied in
//   READ_TENSOR u32 handle                -> tensor
//   FREE_TENSOR u32 handle                -> (unknown handles: error)
//   PIN_MODEL   u32 model, u32 mode (0 unpin, 1 madvise WILLNEED, 2 mlock, 3 GPU residency; all or nothing)
//               -> JSON {bytes, resident, locked, error}: keeps a model's mapped weights in RAM (no re-reads from
//               flash under memory pressure)
// HELLO's JSON gains "features": ["profile", "gpu_inspect", "memory", "run2", "device_tensors"]; older servers answer the newer
// messages with an error ("unknown message type"), which clients treat as "not supported".
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "lokinfer/ir/status.h"

namespace lk::remote {

enum class MsgType : uint16_t {
  Hello = 1,
  HasBlob = 2,
  PutBlob = 3,
  LoadModel = 4,
  Run = 5,
  Shutdown = 6,
  Profile = 7,
  TakeProfile = 8,
  GpuInspect = 9,
  Memory = 10,
  Run2 = 11,
  AllocTensor = 12,
  WriteTensor = 13,
  ReadTensor = 14,
  FreeTensor = 15,
  LoadModelSplit = 16,  // feature "split_models": a .lkm as skeleton + per-constant blobs
  Blobs = 17,           // feature "blobs": report the blob store's size, or remove its blobs except a keep list
  FreeModel = 18,       // feature "free_model": u32 model id; the model (and its memory) goes now, not at disconnect
  PinModel = 19,        // feature "pin_model": u32 model id, u32 mode (0 unpin, 1 madvise WILLNEED, 2 mlock,
                        // 3 GPU residency) the model's mapped weights -> JSON {bytes, resident, locked, error}
};

constexpr uint32_t kMagic = 0x494B4F4C;  // "LOKI"
constexpr uint16_t kVersion = 1;
constexpr uint16_t kDefaultPort = 47800;

struct ServerOptions {
  uint16_t port = kDefaultPort;
  std::string blobDir;  // content-addressed model cache
  std::string deviceInfoJson = "{}";
  // If set, called on every Hello instead of deviceInfoJson (live state: thermal, Low Power Mode).
  std::function<std::string()> deviceInfo;
  bool verbose = true;
};

// Blocks: accepts clients one at a time until a SHUTDOWN request.
Status runServer(const ServerOptions& opts);

// Serves one already-connected stream (e.g. one end of a socketpair) with a fresh session until the peer closes,
// sends a malformed header or SHUTDOWN. Same frame handling as runServer; used by the protocol fuzz test.
Status serveConnection(int fd, const ServerOptions& opts);

}  // namespace lk::remote
