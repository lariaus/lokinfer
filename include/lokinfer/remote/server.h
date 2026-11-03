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
//   tensor := u32 dtype, u32 ndim, i64 shape[ndim], u64 nbytes, bytes
//   SHUTDOWN    -> stops the server
//   PROFILE     u32 model, u32 level, u32 flags (bit 0: serialize GPU kernels) -> (Executable::setProfile)
//   TAKE_PROFILE u32 model                -> UTF-8 JSON (Executable::takeProfileJson, docs/profiling.md)
//   GPU_INSPECT u8 archive, then '\n'-separated metallib function names (empty = all)
//               -> u64 json_len, JSON (rt::gpuInspectJson), then the binary archive's bytes when requested
//   MEMORY      (empty)                    -> UTF-8 JSON {phys_footprint, phys_footprint_peak, resident, resident_peak,
//               gpu_allocated, models} (task_vm_info + MTLDevice.currentAllocatedSize, bytes)
//   RUN2        u32 model, u32 warmup, u32 iters, u32 flags (bit 0: return outputs), u32 n, n * input,
//               u32 c, c * comparison
//               -> u32 n_out, n_out * tensor (0 unless flag bit 0), u32 c, c * stats, u32 iters, iters * f64 wall_ns
//   input      := u8 kind; 0: tensor | 1: sha256hex(64) of a blob holding one encoded tensor (content-addressed test
//                 inputs / references, uploaded once with PUT_BLOB) | 2: generated := u32 dtype (f32/i32/i64),
//                 u32 ndim, i64 shape[ndim], u64 seed, u32 dist (0 normal(a, b), 1 uniform [a, b), 2 ints [a, b)),
//                 f64 a, f64 b (splitmix64; lokinfer.remote.generate reproduces it on the Mac)
//   comparison := u32 output, u8 kind (1: sha256hex(64) of a reference tensor blob | 3: u32 other output),
//                 f64 rtol, f64 atol
//   stats      := u32 output, u64 count, u64 nonfinite, u64 mismatches (|got - ref| > atol + rtol |ref|),
//                 i64 worst index, f64 max_abs_err, f64 max_abs_ref, f64 sum_sq_err, f64 sum_sq_ref
// HELLO's JSON gains "features": ["profile", "gpu_inspect", "memory", "run2"]; older servers answer the newer
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
