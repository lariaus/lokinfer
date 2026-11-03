#include "lokinfer/remote/server.h"

#include <CommonCrypto/CommonDigest.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <mach/mach.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

#include "lokinfer/ir/binary.h"
#include "lokinfer/ir/verify.h"
#include "lokinfer/runtime/runtime.h"
#include "../runtime/gpu_internal.h"

namespace lk::remote {

namespace {

struct Header {
  uint32_t magic;
  uint16_t version;
  uint16_t type;
  uint32_t requestId;
  uint32_t flags;
  uint64_t payloadLen;
};
static_assert(sizeof(Header) == 24);

constexpr uint64_t kMaxPayload = uint64_t(1) << 31;  // 2 GiB per frame

bool readAll(int fd, void* p, size_t n) {
  auto* c = static_cast<char*>(p);
  while (n) {
    ssize_t r = ::recv(fd, c, n, 0);
    if (r <= 0) return false;
    c += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

bool writeAll(int fd, const void* p, size_t n) {
  auto* c = static_cast<const char*>(p);
  while (n) {
    ssize_t w = ::send(fd, c, n, 0);
    if (w <= 0) return false;
    c += w;
    n -= static_cast<size_t>(w);
  }
  return true;
}

// Payload cursor helpers.
struct In {
  const uint8_t* p;
  const uint8_t* end;
  bool ok = true;
  template <typename T>
  T get() {
    T v{};
    if (end - p < static_cast<ptrdiff_t>(sizeof(T))) {
      ok = false;
      return v;
    }
    std::memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return v;
  }
  const uint8_t* take(size_t n) {
    if (static_cast<size_t>(end - p) < n) {
      ok = false;
      return nullptr;
    }
    const uint8_t* r = p;
    p += n;
    return r;
  }
};

struct Out {
  std::vector<uint8_t> b;
  template <typename T>
  void put(T v) {
    size_t at = b.size();
    b.resize(at + sizeof(T));
    std::memcpy(b.data() + at, &v, sizeof(T));
  }
  void bytes(const void* p, size_t n) {
    const auto* c = static_cast<const uint8_t*>(p);
    b.insert(b.end(), c, c + n);
  }
};

std::string sha256Hex(const uint8_t* p, size_t n) {
  unsigned char d[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(p, static_cast<CC_LONG>(n), d);
  static const char* hex = "0123456789abcdef";
  std::string s;
  for (unsigned char c : d) {
    s += hex[c >> 4];
    s += hex[c & 15];
  }
  return s;
}

bool validSha(const std::string& s) {
  if (s.size() != 64) return false;
  for (char c : s)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  return true;
}

class Session {
 public:
  Session(const ServerOptions& o) : opts_(o) {}

  // Returns false to stop the server.
  bool serve(int fd) {
    while (true) {
      Header h;
      if (!readAll(fd, &h, sizeof h)) return true;  // client gone
      if (h.magic != kMagic || h.version != kVersion) return true;
      if (h.payloadLen > kMaxPayload) return true;  // drop the client, never allocate attacker-sized buffers
      // Grow the buffer as bytes actually arrive (geometric, from 16 MiB): a header claiming 2 GiB followed by a
      // closed connection must not make the phone allocate 2 GiB up front.
      std::vector<uint8_t> payload;
      for (uint64_t got = 0; got < h.payloadLen;) {
        const uint64_t want = std::min<uint64_t>(h.payloadLen - got, uint64_t(16) << 20);
        if (payload.capacity() < got + want)
          payload.reserve(static_cast<size_t>(std::min<uint64_t>(h.payloadLen, std::max<uint64_t>(2 * payload.capacity(), got + want))));
        payload.resize(static_cast<size_t>(got + want));
        if (!readAll(fd, payload.data() + got, static_cast<size_t>(want))) return true;
        got += want;
      }
      Out out;
      Status s = handle(static_cast<MsgType>(h.type), payload, out);
      Header r{kMagic, kVersion, h.type, h.requestId, 0, 0};
      if (!s) {
        r.flags = 1;
        out.b.assign(s.message().begin(), s.message().end());
      }
      r.payloadLen = out.b.size();
      if (!writeAll(fd, &r, sizeof r) || !writeAll(fd, out.b.data(), out.b.size())) return true;
      if (static_cast<MsgType>(h.type) == MsgType::Shutdown) return false;
    }
  }

 private:
  std::string blobPath(const std::string& sha) const { return opts_.blobDir + "/" + sha; }

  Status handle(MsgType t, const std::vector<uint8_t>& payload, Out& out) {
    In in{payload.data(), payload.data() + payload.size()};
    switch (t) {
      case MsgType::Hello: {
        std::string info = opts_.deviceInfo ? opts_.deviceInfo() : opts_.deviceInfoJson;
        // Advertise the optional messages (clients of older servers see no "features").
        if (size_t close = info.rfind('}'); close != std::string::npos) {
          bool empty = info.find_first_not_of(" \t\n{", info.find('{') + 1) == close;
          info.insert(close, std::string(empty ? "" : ",") + "\"features\":[\"profile\",\"gpu_inspect\",\"memory\",\"run2\"]");
        }
        out.bytes(info.data(), info.size());
        return Status::ok();
      }
      case MsgType::HasBlob: {
        std::string sha(payload.begin(), payload.end());
        if (!validSha(sha)) return Status::error("bad sha");
        std::error_code ec;  // non-throwing overload (-fno-exceptions)
        out.put<uint8_t>(std::filesystem::exists(blobPath(sha), ec) ? 1 : 0);
        return Status::ok();
      }
      case MsgType::PutBlob: {
        if (payload.size() < 64) return Status::error("bad PUT_BLOB");
        std::string sha(payload.begin(), payload.begin() + 64);
        if (!validSha(sha)) return Status::error("bad sha");
        const uint8_t* data = payload.data() + 64;
        size_t n = payload.size() - 64;
        if (sha256Hex(data, n) != sha) return Status::error("blob hash mismatch");
        std::string tmp = blobPath(sha) + ".tmp";
        std::FILE* f = std::fopen(tmp.c_str(), "wb");
        if (!f) return Status::error("cannot write blob");
        bool ok = std::fwrite(data, 1, n, f) == n;
        ok = (std::fclose(f) == 0) && ok;
        if (!ok || std::rename(tmp.c_str(), blobPath(sha).c_str()) != 0) return Status::error("cannot store blob");
        return Status::ok();
      }
      case MsgType::LoadModel: {
        // sha256hex, then optionally i32 gpu_replay (RuntimeOptions::gpuReplay: -1 env, 0 off, 1 on) and
        // u32 gpu_segments (0 = unchanged; else GpuContext::setMaxSegments, process-wide).
        if (payload.size() != 64 && payload.size() != 68 && payload.size() != 72) return Status::error("bad sha");
        std::string sha(payload.begin(), payload.begin() + 64);
        if (!validSha(sha)) return Status::error("bad sha");
        rt::RuntimeOptions ro;
        In opt{payload.data() + 64, payload.data() + payload.size()};
        if (payload.size() >= 68) {
          const int32_t replay = opt.get<int32_t>();
          if (replay < -1 || replay > 1) return Status::error("bad LOAD_MODEL gpu_replay");
          ro.gpuReplay = replay;
        }
        if (payload.size() == 72) {
          const uint32_t segs = opt.get<uint32_t>();
          if (segs > 4) return Status::error("bad LOAD_MODEL gpu_segments");
          if (segs) {
            auto g = rt::GpuContext::shared();
            if (!g) return g.status();
            (*g)->setMaxSegments(segs);
          }
        }
        auto m = readModelFile(blobPath(sha));
        if (!m) return m.status();
        LK_RETURN_IF_ERROR(verifyModel(**m, VerifyOptions{true}));  // never execute unverified IR
        std::shared_ptr<Model> model(m.take());
        auto e = rt::Executable::create(model, {}, ro);
        if (!e) return e.status();
        uint32_t id = nextId_++;
        models_[id] = e.take();
        out.put<uint32_t>(id);
        return Status::ok();
      }
      case MsgType::Run: return run(in, out);
      case MsgType::Shutdown: return Status::ok();
      case MsgType::Profile: {
        uint32_t id = in.get<uint32_t>(), level = in.get<uint32_t>(), flags = in.get<uint32_t>();
        if (!in.ok) return Status::error("bad PROFILE");
        auto it = models_.find(id);
        if (it == models_.end()) return Status::error("unknown model id");
        it->second->setProfile(rt::ProfileOptions{static_cast<int>(level), (flags & 1) != 0});
        return Status::ok();
      }
      case MsgType::TakeProfile: {
        uint32_t id = in.get<uint32_t>();
        if (!in.ok) return Status::error("bad TAKE_PROFILE");
        auto it = models_.find(id);
        if (it == models_.end()) return Status::error("unknown model id");
        const std::string j = it->second->takeProfileJson();
        out.bytes(j.data(), j.size());
        return Status::ok();
      }
      case MsgType::GpuInspect: return gpuInspect(payload, out);
      case MsgType::Memory: return memory(payload, out);
      case MsgType::Run2: return run2(in, out);
    }
    return Status::error("unknown message type");
  }

  Status gpuInspect(const std::vector<uint8_t>& payload, Out& out) {
    if (payload.empty()) return Status::error("bad GPU_INSPECT");
    const bool wantArchive = payload[0] != 0;
    std::vector<std::string> names;
    std::string cur;
    for (size_t i = 1; i < payload.size(); ++i) {
      if (payload[i] == '\n') {
        if (!cur.empty()) names.push_back(cur);
        cur.clear();
      } else {
        cur += static_cast<char>(payload[i]);
      }
    }
    if (!cur.empty()) names.push_back(cur);
    // Metal function names: identifiers only (found by the protocol fuzz test: invalid UTF-8 made NSString nil and
    // Metal abort on newFunction(nil)).
    if (names.size() > 4096) return Status::error("too many function names");
    for (const std::string& nm : names) {
      if (nm.size() > 256) return Status::error("bad function name");
      for (char ch : nm)
        if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_'))
          return Status::error("bad function name");
    }
    const std::string archive = wantArchive ? opts_.blobDir + "/gpu_inspect.metallib-archive" : std::string();
    std::error_code ec;
    if (wantArchive) std::filesystem::remove(archive, ec);
    const std::string j = rt::gpuInspectJson(names, archive);
    out.put<uint64_t>(j.size());
    out.bytes(j.data(), j.size());
    if (wantArchive) {
      if (std::FILE* f = std::fopen(archive.c_str(), "rb")) {
        char buf[1 << 16];
        size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.bytes(buf, n);
        std::fclose(f);
      }
    }
    return Status::ok();
  }

  // tensor := u32 dtype, u32 ndim, i64 shape[ndim], u64 nbytes, bytes. `data` points into `in` (no copy); scalars
  // (ndim 0) are copied into `scalar`.
  static Status parseTensor(In& in, rt::TensorView& v, int64_t& scalar) {
    const uint32_t rawDt = in.get<uint32_t>();  // checked before the cast: DType is 8 bits (256 would wrap to F32)
    uint32_t ndim = in.get<uint32_t>();
    if (!in.ok || ndim > 16 || rawDt > static_cast<uint32_t>(DType::Bool)) return Status::error("bad tensor header");
    v.dtype = static_cast<DType>(rawDt);
    v.shape.clear();
    for (uint32_t d = 0; d < ndim; ++d) v.shape.push_back(in.get<int64_t>());
    uint64_t nbytes = in.get<uint64_t>();
    if (!in.ok) return Status::error("truncated tensor header");
    uint64_t expect = 0;
    LK_RETURN_IF_ERROR(byteCount(v.dtype, v.shape, expect));
    // (a scalar's byte count is its dtype size; checked like any other shape)
    if (nbytes != expect) return Status::error("tensor byte count doesn't match its shape");
    const uint8_t* data = in.take(nbytes);
    if (!in.ok) return Status::error("truncated tensor");
    if (ndim == 0) {
      scalar = 0;
      std::memcpy(&scalar, data, std::min<uint64_t>(nbytes, 8));
      v.data = &scalar;
    } else {
      v.data = const_cast<uint8_t*>(data);
    }
    return Status::ok();
  }

  static Status byteCount(DType dt, const std::vector<int64_t>& shape, uint64_t& bytes) {
    bytes = dtypeSize(dt);
    for (int64_t d : shape) {
      if (d < 0 || (d && bytes > kMaxPayload / static_cast<uint64_t>(d))) return Status::error("bad tensor shape");
      bytes *= static_cast<uint64_t>(d);
    }
    return Status::ok();
  }

  // The inputs get an owned, page-aligned copy (made once, before the timed calls): with an owner, GPU models wrap
  // them with no copy (cached across the calls), as a Mac caller's numpy / torch inputs and the MLX Swift / ExecuTorch
  // harnesses' pre-built inputs are; without one every timed call copied them (MNIST B=1024: 3.2 MB per call).
  Status runCalls(rt::Executable& e, std::span<const rt::TensorView> in, uint32_t warmup, uint32_t iters,
                  std::vector<rt::OwnedTensor>& outs, std::vector<double>& times) {
    std::vector<rt::TensorView> views(in.begin(), in.end());
    for (rt::TensorView& v : views) {
      if (v.owner || v.mtlBuffer || v.shape.empty()) continue;
      size_t bytes = dtypeSize(v.dtype);
      for (int64_t d : v.shape) bytes *= static_cast<size_t>(d);
      if (bytes == 0) continue;
      void* mem = nullptr;
      if (posix_memalign(&mem, 16384, (bytes + 16383) / 16384 * 16384) != 0) return Status::error("out of memory");
      std::memcpy(mem, v.data, bytes);
      v.owner = std::shared_ptr<void>(mem, std::free);
      v.data = mem;
    }
    for (uint32_t r = 0; r < warmup + std::max<uint32_t>(iters, 1); ++r) {
      auto t0 = std::chrono::steady_clock::now();
      LK_RETURN_IF_ERROR(e.run(views, outs));
      auto t1 = std::chrono::steady_clock::now();
      if (r >= warmup) times.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
    }
    return Status::ok();
  }

  static void putTensor(Out& out, const rt::OwnedTensor& t) {
    out.put<uint32_t>(static_cast<uint32_t>(t.dtype));
    out.put<uint32_t>(static_cast<uint32_t>(t.shape.size()));
    size_t nbytes = dtypeSize(t.dtype);
    for (int64_t d : t.shape) {
      out.put<int64_t>(d);
      nbytes *= static_cast<size_t>(d);
    }
    out.put<uint64_t>(nbytes);
    out.bytes(t.data(), nbytes);
  }

  Status run(In& in, Out& out) {
    uint32_t id = in.get<uint32_t>(), warmup = in.get<uint32_t>(), iters = in.get<uint32_t>(),
             n = in.get<uint32_t>();
    if (!in.ok || n > 1024 || warmup > 100000 || iters > 100000) return Status::error("bad RUN");
    auto it = models_.find(id);
    if (it == models_.end()) return Status::error("unknown model id");
    std::vector<rt::TensorView> views(n);
    std::vector<int64_t> scalars(n);
    for (uint32_t i = 0; i < n; ++i) LK_RETURN_IF_ERROR(parseTensor(in, views[i], scalars[i]));
    std::vector<rt::OwnedTensor> outs;
    std::vector<double> times;
    LK_RETURN_IF_ERROR(runCalls(*it->second, views, warmup, iters, outs, times));
    out.put<uint32_t>(static_cast<uint32_t>(outs.size()));
    for (const rt::OwnedTensor& t : outs) putTensor(out, t);
    out.put<uint32_t>(static_cast<uint32_t>(times.size()));
    for (double t : times) out.put<double>(t);
    return Status::ok();
  }

  // ---- RUN2: content-addressed and generated inputs, on-device comparisons (docs: server.h)

  // A blob holding one encoded tensor, read once and kept in a small in-memory cache (test inputs / references
  // are re-used across many runs).
  Expected<std::shared_ptr<const std::vector<uint8_t>>> tensorBlob(const std::string& sha) {
    if (!validSha(sha)) return Status::error("bad sha");
    if (auto it = tensorCache_.find(sha); it != tensorCache_.end()) return it->second;
    std::error_code ec;
    const auto size = std::filesystem::file_size(blobPath(sha), ec);
    if (ec) return Status::error("unknown blob " + sha.substr(0, 12));
    if (size > kMaxPayload) return Status::error("blob too large");
    auto bytes = std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(size));
    std::FILE* f = std::fopen(blobPath(sha).c_str(), "rb");
    if (!f) return Status::error("cannot read blob");
    const bool ok = std::fread(bytes->data(), 1, bytes->size(), f) == bytes->size();
    std::fclose(f);
    if (!ok) return Status::error("cannot read blob");
    if (tensorCacheBytes_ + bytes->size() > (size_t(256) << 20)) {
      tensorCache_.clear();
      tensorCacheBytes_ = 0;
    }
    tensorCacheBytes_ += bytes->size();
    tensorCache_[sha] = bytes;
    return std::shared_ptr<const std::vector<uint8_t>>(bytes);
  }

  Status tensorFromBlob(const std::string& sha, std::vector<std::shared_ptr<const std::vector<uint8_t>>>& keep,
                        rt::TensorView& v, int64_t& scalar) {
    auto b = tensorBlob(sha);
    if (!b) return b.status();
    keep.push_back(*b);
    In bin{(*b)->data(), (*b)->data() + (*b)->size()};
    LK_RETURN_IF_ERROR(parseTensor(bin, v, scalar));
    if (bin.p != bin.end) return Status::error("tensor blob has trailing bytes");
    return Status::ok();
  }

  Status run2(In& in, Out& out) {
    const uint32_t id = in.get<uint32_t>(), warmup = in.get<uint32_t>(), iters = in.get<uint32_t>(),
                   flags = in.get<uint32_t>(), n = in.get<uint32_t>();
    if (!in.ok || n > 1024 || warmup > 100000 || iters > 100000) return Status::error("bad RUN2");
    auto it = models_.find(id);
    if (it == models_.end()) return Status::error("unknown model id");
    std::vector<rt::TensorView> views(n);
    std::vector<int64_t> scalars(n);
    std::vector<std::shared_ptr<const std::vector<uint8_t>>> keep;  // blob bytes the views point into
    std::vector<std::vector<uint8_t>> generated;
    for (uint32_t i = 0; i < n; ++i) {
      const uint8_t kind = in.get<uint8_t>();
      if (!in.ok) return Status::error("truncated RUN2 input");
      if (kind == 0) {
        LK_RETURN_IF_ERROR(parseTensor(in, views[i], scalars[i]));
      } else if (kind == 1) {
        const uint8_t* sha = in.take(64);
        if (!in.ok) return Status::error("truncated RUN2 input");
        LK_RETURN_IF_ERROR(tensorFromBlob(std::string(sha, sha + 64), keep, views[i], scalars[i]));
      } else if (kind == 2) {
        LK_RETURN_IF_ERROR(generateTensor(in, views[i], scalars[i], generated));
      } else {
        return Status::error("bad RUN2 input kind");
      }
    }
    struct Cmp {
      uint32_t out;
      uint8_t kind;
      uint32_t other;
      std::string sha;
      double rtol, atol;
    };
    const uint32_t nc = in.get<uint32_t>();
    if (!in.ok || nc > 1024) return Status::error("bad RUN2 comparisons");
    std::vector<Cmp> cmps(nc);
    for (Cmp& c : cmps) {
      c.out = in.get<uint32_t>();
      c.kind = in.get<uint8_t>();
      if (c.kind == 1) {
        const uint8_t* sha = in.take(64);
        if (sha) c.sha.assign(sha, sha + 64);
      } else if (c.kind == 3) {
        c.other = in.get<uint32_t>();
      } else {
        return Status::error("bad RUN2 comparison kind");
      }
      c.rtol = in.get<double>();
      c.atol = in.get<double>();
      if (!in.ok) return Status::error("truncated RUN2 comparison");
      if (!(c.rtol >= 0) || !(c.atol >= 0)) return Status::error("bad RUN2 tolerance");
    }
    std::vector<rt::OwnedTensor> outs;
    std::vector<double> times;
    LK_RETURN_IF_ERROR(runCalls(*it->second, views, warmup, iters, outs, times));
    if (flags & 1) {
      out.put<uint32_t>(static_cast<uint32_t>(outs.size()));
      for (const rt::OwnedTensor& t : outs) putTensor(out, t);
    } else {
      out.put<uint32_t>(0);
    }
    out.put<uint32_t>(nc);
    for (const Cmp& c : cmps) {
      if (c.out >= outs.size()) return Status::error("RUN2 comparison: no output " + std::to_string(c.out));
      const rt::OwnedTensor& got = outs[c.out];
      rt::TensorView ref;
      int64_t refScalar = 0;
      if (c.kind == 3) {
        if (c.other >= outs.size()) return Status::error("RUN2 comparison: no output " + std::to_string(c.other));
        ref = rt::TensorView{outs[c.other].data(), outs[c.other].dtype, outs[c.other].shape};
      } else {
        LK_RETURN_IF_ERROR(tensorFromBlob(c.sha, keep, ref, refScalar));
      }
      LK_RETURN_IF_ERROR(compare(c.out, got, ref, c.rtol, c.atol, out));
    }
    out.put<uint32_t>(static_cast<uint32_t>(times.size()));
    for (double t : times) out.put<double>(t);
    return Status::ok();
  }

  static Status compare(uint32_t index, const rt::OwnedTensor& got, const rt::TensorView& ref, double rtol, double atol,
                        Out& out) {
    uint64_t count = 1;
    for (int64_t d : got.shape) count *= static_cast<uint64_t>(d);
    uint64_t refCount = 1;
    for (int64_t d : ref.shape) refCount *= static_cast<uint64_t>(d);
    if (count != refCount || got.dtype != ref.dtype)
      return Status::error("RUN2 comparison of output " + std::to_string(index) + ": shape or dtype differs");
    auto at = [](DType dt, const void* p, uint64_t i) -> double {
      switch (dt) {
        case DType::F32: return static_cast<const float*>(p)[i];
        case DType::I32: return static_cast<const int32_t*>(p)[i];
        case DType::I64: return static_cast<double>(static_cast<const int64_t*>(p)[i]);
        case DType::U8: return static_cast<const uint8_t*>(p)[i];
        case DType::Bool: return static_cast<const uint8_t*>(p)[i];
        default: return 0;
      }
    };
    if (got.dtype != DType::F32 && got.dtype != DType::I32 && got.dtype != DType::I64 && got.dtype != DType::U8 &&
        got.dtype != DType::Bool)
      return Status::error(std::string("RUN2 comparison: unsupported dtype ") + dtypeName(got.dtype));
    uint64_t nonfinite = 0, mismatches = 0;
    int64_t worst = -1;
    double maxErr = 0, maxRef = 0, sse = 0, ssr = 0, worstExcess = -HUGE_VAL;
    for (uint64_t i = 0; i < count; ++i) {
      const double g = at(got.dtype, got.data(), i), r = at(ref.dtype, ref.data, i);
      if (!std::isfinite(g)) {
        ++nonfinite;
        if (!(std::isnan(g) && std::isnan(r)) && g != r) ++mismatches;
        continue;
      }
      const double e = std::fabs(g - r);
      maxErr = std::max(maxErr, e);
      maxRef = std::max(maxRef, std::fabs(r));
      sse += e * e;
      ssr += r * r;
      const double excess = e - (atol + rtol * std::fabs(r));
      if (excess > 0) ++mismatches;
      if (excess > worstExcess) worstExcess = excess, worst = static_cast<int64_t>(i);
    }
    out.put<uint32_t>(index);
    out.put<uint64_t>(count);
    out.put<uint64_t>(nonfinite);
    out.put<uint64_t>(mismatches);
    out.put<int64_t>(worst);
    out.put<double>(maxErr);
    out.put<double>(maxRef);
    out.put<double>(sse);
    out.put<double>(ssr);
    return Status::ok();
  }

  // Generated input: u32 dtype, u32 ndim, i64 shape[ndim], u64 seed, u32 dist, f64 a, f64 b.
  // dist 0 normal(mean a, std b), 1 uniform [a, b), 2 integers in [a, b). Element i uses the splitmix64 values of
  // seed + (2i + 1) * golden and seed + (2i + 2) * golden; python/lokinfer/remote.py `generate` is the same function.
  Status generateTensor(In& in, rt::TensorView& v, int64_t& scalar, std::vector<std::vector<uint8_t>>& store) {
    const uint32_t rawDt = in.get<uint32_t>(), ndim = in.get<uint32_t>();
    if (!in.ok || ndim > 16) return Status::error("bad generated tensor");
    std::vector<int64_t> shape(ndim);
    for (auto& d : shape) d = in.get<int64_t>();
    const uint64_t seed = in.get<uint64_t>();
    const uint32_t dist = in.get<uint32_t>();
    const double a = in.get<double>(), b = in.get<double>();
    if (!in.ok) return Status::error("truncated generated tensor");
    if (rawDt != static_cast<uint32_t>(DType::F32) && rawDt != static_cast<uint32_t>(DType::I32) &&
        rawDt != static_cast<uint32_t>(DType::I64))
      return Status::error("generated tensors are f32, i32 or i64");
    if (dist > 2 || !std::isfinite(a) || !std::isfinite(b)) return Status::error("bad generated distribution");
    const auto dt = static_cast<DType>(rawDt);
    uint64_t bytes = 0;
    LK_RETURN_IF_ERROR(byteCount(dt, shape, bytes));
    if (bytes > (uint64_t(1) << 30)) return Status::error("generated tensor too large");
    if (dist == 2 && !(b > a && b - a < 9.0e15)) return Status::error("bad generated integer range");
    store.emplace_back(static_cast<size_t>(std::max<uint64_t>(bytes, 8)));
    uint8_t* p = store.back().data();
    const uint64_t count = bytes / dtypeSize(dt);
    constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ull;
    auto mix = [](uint64_t z) {
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
      return z ^ (z >> 31);
    };
    for (uint64_t i = 0; i < count; ++i) {
      const double u1 = static_cast<double>(mix(seed + (2 * i + 1) * kGolden) >> 11) * 0x1.0p-53;
      double x;
      if (dist == 0) {
        const double u2 = static_cast<double>(mix(seed + (2 * i + 2) * kGolden) >> 11) * 0x1.0p-53;
        x = a + b * std::sqrt(-2.0 * std::log(1.0 - u1)) * std::cos(6.283185307179586 * u2);
      } else if (dist == 1) {
        x = a + (b - a) * u1;
      } else {
        x = std::floor(a) + std::floor(u1 * (std::floor(b) - std::floor(a)));
      }
      if (dt == DType::F32) reinterpret_cast<float*>(p)[i] = static_cast<float>(x);
      else if (dt == DType::I32) reinterpret_cast<int32_t*>(p)[i] = static_cast<int32_t>(x);
      else reinterpret_cast<int64_t*>(p)[i] = static_cast<int64_t>(x);
    }
    v.dtype = dt;
    v.shape = shape;
    if (ndim == 0) {
      scalar = 0;
      std::memcpy(&scalar, p, dtypeSize(dt));
      v.data = &scalar;
    } else {
      v.data = p;
    }
    return Status::ok();
  }

  // MEMORY: process memory (task_vm_info) and the Metal device's allocated size, as JSON.
  Status memory(const std::vector<uint8_t>& payload, Out& out) {
    if (!payload.empty()) return Status::error("bad MEMORY");
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
      return Status::error("task_info failed");
    const std::string j = "{\"phys_footprint\":" + std::to_string(info.phys_footprint) +
                          ",\"phys_footprint_peak\":" + std::to_string(info.ledger_phys_footprint_peak) +
                          ",\"resident\":" + std::to_string(info.resident_size) +
                          ",\"resident_peak\":" + std::to_string(info.resident_size_peak) +
                          ",\"gpu_allocated\":" + std::to_string(rt::gpuAllocatedBytes()) +
                          ",\"models\":" + std::to_string(models_.size()) + "}";
    out.bytes(j.data(), j.size());
    return Status::ok();
  }

  std::unordered_map<std::string, std::shared_ptr<const std::vector<uint8_t>>> tensorCache_;
  size_t tensorCacheBytes_ = 0;
  const ServerOptions& opts_;
  std::unordered_map<uint32_t, std::unique_ptr<rt::Executable>> models_;
  uint32_t nextId_ = 1;
};

}  // namespace

Status serveConnection(int fd, const ServerOptions& opts) {
  Session session(opts);
  session.serve(fd);
  return Status::ok();
}

Status runServer(const ServerOptions& opts) {
  std::error_code ec;
  std::filesystem::create_directories(opts.blobDir, ec);
  int fd = ::socket(AF_INET6, SOCK_STREAM, 0);
  if (fd < 0) return Status::error("socket() failed");
  int one = 1, zero = 0;
  ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  ::setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &zero, sizeof zero);  // dual-stack
  sockaddr_in6 addr{};
  addr.sin6_family = AF_INET6;
  addr.sin6_port = htons(opts.port);
  addr.sin6_addr = in6addr_any;
  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof addr) != 0 || ::listen(fd, 4) != 0) {
    ::close(fd);
    return Status::error("cannot listen on port " + std::to_string(opts.port));
  }
  if (opts.verbose) std::printf("lokinfer remote executor listening on [::]:%u\n", opts.port), std::fflush(stdout);
  Session session(opts);
  bool keepGoing = true;
  while (keepGoing) {
    int c = ::accept(fd, nullptr, nullptr);
    if (c < 0) continue;
    ::setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    // One client at a time: a client that vanishes without closing (Mac asleep, tunnel reset) must not wedge the
    // loop. Keepalive probes find a dead peer in ~25 s (recv then fails and the next client is accepted); writing to
    // a closed peer returns an error instead of killing the process with SIGPIPE.
    int idle = 10, intvl = 5, cnt = 3;
    ::setsockopt(c, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
    ::setsockopt(c, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one);
    ::setsockopt(c, IPPROTO_TCP, TCP_KEEPALIVE, &idle, sizeof idle);
    ::setsockopt(c, IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof intvl);
    ::setsockopt(c, IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof cnt);
    if (opts.verbose) std::printf("client connected\n"), std::fflush(stdout);
    keepGoing = session.serve(c);
    ::close(c);
  }
  ::close(fd);
  return Status::ok();
}

}  // namespace lk::remote
