// Remote-protocol fuzzing (the iPhone test host listens on the network): hostile frames against the real server
// session (`lk::remote::serveConnection`, the code `lokinfer-serve` and LokiHost run), over a socketpair. Every case
// must end in an error reply or a dropped connection, never a crash, a hang or an attacker-sized allocation; the
// session must keep serving valid requests afterwards.
#include <catch2/catch_test_macros.hpp>

#include <CommonCrypto/CommonDigest.h>
#include <mach/mach.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "lokinfer/compiler/passes.h"
#include "lokinfer/ir/binary.h"
#include "lokinfer/ir/text.h"
#include "lokinfer/remote/server.h"

using namespace lk;
using lk::remote::MsgType;

namespace {

struct Reply {
  uint32_t flags = 0;
  std::vector<uint8_t> body;
  bool ok() const { return flags == 0; }
  std::string text() const { return std::string(body.begin(), body.end()); }
};

#pragma pack(push, 1)
struct Header {
  uint32_t magic;
  uint16_t version;
  uint16_t type;
  uint32_t requestId;
  uint32_t flags;
  uint64_t payloadLen;
};
#pragma pack(pop)
static_assert(sizeof(Header) == 24);

std::string blobDir() {
  static std::string d = [] {
    auto p = std::filesystem::temp_directory_path() / ("lokinfer-fuzz-" + std::to_string(::getpid()));
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
    return p.string();
  }();
  return d;
}

// One client connection to a fresh server session running on its own thread.
class Conn {
 public:
  Conn() {
    int sv[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    int one = 1;
    ::setsockopt(sv[0], SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
    ::setsockopt(sv[1], SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
    fd_ = sv[0];
    serverFd_ = sv[1];
    opts_.blobDir = blobDir();
    opts_.deviceInfoJson = R"({"device": "fuzz"})";
    opts_.verbose = false;
    thread_ = std::thread([this] {
      remote::serveConnection(serverFd_, opts_);
      ::shutdown(serverFd_, SHUT_RDWR);  // the client sees EOF once the session drops it
      done_ = true;
    });
  }
  ~Conn() {
    ::shutdown(fd_, SHUT_RDWR);
    thread_.join();
    ::close(fd_);
    ::close(serverFd_);
  }
  bool serverDone() const { return done_; }

  bool sendRaw(const void* p, size_t n) {
    auto* c = static_cast<const char*>(p);
    while (n) {
      ssize_t w = ::send(fd_, c, n, 0);
      if (w <= 0) return false;
      c += w;
      n -= static_cast<size_t>(w);
    }
    return true;
  }
  bool recvRaw(void* p, size_t n) {
    auto* c = static_cast<char*>(p);
    while (n) {
      ssize_t r = ::recv(fd_, c, n, 0);
      if (r <= 0) return false;
      c += r;
      n -= static_cast<size_t>(r);
    }
    return true;
  }
  // Sends a frame (header fields overridable) and returns the reply, or nullopt when the server dropped us.
  std::optional<Reply> call(uint16_t type, const std::vector<uint8_t>& payload, uint32_t magic = remote::kMagic,
                            uint16_t version = remote::kVersion) {
    Header h{magic, version, type, ++rid_, 0, payload.size()};
    if (!sendRaw(&h, sizeof h) || (!payload.empty() && !sendRaw(payload.data(), payload.size()))) return std::nullopt;
    Header r{};
    if (!recvRaw(&r, sizeof r)) return std::nullopt;
    REQUIRE(r.magic == remote::kMagic);
    REQUIRE(r.requestId == rid_);
    REQUIRE(r.type == type);
    REQUIRE(r.payloadLen < (uint64_t(1) << 32));
    Reply rep;
    rep.flags = r.flags;
    rep.body.resize(r.payloadLen);
    if (r.payloadLen && !recvRaw(rep.body.data(), rep.body.size())) return std::nullopt;
    return rep;
  }
  std::optional<Reply> call(MsgType t, const std::vector<uint8_t>& p) { return call(static_cast<uint16_t>(t), p); }

 private:
  int fd_ = -1, serverFd_ = -1;
  remote::ServerOptions opts_;
  std::thread thread_;
  std::atomic<bool> done_{false};
  uint32_t rid_ = 0;
};

struct Buf {
  std::vector<uint8_t> b;
  template <typename T>
  Buf& put(T v) {
    size_t at = b.size();
    b.resize(at + sizeof(T));
    std::memcpy(b.data() + at, &v, sizeof(T));
    return *this;
  }
  Buf& bytes(const void* p, size_t n) {
    b.insert(b.end(), static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + n);
    return *this;
  }
  Buf& str(const std::string& s) { return bytes(s.data(), s.size()); }
};

std::string sha256Hex(const std::vector<uint8_t>& v) {
  unsigned char d[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(v.data(), static_cast<CC_LONG>(v.size()), d);
  static const char* hex = "0123456789abcdef";
  std::string s;
  for (unsigned char c : d) s += hex[c >> 4], s += hex[c & 15];
  return s;
}

std::vector<uint8_t> modelBytes() {
  static std::vector<uint8_t> bytes = [] {
    auto m = parseModel(R"(model {
  init {
    return
  }
  exec(%x: tensor<f32[N]>, %s: f32) {
    %a = kernel "add_f32_cpu"(%x, %x) : tensor<f32[N]>
    %o = kernel "mul_f32_cpu"(%a, %x) : tensor<f32[N]>
    return %o
  }
}
)");
    REQUIRE(m.isOk());
    REQUIRE(runPasses(**m, defaultPipeline()).isOk());
    auto b = writeModelBytes(**m);
    REQUIRE(b.isOk());
    return b.take();
  }();
  return bytes;
}

std::string putModel(Conn& c, const std::vector<uint8_t>& bytes) {
  std::string sha = sha256Hex(bytes);
  auto r = c.call(MsgType::PutBlob, Buf().str(sha).bytes(bytes.data(), bytes.size()).b);
  REQUIRE(r);
  REQUIRE(r->ok());
  return sha;
}

uint32_t loadModel(Conn& c) {
  std::string sha = putModel(c, modelBytes());
  auto r = c.call(MsgType::LoadModel, Buf().str(sha).b);
  REQUIRE(r);
  INFO(r->text());
  REQUIRE(r->ok());
  uint32_t id;
  std::memcpy(&id, r->body.data(), 4);
  return id;
}

// A valid RUN of the test model: x = [1, 2, ..., n], s = 0.5 (scalar).
std::vector<uint8_t> runPayload(uint32_t id, int64_t n = 4) {
  Buf b;
  b.put<uint32_t>(id).put<uint32_t>(0).put<uint32_t>(1).put<uint32_t>(2);
  b.put<uint32_t>(0).put<uint32_t>(1).put<int64_t>(n).put<uint64_t>(4 * n);
  for (int64_t i = 0; i < n; ++i) b.put<float>(float(i + 1));
  b.put<uint32_t>(0).put<uint32_t>(0).put<uint64_t>(4).put<float>(0.5f);
  return b.b;
}

// Checks a valid RUN still works and computes (x + x) * x.
void checkHealthy(Conn& c, uint32_t id) {
  auto h = c.call(MsgType::Hello, {});
  REQUIRE(h);
  REQUIRE(h->ok());
  auto r = c.call(MsgType::Run, runPayload(id));
  REQUIRE(r);
  INFO(r->text());
  REQUIRE(r->ok());
  const uint8_t* p = r->body.data();
  uint32_t n;
  std::memcpy(&n, p, 4);
  REQUIRE(n == 1);
  // u32 n | u32 dtype, u32 ndim, i64 shape, u64 nbytes, data
  const float* out = reinterpret_cast<const float*>(p + 4 + 4 + 4 + 8 + 8);
  for (int i = 0; i < 4; ++i) REQUIRE(out[i] == 2.f * float(i + 1) * float(i + 1));
}

int64_t physFootprint() {
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) return 0;
  return static_cast<int64_t>(info.phys_footprint);
}

}  // namespace

TEST_CASE("remote fuzz: malformed headers drop the connection without big allocations", "[remote][fuzz]") {
  SECTION("bad magic") {
    Conn c;
    REQUIRE_FALSE(c.call(1, {}, 0xDEADBEEF));
  }
  SECTION("bad version") {
    Conn c;
    REQUIRE_FALSE(c.call(1, {}, remote::kMagic, 99));
  }
  SECTION("payload above the 2 GiB cap") {
    Conn c;
    Header h{remote::kMagic, remote::kVersion, 1, 1, 0, (uint64_t(1) << 31) + 1};
    REQUIRE(c.sendRaw(&h, sizeof h));
    Header r;
    REQUIRE_FALSE(c.recvRaw(&r, sizeof r));
  }
  SECTION("huge claimed payload, then the client hangs up: no up-front allocation") {
    const int64_t before = physFootprint();
    for (int i = 0; i < 4; ++i) {
      Conn c;
      Header h{remote::kMagic, remote::kVersion, static_cast<uint16_t>(MsgType::PutBlob), 1, 0, uint64_t(1) << 31};
      REQUIRE(c.sendRaw(&h, sizeof h));
      std::vector<uint8_t> some(1000, 7);
      REQUIRE(c.sendRaw(some.data(), some.size()));
    }  // ~Conn closes the client: the session sees EOF mid-payload
    // The server buffers what arrived (16 MiB chunks at most), never the claimed 2 GiB.
    REQUIRE(physFootprint() - before < (int64_t(256) << 20));
  }
  SECTION("truncated header") {
    Conn c;
    uint8_t half[10] = {0x4C, 0x4F, 0x4B, 0x49};
    REQUIRE(c.sendRaw(half, sizeof half));
  }  // close mid-header: the session must just end
}

TEST_CASE("remote fuzz: malformed requests get errors and the session keeps serving", "[remote][fuzz]") {
  Conn c;
  const uint32_t id = loadModel(c);
  checkHealthy(c, id);
  const std::string goodSha = sha256Hex(modelBytes());

  struct Case {
    const char* name;
    uint16_t type;
    std::vector<uint8_t> payload;
  };
  auto tensor = [](Buf& b, uint32_t dt, std::vector<int64_t> shape, uint64_t nbytes, size_t actual) {
    b.put<uint32_t>(dt).put<uint32_t>(static_cast<uint32_t>(shape.size()));
    for (int64_t d : shape) b.put<int64_t>(d);
    b.put<uint64_t>(nbytes);
    b.b.resize(b.b.size() + actual, 0);
  };
  auto runHead = [&](uint32_t n, uint32_t mid = 0) {
    Buf b;
    b.put<uint32_t>(mid ? mid : id).put<uint32_t>(0).put<uint32_t>(1).put<uint32_t>(n);
    return b;
  };
  std::vector<Case> cases;
  cases.push_back({"unknown type 0", 0, {}});
  cases.push_back({"unknown type 77", 77, {1, 2, 3}});
  cases.push_back({"unknown type 0xffff", 0xffff, {}});
  cases.push_back({"HAS_BLOB empty", 2, {}});
  cases.push_back({"HAS_BLOB path traversal", 2, Buf().str("../../../../etc/passwd" + std::string(42, 'a')).b});
  cases.push_back({"HAS_BLOB uppercase hex", 2, Buf().str(std::string(64, 'A')).b});
  cases.push_back({"PUT_BLOB short", 3, Buf().str("abc").b});
  cases.push_back({"PUT_BLOB bad sha chars", 3, Buf().str(std::string(63, 'a') + "/").str("data").b});
  cases.push_back({"PUT_BLOB hash mismatch", 3, Buf().str(std::string(64, 'a')).str("data").b});
  cases.push_back({"LOAD_MODEL missing blob", 4, Buf().str(std::string(64, 'b')).b});
  cases.push_back({"LOAD_MODEL bad sha", 4, Buf().str("x").b});
  cases.push_back({"LOAD_MODEL bad replay option", 4, Buf().str(goodSha).put<int32_t>(7).b});
  cases.push_back({"LOAD_MODEL bad segments option", 4, Buf().str(goodSha).put<int32_t>(0).put<uint32_t>(99).b});
  cases.push_back({"LOAD_MODEL odd payload size", 4, Buf().str(goodSha).put<uint16_t>(1).b});
  cases.push_back({"RUN empty", 5, {}});
  cases.push_back({"RUN truncated head", 5, Buf().put<uint32_t>(id).put<uint16_t>(0).b});
  cases.push_back({"RUN unknown model", 5, runPayload(0xFFFFFFF0)});
  cases.push_back({"RUN too many inputs", 5, runHead(5000).b});
  cases.push_back({"RUN huge iters", 5, Buf().put<uint32_t>(id).put<uint32_t>(0).put<uint32_t>(0x7fffffff).put<uint32_t>(0).b});
  cases.push_back({"RUN huge warmup", 5, Buf().put<uint32_t>(id).put<uint32_t>(0xffffffff).put<uint32_t>(1).put<uint32_t>(0).b});
  cases.push_back({"RUN zero inputs", 5, runHead(0).b});
  {
    Buf b = runHead(2);
    tensor(b, 9, {4}, 16, 16);  // dtype past Bool
    cases.push_back({"RUN bad dtype", 5, b.b});
  }
  {
    Buf b = runHead(2);
    tensor(b, 256, {4}, 16, 16);  // wraps to F32 if cast before the check
    cases.push_back({"RUN dtype 256", 5, b.b});
  }
  {
    Buf b = runHead(2);
    tensor(b, 0, std::vector<int64_t>(17, 1), 4, 4);
    cases.push_back({"RUN rank 17", 5, b.b});
  }
  {
    Buf b = runHead(2);
    tensor(b, 0, {-4}, 16, 16);
    cases.push_back({"RUN negative dim", 5, b.b});
  }
  {
    Buf b = runHead(2);
    tensor(b, 0, {int64_t(1) << 62, int64_t(1) << 62}, 0, 0);  // product overflows
    cases.push_back({"RUN shape overflow", 5, b.b});
  }
  {
    Buf b = runHead(2);
    tensor(b, 0, {INT64_MAX}, 0, 0);
    cases.push_back({"RUN INT64_MAX dim", 5, b.b});
  }
  {
    Buf b = runHead(2);
    tensor(b, 0, {4}, 12, 12);  // nbytes != 4 * 4
    cases.push_back({"RUN nbytes < shape", 5, b.b});
  }
  {
    Buf b = runHead(2);
    tensor(b, 0, {4}, 20, 20);
    cases.push_back({"RUN nbytes > shape", 5, b.b});
  }
  {
    Buf b = runHead(2);
    tensor(b, 0, {4}, 16, 8);  // claims 16 bytes, frame ends after 8
    cases.push_back({"RUN truncated tensor data", 5, b.b});
  }
  {
    Buf b = runHead(2);
    tensor(b, 0, {4}, 16, 16);
    tensor(b, 0, {}, 64, 64);  // scalar claiming 64 bytes
    cases.push_back({"RUN scalar nbytes 64", 5, b.b});
  }
  {
    Buf b = runHead(2);
    tensor(b, 0, {4}, 16, 16);
    tensor(b, 0, {}, 0, 0);
    cases.push_back({"RUN scalar nbytes 0", 5, b.b});
  }
  {
    Buf b = runHead(2);
    tensor(b, 7, {4}, 32, 32);  // i64 where the model wants f32
    tensor(b, 0, {}, 4, 4);
    cases.push_back({"RUN dtype mismatch", 5, b.b});
  }
  {
    Buf b = runHead(2);
    tensor(b, 0, {2, 2}, 16, 16);  // rank mismatch
    tensor(b, 0, {}, 4, 4);
    cases.push_back({"RUN rank mismatch", 5, b.b});
  }
  {
    Buf b = runHead(1);
    tensor(b, 0, {4}, 16, 16);
    cases.push_back({"RUN missing input", 5, b.b});
  }
  {
    Buf b = runHead(3);
    tensor(b, 0, {4}, 16, 16);
    tensor(b, 0, {}, 4, 4);
    tensor(b, 0, {4}, 16, 16);
    cases.push_back({"RUN extra input", 5, b.b});
  }
  {
    Buf b = runHead(2);
    tensor(b, 0, {4}, 16, 16);
    tensor(b, 0, {}, 4, 4);
    b.put<uint64_t>(0xdeadbeef);  // trailing garbage is ignored, not read past
    cases.push_back({"RUN trailing bytes", 5, b.b});
  }
  cases.push_back({"PROFILE truncated", 7, Buf().put<uint32_t>(id).b});
  cases.push_back({"PROFILE unknown model", 7, Buf().put<uint32_t>(999).put<uint32_t>(1).put<uint32_t>(0).b});
  cases.push_back({"TAKE_PROFILE empty", 8, {}});
  cases.push_back({"TAKE_PROFILE unknown model", 8, Buf().put<uint32_t>(999).b});
  cases.push_back({"GPU_INSPECT empty", 9, {}});
  cases.push_back({"MEMORY with payload", 10, Buf().put<uint32_t>(999).b});
  auto run2Head = [&](uint32_t n) {
    Buf b;
    b.put<uint32_t>(id).put<uint32_t>(0).put<uint32_t>(1).put<uint32_t>(1).put<uint32_t>(n);
    return b;
  };
  cases.push_back({"RUN2 truncated head", 11, Buf().put<uint32_t>(id).put<uint32_t>(0).b});
  cases.push_back({"RUN2 too many inputs", 11, run2Head(4096).b});
  cases.push_back({"RUN2 bad input kind", 11, run2Head(1).put<uint8_t>(9).b});
  cases.push_back({"RUN2 unknown blob", 11, run2Head(1).put<uint8_t>(1).str(std::string(64, 'c')).b});
  cases.push_back({"RUN2 bad sha", 11, run2Head(1).put<uint8_t>(1).str(std::string(64, 'Z')).b});
  cases.push_back({"RUN2 truncated sha", 11, run2Head(1).put<uint8_t>(1).str("abc").b});
  cases.push_back({"RUN2 model blob as a tensor", 11, run2Head(1).put<uint8_t>(1).str(goodSha).b});
  {
    Buf b = run2Head(2);  // generated: unsupported dtype
    b.put<uint8_t>(2).put<uint32_t>(1).put<uint32_t>(1).put<int64_t>(4).put<uint64_t>(1).put<uint32_t>(0).put<double>(0).put<double>(1);
    cases.push_back({"RUN2 generated f16", 11, b.b});
  }
  {
    Buf b = run2Head(2);  // generated: 2^40 elements
    b.put<uint8_t>(2).put<uint32_t>(0).put<uint32_t>(1).put<int64_t>(int64_t(1) << 40).put<uint64_t>(1).put<uint32_t>(0).put<double>(0).put<double>(1);
    cases.push_back({"RUN2 generated huge", 11, b.b});
  }
  {
    Buf b = run2Head(2);  // generated: NaN parameter
    b.put<uint8_t>(2).put<uint32_t>(0).put<uint32_t>(1).put<int64_t>(4).put<uint64_t>(1).put<uint32_t>(0).put<double>(NAN).put<double>(1);
    cases.push_back({"RUN2 generated NaN param", 11, b.b});
  }
  {
    Buf b = run2Head(2);  // generated: empty integer range
    b.put<uint8_t>(2).put<uint32_t>(7).put<uint32_t>(1).put<int64_t>(4).put<uint64_t>(1).put<uint32_t>(2).put<double>(5).put<double>(5);
    cases.push_back({"RUN2 generated empty int range", 11, b.b});
  }
  {
    Buf b = run2Head(2);  // valid inputs, comparison of a missing output
    b.put<uint8_t>(0);
    tensor(b, 0, {4}, 16, 16);
    b.put<uint8_t>(0);
    tensor(b, 0, {}, 4, 4);
    b.put<uint32_t>(1).put<uint32_t>(5).put<uint8_t>(3).put<uint32_t>(0).put<double>(0).put<double>(0);
    cases.push_back({"RUN2 compare missing output", 11, b.b});
  }
  {
    Buf b = run2Head(2);
    b.put<uint8_t>(0);
    tensor(b, 0, {4}, 16, 16);
    b.put<uint8_t>(0);
    tensor(b, 0, {}, 4, 4);
    b.put<uint32_t>(1).put<uint32_t>(0).put<uint8_t>(3).put<uint32_t>(0).put<double>(-1).put<double>(0);
    cases.push_back({"RUN2 negative tolerance", 11, b.b});
  }
  {
    Buf b = run2Head(2);
    b.put<uint8_t>(0);
    tensor(b, 0, {4}, 16, 16);
    b.put<uint8_t>(0);
    tensor(b, 0, {}, 4, 4);
    b.put<uint32_t>(1).put<uint32_t>(0).put<uint8_t>(7);
    cases.push_back({"RUN2 bad comparison kind", 11, b.b});
  }
  {
    Buf b = run2Head(2);
    b.put<uint8_t>(0);
    tensor(b, 0, {4}, 16, 16);
    b.put<uint8_t>(0);
    tensor(b, 0, {}, 4, 4);
    b.put<uint32_t>(5000);
    cases.push_back({"RUN2 too many comparisons", 11, b.b});
  }

  for (const Case& k : cases) {
    INFO(k.name);
    auto r = c.call(k.type, k.payload);
    REQUIRE(r);  // an error reply, not a dropped session
    if (std::string(k.name) == "RUN trailing bytes") {
      REQUIRE(r->ok());
    } else {
      INFO(r->text());
      REQUIRE_FALSE(r->ok());
      REQUIRE_FALSE(r->body.empty());  // a message says why
    }
    checkHealthy(c, id);
  }
  // The valid blob is still cached and loadable.
  auto has = c.call(MsgType::HasBlob, Buf().str(goodSha).b);
  REQUIRE(has);
  REQUIRE(has->body.at(0) == 1);
}

TEST_CASE("remote fuzz: random and mutated frames never crash the session", "[remote][fuzz]") {
  std::mt19937_64 rng(0x10C1F);  // fixed seed: failures reproduce
  auto conn = std::make_unique<Conn>();
  uint32_t id = loadModel(*conn);
  const std::vector<uint8_t> valid = runPayload(id);
  int dropped = 0, errors = 0, oks = 0;
  for (int iter = 0; iter < 3000; ++iter) {
    uint16_t type;
    std::vector<uint8_t> p;
    const int mode = static_cast<int>(rng() % 3);
    if (mode == 0) {  // random type + random bytes
      do type = static_cast<uint16_t>(rng() % 14);
      while (type == static_cast<uint16_t>(MsgType::Shutdown));
      p.resize(rng() % 200);
      for (auto& b : p) b = static_cast<uint8_t>(rng());
      if (p.size() >= 4 && (rng() & 1)) std::memcpy(p.data(), &id, 4);  // often aim at the real model
    } else {  // mutations of a valid RUN: bit flips, byte sets, truncation, extension
      type = static_cast<uint16_t>(MsgType::Run);
      p = valid;
      const int edits = 1 + static_cast<int>(rng() % 4);
      for (int e = 0; e < edits; ++e) {
        switch (rng() % 4) {
          case 0: p[rng() % p.size()] ^= static_cast<uint8_t>(1u << (rng() % 8)); break;
          case 1: p[rng() % p.size()] = static_cast<uint8_t>(rng()); break;
          case 2: p.resize(rng() % p.size() + 1); break;
          case 3: p.resize(p.size() + rng() % 32, static_cast<uint8_t>(rng())); break;
        }
      }
      // Keep warmup/iters small so a mutated count can't make one frame run for minutes.
      if (p.size() >= 12) {
        uint32_t small = static_cast<uint32_t>(rng() % 3);
        std::memcpy(p.data() + 4, &small, 4);
        std::memcpy(p.data() + 8, &small, 4);
      }
    }
    auto r = conn->call(type, p);
    if (!r) {  // only malformed headers drop a session, and we never send those here
      ++dropped;
      conn = std::make_unique<Conn>();
      id = loadModel(*conn);
      continue;
    }
    (r->ok() ? oks : errors)++;
  }
  INFO("ok " << oks << ", errors " << errors << ", dropped " << dropped);
  REQUIRE(dropped == 0);
  REQUIRE(errors > 1000);
  checkHealthy(*conn, id);
}

TEST_CASE("remote fuzz: corrupted model blobs are rejected or run safely", "[remote][fuzz]") {
  std::mt19937_64 rng(0x5EED);
  Conn c;
  const uint32_t good = loadModel(c);
  const std::vector<uint8_t> base = modelBytes();
  int loaded = 0, rejected = 0;
  for (int iter = 0; iter < 400; ++iter) {
    std::vector<uint8_t> m = base;
    const int edits = 1 + static_cast<int>(rng() % 3);
    for (int e = 0; e < edits; ++e) {
      switch (rng() % 4) {
        case 0: m[rng() % m.size()] ^= static_cast<uint8_t>(1u << (rng() % 8)); break;
        case 1: m[rng() % m.size()] = static_cast<uint8_t>(rng()); break;
        case 2: m.resize(rng() % m.size()); break;
        case 3: {  // a random 8-byte little-endian value (sizes, offsets, counts) somewhere
          if (m.size() < 8) break;
          uint64_t v = rng() >> (rng() % 64);
          std::memcpy(m.data() + rng() % (m.size() - 7), &v, 8);
          break;
        }
      }
    }
    if (m.empty()) m.push_back(0);
    std::string sha = putModel(c, m);
    auto r = c.call(MsgType::LoadModel, Buf().str(sha).b);
    REQUIRE(r);
    if (!r->ok()) {
      ++rejected;
      continue;
    }
    ++loaded;  // a mutation the loader + verifier accept must still be safe to run (errors are fine)
    uint32_t mid;
    std::memcpy(&mid, r->body.data(), 4);
    auto run = c.call(MsgType::Run, runPayload(mid));
    REQUIRE(run);
  }
  INFO("loaded " << loaded << ", rejected " << rejected);
  REQUIRE(rejected > 0);
  checkHealthy(c, good);
}

TEST_CASE("remote: RUN2 with blob, generated and inline inputs and on-device comparisons", "[remote]") {
  Conn c;
  const uint32_t id = loadModel(c);
  // x as a content-addressed tensor blob: the encoded tensor, stored by its sha256.
  Buf xt;
  xt.put<uint32_t>(0).put<uint32_t>(1).put<int64_t>(4).put<uint64_t>(16);
  for (int i = 0; i < 4; ++i) xt.put<float>(float(i + 1));
  const std::string xsha = putModel(c, xt.b);
  // The expected output (x + x) * x as a reference blob, and a wrong one.
  Buf rt, bad;
  for (Buf* b : {&rt, &bad}) b->put<uint32_t>(0).put<uint32_t>(1).put<int64_t>(4).put<uint64_t>(16);
  for (int i = 0; i < 4; ++i) rt.put<float>(2.f * float(i + 1) * float(i + 1)), bad.put<float>(float(i == 2 ? 99 : 2 * (i + 1) * (i + 1)));
  const std::string rsha = putModel(c, rt.b), bsha = putModel(c, bad.b);

  Buf b;
  b.put<uint32_t>(id).put<uint32_t>(1).put<uint32_t>(2).put<uint32_t>(1).put<uint32_t>(2);
  b.put<uint8_t>(1).str(xsha);
  b.put<uint8_t>(0).put<uint32_t>(0).put<uint32_t>(0).put<uint64_t>(4).put<float>(0.5f);
  b.put<uint32_t>(3);
  b.put<uint32_t>(0).put<uint8_t>(1).str(rsha).put<double>(0).put<double>(0);
  b.put<uint32_t>(0).put<uint8_t>(1).str(bsha).put<double>(0).put<double>(1e-3);
  b.put<uint32_t>(0).put<uint8_t>(3).put<uint32_t>(0).put<double>(0).put<double>(0);
  auto r = c.call(MsgType::Run2, b.b);
  REQUIRE(r);
  INFO(r->text());
  REQUIRE(r->ok());
  const uint8_t* p = r->body.data();
  auto u32 = [&] { uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; };
  auto u64 = [&] { uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; };
  auto f64 = [&] { double v; std::memcpy(&v, p, 8); p += 8; return v; };
  REQUIRE(u32() == 1);  // outputs returned (flag bit 0)
  REQUIRE(u32() == 0);  // dtype
  REQUIRE(u32() == 1);
  p += 8;               // shape
  REQUIRE(u64() == 16);
  const float* o = reinterpret_cast<const float*>(p);
  for (int i = 0; i < 4; ++i) REQUIRE(o[i] == 2.f * float(i + 1) * float(i + 1));
  p += 16;
  REQUIRE(u32() == 3);
  for (int k = 0; k < 3; ++k) {
    REQUIRE(u32() == 0);
    REQUIRE(u64() == 4);  // count
    REQUIRE(u64() == 0);  // nonfinite
    const uint64_t mism = u64();
    int64_t worst;
    std::memcpy(&worst, p, 8);
    p += 8;
    const double maxErr = f64(), maxRef = f64();
    f64();
    f64();
    if (k == 1) {
      REQUIRE(mism == 1);
      REQUIRE(worst == 2);
      REQUIRE(maxErr == 99.0 - 18.0);
    } else {
      REQUIRE(mism == 0);
      REQUIRE(maxErr == 0);
    }
    REQUIRE(maxRef > 0);
  }
  REQUIRE(u32() == 2);  // iters timed

  // Generated input: deterministic per seed (two runs agree), different across seeds, roughly N(0, 1).
  auto gen = [&](uint64_t seed) {
    Buf g;
    g.put<uint32_t>(id).put<uint32_t>(0).put<uint32_t>(1).put<uint32_t>(1).put<uint32_t>(2);
    g.put<uint8_t>(2).put<uint32_t>(0).put<uint32_t>(1).put<int64_t>(4096).put<uint64_t>(seed).put<uint32_t>(0).put<double>(0).put<double>(1);
    g.put<uint8_t>(2).put<uint32_t>(0).put<uint32_t>(0).put<uint64_t>(seed).put<uint32_t>(1).put<double>(0).put<double>(1);
    g.put<uint32_t>(0);
    auto rr = c.call(MsgType::Run2, g.b);
    REQUIRE(rr);
    INFO(rr->text());
    REQUIRE(rr->ok());
    const float* out = reinterpret_cast<const float*>(rr->body.data() + 4 + 4 + 4 + 8 + 8);
    return std::vector<float>(out, out + 4096);
  };
  auto a1 = gen(7), a2 = gen(7), b2 = gen(8);
  REQUIRE(a1 == a2);
  REQUIRE(a1 != b2);
  double mean = 0, var = 0;
  for (float v : a1) {
    const double x = std::sqrt(v / 2.0);  // out = 2 x^2
    mean += x;
  }
  for (float v : a1) var += v / 2.0;  // E[x^2] = 1
  REQUIRE(std::fabs(var / 4096 - 1.0) < 0.1);
  REQUIRE(mean > 0);
}
