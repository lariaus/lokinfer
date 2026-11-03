// Compiler passes (see docs/impl-notes/design-compiler-passes.md).
#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lokinfer/ir/ir.h"
#include "lokinfer/ir/status.h"

namespace lk {

using PassFn = Status (*)(Model&);

struct PassInfo {
  const char* name;
  const char* description;
  PassFn fn;
};

std::span<const PassInfo> allPasses();
const PassInfo* findPass(std::string_view name);

// The default pipeline: dce, bufferize, plan-memory, hoist-init.
const std::vector<std::string>& defaultPipeline();
// Runs the passes in order. Verification is not a pass: the caller verifies the model when it loads/builds it, and
// with verifyEach (the default) runPasses re-verifies after every pass, naming the pass that broke the IR.
// With `stats`, appends one PassStats per pass (wall time of the pass and of its verification, op counts after it).
struct PassStats {
  std::string name;
  double passMs = 0, verifyMs = 0;
  size_t initOps = 0, execOps = 0;
  std::vector<std::string> notes;  // pass-specific statistics (passNote), e.g. plan-memory's arena vs lower bound
};
// For passes: true while runPasses collects stats; passNote then appends a line to the running pass's notes.
bool passNotesEnabled();
void passNote(std::string line);
Status runPasses(Model& m, std::span<const std::string> names, bool verifyEach = true,
                 std::vector<PassStats>* stats = nullptr);

// Data of an external constant (internalize).
struct ExternalData {
  std::string name;
  const uint8_t* bytes = nullptr;
  size_t size = 0;
  std::shared_ptr<const void> owner;  // keeps `bytes` alive (copy = false)
};
// const.external -> const.internal for every external constant (src/compiler/internalize.cpp). copy = true: the
// model owns a 16 KiB-aligned copy (later writes to the caller's tensor no longer reach it); false: it references
// `bytes` through `owner`. Errors if an external constant has no data. `count` receives the number of ops changed.
Status internalize(Model& m, std::span<const ExternalData> data, bool copy, size_t* count = nullptr);

Status dce(Model& m);
Status bufferize(Model& m);
Status planMemory(Model& m);
Status hoistInit(Model& m);

}  // namespace lk
