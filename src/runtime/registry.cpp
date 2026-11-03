#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>

#include "lokinfer/runtime/kernel.h"

namespace lk::rt {

// Per-family registration functions (kernels/**).
void registerCpuElementwiseKernels();
void registerCpuGemmKernels();
void registerCpuKokoroConvKernels();
void registerCpuKokoroConvFusedKernels();
void registerCpuKokoroLstmKernels();
void registerCpuKokoroMiscKernels();
void registerCpuKokoroNormKernels();
void registerCpuKokoroSignalKernels();
void registerCpuVisionKernels();
void registerCpuDecodeKernels();
void registerGpuKernels();
void registerGpuKokoroKernels();
void registerGpuMpsKernels();
void registerGpuTransformerKernels();
void registerGpuWinogradKernels();

namespace {

std::unordered_map<std::string, KernelEntry>& table() {
  static std::unordered_map<std::string, KernelEntry> t;
  return t;
}

}  // namespace

void registerKernel(const KernelEntry& e) { table()[e.name] = e; }

void registerBuiltinKernels() {
  static std::once_flag once;
  std::call_once(once, [] {
    registerCpuElementwiseKernels();
    registerCpuGemmKernels();
    registerCpuKokoroConvKernels();
    registerCpuKokoroConvFusedKernels();
    registerCpuKokoroLstmKernels();
    registerCpuKokoroMiscKernels();
    registerCpuKokoroNormKernels();
    registerCpuKokoroSignalKernels();
    registerCpuVisionKernels();
    registerCpuDecodeKernels();
    registerGpuKernels();
    registerGpuKokoroKernels();
    registerGpuMpsKernels();
    registerGpuTransformerKernels();
    registerGpuWinogradKernels();
  });
}

std::vector<std::string> registeredKernels() {
  registerBuiltinKernels();
  std::vector<std::string> names;
  for (auto& [n, e] : table()) names.push_back(n);
  std::sort(names.begin(), names.end());
  return names;
}

const KernelEntry* findKernel(std::string_view name) {
  registerBuiltinKernels();
  auto it = table().find(std::string(name));
  return it == table().end() ? nullptr : &it->second;
}

}  // namespace lk::rt
