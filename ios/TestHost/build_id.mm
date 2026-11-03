// Build id of the running LokiHost for benchmark records (perf/kernels.sqlite `runs.build_id`, docs/benchmarking.md):
// the main executable's Mach-O LC_UUID, which changes on every link. First 12 hex digits.
#include <mach-o/dyld.h>
#include <mach-o/loader.h>

#include <cstdio>
#include <cstring>
#include <string>

std::string lokihost_build_id() {
  const auto* h = reinterpret_cast<const mach_header_64*>(_dyld_get_image_header(0));
  if (!h) return "unknown";
  const auto* lc = reinterpret_cast<const load_command*>(h + 1);
  for (uint32_t i = 0; i < h->ncmds; ++i) {
    if (lc->cmd == LC_UUID) {
      const auto* u = reinterpret_cast<const uuid_command*>(lc);
      char buf[33];
      for (int k = 0; k < 16; ++k) std::snprintf(buf + 2 * k, 3, "%02x", u->uuid[k]);
      return std::string(buf, 12);
    }
    lc = reinterpret_cast<const load_command*>(reinterpret_cast<const char*>(lc) + lc->cmdsize);
  }
  return "unknown";
}
