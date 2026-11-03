// Multi-stream AMX / SME re-verification bench (amx_bench.cpp), shared by `lokinfer-bench-conv amx` and LokiHost
// `--bench-amx`.
#pragma once

#include <vector>

namespace lk::bench {

// Prints AMX_BENCH lines: per GEMM shape, 1 single-threaded stream (QoS user-interactive and background), one call
// with Accelerate's threading, and each N in `streamCounts` concurrent single-threaded streams.
void runAmxBench(const std::vector<int>& streamCounts);

}  // namespace lk::bench
