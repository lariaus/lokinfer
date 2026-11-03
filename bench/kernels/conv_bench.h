// Conv proof benchmarks (M2) shared by the Mac CLI (lokinfer-bench-conv) and
// the iOS test host (`--bench conv`). Kokoro rand510 generator shapes.
#pragma once

#include <string>

namespace lk::bench {

// CPU: time-tiled im2col over N single-threaded Accelerate streams (the M2 CPU winner).
// GPU: vendored MLX steel implicit-GEMM conv, channel-last. Prints one line per case
// and checks every result against a direct computation. `metallib`: path of the
// bench metallib (conv1d_steel.metal).
void runConvBench(const std::string& metallib, bool cpu, bool gpu);

// CPU tuning sweep: tile x streams (raw threads) and the runtime thread pool.
void runConvTune();

}  // namespace lk::bench
