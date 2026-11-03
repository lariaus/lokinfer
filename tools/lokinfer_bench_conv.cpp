// lokinfer-bench-conv: M2 conv proof on this Mac (same code the iOS host runs
// with `--bench conv`). Usage: lokinfer-bench-conv [cpu|gpu|tune|amx [N ...]] [metallib]
//   amx: multi-stream Accelerate sgemm re-verification (bench/kernels/amx_bench.cpp); N = stream counts (default
//        2 3 4 6 8)
#include <cstring>
#include <string>

#include <cstdlib>
#include <vector>

#include "../bench/kernels/amx_bench.h"
#include "../bench/kernels/conv_bench.h"

int main(int argc, char** argv) {
  bool cpu = true, gpu = true;
  std::string lib = LOKINFER_BENCH_METALLIB_PATH;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "cpu")) gpu = false;
    else if (!std::strcmp(argv[i], "gpu")) cpu = false;
    else if (!std::strcmp(argv[i], "tune")) return lk::bench::runConvTune(), 0;
    else if (!std::strcmp(argv[i], "amx")) {
      std::vector<int> n;
      for (int j = i + 1; j < argc; ++j) n.push_back(std::atoi(argv[j]));
      if (n.empty()) n = {2, 3, 4, 6, 8};
      return lk::bench::runAmxBench(n), 0;
    }
    else lib = argv[i];
  }
  lk::bench::runConvBench(lib, cpu, gpu);
  return 0;
}
