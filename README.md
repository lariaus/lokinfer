# lokinfer

An ML inference framework for Apple silicon (macOS and iOS), on the CPU (NEON, SME2, Accelerate) and on the GPU
(Metal 4). Its main target is the **Kokoro-82M** text-to-speech model, which it runs 1.7-3.6x faster than the
fastest other framework tested, on every device and input length measured.

- A small MLIR-style IR with a verifier, a compiler pipeline (DCE, bufferization, memory planning, hoisting of
  weight preparation) and a runtime that runs the compiled model with no copies of inputs, outputs or weights.
- Devices are explicit in the graph: a GPU model is GPU end to end, with Metal kernels written for it.
- A Python layer (nanobind) to build, compile, run, save and load models; a C++ API to load and run saved models
  in an app.
- f32 everywhere: results are compared with PyTorch at f32.

Version **0.0.1** (see [Versions](#versions)).

## Performance

Median time of one forward pass, f32, against the fastest baseline measured on the same device (PyTorch eager /
`torch.compile` / MPS, MLX, MLX Swift, ExecuTorch, ONNX Runtime, Core ML). Mac: M2 Max. iPhone: iPhone 18 Pro Max,
iOS 27.

| Model, case | Device | lokinfer | Fastest baseline |
|---|---|---|---|
| Kokoro-82M, 4 s sentence (59 tokens) | Mac CPU | 107 ms | 324 ms (torch.compile) |
| | Mac GPU | 49 ms | 108 ms (MLX) |
| | iPhone CPU | 108 ms | 289 ms (ExecuTorch XNNPACK) |
| | iPhone GPU | 66 ms | 210 ms (MLX Swift) |
| Kokoro-82M, 26 s paragraph (487 tokens) | Mac CPU | 641 ms | 1909 ms (torch.compile) |
| | Mac GPU | 261 ms | 775 ms (MLX Swift) |
| | iPhone CPU | 912 ms | 2619 ms (ExecuTorch XNNPACK) |
| | iPhone GPU | 617 ms | 1633 ms (MLX Swift) |
| GPT-2 124M, decode 1 token (512-token cache) | Mac CPU | 8.1 ms | 19.8 ms (torch eager) |
| | Mac GPU | 2.0 ms | 3.2 ms (torch.compile MPS) |
| | iPhone CPU | 5.5 ms | 8.7 ms (ExecuTorch XNNPACK) |
| | iPhone GPU | 5.6 ms | 6.0 ms (MLX Swift) |

lokinfer is fastest in every Kokoro and GPT-2 configuration measured (16 Kokoro cases, 3 GPT-2 cases, 4 devices).
The other models (MNIST MLP, ResNet-18, MobileNetV2, DCGAN) are at or near the fastest baseline. Every case and baseline is in
the perf dashboard (`perf/index.html`, data in `perf/results.json`; see [docs/perf-dashboard.md](docs/perf-dashboard.md)).

Kokoro audio quality was checked end to end against the reference Python implementation on 225 texts (one word to
80 s, 10 voices): same durations, and the same spectral similarity as two runs of the reference implementation
([tests/e2e/kokoro/](tests/e2e/kokoro/README.md)).

## Requirements

- macOS 27 / iOS 27 on Apple silicon (the GPU path uses Metal 4).
- Xcode with the Metal toolchain, CMake 3.28+, Ninja.
- Python 3.14 and [uv](https://docs.astral.sh/uv/) for the Python layer, the tests and the benchmarks.

## Building

C++ library, tools and unit tests:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0
cmake --build build
build/lokinfer_unit_tests
```

iOS (static libraries for an app; `scripts/ios_deploy.sh` builds and runs the on-device test host):

```sh
cmake -S . -B build/ios -G Ninja -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos \
      -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0
cmake --build build/ios
```

Python (editable; C++ changes are rebuilt on `import lokinfer`):

```sh
uv venv --python 3.14 && source .venv/bin/activate
uv pip install scikit-build-core==1.1.1 nanobind==3.1.0 numpy torch==2.14.1 lit filecheck
uv pip install --no-build-isolation -e .
```

The GPU kernels are compiled into `lokinfer.metallib`, which ships with the library (the Python package finds it by
itself; C++ apps set `LOKINFER_METALLIB`, see below).

## Using it

### Python: build, compile, run, save

```python
import lokinfer as lk
from lokinfer import ops

w, b = lk.randn(784, 128, seed=1).to("gpu"), lk.randn(128, seed=2).to("gpu")  # constants live on their device
model = lk.compile(lambda x: ops.gemm_bias_relu_f32_gpu(ops.to_gpu(x), w, b),  # device boundaries are explicit
                   inputs=[lk.randn(1, 784)])
y = model(lk.randn(1, 784)).numpy()       # synchronous; model.submit(...) runs it on a worker thread
model.save("model.lkm", bake_init=True)   # self-contained file, weights included
m2 = lk.load("model.lkm")                 # weights are mmapped, not copied
```

Kokoro-82M is built from its PyTorch weights by `models/kokoro/model.py` (`build(fold_weights(...), gpu=True)` for
the GPU model); see [docs/models/kokoro.md](docs/models/kokoro.md) and the end-to-end example in
[tests/e2e/kokoro/](tests/e2e/kokoro/README.md). The whole Python API: [docs/python-api.md](docs/python-api.md).

### C++: load a saved model and run it

```cpp
#include "lokinfer/ir/binary.h"
#include "lokinfer/ir/verify.h"
#include "lokinfer/runtime/runtime.h"

auto model = lk::readModelFile("model.lkm");                     // weights are mmapped
if (!model) return model.status();
LK_RETURN_IF_ERROR(lk::verifyModel(**model, lk::VerifyOptions{true}));  // never run unverified IR
auto exe = lk::rt::Executable::create(std::shared_ptr<const lk::Model>(model.take()), {});
if (!exe) return exe.status();
std::vector<lk::rt::TensorView> inputs = {{x.data(), lk::DType::F32, {1, 784}}};
std::vector<lk::rt::OwnedTensor> outputs;
LK_RETURN_IF_ERROR((*exe)->run(inputs, outputs));                // outputs[0].data(), outputs[0].shape
```

Link `lokinfer_runtime`. Kernels register themselves. GPU models load their kernels from `lokinfer.metallib`: set
`LOKINFER_METALLIB` to its path before the first GPU model is created (an iOS app ships it in its bundle, as
`ios/TestHost/main.mm` does). One executable runs one call at a time.

## Tests

The iPhone, baseline and end-to-end tooling reads machine-local settings (device UDID, signing team, paths of
other checkouts) from a git-ignored `lokinfer.local.json`; copy `lokinfer.local.example.json` and see
[docs/benchmarking.md](docs/benchmarking.md#local-configuration). The library itself needs none of it.

```sh
build/lokinfer_unit_tests       # C++ unit tests (Catch2)
lit -q tests/lit                # IR / pass tests (FileCheck)
python -m pytest -q             # kernels, models, Python API, perf-dashboard goals
scripts/ios_deploy.sh           # the C++ tests on a connected iPhone
```

Model tests compare against the PyTorch reference at f32; Kokoro also has free-running (random-phase) audio tests on
real sentences in both app voices and extra voices.

## Repository layout

| Path | What |
|---|---|
| `include/`, `src/` | IR, verifier, compiler passes, runtime (CPU thread pool, Metal 4 GPU context), remote executor |
| `kernels/` | CPU (NEON, SME2, Accelerate) and GPU (Metal) kernels |
| `python/` | the Python package and its nanobind extension |
| `models/` | model graphs: Kokoro-82M, GPT-2, ResNet-18, MobileNetV2, DCGAN, MNIST |
| `tests/` | unit, lit, Python, model, perf-goal and end-to-end tests |
| `bench/`, `perf/` | benchmarks, baselines and the perf dashboard data |
| `ios/` | the iOS test host (runs the C++ tests on the device, then serves as a remote executor) |
| `tools/` | `lokinfer-opt` (run passes on IR text), `lokinfer-serve` (remote executor on the Mac) |
| `docs/` | design and reference docs: [compiler](docs/compiler.md), [runtime](docs/runtime.md), [verifier](docs/verifier.md), [profiling](docs/profiling.md), [benchmarking](docs/benchmarking.md), models |

## Versions

The version is in `VERSION` (CMake reads it; `pyproject.toml` must match) and each release is a git tag
`v<version>`. To depend on a fixed release from CMake:

```cmake
include(FetchContent)
FetchContent_Declare(lokinfer GIT_REPOSITORY <this repository's URL> GIT_TAG v0.0.1)
set(LOKINFER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(lokinfer)
target_link_libraries(my_app PRIVATE lokinfer_runtime)
```

C++ code sees the version as the `LOKINFER_VERSION` macro, Python as `lokinfer.__version__`.

As a dependency, lokinfer builds only its libraries and `lokinfer.metallib` (`LOKINFER_BUILD_DEV_TOOLS` defaults to
off unless lokinfer is the top-level project), on macOS and iOS.

## License

MIT, see [LICENSE](LICENSE). `third_party/mlx_steel` is from [MLX](https://github.com/ml-explore/mlx) (MIT,
Apple Inc.), license included there.
