# lokinfer: Infrastructure Research (build, bindings, tests, iOS, benchmarks)

Research date: 2026-11-01. Machine: M2 Max, macOS 27.0.1, Xcode 27.0 (27A266a).
The phone (iPhone 18 Pro Max, iOS 27.0, UDID `<udid>`) was only
**queried** (`devicectl list/info`, ping). Nothing was installed or launched on it.

Labels used below:
- **[verified]**: run on this machine during the research.
- **[docs]**: taken from upstream docs or help text, not run end to end.
- **[unverified]**: inference that still needs a test on the device.

The smoke project lived in a local scratch directory (not in the repo). It is
throwaway. Copy the patterns below into lokinfer, not the files.

---

## 0. Toolchain baseline [verified]

| Tool | Version | Notes |
|---|---|---|
| Apple clang (`xcrun clang --version`) | 21.0.0 (clang-2100.3.34.2) | target arm64-apple-darwin27.0.0 |
| Xcode | 27.0 (27A266a) | iphoneos SDK 27.0, macosx SDK 27.0 |
| Metal compiler | metalfe-32023.921.6 | **was NOT installed.** See below |
| CMake | 4.4.3 (Homebrew) | |
| Ninja | 1.13.2 (Homebrew) | |
| uv | 0.9.7 | |
| Python | 3.14.10 (Homebrew, found by `uv venv --python 3.14`) | |

**Gotcha 1: the Metal Toolchain is a separate download in Xcode 26 and later.** Before the
download, `xcrun metal` failed with `cannot execute tool 'metal' due to missing Metal Toolchain`.
The fix is `xcodebuild -downloadComponent MetalToolchain` (839 MB, installed during this research).
Check it with `xcodebuild -showComponent MetalToolchain` (expect `Status: installed`). The
toolchain mounts as a cryptex under `/private/var/run/com.apple.security.cryptexd/...`.
The top-level CMake should fail early with a clear message when `xcrun -sdk macosx metal --version`
fails.

MSL standards accepted by this compiler: `metal3.0` to `metal3.2`, `metal4.0`, `metal4.1`
(`metal4.2` is rejected). **Use `-std=metal4.1`.**

## 1. Project layout and CMake

### 1.1 Pinned dependencies

These are URL tarballs with a SHA256 hash. No git is involved. Every hash was verified by
FetchContent during the smoke build.

| Dep | Version | URL | SHA256 |
|---|---|---|---|
| nanobind | 3.1.0 (2026-09-18) | `https://files.pythonhosted.org/packages/c7/1f/e28fae715401e8f57400e72aeaff4fde12046d31828164f34f61f0816da8/nanobind-3.1.0.tar.gz` (PyPI sdist) | `6661e3dc1434d78781ccf7913442d90361dab395c3424efadb63bbb6ddea7ea3` |
| Catch2 | v3.16.1 (2026-10-08) | `https://github.com/catchorg/Catch2/archive/refs/tags/v3.16.1.tar.gz` | `d16bd1b8b2364918bd472cb216394acb93bde6784890df0d91bccfb3f28d777d` |
| Google Benchmark | v1.10.0 (2026-10-14) | `https://github.com/google/benchmark/archive/refs/tags/v1.10.0.tar.gz` | `acc5c7c0fbed855c4aaa2d7c3c36fb82b0c15c28f4e151e3f679c2fc0aa404af` |
| metal-cpp | tag `release/metal-cpp_macOS27.0.1_iOS27.0.1` | `https://github.com/apple/metal-cpp/archive/refs/tags/release/metal-cpp_macOS27.0.1_iOS27.0.1.tar.gz` | `3a2f456c5373748e1b8de0896d1dbe53d6e0a1d3b8b9e5235c1a8abae6f373f5` |

Python side (PyPI, all have cp314 macOS arm64 wheels where relevant): `nanobind==3.1.0`,
`scikit-build-core==1.1.1`, `lit==23.1.4`, `filecheck==1.0.6`, `numpy==2.5.4`, `torch==2.14.1`
(pinned by design-decisions; 2.15.0 is the newest).

Notes:
- **Gotcha 2:** the nanobind **GitHub archive tarball does not include the `ext/robin_map`
  submodule** (verified: 0 files). Use the **PyPI sdist** (which includes it) or the pip package's
  `python -m nanobind --cmake_dir`. Both paths were verified.
- metal-cpp now has an official GitHub repo, `apple/metal-cpp`, with release tags. The
  developer.apple.com page no longer links ZIPs directly. The repo has no CMakeLists, so wrap it
  as an INTERFACE target. It includes the Metal 4 headers (`MTL4CommandQueue.hpp`, etc.).
- **Gotcha 3:** do **not** share one `FETCHCONTENT_BASE_DIR` between the macOS and iOS build
  dirs. The `<dep>-build` subdirs land there and collide across platforms (seen in the smoke
  test: the iOS Catch2 build went into the shared dir). Keep the default per-build-dir `_deps`.
  The downloads are small (about 3 MB total). To work offline, set `FETCHCONTENT_SOURCE_DIR_<NAME>`
  per dep, which shares sources only.

### 1.2 Recommended layout

```
lokinfer/
  CMakeLists.txt            # top-level; options LOKI_BUILD_{TESTS,BENCH,PYTHON}
  cmake/LokiMetal.cmake     # loki_add_metallib()
  cmake/Deps.cmake          # all FetchContent_Declare with URL+SHA256
  include/lokinfer/...      # public C++ headers (no metal-cpp in public headers)
  src/ir/ src/compiler/ src/runtime/{cpu,gpu}/ src/kernels/{cpu,gpu}/   # -fno-exceptions
  kernels/metal/*.metal     # MSL -> one lokinfer.metallib per platform
  python/lokinfer/          # pure-python package (+ _lokinfer*.so installed beside it)
  python/ext/*.cpp          # nanobind binding layer (exceptions ON here only)
  tools/lokinfer-opt/       # pass driver for lit tests
  tests/cpp/                # Catch2
  tests/lit/                # lit.cfg.py + *.lir
  tests/py/                 # pytest (graph correctness, remote executor client)
  bench/                    # Google Benchmark kernels + scripts (db, compare)
  ios/TestHost/             # main.mm, Info.plist.in (CMake Xcode-generator target)
  pyproject.toml            # scikit-build-core
```

### 1.3 Building for macOS and iOS [verified]

- **macOS dev builds use Ninja:**
  `cmake -S . -B build/mac -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0`.
  Configure took 25 s on a cold cache (downloads included); a full build took 33 s. With no
  changes, `ninja` reports no work to do.
- **iOS static libs and tests with Ninja also work:**
  `cmake -S . -B build/ios -G Ninja -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_SYSROOT=iphoneos -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0`.
  This is the same pattern english-practice's `build-xcode.sh` uses. `LC_BUILD_VERSION` comes out
  as platform 2 (iOS), minos 27.0. Note that CMake sets `MACOSX_BUNDLE` on by default for iOS, so
  every `add_executable` becomes an unsigned `.app`.
- **iOS host app: use the CMake Xcode generator on the same tree**
  (`cmake -G Xcode -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0`). It handled
  FetchContent, the `.metal` custom commands (emitted as script phases) and the metallib as a
  bundle resource. `xcodebuild` built it in 17 s. This is what gives **automatic signing**:

  ```cmake
  set_target_properties(LokiHost PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST ${CMAKE_SOURCE_DIR}/ios/TestHost/Info.plist.in
    MACOSX_BUNDLE_GUI_IDENTIFIER <bundle-id>
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER <bundle-id>
    XCODE_ATTRIBUTE_DEVELOPMENT_TEAM <team-id>
    XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
    XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Apple Development"
    XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET 27.0)
  set_source_files_properties(${METALLIB} PROPERTIES GENERATED TRUE MACOSX_PACKAGE_LOCATION Resources)
  ```

  `xcodebuild -project build/ios-xcode/lokinfer.xcodeproj -scheme LokiHost -configuration Release -destination 'id=<udid>' -allowProvisioningUpdates build`
  produced a correctly signed app ("iOS Team Provisioning Profile: <bundle-id>",
  `get-task-allow=true`). **The App ID `<bundle-id>` is now registered on team
  <team-id>.** Reuse that ID; don't create more.
- **Gotcha 4: team <team-id> is a free Personal Team.** The generated profile expires
  2026-11-08 (7 days) and covers 1 device. Free teams also limit new App IDs (about 10 per 7 days)
  and installed sideloaded apps (3). This explains the recurring weekly "provisioning trust" issue.
  **Use exactly one bundle ID for the whole test host.** Don't make
  per-test apps.
- **Ninja-only alternative (fallback) [verified, sign step only]:** a Ninja-built `.app` can be
  signed by hand. Embed the Xcode-made profile as `embedded.mobileprovision`, extract its
  `Entitlements`, and run `codesign --force --sign <SHA1> --entitlements ent.plist app`.
  `codesign --verify --strict` passes. **Gotcha 5:** the keychain holds two identities with the
  same name ("Apple Development: <apple-id> (<cert-id>)"), so a name-based
  `--sign` is ambiguous. Use the SHA1 (`<SHA1>`) or let Xcode sign. Profile renewal still
  needs `xcodebuild -allowProvisioningUpdates` weekly, so the Xcode-generator path stays the
  default.
- **Gotcha 6 (2026-11-01): Xcode auto-update → hidden first-launch hang.** After Xcode updated
  to 27.2, `xcrun devicectl …` started `xcodebuild -runFirstLaunch`, which needs sudo and hangs
  forever without a terminal (no output, no child process). `xcodebuild -checkFirstLaunchStatus`
  exits with 69 while setup is pending. Fix: the user runs `sudo xcodebuild -runFirstLaunch` (any
  directory, about 1 min), then the Metal Toolchain may need `xcodebuild -downloadComponent MetalToolchain`.
  While any iPhone step is blocked, keep doing other work, retry every 5 min, and re-ask.
- **Recommendation:** Ninja everywhere (mac dev, python ext, iOS lib CI), plus one Xcode-generator
  build dir only for the iOS host app.
- In the Xcode generator, `CMAKE_BUILD_TYPE` is empty (multi-config). Use generator expressions
  (`$<IF:$<CONFIG:Debug>,-O0,-O3>`) for metal flags, not `if(CMAKE_BUILD_TYPE ...)`.
- UIKit warnings on iOS 27: `UIScreen.mainScreen` and `UIWindow initWithFrame:` are deprecated.
  Use a scene delegate (`UIWindowScene`) in the real host app. Add `UIRequiresFullScreen` or all
  orientations to silence the orientation warning.

### 1.4 `.metal` to `.air` to `.metallib` [verified, macOS and iOS]

```cmake
# cmake/LokiMetal.cmake
set(LOKI_METAL_STD "metal4.1" CACHE STRING "")
option(LOKI_METAL_DEBUG_INFO "embed sources+line tables (xctrace/Xcode shader profiling)" ON)
function(loki_add_metallib target)
  cmake_parse_arguments(A "" "OUTPUT" "SOURCES;INCLUDE_DIRS" ${ARGN})
  # LOKI_METAL_SDK = macosx|iphoneos ; LOKI_METAL_MINVER = -mmacosx-version-min=27.0|-mios-version-min=27.0
  set(flags -std=${LOKI_METAL_STD} ${LOKI_METAL_MINVER} -Wall "$<IF:$<CONFIG:Debug>,-O0,-O3>")
  if(LOKI_METAL_DEBUG_INFO)
    list(APPEND flags -frecord-sources -gline-tables-only)
  endif()
  foreach(d IN LISTS A_INCLUDE_DIRS)
    list(APPEND flags -I${d})
  endforeach()
  set(airs)
  foreach(src IN LISTS A_SOURCES)
    get_filename_component(abs ${src} ABSOLUTE)
    get_filename_component(name ${src} NAME_WE)
    set(air ${CMAKE_CURRENT_BINARY_DIR}/metal/${LOKI_METAL_SDK}/${name}.air)
    add_custom_command(OUTPUT ${air}
      COMMAND xcrun -sdk ${LOKI_METAL_SDK} metal ${flags} -MMD -MF ${air}.d -c ${abs} -o ${air}
      DEPENDS ${abs} DEPFILE ${air}.d VERBATIM
      COMMENT "MSL ${name}.metal")
    list(APPEND airs ${air})
  endforeach()
  add_custom_command(OUTPUT ${A_OUTPUT}
    COMMAND xcrun -sdk ${LOKI_METAL_SDK} metallib ${airs} -o ${A_OUTPUT}
    DEPENDS ${airs} VERBATIM)
  add_custom_target(${target} ALL DEPENDS ${A_OUTPUT})
endfunction()
```

- The `-MMD -MF` plus `DEPFILE` pair works with Ninja. Touching a header included by `add.metal`
  rebuilt only `add.air` and the metallib.
- One `.air` per `.metal` gives incremental and parallel builds. Link everything into one
  `lokinfer.metallib` per platform. `xcrun metal-nm lib.metallib` lists the kernels.
- iOS uses the same commands with `-sdk iphoneos -mios-version-min=27.0`. The macOS and iOS
  metallibs are different files (`file` reports "MetalLib executable (MacOS)" vs "MetalLib
  executable").
- Runtime loading: macOS dev builds use the compiled-in absolute path (or env override). The
  Python package uses a path next to the `.so` (see 1.5). iOS uses
  `[[NSBundle mainBundle] pathForResource:@"lokinfer" ofType:@"metallib"]`, passed down to C++ as a
  string. Then call `MTL::Device::newLibrary(NS::URL*, NS::Error**)`.
- `-frecord-sources -gline-tables-only` let Metal System Trace and Xcode map shader cost to MSL
  lines. Keep them on, and benchmark once with them off to confirm there is no runtime cost.

### 1.5 nanobind extension and uv [verified]

**Use scikit-build-core** as the build backend, with editable installs.

```toml
[build-system]
requires = ["scikit-build-core==1.1.1", "nanobind==3.1.0"]
build-backend = "scikit_build_core.build"
[project]
name = "lokinfer"
requires-python = ">=3.14"
[tool.scikit-build]
minimum-version = "build-system.requires"
build-dir = "build/py-{wheel_tag}"     # persistent -> incremental rebuilds
cmake.build-type = "Release"
wheel.packages = ["python/lokinfer"]
editable.mode = "redirect"
editable.rebuild = true                # rebuild C++ on `import lokinfer`
[tool.scikit-build.cmake.define]
LOKI_BUILD_PYTHON = "ON"
LOKI_BUILD_TESTS = "OFF"
LOKI_BUILD_BENCH = "OFF"
CMAKE_OSX_DEPLOYMENT_TARGET = "27.0"
```

In CMake, the pip path is `execute_process(${Python_EXECUTABLE} -m nanobind --cmake_dir)` followed
by `find_package(nanobind CONFIG)`. The FetchContent path is the PyPI sdist plus
`include(${nanobind_SOURCE_DIR}/cmake/nanobind-config.cmake)`. Both work. Then:
`nanobind_add_module(_lokinfer NB_STATIC python/ext/module.cpp)`, and `install(TARGETS ...)` plus
`install(FILES lokinfer.metallib DESTINATION lokinfer)`.

The workflow (all verified):
```
uv venv --python 3.14 .venv
uv pip install scikit-build-core==1.1.1 nanobind==3.1.0 numpy torch==2.14.1 lit filecheck
uv pip install --no-build-isolation -e .     # ~10 s first time
python -c "import lokinfer"                  # edits to .cpp are rebuilt on import (~1.3 s for 1 TU)
```
- `editable.rebuild=true` needs `--no-build-isolation`, so the build deps must be installed in the
  venv. With build isolation, a plain `-e .` also worked but doesn't auto-rebuild.
- **Gotcha 6:** in `redirect` editable mode, the `.py` files come from the source tree while the
  `.so` and the metallib come from site-packages. Resolve data files relative to the **extension
  module**, not `__init__.py`:
  `METALLIB = Path(_lokinfer.__file__).with_name("lokinfer.metallib")`.
- C++ unit tests (Catch2) and lit run from the Ninja `build/mac` dir. They don't depend on the
  wheel build.
- Everything built with Apple clang 21 and Python 3.14.10 with **zero warnings** (nanobind 3.1.0,
  Catch2 3.16.1, benchmark 1.10.0, metal-cpp 27.0.1).

## 2. lit and filecheck [verified]

- **Gotcha 7: pip `filecheck` 1.0.6 is no longer stanislaw's package.** It is AntonLydike/filecheck
  (the xDSL team's), the declared successor of mull-project/FileCheck.py. It is a Python clone of
  LLVM FileCheck that passes about 95.8% of upstream MLIR FileCheck tests. The binary is
  `filecheck` (lowercase).
  - Supported: `CHECK`, `-NEXT`, `-NOT`, `-LABEL`, `-EMPTY`, `-SAME`, `-DAG`, `-COUNT-n`,
    `{{regex}}`, `[[VAR:regex]]` / `[[VAR]]`, numeric captures, `CHECK{LITERAL}`, `--check-prefix(es)`,
    `--match-full-lines`, `--strict-whitespace`, `--enable-var-scope`, `-D`, `--allow-empty`,
    `--dump-input=fail|never`.
  - Missing: numeric substitution arithmetic (`[[#N+1]]`), `--implicit-check-not`, `--ignore-case`,
    `--allow-unused-prefixes`, `-v`.
  - Regexes are **Python/PCRE**, not POSIX. `\d` works; `[[:alnum:]]` is translated on a
    best-effort basis.
  - There are open bugs on `CHECK-NOT` and `CHECK-LABEL` edge cases (issues #8 and #10).
  - Opt-in `FILECHECK_FEATURE_ENABLE=MLIR_REGEX_CLS` adds `\V` (matches an SSA value name).
  - Quirk: on a failure it prints a stray debug line (`moving to N of DiscontigousRange(...)`) on
    stdout. This is harmless.
- `lit` 23.1.4 (pip) works standalone.
- **Verified test set** (4 tests, against a toy `lokinfer-opt`): CHECK-LABEL, CHECK-NEXT with a
  `[[A:alloc[0-9]+]]` capture, CHECK-DAG, CHECK-SAME, CHECK-NOT with `%[[A1]]` reuse,
  CHECK-COUNT-3 under `--match-full-lines`, `not %tool ... 2>&1 | FileCheck` for error tests, and
  `XFAIL`. A deliberately wrong CHECK fails with a clear message, and the whole suite runs in
  about 0.4 s.

`tests/lit/lit.cfg.py`:
```python
import os, shutil
import lit.formats
config.name = "lokinfer"
config.test_format = lit.formats.ShTest(execute_external=False)   # lit internal shell (fast)
config.suffixes = [".lir"]
config.excludes = ["Inputs"]
config.test_source_root = os.path.dirname(__file__)
build_dir = os.path.abspath(lit_config.params.get("build_dir",
            os.path.join(config.test_source_root, "..", "..", "build", "mac")))
config.test_exec_root = os.path.join(build_dir, "lit")          # %t files out of tree
tool = os.path.join(build_dir, "tools", "lokinfer-opt", "lokinfer-opt")
if not os.path.exists(tool):
    lit_config.fatal(f"lokinfer-opt not found: {tool}")
fc = shutil.which("filecheck") or lit_config.fatal("pip filecheck not on PATH (activate .venv)")
config.substitutions.append(("%lokinfer-opt", tool))
config.substitutions.append(("FileCheck", fc))                 # keep LLVM spelling in RUN lines
config.environment["FILECHECK_FEATURE_ENABLE"] = "MLIR_REGEX_CLS"
```
Test file shape:
```
// RUN: %lokinfer-opt --pass=bufferize %s | FileCheck %s
// CHECK-LABEL: graph @add
// CHECK-NEXT:    %[[A:alloc[0-9]+]] = allocate_buffer() -> buffer<{{[0-9]+}}>
```
Hook it into ctest with
`add_test(NAME lit COMMAND ${Python_EXECUTABLE} -m lit -sv --param build_dir=${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/tests/lit)`.
Write `lokinfer-opt` to take `--pass=a,b`, read a file or stdin, print to stdout and errors to
stderr as `error: <msg> at L:C`, and exit non-zero on error.

## 3. nanobind, DLPack, MPS, and no-exceptions

### 3.1 CPU interop [verified]

- **Import** (`nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu>`) accepts
  numpy and torch CPU tensors directly. `data()` equals `a.ctypes.data` and `t.data_ptr()`, so the
  import is zero-copy. In-place writes through the view are visible in both frameworks.
- **Export:** use `nb::ndarray<nb::numpy, ...>(ptr, shape, owner_capsule)` to get a numpy view, or
  **`nb::array_api`** to get an object with `__dlpack__`, `__dlpack_device__` and the buffer
  protocol. `np.from_dlpack(x)` and `torch.from_dlpack(x)` both alias our memory (verified: same
  pointer, a torch write is visible in numpy).
  - **Gotcha 8:** without a framework tag, nanobind returns a raw `dltensor` PyCapsule, and
    `np.from_dlpack(capsule)` fails. Always use `nb::array_api` (or a concrete framework) on
    returns. The lokinfer `Tensor` class should implement `__dlpack__` and `__dlpack_device__`
    itself, following the pattern in nanobind's docs (`ndarray.rst` "Returning arrays" and
    "dlpack" sections).
- **Gotcha 9: silent copies.** By default, a `c_contig` parameter **accepts a strided array by
  making a temporary contiguous copy** (verified: `a[::2]` was accepted). If you write into it,
  the result is lost. Mark tensor arguments `nb::arg("x").noconvert()` wherever aliasing matters.
  With that, the strided input is rejected with a TypeError (verified).
- Release the GIL around kernel runs with `nb::gil_scoped_release`.

### 3.2 Unified memory and Metal [verified]

- `MTL::Device::newBuffer(ptr, len, MTL::ResourceStorageModeShared, nullptr)` (bytesNoCopy) wraps
  page-aligned host memory with no copy. The page size is 16 KiB. lokinfer Tensors on `gpu` should
  own a shared-storage `MTLBuffer` from the start and export `contents()` to the CPU side. A big
  numpy array happened to be page-aligned, but a small one was not and fell back to a copy, so
  never assume alignment for foreign arrays.

### 3.3 torch MPS [verified, torch 2.14.1]: full zero-copy works in both directions

- `mps_tensor.__dlpack_device__()` returns `(kDLMetal=8, 0)`, and `__dlpack__()` works.
  `nb::ndarray<..., nb::device::metal>` accepts it.
- **In DLPack for Metal, `dl.data` is the `id<MTLBuffer>`.** Read it as
  `(char*)nd.data() - nd.byte_offset()`, because nanobind's `data()` adds `byte_offset`, and a
  view such as `big[1024:2048]` has `byte_offset=4096`.
- That MTLBuffer has length 262144, **storageMode 0 = Shared** (so `contents()` is a valid CPU
  pointer), and the **same `MTLDevice`** as `MTLCreateSystemDefaultDevice()`.
- We ran our own metallib kernel on torch's MPS buffers through our own command queue and got
  correct results (0.01 ms). **Call `torch.mps.synchronize()` first**, because torch has its own
  queue. Use buffer offsets from `byte_offset`.
- **Reverse direction:** export our shared `MTLBuffer` as `kDLMetal` (`data=buffer handle`,
  `device::metal`, through `nb::array_api`). `torch.from_dlpack(...)` gives a `mps:0` tensor.
  After `t.mul_(10)` our buffer's `contents()` showed `[0,10,20,...]`, so it really aliases.
  Keep the buffer alive with a capsule that `release()`s it.
- This answers the original-plan worry: **zero-copy MPS interop works in torch 2.14.1.** The
  remaining caveat is synchronization between torch's queue and ours. Use
  `torch.mps.synchronize()` at the hand-off. A future optimisation could use `MTLSharedEvent`.

### 3.4 Exceptions [verified]

- **nanobind needs C++ exceptions.** Compiling a module with `-fno-exceptions` fails in
  `nb_error.h:133: cannot use 'throw' with exceptions disabled`.
- **Pattern (verified):** all core libraries build with `-fno-exceptions` and return
  `Status`/`Expected`. The binding TUs (`nanobind_add_module`) keep exceptions on and translate
  `Status` to `throw` at the boundary. Linking `-fno-exceptions` static libs into the extension
  works. `nm` shows no `__cxa_throw` in `libloki_core.a`.
- Rules:
  1. Core never calls back into Python. If a callback is ever needed, wrap it in a `noexcept`
     trampoline that catches everything and returns a Status.
  2. Catch2 test executables can keep exceptions on (Catch2 needs them for `REQUIRE`
     aborts/`THROWS`). Only the libraries under test are `-fno-exceptions`.
  3. **Keep RTTI on in core** (`-fno-rtti` is not worth it). Verified: binding a polymorphic core
     class with `nb::class_<Base>` while core used `-fno-rtti` fails to link ("Undefined symbols",
     missing typeinfo). The same code with RTTI imports and works.
  4. `-fno-exceptions` doesn't stop `operator new` failure from aborting. That is acceptable.

## 4. iOS test host app

### 4.1 Structure [verified build, unverified on device]

- Write a minimal **UIKit** app (`main.mm`, ObjC++ with `-fobjc-arc`) that links `loki_core` and
  the other static libs, `Catch2::Catch2` (**not** `Catch2WithMain`), the test sources, and
  `lokinfer.metallib` as a resource. Use a real `UIApplicationMain` rather than a bare `main()`.
  A UI-less process risks being killed by the launch watchdog, and the remote executor must stay
  alive in the foreground. Set `idleTimerDisabled = YES` so long benches don't lock the screen.
- Launch args are forwarded to Catch2 from a worker thread:
  `devicectl device process launch --console --device <udid> <bundle-id> --run-tests "[gpu]" --reporter ...`.
  In test mode the app prints `LOKI_TESTS_EXIT_CODE=N` and calls `exit(N)`.
- Info.plist template (verified it builds):
  - `UILaunchScreen` = `{}`
  - `UIRequiredDeviceCapabilities` = `[arm64, metal]`
  - `UIFileSharingEnabled` and `LSSupportsOpeningDocumentsInPlace` = true (Documents/ is
    visible in the Files app and Finder)
  - `NSLocalNetworkUsageDescription`
  - `NSBonjourServices = [_lokinfer._tcp]`
  - `NSAppTransportSecurity/NSAllowsLocalNetworking = true`. This is only needed if
    URLSession/HTTP is ever used, because raw sockets and Network.framework TCP are not subject
    to ATS.

### 4.2 Deploy and run (from devicectl 642.16 help, plus english-practice's workflow)

```
UDID=$(xcrun devicectl list devices --json-output /dev/stdout 2>/dev/null | python3 -c 'import json,sys; print(next(d["hardwareProperties"]["udid"] for d in json.load(sys.stdin)["result"]["devices"] if d["hardwareProperties"].get("reality")=="physical" and d["connectionProperties"].get("pairingState")=="paired" and d["connectionProperties"].get("tunnelState")!="unavailable"))')
xcodebuild -project build/ios-xcode/lokinfer.xcodeproj -scheme LokiHost -configuration Release \
  -destination "id=$UDID" -allowProvisioningUpdates build
xcrun devicectl device install app --device $UDID build/ios-xcode/Release-iphoneos/LokiHost.app
xcrun devicectl device process launch --device $UDID --console --terminate-existing \
  <bundle-id> --run-tests "[gpu]"          # stdout streamed, waits for exit
xcrun devicectl device copy from --device $UDID --domain-type appDataContainer \
  --domain-identifier <bundle-id> --source Documents/loki_tests.junit.xml --destination out/
```
- `--console` bridges the app's stdout to devicectl's stdout and waits for the app to exit. It
  cannot be combined with `--json-output -`; use `--json-output <file>`. Environment variables
  reach the app through `DEVICECTL_CHILD_*` or `-e '{json}'`.
- Use both result channels. Console stdout gives a fast human-readable view. The JUnit/JSON file
  in `Documents/`, pulled with `copy from`, gives machine-readable results (Catch2 multi-reporter:
  `--reporter console::out=-::colour-mode=none --reporter junit::out=<Documents>/x.xml`).
  stdout through `--console` has not been verified to be lossless for large outputs, so the
  harness should rely on the file or socket for data.
- The physical-device filter: `devicectl list devices` also lists an old `unavailable` iPhone 13
  Pro and simulators. Select by `available (paired)` and `reality=physical`.
- **Expect the weekly profile expiry** (free team). Re-running `xcodebuild -allowProvisioningUpdates`
  renews the profile. The on-device "trust" step stays manual, so stop and tell the user.

## 5. Remote executor transport

### 5.1 USB CoreDevice tunnel [verified on Mac side]

- `xcrun devicectl device info details --device <udid> --json-output f.json` brings the tunnel up.
  `result.connectionProperties` then shows `tunnelState: connected`,
  `tunnelIPAddress: fdXX:...::1`, and `transportType: wired`. The Mac side is a `utunN` with
  `...::2`. (`connectionProperties` is deprecated in favour of `result.properties`; read both.)
- **`<UDID>.coredevice.local` resolves to the current tunnel IP** (`dscacheutil`), and
  `ping6` RTT is about 3 ms. A TCP connect to a device port through the tunnel succeeded (lockdownd
  :62078). **Arbitrary TCP Mac to iPhone works over the tunnel.**
  - Caveat: Python's `getaddrinfo("<device-name>.coredevice.local")` failed. Only the UDID host
    name resolved. Pass the UDID host name, or the IP from the JSON.
- **Gotcha 10: the tunnel is on-demand and ephemeral.** It dropped about 20 s after the last
  devicectl command, and **its IPv6 address changes every time it comes up** (3 different
  addresses in 10 minutes). **Fix [unverified]:** keep `devicectl device process launch --console`
  running for the whole session (as the log pump). That should hold the tunnel open. Re-resolve
  `<UDID>.coredevice.local` before every connect, and reconnect with backoff.
- On iOS, listen on `[::]:<port>` (IPv6 any, dual-stack) with BSD sockets or `NWListener`. Use a
  fixed port such as 47800, plus a Bonjour fallback.

### 5.2 Wi-Fi [docs + english-practice experience]

- Apple TN3179 table: **"Listening for and accepting incoming TCP connections: not required"**
  (no Local Network permission), while **outgoing** TCP to a LAN address and **Bonjour/multicast**
  do require it. So a phone-side listener that the Mac connects to works without the prompt.
  Advertising `_lokinfer._tcp` over Bonjour triggers the permission prompt once; tap Allow on
  first run. Keep `NSLocalNetworkUsageDescription` and `NSBonjourServices` in the plist anyway.
  english-practice failed silently because it *initiated* connections from iOS with no usage
  description.
- Mac side: TN3179 says command-line tools run from Terminal or SSH are auto-allowed. Processes
  under other GUI hosts (VS Code terminal, agent shells) may be attributed to that app, which can
  then need the Local Network permission on macOS. If a Wi-Fi connect silently fails from one
  host but works from Terminal, this is why.
- Prefer USB: lower and more stable latency, no permission prompts, and Wi-Fi radio power states
  add jitter to timing runs. Use Wi-Fi only as a fallback.

### 5.3 Protocol recommendation

Use a plain TCP stream with length-prefixed little-endian binary frames. There are no
dependencies, and one implementation in `src/remote/` serves both the server (iOS app) and a
small client. Python gets the client through nanobind, or a 100-line pure-Python `struct` client.

```
Frame header (24 B): magic u32 'LOKI' | version u16 | type u16 | request_id u32 | flags u32 | payload_len u64
Payload: message-specific; tensors as {dtype u8, ndim u8, pad u16, shape i64[ndim], nbytes u64, raw bytes (64B-aligned)}
```
Messages (each request gets a matching response type carrying a Status code and message):

| Message | Purpose |
|---|---|
| `HELLO` | Version check. Reply: device model, OS, GPU name, git commit/build id, metallib UUID, free disk. |
| `HAS_BLOBS [sha256...]` / `PUT_BLOB sha256 + bytes` | Content-addressed upload of models and reference data into `Application Support/lokinfer/blobs/<sha256>`. Only missing blobs are sent (see 6). |
| `LOAD_MODEL blob_sha256` | Returns `model_id`. Mmaps the binary model, so no copy. |
| `SET_INPUT model_id idx tensor` or `GEN_INPUT model_id idx {dtype, shape, rng: philox, seed, dist(uniform/normal), params}` | Seed-generated inputs avoid sending data. Use **one C++ RNG implementation**, exposed to Python through nanobind on the Mac, so both sides are bit-identical. Don't reimplement it in numpy. |
| `RUN model_id {warmup, iters, sync_each}` | Reply: per-iteration wall and GPU times (ns). |
| `GET_OUTPUT model_id idx {mode: full or stats}` | Full tensor bytes, or only `{checksum, min, max, mean, l2}` for big outputs. |
| `COMPARE model_id idx ref_blob {atol, rtol}` | On-device comparison against a reference blob: returns max abs error, MSE, number of mismatches. Avoids shipping outputs back. |
| `RUN_SWEEP {kernel, ref_kernel, shapes..., seeds}` | The "on-device reference" mode. Returns pass/fail plus worst error per shape. |
| `RUN_BENCH {filter, repetitions}` | Runs the on-device kernel benchmark registry. Returns JSON (same schema as Google Benchmark). |
| `LOG` (server push) | Log lines. |
| `SHUTDOWN` | Ends the session. |

- One connection, requests handled sequentially. Use `TCP_NODELAY`. Keep frames below a few
  hundred MB and stream big blobs in 8 MiB chunks so the phone never double-buffers.
- pytest fixture: `device = RemoteDevice.connect(udid)`. It resolves the tunnel host, retries,
  and checks HELLO's commit against the local build, refusing stale apps (english-practice's
  stale-build lesson). The same test bodies run on `local_cpu`, `local_gpu`, and `iphone_gpu`
  through parametrization.

## 6. Shipping large data to the phone

- **Don't bundle reference data or models in the `.app`.** Every byte is re-signed, re-hashed and
  re-transferred on each install. Bundle only the code and the metallib.
- **Primary: push over the executor socket into a content-addressed cache.** Use
  `Application Support/lokinfer/blobs/<sha256>`, not `Caches/`, which iOS can purge. Data in the
  app data container **survives reinstalls/updates of the same bundle ID**. It is lost only when
  the app is deleted. `HAS_BLOBS` makes re-runs free. This needs no extra tooling and is under
  our control.
- **Secondary or bootstrap:** `xcrun devicectl device copy to --device <udid> --domain-type appDataContainer --domain-identifier <bundle-id> --source <file> --destination Library/Application\ Support/lokinfer/blobs/<sha>`.
  This is useful when the app isn't running. [docs; throughput not measured]
- Files must be mmap-friendly: the binary model format is already designed for mmap and
  `newBuffer(bytesNoCopy)`. Page-align the weight blob at 16 KiB.
- Generate inputs from a seed whenever possible. Ship only reference **outputs** (or only their
  stats) when the reference has to come from PyTorch.

## 7. Benchmark infrastructure

### 7.1 Google Benchmark 1.10.0 [verified]
- CPU kernels: `benchmark::DoNotOptimize`, `ClobberMemory`, `SetBytesProcessed`, and custom
  `state.counters` (shape and FLOPs, which give GFLOP/s through `benchmark::Counter::kIsRate`).
- GPU kernels: `->UseManualTime()` with `state.SetIterationTime(cb->GPUEndTime() - cb->GPUStartTime())`
  (verified). For per-dispatch precision inside one command buffer, use `MTLCounterSampleBuffer`
  timestamp sampling at dispatch boundaries (to try during the Metal 3 vs 4 dispatch-overhead
  study).
- Run with:
  `--benchmark_repetitions=N --benchmark_report_aggregates_only=true --benchmark_out=x.json --benchmark_out_format=json --benchmark_min_time=0.05s`.
  The JSON has `context` (host, CPU count, caches, `library_version`, load average) and
  `benchmarks[]` with `run_name`, `aggregate_name` (mean/median/stddev/cv), `real_time`,
  `cpu_time`, and the counters.
- Observed noise: small GPU dispatches (64K elements) had **cv 98%** (GPU clock ramp-up), while
  4M elements had cv 1%. GPU benches need warm-up iterations, report the median, and should reject
  runs with high cv (re-run). This matches the "median plus variance, rerun when noisy" decision.
  CPU at 1M elements had cv 10%. Pin to P-cores with QoS `USER_INTERACTIVE`, and run with the
  machine otherwise idle (load average was 6 to 12 during these tests).
- `benchmark-src/tools/compare.py` (needs numpy and scipy) compares two JSON files with a U-test.
  Use it for quick A/B. The DB below is for history.
- iOS: compile the benchmark library into the test host app too (it builds with iOS CMake). Run
  it with `RUN_BENCH`, or with launch args `--benchmark_out=<Documents>/b.json`, then `copy from`.

### 7.2 Results DB (SQLite, from Python's stdlib `sqlite3`)
```sql
CREATE TABLE run(id INTEGER PRIMARY KEY, ts TEXT, git_commit TEXT, dirty INT, device TEXT,
                 os TEXT, build_type TEXT, host_load REAL, notes TEXT);
CREATE TABLE result(run_id INT REFERENCES run(id), kernel TEXT, variant TEXT, backend TEXT, -- cpu|gpu|torch_cpu|torch_mps|mlx|coreml|ort
                    dtype TEXT, shape TEXT, median_ns REAL, mean_ns REAL, stddev_ns REAL, cv REAL,
                    reps INT, gflops REAL, gbps REAL, extra_json TEXT);
CREATE INDEX result_key ON result(kernel, shape, dtype, backend);
```
- `bench/ingest.py` reads Google Benchmark JSON plus our own baseline-runner JSON (torch, MLX and
  others use the same schema) and inserts a `run` row with the commit. To avoid calling `git` at
  run time, read the commit from a build-time-generated header or file, not from a `git` call.
- `bench/report.py --base <run|commit> --head <run>` prints a per-(kernel, shape, backend) table
  with speedup and significance (difference larger than about 2 × combined stddev, or a U-test).
  `--vs torch_mps` compares across backends on the same device. Report regressions above a
  threshold (tuned from observed cv) and exit non-zero for CI-style checks.

### 7.3 xctrace (headless, CLI only) [verified on Mac]
- `xcrun xctrace version` is 27.0. Templates include `Metal System Trace`, `CPU Counters`,
  `Processor Trace`, `Time Profiler`, `System Trace`, `Game Performance`, and the user's custom
  template `trace-whisper-exec`.
- Record:
  `xcrun xctrace record --template 'Metal System Trace' --output mst.trace --no-prompt --launch -- ./loki_bench --benchmark_filter=gpu`.
  It runs headlessly (no Instruments window opens) and stops when the target exits.
  **Cost:** a few-second run took 84 s total (processing and saving) and produced **234 MB**.
  Keep traced runs short and filtered.
- Export:
  `xcrun xctrace export --input mst.trace --toc` lists the tables, then
  `xcrun xctrace export --input mst.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' --output gpu.xml`
  (12 s, 54 MB, about 50k rows).
  - Rows use **`id`/`ref` deduplication**: the first occurrence carries `id="N"` and later ones
    `ref="N"`. The parser must keep an id-to-value map.
  - Rows include other processes (WindowServer "Vertex" work), so filter by `process`.
  - Useful schemas: `metal-gpu-intervals`, `metal-application-encoders-list`,
    `metal-gpu-counter-intervals` / `gpu-counter-value`, `gpu-shader-profiler-sample`
    (needs `-frecord-sources`), `metal-command-buffer-completed`.
- **CPU Counters:** `--template 'CPU Counters'` worked headlessly (14 s). The default mode is
  **"CPU Bottlenecks"** (top-down breakdown: Useful / Delivery / Discarded / Processing per 1 ms
  and 10 ms bucket, plus cycles), in table `MetricAggregationForProcess`, with P-cores labelled
  "Super". Choosing specific PMU events (instructions, L1D misses, and so on) needs a **custom
  template** saved once from Instruments (the user already has one), then passed by path
  (`--template /path/x.tracetemplate`). Also available: `l1d_miss_sampling` and
  `discarded_sampling` counting modes.
- Device traces [docs, unverified]: `xctrace record --device <udid> --template ... --attach <pid|name>`
  (or `--launch` with the bundle id). The trace is pulled back to the Mac.
- **kperf (in-process counters):** the private `kperf.framework`/`kpc` API (ibireme's well-known
  `kpc_demo` approach) gives exact cycle and instruction counts around a C++ region on macOS. It
  needs **root**, is private API, and doesn't work on iOS. Treat it as an optional Mac-only tool
  for IPC on hot CPU kernels. Default to xctrace CPU Counters.

## 8. Smoke test summary (what was proven)

| Check | Result |
|---|---|
| Ninja macOS build: core (`-fno-exceptions`), metal-cpp, metallib (2 kernels, metal4.1), Catch2 tests, Google Benchmark, `lokinfer-opt`, nanobind module (FetchContent sdist) | built, 0 warnings |
| Catch2 (GENERATE, WARN), ctest via `catch_discover_tests`, JUnit reporter | 3/3 pass, GPU add/mul correct on M2 Max |
| Metal depfile incremental rebuild | only the touched kernel rebuilt |
| Python 3.14.10 (uv venv) + numpy 2.5.4 + torch 2.14.1: ndarray import/export, DLPack, in-place aliasing, Status-to-RuntimeError | all OK |
| `noconvert()` vs silent strided copy | default copies silently; `noconvert` rejects |
| torch MPS to our kernel (DLPack kDLMetal to MTLBuffer) and our MTLBuffer to torch mps tensor | zero-copy both ways, correct |
| nanobind under `-fno-exceptions` | compile error (exceptions required in the binding layer) |
| core with `-fno-rtti` + binding a polymorphic class | link error, so keep RTTI |
| scikit-build-core 1.1.1 editable (`redirect`) via `uv pip install -e`, plus `editable.rebuild` | works; rebuild on import about 1.3 s |
| lit 23.1.4 + filecheck 1.0.6: LABEL/NEXT/DAG/SAME/NOT/COUNT/captures/`not`/XFAIL | 3 pass + 1 xfail; a negative case fails correctly |
| iOS Ninja cross-build (libs, metallib, tests) | OK (`minos 27.0`) |
| iOS host app via CMake Xcode generator plus `xcodebuild -allowProvisioningUpdates` | signed OK (team <team-id>, 7-day profile). **Not installed.** |
| Manual codesign of a Ninja-built `.app` | signature verifies (install not tested) |
| CoreDevice tunnel: IP via `devicectl info details`, `<UDID>.coredevice.local` resolution, TCP to device | works; tunnel is ephemeral (about 20 s idle) and its IP changes |
| xctrace Metal System Trace and CPU Counters, record plus XML export, headless | works |

**Still to verify on the device** (first implementation task for the iOS part):
1. Does a held-open `devicectl ... --console` keep the tunnel up, and does a TCP connect to the
   app's listener over `<UDID>.coredevice.local` work?
2. Is stdout through `--console` reliable for large outputs?
3. How fast is `devicectl copy to` compared with socket push for 100 MB+ blobs?
4. Does `MTLCounterSampleBuffer` dispatch-boundary sampling work on the A-series GPU?
5. Does `xctrace --device` work for Metal System Trace on the phone?
