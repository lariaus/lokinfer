cmake_minimum_required(VERSION 3.28)
# The release version lives in VERSION (pyproject.toml repeats it: checked below); git tags are v<VERSION>.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" LOKINFER_VERSION LIMIT_COUNT 1)
project(lokinfer VERSION ${LOKINFER_VERSION} LANGUAGES C CXX OBJCXX)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml" _lk_pyproject)
if(NOT _lk_pyproject MATCHES "\nversion = \"${LOKINFER_VERSION}\"")
  message(FATAL_ERROR "pyproject.toml version differs from VERSION (${LOKINFER_VERSION})")
endif()
set(CMAKE_OBJCXX_STANDARD 20)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 27.0)
endif()

option(LOKINFER_BUILD_TESTS "Build unit tests" ON)
option(LOKINFER_BUILD_BENCH "Build kernel benchmarks" OFF)
option(LOKINFER_BUILD_PYTHON "Build the nanobind extension" OFF)
# lokinfer's own dev targets: tools, bench library, the iOS test host (LokiHost). Off when lokinfer is a dependency
# (FetchContent / add_subdirectory): a consumer builds only the libraries and lokinfer.metallib.
option(LOKINFER_BUILD_DEV_TOOLS "Build tools, bench lib and the iOS test host" ${PROJECT_IS_TOP_LEVEL})

include(cmake/Deps.cmake)
include(cmake/LokiMetal.cmake)

# Core libraries are exception-free (design-decisions.md); only the Python
# binding layer uses exceptions.
set(LOKI_CORE_FLAGS -Wall -Wextra -fno-exceptions)

add_library(lokinfer_ir STATIC
  src/ir/binary.cpp
  src/ir/dim.cpp
  src/ir/ir.cpp
  src/ir/text.cpp
  src/ir/types.cpp
  src/ir/verify.cpp
  src/ir/verify_devices.cpp
)
target_include_directories(lokinfer_ir PUBLIC include)
target_compile_definitions(lokinfer_ir PUBLIC LOKINFER_VERSION="${PROJECT_VERSION}")  # visible to every consumer
target_compile_options(lokinfer_ir PRIVATE ${LOKI_CORE_FLAGS})

add_library(lokinfer_compiler STATIC
  src/compiler/analysis.cpp
  src/compiler/bufferize.cpp
  src/compiler/dce.cpp
  src/compiler/hoist_init.cpp
  src/compiler/internalize.cpp
  src/compiler/memplan.cpp
  src/compiler/passes.cpp
  src/compiler/plan_memory.cpp
)
target_link_libraries(lokinfer_compiler PUBLIC lokinfer_ir)
target_compile_options(lokinfer_compiler PRIVATE ${LOKI_CORE_FLAGS})

# Kernel metallib (one per platform).
set(LOKINFER_METALLIB ${PROJECT_BINARY_DIR}/lokinfer.metallib)
loki_add_metallib(lokinfer_metallib
  OUTPUT ${LOKINFER_METALLIB}
  SOURCES kernels/gpu/elementwise.metal kernels/gpu/gemm.metal kernels/gpu/transpose.metal
          kernels/gpu/kokoro_lc.metal kernels/gpu/kokoro_tok.metal kernels/gpu/vision.metal
          kernels/gpu/gemm_steel.metal kernels/gpu/winograd.metal
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/third_party/mlx_steel)

add_library(lokinfer_runtime STATIC
  src/runtime/executable.cpp
  src/runtime/gpu.cpp
  src/runtime/gpu_replay.cpp
  src/runtime/registry.cpp
  src/runtime/bake_init.cpp
  src/runtime/thread_pool.cpp
  kernels/cpu/elementwise.cpp
  kernels/cpu/gemm.cpp
  kernels/cpu/kokoro_conv.cpp
  kernels/cpu/kokoro_conv_fused.cpp
  kernels/cpu/kokoro_lstm.cpp
  kernels/cpu/kokoro_misc.cpp
  kernels/cpu/kokoro_norm.cpp
  kernels/cpu/kokoro_signal.cpp
  kernels/cpu/sme2_conv.cpp
  kernels/cpu/sme2_gemm.cpp
  kernels/cpu/conv2d.cpp
  kernels/cpu/decode.cpp
  kernels/gpu/dispatch.cpp
  kernels/gpu/kokoro_lc.cpp
  kernels/gpu/mps.mm
  kernels/gpu/transformer_gpu.cpp
  kernels/gpu/winograd.cpp
)
set_source_files_properties(kernels/gpu/mps.mm PROPERTIES COMPILE_OPTIONS -fobjc-arc)
# SME2 streaming kernels (A19 / M4+; runtime-checked). No +sve: non-streaming SVE doesn't exist on Apple CPUs.
set_source_files_properties(kernels/cpu/sme2_conv.cpp kernels/cpu/sme2_gemm.cpp PROPERTIES COMPILE_OPTIONS -march=armv8.6-a+sme2)
target_link_libraries(lokinfer_runtime PUBLIC lokinfer_compiler metal_cpp "-framework Accelerate" "-framework MetalPerformanceShaders")
target_compile_options(lokinfer_runtime PRIVATE ${LOKI_CORE_FLAGS})
target_compile_definitions(lokinfer_runtime PRIVATE LOKINFER_DEFAULT_METALLIB="${LOKINFER_METALLIB}")
add_dependencies(lokinfer_runtime lokinfer_metallib)

if(LOKINFER_BUILD_PYTHON)
  find_package(Python 3.14 REQUIRED COMPONENTS Interpreter Development.Module)
  if(NOT COMMAND nanobind_add_module)
    include(${nanobind_SOURCE_DIR}/cmake/nanobind-config.cmake)
  endif()
  nanobind_add_module(_lokinfer NB_STATIC python/ext/module.cpp)
  target_link_libraries(_lokinfer PRIVATE lokinfer_runtime)
  install(TARGETS _lokinfer LIBRARY DESTINATION lokinfer)
  install(FILES ${LOKINFER_METALLIB} DESTINATION lokinfer)
endif()

add_library(lokinfer_remote STATIC src/remote/server.cpp)
target_link_libraries(lokinfer_remote PUBLIC lokinfer_runtime)
target_compile_options(lokinfer_remote PRIVATE ${LOKI_CORE_FLAGS})

if(NOT LOKINFER_BUILD_DEV_TOOLS)
  return()  # a dependency: libraries + lokinfer.metallib only (unit tests need the dev targets too)
endif()

# M2 conv-proof benchmarks shared by the Mac CLI and the iOS host (`--bench conv`).
set(LOKINFER_BENCH_METALLIB ${PROJECT_BINARY_DIR}/bench.metallib)
loki_add_metallib(lokinfer_bench_metallib OUTPUT ${LOKINFER_BENCH_METALLIB}
  SOURCES bench/kernels/conv1d_gpu.metal bench/kernels/conv1d_steel.metal
  INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/third_party/mlx_steel)
add_library(lokinfer_bench_lib STATIC bench/kernels/conv_bench.cpp bench/kernels/amx_bench.cpp)
target_link_libraries(lokinfer_bench_lib PUBLIC lokinfer_runtime)
add_dependencies(lokinfer_bench_lib lokinfer_bench_metallib)

set(LOKINFER_UNIT_TEST_SOURCES
  tests/unit/test_binary.cpp
  tests/unit/test_compiler.cpp
  tests/unit/test_dim.cpp
  tests/unit/test_remote_fuzz.cpp
  tests/unit/test_runtime.cpp
  tests/unit/test_sme2_conv.cpp
  tests/unit/test_sme2_gemm.cpp
  tests/unit/test_text.cpp
)

if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  # iOS: only the test host app (Catch2 suite + remote executor), signed by
  # the Xcode generator. See docs/impl-notes/research-infra.md §4.
  # Signing: -D values, else $LOKINFER_TEAM / $LOKINFER_BUNDLE_ID, else the git-ignored lokinfer.local.json
  # (ios_team_id / ios_bundle_id; template lokinfer.local.example.json).
  set(_lk_team "$ENV{LOKINFER_TEAM}")
  set(_lk_bundle "$ENV{LOKINFER_BUNDLE_ID}")
  if(EXISTS "${PROJECT_SOURCE_DIR}/lokinfer.local.json")
    file(READ "${PROJECT_SOURCE_DIR}/lokinfer.local.json" _lk_local)
    if(NOT _lk_team)
      string(JSON _lk_team ERROR_VARIABLE _lk_err GET "${_lk_local}" ios_team_id)
    endif()
    if(NOT _lk_bundle)
      string(JSON _lk_bundle ERROR_VARIABLE _lk_err GET "${_lk_local}" ios_bundle_id)
    endif()
  endif()
  if(NOT _lk_bundle OR _lk_bundle MATCHES "-NOTFOUND$")
    set(_lk_bundle "com.example.lokinfer.testhost")
  endif()
  if(_lk_team MATCHES "-NOTFOUND$")
    set(_lk_team "")
  endif()
  set(LOKINFER_TEAM "${_lk_team}" CACHE STRING "Apple development team (lokinfer.local.json ios_team_id)")
  set(LOKINFER_HOST_BUNDLE_ID "${_lk_bundle}" CACHE STRING "LokiHost bundle id (lokinfer.local.json ios_bundle_id)")
  if(NOT LOKINFER_TEAM)
    message(WARNING "LOKINFER_TEAM is not set: LokiHost can't be signed (set ios_team_id in lokinfer.local.json)")
  endif()
  set_source_files_properties(${LOKINFER_METALLIB} ${LOKINFER_BENCH_METALLIB} PROPERTIES GENERATED TRUE
    MACOSX_PACKAGE_LOCATION Resources)
  add_executable(LokiHost MACOSX_BUNDLE ios/TestHost/main.mm ios/TestHost/build_id.mm ${LOKINFER_UNIT_TEST_SOURCES} ${LOKINFER_METALLIB}
    ${LOKINFER_BENCH_METALLIB})
  add_dependencies(LokiHost lokinfer_metallib lokinfer_bench_metallib)
  target_link_libraries(LokiHost PRIVATE lokinfer_remote lokinfer_bench_lib Catch2::Catch2 "-framework UIKit")
  target_compile_options(LokiHost PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  set_target_properties(LokiHost PROPERTIES
    MACOSX_BUNDLE_INFO_PLIST ${PROJECT_SOURCE_DIR}/ios/TestHost/Info.plist.in
    MACOSX_BUNDLE_GUI_IDENTIFIER ${LOKINFER_HOST_BUNDLE_ID}
    MACOSX_BUNDLE_BUNDLE_NAME LokiHost
    XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER ${LOKINFER_HOST_BUNDLE_ID}
    XCODE_ATTRIBUTE_DEVELOPMENT_TEAM ${LOKINFER_TEAM}
    XCODE_ATTRIBUTE_CODE_SIGN_STYLE Automatic
    XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "Apple Development"
    XCODE_ATTRIBUTE_CODE_SIGN_ENTITLEMENTS ${PROJECT_SOURCE_DIR}/ios/TestHost/LokiHost.entitlements
    XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET 27.0)
  return()
endif()

add_executable(lokinfer-opt tools/lokinfer_opt.cpp)
target_link_libraries(lokinfer-opt PRIVATE lokinfer_compiler)
add_executable(lokinfer-serve tools/lokinfer_serve.cpp)
target_link_libraries(lokinfer-serve PRIVATE lokinfer_remote)
add_executable(lokinfer-bench-conv tools/lokinfer_bench_conv.cpp)
target_link_libraries(lokinfer-bench-conv PRIVATE lokinfer_bench_lib)
target_compile_definitions(lokinfer-bench-conv PRIVATE LOKINFER_BENCH_METALLIB_PATH="${LOKINFER_BENCH_METALLIB}")

if(LOKINFER_BUILD_BENCH)
  add_executable(lokinfer_bench_runtime bench/runtime_overhead.cpp)
  target_link_libraries(lokinfer_bench_runtime PRIVATE lokinfer_runtime benchmark::benchmark)
  add_executable(lokinfer_bench_kernels bench/kernels/kernel_gbench.cpp)
  target_link_libraries(lokinfer_bench_kernels PRIVATE lokinfer_runtime benchmark::benchmark)
  add_executable(lokinfer_bench_conv1d_cpu bench/kernels/conv1d_cpu.cpp)
  target_link_libraries(lokinfer_bench_conv1d_cpu PRIVATE lokinfer_runtime benchmark::benchmark)
  add_executable(lokinfer_bench_conv1d_gpu bench/kernels/conv1d_gpu.cpp)
  target_link_libraries(lokinfer_bench_conv1d_gpu PRIVATE lokinfer_runtime)
  add_dependencies(lokinfer_bench_conv1d_gpu lokinfer_bench_metallib)
endif()

if(LOKINFER_BUILD_TESTS)
  enable_testing()
  add_executable(lokinfer_unit_tests ${LOKINFER_UNIT_TEST_SOURCES})
  target_link_libraries(lokinfer_unit_tests PRIVATE lokinfer_runtime lokinfer_remote Catch2::Catch2WithMain)
  include(Catch)
  catch_discover_tests(lokinfer_unit_tests)
endif()
