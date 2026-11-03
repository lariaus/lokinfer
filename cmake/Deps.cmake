# Third-party dependencies: FetchContent with pinned release tarballs + SHA256
# (no git checkouts). Versions are pinned here and only here; see
# docs/impl-notes/research-infra.md §1.1 for provenance.
include(FetchContent)

# metal-cpp: header-only; wrapped as an INTERFACE target.
FetchContent_Declare(metalcpp
  URL https://github.com/apple/metal-cpp/archive/refs/tags/release/metal-cpp_macOS27.0.1_iOS27.0.1.tar.gz
  URL_HASH SHA256=3a2f456c5373748e1b8de0896d1dbe53d6e0a1d3b8b9e5235c1a8abae6f373f5
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
FetchContent_MakeAvailable(metalcpp)
add_library(metal_cpp INTERFACE)
target_include_directories(metal_cpp SYSTEM INTERFACE ${metalcpp_SOURCE_DIR})
target_link_libraries(metal_cpp INTERFACE "-framework Metal" "-framework Foundation" "-framework QuartzCore")

if(LOKINFER_BUILD_TESTS)
  FetchContent_Declare(Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.16.1.tar.gz
    URL_HASH SHA256=d16bd1b8b2364918bd472cb216394acb93bde6784890df0d91bccfb3f28d777d
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_MakeAvailable(Catch2)
  list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif()

if(LOKINFER_BUILD_BENCH)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.10.0.tar.gz
    URL_HASH SHA256=acc5c7c0fbed855c4aaa2d7c3c36fb82b0c15c28f4e151e3f679c2fc0aa404af
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_MakeAvailable(benchmark)
endif()

if(LOKINFER_BUILD_PYTHON)
  # PyPI sdist: the GitHub archive lacks the robin_map submodule.
  FetchContent_Declare(nanobind
    URL https://files.pythonhosted.org/packages/c7/1f/e28fae715401e8f57400e72aeaff4fde12046d31828164f34f61f0816da8/nanobind-3.1.0.tar.gz
    URL_HASH SHA256=6661e3dc1434d78781ccf7913442d90361dab395c3424efadb63bbb6ddea7ea3
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_MakeAvailable(nanobind)
endif()
