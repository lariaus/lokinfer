# Offline MSL compilation: one .air per .metal, linked into one .metallib.
# See docs/impl-notes/research-infra.md §1.4.
set(LOKI_METAL_STD "metal4.1" CACHE STRING "MSL language standard")
option(LOKI_METAL_DEBUG_INFO "Embed sources + line tables (shader profiling)" ON)

if(CMAKE_SYSTEM_NAME STREQUAL "iOS")
  set(LOKI_METAL_SDK iphoneos)
  set(LOKI_METAL_MINVER -mios-version-min=27.0)
else()
  set(LOKI_METAL_SDK macosx)
  set(LOKI_METAL_MINVER -mmacosx-version-min=27.0)
endif()

execute_process(COMMAND xcrun -sdk ${LOKI_METAL_SDK} -f metal
  RESULT_VARIABLE _loki_metal_found OUTPUT_QUIET ERROR_QUIET)
if(NOT _loki_metal_found EQUAL 0)
  message(FATAL_ERROR "Metal toolchain missing: run `xcodebuild -downloadComponent MetalToolchain`")
endif()

function(loki_add_metallib target)
  cmake_parse_arguments(A "" "OUTPUT" "SOURCES;INCLUDE_DIRS" ${ARGN})
  # Always -O3, Debug builds included (2026-11-06): at -O0 the steel f16 GEMMs (lk_f16o_gemm_swiglu_64x64) fail
  # pipeline creation with "Compute function exceeds available stack space", so a Debug build (build/lokinfer-serve,
  # the dev tests' server) could not run Qwen3-ASR on the GPU. Shader debugging keeps working through
  # LOKI_METAL_DEBUG_INFO (source lines). Override with LOKI_METAL_OPT (e.g. -O0) for a one-off shader debug session.
  set(LOKI_METAL_OPT "-O3" CACHE STRING "Metal shader optimization level")
  set(opt ${LOKI_METAL_OPT})
  set(flags -std=${LOKI_METAL_STD} ${LOKI_METAL_MINVER} -Wall ${opt})
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
    DEPENDS ${airs} VERBATIM
    COMMENT "metallib ${target}")
  add_custom_target(${target} ALL DEPENDS ${A_OUTPUT})
endfunction()
