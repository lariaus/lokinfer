// MLX steel implicit-GEMM conv (vendored, third_party/mlx_steel) instantiated
// for lokinfer's channel-last conv1d: input [L, Ci] (as NHWC with N=H=1),
// weights [Co, K, Ci], output [Lo, Co].
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/mma.h"
#include "mlx/backend/metal/kernels/steel/conv/conv.h"
#include "mlx/backend/metal/kernels/steel/conv/params.h"
#include "mlx/backend/metal/kernels/steel/conv/kernels/steel_conv.h"

template [[host_name("steel_conv1d_lc_f32_bm64_bn64")]] [[kernel]] void
implicit_gemm_conv_2d<float, 64, 64, 16, 2, 2, 0, true>(
    const device float* A [[buffer(0)]], const device float* B [[buffer(1)]], device float* C [[buffer(2)]],
    const constant MLXConvParams<2>* params [[buffer(3)]],
    const constant ImplicitGemmConv2DParams* gemm_params [[buffer(4)]], uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]], uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]);
