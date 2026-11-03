#include <metal_stdlib>
using namespace metal;

struct ElementwiseParams {
  uint n;
};

// 4 elements per thread (vectorized), scalar tail handled by the last threads.
#define LK_BINARY_F32(NAME, EXPR)                                                     \
  kernel void NAME(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], \
                   device float* out [[buffer(2)]], constant ElementwiseParams& p [[buffer(3)]], \
                   uint gid [[thread_position_in_grid]]) {                             \
    uint i = gid * 4;                                                                   \
    if (i + 4 <= p.n) {                                                                 \
      float4 x = *reinterpret_cast<device const float4*>(a + i);                        \
      float4 y = *reinterpret_cast<device const float4*>(b + i);                        \
      *reinterpret_cast<device float4*>(out + i) = EXPR;                                \
    } else {                                                                            \
      for (; i < p.n; ++i) {                                                            \
        float x = a[i], y = b[i];                                                       \
        out[i] = EXPR;                                                                  \
      }                                                                                 \
    }                                                                                   \
  }

LK_BINARY_F32(add_f32, x + y)
LK_BINARY_F32(mul_f32, x * y)
