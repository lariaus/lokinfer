// out[C, R] = in[R, C]^T, f32. 32x32 tiles through threadgroup memory
// (+1 column of padding against bank conflicts); 32x8 threads per tile.
#include <metal_stdlib>
using namespace metal;

struct TransposeParams {
  uint rows, cols;
};

kernel void transpose_f32(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                          constant TransposeParams& p [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint2 t [[thread_position_in_threadgroup]]) {
  threadgroup float tile[32][33];
  uint c0 = tg.x * 32, r0 = tg.y * 32;
  for (uint k = 0; k < 32; k += 8) {
    uint r = r0 + t.y + k, c = c0 + t.x;
    if (r < p.rows && c < p.cols) tile[t.y + k][t.x] = in[r * p.cols + c];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = 0; k < 32; k += 8) {
    uint oc = r0 + t.x, orow = c0 + t.y + k;  // out is [cols, rows]
    if (orow < p.cols && oc < p.rows) out[orow * p.rows + oc] = tile[t.x][t.y + k];
  }
}
