// gfx1100 WMMA rate vs independent accumulator chains (1/2/4/8), f16 and bf16, and vs waves a SIMD.
// The question: is K3's QK phase (2 chains, 8 deep) issue-limited by WMMA dependency latency?
#include <hip/hip_runtime.h>
#include <cstdio>
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
constexpr int IT = 4096;
template <int CH, bool BF> __global__ __launch_bounds__(256) void k(float *out) {
  v16h a, b; v16s sa, sb;
  for (int i = 0; i < 16; i++) { a[i] = (_Float16)(threadIdx.x * 0.001f + i); b[i] = a[i]; sa[i] = threadIdx.x + i; sb[i] = sa[i]; }
  v8f c[CH];
  for (int j = 0; j < CH; j++) c[j] = v8f{0};
  for (int i = 0; i < IT * 8 / CH; i++) {
#pragma unroll
    for (int j = 0; j < CH; j++) {
      if constexpr (BF) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(sa, sb, c[j]);
      else c[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c[j]);
    }
  }
  float r = 0;
  for (int j = 0; j < CH; j++) r += c[j][j % 8];
  if (r == 12345.f) out[threadIdx.x] = r;
}
template <int CH, bool BF> void run(float *o, int blocks, const char *name, int occ) {
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  float ms = 0;
  for (int r = 0; r < 3; r++) {
    hipEventRecord(e0);
    k<CH, BF><<<blocks, 256>>>(o);
    hipEventRecord(e1); hipEventSynchronize(e1);
    hipEventElapsedTime(&ms, e0, e1);
  }
  double ops = (double)blocks * 8 * IT * 8 * 2.0 * 16 * 16 * 16;
  printf("%-5s chains %d  waves/SIMD %2d  %7.1f TFLOP/s\n", name, CH, occ, ops / ms / 1e9);
}
int main() {
  float *o; hipMalloc(&o, 4096);
  hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
  // a WGP (2 CUs) has 4 SIMDs: one 8-wave block = 2 waves a SIMD of one WGP; 48 WGPs
  for (int occ : {2, 6, 12}) {
    int blocks = 48 * occ / 2;
    run<1, false>(o, blocks, "f16", occ); run<2, false>(o, blocks, "f16", occ);
    run<4, false>(o, blocks, "f16", occ); run<8, false>(o, blocks, "f16", occ);
    run<1, true>(o, blocks, "bf16", occ); run<2, true>(o, blocks, "bf16", occ);
    run<4, true>(o, blocks, "bf16", occ); run<8, true>(o, blocks, "bf16", occ);
  }
}
