// gfx12 WMMA issue ceilings: back-to-back WMMAs, ACC independent accumulators, no memory in the loop.
// One line per instruction: bf16, f16, fp8 x fp8, iu8 (all 16x16x16, f32/i32 accumulate), and the
// 16x16x32 sparse bf16 (SWMMAC: A 16x32 at 2:4, B 32x16). Run it beside a clock sampler.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>

typedef __attribute__((__vector_size__(8 * sizeof(short)))) short  s8v;
typedef __attribute__((__vector_size__(16 * sizeof(short)))) short s16v;
typedef __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16 h8v;
typedef __attribute__((__vector_size__(8 * sizeof(float)))) float  f8v;
typedef __attribute__((__vector_size__(8 * sizeof(int)))) int i8v;
typedef __attribute__((__vector_size__(2 * sizeof(int)))) int i2v;

#ifndef ACC
#define ACC 8
#endif
#define ITERS 4096

template<int OP> __global__ __launch_bounds__(256) void k(float *sink, int iters) {
    f8v c[ACC]; i8v ci[ACC];
    #pragma unroll
    for (int j = 0; j < ACC; j++) for (int i = 0; i < 8; i++) { c[j][i] = 0.f; ci[j][i] = 0; }
    s8v a, b; h8v ah, bh; i2v a8, b8; s16v bs;
    for (int i = 0; i < 8; i++) { a[i] = (short)(threadIdx.x + i); b[i] = (short)i; ah[i] = (_Float16)(i * 0.01f); bh[i] = (_Float16)(threadIdx.x * 0.001f); }
    for (int i = 0; i < 16; i++) bs[i] = (short)(i + threadIdx.x);
    a8[0] = threadIdx.x * 0x01010101; a8[1] = 0x02020202; b8[0] = 0x03030303; b8[1] = threadIdx.x;
    int idx = threadIdx.x;
    for (int it = 0; it < iters; it++) {
        #pragma unroll
        for (int j = 0; j < ACC; j++) {
            if constexpr (OP == 0) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c[j]);
            if constexpr (OP == 1) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ah, bh, c[j]);
            if constexpr (OP == 2) c[j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a8, b8, c[j]);
            if constexpr (OP == 3) ci[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a8, true, b8, ci[j], false);
            if constexpr (OP == 4) c[j] = __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(a, bs, c[j], idx);
        }
    }
    float s = 0;
    #pragma unroll
    for (int j = 0; j < ACC; j++) for (int i = 0; i < 8; i++) s += c[j][i] + (float)ci[j][i];
    if (s == 1234.5678f) sink[0] = s;
}

template<int OP> void run(const char *name, double flop_per_inst, int wgs) {
    float *sink; hipMalloc(&sink, 4);
    for (int r = 0; r < 3; r++) { hipLaunchKernelGGL(k<OP>, dim3(wgs), dim3(256), 0, 0, sink, ITERS); }
    hipDeviceSynchronize();
    hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
    const int reps = 30;
    hipEventRecord(s);
    for (int r = 0; r < reps; r++) hipLaunchKernelGGL(k<OP>, dim3(wgs), dim3(256), 0, 0, sink, ITERS);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms; hipEventElapsedTime(&ms, s, e); ms /= reps;
    const double waves = (double)wgs * 256 / 32.0;
    const double flop = waves * ITERS * ACC * flop_per_inst;
    printf("%-14s ACC=%d  %.3f ms  %.1f T(FL)OP/s\n", name, ACC, ms, flop / (ms * 1e9));
    hipFree(sink);
}

int main(int argc, char **argv) {
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    printf("device %s, %d multiprocessors, clockRate %.3f GHz\n", p.gcnArchName, p.multiProcessorCount, p.clockRate / 1e6);
    const int wgs = p.multiProcessorCount * 16;
    const double dense = 2.0 * 16 * 16 * 16;
    run<0>("bf16", dense, wgs);
    run<1>("f16", dense, wgs);
    run<2>("fp8_fp8", dense, wgs);
    run<3>("iu8", dense, wgs);
    run<4>("swmmac_bf16", 2.0 * 16 * 16 * 32, wgs);   // counted as the dense 16x16x32 it stands for
    return 0;
}
