// The two numbers a roofline on this card needs, measured rather than read off
// a datasheet: the dense bf16 MFMA rate and the HBM bandwidth.
//
// MFMA: a register-only loop. Every wave keeps CHAINS independent accumulators
// and issues v_mfma_f32_32x32x16_bf16 (or 16x16x32) on operands that never
// leave its registers, so nothing but the matrix cores is timed. Swept over
// waves per SIMD and chain counts; the best is the peak a kernel can reach.
// Each wave also reads the shader clock (s_memtime) and the 100 MHz wall clock
// (s_memrealtime) around its loop, which gives the clock the loop ran at.
//
// HBM: a copy (read once, write once) and a read, each from buffers far past
// the 256 MB Infinity Cache: every timed call takes the next of four 1 GiB
// buffers, so no call finds its data in the cache. The same kernels at 64 MB,
// inside the cache, are printed for comparison.
//
//   hipcc -O3 --offload-arch=gfx950 -o /tmp/peaks tools/cdna-probes/peaks.hip
//   HIP_VISIBLE_DEVICES=0 /tmp/peaks

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                              \
    do {                                                                      \
        hipError_t e_ = (x);                                                  \
        if (e_ != hipSuccess) {                                               \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__,            \
                         hipGetErrorString(e_));                              \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x16 __attribute__((ext_vector_type(16)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

// ------------------------------------------------------------------ MFMA

template <int CHAINS, bool BIG>
__global__ void mfma_loop(float *out, unsigned long long *clocks, int iters,
                          int flag) {
    bf16x8 a, b;
    for (int i = 0; i < 8; ++i) {
        a[i] = (__bf16)(float)((threadIdx.x + i) & 7);
        b[i] = (__bf16)(float)((threadIdx.x * 3 + i) & 7);
    }
    unsigned long long c0 = __builtin_amdgcn_s_memtime();
    unsigned long long w0 = __builtin_amdgcn_s_memrealtime();
    float s = 0.f;
    if constexpr (BIG) {
        f32x16 acc[CHAINS];
        for (int c = 0; c < CHAINS; ++c)
            for (int i = 0; i < 16; ++i) acc[c][i] = 0.f;
        for (int it = 0; it < iters; ++it)
#pragma unroll
            for (int c = 0; c < CHAINS; ++c)
                acc[c] = __builtin_amdgcn_mfma_f32_32x32x16_bf16(a, b, acc[c], 0,
                                                                 0, 0);
        for (int c = 0; c < CHAINS; ++c) s += acc[c][0] + acc[c][15];
    } else {
        f32x4 acc[CHAINS];
        for (int c = 0; c < CHAINS; ++c)
            for (int i = 0; i < 4; ++i) acc[c][i] = 0.f;
        for (int it = 0; it < iters; ++it)
#pragma unroll
            for (int c = 0; c < CHAINS; ++c)
                acc[c] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, acc[c], 0,
                                                                 0, 0);
        for (int c = 0; c < CHAINS; ++c) s += acc[c][0] + acc[c][3];
    }
    unsigned long long c1 = __builtin_amdgcn_s_memtime();
    unsigned long long w1 = __builtin_amdgcn_s_memrealtime();
    // flag is 0 at run time: keeps the accumulators live without a store.
    if (flag) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        clocks[0] = c1 - c0;
        clocks[1] = w1 - w0;
    }
}

using MfmaLaunch = void (*)(float *, unsigned long long *, int, int, int, int);

template <int CHAINS, bool BIG>
static void mfma_launch(float *out, unsigned long long *clk, int iters,
                        int grid, int block, int flag) {
    mfma_loop<CHAINS, BIG><<<grid, block>>>(out, clk, iters, flag);
}

// ------------------------------------------------------------------ HBM

template <int UNROLL>
__global__ void copy_kernel(const uint4 *__restrict__ src, uint4 *__restrict__ dst,
                            size_t n16) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (UNROLL - 1) * stride < n16; i += UNROLL * stride) {
        uint4 v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) v[u] = src[i + u * stride];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) dst[i + u * stride] = v[u];
    }
    for (; i < n16; i += stride) dst[i] = src[i];
}

template <int UNROLL>
__global__ void read_kernel(const uint4 *__restrict__ p, size_t n16, uint4 *sink) {
    uint4 acc = {0, 0, 0, 0};
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (UNROLL - 1) * stride < n16; i += UNROLL * stride) {
        uint4 v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) v[u] = p[i + u * stride];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w;
        }
    }
    for (; i < n16; i += stride) {
        uint4 v = p[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) sink[0] = acc;
}

using CopyLaunch = void (*)(const uint4 *, uint4 *, size_t, int, int);
using ReadLaunch = void (*)(const uint4 *, size_t, uint4 *, int, int);

template <int U>
static void copy_launch(const uint4 *s, uint4 *d, size_t n16, int grid, int block) {
    copy_kernel<U><<<grid, block>>>(s, d, n16);
}
template <int U>
static void read_launch(const uint4 *p, size_t n16, uint4 *sink, int grid, int block) {
    read_kernel<U><<<grid, block>>>(p, n16, sink);
}

int main(int argc, char **argv) {
    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    const int cus = prop.multiProcessorCount;
    int wall_khz = 0;
    CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
    std::printf("%s, %d CUs, max clock %d MHz, wall clock %d kHz\n", prop.gcnArchName,
                cus, prop.clockRate / 1000, wall_khz);

    hipEvent_t t0, t1;
    CHECK(hipEventCreate(&t0));
    CHECK(hipEventCreate(&t1));
    auto time_ms = [&](auto &&fn, int reps) {
        float best = 1e30f;
        for (int r = 0; r < reps; ++r) {
            CHECK(hipEventRecord(t0));
            fn(r);
            CHECK(hipEventRecord(t1));
            CHECK(hipEventSynchronize(t1));
            float ms;
            CHECK(hipEventElapsedTime(&ms, t0, t1));
            best = std::min(best, ms);
        }
        return best;
    };

    // ---- MFMA
    float *out;
    unsigned long long *clk;
    CHECK(hipMalloc(&out, sizeof(float) * cus * 64 * 64));
    CHECK(hipMalloc(&clk, sizeof(unsigned long long) * 2));
    struct M { const char *name; MfmaLaunch fn; int chains; double flop; };
    const M ms[] = {
        {"32x32x16", mfma_launch<4, true>, 4, 2.0 * 32 * 32 * 16},
        {"32x32x16", mfma_launch<2, true>, 2, 2.0 * 32 * 32 * 16},
        {"16x16x32", mfma_launch<4, false>, 4, 2.0 * 16 * 16 * 32},
        {"16x16x32", mfma_launch<8, false>, 8, 2.0 * 16 * 16 * 32},
    };
    const int iters = 20000;
    std::printf("\nbf16 MFMA, register-only, %d iterations a wave:\n", iters);
    double peak = 0;
    for (const M &m : ms)
        for (int waves_per_simd : {1, 2, 4}) {
            const int block = 256;  // four waves, one per SIMD
            const int grid = cus * waves_per_simd;
            m.fn(out, clk, iters, grid, block, 0);  // warm
            float t = time_ms([&](int) { m.fn(out, clk, iters, grid, block, 0); }, 5);
            unsigned long long h[2];
            CHECK(hipMemcpy(h, clk, sizeof(h), hipMemcpyDeviceToHost));
            const double flops = m.flop * m.chains * (double)iters * grid * (block / 64);
            const double tf = flops / (t * 1e-3) / 1e12;
            const double mhz = h[1] ? (double)h[0] / h[1] * wall_khz / 1e3 : 0;
            peak = std::max(peak, tf);
            std::printf("  %s chains %d, %d wave(s)/SIMD: %7.1f TFLOP/s  (%.3f ms; "
                        "loop at %.0f MHz by s_memtime; %.0f FLOP/clk/CU)\n",
                        m.name, m.chains, waves_per_simd, tf, t, mhz,
                        mhz ? tf * 1e12 / (mhz * 1e6) / cus : 0.0);
        }
    std::printf("best: %.1f TFLOP/s\n", peak);
    {
        // The best shape held for ~0.1-0.2 s a call, as a long prefill GEMM
        // holds it: what the clock settles at under sustained MFMA load.
        const int long_iters = iters * 25, grid = cus * 4, block = 256;
        mfma_launch<4, true>(out, clk, long_iters, grid, block, 0);
        float t = time_ms([&](int) { mfma_launch<4, true>(out, clk, long_iters, grid,
                                                          block, 0); }, 5);
        unsigned long long h[2];
        CHECK(hipMemcpy(h, clk, sizeof(h), hipMemcpyDeviceToHost));
        const double tf = 2.0 * 32 * 32 * 16 * 4 * (double)long_iters * grid * 4 /
                          (t * 1e-3) / 1e12;
        std::printf("sustained, 32x32x16 chains 4, 4 waves/SIMD, %.1f ms a call: "
                    "%.1f TFLOP/s, loop at %.0f MHz\n", t, tf,
                    (double)h[0] / h[1] * wall_khz / 1e3);
    }

    // ---- HBM
    const size_t big = 1ull << 30;
    const int n_bufs = 4;
    std::vector<uint4 *> src(n_bufs), dst(n_bufs);
    for (int i = 0; i < n_bufs; ++i) {
        CHECK(hipMalloc(&src[i], big));
        CHECK(hipMalloc(&dst[i], big));
        CHECK(hipMemset(src[i], 0x5a, big));
        CHECK(hipMemset(dst[i], 0, big));
    }
    uint4 *sink;
    CHECK(hipMalloc(&sink, 16));
    const CopyLaunch copies[] = {copy_launch<1>, copy_launch<2>, copy_launch<4>};
    const ReadLaunch reads[] = {read_launch<1>, read_launch<2>, read_launch<4>,
                                read_launch<8>, read_launch<16>};
    // Calls back to back between two events, so a launch's gap is not timed:
    // past the cache each call takes the next buffer (4 GiB between two reads
    // of one), inside it every call reads the same one.
    struct S { const char *name; size_t bytes; bool rotate; int calls; };
    const S sizes[] = {{"1 GiB, four buffers rotated (past the cache)", big, true, 4},
                       {"64 MiB, one buffer (inside the cache)", 64ull << 20, false, 32}};
    for (const S &s : sizes) {
        const size_t n16 = s.bytes / 16;
        double best_c = 0, best_r = 0;
        int cg = 0, cb = 0, cu_ = 0, rg = 0, rb = 0, ru = 0;
        for (int gpc : {1, 2, 4, 8, 16, 32})
            for (int block : {256, 512, 1024}) {
                const int grid = gpc * cus;
                for (int u = 0; u < 3; ++u) {
                    copies[u](src[0], dst[0], n16, grid, block);
                    float t = time_ms([&](int) {
                        for (int c = 0; c < s.calls; ++c) {
                            int k = s.rotate ? c % n_bufs : 0;
                            copies[u](src[k], dst[k], n16, grid, block);
                        }
                    }, 6) / s.calls;
                    double gbs = 2.0 * s.bytes / (t * 1e-3) / 1e9;
                    if (gbs > best_c) { best_c = gbs; cg = grid; cb = block; cu_ = 1 << u; }
                }
                for (int u = 0; u < 5; ++u) {
                    reads[u](src[0], n16, sink, grid, block);
                    float t = time_ms([&](int) {
                        for (int c = 0; c < s.calls; ++c) {
                            int k = s.rotate ? c % n_bufs : 0;
                            reads[u](src[k], n16, sink, grid, block);
                        }
                    }, 6) / s.calls;
                    double gbs = (double)s.bytes / (t * 1e-3) / 1e9;
                    if (gbs > best_r) { best_r = gbs; rg = grid; rb = block; ru = 1 << u; }
                }
            }
        std::printf("\n%s:\n  copy %.0f GB/s (read + written; grid %d, block %d, unroll %d)\n"
                    "  read %.0f GB/s (grid %d, block %d, unroll %d)\n",
                    s.name, best_c, cg, cb, cu_, best_r, rg, rb, ru);
    }
    return 0;
}
