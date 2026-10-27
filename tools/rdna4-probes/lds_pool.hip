// How many workgroups of a given LDS size are resident at once: the LDS pool, measured.
// Each workgroup (one wave) counts itself in, records the high-water mark, sleeps, counts itself out.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
__global__ void k(int *active, int *peak, int iters) {
    extern __shared__ int lds[];
    if (threadIdx.x == 0) {
        lds[0] = 1;
        int a = atomicAdd(active, 1) + 1;
        atomicMax(peak, a);
        for (int i = 0; i < iters; i++) __builtin_amdgcn_s_sleep(127);
        atomicSub(active, 1);
        if (lds[0] != 1) *peak = -1000000;
    }
}
int main(int argc, char **argv) {
    int *d; hipMalloc(&d, 8);
    int sizes[] = {8, 16, 24, 32, 40, 48, 56, 64};
    for (int kb : sizes) {
        int bytes = kb * 1024 - 64;   // a little under, so static/padding cannot push it over
        hipFuncSetAttribute((const void*)k, hipFuncAttributeMaxDynamicSharedMemorySize, bytes);
        int best = 0;
        for (int rep = 0; rep < 3; rep++) {
            hipMemset(d, 0, 8);
            hipLaunchKernelGGL(k, dim3(4096), dim3(32), bytes, 0, d, d + 1, 400);
            hipDeviceSynchronize();
            int h[2]; hipMemcpy(h, d, 8, hipMemcpyDeviceToHost);
            if (h[1] > best) best = h[1];
        }
        printf("LDS %2d KB/workgroup: peak %4d resident workgroups -> %7.1f KB in use\n", kb, best, best * kb * 1.0);
    }
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    printf("CUs %d, sharedMemPerBlock %zu, maxSharedMemoryPerMultiProcessor %zu\n", p.multiProcessorCount, p.sharedMemPerBlock, p.maxSharedMemoryPerMultiProcessor);
    return 0;
}
