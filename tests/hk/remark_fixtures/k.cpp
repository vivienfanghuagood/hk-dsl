#include <hip/hip_runtime.h>

__global__ void scale(float* x, float a) { x[threadIdx.x] *= a; }

template <int N>
__global__ void stack(float* out, const int* idx) {
    float buf[N];
    for (int i = 0; i < N; ++i) buf[i] = out[i * blockDim.x + threadIdx.x];
    out[threadIdx.x] = buf[idx[threadIdx.x] % N];
}
template __global__ void stack<4>(float*, const int*);
template __global__ void stack<256>(float*, const int*);
