#include "eren/reduction.cuh"

namespace eren {

__global__ void reduceSum(
    const float* input,
    float* output,
    int N
) {
    __shared__ float sdata[kReductionThreads];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reducePartialSums(
    const float* input,
    float* output,
    int N
) {
    __shared__ float sdata[kReductionThreads];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < N) ? input[i] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

}  // namespace eren
