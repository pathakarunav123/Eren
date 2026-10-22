#pragma once

#include <cuda_runtime.h>

namespace eren {

constexpr int kReductionThreads = 256;

__global__ void reduceSum(
    const float* input,
    float* output,
    int N
);

__global__ void reducePartialSums(
    const float* input,
    float* output,
    int N
);

}  // namespace eren
