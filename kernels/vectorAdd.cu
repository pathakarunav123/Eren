
#include "eren/cuda_check.cuh"

#include <cstdlib>
#include <iostream>
#include <vector>

// GPU kernel: this function executes on the GPU.
__global__ void vectorAdd(
    const float* A,
    const float* B,
    float* C,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}



int main() {
    const int N = 10;
    const size_t bytes = N * sizeof(float);

    std::vector<float> A(N);
    std::vector<float> B(N);
    std::vector<float> C(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i * 10);
    }

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    // Allocate memory on the GPU.
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Copy input vectors from CPU memory to GPU memory.
    CUDA_CHECK(cudaMemcpy(
        d_A, A.data(), bytes, cudaMemcpyHostToDevice
    ));
    CUDA_CHECK(cudaMemcpy(
        d_B, B.data(), bytes, cudaMemcpyHostToDevice
    ));

    // Launch the kernel: 256 threads per block.
    constexpr int threadsPerBlock = 256;
    const int blocks =
        (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check launch errors and wait for the GPU to finish.
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result from GPU memory back to CPU memory.
    CUDA_CHECK(cudaMemcpy(
        C.data(), d_C, bytes, cudaMemcpyDeviceToHost
    ));

    bool correct = true;

    for (int i = 0; i < N; ++i) {
        const float expected = A[i] + B[i];

        std::cout << A[i] << " + " << B[i]
                  << " = " << C[i] << '\n';

        if (C[i] != expected) {
            correct = false;
        }
    }

    std::cout << (correct
        ? "Vector addition PASSED\n"
        : "Vector addition FAILED\n");

    // Release GPU memory.
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
