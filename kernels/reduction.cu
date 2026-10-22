
#include "eren/cuda_check.cuh"
#include "eren/reduction.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>



int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <N>\n";
        return EXIT_FAILURE;
    }

    int N = 0;

    try {
        std::size_t parsedChars = 0;
        const long long parsedN = std::stoll(argv[1], &parsedChars);

        if (parsedChars != std::string(argv[1]).size() ||
            parsedN < 1 ||
            parsedN > std::numeric_limits<int>::max()) {
            std::cerr << "Error: N must be a positive integer "
                         "within the supported int range.\n";
            return EXIT_FAILURE;
        }

        N = static_cast<int>(parsedN);
    } catch (const std::exception&) {
        std::cerr << "Error: N must be a valid positive integer.\n";
        return EXIT_FAILURE;
    }

    const int blocks = 1 + (N - 1) / eren::kReductionThreads;

    std::vector<float> input(N);

    // Deterministic varied inputs: positive, negative and zero values.
    for (int i = 0; i < N; ++i) {
        input[i] = static_cast<float>((i % 11) - 5);
    }

    float* d_input = nullptr;
    float* d_bufferA = nullptr;
    float* d_bufferB = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_input,
        static_cast<std::size_t>(N) * sizeof(float)
    ));

    // Both buffers need space for the largest partial-sum count.
    CUDA_CHECK(cudaMalloc(
        &d_bufferA,
        static_cast<std::size_t>(blocks) * sizeof(float)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_bufferB,
        static_cast<std::size_t>(blocks) * sizeof(float)
    ));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        input.data(),
        static_cast<std::size_t>(N) * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // First reduction stage: original input -> partial sums.
    eren::reduceSum<<<blocks, eren::kReductionThreads>>>(d_input, d_bufferA, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int currentCount = blocks;
    float* currentInput = d_bufferA;
    float* nextOutput = d_bufferB;

    // Repeatedly reduce partial sums until one value remains.
    while (currentCount > 1) {
        const int nextCount = 1 + (currentCount - 1) / eren::kReductionThreads;

        eren::reducePartialSums<<<nextCount, eren::kReductionThreads>>>(
            currentInput,
            nextOutput,
            currentCount
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        currentCount = nextCount;

        // Swap input/output buffers for the next stage.
        float* temp = currentInput;
        currentInput = nextOutput;
        nextOutput = temp;
    }

    float gpuResult = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        &gpuResult,
        currentInput,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // Independent CPU reference using double precision.
    double cpuResult = 0.0;

    for (float value : input) {
        cpuResult += static_cast<double>(value);
    }

    const double absoluteError =
        std::abs(static_cast<double>(gpuResult) - cpuResult);

    const double tolerance =
        1e-5 + 1e-6 * std::abs(cpuResult);

    const bool passed = absoluteError <= tolerance;

    std::cout << "Input N: " << N << '\n';
    std::cout << "Initial blocks: " << blocks << '\n';
    std::cout << "CPU sum: " << cpuResult << '\n';
    std::cout << "GPU sum: " << gpuResult << '\n';
    std::cout << (passed
        ? "Reduction PASSED\n"
        : "Reduction FAILED\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_bufferA));
    CUDA_CHECK(cudaFree(d_bufferB));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
