#include "eren/cuda_check.cuh"
#include "eren/reduction.cuh"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

constexpr int kWarmupRuns = 10;
constexpr int kTimedRuns = 100;

int ceilDiv(int n, int d) {
    return 1 + (n - 1) / d;
}

void runBenchmark(int N) {
    const int threads = eren::kReductionThreads;
    const int initialBlocks = ceilDiv(N, threads);

    std::vector<float> input(N);
    for (int i = 0; i < N; ++i) {
        input[i] = static_cast<float>((i % 11) - 5);
    }

    float* dInput = nullptr;
    float* dBufferA = nullptr;
    float* dBufferB = nullptr;

    CUDA_CHECK(cudaMalloc(&dInput, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBufferA, initialBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBufferB, initialBlocks * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        dInput, input.data(), N * sizeof(float), cudaMemcpyHostToDevice
    ));

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto launchReduction = [&]() {
        eren::reduceSum<<<initialBlocks, threads>>>(dInput, dBufferA, N);

        int count = initialBlocks;
        float* current = dBufferA;
        float* next = dBufferB;

        while (count > 1) {
            const int nextCount = ceilDiv(count, threads);
            eren::reducePartialSums<<<nextCount, threads>>>(
                current, next, count
            );

            count = nextCount;
            std::swap(current, next);
        }
    };

    for (int i = 0; i < kWarmupRuns; ++i) {
        launchReduction();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < kTimedRuns; ++i) {
        launchReduction();
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

    const double averageUs =
        static_cast<double>(elapsedMs) * 1000.0 / kTimedRuns;

    float result = 0.0f;
    int count = initialBlocks;
    float* current = dBufferA;
    float* next = dBufferB;

    while (count > 1) {
        std::swap(current, next);
        count = ceilDiv(count, threads);
    }

    // The final output buffer depends on the number of reduction stages.
    // Re-run once and copy the final result for a correctness check.
    launchReduction();

    count = initialBlocks;
    current = dBufferA;
    next = dBufferB;
    while (count > 1) {
        std::swap(current, next);
        count = ceilDiv(count, threads);
    }

    CUDA_CHECK(cudaMemcpy(&result, current, sizeof(float), cudaMemcpyDeviceToHost));

    const double cpuResult = std::accumulate(
        input.begin(), input.end(), 0.0,
        [](double sum, float value) { return sum + value; }
    );

    const double error = std::abs(static_cast<double>(result) - cpuResult);
    const double tolerance = 1e-5 + 1e-6 * std::abs(cpuResult);

    std::cout << std::setw(10) << N
              << std::setw(12) << initialBlocks
              << std::setw(18) << std::fixed << std::setprecision(3)
              << averageUs
              << std::setw(14) << (error <= tolerance ? "PASS" : "FAIL")
              << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dInput));
    CUDA_CHECK(cudaFree(dBufferA));
    CUDA_CHECK(cudaFree(dBufferB));

    if (error > tolerance) {
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main() {
    int device = 0;
    cudaDeviceProp properties{};

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    std::cout << "EREN Reduction Benchmark\n"
              << "GPU: " << properties.name << "\n\n"
              << std::setw(10) << "N"
              << std::setw(12) << "Blocks"
              << std::setw(18) << "Avg time (us)"
              << std::setw(14) << "Correctness" << '\n';

    for (int N : {256, 4096, 65536, 1000003}) {
        runBenchmark(N);
    }

    return EXIT_SUCCESS;
}
