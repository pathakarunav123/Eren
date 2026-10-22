
#include <cuda_runtime.h>

#include <iostream>
#include <string>

int main() {
    const std::string project_name = "EREN";

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "CUDA initialization failed: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable GPU found.\n";
        return 1;
    }

    int device = 0;
    cudaDeviceProp prop{};

    err = cudaGetDeviceProperties(&prop, device);

    if (err != cudaSuccess) {
        std::cerr << "Failed to query GPU: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    int runtimeVersion = 0;
    err = cudaRuntimeGetVersion(&runtimeVersion);

    if (err != cudaSuccess) {
        std::cerr << "Failed to query CUDA Runtime: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << "========== EREN ==========\n";
    std::cout << "Project: " << project_name << '\n';
    std::cout << "C++ environment is working!\n\n";

    std::cout << "========== GPU REPORT ==========\n";
    std::cout << "GPU Name: " << prop.name << '\n';
    std::cout << "Device Count: " << deviceCount << '\n';
    std::cout << "Compute Capability: "
              << prop.major << "." << prop.minor << '\n';
    std::cout << "Multiprocessor (SM) Count: "
              << prop.multiProcessorCount << '\n';
    std::cout << "Global Memory: "
              << prop.totalGlobalMem / (1024 * 1024)
              << " MiB\n";
    std::cout << "Shared Memory per Block: "
              << prop.sharedMemPerBlock / 1024
              << " KiB\n";
    std::cout << "Max Threads per Block: "
              << prop.maxThreadsPerBlock << '\n';
    std::cout << "CUDA Runtime Version: "
              << runtimeVersion / 1000 << "."
              << (runtimeVersion % 1000) / 10 << '\n';

    return 0;
}
