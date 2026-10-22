
# Development Environment

## Hardware

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Physical VRAM specification: 8188 MiB
- Reported global memory: 8187 MiB
- Compute capability: 8.9
- Streaming multiprocessors (SMs): 24
- Shared memory per block: 48 KiB
- Maximum threads per block: 1024

## Operating System

- Host: Windows 11
- Development: WSL2 + Ubuntu
- WSL version: 2

## Windows Toolchain

- NVIDIA driver: 595.97
- CUDA compiler: nvcc 13.3.73
- MSVC: 19.51
- CMake: 4.3.1-msvc1
- Git: 2.50.1.windows.1

## Linux Toolchain

- C++ compiler: g++ 15.2.0
- CMake: 4.2.3
- Git: 2.53.0
- CUDA compiler: nvcc 13.2.86
- CUDA Runtime reported by EREN: 13.2

## Verification Status

- NVIDIA GPU is visible inside Ubuntu; `nvidia-smi` works.
- CUDA compilation and CMake builds have been verified in WSL2.
- EREN detects the NVIDIA GeForce RTX 4060 Laptop GPU.
- GPU capability reporting is implemented in `src/main.cpp`.
- Vector addition has passed its correctness check.
- Multi-stage reduction has passed CPU-reference checks for input sizes 1, 255, 256, 257, and 1,000,003.
- Reduction benchmark correctness has passed for input sizes 256, 4,096, 65,536, and 1,000,003.
- The four registered CTest checks have passed in the Phase 0 build.

## CMake Targets

- `eren_core`: Static library containing reusable reduction kernels.
- `eren`: GPU capability report.
- `vector_add`: CUDA vector-add correctness example.
- `reduction`: Multi-stage reduction with CPU-reference validation.
- `reduction_bench`: Reduction benchmark using CUDA events.

## Build Commands

Configure:

```bash
cmake -S . -B build-phase0
```

Build:

```bash
cmake --build build-phase0 -j2
```

Run the GPU capability report:

```bash
./build-phase0/eren
```

Run vector addition:

```bash
./build-phase0/vector_add
```

Run reduction:

```bash
./build-phase0/reduction 1000
```

Run the benchmark:

```bash
./build-phase0/reduction_bench
```

Run registered tests:

```bash
ctest --test-dir build-phase0 --output-on-failure
```

## Notes

- WSL2 GPU integration uses the Windows NVIDIA driver.
- Do not install a separate Linux NVIDIA display driver in WSL2.
- Windows and WSL2 use different CUDA Toolkit/compiler versions. The Linux build uses the CUDA Toolkit installed inside WSL2.
- Benchmark timings are initial baselines, not optimized performance claims. See `docs/progress.md`.
- Record future toolchain changes here.
