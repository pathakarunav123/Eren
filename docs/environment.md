
# Development Environment

## Hardware
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- VRAM: 8188 MiB(physical specification)
- Reported Global Memory: 8187 MiB

- Compute Capability: 8.9

- Streaming Multiprocessors (SMs): 24

- Shared Memory per Block: 48 KiB

- Maximum Threads per Block: 1024

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
- g++: 15.2.0
- CMake: 4.2.3
- Git: 2.53.0
- CUDA compiler: nvcc 13.2.86
- CUDA Runtime reported by EREN: 13.2

## GPU
- nvidia-smi works inside Ubuntu.
- CUDA compilation is pending verification.
- EREN successfully detects the NVIDIA GeForce RTX 4060 Laptop GPU.

- GPU capability report is implemented in src/main.cpp.

- Vector addition has previously passed its correctness test.

- Multi-stage reduction implementation is present; verify its latest version against multiple input sizes.
- Multi-stage reduction has passed CPU-reference checks for input sizes 1, 255, 256, 257, and 1,000,003.
- Reduction benchmark correctness passed for input sizes 256, 4,096, 65,536, and 1,000,003.
- nitial benchmark timings are recorded in docs/progress.md; they are baseline measurements, not optimized performance claims.

## CMake Targets
- eren: GPU capability report.
- vector_add: CUDA vector addition.
- reduction: Multi-stage reduction with CPU-reference validation.
- reduction_bench: Reduction benchmark using CUDA events.

## Build Commands
- Configure the CUDA project:

- cmake -S . -B build-cuda

- Build all configured targets:

- cmake --build build-cuda -j2

- Run the GPU capability report:

- ./build-cuda/eren

- Run vector addition:

- ./build-cuda/vector_add

- Run reduction with an input size:

- ./build-cuda/reduction 1000

## Notes
- WSL2 GPU integration uses the Windows NVIDIA driver.

- Do not install a separate Linux NVIDIA display driver in WSL.

- Windows and WSL use different CUDA Toolkit/compiler versions; the Linux build uses the WSL-installed CUDA Toolkit.

- Record future toolchain changes here.
