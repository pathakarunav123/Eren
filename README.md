# Project EREN

A build-first C++/CUDA LLM inference engine built from scratch.

## Mission

Build a minimal, GPU-accelerated LLM inference engine incrementally using C++ and CUDA.

The goal is to understand GPU execution, implement inference operators, validate correctness against trusted references, optimize performance, and document engineering decisions.

## Development Environment

- **OS:** Windows 11 + WSL2 + Ubuntu
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
- **Language:** C++17 and CUDA C++
- **CUDA Toolkit:** 13.2
- **Build System:** CMake
- **Version Control:** Git and GitHub

## Project Structure

```text
EREN/
├── bench/       # Benchmark programs
├── docs/        # Documentation and design decisions
├── include/     # Public headers
├── kernels/     # CUDA kernels
├── src/         # C++ host code and engine components
├── tests/       # Correctness tests
├── CMakeLists.txt
├── README.md
└── .gitignore
```

## Build and Run

Configure the project:

```bash
cmake -S . -B build-phase0
```

Build all targets:

```bash
cmake --build build-phase0 -j2
```

Run the GPU capability report:

```bash
./build-phase0/eren
```

Run the CUDA vector-add example:

```bash
./build-phase0/vector_add
```

Run reduction correctness checks:

```bash
./build-phase0/reduction 257
./build-phase0/reduction 1000003
```

Run the reduction benchmark:

```bash
./build-phase0/reduction_bench
```

Run the registered CTest checks:

```bash
ctest --test-dir build-phase0 --output-on-failure
```

The build directory is generated locally and is excluded from version control by `.gitignore`.


> These commands assume the CUDA Toolkit is installed at `/usr/local/cuda-13.2` inside WSL2.

## Current Progress

### Phase 0 — Environment and C++/CUDA Foundation

**Completed**

- [x] C++17/CUDA project configured with CMake
- [x] WSL2 CUDA development environment and GPU detection
- [x] GPU capability report implemented
- [x] CUDA vector-add kernel implemented and verified
- [x] Reusable CUDA error-checking helper
- [x] Multi-stage CUDA reduction with CPU-reference validation
- [x] Reduction benchmark with CUDA-event timing
- [x] All four CMake targets built successfully
- [x] Reusable `eren_core` static library for reduction kernels

**Pending**

- [ ] Final README and environment documentation review
- [ ] Final source and Git cleanup
- [ ] Final clean-build and correctness verification
- [ ] Initial Phase 0 Git commit and push

The reduction benchmark has passed correctness checks for input sizes 256, 4,096, 65,536, and 1,000,003. Recorded timings are initial baselines, not optimized performance claims.

The detailed Project EREN roadmap remains the source of truth for phase requirements and exit criteria.

## Planned Roadmap

The project roadmap is organized into phases:

1. **Environment and foundation:** C++, CUDA, CMake, testing, and benchmarks.
2. **Python reference model:** Establish a trusted model and golden outputs.
3. **CUDA inference operators:** Implement and validate the model's forward pass.
4. **Generation:** Token generation, sampling, and KV cache.
5. **Modern LLM architecture:** Architecture support and mixed precision.
6. **Profiling and optimization:** Measure bottlenecks and improve performance.
7. **CNN and image processing:** A separate GPU programming track.
8. **Serving and advanced optimization:** Batching, KV-cache improvements, and selected fused kernels.
9. **Triton and library literacy:** Understand where specialized tools and libraries fit.
10. **Deployment:** API, Docker, and AWS.

The detailed project roadmap remains the source of truth for phase requirements and exit criteria.

## Engineering Principles

1. Correctness before performance.
2. Understand the implementation before introducing libraries.
3. Test custom operators against trusted references.
4. Benchmark performance reproducibly.
5. Profile before optimizing.
6. Keep experiments, architecture decisions, and limitations documented.
7. Build incrementally, with verifiable milestones.

## Long-Term Goal

Deliver a minimal, working LLM inference engine that demonstrates GPU kernel implementation, model execution, sampling, performance analysis, and deployment.

The focus is not just making it run, but understanding **why it works, how it performs, and where optimization matters**.
