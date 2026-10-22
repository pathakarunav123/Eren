# EREN Architecture

## Repository Layout
- `src/`: C++ host code and GPU capability report CLI
- `include/eren/`: Public headers and reusable CUDA helpers
- `kernels/`: Custom CUDA kernels and reduction correctness executable
- `bench/`: Performance benchmark programs
- `tests/`: Reserved for future standalone correctness and regression tests
- `docs/`: Engineering documentation and design decisions
- `models/`: Local model weights, excluded from Git
- `build-phase0/`: Local CMake build output, excluded from Git

## Current Build Targets

- `eren_core`: Static library containing reusable reduction kernel implementations
- `eren`: GPU capability report executable
- `vector_add`: CUDA vector-add correctness example
- `reduction`: Multi-stage reduction correctness executable, validated against a CPU reference
- `reduction_bench`: Reduction benchmark using CUDA events

The reduction executable and benchmark link against `eren_core`, so they share the same kernel implementation. The GPU report and vector-add targets link against the CUDA Runtime directly.

## Design Principles
- EREN owns its inference flow and KV-cache logic.
- Custom operators are validated against reference outputs.
- cuBLAS provides a matrix multiplication baseline.
- cuDNN is explored in the CNN/image-processing track.
- Every optimization requires correctness tests and benchmarks.
- Correctness comes before performance optimization.
- Build and benchmark results should be reproducible and documented.

## Pending Decisions
- Tensor and device-buffer abstractions
- Weight file format
- CUDA allocator
- Inference operator and runtime API boundaries
