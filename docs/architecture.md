# EREN Architecture

## Repository Layout
- src/: C++ host code
- include/eren/: public headers
- kernels/: custom CUDA kernels
- tests/: correctness and regression tests
- bench/: performance benchmarks
- docs/: engineering documentation
- models/: local model weights, excluded from Git

## Design Principles
- EREN owns its inference flow and KV-cache logic.
- Custom operators are validated against reference outputs.
- cuBLAS provides a matrix multiplication baseline.
- cuDNN is explored in the CNN/image-processing track.
- Every optimization requires correctness tests and benchmarks.

## Pending Decisions
- Tensor and device-buffer abstractions
- Weight file format
- CUDA allocator
- Supported model architecture
