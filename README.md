# Project EREN

A build-first C++/CUDA LLM inference engine.

## Mission
Build a working GPU inference engine incrementally.
Validate correctness, measure performance, and document decisions.

## Development Environment
- OS: Windows 11 with WSL2 and Ubuntu
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Language: C++17 and CUDA C++
- Build system: CMake
- Version control: Git

## Build
cmake -S . -B build
cmake --build build
./build/eren

## Engineering Principles
1. Correctness before performance.
2. Test custom operators against trusted references.
3. Benchmark performance reproducibly.
4. Document architecture and design decisions.

## Current Status
Phase 0: Environment and C++/CUDA foundation.
