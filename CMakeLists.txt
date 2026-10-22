
cmake_minimum_required(VERSION 3.20)

project(EREN LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Existing C++ executable
add_executable(eren src/main.cpp)
target_link_libraries(eren PRIVATE CUDA::cudart)
# First CUDA kernel executable
add_executable(vector_add kernels/vectorAdd.cu)
target_include_directories(vector_add PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# CUDA reduction baseline
add_executable(reduction
    kernels/reduction.cu
    kernels/reduction_kernels.cu
)
target_include_directories(reduction PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Reduction benchmark executable
add_executable(reduction_bench
    bench/reduction_bench.cu
    kernels/reduction_kernels.cu
)
target_include_directories(reduction_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# Testing
enable_testing()

add_test(
    NAME gpu_report
    COMMAND eren
)

add_test(
    NAME vector_add_correctness
    COMMAND vector_add
)

add_test(
    NAME reduction_small
    COMMAND reduction 257
)

add_test(
    NAME reduction_large
    COMMAND reduction 1000003
)
