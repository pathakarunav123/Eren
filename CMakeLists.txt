

cmake_minimum_required(VERSION 3.20)

project(EREN LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Reusable EREN core library
add_library(eren_core STATIC
    kernels/reduction_kernels.cu
)

target_include_directories(eren_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(eren_core
    PUBLIC CUDA::cudart
)

# GPU capability report CLI
add_executable(eren src/main.cpp)

target_link_libraries(eren
    PRIVATE CUDA::cudart
)

# CUDA vector addition example
add_executable(vector_add kernels/vectorAdd.cu)

target_include_directories(vector_add
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(vector_add
    PRIVATE CUDA::cudart
)

# Reduction correctness executable
add_executable(reduction kernels/reduction.cu)

target_link_libraries(reduction
    PRIVATE eren_core
)

# Reduction benchmark
add_executable(reduction_bench bench/reduction_bench.cu)

target_link_libraries(reduction_bench
    PRIVATE eren_core
)

# Smoke tests
enable_testing()

add_test(NAME gpu_report COMMAND eren)
add_test(NAME vector_add_correctness COMMAND vector_add)
add_test(NAME reduction_small COMMAND reduction 257)
add_test(NAME reduction_large COMMAND reduction 1000003)
