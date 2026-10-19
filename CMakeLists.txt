
cmake_minimum_required(VERSION 3.20)

project(EREN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Existing C++ executable
add_executable(eren src/main.cpp)

# First CUDA kernel executable
add_executable(vector_add kernels/vectorAdd.cu)



