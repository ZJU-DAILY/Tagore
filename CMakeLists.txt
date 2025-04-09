cmake_minimum_required(VERSION 3.10)
project(Tagore_test LANGUAGES CXX CUDA)

set(pybind11_DIR "/usr/local/python3.10/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_FLAGS -O3 -fopenmp)
set(CUDA_NVCC_FLAGS -O3 -fopenmp)

set(dso_SOURCE_FILES
 ${PROJECT_SOURCE_DIR}/src/common.cu
 ${PROJECT_SOURCE_DIR}/src/Tagore_src.cu
 ${PROJECT_SOURCE_DIR}/src/large_index.cu
)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)
pybind11_add_module(Tagore pybind.cu ${dso_SOURCE_FILES})

set_target_properties(Tagore PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
