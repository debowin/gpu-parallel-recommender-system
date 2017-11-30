cmake_minimum_required(VERSION 3.5)
project(parallel_recommenders LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        main.cpp
        ratings_util.cpp
        ratings_util.h
        similarity_gold.cpp similarity_gold.h misc_utls.h misc_utils.cpp)

find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --ptxas-options=-v -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\")
OPTION(DBG "Debug Mode" ON)
IF(DBG)
    set(NVCC_FLAGS ${NVCC_FLAGS} -g -G)
ELSE()
    set(NVCC_FLAGS ${NVCC_FLAGS} -O3)
ENDIF(DBG)

cuda_add_executable(parallel_recommenders ${SOURCE_FILES})