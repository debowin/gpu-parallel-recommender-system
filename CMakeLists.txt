cmake_minimum_required(VERSION 3.5)
project(parallel_recommenders)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        main.cpp
        ratings_util.cpp
        ratings_util.h
        recommendations_gold.cpp
        recommendations_gold.h
        misc_utls.h
        misc_utils.cpp)

#find_package(CUDA REQUIRED)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --ptxas-options=-v -I/usr/local/cuda/include -gencode=arch=compute_60,code=\"sm_60\")
#OPTION(DBG "Debug Mode" ON)
#IF (DBG)
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G)
#ELSE ()
#    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)
#ENDIF (DBG)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

add_executable(parallel_recommenders ${SOURCE_FILES})