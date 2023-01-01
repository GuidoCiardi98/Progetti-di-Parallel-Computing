cmake_minimum_required(VERSION 3.23)
project(Parallel_K_means C)

set(CMAKE_CXX_STANDARD 11)

# openMP configuration
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp ")

set(SOURCE_FILES main.c)

add_executable(K_meansPar ${SOURCE_FILES})

