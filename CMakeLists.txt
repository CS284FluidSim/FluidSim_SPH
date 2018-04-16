cmake_minimum_required(VERSION 3.0)
project(FluidSim_SPH)

option(BUILD_CUDA "Build with CUDA support" on)

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Ox")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include/)

if(BUILD_CUDA)
find_package(CUDA REQUIRED)
cuda_add_library(fluidsim_sph_gpu
	${CMAKE_SOURCE_DIR}/src/fluidsim_system_gpu.cu
	${CMAKE_SOURCE_DIR}/src/fluidsim_timer.cpp)
else()
add_library(fluidsim_sph
	${CMAKE_SOURCE_DIR}/src/fluidsim_system.cpp
	${CMAKE_SOURCE_DIR}/src/fluidsim_timer.cpp)
target_include_directories(fluidsim_sph PRIVATE D:/Ironman/eigen-eigen-5a0156e40feb/eigen-eigen-5a0156e40feb)
endif()

add_executable(dambreak ${CMAKE_SOURCE_DIR}/src/main.cpp)
if(BUILD_CUDA)
add_definitions(-DBUILD_CUDA)
target_link_libraries(dambreak fluidsim_sph_gpu)
else()
target_link_libraries(dambreak fluidsim_sph)
endif()

target_include_directories(dambreak PRIVATE D:/Ironman/eigen-eigen-5a0156e40feb/eigen-eigen-5a0156e40feb)
