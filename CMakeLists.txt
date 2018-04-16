cmake_minimum_required(VERSION 3.0)
project(FluidSim_SPH)

option(BUILD_CUDA "Build with CUDA support" on)

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Ox")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include/)

if(BUILD_CUDA)
enable_language(CUDA)
SET(SOURCE
  ${CMAKE_SOURCE_DIR}/src/fluidsim_system_gpu.cu
	${CMAKE_SOURCE_DIR}/src/fluidsim_marchingcube.cpp
	${CMAKE_SOURCE_DIR}/src/fluidsim_timer.cpp
)
SET(HEADERS
	${CMAKE_SOURCE_DIR}/include/fluidsim_system_gpu.cuh
	${CMAKE_SOURCE_DIR}/include/fluidsim_marchingcube.h
	${CMAKE_SOURCE_DIR}/include/fluidsim_marchingcube_list.h
	${CMAKE_SOURCE_DIR}/include/fluidsim_timer.h
)
add_library(fluidsim_sph_gpu ${SOURCE} ${HEADERS})
else()
find_package(EIGEN3 REQUIRED)
if(EIGEN3_FOUND)
	message("EIGEN3 found")
endif()
SET(SOURCE
  ${CMAKE_SOURCE_DIR}/src/fluidsim_system.cpp
  ${CMAKE_SOURCE_DIR}/src/fluidsim_marchingcube.cpp
	${CMAKE_SOURCE_DIR}/src/fluidsim_timer.cpp
)
SET(HEADERS
  ${CMAKE_SOURCE_DIR}/include/fluidsim_system.h
  	${CMAKE_SOURCE_DIR}/include/fluidsim_types.h
	${CMAKE_SOURCE_DIR}/include/fluidsim_marchingcube.h
	${CMAKE_SOURCE_DIR}/include/fluidsim_marchingcube_list.h
	${CMAKE_SOURCE_DIR}/include/fluidsim_timer.h
)
add_library(fluidsim_sph ${SOURCE} ${HEADERS})
target_include_directories(fluidsim_sph PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

add_executable(dambreak ${CMAKE_SOURCE_DIR}/src/main.cpp)
if(BUILD_CUDA)
add_definitions(-DBUILD_CUDA)
target_link_libraries(dambreak fluidsim_sph_gpu)
else()
target_link_libraries(dambreak fluidsim_sph)
endif()

if(EIGEN3_FOUND)
else()
	find_package(EIGEN3 REQUIRED)
	if(EIGEN3_FOUND)
		message("EIGEN3 found")
	endif()
endif()
target_include_directories(dambreak PRIVATE ${EIGEN3_INCLUDE_DIR})
