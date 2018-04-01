cmake_minimum_required(VERSION 3.0)

project(FluidSim_SPH)

find_package(EIGEN3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/include/)

add_executable(FluiSim_SPH 
	${CMAKE_SOURCE_DIR}/src/main.cpp
	${CMAKE_SOURCE_DIR}/src/fluidsim_system.cpp
	${CMAKE_SOURCE_DIR}/src/fluidsim_timer.cpp)

