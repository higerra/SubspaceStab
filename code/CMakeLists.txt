cmake_minimum_required(VERSION 3.0)

project(SubspaceStab)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wno-deprecated-declarations -Wno-deprecated-register")
set(CMAKE_BUILD_TYPE Release)
set(BUILD_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")

include_directories(/usr/local/include/eigen3)
unset(OPENMP_FOUND)
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  message("-- Found OpenMP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(SubspaceStab main.cpp tracking.h tracking.cpp warping.h warping.cpp)

target_link_libraries(SubspaceStab ${OpenCV_LIBS})