cmake_minimum_required(VERSION 3.0)

project(SubspaceStab)

find_package(OpenCV REQUIRED)
find_package(gflags REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -Wno-deprecated-declarations -Wno-deprecated-register")
set(CMAKE_BUILD_TYPE Release)
set(BUILD_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_DIRECTORY})

include_directories(/usr/local/include/eigen3 /usr/include/suitesparse)
link_directories(/usr/local/lib /usr/lib/x86_64-linux-gnu/)

unset(OPENMP_FOUND)
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  message("-- Found OpenMP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(SubspaceStab main.cpp tracking.h tracking.cpp factorization.cpp factorization.h warping.h warping.cpp gridenergy.h utility.h utility.cpp thread_guard.h)

target_link_libraries(SubspaceStab ${OpenCV_LIBS} glog gflags spqr cholmod)
