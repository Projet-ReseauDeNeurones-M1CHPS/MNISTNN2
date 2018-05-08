cmake_minimum_required(VERSION 3.7.2)

project(projet VERSION 1.0 LANGUAGES CXX)

# compilation c++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_BUILD_TYPE Release)

set(TBB_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/TBB)
set(TBB_INCLUDE ${TBB_INSTALL_DIR}/include)
set (TBB_LIBRARY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/tbb_cmake_build/tbb_cmake_build_subdir_release)
set(CMAKE_PREFIX_PATH ${TBB_LIBRARY_RELEASE})
set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_INSTALL_RPATH ${CMAKE_PREFIX_PATH})


set(EIGEN_INCLUDE lib)

include(${TBB_INSTALL_DIR}/cmake/TBBBuild.cmake)
tbb_build(TBB_ROOT ${TBB_INSTALL_DIR} CONFIG_DIR TBB_DIR)
find_package(TBB REQUIRED)

find_library(TBB_LIBRARIES tbb)
link_directories(${TBB_LIBRARIES})

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall" )

add_executable(MNISTNN
  src/main.cpp
  )
link_directories(${TBB_LIBRARIES})
target_include_directories(MNISTNN PRIVATE ${EIGEN_INCLUDE})
target_include_directories(MNISTNN PRIVATE ${TBB_INCLUDE})
target_link_libraries(MNISTNN ${TBB_LIBRARIES})
  
install (TARGETS MNISTNN DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin)

