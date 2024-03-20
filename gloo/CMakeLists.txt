cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(gloo CXX C)

set(GLOO_VERSION_MAJOR 0)
set(GLOO_VERSION_MINOR 5)
set(GLOO_VERSION_PATCH 0)
set(GLOO_VERSION
    "${GLOO_VERSION_MAJOR}.${GLOO_VERSION_MINOR}.${GLOO_VERSION_PATCH}")

# Gloo assumes 64-bit and doesn't run builds/tests for anything else.
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Gloo can only be built on 64-bit systems.")
endif()

# Local CMake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Build target options
option(BUILD_TEST "Build test binary (requires gtest)" OFF)
option(BUILD_BENCHMARK "Build benchmark binary (requires hiredis)" OFF)

# Option defaults (so they can be overwritten before declaring the option)
set(USE_REDIS_DEFAULT OFF)
set(USE_IBVERBS_DEFAULT OFF)
set(USE_NCCL_DEFAULT OFF)
set(USE_RCCL_DEFAULT OFF)
set(USE_LIBUV_DEFAULT OFF)
set(USE_SHARP_DEFAULT OFF)
set(USE_GLEX_DEFAULT OFF)
set(USE_GLEX_RDMA_T_DEFAULT OFF)
#yed
set(USE_PIPE_DEFAULT OFF)


# Options
option(USE_REDIS "Support using Redis for rendezvous" ${USE_REDIS_DEFAULT})
option(USE_IBVERBS "Support ibverbs transport" ${USE_IBVERBS_DEFAULT})
option(USE_NCCL "Support using NCCL for local collectives" ${USE_NCCL_DEFAULT})
option(USE_RCCL "Support using RCCL for local collectives" ${USE_RCCL_DEFAULT})
option(USE_LIBUV "Build libuv transport" ${USE_LIBUV_DEFAULT})
option(USE_SHARP "Build sharp transport" ${USE_SHARP_DEFAULT})
option(USE_GLEX "Build glex transport" ${USE_GLEX_DEFAULT})
option(USE_GLEX_RDMA_T "Build glex_rdma_t transport" ${USE_GLEX_TDMA_T_DEFAULT})
#yed
option(USE_PIPE "Build pipe transport" ${USE_PIPE_DEFAULT})

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set -- defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Use CMAKE_<LANG>_COMPILER_LAUNCHER if available.
find_program(SCCACHE_EXECUTABLE sccache)
mark_as_advanced(SCCACHE_EXECUTABLE)
if(SCCACHE_EXECUTABLE)
  foreach(LANG CXX C)
    if(NOT DEFINED CMAKE_${LANG}_COMPILER_LAUNCHER AND NOT CMAKE_${LANG}_COMPILER MATCHES ".*/s?ccache$")
      message(STATUS "Enabling sccache for ${LANG}")
      set(CMAKE_${LANG}_COMPILER_LAUNCHER ${SCCACHE_EXECUTABLE} CACHE STRING "")
    endif()
  endforeach()
endif()

# Define sanitizer option, if specified.
if(SANITIZE)
  add_definitions("-fsanitize=${SANITIZE}")
  add_definitions("-fno-omit-frame-pointer")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=${SANITIZE}")
endif()

# Add install targets by default (override from parent project)
set(GLOO_INSTALL ON CACHE BOOL "")
mark_as_advanced(GLOO_INSTALL)

# Build shared or static libraries (override from parent project)
if(BUILD_SHARED_LIBS)
  set(GLOO_STATIC_OR_SHARED SHARED CACHE STRING "")
else()
  set(GLOO_STATIC_OR_SHARED STATIC CACHE STRING "")
endif()
mark_as_advanced(GLOO_STATIC_OR_SHARED)

# Process dependencies
include(cmake/Dependencies.cmake)
# Use project root as default include directory
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR})

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

#yed
if(USE_PIPE)
  include_directories(/home/mpi_share/env/mlnx_sharp/hpcx-v2.7.0-gcc-MLNX_OFED_LINUX-5.1-0.6.6.0-redhat7.8-x86_64/sharp/include/)
  link_directories(/home/mpi_share/env/mlnx_sharp/hpcx-v2.7.0-gcc-MLNX_OFED_LINUX-5.1-0.6.6.0-redhat7.8-x86_64/sharp/lib/)
endif()

if(USE_SHARP)
  include_directories(/home/mpi_share/env/mlnx_sharp/hpcx-v2.7.0-gcc-MLNX_OFED_LINUX-5.1-0.6.6.0-redhat7.8-x86_64/sharp/include/)
  link_directories(/home/mpi_share/env/mlnx_sharp/hpcx-v2.7.0-gcc-MLNX_OFED_LINUX-5.1-0.6.6.0-redhat7.8-x86_64/sharp/lib/)
endif()


if(USE_GLEX)
  add_compile_options(-DGLOO_HAVE_TRANSPORT_GLEX)
endif()

if(USE_GLEX_RDMA_T)
  add_compile_options(-DGLOO_HAVE_TRANSPORT_GLEX_RDMA_T)
endif()

if(USE_GLEX_RDMA_S)
  add_compile_options(-DGLOO_HAVE_TRANSPORT_GLEX_RDMA_S)
endif()

# Recurse into main project directory
add_subdirectory(gloo)

# Finally, create the cmake configuration files.
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/GlooConfigVersion.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/GlooConfigVersion.cmake
  @ONLY)
configure_file(
  ${PROJECT_SOURCE_DIR}/cmake/GlooConfig.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/GlooConfig.cmake
  @ONLY)

if(GLOO_INSTALL)
  install(FILES
    ${PROJECT_BINARY_DIR}/cmake/GlooConfig.cmake
    ${PROJECT_BINARY_DIR}/cmake/GlooConfigVersion.cmake
    DESTINATION share/cmake/Gloo
    COMPONENT dev)
  install(EXPORT GlooTargets DESTINATION share/cmake/Gloo
    FILE GlooTargets.cmake
    COMPONENT dev)
endif()
