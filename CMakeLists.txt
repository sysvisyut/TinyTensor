cmake_minimum_required(VERSION 3.20)
project(tinytensor VERSION 0.1.0 LANGUAGES C CXX)

# Enforce C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for language servers (clangd)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Configuration Options
option(TINYTENSOR_ENABLE_CUDA "Enable CUDA backend and GPU runtime" OFF)
option(TINYTENSOR_ENABLE_LLVM "Enable LLVM dependency discovery" ON)
option(TINYTENSOR_BUILD_TESTS "Build test suite" ON)

# Strict warning interface target
add_library(tinytensor_warnings INTERFACE)
target_compile_options(tinytensor_warnings INTERFACE
  $<$<CXX_COMPILER_ID:MSVC>:
    /W4 /WX /permissive-
  >
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wconversion
    -Wsign-conversion
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
  >
)

# -----------------------------------------------------------------------------
# Toolchain Discovery: LLVM
# -----------------------------------------------------------------------------
set(TINYTENSOR_LLVM_FOUND FALSE)
set(TINYTENSOR_LLVM_HAS_NVPTX FALSE)

if(TINYTENSOR_ENABLE_LLVM)
  # Look in common Homebrew keg paths on macOS if LLVM_DIR is unset
  if(APPLE AND NOT DEFINED LLVM_DIR)
    if(EXISTS "/opt/homebrew/opt/llvm/lib/cmake/llvm")
      set(LLVM_DIR "/opt/homebrew/opt/llvm/lib/cmake/llvm")
    elseif(EXISTS "/opt/homebrew/opt/llvm@18/lib/cmake/llvm")
      set(LLVM_DIR "/opt/homebrew/opt/llvm@18/lib/cmake/llvm")
    elseif(EXISTS "/usr/local/opt/llvm/lib/cmake/llvm")
      set(LLVM_DIR "/usr/local/opt/llvm/lib/cmake/llvm")
    endif()
  endif()

  find_package(LLVM CONFIG QUIET)
  if(LLVM_FOUND)
    set(TINYTENSOR_LLVM_FOUND TRUE)
    if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
      set(TINYTENSOR_LLVM_HAS_NVPTX TRUE)
    endif()
  endif()
endif()

# -----------------------------------------------------------------------------
# Toolchain Discovery: CUDA
# -----------------------------------------------------------------------------
set(TINYTENSOR_CUDA_FOUND FALSE)

if(TINYTENSOR_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    set(TINYTENSOR_CUDA_FOUND TRUE)
  else()
    message(WARNING "TINYTENSOR_ENABLE_CUDA=ON, but no CUDA compiler (nvcc) was found. Disabling CUDA runtime.")
    set(TINYTENSOR_ENABLE_CUDA OFF)
  endif()
endif()

# -----------------------------------------------------------------------------
# Configuration Summary
# -----------------------------------------------------------------------------
message(STATUS "========================================================")
message(STATUS " TinyTensor Configuration Summary")
message(STATUS "========================================================")
message(STATUS " C++ Standard:       ${CMAKE_CXX_STANDARD}")
message(STATUS " C++ Compiler:       ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
if(TINYTENSOR_LLVM_FOUND)
  message(STATUS " LLVM:               Found ${LLVM_PACKAGE_VERSION} (${LLVM_DIR})")
  message(STATUS "   NVPTX Target:     ${TINYTENSOR_LLVM_HAS_NVPTX}")
  message(STATUS "   Targets to Build: ${LLVM_TARGETS_TO_BUILD}")
else()
  message(STATUS " LLVM:               Not found (backend code emission disabled)")
endif()
if(TINYTENSOR_CUDA_FOUND)
  message(STATUS " CUDA:               Enabled (${CMAKE_CUDA_COMPILER})")
else()
  message(STATUS " CUDA:               Disabled (CPU-only build mode)")
endif()
message(STATUS " Build Tests:        ${TINYTENSOR_BUILD_TESTS}")
message(STATUS "========================================================")

# -----------------------------------------------------------------------------
# Main Library Target
# -----------------------------------------------------------------------------
add_library(tinytensor
  src/tinytensor.cpp
  src/ir.cpp
)

target_link_libraries(tinytensor
  PRIVATE
    tinytensor_warnings
)

target_include_directories(tinytensor
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# -----------------------------------------------------------------------------
# Test Suite
# -----------------------------------------------------------------------------
if(TINYTENSOR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
