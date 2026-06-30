cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(_C LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 1. Locate System & Package Dependencies
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(CUDAToolkit REQUIRED)

# 2. Locate Pybind11 inside your local .venv
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# 3. Create the isolated CUDA Static Engine
add_library(minitorch_cuda_engine STATIC
  src/tensor.cu
)

set_target_properties(minitorch_cuda_engine PROPERTIES
  CUDA_ARCHITECTURES "86;89"
)

# 4. Create the Python C++ Extension Module
pybind11_add_module(_C
  src/bindings.cpp
  src/tensor.cpp
  src/memory_pool.cpp
)

target_include_directories(_C PRIVATE
  ${CMAKE_SOURCE_DIR}/src
)

# 5. Link everything together
target_link_libraries(_C PRIVATE
  minitorch_cuda_engine
  CUDA::cudart
)

# 6. Apply target-specific compile flags cleanly
target_compile_options(minitorch_cuda_engine PRIVATE
  -O3
  --use_fast_math
  -Xcompiler /wd4819
)

target_compile_options(_C PRIVATE
  /O2
  /W3
)

# Tells CMake where to send the final file when you run the install command
install(TARGETS _C
  LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/minipypy"
  RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/minipypy"
)
