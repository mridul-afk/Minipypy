cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(_C LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 1. Locate System & Package Dependencies
set(Python_FIND_VIRTUALENV FIRST)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(CUDAToolkit REQUIRED)

# 2. Locate pybind11 from the active Python environment
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

# Support common NVIDIA GPU architectures:
# 75 = Turing
# 80 = Ampere data center
# 86 = Ampere consumer GPUs, e.g. RTX 30 series
# 89 = Ada GPUs, e.g. RTX 40 series
set_target_properties(minitorch_cuda_engine PROPERTIES
  CUDA_ARCHITECTURES "75;80;86;89"
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
if(MSVC)
  target_compile_options(minitorch_cuda_engine PRIVATE
    -O3
    --use_fast_math
    -Xcompiler /wd4819
  )

  target_compile_options(_C PRIVATE
    /O2
    /W3
  )
else()
  target_compile_options(minitorch_cuda_engine PRIVATE
    -O3
    --use_fast_math
  )

  target_compile_options(_C PRIVATE
    -O3
    -Wall
  )
endif()

# 7. Install the compiled Python extension into the minipypy package
install(TARGETS _C
  LIBRARY DESTINATION minipypy
  RUNTIME DESTINATION minipypy
)
