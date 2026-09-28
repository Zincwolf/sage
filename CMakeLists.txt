cmake_minimum_required(VERSION 3.28)
project(weight_vq LANGUAGES CXX CUDA)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
# PyTorch's documented CMake prefix from the isolated build environment.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)
find_package(CUDAToolkit 12.8 EXACT REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib"
  NO_DEFAULT_PATH REQUIRED)

# Private module names avoid colliding with the training dependencies' extensions.
Python_add_library(_C MODULE WITH_SOABI
  csrc/bindings.cpp csrc/sphquant/sph_gemv_cuda.cu
  csrc/sage/sage_gemv_cuda.cu csrc/sage/sage_gemm_cuda.cu)
target_include_directories(_C PRIVATE csrc/sphquant csrc/sage)
# Use the patched submodules as the single source of third-party kernels.
# QTIP's Qwen3-4B entry points are maintained in patches/qtip/.
Python_add_library(_qtip MODULE WITH_SOABI
  third_party/qtip/qtip-kernels/src/wrapper.cpp
  third_party/qtip/qtip-kernels/src/qtip_torch.cu)
# qtip_torch.cu includes inference.cu.
Python_add_library(_unisvq MODULE WITH_SOABI
  third_party/unisvq/hadaquant/csrc/v1_cuda.cpp
  third_party/unisvq/hadaquant/csrc/v1_cuda_kernel.cu)

foreach(target IN ITEMS _C _qtip _unisvq)
  target_link_libraries(${target} PRIVATE ${TORCH_LIBRARIES} "${TORCH_PYTHON_LIBRARY}" CUDA::cudart)
  target_compile_features(${target} PRIVATE cxx_std_17 cuda_std_17)
  target_compile_definitions(${target} PRIVATE TORCH_EXTENSION_NAME=${target})
  set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN/../../torch/lib")
  install(TARGETS ${target} LIBRARY DESTINATION weight_vq/inference)
endforeach()

# Preserve the frozen baseline arithmetic and optimization flags.
target_compile_options(_qtip PRIVATE -O3 $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>)
target_compile_options(_unisvq PRIVATE -O3 $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
# QuIP# and FHT use the existing quiptools-cuda and fast-hadamard-transform
# dependencies; do not build/install second copies.
# CUDA architectures are selected by the caller through CMAKE_CUDA_ARCHITECTURES.
