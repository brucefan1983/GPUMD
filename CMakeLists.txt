cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA compute capability (e.g. native, 75, 80, 90)")
project(GPUMD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# MSVC host compiler flags: /utf-8 for Unicode source, /Zc:preprocessor for standards-conforming preprocessor.
if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /utf-8 -Xcompiler /Zc:preprocessor")
endif()

# --- gpumd ---
file(GLOB GPUMD_SOURCES
    src/main_gpumd/*.cu
    src/minimize/*.cu
    src/phonon/*.cu
    src/integrate/*.cu
    src/mc/*.cu
    src/force/*.cu
    src/measure/*.cu
    src/model/*.cu
    src/utilities/*.cu)
add_executable(gpumd ${GPUMD_SOURCES})
target_include_directories(gpumd PRIVATE src)
target_link_libraries(gpumd PRIVATE CUDA::cublas CUDA::cusolver CUDA::cufft)

# --- nep ---
file(GLOB NEP_SOURCES
    src/main_nep/*.cu
    src/utilities/*.cu)
add_executable(nep ${NEP_SOURCES})
target_include_directories(nep PRIVATE src)
target_link_libraries(nep PRIVATE CUDA::cublas CUDA::cusolver CUDA::cufft)

# --- gnep (optional, build with: cmake --build . --target gnep) ---
file(GLOB GNEP_SOURCES
    src/main_gnep/*.cu
    src/utilities/*.cu)
add_executable(gnep EXCLUDE_FROM_ALL ${GNEP_SOURCES})
target_include_directories(gnep PRIVATE src)
target_link_libraries(gnep PRIVATE CUDA::cublas CUDA::cusolver CUDA::cufft)
