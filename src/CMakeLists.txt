cmake_minimum_required(VERSION 3.15)

project(GAS LANGUAGES CXX CUDA)

###################################################
# Set the CUDA compiler path
set(CUDACXX /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_COMPILER ${CUDACXX})

# Specify the Python environment
set(PYTHON_INCLUDE /home/tensor/anaconda3/envs/py38/include/python3.8)
set(PYTHON_LIB /home/tensor/anaconda3/envs/py38/lib)
set(PYTHON_LIB_NAME python3.8)

# Set the libtorch path
list(APPEND CMAKE_PREFIX_PATH "/home/tensor/software/libtorch/share/cmake/Torch")
# list(APPEND CMAKE_PREFIX_PATH "/home/tensor/downloads/libtorch/cuda11.8/share/cmake/Torch")

# you may change the path in this zone according to your environment, with mine offered as a reference.
# if use docker, path above should be consistent with the path in docker.
###################################################
# Set the compiler flags
if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_80 -Xcompiler \"/wd 4819\" -I${PYTHON_INCLUDE}")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_80 -I${PYTHON_INCLUDE} -DUSE_GAS -DTORCH_USE_CUDA_DSA=ON")
endif()

set(INC -I./)
set(LIBS -lcublas -lcusolver)

# Find libtorch
find_package(Torch REQUIRED)

# Gather source files
file(GLOB SOURCES_GPUMD
    "${CMAKE_SOURCE_DIR}/main_gpumd/*.cu"
    "${CMAKE_SOURCE_DIR}/minimize/*.cu"
    "${CMAKE_SOURCE_DIR}/phonon/*.cu"
    "${CMAKE_SOURCE_DIR}/integrate/*.cu"
    "${CMAKE_SOURCE_DIR}/mc/*.cu"
    "${CMAKE_SOURCE_DIR}/force/*.cu"
    "${CMAKE_SOURCE_DIR}/measure/*.cu"
    "${CMAKE_SOURCE_DIR}/model/*.cu"
    "${CMAKE_SOURCE_DIR}/utilities/*.cu"
    # "${CMAKE_SOURCE_DIR}/measure/gasCV/*.cu" for libtorch version
)

file(GLOB SOURCES_NEP
    "${CMAKE_SOURCE_DIR}/main_nep/*.cu"
    "${CMAKE_SOURCE_DIR}/utilities/*.cu"
)

# Gather header files
file(GLOB HEADERS
    "${CMAKE_SOURCE_DIR}/utilities/*.cuh"
    "${CMAKE_SOURCE_DIR}/main_gpumd/*.cuh"
    "${CMAKE_SOURCE_DIR}/integrate/*.cuh"
    "${CMAKE_SOURCE_DIR}/mc/*.cuh"
    "${CMAKE_SOURCE_DIR}/minimize/*.cuh"
    "${CMAKE_SOURCE_DIR}/force/*.cuh"
    "${CMAKE_SOURCE_DIR}/measure/*.cuh"
    "${CMAKE_SOURCE_DIR}/model/*.cuh"
    "${CMAKE_SOURCE_DIR}/phonon/*.cuh"
    "${CMAKE_SOURCE_DIR}/main_nep/*.cuh"
    #gas file
    # "${CMAKE_SOURCE_DIR}/measure/gasCV/*.cuh"
)

# Include directories
include_directories(
    ${PYTHON_INCLUDE}
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/main_gpumd
    ${CMAKE_SOURCE_DIR}/minimize
    ${CMAKE_SOURCE_DIR}/phonon
    ${CMAKE_SOURCE_DIR}/integrate
    ${CMAKE_SOURCE_DIR}/mc
    ${CMAKE_SOURCE_DIR}/force
    ${CMAKE_SOURCE_DIR}/measure
    ${CMAKE_SOURCE_DIR}/model
    ${CMAKE_SOURCE_DIR}/utilities
    ${CMAKE_SOURCE_DIR}/main_nep
    #gas file
    # ${CMAKE_SOURCE_DIR}/measure/gasCV
)

# Create executables
add_executable(gpumd ${SOURCES_GPUMD})
add_executable(nep ${SOURCES_NEP})

# Link libraries for gpumd
target_link_libraries(gpumd
    ${TORCH_LIBRARIES}
    ${LIBS}
    -L${PYTHON_LIB}
    -l${PYTHON_LIB_NAME}
    -static-libgcc
    -static-libstdc++
)

# Link libraries for nep
target_link_libraries(nep
    ${TORCH_LIBRARIES}
    ${LIBS}
    -L${PYTHON_LIB}
    -l${PYTHON_LIB_NAME}
    -static-libgcc
    -static-libstdc++
)

# Ensure include directories are set correctly for both targets
target_include_directories(gpumd PRIVATE ${PYTHON_INCLUDE} ${INC} ${TORCH_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})
target_include_directories(nep PRIVATE ${PYTHON_INCLUDE} ${INC} ${TORCH_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})

# Set C++ standard
set_property(TARGET gpumd PROPERTY CXX_STANDARD 17)
set_property(TARGET nep PROPERTY CXX_STANDARD 17)

# Clean up
add_custom_target(clean_all
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/clean.cmake
)

# Create a custom clean command
file(WRITE ${CMAKE_BINARY_DIR}/clean.cmake "
    file(GLOB_RECURSE OBJECT_FILES \"*.o\" \"*.obj\")
    list(LENGTH OBJECT_FILES OBJECT_FILES_COUNT)
    if(OBJECT_FILES_COUNT GREATER 0)
        file(REMOVE ${OBJECT_FILES})
    endif()
    file(GLOB EXECUTABLES \"gpumd\" \"nep\")
    list(LENGTH EXECUTABLES EXECUTABLES_COUNT)
    if(EXECUTABLES_COUNT GREATER 0)
        file(REMOVE ${EXECUTABLES})
    endif()
")
