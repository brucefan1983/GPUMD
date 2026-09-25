cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA compute capability (e.g. native, 75, 80, 90)")
project(GPUMD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

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

option(GPUMD_WPE "Enable Wisevolve Potential Engine (WPE) integration" OFF)
set(WPE_ROOT "${CMAKE_SOURCE_DIR}/../wpe" CACHE PATH
    "Wisevolve public integration root (sibling wpe directory by default)")
set(WPE_SO_ROOT "${CMAKE_SOURCE_DIR}/../wpe.so" CACHE PATH
    "Wisevolve binary package root (sibling wpe.so directory by default)")

if(GPUMD_WPE)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR
            "GPUMD_WPE currently supports Linux/CUDA builds with the WPE ELF shared library")
    endif()

    set(WPE_GPUMD_INTEGRATION "${WPE_ROOT}/integration/gpumd")
    set(WPE_PUBLIC_INCLUDE "${WPE_ROOT}/include")
    set(WPE_LIBRARY "${WPE_SO_ROOT}/lib/libwisevolve_potential.so" CACHE FILEPATH
        "Prebuilt Wisevolve Potential Engine shared library")

    if(NOT EXISTS "${WPE_GPUMD_INTEGRATION}/wpe_adapter.cu" OR
       NOT EXISTS "${WPE_GPUMD_INTEGRATION}/wpe_stage.cu")
        message(FATAL_ERROR
            "GPUMD_WPE=ON requires the WPE public integration sources under ${WPE_GPUMD_INTEGRATION}")
    endif()
    if(NOT EXISTS "${WPE_LIBRARY}")
        message(FATAL_ERROR
            "GPUMD_WPE=ON requires a prebuilt ${WPE_LIBRARY}; build WPE first")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E compare_files
            "${CMAKE_SOURCE_DIR}/src/lib/wpe/wisevolve_potential_api.h"
            "${WPE_PUBLIC_INCLUDE}/wisevolve_potential_api.h"
        RESULT_VARIABLE WPE_API_COMPARE_RESULT)
    if(NOT WPE_API_COMPARE_RESULT EQUAL 0)
        message(FATAL_ERROR
            "GPUMD and WPE copies of wisevolve_potential_api.h differ")
    endif()

    list(APPEND GPUMD_SOURCES
        "${WPE_GPUMD_INTEGRATION}/wpe_adapter.cu"
        "${WPE_GPUMD_INTEGRATION}/wpe_stage.cu")
endif()

add_executable(gpumd ${GPUMD_SOURCES})
target_include_directories(gpumd PRIVATE src)
target_link_libraries(gpumd PRIVATE CUDA::cublas CUDA::cusolver CUDA::cufft)

if(GPUMD_WPE)
    add_library(wisevolve_potential SHARED IMPORTED GLOBAL)
    set_target_properties(wisevolve_potential PROPERTIES
        IMPORTED_LOCATION "${WPE_LIBRARY}")

    target_compile_definitions(gpumd PRIVATE GPUMD_WPE_ENABLED=1)
    target_include_directories(gpumd PRIVATE
        src/force
        src/lib/wpe
        "${WPE_GPUMD_INTEGRATION}")
    target_link_libraries(gpumd PRIVATE wisevolve_potential)
    set_target_properties(gpumd PROPERTIES BUILD_RPATH "${WPE_SO_ROOT}/lib")
endif()

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
target_link_libraries(gnep PRIVATE CUDA::cublas CUDA::cusolver CUDA::cufft Threads::Threads)
option(GNEP_TEST_DIAGNOSTICS "Export deterministic GNEP training diagnostics" OFF)
if(GNEP_TEST_DIAGNOSTICS)
    target_compile_definitions(gnep PRIVATE GNEP_TEST_DIAGNOSTICS)
endif()
option(GNEP_TEST_FIXED_SCALER "Keep the GNEP descriptor scaler fixed for gradient tests" OFF)
if(GNEP_TEST_FIXED_SCALER)
    target_compile_definitions(gnep PRIVATE USE_FIXED_SCALER)
endif()


# --- NNAP support (enable with -D PKG_NNAP=yes) ---
option(PKG_NNAP "Enable NNAP support" OFF)
if(PKG_NNAP)
    find_package(JNI REQUIRED)
    target_include_directories(gpumd PRIVATE ${JNI_INCLUDE_DIRS})
    target_link_libraries(gpumd PRIVATE ${JNI_LIBRARIES})
    target_compile_definitions(gpumd PRIVATE USE_NNAP)
    
    # delay load jvm.dll on windows
    if(WIN32 AND MSVC)
        target_link_libraries(gpumd PRIVATE delayimp)
        target_link_options(gpumd PRIVATE "/DELAYLOAD:jvm.dll")
        string(REPLACE "\\" "/" JSE_JVM_LIB_PATH_DEF "$ENV{JVM_LIB_PATH}")
        add_definitions(-DJVM_LIB_PATH="${JSE_JVM_LIB_PATH_DEF}")
    endif()

    string(REPLACE "\\" "/" JSE_JAR_PATH_DEF "$ENV{JSE_JAR_PATH}")
    target_compile_definitions(gpumd PRIVATE JVM_CLASS_PATH="-Djava.class.path=${JSE_JAR_PATH_DEF}")
endif()
