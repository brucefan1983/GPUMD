/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#ifdef USE_HIP // HIP for AMD card

#include "hip/hip_runtime.h"

// memory manipulation
#define gpuMalloc hipMalloc 
#define gpuMallocManaged hipMallocManaged
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyFromSymbol hipMemcpyFromSymbol
#define gpuMemcpyToSymbol hipMemcpyToSymbol
#define gpuGetSymbolAddress hipGetSymbolAddress
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToHost hipMemcpyHostToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemset hipMemset

// error handling
#define gpuError_t hipError_t 
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuGetLastError hipGetLastError

// device manipulation
#define gpuSetDevice hipSetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuDeviceProp hipDeviceProp
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceCanAccessPeer hipDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define gpuDeviceSynchronize hipDeviceSynchronize

// stream
#define gpuStream_t hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy

// random numbers
#define gpurandState hiprandState
#define gpurand_normal_double hiprand_normal_double
#define gpurand_normal hiprand_normal
#define gpurand_init hiprand_init

#else // CUDA for Nvidia card

// memory manipulation
#define gpuMalloc cudaMalloc
#define gpuMallocManaged cudaMallocManaged
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyFromSymbol cudaMemcpyFromSymbol
#define gpuMemcpyToSymbol cudaMemcpyToSymbol
#define gpuGetSymbolAddress cudaGetSymbolAddress
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToHost cudaMemcpyHostToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemset cudaMemset

// error handling
#define gpuError_t cudaError_t 
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError

// device manipulation
#define gpuSetDevice cudaSetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuDeviceProp cudaDeviceProp
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceCanAccessPeer cudaDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
#define gpuDeviceSynchronize cudaDeviceSynchronize

// stream
#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy

// random numbers
#define gpurandState curandState
#define gpurand_normal_double curand_normal_double
#define gpurand_normal curand_normal
#define gpurand_init curand_init

#endif
