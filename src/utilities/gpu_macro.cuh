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

#define gpuMalloc hipMalloc 
#define gpuMallocManaged hipMallocManaged
#define gpuFree hipFree

#define gpuMemcpy hipMemcpy
#define gpuMemcpyFromSymbol hipMemcpyFromSymbol
#define gpuMemcpyToSymbol hipMemcpyToSymbol
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToHost hipMemcpyHostToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define gpuError_t hipError_t 
#define gpuSuccess hipSuccess
#define gpuGetErrorString hipGetErrorString
#define gpuGetLastError hipGetLastError

#define gpuDeviceSynchronize hipDeviceSynchronize

#else // CUDA for Nvidia card

#define gpuMalloc cudaMalloc
#define gpuMallocManaged cudaMallocManaged
#define gpuFree cudaFree

#define gpuMemcpy cudaMemcpy
#define gpuMemcpyFromSymbol cudaMemcpyFromSymbol
#define gpuMemcpyToSymbol cudaMemcpyToSymbol
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToHost cudaMemcpyHostToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

#define gpuError_t cudaError_t 
#define gpuSuccess cudaSuccess
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError

#define gpuDeviceSynchronize cudaDeviceSynchronize

#endif
