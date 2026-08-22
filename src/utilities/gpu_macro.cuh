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
#include <hipfft/hipfft.h>
#include <hip/hip_runtime.h>

// memory manipulation
#define gpuMalloc hipMalloc
#define gpuMallocManaged hipMallocManaged
#define gpuHostAlloc hipHostMalloc
#define gpuHostAllocDefault hipHostMallocDefault
#define gpuFree hipFree
#define gpuFreeHost hipHostFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyPeer hipMemcpyPeer
#define gpuMemcpyPeerAsync hipMemcpyPeerAsync
#define gpuMemcpyFromSymbol hipMemcpyFromSymbol
#define gpuMemcpyToSymbol hipMemcpyToSymbol
#define gpuGetSymbolAddress hipGetSymbolAddress
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToHost hipMemcpyHostToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemset hipMemset
#define gpuMemsetAsync hipMemsetAsync

// error handling
#define gpuError_t hipError_t
#define gpuSuccess hipSuccess
#define gpuErrorNotReady hipErrorNotReady
#define gpuGetErrorString hipGetErrorString
#define gpuGetLastError hipGetLastError

// device manipulation
#define gpuSetDevice hipSetDevice
#define gpuGetDevice hipGetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuDeviceProp hipDeviceProp_t
#define gpuGetDeviceProperties hipGetDeviceProperties
#define gpuDeviceCanAccessPeer hipDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define gpuDeviceSynchronize hipDeviceSynchronize

// stream
#define gpuStream_t hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamCreateWithFlags hipStreamCreateWithFlags
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStreamWaitEvent hipStreamWaitEvent
#define gpuStreamQuery hipStreamQuery
#define gpuStreamDefault hipStreamDefault
#define gpuStreamNonBlocking hipStreamNonBlocking

// event
#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventQuery hipEventQuery
#define gpuEventElapsedTime hipEventElapsedTime
#define gpuEventDefault hipEventDefault
#define gpuEventDisableTiming hipEventDisableTiming

// random numbers
#define gpurandState hiprandState
#define gpurand_normal_double hiprand_normal_double
#define gpurand_normal hiprand_normal
#define gpurand_init hiprand_init

// blas
#define gpublasHandle_t hipblasHandle_t
#define gpublasSgemv hipblasSgemv
#define gpublasSgemm hipblasSgemm
#define gpublasSdgmm hipblasSdgmm
#define gpublasDgemvBatched hipblasDgemvBatched
#define gpublasDestroy hipblasDestroy
#define gpublasCreate hipblasCreate
#define gpublasSetStream hipblasSetStream
#define GPUBLAS_SIDE_LEFT HIPBLAS_SIDE_LEFT
#define GPUBLAS_OP_N HIPBLAS_OP_N
#define GPUBLAS_OP_T HIPBLAS_OP_T

// lapack
#define gpuDoubleComplex hipDoubleComplex
#define gpusolverDnHandle_t hipsolverDnHandle_t
#define gpusolverDnCreate hipsolverDnCreate
#define gpusolverDnDestroy hipsolverDnDestroy
#define gpusolverDnSetStream hipsolverDnSetStream
#define gpusolverEigMode_t hipsolverEigMode_t
#define gpusolverFillMode_t hipsolverFillMode_t
#define GPUSOLVER_EIG_MODE_NOVECTOR HIPSOLVER_EIG_MODE_NOVECTOR
#define GPUSOLVER_EIG_MODE_VECTOR HIPSOLVER_EIG_MODE_VECTOR
#define GPUSOLVER_FILL_MODE_LOWER HIPSOLVER_FILL_MODE_LOWER
#define gpusolverSyevjInfo_t hipsolverSyevjInfo_t
#define gpusolverDnCreateSyevjInfo hipsolverDnCreateSyevjInfo
#define gpusolverDnDestroySyevjInfo hipsolverDnDestroySyevjInfo
#define gpusolverDnZheevj_bufferSize hipsolverDnZheevj_bufferSize
#define gpusolverDnZheevj hipsolverDnZheevj
#define gpusolverDnZheevd_bufferSize hipsolverDnZheevd_bufferSize
#define gpusolverDnZheevd hipsolverDnZheevd
#define gpusolverDnDsyevj_bufferSize hipsolverDnDsyevj_bufferSize
#define gpusolverDnDsyevj hipsolverDnDsyevj
#define gpusolverDnZheevjBatched_bufferSize hipsolverDnZheevjBatched_bufferSize
#define gpusolverDnZheevjBatched hipsolverDnZheevjBatched

// FFT
#define gpufftHandle hipfftHandle
#define gpufftComplex hipfftComplex
#define gpufftExecC2C hipfftExecC2C
#define gpufftPlan3d hipfftPlan3d 
#define gpufftPlanMany hipfftPlanMany
#define gpufftDestroy hipfftDestroy
#define gpufftSetStream hipfftSetStream
#define GPUFFT_SUCCESS HIPFFT_SUCCESS
#define GPUFFT_C2C HIPFFT_C2C
#define GPUFFT_FORWARD HIPFFT_FORWARD
#define GPUFFT_INVERSE HIPFFT_BACKWARD

#else // CUDA for Nvidia card

// memory manipulation
#define gpuMalloc cudaMalloc
#define gpuMallocManaged cudaMallocManaged
#define gpuHostAlloc cudaHostAlloc
#define gpuHostAllocDefault cudaHostAllocDefault
#define gpuFree cudaFree
#define gpuFreeHost cudaFreeHost
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyPeer cudaMemcpyPeer
#define gpuMemcpyPeerAsync cudaMemcpyPeerAsync
#define gpuMemcpyFromSymbol cudaMemcpyFromSymbol
#define gpuMemcpyToSymbol cudaMemcpyToSymbol
#define gpuGetSymbolAddress cudaGetSymbolAddress
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToHost cudaMemcpyHostToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemset cudaMemset
#define gpuMemsetAsync cudaMemsetAsync

// error handling
#define gpuError_t cudaError_t
#define gpuSuccess cudaSuccess
#define gpuErrorNotReady cudaErrorNotReady
#define gpuGetErrorString cudaGetErrorString
#define gpuGetLastError cudaGetLastError

// device manipulation
#define gpuSetDevice cudaSetDevice
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuDeviceProp cudaDeviceProp
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuDeviceCanAccessPeer cudaDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
#define gpuDeviceSynchronize cudaDeviceSynchronize

// stream
#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStreamWaitEvent cudaStreamWaitEvent
#define gpuStreamQuery cudaStreamQuery
#define gpuStreamDefault cudaStreamDefault
#define gpuStreamNonBlocking cudaStreamNonBlocking

// event
#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventCreateWithFlags cudaEventCreateWithFlags
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventQuery cudaEventQuery
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuEventDefault cudaEventDefault
#define gpuEventDisableTiming cudaEventDisableTiming

// random numbers
#define gpurandState curandState
#define gpurand_normal_double curand_normal_double
#define gpurand_normal curand_normal
#define gpurand_init curand_init

// blas
#define gpublasHandle_t cublasHandle_t
#define gpublasSgemv cublasSgemv
#define gpublasSgemm cublasSgemm
#define gpublasSdgmm cublasSdgmm
#define gpublasDgemv cublasDgemv
#if (CUDA_VERSION >= 12000)
#define gpublasDgemvBatched cublasDgemvBatched
#endif
#define gpublasDestroy cublasDestroy
#define gpublasCreate cublasCreate
#define gpublasSetStream cublasSetStream
#define GPUBLAS_SIDE_LEFT CUBLAS_SIDE_LEFT
#define GPUBLAS_OP_N CUBLAS_OP_N
#define GPUBLAS_OP_T CUBLAS_OP_T

// lapack
#define gpuDoubleComplex cuDoubleComplex
#define gpusolverDnHandle_t cusolverDnHandle_t
#define gpusolverDnCreate cusolverDnCreate
#define gpusolverDnDestroy cusolverDnDestroy
#define gpusolverDnSetStream cusolverDnSetStream
#define gpusolverEigMode_t cusolverEigMode_t
#define gpusolverFillMode_t cublasFillMode_t // why cublas?
#define GPUSOLVER_EIG_MODE_NOVECTOR CUSOLVER_EIG_MODE_NOVECTOR
#define GPUSOLVER_EIG_MODE_VECTOR CUSOLVER_EIG_MODE_VECTOR
#define GPUSOLVER_FILL_MODE_LOWER CUBLAS_FILL_MODE_LOWER // why cublas?
#define gpusolverSyevjInfo_t syevjInfo_t                 // why not cusolverSyevjInfo_t?
#define gpusolverDnCreateSyevjInfo cusolverDnCreateSyevjInfo
#define gpusolverDnDestroySyevjInfo cusolverDnDestroySyevjInfo
#define gpusolverDnZheevj_bufferSize cusolverDnZheevj_bufferSize
#define gpusolverDnZheevj cusolverDnZheevj
#define gpusolverDnZheevd_bufferSize cusolverDnZheevd_bufferSize
#define gpusolverDnZheevd cusolverDnZheevd
#define gpusolverDnDsyevj_bufferSize cusolverDnDsyevj_bufferSize
#define gpusolverDnDsyevj cusolverDnDsyevj
#define gpusolverDnZheevjBatched_bufferSize cusolverDnZheevjBatched_bufferSize
#define gpusolverDnZheevjBatched cusolverDnZheevjBatched

// FFT
#define gpufftHandle cufftHandle
#define gpufftComplex cufftComplex
#define gpufftExecC2C cufftExecC2C
#define gpufftPlan3d cufftPlan3d 
#define gpufftPlanMany cufftPlanMany
#define gpufftDestroy cufftDestroy
#define gpufftSetStream cufftSetStream
#define GPUFFT_SUCCESS CUFFT_SUCCESS
#define GPUFFT_C2C CUFFT_C2C
#define GPUFFT_FORWARD CUFFT_FORWARD
#define GPUFFT_INVERSE CUFFT_INVERSE

#endif
