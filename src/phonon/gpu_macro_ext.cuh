/*
    Copyright 2017 Zheyong Fan and GPUMD development team.
    This file is part of GPUMD, licensed under GNU GPL version 3 or later.
*/

/*----------------------------------------------------------------------------80
GPU macro extensions used by the hessian module.

This header is included only by the analytic Hessian code so the parent
src/utilities/gpu_macro.cuh file is not modified.
------------------------------------------------------------------------------*/

#pragma once

#ifdef USE_HIP
#ifndef HESSIAN_GPU_MACRO_EXT_DEFINED
#define HESSIAN_GPU_MACRO_EXT_DEFINED
#define gpuDeviceGetLimit hipDeviceGetLimit
#define gpuDeviceSetLimit hipDeviceSetLimit
#define gpuLimitStackSize hipLimitStackSize
#define gpuLimitPrintfFifoSize hipLimitPrintfFifoSize
#endif
#else
#ifndef HESSIAN_GPU_MACRO_EXT_CUDA_DEFINED
#define HESSIAN_GPU_MACRO_EXT_CUDA_DEFINED
#define gpuDeviceGetLimit cudaDeviceGetLimit
#define gpuDeviceSetLimit cudaDeviceSetLimit
#define gpuLimitStackSize cudaLimitStackSize
#define gpuLimitPrintfFifoSize cudaLimitPrintfFifoSize
#endif
#endif

#ifndef HESSIAN_SOLVER_EXT_DEFINED
#define HESSIAN_SOLVER_EXT_DEFINED
#ifdef USE_HIP
#define gpusolverDnDsyevd_bufferSize hipsolverDnDsyevd_bufferSize
#define gpusolverDnDsyevd hipsolverDnDsyevd
#else
#define gpusolverDnDsyevd_bufferSize cusolverDnDsyevd_bufferSize
#define gpusolverDnDsyevd cusolverDnDsyevd
#endif
#endif
