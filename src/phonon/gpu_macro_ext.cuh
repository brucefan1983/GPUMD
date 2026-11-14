/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
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
#endif
#else
#ifndef HESSIAN_GPU_MACRO_EXT_CUDA_DEFINED
#define HESSIAN_GPU_MACRO_EXT_CUDA_DEFINED
#define gpuDeviceGetLimit cudaDeviceGetLimit
#define gpuDeviceSetLimit cudaDeviceSetLimit
#define gpuLimitStackSize cudaLimitStackSize
#endif
#endif
