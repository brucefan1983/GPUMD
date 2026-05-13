/*
    gpumd_python_kernels.cuh

    Declarations for CUDA kernels used by the GPUMD pybind11 wrapper.

    Copyright 2026 Jaafar Mehrez
    (Shanghai Jiao Tong University, Shanghai, China;
     HPQC Labs, Waterloo, Canada;
     jaafarmehrez@sjtu.edu.cn, jaafar@hpqc.org)

    SPDX-License-Identifier: MIT
*/

#pragma once

// Forward declaration of the host-side launcher.
// The actual __global__ kernel lives in the accompanying .cu file so that
// nvcc compiles it; this header can be included from plain C++ code.
void gpu_add_aos_bias_to_soa_forces(int N, double* forces_soa, const double* bias_aos);
