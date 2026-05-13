/*
    gpumd_python_kernels.cu

    CUDA kernels for the GPUMD pybind11 wrapper.

    Copyright 2026 Jaafar Mehrez
    (Shanghai Jiao Tong University, Shanghai, China;
     HPQC Labs, Waterloo, Canada;
     jaafarmehrez@sjtu.edu.cn, jaafar@hpqc.org)

    SPDX-License-Identifier: MIT
*/

// IMPORTANT: cuda_runtime.h must come before any GPUMD .cuh headers.
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA kernel: add AOS bias to SOA forces in-place.
//
// GPUMD force_per_atom is SOA: [fx0..fxN-1, fy0..fyN-1, fz0..fzN-1]
// JAX bias is AOS:             [fx0, fy0, fz0, fx1, fy1, fz1, ...]
// ---------------------------------------------------------------------------
__global__ void gpu_add_aos_bias_to_soa_forces_kernel(
  int N, double* forces_soa, const double* bias_aos)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    forces_soa[i]         += bias_aos[i * 3];     // fx
    forces_soa[i + N]     += bias_aos[i * 3 + 1]; // fy
    forces_soa[i + 2 * N] += bias_aos[i * 3 + 2]; // fz
  }
}

void gpu_add_aos_bias_to_soa_forces(int N, double* forces_soa, const double* bias_aos)
{
  const int block = 256;
  const int grid = (N + block - 1) / block;
  gpu_add_aos_bias_to_soa_forces_kernel<<<grid, block>>>(N, forces_soa, bias_aos);
}
