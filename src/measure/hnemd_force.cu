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

#include "hnemd_force.cuh"
#include "utilities/gpu_macro.cuh"

static __global__ void gpu_add_hnemd_force(
  int N,
  double fe_x,
  double fe_y,
  double fe_z,
  double* g_sxx,
  double* g_sxy,
  double* g_sxz,
  double* g_syx,
  double* g_syy,
  double* g_syz,
  double* g_szx,
  double* g_szy,
  double* g_szz,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    g_fx[i] += fe_x * g_sxx[i] + fe_y * g_syx[i] + fe_z * g_szx[i];
    g_fy[i] += fe_x * g_sxy[i] + fe_y * g_syy[i] + fe_z * g_szy[i];
    g_fz[i] += fe_x * g_sxz[i] + fe_y * g_syz[i] + fe_z * g_szz[i];
  }
}

static __global__ void gpu_sum_force(
  int N, double* g_fx, double* g_fy, double* g_fz, double* g_f)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ double s_f[1024];
  double f = 0.0;

  switch (bid) {
    case 0:
      for (int batch = 0; batch < number_of_batches; ++batch) {
        int n = tid + batch * 1024;
        if (n < N)
          f += g_fx[n];
      }
      break;
    case 1:
      for (int batch = 0; batch < number_of_batches; ++batch) {
        int n = tid + batch * 1024;
        if (n < N)
          f += g_fy[n];
      }
      break;
    case 2:
      for (int batch = 0; batch < number_of_batches; ++batch) {
        int n = tid + batch * 1024;
        if (n < N)
          f += g_fz[n];
      }
      break;
  }
  s_f[tid] = f;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_f[tid] += s_f[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_f[bid] = s_f[0];
  }
}

static __global__ void gpu_correct_force(
  int N, double one_over_N, double* g_fx, double* g_fy, double* g_fz, double* g_f)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    g_fx[i] -= g_f[0] * one_over_N;
    g_fy[i] -= g_f[1] * one_over_N;
    g_fz[i] -= g_f[2] * one_over_N;
  }
}

void apply_hnemd_force(
  const int number_of_atoms,
  const double fe_x,
  const double fe_y,
  const double fe_z,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& force_sum)
{
  // the virial tensor:
  // xx xy xz    0 3 4
  // yx yy yz    6 1 5
  // zx zy zz    7 8 2
  gpu_add_hnemd_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    fe_x,
    fe_y,
    fe_z,
    virial_per_atom.data() + 0 * number_of_atoms,
    virial_per_atom.data() + 3 * number_of_atoms,
    virial_per_atom.data() + 4 * number_of_atoms,
    virial_per_atom.data() + 6 * number_of_atoms,
    virial_per_atom.data() + 1 * number_of_atoms,
    virial_per_atom.data() + 5 * number_of_atoms,
    virial_per_atom.data() + 7 * number_of_atoms,
    virial_per_atom.data() + 8 * number_of_atoms,
    virial_per_atom.data() + 2 * number_of_atoms,
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + 2 * number_of_atoms);

  gpu_sum_force<<<3, 1024>>>(
    number_of_atoms,
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + 2 * number_of_atoms,
    force_sum.data());
  GPU_CHECK_KERNEL

  gpu_correct_force<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    1.0 / number_of_atoms,
    force_per_atom.data(),
    force_per_atom.data() + number_of_atoms,
    force_per_atom.data() + 2 * number_of_atoms,
    force_sum.data());
  GPU_CHECK_KERNEL
}
