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

/*----------------------------------------------------------------------------80
Add random forces with zero mean and specified variance.
------------------------------------------------------------------------------*/

#include "add_random_force.cuh"
#include "model/atom.cuh"
#include "utilities/read_file.cuh"
#include <cstdlib>
#include <iostream>
#include <vector>

static __global__ void initialize_curand_states(gpurandState* state, int N, int seed)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    gpurand_init(seed, n, 0, &state[n]);
  }
}

static __global__ void add_random_force(
  const int N,
  const double force_variance,
  gpurandState* g_state,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    gpurandState state = g_state[n];
    g_fx[n] += force_variance * gpurand_normal_double(&state);
    g_fy[n] += force_variance * gpurand_normal_double(&state);
    g_fz[n] += force_variance * gpurand_normal_double(&state);
    g_state[n] = state;
  }
}

__device__ double device_total_force[3];

// get the total force
static __global__ void gpu_sum_force(int N, double* g_fx, double* g_fy, double* g_fz)
{
  //<<<3, 1024>>>
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_f[1024];
  double f = 0.0;

  switch (bid) {
    case 0:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          f += g_fx[n];
      }
      break;
    case 1:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          f += g_fy[n];
      }
      break;
    case 2:
      for (int patch = 0; patch < number_of_patches; ++patch) {
        int n = tid + patch * 1024;
        if (n < N)
          f += g_fz[n];
      }
      break;
  }
  s_f[tid] = f;
  __syncthreads();

#pragma unroll
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_f[tid] += s_f[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    device_total_force[bid] = s_f[0];
  }
}

// correct the total force
static __global__ void
gpu_correct_force(int N, double one_over_N, double* g_fx, double* g_fy, double* g_fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    g_fx[i] -= device_total_force[0] * one_over_N;
    g_fy[i] -= device_total_force[1] * one_over_N;
    g_fz[i] -= device_total_force[2] * one_over_N;
  }
}

void Add_Random_Force::compute(const int step, Atom& atom)
{
  for (int call = 0; call < num_calls_; ++call) {
    add_random_force<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      force_variance_,
      curand_states_.data(),
      atom.force_per_atom.data(),
      atom.force_per_atom.data() + atom.number_of_atoms,
      atom.force_per_atom.data() + atom.number_of_atoms * 2);
    CUDA_CHECK_KERNEL

    gpu_sum_force<<<3, 1024>>>(
      atom.number_of_atoms,
      atom.force_per_atom.data(),
      atom.force_per_atom.data() + atom.number_of_atoms,
      atom.force_per_atom.data() + 2 * atom.number_of_atoms);
    CUDA_CHECK_KERNEL

    gpu_correct_force<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      1.0 / atom.number_of_atoms,
      atom.force_per_atom.data(),
      atom.force_per_atom.data() + atom.number_of_atoms,
      atom.force_per_atom.data() + 2 * atom.number_of_atoms);
    CUDA_CHECK_KERNEL
  }
}

void Add_Random_Force::parse(const char** param, int num_param, int number_of_atoms)
{
  printf("Add force.\n");

  // check the number of parameters
  if (num_param != 2) {
    PRINT_INPUT_ERROR("add_random_force should have 1 parameter.\n");
  }

  // parse force variance
  if (!is_valid_real(param[1], &force_variance_)) {
    PRINT_INPUT_ERROR("force variance should be a number.\n");
  }
  if (force_variance_ < 0) {
    PRINT_INPUT_ERROR("force variance should >= 0.\n");
  }

  ++num_calls_;

  if (num_calls_ > 1) {
    PRINT_INPUT_ERROR("add_random_force cannot be used more than 1 time in one run.");
  }

  curand_states_.resize(number_of_atoms);
  int grid_size = (number_of_atoms - 1) / 128 + 1;
  initialize_curand_states<<<grid_size, 128>>>(curand_states_.data(), number_of_atoms, rand());
  CUDA_CHECK_KERNEL
}

void Add_Random_Force::finalize() { num_calls_ = 0; }
