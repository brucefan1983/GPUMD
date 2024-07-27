/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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

static __global__ void initialize_curand_states(curandState* state, int N, int seed)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    curand_init(seed, n, 0, &state[n]);
  }
}

static __global__ void add_random_force(
  const int N,
  const double force_variance,
  curandState* g_state,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    curandState state = g_state[n];
    g_fx[n] = force_variance * curand_normal_double(&state);
    g_fy[n] = force_variance * curand_normal_double(&state);
    g_fz[n] = force_variance * curand_normal_double(&state);
    g_state[n] = state;
  }
}

void Add_Random_Force::compute(const int step, Atom& atom)
{
  for (int call = 0; call < num_calls_; ++call) {
    const int num_atoms_total = atom.force_per_atom.size() / 3;
    add_random_force<<<(num_atoms_total - 1) / 64 + 1, 64>>>(
      num_atoms_total,
      force_variance_,
      curand_states_.data(),
      atom.force_per_atom.data(),
      atom.force_per_atom.data() + num_atoms_total,
      atom.force_per_atom.data() + num_atoms_total * 2
    );
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

  // parse grouping method
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

void Add_Random_Force::finalize() 
{ 
  num_calls_ = 0;
}
