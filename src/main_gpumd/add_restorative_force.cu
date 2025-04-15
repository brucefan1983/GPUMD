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

#include "add_restorative_force.cuh"
#include "model/atom.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cstring>

static __global__ void add_force(
  const int N,
  const double force_constant,
  const double z0,
  double* g_z,
  double* g_fz)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    const double d = g_z[n] > z0;
    if (d > 0) {
      g_fz[n] -= force_constant * d;
    }
  }
}

void Add_Restorative_Force::compute(const int step, Atom& atom)
{
  for (int call = 0; call < num_calls_; ++call) {
    add_force<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
      atom.number_of_atoms,
      force_constant_,
      z0_,
      atom.position_per_atom.data() + atom.number_of_atoms * 2,
      atom.force_per_atom.data() + atom.number_of_atoms * 2);
    GPU_CHECK_KERNEL
  }
}

void Add_Restorative_Force::parse(const char** param, int num_param, int number_of_atoms)
{
  printf("Add restorative force.\n");

  // check the number of parameters
  if (num_param != 3) {
    PRINT_INPUT_ERROR("add_restorative_force should have 2 parameters.\n");
  }

  // parse force constant
  if (!is_valid_real(param[1], &force_constant_)) {
    PRINT_INPUT_ERROR("force constant should be a number.\n");
  }
  if (force_constant_ < 0) {
    PRINT_INPUT_ERROR("force constant should >= 0.\n");
  }

  // parse z0
  if (!is_valid_real(param[2], &z0_)) {
    PRINT_INPUT_ERROR("z0 should be a number.\n");
  }

  ++num_calls_;

  if (num_calls_ > 1) {
    PRINT_INPUT_ERROR("add_restorative_force cannot be used more than 1 time in one run.");
  }


  printf("    with force constant %f eV/A^2.\n", force_constant_);
  printf("    for atoms above %f A.\n", z0_);
}

void Add_Restorative_Force::finalize() { num_calls_ = 0; }
