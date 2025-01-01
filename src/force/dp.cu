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
The class dealing with the Deep Potential(DP).
------------------------------------------------------------------------------*/


#include "dp.cuh"
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <sstream>


#define BLOCK_SIZE_FORCE 128
#define MAX_NEIGH_NUM_DP 512    // max neighbor number of an atom for DP



DP::DP(const char* filename_dp, int num_atoms)
{
  // DP setting
  set_dp_coeff();

  // init DP from potential file
  initialize_dp(filename_dp);


  dp_data.NN.resize(num_atoms);
  dp_data.NL.resize(num_atoms * 1024); // the largest supported by CUDA
  dp_data.cell_count.resize(num_atoms);
  dp_data.cell_count_sum.resize(num_atoms);
  dp_data.cell_contents.resize(num_atoms);

  // init dp neighbor list
  dp_nl.inum = num_atoms;
  dp_nl.ilist.resize(num_atoms);
  dp_nl.numneigh.resize(num_atoms);
  dp_nl.firstneigh.resize(num_atoms);
  dp_nl.neigh_storage.resize(num_atoms * MAX_NEIGH_NUM_DP);
  

}


void DP::initialize_dp(const char* filename_dp)
{
  int num_gpus;
  CHECK(gpuGetDeviceCount(&num_gpus));
  printf("\nInitialize deep potential by the file: %s.\n\n", filename_dp);
  deep_pot.init(filename_dp, num_gpus);
  rc = deep_pot.cutoff();
  int numb_types = deep_pot.numb_types();
  int numb_types_spin = deep_pot.numb_types_spin();
  int dim_fparam = deep_pot.dim_fparam();
  int dim_aparam = deep_pot.dim_aparam();

  char* type_map[numb_types];
  std::string type_map_str;
  deep_pot.get_type_map(type_map_str);
  // convert the string to a vector of strings
  std::istringstream iss(type_map_str);
  std::string type_name;
  int i = 0;
  while (iss >> type_name) {
    if (i >= numb_types) break;
    type_map[i] = strdup(type_name.c_str());
    i++;
  }

  printf("=======================================================\n");
  printf("  ++ cutoff: %f ++ \n", rc);
  printf("  ++ numb_types: %d ++ \n", numb_types);
  printf("  ++ numb_types_spin: %d ++ \n", numb_types_spin);
  printf("  ++ dim_fparam: %d ++ \n", dim_fparam);
  printf("  ++ dim_aparam: %d ++ \n  ++ ", dim_aparam);
  for (int i = 0; i < numb_types; ++i)
  {
    printf("%s ", type_map[i]);
  }
  printf("++\n=======================================================\n");
}

DP::~DP(void)
{
  // none
}

void DP::set_dp_coeff(void) {
  ener_unit_cvt_factor=1;      // 1.0 / 8.617343e-5;
  dist_unit_cvt_factor=1;      // 1;
  force_unit_cvt_factor=1;     // ener_unit_cvt_factor / dist_unit_cvt_factor;
  virial_unit_cvt_factor=1;    // ener_unit_cvt_factor
  single_model = true;
  atom_spin_flag = false;
}


static __global__ void create_dp_position(
  const double* gpumd_position,
  const double* dp_position,
  int N)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x; // particle index
  if (n1 < N) {
    dp_position[n1 * 3] = gpumd_position[n1];
    dp_position[n1 * 3 + 1] = gpumd_position[n1 + N];
    dp_position[n1 * 3 + 2] = gpumd_position[n1 + 2 * N];
  }
}


// force and virial need transpose from dp to gpumd
// TODO: use share memory to speed up
static __global__ void transpose_and_update_unit(
  const double* e_f_v_in,
  double* e_out,
  double* f_out,
  double* v_out,
  double e_factor,
  double f_factor,
  double v_factor,
  const int N)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x; // particle index
  if (n1 < N) {
    const int f_in_offset = N;
    const int v_in_offset = N * 4;
    e_out[n1] = e_f_v_in[n1] * e_factor;
    f_out[n1] = e_f_v_in[f_in_offset + n1 * 3] * f_factor;                // fx
    f_out[n1 + N] = e_f_v_in[f_in_offset + n1 * 3 + 1] * f_factor;        // fy
    f_out[n1 + N * 2] = e_f_v_in[f_in_offset + n1 * 3 + 2] * f_factor;    // fz
    // virial
    v_out[n1] = e_f_v_in[v_in_offset + n1 * 9] * v_factor;
    v_out[n1 + N] = e_f_v_in[v_in_offset + n1 * 9 + 4] * v_factor;
    v_out[n1 + N * 2] = e_f_v_in[v_in_offset + n1 * 9 + 8] * v_factor;
    v_out[n1 + N * 3] = e_f_v_in[v_in_offset + n1 * 9 + 3] * v_factor;
    v_out[n1 + N * 4] = e_f_v_in[v_in_offset + n1 * 9 + 6] * v_factor;
    v_out[n1 + N * 5] = e_f_v_in[v_in_offset + n1 * 9 + 7] * v_factor;
    v_out[n1 + N * 6] = e_f_v_in[v_in_offset + n1 * 9 + 1] * v_factor;
    v_out[n1 + N * 7] = e_f_v_in[v_in_offset + n1 * 9 + 2] * v_factor;
    v_out[n1 + N * 8] = e_f_v_in[v_in_offset + n1 * 9 + 5] * v_factor;
  }
}



void DP::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  return;
}