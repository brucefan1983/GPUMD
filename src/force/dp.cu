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
  dp_data.NL.resize(num_atoms * MAX_NEIGH_NUM_DP); // the largest supported by CUDA
  dp_data.cell_count.resize(num_atoms);
  dp_data.cell_count_sum.resize(num_atoms);
  dp_data.cell_contents.resize(num_atoms);
  dp_position_gpu.resize(num_atoms * 3);
  type_cpu.resize(num_atoms);
  e_f_v_gpu.resize(num_atoms * (1 + 3 + 9));    // energy: 1; force: 3; virial: 9

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
  printf("\nInitialize deep potential by the file: %s and %d gpu(s).\n\n", filename_dp, num_gpus);
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
  dist_unit_cvt_factor=1;      // gpumd: angstrom, dp: angstrom;
  force_unit_cvt_factor=ener_unit_cvt_factor / dist_unit_cvt_factor;
  virial_unit_cvt_factor=1;    // ener_unit_cvt_factor
  single_model = true;
  atom_spin_flag = false;
}


static __global__ void create_dp_position(
  const double* gpumd_position,
  double* dp_position,
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
  const int number_of_atoms = type.size();
  int grid_size = (N2 - N1 - 1) / BLOCK_SIZE_FORCE + 1;

#ifdef USE_FIXED_NEIGHBOR
  static int num_calls = 0;
#endif
#ifdef USE_FIXED_NEIGHBOR
  if (num_calls++ == 0) {
#endif
    find_neighbor(
      N1,
      N2,
      rc,
      box,
      type,
      position_per_atom,
      dp_data.cell_count,
      dp_data.cell_count_sum,
      dp_data.cell_contents,
      dp_data.NN,
      dp_data.NL);
#ifdef USE_FIXED_NEIGHBOR
  }
#endif

  // create dp position from gpumd
  create_dp_position<<<grid_size, BLOCK_SIZE_FORCE>>>(
    position_per_atom.data(),
    dp_position_gpu.data(),
    number_of_atoms
  );
  GPU_CHECK_KERNEL

  // Initialize DeepPot computation variables
  std::vector<double> dp_ene_all(1, 0.0);
  std::vector<double> dp_ene_atom(number_of_atoms, 0.0);
  std::vector<double> dp_force(number_of_atoms * 3, 0.0);
  std::vector<double> dp_vir_all(9, 0.0);
  std::vector<double> dp_vir_atom(number_of_atoms * 9, 0.0);


  // copy position and type to CPU
  std::vector<double> dp_position_cpu(number_of_atoms * 3);
  dp_position_gpu.copy_to_host(dp_position_cpu.data());
  // TODO: BUG! argument list does not match, because type is const int?
  // type.copy_to_host(type_cpu.data(), number_of_atoms);
  CHECK(gpuMemcpy(type_cpu.data(), type.data(), number_of_atoms * sizeof(int), gpuMemcpyDeviceToHost));


  // create dp box
  std::vector<double> dp_box(9, 0.0);
  if (box.triclinic == 0) {
    dp_box[0] = box.cpu_h[0];
    dp_box[4] = box.cpu_h[1];
    dp_box[8] = box.cpu_h[2];
  } else {
    dp_box[0] = box.cpu_h[0];
    dp_box[4] = box.cpu_h[1];
    dp_box[8] = box.cpu_h[2];
    dp_box[7] = box.cpu_h[7];
    dp_box[6] = box.cpu_h[6];
    dp_box[3] = box.cpu_h[3];
  }

  // Allocate lmp_ilist and lmp_numneigh
  dp_data.NN.copy_to_host(dp_nl.numneigh.data());
  // gpuMemcpy(lmp_numneigh, deepmd_ghost_data.NN.data(), num_of_all_atoms*sizeof(int), gpuMemcpyDeviceToHost);
  std::vector<int> cpu_NL(dp_data.NL.size());
  dp_data.NL.copy_to_host(cpu_NL.data());
  // gpuMemcpy(cpu_NL.data(), deepmd_ghost_data.NL.data(), total_all_neighs * sizeof(int), gpuMemcpyDeviceToHost);

  int offset = 0;
  for (int i = 0; i < number_of_atoms; ++i) {
    dp_nl.ilist[i] = i;
    dp_nl.firstneigh[i] = dp_nl.neigh_storage.data() + offset;
    for (int j = 0; j < dp_nl.numneigh[i]; ++j) {
        dp_nl.neigh_storage[offset + j] = cpu_NL[i + j * number_of_atoms]; // Copy in column-major order
    }
    offset += dp_nl.numneigh[i];
  }

  // Constructing a neighbor list in LAMMPS format
  // deepmd_compat::InputNlist lmp_list(num_of_all_atoms, lmp_ilist, lmp_numneigh, lmp_firstneigh);
  deepmd_compat::InputNlist lmp_list(dp_nl.inum, dp_nl.ilist.data(), dp_nl.numneigh.data(), dp_nl.firstneigh.data());



  // to calculate the atomic force and energy from deepot
  if (single_model) {
    if (! atom_spin_flag) {
        //deep_pot.compute(dp_ene_all, dp_force, dp_vir_all,dp_cpu_ghost_position, gpumd_cpu_ghost_type,dp_box);
        deep_pot.compute(dp_ene_all, dp_force, dp_vir_all, dp_ene_atom, dp_vir_atom, 
            dp_position_cpu, type_cpu, dp_box,
            0, lmp_list, 0);
    }
  }


  // copy dp output energy, force, and virial to gpu
  size_t size_tmp = number_of_atoms; // size of number_of_atom * 1 in double
  // memory distribution of e_f_v_gpu: e1, e2 ... en, fx1, fy1, fz1, fx2 ... fzn, vxx1 ...
  e_f_v_gpu.copy_from_host(dp_ene_atom.data(), size_tmp, 0);
  e_f_v_gpu.copy_from_host(dp_force.data(), size_tmp * 3, number_of_atoms);
  e_f_v_gpu.copy_from_host(dp_vir_atom.data(), size_tmp * 9, number_of_atoms * 4);

  // std::vector<double> gpumd_ene_atom(number_of_atoms, 0.0);
  // std::vector<double> gpumd_force(number_of_atoms * 3, 0.0);
  // std::vector<double> virial_per_atom_cpu(number_of_atoms * 9, 0.0);
  // const int const_cell = half_const_cell * 2 + 1;
  // for (int g = 0; g < number_of_atoms; g++) {
  //   gpumd_ene_atom[g] += dp_ene_atom[i * real_num_of_atoms + g] * ener_unit_cvt_factor;
  //   for (int o = 0; o < 3; o++)
  //     gpumd_force.data()[g + o * real_num_of_atoms] += dp_force.data()[3*g+o] * force_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 0 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 0] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 1 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 4] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 2 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 8] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 3 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 3] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 4 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 6] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 5 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 7] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 6 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 1] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 7 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 2] * virial_unit_cvt_factor;
  //   virial_per_atom_cpu.data()[g + 8 * real_num_of_atoms] += dp_vir_atom.data()[9 * g + 5] * virial_unit_cvt_factor;
  // }
  transpose_and_update_unit<<<grid_size, BLOCK_SIZE_FORCE>>>(
    e_f_v_gpu.data(),
    potential_per_atom.data(),
    force_per_atom.data(),
    virial_per_atom.data(),
    ener_unit_cvt_factor,
    force_unit_cvt_factor,
    virial_unit_cvt_factor,
    number_of_atoms
  );
  GPU_CHECK_KERNEL

}