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
Add electric field to a group of atoms.
------------------------------------------------------------------------------*/

#include "add_efield.cuh"
#include "model/atom.cuh"
#include "model/group.cuh"
#include "utilities/read_file.cuh"
#include <iostream>
#include <vector>

static void __global__
add_efield(
  const int group_size,
  const int group_size_sum,
  const int* g_group_contents,
  const double Ex, 
  const double Ey,
  const double Ez,
  const double* g_charge,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < group_size) {
    const int atom_id = g_group_contents[group_size_sum + tid];
    const double charge = g_charge[atom_id];
    g_fx[atom_id] += charge * Ex;
    g_fy[atom_id] += charge * Ey;
    g_fz[atom_id] += charge * Ez;
  }
}

void Add_Efeild::compute(const int step, const std::vector<Group>& groups, Atom& atom)
{
  for (int call = 0; call < num_calls_; ++call) {
    const int step_mod_table_length = step % table_length_[call];
    const double Ex = efield_table_[call][0 * table_length_[call] + step_mod_table_length];
    const double Ey = efield_table_[call][1 * table_length_[call] + step_mod_table_length];
    const double Ez = efield_table_[call][2 * table_length_[call] + step_mod_table_length];
    const int num_atoms_total = atom.force_per_atom.size() / 3;
    const int group_size = groups[grouping_method_[call]].cpu_size[group_id_[call]];
    const int group_size_sum = groups[grouping_method_[call]].cpu_size_sum[group_id_[call]];
    add_efield<<<(group_size - 1) / 64 + 1, 64>>>(
      group_size,
      group_size_sum,
      groups[grouping_method_[call]].contents.data(),
      Ex,
      Ey,
      Ez,
      atom.charge.data(),
      atom.force_per_atom.data(),
      atom.force_per_atom.data() + num_atoms_total,
      atom.force_per_atom.data() + num_atoms_total * 2
    );
    CUDA_CHECK_KERNEL
  }
}

void Add_Efeild::parse(const char** param, int num_param, const std::vector<Group>& group)
{
  printf("Add electric field.\n");

  // check the number of parameters
  if (num_param != 6 && num_param != 4) {
    PRINT_INPUT_ERROR("add_efield should have 5 or 3 parameters.\n");
  }

  // parse grouping method
  if (!is_valid_int(param[1], &grouping_method_[num_calls_])) {
    PRINT_INPUT_ERROR("grouping method should be an integer.\n");
  }
  if (grouping_method_[num_calls_] < 0) {
    PRINT_INPUT_ERROR("grouping method should >= 0.\n");
  }
  if (grouping_method_[num_calls_] >= group.size()) {
    PRINT_INPUT_ERROR("grouping method should < maximum number of grouping methods.\n");
  }

  // parse group id
  if (!is_valid_int(param[2], &group_id_[num_calls_])) {
    PRINT_INPUT_ERROR("group id should be an integer.\n");
  }
  if (group_id_[num_calls_] < 0) {
    PRINT_INPUT_ERROR("group id should >= 0.\n");
  }
  if (group_id_[num_calls_] >= group[grouping_method_[num_calls_]].number) {
    PRINT_INPUT_ERROR("group id should < maximum number of groups in the grouping method.\n");
  }

  printf(
    "    for atoms in group %d of grouping method %d.\n", 
    group_id_[num_calls_], 
    grouping_method_[num_calls_]
  );

  if (num_param == 6) {
    table_length_[num_calls_] = 1;
    efield_table_[num_calls_].resize(table_length_[num_calls_] * 3);
    if (!is_valid_real(param[3], &efield_table_[num_calls_][0])) {
      PRINT_INPUT_ERROR("Ex should be a number.\n");
    }
    if (!is_valid_real(param[4], &efield_table_[num_calls_][1])) {
      PRINT_INPUT_ERROR("Ey should be a number.\n");
    }
    if (!is_valid_real(param[5], &efield_table_[num_calls_][2])) {
      PRINT_INPUT_ERROR("Ez should be a number.\n");
    }
    printf("    Ex = %g V/A.\n", efield_table_[num_calls_][0]);
    printf("    Ey = %g V/A.\n", efield_table_[num_calls_][1]);
    printf("    Ez = %g V/A.\n", efield_table_[num_calls_][2]);
  } else {
    std::ifstream input(param[3]);
    if (!input.is_open()) {
      printf("Failed to open %s.\n", param[3]);
      exit(1);
    }

    std::vector<std::string> tokens = get_tokens(input);
    if (tokens.size() != 1) {
      PRINT_INPUT_ERROR("The first line of the add_efield file should have 1 value.");
    }
    table_length_[num_calls_] = get_int_from_token(tokens[0], __FILE__, __LINE__);
    if (table_length_[num_calls_] < 2) {
      PRINT_INPUT_ERROR("Number of steps in the add_efield file should >= 2.\n");
    } else {
      printf("    number of values in the add_efield file = %d.\n", table_length_[num_calls_]);
    }

    efield_table_[num_calls_].resize(table_length_[num_calls_] * 3);
    for (int n = 0; n < table_length_[num_calls_]; ++n) {
      std::vector<std::string> tokens = get_tokens(input);
      if (tokens.size() != 3) {
        PRINT_INPUT_ERROR("Number of electric field components at each step should be 3.");
      }
      for (int t = 0; t < 3; ++t) {
        efield_table_[num_calls_][t * table_length_[num_calls_] + n] = get_double_from_token(tokens[t], __FILE__, __LINE__);
      }
    }
  }

  ++num_calls_;

  if (num_calls_ > 10) {
    PRINT_INPUT_ERROR("add_efield cannot be used more than 10 times in one run.");
  }
}

void Add_Efeild::finalize() 
{ 
  num_calls_ = 0;
}
