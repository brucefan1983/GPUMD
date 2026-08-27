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
Add electric field to a group of atoms.
------------------------------------------------------------------------------*/

#include "add_efield.cuh"
#include "force/force.cuh"
#include "model/atom.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <iostream>
#include <vector>

static void __global__ add_efield(
  const int group_size,
  const int group_size_sum,
  const int* g_group_contents,
  const double Ex,
  const double Ey,
  const double Ez,
  const float* g_charge,
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

static void __global__ add_efield_bec(
  const int num_atoms,
  const int group_size,
  const int group_size_sum,
  const int* g_group_contents,
  const double Ex,
  const double Ey,
  const double Ez,
  const float* g_bec,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < group_size) {
    const int atom_id = g_group_contents[group_size_sum + tid];
    const double bec[9] = {
      g_bec[atom_id + 0 * num_atoms],
      g_bec[atom_id + 1 * num_atoms],
      g_bec[atom_id + 2 * num_atoms],
      g_bec[atom_id + 3 * num_atoms],
      g_bec[atom_id + 4 * num_atoms],
      g_bec[atom_id + 5 * num_atoms],
      g_bec[atom_id + 6 * num_atoms],
      g_bec[atom_id + 7 * num_atoms],
      g_bec[atom_id + 8 * num_atoms]
    };
    g_fx[atom_id] += bec[0] * Ex + bec[3] * Ey + bec[6] * Ez;
    g_fy[atom_id] += bec[1] * Ex + bec[4] * Ey + bec[7] * Ez;
    g_fz[atom_id] += bec[2] * Ex + bec[5] * Ey + bec[8] * Ez;
  }
}

void Add_Efield::post_force(
  const int step,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  const int step_mod_table_length = step % table_length_;
  const double Ex = efield_table_[0 * table_length_ + step_mod_table_length];
  const double Ey = efield_table_[1 * table_length_ + step_mod_table_length];
  const double Ez = efield_table_[2 * table_length_ + step_mod_table_length];
  const int num_atoms_total = atom.force_per_atom.size() / 3;
  const int group_size = group[grouping_method_].cpu_size[group_id_];
  const int group_size_sum = group[grouping_method_].cpu_size_sum[group_id_];

  if (use_bec_) {
    GPU_Vector<float>& bec = force.potentials[0]->get_bec_reference();
    add_efield_bec<<<(group_size - 1) / 64 + 1, 64>>>(
      num_atoms_total,
      group_size,
      group_size_sum,
      group[grouping_method_].contents.data(),
      Ex,
      Ey,
      Ez,
      bec.data(),
      atom.force_per_atom.data(),
      atom.force_per_atom.data() + num_atoms_total,
      atom.force_per_atom.data() + num_atoms_total * 2);
  } else if (is_nep_charge_) {
    GPU_Vector<float>& nep_charge = force.potentials[0]->get_charge_reference();
    add_efield<<<(group_size - 1) / 64 + 1, 64>>>(
      group_size,
      group_size_sum,
      group[grouping_method_].contents.data(),
      Ex,
      Ey,
      Ez,
      nep_charge.data(),
      atom.force_per_atom.data(),
      atom.force_per_atom.data() + num_atoms_total,
      atom.force_per_atom.data() + num_atoms_total * 2);
  } else {
    add_efield<<<(group_size - 1) / 64 + 1, 64>>>(
      group_size,
      group_size_sum,
      group[grouping_method_].contents.data(),
      Ex,
      Ey,
      Ez,
      atom.charge.data(),
      atom.force_per_atom.data(),
      atom.force_per_atom.data() + num_atoms_total,
      atom.force_per_atom.data() + num_atoms_total * 2);
  }
  GPU_CHECK_KERNEL
}

Add_Efield::Add_Efield(const char** param, int num_param, const std::vector<Group>& group)
{
  property_name = "add_efield";
  printf("Add electric field.\n");

  bool use_file_input = false;
  is_nep_charge_ = check_is_nep_charge();
  std::string mode_str = is_nep_charge_ ? "bec" : "charge";

  if (num_param == 7) {
    mode_str = param[6];
    use_file_input = false;
  } else if (num_param == 5) {
    mode_str = param[4];
    use_file_input = true;
  } else if (num_param == 6) {
    use_file_input = false;
  } else if (num_param == 4) {
    use_file_input = true;
  } else {
    PRINT_INPUT_ERROR("add_efield should have 3, 4, 5 or 6 parameters.\n");
  }

  if (mode_str != "charge" && mode_str != "bec") {
    PRINT_INPUT_ERROR("Mode can only be charge or bec.\n");
  }

  if (is_nep_charge_) {
    if (mode_str == "bec") {
      use_bec_ = true;
      printf("    using the BEC values predicted by the NEP-Charge model.\n");
    } else {
      use_bec_ = false;
      printf("    using the charge values predicted by the NEP-Charge model.\n");
    }
  } else {
    if (mode_str == "bec") {
      PRINT_INPUT_ERROR("Cannot use bec for non-qNEP models.\n");
      printf("    using the charge values specified in model.xyz.\n");
    } else {
      use_bec_ = false;
    }
  }

  // parse grouping method
  if (!is_valid_int(param[1], &grouping_method_)) {
    PRINT_INPUT_ERROR("grouping method should be an integer.\n");
  }
  if (grouping_method_ < 0) {
    PRINT_INPUT_ERROR("grouping method should >= 0.\n");
  }
  if (grouping_method_ >= group.size()) {
    PRINT_INPUT_ERROR("grouping method should < maximum number of grouping methods.\n");
  }

  // parse group id
  if (!is_valid_int(param[2], &group_id_)) {
    PRINT_INPUT_ERROR("group id should be an integer.\n");
  }
  if (group_id_ < 0) {
    PRINT_INPUT_ERROR("group id should >= 0.\n");
  }
  if (group_id_ >= group[grouping_method_].number) {
    PRINT_INPUT_ERROR("group id should < maximum number of groups in the grouping method.\n");
  }

  printf(
    "    for atoms in group %d of grouping method %d.\n",
    group_id_,
    grouping_method_);

  if (!use_file_input) {
    table_length_ = 1;
    efield_table_.resize(table_length_ * 3);
    if (!is_valid_real(param[3], &efield_table_[0])) {
      PRINT_INPUT_ERROR("Ex should be a number.\n");
    }
    if (!is_valid_real(param[4], &efield_table_[1])) {
      PRINT_INPUT_ERROR("Ey should be a number.\n");
    }
    if (!is_valid_real(param[5], &efield_table_[2])) {
      PRINT_INPUT_ERROR("Ez should be a number.\n");
    }
    printf("    Ex = %g V/A.\n", efield_table_[0]);
    printf("    Ey = %g V/A.\n", efield_table_[1]);
    printf("    Ez = %g V/A.\n", efield_table_[2]);
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
    table_length_ = get_int_from_token(tokens[0], __FILE__, __LINE__);
    if (table_length_ < 2) {
      PRINT_INPUT_ERROR("Number of steps in the add_efield file should >= 2.\n");
    } else {
      printf("    number of values in the add_efield file = %d.\n", table_length_);
    }

    efield_table_.resize(table_length_ * 3);
    for (int n = 0; n < table_length_; ++n) {
      std::vector<std::string> tokens = get_tokens(input);
      if (tokens.size() != 3) {
        PRINT_INPUT_ERROR("Number of electric field components at each step should be 3.");
      }
      for (int t = 0; t < 3; ++t) {
        efield_table_[t * table_length_ + n] =
          get_double_from_token(tokens[t], __FILE__, __LINE__);
      }
    }
  }
}
