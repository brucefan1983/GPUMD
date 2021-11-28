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
Compute the cohesive energy curve.
------------------------------------------------------------------------------*/

#include "cohesive.cuh"
#include "deform_box.cuh"
#include "force/force.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "model/neighbor.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <vector>

void Cohesive::parse(char** param, int num_param)
{
  printf("Compute cohesive energy.\n");
  if (num_param != 4) {
    PRINT_INPUT_ERROR("compute_cohesive should have 4 parameters.\n");
  }

  if (!is_valid_real(param[1], &start_factor)) {
    PRINT_INPUT_ERROR("start_factor should be a number.\n");
  }
  if (start_factor <= 0) {
    PRINT_INPUT_ERROR("start_factor should be positive.\n");
  }
  printf("    start_factor = %g.\n", start_factor);

  if (!is_valid_real(param[2], &end_factor)) {
    PRINT_INPUT_ERROR("end_factor should be a number.\n");
  }
  if (end_factor <= start_factor) {
    PRINT_INPUT_ERROR("end_factor should > start_factor.\n");
  }
  printf("    end_factor = %g.\n", end_factor);

  if (!is_valid_int(param[3], &num_points)) {
    PRINT_INPUT_ERROR("num_points should be an integer.\n");
  }
  if (num_points < 2) {
    PRINT_INPUT_ERROR("num_points should >= 2.\n");
  }
  printf("    num_points = %d.\n", num_points);

  delta_factor = (end_factor - start_factor) / (num_points - 1);
}

void Cohesive::compute(
  char* input_dir,
  const Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  Neighbor& neighbor,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  Force& force)
{
  const int num_atoms = potential_per_atom.size();

  Box new_box;
  new_box.pbc_x = box.pbc_x;
  new_box.pbc_y = box.pbc_y;
  new_box.pbc_z = box.pbc_z;
  new_box.triclinic = box.triclinic;

  GPU_Vector<double> gpu_D(9);
  GPU_Vector<double> new_position_per_atom(num_atoms * 3);
  std::vector<double> cpu_potential_per_atom(num_atoms);

  char file_cohesive[200];
  strcpy(file_cohesive, input_dir);
  strcat(file_cohesive, "/cohesive.out");
  FILE* fid_cohesive = my_fopen(file_cohesive, "w");

  for (int n = 0; n < num_points; ++n) {
    const double factor = start_factor + delta_factor * n;
    const double cpu_D[9] = {factor, 0, 0, 0, factor, 0, 0, 0, factor};
    gpu_D.copy_from_host(cpu_D);
    deform_box(
      num_atoms, cpu_D, gpu_D.data(), box, new_box, position_per_atom, new_position_per_atom);
    force.compute(
      new_box, new_position_per_atom, type, group, neighbor, potential_per_atom, force_per_atom,
      virial_per_atom);
    potential_per_atom.copy_to_host(cpu_potential_per_atom.data());
    double cpu_potential_total = 0.0;
    for (int i = 0; i < num_atoms; ++i) {
      cpu_potential_total += cpu_potential_per_atom[i];
    }
    fprintf(fid_cohesive, "%15.7e%15.7e\n", factor, cpu_potential_total);
  }
  printf("Cohesive energies have been computed.\n");
}
