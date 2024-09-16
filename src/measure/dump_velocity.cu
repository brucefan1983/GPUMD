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

/*-----------------------------------------------------------------------------------------------100
Dump velocity data to a file at a given interval.
--------------------------------------------------------------------------------------------------*/

#include "dump_velocity.cuh"
#include "model/group.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <vector>

void Dump_Velocity::parse(const char** param, int num_param, const std::vector<Group>& groups)
{
  dump_ = true;
  printf("Dump velocity.\n");

  if (num_param != 2 && num_param != 5) {
    PRINT_INPUT_ERROR("dump_velocity should have 1 or 4 parameters.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("velocity dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("velocity dump interval should > 0.");
  }

  printf("    every %d steps.\n", dump_interval_);

  for (int k = 2; k < num_param; k++) {
    if (strcmp(param[k], "group") == 0) {
      parse_group(param, num_param, false, groups, k, grouping_method_, group_id_);
    } else {
      PRINT_INPUT_ERROR("Unrecognized argument in dump_velocity.\n");
    }
  }

  print_line_1();
  printf("Warning: Starting from GPUMD-v3.4, the velocity data in velocity.out will be in units of "
         "Angstrom/fs\n");
  printf("         Previously they are in units of Angstrom/ps.\n");
  printf("         The reason for this change is to make it consistent with the extendend XYZ "
         "format.\n");
  print_line_2();
}

void Dump_Velocity::preprocess()
{
  if (dump_) {
    fid_ = my_fopen("velocity.out", "a");
  }
}

__global__ void copy_velocity(
  const int num_atoms,
  const int offset,
  const int* g_group_contents,
  const double* g_vx_i,
  const double* g_vy_i,
  const double* g_vz_i,
  double* g_vx_o,
  double* g_vy_o,
  double* g_vz_o)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < num_atoms) {
    const int m = g_group_contents[offset + n];
    g_vx_o[n] = g_vx_i[m];
    g_vy_o[n] = g_vy_i[m];
    g_vz_o[n] = g_vz_i[m];
  }
}

void Dump_Velocity::process(
  const int step,
  const std::vector<Group>& groups,
  GPU_Vector<double>& velocity_per_atom,
  std::vector<double>& cpu_velocity_per_atom)
{
  if (!dump_)
    return;
  if ((step + 1) % dump_interval_ != 0)
    return;

  const int num_atoms_total = velocity_per_atom.size() / 3;
  const double natural_to_A_per_fs = 1.0 / TIME_UNIT_CONVERSION;

  if (grouping_method_ < 0) {
    velocity_per_atom.copy_to_host(cpu_velocity_per_atom.data());
    for (int n = 0; n < num_atoms_total; n++) {
      fprintf(
        fid_,
        "%g %g %g\n",
        cpu_velocity_per_atom[n] * natural_to_A_per_fs,
        cpu_velocity_per_atom[n + num_atoms_total] * natural_to_A_per_fs,
        cpu_velocity_per_atom[n + 2 * num_atoms_total] * natural_to_A_per_fs);
    }
  } else {
    const int group_size = groups[grouping_method_].cpu_size[group_id_];
    const int group_size_sum = groups[grouping_method_].cpu_size_sum[group_id_];
    GPU_Vector<double> gpu_velocity_tmp(group_size * 3);
    copy_velocity<<<(group_size - 1) / 128 + 1, 128>>>(
      group_size,
      group_size_sum,
      groups[grouping_method_].contents.data(),
      velocity_per_atom.data(),
      velocity_per_atom.data() + num_atoms_total,
      velocity_per_atom.data() + 2 * num_atoms_total,
      gpu_velocity_tmp.data(),
      gpu_velocity_tmp.data() + group_size,
      gpu_velocity_tmp.data() + group_size * 2);
    for (int d = 0; d < 3; ++d) {
      double* cpu_v = cpu_velocity_per_atom.data() + num_atoms_total * d;
      double* gpu_v = gpu_velocity_tmp.data() + group_size * d;
      CHECK(cudaMemcpy(cpu_v, gpu_v, sizeof(double) * group_size, cudaMemcpyDeviceToHost));
    }
    for (int n = 0; n < group_size; n++) {
      fprintf(
        fid_,
        "%g %g %g\n",
        cpu_velocity_per_atom[n] * natural_to_A_per_fs,
        cpu_velocity_per_atom[n + num_atoms_total] * natural_to_A_per_fs,
        cpu_velocity_per_atom[n + 2 * num_atoms_total] * natural_to_A_per_fs);
    }
  }

  fflush(fid_);
}

void Dump_Velocity::postprocess()
{
  if (dump_) {
    fclose(fid_);
    dump_ = false;
    grouping_method_ = -1;
  }
}
