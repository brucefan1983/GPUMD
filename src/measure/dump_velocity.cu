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

/*-----------------------------------------------------------------------------------------------100
Dump velocity data to a file at a given interval.
--------------------------------------------------------------------------------------------------*/

#include "dump_velocity.cuh"
#include "model/group.cuh"
#include "parse_group.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <vector>

void Dump_Velocity::parse(char** param, int num_param, const std::vector<Group>& groups)
{
  dump_ = true;
  printf("Dump velocity every %d steps.\n", dump_interval_);

  if (num_param != 2 && num_param != 5) {
    PRINT_INPUT_ERROR("dump_velocity should have 1 or 4 parameters.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("velocity dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("velocity dump interval should > 0.");
  }

  for (int k = 2; k < num_param; k++) {
    if (strcmp(param[k], "group") == 0) {
      parse_group(param, groups, k, grouping_method_, group_id_);
      if (group_id_ < 0) {
        PRINT_INPUT_ERROR("group ID should >= 0.\n");
      }
      printf("    grouping_method is %d and group is %d.\n", grouping_method_, group_id_);
    } else {
      PRINT_INPUT_ERROR("Unrecognized argument in dump_velocity.\n");
    }
  }
}

void Dump_Velocity::preprocess(char* input_dir)
{
  if (dump_) {
    strcpy(filename_, input_dir);
    strcat(filename_, "/velocity.out");
    fid_ = my_fopen(filename_, "a");
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

  if (grouping_method_ < 0) {
    velocity_per_atom.copy_to_host(cpu_velocity_per_atom.data());
    for (int n = 0; n < num_atoms_total; n++) {
      fprintf(
        fid_, "%g %g %g\n", cpu_velocity_per_atom[n], cpu_velocity_per_atom[n + num_atoms_total],
        cpu_velocity_per_atom[n + 2 * num_atoms_total]);
    }
  } else {
    const int group_size = groups[grouping_method_].cpu_size[group_id_];
    const int group_size_sum = groups[grouping_method_].cpu_size_sum[group_id_];

    for (int d = 0; d < 3; ++d) {
      double* cpu_v = cpu_velocity_per_atom.data() + num_atoms_total * d + group_size_sum;
      double* gpu_v = velocity_per_atom.data() + num_atoms_total * d + group_size_sum;
      CHECK(cudaMemcpy(cpu_v, gpu_v, sizeof(double) * group_size, cudaMemcpyDeviceToHost));
    }
    for (int n = 0; n < group_size; n++) {
      fprintf(
        fid_, "%g %g %g\n", cpu_velocity_per_atom[n + group_size_sum],
        cpu_velocity_per_atom[n + num_atoms_total + group_size_sum],
        cpu_velocity_per_atom[n + 2 * num_atoms_total + group_size_sum]);
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
