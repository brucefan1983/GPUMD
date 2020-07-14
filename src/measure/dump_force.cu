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
Dump force data to a file at a given interval.
--------------------------------------------------------------------------------------------------*/

#include "dump_force.cuh"
#include "model/group.cuh"
#include "parse_group.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <vector>

void Dump_Force::parse(char** param, int num_param, const std::vector<Group>& groups)
{
  dump_ = true;
  printf("Dump force every %d steps.\n", dump_interval_);

  if (num_param != 2 && num_param != 5) {
    PRINT_INPUT_ERROR("dump_force should have 1 or 4 parameters.");
  }
  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("force dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("force dump interval should > 0.");
  }

  for (int k = 2; k < num_param; k++) {
    if (strcmp(param[k], "group") == 0) {
      parse_group(param, num_param, false, groups, k, grouping_method_, group_id_);
    } else {
      PRINT_INPUT_ERROR("Unrecognized argument in dump_force.\n");
    }
  }
}

void Dump_Force::preprocess(
  char* input_dir, const int number_of_atoms, const std::vector<Group>& groups)
{
  if (dump_) {
    strcpy(filename_, input_dir);
    strcat(filename_, "/force.out");
    fid_ = my_fopen(filename_, "a");

    if (grouping_method_ < 0)
      cpu_force_per_atom.resize(number_of_atoms * 3);
    else
      cpu_force_per_atom.resize(groups[grouping_method_].cpu_size[group_id_] * 3);
  }
}

void Dump_Force::process(
  const int step, const std::vector<Group>& groups, GPU_Vector<double>& force_per_atom)
{
  if (!dump_)
    return;
  if ((step + 1) % dump_interval_ != 0)
    return;

  const int number_of_atoms = force_per_atom.size() / 3;

  if (grouping_method_ < 0) {
    force_per_atom.copy_to_host(cpu_force_per_atom.data());
    for (int n = 0; n < number_of_atoms; n++) {
      fprintf(
        fid_, "%g %g %g\n", cpu_force_per_atom[n], cpu_force_per_atom[n + number_of_atoms],
        cpu_force_per_atom[n + 2 * number_of_atoms]);
    }
  } else {
    const int group_size = groups[grouping_method_].cpu_size[group_id_];
    const int group_size_sum = groups[grouping_method_].cpu_size_sum[group_id_];

    for (int d = 0; d < 3; ++d) {
      double* cpu_f = cpu_force_per_atom.data() + group_size * d;
      double* gpu_f = force_per_atom.data() + number_of_atoms * d + group_size_sum;
      CHECK(cudaMemcpy(cpu_f, gpu_f, sizeof(double) * group_size, cudaMemcpyDeviceToHost));
    }
    for (int n = 0; n < group_size; n++) {
      fprintf(
        fid_, "%g %g %g\n", cpu_force_per_atom[n], cpu_force_per_atom[n + group_size],
        cpu_force_per_atom[n + 2 * group_size]);
    }
  }

  fflush(fid_);
}

void Dump_Force::postprocess()
{
  if (dump_) {
    fclose(fid_);
    dump_ = false;
    grouping_method_ = -1;
  }
}
