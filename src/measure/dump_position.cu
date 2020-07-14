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
Dump position data to movie.xyz.
--------------------------------------------------------------------------------------------------*/

#include "dump_position.cuh"
#include "model/group.cuh"
#include "parse_group.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <vector>

void Dump_Position::parse(char** param, int num_param, const std::vector<Group>& groups)
{
  dump_ = true;
  printf("Dump position.\n");

  if (num_param < 2) {
    PRINT_INPUT_ERROR("dump_position should have at least 1 parameter.\n");
  }
  if (num_param > 7) {
    PRINT_INPUT_ERROR("dump_position has too many parameters.\n");
  }

  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("position dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("position dump interval should > 0.");
  }

  printf("    every %d steps.\n", dump_interval_);

  for (int k = 2; k < num_param; k++) {
    if (strcmp(param[k], "group") == 0) {
      parse_group(param, num_param, groups, k, grouping_method_, group_id_);
      if (group_id_ < 0) { // TODO: move to parse_group
        PRINT_INPUT_ERROR("group ID should >= 0.\n");
      }
    } else if (strcmp(param[k], "precision") == 0) {
      // TODO: move to parse_precision
      if (k + 2 > num_param) {
        PRINT_INPUT_ERROR("Not enough arguments for option 'precision'.\n");
      }
      if (strcmp(param[k + 1], "single") == 0) {
        precision_ = 1;
        printf("    with single precision.\n");
      } else if (strcmp(param[k + 1], "double") == 0) {
        precision_ = 2;
        printf("    with double  precision.\n");
      } else {
        PRINT_INPUT_ERROR("Invalid precision.\n");
      }
      k++;
    } else {
      PRINT_INPUT_ERROR("Unrecognized argument in dump_position.\n");
    }
  }
}

void Dump_Position::preprocess(char* input_dir)
{
  if (dump_) {
    strcpy(filename_, input_dir);
    strcat(filename_, "/movie.xyz");
    fid_ = my_fopen(filename_, "a");

    if (precision_ == 0)
      strcpy(precision_str_, "%d %g %g %g\n");
    else if (precision_ == 1) // single precision
      strcpy(precision_str_, "%d %0.9g %0.9g %0.9g\n");
    else if (precision_ == 2) // double precision
      strcpy(precision_str_, "%d %.17f %.17f %.17f\n");
  }
}

void Dump_Position::process(
  const int step,
  const std::vector<Group>& groups,
  const std::vector<int>& cpu_type,
  GPU_Vector<double>& position_per_atom,
  std::vector<double>& cpu_position_per_atom)
{
  if (!dump_)
    return;
  if ((step + 1) % dump_interval_ != 0)
    return;

  const int num_atoms_total = position_per_atom.size() / 3;

  if (grouping_method_ < 0) {
    position_per_atom.copy_to_host(cpu_position_per_atom.data());
    fprintf(fid_, "%d\n", num_atoms_total);
    fprintf(fid_, "%d\n", (step + 1) / dump_interval_ - 1);
    for (int n = 0; n < num_atoms_total; n++) {
      fprintf(
        fid_, precision_str_, cpu_type[n], cpu_position_per_atom[n],
        cpu_position_per_atom[n + num_atoms_total], cpu_position_per_atom[n + 2 * num_atoms_total]);
    }
  } else {
    const int group_size = groups[grouping_method_].cpu_size[group_id_];
    const int group_size_sum = groups[grouping_method_].cpu_size_sum[group_id_];

    for (int d = 0; d < 3; ++d) {
      double* cpu_v = cpu_position_per_atom.data() + num_atoms_total * d + group_size_sum;
      double* gpu_v = position_per_atom.data() + num_atoms_total * d + group_size_sum;
      CHECK(cudaMemcpy(cpu_v, gpu_v, sizeof(double) * group_size, cudaMemcpyDeviceToHost));
    }

    fprintf(fid_, "%d\n", group_size);
    fprintf(fid_, "%d\n", (step + 1) / dump_interval_ - 1);
    for (int n = 0; n < group_size; n++) {
      fprintf(
        fid_, precision_str_, cpu_type[n], cpu_position_per_atom[n + group_size_sum],
        cpu_position_per_atom[n + num_atoms_total + group_size_sum],
        cpu_position_per_atom[n + 2 * num_atoms_total + group_size_sum]);
    }
  }

  fflush(fid_);
}

void Dump_Position::postprocess()
{
  if (dump_) {
    fclose(fid_);
    dump_ = false;
    grouping_method_ = -1;
  }
}
