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
#include "model/box.cuh"
#include "model/group.cuh"
#include "parse_utilities.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"

void Dump_Position::parse(const char** param, int num_param, const std::vector<Group>& groups)
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
      parse_group(param, num_param, false, groups, k, grouping_method_, group_id_);
    } else if (strcmp(param[k], "precision") == 0) {
      parse_precision(param, num_param, k, precision_);
    } else {
      PRINT_INPUT_ERROR("Unrecognized argument in dump_position.\n");
    }
  }
}

void Dump_Position::preprocess()
{
  if (dump_) {
    fid_ = my_fopen("movie.xyz", "a");
    if (precision_ == 0)
      strcpy(precision_str_, "%s %g %g %g\n");
    else if (precision_ == 1) // single precision
      strcpy(precision_str_, "%s %0.9g %0.9g %0.9g\n");
    else if (precision_ == 2) // double precision
      strcpy(precision_str_, "%s %.17f %.17f %.17f\n");
  }
}

__global__ void copy_position(
  const int num_atoms,
  const int offset,
  const int* g_group_contents,
  const double* g_x_i,
  const double* g_y_i,
  const double* g_z_i,
  double* g_x_o,
  double* g_y_o,
  double* g_z_o)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < num_atoms) {
    const int m = g_group_contents[offset + n];
    g_x_o[n] = g_x_i[m];
    g_y_o[n] = g_y_i[m];
    g_z_o[n] = g_z_i[m];
  }
}

void Dump_Position::output_line2(const Box& box, const std::vector<std::string>& cpu_atom_symbol)
{
  if (box.triclinic == 0) {
    fprintf(
      fid_,
      "Lattice=\"%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e\" "
      "Properties=species:S:1:pos:R:3\n",
      box.cpu_h[0],
      0.0,
      0.0,
      0.0,
      box.cpu_h[1],
      0.0,
      0.0,
      0.0,
      box.cpu_h[2]);
  } else {
    fprintf(
      fid_,
      "Lattice=\"%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e\" "
      "Properties=species:S:1:pos:R:3\n",
      box.cpu_h[0],
      box.cpu_h[3],
      box.cpu_h[6],
      box.cpu_h[1],
      box.cpu_h[4],
      box.cpu_h[7],
      box.cpu_h[2],
      box.cpu_h[5],
      box.cpu_h[8]);
  }
}

void Dump_Position::process(
  const int step,
  const Box& box,
  const std::vector<Group>& groups,
  const std::vector<std::string>& cpu_atom_symbol,
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
    output_line2(box, cpu_atom_symbol);
    for (int n = 0; n < num_atoms_total; n++) {
      fprintf(
        fid_,
        precision_str_,
        cpu_atom_symbol[n].c_str(),
        cpu_position_per_atom[n],
        cpu_position_per_atom[n + num_atoms_total],
        cpu_position_per_atom[n + 2 * num_atoms_total]);
    }
  } else {
    const int group_size = groups[grouping_method_].cpu_size[group_id_];
    const int group_size_sum = groups[grouping_method_].cpu_size_sum[group_id_];
    GPU_Vector<double> gpu_position_tmp(group_size * 3);
    copy_position<<<(group_size - 1) / 128 + 1, 128>>>(
      group_size,
      group_size_sum,
      groups[grouping_method_].contents.data(),
      position_per_atom.data(),
      position_per_atom.data() + num_atoms_total,
      position_per_atom.data() + 2 * num_atoms_total,
      gpu_position_tmp.data(),
      gpu_position_tmp.data() + group_size,
      gpu_position_tmp.data() + group_size * 2);
    for (int d = 0; d < 3; ++d) {
      double* cpu_data = cpu_position_per_atom.data() + num_atoms_total * d;
      double* gpu_data = gpu_position_tmp.data() + group_size * d;
      CHECK(cudaMemcpy(cpu_data, gpu_data, sizeof(double) * group_size, cudaMemcpyDeviceToHost));
    }
    fprintf(fid_, "%d\n", group_size);
    output_line2(box, cpu_atom_symbol);
    for (int n = 0; n < group_size; n++) {
      fprintf(
        fid_,
        precision_str_,
        cpu_atom_symbol[groups[grouping_method_].cpu_contents[group_size_sum + n]].c_str(),
        cpu_position_per_atom[n],
        cpu_position_per_atom[n + num_atoms_total],
        cpu_position_per_atom[n + 2 * num_atoms_total]);
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
    precision_ = 0;
  }
}
