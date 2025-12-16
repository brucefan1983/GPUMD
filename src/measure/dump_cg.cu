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
Dump training data for NEP-CG
--------------------------------------------------------------------------------------------------*/

#include "dump_cg.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <iostream>
#include <vector>
#include <string>

static __global__ void gpu_sum(const int N, const double* g_data, double* g_data_sum)
{
  int number_of_rounds = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[threadIdx.x] = 0.0;
  for (int round = 0; round < number_of_rounds; ++round) {
    int n = threadIdx.x + round * 1024;
    if (n < N) {
      s_data[threadIdx.x] += g_data[n + blockIdx.x * N];
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      s_data[threadIdx.x] += s_data[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    g_data_sum[blockIdx.x] = s_data[0];
  }
}

Dump_CG::Dump_CG(const char** param, int num_param, std::vector<Group>& group) 
{
  parse(param, num_param, group);
  property_name = "dump_cg";
}

void Dump_CG::parse(const char** param, int num_param, std::vector<Group>& group)
{
  printf("Dump train.xyz for NEP-CG.\n");

  if (num_param < 3) {
    PRINT_INPUT_ERROR("dump_cg should have at least 2 parameters.\n");
  }

  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("dump interval should > 0.");
  }

  printf("    every %d steps.\n", dump_interval_);

  if (!is_valid_int(param[2], &grouping_method_)) {
    PRINT_INPUT_ERROR("grouping method should be an integer.");
  }
  if (grouping_method_ < 0) {
    PRINT_INPUT_ERROR("grouping method should >= 0.");
  }
  if (grouping_method_ >= group.size()) {
    PRINT_INPUT_ERROR("grouping method should < number of grouping methods.");
  }

  printf("    using grouping method %d to define beads.\n", grouping_method_);
}

void Dump_CG::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  fid_ = my_fopen("train.xyz", "a");

  gpu_total_virial_.resize(6);
  cpu_total_virial_.resize(6);
  cpu_force_per_atom_.resize(atom.number_of_atoms * 3);
  bead_name_.resize(atom.number_of_atoms);

  std::ifstream input("bead_name.txt");
  if (!input.is_open()) {
    std::cout << "Failed to open bead_name.txt." << std::endl;
    exit(1);
  } else {
    for (int n = 0; n < atom.number_of_atoms; ++n) {
      std::vector<std::string> tokens = get_tokens(input);
      if (tokens.size() != 1) {
        std::cout << "Each line of bead_name.txt should have one value." << std::endl;
        exit(1);
      }
      bead_name_[n] = tokens[0];
    }
    input.close();
  }
}

void Dump_CG::output_line2(
  FILE* fid,
  const Box& box,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& gpu_thermo,
  double relative_dof)
{
  // PBC
  fprintf(
    fid, " pbc=\"%c %c %c\"", box.pbc_x ? 'T' : 'F', box.pbc_y ? 'T' : 'F', box.pbc_z ? 'T' : 'F');

  // box
  fprintf(
    fid,
    " Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"",
    box.cpu_h[0],
    box.cpu_h[3],
    box.cpu_h[6],
    box.cpu_h[1],
    box.cpu_h[4],
    box.cpu_h[7],
    box.cpu_h[2],
    box.cpu_h[5],
    box.cpu_h[8]);

  // energy and virial (symmetric tensor) in eV
  double cpu_thermo[8];
  gpu_thermo.copy_to_host(cpu_thermo, 8);
  const int N = virial_per_atom.size() / 9;
  gpu_sum<<<6, 1024>>>(N, virial_per_atom.data(), gpu_total_virial_.data());
  gpu_total_virial_.copy_to_host(cpu_total_virial_.data());

  fprintf(fid, " energy=%.8f", cpu_thermo[1] * relative_dof);
  fprintf(
    fid,
    " virial=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"",
    cpu_total_virial_[0] * relative_dof,
    cpu_total_virial_[3] * relative_dof,
    cpu_total_virial_[4] * relative_dof,
    cpu_total_virial_[3] * relative_dof,
    cpu_total_virial_[1] * relative_dof,
    cpu_total_virial_[5] * relative_dof,
    cpu_total_virial_[4] * relative_dof,
    cpu_total_virial_[5] * relative_dof,
    cpu_total_virial_[2] * relative_dof);

  // Properties
  fprintf(fid, " Properties=species:S:1:pos:R:3:forces:R:3\n");
}

void Dump_CG::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  if ((step + 1) % dump_interval_ != 0)
    return;

  Group& g = group[grouping_method_];
  const int num_atoms_total = atom.number_of_atoms;
  const int num_beads = g.number;
  atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  atom.force_per_atom.copy_to_host(cpu_force_per_atom_.data());

  int max_bead_size = 0;
  for (int b = 0; b < num_beads; ++b) {
    if (max_bead_size < g.cpu_size[b]) {
      max_bead_size = g.cpu_size[b];
    }
  }
  double relative_dof = double(num_beads) / num_atoms_total;

  // line 1
  fprintf(fid_, "%d\n", num_beads);

  // line 2
  output_line2(fid_, box, atom.virial_per_atom, thermo, relative_dof);

  std::vector<double> xyz_bead(max_bead_size * 3);
  std::vector<double> mass_bead(max_bead_size);
  std::vector<double> force_bead(max_bead_size * 3);

  // other lines
  for (int b = 0; b < num_beads; b++) {

    for (int k = 0; k < g.cpu_size[b]; ++k) {
      int n = g.cpu_contents[g.cpu_size_sum[b] + k];
      mass_bead[k] = atom.cpu_mass[n];
      for (int d = 0; d < 3; ++d) {
        xyz_bead[k + max_bead_size * d] = atom.cpu_position_per_atom[n + num_atoms_total * d];
        force_bead[k + max_bead_size * d] = cpu_force_per_atom_[n + num_atoms_total * d];
      }
    }

    for (int k = 1; k < g.cpu_size[b]; ++k) {
      double pos_diff[3];
      for (int d = 0; d < 3; ++d) {
        pos_diff[d] = xyz_bead[k + max_bead_size * d] - xyz_bead[0 + max_bead_size * d];
      }
      apply_mic(box, pos_diff[0], pos_diff[1], pos_diff[2]);
      for (int d = 0; d < 3; ++d) {
        xyz_bead[k + max_bead_size * d] = xyz_bead[0 + max_bead_size * d] + pos_diff[d];
      }
    }
   
    double r_com[3] = {0.0, 0.0, 0.0};
    double m_com = 0;
    for (int k = 0; k < g.cpu_size[b]; ++k) {
      m_com += mass_bead[k];
      for (int d = 0; d < 3; ++d) {
        r_com[d] += xyz_bead[k + max_bead_size * d] * mass_bead[k];
      }
    }

    fprintf(fid_, "%s", bead_name_[g.cpu_contents[g.cpu_size_sum[b] + 0]].c_str());
    for (int d = 0; d < 3; ++d) {
      r_com[d] /= m_com;
      fprintf(fid_, " %.8f", r_com[d]);
    }

    double f_com[3] = {0.0, 0.0, 0.0};
    for (int k = 0; k < g.cpu_size[b]; ++k) {
      for (int d = 0; d < 3; ++d) {
        f_com[d] += force_bead[k + max_bead_size * d];
      }
    }

    for (int d = 0; d < 3; ++d) {
      fprintf(fid_, " %.8f", f_com[d]);
    }
    fprintf(fid_, "\n");
  }
  fflush(fid_);
}

void Dump_CG::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  fclose(fid_);
}
