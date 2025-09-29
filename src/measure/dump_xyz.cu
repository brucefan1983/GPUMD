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
Dump per-atom data to user-specified file(s) in the extended XYZ format
--------------------------------------------------------------------------------------------------*/

#include "dump_xyz.cuh"
#include "force/force.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cstring>

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

Dump_XYZ::Dump_XYZ(const char** param, int num_param, const std::vector<Group>& groups, Atom& atom) 
{
  is_nep_charge = check_is_nep_charge();

  parse(param, num_param, groups);
  if (atom.unwrapped_position.size() < atom.number_of_atoms * 3) {
    atom.unwrapped_position.resize(atom.number_of_atoms * 3);
    atom.unwrapped_position.copy_from_device(atom.position_per_atom.data());
  }
  if (atom.position_temp.size() < atom.number_of_atoms * 3) {
    atom.position_temp.resize(atom.number_of_atoms * 3);
  }
  property_name = "dump_xyz";
}

void Dump_XYZ::parse(const char** param, int num_param, const std::vector<Group>& groups)
{
  printf("Dump extended XYZ.\n");

  if (num_param < 5) {
    PRINT_INPUT_ERROR("dump_xyz should have at least 4 parameters.\n");
  }

  // grouping_method
  if (!is_valid_int(param[1], &grouping_method_)) {
    PRINT_INPUT_ERROR("grouping method of dump_xyz should be integer.");
  }
  if (grouping_method_ < 0) {
    printf("    for the whole system.\n");
  } else {
    if (grouping_method_ >= int(groups.size())) {
      PRINT_INPUT_ERROR("grouping method exceeds the bound.");
    }
    printf("    for grouping method %d.\n", grouping_method_);
  }

  // group_id
  if (!is_valid_int(param[2], &group_id_)) {
    PRINT_INPUT_ERROR("group id of dump_xyz should be integer.");
  }
  if (grouping_method_ >= 0) {
    if (group_id_ >= groups[grouping_method_].number) {
      PRINT_INPUT_ERROR("group id exceeds the bound.");
    }
    if (group_id_ < 0) {
      PRINT_INPUT_ERROR("group id is negative.");
    }
    printf("    for group id %d.\n", group_id_);
  }

  if (!is_valid_int(param[3], &dump_interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("dump interval should > 0.");
  } else {
    printf("    every %d steps.\n", dump_interval_);
  }

  // filename
  std::string filename_temp = param[4];
  printf("    into file %s.\n", filename_temp.c_str());
  if (filename_temp.back() == '*') {
    separated_ = 1;
    filename_ = filename_temp.substr(0, filename_temp.size() - 1);
  } else {
    separated_ = 0;
    filename_ = filename_temp;
  }

  for (int m = 5; m < num_param; ++m) {
    if (strcmp(param[m], "velocity") == 0) {
      quantities.has_velocity_ = true;
      printf("    has velocity.\n");
    }
    if (strcmp(param[m], "force") == 0) {
      quantities.has_force_ = true;
      printf("    has force.\n");
    }
    if (strcmp(param[m], "potential") == 0) {
      quantities.has_potential_ = true;
      printf("    has potential.\n");
    }
    if (strcmp(param[m], "unwrapped_position") == 0) {
      quantities.has_unwrapped_position_ = true;
      printf("    has unwrapped position.\n");
    }
    if (strcmp(param[m], "mass") == 0) {
      quantities.has_mass_ = true;
      printf("    has mass.\n");
    }
    if (strcmp(param[m], "charge") == 0) {
      quantities.has_charge_ = true;
      if (is_nep_charge){
        printf("    has charge predicted by NEP-charge.\n");
      } else {
        printf("    has charge specified in model.xyz.\n");
      }
    }
    if (strcmp(param[m], "bec") == 0) {
      quantities.has_bec_ = true;
      if (is_nep_charge){
        printf("    has BEC predicted by NEP-charge.\n");
      } else {
        PRINT_INPUT_ERROR("Cannot output BEC for a non-NEP-charge model.\n");
      }
    }
    if (strcmp(param[m], "virial") == 0) {
      quantities.has_virial_ = true;
      printf("    has virial.\n");
    }
    if (strcmp(param[m], "group") == 0) {
      quantities.has_group_ = true;
      printf("    has group.\n");
    }
  }
}

void Dump_XYZ::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (separated_ == 0) {
    fid_ = my_fopen(filename_.c_str(), "a");
  }

  gpu_total_virial_.resize(6);
  cpu_total_virial_.resize(6);
  if (quantities.has_force_) {
    cpu_force_per_atom_.resize(atom.number_of_atoms * 3);
  }
  if (quantities.has_potential_) {
    cpu_potential_per_atom_.resize(atom.number_of_atoms);
  }
  if (quantities.has_unwrapped_position_) {
    cpu_unwrapped_position_.resize(atom.number_of_atoms * 3);
  }
  if (quantities.has_virial_) {
    cpu_virial_per_atom_.resize(atom.number_of_atoms * 9);
  }
  if (quantities.has_bec_) {
    cpu_bec_.resize(atom.number_of_atoms * 9);
  }
}

void Dump_XYZ::output_line2(
  const double time,
  const Box& box,
  std::vector<Group>& groups,
  const std::vector<std::string>& cpu_atom_symbol,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& gpu_thermo)
{
  // time
  fprintf(fid_, "Time=%.8f", time * TIME_UNIT_CONVERSION); // output time is in units of fs

  // PBC
  fprintf(
    fid_, " pbc=\"%c %c %c\"", box.pbc_x ? 'T' : 'F', box.pbc_y ? 'T' : 'F', box.pbc_z ? 'T' : 'F');

  // box
  fprintf(
    fid_,
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

  // energy and virial (symmetric tensor) in eV, and stress (symmetric tensor) in eV/A^3
  double cpu_thermo[8];
  gpu_thermo.copy_to_host(cpu_thermo, 8);
  const int N = virial_per_atom.size() / 9;
  gpu_sum<<<6, 1024>>>(N, virial_per_atom.data(), gpu_total_virial_.data());
  gpu_total_virial_.copy_to_host(cpu_total_virial_.data());

  fprintf(fid_, " energy=%.8f", cpu_thermo[1]);
  fprintf(
    fid_,
    " virial=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"",
    cpu_total_virial_[0],
    cpu_total_virial_[3],
    cpu_total_virial_[4],
    cpu_total_virial_[3],
    cpu_total_virial_[1],
    cpu_total_virial_[5],
    cpu_total_virial_[4],
    cpu_total_virial_[5],
    cpu_total_virial_[2]);
  fprintf(
    fid_,
    " stress=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"",
    cpu_thermo[2],
    cpu_thermo[5],
    cpu_thermo[6],
    cpu_thermo[5],
    cpu_thermo[3],
    cpu_thermo[7],
    cpu_thermo[6],
    cpu_thermo[7],
    cpu_thermo[4]);

  // Properties
  fprintf(fid_, " Properties=species:S:1:pos:R:3");

  if (quantities.has_mass_) {
    fprintf(fid_, ":mass:R:1");
  }
  if (quantities.has_charge_) {
    fprintf(fid_, ":charge:R:1");
  }
  if (quantities.has_bec_) {
    fprintf(fid_, ":bec:R:9");
  }
  if (quantities.has_velocity_) {
    fprintf(fid_, ":vel:R:3");
  }
  if (quantities.has_force_) {
    fprintf(fid_, ":forces:R:3");
  }
  if (quantities.has_potential_) {
    fprintf(fid_, ":energy_atom:R:1");
  }
  if (quantities.has_unwrapped_position_) {
    fprintf(fid_, ":unwrapped_position:R:3");
  }
  if (quantities.has_virial_) {
    fprintf(fid_, ":virial:R:9");
  }
  if (quantities.has_group_) {
    const int num_grouping_methods = groups.size();
    fprintf(fid_, ":group:I:%d", num_grouping_methods);
  }

  // Over
  fprintf(fid_, "\n");
}

void Dump_XYZ::process(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& groups,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  if ((step + 1) % dump_interval_ != 0)
    return;

  int number_of_atoms_to_dump = atom.number_of_atoms;
  if (grouping_method_ >= 0) {
    number_of_atoms_to_dump = groups[grouping_method_].cpu_size[group_id_];
  }

  atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  if (quantities.has_mass_) {
    atom.mass.copy_to_host(atom.cpu_mass.data());
  }
  if (quantities.has_charge_) {
    if (is_nep_charge) {
      GPU_Vector<float>& nep_charge = force.potentials[0]->get_charge_reference();
      nep_charge.copy_to_host(atom.cpu_charge.data());
    } else {
      atom.charge.copy_to_host(atom.cpu_charge.data());
    }
  }
  if (quantities.has_bec_) {
    GPU_Vector<float>& gpu_bec = force.potentials[0]->get_bec_reference();
    gpu_bec.copy_to_host(cpu_bec_.data());
  }
  if (quantities.has_velocity_) {
    atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());
  }
  if (quantities.has_force_) {
    atom.force_per_atom.copy_to_host(cpu_force_per_atom_.data());
  }
  if (quantities.has_potential_) {
    atom.potential_per_atom.copy_to_host(cpu_potential_per_atom_.data());
  }
  if (quantities.has_unwrapped_position_) {
    atom.unwrapped_position.copy_to_host(cpu_unwrapped_position_.data());
  }
  if (quantities.has_virial_) {
    atom.virial_per_atom.copy_to_host(cpu_virial_per_atom_.data());
  }

  if (separated_) {
    std::string filename = filename_ + std::to_string(step + 1);
    fid_ = my_fopen(filename.data(), "w");
  }

  // line 1
  fprintf(fid_, "%d\n", number_of_atoms_to_dump);

  // line 2
  output_line2(global_time, box, groups, atom.cpu_atom_symbol, atom.virial_per_atom, thermo);

  // other lines
  for (int n = 0; n < number_of_atoms_to_dump; n++) {

    int m = n;
    if (grouping_method_ >= 0) {
      int group_size_sum = groups[grouping_method_].cpu_size_sum[group_id_];
      m = groups[grouping_method_].cpu_contents[group_size_sum + n];
    }

    fprintf(fid_, "%s", atom.cpu_atom_symbol[m].c_str());
    for (int d = 0; d < 3; ++d) {
      fprintf(fid_, " %.8f", atom.cpu_position_per_atom[m + atom.number_of_atoms * d]);
    }
    if (quantities.has_mass_) {
      fprintf(fid_, " %.8f", atom.cpu_mass[m]);
    }
    if (quantities.has_charge_) {
      fprintf(fid_, " %.8f", atom.cpu_charge[m]);
    }
    if (quantities.has_bec_) {
      for (int d = 0; d < 9; ++d) {
        fprintf(fid_, " %.8f", cpu_bec_[m + atom.number_of_atoms * d]);
      }
    }
    if (quantities.has_velocity_) {
      const double natural_to_A_per_fs = 1.0 / TIME_UNIT_CONVERSION;
      for (int d = 0; d < 3; ++d) {
        fprintf(
          fid_, " %.8f", atom.cpu_velocity_per_atom[m + atom.number_of_atoms * d] * natural_to_A_per_fs);
      }
    }
    if (quantities.has_force_) {
      for (int d = 0; d < 3; ++d) {
        fprintf(fid_, " %.8f", cpu_force_per_atom_[m + atom.number_of_atoms * d]);
      }
    }
    if (quantities.has_potential_) {
      fprintf(fid_, " %.8f", cpu_potential_per_atom_[m]);
    }
    if (quantities.has_unwrapped_position_) {
      for (int d = 0; d < 3; ++d) {
        fprintf(fid_, " %.8f", cpu_unwrapped_position_[m + atom.number_of_atoms * d]);
      }
    }
    if (quantities.has_virial_) {
      const int index[9] = {0, 3, 4, 6, 1, 5, 7, 8, 2};
      for (int d = 0; d < 9; ++d) {
        fprintf(fid_, " %.8f", cpu_virial_per_atom_[m + atom.number_of_atoms * index[d]]);
      }
    }
    if (quantities.has_group_) {
      for (int d = 0; d < groups.size(); ++d) {
        fprintf(fid_, " %d", groups[d].cpu_label[m]);
      }
    }
    fprintf(fid_, "\n");
  }
  if (separated_ == 0) {
    fflush(fid_);
  } else {
    fclose(fid_);
  }
}

void Dump_XYZ::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (separated_ == 0) {
    fclose(fid_);
  }
}
