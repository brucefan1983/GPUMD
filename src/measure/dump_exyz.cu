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
Dump some data to dump.xyz in the extended XYZ format
--------------------------------------------------------------------------------------------------*/

#include "dump_exyz.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"

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

void Dump_EXYZ::parse(const char** param, int num_param)
{
  dump_ = true;
  printf("Dump extended XYZ.\n");

  if (num_param < 2) {
    PRINT_INPUT_ERROR("dump_exyz should have at least 1 parameter.\n");
  }

  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("dump interval should > 0.");
  }

  printf("    every %d steps.\n", dump_interval_);

  has_velocity_ = 0;
  has_force_ = 0;
  has_potential_ = 0;

  if (num_param >= 3) {
    if (!is_valid_int(param[2], &has_velocity_)) {
      PRINT_INPUT_ERROR("has_velocity should be an integer.");
    }
    if (has_velocity_ == 0) {
      printf("    without velocity data.\n");
    } else {
      printf("    with velocity data.\n");
    }
  }

  if (num_param >= 4) {
    if (!is_valid_int(param[3], &has_force_)) {
      PRINT_INPUT_ERROR("has_force should be an integer.");
    }
    if (has_force_ == 0) {
      printf("    without force data.\n");
    } else {
      printf("    with force data.\n");
    }
  }

  if (num_param >= 5) {
    if (!is_valid_int(param[4], &has_potential_)) {
      PRINT_INPUT_ERROR("has_potential should be an integer.");
    }
    if (has_potential_ == 0) {
      printf("    without potential data.\n");
    } else {
      printf("    with potential data.\n");
    }
  }
}

void Dump_EXYZ::preprocess(const int number_of_atoms)
{
  if (dump_) {
    fid_ = my_fopen("dump.xyz", "a");
    gpu_total_virial_.resize(6);
    cpu_total_virial_.resize(6);
    if (has_force_) {
      cpu_force_per_atom_.resize(number_of_atoms * 3);
    }
    if (has_potential_) {
      cpu_potential_per_atom_.resize(number_of_atoms);
    }
  }
}

void Dump_EXYZ::output_line2(
  const double time,
  const Box& box,
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
  if (box.triclinic == 0) {
    fprintf(
      fid_,
      " Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"",
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
  }

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

  if (has_velocity_) {
    fprintf(fid_, ":vel:R:3");
  }
  if (has_force_) {
    fprintf(fid_, ":forces:R:3");
  }
  if (has_potential_) {
    fprintf(fid_, ":energy_atom:R:1");
  }

  // Over
  fprintf(fid_, "\n");
}

void Dump_EXYZ::process(
  const int step,
  const double global_time,
  const Box& box,
  Atom& atom,
  GPU_Vector<double>& gpu_thermo)
{
  if (!dump_)
    return;
  if ((step + 1) % dump_interval_ != 0)
    return;

  const int num_atoms_total = atom.position_per_atom.size() / 3;
  atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  if (has_velocity_) {
    atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());
  }
  if (has_force_) {
    atom.force_per_atom.copy_to_host(cpu_force_per_atom_.data());
  }
  if (has_potential_) {
    atom.potential_per_atom.copy_to_host(cpu_potential_per_atom_.data());
  }

  // line 1
  fprintf(fid_, "%d\n", num_atoms_total);

  // line 2
  output_line2(global_time, box, atom.cpu_atom_symbol, atom.virial_per_atom, gpu_thermo);

  // other lines
  for (int n = 0; n < num_atoms_total; n++) {
    fprintf(fid_, "%s", atom.cpu_atom_symbol[n].c_str());
    for (int d = 0; d < 3; ++d) {
      fprintf(fid_, " %.8f", atom.cpu_position_per_atom[n + num_atoms_total * d]);
    }
    if (has_velocity_) {
      const double natural_to_A_per_fs = 1.0 / TIME_UNIT_CONVERSION;
      for (int d = 0; d < 3; ++d) {
        fprintf(
          fid_, " %.8f", atom.cpu_velocity_per_atom[n + num_atoms_total * d] * natural_to_A_per_fs);
      }
    }
    if (has_force_) {
      for (int d = 0; d < 3; ++d) {
        fprintf(fid_, " %.8f", cpu_force_per_atom_[n + num_atoms_total * d]);
      }
    }
    if (has_potential_) {
      fprintf(fid_, " %.8f", cpu_potential_per_atom_[n]);
    }
    fprintf(fid_, "\n");
  }

  fflush(fid_);
}

void Dump_EXYZ::postprocess()
{
  if (dump_) {
    fclose(fid_);
    dump_ = false;
  }
}
