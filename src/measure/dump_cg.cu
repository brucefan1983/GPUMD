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

Dump_CG::Dump_CG(const char** param, int num_param) 
{
  parse(param, num_param);
  property_name = "dump_cg";
}

void Dump_CG::parse(const char** param, int num_param)
{
  printf("Dump train.xyz for NEP-CG.\n");

  if (num_param < 2) {
    PRINT_INPUT_ERROR("dump_cg should have at least 1 parameter.\n");
  }

  if (!is_valid_int(param[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("dump interval should > 0.");
  }

  printf("    every %d steps.\n", dump_interval_);
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
  fid2_ = my_fopen("train2.xyz", "a");

  gpu_total_virial_.resize(6);
  cpu_total_virial_.resize(6);
  cpu_force_per_atom_.resize(atom.number_of_atoms * 3);
}

void Dump_CG::output_line2(
  FILE* fid,
  const Box& box,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& gpu_thermo)
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

  // energy and virial (symmetric tensor) in eV, and stress (symmetric tensor) in eV/A^3
  double cpu_thermo[8];
  gpu_thermo.copy_to_host(cpu_thermo, 8);
  const int N = virial_per_atom.size() / 9;
  gpu_sum<<<6, 1024>>>(N, virial_per_atom.data(), gpu_total_virial_.data());
  gpu_total_virial_.copy_to_host(cpu_total_virial_.data());

  fprintf(fid, " energy=%.8f", cpu_thermo[1]);
  fprintf(
    fid,
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
  process2(
    number_of_steps,
    step,
    fixed_group,
    move_group,
    global_time,
    temperature,
    integrate,
    box,
    group,
    thermo,
    atom,
    force);

  if ((step + 1) % dump_interval_ != 0)
    return;

  const int num_atoms_total = atom.position_per_atom.size() / 3;
  const int num_beads = num_atoms_total / 3;
  atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  atom.force_per_atom.copy_to_host(cpu_force_per_atom_.data());

  // line 1
  fprintf(fid_, "%d\n", num_beads); // water

  // line 2
  output_line2(fid_, box, atom.virial_per_atom, thermo);

  // other lines
  for (int b = 0; b < num_beads; b++) {
    int n1 = b * 3; // O
    int n2 = n1 + 1; // H
    int n3 = n1 + 2; // H
    fprintf(fid_, "O "); // call it O
    for (int d = 0; d < 3; ++d) {
      double r1 = atom.cpu_position_per_atom[n1 + num_atoms_total * d];
      double r2 = atom.cpu_position_per_atom[n2 + num_atoms_total * d];
      double r3 = atom.cpu_position_per_atom[n3 + num_atoms_total * d];
      if (r2 - r1 > box.cpu_h[0]/2) {
        r2 -= box.cpu_h[0];
      } else if (r2 - r1 < -box.cpu_h[0]/2) {
        r2 += box.cpu_h[0];
      }
      if (r3 - r1 > box.cpu_h[0]/2) {
        r3 -= box.cpu_h[0];
      } else if (r3 - r1 < -box.cpu_h[0]/2) {
        r3 += box.cpu_h[0];
      }
      double r_com = (r1 * 16.0 + r2 * 1.0 + r3 * 1.0) / 18.0;
      if (r_com < 0) {
        r_com += box.cpu_h[0];
      } else if (r_com > box.cpu_h[0]) {
        r_com -= box.cpu_h[0];
      }
      fprintf(fid_, " %.8f", r_com);
    }
    for (int d = 0; d < 3; ++d) {
      double f1 = cpu_force_per_atom_[n1 + num_atoms_total * d];
      double f2 = cpu_force_per_atom_[n2 + num_atoms_total * d];
      double f3 = cpu_force_per_atom_[n3 + num_atoms_total * d];
      double f_tot = f1 + f2 + f3;
      fprintf(fid_, " %.8f", f_tot);
    }
    fprintf(fid_, "\n");
  }
  fflush(fid_);
}

void Dump_CG::process2(
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

  const int num_atoms_total = atom.position_per_atom.size() / 3;
  const int num_beads = num_atoms_total / 3;
  atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  atom.force_per_atom.copy_to_host(cpu_force_per_atom_.data());

  // line 1
  fprintf(fid2_, "%d\n", num_beads); // water

  // line 2
  output_line2(fid2_, box, atom.virial_per_atom, thermo);

  // other lines
  for (int b = 0; b < num_beads; b++) {
    int n1 = b * 3; // O
    int n2 = n1 + 1; // H
    int n3 = n1 + 2; // H
    fprintf(fid2_, "O "); // call it O
    for (int d = 0; d < 3; ++d) {
      double r1 = atom.cpu_position_per_atom[n1 + num_atoms_total * d];
      double r2 = atom.cpu_position_per_atom[n2 + num_atoms_total * d];
      double r3 = atom.cpu_position_per_atom[n3 + num_atoms_total * d];
      if (r2 - r1 > box.cpu_h[0]/2) {
        r2 -= box.cpu_h[0];
      } else if (r2 - r1 < -box.cpu_h[0]/2) {
        r2 += box.cpu_h[0];
      }
      if (r3 - r1 > box.cpu_h[0]/2) {
        r3 -= box.cpu_h[0];
      } else if (r3 - r1 < -box.cpu_h[0]/2) {
        r3 += box.cpu_h[0];
      }
      double r_com = (r1 * 16.0 + r2 * 1.0 + r3 * 1.0) / 18.0;
      if (r_com < 0) {
        r_com += box.cpu_h[0];
      } else if (r_com > box.cpu_h[0]) {
        r_com -= box.cpu_h[0];
      }
      fprintf(fid2_, " %.8f", r_com);
    }
    for (int d = 0; d < 3; ++d) {
      double f1 = cpu_force_per_atom_[n1 + num_atoms_total * d];
      double f2 = cpu_force_per_atom_[n2 + num_atoms_total * d];
      double f3 = cpu_force_per_atom_[n3 + num_atoms_total * d];
      double f_tot = f1 + f2 + f3;
      fprintf(fid2_, " %.8f", f_tot);
    }
    fprintf(fid2_, "\n");
  }
  fflush(fid2_);
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
  fclose(fid2_);
}
