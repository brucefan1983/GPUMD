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
Dump energy/force/virial with all loaded potentials at a given interval.
--------------------------------------------------------------------------------------------------*/

#include "dump_observer.cuh"
#include "model/box.cuh"
#include "utilities/common.cuh"
#include "parse_utilities.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <vector>
#include <iostream>


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


void Dump_Observer::parse(const char** param, int num_param)
{
  dump_ = true;
  printf("Dump observer.\n");

  if (num_param != 3) {
    PRINT_INPUT_ERROR("dump_observer should have 2 parameters.");
  }
  mode_ = param[1];
  if (strcmp(mode_, "observe") != 0 && strcmp(mode_, "average") != 0) {
    PRINT_INPUT_ERROR("observer mode should be 'observe' or 'average'");
  }
  if (!is_valid_int(param[2], &dump_interval_)) {
    PRINT_INPUT_ERROR("observer dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("observer dump interval should > 0.");
  }
  
  if (strcmp(mode_, "observe") == 0) {
    printf("    evaluate all potentials every %d steps.\n", dump_interval_);
  }
  else if (strcmp(mode_, "average") == 0){
    printf("    use the average potential in the molecular dynamics run, and dump every %d steps.\n", dump_interval_);
  }
}

void Dump_Observer::preprocess(const int number_of_atoms, const int number_of_potentials, Force& force)
{
  // Setup a dump_exyz with the dump_interval for dump_observer.
  force.set_multiple_potentials_mode(mode_);
  if (dump_) {
    for (int i = 0; i < number_of_potentials; i++){
      const std::string file_number = (number_of_potentials == 1) ? "" : std::to_string(i); 
      std::string filename = "observer" + file_number + ".xyz";
      files_.push_back(my_fopen(filename.c_str(), "a"));
    }
    gpu_total_virial_.resize(6);
    cpu_total_virial_.resize(6);
    if (has_force_) {
      cpu_force_per_atom_.resize(number_of_atoms * 3);
    }
  }
}

void Dump_Observer::process(
    int step,
    const double global_time,
    Box& box,
    Atom& atom,
    Force& force,
    GPU_Vector<double>& thermo)
{
  // Only run if should dump, since forces have to be recomputed with each potential.
  if (!dump_)
    return;
  if ((step + 1) % dump_interval_ != 0)
    return;
  if(strcmp(mode_, "observe") == 0)
  {
    // If observing, calculate properties with all potentials.
    const int number_of_potentials = force.potentials.size();
    for (int potential_index = 0; potential_index < number_of_potentials; potential_index++) {
      force.potentials[potential_index]->compute(box, atom.type, atom.position_per_atom, 
          atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);
      write(step, global_time, box, atom.cpu_atom_symbol, atom.cpu_type, atom.position_per_atom,
        atom.cpu_position_per_atom, atom.velocity_per_atom, atom.cpu_velocity_per_atom,
        atom.force_per_atom, atom.virial_per_atom, thermo, potential_index);
    }
  }
  else if(strcmp(mode_, "average") == 0)
  {
    // If average, dump already computed properties to file.
    write(step, global_time, box, atom.cpu_atom_symbol, atom.cpu_type, atom.position_per_atom,
      atom.cpu_position_per_atom, atom.velocity_per_atom, atom.cpu_velocity_per_atom,
      atom.force_per_atom, atom.virial_per_atom, thermo, 0);
  }
  else {
    PRINT_INPUT_ERROR("Invalid observer mode.\n");
  }
}


void Dump_Observer::output_line2(
  const double time,
  const Box& box,
  const std::vector<std::string>& cpu_atom_symbol,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& gpu_thermo,
  FILE* fid_)
{
  // time
  fprintf(fid_, "Time=%.8f", time * TIME_UNIT_CONVERSION); // output time is in units of fs

  // PBC
  fprintf(
    fid_, " pbc=\"%c %c %c\"", box.pbc_x ? 'T' : 'F', box.pbc_y ? 'T' : 'F', box.pbc_z ? 'T' : 'F');

  // box
  if (box.triclinic == 0) {
    fprintf(
      fid_, " Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"", box.cpu_h[0], 0.0, 0.0,
      0.0, box.cpu_h[1], 0.0, 0.0, 0.0, box.cpu_h[2]);
  } else {
    fprintf(
      fid_, " Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"", box.cpu_h[0], box.cpu_h[3],
      box.cpu_h[6], box.cpu_h[1], box.cpu_h[4], box.cpu_h[7], box.cpu_h[2], box.cpu_h[5],
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
    fid_, " virial=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"", cpu_total_virial_[0],
    cpu_total_virial_[3], cpu_total_virial_[4], cpu_total_virial_[3], cpu_total_virial_[1],
    cpu_total_virial_[5], cpu_total_virial_[4], cpu_total_virial_[5], cpu_total_virial_[2]);
  fprintf(
    fid_, " stress=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\"", cpu_thermo[2], cpu_thermo[5],
    cpu_thermo[6], cpu_thermo[5], cpu_thermo[3], cpu_thermo[7], cpu_thermo[6], cpu_thermo[7],
    cpu_thermo[4]);

  // Properties
  fprintf(fid_, " Properties=species:S:1:pos:R:3");

  if (has_velocity_) {
    fprintf(fid_, ":vel:R:3");
  }
  if (has_force_) {
    fprintf(fid_, ":forces:R:3");
  }

  // Over
  fprintf(fid_, "\n");
}

void Dump_Observer::write(
  const int step,
  const double global_time,
  const Box& box,
  const std::vector<std::string>& cpu_atom_symbol,
  const std::vector<int>& cpu_type,
  GPU_Vector<double>& position_per_atom,
  std::vector<double>& cpu_position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  std::vector<double>& cpu_velocity_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& gpu_thermo,
  const int file_index)
{
  if (!dump_)
    return;
  if ((step + 1) % dump_interval_ != 0)
    return;
 
  const int num_atoms_total = position_per_atom.size() / 3;
  FILE* fid_ = files_[file_index];
  position_per_atom.copy_to_host(cpu_position_per_atom.data());
  if (has_velocity_) {
    velocity_per_atom.copy_to_host(cpu_velocity_per_atom.data());
  }
  if (has_force_) {
    force_per_atom.copy_to_host(cpu_force_per_atom_.data());
  }

  // line 1
  fprintf(fid_, "%d\n", num_atoms_total);

  // line 2
  output_line2(global_time, box, cpu_atom_symbol, virial_per_atom, gpu_thermo, fid_);

  // other lines
  for (int n = 0; n < num_atoms_total; n++) {
    fprintf(fid_, "%s", cpu_atom_symbol[n].c_str());
    for (int d = 0; d < 3; ++d) {
      fprintf(fid_, " %.8f", cpu_position_per_atom[n + num_atoms_total * d]);
    }
    if (has_velocity_) {
      const double natural_to_A_per_fs = 1.0 / TIME_UNIT_CONVERSION;
      for (int d = 0; d < 3; ++d) {
        fprintf(
          fid_, " %.8f", cpu_velocity_per_atom[n + num_atoms_total * d] * natural_to_A_per_fs);
      }
    }
    if (has_force_) {
      for (int d = 0; d < 3; ++d) {
        fprintf(fid_, " %.8f", cpu_force_per_atom_[n + num_atoms_total * d]);
      }
    }
    fprintf(fid_, "\n");
  }

  fflush(fid_);
}


void Dump_Observer::postprocess()
{
  for (int i = 0; i < files_.size(); i++){
    fclose(files_[i]);
  }
  dump_ = false;
}


