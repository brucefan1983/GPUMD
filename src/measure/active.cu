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
    GNU General Public License for more details. You should have received a copy of the GNU General Public License along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

/*-----------------------------------------------------------------------------------------------100
Dump energy/force/virial with all loaded potentials at a given interval.
--------------------------------------------------------------------------------------------------*/

#include "active.cuh"
#include "model/box.cuh"
#include "utilities/error.cuh"
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


static __global__ void initialize_properties(
  int N, double* g_fx, double* g_fy, double* g_fz, double* g_pe, double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    g_fx[n1] = 0.0;
    g_fy[n1] = 0.0;
    g_fz[n1] = 0.0;
    g_pe[n1] = 0.0;
    g_virial[n1 + 0 * N] = 0.0;
    g_virial[n1 + 1 * N] = 0.0;
    g_virial[n1 + 2 * N] = 0.0;
    g_virial[n1 + 3 * N] = 0.0;
    g_virial[n1 + 4 * N] = 0.0;
    g_virial[n1 + 5 * N] = 0.0;
    g_virial[n1 + 6 * N] = 0.0;
    g_virial[n1 + 7 * N] = 0.0;
    g_virial[n1 + 8 * N] = 0.0;
  }
}

static __global__ void initialize_mean_vectors(
  int N, double* g_m, double* g_m_sq)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  // 3*N since 3 cartesian directions
  if (n1 < N*3) {
    g_m[n1] = 0.0;
    g_m_sq[n1] = 0.0;
  }
}


static __global__ void compute_mean(
  int N, int M, double* g_m, double* g_m_sq, double* g_fx, double* g_fy, double* g_fz)
{
  int n1 = blockidx.x * blockdim.x + threadidx.x;
  if (n1 < N) {
    // Average over number of potentials, M
    g_m[n1 + 0 * N] += g_fx[n1]/M;
    g_m[n1 + 1 * N] += g_fy[n1]/M;
    g_m[n1 + 2 * N] += g_fz[n1]/M;
    g_m_sq[n1 + 0 * N] += g_fx[n1]*g_fx[n1]/M;
    g_m_sq[n1 + 1 * N] += g_fy[n1]*g_fy[n1]/M;
    g_m_sq[n1 + 2 * N] += g_fz[n1]*g_fz[n1]/M;
  }
}


static __global__ void compute_uncertainty(
  int N, double* g_m, double* g_m_sq, double* g_u)
{
  int n1 = blockidx.x * blockdim.x + threadidx.x;
  if (n1 < 3*N) {
    g_u[0] += (g_m_sq[n1] - g_m[n1]*g_m[n1]) / (3*N);
  }
}


void Active::parse(const char** param, int num_param)
{
  check_ = true;
  printf("Active learning.\n");

  if (num_param != 5) {
    PRINT_INPUT_ERROR("active should have 4 parameters.");
  }
  if (!is_valid_int(param[1], &check_interval_)) {
    PRINT_INPUT_ERROR("check interval should be an integer.");
  }
  if (check_interval_ <= 0) {
    PRINT_INPUT_ERROR("dump interval should > 0.");
  }
  printf("    check uncertainty every %d steps.\n", check_interval_);

  if (!is_valid_int(param[2], &has_velocity_)) {
    PRINT_INPUT_ERROR("has_velocity should be an integer.");
  }
  if (has_velocity_ == 0) {
    printf("    without velocity data.\n");
  } else {
    printf("    with velocity data.\n");
  }

  if (!is_valid_int(param[3], &has_force_)) {
    print_input_error("has_force should be an integer.");
  }
  if (has_force_ == 0) {
    printf("    without force data.\n");
  } else {
    printf("    with force data.\n");
  }
  if (!is_valid_real(param[4], &threshold_)) {
    PRINT_INPUT_ERROR("threshold should be a real number.\n");
  }
  
  printf("    will check if uncertainties exceed %f every %d iterations.\n", threshold_, check_interval_);
}

void Active::preprocess(const int number_of_atoms, const int number_of_potentials, Force& force)
{
  // Always use mode "observe" with all other potentials for active learning.
  // Only propagate MD with the main potential.
  force.set_multiple_potentials_mode("observe");
  if (check_) {
    std::string exyz_filename = "active.xyz";
    std::string out_filename = "active.out";
    exyz_file_ = my_fopen(exyz_filename.c_str(), "a");
    out_file_ = my_fopen(out_filename.c_str(), "a");
    gpu_total_virial_.resize(6);
    cpu_total_virial_.resize(6);
    if (has_force_) {
      cpu_force_per_atom_.resize(number_of_atoms * 3);
    }
    mean_force_.resize(number_of_atoms * 3);
    mean_force_sq_.resize(number_of_atoms * 3);
    g_uncertainty_.resize(1);
  }
}

void Active::process(
    int step,
    const double global_time,
    const int number_of_atoms_fixed,
    std::vector<Group>& group,
    Box& box,
    Atom& atom,
    Force& force,
    GPU_Vector<double>& thermo)
{
  // Only run if should check, since forces have to be recomputed with each potential.
  if (!check_)
    return;
  if ((step + 1) % check_interval_ != 0)
    return;

  const int number_of_potentials = force.potentials.size();
  const int number_of_atoms = atom.type.size();
  // Reset mean vectors to zero
  initialize_mean_vectors<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms, mean_force_, mean_force_sq);
  CUDA_CHECK_KERNEL

  // Loop backwards over files to evaluate the main potential last, keeping it's properties intact
  for (int potential_index = number_of_potentials-1; potential_index >= 0; potential_index--) {
    // Set potential/force/virials to zero
    initialize_properties<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, atom.force_per_atom.data(), atom.force_per_atom.data() + number_of_atoms,
      atom.force_per_atom.data() + number_of_atoms * 2, atom.potential_per_atom.data(), atom.virial_per_atom.data());
    CUDA_CHECK_KERNEL
    // Compute new potential properties
    force.potentials[potential_index]->compute(box, atom.type, atom.position_per_atom, 
        atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);
    // Write properties to GPU vector 
    compute_mean<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, number_of_porentials, mean_force_, mean_force_sq, atom.force_per_atom.data(), atom.force_per_atom.data() + number_of_atoms,
      atom.force_per_atom.data() + number_of_atoms * 2);
  }
  // Sum mean and mean_sq on GPU, move sum to CPU
  compute_uncertainty<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms, mean_force_, mean_force_sq, g_uncertainty);
  double unc[1];
  g_uncertainty.copy_to_host(unc, 1);
  uncertainty = unc[0];
  if (uncertainty > threshold_){
    write_exyz(step, global_time, box, atom.cpu_atom_symbol, atom.cpu_type, atom.position_per_atom,
      atom.cpu_position_per_atom, atom.velocity_per_atom, atom.cpu_velocity_per_atom,
      atom.force_per_atom, atom.virial_per_atom, thermo, uncertainty);
  write_out()
}


void Active::write_out(
  const int step,
  GPU_Vector<double>& gpu_thermo,
  double uncertainty)
{
  if (!check_)
    return;
  if ((step + 1) % check_interval_ != 0)
    return;

  FILE* fid_ = out_file_;
  double thermo[8];
  gpu_thermo.copy_to_host(thermo, 8);

  // Write time, uncertainty to file
  fprintf(fid_, "%20.10e%20.10e\n", thermo[0], uncertainty);
  fflush(fid_);
}


void Active::output_line2(
  const double time,
  const Box& box,
  const std::vector<std::string>& cpu_atom_symbol,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& gpu_thermo,
  double uncertainty,
  FILE* fid_)
{
  // time
  fprintf(fid_, "Time=%.8f", time * TIME_UNIT_CONVERSION); // output time is in units of fs

  // PBC
  fprintf(
    fid_, " pbc=\"%c %c %c\"", box.pbc_x ? 'T' : 'F', box.pbc_y ? 'T' : 'F', box.pbc_z ? 'T' : 'F');

  // Uncertainty
  fprintf(
    fid_, " uncertainty=%.8f", uncertainty);

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

void Active::write_exyz(
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
  double uncertainty)
{
  if (!check_)
    return;
  if ((step + 1) % check_interval_ != 0)
    return;
 
  const int num_atoms_total = position_per_atom.size() / 3;
  FILE* fid_ = exyz_file_;
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
  output_line2(global_time, box, cpu_atom_symbol, virial_per_atom, gpu_thermo, uncertainty, fid_);

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

void Active::postprocess()
{
  fclose(exyz_file_);
  fclose(out_file_);
  dump_ = false;
}


