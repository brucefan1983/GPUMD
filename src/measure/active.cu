/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details. You should have received a copy of the GNU Lesser General
   Public License along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*-----------------------------------------------------------------------------------------------100
Run active learning on-the-fly during MD
--------------------------------------------------------------------------------------------------*/

#include "active.cuh"
#include "force/force.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/box.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

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

// Welford's update with the forces of the k-th potential.
// Each term added to the sum of squared deviations is the product of two numbers that cannot
// have opposite signs.
// The sum therefore stays non-negative in floating point.
static __global__ void accumulate_force_statistics(
  const int size,
  const int k,
  const double* g_force,
  double* g_mean,
  double* g_squared_deviation_sum)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < size) {
    const double deviation = g_force[n] - g_mean[n];
    g_mean[n] += deviation / k;
    g_squared_deviation_sum[n] += deviation * (g_force[n] - g_mean[n]);
  }
}

// The uncertainty of an atom is the norm of the standard deviations of its three force components
// over the M potentials, with the factor 1/M.
static __global__ void
compute_uncertainty(const int N, const int M, const double* g_squared_deviation_sum, double* g_u)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    g_u[n] = sqrt(
      (g_squared_deviation_sum[n] + g_squared_deviation_sum[n + N] +
       g_squared_deviation_sum[n + 2 * N]) /
      M);
  }
}

Active::Active(const std::vector<std::string>& tokens)
{
  parse(tokens);
  action_name = "active";
}

void Active::parse(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  printf("Active learning.\n");

  if (num_param != 6) {
    PRINT_INPUT_ERROR("active should have 5 parameters.");
  }
  if (!is_valid_int(tokens[1], &check_interval_)) {
    PRINT_INPUT_ERROR("check interval should be an integer.");
  }
  if (check_interval_ <= 0) {
    PRINT_INPUT_ERROR("check interval should > 0.");
  }
  printf("    check uncertainty every %d steps.\n", check_interval_);

  if (!is_valid_int(tokens[2], &has_velocity_)) {
    PRINT_INPUT_ERROR("has_velocity should be an integer.");
  }
  if (has_velocity_ == 0) {
    printf("    without velocity data.\n");
  } else {
    printf("    with velocity data.\n");
  }

  if (!is_valid_int(tokens[3], &has_force_)) {
    PRINT_INPUT_ERROR("has_force should be an integer.");
  }
  if (has_force_ == 0) {
    printf("    without force data.\n");
  } else {
    printf("    with force data.\n");
  }

  if (!is_valid_int(tokens[4], &has_uncertainty_)) {
    PRINT_INPUT_ERROR("has_uncertainty should be an integer.");
  }
  if (has_uncertainty_ == 0) {
    printf("    without per-atom uncertainty data.\n");
  } else {
    printf("    with per-atom uncertainty data.\n");
  }

  if (!is_valid_real(tokens[5], &threshold_)) {
    PRINT_INPUT_ERROR("threshold should be a real number.\n");
  }

  printf(
    "    will check if uncertainties exceed %f every %d iterations.\n",
    threshold_,
    check_interval_);
}

void Active::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  // Force accepts several potentials only when every one of them is a NEP potential.
  if (force.get_number_of_potentials() < 2) {
    PRINT_INPUT_ERROR("active requires at least two potentials.\n");
  }
  std::string exyz_filename = "active.xyz";
  std::string out_filename = "active.out";
  exyz_file_ = my_fopen(exyz_filename.c_str(), "a");
  out_file_ = my_fopen(out_filename.c_str(), "a");
  gpu_total_virial_.resize(6);
  cpu_total_virial_.resize(6);
  if (has_force_) {
    cpu_force_per_atom_.resize(atom.number_of_atoms * 3);
  }
  // The velocity keyword of a later run reads the host copies in Atom, which active leaves alone.
  cpu_position_per_atom_.resize(atom.number_of_atoms * 3);
  if (has_velocity_) {
    cpu_velocity_per_atom_.resize(atom.number_of_atoms * 3);
  }
  mean_force_.resize(atom.number_of_atoms * 3);
  squared_force_deviation_sum_.resize(atom.number_of_atoms * 3);
  gpu_uncertainty_.resize(atom.number_of_atoms);
  cpu_uncertainty_.resize(atom.number_of_atoms);
  active_potential_per_atom_.resize(atom.number_of_atoms);
  active_force_per_atom_.resize(atom.number_of_atoms * 3);
  active_virial_per_atom_.resize(atom.number_of_atoms * 9);
  // Ensemble::find_thermo writes T, U and the six components of the stress.
  active_thermo_.resize(8);
}

void Active::end_of_step(
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
  if ((step + 1) % check_interval_ != 0)
    return;

  const int number_of_potentials = force.get_number_of_potentials();
  const int number_of_atoms = atom.type.size();
  mean_force_.fill(0.0);
  squared_force_deviation_sum_.fill(0.0);

  // Every potential is evaluated into scratch arrays, which leaves the per-atom arrays and the
  // thermo vector of the run unchanged.
  // Potential 0 is evaluated last and leaves its properties in the scratch arrays for active.xyz.
  for (int potential_index = number_of_potentials - 1; potential_index >= 0; potential_index--) {
    force.compute_one_potential(
      potential_index,
      box,
      atom.position_per_atom,
      atom.type,
      group,
      active_potential_per_atom_,
      active_force_per_atom_,
      active_virial_per_atom_);
    accumulate_force_statistics<<<(3 * number_of_atoms - 1) / 128 + 1, 128>>>(
      3 * number_of_atoms,
      number_of_potentials - potential_index,
      active_force_per_atom_.data(),
      mean_force_.data(),
      squared_force_deviation_sum_.data());
    GPU_CHECK_KERNEL
  }
  compute_uncertainty<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    number_of_potentials,
    squared_force_deviation_sum_.data(),
    gpu_uncertainty_.data());
  GPU_CHECK_KERNEL
  gpu_uncertainty_.copy_to_host(cpu_uncertainty_.data());
  // A NaN, from a model with forces that are not finite, ranks above every number.
  const double uncertainty = *std::max_element(
    cpu_uncertainty_.begin(), cpu_uncertainty_.end(), [](const double a, const double b) {
      return std::isnan(b) ? !std::isnan(a) : a < b;
    });
  write_uncertainty(global_time, uncertainty);
  if (std::isnan(uncertainty) || uncertainty > threshold_) {
    integrate.find_thermo(
      box.get_volume(),
      group,
      atom.mass,
      active_potential_per_atom_,
      atom.velocity_per_atom,
      active_virial_per_atom_,
      active_thermo_);
    write_exyz(global_time, box, atom, uncertainty);
  }
}

void Active::write_uncertainty(const double time, double uncertainty)
{
  FILE* fid_ = out_file_;

  // Write time, uncertainty to file
  fprintf(fid_, "%20.10e%20.10e\n", time * TIME_UNIT_CONVERSION, uncertainty);
  fflush(fid_);
}

void Active::output_line2(const double time, const Box& box, double uncertainty)
{
  FILE* fid_ = exyz_file_;
  // time
  fprintf(fid_, "Time=%.8f", time * TIME_UNIT_CONVERSION); // output time is in units of fs

  // PBC
  fprintf(
    fid_, " pbc=\"%c %c %c\"", box.pbc_x ? 'T' : 'F', box.pbc_y ? 'T' : 'F', box.pbc_z ? 'T' : 'F');

  // Uncertainty
  fprintf(fid_, " uncertainty=%.8f", uncertainty);

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
  active_thermo_.copy_to_host(cpu_thermo, 8);
  const int N = active_virial_per_atom_.size() / 9;
  gpu_sum<<<6, 1024>>>(N, active_virial_per_atom_.data(), gpu_total_virial_.data());
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
  if (has_uncertainty_) {
    fprintf(fid_, ":uncertainty:R:1");
  }

  // Over
  fprintf(fid_, "\n");
}

void Active::write_exyz(const double global_time, const Box& box, Atom& atom, double uncertainty)
{
  const int num_atoms_total = atom.number_of_atoms;
  FILE* fid_ = exyz_file_;
  atom.position_per_atom.copy_to_host(cpu_position_per_atom_.data());
  if (has_velocity_) {
    atom.velocity_per_atom.copy_to_host(cpu_velocity_per_atom_.data());
  }
  if (has_force_) {
    active_force_per_atom_.copy_to_host(cpu_force_per_atom_.data());
  }

  // line 1
  fprintf(fid_, "%d\n", num_atoms_total);

  // line 2
  output_line2(global_time, box, uncertainty);

  // other lines
  for (int n = 0; n < num_atoms_total; n++) {
    fprintf(fid_, "%s", atom.cpu_atom_symbol[n].c_str());
    for (int d = 0; d < 3; ++d) {
      fprintf(fid_, " %.8f", cpu_position_per_atom_[n + num_atoms_total * d]);
    }
    if (has_velocity_) {
      const double natural_to_A_per_fs = 1.0 / TIME_UNIT_CONVERSION;
      for (int d = 0; d < 3; ++d) {
        fprintf(
          fid_, " %.8f", cpu_velocity_per_atom_[n + num_atoms_total * d] * natural_to_A_per_fs);
      }
    }
    if (has_force_) {
      for (int d = 0; d < 3; ++d) {
        fprintf(fid_, " %.8f", cpu_force_per_atom_[n + num_atoms_total * d]);
      }
    }
    if (has_uncertainty_) {
      fprintf(fid_, " %.8f", cpu_uncertainty_[n]);
    }
    fprintf(fid_, "\n");
  }

  fflush(fid_);
}

void Active::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  fclose(exyz_file_);
  fclose(out_file_);
}
