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

/*----------------------------------------------------------------------------80
Apply electron stopping.
------------------------------------------------------------------------------*/

#include "electron_stop.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/gpu_macro.cuh"
#include <iostream>
#include <vector>

static void __global__ find_stopping_force(
  const int num_atoms,
  const int num_points,
  const double time_step,
  const double energy_min,
  const double energy_max,
  const double energy_interval_inverse,
  const double* g_stopping_power,
  const int* g_type,
  const double* g_mass,
  const double* g_velocity,
  double* g_force,
  double* g_power_loss)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_atoms) {
    int type = g_type[i];
    double mass = g_mass[i];
    double vx = g_velocity[0 * num_atoms + i];
    double vy = g_velocity[1 * num_atoms + i];
    double vz = g_velocity[2 * num_atoms + i];
    double v2 = vx * vx + vy * vy + vz * vz;
    double energy = 0.5 * mass * v2;

    if (energy < energy_min + 1.0e-6 || energy > energy_max - 1.0e-6) {
      g_force[0 * num_atoms + i] = 0.0;
      g_force[1 * num_atoms + i] = 0.0;
      g_force[2 * num_atoms + i] = 0.0;
      return;
    }

    double fractional_energy = (energy - energy_min) * energy_interval_inverse;
    int index_left = static_cast<int>(fractional_energy);
    int index_right = index_left + 1;
    double weight_right = fractional_energy - index_left;
    double weight_left = 1.0 - weight_right;
    double stopping_power = g_stopping_power[type * num_points + index_left] * weight_left +
                            g_stopping_power[type * num_points + index_right] * weight_right;

    double factor = -stopping_power / sqrt(v2);

    g_force[0 * num_atoms + i] = vx * factor;
    g_force[1 * num_atoms + i] = vy * factor;
    g_force[2 * num_atoms + i] = vz * factor;

    g_power_loss[i] = stopping_power * sqrt(v2) * time_step;
  }
}

__device__ float device_force_average[3];

static __global__ void find_force_average(int num_atoms, double* g_force)
{
  //<<<3, 1024>>>
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_batches = (num_atoms - 1) / 1024 + 1;
  __shared__ double s_f[1024];
  double f = 0.0;

  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < num_atoms) {
      f += g_force[n + bid * num_atoms];
    }
  }

  s_f[tid] = f;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_f[tid] += s_f[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    device_force_average[bid] = s_f[0] / num_atoms;
  }
}

static void __global__
apply_electron_stopping(const int num_atoms, const double* g_stopping_force, double* g_force)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_atoms) {
    for (int d = 0; d < 3; ++d) {
      g_force[d * num_atoms + i] += g_stopping_force[d * num_atoms + i] - device_force_average[d];
    }
  }
}

__device__ double device_power_loss;

static __global__ void find_power_loss(int num_atoms, double* g_power_loss)
{
  //<<<1, 1024>>>
  int tid = threadIdx.x;
  int block_size = blockDim.x;

  int number_of_batches = (num_atoms + block_size - 1) / block_size;
  __shared__ double s_f[1024];
  double f = 0.0;

  for (int batch = 0; batch < number_of_batches; ++batch) {
    int idx = tid + batch * block_size;
    if (idx < num_atoms) {
      f += g_power_loss[idx];
    }
  }

  s_f[tid] = f;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_f[tid] += s_f[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    device_power_loss = s_f[0];
  }
}

void Electron_Stop::compute(double time_step, Atom& atom)
{
  if (!do_electron_stop) {
    return;
  }

  find_stopping_force<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
    atom.number_of_atoms,
    num_points,
    time_step,
    energy_min,
    energy_max,
    1.0 / energy_interval,
    stopping_power_gpu.data(),
    atom.type.data(),
    atom.mass.data(),
    atom.velocity_per_atom.data(),
    stopping_force.data(),
    stopping_loss.data());

  GPU_CHECK_KERNEL

  find_force_average<<<3, 1024>>>(atom.number_of_atoms, stopping_force.data());
  GPU_CHECK_KERNEL

  apply_electron_stopping<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(
    atom.number_of_atoms, stopping_force.data(), atom.force_per_atom.data());
  GPU_CHECK_KERNEL

  find_power_loss<<<1, 1024>>>(atom.number_of_atoms, stopping_loss.data());
  GPU_CHECK_KERNEL

  double power_loss_host;
  CHECK(gpuMemcpyFromSymbol(
    &power_loss_host, device_power_loss, sizeof(double), 0, gpuMemcpyDeviceToHost));
  stopping_power_loss += power_loss_host;
}

void Electron_Stop::parse(
  const char** param, int num_param, const int num_atoms, const int num_types)
{
  printf("Apply electron stopping.\n");
  if (num_param != 2) {
    PRINT_INPUT_ERROR("electron_stop should have 1 parameter.\n");
  }
  printf("    using the stopping power data in %s.\n", param[1]);

  std::ifstream input(param[1]);
  if (!input.is_open()) {
    printf("Failed to open %s.\n", param[1]);
    exit(1);
  }

  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() != 3) {
    PRINT_INPUT_ERROR("The first line of the stopping power file should have 3 values.");
  }
  num_points = get_int_from_token(tokens[0], __FILE__, __LINE__);
  if (num_points < 2) {
    PRINT_INPUT_ERROR("Number of energy values should >= 2.\n");
  } else {
    printf("    number of energy values = %d.\n", num_points);
  }

  energy_min = get_double_from_token(tokens[1], __FILE__, __LINE__);
  if (energy_min <= 0) {
    PRINT_INPUT_ERROR("energy_min should > 0.\n");
  } else {
    printf("    energy_min = %g eV.\n", energy_min);
  }

  energy_max = get_double_from_token(tokens[2], __FILE__, __LINE__);
  if (energy_max <= energy_min) {
    PRINT_INPUT_ERROR("energy_max should > energy_min.\n");
  } else {
    printf("    energy_max = %g eV.\n", energy_max);
  }

  energy_interval = (energy_max - energy_min) / (num_points - 1);
  printf("    energy interval = %g eV.\n", energy_interval);

  stopping_power_cpu.resize(num_points * num_types);
  for (int n = 0; n < num_points; ++n) {
    std::vector<std::string> tokens = get_tokens(input);
    if (tokens.size() != num_types) {
      PRINT_INPUT_ERROR("Number of values does not match with the number of elements.");
    }
    for (int t = 0; t < num_types; ++t) {
      stopping_power_cpu[t * num_points + n] = get_double_from_token(tokens[t], __FILE__, __LINE__);
    }
  }

  stopping_power_gpu.resize(num_points * num_types);
  stopping_power_gpu.copy_from_host(stopping_power_cpu.data());
  stopping_force.resize(num_atoms * 3);
  stopping_loss.resize(num_atoms);
  do_electron_stop = true;
}

void Electron_Stop::finalize()
{
  if (do_electron_stop) {
    printf("Total electron stopping power loss = %g eV.\n", stopping_power_loss);
  }
  do_electron_stop = false;
  stopping_power_loss = 0.0;
}
