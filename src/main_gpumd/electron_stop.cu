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

/*----------------------------------------------------------------------------80
Apply electron stopping.
------------------------------------------------------------------------------*/

#include "electron_stop.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <iostream>
#include <vector>

void __global__ find_stopping_force(
  const int num_atoms,
  const double energy_min,
  const double energy_max,
  const double energy_interval_inverse,
  const double* g_stopping_power,
  const int* g_type,
  const double* g_mass,
  const double* g_velocity,
  double* g_force)
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

    if (energy < energy_min + 1.0e-6) {
      return;
    }
    if (energy > energy_max - 1.0e-6) {
      return;
    }

    double fractional_energy = (energy - energy_min) * energy_interval_inverse;
    int index_left = static_cast<int>(fractional_energy);
    int index_right = index_left + 1;
    double weight_right = fractional_energy - index_left;
    double weight_left = 1.0 - weight_right;
    double stopping_power = g_stopping_power[type * num_points + index_left] * weight_left +
                            g_stopping_power[type * num_points + index_right] * weight_right;

    double factor = stopping_power / sqrt(v2);

    g_force[0 * num_atoms + i] -= vx * factor;
    g_force[1 * num_atoms + i] -= vx * factor;
    g_force[2 * num_atoms + i] -= vx * factor;
  }
}

void __global__ apply_electron_stopping(
  const int num_atoms,
  const double time_step,
  const double energy_min,
  const double energy_max,
  const double energy_interval_inverse,
  const double* g_stopping_power,
  const int* g_type,
  const double* g_mass,
  const double* g_velocity,
  double* g_force)
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

    if (energy < energy_min + 1.0e-6) {
      return;
    }
    if (energy > energy_max - 1.0e-6) {
      return;
    }

    double fractional_energy = (energy - energy_min) * energy_interval_inverse;
    int index_left = static_cast<int>(fractional_energy);
    int index_right = index_left + 1;
    double weight_right = fractional_energy - index_left;
    double weight_left = 1.0 - weight_right;
    double stopping_power = g_stopping_power[type * num_points + index_left] * weight_left +
                            g_stopping_power[type * num_points + index_right] * weight_right;

    double factor = stopping_power / sqrt(v2);

    g_force[0 * num_atoms + i] -= vx * factor;
    g_force[1 * num_atoms + i] -= vx * factor;
    g_force[2 * num_atoms + i] -= vx * factor;
  }
}

void __global__ apply_electron_stopping(
  const int num_atoms,
  const double time_step,
  const double energy_min,
  const double energy_max,
  const double energy_interval_inverse,
  const double* g_stopping_power,
  const int* g_type,
  const double* g_mass,
  const double* g_velocity,
  double* g_force)
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

    if (energy < energy_min + 1.0e-6) {
      return;
    }
    if (energy > energy_max - 1.0e-6) {
      return;
    }

    double fractional_energy = (energy - energy_min) * energy_interval_inverse;
    int index_left = static_cast<int>(fractional_energy);
    int index_right = index_left + 1;
    double weight_right = fractional_energy - index_left;
    double weight_left = 1.0 - weight_right;
    double stopping_power = g_stopping_power[type * num_points + index_left] * weight_left +
                            g_stopping_power[type * num_points + index_right] * weight_right;

    double factor = stopping_power / sqrt(v2);

    g_force[0 * num_atoms + i] -= vx * factor;
    g_force[1 * num_atoms + i] -= vx * factor;
    g_force[2 * num_atoms + i] -= vx * factor;
  }
}

void Electron_Stop::compute(const int num_atoms, const double time_step, Atom& atom)
{
  if (!do_electron_stop) {
    return;
  }

  apply_electron_stopping<<<(num_atoms - 1) / 128 + 1, 128>>>(
    num_atoms,
    time_step,
    energy_min,
    energy_max,
    1.0 / energy_interval_inverse,
    stopping_power_gpu.data(),
    atom.type.data(),
    atom.mass.data(),
    atom.velocity_per_atom.data());
}

void Electron_Stop::parse(const char** param, int num_param, const int num_types)
{
  printf("Apply electron stopping.\n");
  if (num_param != 2) {
    PRINT_INPUT_ERROR("electron_stop should have 1 parameter.\n");
  }
  printf("     using the stopping power data in %s.\n", param[1]);

  std::ifstream input(param[1]);
  if (!input.is_open()) {
    printf("Failed to open %s.\n", param[1]);
  }

  input >> num_points;
  if (num_points < 2) {
    printf("Number of stopping power values should >= 2.\n");
  } else {
    printf("    number of energy points = %d.\n", num_points);
  }

  stopping_power_cpu.resize(num_points * num_types);
  for (int n = 0; n < num_points; ++n) {
    input >> stopping_power_cpu[0 * num_points + n];
    for (int t = 0; t < num_types; ++t) {
      input >> stopping_power_cpu[(t + 1) * num_points + n];
    }
  }

  energy_min = stopping_power_cpu[0];
  energy_max = stopping_power_cpu[num_points - 1];
  energy_interval = (energy_max - energy_min) / (num_points - 1);
  printf("    minimal energy = %g eV.\n", energy_min);
  printf("    maximal energy = %g eV.\n", energy_max);
  printf("    energy interval = %g eV.\n", energy_interval);

  stopping_power_gpu.resize(num_points * num_types);
  stopping_power_gpu.copy_from_host(stopping_power_cpu.data());
  do_electron_stop = true;

  // test
  for (int n = 0; n < num_points; ++n) {
    std::cout << stopping_power_cpu[0 * num_points + n] << " ";
    for (int t = 0; t < num_types; ++t) {
      std::cout << stopping_power_cpu[(t + 1) * num_points + n] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void Electron_Stop::finalize() { do_electron_stop = false; }
