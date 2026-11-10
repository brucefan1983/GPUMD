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
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Thermo reduction and SVR rescaling of the replica thermostat, shared by one
replica (Ensemble_BDP_Replica) and the concatenated batch (Replica_Batch).
------------------------------------------------------------------------------*/

#pragma once

#include "utilities/common.cuh"

// <<<dim3(8, replicas), 1024>>>: block (b, r) computes thermo[r * 8 + b] = T, U, s_xx, s_yy,
// s_zz, s_xy, s_xz, s_yz of replica r. Replica r has atoms [r * N, r * N + N) of arrays whose
// x, y and z (or virial) components are `stride` apart.
static __global__ void gpu_find_thermo_replicas(
  const int N,
  const int stride,
  const int degrees_of_freedom,
  const double volume,
  const double* g_mass,
  const double* g_potential,
  const double* g_velocity,
  const double* g_virial,
  double* g_thermo)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int offset0 = blockIdx.y * N;
  const double* g_vx = g_velocity + offset0;
  const double* g_vy = g_velocity + stride + offset0;
  const double* g_vz = g_velocity + stride * 2 + offset0;
  const double* g_s = g_virial + offset0 + (bid >= 2 ? stride * (bid - 2) : 0);
  const int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    const int n = tid + patch * 1024;
    if (n < N) {
      const double mass = g_mass[offset0 + n];
      const double vx = g_vx[n];
      const double vy = g_vy[n];
      const double vz = g_vz[n];
      switch (bid) {
        case 0:
          s_data[tid] += (vx * vx + vy * vy + vz * vz) * mass;
          break;
        case 1:
          s_data[tid] += g_potential[offset0 + n];
          break;
        case 2:
          s_data[tid] += g_s[n] + vx * vx * mass;
          break;
        case 3:
          s_data[tid] += g_s[n] + vy * vy * mass;
          break;
        case 4:
          s_data[tid] += g_s[n] + vz * vz * mass;
          break;
        case 5:
          s_data[tid] += g_s[n] + vx * vy * mass;
          break;
        case 6:
          s_data[tid] += g_s[n] + vx * vz * mass;
          break;
        default:
          s_data[tid] += g_s[n] + vy * vz * mass;
          break;
      }
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    double* thermo = g_thermo + blockIdx.y * 8;
    if (bid == 0) {
      thermo[0] = s_data[0] / (degrees_of_freedom * K_B);
    } else if (bid == 1) {
      thermo[1] = s_data[0];
    } else {
      thermo[bid] = s_data[0] / volume;
    }
  }
}

// The velocity factor of resamplekin() (integrate/svr_utilities.cuh), with the temperature
// thermo[0] read on the device.
static __device__ double svr_velocity_factor(
  const double thermo_temperature,
  const int ndeg,
  const double temperature,
  const double temperature_coupling,
  const double gaussian,
  const double sum_of_squares)
{
  const double kinetic_energy = thermo_temperature * ndeg * K_B * 0.5;
  const double target_kinetic_energy = ndeg * K_B * temperature * 0.5;
  const double coupling = temperature_coupling > 0.1 ? exp(-1.0 / temperature_coupling) : 0.0;
  const double new_kinetic_energy =
    kinetic_energy +
    (1.0 - coupling) *
      (target_kinetic_energy * (sum_of_squares + gaussian * gaussian) / ndeg - kinetic_energy) +
    2.0 * gaussian *
      sqrt(kinetic_energy * target_kinetic_energy / ndeg * (1.0 - coupling) * coupling);
  return sqrt(new_kinetic_energy / kinetic_energy);
}
