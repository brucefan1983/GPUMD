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
Some CUDA kernels for Langevin thermostats.
------------------------------------------------------------------------------*/

#pragma once
#include "model/box.cuh"
#include "utilities/gpu_macro.cuh"

#define CURAND_NORMAL(a) gpurand_normal_double(a)

// initialize curand states
static __global__ void initialize_curand_states(gpurandState* state, int N, int seed)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    gpurand_init(seed, n, 0, &state[n]);
  }
}

// global Langevin thermostatting
static __global__ void gpu_langevin(
  gpurandState* g_state,
  const int N,
  const double c1,
  const double c2,
  const double* g_mass,
  double* g_vx,
  double* g_vy,
  double* g_vz)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    gpurandState state = g_state[n];
    double c2m = c2 * sqrt(1.0 / g_mass[n]);
    g_vx[n] = c1 * g_vx[n] + c2m * CURAND_NORMAL(&state);
    g_vy[n] = c1 * g_vy[n] + c2m * CURAND_NORMAL(&state);
    g_vz[n] = c1 * g_vz[n] + c2m * CURAND_NORMAL(&state);
    g_state[n] = state;
  }
}

// total momentums and total mass
__device__ double device_momentum[4];

// get the total momentum in each direction
static __global__ void gpu_find_momentum(
  const int N, const double* g_mass, const double* g_vx, const double* g_vy, const double* g_vz)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_rounds = (N - 1) / 1024 + 1;
  __shared__ double s_momentum[1024];
  double momentum = 0.0;

  switch (bid) {
    case 0:
      for (int round = 0; round < number_of_rounds; ++round) {
        int n = tid + round * 1024;
        if (n < N)
          momentum += g_mass[n] * g_vx[n];
      }
      break;
    case 1:
      for (int round = 0; round < number_of_rounds; ++round) {
        int n = tid + round * 1024;
        if (n < N)
          momentum += g_mass[n] * g_vy[n];
      }
      break;
    case 2:
      for (int round = 0; round < number_of_rounds; ++round) {
        int n = tid + round * 1024;
        if (n < N)
          momentum += g_mass[n] * g_vz[n];
      }
      break;
    case 3:
      for (int round = 0; round < number_of_rounds; ++round) {
        int n = tid + round * 1024;
        if (n < N)
          momentum += g_mass[n];
      }
      break;
  }
  s_momentum[tid] = momentum;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_momentum[tid] += s_momentum[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    device_momentum[bid] = s_momentum[0];
  }
}

// correct the momentum
static __global__ void gpu_correct_momentum(const int N, double* g_vx, double* g_vy, double* g_vz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double inverse_of_total_mass = 1.0 / device_momentum[3];
    g_vx[i] -= device_momentum[0] * inverse_of_total_mass;
    g_vy[i] -= device_momentum[1] * inverse_of_total_mass;
    g_vz[i] -= device_momentum[2] * inverse_of_total_mass;
  }
}

// local Langevin thermostatting
static __global__ void gpu_langevin(
  gpurandState* g_state,
  const int N,
  const int offset,
  const int* g_group_contents,
  const double c1,
  const double c2,
  const double* g_mass,
  double* g_vx,
  double* g_vy,
  double* g_vz)
{
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m < N) {
    gpurandState state = g_state[m];
    int n = g_group_contents[offset + m];
    double c2m = c2 * sqrt(1.0 / g_mass[n]);
    g_vx[n] = c1 * g_vx[n] + c2m * CURAND_NORMAL(&state);
    g_vy[n] = c1 * g_vy[n] + c2m * CURAND_NORMAL(&state);
    g_vz[n] = c1 * g_vz[n] + c2m * CURAND_NORMAL(&state);
    g_state[m] = state;
  }
}

static __device__ void get_fractional_position(
  const Box& box,
  const double x,
  const double y,
  const double z,
  double& sa,
  double& sb,
  double& sc)
{
  sa = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
  sb = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
  sc = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;

  if (box.pbc_x == 1) {
    if (sa < 0.0) {
      sa += 1.0;
    } else if (sa >= 1.0) {
      sa -= 1.0;
    }
  }
  if (box.pbc_y == 1) {
    if (sb < 0.0) {
      sb += 1.0;
    } else if (sb >= 1.0) {
      sb -= 1.0;
    }
  }
  if (box.pbc_z == 1) {
    if (sc < 0.0) {
      sc += 1.0;
    } else if (sc >= 1.0) {
      sc -= 1.0;
    }
  }
}

static __device__ bool is_in_region(
  const double sa,
  const double sb,
  const double sc,
  const double amin,
  const double amax,
  const double bmin,
  const double bmax,
  const double cmin,
  const double cmax)
{
  return sa >= amin && sa < amax && sb >= bmin && sb < bmax && sc >= cmin && sc < cmax;
}

// local Langevin thermostatting based on fractional-coordinate regions
static __global__ void gpu_langevin_region(
  gpurandState* g_state,
  const int N,
  const Box box,
  const double source_amin,
  const double source_amax,
  const double source_bmin,
  const double source_bmax,
  const double source_cmin,
  const double source_cmax,
  const double sink_amin,
  const double sink_amax,
  const double sink_bmin,
  const double sink_bmax,
  const double sink_cmin,
  const double sink_cmax,
  const double c1,
  const double c2_source,
  const double c2_sink,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const double* g_mass,
  double* g_vx,
  double* g_vy,
  double* g_vz)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double sa, sb, sc;
    get_fractional_position(box, g_x[n], g_y[n], g_z[n], sa, sb, sc);

    double c2 = 0.0;
    if (is_in_region(
          sa,
          sb,
          sc,
          source_amin,
          source_amax,
          source_bmin,
          source_bmax,
          source_cmin,
          source_cmax)) {
      c2 = c2_source;
    } else if (is_in_region(
                 sa,
                 sb,
                 sc,
                 sink_amin,
                 sink_amax,
                 sink_bmin,
                 sink_bmax,
                 sink_cmin,
                 sink_cmax)) {
      c2 = c2_sink;
    } else {
      return;
    }

    gpurandState state = g_state[n];
    double c2m = c2 * sqrt(1.0 / g_mass[n]);
    g_vx[n] = c1 * g_vx[n] + c2m * CURAND_NORMAL(&state);
    g_vy[n] = c1 * g_vy[n] + c2m * CURAND_NORMAL(&state);
    g_vz[n] = c1 * g_vz[n] + c2m * CURAND_NORMAL(&state);
    g_state[n] = state;
  }
}

// group kinetic energy
static __global__ void find_ke(
  const int* g_group_size,
  const int* g_group_size_sum,
  const int* g_group_contents,
  const double* g_mass,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  double* g_ke)
{
  //<<<number_of_groups, 512>>>
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int group_size = g_group_size[bid];
  int offset = g_group_size_sum[bid];
  int number_of_patches = (group_size - 1) / 512 + 1;
  __shared__ double s_ke[512]; // relative kinetic energy
  s_ke[tid] = 0.0;
  for (int patch = 0; patch < number_of_patches; ++patch) {
    int n = tid + patch * 512;
    if (n < group_size) {
      int index = g_group_contents[offset + n];
      double mass = g_mass[index];
      double vx = g_vx[index];
      double vy = g_vy[index];
      double vz = g_vz[index];
      s_ke[tid] += (vx * vx + vy * vy + vz * vz) * mass;
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_ke[tid] += s_ke[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_ke[bid] = s_ke[0];
  } // kinetic energy times 2
}

// kinetic energy in the source and sink regions
static __global__ void find_ke_region(
  const int N,
  const Box box,
  const double source_amin,
  const double source_amax,
  const double source_bmin,
  const double source_bmax,
  const double source_cmin,
  const double source_cmax,
  const double sink_amin,
  const double sink_amax,
  const double sink_bmin,
  const double sink_bmax,
  const double sink_cmin,
  const double sink_cmax,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const double* g_mass,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  double* g_ke)
{
  //<<<2, 512>>>
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_patches = (N - 1) / 512 + 1;
  __shared__ double s_ke[512];
  s_ke[tid] = 0.0;

  for (int patch = 0; patch < number_of_patches; ++patch) {
    int n = tid + patch * 512;
    if (n < N) {
      double sa, sb, sc;
      get_fractional_position(box, g_x[n], g_y[n], g_z[n], sa, sb, sc);
      bool in_region;
      if (bid == 0) {
        in_region = is_in_region(
          sa,
          sb,
          sc,
          source_amin,
          source_amax,
          source_bmin,
          source_bmax,
          source_cmin,
          source_cmax);
      } else {
        in_region = is_in_region(
          sa,
          sb,
          sc,
          sink_amin,
          sink_amax,
          sink_bmin,
          sink_bmax,
          sink_cmin,
          sink_cmax);
      }
      if (in_region) {
        double mass = g_mass[n];
        double vx = g_vx[n];
        double vy = g_vy[n];
        double vz = g_vz[n];
        s_ke[tid] += (vx * vx + vy * vy + vz * vz) * mass;
      }
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_ke[tid] += s_ke[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_ke[bid] = s_ke[0];
  } // kinetic energy times 2
}

