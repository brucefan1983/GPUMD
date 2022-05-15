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
Some CUDA kernels for Langevin thermostats.
------------------------------------------------------------------------------*/

#define CURAND_NORMAL(a) curand_normal_double(a)

// initialize curand states
static __global__ void initialize_curand_states(curandState* state, int N, int seed)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    curand_init(seed, n, 0, &state[n]);
  }
}

// global Langevin thermostatting
static __global__ void gpu_langevin(
  curandState* g_state,
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
    curandState state = g_state[n];
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

  for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
    if (tid < offset) {
      s_momentum[tid] += s_momentum[tid + offset];
    }
    __syncthreads();
  }
  for (int offset = 32; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_momentum[tid] += s_momentum[tid + offset];
    }
    __syncwarp();
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
  curandState* g_state,
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
    curandState state = g_state[m];
    int n = g_group_contents[offset + m];
    double c2m = c2 * sqrt(1.0 / g_mass[n]);
    g_vx[n] = c1 * g_vx[n] + c2m * CURAND_NORMAL(&state);
    g_vy[n] = c1 * g_vy[n] + c2m * CURAND_NORMAL(&state);
    g_vz[n] = c1 * g_vz[n] + c2m * CURAND_NORMAL(&state);
    g_state[m] = state;
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
#pragma unroll
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