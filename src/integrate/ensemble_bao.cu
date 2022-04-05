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
The Langevin thermostat with the BAOAB splitting:
[1] Leimkuhler, Benedict, and Charles Matthews. Applied Mathematics Research
    eXpress 2013.1 (2013): 34-56.
[2] Fass, Josh, et al. Entropy 20.5 (2018): 318.
------------------------------------------------------------------------------*/

#include "ensemble_bao.cuh"
#include "utilities/common.cuh"
#define CURAND_NORMAL(a) curand_normal_double(a)

// initialize curand states
static __global__ void initialize_curand_states(curandState* state, int N)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  // We can use a fixed seed here.
  if (n < N) {
    curand_init(123456, n, 0, &state[n]);
  }
}

Ensemble_BAO::Ensemble_BAO(int t, int fg, int N, double T, double Tc)
{
  type = t;
  fixed_group = fg;
  temperature = T;
  temperature_coupling = Tc;
  c1 = exp(-1.0 / temperature_coupling);
  c2 = sqrt((1 - c1 * c1) * K_B * T);
  curand_states.resize(N);
  int grid_size = (N - 1) / 128 + 1;
  initialize_curand_states<<<grid_size, 128>>>(curand_states.data(), N);
  CUDA_CHECK_KERNEL
}

Ensemble_BAO::Ensemble_BAO(
  int t,
  int fg,
  int source_input,
  int sink_input,
  int source_size,
  int sink_size,
  int source_offset,
  int sink_offset,
  double T,
  double Tc,
  double dT)
{
  type = t;
  fixed_group = fg;
  temperature = T;
  temperature_coupling = Tc;
  delta_temperature = dT;
  source = source_input;
  sink = sink_input;
  N_source = source_size;
  N_sink = sink_size;
  offset_source = source_offset;
  offset_sink = sink_offset;
  c1 = exp(-1.0 / temperature_coupling);
  c2_source = sqrt((1 - c1 * c1) * K_B * (T + dT));
  c2_sink = sqrt((1 - c1 * c1) * K_B * (T - dT));
  curand_states_source.resize(N_source);
  curand_states_sink.resize(N_sink);
  int grid_size_source = (N_source - 1) / 128 + 1;
  int grid_size_sink = (N_sink - 1) / 128 + 1;
  initialize_curand_states<<<grid_size_source, 128>>>(curand_states_source.data(), N_source);
  CUDA_CHECK_KERNEL
  initialize_curand_states<<<grid_size_sink, 128>>>(curand_states_sink.data(), N_sink);
  CUDA_CHECK_KERNEL
  energy_transferred[0] = 0.0;
  energy_transferred[1] = 0.0;
}

Ensemble_BAO::~Ensemble_BAO(void)
{
  // nothing
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

__device__ double device_momentum[4];

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

// wrapper of the above kernels
void Ensemble_BAO::integrate_nvt_lan(
  const GPU_Vector<double>& mass, GPU_Vector<double>& velocity_per_atom)
{
  const int number_of_atoms = mass.size();

  gpu_langevin<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    curand_states.data(), number_of_atoms, c1, c2, mass.data(), velocity_per_atom.data(),
    velocity_per_atom.data() + number_of_atoms, velocity_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL

  gpu_find_momentum<<<4, 1024>>>(
    number_of_atoms, mass.data(), velocity_per_atom.data(),
    velocity_per_atom.data() + number_of_atoms, velocity_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL

  gpu_correct_momentum<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms, velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL
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

// wrapper of the above two kernels
void Ensemble_BAO::integrate_heat_lan(
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  GPU_Vector<double>& velocity_per_atom)
{
  const int number_of_atoms = mass.size();

  int Ng = group[0].number;

  std::vector<double> ek2(Ng);
  GPU_Vector<double> ke(Ng);

  find_ke<<<Ng, 512>>>(
    group[0].size.data(), group[0].size_sum.data(), group[0].contents.data(), mass.data(),
    velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms, ke.data());
  CUDA_CHECK_KERNEL

  ke.copy_to_host(ek2.data());
  energy_transferred[0] += ek2[source] * 0.5;
  energy_transferred[1] += ek2[sink] * 0.5;

  gpu_langevin<<<(N_source - 1) / 128 + 1, 128>>>(
    curand_states_source.data(), N_source, offset_source, group[0].contents.data(), c1, c2_source,
    mass.data(), velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL

  gpu_langevin<<<(N_sink - 1) / 128 + 1, 128>>>(
    curand_states_sink.data(), N_sink, offset_sink, group[0].contents.data(), c1, c2_sink,
    mass.data(), velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL

  find_ke<<<Ng, 512>>>(
    group[0].size.data(), group[0].size_sum.data(), group[0].contents.data(), mass.data(),
    velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms, ke.data());
  CUDA_CHECK_KERNEL

  ke.copy_to_host(ek2.data());
  energy_transferred[0] -= ek2[source] * 0.5;
  energy_transferred[1] -= ek2[sink] * 0.5;
}

// the A operator.
static __global__ void gpu_operator_A(
  const int number_of_particles,
  const double g_time_step,
  const double* g_mass,
  double* g_x,
  double* g_y,
  double* g_z,
  double* g_vx,
  double* g_vy,
  double* g_vz,
  const double* g_fx,
  const double* g_fy,
  const double* g_fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    const double time_step = g_time_step;
    const double time_step_half = time_step * 0.5;
    double vx = g_vx[i];
    double vy = g_vy[i];
    double vz = g_vz[i];
    g_x[i] += vx * time_step_half;
    g_y[i] += vy * time_step_half;
    g_z[i] += vz * time_step_half;
  }
}

// wrapper of the above kernel
void Ensemble_BAO::operator_A(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom)
{
  const int number_of_atoms = mass.size();

  gpu_operator_A<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms, time_step, mass.data(), position_per_atom.data(),
    position_per_atom.data() + number_of_atoms, position_per_atom.data() + number_of_atoms * 2,
    velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
    velocity_per_atom.data() + 2 * number_of_atoms, force_per_atom.data(),
    force_per_atom.data() + number_of_atoms, force_per_atom.data() + 2 * number_of_atoms);
  CUDA_CHECK_KERNEL
}

// the B operator.
static __global__ void gpu_operator_B(
  const int number_of_particles,
  const int fixed_group,
  const int* group_id,
  const double g_time_step,
  const double* g_mass,
  double* g_x,
  double* g_y,
  double* g_z,
  double* g_vx,
  double* g_vy,
  double* g_vz,
  const double* g_fx,
  const double* g_fy,
  const double* g_fz)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    const double time_step = g_time_step;
    const double time_step_half = time_step * 0.5;
    double vx = g_vx[i];
    double vy = g_vy[i];
    double vz = g_vz[i];
    const double mass_inv = 1.0 / g_mass[i];
    const double ax = g_fx[i] * mass_inv;
    const double ay = g_fy[i] * mass_inv;
    const double az = g_fz[i] * mass_inv;
    if (group_id[i] == fixed_group) {
      vx = 0.0;
      vy = 0.0;
      vz = 0.0;
    } else {
      vx += ax * time_step_half;
      vy += ay * time_step_half;
      vz += az * time_step_half;
    }
    g_vx[i] = vx;
    g_vy[i] = vy;
    g_vz[i] = vz;
  }
}

static __global__ void gpu_operator_B(
  const int number_of_particles,
  const double g_time_step,
  const double* g_mass,
  double* g_x,
  double* g_y,
  double* g_z,
  double* g_vx,
  double* g_vy,
  double* g_vz,
  const double* g_fx,
  const double* g_fy,
  const double* g_fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < number_of_particles) {
    const double time_step = g_time_step;
    const double time_step_half = time_step * 0.5;
    double vx = g_vx[i];
    double vy = g_vy[i];
    double vz = g_vz[i];
    const double mass_inv = 1.0 / g_mass[i];
    const double ax = g_fx[i] * mass_inv;
    const double ay = g_fy[i] * mass_inv;
    const double az = g_fz[i] * mass_inv;
    vx += ax * time_step_half;
    vy += ay * time_step_half;
    vz += az * time_step_half;
    g_vx[i] = vx;
    g_vy[i] = vy;
    g_vz[i] = vz;
  }
}

// wrapper of the above two kernels
void Ensemble_BAO::operator_B(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom)
{
  const int number_of_atoms = mass.size();

  if (fixed_group == -1) {
    gpu_operator_B<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, time_step, mass.data(), position_per_atom.data(),
      position_per_atom.data() + number_of_atoms, position_per_atom.data() + number_of_atoms * 2,
      velocity_per_atom.data(), velocity_per_atom.data() + number_of_atoms,
      velocity_per_atom.data() + 2 * number_of_atoms, force_per_atom.data(),
      force_per_atom.data() + number_of_atoms, force_per_atom.data() + 2 * number_of_atoms);
  } else {
    gpu_operator_B<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
      number_of_atoms, fixed_group, group[0].label.data(), time_step, mass.data(),
      position_per_atom.data(), position_per_atom.data() + number_of_atoms,
      position_per_atom.data() + number_of_atoms * 2, velocity_per_atom.data(),
      velocity_per_atom.data() + number_of_atoms, velocity_per_atom.data() + 2 * number_of_atoms,
      force_per_atom.data(), force_per_atom.data() + number_of_atoms,
      force_per_atom.data() + 2 * number_of_atoms);
  }
  CUDA_CHECK_KERNEL
}

void Ensemble_BAO::compute1(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{
  if (type == 5) {
    operator_B(
      time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

    operator_A(
      time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

    integrate_nvt_lan(mass, velocity_per_atom);

    operator_A(
      time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);
  } else {
    operator_B(
      time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

    operator_A(
      time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

    integrate_heat_lan(group, mass, velocity_per_atom);

    operator_A(
      time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);
  }
}

void Ensemble_BAO::compute2(
  const double time_step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential_per_atom,
  const GPU_Vector<double>& force_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& thermo)
{
  if (type == 5) {
    operator_B(
      time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);

    find_thermo(
      true, box.get_volume(), group, mass, potential_per_atom, velocity_per_atom, virial_per_atom,
      thermo);
  } else {
    operator_B(
      time_step, group, mass, force_per_atom, position_per_atom, velocity_per_atom);
  }
}