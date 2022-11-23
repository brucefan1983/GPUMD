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
Calculate the stress autocorrelation function and viscosity.
------------------------------------------------------------------------------*/

#include "utilities/common.cuh"
#include "utilities/read_file.cuh"
#include "viscosity.cuh"
#include <vector>

#define NUM_OF_COMPONENTS 6

void Viscosity::preprocess(const int number_of_steps)
{
  if (compute) {
    int number_of_frames = number_of_steps / sample_interval;
    stress_all.resize(NUM_OF_COMPONENTS * number_of_frames);
  }
}

static __global__ void gpu_sum_stress(
  const int N,
  const int Nd,
  const int nd,
  const double* g_mass,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  const double* g_virial,
  double* g_stress_all)
{
  const int tid = threadIdx.x;
  const int number_of_rounds = (N - 1) / 1024 + 1;

  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (int round = 0; round < number_of_rounds; ++round) {
    const int n = tid + round * 1024;
    if (n < N) {
      // the virial tensor:
      // xx xy xz    0 3 4
      // yx yy yz    6 1 5
      // zx zy zz    7 8 2
      s_data[tid] += g_virial[n + N * (blockIdx.x + 3)];
      switch (blockIdx.x) {
        case 0:
          s_data[tid] += g_mass[n] * g_vx[n] * g_vy[n];
          break;
        case 1:
          s_data[tid] += g_mass[n] * g_vx[n] * g_vz[n];
          break;
        case 2:
          s_data[tid] += g_mass[n] * g_vy[n] * g_vz[n];
          break;
        case 3:
          s_data[tid] += g_mass[n] * g_vy[n] * g_vx[n];
          break;
        case 4:
          s_data[tid] += g_mass[n] * g_vz[n] * g_vx[n];
          break;
        case 5:
          s_data[tid] += g_mass[n] * g_vz[n] * g_vy[n];
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
    g_stress_all[nd + Nd * blockIdx.x] = s_data[0];
  }
}

void Viscosity::process(
  const int number_of_steps,
  const int step,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& velocity,
  const GPU_Vector<double>& virial)
{
  if (!compute)
    return;
  if ((step + 1) % sample_interval != 0)
    return;

  const int N = velocity.size() / 3;

  int nd = (step + 1) / sample_interval - 1;
  int Nd = number_of_steps / sample_interval;
  gpu_sum_stress<<<NUM_OF_COMPONENTS, 1024>>>(
    N, Nd, nd, mass.data(), velocity.data(), velocity.data() + N, velocity.data() + N * 2,
    virial.data(), stress_all.data());
  CUDA_CHECK_KERNEL
}

static __global__ void
gpu_find_correlation(const int Nc, const int Nd, const double* g_stress, double* g_correlation)
{
  __shared__ double s_correlation[768];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_rounds = (Nd - 1) / 128 + 1;
  int number_of_data = Nd - bid;

  s_correlation[tid] = 0.0;

  for (int round = 0; round < number_of_rounds; ++round) {
    int index = tid + round * 128;
    if (index + bid < Nd) {
      for (int k = 0; k < NUM_OF_COMPONENTS; ++k) {
        s_correlation[tid + k * 128] += g_stress[index + Nd * k] * g_stress[index + bid + Nd * k];
      }
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      for (int k = 0; k < NUM_OF_COMPONENTS; ++k) {
        s_correlation[tid + k * 128] += s_correlation[tid + offset + k * 128];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int k = 0; k < NUM_OF_COMPONENTS; ++k) {
      g_correlation[bid + Nc * k] = s_correlation[0 + k * 128] / number_of_data;
    }
  }
}

static void
find_viscosity(const int Nc, const double factor, const double* correlation, double* viscosity)
{
  for (int k = 0; k < NUM_OF_COMPONENTS; k++) {
    for (int nc = 1; nc < Nc; nc++) {
      const int index = Nc * k + nc;
      viscosity[index] =
        viscosity[index - 1] + (correlation[index - 1] + correlation[index]) * factor;
    }
  }
}

void Viscosity::postprocess(
  const int number_of_steps, const double temperature, const double time_step, const double volume)
{
  if (!compute)
    return;
  print_line_1();
  printf("Start to calculate viscosity.\n");

  const int Nd = number_of_steps / sample_interval;
  const double dt = time_step * sample_interval;
  const double dt_in_ps = dt * TIME_UNIT_CONVERSION / 1000.0; // ps

  std::vector<double> viscosity(Nc * NUM_OF_COMPONENTS, 0.0);
  GPU_Vector<double> correlation_gpu(Nc * NUM_OF_COMPONENTS);
  std::vector<double> correlation_cpu(Nc * NUM_OF_COMPONENTS);

  gpu_find_correlation<<<Nc, 128>>>(Nc, Nd, stress_all.data(), correlation_gpu.data());
  CUDA_CHECK_KERNEL

  correlation_gpu.copy_to_host(correlation_cpu.data());

  double factor = dt * 0.5 / (K_B * temperature * volume);
  factor *= PRESSURE_UNIT_CONVERSION * TIME_UNIT_CONVERSION * 1.0e-6; // Pa s

  find_viscosity(Nc, factor, correlation_cpu.data(), viscosity.data());

  FILE* fid = fopen("viscosity.out", "a");
  for (int nc = 0; nc < Nc; nc++) {
    fprintf(fid, "%25.15e", nc * dt_in_ps);
    for (int m = 0; m < NUM_OF_COMPONENTS; m++) {
      fprintf(fid, "%25.15e", correlation_cpu[Nc * m + nc]);
    }
    for (int m = 0; m < NUM_OF_COMPONENTS; m++) {
      fprintf(fid, "%25.15e", viscosity[Nc * m + nc]);
    }
    fprintf(fid, "\n");
  }
  fflush(fid);
  fclose(fid);

  printf("Viscosity is calculated.\n");
  print_line_2();

  compute = 0;
}

void Viscosity::parse(const char** param, int num_param)
{
  compute = 1;

  printf("Compute Viscosity.\n");

  if (num_param != 3) {
    PRINT_INPUT_ERROR("compute_viscosity should have 2 parameters.\n");
  }

  if (!is_valid_int(param[1], &sample_interval)) {
    PRINT_INPUT_ERROR("sample interval for viscosity should be an integer number.\n");
  }
  printf("    sample interval is %d.\n", sample_interval);

  if (!is_valid_int(param[2], &Nc)) {
    PRINT_INPUT_ERROR("Nc for viscosity should be an integer number.\n");
  }
  printf("    Nc is %d\n", Nc);
}
