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
Calculate the thermal conductivity using the HNEMD method.
Reference:
[1] Z. Fan, H. Dong, A. Harju, T. Ala-Nissila, Homogeneous nonequilibrium
molecular dynamics method for heat transport and spectral decomposition
with many-body potentials, Phys. Rev. B 99, 064308 (2019).
------------------------------------------------------------------------------*/

#include "compute_heat.cuh"
#include "hnemd_kappa.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <vector>

#define NUM_OF_HEAT_COMPONENTS 5
#define FILE_NAME_LENGTH 200

void HNEMD::preprocess()
{
  if (!compute)
    return;
  heat_all.resize(NUM_OF_HEAT_COMPONENTS * output_interval);
}

static __global__ void
gpu_sum_heat(const int N, const int step, const double* g_heat, double* g_heat_sum)
{
  // <<<5, 1024>>>
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;
  for (int patch = 0; patch < number_of_patches; ++patch) {
    const int n = tid + patch * 1024;
    if (n < N) {
      s_data[tid] += g_heat[n + N * bid];
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
    g_heat_sum[step * NUM_OF_HEAT_COMPONENTS + bid] = s_data[0];
  }
}

void HNEMD::process(
  int step,
  const double temperature,
  const double volume,
  const GPU_Vector<double>& velocity_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& heat_per_atom)
{
  if (!compute)
    return;
  const int output_flag = ((step + 1) % output_interval == 0);
  step %= output_interval;

  const int N = velocity_per_atom.size() / 3;

  compute_heat(virial_per_atom, velocity_per_atom, heat_per_atom);

  gpu_sum_heat<<<NUM_OF_HEAT_COMPONENTS, 1024>>>(N, step, heat_per_atom.data(), heat_all.data());
  GPU_CHECK_KERNEL

  if (output_flag) {
    const int num = NUM_OF_HEAT_COMPONENTS * output_interval;
    std::vector<double> heat_cpu(num);
    heat_all.copy_to_host(heat_cpu.data());
    double kappa[NUM_OF_HEAT_COMPONENTS];
    for (int n = 0; n < NUM_OF_HEAT_COMPONENTS; n++) {
      kappa[n] = 0.0;
    }
    for (int m = 0; m < output_interval; m++) {
      for (int n = 0; n < NUM_OF_HEAT_COMPONENTS; n++) {
        kappa[n] += heat_cpu[m * NUM_OF_HEAT_COMPONENTS + n];
      }
    }
    double factor = KAPPA_UNIT_CONVERSION / output_interval;
    factor /= (volume * temperature * fe);

    FILE* fid = fopen("kappa.out", "a");
    for (int n = 0; n < NUM_OF_HEAT_COMPONENTS; n++) {
      fprintf(fid, "%25.15f", kappa[n] * factor);
    }
    fprintf(fid, "\n");
    fflush(fid);
    fclose(fid);
  }
}

void HNEMD::postprocess() { compute = 0; }

void HNEMD::parse(const char** param, int num_param)
{
  compute = 1;

  printf("Compute thermal conductivity using the HNEMD method.\n");

  if (num_param != 5) {
    PRINT_INPUT_ERROR("compute_hnemd should have 4 parameters.\n");
  }

  if (!is_valid_int(param[1], &output_interval)) {
    PRINT_INPUT_ERROR("output_interval for HNEMD should be an integer number.\n");
  }
  printf("    output_interval = %d\n", output_interval);
  if (output_interval < 1) {
    PRINT_INPUT_ERROR("output_interval for HNEMD should be larger than 0.\n");
  }
  if (!is_valid_real(param[2], &fe_x)) {
    PRINT_INPUT_ERROR("fe_x for HNEMD should be a real number.\n");
  }
  printf("    fe_x = %g /A\n", fe_x);
  if (!is_valid_real(param[3], &fe_y)) {
    PRINT_INPUT_ERROR("fe_y for HNEMD should be a real number.\n");
  }
  printf("    fe_y = %g /A\n", fe_y);
  if (!is_valid_real(param[4], &fe_z)) {
    PRINT_INPUT_ERROR("fe_z for HNEMD should be a real number.\n");
  }
  printf("    fe_z = %g /A\n", fe_z);

  // magnitude of the vector
  fe = fe_x * fe_x;
  fe += fe_y * fe_y;
  fe += fe_z * fe_z;
  fe = sqrt(fe);
}
