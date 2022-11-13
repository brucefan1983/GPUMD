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
Calculate the thermal conductivity using the HNEMD method.
Reference:
[1] Z. Fan, H. Dong, A. Harju, T. Ala-Nissila, Homogeneous nonequilibrium
molecular dynamics method for heat transport and spectral decomposition
with many-body potentials, Phys. Rev. B 99, 064308 (2019).
------------------------------------------------------------------------------*/

#include "compute_heat.cuh"
#include "hnemdec_kappa.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <vector>

#define NUM_OF_HEAT_COMPONENTS 3
#define FILE_NAME_LENGTH 200

void HNEMDEC::preprocess(
  const std::vector<double>& mass, const std::vector<int>& type, const std::vector<int>& type_size)
{
  if (compute == -1)
    return;
  heat_all.resize(NUM_OF_HEAT_COMPONENTS * output_interval);

  // find atom types' mass and factor
  number_of_types = type_size.size();
  NUM_OF_DIFFUSION_COMPONENTS = 3 * number_of_types;
  int N = mass.size();
  diffusion_all.resize(3 * number_of_types * output_interval);
  cpu_mass_type.resize(number_of_types);
  mass_type.resize(number_of_types);

  int find_mass_type = 0;
  for (int i = 0; i < N; i++) {
    if (cpu_mass_type[type[i]] != mass[i]) {
      cpu_mass_type[type[i]] = mass[i];
      find_mass_type += 1;
    }
    if (find_mass_type == number_of_types) {
      break;
    }
  }
  mass_type.copy_from_host(cpu_mass_type.data());

  if (compute == 0) {
    FACTOR = 1;
  } else {
    double patial_mass = 0;
    for (int i = 0; i < number_of_types; i++) {
      if (i != compute - 1) {
        patial_mass += type_size[i] * cpu_mass_type[i];
      }
    }
    FACTOR = N * (1.0 / patial_mass + 1.0 / (type_size[compute - 1] * cpu_mass_type[compute - 1]));
    FACTOR = 1.0 / FACTOR;
  }
}

static __global__ void gpu_sum_heat_and_diffusion(
  const int N,
  const int step,
  const int* g_type,
  const double* g_mass_type,
  const double* g_velocity,
  const double* g_heat,
  double* g_heat_sum,
  double* g_diffusion)
{
  // <<<3 + 3 * number_of_types, 1024>>>
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int number_of_patches = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  if (bid < 3) {
    for (int patch = 0; patch < number_of_patches; ++patch) {
      const int n = tid + patch * 1024;
      if (n < N) {
        s_data[tid] += g_heat[n + N * bid];
      }
    }
    __syncthreads();

#pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
      if (tid < offset) {
        s_data[tid] += s_data[tid + offset];
      }
      __syncthreads();
    }

    if (tid == 0) {
      g_heat_sum[step * NUM_OF_HEAT_COMPONENTS + bid] = s_data[0];
    }
  } else {
    int element_index = ((bid - NUM_OF_HEAT_COMPONENTS) / 3);
    int component = bid % 3;
    for (int patch = 0; patch < number_of_patches; ++patch) {
      const int n = tid + patch * 1024;
      if (n < N) {
        if (g_type[n] == element_index) {
          s_data[tid] += g_velocity[n + N * component];
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
      if (tid < offset) {
        s_data[tid] += s_data[tid + offset];
      }
      __syncthreads();
    }

    if (tid == 0) {
      g_diffusion[step * (gridDim.x - NUM_OF_HEAT_COMPONENTS) + bid - NUM_OF_HEAT_COMPONENTS] =
        g_mass_type[element_index] * s_data[0];
    }
  }
}

void HNEMDEC::process(
  int step,
  const char* input_dir,
  const double temperature,
  const double volume,
  const GPU_Vector<double>& velocity_per_atom,
  const GPU_Vector<double>& virial_per_atom,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential,
  GPU_Vector<double>& heat_per_atom)
{
  if (compute == -1)
    return;
  const int output_flag = ((step + 1) % output_interval == 0);
  step %= output_interval;

  const int N = velocity_per_atom.size() / 3;

  compute_heat(mass, potential, virial_per_atom, velocity_per_atom, heat_per_atom);

  gpu_sum_heat_and_diffusion<<<NUM_OF_HEAT_COMPONENTS + NUM_OF_DIFFUSION_COMPONENTS, 1024>>>(
    N, step, type.data(), mass_type.data(), velocity_per_atom.data(), heat_per_atom.data(),
    heat_all.data(), diffusion_all.data());
  CUDA_CHECK_KERNEL

  if (output_flag) {
    const int heat_num = NUM_OF_HEAT_COMPONENTS * output_interval;
    const int diffusion_num = NUM_OF_DIFFUSION_COMPONENTS * output_interval;
    std::vector<double> heat_cpu(heat_num);
    heat_all.copy_to_host(heat_cpu.data());
    std::vector<double> diffusion_cpu(diffusion_num);
    diffusion_all.copy_to_host(diffusion_cpu.data());
    double onsager1[NUM_OF_HEAT_COMPONENTS];
    std::vector<double> onsager2(NUM_OF_DIFFUSION_COMPONENTS);
    for (int n = 0; n < NUM_OF_HEAT_COMPONENTS; n++) {
      onsager1[n] = 0.0;
    }

    for (int m = 0; m < output_interval; m++) {
      for (int n = 0; n < NUM_OF_HEAT_COMPONENTS; n++) {
        onsager1[n] += heat_cpu[m * NUM_OF_HEAT_COMPONENTS + n];
        onsager2[n] += diffusion_cpu[m * NUM_OF_DIFFUSION_COMPONENTS + n];
      }
      for (int n = NUM_OF_HEAT_COMPONENTS; n < NUM_OF_DIFFUSION_COMPONENTS; n++) {
        onsager2[n] += diffusion_cpu[m * NUM_OF_DIFFUSION_COMPONENTS + n];
      }
    }

    double factor1, factor2;
    if (compute == 0) {
      factor1 = KAPPA_UNIT_CONVERSION / output_interval;
      factor1 /= (volume * temperature * fe);
      factor2 = 1631.0961499964144; // from natural to 10e-6 kg/smK
      factor2 *= FACTOR / (output_interval * volume * temperature * fe);
    } else if (compute > 0) {
      factor1 = 1631.0961499964144; // from natural to 10e-6 kg/smK
      factor1 *= FACTOR / (output_interval * volume * temperature * fe);
      factor2 = 16.905134572911963; // from natural to 10e-12 kgs/m^3k
      factor2 *= FACTOR / (output_interval * volume * temperature * fe);
    }

    char file_onsager[FILE_NAME_LENGTH];
    strcpy(file_onsager, input_dir);
    strcat(file_onsager, "/onsager.out");
    FILE* fid = fopen(file_onsager, "a");
    for (int n = 0; n < NUM_OF_HEAT_COMPONENTS; n++) {
      // [Lqq/T^2](W/mK) for compute==1,  [Lq1/T^2](kg/smK) for compute==2
      fprintf(fid, "%25.15f", onsager1[n] * factor1);
    }
    for (int n = 0; n < NUM_OF_DIFFUSION_COMPONENTS; n++) {
      // [L1q/T^2](kg/smK) for compute==1,  [L11/T^2](kgs/m^3k) for compute==2
      fprintf(fid, "%25.15f", onsager2[n] * factor2);
    }
    fprintf(fid, "\n");
    fflush(fid);
    fclose(fid);
  }
}

void HNEMDEC::postprocess() { compute = -1; }

void HNEMDEC::parse(const char** param, int num_param)
{
  printf("Compute thermal conductivity using the HNEMD Evans-Cummings method.\n");

  // compute_hnemdec compute output_interval fe_x fe_y fe_z
  if (num_param != 6) {
    PRINT_INPUT_ERROR("compute_hnemdec should have 5 parameters.\n");
  }

  if (!is_valid_int(param[1], &compute)) {
    PRINT_INPUT_ERROR("compute for HNEMDEC should be an integer number.\n");
  }

  if (!is_valid_int(param[2], &output_interval)) {
    PRINT_INPUT_ERROR("output_interval for HNEMDEC should be an integer number.\n");
  }

  if (output_interval < 1) {
    PRINT_INPUT_ERROR("output_interval for HNEMDEC should be larger than 0.\n");
  }
  if (!is_valid_real(param[3], &fe_x)) {
    PRINT_INPUT_ERROR("fe_x for HNEMDEC should be a real number.\n");
  }

  if (!is_valid_real(param[4], &fe_y)) {
    PRINT_INPUT_ERROR("fe_y for HNEMDEC should be a real number.\n");
  }

  if (!is_valid_real(param[5], &fe_z)) {
    PRINT_INPUT_ERROR("fe_z for HNEMDEC should be a real number.\n");
  }

  if (compute == 0) {
    printf("Using the HNEMD EC heat flow method.\n");
    printf("    output_interval = %d\n", output_interval);
    printf("    fe_x = %g /A\n", fe_x);
    printf("    fe_y = %g /A\n", fe_y);
    printf("    fe_z = %g /A\n", fe_z);
  } else if (compute > 0) {
    printf("Using the HNEMD EC color conductivity method.\n");
    printf("    output_interval = %d\n", output_interval);
    printf("    fe_x = %g /(eV/A)\n", fe_x);
    printf("    fe_y = %g /(eV/A)\n", fe_y);
    printf("    fe_z = %g /(eV/A)\n", fe_z);
  }

  // magnitude of the vector
  fe = fe_x * fe_x;
  fe += fe_y * fe_y;
  fe += fe_z * fe_z;
  fe = sqrt(fe);
}
