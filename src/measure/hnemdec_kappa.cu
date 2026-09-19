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
#include "hnemdec_kappa.cuh"
#include "integrate/integrate.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <vector>

#define NUM_OF_HEAT_COMPONENTS 3
#define FILE_NAME_LENGTH 200

void HNEMDEC::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (compute == -1)
    return;

  number_of_types = atom.cpu_type_size.size();
  if (compute > number_of_types) {
    PRINT_INPUT_ERROR(
      "compute for HNEMDEC should be an integer between 0 and number_of_types.\n");
  }

  heat_all.resize(NUM_OF_HEAT_COMPONENTS * output_interval);
  atom.heat_per_atom.resize(atom.number_of_atoms * 5);

  NUM_OF_DIFFUSION_COMPONENTS = 3 * number_of_types;
  const int N = atom.number_of_atoms;
  diffusion_all.resize(NUM_OF_DIFFUSION_COMPONENTS * output_interval);
  cpu_mass_type.assign(number_of_types, 0.0);
  mass_type.resize(number_of_types);

  double total_mass = 0.0;
  int find_mass_type = 0;
  for (int i = 0; i < N; i++) {
    if (cpu_mass_type[atom.cpu_type[i]] != atom.cpu_mass[i]) {
      cpu_mass_type[atom.cpu_type[i]] = atom.cpu_mass[i];
      find_mass_type += 1;
    }
    total_mass += atom.cpu_mass[i];
  }
  if (find_mass_type != number_of_types) {
    PRINT_INPUT_ERROR("mass type and element type do not match.\n");
  }
  mass_type.copy_from_host(cpu_mass_type.data());

  if (compute == 0) {
    std::vector<double> cpu_coefficient(number_of_types * 2);
    coefficient_.resize(number_of_types * 2);
    tensor_per_atom_.resize(static_cast<size_t>(N) * 9);
    tensor_sum_.resize(9);

    const double temperature = integrate.get_temperature1();
    for (int i = 0; i < number_of_types; i++) {
      double c_hv = (total_mass - N * cpu_mass_type[i]) / total_mass;
      cpu_coefficient[i * 2] = (c_hv - 1) / N;
      cpu_coefficient[i * 2 + 1] = K_B * temperature * c_hv;
    }
    coefficient_.copy_from_host(cpu_coefficient.data());
    FACTOR = 1.0;
  } else {
    const int element_index = compute - 1;
    std::vector<double> cpu_coefficient(number_of_types, 0.0);
    cpu_coefficient[element_index] = double(N) / atom.cpu_type_size[element_index];

    double partial_mass = 0.0;
    for (int i = 0; i < number_of_types; i++) {
      if (i != element_index) {
        partial_mass += cpu_mass_type[i] * atom.cpu_type_size[i];
      }
    }
    for (int i = 0; i < number_of_types; i++) {
      if (i != element_index) {
        cpu_coefficient[i] = -1.0 * N * cpu_mass_type[i] / partial_mass;
      }
    }

    coefficient_.resize(number_of_types);
    coefficient_.copy_from_host(cpu_coefficient.data());

    FACTOR =
      N *
      (1.0 / partial_mass +
       1.0 / (atom.cpu_type_size[element_index] * cpu_mass_type[element_index]));
    FACTOR = 1.0 / FACTOR;
  }
}

static __global__ void gpu_find_per_atom_tensor(
  int N,
  double* g_mass,
  double* g_potential,
  double* g_vx,
  double* g_vy,
  double* g_vz,
  double* g_sxx,
  double* g_sxy,
  double* g_sxz,
  double* g_syx,
  double* g_syy,
  double* g_syz,
  double* g_szx,
  double* g_szy,
  double* g_szz,
  double* g_tensor)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double mass = g_mass[i];
    double potential = g_potential[i];
    double vx = g_vx[i];
    double vy = g_vy[i];
    double vz = g_vz[i];
    double energy = mass * (vx * vx + vy * vy + vz * vz) * 0.5 + potential;
    // the tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_tensor[i] = energy + g_sxx[i];
    g_tensor[i + 3 * N] = g_sxy[i];
    g_tensor[i + 4 * N] = g_sxz[i];
    g_tensor[i + 6 * N] = g_syx[i];
    g_tensor[i + N] = energy + g_syy[i];
    g_tensor[i + 5 * N] = g_syz[i];
    g_tensor[i + 7 * N] = g_szx[i];
    g_tensor[i + 8 * N] = g_szy[i];
    g_tensor[i + 2 * N] = energy + g_szz[i];
  }
}

static __global__ void gpu_sum_tensor(int N, double* g_tensor, double* g_sum_tensor)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ double s_t[1024];
  double t = 0.0;

  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N)
      t += g_tensor[bid * N + n];
  }
  s_t[tid] = t;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_t[tid] += s_t[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_sum_tensor[bid] = s_t[0];
  }
}

static __global__ void gpu_add_heat_flow_driving_force(
  int N,
  const double* g_coefficient,
  const int* g_type,
  double fe_x,
  double fe_y,
  double fe_z,
  double* g_sxx,
  double* g_sxy,
  double* g_sxz,
  double* g_syx,
  double* g_syy,
  double* g_syz,
  double* g_szx,
  double* g_szy,
  double* g_szz,
  double* g_tensor_tot,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int type2 = g_type[i] * 2;
    double coefficient1 = g_coefficient[type2];
    double coefficient2 = g_coefficient[type2 + 1];
    // the tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_fx[i] += fe_x * (g_sxx[i] + coefficient1 * g_tensor_tot[0] + coefficient2) +
               fe_y * (g_syx[i] + coefficient1 * g_tensor_tot[6]) +
               fe_z * (g_szx[i] + coefficient1 * g_tensor_tot[7]);

    g_fy[i] += fe_x * (g_sxy[i] + coefficient1 * g_tensor_tot[3]) +
               fe_y * (g_syy[i] + coefficient1 * g_tensor_tot[1] + coefficient2) +
               fe_z * (g_szy[i] + coefficient1 * g_tensor_tot[8]);

    g_fz[i] += fe_x * (g_sxz[i] + coefficient1 * g_tensor_tot[4]) +
               fe_y * (g_syz[i] + coefficient1 * g_tensor_tot[5]) +
               fe_z * (g_szz[i] + coefficient1 * g_tensor_tot[2] + coefficient2);
  }
}

static __global__ void gpu_add_color_driving_force(
  int N,
  const double* g_coefficient,
  const int* g_type,
  double fe_x,
  double fe_y,
  double fe_z,
  double* g_fx,
  double* g_fy,
  double* g_fz)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double coefficient = g_coefficient[g_type[i]];
    g_fx[i] += fe_x * coefficient;
    g_fy[i] += fe_y * coefficient;
    g_fz[i] += fe_z * coefficient;
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
  const int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  if (bid < 3) {
    for (int batch = 0; batch < number_of_batches; ++batch) {
      const int n = tid + batch * 1024;
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
  } else {
    int element_index = ((bid - NUM_OF_HEAT_COMPONENTS) / 3);
    int component = bid % 3;
    for (int batch = 0; batch < number_of_batches; ++batch) {
      const int n = tid + batch * 1024;
      if (n < N) {
        if (g_type[n] == element_index) {
          s_data[tid] += g_velocity[n + N * component];
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
      g_diffusion[step * (gridDim.x - NUM_OF_HEAT_COMPONENTS) + bid - NUM_OF_HEAT_COMPONENTS] =
        g_mass_type[element_index] * s_data[0];
    }
  }
}

void HNEMDEC::post_force(
  const int step,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (compute == -1)
    return;

  const int N = atom.number_of_atoms;

  if (compute == 0) {
    gpu_find_per_atom_tensor<<<(N - 1) / 128 + 1, 128>>>(
      N,
      atom.mass.data(),
      atom.potential_per_atom.data(),
      atom.velocity_per_atom.data(),
      atom.velocity_per_atom.data() + N,
      atom.velocity_per_atom.data() + 2 * N,
      atom.virial_per_atom.data() + 0 * N,
      atom.virial_per_atom.data() + 3 * N,
      atom.virial_per_atom.data() + 4 * N,
      atom.virial_per_atom.data() + 6 * N,
      atom.virial_per_atom.data() + 1 * N,
      atom.virial_per_atom.data() + 5 * N,
      atom.virial_per_atom.data() + 7 * N,
      atom.virial_per_atom.data() + 8 * N,
      atom.virial_per_atom.data() + 2 * N,
      tensor_per_atom_.data());
    GPU_CHECK_KERNEL

    gpu_sum_tensor<<<9, 1024>>>(N, tensor_per_atom_.data(), tensor_sum_.data());
    GPU_CHECK_KERNEL

    gpu_add_heat_flow_driving_force<<<(N - 1) / 128 + 1, 128>>>(
      N,
      coefficient_.data(),
      atom.type.data(),
      fe_x,
      fe_y,
      fe_z,
      tensor_per_atom_.data() + 0 * N,
      tensor_per_atom_.data() + 3 * N,
      tensor_per_atom_.data() + 4 * N,
      tensor_per_atom_.data() + 6 * N,
      tensor_per_atom_.data() + 1 * N,
      tensor_per_atom_.data() + 5 * N,
      tensor_per_atom_.data() + 7 * N,
      tensor_per_atom_.data() + 8 * N,
      tensor_per_atom_.data() + 2 * N,
      tensor_sum_.data(),
      atom.force_per_atom.data(),
      atom.force_per_atom.data() + N,
      atom.force_per_atom.data() + 2 * N);
    GPU_CHECK_KERNEL
  } else {
    gpu_add_color_driving_force<<<(N - 1) / 128 + 1, 128>>>(
      N,
      coefficient_.data(),
      atom.type.data(),
      fe_x,
      fe_y,
      fe_z,
      atom.force_per_atom.data(),
      atom.force_per_atom.data() + N,
      atom.force_per_atom.data() + 2 * N);
  }
}

void HNEMDEC::end_of_step(
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
  if (compute == -1)
    return;
  const int output_flag = ((step + 1) % output_interval == 0);
  step %= output_interval;

  const int N = atom.number_of_atoms;

  compute_heat(
    atom.mass, 
    atom.potential_per_atom, 
    atom.virial_per_atom, 
    atom.velocity_per_atom, 
    atom.heat_per_atom);

  gpu_sum_heat_and_diffusion<<<NUM_OF_HEAT_COMPONENTS + NUM_OF_DIFFUSION_COMPONENTS, 1024>>>(
    N,
    step,
    atom.type.data(),
    mass_type.data(),
    atom.velocity_per_atom.data(),
    atom.heat_per_atom.data(),
    heat_all.data(),
    diffusion_all.data());
  GPU_CHECK_KERNEL

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

    double volume = box.get_volume();
    double factor1 = 0;
    double factor2 = 0;
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

    FILE* fid = fopen("onsager.out", "a");
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

void HNEMDEC::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature) { compute = -1; }

HNEMDEC::HNEMDEC(const std::vector<std::string>& tokens)
{
  parse(tokens);
  action_name = "compute_hnemdec";
}

void HNEMDEC::parse(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  printf("Compute thermal conductivity using the HNEMD Evans-Cummings method.\n");

  // compute_hnemdec compute output_interval fe_x fe_y fe_z
  if (num_param != 6) {
    PRINT_INPUT_ERROR("compute_hnemdec should have 5 parameters.\n");
  }

  if (!is_valid_int(tokens[1], &compute)) {
    PRINT_INPUT_ERROR("compute for HNEMDEC should be an integer number.\n");
  }

  if (compute < 0) {
    PRINT_INPUT_ERROR(
      "compute for HNEMDEC should be an integer between 0 and number_of_types.\n");
  }

  if (!is_valid_int(tokens[2], &output_interval)) {
    PRINT_INPUT_ERROR("output_interval for HNEMDEC should be an integer number.\n");
  }

  if (output_interval < 1) {
    PRINT_INPUT_ERROR("output_interval for HNEMDEC should be larger than 0.\n");
  }
  if (!is_valid_real(tokens[3], &fe_x)) {
    PRINT_INPUT_ERROR("fe_x for HNEMDEC should be a real number.\n");
  }

  if (!is_valid_real(tokens[4], &fe_y)) {
    PRINT_INPUT_ERROR("fe_y for HNEMDEC should be a real number.\n");
  }

  if (!is_valid_real(tokens[5], &fe_z)) {
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
