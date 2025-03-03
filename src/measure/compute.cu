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
Compute block (space) averages of various per-atom quantities.
------------------------------------------------------------------------------*/

#include "compute.cuh"
#include "integrate/integrate.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <vector>

#define DIM 3

void Compute::preprocess(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  number_of_scalars = 0;
  if (compute_temperature)
    number_of_scalars += 1;
  if (compute_potential)
    number_of_scalars += 1;
  if (compute_force)
    number_of_scalars += 3;
  if (compute_virial)
    number_of_scalars += 9;
  if (compute_jp)
    number_of_scalars += 3;
  if (compute_jk)
    number_of_scalars += 3;
  if (compute_momentum)
    number_of_scalars += 3;
  if (number_of_scalars == 0)
    return;

  int number_of_columns = group[grouping_method].number * number_of_scalars;
  cpu_group_sum.resize(number_of_columns, 0.0);
  cpu_group_sum_ave.resize(number_of_columns, 0.0);

  gpu_group_sum.resize(number_of_columns);
  gpu_per_atom_x.resize(atom.number_of_atoms);
  gpu_per_atom_y.resize(atom.number_of_atoms);
  gpu_per_atom_z.resize(atom.number_of_atoms);

  fid = my_fopen("compute.out", "a");
}

void Compute::postprocess(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (number_of_scalars == 0)
    return;
  fclose(fid);

  compute_temperature = 0;
  compute_potential = 0;
  compute_force = 0;
  compute_virial = 0;
  compute_jp = 0;
  compute_jk = 0;
  compute_momentum = 0;
}

static __global__ void find_per_atom_temperature(
  const int N,
  const double* g_mass,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  double* g_temperature)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double vx = g_vx[n];
    double vy = g_vy[n];
    double vz = g_vz[n];
    double ek2 = g_mass[n] * (vx * vx + vy * vy + vz * vz);
    g_temperature[n] = ek2 / (DIM * K_B);
  }
}

static __global__ void find_per_atom_jp(
  const int N,
  const double* sxx,
  const double* sxy,
  const double* sxz,
  const double* syx,
  const double* syy,
  const double* syz,
  const double* szx,
  const double* szy,
  const double* szz,
  const double* vx,
  const double* vy,
  const double* vz,
  double* jx,
  double* jy,
  double* jz)
{
  int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N) {
    jx[n] = sxx[n] * vx[n] + sxy[n] * vy[n] + sxz[n] * vz[n];
    jy[n] = syx[n] * vx[n] + syy[n] * vy[n] + syz[n] * vz[n];
    jz[n] = szx[n] * vx[n] + szy[n] * vy[n] + szz[n] * vz[n];
  }
}

static __global__ void find_per_atom_jk(
  const int N,
  const double* g_potential,
  const double* g_mass,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  double* g_jx,
  double* g_jy,
  double* g_jz)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double potential = g_potential[n];
    double mass = g_mass[n];
    double vx = g_vx[n];
    double vy = g_vy[n];
    double vz = g_vz[n];
    double energy = mass * (vx * vx + vy * vy + vz * vz) * 0.5 + potential;
    g_jx[n] = vx * energy;
    g_jy[n] = vy * energy;
    g_jz[n] = vz * energy;
  }
}

static __global__ void find_per_atom_momentum(
  const int N,
  const double* g_mass,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  double* g_momentumx,
  double* g_momentumy,
  double* g_momentumz)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double mass = g_mass[n];
    g_momentumx[n] = g_vx[n] * mass;
    g_momentumy[n] = g_vy[n] * mass;
    g_momentumz[n] = g_vz[n] * mass;
  }
}

static __global__ void find_group_sum_1(
  const int* g_group_size,
  const int* g_group_size_sum,
  const int* g_group_contents,
  const double* g_in,
  double* g_out)
{
  // <<<number_of_groups, 256>>> (one CUDA block for one group of atoms)
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int group_size = g_group_size[bid];
  int offset = g_group_size_sum[bid];
  int number_of_patches = (group_size - 1) / 256 + 1;
  __shared__ double s_data[256];
  s_data[tid] = 0.0;

  for (int patch = 0; patch < number_of_patches; patch++) {
    int k = tid + patch * 256;
    if (k < group_size) {
      int n = g_group_contents[offset + k]; // particle index
      s_data[tid] += g_in[n];
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
    g_out[bid] = s_data[0];
  }
}

static __global__ void find_group_sum_3(
  const int* g_group_size,
  const int* g_group_size_sum,
  const int* g_group_contents,
  const double* g_fx,
  const double* g_fy,
  const double* g_fz,
  double* g_out)
{
  // <<<number_of_groups, 256>>> (one CUDA block for one group of atoms)
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int group_size = g_group_size[bid];
  int offset = g_group_size_sum[bid];
  int number_of_patches = (group_size - 1) / 256 + 1;
  __shared__ double s_fx[256];
  __shared__ double s_fy[256];
  __shared__ double s_fz[256];
  s_fx[tid] = 0.0;
  s_fy[tid] = 0.0;
  s_fz[tid] = 0.0;

  for (int patch = 0; patch < number_of_patches; patch++) {
    int k = tid + patch * 256;
    if (k < group_size) {
      int n = g_group_contents[offset + k]; // particle index
      s_fx[tid] += g_fx[n];
      s_fy[tid] += g_fy[n];
      s_fz[tid] += g_fz[n];
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_fx[tid] += s_fx[tid + offset];
      s_fy[tid] += s_fy[tid + offset];
      s_fz[tid] += s_fz[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_out[bid] = s_fx[0];
    g_out[bid + gridDim.x] = s_fy[0];
    g_out[bid + gridDim.x * 2] = s_fz[0];
  }
}

static __global__ void find_group_sum_9(
  const int* g_group_size,
  const int* g_group_size_sum,
  const int* g_group_contents,
  const double* g_xx,
  const double* g_xy,
  const double* g_xz,
  const double* g_yx,
  const double* g_yy,
  const double* g_yz,
  const double* g_zx,
  const double* g_zy,
  const double* g_zz,
  double* g_out)
{
  // <<<number_of_groups, 128>>> (one CUDA block for one group of atoms)
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int group_size = g_group_size[bid];
  int offset = g_group_size_sum[bid];
  int number_of_patches = (group_size - 1) / 128 + 1;
  __shared__ double s_xx[128];
  __shared__ double s_xy[128];
  __shared__ double s_xz[128];
  __shared__ double s_yx[128];
  __shared__ double s_yy[128];
  __shared__ double s_yz[128];
  __shared__ double s_zx[128];
  __shared__ double s_zy[128];
  __shared__ double s_zz[128];
  s_xx[tid] = 0.0;
  s_xy[tid] = 0.0;
  s_xz[tid] = 0.0;
  s_yx[tid] = 0.0;
  s_yy[tid] = 0.0;
  s_yz[tid] = 0.0;
  s_zx[tid] = 0.0;
  s_zy[tid] = 0.0;
  s_zz[tid] = 0.0;

  for (int patch = 0; patch < number_of_patches; patch++) {
    int k = tid + patch * 128;
    if (k < group_size) {
      int n = g_group_contents[offset + k]; // particle index
      s_xx[tid] += g_xx[n];
      s_xy[tid] += g_xy[n];
      s_xz[tid] += g_xz[n];
      s_yx[tid] += g_yx[n];
      s_yy[tid] += g_yy[n];
      s_yz[tid] += g_yz[n];
      s_zx[tid] += g_zx[n];
      s_zy[tid] += g_zy[n];
      s_zz[tid] += g_zz[n];
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_xx[tid] += s_xx[tid + offset];
      s_xy[tid] += s_xy[tid + offset];
      s_xz[tid] += s_xz[tid + offset];
      s_yx[tid] += s_yx[tid + offset];
      s_yy[tid] += s_yy[tid + offset];
      s_yz[tid] += s_yz[tid + offset];
      s_zx[tid] += s_zx[tid + offset];
      s_zy[tid] += s_zy[tid + offset];
      s_zz[tid] += s_zz[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_out[bid + gridDim.x * 0] = s_xx[0];
    g_out[bid + gridDim.x * 1] = s_xy[0];
    g_out[bid + gridDim.x * 2] = s_xz[0];
    g_out[bid + gridDim.x * 3] = s_yx[0];
    g_out[bid + gridDim.x * 4] = s_yy[0];
    g_out[bid + gridDim.x * 5] = s_yz[0];
    g_out[bid + gridDim.x * 6] = s_zx[0];
    g_out[bid + gridDim.x * 7] = s_zy[0];
    g_out[bid + gridDim.x * 8] = s_zz[0];
  }
}

void Compute::process(
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
  if (number_of_scalars == 0)
    return;
  if ((step + 1) % sample_interval != 0)
    return;

  int output_flag = (((step + 1) / sample_interval) % output_interval == 0);

  const int Ng = group[grouping_method].number;
  const int N = atom.number_of_atoms;

  int offset = 0;
  if (compute_temperature) {
    find_per_atom_temperature<<<(N - 1) / 256 + 1, 256>>>(
      N,
      atom.mass.data(),
      atom.velocity_per_atom.data(),
      atom.velocity_per_atom.data() + N,
      atom.velocity_per_atom.data() + 2 * N,
      gpu_per_atom_x.data());
    GPU_CHECK_KERNEL
    find_group_sum_1<<<Ng, 256>>>(
      group[grouping_method].size.data(),
      group[grouping_method].size_sum.data(),
      group[grouping_method].contents.data(),
      gpu_per_atom_x.data(),
      gpu_group_sum.data() + offset);
    GPU_CHECK_KERNEL
    offset += Ng;
  }
  if (compute_potential) {
    find_group_sum_1<<<Ng, 256>>>(
      group[grouping_method].size.data(),
      group[grouping_method].size_sum.data(),
      group[grouping_method].contents.data(),
      atom.potential_per_atom.data(),
      gpu_group_sum.data() + offset);
    GPU_CHECK_KERNEL
    offset += Ng;
  }
  if (compute_force) {
    find_group_sum_3<<<Ng, 256>>>(
      group[grouping_method].size.data(),
      group[grouping_method].size_sum.data(),
      group[grouping_method].contents.data(),
      atom.force_per_atom.data(),
      atom.force_per_atom.data() + N,
      atom.force_per_atom.data() + 2 * N,
      gpu_group_sum.data() + offset);
    GPU_CHECK_KERNEL
    offset += Ng * 3;
  }
  if (compute_virial) {
    // the virial tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    find_group_sum_9<<<Ng, 128>>>(
      group[grouping_method].size.data(),
      group[grouping_method].size_sum.data(),
      group[grouping_method].contents.data(),
      atom.virial_per_atom.data() + N * 0,
      atom.virial_per_atom.data() + N * 3,
      atom.virial_per_atom.data() + N * 4,
      atom.virial_per_atom.data() + N * 6,
      atom.virial_per_atom.data() + N * 1,
      atom.virial_per_atom.data() + N * 5,
      atom.virial_per_atom.data() + N * 7,
      atom.virial_per_atom.data() + N * 8,
      atom.virial_per_atom.data() + N * 2,
      gpu_group_sum.data() + offset);
    GPU_CHECK_KERNEL
    offset += Ng * 9;
  }
  if (compute_jp) {
    // the virial tensor:
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    find_per_atom_jp<<<(N - 1) / 128 + 1, 128>>>(
      N,
      atom.virial_per_atom.data(),
      atom.virial_per_atom.data() + N * 3,
      atom.virial_per_atom.data() + N * 4,
      atom.virial_per_atom.data() + N * 6,
      atom.virial_per_atom.data() + N * 1,
      atom.virial_per_atom.data() + N * 5,
      atom.virial_per_atom.data() + N * 7,
      atom.virial_per_atom.data() + N * 8,
      atom.virial_per_atom.data() + N * 2,
      atom.velocity_per_atom.data(),
      atom.velocity_per_atom.data() + N,
      atom.velocity_per_atom.data() + 2 * N,
      gpu_per_atom_x.data(),
      gpu_per_atom_y.data(),
      gpu_per_atom_z.data());
    GPU_CHECK_KERNEL

    find_group_sum_3<<<Ng, 256>>>(
      group[grouping_method].size.data(),
      group[grouping_method].size_sum.data(),
      group[grouping_method].contents.data(),
      gpu_per_atom_x.data(),
      gpu_per_atom_y.data(),
      gpu_per_atom_z.data(),
      gpu_group_sum.data() + offset);
    GPU_CHECK_KERNEL
    offset += Ng * 3;
  }
  if (compute_jk) {
    find_per_atom_jk<<<(N - 1) / 256 + 1, 256>>>(
      N,
      atom.potential_per_atom.data(),
      atom.mass.data(),
      atom.velocity_per_atom.data(),
      atom.velocity_per_atom.data() + N,
      atom.velocity_per_atom.data() + 2 * N,
      gpu_per_atom_x.data(),
      gpu_per_atom_y.data(),
      gpu_per_atom_z.data());
    GPU_CHECK_KERNEL

    find_group_sum_3<<<Ng, 256>>>(
      group[grouping_method].size.data(),
      group[grouping_method].size_sum.data(),
      group[grouping_method].contents.data(),
      gpu_per_atom_x.data(),
      gpu_per_atom_y.data(),
      gpu_per_atom_z.data(),
      gpu_group_sum.data() + offset);
    GPU_CHECK_KERNEL
    offset += Ng * 3;
  }
  if (compute_momentum) {
    find_per_atom_momentum<<<(N - 1) / 256 + 1, 256>>>(
      N,
      atom.mass.data(),
      atom.velocity_per_atom.data(),
      atom.velocity_per_atom.data() + N,
      atom.velocity_per_atom.data() + 2 * N,
      gpu_per_atom_x.data(),
      gpu_per_atom_y.data(),
      gpu_per_atom_z.data());
    GPU_CHECK_KERNEL

    find_group_sum_3<<<Ng, 256>>>(
      group[grouping_method].size.data(),
      group[grouping_method].size_sum.data(),
      group[grouping_method].contents.data(),
      gpu_per_atom_x.data(),
      gpu_per_atom_y.data(),
      gpu_per_atom_z.data(),
      gpu_group_sum.data() + offset);
    GPU_CHECK_KERNEL
    offset += Ng * 3;
  }

  gpu_group_sum.copy_to_host(cpu_group_sum.data());

  for (int n = 0; n < Ng * number_of_scalars; ++n)
    cpu_group_sum_ave[n] += cpu_group_sum[n];

  if (output_flag) {
    output_results(integrate.ensemble->energy_transferred, group);
    for (int n = 0; n < Ng * number_of_scalars; ++n)
      cpu_group_sum_ave[n] = 0.0;
  }
}

void Compute::output_results(const double energy_transferred[], const std::vector<Group>& group)
{
  int Ng = group[grouping_method].number;
  for (int n = 0; n < number_of_scalars; ++n) {
    int offset = n * Ng;
    for (int k = 0; k < Ng; k++) {
      double tmp = cpu_group_sum_ave[k + offset] / output_interval;
      if (compute_temperature && n == 0) {
        tmp /= group[grouping_method].cpu_size[k];
      }
      fprintf(fid, "%15.6e", tmp);
    }
  }

  if (compute_temperature) {
    fprintf(fid, "%15.6e", energy_transferred[0]);
    fprintf(fid, "%15.6e", energy_transferred[1]);
  }

  fprintf(fid, "\n");
  fflush(fid);
}

Compute::Compute(const char** param, int num_param, const std::vector<Group>& group)
{
  parse(param, num_param, group);
  property_name = "compute";
}

void Compute::parse(const char** param, int num_param, const std::vector<Group>& group)
{
  printf("Compute space and/or time average of:\n");
  if (num_param < 5) {
    PRINT_INPUT_ERROR("compute should have at least 4 parameters.");
  }

  // grouping_method
  if (!is_valid_int(param[1], &grouping_method)) {
    PRINT_INPUT_ERROR("grouping method of compute should be integer.");
  }
  if (grouping_method < 0) {
    PRINT_INPUT_ERROR("grouping method should >= 0.");
  }
  if (grouping_method >= group.size()) {
    PRINT_INPUT_ERROR("grouping method exceeds the bound.");
  }

  // sample_interval
  if (!is_valid_int(param[2], &sample_interval)) {
    PRINT_INPUT_ERROR("sampling interval of compute should be integer.");
  }
  if (sample_interval <= 0) {
    PRINT_INPUT_ERROR("sampling interval of compute should > 0.");
  }

  // output_interval
  if (!is_valid_int(param[3], &output_interval)) {
    PRINT_INPUT_ERROR("output interval of compute should be integer.");
  }
  if (output_interval <= 0) {
    PRINT_INPUT_ERROR("output interval of compute should > 0.");
  }

  // temperature potential force virial jp jk (order is not important)
  for (int k = 0; k < num_param - 4; ++k) {
    if (strcmp(param[k + 4], "temperature") == 0) {
      compute_temperature = 1;
      printf("    temperature\n");
    } else if (strcmp(param[k + 4], "potential") == 0) {
      compute_potential = 1;
      printf("    potential energy\n");
    } else if (strcmp(param[k + 4], "force") == 0) {
      compute_force = 1;
      printf("    force\n");
    } else if (strcmp(param[k + 4], "virial") == 0) {
      compute_virial = 1;
      printf("    virial\n");
    } else if (strcmp(param[k + 4], "jp") == 0) {
      compute_jp = 1;
      printf("    potential part of heat current\n");
    } else if (strcmp(param[k + 4], "jk") == 0) {
      compute_jk = 1;
      printf("    kinetic part of heat current\n");
    } else if (strcmp(param[k + 4], "momentum") == 0) {
      compute_momentum = 1;
      printf("    momentum\n");
    } else {
      PRINT_INPUT_ERROR("Invalid property for compute.");
    }
  }

  printf("    using grouping method %d.\n", grouping_method);
  printf("    with sampling interval %d.\n", sample_interval);
  printf("    and output interval %d.\n", output_interval);
}
