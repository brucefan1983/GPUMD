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

#include "dataset.cuh"
#include "mic.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"

void Dataset::copy_structures(std::vector<Structure>& structures_input)
{
  Nc = structures_input.size();
  structures.resize(Nc);

  for (int n = 0; n < Nc; ++n) {
    structures[n].num_atom = structures_input[n].num_atom;
    structures[n].num_atom_original = structures_input[n].num_atom_original;
    structures[n].has_virial = structures_input[n].has_virial;
    structures[n].energy = structures_input[n].energy;
    for (int k = 0; k < 6; ++k) {
      structures[n].virial[k] = structures_input[n].virial[k];
    }
    for (int k = 0; k < 18; ++k) {
      structures[n].box[k] = structures_input[n].box[k];
    }

    structures[n].atomic_number.resize(structures[n].num_atom);
    structures[n].x.resize(structures[n].num_atom);
    structures[n].y.resize(structures[n].num_atom);
    structures[n].z.resize(structures[n].num_atom);
    structures[n].fx.resize(structures[n].num_atom);
    structures[n].fy.resize(structures[n].num_atom);
    structures[n].fz.resize(structures[n].num_atom);

    for (int na = 0; na < structures[n].num_atom; ++na) {
      structures[n].atomic_number[na] = structures_input[n].atomic_number[na];
      structures[n].x[na] = structures_input[n].x[na];
      structures[n].y[na] = structures_input[n].y[na];
      structures[n].z[na] = structures_input[n].z[na];
      structures[n].fx[na] = structures_input[n].fx[na];
      structures[n].fy[na] = structures_input[n].fy[na];
      structures[n].fz[na] = structures_input[n].fz[na];
    }
  }
}

void Dataset::find_Na()
{
  N = 0;
  max_Na = 0;
  int num_virial_configurations = 0;
  for (int nc = 0; nc < Nc; ++nc) {
    Na[nc] = structures[nc].num_atom;
    Na_sum[nc] = 0;
  }

  for (int nc = 0; nc < Nc; ++nc) {
    N += structures[nc].num_atom;
    if (structures[nc].num_atom > max_Na) {
      max_Na = structures[nc].num_atom;
    }
    num_virial_configurations += structures[nc].has_virial;
  }

  for (int nc = 1; nc < Nc; ++nc) {
    Na_sum[nc] = Na_sum[nc - 1] + Na[nc - 1];
  }

  printf("Total number of atoms = %d.\n", N);
  printf("Number of atoms in the largest configuration = %d.\n", max_Na);
  printf("Number of configurations having virial = %d.\n", num_virial_configurations);
}

void Dataset::initialize_gpu_data(Parameters& para)
{
  type.resize(N, Memory_Type::managed);

  r.resize(N * 3, Memory_Type::managed);
  force.resize(N * 3, 0.0f, Memory_Type::managed);
  force_ref.resize(N * 3, Memory_Type::managed);
  pe.resize(N, 0.0f, Memory_Type::managed);
  virial.resize(N * 6, 0.0f, Memory_Type::managed);

  for (int n = 0; n < Nc; ++n) {
    pe_ref[n] = structures[n].energy;
    for (int k = 0; k < 6; ++k) {
      virial_ref[k * Nc + n] = structures[n].virial[k];
    }
    for (int k = 0; k < 18; ++k) {
      h[k + n * 18] = structures[n].box[k];
    }
    for (int na = 0; na < structures[n].num_atom; ++na) {
      r[Na_sum[n] + na] = structures[n].x[na];
      r[Na_sum[n] + na + N] = structures[n].y[na];
      r[Na_sum[n] + na + N * 2] = structures[n].z[na];
      force_ref[Na_sum[n] + na] = structures[n].fx[na];
      force_ref[Na_sum[n] + na + N] = structures[n].fy[na];
      force_ref[Na_sum[n] + na + N * 2] = structures[n].fz[na];
    }
  }
}

void Dataset::check_types(Parameters& para)
{
  std::vector<int> types;
  for (int nc = 0; nc < Nc; ++nc) {
    for (int na = 0; na < structures[nc].num_atom; ++na) {
      type[Na_sum[nc] + na] = structures[nc].atomic_number[na];
      bool find_a_new_type = true;
      for (int k = 0; k < types.size(); ++k) {
        if (types[k] == structures[nc].atomic_number[na]) {
          find_a_new_type = false;
        }
      }
      if (find_a_new_type) {
        types.emplace_back(structures[nc].atomic_number[na]);
      }
    }
  }
  num_types = types.size();

  if (num_types != para.num_types) {
    PRINT_INPUT_ERROR("mismatching num_types in nep.in and train.in.");
  }
  for (int nc = 0; nc < Nc; ++nc) {
    for (int na = 0; na < structures[nc].num_atom; ++na) {
      if (structures[nc].atomic_number[na] >= num_types) {
        PRINT_INPUT_ERROR("detected atom type (in train.in) >= num_types (in nep.in).");
      }
    }
  }
}

static __global__ void gpu_find_neighbor_number(
  const int N,
  const int* Na,
  const int* Na_sum,
  const float rc2_radial,
  const float rc2_angular,
  const float* __restrict__ box,
  const float* x,
  const float* y,
  const float* z,
  int* NN_radial,
  int* NN_angular)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    const float* __restrict__ h = box + 18 * blockIdx.x;
    float x1 = x[n1];
    float y1 = y[n1];
    float z1 = z[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      if (n2 == n1) {
        continue;
      }
      float x12 = x[n2] - x1;
      float y12 = y[n2] - y1;
      float z12 = z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float distance_square = x12 * x12 + y12 * y12 + z12 * z12;
      if (distance_square < rc2_radial) {
        count_radial++;
      }
      if (distance_square < rc2_angular) {
        count_angular++;
      }
    }
    NN_radial[n1] = count_radial;
    NN_angular[n1] = count_angular;
  }
}

static __global__ void gpu_find_neighbor_list(
  const int N,
  const int* Na,
  const int* Na_sum,
  const float rc2_radial,
  const float rc2_angular,
  const float* __restrict__ box,
  const float* x,
  const float* y,
  const float* z,
  int* NN_radial,
  int* NL_radial,
  int* NN_angular,
  int* NL_angular,
  float* x12_radial,
  float* y12_radial,
  float* z12_radial,
  float* x12_angular,
  float* y12_angular,
  float* z12_angular)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    const float* __restrict__ h = box + 18 * blockIdx.x;
    float x1 = x[n1];
    float y1 = y[n1];
    float z1 = z[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      if (n2 == n1) {
        continue;
      }
      float x12 = x[n2] - x1;
      float y12 = y[n2] - y1;
      float z12 = z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float distance_square = x12 * x12 + y12 * y12 + z12 * z12;
      if (distance_square < rc2_radial) {
        NL_radial[count_radial * N + n1] = n2;
        x12_radial[count_radial * N + n1] = x12;
        y12_radial[count_radial * N + n1] = y12;
        z12_radial[count_radial * N + n1] = z12;
        count_radial++;
      }
      if (distance_square < rc2_angular) {
        NL_angular[count_angular * N + n1] = n2;
        x12_angular[count_angular * N + n1] = x12;
        y12_angular[count_angular * N + n1] = y12;
        z12_angular[count_angular * N + n1] = z12;
        count_angular++;
      }
    }
    NN_radial[n1] = count_radial;
    NN_angular[n1] = count_angular;
  }
}

void Dataset::find_neighbor(Parameters& para)
{
  NN_radial.resize(N, Memory_Type::managed);
  NN_angular.resize(N, Memory_Type::managed);
  float rc2_radial = para.rc_radial * para.rc_radial;
  float rc2_angular = para.rc_angular * para.rc_angular;

  gpu_find_neighbor_number<<<Nc, max_Na>>>(
    N, Na.data(), Na_sum.data(), rc2_radial, rc2_angular, h.data(), r.data(), r.data() + N,
    r.data() + N * 2, NN_radial.data(), NN_angular.data());
  CUDA_CHECK_KERNEL

  CHECK(cudaDeviceSynchronize());
  int min_NN_radial = 10000;
  max_NN_radial = -1;
  for (int n = 0; n < N; ++n) {
    if (NN_radial[n] < min_NN_radial) {
      min_NN_radial = NN_radial[n];
    }
    if (NN_radial[n] > max_NN_radial) {
      max_NN_radial = NN_radial[n];
    }
  }
  int min_NN_angular = 10000;
  max_NN_angular = -1;
  for (int n = 0; n < N; ++n) {
    if (NN_angular[n] < min_NN_angular) {
      min_NN_angular = NN_angular[n];
    }
    if (NN_angular[n] > max_NN_angular) {
      max_NN_angular = NN_angular[n];
    }
  }

  printf("Radial descriptor with a cutoff of %g A:\n", para.rc_radial);
  printf("    Minimum number of neighbors for one atom = %d.\n", min_NN_radial);
  printf("    Maximum number of neighbors for one atom = %d.\n", max_NN_radial);
  printf("Angular descriptor with a cutoff of %g A:\n", para.rc_angular);
  printf("    Minimum number of neighbors for one atom = %d.\n", min_NN_angular);
  printf("    Maximum number of neighbors for one atom = %d.\n", max_NN_angular);

  NL_radial.resize(N * max_NN_radial);
  NL_angular.resize(N * max_NN_angular);
  x12_radial.resize(N * max_NN_radial);
  y12_radial.resize(N * max_NN_radial);
  z12_radial.resize(N * max_NN_radial);
  x12_angular.resize(N * max_NN_angular);
  y12_angular.resize(N * max_NN_angular);
  z12_angular.resize(N * max_NN_angular);

  gpu_find_neighbor_list<<<Nc, max_Na>>>(
    N, Na.data(), Na_sum.data(), rc2_radial, rc2_angular, h.data(), r.data(), r.data() + N,
    r.data() + N * 2, NN_radial.data(), NL_radial.data(), NN_angular.data(), NL_angular.data(),
    x12_radial.data(), y12_radial.data(), z12_radial.data(), x12_angular.data(), y12_angular.data(),
    z12_angular.data());
  CUDA_CHECK_KERNEL
}

void Dataset::construct(char* input_dir, Parameters& para, std::vector<Structure>& structures_input)
{
  CHECK(cudaDeviceSynchronize()); // need this due to unified memory

  copy_structures(structures_input);
  h.resize(Nc * 18, Memory_Type::managed);
  pe_ref.resize(Nc, Memory_Type::managed);
  virial_ref.resize(Nc * 6, Memory_Type::managed);
  Na.resize(Nc, Memory_Type::managed);
  Na_sum.resize(Nc, Memory_Type::managed);
  error_cpu.resize(Nc);
  error_gpu.resize(Nc);

  find_Na();
  initialize_gpu_data(para);
  check_types(para);
  find_neighbor(para);
}

static __global__ void gpu_sum_force_error(
  int N,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_fx_ref,
  float* g_fy_ref,
  float* g_fz_ref,
  float* g_error)
{
  int tid = threadIdx.x;
  int number_of_rounds = (N - 1) / blockDim.x + 1;
  extern __shared__ float s_error[];
  s_error[tid] = 0.0f;
  for (int round = 0; round < number_of_rounds; ++round) {
    int n = tid + round * blockDim.x;
    if (n < N) {
      float dx = g_fx[n] - g_fx_ref[n];
      float dy = g_fy[n] - g_fy_ref[n];
      float dz = g_fz[n] - g_fz_ref[n];
      s_error[tid] += dx * dx + dy * dy + dz * dz;
    }
  }

  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncthreads();
  }

  for (int offset = 32; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncwarp();
  }

  if (tid == 0) {
    g_error[0] = s_error[0];
  }
}

float Dataset::get_rmse_force()
{
  gpu_sum_force_error<<<1, 512, sizeof(float) * 512>>>(
    N, force.data(), force.data() + N, force.data() + N * 2, force_ref.data(), force_ref.data() + N,
    force_ref.data() + N * 2, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), sizeof(float), cudaMemcpyDeviceToHost));
  return sqrt(error_cpu[0] / (N * 3));
}

static __global__ void
gpu_sum_pe_error(int* g_Na, int* g_Na_sum, float* g_pe, float* g_pe_ref, float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int offset = g_Na_sum[bid];
  extern __shared__ float s_pe[];
  s_pe[tid] = 0.0f;
  if (tid < Na) {
    int n = offset + tid; // particle index
    s_pe[tid] += g_pe[n];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
    if (tid < offset) {
      s_pe[tid] += s_pe[tid + offset];
    }
    __syncthreads();
  }

  for (int offset = 32; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_pe[tid] += s_pe[tid + offset];
    }
    __syncwarp();
  }

  if (tid == 0) {
    float diff = s_pe[0] / Na - g_pe_ref[bid];
    error_gpu[bid] = diff * diff;
  }
}

static int get_block_size(int max_num_atom)
{
  int block_size = 64;
  for (int n = 64; n < 1024; n <<= 1) {
    if (max_num_atom > n) {
      block_size = n << 1;
    }
  }
  return block_size;
}

float Dataset::get_rmse_energy()
{
  int block_size = get_block_size(max_Na);
  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), pe.data(), pe_ref.data(), error_gpu.data());
  int mem = sizeof(float) * Nc;
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  float error_ave = 0.0;
  for (int n = 0; n < Nc; ++n) {
    error_ave += error_cpu[n];
  }
  return sqrt(error_ave / Nc);
}

float Dataset::get_rmse_virial()
{
  int num_virial_configurations = 0;
  for (int n = 0; n < Nc; ++n) {
    if (structures[n].has_virial) {
      ++num_virial_configurations;
    }
  }
  if (num_virial_configurations == 0) {
    return 0.0f;
  }

  float error_ave = 0.0;
  int mem = sizeof(float) * Nc;
  int block_size = get_block_size(max_Na);

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data(), virial_ref.data(), error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    if (structures[n].has_virial) {
      error_ave += error_cpu[n];
    }
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N, virial_ref.data() + Nc, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    if (structures[n].has_virial) {
      error_ave += error_cpu[n];
    }
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N * 2, virial_ref.data() + Nc * 2, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    if (structures[n].has_virial) {
      error_ave += error_cpu[n];
    }
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N * 3, virial_ref.data() + Nc * 3, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    if (structures[n].has_virial) {
      error_ave += error_cpu[n];
    }
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N * 4, virial_ref.data() + Nc * 4, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    if (structures[n].has_virial) {
      error_ave += error_cpu[n];
    }
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N * 5, virial_ref.data() + Nc * 5, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    if (structures[n].has_virial) {
      error_ave += error_cpu[n];
    }
  }

  return sqrt(error_ave / (num_virial_configurations * 6));
}
