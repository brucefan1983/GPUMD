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

static void get_inverse(float* cpu_h)
{
  cpu_h[9] = cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7];
  cpu_h[10] = cpu_h[2] * cpu_h[7] - cpu_h[1] * cpu_h[8];
  cpu_h[11] = cpu_h[1] * cpu_h[5] - cpu_h[2] * cpu_h[4];
  cpu_h[12] = cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8];
  cpu_h[13] = cpu_h[0] * cpu_h[8] - cpu_h[2] * cpu_h[6];
  cpu_h[14] = cpu_h[2] * cpu_h[3] - cpu_h[0] * cpu_h[5];
  cpu_h[15] = cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6];
  cpu_h[16] = cpu_h[1] * cpu_h[6] - cpu_h[0] * cpu_h[7];
  cpu_h[17] = cpu_h[0] * cpu_h[4] - cpu_h[1] * cpu_h[3];
  float volume = cpu_h[0] * (cpu_h[4] * cpu_h[8] - cpu_h[5] * cpu_h[7]) +
                 cpu_h[1] * (cpu_h[5] * cpu_h[6] - cpu_h[3] * cpu_h[8]) +
                 cpu_h[2] * (cpu_h[3] * cpu_h[7] - cpu_h[4] * cpu_h[6]);
  for (int n = 9; n < 18; n++) {
    cpu_h[n] /= volume;
  }
}

static void transpose(const int n, const float* h_tmp, float* h)
{
  h[0 + 18 * n] = h_tmp[0];
  h[3 + 18 * n] = h_tmp[1];
  h[6 + 18 * n] = h_tmp[2];
  h[1 + 18 * n] = h_tmp[3];
  h[4 + 18 * n] = h_tmp[4];
  h[7 + 18 * n] = h_tmp[5];
  h[2 + 18 * n] = h_tmp[6];
  h[5 + 18 * n] = h_tmp[7];
  h[8 + 18 * n] = h_tmp[8];
}

void Dataset::read_train_in(char* input_dir, Parameters& para)
{
  print_line_1();
  printf("Started reading train.in.\n");
  print_line_2();

  char file_train[200];
  strcpy(file_train, input_dir);
  strcat(file_train, "/train.in");
  FILE* fid = my_fopen(file_train, "r");

  // get Nc
  read_Nc(fid, para);
  h.resize(Nc * 18, Memory_Type::managed);
  pe_ref.resize(Nc, Memory_Type::managed);
  virial_ref.resize(Nc * 6, Memory_Type::managed);
  Na.resize(Nc, Memory_Type::managed);
  Na_sum.resize(Nc, Memory_Type::managed);
  has_virial.resize(Nc);
  error_cpu.resize(Nc);
  error_gpu.resize(Nc);

  read_Na(fid);
  atomic_number.resize(N, Memory_Type::managed);
  r.resize(N * 3, Memory_Type::managed);
  force.resize(N * 3, 0.0f, Memory_Type::managed);
  force_ref.resize(N * 3, Memory_Type::managed);
  pe.resize(N, 0.0f, Memory_Type::managed);
  virial.resize(N * 6, 0.0f, Memory_Type::managed);

  int atomic_number_max = 0;

  for (int n = 0; n < Nc; ++n) {
    int count;

    // energy, virial
    if (has_virial[n]) {
      count = fscanf(
        fid, "%f%f%f%f%f%f%f", &pe_ref[n], &virial_ref[n + Nc * 0], &virial_ref[n + Nc * 1],
        &virial_ref[n + Nc * 2], &virial_ref[n + Nc * 3], &virial_ref[n + Nc * 4],
        &virial_ref[n + Nc * 5]);
      PRINT_SCANF_ERROR(count, 7, "reading error for train.in.");
      for (int k = 0; k < 6; ++k) {
        virial_ref[n + Nc * k] /= Na[n];
      }
    } else {
      count = fscanf(fid, "%f", &pe_ref[n]);
      PRINT_SCANF_ERROR(count, 1, "reading error for train.in.");
    }
    pe_ref[n] /= Na[n];

    // box (ax, ay, az, bx, by, bz, cx, cy, cz)
    float h_tmp[9];
    for (int k = 0; k < 9; ++k) {
      count = fscanf(fid, "%f", &h_tmp[k]);
      PRINT_SCANF_ERROR(count, 1, "reading error for train.in.");
    }
    transpose(n, h_tmp, h.data());
    get_inverse(h.data() + 18 * n);

    // atomic number, position, force
    for (int k = 0; k < Na[n]; ++k) {
      int atomic_number_tmp = 0;
      count = fscanf(
        fid, "%d%f%f%f%f%f%f", &atomic_number_tmp, &r[Na_sum[n] + k], &r[Na_sum[n] + k + N],
        &r[Na_sum[n] + k + N * 2], &force_ref[Na_sum[n] + k], &force_ref[Na_sum[n] + k + N],
        &force_ref[Na_sum[n] + k + N * 2]);
      PRINT_SCANF_ERROR(count, 7, "reading error for train.in.");
      if (atomic_number_tmp < 1) {
        PRINT_INPUT_ERROR("Atomic number should > 0.\n");
      } else {
        atomic_number[Na_sum[n] + k] = atomic_number_tmp;
        if (atomic_number_tmp > atomic_number_max) {
          atomic_number_max = atomic_number_tmp;
        }
      }
    }
  }

  fclose(fid);

  // normalize the atomic number by the largest one
  for (int n = 0; n < N; ++n) {
    atomic_number[n] /= atomic_number_max;
  }
}

void Dataset::read_Nc(FILE* fid, Parameters& para)
{
  int count = fscanf(fid, "%d", &Nc);
  PRINT_SCANF_ERROR(count, 1, "reading error for xyz.in.");
  if (Nc - para.test_set_size < 100) {
    PRINT_INPUT_ERROR("Number of configurations minus test set size should >= 100");
  }
  if (Nc > 100000) {
    PRINT_INPUT_ERROR("Number of configurations should <= 100000");
  }
  printf("Number of configurations = %d.\n", Nc);
}

void Dataset::read_Na(FILE* fid)
{
  N = 0;
  max_Na = 0;
  int num_virial_configurations = 0;
  for (int nc = 0; nc < Nc; ++nc) {
    Na_sum[nc] = 0;
  }

  for (int nc = 0; nc < Nc; ++nc) {
    int count = fscanf(fid, "%d%d", &Na[nc], &has_virial[nc]);
    PRINT_SCANF_ERROR(count, 2, "reading error for train.in.");
    N += Na[nc];
    if (Na[nc] > max_Na) {
      max_Na = Na[nc];
    }
    if (Na[nc] < 2) {
      PRINT_INPUT_ERROR("Number of atoms for one configuration should >= 2.");
    }
    if (Na[nc] > 1024) {
      PRINT_INPUT_ERROR("Number of atoms for one configuration should <=1024.");
    }
    num_virial_configurations += has_virial[nc];
  }

  for (int nc = 1; nc < Nc; ++nc) {
    Na_sum[nc] = Na_sum[nc - 1] + Na[nc - 1];
  }

  printf("Total number of atoms = %d.\n", N);
  printf("Number of atoms in the largest configuration = %d.\n", max_Na);
  printf("Number of configurations having virial = %d.\n", num_virial_configurations);
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

float Dataset::get_rmse_force(const int n1, const int n2)
{
  gpu_sum_force_error<<<1, 512, sizeof(float) * 512>>>(
    n2 - n1, force.data() + n1, force.data() + N + n1, force.data() + N * 2 + n1,
    force_ref.data() + n1, force_ref.data() + N + n1, force_ref.data() + N * 2 + n1,
    error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), sizeof(float), cudaMemcpyDeviceToHost));
  return sqrt(error_cpu[0] / ((n2 - n1) * 3));
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

float Dataset::get_rmse_energy(const int nc1, const int nc2)
{
  int block_size = get_block_size(max_Na);
  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), pe.data(), pe_ref.data(), error_gpu.data());
  int mem = sizeof(float) * Nc;
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  float error_ave = 0.0;
  for (int n = nc1; n < nc2; ++n) {
    error_ave += error_cpu[n];
  }
  return sqrt(error_ave / (nc2 - nc1));
}

float Dataset::get_rmse_virial(const int nc1, const int nc2)
{
  int num_virial_configurations = 0;
  for (int n = nc1; n < nc2; ++n) {
    if (has_virial[n]) {
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
  for (int n = nc1; n < nc2; ++n) {
    if (has_virial[n]) {
      error_ave += error_cpu[n];
    }
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N, virial_ref.data() + Nc, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = nc1; n < nc2; ++n) {
    if (has_virial[n]) {
      error_ave += error_cpu[n];
    }
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N * 2, virial_ref.data() + Nc * 2, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = nc1; n < nc2; ++n) {
    if (has_virial[n]) {
      error_ave += error_cpu[n];
    }
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N * 3, virial_ref.data() + Nc * 3, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = nc1; n < nc2; ++n) {
    if (has_virial[n]) {
      error_ave += error_cpu[n];
    }
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N * 4, virial_ref.data() + Nc * 4, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = nc1; n < nc2; ++n) {
    if (has_virial[n]) {
      error_ave += error_cpu[n];
    }
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    Na.data(), Na_sum.data(), virial.data() + N * 5, virial_ref.data() + Nc * 5, error_gpu.data());
  CHECK(cudaMemcpy(error_cpu.data(), error_gpu.data(), mem, cudaMemcpyDeviceToHost));
  for (int n = nc1; n < nc2; ++n) {
    if (has_virial[n]) {
      error_ave += error_cpu[n];
    }
  }

  return sqrt(error_ave / (num_virial_configurations * 6));
}

static __global__ void gpu_find_neighbor(
  int N,
  int* Na,
  int* Na_sum,
  float cutoff_square,
  const float* __restrict__ box,
  int* NN,
  int* NL,
  float* x,
  float* y,
  float* z)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    const float* __restrict__ h = box + 18 * blockIdx.x;
    float x1 = x[n1];
    float y1 = y[n1];
    float z1 = z[n1];
    int count = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      if (n2 == n1) {
        continue;
      }
      float x12 = x[n2] - x1;
      float y12 = y[n2] - y1;
      float z12 = z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float distance_square = x12 * x12 + y12 * y12 + z12 * z12;
      if (distance_square < cutoff_square) {
        NL[count++ * N + n1] = n2;
      }
    }
    NN[n1] = count;
  }
}

void Dataset::find_neighbor(Parameters& para)
{
  NN.resize(N, Memory_Type::managed);
  NL.resize(N * max_Na, Memory_Type::managed);
  float rc2 = para.rc * para.rc;
  gpu_find_neighbor<<<Nc, max_Na>>>(
    N, Na.data(), Na_sum.data(), rc2, h.data(), NN.data(), NL.data(), r.data(), r.data() + N,
    r.data() + N * 2);
  CUDA_CHECK_KERNEL

  CHECK(cudaDeviceSynchronize());
  int min_NN = 10000, max_NN = -1;
  for (int n = 0; n < N; ++n) {
    if (NN[n] < min_NN) {
      min_NN = NN[n];
    }
    if (NN[n] > max_NN) {
      max_NN = NN[n];
    }
  }
  printf("Minimum number of neighbors for one atom = %d.\n", min_NN);
  printf("Maximum number of neighbors for one atom = %d.\n", max_NN);
}

void Dataset::make_train_or_test_set(
  Parameters& para, int num, int offset, std::vector<int>& configuration_id, Dataset& train_set)
{
  // get the number of configurations
  train_set.Nc = num;

  // allocate per-configuration memory
  train_set.Na.resize(train_set.Nc, Memory_Type::managed);
  train_set.Na_sum.resize(train_set.Nc, Memory_Type::managed);
  train_set.has_virial.resize(train_set.Nc);
  train_set.h.resize(train_set.Nc * 18, Memory_Type::managed);
  train_set.pe_ref.resize(train_set.Nc, Memory_Type::managed);
  train_set.virial_ref.resize(train_set.Nc * 6, Memory_Type::managed);
  train_set.error_cpu.resize(train_set.Nc);
  train_set.error_gpu.resize(train_set.Nc);

  // determine the number of atoms and per-configuration quantities
  train_set.N = 0;
  train_set.max_Na = -1;
  for (int nc = 0; nc < train_set.Nc; ++nc) {
    int nc_global = configuration_id[nc + offset];
    train_set.Na[nc] = Na[nc_global];
    train_set.N += train_set.Na[nc];
    if (train_set.Na[nc] > train_set.max_Na) {
      train_set.max_Na = train_set.Na[nc];
    }
    train_set.Na_sum[nc] = 0;
    train_set.has_virial[nc] = has_virial[nc_global];
    for (int i = 0; i < 18; ++i) {
      train_set.h[nc * 18 + i] = h[nc_global * 18 + i];
    }
    train_set.pe_ref[nc] = pe_ref[nc_global];
    for (int i = 0; i < 6; ++i) {
      train_set.virial_ref[nc + i * train_set.Nc] = virial_ref[nc_global + i * Nc];
    }
  }
  for (int nc = 1; nc < train_set.Nc; ++nc) {
    train_set.Na_sum[nc] = train_set.Na_sum[nc - 1] + train_set.Na[nc - 1];
  }

  // allocate per-atom memory
  train_set.atomic_number.resize(train_set.N, Memory_Type::managed);
  train_set.r.resize(train_set.N * 3, Memory_Type::managed);
  train_set.force.resize(train_set.N * 3, Memory_Type::managed);
  train_set.pe.resize(train_set.N, Memory_Type::managed);
  train_set.virial.resize(train_set.N * 6, Memory_Type::managed);
  train_set.force_ref.resize(train_set.N * 3, Memory_Type::managed);

  // copy data
  for (int nc = 0; nc < train_set.Nc; ++nc) {
    int nc_global = configuration_id[nc + offset];
    for (int i = 0; i < train_set.Na[nc]; ++i) {
      int index = train_set.Na_sum[nc] + i;
      int index_global = Na_sum[nc_global] + i;
      train_set.atomic_number[index] = atomic_number[index_global];
      train_set.pe[index] = pe[index_global];
      train_set.r[index] = r[index_global];
      train_set.r[index + train_set.N] = r[index_global + N];
      train_set.r[index + train_set.N * 2] = r[index_global + N * 2];
      train_set.force[index] = force[index_global];
      train_set.force[index + train_set.N] = force[index_global + N];
      train_set.force[index + train_set.N * 2] = force[index_global + N * 2];
      train_set.force_ref[index] = force_ref[index_global];
      train_set.force_ref[index + train_set.N] = force_ref[index_global + N];
      train_set.force_ref[index + train_set.N * 2] = force_ref[index_global + N * 2];
      train_set.virial[index] = virial[index_global];
      train_set.virial[index + train_set.N] = virial[index_global + N];
      train_set.virial[index + train_set.N * 2] = virial[index_global + N * 2];
      train_set.virial[index + train_set.N * 3] = virial[index_global + N * 3];
      train_set.virial[index + train_set.N * 4] = virial[index_global + N * 4];
      train_set.virial[index + train_set.N * 5] = virial[index_global + N * 5];
    }
  }

  train_set.find_neighbor(para);
}
