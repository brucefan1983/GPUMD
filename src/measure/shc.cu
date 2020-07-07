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
Spectral heat current (SHC) calculations. Referene:
[1] Z. Fan, H. Dong, A. Harju, T. Ala-Nissila, Homogeneous nonequilibrium
molecular dynamics method for heat transport and spectral decomposition
with many-body potentials, Phys. Rev. B 99, 064308 (2019).
------------------------------------------------------------------------------*/

#include "shc.cuh"
#include "utilities/error.cuh"

const int BLOCK_SIZE_SHC = 128;

void SHC::preprocess(const int N, const std::vector<Group>& group)
{
  if (!compute) {
    return;
  }

  num_time_origins = 0;
  if (-1 == group_method) {
    group_size = N;
  } else {
    group_size = group[group_method].cpu_size[group_id];
  }

  vx.resize(group_size * Nc);
  vy.resize(group_size * Nc);
  vz.resize(group_size * Nc);
  sx.resize(group_size);
  sy.resize(group_size);
  sz.resize(group_size);
  ki.resize(Nc, 0.0, Memory_Type::managed);
  ko.resize(Nc, 0.0, Memory_Type::managed);
}

static __global__ void gpu_find_k(
  const int group_size,
  const int correlation_step,
  const double* g_sx,
  const double* g_sy,
  const double* g_sz,
  const double* g_vx,
  const double* g_vy,
  const double* g_vz,
  double* g_ki,
  double* g_ko)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int size_sum = bid * group_size;
  int number_of_rounds = (group_size - 1) / BLOCK_SIZE_SHC + 1;
  __shared__ double s_ki[BLOCK_SIZE_SHC];
  __shared__ double s_ko[BLOCK_SIZE_SHC];
  double ki = 0.0;
  double ko = 0.0;

  for (int round = 0; round < number_of_rounds; ++round) {
    int n = tid + round * BLOCK_SIZE_SHC;
    if (n < group_size) {
      ki += g_sx[n] * g_vx[size_sum + n] + g_sy[n] * g_vy[size_sum + n];
      ko += g_sz[n] * g_vz[size_sum + n];
    }
  }
  s_ki[tid] = ki;
  s_ko[tid] = ko;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_ki[tid] += s_ki[tid + offset];
      s_ko[tid] += s_ko[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (bid <= correlation_step) {
      g_ki[correlation_step - bid] += s_ki[0];
      g_ko[correlation_step - bid] += s_ko[0];
    } else {
      g_ki[correlation_step + gridDim.x - bid] += s_ki[0];
      g_ko[correlation_step + gridDim.x - bid] += s_ko[0];
    }
  }
}

static __global__ void gpu_copy_data(
  const int group_size,
  const int offset,
  const int* g_group_contents,
  double* g_sx_o,
  double* g_sy_o,
  double* g_sz_o,
  double* g_vx_o,
  double* g_vy_o,
  double* g_vz_o,
  const double* g_sx_i,
  const double* g_sy_i,
  const double* g_sz_i,
  const double* g_vx_i,
  const double* g_vy_i,
  const double* g_vz_i)
{
  int n = threadIdx.x + blockIdx.x * blockDim.x;

  if (n < group_size) {
    int m = g_group_contents[offset + n];
    g_sx_o[n] = g_sx_i[m];
    g_sy_o[n] = g_sy_i[m];
    g_sz_o[n] = g_sz_i[m];
    g_vx_o[n] = g_vx_i[m];
    g_vy_o[n] = g_vy_i[m];
    g_vz_o[n] = g_vz_i[m];
  }
}

void SHC::process(
  const int step,
  const std::vector<Group>& group,
  const GPU_Vector<double>& velocity_per_atom,
  const GPU_Vector<double>& virial_per_atom)
{
  if (!compute) {
    return;
  }
  if ((step + 1) % sample_interval != 0) {
    return;
  }
  int sample_step = step / sample_interval; // 0, 1, ..., Nc-1, Nc, Nc+1, ...
  int correlation_step = sample_step % Nc;  // 0, 1, ..., Nc-1, 0, 1, ...
  int offset = correlation_step * group_size;

  const int N = velocity_per_atom.size() / 3;

  const int tensor[3][3] = {0, 3, 4, 6, 1, 5, 7, 8, 2};
  const double* sx_tmp = virial_per_atom.data() + N * tensor[direction][0];
  const double* sy_tmp = virial_per_atom.data() + N * tensor[direction][1];
  const double* sz_tmp = virial_per_atom.data() + N * tensor[direction][2];

  if (-1 == group_method) {
    sx.copy_from_device(sx_tmp);
    sy.copy_from_device(sy_tmp);
    sz.copy_from_device(sz_tmp);
    CHECK(cudaMemcpy(
      vx.data() + offset, velocity_per_atom.data(), sizeof(double) * N, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(
      vy.data() + offset, velocity_per_atom.data() + N, sizeof(double) * N,
      cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(
      vz.data() + offset, velocity_per_atom.data() + N * 2, sizeof(double) * N,
      cudaMemcpyDeviceToDevice));
  } else {
    gpu_copy_data<<<(group_size - 1) / BLOCK_SIZE_SHC + 1, BLOCK_SIZE_SHC>>>(
      group_size, group[group_method].cpu_size_sum[group_id], group[group_method].contents.data(),
      sx.data(), sy.data(), sz.data(), vx.data() + offset, vy.data() + offset, vz.data() + offset,
      sx_tmp, sy_tmp, sz_tmp, velocity_per_atom.data(), velocity_per_atom.data() + N,
      velocity_per_atom.data() + 2 * N);
    CUDA_CHECK_KERNEL
  }

  if (sample_step >= Nc - 1) {
    ++num_time_origins;

    gpu_find_k<<<Nc, BLOCK_SIZE_SHC>>>(
      group_size, correlation_step, sx.data(), sy.data(), sz.data(), vx.data(), vy.data(),
      vz.data(), ki.data(), ko.data());
    CUDA_CHECK_KERNEL
  }
}

void SHC::postprocess(const char* input_dir)
{
  if (!compute) {
    return;
  }

  CHECK(cudaDeviceSynchronize()); // needed for pre-Pascal GPU
  char file_shc[200];
  strcpy(file_shc, input_dir);
  strcat(file_shc, "/shc.out");
  FILE* fid = my_fopen(file_shc, "a");

  for (int nc = 0; nc < Nc; ++nc) {
    fprintf(fid, "%25.15e%25.15e\n", ki[nc] / num_time_origins, ko[nc] / num_time_origins);
  }
  fflush(fid);
  fclose(fid);
}
