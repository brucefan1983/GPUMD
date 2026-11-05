/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "nep_charge_dataset.cuh"
#include "nep_charge_parameters.cuh"
#include "nep_output.cuh"
#include "utilities/common.cuh"

void NEP_Charge_Dataset::construct(
  const Dataset& dataset, const std::vector<NEP_Charge_Labels>& labels)
{
  int N = dataset.N;
  int Nc = dataset.Nc;
  charge.resize(N);
  charge_shifted.resize(N);
  charge_cpu.resize(N);
  charge_ref_cpu.resize(Nc);
  charge_ref_gpu.resize(Nc);
  bec.resize(N * 9);
  bec_cpu.resize(N * 9);
  bec_ref_cpu.resize(N * 9);
  bec_ref_gpu.resize(N * 9);
  has_bec.resize(Nc);
  for (int n = 0; n < Nc; ++n) {
    const auto& label = labels[dataset.structures[n].index_in_file];
    charge_ref_cpu[n] = label.charge;
    has_bec[n] = label.has_bec;
    for (int na = 0; na < dataset.Na_cpu[n]; ++na) {
      for (int d = 0; d < 9; ++d) {
        bec_ref_cpu[dataset.Na_sum_cpu[n] + na + N * d] = label.bec[na * 9 + d];
      }
    }
  }
  charge_ref_gpu.copy_from_host(charge_ref_cpu.data());
  bec_ref_gpu.copy_from_host(bec_ref_cpu.data());
}

static __global__ void gpu_sum_charge_error(
  int* g_Na,
  int* g_Na_sum,
  float* g_charge,
  float* g_charge_ref,
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ float s_charge[];
  s_charge[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    s_charge[tid] += g_charge[n];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_charge[tid] += s_charge[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float diff = (s_charge[0] - g_charge_ref[bid]) / Na;
    error_gpu[bid] = diff * diff;
  }
}

static __global__ void gpu_sum_bec_error(
  const int N,
  int* g_Na,
  int* g_Na_sum,
  float* g_bec,
  float* g_bec_ref,
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int N1 = g_Na_sum[bid];
  int N2 = N1 + g_Na[bid];
  extern __shared__ float s_error[];
  s_error[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    float diff_square = 0.0f;
    for (int d = 0; d < 9; ++d) {
      const float diff = g_bec[n + N * d] - g_bec_ref[n + N * d];
      diff_square += diff * diff;
    }
    s_error[tid] += diff_square;
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_error[tid] += s_error[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    error_gpu[bid] = s_error[0];
  }
}

std::vector<float> NEP_Charge_Dataset::get_rmse_charge(
  NEP_Charge_Parameters& para, Dataset& dataset, int device_id)
{
  std::vector<float> rmse_array(para.num_types + 1, 0.0f);

  CHECK(gpuSetDevice(device_id));

  std::vector<int> count_array(para.num_types + 1, 0);

  int mem = sizeof(float) * dataset.Nc;
  const int block_size = 256;

  gpu_sum_charge_error<<<dataset.Nc, block_size, sizeof(float) * block_size>>>(
    dataset.Na.data(),
    dataset.Na_sum.data(),
    charge.data(),
    charge_ref_gpu.data(),
    dataset.error_gpu.data());
  CHECK(gpuMemcpy(dataset.error_cpu.data(), dataset.error_gpu.data(), mem, gpuMemcpyDeviceToHost));
  for (int n = 0; n < dataset.Nc; ++n) {
    float rmse_temp = dataset.error_cpu[n];
    for (int t = 0; t < para.num_types + 1; ++t) {
      if (dataset.has_type[t * dataset.Nc + n]) {
        rmse_array[t] += rmse_temp;
        count_array[t] += 1;
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / count_array[t]);
    }
  }
  return rmse_array;
}

std::vector<float> NEP_Charge_Dataset::get_rmse_bec(
  NEP_Charge_Parameters& para, Dataset& dataset, int device_id)
{
  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  if (!para.has_bec) {
    return rmse_array;
  }

  CHECK(gpuSetDevice(device_id));

  std::vector<int> count_array(para.num_types + 1, 0);

  int mem = sizeof(float) * dataset.Nc;
  const int block_size = 256;

  gpu_sum_bec_error<<<dataset.Nc, block_size, sizeof(float) * block_size>>>(
    dataset.N,
    dataset.Na.data(),
    dataset.Na_sum.data(),
    bec.data(),
    bec_ref_gpu.data(),
    dataset.error_gpu.data());
  CHECK(gpuMemcpy(dataset.error_cpu.data(), dataset.error_gpu.data(), mem, gpuMemcpyDeviceToHost));
  for (int n = 0; n < dataset.Nc; ++n) {
    if (has_bec[n]) {
      float rmse_temp = dataset.error_cpu[n];
      for (int t = 0; t < para.num_types + 1; ++t) {
        if (dataset.has_type[t * dataset.Nc + n]) {
          rmse_array[t] += rmse_temp / (dataset.Na_cpu[n]);
          count_array[t] += 9;
        }
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / count_array[t]);
    }
  }
  return rmse_array;
}

static void update_charge(
  FILE* fid_charge, Dataset& dataset, NEP_Charge_Dataset& charge_dataset, const int nc)
{
  for (int m = 0; m < dataset.Na_cpu[nc]; ++m) {
    fprintf(fid_charge, "%g\n", charge_dataset.charge_cpu[dataset.Na_sum_cpu[nc] + m]);
  }
}

static void update_bec(
  FILE* fid_bec, Dataset& dataset, NEP_Charge_Dataset& charge_dataset, const int nc)
{
  write_nep_atomic_predictions(
    9, fid_bec, charge_dataset.bec_cpu.data(), charge_dataset.bec_ref_cpu.data(), dataset, nc);
}

void copy_nep_charge_predictions(NEP_Charge_Dataset& data, bool has_bec)
{
  data.charge.copy_to_host(data.charge_cpu.data());
  if (has_bec) {
    data.bec.copy_to_host(data.bec_cpu.data());
  }
}

void write_nep_charge_predictions(
  const std::string& label, const std::vector<std::pair<Dataset*, int>>& structures,
  NEP_Batch_Data<NEP_Charge_Dataset>& charge_data, bool has_bec)
{
  FILE* fid_charge = my_fopen(("charge_" + label + ".out").c_str(), "w");
  FILE* fid_bec = has_bec ? my_fopen(("bec_" + label + ".out").c_str(), "w") : nullptr;
  for (const auto& structure : structures) {
    update_charge(fid_charge, *structure.first, charge_data.get(*structure.first), structure.second);
    if (fid_bec) {
      update_bec(fid_bec, *structure.first, charge_data.get(*structure.first), structure.second);
    }
  }
  fclose(fid_charge);
  if (fid_bec) {
    fclose(fid_bec);
  }
}
