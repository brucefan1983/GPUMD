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

#include "tnep_dataset.cuh"
#include "main_nep/common/nep_parameters.cuh"
#include "main_nep/common/nep_output.cuh"
#include "utilities/common.cuh"
#include <stdexcept>

void TNEP_Atomic_Dataset::construct(
  const Dataset& dataset, const std::vector<TNEP_Atomic_Labels>& labels)
{
  const auto& first = labels[dataset.structures[0].index_in_file];
  atomic_virial_diag_only = first.atomic_virial_diag_only;
  for (const auto& structure : dataset.structures) {
    const auto& label = labels[structure.index_in_file];
    if (label.has_atomic_virial != first.has_atomic_virial) {
      throw std::runtime_error("All structures must have the same has_atomic_virial flag.");
    }
    if (label.atomic_virial_diag_only != first.atomic_virial_diag_only) {
      throw std::runtime_error("All structures must have the same atomic_virial_diag_only flag.");
    }
  }
  if (!first.has_atomic_virial) {
    return;
  }
  int N = dataset.N;
  avirial_ref_cpu.resize(N * (atomic_virial_diag_only ? 3 : 6));
  for (int n = 0; n < dataset.Nc; ++n) {
    const auto& label = labels[dataset.structures[n].index_in_file];
    for (int na = 0; na < dataset.Na_cpu[n]; ++na) {
      int index = dataset.Na_sum_cpu[n] + na;
      avirial_ref_cpu[index] = label.avirialxx[na];
      avirial_ref_cpu[index + N] = label.avirialyy[na];
      avirial_ref_cpu[index + N * 2] = label.avirialzz[na];
      if (!atomic_virial_diag_only) {
        avirial_ref_cpu[index + N * 3] = label.avirialxy[na];
        avirial_ref_cpu[index + N * 4] = label.avirialyz[na];
        avirial_ref_cpu[index + N * 5] = label.avirialzx[na];
      }
    }
  }
  avirial_ref_gpu.resize(avirial_ref_cpu.size());
  avirial_ref_gpu.copy_from_host(avirial_ref_cpu.data());
}

static __global__ void gpu_sum_avirial_diag_only_error(
  const int N,
  int* g_Na,
  int* g_Na_sum,
  int* g_type,
  float* g_type_weight,
  float* g_virial,
  float* g_avxx_ref,
  float* g_avyy_ref,
  float* g_avzz_ref,
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int N1 = g_Na_sum[bid];
  int N2 = N1 + g_Na[bid];
  extern __shared__ float s_error[];
  s_error[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    float avxx_ref = g_avxx_ref[n];
    float avyy_ref = g_avyy_ref[n];
    float avzz_ref = g_avzz_ref[n];
    float dxx = g_virial[n] - avxx_ref;
    float dyy = g_virial[1 * N + n] - avyy_ref;
    float dzz = g_virial[2 * N + n] - avzz_ref;
    float diff_square = dxx * dxx + dyy * dyy + dzz * dzz;
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

static __global__ void gpu_sum_avirial_error(
  const int N,
  int* g_Na,
  int* g_Na_sum,
  int* g_type,
  float* g_type_weight,
  float* g_virial,
  float* g_avxx_ref,
  float* g_avyy_ref,
  float* g_avzz_ref,
  float* g_avxy_ref,
  float* g_avyz_ref,
  float* g_avzx_ref,
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int N1 = g_Na_sum[bid];
  int N2 = N1 + g_Na[bid];
  extern __shared__ float s_error[];
  s_error[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    float avxx_ref = g_avxx_ref[n];
    float avyy_ref = g_avyy_ref[n];
    float avzz_ref = g_avzz_ref[n];
    float avxy_ref = g_avxy_ref[n];
    float avyz_ref = g_avyz_ref[n];
    float avzx_ref = g_avzx_ref[n];
    float dxx = g_virial[n] - avxx_ref;
    float dyy = g_virial[1 * N + n] - avyy_ref;
    float dzz = g_virial[2 * N + n] - avzz_ref;
    float dxy = g_virial[3 * N + n] - avxy_ref;
    float dyz = g_virial[4 * N + n] - avyz_ref;
    float dzx = g_virial[5 * N + n] - avzx_ref;
    float diff_square = dxx * dxx + dyy * dyy + dzz * dzz + dxy * dxy + dyz * dyz + dzx * dzx;
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

std::vector<float> TNEP_Atomic_Dataset::get_rmse(
  NEP_Parameters& para, Dataset& dataset, const bool use_weight, int device_id)
{
  CHECK(gpuSetDevice(device_id));
  const int block_size = 256;

  if (atomic_virial_diag_only) {
    gpu_sum_avirial_diag_only_error<<<dataset.Nc, block_size, sizeof(float) * block_size>>>(
      dataset.N,
      dataset.Na.data(),
      dataset.Na_sum.data(),
      dataset.type.data(),
      dataset.type_weight_gpu.data(),
      dataset.virial.data(),
      avirial_ref_gpu.data(),
      avirial_ref_gpu.data() + dataset.N,
      avirial_ref_gpu.data() + dataset.N * 2,
      dataset.error_gpu.data());
  } else {
    gpu_sum_avirial_error<<<dataset.Nc, block_size, sizeof(float) * block_size>>>(
      dataset.N,
      dataset.Na.data(),
      dataset.Na_sum.data(),
      dataset.type.data(),
      dataset.type_weight_gpu.data(),
      dataset.virial.data(),
      avirial_ref_gpu.data(),
      avirial_ref_gpu.data() + dataset.N,
      avirial_ref_gpu.data() + dataset.N * 2,
      avirial_ref_gpu.data() + dataset.N * 3,
      avirial_ref_gpu.data() + dataset.N * 4,
      avirial_ref_gpu.data() + dataset.N * 5,
      dataset.error_gpu.data());
  }
  int mem = sizeof(float) * dataset.Nc;
  CHECK(gpuMemcpy(dataset.error_cpu.data(), dataset.error_gpu.data(), mem, gpuMemcpyDeviceToHost));

  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  std::vector<int> count_array(para.num_types + 1, 0);
  for (int n = 0; n < dataset.Nc; ++n) {
    float rmse_temp = use_weight ?
      dataset.weight_cpu[n] * dataset.weight_cpu[n] * dataset.error_cpu[n] : dataset.error_cpu[n];
    for (int t = 0; t < para.num_types + 1; ++t) {
      if (dataset.has_type[t * dataset.Nc + n]) {
        rmse_array[t] += rmse_temp;
        count_array[t] += dataset.Na_cpu[n];
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / (count_array[t] * 6));
    }
  }
  return rmse_array;
}

static void update_dipole(FILE* fid_dipole, Dataset& dataset,
  TNEP_Atomic_Dataset& atomic_data, bool atomic, const int nc)
{
  if (!atomic) {
    write_nep_structure_predictions(
      false, 3, fid_dipole, dataset.virial_cpu.data(), dataset.virial_ref_cpu.data(), dataset, nc);
  } else {
    write_nep_atomic_predictions(
      3, fid_dipole, dataset.virial_cpu.data(), atomic_data.avirial_ref_cpu.data(), dataset, nc);
  }
}

static void update_polarizability(
  FILE* fid_polarizability, Dataset& dataset,
  TNEP_Atomic_Dataset& atomic_data, bool atomic, const int nc)
{
  if (!atomic) {
    write_nep_structure_predictions(
      false,
      6,
      fid_polarizability,
      dataset.virial_cpu.data(),
      dataset.virial_ref_cpu.data(),
      dataset,
      nc);
  } else {
    write_nep_atomic_predictions(
      6,
      fid_polarizability,
      dataset.virial_cpu.data(),
      atomic_data.avirial_ref_cpu.data(),
      dataset,
      nc);
  }
}

void write_tnep_predictions(const std::string& label,
  const std::vector<std::pair<Dataset*, int>>& structures,
  NEP_Batch_Data<TNEP_Atomic_Dataset>& atomic_data, bool dipole, bool atomic)
{
  FILE* file = my_fopen(
    ((dipole ? "dipole_" : "polarizability_") + label + ".out").c_str(), "w");
  for (const auto& structure : structures) {
    if (dipole) {
      update_dipole(
        file, *structure.first, atomic_data.get(*structure.first), atomic, structure.second);
    } else {
      update_polarizability(
        file, *structure.first, atomic_data.get(*structure.first), atomic, structure.second);
    }
  }
  fclose(file);
}
