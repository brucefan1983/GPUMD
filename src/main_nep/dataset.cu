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

#include "dataset.cuh"
#include "mic.cuh"
#include "main_nep/common/nep_parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include <cstring>
#include <iostream>

void Dataset::copy_structures(std::vector<Structure>& structures_input, int n1, int n2)
{
  Nc = n2 - n1;
  structures.assign(structures_input.begin() + n1, structures_input.begin() + n2);
}

void Dataset::find_has_type(NEP_Parameters& para)
{
  has_type.resize((para.num_types + 1) * Nc, false);
  for (int n = 0; n < Nc; ++n) {
    has_type[para.num_types * Nc + n] = true;
    for (int na = 0; na < structures[n].num_atom; ++na) {
      has_type[structures[n].type[na] * Nc + n] = true;
    }
  }
}

void Dataset::find_Na(NEP_Parameters& para)
{
  Na_cpu.resize(Nc);
  Na_sum_cpu.resize(Nc);

  N = 0;
  max_Na = 0;
  for (int nc = 0; nc < Nc; ++nc) {
    Na_cpu[nc] = structures[nc].num_atom;
    Na_sum_cpu[nc] = 0;
  }

  for (int nc = 0; nc < Nc; ++nc) {
    N += structures[nc].num_atom;
    if (structures[nc].num_atom > max_Na) {
      max_Na = structures[nc].num_atom;
    }
  }

  for (int nc = 1; nc < Nc; ++nc) {
    Na_sum_cpu[nc] = Na_sum_cpu[nc - 1] + Na_cpu[nc - 1];
  }

  printf("Total number of atoms = %d.\n", N);
  printf("Number of atoms in the largest configuration = %d.\n", max_Na);

  Na.resize(Nc);
  Na_sum.resize(Nc);
  Na.copy_from_host(Na_cpu.data());
  Na_sum.copy_from_host(Na_sum_cpu.data());
}

void Dataset::initialize_gpu_data(NEP_Parameters& para)
{
  std::vector<float> box_cpu(Nc * 18);
  std::vector<float> box_original_cpu(Nc * 9);
  std::vector<int> num_cell_cpu(Nc * 3);
  std::vector<int> pbc_cpu(Nc);
  std::vector<float> r_cpu(N * 3);
  std::vector<int> type_cpu(N);

  energy.resize(N);
  virial.resize(N * 6);
  force.resize(N * 3);
  energy_cpu.resize(N);
  virial_cpu.resize(N * 6);
  force_cpu.resize(N * 3);
  weight_cpu.resize(Nc);
  energy_ref_cpu.resize(Nc);
  energy_weight_cpu.resize(Nc);
  virial_ref_cpu.resize(Nc * 6);
  force_ref_cpu.resize(N * 3);

  for (int n = 0; n < Nc; ++n) {
    pbc_cpu[n] = structures[n].pbc;
    weight_cpu[n] = structures[n].weight;
    energy_ref_cpu[n] = structures[n].energy;
    energy_weight_cpu[n] = structures[n].energy_weight;
    for (int k = 0; k < 6; ++k) {
      virial_ref_cpu[k * Nc + n] = structures[n].virial[k];
    }
    for (int k = 0; k < 18; ++k) {
      box_cpu[k + n * 18] = structures[n].box[k];
    }
    for (int k = 0; k < 9; ++k) {
      box_original_cpu[k + n * 9] = structures[n].box_original[k];
    }
    for (int k = 0; k < 3; ++k) {
      num_cell_cpu[k + n * 3] = structures[n].num_cell[k];
    }
    for (int na = 0; na < structures[n].num_atom; ++na) {
      type_cpu[Na_sum_cpu[n] + na] = structures[n].type[na];
      r_cpu[Na_sum_cpu[n] + na] = structures[n].x[na];
      r_cpu[Na_sum_cpu[n] + na + N] = structures[n].y[na];
      r_cpu[Na_sum_cpu[n] + na + N * 2] = structures[n].z[na];
      force_ref_cpu[Na_sum_cpu[n] + na] = structures[n].fx[na];
      force_ref_cpu[Na_sum_cpu[n] + na + N] = structures[n].fy[na];
      force_ref_cpu[Na_sum_cpu[n] + na + N * 2] = structures[n].fz[na];
    }
  }

  type_weight_gpu.resize(NUM_ELEMENTS);
  
  energy_ref_gpu.resize(Nc);
  energy_weight_gpu.resize(Nc);
  virial_ref_gpu.resize(Nc * 6);
  force_ref_gpu.resize(N * 3);
  type_weight_gpu.copy_from_host(para.type_weight_cpu.data());
  energy_ref_gpu.copy_from_host(energy_ref_cpu.data());
  energy_weight_gpu.copy_from_host(energy_weight_cpu.data());
  virial_ref_gpu.copy_from_host(virial_ref_cpu.data());
  force_ref_gpu.copy_from_host(force_ref_cpu.data());

  box.resize(Nc * 18);
  box_original.resize(Nc * 9);
  num_cell.resize(Nc * 3);
  pbc.resize(Nc);
  r.resize(N * 3);
  type.resize(N);
  box.copy_from_host(box_cpu.data());
  box_original.copy_from_host(box_original_cpu.data());
  num_cell.copy_from_host(num_cell_cpu.data());
  pbc.copy_from_host(pbc_cpu.data());
  r.copy_from_host(r_cpu.data());
  type.copy_from_host(type_cpu.data());
}

static __global__ void gpu_find_neighbor_number(
  const int N,
  const int* Na,
  const int* Na_sum,
  const int* g_type,
  const float* g_rc_radial,
  const float* g_rc_angular,
  const float* __restrict__ g_box,
  const float* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
  const int* __restrict__ g_pbc,
  const float* x,
  const float* y,
  const float* z,
  int* NN_radial,
  int* NN_angular)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
    const float* __restrict__ box = g_box + 18 * blockIdx.x;
    const float* __restrict__ box_original = g_box_original + 9 * blockIdx.x;
    const int* __restrict__ num_cell = g_num_cell + 3 * blockIdx.x;
    const int is_periodic = g_pbc[blockIdx.x];
    float x1 = x[n1];
    float y1 = y[n1];
    float z1 = z[n1];
    int t1 = g_type[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      for (int ia = 0; ia < num_cell[0]; ++ia) {
        for (int ib = 0; ib < num_cell[1]; ++ib) {
          for (int ic = 0; ic < num_cell[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }
            float delta_x = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
            float delta_y = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
            float delta_z = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;
            float x12 = x[n2] + delta_x - x1;
            float y12 = y[n2] + delta_y - y1;
            float z12 = z[n2] + delta_z - z1;
            if (is_periodic) {
              dev_apply_mic(box, x12, y12, z12);
            }
            float distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            int t2 = g_type[n2];
            float rc_radial = (g_rc_radial[t1] + g_rc_radial[t2]) * 0.5f;
            float rc_angular = (g_rc_angular[t1] + g_rc_angular[t2]) * 0.5f;
            if (distance_square < rc_radial * rc_radial) {
              count_radial++;
            }
            if (distance_square < rc_angular * rc_angular) {
              count_angular++;
            }
          }
        }
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
  const int* g_type,
  const float* g_rc_radial,
  const float* g_rc_angular,
  const float* __restrict__ g_box,
  const float* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
  const int* __restrict__ g_pbc,
  const float* x,
  const float* y,
  const float* z,
  const int* NN_radial_sum,
  const int* NN_angular_sum,
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
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
    const float* __restrict__ box = g_box + 18 * blockIdx.x;
    const float* __restrict__ box_original = g_box_original + 9 * blockIdx.x;
    const int* __restrict__ num_cell = g_num_cell + 3 * blockIdx.x;
    const int is_periodic = g_pbc[blockIdx.x];
    float x1 = x[n1];
    float y1 = y[n1];
    float z1 = z[n1];
    int t1 = g_type[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      for (int ia = 0; ia < num_cell[0]; ++ia) {
        for (int ib = 0; ib < num_cell[1]; ++ib) {
          for (int ic = 0; ic < num_cell[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }
            float delta_x = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
            float delta_y = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
            float delta_z = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;
            float x12 = x[n2] + delta_x - x1;
            float y12 = y[n2] + delta_y - y1;
            float z12 = z[n2] + delta_z - z1;
            if (is_periodic) {
              dev_apply_mic(box, x12, y12, z12);
            }
            float distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            int t2 = g_type[n2];
            float rc_radial = (g_rc_radial[t1] + g_rc_radial[t2]) * 0.5f;
            float rc_angular = (g_rc_angular[t1] + g_rc_angular[t2]) * 0.5f;
            if (distance_square < rc_radial * rc_radial) {
              int index = NN_radial_sum[n1] + count_radial;
              NL_radial[index] = n2;
              x12_radial[index] = x12;
              y12_radial[index] = y12;
              z12_radial[index] = z12;
              count_radial++;
            }
            if (distance_square < rc_angular * rc_angular) {
              int index = NN_angular_sum[n1] + count_angular;
              NL_angular[index] = n2;
              x12_angular[index] = x12;
              y12_angular[index] = y12;
              z12_angular[index] = z12;
              count_angular++;
            }
          }
        }
      }
    }
    NN_radial[n1] = count_radial;
    NN_angular[n1] = count_angular;
  }
}

void NEP_Neighbor::prepare(NEP_Parameters& para, Dataset& dataset, int device_id)
{
  // Training datasets and their positions are immutable after construction.
  if (neighbor_dataset == &dataset) {
    return;
  }

  CHECK(gpuSetDevice(device_id));

  if (rc_radial.size() == 0) {
    rc_radial.resize(para.rc_radial.size());
    rc_radial.copy_from_host(para.rc_radial.data());
    rc_angular.resize(para.rc_angular.size());
    rc_angular.copy_from_host(para.rc_angular.data());
  }

  if (dataset.total_NN_radial > static_cast<int>(NL_radial.size())) {
    NL_radial.resize(dataset.total_NN_radial);
    x12_radial.resize(dataset.total_NN_radial);
    y12_radial.resize(dataset.total_NN_radial);
    z12_radial.resize(dataset.total_NN_radial);
  }

  if (dataset.total_NN_angular > static_cast<int>(NL_angular.size())) {
    NL_angular.resize(dataset.total_NN_angular);
    x12_angular.resize(dataset.total_NN_angular);
    y12_angular.resize(dataset.total_NN_angular);
    z12_angular.resize(dataset.total_NN_angular);
  }

  gpu_find_neighbor_list<<<dataset.Nc, 256>>>(
    dataset.N,
    dataset.Na.data(),
    dataset.Na_sum.data(),
    dataset.type.data(),
    rc_radial.data(),
    rc_angular.data(),
    dataset.box.data(),
    dataset.box_original.data(),
    dataset.num_cell.data(),
    dataset.pbc.data(),
    dataset.r.data(),
    dataset.r.data() + dataset.N,
    dataset.r.data() + dataset.N * 2,
    dataset.NN_radial_sum.data(),
    dataset.NN_angular_sum.data(),
    dataset.NN_radial.data(),
    NL_radial.data(),
    dataset.NN_angular.data(),
    NL_angular.data(),
    x12_radial.data(),
    y12_radial.data(),
    z12_radial.data(),
    x12_angular.data(),
    y12_angular.data(),
    z12_angular.data());
  GPU_CHECK_KERNEL

  neighbor_dataset = &dataset;
}

void Dataset::find_neighbor(NEP_Parameters& para)
{
  NN_radial.resize(N);
  NN_angular.resize(N);
  std::vector<int> NN_radial_cpu(N);
  std::vector<int> NN_angular_cpu(N);

  GPU_Vector<float> rc_radial(para.rc_radial.size());
  rc_radial.copy_from_host(para.rc_radial.data());
  GPU_Vector<float> rc_angular(para.rc_angular.size());
  rc_angular.copy_from_host(para.rc_angular.data());

  gpu_find_neighbor_number<<<Nc, 256>>>(
    N,
    Na.data(),
    Na_sum.data(),
    type.data(),
    rc_radial.data(),
    rc_angular.data(),
    box.data(),
    box_original.data(),
    num_cell.data(),
    pbc.data(),
    r.data(),
    r.data() + N,
    r.data() + N * 2,
    NN_radial.data(),
    NN_angular.data());
  GPU_CHECK_KERNEL

  NN_radial.copy_to_host(NN_radial_cpu.data());
  NN_angular.copy_to_host(NN_angular_cpu.data());

  total_NN_radial = 0;
  int min_NN_radial = 10000;
  max_NN_radial = -1;
  for (int n = 0; n < N; ++n) {
    total_NN_radial += NN_radial_cpu[n];
    if (NN_radial_cpu[n] < min_NN_radial) {
      min_NN_radial = NN_radial_cpu[n];
    }
    if (NN_radial_cpu[n] > max_NN_radial) {
      max_NN_radial = NN_radial_cpu[n];
    }
  }

  total_NN_angular = 0;
  int min_NN_angular = 10000;
  max_NN_angular = -1;
  for (int n = 0; n < N; ++n) {
    total_NN_angular += NN_angular_cpu[n];
    if (NN_angular_cpu[n] < min_NN_angular) {
      min_NN_angular = NN_angular_cpu[n];
    }
    if (NN_angular_cpu[n] > max_NN_angular) {
      max_NN_angular = NN_angular_cpu[n];
    }
  }

  std::vector<int> NN_radial_sum_cpu(N, 0);
  std::vector<int> NN_angular_sum_cpu(N, 0);
  NN_radial_sum.resize(N);
  NN_angular_sum.resize(N);
  for (int n = 1; n < N; ++n) {
    NN_radial_sum_cpu[n] = NN_radial_sum_cpu[n - 1] + NN_radial_cpu[n - 1];
    NN_angular_sum_cpu[n] = NN_angular_sum_cpu[n - 1] + NN_angular_cpu[n - 1];
  }
  NN_radial_sum.copy_from_host(NN_radial_sum_cpu.data());
  NN_angular_sum.copy_from_host(NN_angular_sum_cpu.data());

  printf("Radial descriptor with a cutoff of %g A:\n", para.rc_radial_max);
  printf("    Minimum number of neighbors for one atom = %d.\n", min_NN_radial);
  printf("    Maximum number of neighbors for one atom = %d.\n", max_NN_radial);
  printf("    Average number of neighbors for one atom = %g.\n", total_NN_radial / float(N));
  printf("Angular descriptor with a cutoff of %g A:\n", para.rc_angular_max);
  printf("    Minimum number of neighbors for one atom = %d.\n", min_NN_angular);
  printf("    Maximum number of neighbors for one atom = %d.\n", max_NN_angular);
  printf("    Average number of neighbors for one atom = %g.\n", total_NN_angular / float(N));
#ifdef OUTPUT_NEIGHBOR_FOR_TRAIN
  FILE* fid = fopen("neighbor.txt", "a");
  for (int nc = 0; nc < Nc; ++nc) {
    int max_NN_radial = -1;
    int max_NN_angular = -1;
    for (int n = Na_sum_cpu[nc]; n < Na_sum_cpu[nc] + Na_cpu[nc]; ++n) {
      if (max_NN_radial < NN_radial_cpu[n]) {
        max_NN_radial = NN_radial_cpu[n];
      }
      if (max_NN_angular < NN_angular_cpu[n]) {
        max_NN_angular = NN_angular_cpu[n];
      }
    }
    fprintf(fid, "%d %d\n", max_NN_radial, max_NN_angular);
  }
  fclose(fid);
#endif
}

void Dataset::construct(
  NEP_Parameters& para, std::vector<Structure>& structures_input,
  int n1, int n2, int device_id, int batch_index)
{
  this->device_id = device_id;
  this->batch_index = batch_index;
  CHECK(gpuSetDevice(device_id));
  copy_structures(structures_input, n1, n2);
  find_has_type(para);
  energy_error_cpu.resize(Nc);
  energy_error_gpu.resize(Nc);
  error_cpu.resize(Nc);
  error_gpu.resize(Nc);

  find_Na(para);
  initialize_gpu_data(para);
  find_neighbor(para);
}

static __global__ void gpu_sum_force_error(
  bool use_weight,
  float force_delta,
  int* g_Na,
  int* g_Na_sum,
  int* g_type,
  float* g_type_weight,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_fx_ref,
  float* g_fy_ref,
  float* g_fz_ref,
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int N1 = g_Na_sum[bid];
  int N2 = N1 + g_Na[bid];
  extern __shared__ float s_error[];
  s_error[tid] = 0.0f;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    float fx_ref = g_fx_ref[n];
    float fy_ref = g_fy_ref[n];
    float fz_ref = g_fz_ref[n];
    float dx = g_fx[n] - fx_ref;
    float dy = g_fy[n] - fy_ref;
    float dz = g_fz[n] - fz_ref;
    float diff_square = dx * dx + dy * dy + dz * dz;
    if (use_weight) {
      float type_weight = g_type_weight[g_type[n]];
      diff_square *= type_weight * type_weight;
    }
    if (use_weight && force_delta > 0.0f) {
      float force_magnitude = sqrt(fx_ref * fx_ref + fy_ref * fy_ref + fz_ref * fz_ref);
      diff_square *= force_delta / (force_delta + force_magnitude);
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

std::vector<float> Dataset::get_rmse_force(NEP_Parameters& para, const bool use_weight, int device_id)
{
  CHECK(gpuSetDevice(device_id));
  const int block_size = 256;
  gpu_sum_force_error<<<Nc, block_size, sizeof(float) * block_size>>>(
    use_weight,
    para.force_delta,
    Na.data(),
    Na_sum.data(),
    type.data(),
    type_weight_gpu.data(),
    force.data(),
    force.data() + N,
    force.data() + N * 2,
    force_ref_gpu.data(),
    force_ref_gpu.data() + N,
    force_ref_gpu.data() + N * 2,
    error_gpu.data());
  int mem = sizeof(float) * Nc;
  CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));

  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  std::vector<int> count_array(para.num_types + 1, 0);
  for (int n = 0; n < Nc; ++n) {
    float rmse_temp = use_weight ? weight_cpu[n] * weight_cpu[n] * error_cpu[n] : error_cpu[n];
    for (int t = 0; t < para.num_types + 1; ++t) {
      if (has_type[t * Nc + n]) {
        rmse_array[t] += rmse_temp;
        count_array[t] += Na_cpu[n];
      }
    }
  }

  for (int t = 0; t <= para.num_types; ++t) {
    if (count_array[t] > 0) {
      rmse_array[t] = sqrt(rmse_array[t] / (count_array[t] * 3));
    }
  }
  return rmse_array;
}

static __global__ void
gpu_get_energy_shift(
  const int* g_Na,
  const int* g_Na_sum,
  const double* g_pe,
  const double* g_pe_ref,
  const float* g_pe_weight,
  double* g_energy_shift)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ double s_pe[];
  s_pe[tid] = 0.0;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    s_pe[tid] += g_pe[n];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_pe[tid] += s_pe[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    double diff = s_pe[0] / Na - g_pe_ref[bid];
    g_energy_shift[bid] = diff * g_pe_weight[bid];
  }
}

static __global__ void gpu_sum_pe_error(
  double energy_shift,
  const int* g_Na,
  const int* g_Na_sum,
  const double* g_pe,
  const double* g_pe_ref,
  const float* g_pe_weight,
  double* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ double s_pe[];
  s_pe[tid] = 0.0;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    s_pe[tid] += g_pe[n];
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_pe[tid] += s_pe[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    double diff = s_pe[0] / Na - g_pe_ref[bid] - energy_shift;
    error_gpu[bid] = diff * diff * g_pe_weight[bid];
  }
}

std::vector<double> Dataset::get_rmse_energy(
  NEP_Parameters& para,
  double& energy_shift_per_structure,
  const bool use_weight,
  const bool do_shift,
  int device_id)
{
  CHECK(gpuSetDevice(device_id));
  energy_shift_per_structure = 0.0;

  const int block_size = 256;
  int mem = sizeof(double) * Nc;

  if (do_shift) {
    gpu_get_energy_shift<<<Nc, block_size, sizeof(double) * block_size>>>(
      Na.data(),
      Na_sum.data(),
      energy.data(),
      energy_ref_gpu.data(),
      energy_weight_gpu.data(),
      energy_error_gpu.data());
    CHECK(gpuMemcpy(energy_error_cpu.data(), energy_error_gpu.data(), mem, gpuMemcpyDeviceToHost));
    double Nc_with_weight = 0.0;
    for (int n = 0; n < Nc; ++n) {
      Nc_with_weight += energy_weight_cpu[n];
      energy_shift_per_structure += energy_error_cpu[n];
    }
    if (Nc_with_weight > 0.0) {
      energy_shift_per_structure /= Nc_with_weight;
    }
  }

  gpu_sum_pe_error<<<Nc, block_size, sizeof(double) * block_size>>>(
    energy_shift_per_structure,
    Na.data(),
    Na_sum.data(),
    energy.data(),
    energy_ref_gpu.data(),
    energy_weight_gpu.data(),
    energy_error_gpu.data());
  CHECK(gpuMemcpy(energy_error_cpu.data(), energy_error_gpu.data(), mem, gpuMemcpyDeviceToHost));

  std::vector<double> rmse_array(para.num_types + 1, 0.0);
  std::vector<int> count_array(para.num_types + 1, 0);
  for (int n = 0; n < Nc; ++n) {
    double rmse_temp = energy_error_cpu[n];
    if (use_weight) {
      rmse_temp *= double(weight_cpu[n]) * weight_cpu[n];
    }
    for (int t = 0; t < para.num_types + 1; ++t) {
      if (has_type[t * Nc + n]) {
        rmse_array[t] += rmse_temp;
        ++count_array[t];
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

static __global__ void gpu_sum_virial_error(
  const int N,
  const float shear_weight,
  int* g_Na,
  int* g_Na_sum,
  float* g_virial,
  float* g_virial_ref,
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ float s_virial[];
  for (int d = 0; d < 6; ++d) {
    s_virial[d * blockDim.x + tid] = 0.0f;
  }

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    for (int d = 0; d < 6; ++d) {
      s_virial[d * blockDim.x + tid] += g_virial[d * N + n];
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      for (int d = 0; d < 6; ++d) {
        s_virial[d * blockDim.x + tid] += s_virial[d * blockDim.x + tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    float error_sum = 0.0f;
    for (int d = 0; d < 6; ++d) {
      float diff = s_virial[d * blockDim.x + 0] / Na - g_virial_ref[d * gridDim.x + bid];
      error_sum += (d >= 3) ? (shear_weight * diff * diff) : (diff * diff);
    }
    error_gpu[bid] = error_sum;
  }
}

std::vector<float> Dataset::get_rmse_virial(
  NEP_Parameters& para, const bool use_weight, int device_id, int num_components)
{
  CHECK(gpuSetDevice(device_id));

  std::vector<float> rmse_array(para.num_types + 1, 0.0f);
  std::vector<int> count_array(para.num_types + 1, 0);

  int mem = sizeof(float) * Nc;
  const int block_size = 256;

  float shear_weight =
    (num_components == 6) ? (use_weight ? para.lambda_shear * para.lambda_shear : 1.0f) : 0.0f;
  gpu_sum_virial_error<<<Nc, block_size, sizeof(float) * block_size * 6>>>(
    N,
    shear_weight,
    Na.data(),
    Na_sum.data(),
    virial.data(),
    virial_ref_gpu.data(),
    error_gpu.data());
  CHECK(gpuMemcpy(error_cpu.data(), error_gpu.data(), mem, gpuMemcpyDeviceToHost));
  for (int n = 0; n < Nc; ++n) {
    if (structures[n].has_virial) {
      float rmse_temp = use_weight ? weight_cpu[n] * weight_cpu[n] * error_cpu[n] : error_cpu[n];
      for (int t = 0; t < para.num_types + 1; ++t) {
        if (has_type[t * Nc + n]) {
          rmse_array[t] += rmse_temp;
          count_array[t] += num_components;
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
