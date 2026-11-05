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

#include "nep_common.cuh"
#include "main_nep/dataset.cuh"
#include "nep_parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/nep_utilities.cuh"
#include <algorithm>

void initialize_nep_zbl(const NEP_Parameters& para, NEP_ZBL& zbl)
{
  zbl.use_typewise_cutoff_zbl = para.use_typewise_cutoff_zbl;
  zbl.typewise_cutoff_zbl_factor = para.typewise_cutoff_zbl_factor;
  zbl.enabled = para.enable_zbl;
  zbl.flexible = para.flexible_zbl;
  zbl.rc_inner = para.zbl_rc_inner;
  zbl.rc_outer = para.zbl_rc_outer;
  for (int n = 0; n < para.atomic_numbers.size(); ++n) {
    zbl.atomic_numbers[n] = para.atomic_numbers[n];        // starting from 1
  }
  if (zbl.flexible) {
    zbl.num_types = para.num_types;
    int num_type_zbl = (para.num_types * (para.num_types + 1)) / 2;
    for (int n = 0; n < num_type_zbl * 10; ++n) {
      zbl.para[n] = para.zbl_para[n];
    }
  }
}

static void __global__ find_max_min(const int N, const float* g_q, float* g_q_scaler, float* g_q_scaler_max, float* g_q_scaler_min)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float s_max[1024];
  __shared__ float s_min[1024];
  s_max[tid] = -1000000.0f; // a small number
  s_min[tid] = +1000000.0f; // a large number
  const int stride = 1024;
  const int number_of_rounds = (N - 1) / stride + 1;
  for (int round = 0; round < number_of_rounds; ++round) {
    const int n = round * stride + tid;
    if (n < N) {
      const int m = n + N * bid;
      float q = g_q[m];
      if (q > s_max[tid]) {
        s_max[tid] = q;
      }
      if (q < s_min[tid]) {
        s_min[tid] = q;
      }
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_max[tid] < s_max[tid + offset]) {
        s_max[tid] = s_max[tid + offset];
      }
      if (s_min[tid] > s_min[tid + offset]) {
        s_min[tid] = s_min[tid + offset];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_q_scaler_max[bid] = max(g_q_scaler_max[bid], s_max[0]);
    g_q_scaler_min[bid] = min(g_q_scaler_min[bid], s_min[0]);
    g_q_scaler[bid] = 1.0f / (g_q_scaler_max[bid] - g_q_scaler_min[bid]);
  }
}

static __global__ void zero_force(
  const int N, float* g_fx, float* g_fy, float* g_fz, float* g_vxx, float* g_vyy, float* g_vzz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_fx[n1] = 0.0f;
    g_fy[n1] = 0.0f;
    g_fz[n1] = 0.0f;
    g_vxx[n1] = 0.0f;
    g_vyy[n1] = 0.0f;
    g_vzz[n1] = 0.0f;
  }
}

static __global__ void zero_force_long_range(const int N, float* g_fx, float* g_fy, float* g_fz, float* g_v)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_fx[n1] = 0.0f;
    g_fy[n1] = 0.0f;
    g_fz[n1] = 0.0f;
    for (int d = 0; d < 6; ++d) {
      g_v[n1 + N * d] = 0.0f;
    }
  }
}

static __global__ void find_force_ZBL(
  const int N,
  const NEP_ZBL zbl,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  double* g_pe)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    double s_pe = 0.0;
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1]; // starting from 1
    float pow_zi = pow(float(zi), 0.23f);
    int neighbor_number = g_NN[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2]; // starting from 1
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      if (zbl.flexible) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        float ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        float rc_inner = zbl.rc_inner;
        float rc_outer = zbl.rc_outer;
        if (zbl.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * zbl.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = 0.0f;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};

      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);
      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
      s_pe += f * 0.5;
    }
    g_virial[n1 + N * 0] += s_virial_xx;
    g_virial[n1 + N * 1] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
    g_pe[n1] += s_pe;
  }
}

void update_nep_q_scaler(
  int N, int dim, const float* descriptors,
  float* q_scaler, float* q_scaler_max, float* q_scaler_min)
{
  find_max_min<<<dim, 1024>>>(N, descriptors, q_scaler, q_scaler_max, q_scaler_min);
  GPU_CHECK_KERNEL
}

void output_nep_descriptors(
  const Dataset& dataset,
  int dim,
  int output_mode,
  const std::vector<float>& q_scaler,
  GPU_Vector<float>& descriptors)
{
  FILE* fid_descriptor = my_fopen("descriptor.out", "a");
  std::vector<float> descriptor_cpu(descriptors.size());
  descriptors.copy_to_host(descriptor_cpu.data());
  std::vector<float> q_structure(dim, 0.0f);
  for (int nc = 0; nc < dataset.Nc; ++nc) {
    std::fill(q_structure.begin(), q_structure.end(), 0.0f);
    for (int na = 0; na < dataset.Na_cpu[nc]; ++na) {
      int n = dataset.Na_sum_cpu[nc] + na;
      for (int d = 0; d < dim; ++d) {
        float q = descriptor_cpu[n + d * dataset.N] * q_scaler[d];
        q_structure[d] += q;
        if (output_mode == 2) {
          fprintf(fid_descriptor, "%g ", q);
        }
      }
      if (output_mode == 2) {
        fprintf(fid_descriptor, "\n");
      }
    }
    if (output_mode == 1) {
      for (int d = 0; d < dim; ++d) {
        fprintf(fid_descriptor, "%g ", q_structure[d] / dataset.Na_cpu[nc]);
      }
    }
    if (output_mode == 1) {
      fprintf(fid_descriptor, "\n");
    }
  }
  fclose(fid_descriptor);
}

void zero_nep_outputs(Dataset& dataset)
{
  const int block_size = 32;
  const int grid_size = (dataset.N - 1) / block_size + 1;
  zero_force<<<grid_size, block_size>>>(
    dataset.N,
    dataset.force.data(),
    dataset.force.data() + dataset.N,
    dataset.force.data() + dataset.N * 2,
    dataset.virial.data(),
    dataset.virial.data() + dataset.N,
    dataset.virial.data() + dataset.N * 2);
  GPU_CHECK_KERNEL
}

void add_nep_zbl(const NEP_ZBL& zbl, Dataset& dataset, const NEP_Neighbor& neighbor)
{
  const int block_size = 32;
  const int grid_size = (dataset.N - 1) / block_size + 1;
  find_force_ZBL<<<grid_size, block_size>>>(
    dataset.N,
    zbl,
    dataset.NN_angular_sum.data(),
    dataset.NN_angular.data(),
    neighbor.NL_angular.data(),
    dataset.type.data(),
    neighbor.x12_angular.data(),
    neighbor.y12_angular.data(),
    neighbor.z12_angular.data(),
    dataset.force.data(),
    dataset.force.data() + dataset.N,
    dataset.force.data() + dataset.N * 2,
    dataset.virial.data(),
    dataset.energy.data());
  GPU_CHECK_KERNEL
}

void zero_nep_long_range_outputs(Dataset& dataset)
{
  const int block_size = 32;
  const int grid_size = (dataset.N - 1) / block_size + 1;
  zero_force_long_range<<<grid_size, block_size>>>(
    dataset.N,
    dataset.force.data(),
    dataset.force.data() + dataset.N,
    dataset.force.data() + dataset.N * 2,
    dataset.virial.data());
  GPU_CHECK_KERNEL
}
