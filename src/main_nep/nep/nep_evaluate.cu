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

#include "nep_evaluate.cuh"
#include "main_nep/dataset.cuh"
#include "main_nep/common/nep_compile.cuh"
#include "utilities/error.cuh"
#include "utilities/nep_utilities.cuh"

void bind_nep_parameters(int num_types, const float* parameters, NEP_ANN& ann)
{
  const float* pointer = parameters;
  for (int t = 0; t < num_types; ++t) {
    ann.wb[t] = pointer;
    pointer += ann.one_ann_no_bias;
  }
  ann.b = pointer;
  pointer += 1;
  ann.c = pointer;
}

static __global__ void find_descriptors_radial(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_Descriptor_Parameters paramb,
  const float* c,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_NUM_N] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      float x12 = g_x12[index];
      float y12 = g_y12[index];
      float z12 = g_z12[index];
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);

      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
#ifdef USE_CJ
          int type_index = t2;
#else
          int type_index = t1 * paramb.num_types + t2;
#endif
          int c_index = get_c_index(
            type_index, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gn12 += fn12[k] * c[c_index];
        }
        q[n] += gn12;
      }
    }
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      g_descriptors[n1 + n * N] = q[n];
    }
  }
}

static __global__ void find_descriptors_angular(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_Descriptor_Parameters paramb,
  const float* c,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors,
  float* g_sum_fxyz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_DIM_ANGULAR] = {0.0f};

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < neighbor_number; ++i1) {
        int index = g_NN_sum[n1] + i1;
        int n2 = g_NL[index];
        float x12 = g_x12[index];
        float y12 = g_y12[index];
        float z12 = g_z12[index];
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
#ifdef USE_CJ
          int type_index = t2;
#else
          int type_index = t1 * paramb.num_types + t2;
#endif
          int c_index = get_c_index(
            type_index,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * c[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134, paramb.n_max_angular + 1, n, s, q);
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
      }
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      for (int l = 0; l < paramb.num_L; ++l) {
        int ln = l * (paramb.n_max_angular + 1) + n;
        g_descriptors[n1 + ((paramb.n_max_radial + 1) + ln) * N] = q[ln];
      }
    }
  }
}

static __global__ void apply_ann(
  const int N,
  const NEP_ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  double* g_pe,
  float* g_Fp)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int type = g_type[n1];

    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    const int neu1 = annmb.num_neurons1;
    const int neu1_dim = neu1 * annmb.dim;
    if (annmb.num_hidden_layers == 2) {
      const int neu2 = annmb.num_neurons2;
      apply_ann_two_layers(
        annmb.dim,
        neu1,
        neu2,
        annmb.wb[type],
        annmb.wb[type] + neu1_dim,
        annmb.wb[type] + neu1 * (annmb.dim + 1),
        annmb.wb[type] + neu1 * (annmb.dim + 1 + neu2),
        annmb.wb[type] + neu1 * (annmb.dim + 1 + neu2) + neu2,
        annmb.b,
        q,
        F,
        Fp);
    } else {
      apply_ann_one_layer(
        annmb.dim,
        neu1,
        annmb.wb[type],
        annmb.wb[type] + neu1_dim,
        annmb.wb[type] + neu1 * (annmb.dim + 1),
        annmb.b,
        q,
        F,
        Fp);
    }
    g_pe[n1] = static_cast<double>(F);

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

static __global__ void apply_ann_temperature(
  const int N,
  const NEP_ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  float* __restrict__ g_q_scaler,
  const float* __restrict__ g_temperature,
  double* g_pe,
  float* g_Fp)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int type = g_type[n1];
    float temperature = g_temperature[n1];

    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim - 1; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    g_q_scaler[annmb.dim - 1] = 0.001; // temperature dimension scaler
    q[annmb.dim - 1] = temperature * g_q_scaler[annmb.dim - 1];
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    apply_ann_one_layer(
      annmb.dim,
      annmb.num_neurons1,
      annmb.wb[type],
      annmb.wb[type] + annmb.num_neurons1 * annmb.dim,
      annmb.wb[type] + annmb.num_neurons1 * (annmb.dim + 1),
      annmb.b,
      q,
      F,
      Fp);
    g_pe[n1] = static_cast<double>(F);

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

static __global__ void find_force_radial(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_Descriptor_Parameters paramb,
  const float* c,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int neighbor_number = g_NN[n1];
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};

      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
#ifdef USE_CJ
          int type_index = t2;
#else
          int type_index = t1 * paramb.num_types + t2;
#endif
          int c_index = get_c_index(
            type_index, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gnp12 += fnp12[k] * c[c_index];
        }
        float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }

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
    }
    g_virial[n1] += s_virial_xx;
    g_virial[n1 + N] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] = s_virial_xy;
    g_virial[n1 + N * 4] = s_virial_yz;
    g_virial[n1 + N * 5] = s_virial_zx;
  }
}

static __global__ void find_force_angular(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP_Descriptor_Parameters paramb,
  const float* c,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {

    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
      }
    }
    int neighbor_number = g_NN[n1];
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float f12[3] = {0.0f};

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
#ifdef USE_CJ
          int type_index = t2;
#else
          int type_index = t1 * paramb.num_types + t2;
#endif
          int c_index = get_c_index(
            type_index,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * c[c_index];
          gnp12 += fnp12[k] * c[c_index];
        }
        accumulate_f12(paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
          paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
      }

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
    }
    g_virial[n1] += s_virial_xx;
    g_virial[n1 + N] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
  }
}

void compute_nep_descriptors(
  const NEP_Descriptor_Parameters& paramb,
  const float* c,
  const Dataset& dataset,
  const NEP_Neighbor& neighbor,
  NEP_Data& data,
  NEP_Compiled_Evaluation* compiled_kernel)
{
  const int block_size = 32;
  const int grid_size = (dataset.N - 1) / block_size + 1;

  if (compiled_kernel) {
    compiled_kernel->common.launch_descriptor_radial(
      dataset.N,
      dataset.NN_radial_sum.data(),
      dataset.NN_radial.data(),
      neighbor.NL_radial.data(),
      dataset.type.data(),
      neighbor.x12_radial.data(),
      neighbor.y12_radial.data(),
      neighbor.z12_radial.data(),
      data.parameters.data(),
      data.descriptors.data());
  } else {
    find_descriptors_radial<<<grid_size, block_size>>>(
      dataset.N,
      dataset.NN_radial_sum.data(),
      dataset.NN_radial.data(),
      neighbor.NL_radial.data(),
      paramb,
      c,
      dataset.type.data(),
      neighbor.x12_radial.data(),
      neighbor.y12_radial.data(),
      neighbor.z12_radial.data(),
      data.descriptors.data());
    GPU_CHECK_KERNEL
  }

  if (compiled_kernel) {
    compiled_kernel->common.launch_descriptor_angular(
      dataset.N,
      dataset.NN_angular_sum.data(),
      dataset.NN_angular.data(),
      neighbor.NL_angular.data(),
      dataset.type.data(),
      neighbor.x12_angular.data(),
      neighbor.y12_angular.data(),
      neighbor.z12_angular.data(),
      data.parameters.data(),
      data.descriptors.data(),
      data.sum_fxyz.data());
  } else {
    find_descriptors_angular<<<grid_size, block_size>>>(
      dataset.N,
      dataset.NN_angular_sum.data(),
      dataset.NN_angular.data(),
      neighbor.NL_angular.data(),
      paramb,
      c,
      dataset.type.data(),
      neighbor.x12_angular.data(),
      neighbor.y12_angular.data(),
      neighbor.z12_angular.data(),
      data.descriptors.data(),
      data.sum_fxyz.data());
    GPU_CHECK_KERNEL
  }
}

void evaluate_nep_ann(
  const NEP_ANN& ann,
  const float* temperature,
  Dataset& dataset,
  NEP_Data& data,
  float* q_scaler,
  NEP_Compiled_Evaluation* compiled_kernel)
{
  const int block_size = 32;
  const int grid_size = (dataset.N - 1) / block_size + 1;

  if (temperature) {
    if (compiled_kernel) {
      compiled_kernel->launch_ann_temperature(
        dataset.N,
        dataset.type.data(),
        data.descriptors.data(),
        q_scaler,
        temperature,
        data.parameters.data(),
        dataset.energy.data(),
        data.Fp.data());
    } else {
      apply_ann_temperature<<<grid_size, block_size>>>(
        dataset.N,
        ann,
        dataset.type.data(),
        data.descriptors.data(),
        q_scaler,
        temperature,
        dataset.energy.data(),
        data.Fp.data());
      GPU_CHECK_KERNEL
    }
  } else if (compiled_kernel) {
    compiled_kernel->launch_ann(
      dataset.N,
      dataset.type.data(),
      data.descriptors.data(),
      q_scaler,
      data.parameters.data(),
      dataset.energy.data(),
      data.Fp.data());
  } else {
    apply_ann<<<grid_size, block_size>>>(
      dataset.N,
      ann,
      dataset.type.data(),
      data.descriptors.data(),
      q_scaler,
      dataset.energy.data(),
      data.Fp.data());
    GPU_CHECK_KERNEL
  }
}

void compute_nep_forces(
  const NEP_Descriptor_Parameters& paramb,
  const float* c,
  Dataset& dataset,
  const NEP_Neighbor& neighbor,
  NEP_Data& data,
  NEP_Compiled_Evaluation* compiled_kernel)
{
  const int block_size = 32;
  const int grid_size = (dataset.N - 1) / block_size + 1;

  if (compiled_kernel) {
    compiled_kernel->launch_force_radial(
      dataset.N,
      dataset.NN_radial_sum.data(),
      dataset.NN_radial.data(),
      neighbor.NL_radial.data(),
      dataset.type.data(),
      neighbor.x12_radial.data(),
      neighbor.y12_radial.data(),
      neighbor.z12_radial.data(),
      data.parameters.data(),
      data.Fp.data(),
      dataset.force.data(),
      dataset.force.data() + dataset.N,
      dataset.force.data() + dataset.N * 2,
      dataset.virial.data());
  } else {
    find_force_radial<<<grid_size, block_size>>>(
      dataset.N,
      dataset.NN_radial_sum.data(),
      dataset.NN_radial.data(),
      neighbor.NL_radial.data(),
      paramb,
      c,
      dataset.type.data(),
      neighbor.x12_radial.data(),
      neighbor.y12_radial.data(),
      neighbor.z12_radial.data(),
      data.Fp.data(),
      dataset.force.data(),
      dataset.force.data() + dataset.N,
      dataset.force.data() + dataset.N * 2,
      dataset.virial.data());
    GPU_CHECK_KERNEL
  }

  if (compiled_kernel) {
    compiled_kernel->launch_force_angular(
      dataset.N,
      dataset.NN_angular_sum.data(),
      dataset.NN_angular.data(),
      neighbor.NL_angular.data(),
      dataset.type.data(),
      neighbor.x12_angular.data(),
      neighbor.y12_angular.data(),
      neighbor.z12_angular.data(),
      data.parameters.data(),
      data.Fp.data(),
      data.sum_fxyz.data(),
      dataset.force.data(),
      dataset.force.data() + dataset.N,
      dataset.force.data() + dataset.N * 2,
      dataset.virial.data());
  } else {
    find_force_angular<<<grid_size, block_size>>>(
      dataset.N,
      dataset.NN_angular_sum.data(),
      dataset.NN_angular.data(),
      neighbor.NL_angular.data(),
      paramb,
      c,
      dataset.type.data(),
      neighbor.x12_angular.data(),
      neighbor.y12_angular.data(),
      neighbor.z12_angular.data(),
      data.Fp.data(),
      data.sum_fxyz.data(),
      dataset.force.data(),
      dataset.force.data() + dataset.N,
      dataset.force.data() + dataset.N * 2,
      dataset.virial.data());
    GPU_CHECK_KERNEL
  }
}

void NEP_Compiled_Evaluation::launch_ann(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  double* pe,
  float* Fp)
{
  check_nep_launch(
    ann_nep_(
      N, type, descriptors, q_scaler, parameters, pe, Fp),
    "ann_nep_jit");
}

void NEP_Compiled_Evaluation::launch_ann_temperature(
  int N,
  const int* type,
  const float* descriptors,
  float* q_scaler,
  const float* temperature,
  const float* parameters,
  double* pe,
  float* Fp)
{
  check_nep_launch(
    ann_temperature_(
      N, type, descriptors, q_scaler, temperature,
      parameters, pe, Fp),
    "ann_temperature_jit");
}

void NEP_Compiled_Evaluation::launch_force_radial(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp,
  float* fx, float* fy, float* fz, float* virial)
{
  check_nep_launch(
    common.force_radial_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      nullptr, nullptr, nullptr, nullptr,
      0, fx, fy, fz, virial),
    "force_radial_jit");
}

void NEP_Compiled_Evaluation::launch_force_angular(
  int N, const int* NN_sum, const int* NN, const int* NL, const int* type,
  const float* x12, const float* y12, const float* z12,
  const float* parameters, const float* Fp, const float* sum_fxyz,
  float* fx, float* fy, float* fz, float* virial)
{
  check_nep_launch(
    common.force_angular_(
      N, NN_sum, NN, NL, type, x12, y12, z12,
      parameters, Fp,
      nullptr, nullptr, nullptr, nullptr,
      sum_fxyz, 0, fx, fy, fz, virial),
    "force_angular_jit");
}
