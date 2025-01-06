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
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "dataset.cuh"
#include "mic.cuh"
#include "nep3.cuh"
#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_utilities.cuh"

static __global__ void gpu_find_neighbor_list(
  const NEP3::ParaMB paramb,
  const int N,
  const int* Na,
  const int* Na_sum,
  const bool use_typewise_cutoff,
  const int* g_type,
  const double g_rc_radial,
  const double g_rc_angular,
  const double* __restrict__ g_box,
  const double* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
  const double* x,
  const double* y,
  const double* z,
  int* NN_radial,
  int* NL_radial,
  int* NN_angular,
  int* NL_angular,
  double* x12_radial,
  double* y12_radial,
  double* z12_radial,
  double* x12_angular,
  double* y12_angular,
  double* z12_angular)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
    const double* __restrict__ box = g_box + 18 * blockIdx.x;
    const double* __restrict__ box_original = g_box_original + 9 * blockIdx.x;
    const int* __restrict__ num_cell = g_num_cell + 3 * blockIdx.x;
    double x1 = x[n1];
    double y1 = y[n1];
    double z1 = z[n1];
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
            double delta_x = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
            double delta_y = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
            double delta_z = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;
            double x12 = x[n2] + delta_x - x1;
            double y12 = y[n2] + delta_y - y1;
            double z12 = z[n2] + delta_z - z1;
            dev_apply_mic(box, x12, y12, z12);
            double distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            int t2 = g_type[n2];
            double rc_radial = g_rc_radial;
            double rc_angular = g_rc_angular;
            if (use_typewise_cutoff) {
              int z1 = paramb.atomic_numbers[t1];
              int z2 = paramb.atomic_numbers[t2];
              rc_radial = min(
                (COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * paramb.typewise_cutoff_radial_factor,
                rc_radial);
              rc_angular = min(
                (COVALENT_RADIUS[z1] + COVALENT_RADIUS[z2]) * paramb.typewise_cutoff_angular_factor,
                rc_angular);
            }
            if (distance_square < rc_radial * rc_radial) {
              NL_radial[count_radial * N + n1] = n2;
              x12_radial[count_radial * N + n1] = x12;
              y12_radial[count_radial * N + n1] = y12;
              z12_radial[count_radial * N + n1] = z12;
              count_radial++;
            }
            if (distance_square < rc_angular * rc_angular) {
              NL_angular[count_angular * N + n1] = n2;
              x12_angular[count_angular * N + n1] = x12;
              y12_angular[count_angular * N + n1] = y12;
              z12_angular[count_angular * N + n1] = z12;
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

template <bool IsTraining>
static __global__ void find_descriptors_radial(
  const int N,
  const int max_NN_radial,
  const int* g_NN,
  const int* g_NL,
  const NEP3::ParaMB paramb,
  const NEP3::ANN annmb,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x12,
  const double* __restrict__ g_y12,
  const double* __restrict__ g_z12,
  double* g_descriptors,
  double* g_q_c = nullptr)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    double q[MAX_NUM_N] = {0.0};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = n1 + N * i1;
      int n2 = g_NL[index];
      double x12 = g_x12[index];
      double y12 = g_y12[index];
      double z12 = g_z12[index];
      double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      double fc12;
      int t2 = g_type[n2];
      double rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      double rcinv = 1.0 / rc;
      find_fc(rc, rcinv, d12, fc12);

      double fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gn12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
          if constexpr (IsTraining) {
            int g_q_c_index = n1 +
                              N * i1 +
                              N * max_NN_radial * n +
                              N * max_NN_radial * (paramb.n_max_radial + 1) * k;
            g_q_c[g_q_c_index] = fn12[k];
          }
          // 假设:
          // - N = 1000 (总原子数)
          // - basis_size_radial = 5 (6阶)
          // - n_max_radial = 3 (4个径向描述符)
          // - max_NN_radial = 50 (每个原子最多50个邻居)

          // 访问:
          // - 第2个原子(n1=2)
          // - 第30个邻居(i1=30)
          // - 第1个描述符(n=1)
          // - 第3阶基函数(k=3)

          // index = 2 + 
          //         1000 * 30
          //         1000 * 50 * 1
          //         1000 * 50 * 4 * 3
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
  const int* g_NN,
  const int* g_NL,
  const NEP3::ParaMB paramb,
  const NEP3::ANN annmb,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x12,
  const double* __restrict__ g_y12,
  const double* __restrict__ g_z12,
  double* g_descriptors,
  double* g_sum_fxyz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    double q[MAX_DIM_ANGULAR] = {0.0};

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      double s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < neighbor_number; ++i1) {
        int index = n1 + N * i1;
        int n2 = g_NL[n1 + N * i1];
        double x12 = g_x12[index];
        double y12 = g_y12[index];
        double z12 = g_z12[index];
        double d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        double fc12;
        int t2 = g_type[n2];
        double rc = paramb.rc_angular;
        if (paramb.use_typewise_cutoff) {
          rc = min(
            (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
             COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
              paramb.typewise_cutoff_angular_factor,
            rc);
        }
        double rcinv = 1.0 / rc;
        find_fc(rc, rcinv, d12, fc12);
        double fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        double gn12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular + 1, n, s, q);
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
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

NEP3::NEP3(
  Parameters& para,
  int N,
  int N_times_max_NN_radial,
  int N_times_max_NN_angular,
  int version,
  int deviceCount)
{
  paramb.version = version;
  paramb.rc_radial = para.rc_radial;
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rc_angular = para.rc_angular;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  paramb.use_typewise_cutoff = para.use_typewise_cutoff;
  paramb.use_typewise_cutoff_zbl = para.use_typewise_cutoff_zbl;
  paramb.typewise_cutoff_radial_factor = para.typewise_cutoff_radial_factor;
  paramb.typewise_cutoff_angular_factor = para.typewise_cutoff_angular_factor;
  paramb.typewise_cutoff_zbl_factor = para.typewise_cutoff_zbl_factor;
  paramb.num_types = para.num_types;
  paramb.n_max_radial = para.n_max_radial;
  paramb.n_max_angular = para.n_max_angular;
  paramb.L_max = para.L_max;
  paramb.num_L = paramb.L_max;
  if (para.L_max_4body == 2) {
    paramb.num_L += 1;
  }
  if (para.L_max_5body == 1) {
    paramb.num_L += 1;
  }
  paramb.dim_angular = (para.n_max_angular + 1) * paramb.num_L;

  paramb.basis_size_radial = para.basis_size_radial;
  paramb.basis_size_angular = para.basis_size_angular;
  paramb.num_types_sq = para.num_types * para.num_types;
  paramb.num_c_radial =
    paramb.num_types_sq * (para.n_max_radial + 1) * (para.basis_size_radial + 1);

  zbl.enabled = para.enable_zbl;
  zbl.flexibled = para.flexible_zbl;
  zbl.rc_inner = para.zbl_rc_inner;
  zbl.rc_outer = para.zbl_rc_outer;
  for (int n = 0; n < para.atomic_numbers.size(); ++n) {
    zbl.atomic_numbers[n] = para.atomic_numbers[n];        // starting from 1
    paramb.atomic_numbers[n] = para.atomic_numbers[n] - 1; // starting from 0
  }
  if (zbl.flexibled) {
    zbl.num_types = para.num_types;
    int num_type_zbl = (para.num_types * (para.num_types + 1)) / 2;
    for (int n = 0; n < num_type_zbl * 10; ++n) {
      zbl.para[n] = para.zbl_para[n];
    }
  }

  for (int device_id = 0; device_id < deviceCount; device_id++) {
    cudaSetDevice(device_id);
    annmb[device_id].dim = para.dim;
    annmb[device_id].num_neurons1 = para.num_neurons1;
    annmb[device_id].num_para = para.number_of_variables;
    annmb[device_id].num_ann = para.number_of_variables_ann;

    nep_data[device_id].NN_radial.resize(N);
    nep_data[device_id].NN_angular.resize(N);
    nep_data[device_id].NL_radial.resize(N_times_max_NN_radial);
    nep_data[device_id].NL_angular.resize(N_times_max_NN_angular);
    nep_data[device_id].x12_radial.resize(N_times_max_NN_radial);
    nep_data[device_id].y12_radial.resize(N_times_max_NN_radial);
    nep_data[device_id].z12_radial.resize(N_times_max_NN_radial);
    nep_data[device_id].x12_angular.resize(N_times_max_NN_angular);
    nep_data[device_id].y12_angular.resize(N_times_max_NN_angular);
    nep_data[device_id].z12_angular.resize(N_times_max_NN_angular);
    nep_data[device_id].descriptors.resize(N * annmb[device_id].dim);
    nep_data[device_id].Fp.resize(N * annmb[device_id].dim);
    nep_data[device_id].Fp2.resize(N * annmb[device_id].dim * annmb[device_id].dim);
    nep_data[device_id].sum_fxyz.resize(N * (paramb.n_max_angular + 1) * NUM_OF_ABC);
    nep_data[device_id].parameters.resize(annmb[device_id].num_para);
  }
}

void NEP3::update_potential(Parameters& para, const double* parameters, ANN& ann)
{
  const double* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
    // ann.b1[t] = pointer;
    pointer += 1; // type bias stored in ann.w1[t]
  }

  if (para.train_mode == 2) {
    for (int t = 0; t < paramb.num_types; ++t) {
      ann.w0_pol[t] = pointer;
      pointer += ann.num_neurons1 * ann.dim;
      ann.b0_pol[t] = pointer;
      pointer += ann.num_neurons1;
      ann.w1_pol[t] = pointer;
      pointer += ann.num_neurons1;
      // ann.b1_pol[t] = pointer;
      pointer += 1;
    }
  }
  ann.c = pointer;
}

static void __global__ find_max_min(const int N, const double* g_q, double* g_q_scaler)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ double s_max[1024];
  __shared__ double s_min[1024];
  s_max[tid] = -1000000.0; // a small number
  s_min[tid] = +1000000.0; // a large number
  const int stride = 1024;
  const int number_of_rounds = (N - 1) / stride + 1;
  for (int round = 0; round < number_of_rounds; ++round) {
    const int n = round * stride + tid;
    if (n < N) {
      const int m = n + N * bid;
      double q = g_q[m];
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
    g_q_scaler[bid] = min(g_q_scaler[bid], 1.0 / (s_max[0] - s_min[0]));
  }
}

template <bool IsTraining>
static __global__ void apply_ann(
  const int N,
  const NEP3::ParaMB paramb,
  const NEP3::ANN annmb,
  const int* __restrict__ g_type,
  const double* __restrict__ g_descriptors,
  const double* __restrict__ g_q_scaler,
  double* g_pe,
  double* g_Fp,
  double* g_Fp2 = nullptr,
  double* g_Fp_wb = nullptr,
  double* g_E_wb_grad = nullptr)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  int type = g_type[n1];

  if (n1 < N) {
    // get descriptors
    double q[MAX_DIM] = {0.0};
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    // get energy and energy gradient
    double F = 0.0, Fp[MAX_DIM] = {0.0};
    
    if constexpr (IsTraining) {
      // double Fp2[MAX_DIM * MAX_DIM] = {0.0};// 列优先：Fp2[d2 + d1 * annmb.dim] = Fp2[d2][d1] --> feat0, Fp2[0][0], Fp2[1][0]; feat1, Fp2[1][0], Fp2[1][1] --> g_Fp2[n1 + (d2 + d1 * annmb.dim) * N] = g_Fp2[n1][d2][d1]
      int type_offset = n1 * annmb.num_ann + type * ((annmb.dim + 2) * annmb.num_neurons1 + 1); 
      int type_offset_2 = n1 * annmb.num_ann * annmb.dim + type * ((annmb.dim + 2) * annmb.num_neurons1 + 1) * annmb.dim;
      apply_ann_one_layer_w2nd(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[type],
        annmb.b0[type],
        annmb.w1[type],
        N,
        q,  
        F,
        Fp,
        &g_Fp2[n1],
        g_Fp_wb + type_offset_2,
        g_E_wb_grad + type_offset);
      for (int d1 = 0; d1 < annmb.dim; ++d1) {
        for (int d2 = 0; d2 < annmb.dim; ++d2) {
          // g_Fp2[n1 + (d2 + d1 * annmb.dim) * N] = Fp2[d2 + d1 * annmb.dim] * g_q_scaler[d2];
          g_Fp2[n1 + (d2 + d1 * annmb.dim) * N] *= g_q_scaler[d2];
          // printf("g_Fp2[%d + (%d + %d * %d) * %d] = %f\n", n1, d2, d1, annmb.dim, N, g_Fp2[n1 + (d2 + d1 * annmb.dim) * N]);
        }
      }
    } else {
      one_layer(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[type],
        annmb.b0[type],
        annmb.w1[type],
        q,
        F,
        Fp);
    }
    g_pe[n1] = F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

template <bool IsTraining>
static __global__ void apply_ann_pol(
  const int N,
  const NEP3::ParaMB paramb,
  const NEP3::ANN annmb,
  const int* __restrict__ g_type,
  const double* __restrict__ g_descriptors,
  const double* __restrict__ g_q_scaler,
  double* g_virial,
  double* g_Fp,
  double* g_Fp2 = nullptr,
  double* g_Fp_wb = nullptr,
  double* g_E_wb_grad = nullptr)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  int type = g_type[n1];
  int type_offset = n1 * annmb.num_ann + type * (annmb.dim + 2) * annmb.num_neurons1;
  int type_offset_2 = n1 * annmb.num_ann * annmb.dim + type * (annmb.dim + 2) * annmb.num_neurons1 * annmb.dim;
  if (n1 < N) {
    // get descriptors
    double q[MAX_DIM] = {0.0};
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    // get energy and energy gradient
    double F = 0.0, Fp[MAX_DIM] = {0.0};
    if constexpr (IsTraining) {
      // scalar part
      apply_ann_one_layer_w2nd(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0_pol[type],
        annmb.b0_pol[type],
        annmb.w1_pol[type],
        N,
        q,
        F,
        Fp,
        &g_Fp2[n1],
        g_Fp_wb + type_offset_2,
        g_E_wb_grad + type_offset);

      for (int d1 = 0; d1 < annmb.dim; ++d1) {
        for (int d2 = 0; d2 < annmb.dim; ++d2) {
          g_Fp2[n1 + (d2 + d1 * annmb.dim) * N] *= g_q_scaler[d2];
        }
      }
    } else {
      one_layer(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0_pol[type],
        annmb.b0_pol[type],
        annmb.w1_pol[type],
        q,
        F,
        Fp);
    }

    g_virial[n1] = F;
    g_virial[n1 + N] = F;
    g_virial[n1 + N * 2] = F;

    // tensor part
    for (int d = 0; d < annmb.dim; ++d) {
      Fp[d] = 0.0;
    }
    if constexpr (IsTraining) {
      apply_ann_one_layer_w2nd(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[type],
        annmb.b0[type],
        annmb.w1[type],
        N,
        q,
        F,
        Fp,
        &g_Fp2[n1],
        g_Fp_wb + type_offset_2,
        g_E_wb_grad + type_offset);

      for (int d1 = 0; d1 < annmb.dim; ++d1) {
        for (int d2 = 0; d2 < annmb.dim; ++d2) {
          g_Fp2[n1 + (d2 + d1 * annmb.dim) * N] *= g_q_scaler[d2];
        }
      }
      } else {
        one_layer(
          annmb.dim,
          annmb.num_neurons1,
          annmb.w0[type],
          annmb.b0[type],
          annmb.w1[type],
          q,
          F,
          Fp);
      }

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

template <bool IsTraining>
static __global__ void apply_ann_temperature(
  const int N,
  const NEP3::ParaMB paramb,
  const NEP3::ANN annmb,
  const int* __restrict__ g_type,
  const double* __restrict__ g_descriptors,
  double* __restrict__ g_q_scaler,
  const double* __restrict__ g_temperature,
  double* g_pe,
  double* g_Fp,
  double* g_Fp2 = nullptr,
  double* g_Fp_wb = nullptr,
  double* g_E_wb_grad = nullptr)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  int type = g_type[n1];
  double temperature = g_temperature[n1];
  if (n1 < N) {
    // get descriptors
    double q[MAX_DIM] = {0.0};
    for (int d = 0; d < annmb.dim - 1; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    g_q_scaler[annmb.dim - 1] = 0.001; // temperature dimension scaler
    q[annmb.dim - 1] = temperature * g_q_scaler[annmb.dim - 1];
    // get energy and energy gradient
    double F = 0.0, Fp[MAX_DIM] = {0.0};
    if constexpr (IsTraining) {
      int type_offset = n1 * annmb.num_ann + type * (annmb.dim + 2) * annmb.num_neurons1;
      int type_offset_2 = n1 * annmb.num_ann * annmb.dim + type * (annmb.dim + 2) * annmb.num_neurons1 * annmb.dim;
      apply_ann_one_layer_w2nd(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[type],
        annmb.b0[type],
        annmb.w1[type],
        N,
        q,
        F,
        Fp,
        &g_Fp2[n1],
        g_Fp_wb + type_offset_2,
        g_E_wb_grad + type_offset);

      for (int d1 = 0; d1 < annmb.dim; ++d1) {
        for (int d2 = 0; d2 < annmb.dim; ++d2) {
          g_Fp2[n1 + (d2 + d1 * annmb.dim) * N] *= g_q_scaler[d2];
        }
      }
    } else {
      one_layer(
        annmb.dim,
        annmb.num_neurons1,
        annmb.w0[type],
        annmb.b0[type],
        annmb.w1[type],
        q, F, Fp);
    }
    g_pe[n1] = F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

static __global__ void zero_force(
  const int N, double* g_fx, double* g_fy, double* g_fz, double* g_vxx, double* g_vyy, double* g_vzz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_fx[n1] = 0.0;
    g_fy[n1] = 0.0;
    g_fz[n1] = 0.0;
    g_vxx[n1] = 0.0;
    g_vyy[n1] = 0.0;
    g_vzz[n1] = 0.0;
  }
}

static __global__ void gpu_sum_pe_error(
  int* g_Na, int* g_Na_sum, double* g_pe, double* g_pe_ref, double* diff_gpu, double* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];   // 当前结构的原子数
  int N1 = g_Na_sum[bid]; // 当前结构在全局原子数组中的原子起始索引
  int N2 = N1 + Na;      // 当前结构在全局原子数组中的原子结束索引（不包括）
  extern __shared__ double s_pe[];
  s_pe[tid] = 0.0;

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
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
    double diff = (s_pe[0] - g_pe_ref[bid]) / Na;
    diff_gpu[bid] = diff;
    error_gpu[bid] = diff * diff;
  }
}

static __global__ void gpu_sum_virial_error(
  const int N,
  const double shear_weight,
  int* g_Na,
  int* g_Na_sum,
  double* g_virial,
  double* g_virial_ref,
  double* diff_gpu,
  double* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ double s_virial[];
  for (int d = 0; d < 6; ++d) {
    s_virial[d * blockDim.x + tid] = 0.0; //size of s_virial is 6 * blockDim.x
}                     // sum of atomic contributions to virial tensor, respectively for xx, yy, zz, xy, yz, zx

  for (int n = N1 + tid; n < N2; n += blockDim.x) {
    for (int d = 0; d < 6; ++d) {
      s_virial[d * blockDim.x + tid] += g_virial[d * N + n];
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 32; offset >>= 1) {
    if (tid < offset) {
      for (int d = 0; d < 6; ++d) {
        s_virial[d * blockDim.x + tid] += s_virial[d * blockDim.x + tid + offset];
      }
    }
    __syncthreads();
  }

  for (int offset = 32; offset > 0; offset >>= 1) {
    if (tid < offset) {
      for (int d = 0; d < 6; ++d) {
        s_virial[d * blockDim.x + tid] += s_virial[d * blockDim.x + tid + offset];
      }
    }
    __syncwarp();
  }

  if (tid == 0) {
    double error_sum = 0.0;
    for (int d = 0; d < 6; ++d) {
      double diff = (s_virial[d * blockDim.x + 0] - g_virial_ref[d * gridDim.x + bid]) / Na;
      error_sum += (d >= 3) ? (shear_weight * diff * diff) : (diff * diff);
      diff_gpu[bid * 6 + d] = (d >= 3) ? shear_weight * diff : diff;
    }
    error_gpu[bid] = error_sum;
  }
}

static __global__ void compute_grad_radial(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP3::ParaMB paramb,
  const NEP3::ANN annmb,
  const int* __restrict__ g_Na,
  const int Nc,
  const double lambda_e,
  const double lambda_f,
  const double lambda_v,
  const int virial_nums,
  double* g_type_weight,
  double force_delta,
  const int* __restrict__ g_batch_idx,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x12,
  const double* __restrict__ g_y12,
  const double* __restrict__ g_z12,
  const double* __restrict__ g_Fp,
  double* __restrict__ g_E_wb_grad,
  const double* __restrict__ g_diff_gpu_e,
  const double* __restrict__ g_diff_gpu_v,
  const double* __restrict__ g_fx_ref,
  const double* __restrict__ g_fy_ref,
  const double* __restrict__ g_fz_ref,
  const double* __restrict__ g_weight,
  const double* __restrict__ g_fx,
  const double* __restrict__ g_fy,
  const double* __restrict__ g_fz,
  const int max_NN_radial,
  const double* __restrict__ g_q_scaler,
  const double* __restrict__ g_Fp2,
  const double* __restrict__ g_q_c,
  const double* __restrict__ g_ep_wb,
  double* g_grad_sum)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  const int w0_index = annmb.dim * annmb.num_neurons1;
  const int b0_index = w0_index + annmb.num_neurons1;
  if (n1 < N) {
    int neighbor_number = g_NN[n1];
    int t1 = g_type[n1];
    int batch_idx = g_batch_idx[n1];
    int Na = g_Na[batch_idx];
    double weight = g_weight[batch_idx];

    const double per_Nc_e = g_diff_gpu_e[batch_idx] * weight * 2.0 * lambda_e / Nc;
    const double per_Nc = weight * 2.0 * lambda_f / Na / 3 / Nc;
    const double per_Nc_v = weight * 2.0 * lambda_v / virial_nums;

    double fx_ref_n1 = g_fx_ref[n1];
    double fy_ref_n1 = g_fy_ref[n1];
    double fz_ref_n1 = g_fz_ref[n1];
    double dx_n1 = g_fx[n1] - fx_ref_n1;
    double dy_n1 = g_fy[n1] - fy_ref_n1;
    double dz_n1 = g_fz[n1] - fz_ref_n1;
    double type_weight = g_type_weight[g_type[n1]];
    if (force_delta > 0.0) {
      double force_magnitude = sqrt(fx_ref_n1 * fx_ref_n1 + fy_ref_n1 * fy_ref_n1 + fz_ref_n1 * fz_ref_n1);
      type_weight *= sqrt(force_delta / (force_delta + force_magnitude));
    }
    dx_n1 *= type_weight;
    dy_n1 *= type_weight;
    dz_n1 *= type_weight;

    int t1_net_index = t1 * ((annmb.dim + 2) * annmb.num_neurons1 + 1);
    int n1_net_index = n1 * annmb.num_ann + t1_net_index;
    int n1_net_index_wb = n1 * annmb.num_ann * annmb.dim + t1_net_index * annmb.dim;

    double* e_wb_grad = g_E_wb_grad + n1_net_index;
    
    double diff[6];
    if (virial_nums > 0) {
      diff[0] = g_diff_gpu_v[batch_idx * 6 + 0];
      diff[1] = g_diff_gpu_v[batch_idx * 6 + 1];
      diff[2] = g_diff_gpu_v[batch_idx * 6 + 2];
      diff[3] = g_diff_gpu_v[batch_idx * 6 + 3];
      diff[4] = g_diff_gpu_v[batch_idx * 6 + 4];
      diff[5] = g_diff_gpu_v[batch_idx * 6 + 5];
    }

    double feat_x_sum[MAX_NUM_N] = {0.0};
    double feat_y_sum[MAX_NUM_N] = {0.0}; 
    double feat_z_sum[MAX_NUM_N] = {0.0};
    double feat_123_xx_sum[MAX_NUM_N] = {0.0};
    double feat_123_yy_sum[MAX_NUM_N] = {0.0};
    double feat_123_zz_sum[MAX_NUM_N] = {0.0};
    double feat_123_xy_sum[MAX_NUM_N] = {0.0};
    double feat_123_yz_sum[MAX_NUM_N] = {0.0};
    double feat_123_zx_sum[MAX_NUM_N] = {0.0};
    double feat_xx_sum_i1[MAX_NUM_N] = {0.0};
    double feat_yy_sum_i1[MAX_NUM_N] = {0.0};
    double feat_zz_sum_i1[MAX_NUM_N] = {0.0};
    double feat_xy_sum_i1[MAX_NUM_N] = {0.0};
    double feat_yz_sum_i1[MAX_NUM_N] = {0.0};
    double feat_zx_sum_i1[MAX_NUM_N] = {0.0};

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      double fx_ref_n2 = g_fx_ref[n2];
      double fy_ref_n2 = g_fy_ref[n2];
      double fz_ref_n2 = g_fz_ref[n2];
      double dx_n2 = g_fx[n2] - fx_ref_n2;
      double dy_n2 = g_fy[n2] - fy_ref_n2;
      double dz_n2 = g_fz[n2] - fz_ref_n2;
      double type_weight_n2 = g_type_weight[g_type[n2]];
      if (force_delta > 0.0) {
        double force_magnitude = sqrt(fx_ref_n2 * fx_ref_n2 + fy_ref_n2 * fy_ref_n2 + fz_ref_n2 * fz_ref_n2);
        type_weight_n2 *= sqrt(force_delta / (force_delta + force_magnitude));
      }
      dx_n2 *= type_weight_n2;
      dy_n2 *= type_weight_n2;
      dz_n2 *= type_weight_n2;
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;
      double fc12, fcp12;
      double rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      double tmp_xyz[3] = {d12inv * r12[0], d12inv * r12[1], d12inv * r12[2]};
      double feat_x[MAX_NUM_N] = {0.0};
      double feat_y[MAX_NUM_N] = {0.0};
      double feat_z[MAX_NUM_N] = {0.0};

      double tmp_xyz_123[6] = {
        tmp_xyz[0] * r12[0], // xx
        tmp_xyz[1] * r12[1], // yy
        tmp_xyz[2] * r12[2], // zz
        tmp_xyz[1] * r12[0], // xy
        tmp_xyz[2] * r12[1], // yz
        tmp_xyz[0] * r12[2]  // zx
      };
      double feat_123_xx[MAX_NUM_N] = {0.0};
      double feat_123_yy[MAX_NUM_N] = {0.0};
      double feat_123_zz[MAX_NUM_N] = {0.0};
      double feat_123_xy[MAX_NUM_N] = {0.0};
      double feat_123_yz[MAX_NUM_N] = {0.0};
      double feat_123_zx[MAX_NUM_N] = {0.0};

      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12 = 0.0;
        double gFp_val = g_Fp[n1 + n * N];
        // E'(n) * ∂d_ij/∂α_ij
        double fp_xyz[3] = {
          gFp_val * tmp_xyz[0],
          gFp_val * tmp_xyz[1],
          gFp_val * tmp_xyz[2]
        };
        // E'(n) * ∂d_ij/∂α_ij * α_ij
        double fp_xyz_123[6] = {
          gFp_val * tmp_xyz_123[0],
          gFp_val * tmp_xyz_123[1],
          gFp_val * tmp_xyz_123[2],
          gFp_val * tmp_xyz_123[3],
          gFp_val * tmp_xyz_123[4],
          gFp_val * tmp_xyz_123[5]
        };

        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];

          int q_c_index = n1 +
                        N * i1 +
                        N * max_NN_radial * n +
                        N * max_NN_radial * (paramb.n_max_radial + 1) * k;
          // E'(n) * Q'_{nk}(i,j) * ∂d_ij/∂α_ij 
          double qp_c_tmp[3] = {
                    fnp12[k] * fp_xyz[0],
                    fnp12[k] * fp_xyz[1],
                    fnp12[k] * fp_xyz[2]
                };
          double f_c_n1 = qp_c_tmp[0] * dx_n1 + qp_c_tmp[1] * dy_n1 + qp_c_tmp[2] * dz_n1;
          double f_c_n2 = qp_c_tmp[0] * dx_n2 + qp_c_tmp[1] * dy_n2 + qp_c_tmp[2] * dz_n2;
          int grad_c_index = c_index + annmb.num_ann;
          atomicAdd(&g_grad_sum[grad_c_index], f_c_n1 * per_Nc);
          atomicAdd(&g_grad_sum[grad_c_index], -f_c_n2 * per_Nc);
      
          atomicAdd(&g_grad_sum[grad_c_index], g_Fp[n1 + n * N] * g_q_c[q_c_index] * per_Nc_e);

          // E'(n) * Q'_{nk}(i,j) * ∂d_ij/∂α_ij * α_ij
          if (virial_nums > 0) {
            double v_c_n1 = fnp12[k] * per_Nc_v * (fp_xyz_123[0] * diff[0] + fp_xyz_123[1] * diff[1] + fp_xyz_123[2] * diff[2]
            + fp_xyz_123[3] * diff[3] + fp_xyz_123[4] * diff[4] + fp_xyz_123[5] * diff[5]);
            atomicAdd(&g_grad_sum[grad_c_index], -v_c_n1);
          }
        }
 
        feat_x[n] = gnp12 * tmp_xyz[0];
        feat_y[n] = gnp12 * tmp_xyz[1];
        feat_z[n] = gnp12 * tmp_xyz[2]; 

        feat_123_xx[n] = feat_x[n] * r12[0];
        feat_123_yy[n] = feat_y[n] * r12[1];
        feat_123_zz[n] = feat_z[n] * r12[2];
        feat_123_xy[n] = feat_y[n] * r12[0];
        feat_123_yz[n] = feat_z[n] * r12[1];
        feat_123_zx[n] = feat_x[n] * r12[2];
        feat_xx_sum_i1[n] += feat_123_xx[n];
        feat_yy_sum_i1[n] += feat_123_yy[n];
        feat_zz_sum_i1[n] += feat_123_zz[n];
        feat_xy_sum_i1[n] += feat_123_xy[n];
        feat_yz_sum_i1[n] += feat_123_yz[n];
        feat_zx_sum_i1[n] += feat_123_zx[n];
      }

      for (int j = 0; j < annmb.num_neurons1; ++j) {
        double sum_dfeat_w1b0[6] = {0.0};
        for (int d = 0; d <= paramb.n_max_radial; ++d) {
          double sum_dfeat_w0[3] = {0.0};
          double dfeat_scaler[3] = {feat_x[d] * g_q_scaler[d], feat_y[d] * g_q_scaler[d], feat_z[d] * g_q_scaler[d]};
          int w1_index_dim = n1_net_index_wb + (b0_index + j) * annmb.dim + d;//(N_neu * N_des + N_neu + j) * N_des + n
          int b0_index_dim = n1_net_index_wb + (w0_index + j) * annmb.dim + d;//(N_neu * N_des + j) * N_des + n
          sum_dfeat_w1b0[0] += dfeat_scaler[0] * g_ep_wb[w1_index_dim];
          sum_dfeat_w1b0[1] += dfeat_scaler[1] * g_ep_wb[w1_index_dim];
          sum_dfeat_w1b0[2] += dfeat_scaler[2] * g_ep_wb[w1_index_dim];
          sum_dfeat_w1b0[3] += dfeat_scaler[0] * g_ep_wb[b0_index_dim];
          sum_dfeat_w1b0[4] += dfeat_scaler[1] * g_ep_wb[b0_index_dim];
          sum_dfeat_w1b0[5] += dfeat_scaler[2] * g_ep_wb[b0_index_dim];
          for (int m = 0; m <= paramb.n_max_radial; ++m) {
            double dfeat_w0_scaler[3] = {feat_x[m] * g_q_scaler[m], feat_y[m] * g_q_scaler[m], feat_z[m] * g_q_scaler[m]};
            int w0_index_dim = n1_net_index_wb + (j * annmb.dim + d) * annmb.dim + m;
            sum_dfeat_w0[0] += dfeat_w0_scaler[0] * g_ep_wb[w0_index_dim];
            sum_dfeat_w0[1] += dfeat_w0_scaler[1] * g_ep_wb[w0_index_dim];
            sum_dfeat_w0[2] += dfeat_w0_scaler[2] * g_ep_wb[w0_index_dim];
          }
          double f_w0_n1 = sum_dfeat_w0[0] * dx_n1 + sum_dfeat_w0[1] * dy_n1 + sum_dfeat_w0[2] * dz_n1;
          double f_w0_n2 = -sum_dfeat_w0[0] * dx_n2 - sum_dfeat_w0[1] * dy_n2 - sum_dfeat_w0[2] * dz_n2;
          atomicAdd(&g_grad_sum[t1_net_index + j * annmb.dim + d], f_w0_n1 * per_Nc);
          atomicAdd(&g_grad_sum[t1_net_index + j * annmb.dim + d], f_w0_n2 * per_Nc);
        }
        double f_w1_n1 = sum_dfeat_w1b0[0] * dx_n1 + sum_dfeat_w1b0[1] * dy_n1 + sum_dfeat_w1b0[2] * dz_n1;
        double f_w1_n2 = -sum_dfeat_w1b0[0] * dx_n2 - sum_dfeat_w1b0[1] * dy_n2 - sum_dfeat_w1b0[2] * dz_n2;
        double f_b0_n1 = sum_dfeat_w1b0[3] * dx_n1 + sum_dfeat_w1b0[4] * dy_n1 + sum_dfeat_w1b0[5] * dz_n1;
        double f_b0_n2 = -sum_dfeat_w1b0[3] * dx_n2 - sum_dfeat_w1b0[4] * dy_n2 - sum_dfeat_w1b0[5] * dz_n2;
        atomicAdd(&g_grad_sum[t1_net_index + b0_index + j], f_w1_n1 * per_Nc);
        atomicAdd(&g_grad_sum[t1_net_index + b0_index + j], f_w1_n2 * per_Nc);
        atomicAdd(&g_grad_sum[t1_net_index + w0_index + j], f_b0_n1 * per_Nc);
        atomicAdd(&g_grad_sum[t1_net_index + w0_index + j], f_b0_n2 * per_Nc);
      }

      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double feat_xyz_sum[3] = {0.0};
        for (int m = 0; m <= paramb.n_max_radial; ++m) {
          double E2 = g_Fp2[n1 + (m + n * (paramb.n_max_radial + 1)) * N]; //g_Fp2[n1 + (d2 + d1 * annmb.dim) * N]
          feat_xyz_sum[0] += feat_x[m] * E2;
          feat_xyz_sum[1] += feat_y[m] * E2;
          feat_xyz_sum[2] += feat_z[m] * E2;

          feat_123_xx_sum[n] += feat_123_xx[m] * E2;
          feat_123_yy_sum[n] += feat_123_yy[m] * E2;
          feat_123_zz_sum[n] += feat_123_zz[m] * E2;
          feat_123_xy_sum[n] += feat_123_xy[m] * E2;
          feat_123_yz_sum[n] += feat_123_yz[m] * E2;
          feat_123_zx_sum[n] += feat_123_zx[m] * E2;
        }
        feat_x_sum[n] += feat_xyz_sum[0];
        feat_y_sum[n] += feat_xyz_sum[1];
        feat_z_sum[n] += feat_xyz_sum[2];
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          for (int j = 0; j < neighbor_number; ++j) {
            int index = j * N + n1;
            int n2_tmp = g_NL[index];
            int t2_tmp = g_type[n2_tmp];
            int q_c_index = n1 +
                          N * j +
                          N * max_NN_radial * n +
                          N * max_NN_radial * (paramb.n_max_radial + 1) * k;
            int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2_tmp; 
            double q_c_scaler = g_q_c[q_c_index] * g_q_scaler[n];
            double f_c_n2 = -feat_xyz_sum[0] * q_c_scaler * dx_n2 - feat_xyz_sum[1] * q_c_scaler * dy_n2 - feat_xyz_sum[2] * q_c_scaler * dz_n2;
            int grad_c_index = c_index + annmb.num_ann;
            atomicAdd(&g_grad_sum[grad_c_index], f_c_n2 * per_Nc);
          }
        }
      }
    } // end of loop over neighbors

    for (int j = 0; j < annmb.num_neurons1; ++j) {
      double sum_dfeat_w1b0[12] = {0.0};
      for (int d = 0; d <= paramb.n_max_radial; ++d) {
        double sum_dfeat_w0[6] = {0.0};
        double dfeat_scaler[6] = {feat_xx_sum_i1[d] * g_q_scaler[d], 
                                 feat_yy_sum_i1[d] * g_q_scaler[d], 
                                 feat_zz_sum_i1[d] * g_q_scaler[d], 
                                 feat_xy_sum_i1[d] * g_q_scaler[d], 
                                 feat_yz_sum_i1[d] * g_q_scaler[d], 
                                 feat_zx_sum_i1[d] * g_q_scaler[d]};
        int w1_index_dim = n1_net_index_wb + (b0_index + j) * annmb.dim + d;
        int b0_index_dim = n1_net_index_wb + (w0_index + j) * annmb.dim + d;
        sum_dfeat_w1b0[0] += dfeat_scaler[0] * g_ep_wb[w1_index_dim];
        sum_dfeat_w1b0[1] += dfeat_scaler[1] * g_ep_wb[w1_index_dim];
        sum_dfeat_w1b0[2] += dfeat_scaler[2] * g_ep_wb[w1_index_dim];
        sum_dfeat_w1b0[3] += dfeat_scaler[3] * g_ep_wb[w1_index_dim];
        sum_dfeat_w1b0[4] += dfeat_scaler[4] * g_ep_wb[w1_index_dim];
        sum_dfeat_w1b0[5] += dfeat_scaler[5] * g_ep_wb[w1_index_dim];
        sum_dfeat_w1b0[6] += dfeat_scaler[0] * g_ep_wb[b0_index_dim];
        sum_dfeat_w1b0[7] += dfeat_scaler[1] * g_ep_wb[b0_index_dim];
        sum_dfeat_w1b0[8] += dfeat_scaler[2] * g_ep_wb[b0_index_dim];
        sum_dfeat_w1b0[9] += dfeat_scaler[3] * g_ep_wb[b0_index_dim];
        sum_dfeat_w1b0[10] += dfeat_scaler[4] * g_ep_wb[b0_index_dim];
        sum_dfeat_w1b0[11] += dfeat_scaler[5] * g_ep_wb[b0_index_dim];
        for (int m = 0; m <= paramb.n_max_radial; ++m) {
          double dfeat_w0_scaler[6] = {feat_xx_sum_i1[m] * g_q_scaler[m], 
                                      feat_yy_sum_i1[m] * g_q_scaler[m], 
                                      feat_zz_sum_i1[m] * g_q_scaler[m], 
                                      feat_xy_sum_i1[m] * g_q_scaler[m], 
                                      feat_yz_sum_i1[m] * g_q_scaler[m], 
                                      feat_zx_sum_i1[m] * g_q_scaler[m]};
          int w0_index_dim = n1_net_index_wb + (j * annmb.dim + d) * annmb.dim + m;  // (j * annmb.dim + d) * annmb.dim + m
          sum_dfeat_w0[0] += dfeat_w0_scaler[0] * g_ep_wb[w0_index_dim];
          sum_dfeat_w0[1] += dfeat_w0_scaler[1] * g_ep_wb[w0_index_dim];
          sum_dfeat_w0[2] += dfeat_w0_scaler[2] * g_ep_wb[w0_index_dim];
          sum_dfeat_w0[3] += dfeat_w0_scaler[3] * g_ep_wb[w0_index_dim];
          sum_dfeat_w0[4] += dfeat_w0_scaler[4] * g_ep_wb[w0_index_dim];
          sum_dfeat_w0[5] += dfeat_w0_scaler[5] * g_ep_wb[w0_index_dim];
        }
        double e_w0_n1 = e_wb_grad[j * annmb.dim + d] * per_Nc_e;
        atomicAdd(&g_grad_sum[t1_net_index + j * annmb.dim + d], e_w0_n1);
        if (virial_nums > 0) {
          double v_w0_n1 = per_Nc_v * (sum_dfeat_w0[0] * diff[0] + sum_dfeat_w0[1] * diff[1] 
          + sum_dfeat_w0[2] * diff[2] + sum_dfeat_w0[3] * diff[3] 
          + sum_dfeat_w0[4] * diff[4] + sum_dfeat_w0[5] * diff[5]);
          atomicAdd(&g_grad_sum[t1_net_index + j * annmb.dim + d], -v_w0_n1);
        }
      }
      double e_b0_n1 = e_wb_grad[annmb.num_neurons1 * annmb.dim + j] * per_Nc_e;
      double e_w1_n1 = e_wb_grad[annmb.num_neurons1 * annmb.dim + annmb.num_neurons1 + j] * per_Nc_e;
      atomicAdd(&g_grad_sum[t1_net_index + annmb.num_neurons1 * annmb.dim + j], e_b0_n1);
      atomicAdd(&g_grad_sum[t1_net_index + annmb.num_neurons1 * annmb.dim + annmb.num_neurons1 + j], e_w1_n1);
      if (virial_nums > 0) {
        double v_w1_n1 = per_Nc_v * (sum_dfeat_w1b0[0] * diff[0] + sum_dfeat_w1b0[1] * diff[1] 
        + sum_dfeat_w1b0[2] * diff[2] + sum_dfeat_w1b0[3] * diff[3] 
        + sum_dfeat_w1b0[4] * diff[4] + sum_dfeat_w1b0[5] * diff[5]);
        double v_b0_n1 = per_Nc_v * (sum_dfeat_w1b0[6] * diff[0] + sum_dfeat_w1b0[7] * diff[1] 
        + sum_dfeat_w1b0[8] * diff[2] + sum_dfeat_w1b0[9] * diff[3] 
        + sum_dfeat_w1b0[10] * diff[4] + sum_dfeat_w1b0[11] * diff[5]); 
        atomicAdd(&g_grad_sum[t1_net_index + b0_index + j], -v_w1_n1);
        atomicAdd(&g_grad_sum[t1_net_index + w0_index + j], -v_b0_n1);
      }
    }
    double e_b1_n1 = e_wb_grad[annmb.num_neurons1 * annmb.dim + annmb.num_neurons1 + annmb.num_neurons1] * per_Nc_e;
    atomicAdd(&g_grad_sum[t1_net_index + annmb.num_neurons1 * annmb.dim + annmb.num_neurons1 + annmb.num_neurons1], e_b1_n1);

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;  
          int q_c_index = n1 +
                        N * i1 +
                        N * max_NN_radial * n +
                        N * max_NN_radial * (paramb.n_max_radial + 1) * k;
          double q_c_scaler = g_q_c[q_c_index] * g_q_scaler[n];
          double f_c_n1_sum = feat_x_sum[n] * q_c_scaler * dx_n1 + feat_y_sum[n] * q_c_scaler * dy_n1 + feat_z_sum[n] * q_c_scaler * dz_n1;
          int grad_c_index = c_index + annmb.num_ann;
          atomicAdd(&g_grad_sum[grad_c_index], f_c_n1_sum * per_Nc);
          if (virial_nums > 0) {
            double v_c_n1_sum = q_c_scaler * per_Nc_v * (feat_123_xx_sum[n] * diff[0] 
            + feat_123_yy_sum[n] * diff[1] 
            + feat_123_zz_sum[n] * diff[2] 
            + feat_123_xy_sum[n] * diff[3] 
            + feat_123_yz_sum[n] * diff[4] 
            + feat_123_zx_sum[n] * diff[5]);
            atomicAdd(&g_grad_sum[grad_c_index], -v_c_n1_sum);
          }
        }
      }
    }
  }
}

static __global__ void find_force_radial(
  const bool is_dipole,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP3::ParaMB paramb,
  const NEP3::ANN annmb,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x12,
  const double* __restrict__ g_y12,
  const double* __restrict__ g_z12,
  const double* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int neighbor_number = g_NN[n1];
    double s_virial_xx = 0.0;
    double s_virial_yy = 0.0;
    double s_virial_zz = 0.0;
    double s_virial_xy = 0.0;
    double s_virial_yz = 0.0;
    double s_virial_zx = 0.0;
    int t1 = g_type[n1];

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;
      double fc12, fcp12;
      double rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      double f12[3] = {0.0};
      double tmp_xyz[3] = {d12inv * r12[0], d12inv * r12[1], d12inv * r12[2]};
      double feat_x[MAX_NUM_N] = {0.0};
      double feat_y[MAX_NUM_N] = {0.0};
      double feat_z[MAX_NUM_N] = {0.0};

      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        feat_x[n] = gnp12 * tmp_xyz[0];
        feat_y[n] = gnp12 * tmp_xyz[1];
        feat_z[n] = gnp12 * tmp_xyz[2]; 
        f12[0] += g_Fp[n1 + n * N] * feat_x[n];
        f12[1] += g_Fp[n1 + n * N] * feat_y[n];
        f12[2] += g_Fp[n1 + n * N] * feat_z[n];
      }

      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);

      if (is_dipole) {
        double r12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        s_virial_xx -= r12_square * f12[0];
        s_virial_yy -= r12_square * f12[1];
        s_virial_zz -= r12_square * f12[2];
      } else {
        s_virial_xx -= r12[0] * f12[0];
        s_virial_yy -= r12[1] * f12[1];
        s_virial_zz -= r12[2] * f12[2];
      }
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
  const bool is_dipole,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP3::ParaMB paramb,
  const NEP3::ANN annmb,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x12,
  const double* __restrict__ g_y12,
  const double* __restrict__ g_z12,
  const double* __restrict__ g_Fp,
  const double* __restrict__ g_sum_fxyz,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {

    double s_virial_xx = 0.0;
    double s_virial_yy = 0.0;
    double s_virial_zz = 0.0;
    double s_virial_xy = 0.0;
    double s_virial_yz = 0.0;
    double s_virial_zx = 0.0;

    double Fp[MAX_DIM_ANGULAR] = {0.0};
    double sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }
    int neighbor_number = g_NN[n1];
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double fc12, fcp12;
      int t2 = g_type[n2];
      double rc = paramb.rc_angular;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_angular_factor,
          rc);
      }
      double rcinv = 1.0 / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      double f12[3] = {0.0};

      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
      }

      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);

      if (is_dipole) {
        double r12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
        s_virial_xx -= r12_square * f12[0];
        s_virial_yy -= r12_square * f12[1];
        s_virial_zz -= r12_square * f12[2];
      } else {
        s_virial_xx -= r12[0] * f12[0];
        s_virial_yy -= r12[1] * f12[1];
        s_virial_zz -= r12[2] * f12[2];
      }
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

static __global__ void find_force_ZBL(
  const int N,
  const NEP3::ParaMB paramb,
  const NEP3::ZBL zbl,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x12,
  const double* __restrict__ g_y12,
  const double* __restrict__ g_z12,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    double s_pe = 0.0;
    double s_virial_xx = 0.0;
    double s_virial_yy = 0.0;
    double s_virial_zz = 0.0;
    double s_virial_xy = 0.0;
    double s_virial_yz = 0.0;
    double s_virial_zx = 0.0;
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1]; // starting from 1
    double pow_zi = pow(double(zi), 0.23);
    int neighbor_number = g_NN[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      double r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double d12inv = 1.0 / d12;
      double f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2]; // starting from 1
      double a_inv = (pow_zi + pow(double(zj), 0.23)) * 2.134563;
      double zizj = K_C_SP * zi * zj;
      if (zbl.flexibled) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        double ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        double rc_inner = zbl.rc_inner;
        double rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = rc_outer * 0.5;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      double f2 = fp * d12inv * 0.5;
      double f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};

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
      s_pe += f * 0.5f;
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

void NEP3::find_force(
  Parameters& para,
  const double* parameters,
  bool require_grad,
  std::vector<Dataset>& dataset,
  bool calculate_q_scaler,
  bool calculate_neighbor,
  int device_in_this_iter)
{

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(cudaSetDevice(device_id));
    CHECK(cudaMemset(nep_data[device_id].Fp2.data(), 0, nep_data[device_id].Fp2.size() * sizeof(double)));
    // CHECK(cudaMemset(dataset[device_id].gradients.Fp_wb.data(), 0, dataset[device_id].gradients.Fp_wb.size() * sizeof(double)));
    CHECK(cudaMemset(dataset[device_id].gradients.grad_sum.data(), 0, dataset[device_id].gradients.grad_sum.size() * sizeof(double)));
    CHECK(cudaMemset(dataset[device_id].gradients.E_wb_grad.data(), 0, dataset[device_id].gradients.E_wb_grad.size() * sizeof(double)));
    nep_data[device_id].parameters.copy_from_host(parameters);
    update_potential(para, nep_data[device_id].parameters.data(), annmb[device_id]);
  }

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(cudaSetDevice(device_id));
    const int block_size = 32;
    const int grid_size = (dataset[device_id].N - 1) / block_size + 1;

    if (calculate_neighbor) {
      gpu_find_neighbor_list<<<dataset[device_id].Nc, 256>>>(
        paramb,
        dataset[device_id].N,
        dataset[device_id].Na.data(),
        dataset[device_id].Na_sum.data(),
        para.use_typewise_cutoff,
        dataset[device_id].type.data(),
        para.rc_radial,
        para.rc_angular,
        dataset[device_id].box.data(),
        dataset[device_id].box_original.data(),
        dataset[device_id].num_cell.data(),
        dataset[device_id].r.data(),
        dataset[device_id].r.data() + dataset[device_id].N,
        dataset[device_id].r.data() + dataset[device_id].N * 2,
        nep_data[device_id].NN_radial.data(),
        nep_data[device_id].NL_radial.data(),
        nep_data[device_id].NN_angular.data(),
        nep_data[device_id].NL_angular.data(),
        nep_data[device_id].x12_radial.data(),
        nep_data[device_id].y12_radial.data(),
        nep_data[device_id].z12_radial.data(),
        nep_data[device_id].x12_angular.data(),
        nep_data[device_id].y12_angular.data(),
        nep_data[device_id].z12_angular.data());
      CUDA_CHECK_KERNEL
    }
    if (require_grad) {
      find_descriptors_radial<true><<<grid_size, block_size>>>(
        dataset[device_id].N,
        dataset[device_id].max_NN_radial,
        nep_data[device_id].NN_radial.data(),
        nep_data[device_id].NL_radial.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_radial.data(),
        nep_data[device_id].y12_radial.data(),
        nep_data[device_id].z12_radial.data(),
        nep_data[device_id].descriptors.data(),
        dataset[device_id].gradients.q_c.data());
      CUDA_CHECK_KERNEL

      find_descriptors_angular<<<grid_size, block_size>>>(
        dataset[device_id].N,
        nep_data[device_id].NN_angular.data(),
        nep_data[device_id].NL_angular.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_angular.data(),
        nep_data[device_id].y12_angular.data(),
        nep_data[device_id].z12_angular.data(),
        nep_data[device_id].descriptors.data(),
        nep_data[device_id].sum_fxyz.data());
      CUDA_CHECK_KERNEL
    } else {
      find_descriptors_radial<false><<<grid_size, block_size>>>(
        dataset[device_id].N,
        dataset[device_id].max_NN_radial,
        nep_data[device_id].NN_radial.data(),
        nep_data[device_id].NL_radial.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_radial.data(),
        nep_data[device_id].y12_radial.data(),
        nep_data[device_id].z12_radial.data(),
        nep_data[device_id].descriptors.data());
      CUDA_CHECK_KERNEL

      find_descriptors_angular<<<grid_size, block_size>>>(
        dataset[device_id].N,
        nep_data[device_id].NN_angular.data(),
        nep_data[device_id].NL_angular.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].x12_angular.data(),
        nep_data[device_id].y12_angular.data(),
        nep_data[device_id].z12_angular.data(),
        nep_data[device_id].descriptors.data(),
        nep_data[device_id].sum_fxyz.data());
      CUDA_CHECK_KERNEL
    }

    if (calculate_q_scaler) {
      find_max_min<<<annmb[device_id].dim, 1024>>>(
        dataset[device_id].N,
        nep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data());
      CUDA_CHECK_KERNEL
    }

    zero_force<<<grid_size, block_size>>>(
      dataset[device_id].N,
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2,
      dataset[device_id].virial.data(),
      dataset[device_id].virial.data() + dataset[device_id].N,
      dataset[device_id].virial.data() + dataset[device_id].N * 2);
    CUDA_CHECK_KERNEL

    if (require_grad) {
      if (para.train_mode == 2) {
        apply_ann_pol<true><<<grid_size, block_size>>>(
          dataset[device_id].N,
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].descriptors.data(),
          para.q_scaler_gpu[device_id].data(),
          dataset[device_id].virial.data(),
          nep_data[device_id].Fp.data(),
          nep_data[device_id].Fp2.data(),
          dataset[device_id].gradients.Fp_wb.data(),
          dataset[device_id].gradients.E_wb_grad.data());
        CUDA_CHECK_KERNEL
      } else if (para.train_mode == 3) {
        apply_ann_temperature<true><<<grid_size, block_size>>>(
          dataset[device_id].N,
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].descriptors.data(),
          para.q_scaler_gpu[device_id].data(),
          dataset[device_id].temperature_ref_gpu.data(),
          dataset[device_id].energy.data(),
          nep_data[device_id].Fp.data(),
          nep_data[device_id].Fp2.data(),
          dataset[device_id].gradients.Fp_wb.data(),
          dataset[device_id].gradients.E_wb_grad.data());
        CUDA_CHECK_KERNEL
      } else {
        apply_ann<true><<<grid_size, block_size>>>(
          dataset[device_id].N,
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].descriptors.data(),
          para.q_scaler_gpu[device_id].data(),
          dataset[device_id].energy.data(),
          nep_data[device_id].Fp.data(),
          nep_data[device_id].Fp2.data(),
          dataset[device_id].gradients.Fp_wb.data(),
          dataset[device_id].gradients.E_wb_grad.data());
        CUDA_CHECK_KERNEL
        // std::vector<double> Fp_wb_host(dataset[device_id].N * para.number_of_variables_ann * para.dim);
        // CHECK(cudaMemcpy(Fp_wb_host.data(), dataset[device_id].gradients.Fp_wb.data(), dataset[device_id].N * para.number_of_variables_ann * para.dim * sizeof(double), cudaMemcpyDeviceToHost));
        // for (int i = 0; i < dataset[device_id].N; ++i) {
        //   for (int j = 0; j < para.number_of_variables_ann; ++j) {
        //     for (int k = 0; k < para.dim; ++k) {
        //       printf("Fp_wb[%d][%d][%d] = %f\n", i, j, k, Fp_wb_host[i * para.number_of_variables_ann * para.dim + j * para.dim + k]);
        //     }
        //   }
        // }
        // std::vector<double> E_wb_grad_host(dataset[device_id].N * para.number_of_variables_ann);
        // CHECK(cudaMemcpy(E_wb_grad_host.data(), dataset[device_id].gradients.E_wb_grad.data(), dataset[device_id].N * para.number_of_variables_ann * sizeof(double), cudaMemcpyDeviceToHost));
        // for (int i = 0; i < dataset[device_id].N; ++i) {
        //   for (int j = 0; j < para.number_of_variables_ann; ++j) {
        //     printf("E_wb_grad[%d][%d] = %f\n", i, j, E_wb_grad_host[i * para.number_of_variables_ann + j]);
        //   }
        // }
      }
    } else {
      if (para.train_mode == 2) {
        apply_ann_pol<false><<<grid_size, block_size>>>(
          dataset[device_id].N,
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].descriptors.data(),
          para.q_scaler_gpu[device_id].data(),
          dataset[device_id].virial.data(),
          nep_data[device_id].Fp.data());
        CUDA_CHECK_KERNEL
      } else if (para.train_mode == 3) {
        apply_ann_temperature<false><<<grid_size, block_size>>>(
          dataset[device_id].N,
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].descriptors.data(),
          para.q_scaler_gpu[device_id].data(),
          dataset[device_id].temperature_ref_gpu.data(),
          dataset[device_id].energy.data(),
          nep_data[device_id].Fp.data());
        CUDA_CHECK_KERNEL
      } else {
        apply_ann<false><<<grid_size, block_size>>>(
          dataset[device_id].N,
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].descriptors.data(),
          para.q_scaler_gpu[device_id].data(),
          dataset[device_id].energy.data(),
          nep_data[device_id].Fp.data());
        CUDA_CHECK_KERNEL
      }
    }
    bool is_dipole = para.train_mode == 1;
    find_force_radial<<<grid_size, block_size>>>(
      is_dipole,
      dataset[device_id].N,
      nep_data[device_id].NN_radial.data(),
      nep_data[device_id].NL_radial.data(),
      paramb,
      annmb[device_id],
      dataset[device_id].type.data(),
      nep_data[device_id].x12_radial.data(),
      nep_data[device_id].y12_radial.data(),
      nep_data[device_id].z12_radial.data(),
      nep_data[device_id].Fp.data(),
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2,
      dataset[device_id].virial.data());
    CUDA_CHECK_KERNEL

    gpu_sum_pe_error<<<dataset[device_id].Nc, 256, sizeof(double) * 256>>>(
      dataset[device_id].Na.data(),
      dataset[device_id].Na_sum.data(),
      dataset[device_id].energy.data(),
      dataset[device_id].energy_ref_gpu.data(),
      dataset[device_id].diff_gpu_e.data(),
      dataset[device_id].error_gpu.data());
    CHECK(cudaMemcpy(dataset[device_id].error_cpu_e.data(), dataset[device_id].error_gpu.data(), dataset[device_id].Nc * sizeof(double), cudaMemcpyDeviceToHost));

    double shear_weight = (para.train_mode != 1) ? (require_grad ? para.lambda_shear * para.lambda_shear : 1.0) : 0.0;
    gpu_sum_virial_error<<<dataset[device_id].Nc, 256, sizeof(double) * 256 * 6>>>(
      dataset[device_id].N,
      shear_weight,
      dataset[device_id].Na.data(),
      dataset[device_id].Na_sum.data(),
      dataset[device_id].virial.data(),
      dataset[device_id].virial_ref_gpu.data(),
      dataset[device_id].diff_gpu_v.data(),
      dataset[device_id].error_gpu.data());
    CHECK(cudaMemcpy(dataset[device_id].error_cpu_v.data(), dataset[device_id].error_gpu.data(), dataset[device_id].Nc * sizeof(double), cudaMemcpyDeviceToHost));
    int virial_nums = 0;
    for (int n = 0; n < dataset[device_id].Nc; ++n) {
      if (dataset[device_id].has_virial[n]) {
        virial_nums += (para.train_mode != 1) ? 6 : 3;
      }
    }

    if (require_grad) {
      compute_grad_radial<<<grid_size, block_size>>>(
        dataset[device_id].N,
        nep_data[device_id].NN_radial.data(),
        nep_data[device_id].NL_radial.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].Na.data(),
        dataset[device_id].Nc,
        para.lambda_e,
        para.lambda_f,
        para.lambda_v,
        virial_nums,
        dataset[device_id].type_weight_gpu.data(),
        para.force_delta,
        dataset[device_id].batch_idx.data(),
        dataset[device_id].type.data(),
        nep_data[device_id].x12_radial.data(),
        nep_data[device_id].y12_radial.data(),
        nep_data[device_id].z12_radial.data(),
        nep_data[device_id].Fp.data(),
        dataset[device_id].gradients.E_wb_grad.data(),
        dataset[device_id].diff_gpu_e.data(),
        dataset[device_id].diff_gpu_v.data(),
        dataset[device_id].force_ref_gpu.data(),
        dataset[device_id].force_ref_gpu.data() + dataset[device_id].N,
        dataset[device_id].force_ref_gpu.data() + dataset[device_id].N * 2,
        dataset[device_id].type_weight_gpu.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].max_NN_radial,
        para.q_scaler_gpu[device_id].data(),
        nep_data[device_id].Fp2.data(),
        dataset[device_id].gradients.q_c.data(),
        dataset[device_id].gradients.Fp_wb.data(),
        dataset[device_id].gradients.grad_sum.data());
      CUDA_CHECK_KERNEL
      //   std::vector<double>grad_c_sum(para.number_of_variables);
      // CHECK(cudaMemcpy(grad_c_sum.data(), dataset[device_id].gradients.grad_sum.data(), para.number_of_variables * sizeof(double), cudaMemcpyDeviceToHost));
      // for (int j = 0; j < para.number_of_variables; ++j) {
      //   printf("%d %f\n", j, grad_c_sum[j]);
      // }
    }

    find_force_angular<<<grid_size, block_size>>>(
      is_dipole,
      dataset[device_id].N,
      nep_data[device_id].NN_angular.data(),
      nep_data[device_id].NL_angular.data(),
      paramb,
      annmb[device_id],
      dataset[device_id].type.data(),
      nep_data[device_id].x12_angular.data(),
      nep_data[device_id].y12_angular.data(),
      nep_data[device_id].z12_angular.data(),
      nep_data[device_id].Fp.data(),
      nep_data[device_id].sum_fxyz.data(),
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2,
      dataset[device_id].virial.data());
    CUDA_CHECK_KERNEL

    if (zbl.enabled) {
      find_force_ZBL<<<grid_size, block_size>>>(
        dataset[device_id].N,
        paramb,
        zbl,
        nep_data[device_id].NN_angular.data(),
        nep_data[device_id].NL_angular.data(),
        dataset[device_id].type.data(),
        nep_data[device_id].x12_angular.data(),
        nep_data[device_id].y12_angular.data(),
        nep_data[device_id].z12_angular.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data(),
        dataset[device_id].energy.data());
      CUDA_CHECK_KERNEL
    } 
  }
}
