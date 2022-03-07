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
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "dataset.cuh"
#include "mic.cuh"
#include "nep4.cuh"
#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_utilities.cuh"

static __global__ void gpu_find_neighbor_list(
  const int N,
  const int* Na,
  const int* Na_sum,
  const float rc2_angular,
  const float* __restrict__ g_box,
  const float* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
  const float* x,
  const float* y,
  const float* z,
  int* NN_angular,
  int* NL_angular,
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
    float x1 = x[n1];
    float y1 = y[n1];
    float z1 = z[n1];
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
            dev_apply_mic(box, x12, y12, z12);
            float distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            if (distance_square < rc2_angular) {
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
    NN_angular[n1] = count_angular;
  }
}

static __global__ void find_descriptors_angular(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP4::Para para,
  const NEP4::ANN ann,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_q,
  float* g_s)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_DIM_ANGULAR] = {0.0f};

    for (int n = 0; n <= para.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < neighbor_number; ++i1) {
        int index = n1 + N * i1;
        int n2 = g_NL[n1 + N * i1];
        float x12 = g_x12[index];
        float y12 = g_y12[index];
        float z12 = g_z12[index];
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        find_fc(para.rc_angular, para.rcinv_angular, d12, fc12);
        int t2 = g_type[n2];
        float fn12[MAX_NUM_N];
        find_fn(para.basis_size_radial, para.rcinv_angular, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= para.basis_size_radial; ++k) {
          int c_index = (n * (para.basis_size_radial + 1) + k) * para.num_types_sq;
          c_index += t1 * para.num_types + t2;
          gn12 += fn12[k] * ann.c[c_index];
        }
        accumulate_s(d12, x12, y12, z12, gn12, s);
      }
      find_q(para.n_max_angular + 1, n, s, q);
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_s[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
      }
    }

    for (int n = 0; n <= para.n_max_angular; ++n) {
      for (int l = 0; l < para.L_max; ++l) {
        int ln = l * (para.n_max_angular + 1) + n;
        g_q[n1 + ln * N] = q[ln];
      }
    }
  }
}

NEP4::NEP4(char* input_dir, Parameters& para, int N, int N_times_max_NN_angular)
{
  nep_para.rc_angular = para.rc_angular;
  nep_para.rcinv_angular = 1.0f / nep_para.rc_angular;
  ann.dim = para.dim;
  ann.num_neurons1 = para.num_neurons1;
  nep_para.num_types = para.num_types;
  ann.num_para = para.number_of_variables_ann + para.number_of_variables_descriptor;
  gnn.num_para = para.number_of_variables_gnn;
  nep_para.n_max_angular = para.n_max_angular;
  nep_para.L_max = para.L_max;

  nep_para.basis_size_radial = para.basis_size_radial;
  nep_para.num_types_sq = para.num_types * para.num_types;

  zbl.enabled = para.enable_zbl;
  zbl.rc_inner = para.zbl_rc_inner;
  zbl.rc_outer = para.zbl_rc_outer;
  for (int n = 0; n < para.atomic_numbers.size(); ++n) {
    zbl.atomic_numbers[n] = para.atomic_numbers[n];
  }

  nep_data.NN_angular.resize(N);
  nep_data.NL_angular.resize(N_times_max_NN_angular);
  nep_data.x12_angular.resize(N_times_max_NN_angular);
  nep_data.y12_angular.resize(N_times_max_NN_angular);
  nep_data.z12_angular.resize(N_times_max_NN_angular);
  nep_data.dq_dx.resize(N_times_max_NN_angular * ann.dim);
  nep_data.dq_dy.resize(N_times_max_NN_angular * ann.dim);
  nep_data.dq_dz.resize(N_times_max_NN_angular * ann.dim);
  nep_data.q.resize(N * ann.dim);
  nep_data.gnn_descriptors.resize(N * ann.dim);
  nep_data.gnn_messages.resize(N * ann.dim);
  nep_data.dU_dq.resize(N * ann.dim);
  nep_data.s.resize(N * (nep_para.n_max_angular + 1) * NUM_OF_ABC);
  nep_data.parameters.resize(ann.num_para + gnn.num_para);
}

void NEP4::update_potential(const float* parameters, ANN& ann, GNN& gnn)
{
  // ann
  ann.w0 = parameters;
  ann.b0 = ann.w0 + ann.num_neurons1 * ann.dim;
  ann.w1 = ann.b0 + ann.num_neurons1;
  ann.b1 = ann.w1 + ann.num_neurons1;
  ann.c = ann.b1 + 1;
  // gnn
  gnn.theta = parameters + ann.num_para;
}

static void __global__ find_q_scaler(const int N, const float* g_q, float* g_q_scaler)
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
    g_q_scaler[bid] = min(g_q_scaler[bid], 1.0f / (s_max[0] - s_min[0]));
  }
}

static __global__ void
apply_q_scaler(const int N, const NEP4::ANN ann, const float* __restrict__ g_q_scaler, float* g_q)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    for (int d = 0; d < ann.dim; ++d) {
      g_q[n1 + d * N] *= g_q_scaler[d];
    }
  }
}

// Apply the message passing aggregation between neighbors, A*q_theta
static __global__ void apply_gnn_message_passing(
  const int N,
  const NEP4::Para para,
  const NEP4::ANN ann,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_messages,
  const int* g_NN,
  const int* g_NL,
  float* gnn_descriptors)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int num_neighbors_of_n1 = g_NN[n1];
    const int F = ann.dim; // dimension of q_out, for now dim_out = dim_in.
    for (int nu = 0; nu < F; nu++) {
      float q_out_nu = g_messages[n1 + nu * N]; // fc(r_ii) = 1
      // TODO perhaps normalize weights? Compare Kipf, Welling et al. (2016)
      for (int j = 0; j < num_neighbors_of_n1; ++j) {
        int index = n1 + N * j;
        int n2 = g_NL[index];
        float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
        float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        float fc12;
        find_fc(para.rc_angular, para.rcinv_angular, d12, fc12);
        q_out_nu += fc12 * g_messages[n2 + nu * N];
      }
      gnn_descriptors[n1 + nu * N] = tanh(q_out_nu);
    }
  }
}

// Precompute messages q*theta for all descriptors
static __global__ void apply_gnn_compute_messages(
  const int N,
  const NEP4::ANN ann,
  const NEP4::GNN gnn,
  const float* __restrict__ g_q,
  const int* g_NN,
  const int* g_NL,
  float* gnn_messages)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    const int F = ann.dim; // dimension of q_out, for now dim_out = dim_in.
    for (int nu = 0; nu < F; nu++) {
      float q_theta_nu = 0.0f;
      for (int gamma = 0; gamma < ann.dim; gamma++) {
        q_theta_nu += g_q[n1 + gamma * N] * gnn.theta[gamma + ann.dim * nu];
      }
      gnn_messages[n1 + nu * N] = q_theta_nu;
    }
  }
}

static __global__ void apply_ann(
  const int N, const NEP4::ANN ann, const float* __restrict__ g_q, float* g_pe, float* g_dU_dq)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < ann.dim; ++d) {
      q[d] = g_q[n1 + d * N];
    }
    float U = 0.0f, dU_dq[MAX_DIM] = {0.0f};
    apply_ann_one_layer(ann.dim, ann.num_neurons1, ann.w0, ann.b0, ann.w1, ann.b1, q, U, dU_dq);
    g_pe[n1] = U;
    for (int d = 0; d < ann.dim; ++d) {
      g_dU_dq[n1 + d * N] = dU_dq[d];
    }
  }
}

static __global__ void zero_force(const int N, float* g_fx, float* g_fy, float* g_fz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_fx[n1] = 0.0f;
    g_fy[n1] = 0.0f;
    g_fz[n1] = 0.0f;
  }
}

static __global__ void find_dq_dr(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP4::Para para,
  const NEP4::ANN ann,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_s,
  float* g_dq_dx,
  float* g_dq_dy,
  float* g_dq_dz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    float s[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < (para.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      s[d] = g_s[d * N + n1];
    }
    int neighbor_number = g_NN[n1];
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      find_fc_and_fcp(para.rc_angular, para.rcinv_angular, d12, fc12, fcp12);
      int t2 = g_type[n2];
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(para.basis_size_radial, para.rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= para.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= para.basis_size_radial; ++k) {
          int c_index = (n * (para.basis_size_radial + 1) + k) * para.num_types_sq;
          c_index += t1 * para.num_types + t2;
          gn12 += fn12[k] * ann.c[c_index];
          gnp12 += fnp12[k] * ann.c[c_index];
        }
        find_dq_dr(
          N * (i1 * ann.dim + n) + n1, N * (para.n_max_angular + 1), n, para.n_max_angular + 1, d12,
          r12, gn12, gnp12, s, g_dq_dx, g_dq_dy, g_dq_dz);
      }
    }
  }
}

// TODO: change this function to complete NEP4
static __global__ void find_force_angular(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP4::Para para,
  const NEP4::ANN ann,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_dU_dq,
  const float* __restrict__ g_s,
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

    float dU_dq[MAX_DIM_ANGULAR] = {0.0f};
    float s[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < (para.n_max_angular + 1) * para.L_max; ++d) {
      dU_dq[d] = g_dU_dq[d * N + n1];
    }
    for (int d = 0; d < (para.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      s[d] = g_s[d * N + n1];
    }
    int neighbor_number = g_NN[n1];
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      find_fc_and_fcp(para.rc_angular, para.rcinv_angular, d12, fc12, fcp12);
      int t2 = g_type[n2];
      float f12[3] = {0.0f};
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(para.basis_size_radial, para.rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= para.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= para.basis_size_radial; ++k) {
          int c_index = (n * (para.basis_size_radial + 1) + k) * para.num_types_sq;
          c_index += t1 * para.num_types + t2;
          gn12 += fn12[k] * ann.c[c_index];
          gnp12 += fnp12[k] * ann.c[c_index];
        }
        accumulate_f12(n, para.n_max_angular + 1, d12, r12, gn12, gnp12, dU_dq, s, f12);
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

static __global__ void find_force_ZBL(
  const int N,
  const NEP4::ZBL zbl,
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
  float* g_pe)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    float s_pe = 0.0f;
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    float zi = zbl.atomic_numbers[g_type[n1]];
    float pow_zi = pow(zi, 0.23f);
    int neighbor_number = g_NN[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      float zj = zbl.atomic_numbers[g_type[n2]];
      float a_inv = (pow_zi + pow(zj, 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      find_f_and_fp_zbl(zizj, a_inv, zbl.rc_inner, zbl.rc_outer, d12, d12inv, f, fp);
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

void NEP4::find_force(
  Parameters& para, const float* parameters, Dataset& dataset, bool calculate_q_scaler)
{
  nep_data.parameters.copy_from_host(parameters);
  update_potential(nep_data.parameters.data(), ann, gnn);
  float rc2_angular = para.rc_angular * para.rc_angular;

  gpu_find_neighbor_list<<<dataset.Nc, 256>>>(
    dataset.N, dataset.Na.data(), dataset.Na_sum.data(), rc2_angular, dataset.box.data(),
    dataset.box_original.data(), dataset.num_cell.data(), dataset.r.data(),
    dataset.r.data() + dataset.N, dataset.r.data() + dataset.N * 2, nep_data.NN_angular.data(),
    nep_data.NL_angular.data(), nep_data.x12_angular.data(), nep_data.y12_angular.data(),
    nep_data.z12_angular.data());
  CUDA_CHECK_KERNEL
  const int block_size = 32;
  const int grid_size = (dataset.N - 1) / block_size + 1;

  find_descriptors_angular<<<grid_size, block_size>>>(
    dataset.N, nep_data.NN_angular.data(), nep_data.NL_angular.data(), nep_para, ann,
    dataset.type.data(), nep_data.x12_angular.data(), nep_data.y12_angular.data(),
    nep_data.z12_angular.data(), nep_data.q.data(), nep_data.s.data());
  CUDA_CHECK_KERNEL

  find_dq_dr<<<grid_size, block_size>>>(
    dataset.N, nep_data.NN_angular.data(), nep_data.NL_angular.data(), nep_para, ann,
    dataset.type.data(), nep_data.x12_angular.data(), nep_data.y12_angular.data(),
    nep_data.z12_angular.data(), nep_data.s.data(), nep_data.dq_dx.data(), nep_data.dq_dy.data(),
    nep_data.dq_dz.data());
  CUDA_CHECK_KERNEL

  if (calculate_q_scaler) {
    find_q_scaler<<<ann.dim, 1024>>>(dataset.N, nep_data.q.data(), para.q_scaler_gpu.data());
    CUDA_CHECK_KERNEL
  }

  apply_q_scaler<<<grid_size, block_size>>>(
    dataset.N, ann, para.q_scaler_gpu.data(), nep_data.q.data());
  CUDA_CHECK_KERNEL

  apply_gnn_compute_messages<<<(dataset.N - 1) / 64 + 1, 64>>>(
    dataset.N, ann, gnn, nep_data.q.data(), nep_data.NN_angular.data(), nep_data.NL_angular.data(),
    nep_data.gnn_messages.data());
  CUDA_CHECK_KERNEL

  apply_gnn_message_passing<<<(dataset.N - 1) / 64 + 1, 64>>>(
    dataset.N, nep_para, ann, nep_data.x12_angular.data(), nep_data.y12_angular.data(),
    nep_data.z12_angular.data(), nep_data.gnn_messages.data(), nep_data.NN_angular.data(),
    nep_data.NL_angular.data(), nep_data.gnn_descriptors.data());
  CUDA_CHECK_KERNEL

  apply_ann<<<grid_size, block_size>>>(
    dataset.N, ann, nep_data.gnn_descriptors.data(), dataset.energy.data(), nep_data.dU_dq.data());
  CUDA_CHECK_KERNEL

  zero_force<<<grid_size, block_size>>>(
    dataset.N, dataset.force.data(), dataset.force.data() + dataset.N,
    dataset.force.data() + dataset.N * 2);
  CUDA_CHECK_KERNEL

  // TODO: change this function to complete NEP4
  find_force_angular<<<grid_size, block_size>>>(
    dataset.N, nep_data.NN_angular.data(), nep_data.NL_angular.data(), nep_para, ann,
    dataset.type.data(), nep_data.x12_angular.data(), nep_data.y12_angular.data(),
    nep_data.z12_angular.data(), nep_data.dU_dq.data(), nep_data.s.data(), dataset.force.data(),
    dataset.force.data() + dataset.N, dataset.force.data() + dataset.N * 2, dataset.virial.data());
  CUDA_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL<<<grid_size, block_size>>>(
      dataset.N, zbl, nep_data.NN_angular.data(), nep_data.NL_angular.data(), dataset.type.data(),
      nep_data.x12_angular.data(), nep_data.y12_angular.data(), nep_data.z12_angular.data(),
      dataset.force.data(), dataset.force.data() + dataset.N, dataset.force.data() + dataset.N * 2,
      dataset.virial.data(), dataset.energy.data());
    CUDA_CHECK_KERNEL
  }
}
