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
1. The Gradient-optimized Neuroevolution Potential (GNEP)
Ref: Hongfu Huang, Junhao Peng, Kaiqi Li, Jian Zhou, Zhimei Sun, 
Efficient GPU-Accelerated Training of a Neuroevolution Potential with Analytical Gradients,
arXiv:2507.00528.
2. The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "dataset.cuh"
#include "mic.cuh"
#include "gnep.cuh"
#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/gnep_utilities.cuh"

static __global__ void gpu_find_neighbor_list(
  const GNEP::ParaMB paramb,
  const int N,
  const int* Na,
  const int* Na_sum,
  const bool use_typewise_cutoff,
  const int* g_type,
  const float g_rc_radial,
  const float g_rc_angular,
  const float* __restrict__ g_box,
  const float* __restrict__ g_box_original,
  const int* __restrict__ g_num_cell,
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
  for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x) {
    const float* __restrict__ box = g_box + 18 * blockIdx.x;
    const float* __restrict__ box_original = g_box_original + 9 * blockIdx.x;
    const int* __restrict__ num_cell = g_num_cell + 3 * blockIdx.x;
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
            dev_apply_mic(box, x12, y12, z12);
            float distance_square = x12 * x12 + y12 * y12 + z12 * z12;
            int t2 = g_type[n2];
            float rc_radial = g_rc_radial;
            float rc_angular = g_rc_angular;
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

static __global__ void find_descriptors_radial(
  const int N,
  const int max_NN_radial,
  const int* g_NN,
  const int* g_NL,
  const GNEP::ParaMB paramb,
  const GNEP::ANN annmb,
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
      int index = n1 + N * i1;
      int n2 = g_NL[index];
      float x12 = g_x12[index];
      float y12 = g_y12[index];
      float z12 = g_z12[index];
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);

      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gn12 += fn12[k] * annmb.c[c_index];
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
  const GNEP::ParaMB paramb,
  const GNEP::ANN annmb,
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
      float s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < neighbor_number; ++i1) {
        int index = n1 + N * i1;
        int n2 = g_NL[n1 + N * i1];
        float x12 = g_x12[index];
        float y12 = g_y12[index];
        float z12 = g_z12[index];
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = paramb.rc_angular;
        if (paramb.use_typewise_cutoff) {
          rc = min(
            (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
             COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
              paramb.typewise_cutoff_angular_factor,
            rc);
        }
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(paramb.L_max, paramb.n_max_angular + 1, n, s, q);
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
      }
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      for (int l = 0; l < paramb.L_max; ++l) {
        int ln = l * (paramb.n_max_angular + 1) + n;
        g_descriptors[n1 + ((paramb.n_max_radial + 1) + ln) * N] = q[ln];
      }
    }
  }
}

GNEP::GNEP(
  Parameters& para,
  int N,
  int N_times_max_NN_radial,
  int N_times_max_NN_angular,
  int deviceCount)
{
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
  paramb.N_times_max_NN_radial = N_times_max_NN_radial;
  paramb.N_times_max_NN_angular = N_times_max_NN_angular;
  paramb.dim_angular = para.dim_angular;

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

    gnep_data[device_id].NN_radial.resize(N);
    gnep_data[device_id].NN_angular.resize(N);
    gnep_data[device_id].NL_radial.resize(N_times_max_NN_radial);
    gnep_data[device_id].NL_angular.resize(N_times_max_NN_angular);
    gnep_data[device_id].x12_radial.resize(N_times_max_NN_radial);
    gnep_data[device_id].y12_radial.resize(N_times_max_NN_radial);
    gnep_data[device_id].z12_radial.resize(N_times_max_NN_radial);
    gnep_data[device_id].x12_angular.resize(N_times_max_NN_angular);
    gnep_data[device_id].y12_angular.resize(N_times_max_NN_angular);
    gnep_data[device_id].z12_angular.resize(N_times_max_NN_angular);
    gnep_data[device_id].descriptors.resize(N * annmb[device_id].dim);
    gnep_data[device_id].Fp.resize(N * annmb[device_id].dim);
    gnep_data[device_id].sum_fxyz.resize(N * (paramb.n_max_angular + 1) * NUM_OF_ABC);
    gnep_data[device_id].parameters.resize(annmb[device_id].num_para);
  }
}

void GNEP::update_potential(Parameters& para, const float* parameters, ANN& ann)
{
  const float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
    pointer += 1; // type bias stored in ann.w1[t]
  }
  ann.c = pointer;
}

void GNEP::initialize_gradients(Parameters& para, const int N)
{
  gradients.resize(N, para.number_of_variables, para.number_of_variables_ann, para.dim);
}

static void __global__ find_max_min(const int N, const float* g_q, float* g_s_max, float* g_s_min, float* g_q_scaler)
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
    g_s_max[bid] = max(g_s_max[bid], s_max[0]);
    g_s_min[bid] = min(g_s_min[bid], s_min[0]);
    g_q_scaler[bid] = 1.0f / (g_s_max[bid] - g_s_min[bid]);
  }
}

template <bool IsTraining>
static __global__ void apply_ann(
  const int N,
  const GNEP::ParaMB paramb,
  const GNEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  float* g_pe,
  float* g_Fp,
  float* g_Fp2 = nullptr,
  float* g_Fp_wb = nullptr,
  float* g_E_wb_grad = nullptr)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  int type = g_type[n1];

  if (n1 < N) {
    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    
    if (IsTraining) {
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
    g_pe[n1] = F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
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

static __global__ void gpu_sum_pe_error(
  int* g_Na, int* g_Na_sum, float* g_pe, float* g_pe_ref, float* diff_gpu, float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];  
  int N1 = g_Na_sum[bid]; 
  int N2 = N1 + Na;     
  extern __shared__ float s_pe[];
  s_pe[tid] = 0.0f;

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
    float diff = (s_pe[0] / Na - g_pe_ref[bid]);
    diff_gpu[bid] = diff;
    error_gpu[bid] = diff * diff;
  }
}

static __global__ void gpu_sum_virial_error(
  const int N,
  const float shear_weight,
  int* g_Na,
  int* g_Na_sum,
  float* g_virial,
  float* g_virial_ref,
  float* diff_gpu,
  float* error_gpu)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int Na = g_Na[bid];
  int N1 = g_Na_sum[bid];
  int N2 = N1 + Na;
  extern __shared__ float s_virial[];
  for (int d = 0; d < 6; ++d) {
    s_virial[d * blockDim.x + tid] = 0.0f; //size of s_virial is 6 * blockDim.x
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
    float error_sum = 0.0f;
    for (int d = 0; d < 6; ++d) {
      float diff = (s_virial[d * blockDim.x + 0] / Na - g_virial_ref[d * gridDim.x + bid]);
      error_sum += (d >= 3) ? (shear_weight * diff * diff) : (diff * diff);
      diff_gpu[bid * 6 + d] = (d >= 3) ? shear_weight * diff : diff;
    }
    error_gpu[bid] = error_sum;
  }
}

static __global__ void compute_grad_e_without_neighbor(
  const int N,
  const GNEP::ParaMB paramb,
  const GNEP::ANN annmb,
  const int Nc,
  const float lambda_e,
  const int* __restrict__ g_batch_idx,
  const int* __restrict__ g_type,
  float* __restrict__ g_E_wb_grad,
  const float* __restrict__ g_diff_gpu_e,
  const float* __restrict__ g_weight,
  float* g_grad_sum)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  const int w0_index = annmb.dim * annmb.num_neurons1;
  const int b0_index = w0_index + annmb.num_neurons1;
  if (n1 >= N) return;
  int batch_idx = g_batch_idx[n1];
  int t1 = g_type[n1];
  float weight = g_weight[batch_idx];
  const float per_Nc_e = g_diff_gpu_e[batch_idx] * weight * 2.0f * lambda_e / Nc;

  int t1_net_index = t1 * ((annmb.dim + 2) * annmb.num_neurons1 + 1);
  int n1_net_index = n1 * annmb.num_ann + t1_net_index;

  float* e_wb_grad = g_E_wb_grad + n1_net_index;

  for (int j = 0; j < annmb.num_neurons1; ++j) {
    for (int d = 0; d < annmb.dim; ++d) {
      float grad_w0_sum = per_Nc_e * e_wb_grad[j * annmb.dim + d];
      atomicAdd(&g_grad_sum[t1_net_index + j * annmb.dim + d], grad_w0_sum);
    }
    float grad_w1_sum = e_wb_grad[b0_index + j] * per_Nc_e;
    float grad_b0_sum = e_wb_grad[w0_index + j] * per_Nc_e;
    atomicAdd(&g_grad_sum[t1_net_index + b0_index + j], grad_w1_sum);
    atomicAdd(&g_grad_sum[t1_net_index + w0_index + j], grad_b0_sum);
  }
  float e_b1_n1 = e_wb_grad[annmb.num_neurons1 * annmb.dim + annmb.num_neurons1 + annmb.num_neurons1] * per_Nc_e;
  atomicAdd(&g_grad_sum[t1_net_index + annmb.num_neurons1 * annmb.dim + annmb.num_neurons1 + annmb.num_neurons1], e_b1_n1);
}

static __global__ void compute_grad_radial_NM(
  const int N,
  const int M,
  const int* __restrict__ g_NN,
  const int* __restrict__ g_NL,
  const int* __restrict__ g_NN_ang,
  const int* __restrict__ g_NL_ang,
  const GNEP::ParaMB paramb,
  const GNEP::ANN annmb,
  const int* __restrict__ g_Na,
  const int Nc,
  const float lambda_e,
  const float lambda_f,
  const float lambda_v,
  const int virial_nums,
  const int* __restrict__ g_has_virial,
  const float* __restrict__ g_type_weight,
  const float force_delta,
  const int* __restrict__ g_batch_idx,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_x12_ang,
  const float* __restrict__ g_y12_ang,
  const float* __restrict__ g_z12_ang,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_Fp2,
  const float* __restrict__ g_sum_fxyz,
  float* __restrict__ g_E_wb_grad,
  const float* __restrict__ g_diff_gpu_e,
  const float* __restrict__ g_diff_gpu_v,
  const float* __restrict__ g_fx_ref,
  const float* __restrict__ g_fy_ref,
  const float* __restrict__ g_fz_ref,
  const float* __restrict__ g_weight,
  const float* __restrict__ g_fx,
  const float* __restrict__ g_fy,
  const float* __restrict__ g_fz,
  const float* __restrict__ g_q_scaler,
  const float* __restrict__ g_ep_wb,
  float* g_grad_sum)
{
  int NM = N * M;
  int Idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int w0_index = annmb.dim * annmb.num_neurons1;
  const int b0_index = w0_index + annmb.num_neurons1;
  if (Idx >= NM) return;
  int n1 = Idx / M;
  int i1 = Idx % M;
  int neighbor_number = g_NN[n1];
  int neighbor_number_ang = g_NN_ang[n1];
  int batch_idx = g_batch_idx[n1];
  int Na = g_Na[batch_idx];
  bool no_neighbor_isolated_atom = ((Na == 1) && neighbor_number == 0);
  if (i1 >= neighbor_number && !no_neighbor_isolated_atom) return;
  int t1 = g_type[n1];
  float weight = g_weight[batch_idx];
  const float per_Nc_e = g_diff_gpu_e[batch_idx] * weight * 2.0f * lambda_e / Nc;

  // Cache frequently accessed type-dependent constants in registers
  const float typewise_cutoff_radial_factor = paramb.typewise_cutoff_radial_factor;
  const float typewise_cutoff_angular_factor = paramb.typewise_cutoff_angular_factor;
  const int z1 = paramb.atomic_numbers[t1];
  const float covalent_radius_t1 = COVALENT_RADIUS[z1];
  
  int t1_net_index = t1 * ((annmb.dim + 2) * annmb.num_neurons1 + 1);
  int n1_net_index = n1 * annmb.num_ann + t1_net_index;
  int n1_net_index_wb = n1 * annmb.num_ann * annmb.dim + t1_net_index * annmb.dim;
  float* e_wb_grad = g_E_wb_grad + n1_net_index;
  float grad_w0_sum, grad_w1_sum, grad_b0_sum;
  if (no_neighbor_isolated_atom) {
    if (i1 == 0) {
      for (int j = 0; j < annmb.num_neurons1; ++j) {
        for (int d = 0; d < annmb.dim; ++d) {
          grad_w0_sum = per_Nc_e * e_wb_grad[j * annmb.dim + d];
          atomicAdd(&g_grad_sum[t1_net_index + j * annmb.dim + d], grad_w0_sum);
        }
        grad_w1_sum = e_wb_grad[b0_index + j] * per_Nc_e;
        grad_b0_sum = e_wb_grad[w0_index + j] * per_Nc_e;
        atomicAdd(&g_grad_sum[t1_net_index + b0_index + j], grad_w1_sum);
        atomicAdd(&g_grad_sum[t1_net_index + w0_index + j], grad_b0_sum);
      }
      float e_b1_n1 = e_wb_grad[annmb.num_neurons1 * annmb.dim + annmb.num_neurons1 + annmb.num_neurons1] * per_Nc_e;
      atomicAdd(&g_grad_sum[t1_net_index + annmb.num_neurons1 * annmb.dim + annmb.num_neurons1 + annmb.num_neurons1], e_b1_n1);
    }
  } else {
    const float per_Nc = weight * 2.0f * lambda_f / Na / 3 / Nc;
    const float per_Nc_v = ((g_has_virial[batch_idx] && virial_nums > 0) ? weight * 2.0f * lambda_v / virial_nums : 0.0f);

    float fx_ref_n1 = g_fx_ref[n1];
    float fy_ref_n1 = g_fy_ref[n1];
    float fz_ref_n1 = g_fz_ref[n1];
    float dx_n1 = g_fx[n1] - fx_ref_n1;
    float dy_n1 = g_fy[n1] - fy_ref_n1;
    float dz_n1 = g_fz[n1] - fz_ref_n1;
    float type_weight = g_type_weight[g_type[n1]];
    if (force_delta > 0.0f) {
      float force_magnitude = sqrt(fx_ref_n1 * fx_ref_n1 + fy_ref_n1 * fy_ref_n1 + fz_ref_n1 * fz_ref_n1);
      type_weight *= sqrt(force_delta / (force_delta + force_magnitude));
    }
    dx_n1 *= type_weight;
    dy_n1 *= type_weight;
    dz_n1 *= type_weight;
    
    const float diff[6] = {
      g_diff_gpu_v[batch_idx * 6 + 0] * per_Nc_v,
      g_diff_gpu_v[batch_idx * 6 + 1] * per_Nc_v,
      g_diff_gpu_v[batch_idx * 6 + 2] * per_Nc_v,
      g_diff_gpu_v[batch_idx * 6 + 3] * per_Nc_v,
      g_diff_gpu_v[batch_idx * 6 + 4] * per_Nc_v,
      g_diff_gpu_v[batch_idx * 6 + 5] * per_Nc_v
    };

    int index = i1 * N + n1;
    int n2 = g_NL[index];
    int t2 = g_type[n2];
    int type_base = t1 * paramb.num_types + t2;
    float fx_ref_n2 = g_fx_ref[n2];
    float fy_ref_n2 = g_fy_ref[n2];
    float fz_ref_n2 = g_fz_ref[n2];
    float dx_n2 = g_fx[n2] - fx_ref_n2;
    float dy_n2 = g_fy[n2] - fy_ref_n2;
    float dz_n2 = g_fz[n2] - fz_ref_n2;
    float type_weight_n2 = g_type_weight[g_type[n2]];
    if (force_delta > 0.0f) {
      float force_magnitude = sqrt(fx_ref_n2 * fx_ref_n2 + fy_ref_n2 * fy_ref_n2 + fz_ref_n2 * fz_ref_n2);
      type_weight_n2 *= sqrt(force_delta / (force_delta + force_magnitude));
    }
    dx_n2 *= type_weight_n2;
    dy_n2 *= type_weight_n2;
    dz_n2 *= type_weight_n2;
    const float dx_diff = per_Nc * (dx_n1 - dx_n2);
    const float dy_diff = per_Nc * (dy_n1 - dy_n2);
    const float dz_diff = per_Nc * (dz_n1 - dz_n2);
    float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
    float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    float d12inv = 1.0f / d12;
    float fc12, fcp12;
    float rc = paramb.rc_radial;
    if (paramb.use_typewise_cutoff) {
      int z2 = paramb.atomic_numbers[t2];
      rc = min((covalent_radius_t1 + COVALENT_RADIUS[z2]) * typewise_cutoff_radial_factor, rc);
    }
    float rcinv = 1.0f / rc;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
    float tmp_xyz[3] = {d12inv * r12[0], d12inv * r12[1], d12inv * r12[2]};
    float tmp_xyz_123[6] = {
      tmp_xyz[0] * r12[0], // xx
      tmp_xyz[1] * r12[1], // yy
      tmp_xyz[2] * r12[2], // zz
      tmp_xyz[1] * r12[0], // xy
      tmp_xyz[2] * r12[1], // yz
      tmp_xyz[0] * r12[2]  // zx
    };
    // Cache gnp12 values to avoid recomputation, but compute feat_* on-the-fly to reduce register pressure
    float gnp12_cache[MAX_NUM_N];

    int n_base, c_index, grad_c_index;
    float gnp12, gFp_val, fp_xyz[3], fp_xyz_123[6], qp_c_tmp[3], grad_c_sum;
    int n2_tmp, t2_tmp, ln;
    float E2, q_c_scaler, q_c_ang, q_c_scaler_ang;
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      n_base = n * (paramb.basis_size_radial + 1);
      gnp12 = 0.0f;
      gFp_val = g_Fp[n1 + n * N];
      // E'(n) * ∂d_ij/∂α_ij
      fp_xyz[0] = gFp_val * tmp_xyz[0];
      fp_xyz[1] = gFp_val * tmp_xyz[1];
      fp_xyz[2] = gFp_val * tmp_xyz[2];
      // E'(n) * ∂d_ij/∂α_ij * α_ij
      fp_xyz_123[0] = gFp_val * tmp_xyz_123[0];
      fp_xyz_123[1] = gFp_val * tmp_xyz_123[1];
      fp_xyz_123[2] = gFp_val * tmp_xyz_123[2];
      fp_xyz_123[3] = gFp_val * tmp_xyz_123[3];
      fp_xyz_123[4] = gFp_val * tmp_xyz_123[4];
      fp_xyz_123[5] = gFp_val * tmp_xyz_123[5];

      for (int k = 0; k <= paramb.basis_size_radial; ++k) {
        c_index = (n_base + k) * paramb.num_types_sq + type_base;
        gnp12 += fnp12[k] * annmb.c[c_index];
        // E'(n) * Q'_{nk}(i,j) * ∂d_ij/∂α_ij
        qp_c_tmp[0] = fnp12[k] * fp_xyz[0];
        qp_c_tmp[1] = fnp12[k] * fp_xyz[1];
        qp_c_tmp[2] = fnp12[k] * fp_xyz[2];
        grad_c_index = c_index + annmb.num_ann;
        grad_c_sum = qp_c_tmp[0] * dx_diff + qp_c_tmp[1] * dy_diff + qp_c_tmp[2] * dz_diff;
        // if (!no_neighbor_isolated_atom) {
        grad_c_sum += per_Nc_e * g_Fp[n1 + n * N] * fn12[k];
        // }
        grad_c_sum -= fnp12[k] * (fp_xyz_123[0] * diff[0] + fp_xyz_123[1] * diff[1] + fp_xyz_123[2] * diff[2] + fp_xyz_123[3] * diff[3] + fp_xyz_123[4] * diff[4] + fp_xyz_123[5] * diff[5]);
        atomicAdd(&g_grad_sum[grad_c_index], grad_c_sum);
      }

      gnp12_cache[n] = gnp12;
    }

    // Original loop order: n -> k -> j (neighbor)
    // Optimized loop order: j (neighbor) -> n -> k
    // This avoids recomputing d12, rc, rcinv, fc12, fn12 for same neighbor across different (n,k)
    for (int j = 0; j < neighbor_number; ++j) {
      index = j * N + n1;
      n2_tmp = g_NL[index];
      t2_tmp = g_type[n2_tmp];
      float r12_tmp[3] = {g_x12[index], g_y12[index], g_z12[index]};
      d12 = sqrt(r12_tmp[0] * r12_tmp[0] + r12_tmp[1] * r12_tmp[1] + r12_tmp[2] * r12_tmp[2]);
      rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        int z2_tmp = paramb.atomic_numbers[t2_tmp];
        rc = min((covalent_radius_t1 + COVALENT_RADIUS[z2_tmp]) * typewise_cutoff_radial_factor, rc);
      }
      rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);

      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float feat_xyz_sum[3] = {0.0f};
        float feat_123_sum[6] = {0.0f};
        n_base = n * (paramb.basis_size_radial + 1);
        for (int m = 0; m <= paramb.n_max_radial; ++m) {
          E2 = g_Fp2[n1 + (m + n * annmb.dim) * N];
          // Compute feat_* on-the-fly using cached gnp12
          float feat_x_m = gnp12_cache[m] * tmp_xyz[0];
          float feat_y_m = gnp12_cache[m] * tmp_xyz[1];
          float feat_z_m = gnp12_cache[m] * tmp_xyz[2];
          feat_xyz_sum[0] += feat_x_m * E2;
          feat_xyz_sum[1] += feat_y_m * E2;
          feat_xyz_sum[2] += feat_z_m * E2;
          feat_123_sum[0] += feat_x_m * r12[0] * E2;
          feat_123_sum[1] += feat_y_m * r12[1] * E2;
          feat_123_sum[2] += feat_z_m * r12[2] * E2;
          feat_123_sum[3] += feat_y_m * r12[0] * E2;
          feat_123_sum[4] += feat_z_m * r12[1] * E2;
          feat_123_sum[5] += feat_x_m * r12[2] * E2;
        }
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          q_c_scaler = fn12[k] * g_q_scaler[n];
          grad_c_sum = q_c_scaler * (feat_xyz_sum[0] * dx_diff + feat_xyz_sum[1] * dy_diff + feat_xyz_sum[2] * dz_diff);
          grad_c_sum -= q_c_scaler * (feat_123_sum[0] * diff[0] + feat_123_sum[1] * diff[1] + feat_123_sum[2] * diff[2] + feat_123_sum[3] * diff[3] + feat_123_sum[4] * diff[4] + feat_123_sum[5] * diff[5]);

          type_base = t1 * paramb.num_types + t2_tmp;
          c_index = (n_base + k) * paramb.num_types_sq + type_base;
          grad_c_index = c_index + annmb.num_ann;
          atomicAdd(&g_grad_sum[grad_c_index], grad_c_sum);
        }
      }
    }
    // Hoist angular neighbor loop outside similar to radial case
    if (neighbor_number_ang > 0) {
      for (int ia = 0; ia < neighbor_number_ang; ++ia) {
        // Compute geometric quantities once per angular neighbor
        index = ia * N + n1;
        n2_tmp = g_NL_ang[index];
        t2_tmp = g_type[n2_tmp];
        float r12_tmp[3] = {g_x12_ang[index], g_y12_ang[index], g_z12_ang[index]};
        d12 = sqrt(r12_tmp[0] * r12_tmp[0] + r12_tmp[1] * r12_tmp[1] + r12_tmp[2] * r12_tmp[2]);
        rc = paramb.rc_angular;
        if (paramb.use_typewise_cutoff) {
          int z2_tmp = paramb.atomic_numbers[t2_tmp];
          rc = min((covalent_radius_t1 + COVALENT_RADIUS[z2_tmp]) * typewise_cutoff_angular_factor, rc);
        }
        rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);

        for (int na = 0; na <= paramb.n_max_angular; ++na) {
          n_base = na * (paramb.basis_size_angular + 1);
          for (int ka = 0; ka <= paramb.basis_size_angular; ++ka) {
            float f_c_n1[3] = {0.0f};
            float v_c_n1[6] = {0.0f};
            for (int l = 0; l < paramb.L_max; ++l) {
              float feat_xyz_sum[3] = {0.0f};
              float feat_123_sum[6] = {0.0f};
              ln = l * (paramb.n_max_angular + 1) + na;
              for (int ma = 0; ma <= paramb.n_max_radial; ++ma) {
                E2 = g_Fp2[n1 + (ma + (paramb.n_max_radial + 1 + ln) * annmb.dim) * N]; //g_Fp2[n1 + (d2 + d1 * annmb.dim) * N]
                // Compute feat_* on-the-fly using cached gnp12
                float feat_x_ma = gnp12_cache[ma] * tmp_xyz[0];
                float feat_y_ma = gnp12_cache[ma] * tmp_xyz[1];
                float feat_z_ma = gnp12_cache[ma] * tmp_xyz[2];
                feat_xyz_sum[0] += feat_x_ma * E2;
                feat_xyz_sum[1] += feat_y_ma * E2;
                feat_xyz_sum[2] += feat_z_ma * E2;
                feat_123_sum[0] += feat_x_ma * r12[0] * E2;
                feat_123_sum[1] += feat_y_ma * r12[1] * E2;
                feat_123_sum[2] += feat_z_ma * r12[2] * E2;
                feat_123_sum[3] += feat_y_ma * r12[0] * E2;
                feat_123_sum[4] += feat_z_ma * r12[1] * E2;
                feat_123_sum[5] += feat_x_ma * r12[2] * E2;
              }
              accumulate_qc(N, l + 1, na, paramb.n_max_angular + 1, paramb.basis_size_angular+1, d12, r12_tmp, fn12[ka], &g_sum_fxyz[n1], &q_c_ang);
              q_c_scaler_ang = q_c_ang * g_q_scaler[paramb.n_max_radial + 1 + ln];
              f_c_n1[0] += feat_xyz_sum[0] * q_c_scaler_ang;
              f_c_n1[1] += feat_xyz_sum[1] * q_c_scaler_ang;
              f_c_n1[2] += feat_xyz_sum[2] * q_c_scaler_ang;
              v_c_n1[0] += feat_123_sum[0] * q_c_scaler_ang;
              v_c_n1[1] += feat_123_sum[1] * q_c_scaler_ang;
              v_c_n1[2] += feat_123_sum[2] * q_c_scaler_ang;
              v_c_n1[3] += feat_123_sum[3] * q_c_scaler_ang;
              v_c_n1[4] += feat_123_sum[4] * q_c_scaler_ang;
              v_c_n1[5] += feat_123_sum[5] * q_c_scaler_ang;
            }
            grad_c_sum = f_c_n1[0] * dx_diff + f_c_n1[1] * dy_diff + f_c_n1[2] * dz_diff; // grad_c_sum_3b
            grad_c_sum -= v_c_n1[0] * diff[0] + v_c_n1[1] * diff[1] + v_c_n1[2] * diff[2] + v_c_n1[3] * diff[3] + v_c_n1[4] * diff[4] + v_c_n1[5] * diff[5];

            type_base = t1 * paramb.num_types + t2_tmp + paramb.num_c_radial;
            c_index = (n_base + ka) * paramb.num_types_sq + type_base;
            grad_c_index = c_index + annmb.num_ann;
            atomicAdd(&g_grad_sum[grad_c_index], grad_c_sum);
          }
        }
      }
    }
    int w0_index_dim, w1_index_dim, b0_index_dim;
    float scale, g_ep_w1b, g_ep_wb0, g_ep_w0b;
    for (int j = 0; j < annmb.num_neurons1; ++j) {
      float sum_dfeat_w1b0[6] = {0.0f};
      float sum_dfeat_w1b0_v[12] = {0.0f};
      for (int d = 0; d < annmb.dim; ++d) {
        float sum_dfeat_w0[3] = {0.0f};
        float sum_dfeat_w0_v[6] = {0.0f};
        if (d <= paramb.n_max_radial) {
          scale = g_q_scaler[d];
          // Compute dfeat_scaler on-the-fly using cached gnp12
          float feat_x_d = gnp12_cache[d] * tmp_xyz[0];
          float feat_y_d = gnp12_cache[d] * tmp_xyz[1];
          float feat_z_d = gnp12_cache[d] * tmp_xyz[2];
          float dfeat_scaler[3] = {feat_x_d * scale, feat_y_d * scale, feat_z_d * scale};
          float dfeat_scaler_v[6] = {feat_x_d * r12[0] * scale,
                                    feat_y_d * r12[1] * scale,
                                    feat_z_d * r12[2] * scale,
                                    feat_y_d * r12[0] * scale,
                                    feat_z_d * r12[1] * scale,
                                    feat_x_d * r12[2] * scale};
          w1_index_dim = n1_net_index_wb + (b0_index + j) * annmb.dim + d;//(N_neu * N_des + N_neu + j) * N_des + n
          b0_index_dim = n1_net_index_wb + (w0_index + j) * annmb.dim + d;//(N_neu * N_des + j) * N_des + n
          g_ep_w1b = g_ep_wb[w1_index_dim];
          g_ep_wb0 = g_ep_wb[b0_index_dim];
          sum_dfeat_w1b0[0] += dfeat_scaler[0] * g_ep_w1b;
          sum_dfeat_w1b0[1] += dfeat_scaler[1] * g_ep_w1b;
          sum_dfeat_w1b0[2] += dfeat_scaler[2] * g_ep_w1b;
          sum_dfeat_w1b0[3] += dfeat_scaler[0] * g_ep_wb0;
          sum_dfeat_w1b0[4] += dfeat_scaler[1] * g_ep_wb0;
          sum_dfeat_w1b0[5] += dfeat_scaler[2] * g_ep_wb0;
          sum_dfeat_w1b0_v[0] += dfeat_scaler_v[0] * g_ep_w1b;
          sum_dfeat_w1b0_v[1] += dfeat_scaler_v[1] * g_ep_w1b;
          sum_dfeat_w1b0_v[2] += dfeat_scaler_v[2] * g_ep_w1b;
          sum_dfeat_w1b0_v[3] += dfeat_scaler_v[3] * g_ep_w1b;
          sum_dfeat_w1b0_v[4] += dfeat_scaler_v[4] * g_ep_w1b;
          sum_dfeat_w1b0_v[5] += dfeat_scaler_v[5] * g_ep_w1b;
          sum_dfeat_w1b0_v[6] += dfeat_scaler_v[0] * g_ep_wb0;
          sum_dfeat_w1b0_v[7] += dfeat_scaler_v[1] * g_ep_wb0;
          sum_dfeat_w1b0_v[8] += dfeat_scaler_v[2] * g_ep_wb0;
          sum_dfeat_w1b0_v[9] += dfeat_scaler_v[3] * g_ep_wb0;
          sum_dfeat_w1b0_v[10] += dfeat_scaler_v[4] * g_ep_wb0;
          sum_dfeat_w1b0_v[11] += dfeat_scaler_v[5] * g_ep_wb0;
        }
        for (int m = 0; m <= paramb.n_max_radial; ++m) {
          scale = g_q_scaler[m];
          w0_index_dim = n1_net_index_wb + (j * annmb.dim + d) * annmb.dim + m;
          g_ep_w0b = g_ep_wb[w0_index_dim];
          // Compute feat_* on-the-fly using cached gnp12
          float feat_x_m = gnp12_cache[m] * tmp_xyz[0];
          float feat_y_m = gnp12_cache[m] * tmp_xyz[1];
          float feat_z_m = gnp12_cache[m] * tmp_xyz[2];
          sum_dfeat_w0[0] += feat_x_m * scale * g_ep_w0b;
          sum_dfeat_w0[1] += feat_y_m * scale * g_ep_w0b;
          sum_dfeat_w0[2] += feat_z_m * scale * g_ep_w0b;
          sum_dfeat_w0_v[0] += feat_x_m * r12[0] * scale * g_ep_w0b;
          sum_dfeat_w0_v[1] += feat_y_m * r12[1] * scale * g_ep_w0b;
          sum_dfeat_w0_v[2] += feat_z_m * r12[2] * scale * g_ep_w0b;
          sum_dfeat_w0_v[3] += feat_y_m * r12[0] * scale * g_ep_w0b;
          sum_dfeat_w0_v[4] += feat_z_m * r12[1] * scale * g_ep_w0b;
          sum_dfeat_w0_v[5] += feat_x_m * r12[2] * scale * g_ep_w0b;
        }
        grad_w0_sum = sum_dfeat_w0[0] * dx_diff + sum_dfeat_w0[1] * dy_diff + sum_dfeat_w0[2] * dz_diff;
        grad_w0_sum += i1 == 0 ? per_Nc_e * e_wb_grad[j * annmb.dim + d] : 0.0f;
        grad_w0_sum -= sum_dfeat_w0_v[0] * diff[0] + sum_dfeat_w0_v[1] * diff[1] + sum_dfeat_w0_v[2] * diff[2] + sum_dfeat_w0_v[3] * diff[3] + sum_dfeat_w0_v[4] * diff[4] + sum_dfeat_w0_v[5] * diff[5];
        atomicAdd(&g_grad_sum[t1_net_index + j * annmb.dim + d], grad_w0_sum);
      }
      grad_w1_sum = sum_dfeat_w1b0[0] * dx_diff + sum_dfeat_w1b0[1] * dy_diff + sum_dfeat_w1b0[2] * dz_diff;
      grad_b0_sum = sum_dfeat_w1b0[3] * dx_diff + sum_dfeat_w1b0[4] * dy_diff + sum_dfeat_w1b0[5] * dz_diff;
      if (i1 == 0) {
        grad_w1_sum += e_wb_grad[b0_index + j] * per_Nc_e;
        grad_b0_sum += e_wb_grad[w0_index + j] * per_Nc_e;
      }
      grad_w1_sum -= sum_dfeat_w1b0_v[0] * diff[0] + sum_dfeat_w1b0_v[1] * diff[1] 
        + sum_dfeat_w1b0_v[2] * diff[2] + sum_dfeat_w1b0_v[3] * diff[3] 
        + sum_dfeat_w1b0_v[4] * diff[4] + sum_dfeat_w1b0_v[5] * diff[5];
      grad_b0_sum -= sum_dfeat_w1b0_v[6] * diff[0] + sum_dfeat_w1b0_v[7] * diff[1] 
        + sum_dfeat_w1b0_v[8] * diff[2] + sum_dfeat_w1b0_v[9] * diff[3] 
        + sum_dfeat_w1b0_v[10] * diff[4] + sum_dfeat_w1b0_v[11] * diff[5];
      atomicAdd(&g_grad_sum[t1_net_index + b0_index + j], grad_w1_sum);
      atomicAdd(&g_grad_sum[t1_net_index + w0_index + j], grad_b0_sum);
    }
    if (i1 == 0) {
      float e_b1_n1 = e_wb_grad[annmb.num_neurons1 * annmb.dim + annmb.num_neurons1 + annmb.num_neurons1] * per_Nc_e;
      atomicAdd(&g_grad_sum[t1_net_index + annmb.num_neurons1 * annmb.dim + annmb.num_neurons1 + annmb.num_neurons1], e_b1_n1);
    }
  }
}

static __global__ void compute_grad_angular_NM(
  const int N,
  const int M,
  const int* __restrict__ g_NN,
  const int* __restrict__ g_NL,
  const int* __restrict__ g_NN_rad,
  const int* __restrict__ g_NL_rad,
  const GNEP::ParaMB paramb,
  const GNEP::ANN annmb,
  const int* __restrict__ g_Na,
  const int Nc,
  const float lambda_e,
  const float lambda_f,
  const float lambda_v,
  const int virial_nums,
  const int* __restrict__ g_has_virial,
  const float*  __restrict__ g_type_weight,
  const float force_delta,
  const int* __restrict__ g_batch_idx,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_x12_rad,
  const float* __restrict__ g_y12_rad,
  const float* __restrict__ g_z12_rad,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_Fp2,
  const float* __restrict__ g_sum_fxyz,
  const float* __restrict__ g_sum_s2xyz,
  const float* __restrict__ g_sum_s2xyz123,
  const float* __restrict__ g_diff_gpu_e,
  const float* __restrict__ g_diff_gpu_v,
  const float* __restrict__ g_fx_ref,
  const float* __restrict__ g_fy_ref,
  const float* __restrict__ g_fz_ref,
  const float* __restrict__ g_weight,
  const float* __restrict__ g_fx,
  const float* __restrict__ g_fy,
  const float* __restrict__ g_fz,
  const float* __restrict__ g_q_scaler,
  const float* __restrict__ g_ep_wb,
  float* g_grad_sum)
{
  const int NM = N * M;
  int Idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int w0_index = annmb.dim * annmb.num_neurons1;
  const int b0_index = w0_index + annmb.num_neurons1;
  if (Idx >= NM) return;
  int n1 = Idx / M;
  int i1 = Idx % M;
  int neighbor_number = g_NN[n1];
  int neighbor_number_rad = g_NN_rad[n1];
  if (i1 >= neighbor_number) return;
  float Fp[MAX_DIM_ANGULAR] = {0.0f};
  for (int d = 0; d < paramb.dim_angular; ++d) {
    Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
  }
  int t1 = g_type[n1];
  int batch_idx = g_batch_idx[n1];
  int Na = g_Na[batch_idx];
  float weight = g_weight[batch_idx];
  float per_Nc_e = g_diff_gpu_e[batch_idx] * weight * 2.0f * lambda_e / Nc;
  float per_Nc = weight * 2.0f * lambda_f / Na / 3 / Nc;
  float per_Nc_v = (g_has_virial[batch_idx] && virial_nums > 0) ? weight * 2.0f * lambda_v / virial_nums : 0.0f;

  float fx_ref_n1 = g_fx_ref[n1];
  float fy_ref_n1 = g_fy_ref[n1];
  float fz_ref_n1 = g_fz_ref[n1];
  float dx_n1 = g_fx[n1] - fx_ref_n1;
  float dy_n1 = g_fy[n1] - fy_ref_n1;
  float dz_n1 = g_fz[n1] - fz_ref_n1;
  float type_weight = g_type_weight[g_type[n1]];
  if (force_delta > 0.0f) {
    float force_magnitude = sqrt(fx_ref_n1 * fx_ref_n1 + fy_ref_n1 * fy_ref_n1 + fz_ref_n1 * fz_ref_n1);
    type_weight *= sqrt(force_delta / (force_delta + force_magnitude));
  }
  dx_n1 *= type_weight;
  dy_n1 *= type_weight;
  dz_n1 *= type_weight;

  // Cache frequently accessed type-dependent constants in registers
  const float typewise_cutoff_radial_factor = paramb.typewise_cutoff_radial_factor;
  const float typewise_cutoff_angular_factor = paramb.typewise_cutoff_angular_factor;
  const int z1 = paramb.atomic_numbers[t1];
  const float covalent_radius_t1 = COVALENT_RADIUS[z1];

  int t1_net_index = t1 * ((annmb.dim + 2) * annmb.num_neurons1 + 1);
  int n1_net_index_wb = n1 * annmb.num_ann * annmb.dim + t1_net_index * annmb.dim;
  
  float diff[6] = {
    g_diff_gpu_v[batch_idx * 6 + 0] * per_Nc_v,
    g_diff_gpu_v[batch_idx * 6 + 1] * per_Nc_v,
    g_diff_gpu_v[batch_idx * 6 + 2] * per_Nc_v,
    g_diff_gpu_v[batch_idx * 6 + 3] * per_Nc_v,
    g_diff_gpu_v[batch_idx * 6 + 4] * per_Nc_v,
    g_diff_gpu_v[batch_idx * 6 + 5] * per_Nc_v
  };

  int index = i1 * N + n1;
  int n2 = g_NL[index];
  int t2 = g_type[n2];
  int type_base = t1 * paramb.num_types + t2 + paramb.num_c_radial;
  float fx_ref_n2 = g_fx_ref[n2];
  float fy_ref_n2 = g_fy_ref[n2];
  float fz_ref_n2 = g_fz_ref[n2];
  float dx_n2 = g_fx[n2] - fx_ref_n2;
  float dy_n2 = g_fy[n2] - fy_ref_n2;
  float dz_n2 = g_fz[n2] - fz_ref_n2;
  float type_weight_n2 = g_type_weight[g_type[n2]];
  if (force_delta > 0.0f) {
    float force_magnitude = sqrt(fx_ref_n2 * fx_ref_n2 + fy_ref_n2 * fy_ref_n2 + fz_ref_n2 * fz_ref_n2);
    type_weight_n2 *= sqrt(force_delta / (force_delta + force_magnitude));
  }
  dx_n2 *= type_weight_n2;
  dy_n2 *= type_weight_n2;
  dz_n2 *= type_weight_n2;
  float dx_diff = per_Nc * (dx_n1 - dx_n2);
  float dy_diff = per_Nc * (dy_n1 - dy_n2);
  float dz_diff = per_Nc * (dz_n1 - dz_n2);
  float r12_i1[3] = {g_x12[index], g_y12[index], g_z12[index]};
  float d12_i1 = sqrt(r12_i1[0] * r12_i1[0] + r12_i1[1] * r12_i1[1] + r12_i1[2] * r12_i1[2]);
  float feat_x[MAX_LN];
  float feat_y[MAX_LN];
  float feat_z[MAX_LN];
  float fc12, fcp12;
  float rc = paramb.rc_angular;
  if (paramb.use_typewise_cutoff) {
    int z2 = paramb.atomic_numbers[t2];
    rc = min((covalent_radius_t1 + COVALENT_RADIUS[z2]) * typewise_cutoff_angular_factor, rc);
  }
  float rcinv = 1.0f / rc;
  find_fc_and_fcp(rc, rcinv, d12_i1, fc12, fcp12);

  float fn12[MAX_NUM_N];
  float fnp12[MAX_NUM_N];
  find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12_i1, fc12, fcp12, fn12, fnp12);

  float e_c, grad_c_sum, qp_c_tmp[3], qp_c_tmp123[6], qp_c_tmp1[3], qp_c_tmp2[3], q_c_ang, q_c_scaler;
  int n_base, c_index, grad_c_index;
  int n2_tmp, t2_tmp, ln, feat_offset;
  float dx_n2_tmp, dy_n2_tmp, dz_n2_tmp, E2;
  float gn12[MAX_NUM_N];
  float gnp12[MAX_NUM_N];
  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    gn12[n] = 0.0f;
    gnp12[n] = 0.0f;
    n_base = n * (paramb.basis_size_angular + 1);
    for (int k = 0; k <= paramb.basis_size_angular; ++k) {
      e_c = 0.0f;
      c_index = (n_base + k) * paramb.num_types_sq + type_base;
      gn12[n] += fn12[k] * annmb.c[c_index];
      gnp12[n] += fnp12[k] * annmb.c[c_index];
      accumulate_ec(N, paramb.L_max, n, paramb.n_max_angular + 1, paramb.basis_size_angular+1, d12_i1, r12_i1, fn12[k], fnp12[k], &g_sum_fxyz[n1], &g_sum_s2xyz[n1], &g_sum_s2xyz123[n1], Fp, &e_c, qp_c_tmp, qp_c_tmp123);
      grad_c_index = c_index + annmb.num_ann;
      grad_c_sum = per_Nc * (qp_c_tmp[0] * dx_n1 + qp_c_tmp[1] * dy_n1 + qp_c_tmp[2] * dz_n1);
      grad_c_sum += per_Nc_e * e_c;
      grad_c_sum -= qp_c_tmp123[0] * diff[0] + qp_c_tmp123[1] * diff[1] + qp_c_tmp123[2] * diff[2] + qp_c_tmp123[3] * diff[3] + qp_c_tmp123[4] * diff[4] + qp_c_tmp123[5] * diff[5];
      atomicAdd(&g_grad_sum[grad_c_index], grad_c_sum);
    }
    accumulate_dfe(N, NM, paramb.L_max, n, paramb.n_max_angular + 1, d12_i1, r12_i1, gn12[n], gnp12[n], &g_sum_fxyz[n1], &feat_x[n], &feat_y[n], &feat_z[n]);
  } // end of loop over n_max_ang
  float r12[3] = {0.0f};
  float d12 = 0.0f;
  for (int n = 0; n <= paramb.n_max_angular; ++n) {
    n_base = n * (paramb.basis_size_angular + 1);
    float s[NUM_OF_ABC];
    float sum_s2xyz[NUM_OF_ABC * 3];
    calculate_s_i1(paramb.L_max, d12_i1, r12_i1, gn12[n], gnp12[n], &g_sum_fxyz[n1], s, sum_s2xyz);
    // Hoist neighbor loop outside for angular gradient (3-body term)
    for (int j = 0; j < neighbor_number; ++j) {
      index = j * N + n1;
      n2_tmp = g_NL[index];
      t2_tmp = g_type[n2_tmp];
      fx_ref_n2 = g_fx_ref[n2_tmp];
      fy_ref_n2 = g_fy_ref[n2_tmp];
      fz_ref_n2 = g_fz_ref[n2_tmp];
      dx_n2_tmp = g_fx[n2_tmp] - fx_ref_n2;
      dy_n2_tmp = g_fy[n2_tmp] - fy_ref_n2;
      dz_n2_tmp = g_fz[n2_tmp] - fz_ref_n2;
      type_weight_n2 = g_type_weight[g_type[n2_tmp]];
      if (force_delta > 0.0f) {
        float force_magnitude = sqrt(fx_ref_n2 * fx_ref_n2 + fy_ref_n2 * fy_ref_n2 + fz_ref_n2 * fz_ref_n2);
        type_weight_n2 *= sqrt(force_delta / (force_delta + force_magnitude));
      }
      dx_n2_tmp *= type_weight_n2;
      dy_n2_tmp *= type_weight_n2;
      dz_n2_tmp *= type_weight_n2;
      r12[0] = g_x12[index];
      r12[1] = g_y12[index];
      r12[2] = g_z12[index];
      d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      rc = paramb.rc_angular;
      if (paramb.use_typewise_cutoff) {
        int z2_tmp = paramb.atomic_numbers[t2_tmp];
        rc = min((covalent_radius_t1 + COVALENT_RADIUS[z2_tmp]) * typewise_cutoff_angular_factor, rc);
      }
      rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);

      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        float f_c_n1[3] = {0.0f};
        float v_c_n1[6] = {0.0f};
        for (int l = 0; l < paramb.L_max; ++l) {
          float feat_xyz_sum[3] = {0.0f};
          float feat_123_sum[6] = {0.0f};
          ln = l * (paramb.n_max_angular + 1) + n;
          for (int m = 0; m < paramb.dim_angular; ++m) {
            feat_offset = n1 + ((paramb.n_max_radial + 1 + m) + (paramb.n_max_radial + 1 + ln) * annmb.dim) * N; //g_Fp2[n1 + (d2 + d1 * annmb.dim) * N]
            E2 = g_Fp2[feat_offset];
            float feat_x_m = feat_x[m];
            float feat_y_m = feat_y[m];
            float feat_z_m = feat_z[m];
            feat_xyz_sum[0] += feat_x_m * E2;
            feat_xyz_sum[1] += feat_y_m * E2;
            feat_xyz_sum[2] += feat_z_m * E2;
            feat_123_sum[0] += feat_x_m * r12_i1[0] * E2;
            feat_123_sum[1] += feat_y_m * r12_i1[1] * E2;
            feat_123_sum[2] += feat_z_m * r12_i1[2] * E2;
            feat_123_sum[3] += feat_y_m * r12_i1[0] * E2;
            feat_123_sum[4] += feat_z_m * r12_i1[1] * E2;
            feat_123_sum[5] += feat_x_m * r12_i1[2] * E2;
          }
          accumulate_qc(N, l + 1, n, paramb.n_max_angular + 1, paramb.basis_size_angular+1, d12, r12, fn12[k], &g_sum_fxyz[n1], &q_c_ang);
          q_c_scaler = q_c_ang * g_q_scaler[paramb.n_max_radial + 1 + ln];
          f_c_n1[0] += feat_xyz_sum[0] * q_c_scaler;
          f_c_n1[1] += feat_xyz_sum[1] * q_c_scaler;
          f_c_n1[2] += feat_xyz_sum[2] * q_c_scaler;
          v_c_n1[0] += feat_123_sum[0] * q_c_scaler;
          v_c_n1[1] += feat_123_sum[1] * q_c_scaler;
          v_c_n1[2] += feat_123_sum[2] * q_c_scaler;
          v_c_n1[3] += feat_123_sum[3] * q_c_scaler;
          v_c_n1[4] += feat_123_sum[4] * q_c_scaler;
          v_c_n1[5] += feat_123_sum[5] * q_c_scaler;
        }
        accumulate_fc(N, paramb.L_max, n, paramb.n_max_angular + 1, paramb.basis_size_angular+1, d12, r12, fn12[k], fnp12[k], s, sum_s2xyz, Fp, qp_c_tmp1, qp_c_tmp2);
        grad_c_sum = f_c_n1[0] * dx_diff + f_c_n1[1] * dy_diff + f_c_n1[2] * dz_diff;
        grad_c_sum -= per_Nc * (qp_c_tmp1[0] * dx_n2_tmp + qp_c_tmp1[1] * dy_n2_tmp + qp_c_tmp1[2] * dz_n2_tmp + qp_c_tmp2[0] * dx_n2 + qp_c_tmp2[1] * dy_n2 + qp_c_tmp2[2] * dz_n2);
        grad_c_sum -= v_c_n1[0] * diff[0] + v_c_n1[1] * diff[1] + v_c_n1[2] * diff[2] + v_c_n1[3] * diff[3] + v_c_n1[4] * diff[4] + v_c_n1[5] * diff[5];

        type_base = t1 * paramb.num_types + t2_tmp + paramb.num_c_radial;
        c_index = (n_base + k) * paramb.num_types_sq + type_base;
        grad_c_index = c_index + annmb.num_ann;
        atomicAdd(&g_grad_sum[grad_c_index], grad_c_sum);
      }
    }
  } // end of loop over neighbors' neighbors
  for (int nr = 0; nr <= paramb.n_max_radial; ++nr) {
    float feat_xyz_sum[3] = {0.0f};
    float feat_123_sum[6] = {0.0f};
    n_base = nr * (paramb.basis_size_radial + 1);
    for (int mr = 0; mr < paramb.dim_angular; ++mr) {
      E2 = g_Fp2[n1 + ((paramb.n_max_radial + 1 + mr) + nr * annmb.dim) * N]; //g_Fp2[n1 + (d2 + d1 * annmb.dim) * N]
      float feat_x_mr = feat_x[mr];
      float feat_y_mr = feat_y[mr];
      float feat_z_mr = feat_z[mr];
      feat_xyz_sum[0] += feat_x_mr * E2;
      feat_xyz_sum[1] += feat_y_mr * E2;
      feat_xyz_sum[2] += feat_z_mr * E2;
      feat_123_sum[0] += feat_x_mr * r12_i1[0] * E2;
      feat_123_sum[1] += feat_y_mr * r12_i1[1] * E2;
      feat_123_sum[2] += feat_z_mr * r12_i1[2] * E2;
      feat_123_sum[3] += feat_y_mr * r12_i1[0] * E2;
      feat_123_sum[4] += feat_z_mr * r12_i1[1] * E2;
      feat_123_sum[5] += feat_x_mr * r12_i1[2] * E2;
    }
    for (int ir = 0; ir < neighbor_number_rad; ++ir) {
      index = ir * N + n1;
      n2_tmp = g_NL_rad[index];
      t2_tmp = g_type[n2_tmp];
      r12[0] = g_x12_rad[index];
      r12[1] = g_y12_rad[index];
      r12[2] = g_z12_rad[index];
      d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        int z2_tmp = paramb.atomic_numbers[t2_tmp];
        rc = min((covalent_radius_t1 + COVALENT_RADIUS[z2_tmp]) * typewise_cutoff_radial_factor, rc);
      }
      rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);

      for (int kr = 0; kr <= paramb.basis_size_radial; ++kr) {
        q_c_scaler = fn12[kr] * g_q_scaler[nr];
        grad_c_sum = q_c_scaler * (feat_xyz_sum[0] * dx_diff + feat_xyz_sum[1] * dy_diff + feat_xyz_sum[2] * dz_diff);  // grad_c_sum_2b
        grad_c_sum -= q_c_scaler * (feat_123_sum[0] * diff[0] + feat_123_sum[1] * diff[1] + feat_123_sum[2] * diff[2] + feat_123_sum[3] * diff[3] + feat_123_sum[4] * diff[4] + feat_123_sum[5] * diff[5]);

        type_base = t1 * paramb.num_types + t2_tmp;
        c_index = (n_base + kr) * paramb.num_types_sq + type_base;
        grad_c_index = c_index + annmb.num_ann;
        atomicAdd(&g_grad_sum[grad_c_index], grad_c_sum);
      }
    }
  } // end of loop over neighbors' neighbors
  int index_dim, w0_index_dim, w1_index_dim, b0_index_dim;
  float scale, g_ep_w1b, g_ep_wb0, g_ep_w0b, grad_w0_sum, grad_w1_sum, grad_b0_sum;
  for (int j = 0; j < annmb.num_neurons1; ++j) {
    float sum_dfeat_w1b0[6] = {0.0f};
    float sum_dfeat_w1b0_v[12] = {0.0f};
    for (int d = 0; d < annmb.dim; ++d) {
      float sum_dfeat_w0[3] = {0.0f};
      float sum_dfeat_w0_v[6] = {0.0f};
      if (d < paramb.dim_angular) {
        index_dim = d + paramb.n_max_radial + 1;
        scale = g_q_scaler[index_dim];
        float feat_x_d = feat_x[d];
        float feat_y_d = feat_y[d];
        float feat_z_d = feat_z[d];
        float dfeat_scaler[3] = {feat_x_d * scale, feat_y_d * scale, feat_z_d * scale};
        float dfeat_scaler_v[6] = {feat_x_d * r12_i1[0] * scale, 
                                  feat_y_d * r12_i1[1] * scale, 
                                  feat_z_d * r12_i1[2] * scale, 
                                  feat_y_d * r12_i1[0] * scale, 
                                  feat_z_d * r12_i1[1] * scale, 
                                  feat_x_d * r12_i1[2] * scale};
        w1_index_dim = n1_net_index_wb + (b0_index + j) * annmb.dim + index_dim;//(N_neu * N_des + N_neu + j) * N_des + n
        b0_index_dim = n1_net_index_wb + (w0_index + j) * annmb.dim + index_dim;//(N_neu * N_des + j) * N_des + n
        g_ep_w1b = g_ep_wb[w1_index_dim];
        g_ep_wb0 = g_ep_wb[b0_index_dim];
        sum_dfeat_w1b0[0] += dfeat_scaler[0] * g_ep_w1b;
        sum_dfeat_w1b0[1] += dfeat_scaler[1] * g_ep_w1b;
        sum_dfeat_w1b0[2] += dfeat_scaler[2] * g_ep_w1b;
        sum_dfeat_w1b0[3] += dfeat_scaler[0] * g_ep_wb0;
        sum_dfeat_w1b0[4] += dfeat_scaler[1] * g_ep_wb0;
        sum_dfeat_w1b0[5] += dfeat_scaler[2] * g_ep_wb0;
        sum_dfeat_w1b0_v[0] += dfeat_scaler_v[0] * g_ep_w1b;
        sum_dfeat_w1b0_v[1] += dfeat_scaler_v[1] * g_ep_w1b;
        sum_dfeat_w1b0_v[2] += dfeat_scaler_v[2] * g_ep_w1b;
        sum_dfeat_w1b0_v[3] += dfeat_scaler_v[3] * g_ep_w1b;
        sum_dfeat_w1b0_v[4] += dfeat_scaler_v[4] * g_ep_w1b;
        sum_dfeat_w1b0_v[5] += dfeat_scaler_v[5] * g_ep_w1b;
        sum_dfeat_w1b0_v[6] += dfeat_scaler_v[0] * g_ep_wb0;
        sum_dfeat_w1b0_v[7] += dfeat_scaler_v[1] * g_ep_wb0;
        sum_dfeat_w1b0_v[8] += dfeat_scaler_v[2] * g_ep_wb0;
        sum_dfeat_w1b0_v[9] += dfeat_scaler_v[3] * g_ep_wb0;
        sum_dfeat_w1b0_v[10] += dfeat_scaler_v[4] * g_ep_wb0;
        sum_dfeat_w1b0_v[11] += dfeat_scaler_v[5] * g_ep_wb0;
      }
      for (int m = 0; m < paramb.dim_angular; ++m) {
        index_dim = m + paramb.n_max_radial + 1;
        scale = g_q_scaler[index_dim];
        w0_index_dim = n1_net_index_wb + (j * annmb.dim + d) * annmb.dim + index_dim;
        g_ep_w0b = g_ep_wb[w0_index_dim];
        float feat_x_m = feat_x[m];
        float feat_y_m = feat_y[m];
        float feat_z_m = feat_z[m];
        float scale_g_ep = scale * g_ep_w0b;
        sum_dfeat_w0[0] += feat_x_m * scale_g_ep;
        sum_dfeat_w0[1] += feat_y_m * scale_g_ep;
        sum_dfeat_w0[2] += feat_z_m * scale_g_ep;
        sum_dfeat_w0_v[0] += feat_x_m * r12_i1[0] * scale_g_ep;
        sum_dfeat_w0_v[1] += feat_y_m * r12_i1[1] * scale_g_ep;
        sum_dfeat_w0_v[2] += feat_z_m * r12_i1[2] * scale_g_ep;
        sum_dfeat_w0_v[3] += feat_y_m * r12_i1[0] * scale_g_ep;
        sum_dfeat_w0_v[4] += feat_z_m * r12_i1[1] * scale_g_ep;
        sum_dfeat_w0_v[5] += feat_x_m * r12_i1[2] * scale_g_ep;
      }
      grad_w0_sum = sum_dfeat_w0[0] * dx_diff + sum_dfeat_w0[1] * dy_diff + sum_dfeat_w0[2] * dz_diff;
      grad_w0_sum -= sum_dfeat_w0_v[0] * diff[0] + sum_dfeat_w0_v[1] * diff[1] 
                    + sum_dfeat_w0_v[2] * diff[2] + sum_dfeat_w0_v[3] * diff[3] 
                    + sum_dfeat_w0_v[4] * diff[4] + sum_dfeat_w0_v[5] * diff[5];
      atomicAdd(&g_grad_sum[t1_net_index + j * annmb.dim + d], grad_w0_sum);
    }
    grad_w1_sum = sum_dfeat_w1b0[0] * dx_diff + sum_dfeat_w1b0[1] * dy_diff + sum_dfeat_w1b0[2] * dz_diff;
    grad_b0_sum = sum_dfeat_w1b0[3] * dx_diff + sum_dfeat_w1b0[4] * dy_diff + sum_dfeat_w1b0[5] * dz_diff;
    grad_w1_sum -= sum_dfeat_w1b0_v[0] * diff[0] + sum_dfeat_w1b0_v[1] * diff[1] 
                  + sum_dfeat_w1b0_v[2] * diff[2] + sum_dfeat_w1b0_v[3] * diff[3] 
                  + sum_dfeat_w1b0_v[4] * diff[4] + sum_dfeat_w1b0_v[5] * diff[5];
    grad_b0_sum -= sum_dfeat_w1b0_v[6] * diff[0] + sum_dfeat_w1b0_v[7] * diff[1] 
                  + sum_dfeat_w1b0_v[8] * diff[2] + sum_dfeat_w1b0_v[9] * diff[3] 
                  + sum_dfeat_w1b0_v[10] * diff[4] + sum_dfeat_w1b0_v[11] * diff[5]; 
    atomicAdd(&g_grad_sum[t1_net_index + b0_index + j], grad_w1_sum);
    atomicAdd(&g_grad_sum[t1_net_index + w0_index + j], grad_b0_sum);
  }
}

static __global__ void find_force_radial(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const GNEP::ParaMB paramb,
  const GNEP::ANN annmb,
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
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      float rc = paramb.rc_radial;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_radial_factor,
          rc);
      }
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};
      float tmp_xyz[3] = {d12inv * r12[0], d12inv * r12[1], d12inv * r12[2]};

      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        float fp_xyz = g_Fp[n1 + n * N] * gnp12;
        f12[0] += fp_xyz * tmp_xyz[0];
        f12[1] += fp_xyz * tmp_xyz[1];
        f12[2] += fp_xyz * tmp_xyz[2];
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
  const bool requires_grad,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const GNEP::ParaMB paramb,
  const GNEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  float* g_sum_s2xyz,
  float* g_sum_s2xyz123,
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
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    int neighbor_number = g_NN[n1];
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = paramb.rc_angular;
      if (paramb.use_typewise_cutoff) {
        rc = min(
          (COVALENT_RADIUS[paramb.atomic_numbers[t1]] +
           COVALENT_RADIUS[paramb.atomic_numbers[t2]]) *
            paramb.typewise_cutoff_angular_factor,
          rc);
      }
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
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }          
        accumulate_f12(N, requires_grad, paramb.L_max, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, &g_sum_fxyz[n1], &g_sum_s2xyz[n1], &g_sum_s2xyz123[n1], f12);
      } // end of loop over n_max_ang

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
    } // end of loop over neighbors
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
  const GNEP::ParaMB paramb,
  const GNEP::ZBL zbl,
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
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1]; // starting from 1
    float pow_zi = pow(float(zi), 0.23f);
    int neighbor_number = g_NN[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2]; // starting from 1
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
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
        float ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        float rc_inner = zbl.rc_inner;
        float rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = rc_outer * 0.5f;
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

void GNEP::find_force(
  Parameters& para,
  const float* parameters,
  bool require_grad,
  std::vector<Dataset>& dataset,
  bool calculate_q_scaler,
  bool calculate_neighbor,
  int device_in_this_iter)
{

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(cudaSetDevice(device_id));
    gnep_data[device_id].Fp2.resize(dataset[device_id].N * annmb[device_id].dim * annmb[device_id].dim, 0.0f);
    gnep_data[device_id].sum_s2xyz.resize(dataset[device_id].N * (paramb.n_max_angular + 1) * NUM_OF_ABC * 3, 0.0f);
    gnep_data[device_id].sum_s2xyz123.resize(dataset[device_id].N * (paramb.n_max_angular + 1) * NUM_OF_ABC * 6, 0.0f);
    gnep_data[device_id].parameters.copy_from_host(parameters);
    update_potential(para, gnep_data[device_id].parameters.data(), annmb[device_id]);
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
        gnep_data[device_id].NN_radial.data(),
        gnep_data[device_id].NL_radial.data(),
        gnep_data[device_id].NN_angular.data(),
        gnep_data[device_id].NL_angular.data(),
        gnep_data[device_id].x12_radial.data(),
        gnep_data[device_id].y12_radial.data(),
        gnep_data[device_id].z12_radial.data(),
        gnep_data[device_id].x12_angular.data(),
        gnep_data[device_id].y12_angular.data(),
        gnep_data[device_id].z12_angular.data());
      GPU_CHECK_KERNEL
    }

    find_descriptors_radial<<<grid_size, block_size>>>(
      dataset[device_id].N,
      dataset[device_id].max_NN_radial,
      gnep_data[device_id].NN_radial.data(),
      gnep_data[device_id].NL_radial.data(),
      paramb,
      annmb[device_id],
      dataset[device_id].type.data(),
      gnep_data[device_id].x12_radial.data(),
      gnep_data[device_id].y12_radial.data(),
      gnep_data[device_id].z12_radial.data(),
      gnep_data[device_id].descriptors.data());
    GPU_CHECK_KERNEL

    find_descriptors_angular<<<grid_size, block_size>>>(
      dataset[device_id].N,
      gnep_data[device_id].NN_angular.data(),
      gnep_data[device_id].NL_angular.data(),
      paramb,
      annmb[device_id],
      dataset[device_id].type.data(),
      gnep_data[device_id].x12_angular.data(),
      gnep_data[device_id].y12_angular.data(),
      gnep_data[device_id].z12_angular.data(),
      gnep_data[device_id].descriptors.data(),
      gnep_data[device_id].sum_fxyz.data());
    GPU_CHECK_KERNEL

    if (para.prediction == 1 && para.output_descriptor >= 1) {
      FILE* fid_descriptor = my_fopen("descriptor.out", "a");
      std::vector<float> descriptor_cpu(gnep_data[device_id].descriptors.size());
      gnep_data[device_id].descriptors.copy_to_host(descriptor_cpu.data());
      for (int nc = 0; nc < dataset[device_id].Nc; ++nc) {
        float q_structure[MAX_DIM] = {0.0f};
        for (int na = 0; na < dataset[device_id].Na_cpu[nc]; ++na) {
          int n = dataset[device_id].Na_sum_cpu[nc] + na;
          for (int d = 0; d < annmb[device_id].dim; ++d) {
            float q = descriptor_cpu[n + d * dataset[device_id].N] * para.q_scaler_cpu[d];
            q_structure[d] += q;
            if (para.output_descriptor == 2) {
              fprintf(fid_descriptor, "%g ", q);
            }
          }
          if (para.output_descriptor == 2) {
            fprintf(fid_descriptor, "\n");
          }
        }
        if (para.output_descriptor == 1) {
          for (int d = 0; d < annmb[device_id].dim; ++d) {
            fprintf(fid_descriptor, "%g ", q_structure[d] / dataset[device_id].Na_cpu[nc]);
          }
        }
        if (para.output_descriptor == 1) {
          fprintf(fid_descriptor, "\n");
        }
      }
      fclose(fid_descriptor);
    }

    if (calculate_q_scaler) {
      find_max_min<<<annmb[device_id].dim, 1024>>>(
        dataset[device_id].N,
        gnep_data[device_id].descriptors.data(),
        para.s_max[device_id].data(),
        para.s_min[device_id].data(),
        para.q_scaler_gpu[device_id].data());
      GPU_CHECK_KERNEL
    }

    zero_force<<<grid_size, block_size>>>(
      dataset[device_id].N,
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2,
      dataset[device_id].virial.data(),
      dataset[device_id].virial.data() + dataset[device_id].N,
      dataset[device_id].virial.data() + dataset[device_id].N * 2);
    GPU_CHECK_KERNEL

    if (require_grad) {
      initialize_gradients(para, dataset[device_id].N);
      apply_ann<true><<<grid_size, block_size>>>(
        dataset[device_id].N,
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        gnep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data(),
        dataset[device_id].energy.data(),
        gnep_data[device_id].Fp.data(),
        gnep_data[device_id].Fp2.data(),
        gradients.Fp_wb.data(),
        gradients.E_wb_grad.data());
      GPU_CHECK_KERNEL
    } else {
      apply_ann<false><<<grid_size, block_size>>>(
        dataset[device_id].N,
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        gnep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data(),
        dataset[device_id].energy.data(),
        gnep_data[device_id].Fp.data());
      GPU_CHECK_KERNEL
    }
    find_force_radial<<<grid_size, block_size>>>(
      dataset[device_id].N,
      gnep_data[device_id].NN_radial.data(),
      gnep_data[device_id].NL_radial.data(),
      paramb,
      annmb[device_id],
      dataset[device_id].type.data(),
      gnep_data[device_id].x12_radial.data(),
      gnep_data[device_id].y12_radial.data(),
      gnep_data[device_id].z12_radial.data(),
      gnep_data[device_id].Fp.data(),
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2,
      dataset[device_id].virial.data());
    GPU_CHECK_KERNEL

    find_force_angular<<<grid_size, block_size>>>(
      require_grad,
      dataset[device_id].N,
      gnep_data[device_id].NN_angular.data(),
      gnep_data[device_id].NL_angular.data(),
      paramb,
      annmb[device_id],
      dataset[device_id].type.data(),
      gnep_data[device_id].x12_angular.data(),
      gnep_data[device_id].y12_angular.data(),
      gnep_data[device_id].z12_angular.data(),
      gnep_data[device_id].Fp.data(),
      gnep_data[device_id].sum_fxyz.data(),
      gnep_data[device_id].sum_s2xyz.data(),
      gnep_data[device_id].sum_s2xyz123.data(),
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2,
      dataset[device_id].virial.data());
    GPU_CHECK_KERNEL

    if (zbl.enabled) {
      find_force_ZBL<<<grid_size, block_size>>>(
        dataset[device_id].N,
        paramb,
        zbl,
        gnep_data[device_id].NN_angular.data(),
        gnep_data[device_id].NL_angular.data(),
        dataset[device_id].type.data(),
        gnep_data[device_id].x12_angular.data(),
        gnep_data[device_id].y12_angular.data(),
        gnep_data[device_id].z12_angular.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data(),
        dataset[device_id].energy.data());
      GPU_CHECK_KERNEL
    } 

    gpu_sum_pe_error<<<dataset[device_id].Nc, 256, sizeof(float) * 256>>>(
      dataset[device_id].Na.data(),
      dataset[device_id].Na_sum.data(),
      dataset[device_id].energy.data(),
      dataset[device_id].energy_ref_gpu.data(),
      dataset[device_id].diff_gpu_e.data(),
      dataset[device_id].error_gpu.data());
    CHECK(cudaMemcpy(dataset[device_id].error_cpu_e.data(), dataset[device_id].error_gpu.data(), dataset[device_id].Nc * sizeof(float), cudaMemcpyDeviceToHost));

    float shear_weight = require_grad ? para.lambda_shear * para.lambda_shear : 1.0f;
    gpu_sum_virial_error<<<dataset[device_id].Nc, 256, sizeof(float) * 256 * 6>>>(
      dataset[device_id].N,
      shear_weight,
      dataset[device_id].Na.data(),
      dataset[device_id].Na_sum.data(),
      dataset[device_id].virial.data(),
      dataset[device_id].virial_ref_gpu.data(),
      dataset[device_id].diff_gpu_v.data(),
      dataset[device_id].error_gpu.data());
    CHECK(cudaMemcpy(dataset[device_id].error_cpu_v.data(), dataset[device_id].error_gpu.data(), dataset[device_id].Nc * sizeof(float), cudaMemcpyDeviceToHost));
    int virial_nums = 0;
    for (int n = 0; n < dataset[device_id].Nc; ++n) {
      if (dataset[device_id].has_virial[n]) {
        virial_nums += 6;
      }
    }

    if (require_grad) {
      const int NM_radial = dataset[device_id].N * dataset[device_id].max_NN_radial;
      const int threads_per_block = 32;
      if (NM_radial > 0) {
        const int blocks_per_grid = (NM_radial + threads_per_block - 1) / threads_per_block;
        compute_grad_radial_NM<<<blocks_per_grid, threads_per_block>>>(
          dataset[device_id].N,
          dataset[device_id].max_NN_radial,
          gnep_data[device_id].NN_radial.data(),
          gnep_data[device_id].NL_radial.data(),
          gnep_data[device_id].NN_angular.data(),
          gnep_data[device_id].NL_angular.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].Na.data(),
          dataset[device_id].Nc,
          para.lambda_e,
          para.lambda_f,
          para.lambda_v,
          virial_nums,
          dataset[device_id].has_virial_gpu.data(),
          dataset[device_id].type_weight_gpu.data(),
          para.force_delta,
          dataset[device_id].batch_idx.data(),
          dataset[device_id].type.data(),
          gnep_data[device_id].x12_radial.data(),
          gnep_data[device_id].y12_radial.data(),
          gnep_data[device_id].z12_radial.data(),
          gnep_data[device_id].x12_angular.data(),
          gnep_data[device_id].y12_angular.data(),
          gnep_data[device_id].z12_angular.data(),
          gnep_data[device_id].Fp.data(),
          gnep_data[device_id].Fp2.data(),
          gnep_data[device_id].sum_fxyz.data(),
          gradients.E_wb_grad.data(),
          dataset[device_id].diff_gpu_e.data(),
          dataset[device_id].diff_gpu_v.data(),
          dataset[device_id].force_ref_gpu.data(),
          dataset[device_id].force_ref_gpu.data() + dataset[device_id].N,
          dataset[device_id].force_ref_gpu.data() + dataset[device_id].N * 2,
          dataset[device_id].weight_gpu.data(),
          dataset[device_id].force.data(),
          dataset[device_id].force.data() + dataset[device_id].N,
          dataset[device_id].force.data() + dataset[device_id].N * 2,
          para.q_scaler_gpu[device_id].data(),
          gradients.Fp_wb.data(),
          gradients.grad_sum.data());
        GPU_CHECK_KERNEL
      } else {
        const int blocks_per_grid = (dataset[device_id].N + threads_per_block - 1) / threads_per_block;
        compute_grad_e_without_neighbor<<<blocks_per_grid, threads_per_block>>>(
          dataset[device_id].N,
          paramb,
          annmb[device_id],
          dataset[device_id].Nc,
          para.lambda_e,
          dataset[device_id].batch_idx.data(),
          dataset[device_id].type.data(),
          gradients.E_wb_grad.data(),
          dataset[device_id].diff_gpu_e.data(),
          dataset[device_id].weight_gpu.data(),
          gradients.grad_sum.data());
        GPU_CHECK_KERNEL
      }
      const int NM_angular = dataset[device_id].N * dataset[device_id].max_NN_angular;
      if (NM_angular > 0) {
        const int blocks_per_grid_angular = (NM_angular + threads_per_block - 1) / threads_per_block;
        compute_grad_angular_NM<<<blocks_per_grid_angular, threads_per_block>>>(
          dataset[device_id].N,
          dataset[device_id].max_NN_angular,
          gnep_data[device_id].NN_angular.data(),
          gnep_data[device_id].NL_angular.data(),
          gnep_data[device_id].NN_radial.data(),
          gnep_data[device_id].NL_radial.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].Na.data(),
          dataset[device_id].Nc,
          para.lambda_e,
          para.lambda_f,
          para.lambda_v,
          virial_nums,
          dataset[device_id].has_virial_gpu.data(),
          dataset[device_id].type_weight_gpu.data(),
          para.force_delta,
          dataset[device_id].batch_idx.data(),
          dataset[device_id].type.data(),
          gnep_data[device_id].x12_angular.data(),
          gnep_data[device_id].y12_angular.data(),
          gnep_data[device_id].z12_angular.data(),
          gnep_data[device_id].x12_radial.data(),
          gnep_data[device_id].y12_radial.data(),
          gnep_data[device_id].z12_radial.data(),
          gnep_data[device_id].Fp.data(),
          gnep_data[device_id].Fp2.data(),
          gnep_data[device_id].sum_fxyz.data(),
          gnep_data[device_id].sum_s2xyz.data(),
          gnep_data[device_id].sum_s2xyz123.data(),
          dataset[device_id].diff_gpu_e.data(),
          dataset[device_id].diff_gpu_v.data(),
          dataset[device_id].force_ref_gpu.data(),
          dataset[device_id].force_ref_gpu.data() + dataset[device_id].N,
          dataset[device_id].force_ref_gpu.data() + dataset[device_id].N * 2,
          dataset[device_id].weight_gpu.data(),
          dataset[device_id].force.data(),
          dataset[device_id].force.data() + dataset[device_id].N,
          dataset[device_id].force.data() + dataset[device_id].N * 2,
          para.q_scaler_gpu[device_id].data(),
          gradients.Fp_wb.data(),
          gradients.grad_sum.data());
        GPU_CHECK_KERNEL
      }
    }
  }
}
