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
#include "neptb.cuh"
#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_utilities.cuh"
#ifdef USE_HIP
  #include <hipsolver.h>
#else
  #include <cusolverDn.h>
#endif
#include <cstring>

const int number_of_orbitals_per_atom = 4;

static __global__ void gpu_find_neighbor_list(
  const NEPTB::ParaMB paramb,
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
  const int* g_NN,
  const int* g_NL,
  const NEPTB::ParaMB paramb,
  const NEPTB::ANN annmb,
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
  const NEPTB::ParaMB paramb,
  const NEPTB::ANN annmb,
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

NEPTB::NEPTB(
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
    gpuSetDevice(device_id);
    annmb[device_id].dim = para.dim;
    annmb[device_id].num_neurons1 = para.num_neurons1;
    annmb[device_id].num_para = para.number_of_variables;

    neptb_data[device_id].NN_radial.resize(N);
    neptb_data[device_id].NN_angular.resize(N);
    neptb_data[device_id].NL_radial.resize(N_times_max_NN_radial);
    neptb_data[device_id].NL_angular.resize(N_times_max_NN_angular);
    neptb_data[device_id].x12_radial.resize(N_times_max_NN_radial);
    neptb_data[device_id].y12_radial.resize(N_times_max_NN_radial);
    neptb_data[device_id].z12_radial.resize(N_times_max_NN_radial);
    neptb_data[device_id].x12_angular.resize(N_times_max_NN_angular);
    neptb_data[device_id].y12_angular.resize(N_times_max_NN_angular);
    neptb_data[device_id].z12_angular.resize(N_times_max_NN_angular);
    neptb_data[device_id].descriptors.resize(N * annmb[device_id].dim);
    neptb_data[device_id].Fp.resize(N * annmb[device_id].dim);
    neptb_data[device_id].sum_fxyz.resize(N * (paramb.n_max_angular + 1) * NUM_OF_ABC);
    neptb_data[device_id].parameters.resize(annmb[device_id].num_para);
    neptb_data[device_id].hamiltonian.resize(N * 4 * N * 4);
    neptb_data[device_id].hamiltonian_unscaled.resize(N * 4 * N * 4);
    neptb_data[device_id].eigenvalue.resize(N * 4);
  }
}

void NEPTB::update_potential(Parameters& para, float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
  }
  ann.b1 = pointer;
  pointer += 1;
  ann.c = pointer;
}

static void __global__ find_max_min(const int N, const float* g_q, float* g_q_scaler)
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

static __global__ void apply_ann(
  const int N,
  const NEPTB::ParaMB paramb,
  const NEPTB::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  float* g_pe,
  float* g_Fp)
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

    apply_ann_one_layer(
      annmb.dim,
      annmb.num_neurons1,
      annmb.w0[type],
      annmb.b0[type],
      annmb.w1[type],
      annmb.b1,
      q,
      F,
      Fp);
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

static __global__ void find_force_radial(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEPTB::ParaMB paramb,
  const NEPTB::ANN annmb,
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

      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
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
  const int* g_NN,
  const int* g_NL,
  const NEPTB::ParaMB paramb,
  const NEPTB::ANN annmb,
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
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
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
        accumulate_f12(paramb.L_max, paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
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
  const NEPTB::ParaMB paramb,
  const NEPTB::ZBL zbl,
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

static __device__ float s(float r)
{
  float n=2.0f;
  float nc=6.5f;
  float rc=2.18f;
  float r0=1.536329f;
  return (r0/r) * (r0/r) * exp( n * (-pow(r/rc,nc) + pow(r0/rc, nc)) );
}

static __device__ float s_d(float r)
{
  float n=2.0f;
  float nc=6.5f;
  float rc=2.18f;
  return -n * s(r) * (1.0f + nc * pow(r / rc, nc)) / r;
}

static __global__ void find_hamiltonian(
  const NEPTB::TB tb,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  float* g_hamiltonian,
  float* g_hamiltonian_unscaled)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    for (int k = 0; k < number_of_orbitals_per_atom; ++k) {
      int nk = n1 * number_of_orbitals_per_atom + k;
      g_hamiltonian[nk * N * number_of_orbitals_per_atom + nk] = tb.onsite[k];
    }

    int neighbor_number = g_NN[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;

      float s12 = (tb.r0 * d12inv) * (tb.r0 * d12inv) *
                   exp(2.0f * (-pow(d12 / tb.rc, tb.nc) + pow(tb.r0 / tb.rc, tb.nc)));

      float cos_x = r12[0] * d12inv;
      float cos_y = r12[1] * d12inv;
      float cos_z = r12[2] * d12inv;
      float cos_xx = cos_x * cos_x;
      float cos_yy = cos_y * cos_y;
      float cos_zz = cos_z * cos_z;
      float sin_xx = 1.0f - cos_xx;
      float sin_yy = 1.0f - cos_yy;
      float sin_zz = 1.0f - cos_zz;
      float cos_xy = cos_x * cos_y;
      float cos_yz = cos_y * cos_z;
      float cos_zx = cos_z * cos_x;

      float H12[number_of_orbitals_per_atom][number_of_orbitals_per_atom];
      H12[0][0] = tb.v_sss;
      H12[1][1] = tb.v_pps * cos_xx + tb.v_ppp * sin_xx;
      H12[2][2] = tb.v_pps * cos_yy + tb.v_ppp * sin_yy;
      H12[3][3] = tb.v_pps * cos_zz + tb.v_ppp * sin_zz;
      H12[0][1] = tb.v_sps * cos_x;
      H12[0][2] = tb.v_sps * cos_y;
      H12[0][3] = tb.v_sps * cos_z;
      H12[1][2] = (tb.v_pps - tb.v_ppp) * cos_xy;
      H12[2][3] = (tb.v_pps - tb.v_ppp) * cos_yz;
      H12[3][1] = (tb.v_pps - tb.v_ppp) * cos_zx;
      H12[1][0] = -H12[0][1];
      H12[2][0] = -H12[0][2];
      H12[3][0] = -H12[0][3];
      H12[2][1] = H12[1][2];
      H12[3][2] = H12[2][3];
      H12[1][3] = H12[3][1];

      for (int k1 = 0; k1 < number_of_orbitals_per_atom; ++k1) {
        for (int k2 = 0; k2 < number_of_orbitals_per_atom; ++k2) {
          int n1k1 = n1 * number_of_orbitals_per_atom + k1;
          int n2k2 = n2 * number_of_orbitals_per_atom + k2;
          g_hamiltonian[n1k1 * N * number_of_orbitals_per_atom + n2k2] = H12[k1][k2] * s12;
          g_hamiltonian_unscaled[n1k1 * N * number_of_orbitals_per_atom + n2k2] = H12[k1][k2];
        }
      }
    }
  }
}

static __global__ void find_force_from_eigenvectors(
  const NEPTB::TB tb,
  const int N,
  const int* g_NN,
  const int* g_NL,
  const float* g_x12,
  const float* g_y12,
  const float* g_z12,
  const float* g_hamiltonian_unscaled,
  const float* g_eigenvector,
  float* g_fx,
  float* g_fy,
  float* g_fz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;

  if (n1 < N) {
    float force[3];
    int neighbor_number = g_NN[n1];

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float cos_x = r12[0] * d12inv;
      float cos_y = r12[1] * d12inv;
      float cos_z = r12[2] * d12inv;
      float cos_xx = cos_x * cos_x;
      float cos_yy = cos_y * cos_y;
      float cos_zz = cos_z * cos_z;
      float sin_xx = 1.0f - cos_xx;
      float sin_yy = 1.0f - cos_yy;
      float sin_zz = 1.0f - cos_zz;
      float cos_xy = cos_x * cos_y;
      float cos_yz = cos_y * cos_z;
      float cos_zx = cos_z * cos_x;
      float cos_xyz = cos_xy * cos_z;

      float e_sx[3] = {sin_xx, -cos_xy, -cos_zx};
      float e_sy[3] = {-cos_xy, sin_yy, -cos_yz};
      float e_sz[3] = {-cos_zx, -cos_yz, sin_zz};
      float e_xx[3] = {2.0f*cos_x*e_sx[0], 2.0f*cos_x*e_sx[1], 2.0f*cos_x*e_sx[2]}; 
      float e_yy[3] = {2.0f*cos_y*e_sy[0], 2.0f*cos_y*e_sy[1], 2.0f*cos_y*e_sy[2]}; 
      float e_zz[3] = {2.0f*cos_z*e_sz[0], 2.0f*cos_z*e_sz[1], 2.0f*cos_z*e_sz[2]};
      float e_xy[3] = {cos_y*(1.0f-2.0f*cos_xx), cos_x*(1.0f-2.0f*cos_yy), -2.0f*cos_xyz};
      float e_yz[3] = {-2.0f*cos_xyz, cos_z*(1.0f-2.0f*cos_yy), cos_y*(1.0f-2.0f*cos_zz)};
      float e_zx[3] = {cos_z*(1.0f-2.0f*cos_xx), -2.0f*cos_xyz, cos_x*(1.0f-2.0f*cos_zz)};
      float F[number_of_orbitals_per_atom][number_of_orbitals_per_atom] = {0.0f};
      for (int a = 0; a < number_of_orbitals_per_atom; ++a) {
        for (int b = 0; b < number_of_orbitals_per_atom; ++b) {
          for (int n = 0; n < N * number_of_orbitals_per_atom/2; ++ n) {
            F[a][b] += 
            g_eigenvector[n * N*number_of_orbitals_per_atom + n1*number_of_orbitals_per_atom+a] *
            g_eigenvector[n * N*number_of_orbitals_per_atom + n2*number_of_orbitals_per_atom+b];
          }
        }
      }

      for (int d = 0; d < 3; ++d) {
        float K[4][4]  = {0.0f};
        K[1][1] = s(d12)* d12inv*(tb.v_pps - tb.v_ppp)*e_xx[d];
        K[2][2] = s(d12)* d12inv*(tb.v_pps - tb.v_ppp)*e_yy[d];
        K[3][3] = s(d12)* d12inv*(tb.v_pps - tb.v_ppp)*e_zz[d];
        K[0][1] = s(d12)* d12inv * tb.v_sps * e_sx[d];
        K[0][2] = s(d12)* d12inv * tb.v_sps * e_sy[d];
        K[0][3] = s(d12)* d12inv * tb.v_sps * e_sz[d];
        K[1][2] = s(d12)* d12inv * (tb.v_pps - tb.v_ppp) * e_xy[d];
        K[2][3] = s(d12)* d12inv * (tb.v_pps - tb.v_ppp) * e_yz[d];
        K[3][1] = s(d12)* d12inv * (tb.v_pps - tb.v_ppp) * e_zx[d];
        K[1][0] = - K[0][1];
        K[2][0] = - K[0][2];
        K[3][0] = - K[0][3];
        K[2][1] = + K[1][2];
        K[3][2] = + K[2][3];
        K[1][3] = + K[3][1];
        for (int a = 0; a < number_of_orbitals_per_atom; ++a) {
          for (int b = 0; b < number_of_orbitals_per_atom; ++b) {
            int n1a = n1 * number_of_orbitals_per_atom + a;
            int n2b = n2 * number_of_orbitals_per_atom + b;
            K[a][b] += g_hamiltonian_unscaled[n1a * N*number_of_orbitals_per_atom + n2b] * s_d(d12) * r12[d] * d12inv;
            force[d] += 4.0f * F[a][b] * K[a][b];
          }
        }
      }
    }
    g_fx[n1] += force[0];
    g_fy[n1] += force[1];
    g_fz[n1] += force[2];
  }
}

static void eigenvectors_symmetric_Jacobi(const int N, float* A, float* W)
{
  // get handle
  gpusolverDnHandle_t handle = NULL;
  gpusolverDnCreate(&handle);
  gpusolverEigMode_t jobz = GPUSOLVER_EIG_MODE_VECTOR;
  gpusolverFillMode_t uplo = GPUSOLVER_FILL_MODE_LOWER;

  // some parameters for the Jacobi method
  gpusolverSyevjInfo_t para = NULL;
  gpusolverDnCreateSyevjInfo(&para);

  // get work
  int lwork = 0;
  gpusolverDnSsyevj_bufferSize(handle, jobz, uplo, N, A, N, W, &lwork, para);
  GPU_Vector<float> work(lwork);

  // get W
  GPU_Vector<int> info(1);
  gpusolverDnSsyevj(
    handle, jobz, uplo, N, A, N, W, work.data(), lwork, info.data(), para);

  // free
  gpusolverDnDestroy(handle);
  gpusolverDnDestroySyevjInfo(para);
}

void NEPTB::find_force(
  Parameters& para,
  const float* parameters,
  std::vector<Dataset>& dataset,
  bool calculate_q_scaler,
  bool calculate_neighbor,
  int device_in_this_iter)
{

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    neptb_data[device_id].parameters.copy_from_host(
      parameters + device_id * para.number_of_variables);
    update_potential(para, neptb_data[device_id].parameters.data(), annmb[device_id]);
  }

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
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
        neptb_data[device_id].NN_radial.data(),
        neptb_data[device_id].NL_radial.data(),
        neptb_data[device_id].NN_angular.data(),
        neptb_data[device_id].NL_angular.data(),
        neptb_data[device_id].x12_radial.data(),
        neptb_data[device_id].y12_radial.data(),
        neptb_data[device_id].z12_radial.data(),
        neptb_data[device_id].x12_angular.data(),
        neptb_data[device_id].y12_angular.data(),
        neptb_data[device_id].z12_angular.data());
      GPU_CHECK_KERNEL
    }

    find_descriptors_radial<<<grid_size, block_size>>>(
      dataset[device_id].N,
      neptb_data[device_id].NN_radial.data(),
      neptb_data[device_id].NL_radial.data(),
      paramb,
      annmb[device_id],
      dataset[device_id].type.data(),
      neptb_data[device_id].x12_radial.data(),
      neptb_data[device_id].y12_radial.data(),
      neptb_data[device_id].z12_radial.data(),
      neptb_data[device_id].descriptors.data());
    GPU_CHECK_KERNEL

    find_descriptors_angular<<<grid_size, block_size>>>(
      dataset[device_id].N,
      neptb_data[device_id].NN_angular.data(),
      neptb_data[device_id].NL_angular.data(),
      paramb,
      annmb[device_id],
      dataset[device_id].type.data(),
      neptb_data[device_id].x12_angular.data(),
      neptb_data[device_id].y12_angular.data(),
      neptb_data[device_id].z12_angular.data(),
      neptb_data[device_id].descriptors.data(),
      neptb_data[device_id].sum_fxyz.data());
    GPU_CHECK_KERNEL

    if (para.prediction == 1 && para.output_descriptor >= 1) {
      FILE* fid_descriptor = my_fopen("descriptor.out", "a");
      std::vector<float> descriptor_cpu(neptb_data[device_id].descriptors.size());
      neptb_data[device_id].descriptors.copy_to_host(descriptor_cpu.data());
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
        neptb_data[device_id].descriptors.data(),
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

    find_hamiltonian<<<grid_size, block_size>>>(
      tb, 
      dataset[device_id].N,
      neptb_data[device_id].NN_angular.data(),
      neptb_data[device_id].NL_angular.data(),
      neptb_data[device_id].x12_angular.data(),
      neptb_data[device_id].y12_angular.data(),
      neptb_data[device_id].z12_angular.data(),
      neptb_data[device_id].hamiltonian.data(),
      neptb_data[device_id].hamiltonian_unscaled.data());
    GPU_CHECK_KERNEL

    // hamiltonian will become eigenvector
    eigenvectors_symmetric_Jacobi(
      dataset[device_id].N * number_of_orbitals_per_atom, 
      neptb_data[device_id].hamiltonian.data(),
      neptb_data[device_id].eigenvalue.data());

    // hamiltonian here is eigenvector
    find_force_from_eigenvectors<<<grid_size, block_size>>>(
      tb,
      dataset[device_id].N,
      neptb_data[device_id].NN_angular.data(),
      neptb_data[device_id].NL_angular.data(),
      neptb_data[device_id].x12_angular.data(),
      neptb_data[device_id].y12_angular.data(),
      neptb_data[device_id].z12_angular.data(),
      neptb_data[device_id].hamiltonian_unscaled.data(),
      neptb_data[device_id].hamiltonian.data(),
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2);
    GPU_CHECK_KERNEL

    // now hamiltonian is eigenvector
    FILE* fid_eigenvalue = my_fopen("eigenvalue.out", "w");
    FILE* fid_hamiltonian = my_fopen("hamiltonian.out", "w");
    FILE* fid_force = my_fopen("force.out", "w");
    std::vector<float> eigenvalue_cpu(neptb_data[device_id].eigenvalue.size());
    std::vector<float> hamiltonian_cpu(neptb_data[device_id].hamiltonian.size());
    std::vector<float> force_cpu(dataset[device_id].force.size());
    neptb_data[device_id].eigenvalue.copy_to_host(eigenvalue_cpu.data());
    neptb_data[device_id].hamiltonian.copy_to_host(hamiltonian_cpu.data());
    dataset[device_id].force.copy_to_host(force_cpu.data());
    for (int n1 = 0; n1 < dataset[device_id].N * 4; ++n1) {
      fprintf(fid_eigenvalue, "%g\n", eigenvalue_cpu[n1]);
      for (int n2 = 0; n2 < dataset[device_id].N * 4; ++n2) {
        fprintf(fid_hamiltonian, "%g ", hamiltonian_cpu[n1 * dataset[device_id].N * 4 + n2]);
      }
      fprintf(fid_hamiltonian, "\n");
    }
    for (int n1 = 0; n1 < dataset[device_id].N; ++n1) {
      fprintf(fid_force, "%g %g %g\n", 
      force_cpu[n1], 
      force_cpu[n1 + dataset[device_id].N], 
      force_cpu[n1 + dataset[device_id].N*2]);
    }
    fclose(fid_eigenvalue);
    fclose(fid_hamiltonian);
    fclose(fid_force);
    exit(1);

    apply_ann<<<grid_size, block_size>>>(
      dataset[device_id].N,
      paramb,
      annmb[device_id],
      dataset[device_id].type.data(),
      neptb_data[device_id].descriptors.data(),
      para.q_scaler_gpu[device_id].data(),
      dataset[device_id].energy.data(),
      neptb_data[device_id].Fp.data());
    GPU_CHECK_KERNEL

    find_force_radial<<<grid_size, block_size>>>(
      dataset[device_id].N,
      neptb_data[device_id].NN_radial.data(),
      neptb_data[device_id].NL_radial.data(),
      paramb,
      annmb[device_id],
      dataset[device_id].type.data(),
      neptb_data[device_id].x12_radial.data(),
      neptb_data[device_id].y12_radial.data(),
      neptb_data[device_id].z12_radial.data(),
      neptb_data[device_id].Fp.data(),
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2,
      dataset[device_id].virial.data());
    GPU_CHECK_KERNEL

    find_force_angular<<<grid_size, block_size>>>(
      dataset[device_id].N,
      neptb_data[device_id].NN_angular.data(),
      neptb_data[device_id].NL_angular.data(),
      paramb,
      annmb[device_id],
      dataset[device_id].type.data(),
      neptb_data[device_id].x12_angular.data(),
      neptb_data[device_id].y12_angular.data(),
      neptb_data[device_id].z12_angular.data(),
      neptb_data[device_id].Fp.data(),
      neptb_data[device_id].sum_fxyz.data(),
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
        neptb_data[device_id].NN_angular.data(),
        neptb_data[device_id].NL_angular.data(),
        dataset[device_id].type.data(),
        neptb_data[device_id].x12_angular.data(),
        neptb_data[device_id].y12_angular.data(),
        neptb_data[device_id].z12_angular.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data(),
        dataset[device_id].energy.data());
      GPU_CHECK_KERNEL
    }
  }
}
