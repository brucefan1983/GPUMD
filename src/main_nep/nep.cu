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
#include "nep.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_utilities.cuh"

static __global__ void gpu_find_neighbor_list(
  const int N,
  const int* Na,
  const int* Na_sum,
  const float rc2_radial,
  const float rc2_angular,
  const float* __restrict__ box,
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
    const float* __restrict__ h = box + 18 * blockIdx.x;
    float x1 = x[n1];
    float y1 = y[n1];
    float z1 = z[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      if (n2 == n1) {
        continue;
      }
      float x12 = x[n2] - x1;
      float y12 = y[n2] - y1;
      float z12 = z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float distance_square = x12 * x12 + y12 * y12 + z12 * z12;
      if (distance_square < rc2_radial) {
        NL_radial[count_radial * N + n1] = n2;
        x12_radial[count_radial * N + n1] = x12;
        y12_radial[count_radial * N + n1] = y12;
        z12_radial[count_radial * N + n1] = z12;
        count_radial++;
      }
      if (distance_square < rc2_angular) {
        NL_angular[count_angular * N + n1] = n2;
        x12_angular[count_angular * N + n1] = x12;
        y12_angular[count_angular * N + n1] = y12;
        z12_angular[count_angular * N + n1] = z12;
        count_angular++;
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
  const NEP2::ParaMB paramb,
  const NEP2::ANN annmb,
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
    float q[MAX_DIM] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = n1 + N * i1;
      int n2 = g_NL[index];
      float x12 = g_x12[index];
      float y12 = g_y12[index];
      float z12 = g_z12[index];
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      find_fc(paramb.rc_radial, paramb.rcinv_radial, d12, fc12);
      int t2 = g_type[n2];
      float fn12[MAX_NUM_N];
      find_fn(paramb.n_max_radial, paramb.rcinv_radial, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float c = (paramb.num_types == 1)
                    ? 1.0f
                    : annmb.c[(n * paramb.num_types + t1) * paramb.num_types + t2];
        q[n] += fn12[n] * c;
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
  NEP2::ParaMB paramb,
  const NEP2::ANN annmb,
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
    float q[MAX_DIM] = {0.0f};

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
        find_fc(paramb.rc_angular, paramb.rcinv_angular, d12, fc12);
        int t2 = g_type[n2];
        float fn;
        find_fn(n, paramb.rcinv_angular, d12, fc12, fn);
        fn *=
          (paramb.num_types == 1)
            ? 1.0f
            : annmb
                .c[((paramb.n_max_radial + 1 + n) * paramb.num_types + t1) * paramb.num_types + t2];
        accumulate_s(d12, x12, y12, z12, fn, s);
      }
      find_q(paramb.n_max_angular + 1, n, s, q);
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc] * YLM[abc];
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

NEP2::NEP2(
  char* input_dir, Parameters& para, int N, int N_times_max_NN_radial, int N_times_max_NN_angular)
{
  paramb.rc_radial = para.rc_radial;
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rc_angular = para.rc_angular;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  annmb.dim = (para.n_max_radial + 1) + (para.n_max_angular + 1) * para.L_max;
  annmb.num_neurons1 = para.num_neurons1;
  paramb.num_types = para.num_types;
  annmb.num_para = para.number_of_variables;
  paramb.n_max_radial = para.n_max_radial;
  paramb.n_max_angular = para.n_max_angular;
  paramb.L_max = para.L_max;

  nep_data.NN_radial.resize(N);
  nep_data.NN_angular.resize(N);
  nep_data.NL_radial.resize(N_times_max_NN_radial);
  nep_data.NL_angular.resize(N_times_max_NN_angular);
  nep_data.x12_radial.resize(N_times_max_NN_radial);
  nep_data.y12_radial.resize(N_times_max_NN_radial);
  nep_data.z12_radial.resize(N_times_max_NN_radial);
  nep_data.x12_angular.resize(N_times_max_NN_angular);
  nep_data.y12_angular.resize(N_times_max_NN_angular);
  nep_data.z12_angular.resize(N_times_max_NN_angular);
  nep_data.f12x.resize(N_times_max_NN_angular);
  nep_data.f12y.resize(N_times_max_NN_angular);
  nep_data.f12z.resize(N_times_max_NN_angular);
  nep_data.descriptors.resize(N * annmb.dim);
  nep_data.Fp.resize(N * annmb.dim);
  nep_data.sum_fxyz.resize(N * (paramb.n_max_angular + 1) * NUM_OF_ABC);
}

void NEP2::update_potential(const float* parameters, ANN& ann)
{
  ann.w0 = parameters;
  ann.b0 = ann.w0 + ann.num_neurons1 * ann.dim;
  ann.w1 = ann.b0 + ann.num_neurons1;
  ann.b1 = ann.w1 + ann.num_neurons1;
  if (paramb.num_types > 1) {
    ann.c = ann.b1 + 1;
  }
}

static __device__ void
apply_ann_one_layer(const NEP2::ANN& ann, float* q, float& energy, float* energy_derivative)
{
  for (int n = 0; n < ann.num_neurons1; ++n) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < ann.dim; ++d) {
      w0_times_q += ann.w0[n * ann.dim + d] * q[d];
    }
    float x1 = tanh(w0_times_q - ann.b0[n]);
    energy += ann.w1[n] * x1;
    for (int d = 0; d < ann.dim; ++d) {
      float y1 = (1.0f - x1 * x1) * ann.w0[n * ann.dim + d];
      energy_derivative[d] += ann.w1[n] * y1;
    }
  }
  energy -= ann.b1[0];
}

static __global__ void apply_ann(
  const int N,
  const NEP2::ParaMB paramb,
  const NEP2::ANN annmb,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  float* g_pe,
  float* g_Fp)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    apply_ann_one_layer(annmb, q, F, Fp);
    g_pe[n1] = F;
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

static __global__ void find_force_radial(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP2::ParaMB paramb,
  const NEP2::ANN annmb,
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
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
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
      find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.n_max_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
      float f12[3] = {0.0f};
      float f21[3] = {0.0f};
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float tmp12 = g_Fp[n1 + n * N] * fnp12[n] * d12inv;
        float tmp21 = g_Fp[n2 + n * N] * fnp12[n] * d12inv;
        tmp12 *= (paramb.num_types == 1)
                   ? 1.0f
                   : annmb.c[(n * paramb.num_types + t1) * paramb.num_types + t2];
        tmp21 *= (paramb.num_types == 1)
                   ? 1.0f
                   : annmb.c[(n * paramb.num_types + t2) * paramb.num_types + t1];
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
          f21[d] -= tmp21 * r12[d];
        }
      }
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_virial_xx += r12[0] * f21[0];
      s_virial_yy += r12[1] * f21[1];
      s_virial_zz += r12[2] * f21[2];
      s_virial_xy += r12[0] * f21[1];
      s_virial_yz += r12[1] * f21[2];
      s_virial_zx += r12[2] * f21[0];
    }
    g_fx[n1] = s_fx;
    g_fy[n1] = s_fy;
    g_fz[n1] = s_fz;
    g_virial[n1] = s_virial_xx;
    g_virial[n1 + N] = s_virial_yy;
    g_virial[n1 + N * 2] = s_virial_zz;
    g_virial[n1 + N * 3] = s_virial_xy;
    g_virial[n1 + N * 4] = s_virial_yz;
    g_virial[n1 + N * 5] = s_virial_zx;
  }
}

static __global__ void find_partial_force_angular(
  const int N,
  const int* g_NN,
  const int* g_NL,
  const NEP2::ParaMB paramb,
  const NEP2::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < (paramb.n_max_angular + 1) * paramb.L_max; ++d) {
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
      find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
      int t2 = g_type[n2];
      float f12[3] = {0.0f};
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float fn;
        float fnp;
        find_fn_and_fnp(n, paramb.rcinv_angular, d12, fc12, fcp12, fn, fnp);
        const float c =
          (paramb.num_types == 1)
            ? 1.0f
            : annmb
                .c[((paramb.n_max_radial + 1 + n) * paramb.num_types + t1) * paramb.num_types + t2];
        fn *= c;
        fnp *= c;
        accumulate_f12(
          n, n1, paramb.n_max_radial + 1, paramb.n_max_angular + 1, d12, r12, fn, fnp, Fp, sum_fxyz,
          f12);
      }

      g_f12x[index] = f12[0] * 2.0f;
      g_f12y[index] = f12[1] * 2.0f;
      g_f12z[index] = f12[2] * 2.0f;
    }
  }
}

static __global__ void find_force_manybody(
  const int N,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const float* __restrict__ g_f12x,
  const float* __restrict__ g_f12y,
  const float* __restrict__ g_f12z,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    int neighbor_number = g_neighbor_number[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_neighbor_list[index];
      int neighbor_number_2 = g_neighbor_number[n2];
      float x12 = g_x12[index];
      float y12 = g_y12[index];
      float z12 = g_z12[index];
      float f12x = g_f12x[index];
      float f12y = g_f12y[index];
      float f12z = g_f12z[index];
      int offset = 0;
      for (int k = 0; k < neighbor_number_2; ++k) {
        if (n1 == g_neighbor_list[n2 + N * k]) {
          offset = k;
          break;
        }
      }
      index = offset * N + n2;
      float f21x = g_f12x[index];
      float f21y = g_f12y[index];
      float f21z = g_f12z[index];
      s_fx += f12x - f21x;
      s_fy += f12y - f21y;
      s_fz += f12z - f21z;
      s_virial_xx += x12 * f21x;
      s_virial_yy += y12 * f21y;
      s_virial_zz += z12 * f21z;
      s_virial_xy += x12 * f21y;
      s_virial_yz += y12 * f21z;
      s_virial_zx += z12 * f21x;
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1] += s_virial_xx;
    g_virial[n1 + N] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
  }
}

void NEP2::find_force(Parameters& para, const float* parameters, Dataset& dataset)
{
  CHECK(cudaMemcpyToSymbol(c_parameters, parameters, sizeof(float) * annmb.num_para));
  float* address_c_parameters;
  CHECK(cudaGetSymbolAddress((void**)&address_c_parameters, c_parameters));
  update_potential(address_c_parameters, annmb);

  float rc2_radial = para.rc_radial * para.rc_radial;
  float rc2_angular = para.rc_angular * para.rc_angular;

  gpu_find_neighbor_list<<<dataset.Nc, 256>>>(
    dataset.N, dataset.Na.data(), dataset.Na_sum.data(), rc2_radial, rc2_angular, dataset.h.data(),
    dataset.r.data(), dataset.r.data() + dataset.N, dataset.r.data() + dataset.N * 2,
    nep_data.NN_radial.data(), nep_data.NL_radial.data(), nep_data.NN_angular.data(),
    nep_data.NL_angular.data(), nep_data.x12_radial.data(), nep_data.y12_radial.data(),
    nep_data.z12_radial.data(), nep_data.x12_angular.data(), nep_data.y12_angular.data(),
    nep_data.z12_angular.data());
  CUDA_CHECK_KERNEL

  const int block_size = 32;
  const int grid_size = (dataset.N - 1) / block_size + 1;

  find_descriptors_radial<<<grid_size, block_size>>>(
    dataset.N, nep_data.NN_radial.data(), nep_data.NL_radial.data(), paramb, annmb,
    dataset.type.data(), nep_data.x12_radial.data(), nep_data.y12_radial.data(),
    nep_data.z12_radial.data(), nep_data.descriptors.data());
  CUDA_CHECK_KERNEL

  find_descriptors_angular<<<grid_size, block_size>>>(
    dataset.N, nep_data.NN_angular.data(), nep_data.NL_angular.data(), paramb, annmb,
    dataset.type.data(), nep_data.x12_angular.data(), nep_data.y12_angular.data(),
    nep_data.z12_angular.data(), nep_data.descriptors.data(), nep_data.sum_fxyz.data());
  CUDA_CHECK_KERNEL

  apply_ann<<<grid_size, block_size>>>(
    dataset.N, paramb, annmb, nep_data.descriptors.data(), para.q_scaler_gpu.data(),
    dataset.energy.data(), nep_data.Fp.data());
  CUDA_CHECK_KERNEL

  find_force_radial<<<grid_size, block_size>>>(
    dataset.N, nep_data.NN_radial.data(), nep_data.NL_radial.data(), paramb, annmb,
    dataset.type.data(), nep_data.x12_radial.data(), nep_data.y12_radial.data(),
    nep_data.z12_radial.data(), nep_data.Fp.data(), dataset.force.data(),
    dataset.force.data() + dataset.N, dataset.force.data() + dataset.N * 2, dataset.virial.data());
  CUDA_CHECK_KERNEL

  find_partial_force_angular<<<grid_size, block_size>>>(
    dataset.N, nep_data.NN_angular.data(), nep_data.NL_angular.data(), paramb, annmb,
    dataset.type.data(), nep_data.x12_angular.data(), nep_data.y12_angular.data(),
    nep_data.z12_angular.data(), nep_data.Fp.data(), nep_data.sum_fxyz.data(), nep_data.f12x.data(),
    nep_data.f12y.data(), nep_data.f12z.data());
  CUDA_CHECK_KERNEL

  find_force_manybody<<<grid_size, block_size>>>(
    dataset.N, nep_data.NN_angular.data(), nep_data.NL_angular.data(), nep_data.f12x.data(),
    nep_data.f12y.data(), nep_data.f12z.data(), nep_data.x12_angular.data(),
    nep_data.y12_angular.data(), nep_data.z12_angular.data(), dataset.force.data(),
    dataset.force.data() + dataset.N, dataset.force.data() + dataset.N * 2, dataset.virial.data());
  CUDA_CHECK_KERNEL
}
