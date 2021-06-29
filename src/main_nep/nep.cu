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
Ref: Zheyong Fan et al., in preparation.
------------------------------------------------------------------------------*/

#include "dataset.cuh"
#include "mic.cuh"
#include "nep.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_utilities.cuh"

const int NUM_OF_ABC = 8; // 3 + 5 for L_max = 2
__constant__ float YLM[NUM_OF_ABC] = {0.238732414637843f, 0.119366207318922f, 0.119366207318922f,
                                      0.099471839432435f, 0.596831036594608f, 0.596831036594608f,
                                      0.149207759148652f, 0.149207759148652f};

const int SIZE_BOX_AND_INVERSE_BOX = 18;  // (3 * 3) * 2
const int MAX_NUM_NEURONS_PER_LAYER = 50; // largest ANN: input-50-50-output
const int MAX_NUM_N = 13;                 // n_max+1 = 12+1
const int MAX_NUM_L = 2;                  // L_max=2
const int MAX_DIM = MAX_NUM_N * MAX_NUM_L;
__constant__ float c_parameters[10000]; // less than 64 KB maximum

static __global__ void find_descriptors_radial(
  const int N,
  const int* Na,
  const int* Na_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP2::ParaMB paramb,
  const float* __restrict__ g_atomic_number,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box,
  float* g_descriptors)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    const float* __restrict__ h = g_box + SIZE_BOX_AND_INVERSE_BOX * blockIdx.x;
    float atomic_number_n1 = g_atomic_number[n1];
    int neighbor_number = g_NN[n1];
    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      find_fc(paramb.rc_radial, paramb.rcinv_radial, d12, fc12);
      fc12 *= atomic_number_n1 * g_atomic_number[n2];
      float fn12[MAX_NUM_N];
      find_fn(paramb.n_max_radial, paramb.rcinv_radial, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        q[n] += fn12[n];
      }
    }
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      g_descriptors[n1 + n * N] = q[n];
    }
  }
}

static __global__ void find_descriptors_angular(
  const int N,
  const int* Na,
  const int* Na_sum,
  const int* g_NN,
  const int* g_NL,
  NEP2::ParaMB paramb,
  const float* __restrict__ g_atomic_number,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box,
  float* g_descriptors,
  float* g_sum_fxyz)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    const float* __restrict__ h = g_box + SIZE_BOX_AND_INVERSE_BOX * blockIdx.x;
    float atomic_number_n1 = g_atomic_number[n1];
    int neighbor_number = g_NN[n1];
    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < neighbor_number; ++i1) {
        int n2 = g_NL[n1 + N * i1];
        float x12 = g_x[n2] - x1;
        float y12 = g_y[n2] - y1;
        float z12 = g_z[n2] - z1;
        dev_apply_mic(h, x12, y12, z12);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        find_fc(paramb.rc_angular, paramb.rcinv_angular, d12, fc12);
        fc12 *= atomic_number_n1 * g_atomic_number[n2];
        float fn;
        find_fn(n, paramb.rcinv_angular, d12, fc12, fn);
        float d12inv = 1.0f / d12;
        x12 *= d12inv;
        y12 *= d12inv;
        z12 *= d12inv;
        s[0] += z12 * fn;                       // Y10
        s[1] += x12 * fn;                       // Y11_real
        s[2] += y12 * fn;                       // Y11_imag
        s[3] += (3.0f * z12 * z12 - 1.0f) * fn; // Y20
        s[4] += x12 * z12 * fn;                 // Y21_real
        s[5] += y12 * z12 * fn;                 // Y21_imag
        s[6] += (x12 * x12 - y12 * y12) * fn;   // Y22_real
        s[7] += 2.0f * x12 * y12 * fn;          // Y22_imag
      }
      q[n] = YLM[0] * s[0] * s[0] + 2.0f * (YLM[1] * s[1] * s[1] + YLM[2] * s[2] * s[2]);
      q[(paramb.n_max_angular + 1) + n] =
        YLM[3] * s[3] * s[3] + 2.0f * (YLM[4] * s[4] * s[4] + YLM[5] * s[5] * s[5] +
                                       YLM[6] * s[6] * s[6] + YLM[7] * s[7] * s[7]);
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

void __global__ find_max_min(const int N, const float* g_q, float* g_q_scaler, float* g_q_min)
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
    g_q_scaler[bid] = 1.0f / (s_max[0] - s_min[0]);
    g_q_min[bid] = s_min[0];
  }
}

void __global__ normalize_descriptors(
  NEP2::ANN annmb, const int N, const float* g_q_scaler, const float* g_q_min, float* g_q)
{
  int n1 = blockDim.x * blockIdx.x + threadIdx.x;
  if (n1 < N) {
    for (int d = 0; d < annmb.dim; ++d) {
      g_q[n1 + d * N] = (g_q[n1 + d * N] - g_q_min[d]) * g_q_scaler[d];
    }
  }
}

NEP2::NEP2(char* input_dir, Parameters& para, Dataset& dataset)
{
  paramb.rc_radial = para.rc_radial;
  paramb.rcinv_radial = 1.0f / paramb.rc_radial;
  paramb.rc_angular = para.rc_angular;
  paramb.rcinv_angular = 1.0f / paramb.rc_angular;
  annmb.dim = (para.n_max_radial + 1) + (para.n_max_angular + 1) * para.L_max;
  annmb.num_neurons1 = para.num_neurons1;
  annmb.num_neurons2 = para.num_neurons2;
  annmb.num_para = (annmb.dim + 1) * annmb.num_neurons1;
  annmb.num_para += (annmb.num_neurons1 + 1) * annmb.num_neurons2;
  annmb.num_para += (annmb.num_neurons2 == 0 ? annmb.num_neurons1 : annmb.num_neurons2) + 1;
  paramb.n_max_radial = para.n_max_radial;
  paramb.n_max_angular = para.n_max_angular;
  paramb.L_max = para.L_max;
  nep_data.f12x.resize(dataset.N * dataset.max_NN_angular);
  nep_data.f12y.resize(dataset.N * dataset.max_NN_angular);
  nep_data.f12z.resize(dataset.N * dataset.max_NN_angular);
  nep_data.descriptors.resize(dataset.N * annmb.dim);
  nep_data.Fp.resize(dataset.N * annmb.dim);
  nep_data.sum_fxyz.resize(dataset.N * (paramb.n_max_angular + 1) * NUM_OF_ABC);

  // use radial neighbor list
  find_descriptors_radial<<<dataset.Nc, dataset.max_Na>>>(
    dataset.N, dataset.Na.data(), dataset.Na_sum.data(), dataset.NN_radial.data(),
    dataset.NL_radial.data(), paramb, dataset.atomic_number.data(), dataset.r.data(),
    dataset.r.data() + dataset.N, dataset.r.data() + dataset.N * 2, dataset.h.data(),
    nep_data.descriptors.data());
  CUDA_CHECK_KERNEL

  // use angular neighbor list
  find_descriptors_angular<<<dataset.Nc, dataset.max_Na>>>(
    dataset.N, dataset.Na.data(), dataset.Na_sum.data(), dataset.NN_angular.data(),
    dataset.NL_angular.data(), paramb, dataset.atomic_number.data(), dataset.r.data(),
    dataset.r.data() + dataset.N, dataset.r.data() + dataset.N * 2, dataset.h.data(),
    nep_data.descriptors.data(), nep_data.sum_fxyz.data());
  CUDA_CHECK_KERNEL

  // output descriptors
  char file_descriptors[200];
  strcpy(file_descriptors, input_dir);
  strcat(file_descriptors, "/descriptors.out");
  FILE* fid = my_fopen(file_descriptors, "w");
  std::vector<float> descriptors(dataset.N * annmb.dim);
  nep_data.descriptors.copy_to_host(descriptors.data());
  for (int n = 0; n < dataset.N; ++n) {
    for (int d = 0; d < annmb.dim; ++d) {
      fprintf(fid, "%g ", descriptors[d * dataset.N + n]);
    }
    fprintf(fid, "\n");
  }
  fclose(fid);

  para.q_scaler.resize(annmb.dim, Memory_Type::managed);
  para.q_min.resize(annmb.dim, Memory_Type::managed);
  find_max_min<<<annmb.dim, 1024>>>(
    dataset.N, nep_data.descriptors.data(), para.q_scaler.data(), para.q_min.data());
  CUDA_CHECK_KERNEL
  normalize_descriptors<<<(dataset.N - 1) / 64 + 1, 64>>>(
    annmb, dataset.N, para.q_scaler.data(), para.q_min.data(), nep_data.descriptors.data());
  CUDA_CHECK_KERNEL
}

void NEP2::update_potential(const float* parameters, ANN& ann)
{
  ann.w0 = parameters;
  ann.b0 = ann.w0 + ann.num_neurons1 * ann.dim;
  ann.w1 = ann.b0 + ann.num_neurons1;
  if (ann.num_neurons2 == 0) {
    ann.b1 = ann.w1 + ann.num_neurons1;
  } else {
    ann.b1 = ann.w1 + ann.num_neurons1 * ann.num_neurons2;
    ann.w2 = ann.b1 + ann.num_neurons2;
    ann.b2 = ann.w2 + ann.num_neurons2;
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

static __device__ void
apply_ann(const NEP2::ANN& ann, float* q, float& energy, float* energy_derivative)
{
  // energy
  float x1[MAX_NUM_NEURONS_PER_LAYER] = {0.0f}; // states of the 1st hidden layer neurons
  float x2[MAX_NUM_NEURONS_PER_LAYER] = {0.0f}; // states of the 2nd hidden layer neurons
  for (int n = 0; n < ann.num_neurons1; ++n) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < ann.dim; ++d) {
      w0_times_q += ann.w0[n * ann.dim + d] * q[d];
    }
    x1[n] = tanh(w0_times_q - ann.b0[n]);
  }
  for (int n = 0; n < ann.num_neurons2; ++n) {
    for (int m = 0; m < ann.num_neurons1; ++m) {
      x2[n] += ann.w1[n * ann.num_neurons1 + m] * x1[m];
    }
    x2[n] = tanh(x2[n] - ann.b1[n]);
    energy += ann.w2[n] * x2[n];
  }
  energy -= ann.b2[0];
  // energy gradient (compute it component by component)
  for (int d = 0; d < ann.dim; ++d) {
    float y2[MAX_NUM_NEURONS_PER_LAYER] = {0.0f};
    for (int n1 = 0; n1 < ann.num_neurons1; ++n1) {
      float y1 = (1.0f - x1[n1] * x1[n1]) * ann.w0[n1 * ann.dim + d];
      for (int n2 = 0; n2 < ann.num_neurons2; ++n2) {
        y2[n2] += ann.w1[n2 * ann.num_neurons1 + n1] * y1;
      }
    }
    for (int n2 = 0; n2 < ann.num_neurons2; ++n2) {
      energy_derivative[d] += ann.w2[n2] * (y2[n2] * (1.0f - x2[n2] * x2[n2]));
    }
  }
}

static __global__ void apply_ann(
  const int N,
  const int* Na,
  const int* Na_sum,
  const NEP2::ParaMB paramb,
  const NEP2::ANN annmb,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  float* g_pe,
  float* g_Fp)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = g_descriptors[n1 + d * N];
    }
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    if (annmb.num_neurons2 == 0) {
      apply_ann_one_layer(annmb, q, F, Fp);
    } else {
      apply_ann(annmb, q, F, Fp);
    }
    g_pe[n1] = F;
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

static __global__ void find_force_radial(
  const int N,
  const int* Na,
  const int* Na_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP2::ParaMB paramb,
  const NEP2::ANN annmb,
  const float* __restrict__ g_atomic_number,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box,
  const float* __restrict__ g_Fp,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    const float* __restrict__ h = g_box + SIZE_BOX_AND_INVERSE_BOX * blockIdx.x;
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
    float atomic_number_n1 = g_atomic_number[n1];
    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float atomic_number_n12 = atomic_number_n1 * g_atomic_number[n2];
      float r12[3] = {g_x[n2] - x1, g_y[n2] - y1, g_z[n2] - z1};
      dev_apply_mic(h, r12[0], r12[1], r12[2]);
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
        float tmp12 = g_Fp[n1 + n * N] * fnp12[n] * atomic_number_n12 * d12inv;
        float tmp21 = g_Fp[n2 + n * N] * fnp12[n] * atomic_number_n12 * d12inv;
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
  const int* Na,
  const int* Na_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP2::ParaMB paramb,
  const NEP2::ANN annmb,
  const float* __restrict__ g_atomic_number,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    const float* __restrict__ h = g_box + SIZE_BOX_AND_INVERSE_BOX * blockIdx.x;
    int neighbor_number = g_NN[n1];
    float atomic_number_n1 = g_atomic_number[n1];
    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x[n2] - x1, g_y[n2] - y1, g_z[n2] - z1};
      dev_apply_mic(h, r12[0], r12[1], r12[2]);
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
      float atomic_number_n12 = atomic_number_n1 * g_atomic_number[n2];
      fc12 *= atomic_number_n12;
      fcp12 *= atomic_number_n12;
      float f12[3] = {0.0f};
      for (int n = 0; n <= paramb.n_max_angular; ++n) {

        float fn;
        float fnp;
        find_fn_and_fnp(n, paramb.rcinv_angular, d12, fc12, fcp12, fn, fnp);

        float s[8] = {
          g_sum_fxyz[(n * NUM_OF_ABC + 0) * N + n1], g_sum_fxyz[(n * NUM_OF_ABC + 1) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 2) * N + n1], g_sum_fxyz[(n * NUM_OF_ABC + 3) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 4) * N + n1], g_sum_fxyz[(n * NUM_OF_ABC + 5) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 6) * N + n1], g_sum_fxyz[(n * NUM_OF_ABC + 7) * N + n1]};
        // l=1
        float fn1 = fn * d12inv;
        float fn1p = fnp * d12inv - fn * d12inv * d12inv;
        float Fp1 = g_Fp[n1 + ((paramb.n_max_radial + 1) + n) * N];
        float tmp =
          Fp1 * fn1p * d12inv * (s[0] * r12[2] + 2.0f * s[1] * r12[0] + 2.0f * s[2] * r12[1]);
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp * r12[d];
        }
        tmp = Fp1 * fn1;
        f12[0] += tmp * 2.0f * s[1];
        f12[1] += tmp * 2.0f * s[2];
        f12[2] += tmp * s[0];
        // l=2
        float fn2 = fn1 * d12inv;
        float fn2p = fn1p * d12inv - fn1 * d12inv * d12inv;
        float Fp2 = g_Fp[n1 + ((paramb.n_max_radial + 1) + (paramb.n_max_angular + 1) + n) * N];
        tmp = Fp2 * fn2p * d12inv *
              (s[3] * (3.0f * r12[2] * r12[2] - d12 * d12) + 2.0f * s[4] * r12[0] * r12[2] +
               2.0f * s[5] * r12[1] * r12[2] + 2.0f * s[6] * (r12[0] * r12[0] - r12[1] * r12[1]) +
               2.0f * s[7] * 2.0f * r12[0] * r12[1]);
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp * r12[d];
        }
        tmp = Fp2 * fn2;
        f12[0] += tmp * (-2.0f * s[3] * r12[0] + 2.0f * s[4] * r12[2] + 4.0f * s[6] * r12[0] +
                         4.0f * s[7] * r12[1]);
        f12[1] += tmp * (-2.0f * s[3] * r12[1] + 2.0f * s[5] * r12[2] - 4.0f * s[6] * r12[1] +
                         4.0f * s[7] * r12[0]);
        f12[2] += tmp * (4.0f * s[3] * r12[2] + 2.0f * s[4] * r12[0] + 2.0f * s[5] * r12[1]);
      }
      g_f12x[index] = f12[0] * 2.0f;
      g_f12y[index] = f12[1] * 2.0f;
      g_f12z[index] = f12[2] * 2.0f;
    }
  }
}

static __global__ void find_force_manybody(
  const int N,
  const int* Na,
  const int* Na_sum,
  const int* g_neighbor_number,
  const int* g_neighbor_list,
  const float* __restrict__ g_f12x,
  const float* __restrict__ g_f12y,
  const float* __restrict__ g_f12z,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    const float* __restrict__ h = g_box + SIZE_BOX_AND_INVERSE_BOX * blockIdx.x;
    int neighbor_number = g_neighbor_number[n1];
    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_neighbor_list[index];
      int neighbor_number_2 = g_neighbor_number[n2];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
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

void NEP2::find_force(
  Parameters& para,
  const int configuration_start,
  const int configuration_end,
  const float* parameters,
  Dataset& dataset)
{
  CHECK(cudaMemcpyToSymbol(c_parameters, parameters, sizeof(float) * annmb.num_para));
  float* address_c_parameters;
  CHECK(cudaGetSymbolAddress((void**)&address_c_parameters, c_parameters));
  update_potential(address_c_parameters, annmb);

  apply_ann<<<configuration_end - configuration_start, dataset.max_Na>>>(
    dataset.N, dataset.Na.data() + configuration_start, dataset.Na_sum.data() + configuration_start,
    paramb, annmb, nep_data.descriptors.data(), para.q_scaler.data(), dataset.pe.data(),
    nep_data.Fp.data());
  CUDA_CHECK_KERNEL

  // use radial neighbor list
  find_force_radial<<<configuration_end - configuration_start, dataset.max_Na>>>(
    dataset.N, dataset.Na.data() + configuration_start, dataset.Na_sum.data() + configuration_start,
    dataset.NN_radial.data(), dataset.NL_radial.data(), paramb, annmb, dataset.atomic_number.data(),
    dataset.r.data(), dataset.r.data() + dataset.N, dataset.r.data() + dataset.N * 2,
    dataset.h.data(), nep_data.Fp.data(), dataset.force.data(), dataset.force.data() + dataset.N,
    dataset.force.data() + dataset.N * 2, dataset.virial.data());
  CUDA_CHECK_KERNEL

  // use angular neighbor list
  find_partial_force_angular<<<configuration_end - configuration_start, dataset.max_Na>>>(
    dataset.N, dataset.Na.data() + configuration_start, dataset.Na_sum.data() + configuration_start,
    dataset.NN_angular.data(), dataset.NL_angular.data(), paramb, annmb,
    dataset.atomic_number.data(), dataset.r.data(), dataset.r.data() + dataset.N,
    dataset.r.data() + dataset.N * 2, dataset.h.data(), nep_data.Fp.data(),
    nep_data.sum_fxyz.data(), nep_data.f12x.data(), nep_data.f12y.data(), nep_data.f12z.data());
  CUDA_CHECK_KERNEL

  // use angular neighbor list
  find_force_manybody<<<configuration_end - configuration_start, dataset.max_Na>>>(
    dataset.N, dataset.Na.data() + configuration_start, dataset.Na_sum.data() + configuration_start,
    dataset.NN_angular.data(), dataset.NL_angular.data(), nep_data.f12x.data(),
    nep_data.f12y.data(), nep_data.f12z.data(), dataset.r.data(), dataset.r.data() + dataset.N,
    dataset.r.data() + dataset.N * 2, dataset.h.data(), dataset.force.data(),
    dataset.force.data() + dataset.N, dataset.force.data() + dataset.N * 2, dataset.virial.data());
  CUDA_CHECK_KERNEL
}
