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
Ref: Zheyong Fan et al., in preparison.
------------------------------------------------------------------------------*/

#include "mic.cuh"
#include "neighbor.cuh"
#include "nep2.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"

const int SIZE_BOX_AND_INVERSE_BOX = 18; // (3 * 3) * 2
// set by me:
const int MAX_NUM_NEURONS_PER_LAYER = 40; // largest ANN: input-40-40-output
const int MAX_NUM_N = 9;                  // n_max+1 = 8+1
const int MAX_NUM_L = 9;                  // L_max+1 = 8+1
// calculated:
const int MAX_DIM = MAX_NUM_N * MAX_NUM_L;
const int MAX_2B_SIZE = MAX_NUM_NEURONS_PER_LAYER * (MAX_NUM_NEURONS_PER_LAYER + 3 + 1) + 1;
const int MAX_3B_SIZE = MAX_NUM_NEURONS_PER_LAYER * (MAX_NUM_NEURONS_PER_LAYER + 3 + 3) + 1;
const int MAX_MB_SIZE = MAX_NUM_NEURONS_PER_LAYER * (MAX_NUM_NEURONS_PER_LAYER + 3 + MAX_DIM) + 1;
const int MAX_ANN_SIZE = MAX_2B_SIZE + MAX_3B_SIZE + MAX_MB_SIZE;
// constant memory
__constant__ float c_parameters[MAX_ANN_SIZE];

NEP2::NEP2(
  int num_neurons_2b,
  float rc_2b,
  int num_neurons_3b,
  float rc_3b,
  int num_neurons_mb,
  int n_max,
  int L_max)
{
  // 2body
  ann2b.dim = 1;
  ann2b.num_neurons_per_layer = num_neurons_2b;
  ann2b.num_para =
    ann2b.num_neurons_per_layer > 0
      ? ann2b.num_neurons_per_layer * (ann2b.num_neurons_per_layer + ann2b.dim + 3) + 1
      : 0;
  ann2b.num_neurons1 = num_neurons_2b;
  ann2b.num_neurons2 = num_neurons_2b;
  para2b.rc = rc_2b;
  para2b.rcinv = 1.0f / para2b.rc;
  // 3body
  ann3b.dim = 3;
  ann3b.num_neurons_per_layer = num_neurons_3b;
  ann3b.num_para =
    ann3b.num_neurons_per_layer > 0
      ? ann3b.num_neurons_per_layer * (ann3b.num_neurons_per_layer + ann3b.dim + 3) + 1
      : 0;
  ann3b.num_neurons1 = num_neurons_3b;
  ann3b.num_neurons2 = num_neurons_3b;
  para3b.rc = rc_3b;
  para3b.rcinv = 1.0f / para3b.rc;
  // manybody
  paramb.n_max = n_max;
  paramb.L_max = L_max;
  paramb.rc = rc_2b; // manybody has the same cutoff as twobody
  paramb.rcinv = 1.0f / paramb.rc;
  annmb.dim = (n_max + 1) * (L_max + 1);
  annmb.num_neurons_per_layer = num_neurons_mb;
  annmb.num_para =
    annmb.num_neurons_per_layer > 0
      ? annmb.num_neurons_per_layer * (annmb.num_neurons_per_layer + annmb.dim + 3) + 1
      : 0;
  annmb.num_neurons1 = num_neurons_mb;
  annmb.num_neurons2 = num_neurons_mb;
};

void NEP2::initialize(int N, int MAX_ATOM_NUMBER)
{
  if (ann3b.num_neurons_per_layer > 0) {
    nep_data.NN3b.resize(N);
    nep_data.NL3b.resize(N * MAX_ATOM_NUMBER);
  }
  if (ann3b.num_neurons_per_layer > 0 || annmb.num_neurons_per_layer > 0) {
    nep_data.f12x.resize(N * MAX_ATOM_NUMBER);
    nep_data.f12y.resize(N * MAX_ATOM_NUMBER);
    nep_data.f12z.resize(N * MAX_ATOM_NUMBER);
  }
}

void NEP2::update_potential(const float* parameters)
{
  const int num_para = ann2b.num_para + ann3b.num_para + annmb.num_para;
  CHECK(cudaMemcpyToSymbol(c_parameters, parameters, sizeof(float) * num_para));
  float* address_c_parameters;
  CHECK(cudaGetSymbolAddress((void**)&address_c_parameters, c_parameters));
  if (ann2b.num_neurons_per_layer > 0) {
    update_potential(address_c_parameters, ann2b);
  }
  if (ann3b.num_neurons_per_layer > 0) {
    update_potential(address_c_parameters + ann2b.num_para, ann3b);
  }
  if (annmb.num_neurons_per_layer > 0) {
    update_potential(address_c_parameters + ann2b.num_para + ann3b.num_para, annmb);
  }
}

void NEP2::update_potential(const float* parameters, ANN& ann)
{
  ann.w0 = parameters;
  ann.b0 = ann.w0 + ann.num_neurons_per_layer * ann.dim;
  ann.w1 = ann.b0 + ann.num_neurons_per_layer;
  ann.b1 = ann.w1 + ann.num_neurons_per_layer * ann.num_neurons_per_layer;
  ann.w2 = ann.b1 + ann.num_neurons_per_layer;
  ann.b2 = ann.w2 + ann.num_neurons_per_layer;
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

static __device__ void find_fc(float rc, float rcinv, float d12, float& fc)
{
  if (d12 < rc) {
    float x = d12 * rcinv;
    float y = 1.0f + x * x * (2.0f * x - 3.0f);
    fc = y * y;
  } else {
    fc = 0.0f;
  }
}

static __device__ void find_fc_and_fcp(float rc, float rcinv, float d12, float& fc, float& fcp)
{
  if (d12 < rc) {
    float x = d12 * rcinv;
    float y = 1.0f + x * x * (2.0f * x - 3.0f);
    fc = y * y;
    fcp = 12.0f * y * x * (x - 1.0f);
    fcp *= rcinv;
  } else {
    fc = 0.0f;
    fcp = 0.0f;
  }
}

static __global__ void find_force_2body(
  int N,
  int* Na,
  int* Na_sum,
  int* g_NN2b,
  int* g_NL2b,
  NEP2::Para2B para2b,
  NEP2::ANN ann2b,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  float* g_pe)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    const float* __restrict__ h = g_box + SIZE_BOX_AND_INVERSE_BOX * blockIdx.x;
    int neighbor_number = g_NN2b[n1];
    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];
    float pe = 0.0f;
    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;
    float virial_xx = 0.0f;
    float virial_yy = 0.0f;
    float virial_zz = 0.0f;
    float virial_xy = 0.0f;
    float virial_yz = 0.0f;
    float virial_zx = 0.0f;
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_NL2b[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float p2 = 0.0f, f2[1] = {0.0f};
      float q[1] = {d12 * para2b.rcinv};
      apply_ann(ann2b, q, p2, f2);
      f2[0] *= para2b.rcinv;
      float fc, fcp;
      find_fc_and_fcp(para2b.rc, para2b.rcinv, d12, fc, fcp);
      f2[0] = (f2[0] * fc + p2 * fcp) / d12;
      fx += x12 * f2[0];
      fy += y12 * f2[0];
      fz += z12 * f2[0];
      virial_xx -= x12 * x12 * f2[0] * 0.5f;
      virial_yy -= y12 * y12 * f2[0] * 0.5f;
      virial_zz -= z12 * z12 * f2[0] * 0.5f;
      virial_xy -= x12 * y12 * f2[0] * 0.5f;
      virial_yz -= y12 * z12 * f2[0] * 0.5f;
      virial_zx -= z12 * x12 * f2[0] * 0.5f;
      pe += p2 * fc * 0.5f;
    }
    g_fx[n1] = fx;
    g_fy[n1] = fy;
    g_fz[n1] = fz;
    g_virial[n1 + N * 0] = virial_xx;
    g_virial[n1 + N * 1] = virial_yy;
    g_virial[n1 + N * 2] = virial_zz;
    g_virial[n1 + N * 3] = virial_xy;
    g_virial[n1 + N * 4] = virial_yz;
    g_virial[n1 + N * 5] = virial_zx;
    g_pe[n1] = pe;
  }
}

static __global__ void find_neighbor_list_3body(
  int N,
  int* Na,
  int* Na_sum,
  int* g_NN2b,
  int* g_NL2b,
  NEP2::Para3B para3b,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box,
  int* g_NN3b,
  int* g_NL3b)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    const float* __restrict__ h = g_box + SIZE_BOX_AND_INVERSE_BOX * blockIdx.x;
    int neighbor_number = g_NN2b[n1];
    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];
    int count = 0;
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_NL2b[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float d12sq = x12 * x12 + y12 * y12 + z12 * z12;
      if (d12sq < para3b.rc * para3b.rc) {
        g_NL3b[n1 + N * (count++)] = n2;
      }
    }
    g_NN3b[n1] = count;
  }
}

static __global__ void find_partial_force_3body(
  int N,
  int* Na,
  int* Na_sum,
  int* g_neighbor_number,
  int* g_neighbor_list,
  NEP2::Para3B para3b,
  NEP2::ANN ann3b,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box,
  float* g_potential,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    const float* __restrict__ h = g_box + SIZE_BOX_AND_INVERSE_BOX * blockIdx.x;
    int neighbor_number = g_neighbor_number[n1];
    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];
    float pot_energy = 0.0f;
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_neighbor_list[index];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      find_fc_and_fcp(para3b.rc, para3b.rcinv, d12, fc12, fcp12);
      float p12 = 0.0f, f12[3] = {0.0f, 0.0f, 0.0f};
      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int n3 = g_neighbor_list[n1 + N * i2];
        if (n3 == n2) {
          continue;
        }
        float x13 = g_x[n3] - x1;
        float y13 = g_y[n3] - y1;
        float z13 = g_z[n3] - z1;
        dev_apply_mic(h, x13, y13, z13);
        float d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        float fc13;
        find_fc(para3b.rc, para3b.rcinv, d13, fc13);
        float x23 = x13 - x12;
        float y23 = y13 - y12;
        float z23 = z13 - z12;
        float d23 = sqrt(x23 * x23 + y23 * y23 + z23 * z23);
        float d23inv = 1.0f / d23;
        float q[3] = {d12 + d13, (d12 - d13) * (d12 - d13), d23};
        float p123 = 0.0f, f123[3] = {0.0f};
        apply_ann(ann3b, q, p123, f123);
        p12 += p123 * fc12 * fc13;
        float tmp = p123 * fcp12 * fc13 + (f123[0] + f123[1] * (d12 - d13) * 2.0f) * fc12 * fc13;
        f12[0] += 2.0f * (tmp * x12 * d12inv - f123[2] * fc12 * fc13 * x23 * d23inv);
        f12[1] += 2.0f * (tmp * y12 * d12inv - f123[2] * fc12 * fc13 * y23 * d23inv);
        f12[2] += 2.0f * (tmp * z12 * d12inv - f123[2] * fc12 * fc13 * z23 * d23inv);
      }
      pot_energy += p12;
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
    g_potential[n1] += pot_energy;
  }
}

static __global__ void find_force_3body_or_manybody(
  int N,
  int* Na,
  int* Na_sum,
  int* g_neighbor_number,
  int* g_neighbor_list,
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

static __device__ __forceinline__ void
find_fn(const int n_max, const float rcinv, const float d12, const float fc12, float* fn)
{
  float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
  fn[0] = 1.0f;
  fn[1] = x;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0f * x * fn[m - 1] - fn[m - 2];
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0f) * 0.5f * fc12;
  }
}

static __device__ __forceinline__ void find_fn_and_fnp(
  const int n_max,
  const float rcinv,
  const float d12,
  const float fc12,
  const float fcp12,
  float* fn,
  float* fnp)
{
  float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
  fn[0] = 1.0f;
  fnp[0] = 0.0f;
  fn[1] = x;
  fnp[1] = 1.0f;
  float u0 = 1.0f;
  float u1 = 2.0f * x;
  float u2;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0f * x * fn[m - 1] - fn[m - 2];
    fnp[m] = m * u1;
    u2 = 2.0f * x * u1 - u0;
    u0 = u1;
    u1 = u2;
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0f) * 0.5f;
    fnp[m] *= 2.0f * (d12 * rcinv - 1.0f) * rcinv;
    fnp[m] = fnp[m] * fc12 + fn[m] * fcp12;
    fn[m] *= fc12;
  }
}

#if USE_POLY

static __device__ __forceinline__ void
find_poly_cos(const int L_max, const float x, float* poly_cos)
{
  poly_cos[0] = 1.0f;
  poly_cos[1] = x;
  for (int l = 2; l <= L_max; ++l) {
    poly_cos[l] = poly_cos[l - 1] * x;
  }
}

static __device__ __forceinline__ void
find_poly_cos_der(const int L_max, const float x, float* poly_cos_der)
{
  poly_cos_der[0] = 0.0f;
  poly_cos_der[1] = 1.0f;
  float tmp = 1.0f;
  for (int l = 2; l <= L_max; ++l) {
    tmp *= x;
    poly_cos_der[l] = l * tmp;
  }
}

#else

static __device__ __forceinline__ void
find_poly_cos(const int L_max, const float x, float* poly_cos)
{
  poly_cos[0] = 0.079577471545948f;
  poly_cos[1] = 0.238732414637843f * x;
  float x2 = x * x;
  poly_cos[2] = 0.596831036594608f * x2 - 0.198943678864869f;
  float x3 = x2 * x;
  poly_cos[3] = 1.392605752054084f * x3 - 0.835563451232451f * x;
  float x4 = x3 * x;
  poly_cos[4] = 3.133362942121690f * x4 - 2.685739664675734f * x2 + 0.268573966467573f;
  float x5 = x4 * x;
  poly_cos[5] = 6.893398472667717f * x5 - 7.659331636297464f * x3 + 1.641285350635171f * x;
  float x6 = x5 * x;
  poly_cos[6] = 14.935696690780054f * x6 - 20.366859123790981f * x4 + 6.788953041263660f * x2 -
                0.323283478155412f;
}

static __device__ __forceinline__ void
find_poly_cos_der(const int L_max, const float x, float* poly_cos_der)
{
  poly_cos_der[0] = 0.0f;
  poly_cos_der[1] = 0.238732414637843f;
  poly_cos_der[2] = 1.193662073189215f * x;
  float x2 = x * x;
  poly_cos_der[3] = 4.177817256162252f * x2 - 0.835563451232451f;
  float x3 = x2 * x;
  poly_cos_der[4] = 12.533451768486758f * x3 - 5.371479329351468f * x;
  float x4 = x3 * x;
  poly_cos_der[5] = 34.466992363338584f * x4 - 22.977994908892391f * x2 + 1.641285350635171f;
  float x5 = x4 * x;
  poly_cos_der[6] = 89.614180144680319f * x5 - 81.467436495163923f * x3 + 13.577906082527321f * x;
}
#endif

static __global__ void find_partial_force_manybody(
  int N,
  int* Na,
  int* Na_sum,
  int* g_NN,
  int* g_NL,
  NEP2::ParaMB paramb,
  NEP2::ANN annmb,
  const float* __restrict__ g_atomic_number,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box,
  float* g_pe,
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
    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];
    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      find_fc(paramb.rc, paramb.rcinv, d12, fc12);
      fc12 *= g_atomic_number[n2];
      float fn12[MAX_NUM_N];
      find_fn(paramb.n_max, paramb.rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max; ++n) {
        q[n * (paramb.L_max + 1) + 0] += fn12[n];
      }
      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int n3 = g_NL[n1 + N * i2];
        float x13 = g_x[n3] - x1;
        float y13 = g_y[n3] - y1;
        float z13 = g_z[n3] - z1;
        dev_apply_mic(h, x13, y13, z13);
        float d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        float fc13;
        find_fc(paramb.rc, paramb.rcinv, d13, fc13);
        fc13 *= g_atomic_number[n3];
        float cos123 = (x12 * x13 + y12 * y13 + z12 * z13) / (d12 * d13);
        float poly_cos[MAX_NUM_L];
        find_poly_cos(paramb.L_max, cos123, poly_cos);
        for (int n = 0; n <= paramb.n_max; ++n) {
          for (int l = 1; l <= paramb.L_max; ++l) {
            q[n * (paramb.L_max + 1) + l] += fn12[n] * fc13 * poly_cos[l];
          }
        }
      }
    }
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    apply_ann(annmb, q, F, Fp);
    g_pe[n1] += F;
    // get partial force
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[n1 + N * i1];
      float r12[3] = {g_x[n2] - x1, g_y[n2] - y1, g_z[n2] - z1};
      dev_apply_mic(h, r12[0], r12[1], r12[2]);
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc, paramb.rcinv, d12, fc12, fcp12);
      float atomic_number_n2 = g_atomic_number[n2];
      fc12 *= atomic_number_n2;
      fcp12 *= atomic_number_n2;
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.n_max, paramb.rcinv, d12, fc12, fcp12, fn12, fnp12);
      float f12[3] = {0.0f};
      for (int n = 0; n <= paramb.n_max; ++n) {
        float tmp = Fp[n * (paramb.L_max + 1) + 0] * fnp12[n] * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp * r12[d];
        }
      }
      for (int i2 = 0; i2 < neighbor_number; ++i2) {
        int n3 = g_NL[n1 + N * i2];
        float x13 = g_x[n3] - x1;
        float y13 = g_y[n3] - y1;
        float z13 = g_z[n3] - z1;
        dev_apply_mic(h, x13, y13, z13);
        float d13 = sqrt(x13 * x13 + y13 * y13 + z13 * z13);
        float d13inv = 1.0f / d13;
        float fc13;
        find_fc(paramb.rc, paramb.rcinv, d13, fc13);
        fc13 *= g_atomic_number[n3];
        float cos123 = (r12[0] * x13 + r12[1] * y13 + r12[2] * z13) / (d12 * d13);
        float fn13[MAX_NUM_N];
        find_fn(paramb.n_max, paramb.rcinv, d13, fc13, fn13);
        float poly_cos[MAX_NUM_L];
        find_poly_cos(paramb.L_max, cos123, poly_cos);
        float poly_cos_der[MAX_NUM_L];
        find_poly_cos_der(paramb.L_max, cos123, poly_cos_der);
        float cos_der[3] = {
          x13 * d13inv - r12[0] * d12inv * cos123, y13 * d13inv - r12[1] * d12inv * cos123,
          z13 * d13inv - r12[2] * d12inv * cos123};
        for (int n = 0; n <= paramb.n_max; ++n) {
          float tmp_n_a = (fnp12[n] * fn13[0] + fnp12[0] * fn13[n]) * d12inv;
          float tmp_n_b = (fn12[n] * fn13[0] + fn12[0] * fn13[n]) * d12inv;
          for (int l = 1; l <= paramb.L_max; ++l) {
            float tmp_nl_a = Fp[n * (paramb.L_max + 1) + l] * tmp_n_a * poly_cos[l];
            float tmp_nl_b = Fp[n * (paramb.L_max + 1) + l] * tmp_n_b * poly_cos_der[l];
            for (int d = 0; d < 3; ++d) {
              f12[d] += tmp_nl_a * r12[d] + tmp_nl_b * cos_der[d];
            }
          }
        }
      }
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}

static __global__ void
initialize_properties(int N, float* g_pe, float* g_fx, float* g_fy, float* g_fz, float* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    g_pe[n1] = 0.0f;
    g_fx[n1] = 0.0f;
    g_fy[n1] = 0.0f;
    g_fz[n1] = 0.0f;
    g_virial[n1 + 0 * N] = 0.0f;
    g_virial[n1 + 1 * N] = 0.0f;
    g_virial[n1 + 2 * N] = 0.0f;
    g_virial[n1 + 3 * N] = 0.0f;
    g_virial[n1 + 4 * N] = 0.0f;
    g_virial[n1 + 5 * N] = 0.0f;
  }
}

void NEP2::find_force(
  int Nc,
  int N,
  int* Na,
  int* Na_sum,
  int max_Na,
  float* atomic_number,
  float* h,
  Neighbor* neighbor,
  float* r,
  GPU_Vector<float>& f,
  GPU_Vector<float>& virial,
  GPU_Vector<float>& pe)
{
  if (ann2b.num_neurons_per_layer > 0) {
    find_force_2body<<<Nc, max_Na>>>(
      N, Na, Na_sum, neighbor->NN, neighbor->NL, para2b, ann2b, r, r + N, r + N * 2, h, f.data(),
      f.data() + N, f.data() + N * 2, virial.data(), pe.data());
    CUDA_CHECK_KERNEL
  } else {
    initialize_properties<<<(N - 1) / 64 + 1, 64>>>(
      N, pe.data(), f.data(), f.data() + N, f.data() + N * 2, virial.data());
    CUDA_CHECK_KERNEL
  }
  if (ann3b.num_neurons_per_layer > 0) {
    find_neighbor_list_3body<<<Nc, max_Na>>>(
      N, Na, Na_sum, neighbor->NN, neighbor->NL, para3b, r, r + N, r + N * 2, h,
      nep_data.NN3b.data(), nep_data.NL3b.data());
    CUDA_CHECK_KERNEL
    find_partial_force_3body<<<Nc, max_Na>>>(
      N, Na, Na_sum, nep_data.NN3b.data(), nep_data.NL3b.data(), para3b, ann3b, r, r + N, r + N * 2,
      h, pe.data(), nep_data.f12x.data(), nep_data.f12y.data(), nep_data.f12z.data());
    CUDA_CHECK_KERNEL
    find_force_3body_or_manybody<<<Nc, max_Na>>>(
      N, Na, Na_sum, nep_data.NN3b.data(), nep_data.NL3b.data(), nep_data.f12x.data(),
      nep_data.f12y.data(), nep_data.f12z.data(), r, r + N, r + N * 2, h, f.data(), f.data() + N,
      f.data() + N * 2, virial.data());
    CUDA_CHECK_KERNEL
  }
  if (annmb.num_neurons_per_layer > 0) {
    find_partial_force_manybody<<<Nc, max_Na>>>(
      N, Na, Na_sum, neighbor->NN, neighbor->NL, paramb, annmb, atomic_number, r, r + N, r + N * 2,
      h, pe.data(), nep_data.f12x.data(), nep_data.f12y.data(), nep_data.f12z.data());
    CUDA_CHECK_KERNEL
    find_force_3body_or_manybody<<<Nc, max_Na>>>(
      N, Na, Na_sum, neighbor->NN, neighbor->NL, nep_data.f12x.data(), nep_data.f12y.data(),
      nep_data.f12z.data(), r, r + N, r + N * 2, h, f.data(), f.data() + N, f.data() + N * 2,
      virial.data());
    CUDA_CHECK_KERNEL
  }
}
