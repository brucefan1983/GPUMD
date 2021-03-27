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

#include "error.cuh"
#include "gpu_vector.cuh"
#include "mic.cuh"
#include "neighbor.cuh"
#include "nep.cuh"

const int NUM_OF_ABC = 10;               // 1 + 3 + 6 for L_max = 2
const int SIZE_BOX_AND_INVERSE_BOX = 18; // (3 * 3) * 2

NEP::NEP(
  int num_neurons_2b,
  float r1_2b,
  float r2_2b,
  int num_neurons_3b,
  float r1_3b,
  float r2_3b,
  int num_neurons_mb,
  int n_max,
  int L_max)
{
  // 2body
  ann2b.dim = 1;
  ann2b.num_neurons_per_layer = num_neurons_2b;
  para2b.r1 = r1_2b;
  para2b.r2 = r2_2b;
  para2b.r2inv = 1.0f / para2b.r2;
  para2b.pi_factor = 3.1415927f / (r2_2b - r1_2b);
  // 3body
  ann3b.dim = 3;
  ann3b.num_neurons_per_layer = num_neurons_3b;
  para3b.r1 = r1_3b;
  para3b.r2 = r2_3b;
  para3b.r2inv = 1.0f / para3b.r2;
  para3b.pi_factor = 3.1415927f / (r2_3b - r1_3b);
  // manybody
  paramb.n_max = n_max;
  paramb.L_max = L_max;
  paramb.r1 = 0.0f;  // inner cutoff for manybody is fixed to 0
  paramb.r2 = r2_2b; // manybody has the same outer cutoff as twobody
  paramb.r2inv = 1.0f / paramb.r2;
  paramb.pi_factor = 3.1415927f / (paramb.r2 - paramb.r1);
  paramb.delta_r = paramb.r2 / paramb.n_max;
  annmb.dim = (n_max + 1) * (L_max + 1);
  annmb.num_neurons_per_layer = num_neurons_mb;
};

void NEP::initialize(int N, int MAX_ATOM_NUMBER)
{
  if (ann3b.num_neurons_per_layer > 0) {
    nep_data.NN3b.resize(N);
    nep_data.NL3b.resize(N * MAX_ATOM_NUMBER);
  }
  if (annmb.num_neurons_per_layer > 0) {
    nep_data.Fp.resize(N * annmb.dim);
    nep_data.sum_fxyz.resize(N * (paramb.n_max + 1) * NUM_OF_ABC);
  }
  if (ann3b.num_neurons_per_layer > 0 || annmb.num_neurons_per_layer > 0) {
    nep_data.f12x.resize(N * MAX_ATOM_NUMBER);
    nep_data.f12y.resize(N * MAX_ATOM_NUMBER);
    nep_data.f12z.resize(N * MAX_ATOM_NUMBER);
  }
}

void NEP::update_potential(const float* parameters)
{
  if (ann2b.num_neurons_per_layer > 0) {
    int offset = 0;
    update_potential(parameters, offset, ann2b);
  }
  if (ann3b.num_neurons_per_layer > 0) {
    int offset = 0;
    if (ann2b.num_neurons_per_layer > 0) {
      offset += ann2b.num_neurons_per_layer * (ann2b.num_neurons_per_layer + 4) + 1;
    }
    update_potential(parameters, offset, ann3b);
  }
  if (annmb.num_neurons_per_layer > 0) {
    int offset = 0;
    if (ann2b.num_neurons_per_layer > 0) {
      offset += ann2b.num_neurons_per_layer * (ann2b.num_neurons_per_layer + 4) + 1;
    }
    if (ann3b.num_neurons_per_layer > 0) {
      offset += ann3b.num_neurons_per_layer * (ann3b.num_neurons_per_layer + 6) + 1;
    }
    update_potential(parameters, offset, annmb);
  }
}

void NEP::update_potential(const float* parameters, const int offset, NEP::ANN& ann)
{
  for (int n = 0; n < ann.num_neurons_per_layer; ++n) {
    for (int d = 0; d < ann.dim; ++d) {
      ann.w0[n * ann.dim + d] = parameters[n * ann.dim + d + offset];
    }
    ann.b0[n] = parameters[n + ann.num_neurons_per_layer * ann.dim + offset];
    for (int m = 0; m < ann.num_neurons_per_layer; ++m) {
      int nm = n * ann.num_neurons_per_layer + m;
      ann.w1[nm] = parameters[nm + ann.num_neurons_per_layer * (ann.dim + 1) + offset];
    }
    ann.b1[n] = parameters
      [n + ann.num_neurons_per_layer * (ann.num_neurons_per_layer + (ann.dim + 1)) + offset];
    ann.w2[n] = parameters
      [n + ann.num_neurons_per_layer * (ann.num_neurons_per_layer + (ann.dim + 2)) + offset];
  }
  ann.b2 =
    parameters[ann.num_neurons_per_layer * (ann.num_neurons_per_layer + (ann.dim + 3)) + offset];
}

static __device__ void
apply_ann(const NEP::ANN& ann, float* q, float& energy, float* energy_derivative)
{
  // energy
  float x1[MAX_NUM_NEURONS_PER_LAYER] = {0.0f}; // states of the 1st hidden layer neurons
  float x2[MAX_NUM_NEURONS_PER_LAYER] = {0.0f}; // states of the 2nd hidden layer neurons
  for (int n = 0; n < ann.num_neurons_per_layer; ++n) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < ann.dim; ++d) {
      w0_times_q += ann.w0[n * ann.dim + d] * q[d];
    }
    x1[n] = tanh(w0_times_q - ann.b0[n]);
  }
  for (int n = 0; n < ann.num_neurons_per_layer; ++n) {
    for (int m = 0; m < ann.num_neurons_per_layer; ++m) {
      x2[n] += ann.w1[n * ann.num_neurons_per_layer + m] * x1[m];
    }
    x2[n] = tanh(x2[n] - ann.b1[n]);
  }
  for (int n = 0; n < ann.num_neurons_per_layer; ++n) {
    energy += ann.w2[n] * x2[n];
  }
  energy -= ann.b2;
  // energy gradient (compute it component by component)
  for (int d = 0; d < ann.dim; ++d) {
    float y1[MAX_NUM_NEURONS_PER_LAYER] = {0.0f}; // derivatives of the 1st hidden layer neurons
    float y2[MAX_NUM_NEURONS_PER_LAYER] = {0.0f}; // derivatives of the 2nd hidden layer neurons
    for (int n = 0; n < ann.num_neurons_per_layer; ++n) {
      y1[n] = (1.0f - x1[n] * x1[n]) * ann.w0[n * ann.dim + d];
    }
    for (int n = 0; n < ann.num_neurons_per_layer; ++n) {
      for (int m = 0; m < ann.num_neurons_per_layer; ++m) {
        y2[n] += ann.w1[n * ann.num_neurons_per_layer + m] * y1[m];
      }
      y2[n] *= 1.0f - x2[n] * x2[n];
    }
    for (int n = 0; n < ann.num_neurons_per_layer; ++n) {
      energy_derivative[d] += ann.w2[n] * y2[n];
    }
  }
}

static __device__ void find_fc(float r1, float r2, float pi_factor, float d12, float& fc)
{
  if (d12 < r1) {
    fc = 1.0f;
  } else if (d12 < r2) {
    fc = 0.5f * cos(pi_factor * (d12 - r1)) + 0.5f;
  } else {
    fc = 0.0f;
  }
}

static __device__ void
find_fc_and_fcp(float r1, float r2, float pi_factor, float d12, float& fc, float& fcp)
{
  if (d12 < r1) {
    fc = 1.0f;
    fcp = 0.0f;
  } else if (d12 < r2) {
    fc = 0.5f * cos(pi_factor * (d12 - r1)) + 0.5f;
    fcp = -sin(pi_factor * (d12 - r1)) * pi_factor * 0.5f;
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
  int* g_type,
  NEP::Para2B para2b,
  NEP::ANN ann2b,
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
      float q[1] = {d12 * para2b.r2inv};
      apply_ann(ann2b, q, p2, f2);
      f2[0] *= para2b.r2inv;
      float fc, fcp;
      find_fc_and_fcp(para2b.r1, para2b.r2, para2b.pi_factor, d12, fc, fcp);
      p2 *= fc;
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
      pe += p2 * 0.5f;
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
  NEP::Para3B para3b,
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
      if (d12sq < para3b.r2 * para3b.r2) {
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
  int* g_type,
  NEP::Para3B para3b,
  NEP::ANN ann3b,
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
      find_fc_and_fcp(para3b.r1, para3b.r2, para3b.pi_factor, d12, fc12, fcp12);
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
        find_fc(para3b.r1, para3b.r2, para3b.pi_factor, d13, fc13);
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

static __device__ void find_fn(const int n, const float delta_r, const int d12, float& fn)
{
  float tmp = d12 - n * delta_r;
  fn = exp(-tmp * tmp);
}

static __device__ void
find_fn_and_fnp(const int n, const float delta_r, const int d12, float& fn, float& fnp)
{
  float tmp = d12 - n * delta_r;
  fn = exp(-tmp * tmp);
  fnp = -2.0f * tmp * fn;
}

static __global__ void find_energy_manybody(
  int N,
  int* Na,
  int* Na_sum,
  int* g_NN,
  int* g_NL,
  int* g_type,
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const float* __restrict__ g_x,
  const float* __restrict__ g_y,
  const float* __restrict__ g_z,
  const float* __restrict__ g_box,
  float* g_pe,
  float* g_Fp,
  float* g_sum_fxyz)
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
    float q[MAX_DIM] = {0.0f};
    for (int n = 0; n <= paramb.n_max; ++n) {
      float sum_xyz[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < neighbor_number; ++i1) {
        int n2 = g_NL[n1 + N * i1];
        float x12 = g_x[n2] - x1;
        float y12 = g_y[n2] - y1;
        float z12 = g_z[n2] - z1;
        dev_apply_mic(h, x12, y12, z12);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        find_fc(paramb.r1, paramb.r2, paramb.pi_factor, d12, fc12);
        float fn;
        find_fn(n, paramb.delta_r, d12, fn);
        fn *= fc12;
        float d12inv = 1.0f / d12;
        x12 *= d12inv;
        y12 *= d12inv;
        z12 *= d12inv;
        sum_xyz[0] += fn;
        sum_xyz[1] += x12 * fn;
        sum_xyz[2] += y12 * fn;
        sum_xyz[3] += z12 * fn;
        sum_xyz[4] += x12 * x12 * fn;
        sum_xyz[5] += y12 * y12 * fn;
        sum_xyz[6] += z12 * z12 * fn;
        sum_xyz[7] += x12 * y12 * fn;
        sum_xyz[8] += x12 * z12 * fn;
        sum_xyz[9] += y12 * z12 * fn;
      }
      q[n * 3 + 0] = sum_xyz[0] * sum_xyz[0];
      q[n * 3 + 1] = sum_xyz[1] * sum_xyz[1] + sum_xyz[2] * sum_xyz[2] + sum_xyz[3] * sum_xyz[3];
      q[n * 3 + 2] = sum_xyz[7] * sum_xyz[7] + sum_xyz[8] * sum_xyz[8] + sum_xyz[9] * sum_xyz[9];
      q[n * 3 + 2] *= 2.0f;
      q[n * 3 + 2] += sum_xyz[4] * sum_xyz[4] + sum_xyz[5] * sum_xyz[5] + sum_xyz[6] * sum_xyz[6];
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = sum_xyz[abc];
      }
    }
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    apply_ann(annmb, q, F, Fp);
    g_pe[n1] += F;
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * N + n1] = Fp[d];
    }
  }
}

static __global__ void find_partial_force_manybody(
  int N,
  int* Na,
  int* Na_sum,
  int* g_NN,
  int* g_NL,
  int* g_type,
  NEP::ParaMB paramb,
  NEP::ANN annmb,
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
    float x1 = g_x[n1];
    float y1 = g_y[n1];
    float z1 = g_z[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      float r12[3] = {g_x[n2] - x1, g_y[n2] - y1, g_z[n2] - z1};
      dev_apply_mic(h, r12[0], r12[1], r12[2]);
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      find_fc_and_fcp(paramb.r1, paramb.r2, paramb.pi_factor, d12, fc12, fcp12);
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f};
      for (int n = 0; n <= paramb.n_max; ++n) {
        float fn;
        float fnp;
        find_fn_and_fnp(n, paramb.delta_r, d12, fn, fnp);
        // l=0
        float fn0 = fn * fc12;
        float fn0p = fnp * fc12 + fn * fcp12;
        float Fp0 = g_Fp[(n * 3 + 0) * N + n1];
        float sum_f0 = g_sum_fxyz[(n * NUM_OF_ABC + 0) * N + n1];
        float tmp = Fp0 * sum_f0 * fn0p * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp * r12[d];
        }
        // l=1
        float fn1 = fn0 * d12inv;
        float fn1p = fn0p * d12inv - fn0 * d12inv * d12inv;
        float Fp1 = g_Fp[(n * 3 + 1) * N + n1];
        float sum_f1[3] = {
          g_sum_fxyz[(n * NUM_OF_ABC + 1) * N + n1], g_sum_fxyz[(n * NUM_OF_ABC + 2) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 3) * N + n1]};
        float tmp1 =
          Fp1 * fn1p * (sum_f1[0] * r12[0] + sum_f1[1] * r12[1] + sum_f1[2] * r12[2]) * d12inv;
        float tmp2 = Fp1 * fn1;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp1 * r12[d] + tmp2 * sum_f1[d];
        }
        // l=2
        float fn2 = fn1 * d12inv;
        float fn2p = fn1p * d12inv - fn1 * d12inv * d12inv;
        float Fp2 = g_Fp[(n * 3 + 2) * N + n1];
        float sum_f2[6] = {
          g_sum_fxyz[(n * NUM_OF_ABC + 4) * N + n1], g_sum_fxyz[(n * NUM_OF_ABC + 5) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 6) * N + n1], g_sum_fxyz[(n * NUM_OF_ABC + 7) * N + n1],
          g_sum_fxyz[(n * NUM_OF_ABC + 8) * N + n1], g_sum_fxyz[(n * NUM_OF_ABC + 9) * N + n1]};
        tmp1 = Fp2 * fn2p *
               (sum_f2[0] * r12[0] * r12[0] + sum_f2[1] * r12[1] * r12[1] +
                sum_f2[2] * r12[2] * r12[2] + 2.0f * sum_f2[3] * r12[0] * r12[1] +
                2.0f * sum_f2[4] * r12[0] * r12[2] + 2.0f * sum_f2[5] * r12[1] * r12[2]) *
               d12inv;
        tmp2 = 2.0f * Fp2 * fn2;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp1 * r12[d] + tmp2 * sum_f2[d] * r12[d];
        }
        f12[0] += tmp2 * (sum_f2[3] * r12[1] + sum_f2[4] * r12[2]);
        f12[1] += tmp2 * (sum_f2[3] * r12[0] + sum_f2[5] * r12[2]);
        f12[2] += tmp2 * (sum_f2[4] * r12[0] + sum_f2[5] * r12[1]);
      }
      g_f12x[index] = f12[0] * 2.0f;
      g_f12y[index] = f12[1] * 2.0f;
      g_f12z[index] = f12[2] * 2.0f;
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

void NEP::find_force(
  int Nc,
  int N,
  int* Na,
  int* Na_sum,
  int max_Na,
  int* type,
  float* h,
  Neighbor* neighbor,
  float* r,
  GPU_Vector<float>& f,
  GPU_Vector<float>& virial,
  GPU_Vector<float>& pe)
{
  if (ann2b.num_neurons_per_layer > 0) {
    find_force_2body<<<Nc, max_Na>>>(
      N, Na, Na_sum, neighbor->NN, neighbor->NL, type, para2b, ann2b, r, r + N, r + N * 2, h,
      f.data(), f.data() + N, f.data() + N * 2, virial.data(), pe.data());
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
      N, Na, Na_sum, nep_data.NN3b.data(), nep_data.NL3b.data(), type, para3b, ann3b, r, r + N,
      r + N * 2, h, pe.data(), nep_data.f12x.data(), nep_data.f12y.data(), nep_data.f12z.data());
    CUDA_CHECK_KERNEL
    find_force_3body_or_manybody<<<Nc, max_Na>>>(
      N, Na, Na_sum, nep_data.NN3b.data(), nep_data.NL3b.data(), nep_data.f12x.data(),
      nep_data.f12y.data(), nep_data.f12z.data(), r, r + N, r + N * 2, h, f.data(), f.data() + N,
      f.data() + N * 2, virial.data());
    CUDA_CHECK_KERNEL
  }
  if (annmb.num_neurons_per_layer > 0) {
    find_energy_manybody<<<Nc, max_Na>>>(
      N, Na, Na_sum, neighbor->NN, neighbor->NL, type, paramb, annmb, r, r + N, r + N * 2, h,
      pe.data(), nep_data.Fp.data(), nep_data.sum_fxyz.data());
    CUDA_CHECK_KERNEL
    find_partial_force_manybody<<<Nc, max_Na>>>(
      N, Na, Na_sum, neighbor->NN, neighbor->NL, type, paramb, annmb, r, r + N, r + N * 2, h,
      nep_data.Fp.data(), nep_data.sum_fxyz.data(), nep_data.f12x.data(), nep_data.f12y.data(),
      nep_data.f12z.data());
    CUDA_CHECK_KERNEL
    find_force_3body_or_manybody<<<Nc, max_Na>>>(
      N, Na, Na_sum, neighbor->NN, neighbor->NL, nep_data.f12x.data(), nep_data.f12y.data(),
      nep_data.f12z.data(), r, r + N, r + N * 2, h, f.data(), f.data() + N, f.data() + N * 2,
      virial.data());
    CUDA_CHECK_KERNEL
  }
}
