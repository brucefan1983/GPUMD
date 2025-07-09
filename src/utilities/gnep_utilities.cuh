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

#pragma once

const int NUM_OF_ABC = 80; // 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 for L_max = 8
__constant__ float C3B[NUM_OF_ABC] = {
  0.238732414637843f, 0.119366207318922f, 0.119366207318922f, 0.099471839432435f,
  0.596831036594608f, 0.596831036594608f, 0.149207759148652f, 0.149207759148652f,
  0.139260575205408f, 0.104445431404056f, 0.104445431404056f, 1.044454314040563f,
  1.044454314040563f, 0.174075719006761f, 0.174075719006761f, 0.011190581936149f,
  0.223811638722978f, 0.223811638722978f, 0.111905819361489f, 0.111905819361489f,
  1.566681471060845f, 1.566681471060845f, 0.195835183882606f, 0.195835183882606f,
  0.013677377921960f, 0.102580334414698f, 0.102580334414698f, 2.872249363611549f,
  2.872249363611549f, 0.119677056817148f, 0.119677056817148f, 2.154187022708661f,
  2.154187022708661f, 0.215418702270866f, 0.215418702270866f, 0.004041043476943f,
  0.169723826031592f, 0.169723826031592f, 0.106077391269745f, 0.106077391269745f,
  0.424309565078979f, 0.424309565078979f, 0.127292869523694f, 0.127292869523694f,
  2.800443129521260f, 2.800443129521260f, 0.233370260793438f, 0.233370260793438f,
  0.004662742473395f, 0.004079899664221f, 0.004079899664221f, 0.024479397985326f,
  0.024479397985326f, 0.012239698992663f, 0.012239698992663f, 0.538546755677165f,
  0.538546755677165f, 0.134636688919291f, 0.134636688919291f, 3.500553911901575f,
  3.500553911901575f, 0.250039565135827f, 0.250039565135827f, 0.000082569397966f,
  0.005944996653579f, 0.005944996653579f, 0.104037441437634f, 0.104037441437634f,
  0.762941237209318f, 0.762941237209318f, 0.114441185581398f, 0.114441185581398f,
  5.950941650232678f, 5.950941650232678f, 0.141689086910302f, 0.141689086910302f,
  4.250672607309055f, 4.250672607309055f, 0.265667037956816f, 0.265667037956816f
};

__constant__ float Z_COEFFICIENT_1[2][2] = {
  {0.0f, 1.0f},
  {1.0f, 0.0f}
};

__constant__ float Z_COEFFICIENT_2[3][3] = {
  {-1.0f, 0.0f, 3.0f},
  {0.0f, 1.0f, 0.0f},
  {1.0f, 0.0f, 0.0f}
};

__constant__ float Z_COEFFICIENT_3[4][4] = {
  {0.0f, -3.0f, 0.0f, 5.0f},
  {-1.0f, 0.0f, 5.0f, 0.0f},
  {0.0f, 1.0f, 0.0f, 0.0f},
  {1.0f, 0.0f, 0.0f, 0.0f}
};

__constant__ float Z_COEFFICIENT_4[5][5] = {
  {3.0f, 0.0f, -30.0f, 0.0f, 35.0f},
  {0.0f, -3.0f, 0.0f, 7.0f, 0.0f},
  {-1.0f, 0.0f, 7.0f, 0.0f, 0.0f},
  {0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
  {1.0f, 0.0f, 0.0f, 0.0f, 0.0f}
};

__constant__ float Z_COEFFICIENT_5[6][6] = {
  {0.0f, 15.0f, 0.0f, -70.0f, 0.0f, 63.0f},
  {1.0f, 0.0f, -14.0f, 0.0f, 21.0f, 0.0f},
  {0.0f, -1.0f, 0.0f, 3.0f, 0.0f, 0.0f},
  {-1.0f, 0.0f, 9.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
  {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
};

__constant__ float Z_COEFFICIENT_6[7][7] = {
  {-5.0f, 0.0f, 105.0f, 0.0f, -315.0f, 0.0f, 231.0f},
  {0.0f, 5.0f, 0.0f, -30.0f, 0.0f, 33.0f, 0.0f},
  {1.0f, 0.0f, -18.0f, 0.0f, 33.0f, 0.0f, 0.0f},
  {0.0f, -3.0f, 0.0f, 11.0f, 0.0f, 0.0f, 0.0f},
  {-1.0f, 0.0f, 11.0f, 0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
  {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
};

__constant__ float Z_COEFFICIENT_7[8][8] = {
  {0.0f, -35.0f, 0.0f, 315.0f, 0.0f, -693.0f, 0.0f, 429.0f},
  {-5.0f, 0.0f, 135.0f, 0.0f, -495.0f, 0.0f, 429.0f, 0.0f},
  {0.0f, 15.0f, 0.0f, -110.0f, 0.0f, 143.0f, 0.0f, 0.0f},
  {3.0f, 0.0f, -66.0f, 0.0f, 143.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, -3.0f, 0.0f, 13.0f, 0.0f, 0.0f, 0.0f, 0.0f},
  {-1.0f, 0.0f, 13.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
  {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
};

__constant__ float Z_COEFFICIENT_8[9][9] = {
  {35.0f, 0.0f, -1260.0f, 0.0f, 6930.0f, 0.0f, -12012.0f, 0.0f, 6435.0f},
  {0.0f, -35.0f, 0.0f, 385.0f, 0.0f, -1001.0f, 0.0f, 715.0f, 0.0f},
  {-1.0f, 0.0f, 33.0f, 0.0f, -143.0f, 0.0f, 143.0f, 0.0f, 0.0f},
  {0.0f, 3.0f, 0.0f, -26.0f, 0.0f, 39.0f, 0.0f, 0.0f, 0.0f},
  {1.0f, 0.0f, -26.0f, 0.0f, 65.0f, 0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, -1.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
  {-1.0f, 0.0f, 15.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
  {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
  {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
};

__constant__ float COVALENT_RADIUS[94] = {
  0.426667f, 0.613333f, 1.6f,     1.25333f, 1.02667f, 1.0f,     0.946667f, 0.84f,    0.853333f,
  0.893333f, 1.86667f,  1.66667f, 1.50667f, 1.38667f, 1.46667f, 1.36f,     1.32f,    1.28f,
  2.34667f,  2.05333f,  1.77333f, 1.62667f, 1.61333f, 1.46667f, 1.42667f,  1.38667f, 1.33333f,
  1.32f,     1.34667f,  1.45333f, 1.49333f, 1.45333f, 1.53333f, 1.46667f,  1.52f,    1.56f,
  2.52f,     2.22667f,  1.96f,    1.85333f, 1.76f,    1.65333f, 1.53333f,  1.50667f, 1.50667f,
  1.44f,     1.53333f,  1.64f,    1.70667f, 1.68f,    1.68f,    1.64f,     1.76f,    1.74667f,
  2.78667f,  2.34667f,  2.16f,    1.96f,    2.10667f, 2.09333f, 2.08f,     2.06667f, 2.01333f,
  2.02667f,  2.01333f,  2.0f,     1.98667f, 1.98667f, 1.97333f, 2.04f,     1.94667f, 1.82667f,
  1.74667f,  1.64f,     1.57333f, 1.54667f, 1.48f,    1.49333f, 1.50667f,  1.76f,    1.73333f,
  1.73333f,  1.81333f,  1.74667f, 1.84f,    1.89333f, 2.68f,    2.41333f,  2.22667f, 2.10667f,
  2.02667f,  2.04f,     2.05333f, 2.06667f};

const int SIZE_BOX_AND_INVERSE_BOX = 18; // (3 * 3) * 2
const int MAX_NUM_N = 20;                // n_max+1 = 19+1
const int MAX_DIM = MAX_NUM_N * 7;
const int MAX_DIM_ANGULAR = MAX_NUM_N * 6;
const int MAX_LN = MAX_NUM_N * 8; 

static __device__ __forceinline__ void
complex_product(const float a, const float b, float& real_part, float& imag_part)
{
  const float real_temp = real_part;
  real_part = a * real_temp - b * imag_part;
  imag_part = a * imag_part + b * real_temp;
}

static __device__ void one_layer(
  const int N_des,
  const int N_neu,
  const float* w0,
  const float* b0,
  const float* w1,
  float* q,
  float& energy,
  float* energy_derivative)
{
  for (int n = 0; n < N_neu; ++n) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < N_des; ++d) {
      w0_times_q += w0[n * N_des + d] * q[d];
    }
    float x1 = tanh(w0_times_q - b0[n]);
    float tanh_der = 1.0f - x1 * x1;
    energy += w1[n] * x1;
    for (int d = 0; d < N_des; ++d) {
      float y1 = tanh_der * w0[n * N_des + d];
      energy_derivative[d] += w1[n] * y1;
    }
  }
  energy -= w1[N_neu];
}

static __device__ void apply_ann_one_layer_w2nd(
  const int N_des,
  const int N_neu,
  const float* w0,
  const float* b0,
  const float* w1,
  const int N,
  float* q,
  float& energy,
  float* energy_derivative,
  float* energy_derivative2,
  float* ep_wb,   // derivative of e_wb_grad w.r.t q[n]
  float* e_wb_grad) // energy w.r.t. w0, b0, w1, b1
{
  const int offset_b0 = N_des * N_neu;
  const int offset_w1 = offset_b0 + N_neu;
  for (int j = 0; j < N_neu; ++j) {
    const int j_N_des = j * N_des;
    float w0_times_q = 0.0f;
    for (int n = 0; n < N_des; ++n) {
      w0_times_q += w0[j_N_des + n] * q[n];
    }
    const float x1 = tanh(w0_times_q - b0[j]);
    const float tanh_der = 1.0f - x1 * x1;
    const float tanh_der2 = -2.0f * x1 * tanh_der;  // second derivative of tanh
    const float w1j = w1[j];
    const float delta_1 = w1j * tanh_der;
    energy += w1j * x1;
    for (int n = 0; n < N_des; ++n) {
      const int idx_w0 = j_N_des + n;
      const float w0jn = w0[idx_w0]; 
      float tmp1 = tanh_der * w0jn; // derivative of tanh w.r.t. q[n]
      float tmp2 = w1j * tanh_der2;
      energy_derivative[n] += w1j * tmp1;
      ep_wb[(offset_w1 + j) * N_des + n] = tmp1; // derivative of e_wb_grad[w1] w.r.t. q[n]
      ep_wb[(offset_b0 + j) * N_des + n] = -tmp2 * w0jn; // derivative of e_wb_grad[b0] w.r.t. q[n]
      // second derivative
      const float tmp2_qn = tmp2 * q[n];
      for (int m = 0; m < N_des; ++m) {
        const int idx_m = j_N_des + m;
        const float w0jm = w0[idx_m];
        const float tmp3 = tanh_der2 * w0jn * w0jm;
        energy_derivative2[(n * N_des + m) * N] += w1j * tmp3;
        ep_wb[idx_w0 * N_des + m] = tmp2_qn * w0jm; // derivative of e_wb_grad[w0] w.r.t. q[n]
        ep_wb[idx_w0 * N_des + m] += (m == n) ? delta_1 : 0.0f; 
      }
      e_wb_grad[idx_w0] += delta_1 * q[n]; // energy w.r.t. w0
    }
    e_wb_grad[offset_b0 + j] -= delta_1; // energy w.r.t. b0
    e_wb_grad[offset_w1 + j] += x1; // energy w.r.t. w1
    // w0 (N_neu * N_des), b0 (N_neu), w1 (N_neu), b1 (1)
  }
  e_wb_grad[offset_w1 + N_neu] = -1.0f; // energy w.r.t. b1
  energy -= w1[N_neu];
}
static __device__ void apply_ann_one_layer(
  const int N_des,
  const int N_neu,
  const float* w0,
  const float* b0,
  const float* w1,
  const float* b1,
  float* q,
  float& energy,
  float* energy_derivative)
{
  for (int n = 0; n < N_neu; ++n) {
    float w0_times_q = 0.0f;
    for (int d = 0; d < N_des; ++d) {
      w0_times_q += w0[n * N_des + d] * q[d];
    }
    float x1 = tanh(w0_times_q - b0[n]);
    float tanh_der = 1.0f - x1 * x1;
    energy += w1[n] * x1;
    for (int d = 0; d < N_des; ++d) {
      float y1 = tanh_der * w0[n * N_des + d];
      energy_derivative[d] += w1[n] * y1;
    }
  }
  energy -= b1[0];
}

static __device__ __forceinline__ void find_fc(float rc, float rcinv, float d12, float& fc)
{
  if (d12 < rc) {
    float x = d12 * rcinv;
    fc = 0.5f * cos(PI * x) + 0.5f;
  } else {
    fc = 0.0f;
  }
}

static __device__ __host__ __forceinline__ void
find_fc_and_fcp(float rc, float rcinv, float d12, float& fc, float& fcp)
{
  if (d12 < rc) {
    float x = d12 * rcinv;
    fc = 0.5f * cos(PI * x) + 0.5f;
    fcp = -1.5707963f * sin(PI * x);
    fcp *= rcinv;
  } else {
    fc = 0.0f;
    fcp = 0.0f;
  }
}

static __device__ __forceinline__ void
find_fc_and_fcp_zbl(float r1, float r2, float d12, float& fc, float& fcp)
{
  if (d12 < r1) {
    fc = 1.0f;
    fcp = 0.0f;
  } else if (d12 < r2) {
    float pi_factor = PI / (r2 - r1);
    fc = cos(pi_factor * (d12 - r1)) * 0.5f + 0.5f;
    fcp = -sin(pi_factor * (d12 - r1)) * pi_factor * 0.5f;
  } else {
    fc = 0.0f;
    fcp = 0.0f;
  }
}

static __device__ __forceinline__ void
find_phi_and_phip_zbl(float a, float b, float x, float& phi, float& phip)
{
  float tmp = a * exp(-b * x);
  phi += tmp;
  phip -= b * tmp;
}

static __device__ __forceinline__ void find_f_and_fp_zbl(
  const float zizj,
  const float a_inv,
  const float rc_inner,
  const float rc_outer,
  const float d12,
  const float d12inv,
  float& f,
  float& fp)
{
  const float x = d12 * a_inv;
  f = fp = 0.0f;
  const float Zbl_para[8] = {
    0.18175f, 3.1998f, 0.50986f, 0.94229f, 0.28022f, 0.4029f, 0.02817f, 0.20162f};
  find_phi_and_phip_zbl(Zbl_para[0], Zbl_para[1], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[2], Zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[4], Zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[6], Zbl_para[7], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  float fc, fcp;
  find_fc_and_fcp_zbl(rc_inner, rc_outer, d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

static __device__ __forceinline__ void find_f_and_fp_zbl(
  const float* zbl_para,
  const float zizj,
  const float a_inv,
  const float d12,
  const float d12inv,
  float& f,
  float& fp)
{
  const float x = d12 * a_inv;
  f = fp = 0.0f;
  find_phi_and_phip_zbl(zbl_para[2], zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[4], zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[6], zbl_para[7], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[8], zbl_para[9], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  float fc, fcp;
  find_fc_and_fcp_zbl(zbl_para[0], zbl_para[1], d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

static __device__ __forceinline__ void
find_fn(const int n, const float rcinv, const float d12, const float fc12, float& fn)
{
  if (n == 0) {
    fn = fc12;
  } else if (n == 1) {
    float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
    fn = (x + 1.0f) * 0.5f * fc12;
  } else {
    float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
    float t0 = 1.0f;
    float t1 = x;
    float t2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0f * x * t1 - t0;
      t0 = t1;
      t1 = t2;
    }
    fn = (t2 + 1.0f) * 0.5f * fc12;
  }
}

static __device__ __forceinline__ void find_fn_and_fnp(
  const int n,
  const float rcinv,
  const float d12,
  const float fc12,
  const float fcp12,
  float& fn,
  float& fnp)
{
  if (n == 0) {
    fn = fc12;
    fnp = fcp12;
  } else if (n == 1) {
    float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
    fn = (x + 1.0f) * 0.5;
    fnp = 2.0f * (d12 * rcinv - 1.0f) * rcinv * fc12 + fn * fcp12;
    fn *= fc12;
  } else {
    float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
    float t0 = 1.0f;
    float t1 = x;
    float t2;
    float u0 = 1.0f;
    float u1 = 2.0f * x;
    float u2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0f * x * t1 - t0;
      t0 = t1;
      t1 = t2;
      u2 = 2.0f * x * u1 - u0;
      u0 = u1;
      u1 = u2;
    }
    fn = (t2 + 1.0f) * 0.5f;
    fnp = n * u0 * 2.0f * (d12 * rcinv - 1.0f) * rcinv;
    fnp = fnp * fc12 + fn * fcp12;
    fn *= fc12;
  }
}

static __device__ __forceinline__ void
find_fn(const int n_max, const float rcinv, const float d12, const float fc12, float* fn)
{
  float x = 2.0f * (d12 * rcinv - 1.0f) * (d12 * rcinv - 1.0f) - 1.0f;
  float half_fc12 = 0.5f * fc12;
  fn[0] = fc12;
  fn[1] = (x + 1.0f) * half_fc12;
  float fn_m_minus_2 = 1.0f;
  float fn_m_minus_1 = x;
  float tmp = 0.0f;
  for (int m = 2; m <= n_max; ++m) {
    tmp = 2.0f * x * fn_m_minus_1 - fn_m_minus_2;
    fn_m_minus_2 = fn_m_minus_1;
    fn_m_minus_1 = tmp;
    fn[m] = (tmp + 1.0f) * half_fc12;
  }
}

static __device__ __host__ __forceinline__ void find_fn_and_fnp(
  const int n_max,
  const float rcinv,
  const float d12,
  const float fc12,
  const float fcp12,
  float* fn,
  float* fnp)
{
  float d12_mul_rcinv = d12 * rcinv;
  float x = 2.0f * (d12_mul_rcinv - 1.0f) * (d12_mul_rcinv - 1.0f) - 1.0f;
  fn[0] = fc12;
  fnp[0] = fcp12;
  fn[1] = (x + 1.0f) * 0.5f * fc12;
  fnp[1] = 2.0f * (d12_mul_rcinv - 1.0f) * rcinv * fc12 + (x + 1.0f) * 0.5f * fcp12;
  float u0 = 1.0f;
  float u1 = 2.0f * x;
  float u2;
  float fn_m_minus_2 = 1.0f;
  float fn_m_minus_1 = x;
  for (int m = 2; m <= n_max; ++m) {
    float fn_tmp1 = 2.0f * x * fn_m_minus_1 - fn_m_minus_2;
    fn_m_minus_2 = fn_m_minus_1;
    fn_m_minus_1 = fn_tmp1;
    float fnp_tmp = m * u1;
    u2 = 2.0f * x * u1 - u0;
    u0 = u1;
    u1 = u2;

    float fn_tmp2 = (fn_tmp1 + 1.0f) * 0.5f;
    fnp[m] = (fnp_tmp * 2.0f * (d12 * rcinv - 1.0f) * rcinv) * fc12 + fn_tmp2 * fcp12;
    fn[m] = fn_tmp2 * fc12;
  }
}

template <int L>
static __device__ __forceinline__ void calculate_s_one(
  const int N,
  const int n,
  const int n_max_angular_plus_1,
  const float* Fp,
  const float* sum_fxyz,
  float* s)
{
  const int L_minus_1 = L - 1;
  const int L_twice_plus_1 = 2 * L + 1;
  const int L_square_minus_1 = L * L - 1;
  const int index_base = n * NUM_OF_ABC + L_square_minus_1;
  const int index_0 = index_base * N;
  float Fp_factor = 2.0f * Fp[L_minus_1 * n_max_angular_plus_1 + n];
  s[0] = sum_fxyz[index_0] * C3B[L_square_minus_1] * Fp_factor;
  Fp_factor *= 2.0f;
  for (int k = 1; k < L_twice_plus_1; ++k) {
    int index_s = index_base + k;
    int index_s0 = index_s * N;
    s[k] = sum_fxyz[index_s0] * C3B[L_square_minus_1 + k] * Fp_factor;
  }
}

template <int L>
static __device__ __forceinline__ void calculate_s_one(
  const int N,
  const int n,
  const int n_max_angular_plus_1,
  const float* sum_fxyz,
  float* s)
{
  const int L_twice_plus_1 = 2 * L + 1;
  const int L_square_minus_1 = L * L - 1;
  const int index_base = n * NUM_OF_ABC + L_square_minus_1;
  const int index_0 = index_base * N;
  s[0] = 2.0f * sum_fxyz[index_0] * C3B[L_square_minus_1];
  for (int k = 1; k < L_twice_plus_1; ++k) {
    int index_s = index_base + k;
    int index_s0 = index_s * N;
    s[k] = 4.0f * sum_fxyz[index_s0] * C3B[L_square_minus_1 + k];
  }
}

template <int L>
static __device__ __forceinline__ void calculate_ec_one(
  const int N,
  const int n,
  const int n_max_angular_plus_1,
  const float* Fp,
  const float* sum_fxyz,
  const float* sum_s2xyz,
  const float* sum_s2xyz123,
  const float* s_c,
  float* s,
  float* ec,
  float* f,
  float* f123)
{
  const int L_minus_1 = L - 1;
  const int L_twice_plus_1 = 2 * L + 1;
  const int L_square_minus_1 = L * L - 1;
  float Fp_factor = 2.0f * Fp[L_minus_1 * n_max_angular_plus_1 + n];
  const int index_base = n * NUM_OF_ABC + L_square_minus_1;
  const int index_0 = index_base * N;
  const int index_1 = index_0 * 3;
  const int index_123 = index_1 * 2;
  const float base_factor = C3B[L_square_minus_1] * Fp_factor;
  const float base_factor_s_c = base_factor * s_c[0];
  (*ec) += sum_fxyz[index_0] * base_factor_s_c;
  s[0] = sum_fxyz[index_0] * base_factor;
  f[0] = sum_s2xyz[index_1] * base_factor_s_c;
  f[1] = sum_s2xyz[index_1 + N * 1] * base_factor_s_c;
  f[2] = sum_s2xyz[index_1 + N * 2] * base_factor_s_c;
  f123[0] = sum_s2xyz123[index_123] * base_factor_s_c; 
  f123[1] = sum_s2xyz123[index_123 + N * 1] * base_factor_s_c;
  f123[2] = sum_s2xyz123[index_123 + N * 2] * base_factor_s_c;
  f123[3] = sum_s2xyz123[index_123 + N * 3] * base_factor_s_c;
  f123[4] = sum_s2xyz123[index_123 + N * 4] * base_factor_s_c;
  f123[5] = sum_s2xyz123[index_123 + N * 5] * base_factor_s_c;
  Fp_factor *= 2.0f;
  for (int k = 1; k < L_twice_plus_1; ++k) {
    int index_s = index_base + k;
    int index_s0 = index_s * N;
    int index_s1 = index_s0 * 3;
    int index_s123 = index_s1 * 2;
    float c3b_val = C3B[L_square_minus_1 + k] * Fp_factor;
    float s_c_val = c3b_val * s_c[k];
    (*ec) += sum_fxyz[index_s0] * s_c_val;
    s[k] = sum_fxyz[index_s0] * c3b_val;
    f[0] += sum_s2xyz[index_s1] * s_c_val;
    f[1] += sum_s2xyz[index_s1 + N * 1] * s_c_val;
    f[2] += sum_s2xyz[index_s1 + N * 2] * s_c_val;
    f123[0] += sum_s2xyz123[index_s123] * s_c_val;
    f123[1] += sum_s2xyz123[index_s123 + N * 1] * s_c_val;
    f123[2] += sum_s2xyz123[index_s123 + N * 2] * s_c_val;
    f123[3] += sum_s2xyz123[index_s123 + N * 3] * s_c_val;
    f123[4] += sum_s2xyz123[index_s123 + N * 4] * s_c_val;
    f123[5] += sum_s2xyz123[index_s123 + N * 5] * s_c_val;
  }
}

template <int L>
static __device__ __forceinline__ void calculate_fc_one(
  const int N,
  const int n,
  const int n_max_angular_plus_1,
  const float* Fp,
  const float* sum_fxyz,
  const float* sum_s2xyz,
  const float* s_c,
  float* s,
  float* f)
{
  const int L_minus_1 = L - 1;
  const int L_twice_plus_1 = 2 * L + 1;
  const int L_square_minus_1 = L * L - 1;
  float Fp_factor = 2.0f * Fp[L_minus_1 * n_max_angular_plus_1 + n];
  const int index_1 = L_square_minus_1 * 3;
  const float base_factor = C3B[L_square_minus_1] * Fp_factor;
  const float base_factor_s_c = base_factor * s_c[0];
  s[0] = sum_fxyz[L_square_minus_1] * base_factor;
  f[0] += sum_s2xyz[index_1] * base_factor_s_c;
  f[1] += sum_s2xyz[index_1 + 1] * base_factor_s_c;
  f[2] += sum_s2xyz[index_1 + 2] * base_factor_s_c;
  Fp_factor *= 2.0f;
  for (int k = 1; k < L_twice_plus_1; ++k) {
    int index_s = L_square_minus_1 + k;
    int index_s1 = index_s * 3;
    float c3b_val = C3B[L_square_minus_1 + k] * Fp_factor;
    float s_c_val = c3b_val * s_c[k];
    s[k] = sum_fxyz[index_s] * c3b_val;
    f[0] += sum_s2xyz[index_s1] * s_c_val;
    f[1] += sum_s2xyz[index_s1 + 1] * s_c_val;
    f[2] += sum_s2xyz[index_s1 + 2] * s_c_val;
  }
}

template <int L>
static __device__ __forceinline__ void calculate_qc_one(
  const int N,
  const int n,
  const int n_max_angular_plus_1,
  const float* sum_fxyz,
  const float* s_c,
  float* qc)
{
  const int L_twice_plus_1 = 2 * L + 1;
  const int L_square_minus_1 = L * L - 1;
  const int index_base = n * NUM_OF_ABC + L_square_minus_1;
  const int index_0 = index_base * N;
  float qc_sum = 2.0f * sum_fxyz[index_0] * C3B[L_square_minus_1] * s_c[0];
  for (int k = 1; k < L_twice_plus_1; ++k) {
    int index_s = index_base + k;
    int index_s0 = index_s * N;
    qc_sum += 4.0f * sum_fxyz[index_s0] * C3B[L_square_minus_1 + k] * s_c[k];
  }
  (*qc) = qc_sum;
}

template <int L>
static __device__ __forceinline__ void accumulate_f12_one(
  const float d12inv,
  const float fn,
  const float fnp,
  const float* s,
  const float* r12,
  float* f12)
{
  const float dx[3] = {(1.0f - r12[0] * r12[0]) * d12inv, -r12[0] * r12[1] * d12inv, -r12[0] * r12[2] * d12inv};
  const float dy[3] = {-r12[0] * r12[1] * d12inv, (1.0f - r12[1] * r12[1]) * d12inv, -r12[1] * r12[2] * d12inv};
  const float dz[3] = {-r12[0] * r12[2] * d12inv, -r12[1] * r12[2] * d12inv, (1.0f - r12[2] * r12[2]) * d12inv};

  float z_pow[L + 1] = {1.0f};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = r12[2] * z_pow[n - 1];
  }

  float real_part = 1.0f;
  float imag_part = 0.0f;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    float z_factor = 0.0f;
    float dz_factor = 0.0f;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      float coeff;
      switch(L) {
        case 1: coeff = Z_COEFFICIENT_1[n1][n2]; break;
        case 2: coeff = Z_COEFFICIENT_2[n1][n2]; break;
        case 3: coeff = Z_COEFFICIENT_3[n1][n2]; break;
        case 4: coeff = Z_COEFFICIENT_4[n1][n2]; break;
        case 5: coeff = Z_COEFFICIENT_5[n1][n2]; break;
        case 6: coeff = Z_COEFFICIENT_6[n1][n2]; break;
        case 7: coeff = Z_COEFFICIENT_7[n1][n2]; break;
        case 8: coeff = Z_COEFFICIENT_8[n1][n2]; break;
      }
      z_factor += coeff * z_pow[n2];
      if (n2 > 0) {
        dz_factor += coeff * n2 * z_pow[n2 - 1];
      }
    }
    if (n1 == 0) {
      float factor1 = z_factor * fnp;
      float factor2 = fn * dz_factor;
      for (int d = 0; d < 3; ++d) {
        f12[d] += s[0] * (factor1 * r12[d] + factor2 * dz[d]);
      }
    } else {
      float real_part_n1 = n1 * real_part;
      float imag_part_n1 = n1 * imag_part;
      float z_factor_fn = z_factor * fn;
      for (int d = 0; d < 3; ++d) {
        float real_part_dx = dx[d];
        float imag_part_dy = dy[d];
        complex_product(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        f12[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor_fn;
      }
      complex_product(r12[0], r12[1], real_part, imag_part);
      float xy_temp = s[2 * n1 - 1] * real_part + s[2 * n1 - 0] * imag_part;
      float factor1 = xy_temp * z_factor * fnp;
      float factor2 = xy_temp * fn * dz_factor;
      for (int d = 0; d < 3; ++d) {
        f12[d] += factor1 * r12[d] + factor2 * dz[d];
      }
    }
  }
}

template <int L>
static __device__ __forceinline__ void accumulate_f12_one(
  const float d12inv,
  const float fn,
  const float fnp,
  const float* s,
  const float* r12,
  const float* r12_original,
  float* f12,
  float* f123)
{
  const float dx[3] = {(1.0f - r12[0] * r12[0]) * d12inv, -r12[0] * r12[1] * d12inv, -r12[0] * r12[2] * d12inv};
  const float dy[3] = {-r12[0] * r12[1] * d12inv, (1.0f - r12[1] * r12[1]) * d12inv, -r12[1] * r12[2] * d12inv};
  const float dz[3] = {-r12[0] * r12[2] * d12inv, -r12[1] * r12[2] * d12inv, (1.0f - r12[2] * r12[2]) * d12inv};

  float z_pow[L + 1] = {1.0f};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = r12[2] * z_pow[n - 1];
  }

  float real_part = 1.0f;
  float imag_part = 0.0f;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    float z_factor = 0.0f;
    float dz_factor = 0.0f;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      float coeff;
      switch(L) {
        case 1: coeff = Z_COEFFICIENT_1[n1][n2]; break;
        case 2: coeff = Z_COEFFICIENT_2[n1][n2]; break;
        case 3: coeff = Z_COEFFICIENT_3[n1][n2]; break;
        case 4: coeff = Z_COEFFICIENT_4[n1][n2]; break;
        case 5: coeff = Z_COEFFICIENT_5[n1][n2]; break;
        case 6: coeff = Z_COEFFICIENT_6[n1][n2]; break;
        case 7: coeff = Z_COEFFICIENT_7[n1][n2]; break;
        case 8: coeff = Z_COEFFICIENT_8[n1][n2]; break;
      }
      z_factor += coeff * z_pow[n2];
      if (n2 > 0) {
        dz_factor += coeff * n2 * z_pow[n2 - 1];
      }
    }
    if (n1 == 0) {
      float factor1 = z_factor * fnp;
      float factor2 = fn * dz_factor;
      for (int d = 0; d < 3; ++d) {
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        f12[d] += s[0] * (factor1 * r12[d] + factor2 * dz[d]);
        f123[d] += s[0] * r12_original[d] * (factor1 * r12[d] + factor2 * dz[d]);
        f123[d1+3] += s[0] * r12_original[d1] * (factor1 * r12[d] + factor2 * dz[d]);
      }
    } else {
      float real_part_n1 = n1 * real_part;
      float imag_part_n1 = n1 * imag_part;
      float z_factor_fn = z_factor * fn;
      for (int d = 0; d < 3; ++d) {
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        float real_part_dx = dx[d];
        float imag_part_dy = dy[d];
        complex_product(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        f12[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor_fn;
        f123[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor_fn * r12_original[d];
        f123[d1+3] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor_fn * r12_original[d1];
      }
      complex_product(r12[0], r12[1], real_part, imag_part);
      float xy_temp = s[2 * n1 - 1] * real_part + s[2 * n1 - 0] * imag_part;
      float factor1 = xy_temp * z_factor * fnp;
      float factor2 = xy_temp * fn * dz_factor;
      for (int d = 0; d < 3; ++d) {
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        f12[d] += factor1 * r12[d] + factor2 * dz[d];
        f123[d] += (factor1 * r12[d] + factor2 * dz[d]) * r12_original[d];
        f123[d1+3] += (factor1 * r12[d] + factor2 * dz[d]) * r12_original[d1];
      }
    }
  }
}

template <int L>
static __device__ __forceinline__ void calculate_fxyz_one(
  const float d12inv,
  const float fn,
  const float fnp,
  const float* s,
  const float* r12,
  const float* r12_original,
  float* f12,
  float* f123)
{
  const float dx[3] = {(1.0f - r12[0] * r12[0]) * d12inv, -r12[0] * r12[1] * d12inv, -r12[0] * r12[2] * d12inv};
  const float dy[3] = {-r12[0] * r12[1] * d12inv, (1.0f - r12[1] * r12[1]) * d12inv, -r12[1] * r12[2] * d12inv};
  const float dz[3] = {-r12[0] * r12[2] * d12inv, -r12[1] * r12[2] * d12inv, (1.0f - r12[2] * r12[2]) * d12inv};
  float z_pow[L + 1] = {1.0f};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = r12[2] * z_pow[n - 1];
  }

  float real_part = 1.0f;
  float imag_part = 0.0f;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    float z_factor = 0.0f;
    float dz_factor = 0.0f;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      float coeff;
      switch(L) {
        case 1: coeff = Z_COEFFICIENT_1[n1][n2]; break;
        case 2: coeff = Z_COEFFICIENT_2[n1][n2]; break;
        case 3: coeff = Z_COEFFICIENT_3[n1][n2]; break;
        case 4: coeff = Z_COEFFICIENT_4[n1][n2]; break;
        case 5: coeff = Z_COEFFICIENT_5[n1][n2]; break;
        case 6: coeff = Z_COEFFICIENT_6[n1][n2]; break;
        case 7: coeff = Z_COEFFICIENT_7[n1][n2]; break;
        case 8: coeff = Z_COEFFICIENT_8[n1][n2]; break;
      }
      z_factor += coeff * z_pow[n2];
      if (n2 > 0) {
        dz_factor += coeff * n2 * z_pow[n2 - 1];
      }
    }
    if (n1 == 0) {
      float factor1 = z_factor * fnp;
      float factor2 = fn * dz_factor;
      for (int d = 0; d < 3; ++d) {
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        f12[d] += s[0] * (factor1 * r12[d] + factor2 * dz[d]);
        f123[d] += s[0] * r12_original[d] * (factor1 * r12[d] + factor2 * dz[d]);
        f123[d1+3] += s[0] * r12_original[d1] * (factor1 * r12[d] + factor2 * dz[d]);
      }
    } else {
      float real_part_n1 = n1 * real_part;
      float imag_part_n1 = n1 * imag_part;
      float z_factor_fn = z_factor * fn;
      for (int d = 0; d < 3; ++d) {
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        float real_part_dx = dx[d];
        float imag_part_dy = dy[d];
        complex_product(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        f12[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor_fn;
        f123[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor_fn * r12_original[d];
        f123[d1+3] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor_fn * r12_original[d1];
      }
      complex_product(r12[0], r12[1], real_part, imag_part);
      float xy_temp = s[2 * n1 - 1] * real_part + s[2 * n1 - 0] * imag_part;
      float factor1 = z_factor * fnp;
      float factor2 = fn * dz_factor;
      for (int d = 0; d < 3; ++d) {
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        f12[d] += xy_temp * (factor1 * r12[d] + factor2 * dz[d]);
        f123[d] += xy_temp * (factor1 * r12[d] + factor2 * dz[d]) * r12_original[d];
        f123[d1+3] += xy_temp * (factor1 * r12[d] + factor2 * dz[d]) * r12_original[d1];
      }
    }
  }
}

template <int L>
static __device__ __forceinline__ void calculate_s_i1_one(
  const float d12inv,
  const float fn,
  const float fnp,
  const float* s,
  const float* r12,
  float* s_i1,
  float* sf)
{
  const float dx[3] = {(1.0f - r12[0] * r12[0]) * d12inv, -r12[0] * r12[1] * d12inv, -r12[0] * r12[2] * d12inv};
  const float dy[3] = {-r12[0] * r12[1] * d12inv, (1.0f - r12[1] * r12[1]) * d12inv, -r12[1] * r12[2] * d12inv};
  const float dz[3] = {-r12[0] * r12[2] * d12inv, -r12[1] * r12[2] * d12inv, (1.0f - r12[2] * r12[2]) * d12inv};
  const int start_index = L * L - 1;
  int s_index = L * L - 1;
  float z_pow[L + 1] = {1.0f};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = r12[2] * z_pow[n - 1];
  }

  float real_part = 1.0f;
  float imag_part = 0.0f;
  float real_part_s_i1 = r12[0];
  float imag_part_s_i1 = r12[1];
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    float z_factor = 0.0f;
    float dz_factor = 0.0f;
    float z_factor_i1 = 0.0f;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      float coeff;
      switch(L) {
        case 1: coeff = Z_COEFFICIENT_1[n1][n2]; break;
        case 2: coeff = Z_COEFFICIENT_2[n1][n2]; break;
        case 3: coeff = Z_COEFFICIENT_3[n1][n2]; break;
        case 4: coeff = Z_COEFFICIENT_4[n1][n2]; break;
        case 5: coeff = Z_COEFFICIENT_5[n1][n2]; break;
        case 6: coeff = Z_COEFFICIENT_6[n1][n2]; break;
        case 7: coeff = Z_COEFFICIENT_7[n1][n2]; break;
        case 8: coeff = Z_COEFFICIENT_8[n1][n2]; break;
      }
      z_factor += coeff * z_pow[n2];
      if (n2 > 0) {
        dz_factor += coeff * n2 * z_pow[n2 - 1];
      }
    }
    z_factor_i1 = z_factor * fn;
    if (n1 == 0) {
      float factor1 = z_factor * fnp;
      float factor2 = fn * dz_factor;
      s_i1[s_index++] = z_factor_i1;
      for (int d = 0; d < 3; ++d) {
        sf[d + 3 * start_index] = factor1 * r12[d] + factor2 * dz[d];
      }
    } else {
      float real_part_n1 = n1 * real_part;
      float imag_part_n1 = n1 * imag_part;
      float z_factor_fn = z_factor * fn;
      int abc = 3 * (start_index + 2 * n1 - 1);
      s_i1[s_index++] = z_factor_i1 * real_part_s_i1;
      s_i1[s_index++] = z_factor_i1 * imag_part_s_i1;
      complex_product(r12[0], r12[1], real_part_s_i1, imag_part_s_i1);
      for (int d = 0; d < 3; ++d) {
        float real_part_dx = dx[d];
        float imag_part_dy = dy[d];
        complex_product(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        sf[d + abc] = real_part_dx * z_factor_fn;
        sf[d + abc + 3] = imag_part_dy * z_factor_fn;
      }
      complex_product(r12[0], r12[1], real_part, imag_part);
      float factor1 = z_factor * fnp;
      float factor2 = fn * dz_factor;
      for (int d = 0; d < 3; ++d) {
        sf[d + abc] += real_part * (factor1 * r12[d] + factor2 * dz[d]);
        sf[d + abc + 3] += imag_part * (factor1 * r12[d] + factor2 * dz[d]);
      }
    }
  }
}


static __device__ __forceinline__ void calculate_s_i1(
  const int L_max,
  const float d12_i1,
  const float* r12_i1,
  const float gn,
  const float gnp,
  const float* sum_fxyz,
  float* s_i1,
  float* sum_s2xyz)
{
  const float d12inv_i1 = 1.0f / d12_i1;
  const float r12unit_i1[3] = {r12_i1[0]*d12inv_i1, r12_i1[1]*d12inv_i1, r12_i1[2]*d12inv_i1};
  if (L_max >= 1) {
    calculate_s_i1_one<1>(d12inv_i1, gn, gnp, sum_fxyz, r12unit_i1, s_i1, sum_s2xyz);
  }

  if (L_max >= 2) {
    calculate_s_i1_one<2>(d12inv_i1, gn, gnp, sum_fxyz, r12unit_i1, s_i1, sum_s2xyz);
  }

  if (L_max >= 3) {
    calculate_s_i1_one<3>(d12inv_i1, gn, gnp, sum_fxyz, r12unit_i1, s_i1, sum_s2xyz);
  }

  if (L_max >= 4) {
    calculate_s_i1_one<4>(d12inv_i1, gn, gnp, sum_fxyz, r12unit_i1, s_i1, sum_s2xyz);
  }

  if (L_max >= 5) {
    calculate_s_i1_one<5>(d12inv_i1, gn, gnp, sum_fxyz, r12unit_i1, s_i1, sum_s2xyz);
  }

  if (L_max >= 6) {
    calculate_s_i1_one<6>(d12inv_i1, gn, gnp, sum_fxyz, r12unit_i1, s_i1, sum_s2xyz);
  }

  if (L_max >= 7) {
    calculate_s_i1_one<7>(d12inv_i1, gn, gnp, sum_fxyz, r12unit_i1, s_i1, sum_s2xyz);
  }

  if (L_max >= 8) {
    calculate_s_i1_one<8>(d12inv_i1, gn, gnp, sum_fxyz, r12unit_i1, s_i1, sum_s2xyz);
  }
}

template <int L>
static __device__ __forceinline__ void accumulate_f12_one(
  const int N,
  const int n_max,
  const float d12inv,
  const float fn,
  const float fnp,
  const float* s,
  const float* r12,
  const float* r12_original,
  float* sf,
  float* sf123,
  float* f12)
{
  const float dx[3] = {(1.0f - r12[0] * r12[0]) * d12inv, -r12[0] * r12[1] * d12inv, -r12[0] * r12[2] * d12inv};
  const float dy[3] = {-r12[0] * r12[1] * d12inv, (1.0f - r12[1] * r12[1]) * d12inv, -r12[1] * r12[2] * d12inv};
  const float dz[3] = {-r12[0] * r12[2] * d12inv, -r12[1] * r12[2] * d12inv, (1.0f - r12[2] * r12[2]) * d12inv};
  const int N_ABC = 3 * NUM_OF_ABC * n_max;
  const int N_ABC123 = 2 * N_ABC;
  const int start_index = L * L - 1;
  float z_pow[L + 1] = {1.0f};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = r12[2] * z_pow[n - 1];
  }

  float real_part = 1.0f;
  float imag_part = 0.0f;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    float z_factor = 0.0f;
    float dz_factor = 0.0f;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      float coeff;
      switch(L) {
        case 1: coeff = Z_COEFFICIENT_1[n1][n2]; break;
        case 2: coeff = Z_COEFFICIENT_2[n1][n2]; break;
        case 3: coeff = Z_COEFFICIENT_3[n1][n2]; break;
        case 4: coeff = Z_COEFFICIENT_4[n1][n2]; break;
        case 5: coeff = Z_COEFFICIENT_5[n1][n2]; break;
        case 6: coeff = Z_COEFFICIENT_6[n1][n2]; break;
        case 7: coeff = Z_COEFFICIENT_7[n1][n2]; break;
        case 8: coeff = Z_COEFFICIENT_8[n1][n2]; break;
      }
      z_factor += coeff * z_pow[n2];
      if (n2 > 0) {
        dz_factor += coeff * n2 * z_pow[n2 - 1];
      }
    }
    if (n1 == 0) {
      float factor1 = z_factor * fnp;
      float factor2 = fn * dz_factor;
      for (int d = 0; d < 3; ++d) {
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        f12[d] += s[0] * (factor1 * r12[d] + factor2 * dz[d]);
        sf[N * (d + 3 * start_index + N_ABC)] += factor1 * r12[d] + factor2 * dz[d];
        sf123[N * (d + 6 * start_index + N_ABC123)] += (factor1 * r12[d] + factor2 * dz[d]) * r12_original[d];
        sf123[N * ((d1 + 3) + 6 * start_index + N_ABC123)] += (factor1 * r12[d] + factor2 * dz[d]) * r12_original[d1];
      }
    } else {
      float real_part_n1 = n1 * real_part;
      float imag_part_n1 = n1 * imag_part;
      float z_factor_fn = z_factor * fn;
      int abc = 3 * (start_index + 2 * n1 - 1);
      int abc123 = 2 * abc;
      for (int d = 0; d < 3; ++d) {
        float real_part_dx = dx[d];
        float imag_part_dy = dy[d];
        int index = N * (d + abc + N_ABC);
        int index123 = N * (d + abc123 + N_ABC123);
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        int index123_off = N * ((d1 + 3) + abc123 + N_ABC123); 
        complex_product(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        f12[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor_fn;
        sf[index] += real_part_dx * z_factor_fn;
        sf[index + 3 * N] += imag_part_dy * z_factor_fn;
        sf123[index123] += real_part_dx * z_factor_fn * r12_original[d];
        sf123[index123 + 6 * N] += imag_part_dy * z_factor_fn * r12_original[d];
        sf123[index123_off] += real_part_dx * z_factor_fn * r12_original[d1];
        sf123[index123_off + 6 * N] += imag_part_dy * z_factor_fn * r12_original[d1];
      }
      complex_product(r12[0], r12[1], real_part, imag_part);
      float xy_temp = s[2 * n1 - 1] * real_part + s[2 * n1 - 0] * imag_part;
      float factor1 = z_factor * fnp;
      float factor2 = fn * dz_factor;
      for (int d = 0; d < 3; ++d) {
        int index = N * (d + abc + N_ABC);
        int index123 = N * (d + abc123 + N_ABC123);
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        int index123_off = N * ((d1 + 3) + abc123 + N_ABC123); 
        f12[d] += xy_temp * (factor1 * r12[d] + factor2 * dz[d]);
        sf[index] += real_part * (factor1 * r12[d] + factor2 * dz[d]);
        sf[index + 3 * N] += imag_part * (factor1 * r12[d] + factor2 * dz[d]);
        sf123[index123] += real_part * (factor1 * r12[d] + factor2 * dz[d]) * r12_original[d];
        sf123[index123 + 6 * N] += imag_part * (factor1 * r12[d] + factor2 * dz[d]) * r12_original[d];
        sf123[index123_off] += real_part * (factor1 * r12[d] + factor2 * dz[d]) * r12_original[d1];
        sf123[index123_off + 6 * N] += imag_part * (factor1 * r12[d] + factor2 * dz[d]) * r12_original[d1];
      }
    }
  }
}

static __device__ __forceinline__ void accumulate_dfe(
  const int N,
  const int NM,
  const int L_max,
  const int n,
  const int n_max_angular_plus_1,
  const float d12,
  const float* r12,
  float fn,
  float fnp,
  const float* sum_fxyz,
  float* feat_x,
  float* feat_y,
  float* feat_z,
  float* feat_123_xx,
  float* feat_123_yy,
  float* feat_123_zz,
  float* feat_123_xy,
  float* feat_123_yz,
  float* feat_123_zx)
{
  const float d12inv = 1.0f / d12;
  const float r12unit[3] = {r12[0]*d12inv, r12[1]*d12inv, r12[2]*d12inv};
  // const int NM_nmax = n_max_angular_plus_1 * NM;

  if (L_max >= 1) {
    float s1[3];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_s_one<1>(N, n, n_max_angular_plus_1, sum_fxyz, s1);
    calculate_fxyz_one<1>(d12inv, fn, fnp, s1, r12unit, r12, f, f123);
    feat_x[0] = f[0];
    feat_y[0] = f[1];
    feat_z[0] = f[2];
    feat_123_xx[0] = f123[0];
    feat_123_yy[0] = f123[1];
    feat_123_zz[0] = f123[2];
    feat_123_xy[0] = f123[3];
    feat_123_yz[0] = f123[4];
    feat_123_zx[0] = f123[5];
  }

  if (L_max >= 2) {
    float s2[5];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_s_one<2>(N, n, n_max_angular_plus_1, sum_fxyz, s2);
    calculate_fxyz_one<2>(d12inv, fn, fnp, s2, r12unit, r12, f, f123);
    feat_x[n_max_angular_plus_1] = f[0];
    feat_y[n_max_angular_plus_1] = f[1];
    feat_z[n_max_angular_plus_1] = f[2];
    feat_123_xx[n_max_angular_plus_1] = f123[0];
    feat_123_yy[n_max_angular_plus_1] = f123[1];
    feat_123_zz[n_max_angular_plus_1] = f123[2];
    feat_123_xy[n_max_angular_plus_1] = f123[3];
    feat_123_yz[n_max_angular_plus_1] = f123[4];
    feat_123_zx[n_max_angular_plus_1] = f123[5];
  }

  if (L_max >= 3) {
    float s3[7];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_s_one<3>(N, n, n_max_angular_plus_1, sum_fxyz, s3);
    calculate_fxyz_one<3>(d12inv, fn, fnp, s3, r12unit, r12, f, f123);
    feat_x[2 * n_max_angular_plus_1] = f[0];
    feat_y[2 * n_max_angular_plus_1] = f[1];
    feat_z[2 * n_max_angular_plus_1] = f[2];
    feat_123_xx[2 * n_max_angular_plus_1] = f123[0];
    feat_123_yy[2 * n_max_angular_plus_1] = f123[1];
    feat_123_zz[2 * n_max_angular_plus_1] = f123[2];
    feat_123_xy[2 * n_max_angular_plus_1] = f123[3];
    feat_123_yz[2 * n_max_angular_plus_1] = f123[4];
    feat_123_zx[2 * n_max_angular_plus_1] = f123[5];
  }

  if (L_max >= 4) {
    float s4[9];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_s_one<4>(N, n, n_max_angular_plus_1, sum_fxyz, s4);
    calculate_fxyz_one<4>(d12inv, fn, fnp, s4, r12unit, r12, f, f123);
    feat_x[3 * n_max_angular_plus_1] = f[0];
    feat_y[3 * n_max_angular_plus_1] = f[1];
    feat_z[3 * n_max_angular_plus_1] = f[2];
    feat_123_xx[3 * n_max_angular_plus_1] = f123[0];
    feat_123_yy[3 * n_max_angular_plus_1] = f123[1];
    feat_123_zz[3 * n_max_angular_plus_1] = f123[2];
    feat_123_xy[3 * n_max_angular_plus_1] = f123[3];
    feat_123_yz[3 * n_max_angular_plus_1] = f123[4];
    feat_123_zx[3 * n_max_angular_plus_1] = f123[5];
  }

  if (L_max >= 5) {
    float s5[11];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_s_one<5>(N, n, n_max_angular_plus_1, sum_fxyz, s5);
    calculate_fxyz_one<5>(d12inv, fn, fnp, s5, r12unit, r12, f, f123);
    feat_x[4 * n_max_angular_plus_1] = f[0];
    feat_y[4 * n_max_angular_plus_1] = f[1];
    feat_z[4 * n_max_angular_plus_1] = f[2];
    feat_123_xx[4 * n_max_angular_plus_1] = f123[0];
    feat_123_yy[4 * n_max_angular_plus_1] = f123[1];
    feat_123_zz[4 * n_max_angular_plus_1] = f123[2];
    feat_123_xy[4 * n_max_angular_plus_1] = f123[3];
    feat_123_yz[4 * n_max_angular_plus_1] = f123[4];
    feat_123_zx[4 * n_max_angular_plus_1] = f123[5];
  }

  if (L_max >= 6) {
    float s6[13];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_s_one<6>(N, n, n_max_angular_plus_1, sum_fxyz, s6);
    calculate_fxyz_one<6>(d12inv, fn, fnp, s6, r12unit, r12, f, f123);
    feat_x[5 * n_max_angular_plus_1] = f[0];
    feat_y[5 * n_max_angular_plus_1] = f[1];
    feat_z[5 * n_max_angular_plus_1] = f[2];
    feat_123_xx[5 * n_max_angular_plus_1] = f123[0];
    feat_123_yy[5 * n_max_angular_plus_1] = f123[1];
    feat_123_zz[5 * n_max_angular_plus_1] = f123[2];
    feat_123_xy[5 * n_max_angular_plus_1] = f123[3];
    feat_123_yz[5 * n_max_angular_plus_1] = f123[4];
    feat_123_zx[5 * n_max_angular_plus_1] = f123[5];
  }

  if (L_max >= 7) {
    float s7[15];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_s_one<7>(N, n, n_max_angular_plus_1, sum_fxyz, s7);
    calculate_fxyz_one<7>(d12inv, fn, fnp, s7, r12unit, r12, f, f123);
    feat_x[6 * n_max_angular_plus_1] = f[0];
    feat_y[6 * n_max_angular_plus_1] = f[1];
    feat_z[6 * n_max_angular_plus_1] = f[2];
    feat_123_xx[6 * n_max_angular_plus_1] = f123[0];
    feat_123_yy[6 * n_max_angular_plus_1] = f123[1];
    feat_123_zz[6 * n_max_angular_plus_1] = f123[2];
    feat_123_xy[6 * n_max_angular_plus_1] = f123[3];
    feat_123_yz[6 * n_max_angular_plus_1] = f123[4];
    feat_123_zx[6 * n_max_angular_plus_1] = f123[5];
  }

  if (L_max >= 8) {
    float s8[17];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_s_one<8>(N, n, n_max_angular_plus_1, sum_fxyz, s8);
    calculate_fxyz_one<8>(d12inv, fn, fnp, s8, r12unit, r12, f, f123);
    feat_x[7 * n_max_angular_plus_1] = f[0];
    feat_y[7 * n_max_angular_plus_1] = f[1];
    feat_z[7 * n_max_angular_plus_1] = f[2];
    feat_123_xx[7 * n_max_angular_plus_1] = f123[0];
    feat_123_yy[7 * n_max_angular_plus_1] = f123[1];
    feat_123_zz[7 * n_max_angular_plus_1] = f123[2];
    feat_123_xy[7 * n_max_angular_plus_1] = f123[3];
    feat_123_yz[7 * n_max_angular_plus_1] = f123[4];
    feat_123_zx[7 * n_max_angular_plus_1] = f123[5];
  }
}

static __device__ __forceinline__ void accumulate_f12(
  const int N,
  const bool requires_grad,
  const int L_max,
  const int n,
  const int n_max_angular_plus_1,
  const float d12,
  const float* r12,
  const float fn,
  const float fnp,
  const float* Fp,
  const float* sum_fxyz,
  float* sum_s2xyz,
  float* sum_s2xyz123,
  float* f12)
{
  const float d12inv = 1.0f / d12;
  const float r12unit[3] = {r12[0]*d12inv, r12[1]*d12inv, r12[2]*d12inv};

  if (L_max >= 1) {
    float s1[3];
    calculate_s_one<1>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s1);
    if (requires_grad) {
      accumulate_f12_one<1>(N, n, d12inv, fn, fnp, s1, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<1>(d12inv, fn, fnp, s1, r12unit, f12);
    }
  }

  if (L_max >= 2) {
    float s2[5];
    calculate_s_one<2>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s2);
    if (requires_grad) {
      accumulate_f12_one<2>(N, n, d12inv, fn, fnp, s2, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<2>(d12inv, fn, fnp, s2, r12unit, f12);
    }
  }

  if (L_max >= 3) {
    float s3[7];
    calculate_s_one<3>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s3);
    if (requires_grad) {
      accumulate_f12_one<3>(N, n, d12inv, fn, fnp, s3, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<3>(d12inv, fn, fnp, s3, r12unit, f12);
    }
  }

  if (L_max >= 4) {
    float s4[9];
    calculate_s_one<4>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s4);
    if (requires_grad) {
      accumulate_f12_one<4>(N, n, d12inv, fn, fnp, s4, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<4>(d12inv, fn, fnp, s4, r12unit, f12);
    }
  }

  if (L_max >= 5) {
    float s5[11];
    calculate_s_one<5>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s5);
    if (requires_grad) {
      accumulate_f12_one<5>(N, n, d12inv, fn, fnp, s5, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<5>(d12inv, fn, fnp, s5, r12unit, f12);
    }
  }

  if (L_max >= 6) {
    float s6[13];
    calculate_s_one<6>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s6);
    if (requires_grad) {
      accumulate_f12_one<6>(N, n, d12inv, fn, fnp, s6, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<6>(d12inv, fn, fnp, s6, r12unit, f12);
    }
  }

  if (L_max >= 7) {
    float s7[15];
    calculate_s_one<7>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s7);
    if (requires_grad) {
      accumulate_f12_one<7>(N, n, d12inv, fn, fnp, s7, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<7>(d12inv, fn, fnp, s7, r12unit, f12);
    }
  }

  if (L_max >= 8) {
    float s8[17];
    calculate_s_one<8>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s8);
    if (requires_grad) {
      accumulate_f12_one<8>(N, n, d12inv, fn, fnp, s8, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<8>(d12inv, fn, fnp, s8, r12unit, f12);
    }
  }
}


template <int L>
static __device__ __forceinline__ void
calculate_sc_one(
  const float x12,
  const float y12,
  const float z12,
  const float fn,
  float* s)
{
  int s_index = 0;
  float z_pow[L + 1] = {1.0f};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = z12 * z_pow[n - 1];
  }
  float real_part = x12;
  float imag_part = y12;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    float z_factor = 0.0f;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      switch(L) {
        case 1: z_factor += Z_COEFFICIENT_1[n1][n2] * z_pow[n2]; break;
        case 2: z_factor += Z_COEFFICIENT_2[n1][n2] * z_pow[n2]; break;
        case 3: z_factor += Z_COEFFICIENT_3[n1][n2] * z_pow[n2]; break;
        case 4: z_factor += Z_COEFFICIENT_4[n1][n2] * z_pow[n2]; break;
        case 5: z_factor += Z_COEFFICIENT_5[n1][n2] * z_pow[n2]; break;
        case 6: z_factor += Z_COEFFICIENT_6[n1][n2] * z_pow[n2]; break;
        case 7: z_factor += Z_COEFFICIENT_7[n1][n2] * z_pow[n2]; break;
        case 8: z_factor += Z_COEFFICIENT_8[n1][n2] * z_pow[n2]; break;
      }
    }
    z_factor *= fn;
    if (n1 == 0) {
      s[s_index++] = z_factor;
    } else {
      s[s_index++] = z_factor * real_part;
      s[s_index++] = z_factor * imag_part;
      complex_product(x12, y12, real_part, imag_part);
    }
  }
}

static __device__ __forceinline__ void accumulate_ec(
  const int N,
  const int L_max,
  const int n,
  const int n_max_angular_plus_1,
  const int basis_size_angular_plus_1,
  const float d12,
  const float* r12,
  float fn,
  float fnp,
  const float* sum_fxyz,
  const float* sum_s2xyz,
  const float* sum_s2xyz123,
  const float* Fp,
  float* e_c,
  float* qp_c,
  float* qp_c123)
{
  const float d12inv = 1.0f / d12;
  const float r12unit[3] = {r12[0]*d12inv, r12[1]*d12inv, r12[2]*d12inv};
  if (L_max >= 1) {
    float s1[3];
    float sc1[3];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_sc_one<1>(r12unit[0], r12unit[1], r12unit[2], fn, sc1);
    calculate_ec_one<1>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sum_s2xyz123, sc1, s1, e_c, f, f123);
    accumulate_f12_one<1>(d12inv, fn, fnp, s1, r12unit, r12, f, f123);
    qp_c[0] = f[0];
    qp_c[1] = f[1];
    qp_c[2] = f[2];
    qp_c123[0] = f123[0];
    qp_c123[1] = f123[1];
    qp_c123[2] = f123[2];
    qp_c123[3] = f123[3];
    qp_c123[4] = f123[4];
    qp_c123[5] = f123[5];
  }

  if (L_max >= 2) {
    float s2[5];
    float sc2[5];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_sc_one<2>(r12unit[0], r12unit[1], r12unit[2], fn, sc2);
    calculate_ec_one<2>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sum_s2xyz123, sc2, s2, e_c, f, f123);
    accumulate_f12_one<2>(d12inv, fn, fnp, s2, r12unit, r12, f, f123);
    qp_c[0] += f[0];
    qp_c[1] += f[1];
    qp_c[2] += f[2];
    qp_c123[0] += f123[0];
    qp_c123[1] += f123[1];
    qp_c123[2] += f123[2];
    qp_c123[3] += f123[3];
    qp_c123[4] += f123[4];
    qp_c123[5] += f123[5];
  }

  if (L_max >= 3) {
    float s3[7];
    float sc3[7];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_sc_one<3>(r12unit[0], r12unit[1], r12unit[2], fn, sc3);
    calculate_ec_one<3>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sum_s2xyz123, sc3, s3, e_c, f, f123);
    accumulate_f12_one<3>(d12inv, fn, fnp, s3, r12unit, r12, f, f123);
    qp_c[0] += f[0];
    qp_c[1] += f[1];
    qp_c[2] += f[2];
    qp_c123[0] += f123[0];
    qp_c123[1] += f123[1];
    qp_c123[2] += f123[2];
    qp_c123[3] += f123[3];
    qp_c123[4] += f123[4];
    qp_c123[5] += f123[5];
  }

  if (L_max >= 4) {
    float s4[9];
    float sc4[9];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_sc_one<4>(r12unit[0], r12unit[1], r12unit[2], fn, sc4);
    calculate_ec_one<4>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sum_s2xyz123, sc4, s4, e_c, f, f123);
    accumulate_f12_one<4>(d12inv, fn, fnp, s4, r12unit, r12, f, f123);
    qp_c[0] += f[0];
    qp_c[1] += f[1];
    qp_c[2] += f[2];
    qp_c123[0] += f123[0];
    qp_c123[1] += f123[1];
    qp_c123[2] += f123[2];
    qp_c123[3] += f123[3];
    qp_c123[4] += f123[4];
    qp_c123[5] += f123[5];
  }

  if (L_max >= 5) {
    float s5[11];
    float sc5[11];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_sc_one<5>(r12unit[0], r12unit[1], r12unit[2], fn, sc5);
    calculate_ec_one<5>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sum_s2xyz123, sc5, s5, e_c, f, f123);
    accumulate_f12_one<5>(d12inv, fn, fnp, s5, r12unit, r12, f, f123);
    qp_c[0] += f[0];
    qp_c[1] += f[1];
    qp_c[2] += f[2];
    qp_c123[0] += f123[0];
    qp_c123[1] += f123[1];
    qp_c123[2] += f123[2];
    qp_c123[3] += f123[3];
    qp_c123[4] += f123[4];
    qp_c123[5] += f123[5];
  }

  if (L_max >= 6) {
    float s6[13];
    float sc6[13];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_sc_one<6>(r12unit[0], r12unit[1], r12unit[2], fn, sc6);
    calculate_ec_one<6>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sum_s2xyz123, sc6, s6, e_c, f, f123);
    accumulate_f12_one<6>(d12inv, fn, fnp, s6, r12unit, r12, f, f123);
    qp_c[0] += f[0];
    qp_c[1] += f[1];
    qp_c[2] += f[2];
    qp_c123[0] += f123[0];
    qp_c123[1] += f123[1];
    qp_c123[2] += f123[2];
    qp_c123[3] += f123[3];
    qp_c123[4] += f123[4];
    qp_c123[5] += f123[5];
  }

  if (L_max >= 7) {
    float s7[15];
    float sc7[15];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_sc_one<7>(r12unit[0], r12unit[1], r12unit[2], fn, sc7);
    calculate_ec_one<7>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sum_s2xyz123, sc7, s7, e_c, f, f123);
    accumulate_f12_one<7>(d12inv, fn, fnp, s7, r12unit, r12, f, f123);
    qp_c[0] += f[0];
    qp_c[1] += f[1];
    qp_c[2] += f[2];
    qp_c123[0] += f123[0];
    qp_c123[1] += f123[1];
    qp_c123[2] += f123[2];
    qp_c123[3] += f123[3];
    qp_c123[4] += f123[4];
    qp_c123[5] += f123[5];
  }

  if (L_max >= 8) {
    float s8[17];
    float sc8[17];
    float f[3] = {0.0f};
    float f123[6] = {0.0f};
    calculate_sc_one<8>(r12unit[0], r12unit[1], r12unit[2], fn, sc8);
    calculate_ec_one<8>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sum_s2xyz123, sc8, s8, e_c, f, f123);
    accumulate_f12_one<8>(d12inv, fn, fnp, s8, r12unit, r12, f, f123);
    qp_c[0] += f[0];
    qp_c[1] += f[1];
    qp_c[2] += f[2];
    qp_c123[0] += f123[0];
    qp_c123[1] += f123[1];
    qp_c123[2] += f123[2];
    qp_c123[3] += f123[3];
    qp_c123[4] += f123[4];
    qp_c123[5] += f123[5];
  }
}

static __device__ __forceinline__ void accumulate_fc(
  const int N,
  const int L_max,
  const int n,
  const int n_max_angular_plus_1,
  const int basis_size_angular_plus_1,
  const float d12,
  const float* r12,
  float fn,
  float fnp,
  const float* sum_fxyz,
  const float* sum_s2xyz,
  const float* Fp,
  float* qp_c1,
  float* qp_c2)
{
  const float d12inv = 1.0f / d12;
  const float r12unit[3] = {r12[0]*d12inv, r12[1]*d12inv, r12[2]*d12inv};
  if (L_max >= 1) {
    float s1[3];
    float sc1[3];
    float f[3] = {0.0f};
    qp_c2[0] = 0.0f;
    qp_c2[1] = 0.0f;
    qp_c2[2] = 0.0f;
    calculate_sc_one<1>(r12unit[0], r12unit[1], r12unit[2], fn, sc1);
    calculate_fc_one<1>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc1, s1, qp_c2);
    accumulate_f12_one<1>(d12inv, fn, fnp, s1, r12unit, f);
    qp_c1[0] = f[0];
    qp_c1[1] = f[1];
    qp_c1[2] = f[2];
  }

  if (L_max >= 2) {
    float s2[5];
    float sc2[5];
    float f[3] = {0.0f};
    calculate_sc_one<2>(r12unit[0], r12unit[1], r12unit[2], fn, sc2);
    calculate_fc_one<2>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc2, s2, qp_c2);
    accumulate_f12_one<2>(d12inv, fn, fnp, s2, r12unit, f);
    qp_c1[0] += f[0];
    qp_c1[1] += f[1];
    qp_c1[2] += f[2];
  }

  if (L_max >= 3) {
    float s3[7];
    float sc3[7];
    float f[3] = {0.0f};
    calculate_sc_one<3>(r12unit[0], r12unit[1], r12unit[2], fn, sc3);
    calculate_fc_one<3>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc3, s3, qp_c2);
    accumulate_f12_one<3>(d12inv, fn, fnp, s3, r12unit, f);
    qp_c1[0] += f[0];
    qp_c1[1] += f[1];
    qp_c1[2] += f[2];
  }

  if (L_max >= 4) {
    float s4[9];
    float sc4[9];
    float f[3] = {0.0f};
    calculate_sc_one<4>(r12unit[0], r12unit[1], r12unit[2], fn, sc4);
    calculate_fc_one<4>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc4, s4, qp_c2);
    accumulate_f12_one<4>(d12inv, fn, fnp, s4, r12unit, f);
    qp_c1[0] += f[0];
    qp_c1[1] += f[1];
    qp_c1[2] += f[2];
  }

  if (L_max >= 5) {
    float s5[11];
    float sc5[11];
    float f[3] = {0.0f};
    calculate_sc_one<5>(r12unit[0], r12unit[1], r12unit[2], fn, sc5);
    calculate_fc_one<5>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc5, s5, qp_c2);
    accumulate_f12_one<5>(d12inv, fn, fnp, s5, r12unit, f);
    qp_c1[0] += f[0];
    qp_c1[1] += f[1];
    qp_c1[2] += f[2];
  }

  if (L_max >= 6) {
    float s6[13];
    float sc6[13];
    float f[3] = {0.0f};
    calculate_sc_one<6>(r12unit[0], r12unit[1], r12unit[2], fn, sc6);
    calculate_fc_one<6>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc6, s6, qp_c2);
    accumulate_f12_one<6>(d12inv, fn, fnp, s6, r12unit, f);
    qp_c1[0] += f[0];
    qp_c1[1] += f[1];
    qp_c1[2] += f[2];
  }

  if (L_max >= 7) {
    float s7[15];
    float sc7[15];
    float f[3] = {0.0f};
    calculate_sc_one<7>(r12unit[0], r12unit[1], r12unit[2], fn, sc7);
    calculate_fc_one<7>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc7, s7, qp_c2);
    accumulate_f12_one<7>(d12inv, fn, fnp, s7, r12unit, f);
    qp_c1[0] += f[0];
    qp_c1[1] += f[1];
    qp_c1[2] += f[2];
  }

  if (L_max >= 8) {
    float s8[17];
    float sc8[17];
    float f[3] = {0.0f};
    calculate_sc_one<8>(r12unit[0], r12unit[1], r12unit[2], fn, sc8);
    calculate_fc_one<8>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc8, s8, qp_c2);
    accumulate_f12_one<8>(d12inv, fn, fnp, s8, r12unit, f);
    qp_c1[0] += f[0];
    qp_c1[1] += f[1];
    qp_c1[2] += f[2];
  }
}

static __device__ __forceinline__ void accumulate_qc(
  const int N,
  const int L,
  const int n,
  const int n_max_angular_plus_1,
  const int basis_size_angular_plus_1,
  const float d12,
  const float* r12,
  const float fn,
  const float* sum_fxyz,
  float* q_c)
{
  const float d12inv = 1.0f / d12;
  const float r12unit[3] = {r12[0]*d12inv, r12[1]*d12inv, r12[2]*d12inv};
  switch(L) {
    case 1: {
      float sc1[3];
      calculate_sc_one<1>(r12unit[0], r12unit[1], r12unit[2], fn, sc1);
      calculate_qc_one<1>(N, n, n_max_angular_plus_1, sum_fxyz, sc1, q_c);
      break;
    }
    case 2: {
      float sc2[5];
      calculate_sc_one<2>(r12unit[0], r12unit[1], r12unit[2], fn, sc2);
      calculate_qc_one<2>(N, n, n_max_angular_plus_1, sum_fxyz, sc2, q_c);
      break;
    }
    case 3: {
      float sc3[7];
      calculate_sc_one<3>(r12unit[0], r12unit[1], r12unit[2], fn, sc3);
      calculate_qc_one<3>(N, n, n_max_angular_plus_1, sum_fxyz, sc3, q_c);
      break;
    }
    case 4: {
      float sc4[9];
      calculate_sc_one<4>(r12unit[0], r12unit[1], r12unit[2], fn, sc4);
      calculate_qc_one<4>(N, n, n_max_angular_plus_1, sum_fxyz, sc4, q_c);
      break;
    }
    case 5: {
      float sc5[11];
      calculate_sc_one<5>(r12unit[0], r12unit[1], r12unit[2], fn, sc5);
      calculate_qc_one<5>(N, n, n_max_angular_plus_1, sum_fxyz, sc5, q_c);
      break;
    }
    case 6: {
      float sc6[13];
      calculate_sc_one<6>(r12unit[0], r12unit[1], r12unit[2], fn, sc6);
      calculate_qc_one<6>(N, n, n_max_angular_plus_1, sum_fxyz, sc6, q_c);
      break;
    }
    case 7: {
      float sc7[15];
      calculate_sc_one<7>(r12unit[0], r12unit[1], r12unit[2], fn, sc7);
      calculate_qc_one<7>(N, n, n_max_angular_plus_1, sum_fxyz, sc7, q_c);
      break;
    }
    case 8: {
      float sc8[17];
      calculate_sc_one<8>(r12unit[0], r12unit[1], r12unit[2], fn, sc8);
      calculate_qc_one<8>(N, n, n_max_angular_plus_1, sum_fxyz, sc8, q_c);
      break;
    }
  }
}

template <int L>
static __device__ __forceinline__ void
accumulate_s_one(
  const float x12,
  const float y12,
  const float z12,
  const float fn,
  float* s)
{
  int s_index = L * L - 1;
  float z_pow[L + 1] = {1.0f};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = z12 * z_pow[n - 1];
  }
  float real_part = x12;
  float imag_part = y12;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    float z_factor = 0.0f;
    for (int n2 = n2_start; n2 <= L - n1; n2 += 2) {
      if (L == 1) {
        z_factor += Z_COEFFICIENT_1[n1][n2] * z_pow[n2];
      }
      if (L == 2) {
        z_factor += Z_COEFFICIENT_2[n1][n2] * z_pow[n2];
      }
      if (L == 3) {
        z_factor += Z_COEFFICIENT_3[n1][n2] * z_pow[n2];
      }
      if (L == 4) {
        z_factor += Z_COEFFICIENT_4[n1][n2] * z_pow[n2];
      }
      if (L == 5) {
        z_factor += Z_COEFFICIENT_5[n1][n2] * z_pow[n2];
      }
      if (L == 6) {
        z_factor += Z_COEFFICIENT_6[n1][n2] * z_pow[n2];
      }
      if (L == 7) {
        z_factor += Z_COEFFICIENT_7[n1][n2] * z_pow[n2];
      }
      if (L == 8) {
        z_factor += Z_COEFFICIENT_8[n1][n2] * z_pow[n2];
      }
    }
    z_factor *= fn;
    if (n1 == 0) {
      s[s_index++] += z_factor;
    } else {
      s[s_index++] += z_factor * real_part;
      s[s_index++] += z_factor * imag_part;
      complex_product(x12, y12, real_part, imag_part);
    }
  }
}

static __device__ __forceinline__ void
accumulate_s(const int L_max, const float d12, float x12, float y12, float z12, const float fn, float* s)
{
  float d12inv = 1.0f / d12;
  x12 *= d12inv;
  y12 *= d12inv;
  z12 *= d12inv;
  if (L_max >= 1) {
    accumulate_s_one<1>(x12, y12, z12, fn, s);
  }
  if (L_max >= 2) {
    accumulate_s_one<2>(x12, y12, z12, fn, s);
  }
  if (L_max >= 3) {
    accumulate_s_one<3>(x12, y12, z12, fn, s);
  }
  if (L_max >= 4) {
    accumulate_s_one<4>(x12, y12, z12, fn, s);
  }
  if (L_max >= 5) {
    accumulate_s_one<5>(x12, y12, z12, fn, s);
  }
  if (L_max >= 6) {
    accumulate_s_one<6>(x12, y12, z12, fn, s);
  }
  if (L_max >= 7) {
    accumulate_s_one<7>(x12, y12, z12, fn, s);
  }
  if (L_max >= 8) {
    accumulate_s_one<8>(x12, y12, z12, fn, s);
  }
}

template<int L>
static __device__ __forceinline__ float find_q_one(const float* s)
{
  const int start_index = L * L-1;
  const int num_terms = 2 * L + 1;
  float q = 0.0f;
  for (int k = 1; k < num_terms; ++k) {
    q += C3B[start_index + k] * s[start_index + k] * s[start_index + k];
  }
  q *= 2.0f;
  q += C3B[start_index] * s[start_index] * s[start_index];
  return q;
}

static __device__ __forceinline__ void
find_q(
  const int L_max, 
  const int n_max_angular_plus_1, 
  const int n, 
  const float* s, 
  float* q)
{
  if (L_max >= 1) {
    q[0 * n_max_angular_plus_1 + n] = find_q_one<1>(s);
  }
  if (L_max >= 2) {
    q[1 * n_max_angular_plus_1 + n] = find_q_one<2>(s);
  }
  if (L_max >= 3) {
    q[2 * n_max_angular_plus_1 + n] = find_q_one<3>(s);
  }
  if (L_max >= 4) {
    q[3 * n_max_angular_plus_1 + n] = find_q_one<4>(s);
  }
  if (L_max >= 5) {
    q[4 * n_max_angular_plus_1 + n] = find_q_one<5>(s);
  }
  if (L_max >= 6) {
    q[5 * n_max_angular_plus_1 + n] = find_q_one<6>(s);
  }
  if (L_max >= 7) {
    q[6 * n_max_angular_plus_1 + n] = find_q_one<7>(s);
  }
  if (L_max >= 8) {
    q[7 * n_max_angular_plus_1 + n] = find_q_one<8>(s);
  }
}
