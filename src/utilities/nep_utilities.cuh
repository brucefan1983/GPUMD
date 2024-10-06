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

#pragma once

const int NUM_OF_ABC = 24; // 3 + 5 + 7 + 9 for L_max = 4
__constant__ float C3B[NUM_OF_ABC] = {
  0.238732414637843f, 0.119366207318922f, 0.119366207318922f, 0.099471839432435f,
  0.596831036594608f, 0.596831036594608f, 0.149207759148652f, 0.149207759148652f,
  0.139260575205408f, 0.104445431404056f, 0.104445431404056f, 1.044454314040563f,
  1.044454314040563f, 0.174075719006761f, 0.174075719006761f, 0.011190581936149f,
  0.223811638722978f, 0.223811638722978f, 0.111905819361489f, 0.111905819361489f,
  1.566681471060845f, 1.566681471060845f, 0.195835183882606f, 0.195835183882606f};
__constant__ float C4B[5] = {
  -0.007499480826664f,
  -0.134990654879954f,
  0.067495327439977f,
  0.404971964639861f,
  -0.809943929279723f};
__constant__ float C5B[3] = {0.026596810706114f, 0.053193621412227f, 0.026596810706114f};

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

static __device__ __forceinline__ void
complex_product(const float a, const float b, float& real_part, float& imag_part)
{
  const float real_temp = real_part;
  real_part = a * real_temp - b * imag_part;
  imag_part = a * imag_part + b * real_temp;
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

static __device__ void apply_ann_one_layer_nep5(
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
  energy -= w1[N_neu] + b1[0]; // typewise bias + common bias
}

static __device__ __forceinline__ void find_fc(float rc, float rcinv, float d12, float& fc)
{
  if (d12 < rc) {
    float x = d12 * rcinv;
    fc = 0.5f * cos(3.1415927f * x) + 0.5f;
  } else {
    fc = 0.0f;
  }
}

static __device__ __host__ __forceinline__ void
find_fc_and_fcp(float rc, float rcinv, float d12, float& fc, float& fcp)
{
  if (d12 < rc) {
    float x = d12 * rcinv;
    fc = 0.5f * cos(3.1415927f * x) + 0.5f;
    fcp = -1.5707963f * sin(3.1415927f * x);
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
    float pi_factor = 3.1415927f / (r2 - r1);
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
    fn = (x + 1.0f) * 0.5f;
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
  fn[0] = 1.0f;
  fn[1] = x;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0f * x * fn[m - 1] - fn[m - 2];
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0f) * 0.5f * fc12;
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

static __device__ __forceinline__ void get_f12_1(
  const float d12inv,
  const float fn,
  const float fnp,
  const float* s,
  const float* r12,
  float* f12)
{
  float tmp = s[1] * r12[0];
  tmp += s[2] * r12[1];
  tmp += s[0] * r12[2];
  tmp *= fnp * d12inv;
  for (int d = 0; d < 3; ++d) {
    f12[d] += tmp * r12[d];
  }
  f12[0] += fn * s[1];
  f12[1] += fn * s[2];
  f12[2] += fn * s[0];
}

static __device__ __forceinline__ void get_f12_2(
  const float d12,
  const float d12inv,
  const float fn,
  const float fnp,
  const float* s,
  const float* r12,
  float* f12)
{
  float tmp = s[1] * r12[0] * r12[2];                // Re[Y21]
  tmp += s[2] * r12[1] * r12[2];                     // Im[Y21]
  tmp += s[3] * (r12[0] * r12[0] - r12[1] * r12[1]); // Re[Y22]
  tmp += s[4] * 2.0f * r12[0] * r12[1];              // Im[Y22]
  tmp += s[0] * (3.0f * r12[2] * r12[2] - d12 * d12); // Y20
  tmp *= fnp * d12inv;
  for (int d = 0; d < 3; ++d) {
    f12[d] += tmp * r12[d];
  }
  f12[0] += fn * (-2.0f * s[0] * r12[0] + s[1] * r12[2] + 2.0f * s[3] * r12[0] + 2.0f * s[4] * r12[1]);
  f12[1] += fn * (-2.0f * s[0] * r12[1] + s[2] * r12[2] - 2.0f * s[3] * r12[1] + 2.0f * s[4] * r12[0]);
  f12[2] += fn * (4.0f * s[0] * r12[2] + s[1] * r12[0] + s[2] * r12[1]);
}

static __device__ __forceinline__ void get_f12_4body(
  const float d12,
  const float d12inv,
  const float fn,
  const float fnp,
  const float Fp,
  const float* s,
  const float* r12,
  float* f12)
{
  float fn_factor = Fp * fn;
  float fnp_factor = Fp * fnp * d12inv;
  float y20 = (3.0f * r12[2] * r12[2] - d12 * d12);

  // derivative wrt s[0]
  float tmp0 = C4B[0] * 3.0f * s[0] * s[0] + C4B[1] * (s[1] * s[1] + s[2] * s[2]) +
               C4B[2] * (s[3] * s[3] + s[4] * s[4]);
  float tmp1 = tmp0 * y20 * fnp_factor;
  float tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * 2.0f * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0f * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * 4.0f * r12[2];

  // derivative wrt s[1]
  tmp0 = C4B[1] * s[0] * s[1] * 2.0f - C4B[3] * s[3] * s[1] * 2.0f + C4B[4] * s[2] * s[4];
  tmp1 = tmp0 * r12[0] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * r12[2];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[0];

  // derivative wrt s[2]
  tmp0 = C4B[1] * s[0] * s[2] * 2.0f + C4B[3] * s[3] * s[2] * 2.0f + C4B[4] * s[1] * s[4];
  tmp1 = tmp0 * r12[1] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2 * r12[2];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[1];

  // derivative wrt s[3]
  tmp0 = C4B[2] * s[0] * s[3] * 2.0f + C4B[3] * (s[2] * s[2] - s[1] * s[1]);
  tmp1 = tmp0 * (r12[0] * r12[0] - r12[1] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0f * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0f * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s[4]
  tmp0 = C4B[2] * s[0] * s[4] * 2.0f + C4B[4] * s[1] * s[2];
  tmp1 = tmp0 * (2.0f * r12[0] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0f * r12[1];
  f12[1] += tmp1 * r12[1] + tmp2 * 2.0f * r12[0];
  f12[2] += tmp1 * r12[2];
}

static __device__ __forceinline__ void get_f12_5body(
  const float d12,
  const float d12inv,
  const float fn,
  const float fnp,
  const float Fp,
  const float* s,
  const float* r12,
  float* f12)
{
  float fn_factor = Fp * fn;
  float fnp_factor = Fp * fnp * d12inv;
  float s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];

  // derivative wrt s[0]
  float tmp0 = C5B[0] * 4.0f * s[0] * s[0] * s[0] + C5B[1] * s1_sq_plus_s2_sq * 2.0f * s[0];
  float tmp1 = tmp0 * r12[2] * fnp_factor;
  float tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2;

  // derivative wrt s[1]
  tmp0 = C5B[1] * s[0] * s[0] * s[1] * 2.0f + C5B[2] * s1_sq_plus_s2_sq * s[1] * 4.0f;
  tmp1 = tmp0 * r12[0] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2;
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s[2]
  tmp0 = C5B[1] * s[0] * s[0] * s[2] * 2.0f + C5B[2] * s1_sq_plus_s2_sq * s[2] * 4.0f;
  tmp1 = tmp0 * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2;
  f12[2] += tmp1 * r12[2];
}

static __device__ __forceinline__ void get_f12_3(
  const float d12,
  const float d12inv,
  const float fn,
  const float fnp,
  const float* s,
  const float* r12,
  float* f12)
{
  float d12sq = d12 * d12;
  float x2 = r12[0] * r12[0];
  float y2 = r12[1] * r12[1];
  float z2 = r12[2] * r12[2];
  float xy = r12[0] * r12[1];
  float xz = r12[0] * r12[2];
  float yz = r12[1] * r12[2];

  float tmp = s[1] * (5.0f * z2 - d12sq) * r12[0];
  tmp += s[2] * (5.0f * z2 - d12sq) * r12[1];
  tmp += s[3] * (x2 - y2) * r12[2];
  tmp += s[4] * 2.0f * xy * r12[2];
  tmp += s[5] * r12[0] * (x2 - 3.0f * y2);
  tmp += s[6] * r12[1] * (3.0f * x2 - y2);
  tmp += s[0] * (5.0f * z2 - 3.0f * d12sq) * r12[2];
  tmp *= fnp * d12inv;
  for (int d = 0; d < 3; ++d) {
    f12[d] += tmp * r12[d];
  }

  // x
  tmp = s[1] * (4.0f * z2 - 3.0f * x2 - y2);
  tmp += s[2] * (-2.0f * xy);
  tmp += s[3] * 2.0f * xz;
  tmp += s[4] * (2.0f * yz);
  tmp += s[5] * (3.0f * (x2 - y2));
  tmp += s[6] * (6.0f * xy);
  tmp += s[0] * (-6.0f * xz);
  f12[0] += tmp * fn;
  // y
  tmp = s[1] * (-2.0f * xy);
  tmp += s[2] * (4.0f * z2 - 3.0f * y2 - x2);
  tmp += s[3] * (-2.0f * yz);
  tmp += s[4] * (2.0f * xz);
  tmp += s[5] * (-6.0f * xy);
  tmp += s[6] * (3.0f * (x2 - y2));
  tmp += s[0] * (-6.0f * yz);
  f12[1] += tmp * fn;
  // z
  tmp = s[1] * (8.0f * xz);
  tmp += s[2] * (8.0f * yz);
  tmp += s[3] * (x2 - y2);
  tmp += s[4] * (2.0f * xy);
  tmp += s[0] * (9.0f * z2 - 3.0f * d12sq);
  f12[2] += tmp * fn;
}

static __device__ __forceinline__ void get_f12_4(
  const float x,
  const float y,
  const float z,
  const float r,
  const float rinv,
  const float fn,
  const float fnp,
  const float* s,
  float* f12)
{
  const float r2 = r * r;
  const float x2 = x * x;
  const float y2 = y * y;
  const float z2 = z * z;
  const float xy = x * y;
  const float xz = x * z;
  const float yz = y * z;
  const float xyz = x * yz;
  const float x2my2 = x2 - y2;

  float tmp = s[1] * (7.0f * z2 - 3.0f * r2) * xz; // Y41_real
  tmp += s[2] * (7.0f * z2 - 3.0f * r2) * yz;      // Y41_imag
  tmp += s[3] * (7.0f * z2 - r2) * x2my2;          // Y42_real
  tmp += s[4] * (7.0f * z2 - r2) * 2.0f * xy;      // Y42_imag
  tmp += s[5] * (x2 - 3.0f * y2) * xz;             // Y43_real
  tmp += s[6] * (3.0f * x2 - y2) * yz;             // Y43_imag
  tmp += s[7] * (x2my2 * x2my2 - 4.0f * x2 * y2);  // Y44_real
  tmp += s[8] * (4.0f * xy * x2my2);               // Y44_imag
  tmp += s[0] * ((35.0f * z2 - 30.0f * r2) * z2 + 3.0f * r2 * r2); // Y40
  tmp *= fnp * rinv;
  f12[0] += tmp * x;
  f12[1] += tmp * y;
  f12[2] += tmp * z;

  // x
  tmp = s[1] * z * (7.0f * z2 - 3.0f * r2 - 6.0f * x2);  // Y41_real
  tmp += s[2] * (-6.0f * xyz);                           // Y41_imag
  tmp += s[3] * 4.0f * x * (3.0f * z2 - x2);             // Y42_real
  tmp += s[4] * 2.0f * y * (7.0f * z2 - r2 - 2.0f * x2); // Y42_imag
  tmp += s[5] * 3.0f * z * x2my2;                        // Y43_real
  tmp += s[6] * 6.0f * xyz;                              // Y43_imag
  tmp += s[7] * 4.0f * x * (x2 - 3.0f * y2);             // Y44_real
  tmp += s[8] * 4.0f * y * (3.0f * x2 - y2);             // Y44_imag
  tmp += s[0] * 12.0f * x * (r2 - 5.0f * z2); // Y40
  f12[0] += tmp * fn;
  // y
  tmp = s[1] * (-6.0f * xyz);                            // Y41_real
  tmp += s[2] * z * (7.0f * z2 - 3.0f * r2 - 6.0f * y2); // Y41_imag
  tmp += s[3] * 4.0f * y * (y2 - 3.0f * z2);             // Y42_real
  tmp += s[4] * 2.0f * x * (7.0f * z2 - r2 - 2.0f * y2); // Y42_imag
  tmp += s[5] * (-6.0f * xyz);                           // Y43_real
  tmp += s[6] * 3.0f * z * x2my2;                        // Y43_imag
  tmp += s[7] * 4.0f * y * (y2 - 3.0f * x2);             // Y44_real
  tmp += s[8] * 4.0f * x * (x2 - 3.0f * y2);             // Y44_imag
  tmp += s[0] * 12.0f * y * (r2 - 5.0f * z2); // Y40
  f12[1] += tmp * fn;
  // z
  tmp = s[1] * 3.0f * x * (5.0f * z2 - r2);  // Y41_real
  tmp += s[2] * 3.0f * y * (5.0f * z2 - r2); // Y41_imag
  tmp += s[3] * 12.0f * z * x2my2;           // Y42_real
  tmp += s[4] * 24.0f * xyz;                 // Y42_imag
  tmp += s[5] * x * (x2 - 3.0f * y2);        // Y43_real
  tmp += s[6] * y * (3.0f * x2 - y2);        // Y43_imag
  tmp += s[0] * 16.0f * z * (5.0f * z2 - 3.0f * r2); // Y40
  f12[2] += tmp * fn;
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
      if (L == 1) {
        z_factor += Z_COEFFICIENT_1[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_1[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 2) {
        z_factor += Z_COEFFICIENT_2[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_2[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 3) {
        z_factor += Z_COEFFICIENT_3[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_3[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 4) {
        z_factor += Z_COEFFICIENT_4[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_4[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
    }
    if (n1 == 0) {
      for (int d = 0; d < 3; ++d) {
        f12[d] += s[0] * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
      }
    } else {
      float real_part_n1 = n1 * real_part;
      float imag_part_n1 = n1 * imag_part;
      for (int d = 0; d < 3; ++d) {
        float real_part_dx = dx[d];
        float imag_part_dy = dy[d];
        complex_product(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        f12[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor * fn;
      }
      complex_product(r12[0], r12[1], real_part, imag_part);
      for (int d = 0; d < 3; ++d) {
        const float tmp = (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
        f12[d] += (s[2 * n1 - 1] * real_part + s[2 * n1 - 0] * imag_part) * tmp;
      }
    }
  }
}

static __device__ __forceinline__ void accumulate_f12(
  const int L_max,
  const int num_L,
  const int n,
  const int n_max_angular_plus_1,
  const float d12,
  const float* r12,
  float fn,
  float fnp,
  const float* Fp,
  const float* sum_fxyz,
  float* f12)
{
  const float fn_original = fn;
  const float fnp_original = fnp;
  const float d12inv = 1.0f / d12;
  const float r12unit[3] = {r12[0]*d12inv, r12[1]*d12inv, r12[2]*d12inv};
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0], sum_fxyz[n * NUM_OF_ABC + 1], sum_fxyz[n * NUM_OF_ABC + 2]};
  if (num_L >= L_max + 2) {
    get_f12_5body(d12, d12inv, fn, fnp, Fp[(L_max + 1) * n_max_angular_plus_1 + n], s1, r12, f12);
  }
  s1[0] *= C3B[0] * 2.0f * Fp[n]; // (2 - delta_0m)
  s1[1] *= C3B[1] * 4.0f * Fp[n]; // (2 - delta_0m)
  s1[2] *= C3B[2] * 4.0f * Fp[n]; // (2 - delta_0m)
  if (L_max >= 1) {
    //get_f12_1(d12inv, fn, fnp, s1, r12, f12);
    accumulate_f12_one<1>(d12inv, fn_original, fnp_original, s1, r12unit, f12);
  }
  
  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3],
    sum_fxyz[n * NUM_OF_ABC + 4],
    sum_fxyz[n * NUM_OF_ABC + 5],
    sum_fxyz[n * NUM_OF_ABC + 6],
    sum_fxyz[n * NUM_OF_ABC + 7]};
  if (num_L >= L_max + 1) {
    get_f12_4body(d12, d12inv, fn, fnp, Fp[L_max * n_max_angular_plus_1 + n], s2, r12, f12);
  }
  s2[0] *= C3B[3] * 2.0f * Fp[n_max_angular_plus_1 + n]; // (2 - delta_0m)
  s2[1] *= C3B[4] * 4.0f * Fp[n_max_angular_plus_1 + n]; // (2 - delta_0m)
  s2[2] *= C3B[5] * 4.0f * Fp[n_max_angular_plus_1 + n]; // (2 - delta_0m)
  s2[3] *= C3B[6] * 4.0f * Fp[n_max_angular_plus_1 + n]; // (2 - delta_0m)
  s2[4] *= C3B[7] * 4.0f * Fp[n_max_angular_plus_1 + n]; // (2 - delta_0m)
  if (L_max >= 2) {
    //get_f12_2(d12, d12inv, fn, fnp, s2, r12, f12);
    accumulate_f12_one<2>(d12inv, fn_original, fnp_original, s2, r12unit, f12);
  }
  
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8] * 2.0f * Fp[2 * n_max_angular_plus_1 + n],   // (2 - delta_0m)
    sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9] * 4.0f * Fp[2 * n_max_angular_plus_1 + n],   // (2 - delta_0m)
    sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10] * 4.0f * Fp[2 * n_max_angular_plus_1 + n], // (2 - delta_0m)
    sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11] * 4.0f * Fp[2 * n_max_angular_plus_1 + n], // (2 - delta_0m)
    sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12] * 4.0f * Fp[2 * n_max_angular_plus_1 + n], // (2 - delta_0m)
    sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13] * 4.0f * Fp[2 * n_max_angular_plus_1 + n], // (2 - delta_0m)
    sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14] * 4.0f * Fp[2 * n_max_angular_plus_1 + n]  // (2 - delta_0m)
  };
  if (L_max >= 3) {  
    //get_f12_3(d12, d12inv, fn, fnp, s3, r12, f12);
    accumulate_f12_one<3>(d12inv, fn_original, fnp_original, s3, r12unit, f12);
  }

  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  float s4[9] = {
    sum_fxyz[n * NUM_OF_ABC + 15] * C3B[15] * 2.0f * Fp[3 * n_max_angular_plus_1 + n], // (2 - delta_0m)
    sum_fxyz[n * NUM_OF_ABC + 16] * C3B[16] * 4.0f * Fp[3 * n_max_angular_plus_1 + n], // (2 - delta_0m)
    sum_fxyz[n * NUM_OF_ABC + 17] * C3B[17] * 4.0f * Fp[3 * n_max_angular_plus_1 + n], // (2 - delta_0m)
    sum_fxyz[n * NUM_OF_ABC + 18] * C3B[18] * 4.0f * Fp[3 * n_max_angular_plus_1 + n], // (2 - delta_0m)
    sum_fxyz[n * NUM_OF_ABC + 19] * C3B[19] * 4.0f * Fp[3 * n_max_angular_plus_1 + n], // (2 - delta_0m)
    sum_fxyz[n * NUM_OF_ABC + 20] * C3B[20] * 4.0f * Fp[3 * n_max_angular_plus_1 + n], // (2 - delta_0m)
    sum_fxyz[n * NUM_OF_ABC + 21] * C3B[21] * 4.0f * Fp[3 * n_max_angular_plus_1 + n], // (2 - delta_0m)
    sum_fxyz[n * NUM_OF_ABC + 22] * C3B[22] * 4.0f * Fp[3 * n_max_angular_plus_1 + n], // (2 - delta_0m)
    sum_fxyz[n * NUM_OF_ABC + 23] * C3B[23] * 4.0f * Fp[3 * n_max_angular_plus_1 + n]  // (2 - delta_0m)
  };
  if (L_max >= 4) {
    //get_f12_4(r12[0], r12[1], r12[2], d12, d12inv, fn, fnp, s4, f12);
    accumulate_f12_one<4>(d12inv, fn_original, fnp_original, s4, r12unit, f12);
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
}

static __device__ __forceinline__ void
find_q(
  const int L_max, 
  const int num_L, 
  const int n_max_angular_plus_1, 
  const int n, 
  const float* s, 
  float* q)
{
  if (L_max >= 1) {
    q[0 * n_max_angular_plus_1 + n] = C3B[0] * s[0] * s[0] + 2.0f * (C3B[1] * s[1] * s[1] + C3B[2] * s[2] * s[2]);
  }
  if (L_max >= 2) {
    q[1 * n_max_angular_plus_1 + n] =
      C3B[3] * s[3] * s[3] + 2.0f * (C3B[4] * s[4] * s[4] + C3B[5] * s[5] * s[5] +
                                   C3B[6] * s[6] * s[6] + C3B[7] * s[7] * s[7]);
  }
  if (L_max >= 3) {
    q[2 * n_max_angular_plus_1 + n] =
      C3B[8] * s[8] * s[8] +
      2.0f * (C3B[9] * s[9] * s[9] + C3B[10] * s[10] * s[10] + C3B[11] * s[11] * s[11] +
              C3B[12] * s[12] * s[12] + C3B[13] * s[13] * s[13] + C3B[14] * s[14] * s[14]);
  }
  if (L_max >= 4) {
    q[3 * n_max_angular_plus_1 + n] =
      C3B[15] * s[15] * s[15] +
      2.0f * (C3B[16] * s[16] * s[16] + C3B[17] * s[17] * s[17] + C3B[18] * s[18] * s[18] +
              C3B[19] * s[19] * s[19] + C3B[20] * s[20] * s[20] + C3B[21] * s[21] * s[21] +
              C3B[22] * s[22] * s[22] + C3B[23] * s[23] * s[23]);
  }
  if (num_L >= L_max + 1) {
    q[L_max * n_max_angular_plus_1 + n] =
      C4B[0] * s[3] * s[3] * s[3] + C4B[1] * s[3] * (s[4] * s[4] + s[5] * s[5]) +
      C4B[2] * s[3] * (s[6] * s[6] + s[7] * s[7]) + C4B[3] * s[6] * (s[5] * s[5] - s[4] * s[4]) +
      C4B[4] * s[4] * s[5] * s[7];
  }
  if (num_L >= L_max + 2) {
    float s0_sq = s[0] * s[0];
    float s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];
    q[(L_max + 1) * n_max_angular_plus_1 + n] = C5B[0] * s0_sq * s0_sq + 
                                                C5B[1] * s0_sq * s1_sq_plus_s2_sq +
                                                C5B[2] * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq;
  }
}

#ifdef USE_TABLE
namespace
{
const int table_length = 2001;
const int table_segments = table_length - 1;
const float table_resolution = 0.0005f;

__device__ void find_index_and_weight(
  const float d12_reduced,
  int& index_left,
  int& index_right,
  float& weight_left,
  float& weight_right)
{
  float d12_index = d12_reduced * table_segments;
  index_left = int(d12_index);
  if (index_left == table_segments) {
    --index_left;
  }
  index_right = index_left + 1;
  weight_right = d12_index - index_left;
  weight_left = 1.0f - weight_right;
}

static void construct_table_radial_or_angular(
  const int num_types,
  const int num_types_sq,
  const int n_max,
  const int basis_size,
  const float rc,
  const float rcinv,
  const float* c,
  float* gn,
  float* gnp)
{
  for (int table_index = 0; table_index < table_length; ++table_index) {
    float d12 = table_index * table_resolution * rc;
    float fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    for (int t1 = 0; t1 < num_types; ++t1) {
      for (int t2 = 0; t2 < num_types; ++t2) {
        int t12 = t1 * num_types + t2;
        float fn12[MAX_NUM_N];
        float fnp12[MAX_NUM_N];
        find_fn_and_fnp(basis_size, rcinv, d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= n_max; ++n) {
          float gn12 = 0.0f;
          float gnp12 = 0.0f;
          for (int k = 0; k <= basis_size; ++k) {
            gn12 += fn12[k] * c[(n * (basis_size + 1) + k) * num_types_sq + t12];
            gnp12 += fnp12[k] * c[(n * (basis_size + 1) + k) * num_types_sq + t12];
          }
          int index_all = (table_index * num_types_sq + t12) * (n_max + 1) + n;
          gn[index_all] = gn12;
          gnp[index_all] = gnp12;
        }
      }
    }
  }
}
} // namespace
#endif