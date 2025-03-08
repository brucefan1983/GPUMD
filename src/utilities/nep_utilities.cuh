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

const int NUM_OF_ABC = 80; // 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 for L_max = 8
__constant__ double C3B[NUM_OF_ABC] = {
  0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435,
  0.596831036594608, 0.596831036594608, 0.149207759148652, 0.149207759148652,
  0.139260575205408, 0.104445431404056, 0.104445431404056, 1.044454314040563,
  1.044454314040563, 0.174075719006761, 0.174075719006761, 0.011190581936149,
  0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
  1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606,
  0.013677377921960, 0.102580334414698, 0.102580334414698, 2.872249363611549,
  2.872249363611549, 0.119677056817148, 0.119677056817148, 2.154187022708661,
  2.154187022708661, 0.215418702270866, 0.215418702270866, 0.004041043476943,
  0.169723826031592, 0.169723826031592, 0.106077391269745, 0.106077391269745,
  0.424309565078979, 0.424309565078979, 0.127292869523694, 0.127292869523694,
  2.800443129521260, 2.800443129521260, 0.233370260793438, 0.233370260793438,
  0.004662742473395, 0.004079899664221, 0.004079899664221, 0.024479397985326,
  0.024479397985326, 0.012239698992663, 0.012239698992663, 0.538546755677165,
  0.538546755677165, 0.134636688919291, 0.134636688919291, 3.500553911901575,
  3.500553911901575, 0.250039565135827, 0.250039565135827, 0.000082569397966,
  0.005944996653579, 0.005944996653579, 0.104037441437634, 0.104037441437634,
  0.762941237209318, 0.762941237209318, 0.114441185581398, 0.114441185581398,
  5.950941650232678, 5.950941650232678, 0.141689086910302, 0.141689086910302,
  4.250672607309055, 4.250672607309055, 0.265667037956816, 0.265667037956816
};
__constant__ double C4B[5] = {
  -0.007499480826664,
  -0.134990654879954,
  0.067495327439977,
  0.404971964639861,
  -0.809943929279723};
__constant__ double C5B[3] = {0.026596810706114, 0.053193621412227, 0.026596810706114};

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

__constant__ double COVALENT_RADIUS[94] = {
  0.426667, 0.613333, 1.6,     1.25333, 1.02667, 1.0,     0.946667, 0.84,    0.853333,
  0.893333, 1.86667,  1.66667, 1.50667, 1.38667, 1.46667, 1.36,     1.32,    1.28,
  2.34667,  2.05333,  1.77333, 1.62667, 1.61333, 1.46667, 1.42667,  1.38667, 1.33333,
  1.32,     1.34667,  1.45333, 1.49333, 1.45333, 1.53333, 1.46667,  1.52,    1.56,
  2.52,     2.22667,  1.96,    1.85333, 1.76,    1.65333, 1.53333,  1.50667, 1.50667,
  1.44,     1.53333,  1.64,    1.70667, 1.68,    1.68,    1.64,     1.76,    1.74667,
  2.78667,  2.34667,  2.16,    1.96,    2.10667, 2.09333, 2.08,     2.06667, 2.01333,
  2.02667,  2.01333,  2.0,     1.98667, 1.98667, 1.97333, 2.04,     1.94667, 1.82667,
  1.74667,  1.64,     1.57333, 1.54667, 1.48,    1.49333, 1.50667,  1.76,    1.73333,
  1.73333,  1.81333,  1.74667, 1.84,    1.89333, 2.68,    2.41333,  2.22667, 2.10667,
  2.02667,  2.04,     2.05333, 2.06667};

const int SIZE_BOX_AND_INVERSE_BOX = 18; // (3 * 3) * 2
const int MAX_NUM_N = 20;                // n_max+1 = 19+1
const int MAX_DIM = MAX_NUM_N * 7;
const int MAX_DIM_ANGULAR = MAX_NUM_N * 6;
const int MAX_LN = MAX_NUM_N * 8; 

static __device__ __forceinline__ void
complex_product(const double a, const double b, double& real_part, double& imag_part)
{
  const double real_temp = real_part;
  real_part = a * real_temp - b * imag_part;
  imag_part = a * imag_part + b * real_temp;
}

static __device__ void one_layer(
  const int N_des,
  const int N_neu,
  const double* w0,
  const double* b0,
  const double* w1,
  double* q,
  double& energy,
  double* energy_derivative)
{
  for (int n = 0; n < N_neu; ++n) {
    double w0_times_q = 0.0;
    for (int d = 0; d < N_des; ++d) {
      w0_times_q += w0[n * N_des + d] * q[d];
    }
    double x1 = tanh(w0_times_q + b0[n]);
    double tanh_der = 1.0 - x1 * x1;
    energy += w1[n] * x1;
    for (int d = 0; d < N_des; ++d) {
      double y1 = tanh_der * w0[n * N_des + d];
      energy_derivative[d] += w1[n] * y1;
    }
  }
  energy += w1[N_neu];
}

static __device__ void apply_ann_one_layer_w2nd(
  const int N_des,
  const int N_neu,
  const double* w0,
  const double* b0,
  const double* w1,
  const int N,
  double* q,
  double& energy,
  double* energy_derivative,
  double* energy_derivative2,
  double* ep_wb,   // derivative of e_wb_grad w.r.t q[n]
  double* e_wb_grad) // energy w.r.t. w0, b0, w1, b1
{
  for (int j = 0; j < N_neu; ++j) {
    double w0_times_q = 0.0;
    for (int n = 0; n < N_des; ++n) {
      w0_times_q += w0[j * N_des + n] * q[n];
    }
    double x1 = tanh(w0_times_q + b0[j]);
    double tanh_der = 1.0 - x1 * x1;
    double tanh_der2 = -2.0 * x1 * tanh_der;  // second derivative of tanh
    double delta_1 = w1[j] * tanh_der;
    energy += w1[j] * x1;
    for (int n = 0; n < N_des; ++n) {
      double tmp1 = tanh_der * w0[j * N_des + n]; // derivative of tanh w.r.t. q[n]
      double tmp2 = w1[j] * tanh_der2;
      energy_derivative[n] += w1[j] * tmp1;
      ep_wb[(N_neu * N_des + N_neu + j) * N_des + n] = tmp1; // derivative of e_wb_grad[w1] w.r.t. q[n]
      ep_wb[(N_neu * N_des + j) * N_des + n] = tmp2 * w0[j * N_des + n]; // derivative of e_wb_grad[b0] w.r.t. q[n]
      // second derivative
      for (int m = 0; m < N_des; ++m) {
        double tmp3 = tanh_der2 * w0[j * N_des + n] * w0[j * N_des + m];
        energy_derivative2[(n * N_des + m) * N] += w1[j] * tmp3;
        ep_wb[(j * N_des + n) * N_des + m] = tmp2 * w0[j * N_des + m] * q[n]; // derivative of e_wb_grad[w0] w.r.t. q[n]
        ep_wb[(j * N_des + n) * N_des + m] += (m == n) ? delta_1 : 0.0f; 
      }
      e_wb_grad[j * N_des + n] += delta_1 * q[n]; // energy w.r.t. w0
    }
    e_wb_grad[N_neu * N_des + j] += delta_1; // energy w.r.t. b0
    e_wb_grad[N_neu * N_des + N_neu + j] += x1; // energy w.r.t. w1
    e_wb_grad[N_neu * N_des + N_neu + N_neu] = 1.0; // energy w.r.t. b1
    // w0 (N_neu * N_des), b0 (N_neu), w1 (N_neu), b1 (1)
  }
  energy += w1[N_neu];
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

static __device__ __forceinline__ void find_fc(double rc, double rcinv, double d12, double& fc)
{
  if (d12 < rc) {
    double x = d12 * rcinv;
    fc = 0.5 * cos(PI * x) + 0.5;
  } else {
    fc = 0.0;
  }
}

static __device__ __host__ __forceinline__ void
find_fc_and_fcp(double rc, double rcinv, double d12, double& fc, double& fcp)
{
  if (d12 < rc) {
    double x = d12 * rcinv;
    fc = 0.5 * cos(PI * x) + 0.5;
    fcp = -1.5707963 * sin(PI * x);
    fcp *= rcinv;
  } else {
    fc = 0.0;
    fcp = 0.0;
  }
}

static __device__ __forceinline__ void
find_fc_and_fcp_zbl(double r1, double r2, double d12, double& fc, double& fcp)
{
  if (d12 < r1) {
    fc = 1.0;
    fcp = 0.0;
  } else if (d12 < r2) {
    double pi_factor = PI / (r2 - r1);
    fc = cos(pi_factor * (d12 - r1)) * 0.5 + 0.5;
    fcp = -sin(pi_factor * (d12 - r1)) * pi_factor * 0.5;
  } else {
    fc = 0.0;
    fcp = 0.0;
  }
}

static __device__ __forceinline__ void
find_phi_and_phip_zbl(double a, double b, double x, double& phi, double& phip)
{
  double tmp = a * exp(-b * x);
  phi += tmp;
  phip -= b * tmp;
}

static __device__ __forceinline__ void find_f_and_fp_zbl(
  const double zizj,
  const double a_inv,
  const double rc_inner,
  const double rc_outer,
  const double d12,
  const double d12inv,
  double& f,
  double& fp)
{
  const double x = d12 * a_inv;
  f = fp = 0.0;
  const double Zbl_para[8] = {
    0.18175, 3.1998, 0.50986, 0.94229, 0.28022, 0.4029, 0.02817, 0.20162};
  find_phi_and_phip_zbl(Zbl_para[0], Zbl_para[1], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[2], Zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[4], Zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(Zbl_para[6], Zbl_para[7], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  double fc, fcp;
  find_fc_and_fcp_zbl(rc_inner, rc_outer, d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

static __device__ __forceinline__ void find_f_and_fp_zbl(
  const double* zbl_para,
  const double zizj,
  const double a_inv,
  const double d12,
  const double d12inv,
  double& f,
  double& fp)
{
  const double x = d12 * a_inv;
  f = fp = 0.0;
  find_phi_and_phip_zbl(zbl_para[2], zbl_para[3], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[4], zbl_para[5], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[6], zbl_para[7], x, f, fp);
  find_phi_and_phip_zbl(zbl_para[8], zbl_para[9], x, f, fp);
  f *= zizj;
  fp *= zizj * a_inv;
  fp = fp * d12inv - f * d12inv * d12inv;
  f *= d12inv;
  double fc, fcp;
  find_fc_and_fcp_zbl(zbl_para[0], zbl_para[1], d12, fc, fcp);
  fp = fp * fc + f * fcp;
  f *= fc;
}

static __device__ __forceinline__ void
find_fn(const int n, const double rcinv, const double d12, const double fc12, double& fn)
{
  if (n == 0) {
    fn = fc12;
  } else if (n == 1) {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    fn = (x + 1.0) * 0.5 * fc12;
  } else {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    double t0 = 1.0;
    double t1 = x;
    double t2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0 * x * t1 - t0;
      t0 = t1;
      t1 = t2;
    }
    fn = (t2 + 1.0) * 0.5 * fc12;
  }
}

static __device__ __forceinline__ void find_fn_and_fnp(
  const int n,
  const double rcinv,
  const double d12,
  const double fc12,
  const double fcp12,
  double& fn,
  double& fnp)
{
  if (n == 0) {
    fn = fc12;
    fnp = fcp12;
  } else if (n == 1) {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    fn = (x + 1.0) * 0.5;
    fnp = 2.0 * (d12 * rcinv - 1.0) * rcinv * fc12 + fn * fcp12;
    fn *= fc12;
  } else {
    double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
    double t0 = 1.0;
    double t1 = x;
    double t2;
    double u0 = 1.0;
    double u1 = 2.0 * x;
    double u2;
    for (int m = 2; m <= n; ++m) {
      t2 = 2.0 * x * t1 - t0;
      t0 = t1;
      t1 = t2;
      u2 = 2.0 * x * u1 - u0;
      u0 = u1;
      u1 = u2;
    }
    fn = (t2 + 1.0) * 0.5;
    fnp = n * u0 * 2.0 * (d12 * rcinv - 1.0) * rcinv;
    fnp = fnp * fc12 + fn * fcp12;
    fn *= fc12;
  }
}

static __device__ __forceinline__ void
find_fn(const int n_max, const double rcinv, const double d12, const double fc12, double* fn)
{
  double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
  double half_fc12 = 0.5 * fc12;
  fn[0] = fc12;
  fn[1] = (x + 1.0) * half_fc12;
  double fn_m_minus_2 = 1.0;
  double fn_m_minus_1 = x;
  double tmp = 0.0;
  for (int m = 2; m <= n_max; ++m) {
    tmp = 2.0 * x * fn_m_minus_1 - fn_m_minus_2;
    fn_m_minus_2 = fn_m_minus_1;
    fn_m_minus_1 = tmp;
    fn[m] = (tmp + 1.0) * half_fc12;
  }
}

static __device__ __host__ __forceinline__ void find_fn_and_fnp(
  const int n_max,
  const double rcinv,
  const double d12,
  const double fc12,
  const double fcp12,
  double* fn,
  double* fnp)
{
  double d12_mul_rcinv = d12 * rcinv;
  double x = 2.0 * (d12_mul_rcinv - 1.0) * (d12_mul_rcinv - 1.0) - 1.0;
  fn[0] = fc12;
  fnp[0] = fcp12;
  fn[1] = (x + 1.0) * 0.5 * fc12;
  fnp[1] = 2.0 * (d12_mul_rcinv - 1.0) * rcinv * fc12 + (x + 1.0) * 0.5 * fcp12;
  double u0 = 1.0;
  double u1 = 2.0 * x;
  double u2;
  double fn_m_minus_2 = 1.0;
  double fn_m_minus_1 = x;
  for (int m = 2; m <= n_max; ++m) {
    double fn_tmp1 = 2.0 * x * fn_m_minus_1 - fn_m_minus_2;
    fn_m_minus_2 = fn_m_minus_1;
    fn_m_minus_1 = fn_tmp1;
    double fnp_tmp = m * u1;
    u2 = 2.0 * x * u1 - u0;
    u0 = u1;
    u1 = u2;

    double fn_tmp2 = (fn_tmp1 + 1.0) * 0.5;
    fnp[m] = (fnp_tmp * 2.0 * (d12 * rcinv - 1.0) * rcinv) * fc12 + fn_tmp2 * fcp12;
    fn[m] = fn_tmp2 * fc12;
  }
}

static __device__ __forceinline__ void get_f12_4body(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12)
{
  double fn_factor = Fp * fn;
  double fnp_factor = Fp * fnp * d12inv;
  double y20 = (3.0 * r12[2] * r12[2] - d12 * d12);

  // derivative wrt s[0]
  double tmp0 = C4B[0] * 3.0 * s[0] * s[0] + C4B[1] * (s[1] * s[1] + s[2] * s[2]) +
               C4B[2] * (s[3] * s[3] + s[4] * s[4]);
  double tmp1 = tmp0 * y20 * fnp_factor;
  double tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] - tmp2 * 2.0 * r12[0];
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
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12)
{
  double fn_factor = Fp * fn;
  double fnp_factor = Fp * fnp * d12inv;
  double s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];

  // derivative wrt s[0]
  double tmp0 = C5B[0] * 4.0 * s[0] * s[0] * s[0] + C5B[1] * s1_sq_plus_s2_sq * 2.0 * s[0];
  double tmp1 = tmp0 * r12[2] * fnp_factor;
  double tmp2 = tmp0 * fn_factor;
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

template <int L>
static __device__ __forceinline__ void calculate_s_one(
  const int N,
  const int n,
  const int n_max_angular_plus_1,
  const double* Fp,
  const double* sum_fxyz,
  double* s)
{
  const int L_minus_1 = L - 1;
  const int L_twice_plus_1 = 2 * L + 1;
  const int L_square_minus_1 = L * L - 1;
  const int index_base = n * NUM_OF_ABC + L_square_minus_1;
  const int index_0 = index_base * N;
  double Fp_factor = 2.0 * Fp[L_minus_1 * n_max_angular_plus_1 + n];
  s[0] = sum_fxyz[index_0] * C3B[L_square_minus_1] * Fp_factor;
  Fp_factor *= 2.0;
  for (int k = 1; k < L_twice_plus_1; ++k) {
    const int index_s = index_base + k;
    const int index_s0 = index_s * N;
    s[k] = sum_fxyz[index_s0] * C3B[L_square_minus_1 + k] * Fp_factor;
  }
}

template <int L>
static __device__ __forceinline__ void calculate_s_one(
  const int N,
  const int n,
  const int n_max_angular_plus_1,
  const double* sum_fxyz,
  double* s)
{
  const int L_twice_plus_1 = 2 * L + 1;
  const int L_square_minus_1 = L * L - 1;
  const int index_base = n * NUM_OF_ABC + L_square_minus_1;
  const int index_0 = index_base * N;
  s[0] = 2.0 * sum_fxyz[index_0] * C3B[L_square_minus_1];
  for (int k = 1; k < L_twice_plus_1; ++k) {
    const int index_s = index_base + k;
    const int index_s0 = index_s * N;
    s[k] = 4.0 * sum_fxyz[index_s0] * C3B[L_square_minus_1 + k];
  }
}

template <int L>
static __device__ __forceinline__ void calculate_ec_one(
  const int N,
  const int n,
  const int n_max_angular_plus_1,
  const double* Fp,
  const double* sum_fxyz,
  const double* sum_s2xyz,
  const double* sum_s2xyz123,
  const double* s_c,
  double* s,
  double* ec,
  double* f,
  double* f123)
{
  const int L_minus_1 = L - 1;
  const int L_twice_plus_1 = 2 * L + 1;
  const int L_square_minus_1 = L * L - 1;
  double Fp_factor = 2.0 * Fp[L_minus_1 * n_max_angular_plus_1 + n];
  const int index_base = n * NUM_OF_ABC + L_square_minus_1;
  const int index_0 = index_base * N;
  const int index_1 = index_0 * 3;
  const int index_123 = index_1 * 2;
  const double baseC3B = C3B[L_square_minus_1];
  const double base_s_c = s_c[0];
  (*ec) += sum_fxyz[index_0] * baseC3B * base_s_c * Fp_factor;
  s[0] = sum_fxyz[index_0] * baseC3B * Fp_factor;
  f[0] = sum_s2xyz[index_1] * baseC3B * base_s_c * Fp_factor;
  f[1] = sum_s2xyz[index_1 + N * 1] * baseC3B * base_s_c * Fp_factor;
  f[2] = sum_s2xyz[index_1 + N * 2] * baseC3B * base_s_c * Fp_factor;
  f123[0] = sum_s2xyz123[index_123] * baseC3B * base_s_c * Fp_factor; // index合并访问测试?
  f123[1] = sum_s2xyz123[index_123 + N * 1] * baseC3B * base_s_c * Fp_factor;
  f123[2] = sum_s2xyz123[index_123 + N * 2] * baseC3B * base_s_c * Fp_factor;
  f123[3] = sum_s2xyz123[index_123 + N * 3] * baseC3B * base_s_c * Fp_factor;
  f123[4] = sum_s2xyz123[index_123 + N * 4] * baseC3B * base_s_c * Fp_factor;
  f123[5] = sum_s2xyz123[index_123 + N * 5] * baseC3B * base_s_c * Fp_factor;
  Fp_factor *= 2.0;
  for (int k = 1; k < L_twice_plus_1; ++k) {
    const int index_s = index_base + k;
    const int index_s0 = index_s * N;
    const int index_s1 = index_s0 * 3;
    const int index_s123 = index_s1 * 2;
    const double c3b_val = C3B[L_square_minus_1 + k];
    const double s_c_val = s_c[k];
    (*ec) += sum_fxyz[index_s0] * c3b_val * s_c_val * Fp_factor;
    s[k] = sum_fxyz[index_s0] * c3b_val * Fp_factor;
    f[0] += sum_s2xyz[index_s1] * c3b_val * s_c_val * Fp_factor;
    f[1] += sum_s2xyz[index_s1 + N * 1] * c3b_val * s_c_val * Fp_factor;
    f[2] += sum_s2xyz[index_s1 + N * 2] * c3b_val * s_c_val * Fp_factor;
    f123[0] += sum_s2xyz123[index_s123] * c3b_val * s_c_val * Fp_factor;
    f123[1] += sum_s2xyz123[index_s123 + N * 1] * c3b_val * s_c_val * Fp_factor;
    f123[2] += sum_s2xyz123[index_s123 + N * 2] * c3b_val * s_c_val * Fp_factor;
    f123[3] += sum_s2xyz123[index_s123 + N * 3] * c3b_val * s_c_val * Fp_factor;
    f123[4] += sum_s2xyz123[index_s123 + N * 4] * c3b_val * s_c_val * Fp_factor;
    f123[5] += sum_s2xyz123[index_s123 + N * 5] * c3b_val * s_c_val * Fp_factor;
  }
}

template <int L>
static __device__ __forceinline__ void calculate_fc_one(
  const int N,
  const int n,
  const int n_max_angular_plus_1,
  const double* Fp,
  const double* sum_fxyz,
  const double* sum_s2xyz,
  const double* s_c,
  double* s,
  double* f)
{
  const int L_minus_1 = L - 1;
  const int L_twice_plus_1 = 2 * L + 1;
  const int L_square_minus_1 = L * L - 1;
  double Fp_factor = 2.0 * Fp[L_minus_1 * n_max_angular_plus_1 + n];
  int index_1 = L_square_minus_1 * 3;
  const double baseC3B = C3B[L_square_minus_1];
  const double base_s_c = s_c[0];
  s[0] = sum_fxyz[L_square_minus_1] * baseC3B * Fp_factor;
  f[0] += sum_s2xyz[index_1] * baseC3B * base_s_c * Fp_factor;
  f[1] += sum_s2xyz[index_1 + 1] * baseC3B * base_s_c * Fp_factor;
  f[2] += sum_s2xyz[index_1 + 2] * baseC3B * base_s_c * Fp_factor;
  Fp_factor *= 2.0;
  for (int k = 1; k < L_twice_plus_1; ++k) {
    int index_s = L_square_minus_1 + k;
    int index_s1 = index_s * 3;
    double c3b_val = C3B[L_square_minus_1 + k];
    double s_c_val = s_c[k];
    s[k] = sum_fxyz[index_s] * c3b_val * Fp_factor;
    f[0] += sum_s2xyz[index_s1] * c3b_val * s_c_val * Fp_factor;
    f[1] += sum_s2xyz[index_s1 + 1] * c3b_val * s_c_val * Fp_factor;
    f[2] += sum_s2xyz[index_s1 + 2] * c3b_val * s_c_val * Fp_factor;
  }
}

template <int L>
static __device__ __forceinline__ void calculate_qc_one(
  const int N,
  const int n,
  const int n_max_angular_plus_1,
  const double* sum_fxyz,
  const double* s_c,
  double* qc)
{
  const int L_twice_plus_1 = 2 * L + 1;
  const int L_square_minus_1 = L * L - 1;
  const int index_base = n * NUM_OF_ABC + L_square_minus_1;
  const int index_0 = index_base * N;
  (*qc) = 2.0 * sum_fxyz[index_0] * C3B[L_square_minus_1] * s_c[0];
  for (int k = 1; k < L_twice_plus_1; ++k) {
    const int index_s = index_base + k;
    const int index_s0 = index_s * N;
    (*qc) += 4.0 * sum_fxyz[index_s0] * C3B[L_square_minus_1 + k] * s_c[k];
  }
}

template <int L>
static __device__ __forceinline__ void accumulate_f12_one(
  const double d12inv,
  const double fn,
  const double fnp,
  const double* s,
  const double* r12,
  double* f12)
{
  const double dx[3] = {(1.0 - r12[0] * r12[0]) * d12inv, -r12[0] * r12[1] * d12inv, -r12[0] * r12[2] * d12inv};
  const double dy[3] = {-r12[0] * r12[1] * d12inv, (1.0 - r12[1] * r12[1]) * d12inv, -r12[1] * r12[2] * d12inv};
  const double dz[3] = {-r12[0] * r12[2] * d12inv, -r12[1] * r12[2] * d12inv, (1.0 - r12[2] * r12[2]) * d12inv};

  double z_pow[L + 1] = {1.0};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = r12[2] * z_pow[n - 1];
  }

  double real_part = 1.0;
  double imag_part = 0.0;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    double z_factor = 0.0;
    double dz_factor = 0.0;
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
      if (L == 5) {
        z_factor += Z_COEFFICIENT_5[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_5[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 6) {
        z_factor += Z_COEFFICIENT_6[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_6[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 7) {
        z_factor += Z_COEFFICIENT_7[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_7[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 8) {
        z_factor += Z_COEFFICIENT_8[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_8[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
    }
    if (n1 == 0) {
      for (int d = 0; d < 3; ++d) {
        f12[d] += s[0] * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
      }
    } else {
      double real_part_n1 = n1 * real_part;
      double imag_part_n1 = n1 * imag_part;
      for (int d = 0; d < 3; ++d) {
        double real_part_dx = dx[d];
        double imag_part_dy = dy[d];
        complex_product(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        f12[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor * fn;
      }
      complex_product(r12[0], r12[1], real_part, imag_part);
      const float xy_temp = s[2 * n1 - 1] * real_part + s[2 * n1 - 0] * imag_part;
      for (int d = 0; d < 3; ++d) {
        f12[d] += xy_temp * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
      }
    }
  }
}

template <int L>
static __device__ __forceinline__ void accumulate_f12_one(
  const double d12inv,
  const double fn,
  const double fnp,
  const double* s,
  const double* r12,
  const double* r12_original,
  double* f12,
  double* f123)
{
  const double dx[3] = {(1.0 - r12[0] * r12[0]) * d12inv, -r12[0] * r12[1] * d12inv, -r12[0] * r12[2] * d12inv};
  const double dy[3] = {-r12[0] * r12[1] * d12inv, (1.0 - r12[1] * r12[1]) * d12inv, -r12[1] * r12[2] * d12inv};
  const double dz[3] = {-r12[0] * r12[2] * d12inv, -r12[1] * r12[2] * d12inv, (1.0 - r12[2] * r12[2]) * d12inv};

  double z_pow[L + 1] = {1.0};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = r12[2] * z_pow[n - 1];
  }

  double real_part = 1.0;
  double imag_part = 0.0;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    double z_factor = 0.0;
    double dz_factor = 0.0;
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
      if (L == 5) {
        z_factor += Z_COEFFICIENT_5[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_5[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 6) {
        z_factor += Z_COEFFICIENT_6[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_6[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 7) {
        z_factor += Z_COEFFICIENT_7[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_7[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 8) {
        z_factor += Z_COEFFICIENT_8[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_8[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
    }
    if (n1 == 0) {
      for (int d = 0; d < 3; ++d) {
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        f12[d] += s[0] * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
        f123[d] += s[0] * r12_original[d] * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
        f123[d1+3] += s[0] * r12_original[d1] * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
      }
    } else {
      double real_part_n1 = n1 * real_part;
      double imag_part_n1 = n1 * imag_part;
      for (int d = 0; d < 3; ++d) {
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        double real_part_dx = dx[d];
        double imag_part_dy = dy[d];
        complex_product(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        f12[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor * fn;
        f123[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor * fn * r12_original[d];
        f123[d1+3] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor * fn * r12_original[d1];
      }
      complex_product(r12[0], r12[1], real_part, imag_part);
      const float xy_temp = s[2 * n1 - 1] * real_part + s[2 * n1 - 0] * imag_part;
      for (int d = 0; d < 3; ++d) {
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        f12[d] += xy_temp * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
        f123[d] += xy_temp * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]) * r12_original[d];
        f123[d1+3] += xy_temp * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]) * r12_original[d1];
      }
    }
  }
}

template <int L>
static __device__ __forceinline__ void calculate_fxyz_one(
  const double d12inv,
  const double fn,
  const double fnp,
  const double* s,
  const double* r12,
  const double* r12_original,
  double* s_i1,
  double* sf,
  double* f12,
  double* f123)
{
  const double dx[3] = {(1.0 - r12[0] * r12[0]) * d12inv, -r12[0] * r12[1] * d12inv, -r12[0] * r12[2] * d12inv};
  const double dy[3] = {-r12[0] * r12[1] * d12inv, (1.0 - r12[1] * r12[1]) * d12inv, -r12[1] * r12[2] * d12inv};
  const double dz[3] = {-r12[0] * r12[2] * d12inv, -r12[1] * r12[2] * d12inv, (1.0 - r12[2] * r12[2]) * d12inv};
  const int start_index = L * L - 1;
  int s_index = L * L - 1;
  double z_pow[L + 1] = {1.0};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = r12[2] * z_pow[n - 1];
  }

  double real_part = 1.0;
  double imag_part = 0.0;
  double real_part_s_i1 = r12[0];
  double imag_part_s_i1 = r12[1];
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    double z_factor = 0.0;
    double dz_factor = 0.0;
    double z_factor_i1 = 0.0;
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
      if (L == 5) {
        z_factor += Z_COEFFICIENT_5[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_5[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 6) {
        z_factor += Z_COEFFICIENT_6[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_6[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 7) {
        z_factor += Z_COEFFICIENT_7[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_7[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 8) {
        z_factor += Z_COEFFICIENT_8[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_8[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
    }
    z_factor_i1 = z_factor * fn;
    if (n1 == 0) {
      s_i1[s_index++] = z_factor_i1;
      for (int d = 0; d < 3; ++d) {
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        f12[d] += s[0] * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
        sf[d + 3 * start_index] = z_factor * fnp * r12[d] + fn * dz_factor * dz[d];
        f123[d] += s[0] * r12_original[d] * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
        f123[d1+3] += s[0] * r12_original[d1] * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
      }
    } else {
      double real_part_n1 = n1 * real_part;
      double imag_part_n1 = n1 * imag_part;
      int abc = 3 * (start_index + 2 * n1 - 1);
      s_i1[s_index++] = z_factor_i1 * real_part_s_i1;
      s_i1[s_index++] = z_factor_i1 * imag_part_s_i1;
      complex_product(r12[0], r12[1], real_part_s_i1, imag_part_s_i1);
      for (int d = 0; d < 3; ++d) {
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        double real_part_dx = dx[d];
        double imag_part_dy = dy[d];
        complex_product(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        f12[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor * fn;
        sf[d + abc] = real_part_dx * z_factor * fn;
        sf[d + abc + 3] = imag_part_dy * z_factor * fn;
        f123[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor * fn * r12_original[d];
        f123[d1+3] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor * fn * r12_original[d1];
      }
      complex_product(r12[0], r12[1], real_part, imag_part);
      const float xy_temp = s[2 * n1 - 1] * real_part + s[2 * n1 - 0] * imag_part;
      for (int d = 0; d < 3; ++d) {
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        f12[d] += xy_temp * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
        sf[d + abc] += real_part * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
        sf[d + abc + 3] += imag_part * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
        f123[d] += xy_temp * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]) * r12_original[d];
        f123[d1+3] += xy_temp * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]) * r12_original[d1];
      }
    }
  }
}

template <int L>
static __device__ __forceinline__ void accumulate_f12_one(
  const int N,
  const int n_max,
  const double d12inv,
  const double fn,
  const double fnp,
  const double* s,
  const double* r12,
  const double* r12_original,
  double* sf,
  double* sf123,
  double* f12)
{
  const double dx[3] = {(1.0 - r12[0] * r12[0]) * d12inv, -r12[0] * r12[1] * d12inv, -r12[0] * r12[2] * d12inv};
  const double dy[3] = {-r12[0] * r12[1] * d12inv, (1.0 - r12[1] * r12[1]) * d12inv, -r12[1] * r12[2] * d12inv};
  const double dz[3] = {-r12[0] * r12[2] * d12inv, -r12[1] * r12[2] * d12inv, (1.0 - r12[2] * r12[2]) * d12inv};
  const int N_ABC = 3 * NUM_OF_ABC * n_max;
  const int N_ABC123 = 2 * N_ABC;
  const int start_index = L * L - 1;
  double z_pow[L + 1] = {1.0};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = r12[2] * z_pow[n - 1];
  }

  double real_part = 1.0;
  double imag_part = 0.0;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    double z_factor = 0.0;
    double dz_factor = 0.0;
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
      if (L == 5) {
        z_factor += Z_COEFFICIENT_5[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_5[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 6) {
        z_factor += Z_COEFFICIENT_6[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_6[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 7) {
        z_factor += Z_COEFFICIENT_7[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_7[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
      if (L == 8) {
        z_factor += Z_COEFFICIENT_8[n1][n2] * z_pow[n2];
        if (n2 > 0) {
          dz_factor += Z_COEFFICIENT_8[n1][n2] * n2 * z_pow[n2 - 1];
        }
      }
    }
    if (n1 == 0) {
      for (int d = 0; d < 3; ++d) {
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        f12[d] += s[0] * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
        sf[N * (d + 3 * start_index + N_ABC)] += z_factor * fnp * r12[d] + fn * dz_factor * dz[d];
        sf123[N * (d + 6 * start_index + N_ABC123)] += (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]) * r12_original[d];
        sf123[N * ((d1 + 3) + 6 * start_index + N_ABC123)] += (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]) * r12_original[d1];
      }
    } else {
      double real_part_n1 = n1 * real_part;
      double imag_part_n1 = n1 * imag_part;
      int abc = 3 * (start_index + 2 * n1 - 1);
      int abc123 = 2 * abc;
      for (int d = 0; d < 3; ++d) {
        double real_part_dx = dx[d];
        double imag_part_dy = dy[d];
        int index = N * (d + abc + N_ABC);
        int index123 = N * (d + abc123 + N_ABC123);
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        int index123_off = N * ((d1 + 3) + abc123 + N_ABC123); 
        complex_product(real_part_n1, imag_part_n1, real_part_dx, imag_part_dy);
        f12[d] += (s[2 * n1 - 1] * real_part_dx + s[2 * n1 - 0] * imag_part_dy) * z_factor * fn;
        sf[index] += real_part_dx * z_factor * fn;
        sf[index + 3 * N] += imag_part_dy * z_factor * fn;
        sf123[index123] += real_part_dx * z_factor * fn * r12_original[d];
        sf123[index123 + 6 * N] += imag_part_dy * z_factor * fn * r12_original[d];
        sf123[index123_off] += real_part_dx * z_factor * fn * r12_original[d1];
        sf123[index123_off + 6 * N] += imag_part_dy * z_factor * fn * r12_original[d1];
      }
      complex_product(r12[0], r12[1], real_part, imag_part);
      const float xy_temp = s[2 * n1 - 1] * real_part + s[2 * n1 - 0] * imag_part;
      for (int d = 0; d < 3; ++d) {
        int index = N * (d + abc + N_ABC);
        int index123 = N * (d + abc123 + N_ABC123);
        int d1 = (d + 2) % 3; // 0 -> 2, 1 -> 0, 2 -> 1
        int index123_off = N * ((d1 + 3) + abc123 + N_ABC123); 
        f12[d] += xy_temp * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
        sf[index] += real_part * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
        sf[index + 3 * N] += imag_part * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]);
        sf123[index123] += real_part * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]) * r12_original[d];
        sf123[index123 + 6 * N] += imag_part * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]) * r12_original[d];
        sf123[index123_off] += real_part * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]) * r12_original[d1];
        sf123[index123_off + 6 * N] += imag_part * (z_factor * fnp * r12[d] + fn * dz_factor * dz[d]) * r12_original[d1];
      }
    }
  }
}

static __device__ __forceinline__ void accumulate_dfe(
  const int N,
  const int L_max,
  const int n,
  const int n_max_angular_plus_1,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* sum_fxyz,
  double* s_i1,
  double* sum_s2xyz,
  double* feat_x,
  double* feat_y,
  double* feat_z,
  double* feat_123_xx,
  double* feat_123_yy,
  double* feat_123_zz,
  double* feat_123_xy,
  double* feat_123_yz,
  double* feat_123_zx)
{
  const double d12inv = 1.0 / d12;
  const double r12unit[3] = {r12[0]*d12inv, r12[1]*d12inv, r12[2]*d12inv};

  if (L_max >= 1) {
    double s1[3];
    double f[3] = {0.0};
    double f123[6] = {0.0};
    calculate_s_one<1>(N, n, n_max_angular_plus_1, sum_fxyz, s1);
    calculate_fxyz_one<1>(d12inv, fn, fnp, s1, r12unit, r12, s_i1, sum_s2xyz, f, f123);
    feat_x[0] += f[0];
    feat_y[0] += f[1];
    feat_z[0] += f[2];
    feat_123_xx[0] += f123[0];
    feat_123_yy[0] += f123[1];
    feat_123_zz[0] += f123[2];
    feat_123_xy[0] += f123[3];
    feat_123_yz[0] += f123[4];
    feat_123_zx[0] += f123[5];
  }

  if (L_max >= 2) {
    double s2[5];
    double f[3] = {0.0};
    double f123[6] = {0.0};
    calculate_s_one<2>(N, n, n_max_angular_plus_1, sum_fxyz, s2);
    calculate_fxyz_one<2>(d12inv, fn, fnp, s2, r12unit, r12, s_i1, sum_s2xyz, f, f123);
    feat_x[n_max_angular_plus_1] += f[0];
    feat_y[n_max_angular_plus_1] += f[1];
    feat_z[n_max_angular_plus_1] += f[2];
    feat_123_xx[n_max_angular_plus_1] += f123[0];
    feat_123_yy[n_max_angular_plus_1] += f123[1];
    feat_123_zz[n_max_angular_plus_1] += f123[2];
    feat_123_xy[n_max_angular_plus_1] += f123[3];
    feat_123_yz[n_max_angular_plus_1] += f123[4];
    feat_123_zx[n_max_angular_plus_1] += f123[5];
  }

  if (L_max >= 3) {
    double s3[7];
    double f[3] = {0.0};
    double f123[6] = {0.0};
    calculate_s_one<3>(N, n, n_max_angular_plus_1, sum_fxyz, s3);
    calculate_fxyz_one<3>(d12inv, fn, fnp, s3, r12unit, r12, s_i1, sum_s2xyz, f, f123);
    feat_x[2 * n_max_angular_plus_1] += f[0];
    feat_y[2 * n_max_angular_plus_1] += f[1];
    feat_z[2 * n_max_angular_plus_1] += f[2];
    feat_123_xx[2 * n_max_angular_plus_1] += f123[0];
    feat_123_yy[2 * n_max_angular_plus_1] += f123[1];
    feat_123_zz[2 * n_max_angular_plus_1] += f123[2];
    feat_123_xy[2 * n_max_angular_plus_1] += f123[3];
    feat_123_yz[2 * n_max_angular_plus_1] += f123[4];
    feat_123_zx[2 * n_max_angular_plus_1] += f123[5];
  }

  if (L_max >= 4) {
    double s4[9];
    double f[3] = {0.0};
    double f123[6] = {0.0};
    calculate_s_one<4>(N, n, n_max_angular_plus_1, sum_fxyz, s4);
    calculate_fxyz_one<4>(d12inv, fn, fnp, s4, r12unit, r12, s_i1, sum_s2xyz, f, f123);
    feat_x[3 * n_max_angular_plus_1] += f[0];
    feat_y[3 * n_max_angular_plus_1] += f[1];
    feat_z[3 * n_max_angular_plus_1] += f[2];
    feat_123_xx[3 * n_max_angular_plus_1] += f123[0];
    feat_123_yy[3 * n_max_angular_plus_1] += f123[1];
    feat_123_zz[3 * n_max_angular_plus_1] += f123[2];
    feat_123_xy[3 * n_max_angular_plus_1] += f123[3];
    feat_123_yz[3 * n_max_angular_plus_1] += f123[4];
    feat_123_zx[3 * n_max_angular_plus_1] += f123[5];
  }

  if (L_max >= 5) {
    double s5[11];
    double f[3] = {0.0};
    double f123[6] = {0.0};
    calculate_s_one<5>(N, n, n_max_angular_plus_1, sum_fxyz, s5);
    calculate_fxyz_one<5>(d12inv, fn, fnp, s5, r12unit, r12, s_i1, sum_s2xyz, f, f123);
    feat_x[4 * n_max_angular_plus_1] += f[0];
    feat_y[4 * n_max_angular_plus_1] += f[1];
    feat_z[4 * n_max_angular_plus_1] += f[2];
    feat_123_xx[4 * n_max_angular_plus_1] += f123[0];
    feat_123_yy[4 * n_max_angular_plus_1] += f123[1];
    feat_123_zz[4 * n_max_angular_plus_1] += f123[2];
    feat_123_xy[4 * n_max_angular_plus_1] += f123[3];
    feat_123_yz[4 * n_max_angular_plus_1] += f123[4];
    feat_123_zx[4 * n_max_angular_plus_1] += f123[5];
  }

  if (L_max >= 6) {
    double s6[13];
    double f[3] = {0.0};
    double f123[6] = {0.0};
    calculate_s_one<6>(N, n, n_max_angular_plus_1, sum_fxyz, s6);
    calculate_fxyz_one<6>(d12inv, fn, fnp, s6, r12unit, r12, s_i1, sum_s2xyz, f, f123);
    feat_x[5 * n_max_angular_plus_1] += f[0];
    feat_y[5 * n_max_angular_plus_1] += f[1];
    feat_z[5 * n_max_angular_plus_1] += f[2];
    feat_123_xx[5 * n_max_angular_plus_1] += f123[0];
    feat_123_yy[5 * n_max_angular_plus_1] += f123[1];
    feat_123_zz[5 * n_max_angular_plus_1] += f123[2];
    feat_123_xy[5 * n_max_angular_plus_1] += f123[3];
    feat_123_yz[5 * n_max_angular_plus_1] += f123[4];
    feat_123_zx[5 * n_max_angular_plus_1] += f123[5];
  }

  if (L_max >= 7) {
    double s7[15];
    double f[3] = {0.0};
    double f123[6] = {0.0};
    calculate_s_one<7>(N, n, n_max_angular_plus_1, sum_fxyz, s7);
    calculate_fxyz_one<7>(d12inv, fn, fnp, s7, r12unit, r12, s_i1, sum_s2xyz, f, f123);
    feat_x[6 * n_max_angular_plus_1] += f[0];
    feat_y[6 * n_max_angular_plus_1] += f[1];
    feat_z[6 * n_max_angular_plus_1] += f[2];
    feat_123_xx[6 * n_max_angular_plus_1] += f123[0];
    feat_123_yy[6 * n_max_angular_plus_1] += f123[1];
    feat_123_zz[6 * n_max_angular_plus_1] += f123[2];
    feat_123_xy[6 * n_max_angular_plus_1] += f123[3];
    feat_123_yz[6 * n_max_angular_plus_1] += f123[4];
    feat_123_zx[6 * n_max_angular_plus_1] += f123[5];
  }

  if (L_max >= 8) {
    double s8[17];
    double f[3] = {0.0};
    double f123[6] = {0.0};
    calculate_s_one<8>(N, n, n_max_angular_plus_1, sum_fxyz, s8);
    calculate_fxyz_one<8>(d12inv, fn, fnp, s8, r12unit, r12, s_i1, sum_s2xyz, f, f123);
    feat_x[7 * n_max_angular_plus_1] += f[0];
    feat_y[7 * n_max_angular_plus_1] += f[1];
    feat_z[7 * n_max_angular_plus_1] += f[2];
    feat_123_xx[7 * n_max_angular_plus_1] += f123[0];
    feat_123_yy[7 * n_max_angular_plus_1] += f123[1];
    feat_123_zz[7 * n_max_angular_plus_1] += f123[2];
    feat_123_xy[7 * n_max_angular_plus_1] += f123[3];
    feat_123_yz[7 * n_max_angular_plus_1] += f123[4];
    feat_123_zx[7 * n_max_angular_plus_1] += f123[5];
  }
}

static __device__ __forceinline__ void accumulate_f12(
  const int N,
  const bool requires_grad,
  const int L_max,
  const int num_L,
  const int n,
  const int n_max_angular_plus_1,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  double* sum_s2xyz,
  double* sum_s2xyz123,
  double* f12)
{
  const double fn_original = fn;
  const double fnp_original = fnp;
  const double d12inv = 1.0 / d12;
  const double r12unit[3] = {r12[0]*d12inv, r12[1]*d12inv, r12[2]*d12inv};

  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  if (num_L >= L_max + 2) {
    double s1[3] = {
      sum_fxyz[n * NUM_OF_ABC + 0], sum_fxyz[n * NUM_OF_ABC + 1], sum_fxyz[n * NUM_OF_ABC + 2]};
    get_f12_5body(d12, d12inv, fn, fnp, Fp[(L_max + 1) * n_max_angular_plus_1 + n], s1, r12, f12);
  }

  if (L_max >= 1) {
    double s1[3];
    calculate_s_one<1>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s1);
    if (requires_grad) {
      accumulate_f12_one<1>(N, n, d12inv, fn_original, fnp_original, s1, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<1>(d12inv, fn_original, fnp_original, s1, r12unit, f12);
    }
  }
  
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  if (num_L >= L_max + 1) {
    double s2[5] = {
      sum_fxyz[n * NUM_OF_ABC + 3],
      sum_fxyz[n * NUM_OF_ABC + 4],
      sum_fxyz[n * NUM_OF_ABC + 5],
      sum_fxyz[n * NUM_OF_ABC + 6],
      sum_fxyz[n * NUM_OF_ABC + 7]};
    get_f12_4body(d12, d12inv, fn, fnp, Fp[L_max * n_max_angular_plus_1 + n], s2, r12, f12);
  }

  if (L_max >= 2) {
    double s2[5];
    calculate_s_one<2>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s2);
    if (requires_grad) {
      accumulate_f12_one<2>(N, n, d12inv, fn_original, fnp_original, s2, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<2>(d12inv, fn_original, fnp_original, s2, r12unit, f12);
    }
  }

  if (L_max >= 3) {
    double s3[7];
    calculate_s_one<3>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s3);
    if (requires_grad) {
      accumulate_f12_one<3>(N, n, d12inv, fn_original, fnp_original, s3, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<3>(d12inv, fn_original, fnp_original, s3, r12unit, f12);
    }
  }

  if (L_max >= 4) {
    double s4[9];
    calculate_s_one<4>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s4);
    if (requires_grad) {
      accumulate_f12_one<4>(N, n, d12inv, fn_original, fnp_original, s4, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<4>(d12inv, fn_original, fnp_original, s4, r12unit, f12);
    }
  }

  if (L_max >= 5) {
    double s5[11];
    calculate_s_one<5>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s5);
    if (requires_grad) {
      accumulate_f12_one<5>(N, n, d12inv, fn_original, fnp_original, s5, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<5>(d12inv, fn_original, fnp_original, s5, r12unit, f12);
    }
  }

  if (L_max >= 6) {
    double s6[13];
    calculate_s_one<6>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s6);
    if (requires_grad) {
      accumulate_f12_one<6>(N, n, d12inv, fn_original, fnp_original, s6, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<6>(d12inv, fn_original, fnp_original, s6, r12unit, f12);
    }
  }

  if (L_max >= 7) {
    double s7[15];
    calculate_s_one<7>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s7);
    if (requires_grad) {
      accumulate_f12_one<7>(N, n, d12inv, fn_original, fnp_original, s7, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<7>(d12inv, fn_original, fnp_original, s7, r12unit, f12);
    }
  }

  if (L_max >= 8) {
    double s8[17];
    calculate_s_one<8>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, s8);
    if (requires_grad) {
      accumulate_f12_one<8>(N, n, d12inv, fn_original, fnp_original, s8, r12unit, r12, sum_s2xyz, sum_s2xyz123, f12);
    } else {
      accumulate_f12_one<8>(d12inv, fn_original, fnp_original, s8, r12unit, f12);
    }
  }
}


template <int L>
static __device__ __forceinline__ void
calculate_sc_one(
  const double x12,
  const double y12,
  const double z12,
  const double fn,
  double* s)
{
  int s_index = 0;
  double z_pow[L + 1] = {1.0};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = z12 * z_pow[n - 1];
  }
  double real_part = x12;
  double imag_part = y12;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    double z_factor = 0.0;
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
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* sum_fxyz,
  const double* sum_s2xyz,
  const double* sum_s2xyz123,
  const double* Fp,
  double* e_c,
  double* qp_c,
  double* qp_c123)
{
  const double d12inv = 1.0 / d12;
  const double r12unit[3] = {r12[0]*d12inv, r12[1]*d12inv, r12[2]*d12inv};
  if (L_max >= 1) {
    double s1[3];
    double sc1[3];
    double f[3] = {0.0};
    double f123[6] = {0.0};
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
    double s2[5];
    double sc2[5];
    double f[3] = {0.0};
    double f123[6] = {0.0};
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
    double s3[7];
    double sc3[7];
    double f[3] = {0.0};
    double f123[6] = {0.0};
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
    double s4[9];
    double sc4[9];
    double f[3] = {0.0};
    double f123[6] = {0.0};
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
    double s5[11];
    double sc5[11];
    double f[3] = {0.0};
    double f123[6] = {0.0};
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
    double s6[13];
    double sc6[13];
    double f[3] = {0.0};
    double f123[6] = {0.0};
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
    double s7[15];
    double sc7[15];
    double f[3] = {0.0};
    double f123[6] = {0.0};
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
    double s8[17];
    double sc8[17];
    double f[3] = {0.0};
    double f123[6] = {0.0};
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
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* sum_fxyz,
  const double* sum_s2xyz,
  const double* Fp,
  double* qp_c1,
  double* qp_c2)
{
  const double d12inv = 1.0 / d12;
  const double r12unit[3] = {r12[0]*d12inv, r12[1]*d12inv, r12[2]*d12inv};
  if (L_max >= 1) {
    double s1[3];
    double sc1[3];
    double f[3] = {0.0};
    calculate_sc_one<1>(r12unit[0], r12unit[1], r12unit[2], fn, sc1);
    calculate_fc_one<1>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc1, s1, qp_c2);
    accumulate_f12_one<1>(d12inv, fn, fnp, s1, r12unit, f);
    qp_c1[0] = f[0];
    qp_c1[1] = f[1];
    qp_c1[2] = f[2];
  }

  if (L_max >= 2) {
    double s2[5];
    double sc2[5];
    double f[3] = {0.0};
    calculate_sc_one<2>(r12unit[0], r12unit[1], r12unit[2], fn, sc2);
    calculate_fc_one<2>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc2, s2, qp_c2);
    accumulate_f12_one<2>(d12inv, fn, fnp, s2, r12unit, f);
    qp_c1[0] += f[0];
    qp_c1[1] += f[1];
    qp_c1[2] += f[2];
  }

  if (L_max >= 3) {
    double s3[7];
    double sc3[7];
    double f[3] = {0.0};
    calculate_sc_one<3>(r12unit[0], r12unit[1], r12unit[2], fn, sc3);
    calculate_fc_one<3>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc3, s3, qp_c2);
    accumulate_f12_one<3>(d12inv, fn, fnp, s3, r12unit, f);
    qp_c1[0] += f[0];
    qp_c1[1] += f[1];
    qp_c1[2] += f[2];
  }

  if (L_max >= 4) {
    double s4[9];
    double sc4[9];
    double f[3] = {0.0};
    calculate_sc_one<4>(r12unit[0], r12unit[1], r12unit[2], fn, sc4);
    calculate_fc_one<4>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc4, s4, qp_c2);
    accumulate_f12_one<4>(d12inv, fn, fnp, s4, r12unit, f);
    qp_c1[0] += f[0];
    qp_c1[1] += f[1];
    qp_c1[2] += f[2];
  }

  if (L_max >= 5) {
    double s5[11];
    double sc5[11];
    double f[3] = {0.0};
    calculate_sc_one<5>(r12unit[0], r12unit[1], r12unit[2], fn, sc5);
    calculate_fc_one<5>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc5, s5, qp_c2);
    accumulate_f12_one<5>(d12inv, fn, fnp, s5, r12unit, f);
    qp_c1[0] += f[0];
    qp_c1[1] += f[1];
    qp_c1[2] += f[2];
  }

  if (L_max >= 6) {
    double s6[13];
    double sc6[13];
    double f[3] = {0.0};
    calculate_sc_one<6>(r12unit[0], r12unit[1], r12unit[2], fn, sc6);
    calculate_fc_one<6>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc6, s6, qp_c2);
    accumulate_f12_one<6>(d12inv, fn, fnp, s6, r12unit, f);
    qp_c1[0] += f[0];
    qp_c1[1] += f[1];
    qp_c1[2] += f[2];
  }

  if (L_max >= 7) {
    double s7[15];
    double sc7[15];
    double f[3] = {0.0};
    calculate_sc_one<7>(r12unit[0], r12unit[1], r12unit[2], fn, sc7);
    calculate_fc_one<7>(N, n, n_max_angular_plus_1, Fp, sum_fxyz, sum_s2xyz, sc7, s7, qp_c2);
    accumulate_f12_one<7>(d12inv, fn, fnp, s7, r12unit, f);
    qp_c1[0] += f[0];
    qp_c1[1] += f[1];
    qp_c1[2] += f[2];
  }

  if (L_max >= 8) {
    double s8[17];
    double sc8[17];
    double f[3] = {0.0};
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
  const double d12,
  const double* r12,
  const double fn,
  const double* sum_fxyz,
  double* q_c)
{
  const double d12inv = 1.0 / d12;
  const double r12unit[3] = {r12[0]*d12inv, r12[1]*d12inv, r12[2]*d12inv};
  if (L == 1) {
    double sc1[3];
    calculate_sc_one<1>(r12unit[0], r12unit[1], r12unit[2], fn, sc1);
    calculate_qc_one<1>(N, n, n_max_angular_plus_1, sum_fxyz, sc1, q_c);
  }

  if (L == 2) {
    double sc2[5];
    calculate_sc_one<2>(r12unit[0], r12unit[1], r12unit[2], fn, sc2);
    calculate_qc_one<2>(N, n, n_max_angular_plus_1, sum_fxyz, sc2, q_c);
  }

  if (L == 3) {
    double sc3[7];
    calculate_sc_one<3>(r12unit[0], r12unit[1], r12unit[2], fn, sc3);
    calculate_qc_one<3>(N, n, n_max_angular_plus_1, sum_fxyz, sc3, q_c);
  }

  if (L == 4) {
    double sc4[9];
    calculate_sc_one<4>(r12unit[0], r12unit[1], r12unit[2], fn, sc4);
    calculate_qc_one<4>(N, n, n_max_angular_plus_1, sum_fxyz, sc4, q_c);
  }

  if (L == 5) {
    double sc5[11];
    calculate_sc_one<5>(r12unit[0], r12unit[1], r12unit[2], fn, sc5);
    calculate_qc_one<5>(N, n, n_max_angular_plus_1, sum_fxyz, sc5, q_c);
  }

  if (L == 6) {
    double sc6[13];
    calculate_sc_one<6>(r12unit[0], r12unit[1], r12unit[2], fn, sc6);
    calculate_qc_one<6>(N, n, n_max_angular_plus_1, sum_fxyz, sc6, q_c);
  }

  if (L == 7) {
    double sc7[15];
    calculate_sc_one<7>(r12unit[0], r12unit[1], r12unit[2], fn, sc7);
    calculate_qc_one<7>(N, n, n_max_angular_plus_1, sum_fxyz, sc7, q_c);
  }

  if (L == 8) {
    double sc8[17];
    calculate_sc_one<8>(r12unit[0], r12unit[1], r12unit[2], fn, sc8);
    calculate_qc_one<8>(N, n, n_max_angular_plus_1, sum_fxyz, sc8, q_c);
  }
}

template <int L>
static __device__ __forceinline__ void
accumulate_s_one(
  const double x12,
  const double y12,
  const double z12,
  const double fn,
  double* s)
{
  int s_index = L * L - 1;
  double z_pow[L + 1] = {1.0};
  for (int n = 1; n <= L; ++n) {
    z_pow[n] = z12 * z_pow[n - 1];
  }
  double real_part = x12;
  double imag_part = y12;
  for (int n1 = 0; n1 <= L; ++n1) {
    int n2_start = (L + n1) % 2 == 0 ? 0 : 1;
    double z_factor = 0.0;
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
accumulate_s(const int L_max, const double d12, double x12, double y12, double z12, const double fn, double* s)
{
  double d12inv = 1.0 / d12;
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
static __device__ __forceinline__ double find_q_one(const double* s)
{
  const int start_index = L * L-1;
  const int num_terms = 2 * L + 1;
  double q = 0.0;
  for (int k = 1; k < num_terms; ++k) {
    q += C3B[start_index + k] * s[start_index + k] * s[start_index + k];
  }
  q *= 2.0;
  q += C3B[start_index] * s[start_index] * s[start_index];
  return q;
}

static __device__ __forceinline__ void
find_q(
  const int L_max, 
  const int num_L, 
  const int n_max_angular_plus_1, 
  const int n, 
  const double* s, 
  double* q)
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
  if (num_L >= L_max + 1) {
    q[L_max * n_max_angular_plus_1 + n] =
      C4B[0] * s[3] * s[3] * s[3] + C4B[1] * s[3] * (s[4] * s[4] + s[5] * s[5]) +
      C4B[2] * s[3] * (s[6] * s[6] + s[7] * s[7]) + C4B[3] * s[6] * (s[5] * s[5] - s[4] * s[4]) +
      C4B[4] * s[4] * s[5] * s[7];
  }
  if (num_L >= L_max + 2) {
    double s0_sq = s[0] * s[0];
    double s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];
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
