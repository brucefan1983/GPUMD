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

#pragma once

const int NUM_OF_ABC = 24; // 3 + 5 + 7 + 9 for L_max = 4
__constant__ double C3B[NUM_OF_ABC] = {
  0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435,
  0.596831036594608, 0.596831036594608, 0.149207759148652, 0.149207759148652,
  0.139260575205408, 0.104445431404056, 0.104445431404056, 1.044454314040563,
  1.044454314040563, 0.174075719006761, 0.174075719006761, 0.011190581936149,
  0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
  1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606};
__constant__ double C4B[5] = {
  -0.007499480826664,
  -0.134990654879954,
  0.067495327439977,
  0.404971964639861,
  -0.809943929279723};
__constant__ double C5B[3] = {0.026596810706114, 0.053193621412227, 0.026596810706114};
// __constant__ double PI = 3.141592653589793;
// __constant__ double PI_HALF = 1.570796326794897;
const int SIZE_BOX_AND_INVERSE_BOX = 18; // (3 * 3) * 2
const int MAX_NUM_N = 20;                // n_max+1 = 19+1
const int MAX_DIM = MAX_NUM_N * 7;
const int MAX_DIM_ANGULAR = MAX_NUM_N * 6;

static __device__ void apply_ann_one_layer(
  const int N_des,
  const int N_neu,
  const double* w0,
  const double* b0,
  const double* w1,
  const double* b1,
  double* q,
  double& energy,
  double* energy_derivative)
{
  for (int n = 0; n < N_neu; ++n) {
    double w0_times_q = 0.0;
    for (int d = 0; d < N_des; ++d) {
      w0_times_q += w0[n * N_des + d] * q[d];
    }
    double x1 = tanh(w0_times_q - b0[n]);
    double tanh_der = 1.0 - x1 * x1;
    energy += w1[n] * x1;
    for (int d = 0; d < N_des; ++d) {
      double y1 = tanh_der * w0[n * N_des + d];
      energy_derivative[d] += w1[n] * y1;
    }
  }
  energy -= b1[0];
}

static __device__ __forceinline__ void find_fc(double rc, double rcinv, double d12, double& fc)
{
  if (d12 < rc) {
    double x = d12 * rcinv;
    fc = 0.5 * cos(3.141592653589793 * x) + 0.5;
  } else {
    fc = 0.0;
  }
}

static __device__ __host__ __forceinline__ void
find_fc_and_fcp(double rc, double rcinv, double d12, double& fc, double& fcp)
{
  if (d12 < rc) {
    double x = d12 * rcinv;
    fc = 0.5 * cos(3.141592653589793 * x) + 0.5;
    fcp = -1.570796326794897 * sin(3.141592653589793 * x);
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
    double pi_factor = 3.141592653589793 / (r2 - r1);
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
  fn[0] = 1.0;
  fn[1] = x;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0 * x * fn[m - 1] - fn[m - 2];
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0) * 0.5 * fc12;
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
  double x = 2.0 * (d12 * rcinv - 1.0) * (d12 * rcinv - 1.0) - 1.0;
  fn[0] = 1.0;
  fnp[0] = 0.0;
  fn[1] = x;
  fnp[1] = 1.0;
  double u0 = 1.0;
  double u1 = 2.0 * x;
  double u2;
  for (int m = 2; m <= n_max; ++m) {
    fn[m] = 2.0 * x * fn[m - 1] - fn[m - 2];
    fnp[m] = m * u1;
    u2 = 2.0 * x * u1 - u0;
    u0 = u1;
    u1 = u2;
  }
  for (int m = 0; m <= n_max; ++m) {
    fn[m] = (fn[m] + 1.0) * 0.5;
    fnp[m] *= 2.0 * (d12 * rcinv - 1.0) * rcinv;
    fnp[m] = fnp[m] * fc12 + fn[m] * fcp12;
    fn[m] *= fc12;
  }
}

static __device__ __forceinline__ void get_f12_1(
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12)
{
  double tmp = s[1] * r12[0];
  tmp += s[2] * r12[1];
  tmp *= 2.0;
  tmp += s[0] * r12[2];
  tmp *= Fp * fnp * d12inv * 2.0;
  for (int d = 0; d < 3; ++d) {
    f12[d] += tmp * r12[d];
  }
  tmp = Fp * fn * 2.0;
  f12[0] += tmp * 2.0 * s[1];
  f12[1] += tmp * 2.0 * s[2];
  f12[2] += tmp * s[0];
}

static __device__ __forceinline__ void get_f12_2(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12)
{
  double tmp = s[1] * r12[0] * r12[2];                // Re[Y21]
  tmp += s[2] * r12[1] * r12[2];                     // Im[Y21]
  tmp += s[3] * (r12[0] * r12[0] - r12[1] * r12[1]); // Re[Y22]
  tmp += s[4] * 2.0 * r12[0] * r12[1];              // Im[Y22]
  tmp *= 2.0;
  tmp += s[0] * (3.0 * r12[2] * r12[2] - d12 * d12); // Y20
  tmp *= Fp * fnp * d12inv * 2.0;
  for (int d = 0; d < 3; ++d) {
    f12[d] += tmp * r12[d];
  }
  tmp = Fp * fn * 4.0;
  f12[0] += tmp * (-s[0] * r12[0] + s[1] * r12[2] + 2.0 * s[3] * r12[0] + 2.0 * s[4] * r12[1]);
  f12[1] += tmp * (-s[0] * r12[1] + s[2] * r12[2] - 2.0 * s[3] * r12[1] + 2.0 * s[4] * r12[0]);
  f12[2] += tmp * (2.0 * s[0] * r12[2] + s[1] * r12[0] + s[2] * r12[1]);
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
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * 4.0 * r12[2];

  // derivative wrt s[1]
  tmp0 = C4B[1] * s[0] * s[1] * 2.0 - C4B[3] * s[3] * s[1] * 2.0 + C4B[4] * s[2] * s[4];
  tmp1 = tmp0 * r12[0] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * r12[2];
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[0];

  // derivative wrt s[2]
  tmp0 = C4B[1] * s[0] * s[2] * 2.0 + C4B[3] * s[3] * s[2] * 2.0 + C4B[4] * s[1] * s[4];
  tmp1 = tmp0 * r12[1] * r12[2] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2 * r12[2];
  f12[2] += tmp1 * r12[2] + tmp2 * r12[1];

  // derivative wrt s[3]
  tmp0 = C4B[2] * s[0] * s[3] * 2.0 + C4B[3] * (s[2] * s[2] - s[1] * s[1]);
  tmp1 = tmp0 * (r12[0] * r12[0] - r12[1] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0 * r12[0];
  f12[1] += tmp1 * r12[1] - tmp2 * 2.0 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s[4]
  tmp0 = C4B[2] * s[0] * s[4] * 2.0 + C4B[4] * s[1] * s[2];
  tmp1 = tmp0 * (2.0 * r12[0] * r12[1]) * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2 * 2.0 * r12[1];
  f12[1] += tmp1 * r12[1] + tmp2 * 2.0 * r12[0];
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
  tmp0 = C5B[1] * s[0] * s[0] * s[1] * 2.0 + C5B[2] * s1_sq_plus_s2_sq * s[1] * 4.0;
  tmp1 = tmp0 * r12[0] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0] + tmp2;
  f12[1] += tmp1 * r12[1];
  f12[2] += tmp1 * r12[2];

  // derivative wrt s[2]
  tmp0 = C5B[1] * s[0] * s[0] * s[2] * 2.0 + C5B[2] * s1_sq_plus_s2_sq * s[2] * 4.0;
  tmp1 = tmp0 * r12[1] * fnp_factor;
  tmp2 = tmp0 * fn_factor;
  f12[0] += tmp1 * r12[0];
  f12[1] += tmp1 * r12[1] + tmp2;
  f12[2] += tmp1 * r12[2];
}

static __device__ __forceinline__ void get_f12_3(
  const double d12,
  const double d12inv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  const double* r12,
  double* f12)
{
  double d12sq = d12 * d12;
  double x2 = r12[0] * r12[0];
  double y2 = r12[1] * r12[1];
  double z2 = r12[2] * r12[2];
  double xy = r12[0] * r12[1];
  double xz = r12[0] * r12[2];
  double yz = r12[1] * r12[2];

  double tmp = s[1] * (5.0 * z2 - d12sq) * r12[0];
  tmp += s[2] * (5.0 * z2 - d12sq) * r12[1];
  tmp += s[3] * (x2 - y2) * r12[2];
  tmp += s[4] * 2.0 * xy * r12[2];
  tmp += s[5] * r12[0] * (x2 - 3.0 * y2);
  tmp += s[6] * r12[1] * (3.0 * x2 - y2);
  tmp *= 2.0;
  tmp += s[0] * (5.0 * z2 - 3.0 * d12sq) * r12[2];
  tmp *= Fp * fnp * d12inv * 2.0;
  for (int d = 0; d < 3; ++d) {
    f12[d] += tmp * r12[d];
  }

  // x
  tmp = s[1] * (4.0 * z2 - 3.0 * x2 - y2);
  tmp += s[2] * (-2.0 * xy);
  tmp += s[3] * 2.0 * xz;
  tmp += s[4] * (2.0 * yz);
  tmp += s[5] * (3.0 * (x2 - y2));
  tmp += s[6] * (6.0 * xy);
  tmp *= 2.0;
  tmp += s[0] * (-6.0 * xz);
  f12[0] += tmp * Fp * fn * 2.0;
  // y
  tmp = s[1] * (-2.0 * xy);
  tmp += s[2] * (4.0 * z2 - 3.0 * y2 - x2);
  tmp += s[3] * (-2.0 * yz);
  tmp += s[4] * (2.0 * xz);
  tmp += s[5] * (-6.0 * xy);
  tmp += s[6] * (3.0 * (x2 - y2));
  tmp *= 2.0;
  tmp += s[0] * (-6.0 * yz);
  f12[1] += tmp * Fp * fn * 2.0;
  // z
  tmp = s[1] * (8.0 * xz);
  tmp += s[2] * (8.0 * yz);
  tmp += s[3] * (x2 - y2);
  tmp += s[4] * (2.0 * xy);
  tmp *= 2.0;
  tmp += s[0] * (9.0 * z2 - 3.0 * d12sq);
  f12[2] += tmp * Fp * fn * 2.0;
}

static __device__ __forceinline__ void get_f12_4(
  const double x,
  const double y,
  const double z,
  const double r,
  const double rinv,
  const double fn,
  const double fnp,
  const double Fp,
  const double* s,
  double* f12)
{
  const double r2 = r * r;
  const double x2 = x * x;
  const double y2 = y * y;
  const double z2 = z * z;
  const double xy = x * y;
  const double xz = x * z;
  const double yz = y * z;
  const double xyz = x * yz;
  const double x2my2 = x2 - y2;

  double tmp = s[1] * (7.0 * z2 - 3.0 * r2) * xz; // Y41_real
  tmp += s[2] * (7.0 * z2 - 3.0 * r2) * yz;      // Y41_imag
  tmp += s[3] * (7.0 * z2 - r2) * x2my2;          // Y42_real
  tmp += s[4] * (7.0 * z2 - r2) * 2.0 * xy;      // Y42_imag
  tmp += s[5] * (x2 - 3.0 * y2) * xz;             // Y43_real
  tmp += s[6] * (3.0 * x2 - y2) * yz;             // Y43_imag
  tmp += s[7] * (x2my2 * x2my2 - 4.0 * x2 * y2);  // Y44_real
  tmp += s[8] * (4.0 * xy * x2my2);               // Y44_imag
  tmp *= 2.0;
  tmp += s[0] * ((35.0 * z2 - 30.0 * r2) * z2 + 3.0 * r2 * r2); // Y40
  tmp *= Fp * fnp * rinv * 2.0;
  f12[0] += tmp * x;
  f12[1] += tmp * y;
  f12[2] += tmp * z;

  // x
  tmp = s[1] * z * (7.0 * z2 - 3.0 * r2 - 6.0 * x2);  // Y41_real
  tmp += s[2] * (-6.0 * xyz);                           // Y41_imag
  tmp += s[3] * 4.0 * x * (3.0 * z2 - x2);             // Y42_real
  tmp += s[4] * 2.0 * y * (7.0 * z2 - r2 - 2.0 * x2); // Y42_imag
  tmp += s[5] * 3.0 * z * x2my2;                        // Y43_real
  tmp += s[6] * 6.0 * xyz;                              // Y43_imag
  tmp += s[7] * 4.0 * x * (x2 - 3.0 * y2);             // Y44_real
  tmp += s[8] * 4.0 * y * (3.0 * x2 - y2);             // Y44_imag
  tmp *= 2.0;
  tmp += s[0] * 12.0 * x * (r2 - 5.0 * z2); // Y40
  f12[0] += tmp * Fp * fn * 2.0;
  // y
  tmp = s[1] * (-6.0 * xyz);                            // Y41_real
  tmp += s[2] * z * (7.0 * z2 - 3.0 * r2 - 6.0 * y2); // Y41_imag
  tmp += s[3] * 4.0 * y * (y2 - 3.0 * z2);             // Y42_real
  tmp += s[4] * 2.0 * x * (7.0 * z2 - r2 - 2.0 * y2); // Y42_imag
  tmp += s[5] * (-6.0 * xyz);                           // Y43_real
  tmp += s[6] * 3.0 * z * x2my2;                        // Y43_imag
  tmp += s[7] * 4.0 * y * (y2 - 3.0 * x2);             // Y44_real
  tmp += s[8] * 4.0 * x * (x2 - 3.0 * y2);             // Y44_imag
  tmp *= 2.0;
  tmp += s[0] * 12.0 * y * (r2 - 5.0 * z2); // Y40
  f12[1] += tmp * Fp * fn * 2.0;
  // z
  tmp = s[1] * 3.0 * x * (5.0 * z2 - r2);  // Y41_real
  tmp += s[2] * 3.0 * y * (5.0 * z2 - r2); // Y41_imag
  tmp += s[3] * 12.0 * z * x2my2;           // Y42_real
  tmp += s[4] * 24.0 * xyz;                 // Y42_imag
  tmp += s[5] * x * (x2 - 3.0 * y2);        // Y43_real
  tmp += s[6] * y * (3.0 * x2 - y2);        // Y43_imag
  tmp *= 2.0;
  tmp += s[0] * 16.0 * z * (5.0 * z2 - 3.0 * r2); // Y40
  f12[2] += tmp * Fp * fn * 2.0;
}

static __device__ __forceinline__ void accumulate_f12(
  const int n,
  const int n_max_angular_plus_1,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  double* f12)
{
  const double d12inv = 1.0 / d12;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0] * C3B[0],
    sum_fxyz[n * NUM_OF_ABC + 1] * C3B[1],
    sum_fxyz[n * NUM_OF_ABC + 2] * C3B[2]};
  get_f12_1(d12inv, fn, fnp, Fp[n], s1, r12, f12);
  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3] * C3B[3],
    sum_fxyz[n * NUM_OF_ABC + 4] * C3B[4],
    sum_fxyz[n * NUM_OF_ABC + 5] * C3B[5],
    sum_fxyz[n * NUM_OF_ABC + 6] * C3B[6],
    sum_fxyz[n * NUM_OF_ABC + 7] * C3B[7]};
  get_f12_2(d12, d12inv, fn, fnp, Fp[n_max_angular_plus_1 + n], s2, r12, f12);
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],
    sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
    sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10],
    sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
    sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12],
    sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
    sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  get_f12_3(d12, d12inv, fn, fnp, Fp[2 * n_max_angular_plus_1 + n], s3, r12, f12);
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s4[9] = {
    sum_fxyz[n * NUM_OF_ABC + 15] * C3B[15],
    sum_fxyz[n * NUM_OF_ABC + 16] * C3B[16],
    sum_fxyz[n * NUM_OF_ABC + 17] * C3B[17],
    sum_fxyz[n * NUM_OF_ABC + 18] * C3B[18],
    sum_fxyz[n * NUM_OF_ABC + 19] * C3B[19],
    sum_fxyz[n * NUM_OF_ABC + 20] * C3B[20],
    sum_fxyz[n * NUM_OF_ABC + 21] * C3B[21],
    sum_fxyz[n * NUM_OF_ABC + 22] * C3B[22],
    sum_fxyz[n * NUM_OF_ABC + 23] * C3B[23]};
  get_f12_4(
    r12[0], r12[1], r12[2], d12, d12inv, fn, fnp, Fp[3 * n_max_angular_plus_1 + n], s4, f12);
}

static __device__ __forceinline__ void accumulate_f12_with_4body(
  const int n,
  const int n_max_angular_plus_1,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  double* f12)
{
  const double d12inv = 1.0 / d12;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0] * C3B[0],
    sum_fxyz[n * NUM_OF_ABC + 1] * C3B[1],
    sum_fxyz[n * NUM_OF_ABC + 2] * C3B[2]};
  get_f12_1(d12inv, fn, fnp, Fp[n], s1, r12, f12);
  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3],
    sum_fxyz[n * NUM_OF_ABC + 4],
    sum_fxyz[n * NUM_OF_ABC + 5],
    sum_fxyz[n * NUM_OF_ABC + 6],
    sum_fxyz[n * NUM_OF_ABC + 7]};
  get_f12_4body(d12, d12inv, fn, fnp, Fp[4 * n_max_angular_plus_1 + n], s2, r12, f12);
  s2[0] *= C3B[3];
  s2[1] *= C3B[4];
  s2[2] *= C3B[5];
  s2[3] *= C3B[6];
  s2[4] *= C3B[7];
  get_f12_2(d12, d12inv, fn, fnp, Fp[n_max_angular_plus_1 + n], s2, r12, f12);
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],
    sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
    sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10],
    sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
    sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12],
    sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
    sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  get_f12_3(d12, d12inv, fn, fnp, Fp[2 * n_max_angular_plus_1 + n], s3, r12, f12);
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s4[9] = {
    sum_fxyz[n * NUM_OF_ABC + 15] * C3B[15],
    sum_fxyz[n * NUM_OF_ABC + 16] * C3B[16],
    sum_fxyz[n * NUM_OF_ABC + 17] * C3B[17],
    sum_fxyz[n * NUM_OF_ABC + 18] * C3B[18],
    sum_fxyz[n * NUM_OF_ABC + 19] * C3B[19],
    sum_fxyz[n * NUM_OF_ABC + 20] * C3B[20],
    sum_fxyz[n * NUM_OF_ABC + 21] * C3B[21],
    sum_fxyz[n * NUM_OF_ABC + 22] * C3B[22],
    sum_fxyz[n * NUM_OF_ABC + 23] * C3B[23]};
  get_f12_4(
    r12[0], r12[1], r12[2], d12, d12inv, fn, fnp, Fp[3 * n_max_angular_plus_1 + n], s4, f12);
}

static __device__ __forceinline__ void accumulate_f12_with_5body(
  const int n,
  const int n_max_angular_plus_1,
  const double d12,
  const double* r12,
  double fn,
  double fnp,
  const double* Fp,
  const double* sum_fxyz,
  double* f12)
{
  const double d12inv = 1.0 / d12;
  // l = 1
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s1[3] = {
    sum_fxyz[n * NUM_OF_ABC + 0], sum_fxyz[n * NUM_OF_ABC + 1], sum_fxyz[n * NUM_OF_ABC + 2]};
  get_f12_5body(d12, d12inv, fn, fnp, Fp[5 * n_max_angular_plus_1 + n], s1, r12, f12);
  s1[0] *= C3B[0];
  s1[1] *= C3B[1];
  s1[2] *= C3B[2];
  get_f12_1(d12inv, fn, fnp, Fp[n], s1, r12, f12);
  // l = 2
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s2[5] = {
    sum_fxyz[n * NUM_OF_ABC + 3],
    sum_fxyz[n * NUM_OF_ABC + 4],
    sum_fxyz[n * NUM_OF_ABC + 5],
    sum_fxyz[n * NUM_OF_ABC + 6],
    sum_fxyz[n * NUM_OF_ABC + 7]};
  get_f12_4body(d12, d12inv, fn, fnp, Fp[4 * n_max_angular_plus_1 + n], s2, r12, f12);
  s2[0] *= C3B[3];
  s2[1] *= C3B[4];
  s2[2] *= C3B[5];
  s2[3] *= C3B[6];
  s2[4] *= C3B[7];
  get_f12_2(d12, d12inv, fn, fnp, Fp[n_max_angular_plus_1 + n], s2, r12, f12);
  // l = 3
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s3[7] = {
    sum_fxyz[n * NUM_OF_ABC + 8] * C3B[8],
    sum_fxyz[n * NUM_OF_ABC + 9] * C3B[9],
    sum_fxyz[n * NUM_OF_ABC + 10] * C3B[10],
    sum_fxyz[n * NUM_OF_ABC + 11] * C3B[11],
    sum_fxyz[n * NUM_OF_ABC + 12] * C3B[12],
    sum_fxyz[n * NUM_OF_ABC + 13] * C3B[13],
    sum_fxyz[n * NUM_OF_ABC + 14] * C3B[14]};
  get_f12_3(d12, d12inv, fn, fnp, Fp[2 * n_max_angular_plus_1 + n], s3, r12, f12);
  // l = 4
  fnp = fnp * d12inv - fn * d12inv * d12inv;
  fn = fn * d12inv;
  double s4[9] = {
    sum_fxyz[n * NUM_OF_ABC + 15] * C3B[15],
    sum_fxyz[n * NUM_OF_ABC + 16] * C3B[16],
    sum_fxyz[n * NUM_OF_ABC + 17] * C3B[17],
    sum_fxyz[n * NUM_OF_ABC + 18] * C3B[18],
    sum_fxyz[n * NUM_OF_ABC + 19] * C3B[19],
    sum_fxyz[n * NUM_OF_ABC + 20] * C3B[20],
    sum_fxyz[n * NUM_OF_ABC + 21] * C3B[21],
    sum_fxyz[n * NUM_OF_ABC + 22] * C3B[22],
    sum_fxyz[n * NUM_OF_ABC + 23] * C3B[23]};
  get_f12_4(
    r12[0], r12[1], r12[2], d12, d12inv, fn, fnp, Fp[3 * n_max_angular_plus_1 + n], s4, f12);
}

static __device__ __forceinline__ void
accumulate_s(const double d12, double x12, double y12, double z12, const double fn, double* s)
{
  double d12inv = 1.0 / d12;
  x12 *= d12inv;
  y12 *= d12inv;
  z12 *= d12inv;
  double x12sq = x12 * x12;
  double y12sq = y12 * y12;
  double z12sq = z12 * z12;
  double x12sq_minus_y12sq = x12sq - y12sq;
  s[0] += z12 * fn;                                                             // Y10
  s[1] += x12 * fn;                                                             // Y11_real
  s[2] += y12 * fn;                                                             // Y11_imag
  s[3] += (3.0 * z12sq - 1.0) * fn;                                           // Y20
  s[4] += x12 * z12 * fn;                                                       // Y21_real
  s[5] += y12 * z12 * fn;                                                       // Y21_imag
  s[6] += x12sq_minus_y12sq * fn;                                               // Y22_real
  s[7] += 2.0 * x12 * y12 * fn;                                                // Y22_imag
  s[8] += (5.0 * z12sq - 3.0) * z12 * fn;                                     // Y30
  s[9] += (5.0 * z12sq - 1.0) * x12 * fn;                                     // Y31_real
  s[10] += (5.0 * z12sq - 1.0) * y12 * fn;                                    // Y31_imag
  s[11] += x12sq_minus_y12sq * z12 * fn;                                        // Y32_real
  s[12] += 2.0 * x12 * y12 * z12 * fn;                                         // Y32_imag
  s[13] += (x12 * x12 - 3.0 * y12 * y12) * x12 * fn;                           // Y33_real
  s[14] += (3.0 * x12 * x12 - y12 * y12) * y12 * fn;                           // Y33_imag
  s[15] += ((35.0 * z12sq - 30.0) * z12sq + 3.0) * fn;                       // Y40
  s[16] += (7.0 * z12sq - 3.0) * x12 * z12 * fn;                              // Y41_real
  s[17] += (7.0 * z12sq - 3.0) * y12 * z12 * fn;                              // Y41_iamg
  s[18] += (7.0 * z12sq - 1.0) * x12sq_minus_y12sq * fn;                      // Y42_real
  s[19] += (7.0 * z12sq - 1.0) * x12 * y12 * 2.0 * fn;                       // Y42_imag
  s[20] += (x12sq - 3.0 * y12sq) * x12 * z12 * fn;                             // Y43_real
  s[21] += (3.0 * x12sq - y12sq) * y12 * z12 * fn;                             // Y43_imag
  s[22] += (x12sq_minus_y12sq * x12sq_minus_y12sq - 4.0 * x12sq * y12sq) * fn; // Y44_real
  s[23] += (4.0 * x12 * y12 * x12sq_minus_y12sq) * fn;                         // Y44_imag
}

static __device__ __forceinline__ void
find_q(const int n_max_angular_plus_1, const int n, const double* s, double* q)
{
  q[n] = C3B[0] * s[0] * s[0] + 2.0 * (C3B[1] * s[1] * s[1] + C3B[2] * s[2] * s[2]);
  q[n_max_angular_plus_1 + n] =
    C3B[3] * s[3] * s[3] + 2.0 * (C3B[4] * s[4] * s[4] + C3B[5] * s[5] * s[5] +
                                   C3B[6] * s[6] * s[6] + C3B[7] * s[7] * s[7]);
  q[2 * n_max_angular_plus_1 + n] =
    C3B[8] * s[8] * s[8] +
    2.0 * (C3B[9] * s[9] * s[9] + C3B[10] * s[10] * s[10] + C3B[11] * s[11] * s[11] +
            C3B[12] * s[12] * s[12] + C3B[13] * s[13] * s[13] + C3B[14] * s[14] * s[14]);
  q[3 * n_max_angular_plus_1 + n] =
    C3B[15] * s[15] * s[15] +
    2.0 * (C3B[16] * s[16] * s[16] + C3B[17] * s[17] * s[17] + C3B[18] * s[18] * s[18] +
            C3B[19] * s[19] * s[19] + C3B[20] * s[20] * s[20] + C3B[21] * s[21] * s[21] +
            C3B[22] * s[22] * s[22] + C3B[23] * s[23] * s[23]);
}

static __device__ __forceinline__ void
find_q_with_4body(const int n_max_angular_plus_1, const int n, const double* s, double* q)
{
  find_q(n_max_angular_plus_1, n, s, q);
  q[4 * n_max_angular_plus_1 + n] =
    C4B[0] * s[3] * s[3] * s[3] + C4B[1] * s[3] * (s[4] * s[4] + s[5] * s[5]) +
    C4B[2] * s[3] * (s[6] * s[6] + s[7] * s[7]) + C4B[3] * s[6] * (s[5] * s[5] - s[4] * s[4]) +
    C4B[4] * s[4] * s[5] * s[7];
}

static __device__ __forceinline__ void
find_q_with_5body(const int n_max_angular_plus_1, const int n, const double* s, double* q)
{
  find_q_with_4body(n_max_angular_plus_1, n, s, q);
  double s0_sq = s[0] * s[0];
  double s1_sq_plus_s2_sq = s[1] * s[1] + s[2] * s[2];
  q[5 * n_max_angular_plus_1 + n] = C5B[0] * s0_sq * s0_sq + C5B[1] * s0_sq * s1_sq_plus_s2_sq +
                                    C5B[2] * s1_sq_plus_s2_sq * s1_sq_plus_s2_sq;
}

#ifdef USE_TABLE
namespace
{
const int table_length = 2001;
const int table_segments = table_length - 1;
const double table_resolution = 0.0005;

__device__ void find_index_and_weight(
  const double d12_reduced,
  int& index_left,
  int& index_right,
  double& weight_left,
  double& weight_right)
{
  double d12_index = d12_reduced * table_segments;
  index_left = int(d12_index);
  if (index_left == table_segments) {
    --index_left;
  }
  index_right = index_left + 1;
  weight_right = d12_index - index_left;
  weight_left = 1.0 - weight_right;
}

static void construct_table_radial_or_angular(
  const int version,
  const int num_types,
  const int num_types_sq,
  const int n_max,
  const int basis_size,
  const double rc,
  const double rcinv,
  const double* c,
  double* gn,
  double* gnp)
{
  for (int table_index = 0; table_index < table_length; ++table_index) {
    double d12 = table_index * table_resolution * rc;
    double fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    for (int t1 = 0; t1 < num_types; ++t1) {
      for (int t2 = 0; t2 < num_types; ++t2) {
        int t12 = t1 * num_types + t2;
        double fn12[MAX_NUM_N];
        double fnp12[MAX_NUM_N];
        if (version == 2) {
          find_fn_and_fnp(n_max, rcinv, d12, fc12, fcp12, fn12, fnp12);
          for (int n = 0; n <= n_max; ++n) {
            int index_all = (table_index * num_types_sq + t12) * (n_max + 1) + n;
            gn[index_all] = fn12[n] * ((num_types == 1) ? 1.0 : c[n * num_types_sq + t12]);
            gnp[index_all] = fnp12[n] * ((num_types == 1) ? 1.0 : c[n * num_types_sq + t12]);
          }
        } else {
          find_fn_and_fnp(basis_size, rcinv, d12, fc12, fcp12, fn12, fnp12);
          for (int n = 0; n <= n_max; ++n) {
            double gn12 = 0.0;
            double gnp12 = 0.0;
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
}
} // namespace
#endif
