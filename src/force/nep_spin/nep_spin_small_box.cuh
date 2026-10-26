/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once

#include "force/nep_small_box.cuh"

// Spin retains double displacement and force accumulation in small periodic cells.
// Ordinary NEP uses its unchanged small-box kernels.
namespace nep_spin_small_box {
template <int L>
static __device__ __forceinline__ void accumulate_one_angular_channel_in_double(
  const int n,
  const int n_max_angular_plus_1,
  const double d12inv,
  const double fn,
  const double fnp,
  const float* Fp,
  const float* sum_fxyz,
  const double* r12unit,
  double* f12)
{
  double s[2 * L + 1];
  double contribution[3] = {0.0, 0.0, 0.0};
  calculate_s_one<L>(n, n_max_angular_plus_1, Fp, sum_fxyz, s);
  accumulate_f12_one<L>(d12inv, fn, fnp, s, r12unit, contribution);
  for (int d = 0; d < 3; ++d) {
    f12[d] += static_cast<double>(contribution[d]);
  }
}
static __device__ __forceinline__ void find_fn_and_fnp_in_double(
  const int n_max,
  const double rc,
  const double distance,
  double* fn,
  double* fnp)
{
  const double rcinv = 1.0 / rc;
  const double scaled = distance * rcinv;
  const double fc =
    distance < rc ? 0.5 * cos(3.14159265358979323846 * scaled) + 0.5 : 0.0;
  const double fcp =
    distance < rc
      ? -1.57079632679489661923 * sin(3.14159265358979323846 * scaled) * rcinv
      : 0.0;
  const double x = 2.0 * (scaled - 1.0) * (scaled - 1.0) - 1.0;
  fn[0] = fc;
  fnp[0] = fcp;
  fn[1] = (x + 1.0) * 0.5 * fc;
  fnp[1] =
    2.0 * (scaled - 1.0) * rcinv * fc + (x + 1.0) * 0.5 * fcp;
  double u0 = 1.0;
  double u1 = 2.0 * x;
  double t0 = 1.0;
  double t1 = x;
  for (int m = 2; m <= n_max; ++m) {
    const double t2 = 2.0 * x * t1 - t0;
    t0 = t1;
    t1 = t2;
    const double derivative_chebyshev = m * u1;
    const double u2 = 2.0 * x * u1 - u0;
    u0 = u1;
    u1 = u2;
    const double basis = (t2 + 1.0) * 0.5;
    fnp[m] =
      derivative_chebyshev * 2.0 * (scaled - 1.0) * rcinv * fc + basis * fcp;
    fn[m] = basis * fc;
  }
}
static __device__ __forceinline__ void accumulate_f12_small_box_in_double(
  const int L_max,
  const int has_q_222,
  const int has_q_1111,
  const int has_q_112,
  const int has_q_123,
  const int has_q_233,
  const int has_q_134,
  const int num_L,
  const int n,
  const int n_max_angular_plus_1,
  const double d12,
  const double* r12,
  const double fn,
  const double fnp,
  const float* Fp,
  const float* sum_fxyz,
  double* f12)
{
  const bool only_supported_extra =
    !has_q_1111 && !has_q_112 && !has_q_123 && !has_q_233 && !has_q_134 &&
    num_L == L_max + static_cast<int>(has_q_222);
  if (!only_supported_extra) {
    float fallback[3] = {0.0f};
    const float d12_float = static_cast<float>(d12);
    const float r12_float[3] = {
      static_cast<float>(r12[0]),
      static_cast<float>(r12[1]),
      static_cast<float>(r12[2])};
    accumulate_f12(
      L_max,
      has_q_222,
      has_q_1111,
      has_q_112,
      has_q_123,
      has_q_233,
      has_q_134,
      num_L,
      n,
      n_max_angular_plus_1,
      d12_float,
      r12_float,
      static_cast<float>(fn),
      static_cast<float>(fnp),
      Fp,
      sum_fxyz,
      fallback);
    for (int d = 0; d < 3; ++d) {
      f12[d] += static_cast<double>(fallback[d]);
    }
    return;
  }
  const double d12inv = 1.0 / d12;
  const double r12unit[3] = {
    r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};
  if (L_max >= 1)
    accumulate_one_angular_channel_in_double<1>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (L_max >= 2)
    accumulate_one_angular_channel_in_double<2>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (L_max >= 3)
    accumulate_one_angular_channel_in_double<3>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (L_max >= 4)
    accumulate_one_angular_channel_in_double<4>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (L_max >= 5)
    accumulate_one_angular_channel_in_double<5>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (L_max >= 6)
    accumulate_one_angular_channel_in_double<6>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (L_max >= 7)
    accumulate_one_angular_channel_in_double<7>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (L_max >= 8)
    accumulate_one_angular_channel_in_double<8>(
      n, n_max_angular_plus_1, d12inv, fn, fnp, Fp, sum_fxyz, r12unit, f12);
  if (has_q_222) {
    double radial_fn = fn * d12inv;
    double radial_fnp = fnp * d12inv - fn * d12inv * d12inv;
    const double fn2 = radial_fn * d12inv;
    const double fnp2 =
      radial_fnp * d12inv - radial_fn * d12inv * d12inv;
    const double s2[5] = {
      sum_fxyz[n * NUM_OF_ABC + 3],
      sum_fxyz[n * NUM_OF_ABC + 4],
      sum_fxyz[n * NUM_OF_ABC + 5],
      sum_fxyz[n * NUM_OF_ABC + 6],
      sum_fxyz[n * NUM_OF_ABC + 7]};
    double contribution[3] = {0.0, 0.0, 0.0};
    get_f12_4body(
      d12,
      d12inv,
      fn2,
      fnp2,
      static_cast<double>(Fp[L_max * n_max_angular_plus_1 + n]),
      s2,
      r12,
      contribution);
    for (int d = 0; d < 3; ++d) {
      f12[d] += static_cast<double>(contribution[d]);
    }
  }
}
static __global__ void find_force_radial_small_box(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x12,
  const double* __restrict__ g_y12,
  const double* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      double r12_double[3] = {
        static_cast<double>(g_x12[index]),
        static_cast<double>(g_y12[index]),
        static_cast<double>(g_z12[index])};
      const double d12_double = sqrt(
        r12_double[0] * r12_double[0] +
        r12_double[1] * r12_double[1] +
        r12_double[2] * r12_double[2]);
      double f12_double[3] = {0.0, 0.0, 0.0};
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      double fn12_double[MAX_NUM_N] = {0.0};
      double fnp12_double[MAX_NUM_N] = {0.0};
      find_fn_and_fnp_in_double(
        paramb.basis_size_radial,
        static_cast<double>(rc),
        d12_double,
        fn12_double,
        fnp12_double);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        double gnp12_double = 0.0;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gnp12_double +=
            fnp12_double[k] * static_cast<double>(annmb.c_type_pair[c_index]);
        }
        const double tmp12_double =
          static_cast<double>(g_Fp[n1 + n * N]) * gnp12_double / d12_double;
        for (int d = 0; d < 3; ++d) {
          f12_double[d] += tmp12_double * r12_double[d];
        }
      }
      double s_sxx = 0.0;
      double s_sxy = 0.0;
      double s_sxz = 0.0;
      double s_syx = 0.0;
      double s_syy = 0.0;
      double s_syz = 0.0;
      double s_szx = 0.0;
      double s_szy = 0.0;
      double s_szz = 0.0;
      s_sxx -= r12_double[0] * f12_double[0];
      s_syy -= r12_double[1] * f12_double[1];
      s_szz -= r12_double[2] * f12_double[2];
      s_sxy -= r12_double[0] * f12_double[1];
      s_sxz -= r12_double[0] * f12_double[2];
      s_syz -= r12_double[1] * f12_double[2];
      s_syx -= r12_double[1] * f12_double[0];
      s_szx -= r12_double[2] * f12_double[0];
      s_szy -= r12_double[2] * f12_double[1];
      atomicAdd(&g_fx[n1], f12_double[0]);
      atomicAdd(&g_fy[n1], f12_double[1]);
      atomicAdd(&g_fz[n1], f12_double[2]);
      atomicAdd(&g_fx[n2], -f12_double[0]);
      atomicAdd(&g_fy[n2], -f12_double[1]);
      atomicAdd(&g_fz[n2], -f12_double[2]);
      // save virial
      // xx xy xz    0 3 4
      // yx yy yz    6 1 5
      // zx zy zz    7 8 2
      atomicAdd(&g_virial[n2 + 0 * N], s_sxx);
      atomicAdd(&g_virial[n2 + 1 * N], s_syy);
      atomicAdd(&g_virial[n2 + 2 * N], s_szz);
      atomicAdd(&g_virial[n2 + 3 * N], s_sxy);
      atomicAdd(&g_virial[n2 + 4 * N], s_sxz);
      atomicAdd(&g_virial[n2 + 5 * N], s_syz);
      atomicAdd(&g_virial[n2 + 6 * N], s_syx);
      atomicAdd(&g_virial[n2 + 7 * N], s_szx);
      atomicAdd(&g_virial[n2 + 8 * N], s_szy);
    }
  }
}
static __global__ void find_force_angular_small_box(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x12,
  const double* __restrict__ g_y12,
  const double* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
      }
    }
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      double r12_double[3] = {
        static_cast<double>(g_x12[index]),
        static_cast<double>(g_y12[index]),
        static_cast<double>(g_z12[index])};
      double f12_double[3] = {0.0, 0.0, 0.0};
      int t2 = g_type[n2];
      float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      const double d12_double = sqrt(
        r12_double[0] * r12_double[0] +
        r12_double[1] * r12_double[1] +
        r12_double[2] * r12_double[2]);
      double fn12_double[MAX_NUM_N] = {0.0};
      double fnp12_double[MAX_NUM_N] = {0.0};
      find_fn_and_fnp_in_double(
        paramb.basis_size_angular,
        static_cast<double>(rc),
        d12_double,
        fn12_double,
        fnp12_double);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        double gn12_double = 0.0;
        double gnp12_double = 0.0;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12_double +=
            fn12_double[k] * static_cast<double>(annmb.c_type_pair[c_index]);
          gnp12_double +=
            fnp12_double[k] * static_cast<double>(annmb.c_type_pair[c_index]);
        }
        accumulate_f12_small_box_in_double(
          paramb.L_max,
          paramb.has_q_222,
          paramb.has_q_1111,
          paramb.has_q_112,
          paramb.has_q_123,
          paramb.has_q_233,
          paramb.has_q_134,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12_double,
          r12_double,
          gn12_double,
          gnp12_double,
          Fp,
          sum_fxyz,
          f12_double);
      }
      double s_sxx = 0.0;
      double s_sxy = 0.0;
      double s_sxz = 0.0;
      double s_syx = 0.0;
      double s_syy = 0.0;
      double s_syz = 0.0;
      double s_szx = 0.0;
      double s_szy = 0.0;
      double s_szz = 0.0;
      s_sxx -= r12_double[0] * f12_double[0];
      s_syy -= r12_double[1] * f12_double[1];
      s_szz -= r12_double[2] * f12_double[2];
      s_sxy -= r12_double[0] * f12_double[1];
      s_sxz -= r12_double[0] * f12_double[2];
      s_syz -= r12_double[1] * f12_double[2];
      s_syx -= r12_double[1] * f12_double[0];
      s_szx -= r12_double[2] * f12_double[0];
      s_szy -= r12_double[2] * f12_double[1];
      atomicAdd(&g_fx[n1], f12_double[0]);
      atomicAdd(&g_fy[n1], f12_double[1]);
      atomicAdd(&g_fz[n1], f12_double[2]);
      atomicAdd(&g_fx[n2], -f12_double[0]);
      atomicAdd(&g_fy[n2], -f12_double[1]);
      atomicAdd(&g_fz[n2], -f12_double[2]);
      // save virial
      // xx xy xz    0 3 4
      // yx yy yz    6 1 5
      // zx zy zz    7 8 2
      atomicAdd(&g_virial[n2 + 0 * N], s_sxx);
      atomicAdd(&g_virial[n2 + 1 * N], s_syy);
      atomicAdd(&g_virial[n2 + 2 * N], s_szz);
      atomicAdd(&g_virial[n2 + 3 * N], s_sxy);
      atomicAdd(&g_virial[n2 + 4 * N], s_sxz);
      atomicAdd(&g_virial[n2 + 5 * N], s_syz);
      atomicAdd(&g_virial[n2 + 6 * N], s_syx);
      atomicAdd(&g_virial[n2 + 7 * N], s_szx);
      atomicAdd(&g_virial[n2 + 8 * N], s_szy);
    }
  }
}
} // namespace nep_spin_small_box
