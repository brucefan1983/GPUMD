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
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include "utilities/nep_utilities.cuh"
#include <cmath>

namespace nep_spin
{
using ::find_fc_and_fcp;

static __device__ __host__ __forceinline__ void
find_fc_and_fcp(double rc, double rcinv, double d12, double& fc, double& fcp)
{
  if (d12 < rc) {
    const double pi = 3.14159265358979323846;
    const double phase = pi * d12 * rcinv;
    fc = 0.5 * (1.0 + cos(phase));
    fcp = -0.5 * pi * rcinv * sin(phase);
  } else {
    fc = fcp = 0.0;
  }
}

template <typename Real>
static __device__ __host__ __forceinline__ void find_fn_and_fnp(
  const int n_max,
  const Real rcinv,
  const Real d12,
  const Real fc12,
  const Real fcp12,
  Real* fn,
  Real* fnp)
{
  Real d12_mul_rcinv = d12 * rcinv;
  Real x = 2.0f * (d12_mul_rcinv - 1.0f) * (d12_mul_rcinv - 1.0f) - 1.0f;
  fn[0] = fc12;
  fnp[0] = fcp12;
  if (n_max == 0)
    return;
  fn[1] = (x + 1.0f) * 0.5f * fc12;
  fnp[1] = 2.0f * (d12_mul_rcinv - 1.0f) * rcinv * fc12 + (x + 1.0f) * 0.5f * fcp12;
  Real u0 = 1.0f;
  Real u1 = 2.0f * x;
  Real u2;
  Real fn_m_minus_2 = 1.0f;
  Real fn_m_minus_1 = x;
  for (int m = 2; m <= n_max; ++m) {
    Real fn_tmp1 = 2.0f * x * fn_m_minus_1 - fn_m_minus_2;
    fn_m_minus_2 = fn_m_minus_1;
    fn_m_minus_1 = fn_tmp1;
    Real fnp_tmp = m * u1;
    u2 = 2.0f * x * u1 - u0;
    u0 = u1;
    u1 = u2;

    Real fn_tmp2 = (fn_tmp1 + 1.0f) * 0.5f;
    fnp[m] = (fnp_tmp * 2.0f * (d12 * rcinv - 1.0f) * rcinv) * fc12 + fn_tmp2 * fcp12;
    fn[m] = fn_tmp2 * fc12;
  }
}

template <typename Input>
static __device__ void apply_ann_one_layer(
  const int N_des,
  const int N_neu,
  const float* w0,
  const float* b0,
  const float* w1,
  const float* b1,
  const Input* q,
  double& energy,
  double* energy_derivative)
{
  for (int n = 0; n < N_neu; ++n) {
    double w0_times_q = 0.0f;
    for (int d = 0; d < N_des; ++d) {
      w0_times_q += static_cast<double>(w0[n * N_des + d]) * q[d];
    }
    double x1 = tanh(w0_times_q - b0[n]);
    double tanh_der = 1.0f - x1 * x1;
    energy += w1[n] * x1;
    for (int d = 0; d < N_des; ++d) {
      double y1 = tanh_der * w0[n * N_des + d];
      energy_derivative[d] += w1[n] * y1;
    }
  }
  energy -= b1[0];
}
} // namespace nep_spin
