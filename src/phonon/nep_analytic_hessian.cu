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

#include "nep_analytic_hessian.cuh"
#include "sparse_hessian.cuh"
#include "force/force.cuh"
#include "force/nep/nep.cuh"
#include "model/atom.cuh"
#include "model/group.cuh"
#include "utilities/error.cuh"
#include "utilities/nep_utilities.cuh"
#include "gpu_macro_ext.cuh"
#include "nep_hessian_descriptor.cuh"
#include "nep_hessian_model.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#define BLOCK_SIZE 64
#define MAX_DIM 256
#define MAX_NUM_N 17
#define MAX_ANGULAR_DESCRIPTORS 14
#define MAX_ANGULAR_MONOMIAL_DEGREE 4
#define LOW_L_MAX_S_TERMS 24
#define MAX_S_TERMS 80
#define ANGULAR_PREPARE_BLOCK_SIZE 32
#define ANGULAR_PAIR_BLOCK_SIZE 64
#define ANGULAR_RADIAL_BLOCK_SIZE 16


// ============================================================================
// Second derivatives of utility functions
// ============================================================================

// Cutoff function: fc, fcp (1st deriv), fcpp (2nd deriv)
static __device__ __forceinline__ void
find_fc_fcp_fcpp(float rc, float rcinv, float d12, float& fc, float& fcp, float& fcpp)
{
  if (d12 < rc) {
    float x = d12 * rcinv;
    float pix = 3.1415927f * x;
    fc = 0.5f * cosf(pix) + 0.5f;
    fcp = -1.5707963f * sinf(pix) * rcinv;
    fcpp = -4.9348022f * cosf(pix) * rcinv * rcinv;
  } else {
    fc = 0.0f; fcp = 0.0f; fcpp = 0.0f;
  }
}

// Chebyshev basis: fn, fnp (1st deriv), fnpp (2nd deriv) w.r.t. d12
// x(d) = 2*(d*rcinv-1)^2 - 1, fn = 0.5*(T_n(x)+1)*fc
// Recurrences: T_{m+1}=2x*T_m-T_{m-1}, T'_{m+1}=2x*T'_m+2*T_m-T'_{m-1},
//             T''_{m+1}=2x*T''_m+4*T'_m-T''_{m-1}
static __device__ __forceinline__ void find_fn_fnp_fnpp(
  const int n, const float rcinv, const float d12,
  const float fc12, const float fcp12, const float fcpp12,
  float& fn, float& fnp, float& fnpp)
{
  float t = d12 * rcinv - 1.0f;
  float x = 2.0f * t * t - 1.0f;
  float dx_dd = 4.0f * t * rcinv;
  float d2x_dd2 = 4.0f * rcinv * rcinv;

  if (n == 0) {
    fn = fc12; fnp = fcp12; fnpp = fcpp12;
  } else if (n == 1) {
    float Tn = x, Tnp = 1.0f, Tnpp = 0.0f;
    fn = 0.5f * (Tn + 1.0f) * fc12;
    fnp = 0.5f * (Tnp * dx_dd * fc12 + (Tn + 1.0f) * fcp12);
    fnpp = 0.5f * (Tnpp * dx_dd * dx_dd * fc12 + Tnp * d2x_dd2 * fc12
      + 2.0f * Tnp * dx_dd * fcp12 + (Tn + 1.0f) * fcpp12);
  } else {
    float t0=1,t1=x,t2, tp0=0,tp1=1,tp2, tpp0=0,tpp1=0,tpp2;
    for (int m = 1; m < n; ++m) {
      t2 = 2.0f*x*t1 - t0;
      tp2 = 2.0f*x*tp1 + 2.0f*t1 - tp0;
      tpp2 = 2.0f*x*tpp1 + 4.0f*tp1 - tpp0;
      t0=t1; t1=t2; tp0=tp1; tp1=tp2; tpp0=tpp1; tpp1=tpp2;
    }
    float Tn=t1, Tnp=tp1, Tnpp=tpp1;
    fn = 0.5f*(Tn+1.0f)*fc12;
    fnp = 0.5f*(Tnp*dx_dd*fc12 + (Tn+1.0f)*fcp12);
    fnpp = 0.5f*(Tnpp*dx_dd*dx_dd*fc12 + Tnp*d2x_dd2*fc12
      + 2.0f*Tnp*dx_dd*fcp12 + (Tn+1.0f)*fcpp12);
  }
}

static __device__ __forceinline__ float radial_second_derivative(
  const float first_derivative,
  const float second_derivative,
  const float distance_derivative_a,
  const float distance_derivative_b,
  const float distance_second_derivative)
{
  return second_derivative * distance_derivative_a * distance_derivative_b +
    first_derivative * distance_second_derivative;
}

static __device__ __forceinline__ int angular_fp_index(
  const int n_max_radial,
  const int n_max_angular_plus_1,
  const int descriptor,
  const int angular_basis,
  const int atom,
  const int num_atoms)
{
  return (n_max_radial + 1 + descriptor * n_max_angular_plus_1 + angular_basis) *
    num_atoms + atom;
}

static __device__ __forceinline__ void find_zbl_cutoff_derivatives(
  const float rc_inner,
  const float rc_outer,
  const float distance,
  float& cutoff,
  float& cutoff_first,
  float& cutoff_second)
{
  cutoff = 0.0f;
  cutoff_first = 0.0f;
  cutoff_second = 0.0f;
  if (distance < rc_inner) {
    cutoff = 1.0f;
  } else if (distance < rc_outer) {
    const float factor = 3.1415927f / (rc_outer - rc_inner);
    const float phase = factor * (distance - rc_inner);
    cutoff = 0.5f * cosf(phase) + 0.5f;
    cutoff_first = -0.5f * factor * sinf(phase);
    cutoff_second = -0.5f * factor * factor * cosf(phase);
  }
}

static __device__ __forceinline__ void find_zbl_energy_derivatives(
  const NEP::ParaMB paramb,
  const NEP::ZBL zbl,
  const int type1,
  const int type2,
  const float distance,
  float& energy,
  float& energy_first,
  float& energy_second)
{
  const int atomic_number1 = zbl.atomic_numbers[type1];
  const int atomic_number2 = zbl.atomic_numbers[type2];
  const float a_inv =
    (powf((float)atomic_number1, 0.23f) +
     powf((float)atomic_number2, 0.23f)) * 2.134563f;
  const float charge_product = K_C_SP * atomic_number1 * atomic_number2;

  float amplitude[4];
  float exponent[4];
  float rc_inner = zbl.rc_inner;
  float rc_outer = zbl.rc_outer;
  if (zbl.flexible) {
    int lower_type = type1 < type2 ? type1 : type2;
    int upper_type = type1 < type2 ? type2 : type1;
    const int index =
      lower_type * zbl.num_types - lower_type * (lower_type - 1) / 2 +
      upper_type - lower_type;
    rc_inner = zbl.para[10 * index];
    rc_outer = zbl.para[10 * index + 1];
    for (int term = 0; term < 4; ++term) {
      amplitude[term] = zbl.para[10 * index + 2 + 2 * term];
      exponent[term] = zbl.para[10 * index + 3 + 2 * term];
    }
  } else {
    if (paramb.use_typewise_cutoff_zbl) {
      rc_outer = min(
        (COVALENT_RADIUS[atomic_number1 - 1] +
         COVALENT_RADIUS[atomic_number2 - 1]) *
          paramb.typewise_cutoff_zbl_factor,
        rc_outer);
      rc_inner = 0.0f;
    }
    const float parameters[8] = {
      0.18175f, 3.1998f, 0.50986f, 0.94229f,
      0.28022f, 0.4029f, 0.02817f, 0.20162f};
    for (int term = 0; term < 4; ++term) {
      amplitude[term] = parameters[2 * term];
      exponent[term] = parameters[2 * term + 1];
    }
  }

  const float x = distance * a_inv;
  float phi = 0.0f;
  float phi_first = 0.0f;
  float phi_second = 0.0f;
  for (int term = 0; term < 4; ++term) {
    const float value = charge_product * amplitude[term] * expf(-exponent[term] * x);
    phi += value;
    phi_first -= exponent[term] * a_inv * value;
    phi_second += exponent[term] * exponent[term] * a_inv * a_inv * value;
  }

  const float distanceinv = 1.0f / distance;
  const float uncut_first = phi_first * distanceinv - phi * distanceinv * distanceinv;
  const float uncut_second = phi_second * distanceinv -
    2.0f * phi_first * distanceinv * distanceinv +
    2.0f * phi * distanceinv * distanceinv * distanceinv;

  float cutoff;
  float cutoff_first;
  float cutoff_second;
  find_zbl_cutoff_derivatives(rc_inner, rc_outer, distance, cutoff, cutoff_first, cutoff_second);
  const float uncut_energy = phi * distanceinv;
  energy = uncut_energy * cutoff;
  energy_first = uncut_first * cutoff + uncut_energy * cutoff_first;
  energy_second = uncut_second * cutoff + 2.0f * uncut_first * cutoff_first +
    uncut_energy * cutoff_second;
}

static __device__ __forceinline__ float z_coefficient(const int L, const int m, const int p)
{
  if (L == 1) return Z_COEFFICIENT_1[m][p];
  if (L == 2) return Z_COEFFICIENT_2[m][p];
  if (L == 3) return Z_COEFFICIENT_3[m][p];
  if (L == 4) return Z_COEFFICIENT_4[m][p];
  if (L == 5) return Z_COEFFICIENT_5[m][p];
  if (L == 6) return Z_COEFFICIENT_6[m][p];
  if (L == 7) return Z_COEFFICIENT_7[m][p];
  return Z_COEFFICIENT_8[m][p];
}

static __device__ __forceinline__ void
find_z_polynomial_derivatives(const int L, const int m, const float z, float& p0, float& p1, float& p2)
{
  p0 = 0.0f;
  p1 = 0.0f;
  p2 = 0.0f;
  float z_power[9] = {1.0f};
  for (int power = 1; power <= L; ++power) z_power[power] = z * z_power[power - 1];
  for (int power = (L + m) % 2; power <= L - m; power += 2) {
    const float coefficient = z_coefficient(L, m, power);
    p0 += coefficient * z_power[power];
    if (power > 0) p1 += coefficient * power * z_power[power - 1];
    if (power > 1) p2 += coefficient * power * (power - 1) * z_power[power - 2];
  }
}

static __device__ __forceinline__ float monomial_s_product(
  const int degree,
  const int* variables,
  const float* s,
  const float coefficient,
  const int skip_first,
  const int skip_second,
  const int s_stride)
{
  float value = coefficient;
  for (int term = 0; term < degree; ++term) {
    if (term == skip_first || term == skip_second) continue;
    value *= s[variables[term] * s_stride];
  }
  return value;
}

template <typename Accumulator>
static __device__ __forceinline__ void emit_q_monomial(
  Accumulator& accumulator,
  const float coefficient,
  const int degree,
  const int variable_0,
  const int variable_1,
  const int variable_2,
  const int variable_3)
{
  const int variables[MAX_ANGULAR_MONOMIAL_DEGREE] = {
    variable_0, variable_1, variable_2, variable_3};
  accumulator.add(coefficient, degree, variables);
}

template <typename Accumulator>
static __device__ __forceinline__ void visit_q_monomials(
  const NEP::ParaMB paramb,
  const int descriptor,
  Accumulator& accumulator)
{
  if (descriptor < paramb.L_max) {
    const int L = descriptor + 1;
    const int start = L * L - 1;
    emit_q_monomial(
      accumulator, C3B[start], 2, start, start, -1, -1);
    for (int term = 1; term < 2 * L + 1; ++term) {
      const int component = start + term;
      emit_q_monomial(
        accumulator, 2.0f * C3B[component], 2,
        component, component, -1, -1);
    }
    return;
  }

  int current_descriptor = paramb.L_max;
  if (paramb.has_q_222) {
    if (descriptor == current_descriptor) {
      emit_q_monomial(accumulator, C4B[0], 3, 3, 3, 3, -1);
      emit_q_monomial(accumulator, C4B[1], 3, 3, 4, 4, -1);
      emit_q_monomial(accumulator, C4B[1], 3, 3, 5, 5, -1);
      emit_q_monomial(accumulator, C4B[2], 3, 3, 6, 6, -1);
      emit_q_monomial(accumulator, C4B[2], 3, 3, 7, 7, -1);
      emit_q_monomial(accumulator, C4B[3], 3, 6, 5, 5, -1);
      emit_q_monomial(accumulator, -C4B[3], 3, 6, 4, 4, -1);
      emit_q_monomial(accumulator, C4B[4], 3, 4, 5, 7, -1);
      return;
    }
    ++current_descriptor;
  }

  if (paramb.has_q_1111) {
    if (descriptor == current_descriptor) {
      emit_q_monomial(accumulator, C5B[0], 4, 0, 0, 0, 0);
      emit_q_monomial(accumulator, C5B[1], 4, 0, 0, 1, 1);
      emit_q_monomial(accumulator, C5B[1], 4, 0, 0, 2, 2);
      emit_q_monomial(accumulator, C5B[2], 4, 1, 1, 1, 1);
      emit_q_monomial(accumulator, 2.0f * C5B[2], 4, 1, 1, 2, 2);
      emit_q_monomial(accumulator, C5B[2], 4, 2, 2, 2, 2);
      return;
    }
    ++current_descriptor;
  }

  if (paramb.has_q_112) {
    if (descriptor == current_descriptor) {
      emit_q_monomial(accumulator, C4B2[0], 3, 0, 0, 3, -1);
      emit_q_monomial(accumulator, C4B2[1], 3, 0, 1, 4, -1);
      emit_q_monomial(accumulator, C4B2[1], 3, 0, 2, 5, -1);
      emit_q_monomial(accumulator, C4B2[2], 3, 3, 1, 1, -1);
      emit_q_monomial(accumulator, C4B2[2], 3, 3, 2, 2, -1);
      emit_q_monomial(accumulator, C4B2[3], 3, 6, 1, 1, -1);
      emit_q_monomial(accumulator, -C4B2[3], 3, 6, 2, 2, -1);
      emit_q_monomial(accumulator, C4B2[4], 3, 1, 2, 7, -1);
      return;
    }
    ++current_descriptor;
  }

  if (paramb.has_q_123) {
    if (descriptor == current_descriptor) {
      emit_q_monomial(accumulator, C4B_123[6], 3, 12, 2, 4, -1);
      emit_q_monomial(accumulator, -C4B_123[6], 3, 11, 2, 5, -1);
      emit_q_monomial(accumulator, C4B_123[6], 3, 1, 11, 4, -1);
      emit_q_monomial(accumulator, C4B_123[6], 3, 1, 12, 5, -1);
      emit_q_monomial(accumulator, C4B_123[5], 3, 0, 11, 6, -1);
      emit_q_monomial(accumulator, C4B_123[5], 3, 0, 12, 7, -1);
      emit_q_monomial(accumulator, C4B_123[3], 3, 14, 2, 6, -1);
      emit_q_monomial(accumulator, -C4B_123[3], 3, 13, 2, 7, -1);
      emit_q_monomial(accumulator, C4B_123[3], 3, 1, 13, 6, -1);
      emit_q_monomial(accumulator, C4B_123[3], 3, 1, 14, 7, -1);
      emit_q_monomial(accumulator, C4B_123[4], 3, 10, 0, 5, -1);
      emit_q_monomial(accumulator, C4B_123[4], 3, 0, 4, 9, -1);
      emit_q_monomial(accumulator, C4B_123[1], 3, 10, 2, 3, -1);
      emit_q_monomial(accumulator, C4B_123[1], 3, 0, 3, 8, -1);
      emit_q_monomial(accumulator, C4B_123[1], 3, 1, 3, 9, -1);
      emit_q_monomial(accumulator, C4B_123[0], 3, 10, 2, 6, -1);
      emit_q_monomial(accumulator, -C4B_123[0], 3, 10, 1, 7, -1);
      emit_q_monomial(accumulator, -C4B_123[0], 3, 2, 7, 9, -1);
      emit_q_monomial(accumulator, -C4B_123[0], 3, 1, 6, 9, -1);
      emit_q_monomial(accumulator, -C4B_123[2], 3, 2, 5, 8, -1);
      emit_q_monomial(accumulator, -C4B_123[2], 3, 1, 4, 8, -1);
      return;
    }
    ++current_descriptor;
  }

  if (paramb.has_q_233) {
    if (descriptor == current_descriptor) {
      emit_q_monomial(accumulator, C4B_233[0], 3, 3, 8, 8, -1);
      emit_q_monomial(accumulator, C4B_233[1], 3, 10, 10, 3, -1);
      emit_q_monomial(accumulator, C4B_233[1], 3, 3, 9, 9, -1);
      emit_q_monomial(accumulator, -C4B_233[2], 3, 10, 10, 6, -1);
      emit_q_monomial(accumulator, C4B_233[2], 3, 6, 9, 9, -1);
      emit_q_monomial(accumulator, C4B_233[3], 3, 4, 8, 9, -1);
      emit_q_monomial(accumulator, C4B_233[3], 3, 10, 5, 8, -1);
      emit_q_monomial(accumulator, -C4B_233[4], 3, 13, 13, 3, -1);
      emit_q_monomial(accumulator, -C4B_233[4], 3, 14, 14, 3, -1);
      emit_q_monomial(accumulator, -C4B_233[5], 3, 14, 7, 9, -1);
      emit_q_monomial(accumulator, -C4B_233[5], 3, 13, 6, 9, -1);
      emit_q_monomial(accumulator, -C4B_233[5], 3, 10, 14, 6, -1);
      emit_q_monomial(accumulator, C4B_233[5], 3, 10, 13, 7, -1);
      emit_q_monomial(accumulator, C4B_233[6], 3, 10, 7, 9, -1);
      emit_q_monomial(accumulator, -C4B_233[7], 3, 11, 6, 8, -1);
      emit_q_monomial(accumulator, -C4B_233[7], 3, 12, 7, 8, -1);
      emit_q_monomial(accumulator, C4B_233[8], 3, 11, 4, 9, -1);
      emit_q_monomial(accumulator, C4B_233[8], 3, 12, 5, 9, -1);
      emit_q_monomial(accumulator, C4B_233[8], 3, 10, 12, 4, -1);
      emit_q_monomial(accumulator, -C4B_233[8], 3, 10, 11, 5, -1);
      emit_q_monomial(accumulator, C4B_233[9], 3, 12, 14, 4, -1);
      emit_q_monomial(accumulator, C4B_233[9], 3, 11, 14, 5, -1);
      emit_q_monomial(accumulator, C4B_233[9], 3, 13, 11, 4, -1);
      emit_q_monomial(accumulator, -C4B_233[9], 3, 13, 12, 5, -1);
      return;
    }
    ++current_descriptor;
  }

  if (paramb.has_q_134 && descriptor == current_descriptor) {
    emit_q_monomial(accumulator, -C4B_134[0], 3, 10, 15, 2, -1);
    emit_q_monomial(accumulator, -C4B_134[0], 3, 1, 15, 9, -1);
    emit_q_monomial(accumulator, C4B_134[1], 3, 0, 15, 8, -1);
    emit_q_monomial(accumulator, -C4B_134[2], 3, 1, 13, 18, -1);
    emit_q_monomial(accumulator, -C4B_134[2], 3, 1, 14, 19, -1);
    emit_q_monomial(accumulator, -C4B_134[2], 3, 2, 14, 18, -1);
    emit_q_monomial(accumulator, C4B_134[2], 3, 2, 13, 19, -1);
    emit_q_monomial(accumulator, -C4B_134[3], 3, 10, 18, 2, -1);
    emit_q_monomial(accumulator, C4B_134[3], 3, 1, 10, 19, -1);
    emit_q_monomial(accumulator, C4B_134[3], 3, 1, 18, 9, -1);
    emit_q_monomial(accumulator, C4B_134[3], 3, 2, 19, 9, -1);
    emit_q_monomial(accumulator, C4B_134[4], 3, 1, 16, 8, -1);
    emit_q_monomial(accumulator, C4B_134[4], 3, 2, 17, 8, -1);
    emit_q_monomial(accumulator, C4B_134[5], 3, 0, 10, 17, -1);
    emit_q_monomial(accumulator, C4B_134[5], 3, 0, 16, 9, -1);
    emit_q_monomial(accumulator, -C4B_134[5], 3, 1, 11, 16, -1);
    emit_q_monomial(accumulator, -C4B_134[5], 3, 1, 12, 17, -1);
    emit_q_monomial(accumulator, -C4B_134[5], 3, 2, 12, 16, -1);
    emit_q_monomial(accumulator, C4B_134[5], 3, 2, 11, 17, -1);
    emit_q_monomial(accumulator, C4B_134[6], 3, 1, 13, 22, -1);
    emit_q_monomial(accumulator, C4B_134[6], 3, 1, 14, 23, -1);
    emit_q_monomial(accumulator, -C4B_134[6], 3, 2, 14, 22, -1);
    emit_q_monomial(accumulator, C4B_134[6], 3, 2, 13, 23, -1);
    emit_q_monomial(accumulator, C4B_134[7], 3, 0, 11, 18, -1);
    emit_q_monomial(accumulator, C4B_134[7], 3, 0, 12, 19, -1);
    emit_q_monomial(accumulator, C4B_134[8], 3, 0, 13, 20, -1);
    emit_q_monomial(accumulator, C4B_134[8], 3, 0, 14, 21, -1);
    emit_q_monomial(accumulator, C4B_134[9], 3, 1, 11, 20, -1);
    emit_q_monomial(accumulator, C4B_134[9], 3, 1, 12, 21, -1);
    emit_q_monomial(accumulator, -C4B_134[9], 3, 2, 12, 20, -1);
    emit_q_monomial(accumulator, C4B_134[9], 3, 2, 11, 21, -1);
  }
}

struct LocalQChainAccumulator
{
  const float* s;
  const float (*s_gradient)[3];
  const float (*s_hessian)[3][3];
  float* gradient;
  float (*hessian)[3];

  static __device__ __forceinline__ float product(
    const float coefficient,
    const int degree,
    const int* variables,
    const float* s,
    const int skip_first,
    const int skip_second)
  {
    return monomial_s_product(
      degree, variables, s, coefficient, skip_first, skip_second, 1);
  }

  __device__ __forceinline__ void add(
    const float coefficient, const int degree, const int* variables)
  {
    for (int first_factor = 0; first_factor < degree; ++first_factor) {
      const int first_variable = variables[first_factor];
      const float first = product(
        coefficient, degree, variables, s, first_factor, -1);
      for (int axis_a = 0; axis_a < 3; ++axis_a) {
        gradient[axis_a] += first * s_gradient[first_variable][axis_a];
        for (int axis_b = 0; axis_b < 3; ++axis_b) {
          hessian[axis_a][axis_b] +=
            first * s_hessian[first_variable][axis_a][axis_b];
        }
      }

      for (int second_factor = 0; second_factor < degree; ++second_factor) {
        if (first_factor == second_factor) continue;
        const int second_variable = variables[second_factor];
        const float second = product(
          coefficient, degree, variables, s, first_factor, second_factor);
        for (int axis_a = 0; axis_a < 3; ++axis_a)
          for (int axis_b = 0; axis_b < 3; ++axis_b)
            hessian[axis_a][axis_b] += second *
              s_gradient[first_variable][axis_a] *
              s_gradient[second_variable][axis_b];
      }
    }
  }
};

struct CrossQChainAccumulator
{
  const float* s;
  int s_stride;
  const float (*first_gradient)[3];
  int first_axis;
  const float (*second_gradient)[3];
  int second_axis;
  float value;

  __device__ __forceinline__ void add(
    const float coefficient, const int degree, const int* variables)
  {
    for (int first_factor = 0; first_factor < degree; ++first_factor) {
      for (int second_factor = 0; second_factor < degree; ++second_factor) {
        if (first_factor == second_factor) continue;
        value += monomial_s_product(
                   degree, variables, s, coefficient,
                   first_factor, second_factor, s_stride) *
          first_gradient[variables[first_factor]][first_axis] *
          second_gradient[variables[second_factor]][second_axis];
      }
    }
  }
};

static __device__ __forceinline__ float contract_q_s_hessian(
  const NEP::ParaMB paramb,
  const int descriptor,
  const float* s,
  const int s_stride,
  const float (*first_gradient)[3],
  const int first_axis,
  const float (*second_gradient)[3],
  const int second_axis)
{
  CrossQChainAccumulator accumulator = {
    s, s_stride, first_gradient, first_axis,
    second_gradient, second_axis, 0.0f};
  visit_q_monomials(paramb, descriptor, accumulator);
  return accumulator.value;
}


static __device__ __forceinline__ void add_angular_basis_hessian(
  const int L,
  const int m,
  const bool imaginary,
  const float g,
  const float* g_gradient,
  const float (*g_hessian)[3],
  const float* unit,
  const float (*unit_gradient)[3],
  const float (*unit_hessian)[3][3],
  const int s_index,
  float (*s_gradient)[3],
  float (*s_hessian)[3][3])
{
  float p0, p1, p2;
  find_z_polynomial_derivatives(L, m, unit[2], p0, p1, p2);

  float value = 0.0f, value_gradient[3] = {0.0f, 0.0f, 0.0f},
        value_hessian[3][3] = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
  if (m == 0) {
    value = p0;
    for (int a = 0; a < 3; ++a) value_gradient[a] = p1 * unit_gradient[2][a];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        value_hessian[a][b] =
          p2 * unit_gradient[2][a] * unit_gradient[2][b] + p1 * unit_hessian[2][a][b];
      }
    }
  } else {
    const float x = unit[0];
    const float y = unit[1];
    float x_power[9] = {1.0f}, y_power[9] = {1.0f};
    for (int power = 1; power <= m; ++power) {
      x_power[power] = x * x_power[power - 1];
      y_power[power] = y * y_power[power - 1];
    }

    int binomial[9] = {1, m, 0, 0, 0, 0, 0, 0, 0};
    for (int power = 2; power <= m; ++power) {
      binomial[power] = binomial[power - 1] * (m - power + 1) / power;
    }
    for (int x_power_index = 0; x_power_index <= m; ++x_power_index) {
      const int y_power_index = m - x_power_index;
      const bool selected = imaginary ? (x_power_index % 2 != m % 2) : (x_power_index % 2 == m % 2);
      if (!selected) continue;
      float coefficient = (float)binomial[y_power_index];
      if (y_power_index / 2 % 2 == 1) coefficient = -coefficient;
      const float powers[2] = {x_power[x_power_index], y_power[y_power_index]};
	      const float x_without_one = x_power[x_power_index > 0 ? x_power_index - 1 : 0];
	      const float y_without_one = y_power[y_power_index > 0 ? y_power_index - 1 : 0];
	      const float x_without_two = x_power[x_power_index > 1 ? x_power_index - 2 : 0];
	      const float y_without_two = y_power[y_power_index > 1 ? y_power_index - 2 : 0];
	      const float term_gradient_x = x_power_index > 0 ?
	        coefficient * x_power_index * x_without_one * y_power[y_power_index] : 0.0f;
	      const float term_gradient_y = y_power_index > 0 ?
	        coefficient * y_power_index * y_without_one * x_power[x_power_index] : 0.0f;
	      const float term_hessian_xx = x_power_index > 1 ?
	        coefficient * x_power_index * (x_power_index - 1) * x_without_two *
	        y_power[y_power_index] : 0.0f;
	      const float term_hessian_xy = x_power_index > 0 && y_power_index > 0 ?
	        coefficient * x_power_index * y_power_index * x_without_one * y_without_one : 0.0f;
	      const float term_hessian_yy = y_power_index > 1 ?
	        coefficient * y_power_index * (y_power_index - 1) * y_without_two *
	        x_power[x_power_index] : 0.0f;
	      value += coefficient * powers[0] * powers[1];
	      for (int a = 0; a < 3; ++a) {
	        value_gradient[a] += term_gradient_x * unit_gradient[0][a] +
	          term_gradient_y * unit_gradient[1][a];
	        for (int b = 0; b < 3; ++b) {
          value_hessian[a][b] +=
	            term_gradient_x * unit_hessian[0][a][b] +
	            term_gradient_y * unit_hessian[1][a][b] +
	            term_hessian_xx * unit_gradient[0][a] * unit_gradient[0][b] +
	            term_hessian_xy * (unit_gradient[0][a] * unit_gradient[1][b] +
	              unit_gradient[1][a] * unit_gradient[0][b]) +
	            term_hessian_yy * unit_gradient[1][a] * unit_gradient[1][b];
        }
      }
    }

    // Save the xy-part (value, gradient, hessian) BEFORE multiplying by p0,
    // so the product rule for f(z) = p0(z) * xy(x,y) is applied correctly
    // even when p0 = 0 (e.g. z_unit = 0, m >= 1 odd).
    const float value_xy = value;
    float xy_gradient[3] = {value_gradient[0], value_gradient[1], value_gradient[2]};
    float xy_hessian[3][3];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) xy_hessian[a][b] = value_hessian[a][b];

    // total = p0 * value_xy
    // total_gradient = p0 * xy_gradient + p1 * unit_grad[2] * value_xy
    // total_hessian  = p0 * xy_hessian
    //               + p1 * (unit_grad[2] (x) xy_gradient + xy_gradient (x) unit_grad[2])
    //               + (p2 * unit_grad[2] (x) unit_grad[2] + p1 * unit_hessian[2]) * value_xy
    value = p0 * value_xy;
    for (int a = 0; a < 3; ++a) {
      value_gradient[a] = p0 * xy_gradient[a] + p1 * unit_gradient[2][a] * value_xy;
      for (int b = 0; b < 3; ++b) {
        value_hessian[a][b] =
          p0 * xy_hessian[a][b] +
          p1 * (unit_gradient[2][a] * xy_gradient[b] + unit_gradient[2][b] * xy_gradient[a]) +
          (p2 * unit_gradient[2][a] * unit_gradient[2][b] + p1 * unit_hessian[2][a][b]) * value_xy;
      }
    }
  }

  for (int a = 0; a < 3; ++a) {
    s_gradient[s_index][a] += g_gradient[a] * value + g * value_gradient[a];
    for (int b = 0; b < 3; ++b)
      s_hessian[s_index][a][b] += g_hessian[a][b] * value +
        g_gradient[a] * value_gradient[b] + g_gradient[b] * value_gradient[a] +
        g * value_hessian[a][b];
  }
}

static __device__ __forceinline__ int angular_s_index(const int L, const int m, const bool imaginary)
{
  const int start = L * L - 1;
  return m == 0 ? start : start + 1 + 2 * (m - 1) + (imaginary ? 1 : 0);
}


static __device__ __forceinline__ void compute_angular_s_derivatives_for_n(
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int center,
  const int neighbor,
  const int angular_n,
  const int* g_type,
  const float* r12,
  const float distance,
  float (*s_gradient)[3],
  float (*s_hessian)[3][3])
{
  const int center_type = g_type[center];
  const int neighbor_type = g_type[neighbor];
  const float d12inv = 1.0f / distance;
  const float d12inv3 = d12inv * d12inv * d12inv;
  const float rc =
    (paramb.rc_angular[center_type] + paramb.rc_angular[neighbor_type]) * 0.5f;
  const float rcinv = 1.0f / rc;
  float fc, fcp, fcpp;
  find_fc_fcp_fcpp(rc, rcinv, distance, fc, fcp, fcpp);

  float g = 0.0f;
  float g_gradient[3] = {0.0f, 0.0f, 0.0f};
  float g_hessian[3][3] = {{0.0f}};
  for (int basis = 0; basis <= paramb.basis_size_angular; ++basis) {
    float fn, fnp, fnpp;
    find_fn_fnp_fnpp(
      basis, rcinv, distance, fc, fcp, fcpp, fn, fnp, fnpp);
    const int c_index =
      get_c_index(
        center_type * paramb.num_types + neighbor_type, angular_n, basis,
        paramb.n_max_angular, paramb.basis_size_angular, paramb.num_c_radial);
    const float coefficient = annmb.c_type_pair[c_index];
    g += coefficient * fn;
    for (int axis_a = 0; axis_a < 3; ++axis_a) {
      g_gradient[axis_a] +=
        coefficient * fnp * r12[axis_a] * d12inv;
      for (int axis_b = 0; axis_b < 3; ++axis_b) {
        const float delta = axis_a == axis_b ? 1.0f : 0.0f;
        const float distance_hessian = delta * d12inv -
          r12[axis_a] * r12[axis_b] * d12inv3;
        g_hessian[axis_a][axis_b] += radial_second_derivative(
          coefficient * fnp, coefficient * fnpp,
          r12[axis_a] * d12inv, r12[axis_b] * d12inv,
          distance_hessian);
      }
    }
  }

  const int num_s_terms =
    (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
  for (int component = 0; component < num_s_terms; ++component) {
    for (int axis_a = 0; axis_a < 3; ++axis_a) {
      s_gradient[component][axis_a] = 0.0f;
      for (int axis_b = 0; axis_b < 3; ++axis_b)
        s_hessian[component][axis_a][axis_b] = 0.0f;
    }
  }

  const float unit[3] = {
    r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};
  float unit_gradient[3][3];
  float unit_hessian[3][3][3];
  for (int component = 0; component < 3; ++component) {
    for (int axis_a = 0; axis_a < 3; ++axis_a) {
      unit_gradient[component][axis_a] =
        (component == axis_a ? 1.0f : 0.0f) * d12inv -
        unit[component] * unit[axis_a] * d12inv;
      for (int axis_b = 0; axis_b < 3; ++axis_b) {
        const float ua = unit[axis_a];
        const float ub = unit[axis_b];
        const float uc = unit[component];
        const float delta_ab = axis_a == axis_b ? 1.0f : 0.0f;
        const float delta_ac = axis_a == component ? 1.0f : 0.0f;
        const float delta_bc = axis_b == component ? 1.0f : 0.0f;
        unit_hessian[component][axis_a][axis_b] =
          (3.0f * ua * ub * uc - delta_ab * uc - delta_ac * ub -
           delta_bc * ua) /
          (distance * distance);
      }
    }
  }

  for (int L = 1; L <= paramb.L_max; ++L) {
    for (int m = 0; m <= L; ++m) {
      add_angular_basis_hessian(
        L, m, false, g, g_gradient, g_hessian, unit, unit_gradient,
        unit_hessian, angular_s_index(L, m, false), s_gradient,
        s_hessian);
      if (m > 0)
        add_angular_basis_hessian(
          L, m, true, g, g_gradient, g_hessian, unit, unit_gradient,
          unit_hessian, angular_s_index(L, m, true), s_gradient,
          s_hessian);
    }
  }
}

static __device__ __forceinline__ void compute_angular_q_derivatives_for_n(
  const NEP::ParaMB paramb,
  const int descriptor,
  const float* sum_fxyz,
  const float (*s_gradient)[3],
  const float (*s_hessian)[3][3],
  float* neighbor_gradient,
  float (*local_hessian)[3])
{
  for (int axis_a = 0; axis_a < 3; ++axis_a) {
    neighbor_gradient[axis_a] = 0.0f;
    for (int axis_b = 0; axis_b < 3; ++axis_b)
      local_hessian[axis_a][axis_b] = 0.0f;
  }

  LocalQChainAccumulator accumulator = {
    sum_fxyz, s_gradient, s_hessian,
    neighbor_gradient, local_hessian};
  visit_q_monomials(paramb, descriptor, accumulator);

}

static __device__ __forceinline__ void compute_radial_neighbor_derivatives(
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int center_type,
  const int neighbor_type,
  const float* r12,
  const float distance,
  float* radial_derivative)
{
  const float rc =
    (paramb.rc_radial[center_type] + paramb.rc_radial[neighbor_type]) * 0.5f;
  const float rcinv = 1.0f / rc;
  float fc, fcp, fcpp;
  find_fc_fcp_fcpp(rc, rcinv, distance, fc, fcp, fcpp);

  for (int n = 0; n <= paramb.n_max_radial; ++n) {
    radial_derivative[n] = 0.0f;
    for (int basis = 0; basis <= paramb.basis_size_radial; ++basis) {
      const int c_index =
        get_c_index(center_type * paramb.num_types + neighbor_type, n, basis,
            paramb.n_max_radial, paramb.basis_size_radial);
      float fn, fnp, fnpp;
      find_fn_fnp_fnpp(basis, rcinv, distance, fc, fcp, fcpp, fn, fnp, fnpp);
      radial_derivative[n] += annmb.c_type_pair[c_index] * fnp;
    }
  }
}

static __device__ __forceinline__ void load_neighbor_vector(
  const Box box,
  const int N,
  const int center,
  const int neighbor_slot,
  const int neighbor,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_r12_x,
  const float* g_r12_y,
  const float* g_r12_z,
  float* r12)
{
  const int edge_index = center + N * neighbor_slot;
  if (g_r12_x != nullptr) {
    r12[0] = g_r12_x[edge_index];
    r12[1] = g_r12_y[edge_index];
    r12[2] = g_r12_z[edge_index];
    return;
  }

  double x12 = g_x[neighbor] - g_x[center];
  double y12 = g_y[neighbor] - g_y[center];
  double z12 = g_z[neighbor] - g_z[center];
  apply_mic(box, x12, y12, z12);
  r12[0] = static_cast<float>(x12);
  r12[1] = static_cast<float>(y12);
  r12[2] = static_cast<float>(z12);
}

// Validate the exact vectors consumed by the derivative kernels, including
// explicit periodic images in small boxes. Do not silently omit close pairs.
static __global__ void validate_hessian_neighbors(
  const int N, const Box box, const int* counts, const int* neighbors,
  const double* x, const double* y, const double* z,
  const float* r12_x, const float* r12_y, const float* r12_z, int* invalid)
{
  const int center = blockIdx.x * blockDim.x + threadIdx.x;
  if (center >= N) return;
  for (int slot = 0; slot < counts[center]; ++slot) {
    const int neighbor = neighbors[center + N * slot];
    if (neighbor < 0 || neighbor >= N) {
      atomicOr(invalid, 1);
      continue;
    }
    float r12[3];
    load_neighbor_vector(
      box, N, center, slot, neighbor, x, y, z, r12_x, r12_y, r12_z, r12);
    const float distance = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (!isfinite(distance)) atomicOr(invalid, 2);
    else if (distance < 1.0e-6f) atomicOr(invalid, 4);
  }
}

// ============================================================================
// GPU kernel: compute radial Hessian (Term 1 + Term 2 for radial descriptors)
//
// Term 1: sum_{n,n'} Hqq[n,n'] * (dqn/dxa) * (dqn'/dxb)   [ANN Hessian cross term]
// Term 2: sum_n Fp[n] * d2qn/dxa dxb                       [descriptor 2nd deriv]
//
// For radial: qn = sum_k c[n,k] * fn_k(d12)
//   dqn/dx1a = gnp[n] * (-r12a/d12)  (where gnp includes c sum)
//   d2qn/dx1a dx1b = gnpp[n]*r12a*r12b*d12inv^2 + gnp[n]*(delta_ab/d12 - r12a*r12b/d12^3)
// ============================================================================
static __global__ void find_radial_hessian_kernel(
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_r12_x,
  const float* g_r12_y,
  const float* g_r12_z,
  const float* g_Fp,
  const float* g_Hqq,    // per-atom ANN Hessian: [atom, i, j] = g_Hqq[(n1*annmb.dim + i)*annmb.dim + j]
  const int* g_sparse_row_offsets,
  const int* g_sparse_columns,
  double* g_hessian)     // N3 x N3
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;

  const int t1 = g_type[n1];
  const int n_max_r = paramb.n_max_radial;
  const float* q_scaler = annmb.q_scaler;
  float center_gradient[MAX_NUM_N][3] = {};
  float total_hessian[MAX_NUM_N][3][3] = {};
  float gnp[MAX_NUM_N];
  float gnpp[MAX_NUM_N];
  float other_radial_derivative[MAX_NUM_N];

  for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
    const int n2 = g_NL_radial[n1 + N * i1];
    const int t2 = g_type[n2];
    float r12[3];
    load_neighbor_vector(
      box, N, n1, i1, n2, g_x, g_y, g_z,
      g_r12_x, g_r12_y, g_r12_z, r12);
    const float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    const float d12inv = 1.0f / d12;
    const float d12inv3 = d12inv * d12inv * d12inv;
    const float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
    const float rcinv = 1.0f / rc;
    float fc, fcp, fcpp;
    find_fc_fcp_fcpp(rc, rcinv, d12, fc, fcp, fcpp);

    for (int n = 0; n <= n_max_r; ++n) {
      float gp = 0.0f, gpp = 0.0f;
      for (int basis = 0; basis <= paramb.basis_size_radial; ++basis) {
        const int c_index =
          get_c_index(t1 * paramb.num_types + t2, n, basis,
            paramb.n_max_radial, paramb.basis_size_radial);
        float fn, fnp, fnpp;
        find_fn_fnp_fnpp(basis, rcinv, d12, fc, fcp, fcpp, fn, fnp, fnpp);
        gp += annmb.c_type_pair[c_index] * fnp;
        gpp += annmb.c_type_pair[c_index] * fnpp;
      }
      gnp[n] = gp;
      gnpp[n] = gpp;
      for (int a = 0; a < 3; ++a) {
        const float dd12_a = r12[a] * d12inv;
        center_gradient[n][a] -= gp * dd12_a;
        for (int b = 0; b < 3; ++b) {
          const float dd12_b = r12[b] * d12inv;
          const float d2d12 = ((a == b) ? 1.0f : 0.0f) * d12inv -
            r12[a] * r12[b] * d12inv3;
          total_hessian[n][a][b] +=
            gpp * dd12_a * dd12_b + gp * d2d12;
        }
      }
    }
  }

  float Fp1[MAX_DIM];
  for (int d = 0; d < annmb.dim; ++d) Fp1[d] = g_Fp[d * N + n1];
  const float* Hqq1 = g_Hqq + (size_t)n1 * annmb.dim * annmb.dim;

  for (int a = 0; a < 3; ++a) {
    for (int b = 0; b < 3; ++b) {
      float center_value = 0.0f;
      for (int n = 0; n <= n_max_r; ++n)
      center_value += Fp1[n] * total_hessian[n][a][b];
      for (int n = 0; n <= n_max_r; ++n) {
        const float center_a = q_scaler[n] * center_gradient[n][a];
        for (int np = 0; np <= n_max_r; ++np) {
          center_value += Hqq1[n * annmb.dim + np] * center_a *
            q_scaler[np] * center_gradient[np][b];
        }
      }
      const int row_center = a * N + n1;
      const int column_center = b * N + n1;
      sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
        g_sparse_columns, N, row_center, column_center, (double)center_value);
    }
  }

  for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
    const int n2 = g_NL_radial[n1 + N * i1];
    const int t2 = g_type[n2];
    float r12[3];
    load_neighbor_vector(
      box, N, n1, i1, n2, g_x, g_y, g_z,
      g_r12_x, g_r12_y, g_r12_z, r12);
    const float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    const float d12inv = 1.0f / d12;
    const float d12inv3 = d12inv * d12inv * d12inv;
    const float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
    const float rcinv = 1.0f / rc;
    float fc, fcp, fcpp;
    find_fc_fcp_fcpp(rc, rcinv, d12, fc, fcp, fcpp);

    for (int n = 0; n <= n_max_r; ++n) {
      float gp = 0.0f, gpp = 0.0f;
      for (int basis = 0; basis <= paramb.basis_size_radial; ++basis) {
        const int c_index =
          get_c_index(t1 * paramb.num_types + t2, n, basis,
            paramb.n_max_radial, paramb.basis_size_radial);
        float fn, fnp, fnpp;
        find_fn_fnp_fnpp(basis, rcinv, d12, fc, fcp, fcpp, fn, fnp, fnpp);
        gp += annmb.c_type_pair[c_index] * fnp;
        gpp += annmb.c_type_pair[c_index] * fnpp;
      }
      gnp[n] = gp;
      gnpp[n] = gpp;
    }

    for (int a = 0; a < 3; ++a) {
      const float dd12_a = r12[a] * d12inv;
      for (int b = 0; b < 3; ++b) {
        const float dd12_b = r12[b] * d12inv;
        const float d2d12 = ((a == b) ? 1.0f : 0.0f) * d12inv -
          r12[a] * r12[b] * d12inv3;

        float neighbor_value = 0.0f;
        float center_neighbor_value = 0.0f;
        float neighbor_center_value = 0.0f;
        for (int n = 0; n <= n_max_r; ++n) {
          const float local_hessian =
            radial_second_derivative(gnp[n], gnpp[n], dd12_a, dd12_b, d2d12);
          neighbor_value += Fp1[n] * local_hessian;
          center_neighbor_value -= Fp1[n] * local_hessian;
          neighbor_center_value -= Fp1[n] * local_hessian;
        }
        for (int n = 0; n <= n_max_r; ++n) {
          const float neighbor_a = gnp[n] * dd12_a;
          const float center_a = q_scaler[n] * center_gradient[n][a];
          for (int np = 0; np <= n_max_r; ++np) {
            const float neighbor_b = gnp[np] * dd12_b;
            neighbor_value += Hqq1[n * annmb.dim + np] * q_scaler[n] *
              neighbor_a * q_scaler[np] * neighbor_b;
            center_neighbor_value += Hqq1[n * annmb.dim + np] * center_a *
              q_scaler[np] * neighbor_b;
            neighbor_center_value += Hqq1[n * annmb.dim + np] *
              q_scaler[n] * neighbor_a * q_scaler[np] * center_gradient[np][b];
          }
        }

        const int row_center_a = a * N + n1;
        const int row_neighbor_a = a * N + n2;
        const int col_center_b = b * N + n1;
        const int col_neighbor_b = b * N + n2;
        sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
          g_sparse_columns, N, row_neighbor_a, col_neighbor_b, (double)neighbor_value);
        sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
          g_sparse_columns, N, row_center_a, col_neighbor_b, (double)center_neighbor_value);
        sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
          g_sparse_columns, N, row_neighbor_a, col_center_b, (double)neighbor_center_value);
      }
    }
  }

  for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
    const int n2 = g_NL_radial[n1 + N * i1];
    const int t2 = g_type[n2];
    float r12[3];
    load_neighbor_vector(
      box, N, n1, i1, n2, g_x, g_y, g_z,
      g_r12_x, g_r12_y, g_r12_z, r12);
    const float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    const float d12inv = 1.0f / d12;
    const float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
    const float rcinv = 1.0f / rc;
    float fc, fcp, fcpp;
    find_fc_fcp_fcpp(rc, rcinv, d12, fc, fcp, fcpp);
    for (int n = 0; n <= n_max_r; ++n) {
      gnp[n] = 0.0f;
      for (int basis = 0; basis <= paramb.basis_size_radial; ++basis) {
        const int c_index =
          get_c_index(t1 * paramb.num_types + t2, n, basis,
            paramb.n_max_radial, paramb.basis_size_radial);
        float fn, fnp, fnpp;
        find_fn_fnp_fnpp(basis, rcinv, d12, fc, fcp, fcpp, fn, fnp, fnpp);
        gnp[n] += annmb.c_type_pair[c_index] * fnp;
      }
    }

    for (int i2 = i1 + 1; i2 < g_NN_radial[n1]; ++i2) {
      const int n3 = g_NL_radial[n1 + N * i2];
      const int t3 = g_type[n3];
      float r13[3];
      load_neighbor_vector(
        box, N, n1, i2, n3, g_x, g_y, g_z,
        g_r12_x, g_r12_y, g_r12_z, r13);
      const float d13 = sqrtf(r13[0] * r13[0] + r13[1] * r13[1] + r13[2] * r13[2]);
      const float d13inv = 1.0f / d13;
      const float rc13 = (paramb.rc_radial[t1] + paramb.rc_radial[t3]) * 0.5f;
      const float rc13inv = 1.0f / rc13;
      find_fc_fcp_fcpp(rc13, rc13inv, d13, fc, fcp, fcpp);
      for (int n = 0; n <= n_max_r; ++n) {
        other_radial_derivative[n] = 0.0f;
        for (int basis = 0; basis <= paramb.basis_size_radial; ++basis) {
          const int c_index =
            get_c_index(t1 * paramb.num_types + t3, n, basis,
            paramb.n_max_radial, paramb.basis_size_radial);
          float fn, fnp, fnpp;
          find_fn_fnp_fnpp(basis, rc13inv, d13, fc, fcp, fcpp, fn, fnp, fnpp);
          other_radial_derivative[n] += annmb.c_type_pair[c_index] * fnp;
        }
      }

      for (int a = 0; a < 3; ++a) {
        const float unit12_a = r12[a] * d12inv;
        const float unit13_a = r13[a] * d13inv;
        for (int b = 0; b < 3; ++b) {
          const float unit12_b = r12[b] * d12inv;
          const float unit13_b = r13[b] * d13inv;
          float value_12_13 = 0.0f;
          float value_13_12 = 0.0f;
          for (int n = 0; n <= n_max_r; ++n) {
            const float derivative12_a = q_scaler[n] * gnp[n] * unit12_a;
            const float derivative13_a =
              q_scaler[n] * other_radial_derivative[n] * unit13_a;
            for (int np = 0; np <= n_max_r; ++np) {
              value_12_13 += Hqq1[n * annmb.dim + np] * derivative12_a *
                q_scaler[np] * other_radial_derivative[np] * unit13_b;
              value_13_12 += Hqq1[n * annmb.dim + np] * derivative13_a *
                q_scaler[np] * gnp[np] * unit12_b;
            }
          }
          const int row12 = a * N + n2;
          const int row13 = a * N + n3;
          const int column13 = b * N + n3;
          const int column12 = b * N + n2;
          sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
            g_sparse_columns, N, row12, column13, (double)value_12_13);
          sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
            g_sparse_columns, N, row13, column12, (double)value_13_12);
        }
      }
    }
  }
}

static __global__ void find_zbl_hessian_kernel(
  const NEP::ParaMB paramb,
  const NEP::ZBL zbl,
  const int N,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_r12_x,
  const float* g_r12_y,
  const float* g_r12_z,
  const int* g_sparse_row_offsets,
  const int* g_sparse_columns,
  double* g_hessian)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) return;


  for (int neighbor_index = 0; neighbor_index < g_NN[n1]; ++neighbor_index) {
    const int n2 = g_NL[n1 + N * neighbor_index];
    float r12[3];
    load_neighbor_vector(
      box, N, n1, neighbor_index, n2, g_x, g_y, g_z,
      g_r12_x, g_r12_y, g_r12_z, r12);
    const float d12 = sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);

    float energy;
    float energy_first;
    float energy_second;
    find_zbl_energy_derivatives(
      paramb, zbl, g_type[n1], g_type[n2], d12, energy, energy_first, energy_second);
    if (energy == 0.0f && energy_first == 0.0f && energy_second == 0.0f) continue;

    const float distanceinv = 1.0f / d12;
    for (int a = 0; a < 3; ++a) {
      const float unit_a = r12[a] * distanceinv;
      for (int b = 0; b < 3; ++b) {
        const float unit_b = r12[b] * distanceinv;
        const float delta = a == b ? 1.0f : 0.0f;
        const float value = 0.5f *
          (energy_second * unit_a * unit_b +
           energy_first * distanceinv * (delta - unit_a * unit_b));
        const int row_center = a * N + n1;
        const int column_center = b * N + n1;
        const int row_neighbor = a * N + n2;
        const int column_neighbor = b * N + n2;
        sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
          g_sparse_columns, N, row_center, column_center, (double)value);
        sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
          g_sparse_columns, N, row_neighbor, column_neighbor, (double)value);
        sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
          g_sparse_columns, N, row_center, column_neighbor, (double)-value);
        sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
          g_sparse_columns, N, row_neighbor, column_center, (double)-value);
      }
    }
  }
}

// ============================================================================
// GPU kernel: compute per-atom ANN Hessian d2E/dq2
// Hqq[n1, i, j] = sum_n w1[type1, n] * (-2*tanh(z)*sech^2(z)) * w0[type1, n, i] * w0[type1, n, j]
// ============================================================================
static __global__ void compute_ann_hessian_kernel(
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int N,
  const int* g_type,
  const float* g_q_desc,  // stored descriptors [d*N + n1]
  float* g_Hqq)           // N * dim * dim
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) return;

  int t1 = g_type[n1];
  int dim = annmb.dim;
  int N_neu = annmb.num_neurons1;

  // g_q_desc stores the descriptor after the force kernel has applied
  // q_scaler, i.e. exactly the ANN input.  Do not scale it a second time.
  float q[MAX_DIM];
  for (int d = 0; d < dim; ++d)
    q[d] = g_q_desc[d * N + n1];

  // Compute ANN Hessian:
  // d2E/dqi*dqj = sum_n w1[n] * (-2*tanh(z_n)*sech^2(z_n)) * w0[n,i] * w0[n,j]
  // z_n = sum_d w0[n,d]*q[d] - b0[n]
  for (int i = 0; i < dim; ++i)
    for (int j = 0; j < dim; ++j)
      g_Hqq[(n1 * dim + i) * dim + j] = 0.0f;

  for (int n = 0; n < N_neu; ++n) {
    float z = 0.0f;
    for (int d = 0; d < dim; ++d)
      z += annmb.w0[t1][n * dim + d] * q[d];
    z -= annmb.b0[t1][n];
    float th = tanhf(z);
    float sech2 = 1.0f - th * th;
    float factor = annmb.w1[t1][n] * (-2.0f * th * sech2);

    for (int i = 0; i < dim; ++i) {
      float wi = annmb.w0[t1][n * dim + i] * factor;
      for (int j = i; j < dim; ++j) {
        float val = wi * annmb.w0[t1][n * dim + j];
        g_Hqq[(n1 * dim + i) * dim + j] += val;
        if (i != j)
          g_Hqq[(n1 * dim + j) * dim + i] += val;
      }
    }
  }
}


static __device__ __forceinline__ size_t angular_gradient_cache_index(
  const int edge,
  const int angular_descriptor_count,
  const int angular_descriptor,
  const int axis)
{
  return (static_cast<size_t>(edge) * angular_descriptor_count +
          angular_descriptor) * 3 + axis;
}

static __device__ __forceinline__ size_t angular_local_hessian_cache_index(
  const int edge,
  const int angular_descriptor_count,
  const int angular_descriptor,
  const int axis_a,
  const int axis_b)
{
  return (static_cast<size_t>(edge) * angular_descriptor_count +
          angular_descriptor) * 9 + axis_a * 3 + axis_b;
}

static __device__ __forceinline__ size_t angular_s_gradient_cache_index(
  const int edge,
  const int n_mp1,
  const int num_s_terms,
  const int angular_n,
  const int component,
  const int axis)
{
  return ((static_cast<size_t>(edge) * n_mp1 + angular_n) * num_s_terms +
          component) * 3 + axis;
}

// Prepare each center-neighbor-n edge once for both angular pair assembly
// and radial-angular contractions.
template <int S_CAPACITY>
static __global__ void prepare_angular_neighbor_derivatives_kernel(
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int N,
  const int center_begin,
  const int center_count,
  const int max_neighbors,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_r12_x,
  const float* g_r12_y,
  const float* g_r12_z,
  const float* g_sum_fxyz,
  float* g_neighbor_gradient,
  float* g_neighbor_s_gradient,
  float* g_local_hessian,
  int* g_valid)
{
  const int work_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_mp1 = paramb.n_max_angular + 1;
  const int edge_count = center_count * max_neighbors;
  if (work_index >= edge_count * n_mp1) return;

  const int edge = work_index / n_mp1;
  const int angular_n = work_index - edge * n_mp1;
  const int local_center = edge / max_neighbors;
  const int neighbor_slot = edge - local_center * max_neighbors;
  const int center = center_begin + local_center;
  if (angular_n == 0) g_valid[edge] = 0;
  if (neighbor_slot >= g_NN_angular[center]) return;

  const int neighbor = g_NL_angular[center + N * neighbor_slot];
  float r12[3];
  load_neighbor_vector(
    box, N, center, neighbor_slot, neighbor, g_x, g_y, g_z,
    g_r12_x, g_r12_y, g_r12_z, r12);
  const float distance =
    sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);

  const int num_s_terms =
    (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
  const int angular_descriptor_count = paramb.num_L * n_mp1;
  float sum_fxyz[S_CAPACITY] = {};
  float s_gradient[S_CAPACITY][3] = {};
  float s_hessian[S_CAPACITY][3][3] = {};
  float neighbor_gradient[MAX_ANGULAR_DESCRIPTORS][3] = {};
  float local_hessian[MAX_ANGULAR_DESCRIPTORS][3][3] = {};

  for (int component = 0; component < num_s_terms; ++component)
    sum_fxyz[component] = g_sum_fxyz[
      (angular_n * num_s_terms + component) * N + center];

  compute_angular_s_derivatives_for_n(
    paramb, annmb, center, neighbor, angular_n, g_type, r12, distance,
    s_gradient, s_hessian);
  for (int descriptor = 0; descriptor < paramb.num_L; ++descriptor) {
    compute_angular_q_derivatives_for_n(
      paramb, descriptor, sum_fxyz,
      s_gradient, s_hessian,
      neighbor_gradient[descriptor], local_hessian[descriptor]);
    const int angular_descriptor = descriptor * n_mp1 + angular_n;
    for (int axis = 0; axis < 3; ++axis) {
      g_neighbor_gradient[angular_gradient_cache_index(
        edge, angular_descriptor_count, angular_descriptor, axis)] =
        neighbor_gradient[descriptor][axis];
      for (int other_axis = 0; other_axis < 3; ++other_axis) {
        g_local_hessian[angular_local_hessian_cache_index(
          edge, angular_descriptor_count, angular_descriptor, axis,
          other_axis)] = local_hessian[descriptor][axis][other_axis];
      }
    }
  }
  for (int component = 0; component < num_s_terms; ++component)
    for (int axis = 0; axis < 3; ++axis)
      g_neighbor_s_gradient[angular_s_gradient_cache_index(
        edge, n_mp1, num_s_terms, angular_n, component, axis)] =
        s_gradient[component][axis];
  if (angular_n == 0) g_valid[edge] = 1;
}

static __global__ void assemble_angular_neighbor_pairs_kernel(
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int N,
  const int center_begin,
  const int center_count,
  const int max_neighbors,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const float* g_Fp,
  const float* g_sum_fxyz,
  const float* g_Hqq,
  const float* g_neighbor_gradient,
  const float* g_neighbor_s_gradient,
  const float* g_local_hessian,
  const int* g_valid,
  const int* g_sparse_row_offsets,
  const int* g_sparse_columns,
  double* g_hessian)
{
  const int work_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int pairs_per_center = max_neighbors * max_neighbors;
  const int work_count = center_count * pairs_per_center;
  if (work_index >= work_count) return;

  const int local_center = work_index / pairs_per_center;
  const int pair_index = work_index - local_center * pairs_per_center;
  const int first_slot = pair_index / max_neighbors;
  const int second_slot = pair_index - first_slot * max_neighbors;
  const int center = center_begin + local_center;
  const int neighbor_count = g_NN_angular[center];
  if (first_slot >= neighbor_count || second_slot >= neighbor_count) return;

  const int first_edge = local_center * max_neighbors + first_slot;
  const int second_edge = local_center * max_neighbors + second_slot;
  if (!g_valid[first_edge] || !g_valid[second_edge]) return;

  const int first_neighbor = g_NL_angular[center + N * first_slot];
  const int second_neighbor = g_NL_angular[center + N * second_slot];
  const int n_mp1 = paramb.n_max_angular + 1;
  const int n_max_radial = paramb.n_max_radial;
  const int num_s_terms =
    (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
  const int angular_descriptor_count = paramb.num_L * n_mp1;
  const float* q_scaler = annmb.q_scaler;
  const float* Hqq =
    g_Hqq + static_cast<size_t>(center) * annmb.dim * annmb.dim;
  const bool same_neighbor = first_slot == second_slot;

  auto cached_gradient = [=](
                           const int edge,
                           const int descriptor,
                           const int n,
                           const int axis) {
    return g_neighbor_gradient[angular_gradient_cache_index(
      edge, angular_descriptor_count, descriptor * n_mp1 + n, axis)];
  };
  auto cached_local_hessian = [=](
                                const int edge,
                                const int descriptor,
                                const int n,
                                const int axis_a,
                                const int axis_b) {
    return g_local_hessian[angular_local_hessian_cache_index(
      edge, angular_descriptor_count, descriptor * n_mp1 + n,
      axis_a, axis_b)];
  };
  auto cached_s_gradient_base = [=](const int edge, const int n) {
    const size_t offset = angular_s_gradient_cache_index(
      edge, n_mp1, num_s_terms, n, 0, 0);
    return reinterpret_cast<const float (*)[3]>(
      g_neighbor_s_gradient + offset);
  };
  auto descriptor_fp = [&](const int descriptor, const int n) {
    return g_Fp[angular_fp_index(
      n_max_radial, n_mp1, descriptor, n, center, N)];
  };

  for (int axis_a = 0; axis_a < 3; ++axis_a) {
    for (int axis_b = 0; axis_b < 3; ++axis_b) {
      float value = 0.0f;
      for (int descriptor = 0; descriptor < paramb.num_L; ++descriptor) {
        for (int n = 0; n < n_mp1; ++n) {
          const float fp = descriptor_fp(descriptor, n);
          if (same_neighbor) {
            value += fp * cached_local_hessian(
                first_edge, descriptor, n, axis_a, axis_b);
          } else {
            const float* sum_fxyz = g_sum_fxyz +
              static_cast<size_t>(n * num_s_terms) * N + center;
            value += fp * contract_q_s_hessian(
              paramb, descriptor, sum_fxyz, N,
              cached_s_gradient_base(first_edge, n), axis_a,
              cached_s_gradient_base(second_edge, n), axis_b);
          }
        }
      }

      for (int descriptor = 0; descriptor < paramb.num_L; ++descriptor) {
        for (int n = 0; n < n_mp1; ++n) {
          const int descriptor_i =
            n_max_radial + 1 + descriptor * n_mp1 + n;
          const float gradient_i = q_scaler[descriptor_i] *
            cached_gradient(first_edge, descriptor, n, axis_a);
          for (int other_descriptor = 0;
               other_descriptor < paramb.num_L; ++other_descriptor) {
            for (int other_n = 0; other_n < n_mp1; ++other_n) {
              const int descriptor_j = n_max_radial + 1 +
                other_descriptor * n_mp1 + other_n;
              value += Hqq[descriptor_i * annmb.dim + descriptor_j] *
                gradient_i * q_scaler[descriptor_j] *
                cached_gradient(
                  second_edge, other_descriptor, other_n, axis_b);
            }
          }
        }
      }

      const double contribution = static_cast<double>(value);
      sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
        g_sparse_columns, N, axis_a * N + first_neighbor,
        axis_b * N + second_neighbor, contribution);
      sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
        g_sparse_columns, N, axis_a * N + center,
        axis_b * N + second_neighbor, -contribution);
      sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
        g_sparse_columns, N, axis_a * N + first_neighbor,
        axis_b * N + center, -contribution);
      sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
        g_sparse_columns, N, axis_a * N + center,
        axis_b * N + center, contribution);
    }
  }
}

static __global__ void assemble_angular_radial_cross_kernel(
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int N,
  const int center_begin,
  const int center_count,
  const int max_neighbors,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_type,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_r12_radial_x,
  const float* g_r12_radial_y,
  const float* g_r12_radial_z,
  const float* g_Hqq,
  const float* g_neighbor_gradient,
  const int* g_valid,
  const int* g_sparse_row_offsets,
  const int* g_sparse_columns,
  double* g_hessian)
{
  const int local_center = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_center >= center_count) return;
  const int n1 = center_begin + local_center;
  const int neighbor_count = g_NN_angular[n1];
  const int n_mp1 = paramb.n_max_angular + 1;
  const int n_max_radial = paramb.n_max_radial;
  const int angular_descriptor_count = paramb.num_L * n_mp1;
  const float* q_scaler = annmb.q_scaler;
  const float* Hqq =
    g_Hqq + static_cast<size_t>(n1) * annmb.dim * annmb.dim;

  float center_gradient[MAX_ANGULAR_DESCRIPTORS][MAX_NUM_N][3] = {};
  float radial_center_gradient[MAX_NUM_N][3] = {};
  float radial_derivative[MAX_NUM_N] = {};
  float radial_source_gradient[MAX_NUM_N][3] = {};
  float angular_source_gradient[MAX_ANGULAR_DESCRIPTORS][MAX_NUM_N][3] = {};

  auto edge_index = [=](const int neighbor_slot) {
    return local_center * max_neighbors + neighbor_slot;
  };
  auto cached_gradient = [=](
                           const int edge,
                           const int descriptor,
                           const int n,
                           const int axis) {
    return g_neighbor_gradient[angular_gradient_cache_index(
      edge, angular_descriptor_count, descriptor * n_mp1 + n, axis)];
  };


  for (int radial_index = 0; radial_index < g_NN_radial[n1]; ++radial_index) {
    const int radial_neighbor = g_NL_radial[n1 + N * radial_index];
    float r12[3];
    load_neighbor_vector(
      box, N, n1, radial_index, radial_neighbor, g_x, g_y, g_z,
      g_r12_radial_x, g_r12_radial_y, g_r12_radial_z, r12);
    const float d12 =
      sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    const float d12inv = 1.0f / d12;
    compute_radial_neighbor_derivatives(
      paramb, annmb, g_type[n1], g_type[radial_neighbor], r12, d12,
      radial_derivative);
    for (int n = 0; n <= n_max_radial; ++n)
      for (int axis = 0; axis < 3; ++axis)
        radial_center_gradient[n][axis] -=
          radial_derivative[n] * r12[axis] * d12inv;
  }

  for (int neighbor_slot = 0; neighbor_slot < neighbor_count;
       ++neighbor_slot) {
    const int edge = edge_index(neighbor_slot);
    if (!g_valid[edge]) continue;
    for (int descriptor = 0; descriptor < paramb.num_L; ++descriptor) {
      for (int n = 0; n < n_mp1; ++n) {
        for (int axis = 0; axis < 3; ++axis) {
          const float gradient =
            cached_gradient(edge, descriptor, n, axis);
          center_gradient[descriptor][n][axis] -= gradient;
        }
      }
    }
  }


  // Radial-angular terms use the same cached angular Jacobian as the
  // neighbor-pair assembly.
  for (int radial_source = 0; radial_source <= g_NN_radial[n1];
       ++radial_source) {
    int source_atom = n1;
    if (radial_source == 0) {
      for (int n = 0; n <= n_max_radial; ++n)
        for (int axis = 0; axis < 3; ++axis)
          radial_source_gradient[n][axis] =
            radial_center_gradient[n][axis];
    } else {
      const int radial_slot = radial_source - 1;
      source_atom = g_NL_radial[n1 + N * radial_slot];
      float r12[3];
      load_neighbor_vector(
        box, N, n1, radial_slot, source_atom, g_x, g_y, g_z,
        g_r12_radial_x, g_r12_radial_y, g_r12_radial_z, r12);
      const float d12 =
        sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      const float d12inv = 1.0f / d12;
      compute_radial_neighbor_derivatives(
        paramb, annmb, g_type[n1], g_type[source_atom], r12, d12,
        radial_derivative);
      for (int n = 0; n <= n_max_radial; ++n)
        for (int axis = 0; axis < 3; ++axis)
          radial_source_gradient[n][axis] =
            radial_derivative[n] * r12[axis] * d12inv;
    }

    for (int angular_source = 0; angular_source <= neighbor_count;
         ++angular_source) {
      int target_atom = n1;
      if (angular_source == 0) {
        for (int descriptor = 0; descriptor < paramb.num_L; ++descriptor)
          for (int n = 0; n < n_mp1; ++n)
            for (int axis = 0; axis < 3; ++axis)
              angular_source_gradient[descriptor][n][axis] =
                center_gradient[descriptor][n][axis];
      } else {
        const int neighbor_slot = angular_source - 1;
        const int edge = edge_index(neighbor_slot);
        if (!g_valid[edge]) continue;
        target_atom = g_NL_angular[n1 + N * neighbor_slot];
        for (int descriptor = 0; descriptor < paramb.num_L; ++descriptor)
          for (int n = 0; n < n_mp1; ++n)
            for (int axis = 0; axis < 3; ++axis)
              angular_source_gradient[descriptor][n][axis] =
                cached_gradient(edge, descriptor, n, axis);
      }

      for (int axis_a = 0; axis_a < 3; ++axis_a) {
        for (int axis_b = 0; axis_b < 3; ++axis_b) {
          float value = 0.0f;
          for (int n = 0; n <= n_max_radial; ++n) {
            const float radial_i =
              q_scaler[n] * radial_source_gradient[n][axis_a];
            for (int descriptor = 0; descriptor < paramb.num_L; ++descriptor) {
              const int angular_base =
                n_max_radial + 1 + descriptor * n_mp1;
              for (int angular_n = 0; angular_n < n_mp1; ++angular_n) {
                const int angular_descriptor = angular_base + angular_n;
                value += Hqq[n * annmb.dim + angular_descriptor] * radial_i *
                  q_scaler[angular_descriptor] *
                  angular_source_gradient[descriptor][angular_n][axis_b];
              }
            }
          }

          const int row = axis_a * N + source_atom;
          const int column = axis_b * N + target_atom;
          sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
            g_sparse_columns, N, row, column, static_cast<double>(value));
          // Hqq is symmetric.  Scattering the transpose of each sparse
          // radial-angular outer product covers the angular-radial block
          // without assuming one periodic image per atom.
          sparse_hessian::atomic_add_any(g_hessian, g_sparse_row_offsets,
            g_sparse_columns, N, column, row, static_cast<double>(value));
        }
      }
    }
  }
}

namespace {

class AtomStateBackup
{
public:
  explicit AtomStateBackup(Atom& atom)
    : position_(atom.position_per_atom.data(), atom.position_per_atom.size()),
      force_(atom.force_per_atom.data(), atom.force_per_atom.size()),
      potential_(atom.potential_per_atom.data(), atom.potential_per_atom.size()),
      virial_(atom.virial_per_atom.data(), atom.virial_per_atom.size())
  {
  }

  bool save()
  {
    position_host_.resize(position_.second);
    force_host_.resize(force_.second);
    potential_host_.resize(potential_.second);
    virial_host_.resize(virial_.second);
    return copy_to_host(position_.first, position_host_.data(), position_.second) &&
           copy_to_host(force_.first, force_host_.data(), force_.second) &&
           copy_to_host(potential_.first, potential_host_.data(), potential_.second) &&
           copy_to_host(virial_.first, virial_host_.data(), virial_.second);
  }

  bool restore()
  {
    return copy_to_device(force_host_.data(), force_.first, force_.second) &&
           copy_to_device(potential_host_.data(), potential_.first, potential_.second) &&
           copy_to_device(virial_host_.data(), virial_.first, virial_.second) &&
           copy_to_device(position_host_.data(), position_.first, position_.second);
  }

  const std::vector<double>& positions() const { return position_host_; }

private:
  using DeviceArray = std::pair<double*, size_t>;

  static bool copy_to_host(
    const void* source, void* destination, const size_t count)
  {
    if (count == 0) return true;
    const gpuError_t error = gpuMemcpy(
      destination, source, count * sizeof(double), gpuMemcpyDeviceToHost);
    if (error != gpuSuccess) {
      fprintf(
        stderr, "Failed to restore Atom state: %s.\n",
        gpuGetErrorString(error));
      return false;
    }
    return true;
  }

  static bool copy_to_device(
    const void* source, void* destination, const size_t count)
  {
    if (count == 0) return true;
    const gpuError_t error = gpuMemcpy(
      destination, source, count * sizeof(double), gpuMemcpyHostToDevice);
    if (error != gpuSuccess) {
      fprintf(
        stderr, "Failed to restore Atom state: %s.\n",
        gpuGetErrorString(error));
      return false;
    }
    return true;
  }

  DeviceArray position_;
  DeviceArray force_;
  DeviceArray potential_;
  DeviceArray virial_;
  std::vector<double> position_host_;
  std::vector<double> force_host_;
  std::vector<double> potential_host_;
  std::vector<double> virial_host_;
};

} // namespace

// ============================================================================
// Main entry: NEP analytic Hessian for explicitly supported model classes
// ============================================================================
bool compute_nep_analytic_hessian(
  Force& force,
  NEP_Hessian_Model& model,
  Box& box,
  Atom& atom,
  std::vector<double>& hessian_out,
  std::vector<double>& force_out,
  double& pe_out,
  nep_hessian::Diagnostics& diagnostics,
  std::string& error,
  sparse_hessian::BlockPattern* sparse_pattern)
{
  error.clear();
  hessian_out.clear();
  force_out.clear();
  pe_out = 0.0;
  const int N = atom.number_of_atoms;
  auto fail = [&](const char* reason) { error = reason; return false; };
  if (N <= 0 || N > std::numeric_limits<int>::max() / 3)
    return fail("Analytic Hessian: atom count exceeds supported index range.");
  const int N3 = 3 * N;
  if (force.get_number_of_potentials() != 1)
    return fail("Analytic Hessian requires exactly one potential.");
  auto* nep = dynamic_cast<NEP*>(&force.get_potential(0));
  if (nep == nullptr || (nep->nep_model_type != 0 && nep->nep_model_type != 3) ||
      nep->ilp_flag != 0)
    return fail("Analytic Hessian requires ordinary single-GPU NEP or temperature NEP without ILP.");
  const bool use_expanded_box = model_requires_expanded_box(model, box);
  if (model.paramb.L_max < 0 || model.paramb.L_max > 8 ||
      (model.paramb.L_max + 1) * (model.paramb.L_max + 1) - 1 >
        MAX_S_TERMS) {
    printf(
      "NEP analytic Hessian: unsupported L_max=%d (supported range is 0..8).\n",
      model.paramb.L_max);
    return fail("Unsupported analytic Hessian angular expansion (L_max).");
  }
  if (model.paramb.n_max_radial < 0 ||
      model.paramb.n_max_radial + 1 > MAX_NUM_N ||
      model.paramb.n_max_angular < 0 ||
      model.paramb.n_max_angular + 1 > MAX_NUM_N ||
      model.paramb.num_L < 0 ||
      model.paramb.num_L > MAX_ANGULAR_DESCRIPTORS) {
    return fail("Analytic Hessian descriptor orders exceed kernel limits.");
  }
  const int radial_descriptor_count = model.paramb.n_max_radial + 1;
  const int angular_descriptor_stride = model.paramb.n_max_angular + 1;
  if (model.paramb.num_L > 0 && angular_descriptor_stride >
      std::numeric_limits<int>::max() / model.paramb.num_L) {
    return fail("Analytic Hessian angular descriptor count exceeds index range.");
  }
  const int angular_descriptor_count =
    model.paramb.num_L * angular_descriptor_stride;
  if (radial_descriptor_count >
      std::numeric_limits<int>::max() - angular_descriptor_count) {
    return fail("Analytic Hessian descriptor count exceeds index range.");
  }
  const int expected_descriptor_count = radial_descriptor_count +
    angular_descriptor_count + (nep->nep_model_type == 3 ? 1 : 0);
  if (model.annmb.dim <= 0 || model.annmb.dim > MAX_DIM ||
      model.annmb.dim != expected_descriptor_count ||
      expected_descriptor_count > std::numeric_limits<int>::max() / N ||
      N > std::numeric_limits<int>::max() / expected_descriptor_count ||
      N * expected_descriptor_count >
        std::numeric_limits<int>::max() / expected_descriptor_count) {
    printf(
      "NEP analytic Hessian: descriptor dimension mismatch "
      "(annmb.dim=%d, expected=%d).\n",
      model.annmb.dim, expected_descriptor_count);
    return fail("Analytic Hessian descriptor dimensions are inconsistent or exceed kernel index limits.");
  }
  // Only the dense kernels address a full (3N)^2 matrix using int indices.
  if (sparse_pattern == nullptr && N3 > std::numeric_limits<int>::max() / N3) {
    printf("NEP analytic Hessian: dense matrix exceeds supported index range.\n");
    return fail("Analytic Hessian dense matrix exceeds supported index range.");
  }
  const size_t expected_position_size = static_cast<size_t>(N3);
  const size_t expected_atom_size = static_cast<size_t>(N);
  if (atom.position_per_atom.size() != expected_position_size ||
      atom.force_per_atom.size() != expected_position_size ||
      atom.potential_per_atom.size() != expected_atom_size ||
      atom.virial_per_atom.size() != 9 * expected_atom_size ||
      atom.type.size() != expected_atom_size) {
    printf("NEP analytic Hessian: Atom array dimensions are inconsistent.\n");
    return fail("Analytic Hessian Atom array dimensions are inconsistent.");
  }

  AtomStateBackup atom_backup(atom);
  if (!atom_backup.save()) {
    printf("NEP analytic Hessian: failed to save Atom state.\n");
    return fail("Analytic Hessian failed to save Atom state.");
  }

  auto finish = [&](bool success) {
    // Restore the caller geometry and rebuild force-owned caches.
    bool state_restored = atom_backup.restore();
    if (state_restored) {
      // Refresh caches at the original geometry, then restore the saved outputs.
      std::vector<Group> empty_groups;
      force.compute(
        box, atom.position_per_atom, atom.type, empty_groups,
        atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);
      const gpuError_t error = gpuDeviceSynchronize();
      if (error != gpuSuccess) {
        fprintf(
          stderr, "Failed to refresh NEP state: %s.\n",
          gpuGetErrorString(error));
        state_restored = false;
      }
    }
    state_restored = atom_backup.restore() && state_restored;
    if (!state_restored) {
      hessian_out.clear();
      force_out.clear();
      pe_out = 0.0;
      return fail("Analytic Hessian failed to restore Atom state or refresh force caches.");
    }
    return success;
  };

  // The only supported geometry is Atom's current state. No upload is needed.
  for (const double coordinate : atom_backup.positions())
    if (!std::isfinite(coordinate)) {
      return fail("Analytic Hessian found a non-finite coordinate.");
    }

  // 1. Run normal NEP force computation to get Fp, sum_fxyz, forces
  std::vector<Group> empty_groups;
  force.compute(
    box, atom.position_per_atom, atom.type, empty_groups,
    atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);
  CHECK(gpuDeviceSynchronize());

  GPU_Vector<int>& radial_neighbor_counts_gpu = use_expanded_box
    ? nep->small_box_data.NN_radial
    : nep->nep_data.NN_radial;
  GPU_Vector<int>& radial_neighbor_list_gpu = use_expanded_box
    ? nep->small_box_data.NL_radial
    : nep->nep_data.NL_radial;
  GPU_Vector<int>& angular_neighbor_counts_gpu = use_expanded_box
    ? nep->small_box_data.NN_angular
    : nep->nep_data.NN_angular;
  GPU_Vector<int>& angular_neighbor_list_gpu = use_expanded_box
    ? nep->small_box_data.NL_angular
    : nep->nep_data.NL_angular;

  // Hessian kernels use int offsets into the neighbor lists, including in
  // sparse mode where the dense matrix size no longer bounds N.
  const size_t max_neighbor_entries = std::numeric_limits<int>::max();
  if (radial_neighbor_list_gpu.size() > max_neighbor_entries ||
      angular_neighbor_list_gpu.size() > max_neighbor_entries) {
    printf("NEP analytic Hessian: neighbor lists exceed supported index range.\n");
    error = "Analytic Hessian neighbor lists exceed supported index range.";
    return finish(false);
  }

  const float* radial_r12_x = nullptr;
  const float* radial_r12_y = nullptr;
  const float* radial_r12_z = nullptr;
  const float* angular_r12_x = nullptr;
  const float* angular_r12_y = nullptr;
  const float* angular_r12_z = nullptr;
  if (radial_neighbor_counts_gpu.size() != static_cast<size_t>(N) ||
      angular_neighbor_counts_gpu.size() != static_cast<size_t>(N) ||
      radial_neighbor_list_gpu.size() % static_cast<size_t>(N) != 0 ||
      angular_neighbor_list_gpu.size() % static_cast<size_t>(N) != 0) {
    printf("NEP analytic Hessian: neighbor-list dimensions are invalid.\n");
    error = "Analytic Hessian neighbor-list dimensions are invalid.";
    return finish(false);
  }
  if (use_expanded_box) {
    if (nep->small_box_data.r12.size() % 6 != 0) {
      printf("NEP analytic Hessian: expanded-box displacement cache is invalid.\n");
      error = "Analytic Hessian expanded-box displacement cache is invalid.";
      return finish(false);
    }
    const size_t r12_plane_size = nep->small_box_data.r12.size() / 6;
    if (r12_plane_size < radial_neighbor_list_gpu.size() ||
        r12_plane_size < angular_neighbor_list_gpu.size()) {
      printf("NEP analytic Hessian: expanded-box displacement cache is too small.\n");
      error = "Analytic Hessian expanded-box displacement cache is too small.";
      return finish(false);
    }
    radial_r12_x = nep->small_box_data.r12.data();
    radial_r12_y = radial_r12_x + r12_plane_size;
    radial_r12_z = radial_r12_y + r12_plane_size;
    angular_r12_x = radial_r12_z + r12_plane_size;
    angular_r12_y = angular_r12_x + r12_plane_size;
    angular_r12_z = angular_r12_y + r12_plane_size;
  }

  std::vector<int> angular_neighbor_counts(N);
  angular_neighbor_counts_gpu.copy_to_host(angular_neighbor_counts.data(), N);
  std::vector<int> radial_neighbor_counts(N);
  radial_neighbor_counts_gpu.copy_to_host(radial_neighbor_counts.data(), N);
  const size_t radial_neighbor_capacity =
    radial_neighbor_list_gpu.size() / static_cast<size_t>(N);
  const size_t angular_neighbor_capacity =
    angular_neighbor_list_gpu.size() / static_cast<size_t>(N);
  int max_angular_neighbors = 0;
  for (int atom_index = 0; atom_index < N; ++atom_index) {
    const int radial_count = radial_neighbor_counts[atom_index];
    if (radial_count < 0 ||
        static_cast<size_t>(radial_count) > radial_neighbor_capacity) {
      printf(
        "NEP analytic Hessian: invalid radial neighbor count=%d "
        "for atom=%d (capacity=%zu).\n",
        radial_count, atom_index, radial_neighbor_capacity);
      error = "Analytic Hessian radial neighbor count exceeds capacity.";
      return finish(false);
    }
    const int count = angular_neighbor_counts[atom_index];
    if (count < 0 || static_cast<size_t>(count) > angular_neighbor_capacity) {
      printf(
        "NEP analytic Hessian: invalid angular neighbor count=%d "
        "for atom=%d (capacity=%zu).\n",
        count, atom_index, angular_neighbor_capacity);
      error = "Analytic Hessian angular neighbor count exceeds capacity.";
      return finish(false);
    }
    max_angular_neighbors = std::max(max_angular_neighbors, count);
  }
  // Check both lists before allocating the Hessian or evaluating derivatives.
  GPU_Vector<int> invalid_neighbors_gpu(1);
  int invalid_neighbors = 0;
  invalid_neighbors_gpu.copy_from_host(&invalid_neighbors, 1);
  const int neighbor_grid = (N - 1) / BLOCK_SIZE + 1;
  validate_hessian_neighbors<<<neighbor_grid, BLOCK_SIZE>>>(
    N, box, radial_neighbor_counts_gpu.data(), radial_neighbor_list_gpu.data(),
    atom.position_per_atom.data(), atom.position_per_atom.data() + N,
    atom.position_per_atom.data() + 2 * N,
    radial_r12_x, radial_r12_y, radial_r12_z, invalid_neighbors_gpu.data());
  GPU_CHECK_KERNEL
  validate_hessian_neighbors<<<neighbor_grid, BLOCK_SIZE>>>(
    N, box, angular_neighbor_counts_gpu.data(), angular_neighbor_list_gpu.data(),
    atom.position_per_atom.data(), atom.position_per_atom.data() + N,
    atom.position_per_atom.data() + 2 * N,
    angular_r12_x, angular_r12_y, angular_r12_z, invalid_neighbors_gpu.data());
  GPU_CHECK_KERNEL
  invalid_neighbors_gpu.copy_to_host(&invalid_neighbors, 1);
  if (invalid_neighbors != 0) {
    if (invalid_neighbors & 1)
      error = "Analytic Hessian neighbor index is outside the atom range.";
    else if (invalid_neighbors & 2)
      error = "Analytic Hessian found a non-finite neighbor distance.";
    else
      error = "Analytic Hessian requires neighbor distances of at least 1e-6 Angstrom, "
              "including periodic images.";
    return finish(false);
  }
  // Reconstruct the normalized ANN input in command-owned storage.
  GPU_Vector<float> q_cache(static_cast<size_t>(model.annmb.dim) * N);
  const int descriptor_grid = (N - 1) / 64 + 1;
  if (use_expanded_box) {
    const size_t r12_plane_size = nep->small_box_data.r12.size() / 6;
    nep_hessian::find_descriptor_only_small_box<<<descriptor_grid, 64>>>(
      model.paramb, model.annmb, static_cast<float>(force.get_temperature()), N, 0, N,
      radial_neighbor_counts_gpu.data(), radial_neighbor_list_gpu.data(),
      angular_neighbor_counts_gpu.data(), angular_neighbor_list_gpu.data(),
      atom.type.data(),
      nep->small_box_data.r12.data(),
      nep->small_box_data.r12.data() + r12_plane_size,
      nep->small_box_data.r12.data() + 2 * r12_plane_size,
      nep->small_box_data.r12.data() + 3 * r12_plane_size,
      nep->small_box_data.r12.data() + 4 * r12_plane_size,
      nep->small_box_data.r12.data() + 5 * r12_plane_size,
      q_cache.data());
  } else {
    nep_hessian::find_descriptor_only<<<descriptor_grid, 64>>>(
      model.paramb, model.annmb, static_cast<float>(force.get_temperature()), N, 0, N, box,
      radial_neighbor_counts_gpu.data(), radial_neighbor_list_gpu.data(),
      angular_neighbor_counts_gpu.data(), angular_neighbor_list_gpu.data(),
      atom.type.data(),
      atom.position_per_atom.data(), atom.position_per_atom.data() + N,
      atom.position_per_atom.data() + 2 * N,
      q_cache.data());
  }
  GPU_CHECK_KERNEL

  // Extract force and PE
  std::vector<double> force_soa(N3);
  atom.force_per_atom.copy_to_host(force_soa.data());
  std::vector<double> pe_host(N);
  atom.potential_per_atom.copy_to_host(pe_host.data());
  pe_out = 0.0;
  for (int i = 0; i < N; ++i) pe_out += pe_host[i];

  force_out = std::move(force_soa);

  const int dim = model.annmb.dim;
  const int N_total = N;

  GPU_Vector<int> sparse_row_offsets_gpu;
  GPU_Vector<int> sparse_columns_gpu;
  const int* sparse_row_offsets = nullptr;
  const int* sparse_columns = nullptr;
  if (sparse_pattern != nullptr) {
    std::vector<int> radial_list(radial_neighbor_list_gpu.size());
    std::vector<int> angular_list(angular_neighbor_list_gpu.size());
    radial_neighbor_list_gpu.copy_to_host(radial_list.data());
    angular_neighbor_list_gpu.copy_to_host(angular_list.data());
    *sparse_pattern = sparse_hessian::build_pattern_from_neighbors(
      N, radial_neighbor_counts, radial_list, radial_neighbor_capacity,
      angular_neighbor_counts, angular_list, angular_neighbor_capacity);
    sparse_row_offsets_gpu.resize(sparse_pattern->row_offsets.size(), 0);
    sparse_columns_gpu.resize(sparse_pattern->columns.size(), 0);
    sparse_row_offsets_gpu.copy_from_host(sparse_pattern->row_offsets.data(),
      sparse_pattern->row_offsets.size());
    sparse_columns_gpu.copy_from_host(sparse_pattern->columns.data(),
      sparse_pattern->columns.size());
    sparse_row_offsets = sparse_row_offsets_gpu.data();
    sparse_columns = sparse_columns_gpu.data();
  }

  // 3. Allocate Hessian on GPU (zero-initialized)
  GPU_Vector<double> hessian_gpu(
    sparse_pattern != nullptr ? sparse_pattern->value_count() :
      static_cast<size_t>(N3) * N3, 0.0);

  // 4. Compute per-atom ANN Hessian from the current descriptor cache
  GPU_Vector<float> hqq_gpu(N_total * dim * dim, 0.0f);
  compute_ann_hessian_kernel<<<(N_total - 1) / 64 + 1, 64>>>(
    model.paramb, model.annmb, N_total, atom.type.data(),
    q_cache.data(), hqq_gpu.data());
  GPU_CHECK_KERNEL
  // 5. Compute radial Hessian
  int grid = (N_total - 1) / BLOCK_SIZE + 1;
  {
    find_radial_hessian_kernel<<<grid, BLOCK_SIZE>>>(
      model.paramb, model.annmb, N_total, 0, N_total, box,
      radial_neighbor_counts_gpu.data(), radial_neighbor_list_gpu.data(),
      atom.type.data(),
      atom.position_per_atom.data(), atom.position_per_atom.data() + N_total,
      atom.position_per_atom.data() + 2 * N_total,
      radial_r12_x, radial_r12_y, radial_r12_z,
      nep->nep_data.Fp.data(), hqq_gpu.data(),
      sparse_row_offsets, sparse_columns,
      hessian_gpu.data());
    GPU_CHECK_KERNEL
  }

  size_t original_stack_limit = 0;
  bool restore_stack_limit = false;
  if (angular_descriptor_count > 0) {
    CHECK(gpuDeviceGetLimit(&original_stack_limit, gpuLimitStackSize));
    // ptxas reports 5.3 KiB for the largest staged angular kernel on sm_86.
    // Keep headroom for the staged angular kernels.
    constexpr size_t kRequiredStackBytes = 16ULL * 1024;
    if (original_stack_limit < kRequiredStackBytes) {
      CHECK(gpuDeviceSetLimit(gpuLimitStackSize, kRequiredStackBytes));
      restore_stack_limit = true;
    }

    if (max_angular_neighbors > 0) {
      const int num_s_terms =
        (model.paramb.L_max + 1) * (model.paramb.L_max + 1) - 1;
      const int cached_angular_descriptors =
        model.paramb.num_L * (model.paramb.n_max_angular + 1);
      const size_t floats_per_edge =
        static_cast<size_t>(cached_angular_descriptors) * (3 + 9) +
        static_cast<size_t>(model.paramb.n_max_angular + 1) *
          num_s_terms * 3;
      const size_t bytes_per_edge =
        floats_per_edge * sizeof(float) + sizeof(int);
      const size_t bytes_per_center =
        bytes_per_edge * static_cast<size_t>(max_angular_neighbors);
      constexpr size_t kAngularWorkspaceBytes = 64ULL * 1024 * 1024;
      const size_t centers_by_workspace =
        bytes_per_center == 0 ? static_cast<size_t>(N_total) :
        std::max<size_t>(1, kAngularWorkspaceBytes / bytes_per_center);
      const int tile_centers = static_cast<int>(std::min<size_t>(
        static_cast<size_t>(N_total), centers_by_workspace));
      const size_t edge_capacity =
        static_cast<size_t>(tile_centers) * max_angular_neighbors;

      GPU_Vector<float> neighbor_gradient_cache(
        edge_capacity * cached_angular_descriptors * 3);
      GPU_Vector<float> neighbor_s_gradient_cache(
        edge_capacity * (model.paramb.n_max_angular + 1) * num_s_terms * 3);
      GPU_Vector<float> local_hessian_cache(
        edge_capacity * cached_angular_descriptors * 9);
      GPU_Vector<int> valid_cache(edge_capacity);

      for (int center_begin = 0; center_begin < N_total;
           center_begin += tile_centers) {
        const int center_count =
          std::min(tile_centers, N_total - center_begin);
        const int edge_count = center_count * max_angular_neighbors;
        const int prepare_work_count =
          edge_count * (model.paramb.n_max_angular + 1);
        if (num_s_terms <= LOW_L_MAX_S_TERMS) {
          prepare_angular_neighbor_derivatives_kernel<LOW_L_MAX_S_TERMS><<<
            (prepare_work_count - 1) / ANGULAR_PREPARE_BLOCK_SIZE + 1,
            ANGULAR_PREPARE_BLOCK_SIZE>>>(
            model.paramb, model.annmb, N_total, center_begin, center_count,
            max_angular_neighbors, box,
            angular_neighbor_counts_gpu.data(), angular_neighbor_list_gpu.data(),
            atom.type.data(),
            atom.position_per_atom.data(),
            atom.position_per_atom.data() + N_total,
            atom.position_per_atom.data() + 2 * N_total,
            angular_r12_x, angular_r12_y, angular_r12_z,
            nep->nep_data.sum_fxyz.data(), neighbor_gradient_cache.data(),
            neighbor_s_gradient_cache.data(), local_hessian_cache.data(),
            valid_cache.data());
        } else {
          prepare_angular_neighbor_derivatives_kernel<MAX_S_TERMS><<<
            (prepare_work_count - 1) / ANGULAR_PREPARE_BLOCK_SIZE + 1,
            ANGULAR_PREPARE_BLOCK_SIZE>>>(
            model.paramb, model.annmb, N_total, center_begin, center_count,
            max_angular_neighbors, box,
            angular_neighbor_counts_gpu.data(), angular_neighbor_list_gpu.data(),
            atom.type.data(),
            atom.position_per_atom.data(),
            atom.position_per_atom.data() + N_total,
            atom.position_per_atom.data() + 2 * N_total,
            angular_r12_x, angular_r12_y, angular_r12_z,
            nep->nep_data.sum_fxyz.data(), neighbor_gradient_cache.data(),
            neighbor_s_gradient_cache.data(), local_hessian_cache.data(),
            valid_cache.data());
        }
        GPU_CHECK_KERNEL

        const int pair_work_count = center_count * max_angular_neighbors *
          max_angular_neighbors;
        assemble_angular_neighbor_pairs_kernel<<<
          (pair_work_count - 1) / ANGULAR_PAIR_BLOCK_SIZE + 1,
          ANGULAR_PAIR_BLOCK_SIZE>>>(
          model.paramb, model.annmb, N_total, center_begin, center_count,
          max_angular_neighbors,
          angular_neighbor_counts_gpu.data(), angular_neighbor_list_gpu.data(),
          nep->nep_data.Fp.data(), nep->nep_data.sum_fxyz.data(),
          hqq_gpu.data(),
          neighbor_gradient_cache.data(), neighbor_s_gradient_cache.data(),
          local_hessian_cache.data(), valid_cache.data(),
          sparse_row_offsets, sparse_columns,
          hessian_gpu.data());
        GPU_CHECK_KERNEL

        assemble_angular_radial_cross_kernel<<<
          (center_count - 1) / ANGULAR_RADIAL_BLOCK_SIZE + 1,
          ANGULAR_RADIAL_BLOCK_SIZE>>>(
          model.paramb, model.annmb, N_total, center_begin, center_count,
          max_angular_neighbors, box,
          angular_neighbor_counts_gpu.data(), angular_neighbor_list_gpu.data(),
          radial_neighbor_counts_gpu.data(), radial_neighbor_list_gpu.data(),
          atom.type.data(),
          atom.position_per_atom.data(),
          atom.position_per_atom.data() + N_total,
          atom.position_per_atom.data() + 2 * N_total,
          radial_r12_x, radial_r12_y, radial_r12_z,
          hqq_gpu.data(), neighbor_gradient_cache.data(),
          valid_cache.data(), sparse_row_offsets, sparse_columns,
          hessian_gpu.data());
        GPU_CHECK_KERNEL
      }
      CHECK(gpuDeviceSynchronize());
    }
  }
  // The ZBL parameters come from the model that NEP_Hessian_Model loaded.
  const NEP::ZBL& zbl_state = model.zbl;
  if (zbl_state.enabled) {
    find_zbl_hessian_kernel<<<grid, BLOCK_SIZE>>>(
      model.paramb, zbl_state, N_total, box,
      angular_neighbor_counts_gpu.data(),
      angular_neighbor_list_gpu.data(),
      atom.type.data(),
      atom.position_per_atom.data(), atom.position_per_atom.data() + N_total,
      atom.position_per_atom.data() + 2 * N_total,
      angular_r12_x, angular_r12_y, angular_r12_z,
      sparse_row_offsets, sparse_columns,
      hessian_gpu.data());
    GPU_CHECK_KERNEL
  }
  CHECK(gpuDeviceSynchronize());
  if (restore_stack_limit)
    CHECK(gpuDeviceSetLimit(gpuLimitStackSize, original_stack_limit));

  // 6. Copy Hessian to host
  hessian_out.resize(sparse_pattern != nullptr
    ? sparse_pattern->value_count() : static_cast<size_t>(N3) * N3);
  hessian_gpu.copy_to_host(hessian_out.data());
  diagnostics = nep_hessian::inspect(hessian_out, N, sparse_pattern);
  if (!diagnostics.finite || diagnostics.asymmetry_frobenius_relative > 1.0e-6 ||
      diagnostics.asymmetry_max_relative > 1.0e-5 || diagnostics.translation_relative > 1.0e-5) {
    printf(
      "NEP analytic Hessian failed validation: finite=%d symmetry=%.12g "
      "symmetry_frobenius=%.12g translation=%.12g.\n",
      diagnostics.finite, diagnostics.asymmetry_max_relative,
      diagnostics.asymmetry_frobenius_relative, diagnostics.translation_relative);
    error = "Analytic Hessian matrix failed numerical diagnostics:";
    if (!diagnostics.finite) error += " non-finite matrix;";
    if (diagnostics.asymmetry_frobenius_relative > 1.0e-6 ||
        diagnostics.asymmetry_max_relative > 1.0e-5)
      error += " symmetry tolerance exceeded;";
    if (diagnostics.translation_relative > 1.0e-5)
      error += " translation tolerance exceeded;";
    return finish(false);
  }
  return finish(true);
}
