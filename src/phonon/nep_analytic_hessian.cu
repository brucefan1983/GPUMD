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
#include "force/nep.cuh"
#include "model/atom.cuh"
#include "model/group.cuh"
#include "utilities/error.cuh"
#include "utilities/nep_utilities.cuh"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
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

__device__ bool g_disable_s_hessian = false;
__device__ bool g_disable_q_s_hessian = false;
__device__ bool g_dump_s_derivatives = false;

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
  float* q_gradient;
  float* q_hessian_diagonal;
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
      q_gradient[first_variable] += first;
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
        if (first_variable == second_variable)
          q_hessian_diagonal[first_variable] += second;
        if (g_disable_q_s_hessian) continue;
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
  float* s,
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

  s[s_index] += g * value;
  for (int a = 0; a < 3; ++a) {
    s_gradient[s_index][a] += g_gradient[a] * value + g * value_gradient[a];
    for (int b = 0; b < 3; ++b)
      s_hessian[s_index][a][b] += g_hessian[a][b] * value +
        g_gradient[a] * value_gradient[b] + g_gradient[b] * value_gradient[a] +
        (g_disable_s_hessian ? 0.0f : g * value_hessian[a][b]);
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
  float* s,
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
      (angular_n * (paramb.basis_size_angular + 1) + basis) *
        paramb.num_types_sq +
      center_type * paramb.num_types + neighbor_type + paramb.num_c_radial;
    const float coefficient = annmb.c[c_index];
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
    s[component] = 0.0f;
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

  if (g_dump_s_derivatives && angular_n == 0) {
    const int debug_c_index = center_type * paramb.num_types + neighbor_type +
      paramb.num_c_radial;
    printf(
      "QINTERC %d %d %d %d %d %.10g\n",
      center, neighbor, debug_c_index, paramb.num_c_radial,
      paramb.basis_size_angular, annmb.c[debug_c_index]);
    printf(
      "QINTERP %d %d %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n",
      center, neighbor, g, g_gradient[0], g_gradient[1], g_gradient[2],
      g_hessian[0][0], g_hessian[0][2], g_hessian[2][2]);
    printf(
      "QINTERU %d %d %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n",
      center, neighbor, unit[0], unit[1], unit[2], unit_gradient[0][2],
      unit_gradient[2][2], unit_hessian[2][0][2],
      unit_hessian[2][2][2]);
    printf(
      "QINTERA %d %d %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n",
      center, neighbor, 1.0f, unit_gradient[0][0], unit_gradient[0][2],
      unit_gradient[2][0], unit_gradient[2][2], 0.0f, 0.0f);
  }

  for (int L = 1; L <= paramb.L_max; ++L) {
    for (int m = 0; m <= L; ++m) {
      add_angular_basis_hessian(
        L, m, false, g, g_gradient, g_hessian, unit, unit_gradient,
        unit_hessian, angular_s_index(L, m, false), s, s_gradient,
        s_hessian);
      if (m > 0)
        add_angular_basis_hessian(
          L, m, true, g, g_gradient, g_hessian, unit, unit_gradient,
          unit_hessian, angular_s_index(L, m, true), s, s_gradient,
          s_hessian);
    }
  }
}

static __device__ __forceinline__ void compute_angular_q_derivatives_for_n(
  const NEP::ParaMB paramb,
  const int center,
  const int angular_n,
  const int descriptor,
  const float* sum_fxyz,
  const float (*s_gradient)[3],
  const float (*s_hessian)[3][3],
  float* q_gradient_s,
  float* q_hessian_diagonal,
  float* neighbor_gradient,
  float (*local_hessian)[3])
{
  const int num_s_terms =
    (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
  for (int axis_a = 0; axis_a < 3; ++axis_a) {
    neighbor_gradient[axis_a] = 0.0f;
    for (int axis_b = 0; axis_b < 3; ++axis_b)
      local_hessian[axis_a][axis_b] = 0.0f;
  }
  for (int component = 0; component < num_s_terms; ++component) {
    q_gradient_s[component] = 0.0f;
    q_hessian_diagonal[component] = 0.0f;
  }

  LocalQChainAccumulator accumulator = {
    sum_fxyz, s_gradient, s_hessian,
    q_gradient_s, q_hessian_diagonal,
    neighbor_gradient, local_hessian};
  visit_q_monomials(paramb, descriptor, accumulator);

  if (g_dump_s_derivatives) {
    for (int component = 0; component < num_s_terms; ++component) {
      const float first = q_gradient_s[component];
      for (int axis_a = 0; axis_a < 3; ++axis_a) {
        for (int axis_b = 0; axis_b < 3; ++axis_b) {
          printf(
            "QCOMP %d %d %d %d %d %d %.10g %.10g %.10g %.10g\n",
            center, descriptor, angular_n, component, axis_a, axis_b,
            first, first * s_hessian[component][axis_a][axis_b],
            q_hessian_diagonal[component] *
              s_gradient[component][axis_a] *
              s_gradient[component][axis_b],
            local_hessian[axis_a][axis_b]);
        }
      }
    }
    printf(
      "QCHAIN %d %d %d %.10g %.10g\n",
      center, descriptor, angular_n, neighbor_gradient[0],
      local_hessian[0][0]);
  }
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
        (n * (paramb.basis_size_radial + 1) + basis) * paramb.num_types_sq +
        center_type * paramb.num_types + neighbor_type;
      float fn, fnp, fnpp;
      find_fn_fnp_fnpp(basis, rcinv, distance, fc, fcp, fcpp, fn, fnp, fnpp);
      radial_derivative[n] += annmb.c[c_index] * fnp;
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
    if (d12 < 1.0e-6f) continue;
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
          (n * (paramb.basis_size_radial + 1) + basis) * paramb.num_types_sq +
          t1 * paramb.num_types + t2;
        float fn, fnp, fnpp;
        find_fn_fnp_fnpp(basis, rcinv, d12, fc, fcp, fcpp, fn, fnp, fnpp);
        gp += annmb.c[c_index] * fnp;
        gpp += annmb.c[c_index] * fnpp;
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
    if (d12 < 1.0e-6f) continue;
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
          (n * (paramb.basis_size_radial + 1) + basis) * paramb.num_types_sq +
          t1 * paramb.num_types + t2;
        float fn, fnp, fnpp;
        find_fn_fnp_fnpp(basis, rcinv, d12, fc, fcp, fcpp, fn, fnp, fnpp);
        gp += annmb.c[c_index] * fnp;
        gpp += annmb.c[c_index] * fnpp;
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
    if (d12 < 1.0e-6f) continue;
    const float d12inv = 1.0f / d12;
    const float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
    const float rcinv = 1.0f / rc;
    float fc, fcp, fcpp;
    find_fc_fcp_fcpp(rc, rcinv, d12, fc, fcp, fcpp);
    for (int n = 0; n <= n_max_r; ++n) {
      gnp[n] = 0.0f;
      for (int basis = 0; basis <= paramb.basis_size_radial; ++basis) {
        const int c_index =
          (n * (paramb.basis_size_radial + 1) + basis) * paramb.num_types_sq +
          t1 * paramb.num_types + t2;
        float fn, fnp, fnpp;
        find_fn_fnp_fnpp(basis, rcinv, d12, fc, fcp, fcpp, fn, fnp, fnpp);
        gnp[n] += annmb.c[c_index] * fnp;
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
      if (d13 < 1.0e-6f) continue;
      const float d13inv = 1.0f / d13;
      const float rc13 = (paramb.rc_radial[t1] + paramb.rc_radial[t3]) * 0.5f;
      const float rc13inv = 1.0f / rc13;
      find_fc_fcp_fcpp(rc13, rc13inv, d13, fc, fcp, fcpp);
      for (int n = 0; n <= n_max_r; ++n) {
        other_radial_derivative[n] = 0.0f;
        for (int basis = 0; basis <= paramb.basis_size_radial; ++basis) {
          const int c_index =
            (n * (paramb.basis_size_radial + 1) + basis) * paramb.num_types_sq +
            t1 * paramb.num_types + t3;
          float fn, fnp, fnpp;
          find_fn_fnp_fnpp(basis, rc13inv, d13, fc, fcp, fcpp, fn, fnp, fnpp);
          other_radial_derivative[n] += annmb.c[c_index] * fnp;
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
    if (d12 < 1.0e-6f) continue;

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

// Each center-neighbor edge is prepared exactly once. The old monolithic
// kernel rebuilt the same spherical and descriptor derivatives in every
// neighbor pair and again in the radial-angular loop.
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
  if (distance < 1.0e-6f) return;

  const int num_s_terms =
    (paramb.L_max + 1) * (paramb.L_max + 1) - 1;
  const int angular_descriptor_count = paramb.num_L * n_mp1;
  float sum_fxyz[S_CAPACITY] = {};
  float s_values[S_CAPACITY] = {};
  float s_gradient[S_CAPACITY][3] = {};
  float s_hessian[S_CAPACITY][3][3] = {};
  float q_gradient_s[S_CAPACITY];
  float q_hessian_diagonal[S_CAPACITY];
  float neighbor_gradient[MAX_ANGULAR_DESCRIPTORS][3] = {};
  float local_hessian[MAX_ANGULAR_DESCRIPTORS][3][3] = {};

  for (int component = 0; component < num_s_terms; ++component)
    sum_fxyz[component] = g_sum_fxyz[
      (angular_n * num_s_terms + component) * N + center];

  compute_angular_s_derivatives_for_n(
    paramb, annmb, center, neighbor, angular_n, g_type, r12, distance,
    s_values, s_gradient, s_hessian);
  if (g_dump_s_derivatives) {
    if (angular_n == 0)
      printf(
        "QPOS %d %d %.10g %.10g %.10g %.10g\n",
        center, neighbor, r12[0], r12[1], r12[2], distance);
    for (int component = 0; component < num_s_terms; ++component)
      for (int axis_a = 0; axis_a < 3; ++axis_a)
        for (int axis_b = 0; axis_b < 3; ++axis_b)
          printf(
            "QSDERIV %d %d %d %d %d %d %.10g %.10g\n",
            center, neighbor, angular_n, component, axis_a, axis_b,
            s_gradient[component][axis_a],
            s_hessian[component][axis_a][axis_b]);
  }

  for (int descriptor = 0; descriptor < paramb.num_L; ++descriptor) {
    compute_angular_q_derivatives_for_n(
      paramb, center, angular_n, descriptor, sum_fxyz,
      s_gradient, s_hessian, q_gradient_s, q_hessian_diagonal,
      neighbor_gradient[descriptor], local_hessian[descriptor]);
    if (g_dump_s_derivatives) {
      for (int component = 0; component < num_s_terms; ++component) {
        printf(
          "QSTRANS %d %d %d %d %.10g %.10g %.10g\n",
          center, neighbor, descriptor * n_mp1 + angular_n, component,
          sum_fxyz[component], q_gradient_s[component],
          q_hessian_diagonal[component]);
      }
    }

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
  const bool g_disable_q_local,
  const bool g_disable_q_cross,
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
            if (!g_disable_q_local)
              value += fp * cached_local_hessian(
                first_edge, descriptor, n, axis_a, axis_b);
          } else if (!g_disable_q_cross) {
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
  const float* g_Fp,
  const float* g_Hqq,
  const bool g_dump_descriptor_derivatives,
  const float* g_neighbor_gradient,
  const float* g_local_hessian,
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
  auto descriptor_fp = [&](const int descriptor, const int n) {
    return g_Fp[angular_fp_index(
      n_max_radial, n_mp1, descriptor, n, n1, N)];
  };

  for (int radial_index = 0; radial_index < g_NN_radial[n1]; ++radial_index) {
    const int radial_neighbor = g_NL_radial[n1 + N * radial_index];
    float r12[3];
    load_neighbor_vector(
      box, N, n1, radial_index, radial_neighbor, g_x, g_y, g_z,
      g_r12_radial_x, g_r12_radial_y, g_r12_radial_z, r12);
    const float d12 =
      sqrtf(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    if (d12 < 1.0e-6f) continue;
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
    const int neighbor = g_NL_angular[n1 + N * neighbor_slot];
    for (int descriptor = 0; descriptor < paramb.num_L; ++descriptor) {
      for (int n = 0; n < n_mp1; ++n) {
        const int descriptor_index =
          n_max_radial + 1 + descriptor * n_mp1 + n;
        for (int axis = 0; axis < 3; ++axis) {
          const float gradient =
            cached_gradient(edge, descriptor, n, axis);
          center_gradient[descriptor][n][axis] -= gradient;
          if (g_dump_descriptor_derivatives) {
            for (int other_axis = 0; other_axis < 3; ++other_axis) {
              const float hessian = cached_local_hessian(
                edge, descriptor, n, axis, other_axis);
              printf(
                "QDERIV %d %d %d %d %d %d %.10g %.10g %.10g\n",
                n1, neighbor, descriptor_index, n, axis, other_axis,
                q_scaler[descriptor_index] * gradient,
                q_scaler[descriptor_index] * hessian, hessian);
            }
          }
        }
      }
    }
  }


  if (g_dump_descriptor_derivatives) {
    for (int descriptor = 0; descriptor < paramb.num_L; ++descriptor) {
      for (int n = 0; n < n_mp1; ++n) {
        const int descriptor_index =
          n_max_radial + 1 + descriptor * n_mp1 + n;
        for (int axis_a = 0; axis_a < 3; ++axis_a) {
          for (int axis_b = 0; axis_b < 3; ++axis_b) {
            float hessian_sum = 0.0f;
            for (int neighbor_slot = 0; neighbor_slot < neighbor_count;
                 ++neighbor_slot) {
              const int edge = edge_index(neighbor_slot);
              if (g_valid[edge])
                hessian_sum += cached_local_hessian(
                  edge, descriptor, n, axis_a, axis_b);
            }
            printf(
              "QHESS %d %d %d %d %.10g %.10g %.10g\n",
              n1, descriptor_index, axis_a, axis_b, hessian_sum,
              q_scaler[descriptor_index] * hessian_sum,
              descriptor_fp(descriptor, n));
          }
        }
      }
    }
  }

  // Radial-angular terms use the same cached angular Jacobian as the
  // neighbor-pair assembly. This removes the previous O(R*M) derivative
  // reconstruction while preserving the ordered descriptor blocks.
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
      if (d12 < 1.0e-6f) continue;
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

class QScalerOverride
{
public:
  QScalerOverride(const float* scaler, int dimension)
    : scaler_(scaler), dimension_(dimension)
  {
  }

  bool save_original(std::vector<float>& working_scaler)
  {
    working_scaler.resize(dimension_);
    original_.resize(dimension_);
    gpuError_t error = gpuMemcpy(
      original_.data(), scaler_, dimension_ * sizeof(float),
      gpuMemcpyDeviceToHost);
    if (error != gpuSuccess) {
      fprintf(
        stderr, "Failed to save NEP q_scaler: %s.\n",
        gpuGetErrorString(error));
      return false;
    }
    working_scaler = original_;
    return true;
  }

  bool install(const std::vector<float>& working_scaler)
  {
    if (working_scaler.size() != static_cast<size_t>(dimension_)) return false;
    installed_ = true; // A failed copy may still have partially modified state.
    gpuError_t error = gpuMemcpy(
      const_cast<float*>(scaler_), working_scaler.data(),
      dimension_ * sizeof(float), gpuMemcpyHostToDevice);
    if (error != gpuSuccess) {
      fprintf(
        stderr, "Failed to install diagnostic NEP q_scaler: %s.\n",
        gpuGetErrorString(error));
      return false;
    }
    return true;
  }

  bool restore()
  {
    if (!installed_) return !restore_failed_;
    gpuError_t error = gpuMemcpy(
      const_cast<float*>(scaler_), original_.data(),
      dimension_ * sizeof(float), gpuMemcpyHostToDevice);
    if (error != gpuSuccess) {
      restore_failed_ = true;
      fprintf(
        stderr, "Failed to restore NEP q_scaler: %s.\n",
        gpuGetErrorString(error));
      return false;
    }
    installed_ = false;
    return true;
  }

  ~QScalerOverride()
  {
    restore();
  }

private:
  const float* scaler_;
  int dimension_;
  std::vector<float> original_;
  bool installed_ = false;
  bool restore_failed_ = false;
};

bool parse_zero_angular_groups(
  const std::string& text, int angular_group_count,
  int angular_descriptor_stride, std::vector<bool>& zero_angular_groups)
{
  if (angular_group_count <= 0 || angular_descriptor_stride <= 0 ||
      angular_group_count >
        std::numeric_limits<int>::max() / angular_descriptor_stride) {
    return false;
  }
  std::vector<bool> groups(
    angular_group_count * angular_descriptor_stride, false);
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    size_t first = 0;
    while (first < item.size() && std::isspace(static_cast<unsigned char>(item[first])))
      ++first;
    if (first == item.size()) return false;
    try {
      size_t consumed = 0;
      const long value = std::stol(item, &consumed);
      while (consumed < item.size() &&
             std::isspace(static_cast<unsigned char>(item[consumed]))) {
        ++consumed;
      }
      if (consumed != item.size() || value < 0 ||
          value >= angular_group_count) {
        return false;
      }
      const int begin =
        static_cast<int>(value) * angular_descriptor_stride;
      std::fill(
        groups.begin() + begin, groups.begin() + begin + angular_descriptor_stride,
        true);
    } catch (const std::exception&) {
      return false;
    }
  }
  zero_angular_groups.swap(groups);
  return true;
}

void write_hessian_matrix(
  const std::string& path, const std::vector<double>& hessian_soa,
  int N, bool atom_major)
{
  std::vector<double> output_values;
  if (atom_major) {
    output_values.resize(hessian_soa.size());
    for (int atom_i = 0; atom_i < N; ++atom_i) {
      for (int axis_i = 0; axis_i < 3; ++axis_i) {
        const int row_soa = axis_i * N + atom_i;
        const int row_atom_major = atom_i * 3 + axis_i;
        for (int atom_j = 0; atom_j < N; ++atom_j) {
          for (int axis_j = 0; axis_j < 3; ++axis_j) {
            output_values[row_atom_major * (3 * N) + atom_j * 3 + axis_j] =
              hessian_soa[row_soa * (3 * N) + axis_j * N + atom_j];
          }
        }
      }
    }
  }
  const std::vector<double>& values = atom_major ? output_values : hessian_soa;
  std::ofstream output(path);
  output << std::scientific << std::setprecision(17);
  output << "# coordinate_order=" << (atom_major ? "atom_major" : "soa") << "\n";
  output << "# matrix_order=row_major\n";
  output << "# definition=minus_force_jacobian\n";
  output << "# unit=eV/A^2\n";
  for (int row = 0; row < 3 * N; ++row) {
    for (int column = 0; column < 3 * N; ++column)
      output << (column ? " " : "") << values[row * (3 * N) + column];
    output << "\n";
  }
}

} // namespace

// ============================================================================
// Main entry: NEP analytic Hessian for explicitly supported model classes
// ============================================================================
bool compute_nep_analytic_hessian(
  Force& force,
  Box& box,
  Atom& atom,
  const std::vector<double>& position_soa,
  int N,
  std::vector<double>& hessian_out,
  std::vector<double>& force_out,
  double& pe_out,
  std::vector<double>* validation_hessian_soa,
  std::vector<Group>* groups,
  sparse_hessian::BlockPattern* sparse_pattern)
{
  hessian_out.clear();
  force_out.clear();
  pe_out = 0.0;
  if (validation_hessian_soa)
    validation_hessian_soa->clear();
  if (N <= 0 || N > std::numeric_limits<int>::max() / 3 ||
      atom.number_of_atoms != N)
    return false;
  const int N3 = 3 * N;
  if (position_soa.size() != static_cast<size_t>(N3)) {
    printf("NEP analytic Hessian: explicit position array size is invalid.\n");
    return false;
  }

  if (force.get_number_of_potentials() != 1) return false;
  auto* nep = dynamic_cast<NEP*>(&force.get_potential(0));
  if (nep == nullptr || nep->nep_model_type < 0 || nep->nep_model_type > 3 ||
      nep->ilp_flag != 0 || nep->uses_dftd3()) return false;

  bool dump_analytic_atom_major = false;
  if (const char* layout =
        std::getenv("GPUMD_DUMP_ANALYTIC_HESSIAN_LAYOUT")) {
    if (std::string(layout) == "soa") {
      dump_analytic_atom_major = false;
    } else if (std::string(layout) == "atom_major") {
      dump_analytic_atom_major = true;
    } else {
      printf(
        "NEP analytic Hessian: invalid dump layout %s "
        "(expected soa or atom_major).\n",
        layout);
      return false;
    }
  }

  const bool use_expanded_box = nep->requires_expanded_box(box);
  if (nep->paramb.L_max < 0 || nep->paramb.L_max > 8 ||
      (nep->paramb.L_max + 1) * (nep->paramb.L_max + 1) - 1 >
        MAX_S_TERMS) {
    printf(
      "NEP analytic Hessian: unsupported L_max=%d (supported range is 0..8).\n",
      nep->paramb.L_max);
    return false;
  }
  if (nep->paramb.n_max_radial < 0 ||
      nep->paramb.n_max_radial + 1 > MAX_NUM_N ||
      nep->paramb.n_max_angular < 0 ||
      nep->paramb.n_max_angular + 1 > MAX_NUM_N ||
      nep->paramb.num_L < 0 ||
      nep->paramb.num_L > MAX_ANGULAR_DESCRIPTORS) {
    return false;
  }
  const int radial_descriptor_count = nep->paramb.n_max_radial + 1;
  const int angular_descriptor_stride = nep->paramb.n_max_angular + 1;
  if (angular_descriptor_stride >
      std::numeric_limits<int>::max() / nep->paramb.num_L) {
    return false;
  }
  const int angular_descriptor_count =
    nep->paramb.num_L * angular_descriptor_stride;
  if (radial_descriptor_count >
      std::numeric_limits<int>::max() - angular_descriptor_count) {
    return false;
  }
  const int expected_descriptor_count = radial_descriptor_count +
    angular_descriptor_count + (nep->nep_model_type == 3 ? 1 : 0);
  if (nep->annmb.dim <= 0 || nep->annmb.dim > MAX_DIM ||
      nep->annmb.dim != expected_descriptor_count ||
      expected_descriptor_count > std::numeric_limits<int>::max() / N ||
      N > std::numeric_limits<int>::max() / expected_descriptor_count ||
      N * expected_descriptor_count >
        std::numeric_limits<int>::max() / expected_descriptor_count ||
      N3 > std::numeric_limits<int>::max() / N3) {
    printf(
      "NEP analytic Hessian: descriptor dimension mismatch "
      "(annmb.dim=%d, expected=%d).\n",
      nep->annmb.dim, expected_descriptor_count);
    return false;
  }
  const size_t expected_position_size = static_cast<size_t>(N3);
  const size_t expected_atom_size = static_cast<size_t>(N);
  if (atom.position_per_atom.size() != expected_position_size ||
      atom.force_per_atom.size() != expected_position_size ||
      atom.potential_per_atom.size() != expected_atom_size ||
      atom.virial_per_atom.size() != 9 * expected_atom_size ||
      atom.type.size() != expected_atom_size) {
    printf("NEP analytic Hessian: Atom array dimensions are inconsistent.\n");
    return false;
  }

  for (int index = 0; index < N3; ++index) {
    if (!std::isfinite(position_soa[index])) {
      printf(
        "NEP analytic Hessian: requested position %d is non-finite.\n", index);
      return false;
    }
  }

  AtomStateBackup atom_backup(atom);
  if (!atom_backup.save()) {
    printf("NEP analytic Hessian: failed to save Atom state.\n");
    return false;
  }

  const bool zero_radial_descriptors =
    std::getenv("GPUMD_ZERO_RADIAL_DESCRIPTORS") != nullptr;
  const bool zero_angular_descriptors =
    std::getenv("GPUMD_ZERO_ANGULAR_DESCRIPTORS") != nullptr;
  std::vector<bool> zero_angular_groups;
  if (const char* groups = std::getenv("GPUMD_ZERO_ANGULAR_GROUPS")) {
    if (!parse_zero_angular_groups(
          groups, nep->paramb.num_L, angular_descriptor_stride,
          zero_angular_groups)) {
      printf(
        "NEP analytic Hessian: invalid GPUMD_ZERO_ANGULAR_GROUPS=%s.\n",
        groups);
      return false;
    }
  }
  const bool scaler_override_used =
    zero_radial_descriptors || zero_angular_descriptors ||
    !zero_angular_groups.empty();
  QScalerOverride scaler_override(nep->annmb.q_scaler, nep->annmb.dim);
  std::vector<float> scaler_for_test;
  if (scaler_override_used &&
      !scaler_override.save_original(scaler_for_test)) {
    return false;
  }
  if (zero_radial_descriptors) {
    std::fill(
      scaler_for_test.begin(),
      scaler_for_test.begin() + radial_descriptor_count, 0.0f);
  }
  if (zero_angular_descriptors) {
    std::fill(
      scaler_for_test.begin() + radial_descriptor_count,
      scaler_for_test.end(), 0.0f);
  } else {
    for (int index = 0;
         index < static_cast<int>(zero_angular_groups.size()); ++index) {
      if (zero_angular_groups[index])
        scaler_for_test[radial_descriptor_count + index] = 0.0f;
    }
  }

  auto finish = [&](bool success) {
    bool state_restored = scaler_override.restore();
    // Return to the caller's geometry before rebuilding Force-owned caches.
    state_restored = atom_backup.restore() && state_restored;
    if (state_restored) {
      // Refresh neighbor/descriptor caches after diagnostic displacements or a
      // q_scaler override. Atom outputs are restored again below.
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
    if (!state_restored || !success) {
      hessian_out.clear();
      force_out.clear();
      pe_out = 0.0;
      return false;
    }
    return true;
  };

  if (scaler_override_used && !scaler_override.install(scaler_for_test))
    return finish(false);

  const gpuError_t position_upload_error = gpuMemcpy(
    atom.position_per_atom.data(), position_soa.data(),
    N3 * sizeof(double), gpuMemcpyHostToDevice);
  if (position_upload_error != gpuSuccess) {
    fprintf(
      stderr, "NEP analytic Hessian: failed to upload explicit position: %s.\n",
      gpuGetErrorString(position_upload_error));
    return finish(false);
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
    return finish(false);
  }
  if (use_expanded_box) {
    if (nep->small_box_data.r12.size() % 6 != 0) {
      printf("NEP analytic Hessian: expanded-box displacement cache is invalid.\n");
      return finish(false);
    }
    const size_t r12_plane_size = nep->small_box_data.r12.size() / 6;
    if (r12_plane_size < radial_neighbor_list_gpu.size() ||
        r12_plane_size < angular_neighbor_list_gpu.size()) {
      printf("NEP analytic Hessian: expanded-box displacement cache is too small.\n");
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
      return finish(false);
    }
    const int count = angular_neighbor_counts[atom_index];
    if (count < 0 || static_cast<size_t>(count) > angular_neighbor_capacity) {
      printf(
        "NEP analytic Hessian: invalid angular neighbor count=%d "
        "for atom=%d (capacity=%zu).\n",
        count, atom_index, angular_neighbor_capacity);
      return finish(false);
    }
    max_angular_neighbors = std::max(max_angular_neighbors, count);
  }
  // Extract force and PE
  std::vector<double> force_soa(N3);
  atom.force_per_atom.copy_to_host(force_soa.data());
  std::vector<double> pe_host(N);
  atom.potential_per_atom.copy_to_host(pe_host.data());
  pe_out = 0.0;
  for (int i = 0; i < N; ++i) pe_out += pe_host[i];

  force_out = std::move(force_soa);

  const int dim = nep->annmb.dim;
  const int N_total = N;

  GPU_Vector<int> sparse_row_offsets_gpu;
  GPU_Vector<int> sparse_columns_gpu;
  const int* sparse_row_offsets = nullptr;
  const int* sparse_columns = nullptr;
  if (sparse_pattern != nullptr) {
    if (std::getenv("GPUMD_VALIDATE_ANALYTIC_HESSIAN") != nullptr ||
        std::getenv("GPUMD_DUMP_ANALYTIC_HESSIAN") != nullptr ||
        std::getenv("GPUMD_DUMP_NEP_HESSIAN") != nullptr)
      return false;
    std::vector<int> radial_list(radial_neighbor_list_gpu.size());
    std::vector<int> angular_list(angular_neighbor_list_gpu.size());
    radial_neighbor_list_gpu.copy_to_host(radial_list.data());
    angular_neighbor_list_gpu.copy_to_host(angular_list.data());
    try {
      *sparse_pattern = sparse_hessian::build_pattern_from_neighbors(
        N, radial_neighbor_counts, radial_list, radial_neighbor_capacity,
        angular_neighbor_counts, angular_list, angular_neighbor_capacity);
    } catch (const std::exception& error) {
      printf("Sparse NEP Hessian pattern failed: %s\n", error.what());
      return false;
    }
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
    nep->paramb, nep->annmb, N_total, atom.type.data(),
    nep->nep_data.q_descriptors.data(), hqq_gpu.data());
  GPU_CHECK_KERNEL
  if (std::getenv("GPUMD_ZERO_ANN_CROSS") != nullptr) {
    CHECK(gpuMemset(hqq_gpu.data(), 0, hqq_gpu.size() * sizeof(float)));
  }
  if (std::getenv("GPUMD_NEGATE_ANN_CROSS") != nullptr) {
    std::vector<float> hqq_host(hqq_gpu.size());
    hqq_gpu.copy_to_host(hqq_host.data());
    for (float& value : hqq_host) value = -value;
    hqq_gpu.copy_from_host(hqq_host.data());
  }

  // Debug: dump descriptors, Fp, and Hqq for analysis
  if (std::getenv("GPUMD_DUMP_NEP_DEBUG") != nullptr) {
    std::vector<int> nn_radial_host(N_total);
    std::vector<int> nn_angular_host(N_total);
    radial_neighbor_counts_gpu.copy_to_host(nn_radial_host.data(), N_total);
    angular_neighbor_counts_gpu.copy_to_host(nn_angular_host.data(), N_total);
    int radial_min = nn_radial_host[0], radial_max = nn_radial_host[0];
    int angular_min = nn_angular_host[0], angular_max = nn_angular_host[0];
    for (int count : nn_radial_host) {
      radial_min = std::min(radial_min, count);
      radial_max = std::max(radial_max, count);
    }
    for (int count : nn_angular_host) {
      angular_min = std::min(angular_min, count);
      angular_max = std::max(angular_max, count);
    }
    printf(
      "NEP analytic Hessian neighbor counts: radial_min=%d radial_max=%d "
      "angular_min=%d angular_max=%d\n",
      radial_min, radial_max, angular_min, angular_max);
    std::vector<float> q_host(N_total * dim);
    nep->nep_data.q_descriptors.copy_to_host(q_host.data());
    std::vector<float> Fp_host(N_total * dim);
    nep->nep_data.Fp.copy_to_host(Fp_host.data());
    std::vector<float> hqq_host(N_total * dim * dim);
    hqq_gpu.copy_to_host(hqq_host.data());
    if (const char* path = std::getenv("GPUMD_DUMP_NEP_HQQ")) {
      std::ofstream output(path);
      for (int atom = 0; atom < N_total; ++atom) {
        for (int i = 0; i < dim; ++i) {
          for (int j = 0; j < dim; ++j) {
            output << (j ? " " : "") <<
              hqq_host[(atom * dim + i) * dim + j];
          }
          output << "\n";
        }
      }
    }
    printf("=== NEP Debug Dump (N=%d, dim=%d) ===\n", N_total, dim);
    {
      std::vector<float> q_scaler_host(dim);
      CHECK(gpuMemcpy(
        q_scaler_host.data(), nep->annmb.q_scaler, sizeof(float) * dim,
        gpuMemcpyDeviceToHost));
      printf("  q_scaler:");
      for (int d = 0; d < dim; ++d)
        printf(" %.10g", q_scaler_host[d]);
      printf("\n");
    }
    for (int n1 = 0; n1 < N_total; ++n1) {
      std::vector<int> type_host(N_total); atom.type.copy_to_host(type_host.data()); printf("Atom %d (type %d):\n", n1, type_host[n1]);
      printf("  q_desc:");
      for (int d = 0; d < dim; ++d) printf(" %.8g", q_host[d * N_total + n1]);
      printf("\n");
      printf("  Fp   :");
      for (int d = 0; d < dim; ++d) printf(" %.8g", Fp_host[d * N_total + n1]);
      printf("\n");
      // Print Hqq diagonal and a few off-diagonal
      printf("  Hqq_diag:");
      for (int d = 0; d < dim; ++d)
        printf(" %.8g", hqq_host[(n1 * dim + d) * dim + d]);
      printf("\n");
    }
    // Also dump positions
    std::vector<double> pos_host(N_total * 3);
    atom.position_per_atom.copy_to_host(pos_host.data());
    for (int n1 = 0; n1 < N_total; ++n1)
      printf("Pos %d: %.10f %.10f %.10f\n", n1,
        pos_host[n1], pos_host[n1 + N_total], pos_host[n1 + 2 * N_total]);
  }

  // 5. Compute radial Hessian
  int grid = (N_total - 1) / BLOCK_SIZE + 1;
  const bool disable_radial = std::getenv("GPUMD_DISABLE_RADIAL_HESSIAN") != nullptr;
  const bool disable_angular = std::getenv("GPUMD_DISABLE_ANGULAR_HESSIAN") != nullptr;
  const bool disable_zbl = std::getenv("GPUMD_DISABLE_ZBL_HESSIAN") != nullptr;
  const bool disable_s_hessian =
    std::getenv("GPUMD_DISABLE_S_HESSIAN") != nullptr;
  const bool disable_q_s_hessian =
    std::getenv("GPUMD_DISABLE_Q_S_HESSIAN") != nullptr;
  CHECK(gpuMemcpyToSymbol(
    g_disable_s_hessian, &disable_s_hessian, sizeof(bool)));
  CHECK(gpuMemcpyToSymbol(
    g_disable_q_s_hessian, &disable_q_s_hessian, sizeof(bool)));
  const bool dump_s_derivatives =
    std::getenv("GPUMD_DUMP_ANALYTIC_S_DERIVATIVES") != nullptr;
  const bool dump_q_derivatives =
    std::getenv("GPUMD_DUMP_ANALYTIC_Q_DERIVATIVES") != nullptr;
  CHECK(gpuMemcpyToSymbol(
    g_dump_s_derivatives, &dump_s_derivatives, sizeof(bool)));
  if (!disable_radial) {
    find_radial_hessian_kernel<<<grid, BLOCK_SIZE>>>(
      nep->paramb, nep->annmb, N_total, 0, N_total, box,
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
  size_t original_printf_limit = 0;
  bool restore_stack_limit = false;
  bool restore_printf_limit = false;
  if (!disable_angular && angular_descriptor_count > 0) {
    CHECK(gpuDeviceGetLimit(&original_stack_limit, gpuLimitStackSize));
    // ptxas reports 5.3 KiB for the largest staged angular kernel on sm_86.
    // Keep headroom without retaining the legacy monolithic kernel's 256 KiB
    // per-thread stack reservation.
    constexpr size_t kRequiredStackBytes = 16ULL * 1024;
    if (original_stack_limit < kRequiredStackBytes) {
      CHECK(gpuDeviceSetLimit(gpuLimitStackSize, kRequiredStackBytes));
      restore_stack_limit = true;
    }

    if (dump_s_derivatives || dump_q_derivatives) {
      CHECK(gpuDeviceGetLimit(&original_printf_limit, gpuLimitPrintfFifoSize));
      size_t requested_fifo_bytes = 512ULL * 1024 * 1024;
      if (const char* value = std::getenv("GPUMD_PRINTF_FIFO_MB")) {
        const long megabytes = std::atol(value);
        if (megabytes > 0 &&
            static_cast<unsigned long>(megabytes) <=
              std::numeric_limits<size_t>::max() / (1024ULL * 1024)) {
          requested_fifo_bytes = static_cast<size_t>(megabytes) * 1024 * 1024;
        }
      }
      const size_t fifo_candidates[] = {
        requested_fifo_bytes,
        256ULL * 1024 * 1024,
        64ULL * 1024 * 1024,
        8ULL * 1024 * 1024};
      bool fifo_configured = requested_fifo_bytes <= original_printf_limit;
      for (const size_t requested_bytes : fifo_candidates) {
        if (fifo_configured) break;
        const size_t fifo_bytes = std::max(requested_bytes, original_printf_limit);
        gpuGetLastError();
        if (gpuDeviceSetLimit(gpuLimitPrintfFifoSize, fifo_bytes) == gpuSuccess) {
          fifo_configured = true;
          restore_printf_limit = fifo_bytes != original_printf_limit;
        }
      }
      if (!fifo_configured) {
        fprintf(
          stderr,
          "Warning: could not enlarge the GPU printf FIFO; diagnostic output "
          "may be truncated.\n");
        gpuGetLastError();
      }
    }
    if (max_angular_neighbors > 0) {
      const int num_s_terms =
        (nep->paramb.L_max + 1) * (nep->paramb.L_max + 1) - 1;
      const int cached_angular_descriptors =
        nep->paramb.num_L * (nep->paramb.n_max_angular + 1);
      const size_t floats_per_edge =
        static_cast<size_t>(cached_angular_descriptors) * (3 + 9) +
        static_cast<size_t>(nep->paramb.n_max_angular + 1) *
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
        edge_capacity * (nep->paramb.n_max_angular + 1) * num_s_terms * 3);
      GPU_Vector<float> local_hessian_cache(
        edge_capacity * cached_angular_descriptors * 9);
      GPU_Vector<int> valid_cache(edge_capacity);

      if (std::getenv("GPUMD_PRINT_ANALYTIC_HESSIAN_DIAGNOSTICS") != nullptr) {
        printf(
          "NEP angular Hessian cache: max_neighbors=%d tile_centers=%d "
          "workspace_bytes=%zu\n",
          max_angular_neighbors, tile_centers,
          edge_capacity * bytes_per_edge);
      }

      const bool disable_q_local =
        std::getenv("GPUMD_DISABLE_Q_LOCAL") != nullptr;
      const bool disable_q_cross =
        std::getenv("GPUMD_DISABLE_Q_CROSS") != nullptr;
      for (int center_begin = 0; center_begin < N_total;
           center_begin += tile_centers) {
        const int center_count =
          std::min(tile_centers, N_total - center_begin);
        const int edge_count = center_count * max_angular_neighbors;
        const int prepare_work_count =
          edge_count * (nep->paramb.n_max_angular + 1);
        if (num_s_terms <= LOW_L_MAX_S_TERMS) {
          prepare_angular_neighbor_derivatives_kernel<LOW_L_MAX_S_TERMS><<<
            (prepare_work_count - 1) / ANGULAR_PREPARE_BLOCK_SIZE + 1,
            ANGULAR_PREPARE_BLOCK_SIZE>>>(
            nep->paramb, nep->annmb, N_total, center_begin, center_count,
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
            nep->paramb, nep->annmb, N_total, center_begin, center_count,
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
          nep->paramb, nep->annmb, N_total, center_begin, center_count,
          max_angular_neighbors,
          angular_neighbor_counts_gpu.data(), angular_neighbor_list_gpu.data(),
          nep->nep_data.Fp.data(), nep->nep_data.sum_fxyz.data(),
          hqq_gpu.data(), disable_q_local, disable_q_cross,
          neighbor_gradient_cache.data(), neighbor_s_gradient_cache.data(),
          local_hessian_cache.data(), valid_cache.data(),
          sparse_row_offsets, sparse_columns,
          hessian_gpu.data());
        GPU_CHECK_KERNEL

        assemble_angular_radial_cross_kernel<<<
          (center_count - 1) / ANGULAR_RADIAL_BLOCK_SIZE + 1,
          ANGULAR_RADIAL_BLOCK_SIZE>>>(
          nep->paramb, nep->annmb, N_total, center_begin, center_count,
          max_angular_neighbors, box,
          angular_neighbor_counts_gpu.data(), angular_neighbor_list_gpu.data(),
          radial_neighbor_counts_gpu.data(), radial_neighbor_list_gpu.data(),
          atom.type.data(),
          atom.position_per_atom.data(),
          atom.position_per_atom.data() + N_total,
          atom.position_per_atom.data() + 2 * N_total,
          radial_r12_x, radial_r12_y, radial_r12_z,
          nep->nep_data.Fp.data(), hqq_gpu.data(), dump_q_derivatives,
          neighbor_gradient_cache.data(), local_hessian_cache.data(),
          valid_cache.data(), sparse_row_offsets, sparse_columns,
          hessian_gpu.data());
        GPU_CHECK_KERNEL
      }
      CHECK(gpuDeviceSynchronize());
    }
  }
  if (nep->get_zbl().enabled && !disable_zbl) {
    find_zbl_hessian_kernel<<<grid, BLOCK_SIZE>>>(
      nep->paramb, nep->get_zbl(), N_total, box,
      // Use the same angular neighbor list as find_force_ZBL.  This keeps
      // the analytic derivative on exactly the same force surface, including
      // any cutoff-induced truncation.
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
  if (restore_printf_limit)
    CHECK(gpuDeviceSetLimit(gpuLimitPrintfFifoSize, original_printf_limit));
  if (restore_stack_limit)
    CHECK(gpuDeviceSetLimit(gpuLimitStackSize, original_stack_limit));

  // 6. Copy Hessian to host
  hessian_out.resize(sparse_pattern != nullptr
    ? sparse_pattern->value_count() : static_cast<size_t>(N3) * N3);
  hessian_gpu.copy_to_host(hessian_out.data());
  double maximum_asymmetry = 0.0;
  double maximum_absolute_value = 0.0;
  double frobenius_square = 0.0;
  double asymmetry_frobenius_square = 0.0;
  double maximum_translation_row_sum = 0.0;
  bool validation_failed = false;
  if (std::getenv("GPUMD_DUMP_NEP_DEBUG") != nullptr) {
    double hessian_norm = 0.0;
    int nonzeros = 0;
    for (double value : hessian_out) {
      hessian_norm += value * value;
      if (value != 0.0) ++nonzeros;
    }
    printf(
      "NEP analytic Hessian kernel output: nonzeros=%d frobenius=%.12g\n",
      nonzeros, std::sqrt(hessian_norm));
  }
  if (sparse_pattern != nullptr) {
    const auto& pattern = *sparse_pattern;
    for (int atom = 0; atom < N_total; ++atom) {
      for (int block = pattern.row_offsets[atom];
           block < pattern.row_offsets[atom + 1]; ++block) {
        const int other = pattern.columns[block];
        const int transpose = pattern.find_block(other, atom);
        if (transpose < 0) return finish(false);
        for (int a = 0; a < 3; ++a) {
          for (int b = 0; b < 3; ++b) {
            const size_t index = static_cast<size_t>(block) * 9 + a * 3 + b;
            const double value = hessian_out[index];
            if (!std::isfinite(value)) {
              hessian_out.clear();
              return finish(false);
            }
            maximum_absolute_value = std::max(maximum_absolute_value, std::abs(value));
            frobenius_square += value * value;
            const double transposed = hessian_out[
              static_cast<size_t>(transpose) * 9 + b * 3 + a];
            const double difference = value - transposed;
            maximum_asymmetry = std::max(maximum_asymmetry, std::abs(difference));
            asymmetry_frobenius_square += difference * difference;
            double row_sum = 0.0;
            for (int column_block = pattern.row_offsets[atom];
                 column_block < pattern.row_offsets[atom + 1]; ++column_block)
              row_sum += hessian_out[static_cast<size_t>(column_block) * 9 + a * 3 + b];
            maximum_translation_row_sum = std::max(
              maximum_translation_row_sum, std::abs(row_sum));
          }
        }
      }
    }
  } else {
    for (int row = 0; row < N3; ++row) {
      for (int column = row + 1; column < N3; ++column) {
        const double asymmetry = std::abs(
          hessian_out[row * N3 + column] - hessian_out[column * N3 + row]);
        maximum_asymmetry = std::max(maximum_asymmetry, asymmetry);
        asymmetry_frobenius_square += 2.0 * asymmetry * asymmetry;
      }
    }
    for (int row = 0; row < N3; ++row) {
      for (int column = 0; column < N3; ++column) {
        const double value = hessian_out[row * N3 + column];
        if (!std::isfinite(value)) {
          hessian_out.clear();
          return finish(false);
        }
        maximum_absolute_value = std::max(maximum_absolute_value, std::abs(value));
        frobenius_square += value * value;
      }
    }
    for (int atom = 0; atom < N_total; ++atom)
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
          double row_sum = 0.0;
          for (int other = 0; other < N_total; ++other)
            row_sum += hessian_out[(a * N_total + atom) * N3 + b * N_total + other];
          maximum_translation_row_sum = std::max(
            maximum_translation_row_sum, std::abs(row_sum));
        }
  }

  const double raw_frobenius = std::sqrt(frobenius_square);
  const double raw_scale = std::max(
    maximum_absolute_value, std::numeric_limits<double>::min());
  const double asymmetry_frobenius = std::sqrt(asymmetry_frobenius_square);
  const double asymmetry_max_relative = maximum_asymmetry / raw_scale;
  const double asymmetry_frobenius_relative =
    asymmetry_frobenius / std::max(raw_frobenius, std::numeric_limits<double>::min());
  const double translation_relative = maximum_translation_row_sum / raw_scale;

  // Structural checks operate on H_raw and are enabled by default.  Absolute
  // thresholds remain available as additional diagnostic constraints.
  constexpr double kDefaultSymmetryFrobeniusRelative = 1.0e-6;
  constexpr double kDefaultSymmetryMaxRelative = 1.0e-5;
  constexpr double kDefaultTranslationRelative = 1.0e-5;
  auto parse_threshold = [](const char* text, double& value) {
    if (!text) return true;
    char* end = nullptr;
    value = std::strtod(text, &end);
    return end != text && *end == '\0' && std::isfinite(value) && value >= 0.0;
  };
  auto validate_threshold = [
    &validation_failed
  ](bool present, bool valid, double value, double actual, const char* name) {
    if (!present) return;
    if (!valid) {
      printf("NEP analytic Hessian: invalid %s threshold.\n", name);
      validation_failed = true;
    } else if (actual > value) {
      validation_failed = true;
    }
  };

  if (std::getenv("GPUMD_PRINT_ANALYTIC_HESSIAN_DIAGNOSTICS") != nullptr ||
      std::getenv("GPUMD_VALIDATE_ANALYTIC_HESSIAN") != nullptr ||
      std::getenv("GPUMD_DUMP_NEP_DEBUG") != nullptr) {
    printf(
      "NEP analytic Hessian raw diagnostics: max_abs=%.12g frobenius=%.12g "
      "asymmetry_max=%.12g asymmetry_max_relative=%.12g "
      "asymmetry_frobenius=%.12g asymmetry_frobenius_relative=%.12g "
      "translation_max=%.12g translation_relative=%.12g\n",
      maximum_absolute_value, raw_frobenius, maximum_asymmetry,
      asymmetry_max_relative, asymmetry_frobenius,
      asymmetry_frobenius_relative, maximum_translation_row_sum,
      translation_relative);
  }

  validate_threshold(
    true, true, kDefaultSymmetryFrobeniusRelative,
    asymmetry_frobenius_relative, "default symmetry Frobenius");
  validate_threshold(
    true, true, kDefaultSymmetryMaxRelative, asymmetry_max_relative,
    "default symmetry max");
  validate_threshold(
    true, true, kDefaultTranslationRelative, translation_relative,
    "default translation");

  auto validate_environment_threshold = [&](const char* name, double actual) {
    const char* text = std::getenv(name);
    if (text == nullptr) return;
    double value = 0.0;
    const bool valid = parse_threshold(text, value);
    validate_threshold(true, valid, value, actual, name);
  };
  validate_environment_threshold(
    "GPUMD_VALIDATE_MAX_ASYMMETRY", maximum_asymmetry);
  validate_environment_threshold(
    "GPUMD_VALIDATE_MAX_TRANSLATION_ROW_SUM", maximum_translation_row_sum);
  validate_environment_threshold(
    "GPUMD_VALIDATE_MAX_ASYMMETRY_RELATIVE", asymmetry_max_relative);
  validate_environment_threshold(
    "GPUMD_VALIDATE_MAX_ASYMMETRY_FROBENIUS_RELATIVE",
    asymmetry_frobenius_relative);
  validate_environment_threshold(
    "GPUMD_VALIDATE_MAX_TRANSLATION_RELATIVE", translation_relative);

  if (std::getenv("GPUMD_VALIDATE_ANALYTIC_HESSIAN") != nullptr) {
    const int descriptor_count = N_total * dim;
    GPU_Vector<double> position_save(atom.position_per_atom.size());
    GPU_Vector<double> force_save(atom.force_per_atom.size());
    GPU_Vector<double> potential_save(atom.potential_per_atom.size());
    GPU_Vector<double> virial_save(atom.virial_per_atom.size());
    position_save.copy_from_device(atom.position_per_atom.data());
    force_save.copy_from_device(atom.force_per_atom.data());
    potential_save.copy_from_device(atom.potential_per_atom.data());
    virial_save.copy_from_device(atom.virial_per_atom.data());
    double epsilon = std::getenv("GPUMD_VALIDATE_EPSILON")
      ? std::atof(std::getenv("GPUMD_VALIDATE_EPSILON"))
      : 1.0e-3;
    if (!std::isfinite(epsilon) || epsilon <= 0.0) {
      printf("Analytic force validation skipped: epsilon must be finite and positive.\n");
      validation_failed = true;
      // Keep the diagnostic path numerically defined; the invalid user input
      // still forces a failed validation below.
      epsilon = 1.0e-3;
    }
    std::vector<double> finite_difference_hessian(N3 * N3, 0.0);
    double maximum_error = 0.0;
    int maximum_row = -1;
    double finite_difference_frobenius = 0.0;
    double difference_frobenius = 0.0;

    for (int column = 0; column < N3; ++column) {
      std::vector<double> force_plus(N3);
      std::vector<double> force_minus(N3);
      std::vector<double> displacement(position_save.size());
      // Cartesian row/column indices and Atom buffer indices are both SoA.
      const int displacement_index = column;
      position_save.copy_to_host(displacement.data());
      displacement[displacement_index] += epsilon;
      atom.position_per_atom.copy_from_host(displacement.data());
      force.compute(
        box, atom.position_per_atom, atom.type, empty_groups,
        atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);
      CHECK(gpuDeviceSynchronize());
      atom.force_per_atom.copy_to_host(force_plus.data());
      displacement[displacement_index] -= 2.0 * epsilon;
      atom.position_per_atom.copy_from_host(displacement.data());
      force.compute(
        box, atom.position_per_atom, atom.type, empty_groups,
        atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);
      CHECK(gpuDeviceSynchronize());
      atom.force_per_atom.copy_to_host(force_minus.data());
      atom.position_per_atom.copy_from_device(position_save.data());
      atom.force_per_atom.copy_from_device(force_save.data());
      atom.potential_per_atom.copy_from_device(potential_save.data());
      atom.virial_per_atom.copy_from_device(virial_save.data());

      for (int row = 0; row < N3; ++row) {
        // Force and Hessian both use the public Cartesian SoA row directly.
        const int force_row = row;
        const double finite_difference =
          -(force_plus[force_row] - force_minus[force_row]) / (2.0 * epsilon);
        finite_difference_hessian[row * N3 + column] = finite_difference;
        finite_difference_frobenius += finite_difference * finite_difference;
        const double difference = hessian_out[row * N3 + column] - finite_difference;
        difference_frobenius += difference * difference;
        const double error = std::abs(difference);
        if (error > maximum_error) {
          maximum_error = error;
          maximum_row = row * N3 + column;
        }
      }
    }
    finite_difference_frobenius = std::sqrt(finite_difference_frobenius);
    difference_frobenius = std::sqrt(difference_frobenius);
    const double relative_error =
      difference_frobenius /
      std::max(finite_difference_frobenius, std::numeric_limits<double>::min());
    printf(
      "Analytic force validation: max_abs_error=%.12g max_row=%d "
      "epsilon=%.12g fd_frobenius=%.12g difference_frobenius=%.12g "
      "relative_frobenius_error=%.12g\n",
      maximum_error, maximum_row, epsilon, finite_difference_frobenius,
      difference_frobenius, relative_error);
    if (const char* threshold = std::getenv("GPUMD_VALIDATE_MAX_ABS_ERROR")) {
      const double value = std::atof(threshold);
      if (!std::isfinite(value) || value < 0.0 || maximum_error > value)
        validation_failed = true;
    }
    if (const char* threshold = std::getenv("GPUMD_VALIDATE_MAX_REL_ERROR")) {
      const double value = std::atof(threshold);
      if (!std::isfinite(value) || value < 0.0 || relative_error > value)
        validation_failed = true;
    }
    if (validation_hessian_soa)
      validation_hessian_soa->assign(
        finite_difference_hessian.begin(), finite_difference_hessian.end());
    if (const char* path = std::getenv("GPUMD_DUMP_FD_HESSIAN")) {
      std::ofstream output(path);
      output << std::scientific << std::setprecision(17);
      output << "# coordinate_order=soa\n";
      output << "# matrix_order=row_major\n";
      output << "# definition=minus_force_jacobian\n";
      output << "# unit=eV/A^2\n";
      for (int row = 0; row < N3; ++row) {
        for (int column = 0; column < N3; ++column)
          output << (column ? " " : "") << finite_difference_hessian[row * N3 + column];
        output << "\n";
      }
    }
    if (const char* path = std::getenv("GPUMD_DUMP_DESCRIPTOR_FD_HESSIAN")) {
      std::ofstream output(path);
      output << "# coordinate_order=atom_major\n";
      output << "# matrix_order=row_major\n";
      output << "# unit=descriptor/A^2\n";
      double descriptor_epsilon =
        std::getenv("GPUMD_DESCRIPTOR_FD_EPSILON")
        ? std::atof(std::getenv("GPUMD_DESCRIPTOR_FD_EPSILON"))
        : 1.0e-2;
      if (!std::isfinite(descriptor_epsilon) || descriptor_epsilon <= 0.0) {
        printf(
          "Descriptor Hessian validation: epsilon must be finite and positive.\n");
        validation_failed = true;
        descriptor_epsilon = 1.0e-2;
      }
      std::vector<float> descriptor_zero(descriptor_count);
      const int num_s_terms =
        (nep->paramb.L_max + 1) * (nep->paramb.L_max + 1) - 1;
      const int sum_count = N_total * (nep->paramb.n_max_angular + 1) *
        num_s_terms;
      std::vector<float> sum_zero(sum_count);
      std::vector<float> corner_sums(4 * sum_count);
      force.compute(
        box, atom.position_per_atom, atom.type, empty_groups,
        atom.potential_per_atom, atom.force_per_atom,
        atom.virial_per_atom);
      CHECK(gpuDeviceSynchronize());
      nep->nep_data.q_descriptors.copy_to_host(descriptor_zero.data());
      nep->nep_data.sum_fxyz.copy_to_host(sum_zero.data());
      std::vector<double> base_position(position_save.size());
      std::vector<double> displaced_position(position_save.size());
      std::vector<float> corner_descriptors(4 * descriptor_count);
      position_save.copy_to_host(base_position.data());
      if (std::getenv("GPUMD_DUMP_ANALYTIC_Q_DERIVATIVES") != nullptr) {
        for (int atom = 0; atom < N_total; ++atom) {
          printf(
            "QFDPOS %d %.17g %.17g %.17g\n",
            atom,
            base_position[atom],
            base_position[N_total + atom],
            base_position[2 * N_total + atom]);
        }
      }
      for (int dump_atom = 0; dump_atom < N_total; ++dump_atom) {
        std::vector<double> descriptor_hessians(dim * N3 * N3, 0.0);
        for (int row = 0; row < N3; ++row) {
          for (int column = 0; column < N3; ++column) {
            if (row == column) {
              for (int sign = 0; sign < 2; ++sign) {
                displaced_position = base_position;
                const int displaced_coordinate =
                  (column % 3) * N + column / 3;
                displaced_position[displaced_coordinate] +=
                  sign ? -descriptor_epsilon : descriptor_epsilon;
                atom.position_per_atom.copy_from_host(displaced_position.data());
                force.compute(
                  box, atom.position_per_atom, atom.type, empty_groups,
                  atom.potential_per_atom, atom.force_per_atom,
                  atom.virial_per_atom);
                CHECK(gpuDeviceSynchronize());
                nep->nep_data.q_descriptors.copy_to_host(
                  corner_descriptors.data() + sign * descriptor_count);
                nep->nep_data.sum_fxyz.copy_to_host(
                  corner_sums.data() + sign * sum_count);
                if (std::getenv("GPUMD_DUMP_ANALYTIC_Q_DERIVATIVES") != nullptr) {
                  const int sum_index = dump_atom;
                  printf(
                    "QSCORNER row=%d col=%d sign=%d base=%.9g value=%.9g\n",
                    row, column, sign, sum_zero[sum_index],
                    corner_sums[sign * sum_count + sum_index]);
                }
              }
              if (std::getenv("GPUMD_DUMP_ANALYTIC_Q_DERIVATIVES") != nullptr) {
                for (int descriptor = 0; descriptor < dim; ++descriptor) {
                  const int index = descriptor * N_total + dump_atom;
                  const double first_difference =
                    (static_cast<double>(corner_descriptors[index]) -
                     static_cast<double>(
                       corner_descriptors[descriptor_count + index])) /
                    (2.0 * descriptor_epsilon);
                  printf(
                    "QGRADF %d %d %d %.10g\n",
                    dump_atom, row, descriptor, first_difference);
                }
              }
            } else {
              for (int sign = 0; sign < 4; ++sign) {
                const double row_sign =
                  (sign == 0 || sign == 1) ? descriptor_epsilon :
                                             -descriptor_epsilon;
                const double column_sign =
                  (sign == 0 || sign == 2) ? descriptor_epsilon :
                                             -descriptor_epsilon;
                displaced_position = base_position;
                const int displaced_row = (row % 3) * N + row / 3;
                const int displaced_column =
                  (column % 3) * N + column / 3;
                displaced_position[displaced_row] += row_sign;
                displaced_position[displaced_column] += column_sign;
                atom.position_per_atom.copy_from_host(displaced_position.data());
                force.compute(
                  box, atom.position_per_atom, atom.type, empty_groups,
                  atom.potential_per_atom, atom.force_per_atom,
                  atom.virial_per_atom);
                CHECK(gpuDeviceSynchronize());
                nep->nep_data.q_descriptors.copy_to_host(
                  corner_descriptors.data() + sign * descriptor_count);
                nep->nep_data.sum_fxyz.copy_to_host(
                  corner_sums.data() + sign * sum_count);
                if (std::getenv("GPUMD_DUMP_ANALYTIC_Q_DERIVATIVES") != nullptr) {
                  const int sum_index = dump_atom;
                  printf(
                    "QSCORNER row=%d col=%d sign=%d base=%.9g value=%.9g\n",
                    row, column, sign, sum_zero[sum_index],
                    corner_sums[sign * sum_count + sum_index]);
                }
              }
        }

        if (dump_atom == 0) {
          const int angular_descriptor = nep->paramb.n_max_radial + 1;
          const int index = angular_descriptor * N_total + dump_atom;
          printf(
            "QCORNER row=%d col=%d base=%.9g pp=%.9g mp=%.9g pm=%.9g mm=%.9g\n",
            row, column, descriptor_zero[index],
            corner_descriptors[index],
            corner_descriptors[descriptor_count + index],
            corner_descriptors[2 * descriptor_count + index],
            corner_descriptors[3 * descriptor_count + index]);
        }
        atom.position_per_atom.copy_from_device(position_save.data());
            for (int descriptor = 0; descriptor < dim; ++descriptor) {
              const int index = descriptor * N_total + dump_atom;
              double second_difference;
              if (row == column) {
                second_difference =
                  (static_cast<double>(corner_descriptors[index]) +
                   static_cast<double>(
                     corner_descriptors[descriptor_count + index]) -
                   2.0 * descriptor_zero[index]) /
                  (descriptor_epsilon * descriptor_epsilon);
              } else {
                second_difference =
                  (static_cast<double>(corner_descriptors[index]) -
                   static_cast<double>(
                     corner_descriptors[descriptor_count + index]) -
                   static_cast<double>(
                     corner_descriptors[2 * descriptor_count + index]) +
                   static_cast<double>(
                     corner_descriptors[3 * descriptor_count + index])) /
                  (4.0 * descriptor_epsilon * descriptor_epsilon);
              }
              descriptor_hessians[(descriptor * N3 + row) * N3 + column] =
                second_difference;
            }
            if (std::getenv("GPUMD_DUMP_ANALYTIC_Q_DERIVATIVES") != nullptr) {
              for (int n = 0; n <= nep->paramb.n_max_angular; ++n) {
                for (int component = 0; component < num_s_terms; ++component) {
                  const int index =
                    (n * num_s_terms + component) * N_total + dump_atom;
                  double second_difference;
                  if (row == column) {
                    second_difference =
                      (static_cast<double>(corner_sums[index]) +
                       static_cast<double>(
                         corner_sums[sum_count + index]) -
                       2.0 * sum_zero[index]) /
                      (descriptor_epsilon * descriptor_epsilon);
                  } else {
                    second_difference =
                      (static_cast<double>(corner_sums[index]) -
                       static_cast<double>(corner_sums[sum_count + index]) -
                       static_cast<double>(corner_sums[2 * sum_count + index]) +
                       static_cast<double>(corner_sums[3 * sum_count + index])) /
                      (4.0 * descriptor_epsilon * descriptor_epsilon);
                  }
                  output << "QSFD " << dump_atom << ' ' << n << ' '
                    << component << ' ' << row << ' ' << column << ' '
                    << second_difference << "\n";
                }
              }
            }
          }
        }

        for (int descriptor = 0; descriptor < dim; ++descriptor) {
          output << "# atom=" << dump_atom << " descriptor=" << descriptor << "\n";
          for (int row = 0; row < N3; ++row) {
            for (int column = 0; column < N3; ++column) {
              output << (column ? " " : "") <<
                descriptor_hessians[(descriptor * N3 + row) * N3 + column];
            }
            output << "\n";
          }
        }
      }
      atom.position_per_atom.copy_from_device(position_save.data());
      atom.force_per_atom.copy_from_device(force_save.data());
      atom.potential_per_atom.copy_from_device(potential_save.data());
      atom.virial_per_atom.copy_from_device(virial_save.data());
      force.compute(
        box, atom.position_per_atom, atom.type, empty_groups,
        atom.potential_per_atom, atom.force_per_atom,
        atom.virial_per_atom);
      CHECK(gpuDeviceSynchronize());
    }
  }
  if (const char* path = std::getenv("GPUMD_DUMP_ANALYTIC_HESSIAN")) {
    write_hessian_matrix(
      path, hessian_out, N_total, dump_analytic_atom_major);
  }

  if (validation_failed) {
    printf(
      "NEP analytic Hessian validation failed; clearing result for fallback.\n");
    hessian_out.clear();
    return finish(false);
  }

  return finish(true);
}
