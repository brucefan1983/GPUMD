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

#ifdef NEP_SPECIALIZED_RUNTIME_BUILD
/*
 * Runtime-specialized kernels for NEP-family training.
 *
 * The normal nep executable compiles this file as an empty translation unit.
 * NEP_Compile invokes nvcc again at runtime with:
 *   -DNEP_SPECIALIZED_RUNTIME_BUILD
 * and a generated nep_special_config.cuh containing compile-time constants.
 *
 * Generic NEP-family kernels remain the reference implementation.
 */

#include <nep_special_config.cuh>
#define NEP_SPECIALIZED_SOURCE_INTERFACE_VERSION 1
#if !defined(NEP_SPECIAL_CONFIG_INTERFACE_VERSION)
#error "Missing NEP specialization interface version."
#elif NEP_SPECIAL_CONFIG_INTERFACE_VERSION != NEP_SPECIALIZED_SOURCE_INTERFACE_VERSION
#error "Incompatible NEP specialization interface version."
#endif
#include "utilities/nep_utilities.cuh"
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float get_rc_radial_jit(const int t1, const int t2)
{
#if RC_RADIAL_COMMON_JIT
  (void)t1;
  (void)t2;
  return RC_RADIAL_COMMON_VALUE_JIT;
#else
  return 0.5f * (RC_RADIAL_JIT[t1] + RC_RADIAL_JIT[t2]);
#endif
}

__device__ __forceinline__ float get_rc_angular_jit(const int t1, const int t2)
{
#if RC_ANGULAR_COMMON_JIT
  (void)t1;
  (void)t2;
  return RC_ANGULAR_COMMON_VALUE_JIT;
#else
  return 0.5f * (RC_ANGULAR_JIT[t1] + RC_ANGULAR_JIT[t2]);
#endif
}

__device__ __forceinline__ int descriptor_type_index_jit(const int t1, const int t2)
{
#if DESCRIPTOR_USE_CJ_JIT
  (void)t1;
  return t2;
#else
  return t1 * NUM_TYPES_JIT + t2;
#endif
}

__device__ __forceinline__ const float* get_c_jit(const float* parameters)
{
  return parameters + C_OFFSET_JIT;
}

__global__ void descriptor_radial_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  float* descriptors)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c_jit(parameters);
  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];
  float q[DIM_RADIAL_JIT] = {0.0f};

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = NN_sum[n1] + i1;
    const int n2 = NL[index];
    const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
    const float x12 = x12_all[index];
    const float y12 = y12_all[index];
    const float z12 = z12_all[index];
    const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
    const float rc = get_rc_radial_jit(t1, t2);
    const float rcinv = 1.0f / rc;
    float fc12;
    find_fc(rc, rcinv, d12, fc12);
    float fn12[BASIS_SIZE_RADIAL_JIT + 1];
    find_fn(BASIS_SIZE_RADIAL_JIT, rcinv, d12, fc12, fn12);

    const int type_index = descriptor_type_index_jit(t1, t2);
#pragma unroll
    for (int n = 0; n <= N_MAX_RADIAL_JIT; ++n) {
      float gn12 = 0.0f;
#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_RADIAL_JIT; ++k) {
        const int c_index =
          get_c_index(type_index, n, k, N_MAX_RADIAL_JIT, BASIS_SIZE_RADIAL_JIT);
        gn12 += fn12[k] * c[c_index];
      }
      q[n] += gn12;
    }
  }

#pragma unroll
  for (int n = 0; n <= N_MAX_RADIAL_JIT; ++n) {
    descriptors[n1 + n * N] = q[n];
  }
}

__global__ void descriptor_angular_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  float* descriptors,
  float* sum_fxyz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c_jit(parameters);
  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];
  float q[DIM_ANGULAR_JIT] = {0.0f};

#pragma unroll
  for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
    float s[NUM_ABC_JIT] = {0.0f};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      const int index = NN_sum[n1] + i1;
      const int n2 = NL[index];
      const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
      const float x12 = x12_all[index];
      const float y12 = y12_all[index];
      const float z12 = z12_all[index];
      const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      const float rc = get_rc_angular_jit(t1, t2);
      const float rcinv = 1.0f / rc;
      float fc12;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[BASIS_SIZE_ANGULAR_JIT + 1];
      find_fn(BASIS_SIZE_ANGULAR_JIT, rcinv, d12, fc12, fn12);

      const int type_index = descriptor_type_index_jit(t1, t2);
      float gn12 = 0.0f;
#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_ANGULAR_JIT; ++k) {
        const int c_index = get_c_index(
          type_index,
          n,
          k,
          N_MAX_ANGULAR_JIT,
          BASIS_SIZE_ANGULAR_JIT,
          NUM_C_RADIAL_JIT);
        gn12 += fn12[k] * c[c_index];
      }
      accumulate_s(L_MAX_JIT, d12, x12, y12, z12, gn12, s);
    }

    find_q(
      L_MAX_JIT,
      HAS_Q_222_JIT,
      HAS_Q_1111_JIT,
      HAS_Q_112_JIT,
      HAS_Q_123_JIT,
      HAS_Q_233_JIT,
      HAS_Q_134_JIT,
      N_MAX_ANGULAR_JIT + 1,
      n,
      s,
      q);

#pragma unroll
    for (int abc = 0; abc < NUM_ABC_JIT; ++abc) {
      sum_fxyz[(n * NUM_ABC_JIT + abc) * N + n1] = s[abc];
    }
  }

#pragma unroll
  for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
#pragma unroll
    for (int l = 0; l < NUM_L_JIT; ++l) {
      const int ln = l * (N_MAX_ANGULAR_JIT + 1) + n;
      descriptors[n1 + (DIM_RADIAL_JIT + ln) * N] = q[ln];
    }
  }
}

__device__ __forceinline__ void apply_scalar_ann_jit(
  const float* parameters,
  const int set_offset,
  const int type,
  float* q,
  float& F,
  float* Fp)
{
  const float* wb = parameters + set_offset + type * ONE_ANN_NO_BIAS_JIT;
  const float* b =
    parameters + set_offset + NUM_TYPES_JIT * ONE_ANN_NO_BIAS_JIT;

  if (NUM_HIDDEN_LAYERS_JIT == 2) {
    apply_ann_two_layers(
      ANN_DIM_JIT,
      NUM_NEURONS1_JIT,
      NUM_NEURONS2_JIT,
      wb,
      wb + NUM_NEURONS1_JIT * ANN_DIM_JIT,
      wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1),
      wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1 + NUM_NEURONS2_JIT),
      wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1 + NUM_NEURONS2_JIT) +
        NUM_NEURONS2_JIT,
      b,
      q,
      F,
      Fp);
  } else {
    apply_ann_one_layer(
      ANN_DIM_JIT,
      NUM_NEURONS1_JIT,
      wb,
      wb + NUM_NEURONS1_JIT * ANN_DIM_JIT,
      wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1),
      b,
      q,
      F,
      Fp);
  }
}

__global__ void ann_nep_jit(
  const int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp_out)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  float q[ANN_DIM_JIT] = {0.0f};
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    q[d] = descriptors[n1 + d * N] * q_scaler[d];
  }

  float F = 0.0f;
  float Fp[ANN_DIM_JIT] = {0.0f};
  apply_scalar_ann_jit(parameters, 0, t, q, F, Fp);
  pe[n1] = F;

#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    Fp_out[n1 + d * N] = Fp[d] * q_scaler[d];
  }
}

__global__ void ann_temperature_jit(
  const int N,
  const int* type,
  const float* descriptors,
  float* q_scaler,
  const float* temperature,
  const float* parameters,
  float* pe,
  float* Fp_out)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  float q[ANN_DIM_JIT] = {0.0f};
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT - 1; ++d) {
    q[d] = descriptors[n1 + d * N] * q_scaler[d];
  }
  q_scaler[ANN_DIM_JIT - 1] = 0.001f;
  q[ANN_DIM_JIT - 1] = temperature[n1] * 0.001f;

  float F = 0.0f;
  float Fp[ANN_DIM_JIT] = {0.0f};
  const float* wb = parameters + t * ONE_ANN_NO_BIAS_JIT;
  const float* b = parameters + NUM_TYPES_JIT * ONE_ANN_NO_BIAS_JIT;
  apply_ann_one_layer(
    ANN_DIM_JIT,
    NUM_NEURONS1_JIT,
    wb,
    wb + NUM_NEURONS1_JIT * ANN_DIM_JIT,
    wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1),
    b,
    q,
    F,
    Fp);

  pe[n1] = F;
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    Fp_out[n1 + d * N] = Fp[d] * q_scaler[d];
  }
}

__global__ void ann_charge_jit(
  const int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp_out,
  float* charge_out,
  float* charge_derivative_out)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  float q[ANN_DIM_JIT] = {0.0f};
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    q[d] = descriptors[n1 + d * N] * q_scaler[d];
  }

  const float* block = parameters + t * ONE_ANN_NO_BIAS_JIT;
  const float* w0 = block;
  const float* b0 = w0 + NUM_NEURONS1_JIT * ANN_DIM_JIT;
  const float* w1 = b0 + NUM_NEURONS1_JIT;
  const float* b1 = parameters + NUM_TYPES_JIT * ONE_ANN_NO_BIAS_JIT + 1;

  float F = 0.0f;
  float charge = 0.0f;
  float Fp[ANN_DIM_JIT] = {0.0f};
  float charge_derivative[ANN_DIM_JIT] = {0.0f};

  apply_ann_one_layer_charge(
    ANN_DIM_JIT,
    NUM_NEURONS1_JIT,
    w0,
    b0,
    w1,
    b1,
    q,
    F,
    Fp,
    charge,
    charge_derivative);

  pe[n1] = F;
  charge_out[n1] = charge;
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    Fp_out[n1 + d * N] = Fp[d] * q_scaler[d];
    charge_derivative_out[n1 + d * N] =
      charge_derivative[d] * q_scaler[d];
  }
}

__global__ void ann_vdw_jit(
  const int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp_out,
  float* C6_out,
  float* C6_derivative_out)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  float q[ANN_DIM_JIT] = {0.0f};
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    q[d] = descriptors[n1 + d * N] * q_scaler[d];
  }

  const float* block = parameters + t * ONE_ANN_NO_BIAS_JIT;
  const float* w0 = block;
  const float* b0 = w0 + NUM_NEURONS1_JIT * ANN_DIM_JIT;
  const float* w1 = b0 + NUM_NEURONS1_JIT;
  const float* b1 = parameters + NUM_TYPES_JIT * ONE_ANN_NO_BIAS_JIT;

  float F = 0.0f;
  float C6 = 0.0f;
  float Fp[ANN_DIM_JIT] = {0.0f};
  float C6_derivative[ANN_DIM_JIT] = {0.0f};

  apply_ann_one_layer_vdw(
    ANN_DIM_JIT,
    NUM_NEURONS1_JIT,
    w0,
    b0,
    w1,
    b1,
    q,
    F,
    Fp,
    C6,
    C6_derivative);

  pe[n1] = F;
  const float C6_exp =
    C6_REF_SQRT_JIT[t] * expf(C6 * C6_SCALING_FACTOR_JIT);
  C6_out[n1] = C6_exp;

#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    Fp_out[n1 + d * N] = Fp[d] * q_scaler[d];
    C6_derivative_out[n1 + d * N] =
      C6_exp * C6_SCALING_FACTOR_JIT * C6_derivative[d] * q_scaler[d];
  }
}

__global__ void ann_charge_vdw_jit(
  const int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp_out,
  float* charge_out,
  float* charge_derivative_out,
  float* C6_out,
  float* C6_derivative_out)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  float q[ANN_DIM_JIT] = {0.0f};
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    q[d] = descriptors[n1 + d * N] * q_scaler[d];
  }

  const float* block = parameters + t * ONE_ANN_NO_BIAS_JIT;
  const float* w0 = block;
  const float* b0 = w0 + NUM_NEURONS1_JIT * ANN_DIM_JIT;
  const float* w1 = b0 + NUM_NEURONS1_JIT;
  const float* b1 = parameters + NUM_TYPES_JIT * ONE_ANN_NO_BIAS_JIT + 1;

  float F = 0.0f;
  float charge = 0.0f;
  float C6 = 0.0f;
  float Fp[ANN_DIM_JIT] = {0.0f};
  float charge_derivative[ANN_DIM_JIT] = {0.0f};
  float C6_derivative[ANN_DIM_JIT] = {0.0f};

  apply_ann_one_layer_charge_vdw(
    ANN_DIM_JIT,
    NUM_NEURONS1_JIT,
    w0,
    b0,
    w1,
    b1,
    q,
    F,
    Fp,
    charge,
    charge_derivative,
    C6,
    C6_derivative);

  pe[n1] = F;
  charge_out[n1] = charge;
  const float C6_exp =
    C6_REF_SQRT_JIT[t] * expf(C6 * C6_SCALING_FACTOR_JIT);
  C6_out[n1] = C6_exp;

#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    Fp_out[n1 + d * N] = Fp[d] * q_scaler[d];
    charge_derivative_out[n1 + d * N] =
      charge_derivative[d] * q_scaler[d];
    C6_derivative_out[n1 + d * N] =
      C6_exp * C6_SCALING_FACTOR_JIT * C6_derivative[d] * q_scaler[d];
  }
}

__global__ void ann_tnep_pol_jit(
  const int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* virial,
  float* Fp_out)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  float q[ANN_DIM_JIT] = {0.0f};
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    q[d] = descriptors[n1 + d * N] * q_scaler[d];
  }

  // Generic TNEP train_mode=2:
  // second ANN set -> scalar polarizability;
  // first ANN set -> tensor descriptor derivative.
  float F = 0.0f;
  float Fp_pol[ANN_DIM_JIT] = {0.0f};
  apply_scalar_ann_jit(
    parameters, ANN_SET_SIZE_JIT, t, q, F, Fp_pol);

  virial[n1] = F;
  virial[n1 + N] = F;
  virial[n1 + 2 * N] = F;

  float Fp[ANN_DIM_JIT] = {0.0f};
  apply_scalar_ann_jit(parameters, 0, t, q, F, Fp);

#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    Fp_out[n1 + d * N] = Fp[d] * q_scaler[d];
  }
}

__device__ __forceinline__ float effective_fp_jit(
  const int index,
  const int atom,
  const float* Fp,
  const float* charge_derivative,
  const float* D_real,
  const float* C6_derivative,
  const float* D_C6)
{
  float value = Fp[index];

#if NEP_MODEL_MODE_JIT == NEP_MODEL_CHARGE || \
    NEP_MODEL_MODE_JIT == NEP_MODEL_CHARGE_VDW
  value += charge_derivative[index] * D_real[atom];
#else
  (void)charge_derivative;
  (void)D_real;
#endif

#if NEP_MODEL_MODE_JIT == NEP_MODEL_VDW || \
    NEP_MODEL_MODE_JIT == NEP_MODEL_CHARGE_VDW
  value += C6_derivative[index] * D_C6[atom];
#else
  (void)C6_derivative;
  (void)D_C6;
#endif

  return value;
}

__global__ void force_radial_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  const float* Fp,
  const float* charge_derivative,
  const float* D_real,
  const float* C6_derivative,
  const float* D_C6,
  const int is_dipole,
  float* fx,
  float* fy,
  float* fz,
  float* virial)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c_jit(parameters);
  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];

  float sxx = 0.0f;
  float syy = 0.0f;
  float szz = 0.0f;
  float sxy = 0.0f;
  float syz = 0.0f;
  float szx = 0.0f;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = NN_sum[n1] + i1;
    const int n2 = NL[index];
    const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
    const float r12[3] = {
      x12_all[index], y12_all[index], z12_all[index]};
    const float d12 =
      sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    const float d12inv = 1.0f / d12;

    const float rc = get_rc_radial_jit(t1, t2);
    const float rcinv = 1.0f / rc;
    float fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

    float fn12[BASIS_SIZE_RADIAL_JIT + 1];
    float fnp12[BASIS_SIZE_RADIAL_JIT + 1];
    find_fn_and_fnp(
      BASIS_SIZE_RADIAL_JIT,
      rcinv,
      d12,
      fc12,
      fcp12,
      fn12,
      fnp12);

    const int type_index = descriptor_type_index_jit(t1, t2);
    float f12[3] = {0.0f};

#pragma unroll
    for (int n = 0; n <= N_MAX_RADIAL_JIT; ++n) {
      float gnp12 = 0.0f;
#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_RADIAL_JIT; ++k) {
        const int c_index =
          get_c_index(type_index, n, k, N_MAX_RADIAL_JIT, BASIS_SIZE_RADIAL_JIT);
        gnp12 += fnp12[k] * c[c_index];
      }

      const int fp_index = n1 + n * N;
      float tmp12 = effective_fp_jit(
        fp_index,
        n1,
        Fp,
        charge_derivative,
        D_real,
        C6_derivative,
        D_C6);
      tmp12 *= gnp12 * d12inv;

      f12[0] += tmp12 * r12[0];
      f12[1] += tmp12 * r12[1];
      f12[2] += tmp12 * r12[2];
    }

    atomicAdd(&fx[n1], f12[0]);
    atomicAdd(&fy[n1], f12[1]);
    atomicAdd(&fz[n1], f12[2]);
    atomicAdd(&fx[n2], -f12[0]);
    atomicAdd(&fy[n2], -f12[1]);
    atomicAdd(&fz[n2], -f12[2]);

#if NEP_MODEL_MODE_JIT == NEP_MODEL_TNEP
    if (is_dipole) {
      const float r2 =
        r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      sxx -= r2 * f12[0];
      syy -= r2 * f12[1];
      szz -= r2 * f12[2];
    } else {
      sxx -= r12[0] * f12[0];
      syy -= r12[1] * f12[1];
      szz -= r12[2] * f12[2];
    }
#else
    (void)is_dipole;
    sxx -= r12[0] * f12[0];
    syy -= r12[1] * f12[1];
    szz -= r12[2] * f12[2];
#endif
    sxy -= r12[0] * f12[1];
    syz -= r12[1] * f12[2];
    szx -= r12[2] * f12[0];
  }

  virial[n1] += sxx;
  virial[n1 + N] += syy;
  virial[n1 + 2 * N] += szz;

#if NEP_MODEL_MODE_JIT == NEP_MODEL_NEP || \
    NEP_MODEL_MODE_JIT == NEP_MODEL_TNEP
  // These paths zero only the diagonal virial before radial force,
  // so radial force initializes the shear components.
  virial[n1 + 3 * N] = sxy;
  virial[n1 + 4 * N] = syz;
  virial[n1 + 5 * N] = szx;
#else
  // qNEP/vdW paths can already contain long-range virial contributions.
  virial[n1 + 3 * N] += sxy;
  virial[n1 + 4 * N] += syz;
  virial[n1 + 5 * N] += szx;
#endif
}

__global__ void force_angular_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  const float* Fp_in,
  const float* charge_derivative,
  const float* D_real,
  const float* C6_derivative,
  const float* D_C6,
  const float* sum_fxyz_in,
  const int is_dipole,
  float* fx,
  float* fy,
  float* fz,
  float* virial)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c_jit(parameters);
  float Fp[DIM_ANGULAR_JIT] = {0.0f};
  float sum_fxyz[(N_MAX_ANGULAR_JIT + 1) * NUM_OF_ABC];

#pragma unroll
  for (int d = 0; d < DIM_ANGULAR_JIT; ++d) {
    const int index = (DIM_RADIAL_JIT + d) * N + n1;
    Fp[d] = effective_fp_jit(
      index,
      n1,
      Fp_in,
      charge_derivative,
      D_real,
      C6_derivative,
      D_C6);
  }

#pragma unroll
  for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
#pragma unroll
    for (int abc = 0; abc < NUM_ABC_JIT; ++abc) {
      sum_fxyz[n * NUM_OF_ABC + abc] =
        sum_fxyz_in[(n * NUM_ABC_JIT + abc) * N + n1];
    }
  }

  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];

  float sxx = 0.0f;
  float syy = 0.0f;
  float szz = 0.0f;
  float sxy = 0.0f;
  float syz = 0.0f;
  float szx = 0.0f;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = NN_sum[n1] + i1;
    const int n2 = NL[index];
    const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
    const float r12[3] = {
      x12_all[index], y12_all[index], z12_all[index]};
    const float d12 =
      sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);

    const float rc = get_rc_angular_jit(t1, t2);
    const float rcinv = 1.0f / rc;
    float fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

    float fn12[BASIS_SIZE_ANGULAR_JIT + 1];
    float fnp12[BASIS_SIZE_ANGULAR_JIT + 1];
    find_fn_and_fnp(
      BASIS_SIZE_ANGULAR_JIT,
      rcinv,
      d12,
      fc12,
      fcp12,
      fn12,
      fnp12);

    const int type_index = descriptor_type_index_jit(t1, t2);
    float f12[3] = {0.0f};

#pragma unroll
    for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
      float gn12 = 0.0f;
      float gnp12 = 0.0f;

#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_ANGULAR_JIT; ++k) {
        const int c_index = get_c_index(
          type_index,
          n,
          k,
          N_MAX_ANGULAR_JIT,
          BASIS_SIZE_ANGULAR_JIT,
          NUM_C_RADIAL_JIT);
        const float c_value = c[c_index];
        gn12 += fn12[k] * c_value;
        gnp12 += fnp12[k] * c_value;
      }

      accumulate_f12(
        L_MAX_JIT,
        HAS_Q_222_JIT,
        HAS_Q_1111_JIT,
        HAS_Q_112_JIT,
        HAS_Q_123_JIT,
        HAS_Q_233_JIT,
        HAS_Q_134_JIT,
        NUM_L_JIT,
        n,
        N_MAX_ANGULAR_JIT + 1,
        d12,
        r12,
        gn12,
        gnp12,
        Fp,
        sum_fxyz,
        f12);
    }

    atomicAdd(&fx[n1], f12[0]);
    atomicAdd(&fy[n1], f12[1]);
    atomicAdd(&fz[n1], f12[2]);
    atomicAdd(&fx[n2], -f12[0]);
    atomicAdd(&fy[n2], -f12[1]);
    atomicAdd(&fz[n2], -f12[2]);

#if NEP_MODEL_MODE_JIT == NEP_MODEL_TNEP
    if (is_dipole) {
      const float r2 =
        r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      sxx -= r2 * f12[0];
      syy -= r2 * f12[1];
      szz -= r2 * f12[2];
    } else {
      sxx -= r12[0] * f12[0];
      syy -= r12[1] * f12[1];
      szz -= r12[2] * f12[2];
    }
#else
    (void)is_dipole;
    sxx -= r12[0] * f12[0];
    syy -= r12[1] * f12[1];
    szz -= r12[2] * f12[2];
#endif
    sxy -= r12[0] * f12[1];
    syz -= r12[1] * f12[2];
    szx -= r12[2] * f12[0];
  }

  virial[n1] += sxx;
  virial[n1 + N] += syy;
  virial[n1 + 2 * N] += szz;
  virial[n1 + 3 * N] += sxy;
  virial[n1 + 4 * N] += syz;
  virial[n1 + 5 * N] += szx;
}

extern "C" int nep_train_specialized_interface_version()
{
  return NEP_SPECIALIZED_SOURCE_INTERFACE_VERSION;
}

extern "C" int nep_train_launch_descriptor_radial(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  float* descriptors)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  descriptor_radial_jit<<<grid_size, block_size>>>(
    N, NN_sum, NN, NL, type, x12, y12, z12, parameters, descriptors);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_descriptor_angular(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  float* descriptors,
  float* sum_fxyz)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  descriptor_angular_jit<<<grid_size, block_size>>>(
    N,
    NN_sum,
    NN,
    NL,
    type,
    x12,
    y12,
    z12,
    parameters,
    descriptors,
    sum_fxyz);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_ann_nep(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  ann_nep_jit<<<grid_size, block_size>>>(
    N, type, descriptors, q_scaler, parameters, pe, Fp);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_ann_temperature(
  int N,
  const int* type,
  const float* descriptors,
  float* q_scaler,
  const float* temperature,
  const float* parameters,
  float* pe,
  float* Fp)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  ann_temperature_jit<<<grid_size, block_size>>>(
    N,
    type,
    descriptors,
    q_scaler,
    temperature,
    parameters,
    pe,
    Fp);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_ann_charge(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp,
  float* charge,
  float* charge_derivative)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  ann_charge_jit<<<grid_size, block_size>>>(
    N,
    type,
    descriptors,
    q_scaler,
    parameters,
    pe,
    Fp,
    charge,
    charge_derivative);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_ann_vdw(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp,
  float* C6,
  float* C6_derivative)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  ann_vdw_jit<<<grid_size, block_size>>>(
    N,
    type,
    descriptors,
    q_scaler,
    parameters,
    pe,
    Fp,
    C6,
    C6_derivative);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_ann_charge_vdw(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp,
  float* charge,
  float* charge_derivative,
  float* C6,
  float* C6_derivative)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  ann_charge_vdw_jit<<<grid_size, block_size>>>(
    N,
    type,
    descriptors,
    q_scaler,
    parameters,
    pe,
    Fp,
    charge,
    charge_derivative,
    C6,
    C6_derivative);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_ann_tnep_pol(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* virial,
  float* Fp)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  ann_tnep_pol_jit<<<grid_size, block_size>>>(
    N, type, descriptors, q_scaler, parameters, virial, Fp);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_force_radial(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  const float* Fp,
  const float* charge_derivative,
  const float* D_real,
  const float* C6_derivative,
  const float* D_C6,
  int is_dipole,
  float* fx,
  float* fy,
  float* fz,
  float* virial)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  force_radial_jit<<<grid_size, block_size>>>(
    N,
    NN_sum,
    NN,
    NL,
    type,
    x12,
    y12,
    z12,
    parameters,
    Fp,
    charge_derivative,
    D_real,
    C6_derivative,
    D_C6,
    is_dipole,
    fx,
    fy,
    fz,
    virial);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_force_angular(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  const float* Fp,
  const float* charge_derivative,
  const float* D_real,
  const float* C6_derivative,
  const float* D_C6,
  const float* sum_fxyz,
  int is_dipole,
  float* fx,
  float* fy,
  float* fz,
  float* virial)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  force_angular_jit<<<grid_size, block_size>>>(
    N,
    NN_sum,
    NN,
    NL,
    type,
    x12,
    y12,
    z12,
    parameters,
    Fp,
    charge_derivative,
    D_real,
    C6_derivative,
    D_C6,
    sum_fxyz,
    is_dipole,
    fx,
    fy,
    fz,
    virial);
  return static_cast<int>(cudaGetLastError());
}

#endif // NEP_SPECIALIZED_RUNTIME_BUILD
