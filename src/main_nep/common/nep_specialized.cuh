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
#include <nep_special_config.cuh>
#define NEP_SPECIALIZED_SOURCE_INTERFACE_VERSION 4
#if !defined(NEP_SPECIAL_CONFIG_INTERFACE_VERSION)
#error "Missing NEP specialization interface version."
#elif NEP_SPECIAL_CONFIG_INTERFACE_VERSION != NEP_SPECIALIZED_SOURCE_INTERFACE_VERSION
#error "Incompatible NEP specialization interface version."
#endif
#include "utilities/nep_utilities.cuh"
#include <cuda_runtime.h>
#include <cmath>

static __device__ __forceinline__ float get_rc_radial_jit(const int t1, const int t2)
{
#if RC_RADIAL_COMMON_JIT
  (void)t1;
  (void)t2;
  return RC_RADIAL_COMMON_VALUE_JIT;
#else
  return 0.5f * (RC_RADIAL_JIT[t1] + RC_RADIAL_JIT[t2]);
#endif
}

static __device__ __forceinline__ float get_rc_angular_jit(const int t1, const int t2)
{
#if RC_ANGULAR_COMMON_JIT
  (void)t1;
  (void)t2;
  return RC_ANGULAR_COMMON_VALUE_JIT;
#else
  return 0.5f * (RC_ANGULAR_JIT[t1] + RC_ANGULAR_JIT[t2]);
#endif
}

static __device__ __forceinline__ int descriptor_type_index_jit(const int t1, const int t2)
{
#if DESCRIPTOR_USE_CJ_JIT
  (void)t1;
  return t2;
#else
  return t1 * NUM_TYPES_JIT + t2;
#endif
}

static __device__ __forceinline__ const float* get_c_jit(const float* parameters)
{
  return parameters + C_OFFSET_JIT;
}

static __global__ void descriptor_radial_jit(
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
  // Eight lanes own one center and load consecutive CSR neighbors.
  const int lane = threadIdx.x % 8;
  const unsigned mask = 0xffu << ((threadIdx.x % 32) / 8 * 8);
  const int n1 = (threadIdx.x + blockIdx.x * blockDim.x) / 8;
  if (n1 >= N) return;

  const float* c = get_c_jit(parameters);
  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];
  float q[DIM_RADIAL_JIT] = {0.0f};

  for (int first = 0; first < neighbor_number; first += 8) {
    float edge_q[DIM_RADIAL_JIT] = {0.0f};
    const int i1 = first + lane;
    if (i1 < neighbor_number) {
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
        edge_q[n] = gn12;
      }
    }
    // Preserve the baseline neighbor and basis contraction order.
    for (int source_lane = 0;
         source_lane < 8 && first + source_lane < neighbor_number;
         ++source_lane) {
#pragma unroll
      for (int n = 0; n <= N_MAX_RADIAL_JIT; ++n) {
        q[n] += __shfl_sync(mask, edge_q[n], source_lane, 8);
      }
    }
  }
  if (lane != 0) return;

#pragma unroll
  for (int n = 0; n <= N_MAX_RADIAL_JIT; ++n) {
    descriptors[n1 + n * N] = q[n];
  }
}

static __global__ void descriptor_angular_jit(
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
  // Share edge geometry and radial basis across five angular channels.
  for (int base = 0; base <= N_MAX_ANGULAR_JIT; base += 5) {
    float sums[5][NUM_ABC_JIT] = {0.0f};
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
#pragma unroll
      for (int lane = 0; lane < 5; ++lane) {
        const int n = base + lane;
        if (n > N_MAX_ANGULAR_JIT) continue;
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
        accumulate_s(L_MAX_JIT, d12, x12, y12, z12, gn12, sums[lane]);
      }
    }

#pragma unroll
    for (int lane = 0; lane < 5; ++lane) {
      const int n = base + lane;
      if (n > N_MAX_ANGULAR_JIT) continue;
      float* s = sums[lane];
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

static __device__ __forceinline__ float effective_fp_jit(
  const int index,
  const int atom,
  const float* Fp,
  const float* charge_derivative,
  const float* D_real,
  const float* C6_derivative,
  const float* D_C6)
{
  float value = Fp[index];

#if HAS_CHARGE_JIT
  value += charge_derivative[index] * D_real[atom];
#else
  (void)charge_derivative;
  (void)D_real;
#endif

#if HAS_VDW_JIT
  value += C6_derivative[index] * D_C6[atom];
#else
  (void)C6_derivative;
  (void)D_C6;
#endif

  return value;
}

static __device__ __forceinline__ void find_fn_and_fnp_radial_jit(
  const int n_max,
  const float rcinv,
  const float d12,
  const float fc12,
  const float fcp12,
  float* fn,
  float* fnp)
{
  // Keep coordinate rounding independent of the cooperative kernel layout.
  const float shifted = __fmaf_rn(d12, rcinv, -1.0f);
  float x = 2.0f * shifted * shifted - 1.0f;
  fn[0] = fc12;
  fnp[0] = fcp12;
  fn[1] = (x + 1.0f) * 0.5f * fc12;
  fnp[1] = 2.0f * shifted * rcinv * fc12 + (x + 1.0f) * 0.5f * fcp12;
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
    fnp[m] = (fnp_tmp * 2.0f * shifted * rcinv) * fc12 + fn_tmp2 * fcp12;
    fn[m] = fn_tmp2 * fc12;
  }
}

static __global__ void force_radial_jit(
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
  // Eight lanes own one center and load consecutive CSR neighbors.
  const int lane = threadIdx.x % 8;
  const unsigned mask = 0xffu << ((threadIdx.x % 32) / 8 * 8);
  const int n1 = (threadIdx.x + blockIdx.x * blockDim.x) / 8;
  if (n1 >= N) return;

  const float* c = get_c_jit(parameters);
  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];

  float center_force[3] = {0.0f};
  float sxx = 0.0f;
  float syy = 0.0f;
  float szz = 0.0f;
  float sxy = 0.0f;
  float syz = 0.0f;
  float szx = 0.0f;

  for (int first = 0; first < neighbor_number; first += 8) {
    float edge_r[3] = {0.0f};
    float edge_f[3] = {0.0f};
    const int i1 = first + lane;
    if (i1 < neighbor_number) {
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
      find_fn_and_fnp_radial_jit(
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

      atomicAdd(&fx[n2], -f12[0]);
      atomicAdd(&fy[n2], -f12[1]);
      atomicAdd(&fz[n2], -f12[2]);
#pragma unroll
      for (int axis = 0; axis < 3; ++axis) {
        edge_r[axis] = r12[axis];
        edge_f[axis] = f12[axis];
      }
    }
    // Lane zero consumes edges in their original order, including virial FMAs.
    for (int source_lane = 0;
         source_lane < 8 && first + source_lane < neighbor_number;
         ++source_lane) {
      float r12[3], f12[3];
#pragma unroll
      for (int axis = 0; axis < 3; ++axis) {
        r12[axis] = __shfl_sync(mask, edge_r[axis], source_lane, 8);
        f12[axis] = __shfl_sync(mask, edge_f[axis], source_lane, 8);
      }
      if (lane == 0) {
        center_force[0] += f12[0];
        center_force[1] += f12[1];
        center_force[2] += f12[2];
#if TENSOR_VIRIAL_JIT
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
    }
  }
  if (lane != 0) return;

  atomicAdd(&fx[n1], center_force[0]);
  atomicAdd(&fy[n1], center_force[1]);
  atomicAdd(&fz[n1], center_force[2]);

  virial[n1] += sxx;
  virial[n1 + N] += syy;
  virial[n1 + 2 * N] += szz;

#if INITIALIZE_RADIAL_SHEAR_JIT
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

static __global__ void force_angular_jit(
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
  float sum_fxyz[(N_MAX_ANGULAR_JIT + 1) * NUM_ABC_JIT];

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
      sum_fxyz[n * NUM_ABC_JIT + abc] =
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

      accumulate_f12<NUM_ABC_JIT>(
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

#if TENSOR_VIRIAL_JIT
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
  const int block_size = 128;
  const int grid_size = (N * 8 - 1) / block_size + 1;
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
  const int block_size = 128;
  const int grid_size = (N * 8 - 1) / block_size + 1;
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
