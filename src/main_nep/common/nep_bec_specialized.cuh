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
static __global__ void bec_radial_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  const float* charge_derivative,
  float* bec)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c_jit(parameters);
  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];

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

      const float tmp12 =
        charge_derivative[n1 + n * N] * gnp12 * d12inv;
      f12[0] += tmp12 * r12[0];
      f12[1] += tmp12 * r12[1];
      f12[2] += tmp12 * r12[2];
    }

    const float bec_xx = 0.5f * r12[0] * f12[0];
    const float bec_xy = 0.5f * r12[0] * f12[1];
    const float bec_xz = 0.5f * r12[0] * f12[2];
    const float bec_yx = 0.5f * r12[1] * f12[0];
    const float bec_yy = 0.5f * r12[1] * f12[1];
    const float bec_yz = 0.5f * r12[1] * f12[2];
    const float bec_zx = 0.5f * r12[2] * f12[0];
    const float bec_zy = 0.5f * r12[2] * f12[1];
    const float bec_zz = 0.5f * r12[2] * f12[2];

    atomicAdd(&bec[n1], bec_xx);
    atomicAdd(&bec[n1 + N], bec_xy);
    atomicAdd(&bec[n1 + 2 * N], bec_xz);
    atomicAdd(&bec[n1 + 3 * N], bec_yx);
    atomicAdd(&bec[n1 + 4 * N], bec_yy);
    atomicAdd(&bec[n1 + 5 * N], bec_yz);
    atomicAdd(&bec[n1 + 6 * N], bec_zx);
    atomicAdd(&bec[n1 + 7 * N], bec_zy);
    atomicAdd(&bec[n1 + 8 * N], bec_zz);

    atomicAdd(&bec[n2], -bec_xx);
    atomicAdd(&bec[n2 + N], -bec_xy);
    atomicAdd(&bec[n2 + 2 * N], -bec_xz);
    atomicAdd(&bec[n2 + 3 * N], -bec_yx);
    atomicAdd(&bec[n2 + 4 * N], -bec_yy);
    atomicAdd(&bec[n2 + 5 * N], -bec_yz);
    atomicAdd(&bec[n2 + 6 * N], -bec_zx);
    atomicAdd(&bec[n2 + 7 * N], -bec_zy);
    atomicAdd(&bec[n2 + 8 * N], -bec_zz);
  }
}

static __global__ void bec_angular_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  const float* charge_derivative,
  const float* sum_fxyz_in,
  float* bec)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c_jit(parameters);
  float Fp[DIM_ANGULAR_JIT] = {0.0f};
  float sum_fxyz[(N_MAX_ANGULAR_JIT + 1) * NUM_ABC_JIT];

#pragma unroll
  for (int d = 0; d < DIM_ANGULAR_JIT; ++d) {
    Fp[d] = charge_derivative[(DIM_RADIAL_JIT + d) * N + n1];
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

    const float bec_xx = 0.5f * r12[0] * f12[0];
    const float bec_xy = 0.5f * r12[0] * f12[1];
    const float bec_xz = 0.5f * r12[0] * f12[2];
    const float bec_yx = 0.5f * r12[1] * f12[0];
    const float bec_yy = 0.5f * r12[1] * f12[1];
    const float bec_yz = 0.5f * r12[1] * f12[2];
    const float bec_zx = 0.5f * r12[2] * f12[0];
    const float bec_zy = 0.5f * r12[2] * f12[1];
    const float bec_zz = 0.5f * r12[2] * f12[2];

    atomicAdd(&bec[n1], bec_xx);
    atomicAdd(&bec[n1 + N], bec_xy);
    atomicAdd(&bec[n1 + 2 * N], bec_xz);
    atomicAdd(&bec[n1 + 3 * N], bec_yx);
    atomicAdd(&bec[n1 + 4 * N], bec_yy);
    atomicAdd(&bec[n1 + 5 * N], bec_yz);
    atomicAdd(&bec[n1 + 6 * N], bec_zx);
    atomicAdd(&bec[n1 + 7 * N], bec_zy);
    atomicAdd(&bec[n1 + 8 * N], bec_zz);

    atomicAdd(&bec[n2], -bec_xx);
    atomicAdd(&bec[n2 + N], -bec_xy);
    atomicAdd(&bec[n2 + 2 * N], -bec_xz);
    atomicAdd(&bec[n2 + 3 * N], -bec_yx);
    atomicAdd(&bec[n2 + 4 * N], -bec_yy);
    atomicAdd(&bec[n2 + 5 * N], -bec_yz);
    atomicAdd(&bec[n2 + 6 * N], -bec_zx);
    atomicAdd(&bec[n2 + 7 * N], -bec_zy);
    atomicAdd(&bec[n2 + 8 * N], -bec_zz);
  }
}

extern "C" int nep_train_launch_bec_radial(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  const float* charge_derivative,
  float* bec)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  bec_radial_jit<<<grid_size, block_size>>>(
    N, NN_sum, NN, NL, type, x12, y12, z12,
    parameters, charge_derivative, bec);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_bec_angular(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  const float* charge_derivative,
  const float* sum_fxyz,
  float* bec)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  bec_angular_jit<<<grid_size, block_size>>>(
    N, NN_sum, NN, NL, type, x12, y12, z12,
    parameters, charge_derivative, sum_fxyz, bec);
  return static_cast<int>(cudaGetLastError());
}
