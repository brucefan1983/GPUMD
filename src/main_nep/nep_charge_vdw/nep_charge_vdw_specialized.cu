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

#ifdef NEP_SPECIALIZED_RUNTIME_BUILD
#define HAS_CHARGE_JIT 1
#define HAS_VDW_JIT 1
#define TENSOR_VIRIAL_JIT 0
#define INITIALIZE_RADIAL_SHEAR_JIT 0
#include "main_nep/common/nep_specialized.cuh"
#include "main_nep/common/nep_bec_specialized.cuh"

static __global__ void ann_charge_vdw_jit(
  const int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  double* pe,
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

  pe[n1] = static_cast<double>(F);
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

extern "C" int nep_train_launch_ann_charge_vdw(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  double* pe,
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

#endif // NEP_SPECIALIZED_RUNTIME_BUILD
