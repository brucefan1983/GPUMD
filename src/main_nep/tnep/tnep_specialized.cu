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
#define HAS_CHARGE_JIT 0
#define HAS_VDW_JIT 0
#define TENSOR_VIRIAL_JIT 1
#define INITIALIZE_RADIAL_SHEAR_JIT 1
#include "main_nep/common/nep_specialized.cuh"
#include "main_nep/common/nep_scalar_ann_specialized.cuh"

static __global__ void ann_tnep_pol_jit(
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

  // Generic TNEP model_type=2:
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

#endif // NEP_SPECIALIZED_RUNTIME_BUILD
