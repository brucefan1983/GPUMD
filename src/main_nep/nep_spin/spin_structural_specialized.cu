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

/*----------------------------------------------------------------------------80
Major contributors:
  Chengbing Chen: Initial implementation.
------------------------------------------------------------------------------*/

#ifdef NEP_SPECIALIZED_RUNTIME_BUILD
#define HAS_CHARGE_JIT 0
#define HAS_VDW_JIT 0
#define TENSOR_VIRIAL_JIT 0
#define INITIALIZE_RADIAL_SHEAR_JIT 1
#include "main_nep/common/nep_specialized.cuh"
#include "utilities/nep_spin/spin_math.cuh"

static_assert(NUM_HIDDEN_LAYERS_JIT == 1, "NEP-spin supports one hidden layer");

static __global__ void spin_ann_jit(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  double* pe,
  float* Fp_out)
{
  const int atom = threadIdx.x + blockIdx.x * blockDim.x;
  if (atom >= N)
    return;
  const int t = NUM_TYPES_JIT == 1 ? 0 : type[atom];
  float q[ANN_DIM_JIT];
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d)
    q[d] = descriptors[atom + d * N] * q_scaler[d];
  const float* wb = parameters + t * ONE_ANN_NO_BIAS_JIT;
  const float* bias = parameters + NUM_TYPES_JIT * ONE_ANN_NO_BIAS_JIT;
  double energy = 0, derivative[ANN_DIM_JIT] = {};
  nep_spin::apply_ann_one_layer(
    ANN_DIM_JIT,
    NUM_NEURONS1_JIT,
    wb,
    wb + NUM_NEURONS1_JIT * ANN_DIM_JIT,
    wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1),
    bias,
    q,
    energy,
    derivative);
  pe[atom] = energy;
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d)
    Fp_out[atom + d * N] = derivative[d] * q_scaler[d];
}

extern "C" int nep_train_launch_ann_nep(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  double* pe,
  float* Fp)
{
  spin_ann_jit<<<(N + 31) / 32, 32>>>(N, type, descriptors, q_scaler, parameters, pe, Fp);
  return static_cast<int>(cudaGetLastError());
}
// The shared loader requires this symbol; spin has no temperature ANN input.
extern "C" int nep_train_launch_ann_temperature(
  int, const int*, const float*, float*, const float*, const float*, double*, float*)
{
  return static_cast<int>(cudaErrorNotSupported);
}
#endif
