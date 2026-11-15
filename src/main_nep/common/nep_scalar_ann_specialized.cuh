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
static __device__ __forceinline__ void apply_scalar_ann_jit(
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

static __global__ void ann_nep_jit(
  const int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  double* pe,
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
  pe[n1] = static_cast<double>(F);

#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    Fp_out[n1 + d * N] = Fp[d] * q_scaler[d];
  }
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
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  ann_nep_jit<<<grid_size, block_size>>>(
    N, type, descriptors, q_scaler, parameters, pe, Fp);
  return static_cast<int>(cudaGetLastError());
}
