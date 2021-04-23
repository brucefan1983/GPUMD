/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
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

/*----------------------------------------------------------------------------80
find the neighbor list
------------------------------------------------------------------------------*/

#include "dataset.cuh"
#include "mic.cuh"
#include "neighbor.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"

static __global__ void gpu_find_neighbor(
  int N,
  int* Na,
  int* Na_sum,
  float cutoff_square,
  const float* __restrict__ box,
  int* NN,
  int* NL,
  float* x,
  float* y,
  float* z)
{
  int N1 = Na_sum[blockIdx.x];
  int N2 = N1 + Na[blockIdx.x];
  int n1 = N1 + threadIdx.x;
  if (n1 < N2) {
    const float* __restrict__ h = box + 18 * blockIdx.x;
    float x1 = x[n1];
    float y1 = y[n1];
    float z1 = z[n1];
    int count = 0;
    for (int n2 = N1; n2 < N2; ++n2) {
      if (n2 == n1) {
        continue;
      }
      float x12 = x[n2] - x1;
      float y12 = y[n2] - y1;
      float z12 = z[n2] - z1;
      dev_apply_mic(h, x12, y12, z12);
      float distance_square = x12 * x12 + y12 * y12 + z12 * z12;
      if (distance_square < cutoff_square) {
        NL[count++ * N + n1] = n2;
      }
    }
    NN[n1] = count;
  }
}

void Neighbor::compute(Parameters& para, Dataset& dataset)
{
  NN.resize(dataset.N, Memory_Type::managed);
  NL.resize(dataset.N * dataset.max_Na, Memory_Type::managed);
  float rc2 = para.rc * para.rc;
  gpu_find_neighbor<<<dataset.Nc, dataset.max_Na>>>(
    dataset.N, dataset.Na.data(), dataset.Na_sum.data(), rc2, dataset.h.data(), NN.data(),
    NL.data(), dataset.r.data(), dataset.r.data() + dataset.N, dataset.r.data() + dataset.N * 2);
  CUDA_CHECK_KERNEL

  CHECK(cudaDeviceSynchronize());
  int min_NN = 10000, max_NN = -1;
  for (int n = 0; n < dataset.N; ++n) {
    if (NN[n] < min_NN) {
      min_NN = NN[n];
    }
    if (NN[n] > max_NN) {
      max_NN = NN[n];
    }
  }
  printf("Minimum number of neighbors for one atom = %d.\n", min_NN);
  printf("Maximum number of neighbors for one atom = %d.\n", max_NN);
}
