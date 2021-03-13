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

#include "error.cuh"
#include "mic.cuh"
#include "neighbor.cuh"

Neighbor::~Neighbor(void)
{
  CHECK(cudaFree(NN));
  CHECK(cudaFree(NL));
}

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

void Neighbor::compute(int Nc, int N, int max_Na, int* Na, int* Na_sum, float* r, float* h)
{
  int m1 = sizeof(int) * N;
  CHECK(cudaMallocManaged((void**)&NN, m1));
  CHECK(cudaMallocManaged((void**)&NL, m1 * max_Na));
  float rc2 = cutoff * cutoff;
  gpu_find_neighbor<<<Nc, max_Na>>>(N, Na, Na_sum, rc2, h, NN, NL, r, r + N, r + N * 2);
  CUDA_CHECK_KERNEL

  CHECK(cudaDeviceSynchronize());
  for (int nc = 0; nc < Nc; ++nc) {
    printf("NN[%d]=%d,", nc, NN[Na_sum[nc]]);
    if (0 == (nc + 1) % 8)
      printf("\n");
  }
}
