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

/*
Sums of per-pair contributions that do not depend on the order in which threads add them.

A float atomicAdd rounds after every addition, and the order of the additions varies between
runs. A sum in fixed point with 32 fractional bits is a sum of integers, which is associative.
Its resolution is 2^-32 = 2.3e-10 in the unit of the summed quantity, and its range is
+-2^31 = +-2.1e9.
*/

#pragma once

const float FIXED_POINT_SCALE = 4294967296.0f; // 2^32

static __device__ __forceinline__ void
atomic_add_fixed_point(unsigned long long* address, float value)
{
  // two's complement makes the unsigned sum equal to the signed one
  atomicAdd(address, static_cast<unsigned long long>(__float2ll_rn(value * FIXED_POINT_SCALE)));
}

// Adds (fx, fy, fz) to the force on atom n, in fixed point when g_force_fixed is set.
// g_force_fixed holds the x, y and z components at n, n + N and n + 2 * N.
static __device__ __forceinline__ void atomic_add_force(
  const int N,
  const int n,
  const float fx,
  const float fy,
  const float fz,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  unsigned long long* g_force_fixed)
{
  if (g_force_fixed) {
    atomic_add_fixed_point(&g_force_fixed[n], fx);
    atomic_add_fixed_point(&g_force_fixed[n + N], fy);
    atomic_add_fixed_point(&g_force_fixed[n + 2 * N], fz);
  } else {
    atomicAdd(&g_fx[n], fx);
    atomicAdd(&g_fy[n], fy);
    atomicAdd(&g_fz[n], fz);
  }
}

// Adds the fixed-point sums to g_float and resets them to zero for the next evaluation.
static __global__ void
add_fixed_point_sums(const int size, unsigned long long* g_fixed, float* g_float)
{
  int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < size) {
    g_float[n] +=
      static_cast<float>(static_cast<long long>(g_fixed[n]) / double(FIXED_POINT_SCALE));
    g_fixed[n] = 0;
  }
}
