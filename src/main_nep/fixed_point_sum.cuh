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

A contribution that is not finite, or whose magnitude is at least 2^20 = 1.0e6, is added in
floating point instead. NaN and inf, and the forces of overlapping atoms, then reach the result
as they do without fixed point. Up to 2^11 contributions below that bound fit in the range.
*/

#pragma once

const float FIXED_POINT_SCALE = 4294967296.0f; // 2^32
const float FIXED_POINT_LIMIT = 1048576.0f;    // 2^20

static __device__ __forceinline__ void
atomic_add_fixed_point(unsigned long long* address, float value)
{
  // two's complement makes the unsigned sum equal to the signed one
  atomicAdd(address, static_cast<unsigned long long>(__float2ll_rn(value * FIXED_POINT_SCALE)));
}

// Adds value to *fixed_address in fixed point, or to *float_address when it is out of range.
static __device__ __forceinline__ void
atomic_add_in_range(float* float_address, unsigned long long* fixed_address, const float value)
{
  // false for NaN
  if (fabsf(value) < FIXED_POINT_LIMIT) {
    atomic_add_fixed_point(fixed_address, value);
  } else {
    atomicAdd(float_address, value);
  }
}

// Adds value to g_float[index], or to g_fixed[index] in fixed point when use_fixed_point is set.
template <bool use_fixed_point>
static __device__ __forceinline__ void atomic_add_float_or_fixed(
  float* g_float, unsigned long long* g_fixed, const int index, const float value)
{
  if (use_fixed_point) {
    atomic_add_in_range(&g_float[index], &g_fixed[index], value);
  } else {
    atomicAdd(&g_float[index], value);
  }
}

// Adds (fx, fy, fz) to the force on atom n, in fixed point when use_fixed_point is set.
// g_force_fixed holds the x, y and z components at n, n + N and n + 2 * N.
template <bool use_fixed_point>
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
  if (use_fixed_point) {
    atomic_add_in_range(&g_fx[n], &g_force_fixed[n], fx);
    atomic_add_in_range(&g_fy[n], &g_force_fixed[n + N], fy);
    atomic_add_in_range(&g_fz[n], &g_force_fixed[n + 2 * N], fz);
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
