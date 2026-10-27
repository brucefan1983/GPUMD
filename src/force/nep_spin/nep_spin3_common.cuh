/*
    This file is part of GPUMD.
    SPDX-License-Identifier: LGPL-3.0-or-later
*/

#pragma once

// Small device helpers shared by the Spin3 runtime and training paths.

__device__ __forceinline__ void minimum_image_delta(
  const SimulationBox& box,
  const double x,
  const double y,
  const double z,
  float& dx,
  float& dy,
  float& dz)
{
  dx = static_cast<float>(x);
  dy = static_cast<float>(y);
  dz = static_cast<float>(z);
  apply_mic(box, dx, dy, dz);
}

__device__ __forceinline__ void compute_spin_edge_geometry_f32(
  const int atom,
  const int neighbor,
  const int atom_stride,
  const SimulationBox box,
  const double* __restrict__ positions,
  float* rhat,
  float& distance)
{
  float dx;
  float dy;
  float dz;
  minimum_image_delta(
    box,
    positions[neighbor] - positions[atom],
    positions[atom_stride + neighbor] - positions[atom_stride + atom],
    positions[2 * atom_stride + neighbor] - positions[2 * atom_stride + atom],
    dx,
    dy,
    dz);
  distance = sqrtf(dx * dx + dy * dy + dz * dz);
  const float inverse_distance = distance > 0.0f ? 1.0f / distance : 0.0f;
  rhat[0] = dx * inverse_distance;
  rhat[1] = dy * inverse_distance;
  rhat[2] = dz * inverse_distance;
}

__device__ __forceinline__ int virial_internal_component(const int row_major)
{
  return row_major == 0   ? 0
         : row_major == 1 ? 3
         : row_major == 2 ? 4
         : row_major == 3 ? 6
         : row_major == 4 ? 1
         : row_major == 5 ? 5
         : row_major == 6 ? 7
         : row_major == 7 ? 8
                          : 2;
}
