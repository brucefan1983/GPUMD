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

#pragma once
#include "model/box.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

static void __global__ deform_position(
  const int N,
  const double* D,
  const double* old_x,
  const double* old_y,
  const double* old_z,
  double* new_x,
  double* new_y,
  double* new_z)
{
  const int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < N) {
    new_x[n] = D[0] * old_x[n] + D[1] * old_y[n] + D[2] * old_z[n];
    new_y[n] = D[3] * old_x[n] + D[4] * old_y[n] + D[5] * old_z[n];
    new_z[n] = D[6] * old_x[n] + D[7] * old_y[n] + D[8] * old_z[n];
  }
}

static void deform_box(
  const int N,
  const double* cpu_D,
  const double* gpu_D,
  const Box& old_box,
  Box& new_box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& new_position_per_atom)
{
  if (new_box.triclinic == 0) {
    new_box.cpu_h[0] = cpu_D[0] * old_box.cpu_h[0];
    new_box.cpu_h[1] = cpu_D[4] * old_box.cpu_h[1];
    new_box.cpu_h[2] = cpu_D[8] * old_box.cpu_h[2];
    for (int d = 0; d < 3; ++d) {
      new_box.cpu_h[d + 3] = new_box.cpu_h[d] * 0.5;
    }
  } else {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        for (int d = 0; d < 3; ++d) {
          new_box.cpu_h[r * 3 + c] += cpu_D[r * 3 + d] * old_box.cpu_h[d * 3 + c];
        }
      }
      new_box.get_inverse();
    }
  }

  deform_position<<<(N - 1) / 128 + 1, 128>>>(
    N, gpu_D, position_per_atom.data(), position_per_atom.data() + N,
    position_per_atom.data() + N * 2, new_position_per_atom.data(),
    new_position_per_atom.data() + N, new_position_per_atom.data() + N * 2);
}