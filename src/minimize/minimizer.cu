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
The abstract base class (ABC) for the minimizer classes.
------------------------------------------------------------------------------*/

#include "minimizer.cuh"

namespace
{

__global__ void gpu_calculate_potential_difference(
  const int size,
  const int number_of_rounds,
  const double* potential_per_atom,
  const double* potential_per_atom_temp,
  double* potential_difference)
{
  __shared__ double s_diff[1024];
  s_diff[threadIdx.x] = 0.0;

  double diff = 0.0f;

  for (int round = 0; round < number_of_rounds; ++round) {
    const int n = threadIdx.x + round * 1024;
    if (n < size) {
      diff += potential_per_atom_temp[n] - potential_per_atom[n];
    }
  }

  s_diff[threadIdx.x] = diff;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      s_diff[threadIdx.x] += s_diff[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    potential_difference[0] = s_diff[0];
  }
}

__global__ void gpu_calculate_force_square_max(
  const int size,
  const int number_of_rounds,
  const double* force_per_atom,
  double* force_square_max)
{
  const int tid = threadIdx.x;

  __shared__ double s_force_square[1024];
  s_force_square[tid] = 0.0;

  double force_square = 0.0;

  for (int round = 0; round < number_of_rounds; ++round) {
    const int n = tid + round * 1024;
    if (n < size) {
      const double f = force_per_atom[n];
      if (f * f > force_square)
        force_square = f * f;
    }
  }

  s_force_square[tid] = force_square;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_force_square[tid + offset] > s_force_square[tid]) {
        s_force_square[tid] = s_force_square[tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    force_square_max[0] = s_force_square[0];
  }
}

} // namespace

void Minimizer::calculate_potential_difference(const GPU_Vector<double>& potential_per_atom)
{
  const int size = potential_per_atom.size();
  const int number_of_rounds = (size - 1) / 1024 + 1;
  gpu_calculate_potential_difference<<<1, 1024>>>(
    size, number_of_rounds, potential_per_atom.data(), potential_per_atom_temp_.data(),
    potential_difference_.data());

  potential_difference_.copy_to_host(cpu_potential_difference_.data());
}

void Minimizer::calculate_force_square_max(const GPU_Vector<double>& force_per_atom)
{
  const int size = force_per_atom.size();
  const int number_of_rounds = (size - 1) / 1024 + 1;
  gpu_calculate_force_square_max<<<1, 1024>>>(
    size, number_of_rounds, force_per_atom.data(), force_square_max_.data());

  force_square_max_.copy_to_host(cpu_force_square_max_.data());
}
