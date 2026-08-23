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

/*----------------------------------------------------------------------------80
The class defining the simulation box.
------------------------------------------------------------------------------*/

#include "atom.cuh"
#include "box.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <cstring>

int Atom::number_of_type(std::string& symbol)
{
  int sum = 0;
  for (int i = 0; i < number_of_atoms; i++) {
    if (cpu_atom_symbol[i] == symbol)
      sum++;
  }
  return sum;
}

static __global__ void gpu_update_unwrapped_position(
  const int number_of_atoms,
  const Box box,
  const double* position,
  const int* image,
  double* unwrapped_position)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < number_of_atoms) {
    const int ix = image[n];
    const int iy = image[n + number_of_atoms];
    const int iz = image[n + number_of_atoms * 2];
    unwrapped_position[n] =
      position[n] + box.cpu_h[0] * ix + box.cpu_h[1] * iy + box.cpu_h[2] * iz;
    unwrapped_position[n + number_of_atoms] =
      position[n + number_of_atoms] + box.cpu_h[3] * ix + box.cpu_h[4] * iy + box.cpu_h[5] * iz;
    unwrapped_position[n + number_of_atoms * 2] =
      position[n + number_of_atoms * 2] +
      box.cpu_h[6] * ix + box.cpu_h[7] * iy + box.cpu_h[8] * iz;
  }
}

void Atom::enable_unwrapped_position()
{
  if (unwrapped_position.size() < number_of_atoms * 3) {
    unwrapped_position.resize(number_of_atoms * 3);
    unwrapped_position.copy_from_device(position_per_atom.data());
    position_image.resize(number_of_atoms * 3, 0);
  }
}

void Atom::update_unwrapped_position(const Box& box)
{
  if (unwrapped_position.size() == 0)
    return;

  gpu_update_unwrapped_position<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    box,
    position_per_atom.data(),
    position_image.data(),
    unwrapped_position.data());
  GPU_CHECK_KERNEL
}
