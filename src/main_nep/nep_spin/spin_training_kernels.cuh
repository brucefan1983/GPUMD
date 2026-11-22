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
namespace
{
__global__ void add_spin_training_outputs(
  int atom_count,
  const double* spin_force,
  const double* spin_mforce,
  const double* spin_virial,
  float* force,
  float* mforce,
  float* virial)
{
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= atom_count) {
    return;
  }
  for (int component = 0; component < 3; ++component) {
    const int index = component * atom_count + atom;
    force[index] += static_cast<float>(spin_force[index]);
    mforce[index] = static_cast<float>(spin_mforce[index]);
  }
  for (int component = 0; component < 3; ++component) {
    virial[component * atom_count + atom] +=
      static_cast<float>(spin_virial[component * atom_count + atom]);
  }
  // Fixed spins with SOC allow a torque. Symmetric strain couples to (W + W^T)/2.
  constexpr int forward[3] = {3, 5, 7}; // xy, yz, zx
  constexpr int reverse[3] = {6, 8, 4}; // yx, zy, xz
  for (int shear = 0; shear < 3; ++shear) {
    const double a = spin_virial[forward[shear] * atom_count + atom];
    const double b = spin_virial[reverse[shear] * atom_count + atom];
    virial[(shear + 3) * atom_count + atom] += static_cast<float>(0.5 * (a + b));
    // The structural and ZBL contributions are symmetric after summing atoms.
    virial[(shear + 6) * atom_count + atom] = static_cast<float>(0.5 * (a - b));
  }
}

} // namespace
