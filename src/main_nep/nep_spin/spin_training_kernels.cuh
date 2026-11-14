/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD and is distributed under LGPLv3 or later.
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
  constexpr int upper[3] = {3, 5, 4}; // xy, yz, xz
  constexpr int lower[3] = {6, 8, 7}; // yx, zy, zx
  for (int shear = 0; shear < 3; ++shear) {
    const double symmetric = 0.5 * (spin_virial[upper[shear] * atom_count + atom] +
                                    spin_virial[lower[shear] * atom_count + atom]);
    virial[(shear + 3) * atom_count + atom] += static_cast<float>(symmetric);
  }
}

} // namespace
