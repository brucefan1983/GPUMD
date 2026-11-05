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

/*
Spin step implementation.

Contributors:
  Chengbing Chen: 1747193328@qq.com
*/

#include "spin_ensemble.cuh"
#include "utilities/gpu_macro.cuh"

namespace
{
__global__ void form_delta(int size, const double* current, double* base)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    base[i] = current[i] - base[i];
}
__global__ void restore_modifiers(
  int n,
  const double* df,
  const double* db,
  const double* spin,
  double* force,
  double* mforce,
  double* potential)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double zeeman = 0.0;
    for (int d = 0; d < 3; ++d) {
      const int k = i + d * n;
      force[k] += df[k];
      mforce[k] += db[k];
      zeeman -= db[k] * spin[k];
    }
    potential[i] += zeeman;
  }
}
} // namespace

void Spin_Ensemble::refresh_endpoint(
  Force& force, Box& box, Atom& atom, const std::vector<Group>& group, GPU_Vector<double>& thermo)
{
  // Reuse each Action's already sampled increment. Replaying post_force would
  // resample mechanical noise. Magnetic Actions add a constant field on a group
  // or fractional region; the corrector preserves membership under cell scaling.
  const int n = atom.number_of_atoms;
  form_delta<<<(3 * n + 127) / 128, 128>>>(3 * n, atom.force_per_atom.data(), base_force_.data());
  GPU_CHECK_KERNEL
  form_delta<<<(3 * n + 127) / 128, 128>>>(3 * n, atom.mforce_per_atom.data(), base_mforce_.data());
  GPU_CHECK_KERNEL
  force.compute(box, atom, group);
  restore_modifiers<<<(n + 127) / 128, 128>>>(
    n,
    base_force_.data(),
    base_mforce_.data(),
    atom.spin_per_atom.data(),
    atom.force_per_atom.data(),
    atom.mforce_per_atom.data(),
    atom.potential_per_atom.data());
  GPU_CHECK_KERNEL
  find_thermo(
    box.get_volume(),
    group,
    atom.mass,
    atom.potential_per_atom,
    atom.velocity_per_atom,
    atom.virial_per_atom,
    thermo);
}
