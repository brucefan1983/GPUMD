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

#include "step.cuh"
#include "force/force.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "utilities/gpu_macro.cuh"

static __global__ void gpu_form_force_delta(
  const int size,
  const double* force_with_modifiers,
  double* base_force)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    base_force[index] = force_with_modifiers[index] - base_force[index];
  }
}

static __global__ void gpu_add_force_delta(
  const int size,
  const double* force_delta,
  double* force)
{
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    force[index] += force_delta[index];
  }
}

void refresh_spin_endpoint(
  Force& force, Integrate& integrate, Box& box, Atom& atom, std::vector<Group>& group,
  GPU_Vector<double>& thermo, GPU_Vector<double>& spin_predictor_base_force)
{
  // The saved force is at SIB's chord midpoint (GLSD: Euler predictor). The
  // endpoint field is the starting field of the next predictor and the
  // state consumed by measurements, so refresh the complete endpoint state.
  // Preserve the already-sampled external-force increment instead of
  // invoking stateful modifiers (random force/electron stopping) twice.
  const int force_size = 3 * atom.number_of_atoms;
  gpu_form_force_delta<<<(force_size - 1) / 128 + 1, 128>>>(
    force_size,
    atom.force_per_atom.data(),
    spin_predictor_base_force.data());
  GPU_CHECK_KERNEL
  force.compute_spin(
    box, atom.position_per_atom, atom.type, group, atom.potential_per_atom,
    atom.force_per_atom, atom.virial_per_atom, atom.spin_per_atom, atom.mforce_per_atom,
    atom.position_image.size() > 0 ? atom.position_image.data() : nullptr);
  gpu_add_force_delta<<<(force_size - 1) / 128 + 1, 128>>>(
    force_size,
    spin_predictor_base_force.data(),
    atom.force_per_atom.data());
  GPU_CHECK_KERNEL
  integrate.find_thermo(
    box.get_volume(),
    group,
    atom.mass,
    atom.potential_per_atom,
    atom.velocity_per_atom,
    atom.virial_per_atom,
    thermo);
}
