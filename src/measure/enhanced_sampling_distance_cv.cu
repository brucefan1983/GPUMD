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
Distance collective variable for enhanced sampling.
------------------------------------------------------------------------------*/

#include "enhanced_sampling_distance_cv.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>

static __global__ void gpu_enhanced_sampling_compute_distance(
  const int number_of_atoms,
  const int atom_i,
  const int atom_j,
  const bool use_pbc,
  const Box box,
  const double* g_position_per_atom,
  double* g_distance_data,
  double* g_value,
  int* g_invalid_distance)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    double dx = g_position_per_atom[atom_j] - g_position_per_atom[atom_i];
    double dy = g_position_per_atom[atom_j + number_of_atoms] -
                g_position_per_atom[atom_i + number_of_atoms];
    double dz = g_position_per_atom[atom_j + number_of_atoms * 2] -
                g_position_per_atom[atom_i + number_of_atoms * 2];

    if (use_pbc) {
      apply_mic(box, dx, dy, dz);
    }

    const double distance_squared = dx * dx + dy * dy + dz * dz;
    const double distance = sqrt(distance_squared);

    g_distance_data[0] = dx;
    g_distance_data[1] = dy;
    g_distance_data[2] = dz;
    g_value[0] = distance;

    if (!(distance_squared > 1.0e-20)) {
      g_distance_data[3] = 0.0;
      // Keep this flag set until the next output or post_run check. Checking it
      // on the host every step would introduce a synchronization in the force path.
      g_invalid_distance[0] = 1;
    } else {
      g_distance_data[3] = 1.0 / distance;
    }
  }
}

static __global__ void gpu_enhanced_sampling_add_distance_force_virial(
  const int number_of_atoms,
  const int atom_i,
  const int atom_j,
  const double* g_derivative,
  const double* g_distance_data,
  double* g_force_per_atom,
  double* g_virial_per_atom,
  double* g_bias_virial)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const double dx = g_distance_data[0];
    const double dy = g_distance_data[1];
    const double dz = g_distance_data[2];
    const double inverse_distance = g_distance_data[3];
    const double derivative = g_derivative[0];
    const double force_factor = derivative * inverse_distance;

    g_force_per_atom[atom_i] += force_factor * dx;
    g_force_per_atom[atom_i + number_of_atoms] += force_factor * dy;
    g_force_per_atom[atom_i + number_of_atoms * 2] += force_factor * dz;
    g_force_per_atom[atom_j] -= force_factor * dx;
    g_force_per_atom[atom_j + number_of_atoms] -= force_factor * dy;
    g_force_per_atom[atom_j + number_of_atoms * 2] -= force_factor * dz;

    const double virial_factor = -derivative * inverse_distance;
    const double virial[9] = {
      virial_factor * dx * dx,
      virial_factor * dy * dy,
      virial_factor * dz * dz,
      virial_factor * dx * dy,
      virial_factor * dx * dz,
      virial_factor * dy * dz,
      virial_factor * dy * dx,
      virial_factor * dz * dx,
      virial_factor * dz * dy};

    for (int component = 0; component < 9; ++component) {
      const double half_virial = 0.5 * virial[component];
      g_virial_per_atom[atom_i + component * number_of_atoms] += half_virial;
      g_virial_per_atom[atom_j + component * number_of_atoms] += half_virial;
      g_bias_virial[component] += virial[component];
    }
  }
}

EnhancedSamplingDistanceCV::EnhancedSamplingDistanceCV(
  const std::string& name,
  const int atom_i,
  const int atom_j,
  const bool use_pbc)
  : EnhancedSamplingCV(name),
    atom_i_(atom_i),
    atom_j_(atom_j),
    use_pbc_(use_pbc),
    number_of_atoms_(0)
{
}

void EnhancedSamplingDistanceCV::prepare(const int number_of_atoms)
{
  if (number_of_atoms <= 0) {
    PRINT_INPUT_ERROR("The number of atoms should be positive.\n");
  }
  if (atom_i_ < 0 || atom_i_ >= number_of_atoms || atom_j_ < 0 ||
      atom_j_ >= number_of_atoms) {
    PRINT_INPUT_ERROR("Atom index for a distance CV is out of range.\n");
  }
  if (atom_i_ == atom_j_) {
    PRINT_INPUT_ERROR("The two atoms in a distance CV should be different.\n");
  }

  number_of_atoms_ = number_of_atoms;
  distance_data_.resize(4);
  invalid_distance_.resize(1, 0);
}

void EnhancedSamplingDistanceCV::compute(
  const Box& box,
  const GPU_Vector<double>& position_per_atom,
  double* g_value)
{
  gpu_enhanced_sampling_compute_distance<<<1, 1>>>(
    number_of_atoms_,
    atom_i_,
    atom_j_,
    use_pbc_,
    box,
    position_per_atom.data(),
    distance_data_.data(),
    g_value,
    invalid_distance_.data());
  GPU_CHECK_KERNEL
}

void EnhancedSamplingDistanceCV::add_force_virial(
  const double* g_derivative,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  double* g_bias_virial)
{
  gpu_enhanced_sampling_add_distance_force_virial<<<1, 1>>>(
    number_of_atoms_,
    atom_i_,
    atom_j_,
    g_derivative,
    distance_data_.data(),
    force_per_atom.data(),
    virial_per_atom.data(),
    g_bias_virial);
  GPU_CHECK_KERNEL
}

void EnhancedSamplingDistanceCV::check()
{
  int invalid_distance = 0;
  invalid_distance_.copy_to_host(&invalid_distance);
  if (invalid_distance != 0) {
    fprintf(stderr, "Runtime Error:\n");
    fprintf(
      stderr,
      "    Error text: Distance CV '%s' has an invalid distance between "
      "atoms %d and %d. The atoms may be coincident or nearly coincident.\n",
      get_name().c_str(),
      atom_i_,
      atom_j_);
    exit(1);
  }
}
