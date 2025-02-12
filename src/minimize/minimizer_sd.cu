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
The SD (steepest decent) minimizer.
------------------------------------------------------------------------------*/

#include "force/force.cuh"
#include "minimizer_sd.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>

const double decreasing_factor = 0.2;
const double increasing_factor = 1.2;

namespace
{

__global__ void update_positions(
  const int size,
  const double position_step,
  const double* force_per_atom,
  const double* position_per_atom,
  double* position_per_atom_temp)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < size) {
    const double position_change = force_per_atom[n] * position_step;
    position_per_atom_temp[n] = position_per_atom[n] + position_change;
  }
}

__global__ void gpu_pairwise_product(const int size, double* a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = a[n] * b[n];
}
void pairwise_product(GPU_Vector<double>& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = a.size();
  gpu_pairwise_product<<<(size - 1) / 128 + 1, 128>>>(size, a.data(), b.data(), c.data());
}

} // namespace

void Minimizer_SD::compute(
  Force& force,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  force.compute(
    box, position_per_atom, type, group, potential_per_atom, force_per_atom, virial_per_atom);

  int number_of_force_evaluations = 1;
  double position_step = 0.1;

  printf("\nEnergy minimization started.\n");

  for (int step = 0; step < number_of_steps_; ++step) {
    calculate_force_square_max(force_per_atom);
    const double force_max = sqrt(cpu_force_square_max_[0]);

    if (force_max < force_tolerance_) {
      printf("    step %d: f_max = %g eV/A.\n", step, force_max);
      break;
    }

    const int size = number_of_atoms_ * 3;
    update_positions<<<(size - 1) / 128 + 1, 128>>>(
      size,
      position_step / force_max,
      force_per_atom.data(),
      position_per_atom.data(),
      position_per_atom_temp_.data());

    force.compute(
      box,
      position_per_atom_temp_,
      type,
      group,
      potential_per_atom_temp_,
      force_per_atom_temp_,
      virial_per_atom);

    ++number_of_force_evaluations;

    calculate_total_potential(potential_per_atom);

    if (cpu_total_potential_[1] > cpu_total_potential_[0]) {
      position_step *= decreasing_factor;
    } else {
      position_per_atom_temp_.copy_to_device(position_per_atom.data());
      force_per_atom_temp_.copy_to_device(force_per_atom.data());
      potential_per_atom_temp_.copy_to_device(potential_per_atom.data());
      position_step *= increasing_factor;
    }

    int base = (number_of_steps_ >= 10) ? (number_of_steps_ / 10) : 1;

    double total_potential_smaller = (cpu_total_potential_[1] > cpu_total_potential_[0])
                                       ? cpu_total_potential_[0]
                                       : cpu_total_potential_[1];

    if (step == 0) {
      printf(
        "    step 0: total_potential = %.10f eV, f_max = %.10f eV/A.\n",
        total_potential_smaller,
        force_max);
    }
    if ((step + 1) % base == 0) {
      printf(
        "    step %d: total_potential = %.10f eV, f_max = %.10f eV/A.\n",
        step + 1,
        total_potential_smaller,
        force_max);
    }
  }

  printf("Energy minimization finished.\n");
}

void Minimizer_SD::compute_label_atoms(
  Force& force,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& local_flags,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  force.compute(
    box, position_per_atom, type, group, potential_per_atom, force_per_atom, virial_per_atom);
  pairwise_product(force_per_atom, local_flags, force_per_atom);
  pairwise_product(potential_per_atom, local_flags, potential_per_atom);
    

  int number_of_force_evaluations = 1;
  double position_step = 0.1;

  //printf("\nEnergy minimization started.\n");

  for (int step = 0; step < number_of_steps_; ++step) {
    calculate_force_square_max(force_per_atom);
    const double force_max = sqrt(cpu_force_square_max_[0]);

    if (force_max < force_tolerance_) {
      //printf("    step %d: f_max = %g eV/A.\n", step, force_max);
      break;
    }

    const int size = number_of_atoms_ * 3;
    update_positions<<<(size - 1) / 128 + 1, 128>>>(
      size,
      position_step / force_max,
      force_per_atom.data(),
      position_per_atom.data(),
      position_per_atom_temp_.data());

    force.compute(
      box,
      position_per_atom_temp_,
      type,
      group,
      potential_per_atom_temp_,
      force_per_atom_temp_,
      virial_per_atom);
      pairwise_product(force_per_atom_temp_, local_flags, force_per_atom_temp_);
      pairwise_product(potential_per_atom_temp_, local_flags, potential_per_atom_temp_);

    ++number_of_force_evaluations;

    calculate_total_potential(potential_per_atom);

    if (cpu_total_potential_[1] > cpu_total_potential_[0]) {
      position_step *= decreasing_factor;
    } else {
      position_per_atom_temp_.copy_to_device(position_per_atom.data());
      force_per_atom_temp_.copy_to_device(force_per_atom.data());
      potential_per_atom_temp_.copy_to_device(potential_per_atom.data());
      position_step *= increasing_factor;
    }
/*
    int base = (number_of_steps_ >= 10) ? (number_of_steps_ / 10) : 1;

    double total_potential_smaller = (cpu_total_potential_[1] > cpu_total_potential_[0])
                                       ? cpu_total_potential_[0]
                                       : cpu_total_potential_[1];

    if (step == 0) {
      printf(
        "    step 0: total_potential = %.10f eV, f_max = %.10f eV/A.\n",
        total_potential_smaller,
        force_max);
    }
    if ((step + 1) % base == 0) {
      printf(
        "    step %d: total_potential = %.10f eV, f_max = %.10f eV/A.\n",
        step + 1,
        total_potential_smaller,
        force_max);
    }
        */
  }
  //printf("Energy minimization finished.\n");
}
