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

#include "nhc.cuh"
#include <cmath>

double nhc(
  int chain_length,
  double* position,
  double* velocity,
  double* mass,
  double twice_kinetic_energy,
  double thermal_energy,
  double degrees_of_freedom,
  double half_time_step)
{
  const int number_of_yoshida_suzuki_stages = 7;
  const int number_of_respa_steps = 4;
  const double weight[7] = {
    0.784513610477560,
    0.235573213359357,
    -1.17767998417887,
    1.31518632068391,
    -1.17767998417887,
    0.235573213359357,
    0.784513610477560};

  double factor = 1.0;
  for (int stage = 0; stage < number_of_yoshida_suzuki_stages; ++stage) {
    const double dt2 =
      half_time_step * weight[stage] / number_of_respa_steps;
    const double dt4 = dt2 * 0.5;
    const double dt8 = dt4 * 0.5;
    for (int respa = 0; respa < number_of_respa_steps; ++respa) {
      double acceleration =
        velocity[chain_length - 2] * velocity[chain_length - 2] /
          mass[chain_length - 2] -
        thermal_energy;
      velocity[chain_length - 1] += dt4 * acceleration;

      for (int index = chain_length - 2; index >= 0; --index) {
        const double scale =
          std::exp(-dt8 * velocity[index + 1] / mass[index + 1]);
        if (index == 0) {
          acceleration =
            twice_kinetic_energy - degrees_of_freedom * thermal_energy;
        } else {
          acceleration =
            velocity[index - 1] * velocity[index - 1] / mass[index - 1] -
            thermal_energy;
        }
        velocity[index] =
          scale * (scale * velocity[index] + dt4 * acceleration);
      }

      for (int index = chain_length - 1; index >= 0; --index) {
        position[index] += dt2 * velocity[index] / mass[index];
      }

      const double factor_local =
        std::exp(-dt2 * velocity[0] / mass[0]);
      twice_kinetic_energy *= factor_local * factor_local;
      factor *= factor_local;

      for (int index = 0; index < chain_length - 1; ++index) {
        const double scale =
          std::exp(-dt8 * velocity[index + 1] / mass[index + 1]);
        if (index == 0) {
          acceleration =
            twice_kinetic_energy - degrees_of_freedom * thermal_energy;
        } else {
          acceleration =
            velocity[index - 1] * velocity[index - 1] / mass[index - 1] -
            thermal_energy;
        }
        velocity[index] =
          scale * (scale * velocity[index] + dt4 * acceleration);
      }

      acceleration =
        velocity[chain_length - 2] * velocity[chain_length - 2] /
          mass[chain_length - 2] -
        thermal_energy;
      velocity[chain_length - 1] += dt4 * acceleration;
    }
  }
  return factor;
}
