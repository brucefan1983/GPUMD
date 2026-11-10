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

/*----------------------------------------------------------------------------80
Canonical velocity sampling for PRD. REMD initializes velocities with Velocity
(main_gpumd/velocity.cu), as gpumd does.
------------------------------------------------------------------------------*/

#include "replica_velocity.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include <cmath>
#include <random>
#include <vector>

static void zero_linear_momentum(const int N, const double* m, double* vx, double* vy, double* vz)
{
  double center_of_mass_velocity[3] = {0.0, 0.0, 0.0};
  double total_mass = 0.0;
  for (int n = 0; n < N; ++n) {
    total_mass += m[n];
    center_of_mass_velocity[0] += m[n] * vx[n];
    center_of_mass_velocity[1] += m[n] * vy[n];
    center_of_mass_velocity[2] += m[n] * vz[n];
  }
  center_of_mass_velocity[0] /= total_mass;
  center_of_mass_velocity[1] /= total_mass;
  center_of_mass_velocity[2] /= total_mass;
  for (int n = 0; n < N; ++n) {
    vx[n] -= center_of_mass_velocity[0];
    vy[n] -= center_of_mass_velocity[1];
    vz[n] -= center_of_mass_velocity[2];
  }
}

void Replica_Velocity::generate_canonical(
  const double temperature,
  const std::vector<double>& mass,
  std::vector<double>& velocity,
  const unsigned int seed)
{
  if (mass.size() < 2 || !std::isfinite(temperature) || temperature <= 0.0)
    PRINT_INPUT_ERROR("Canonical velocity sampling requires at least two atoms and a positive finite temperature.");
  for (const double m : mass) {
    if (!std::isfinite(m) || m <= 0.0)
      PRINT_INPUT_ERROR("Canonical velocity sampling requires positive finite masses.");
  }

  const int N = mass.size();
  velocity.resize(3 * N);
  // Keep the distribution local so no cached Gaussian survives a sampling call.
  std::mt19937 rng(seed);
  std::normal_distribution<double> gaussian(0.0, 1.0);
  for (int n = 0; n < N; ++n) {
    const double sigma = std::sqrt(K_B * temperature / mass[n]);
    for (int d = 0; d < 3; ++d)
      velocity[n + d * N] = sigma * gaussian(rng);
  }
  zero_linear_momentum(
    N, mass.data(), velocity.data(), velocity.data() + N, velocity.data() + 2 * N);
  // Do not remove angular momentum or rescale to a fixed kinetic energy:
  // the constrained canonical distribution must retain its kinetic fluctuations.
}
