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

#include "model.cuh"
#include "vector.cuh"
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#define PI 3.141592653589793

void Model::initialize()
{
#ifdef DEBUG
  // use the same seed for different runs
  generator = std::mt19937(12345678);
#else
  // use different seeds for different runs
  generator = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif

  energy.resize(number_of_energy_points);
  time_step.resize(number_of_steps_correlation);
}

// This function is called by the lsqt function in the lsqt.cu file
// It initializes a random vector
void Model::initialize_state(Vector& random_state)
{
  std::uniform_real_distribution<real> phase(0, 2 * PI);
  real* random_state_real = new real[number_of_atoms];
  real* random_state_imag = new real[number_of_atoms];

  for (int n = 0; n < number_of_atoms; ++n) {
    real random_phase = phase(generator);
    random_state_real[n] = cos(random_phase);
    random_state_imag[n] = sin(random_phase);
  }

  random_state.copy_from_host(random_state_real, random_state_imag);
  delete[] random_state_real;
  delete[] random_state_imag;
}
