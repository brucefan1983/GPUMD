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
#include <fstream>
#include <iostream>
#include <sstream>

#define PI 3.141592653589793

void Model::initialize()
{

  energy.resize(number_of_energy_points); // in units of eV
  double delta_energy = 20.0 / (number_of_energy_points - 1);
  for (int n = 0; n < number_of_energy_points; ++n) {
    energy[n] = delta_energy * (n - number_of_energy_points / 2);
  }
  time_step.resize(number_of_steps_correlation); // in units of hbar/eV
  for (int n = 0; n < number_of_energy_points; ++n) {
    time_step[n] = 1.0;
  }

  std::cout << "energy= " << std::endl;
  for (int n = 0; n < number_of_energy_points; ++n) {
    std::cout << energy[n] << " ";
  }
  std::cout << std::endl;
  std::cout << "time_step= " << std::endl;
  for (int n = 0; n < number_of_steps_correlation; ++n) {
    std::cout << time_step[n] << " ";
  }
  std::cout << std::endl;
  std::cout << "done================================= " << std::endl;
  exit(1);
}
