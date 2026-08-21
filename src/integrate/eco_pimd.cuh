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

#pragma once
#include <vector>

struct Eco_PIMD_Result
{
  // omega_k / omega_P for k = 0, ..., P - 1.
  std::vector<double> mode_factors;

  // Independent dimensionless frequencies y_k = beta * hbar * omega_k.
  // Keeping these allows a temperature-dependent calculation to use the
  // previous solution as the next Newton starting point.
  std::vector<double> independent_frequencies;

  double rmse_trotter = 0.0;
  double rmse_matsubara = 0.0;
  double rmse_eco = 0.0;
  int number_of_iterations = 0;
};

Eco_PIMD_Result find_eco_pimd_frequencies(
  int number_of_beads,
  double x_max,
  const std::vector<double>& initial_frequencies = std::vector<double>());
