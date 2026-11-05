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

#pragma once
#include <random>
#include <vector>

class NEP_Parameters;

enum class NEP_Initialization { RANDOM, RESTART, FINE_TUNE };

NEP_Initialization initialize_nep_mu_and_sigma(
  const NEP_Parameters& para,
  std::mt19937& rng,
  std::vector<float>& mu,
  std::vector<float>& sigma,
  int global_parameters);

void find_nep_type_of_variable(
  const NEP_Parameters& para, std::vector<int>& type_of_variable, int global_parameters, int ann_count);
