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
#include "utilities/gpu_vector.cuh"
#include "utilities/gpu_macro.cuh"
#ifdef USE_HIP
  #include <hiprand/hiprand_kernel.h>
#else
  #include <curand_kernel.h>
#endif
#include <random>
#include <vector>
class Fitness;
class Potential;
class Parameters;

class SNES
{
public:
  SNES(Parameters& para, Potential& potential);
  void run(Parameters& para, Fitness* fitness_function);

protected:
  Potential& potential;
  std::mt19937 rng;
  int maximum_generation = 10000;
  int number_of_variables = 10;
  int population_size = 20;
  float eta_sigma = 0.1f;

  std::vector<int> index;
  std::vector<float> fitness_total;
  std::vector<float> population;
  std::vector<float> mu;
  std::vector<float> sigma;
  std::vector<float> utility;
  std::vector<int> type_of_variable;

  GPU_Vector<gpurandState> curand_states;
  GPU_Vector<int> gpu_type_of_variable;
  GPU_Vector<int> gpu_index;
  GPU_Vector<float> gpu_utility;
  GPU_Vector<float> gpu_population;
  GPU_Vector<float> gpu_s;
  GPU_Vector<float> gpu_sigma;
  GPU_Vector<float> gpu_mu;

  void initialize_rng(const int seed);
  void calculate_utility();
  void create_population();
  void sort_population(Parameters& para);
  void update_mu_and_sigma();
  void output_mu_and_sigma(Parameters& para, const char* filename);
};
