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
#include "potential.cuh"
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>

class Parameters;

class Fitness
{
public:
  explicit Fitness(Potential& potential);
  Fitness(const Fitness&) = delete;
  Fitness& operator=(const Fitness&) = delete;
  void print_training_header();
  void compute(
    const int generation,
    Parameters& para,
    const float* population,
    const float* gpu_population,
    const int* gpu_type_of_variable,
    float* fitness_total);
  void report_error(
    Parameters& para,
    const int generation,
    const float loss_total,
    const int best_index,
    float* elite,
    const std::string& checkpoint_base);
  void predict(float* elite);
  void get_save_potential_label(Parameters& para, const int generation, std::string& filename);

private:
  Potential& potential;
  int num_batches = 0;
  std::vector<float> fitness_L1;
  std::vector<float> fitness_L2;
  std::vector<float> cost_L1reg;
  std::vector<float> cost_L2reg;
  GPU_Vector<float> gpu_cost_L1reg;
  GPU_Vector<float> gpu_cost_L2reg;
  void regularize(
    Parameters& para,
    const float* gpu_population,
    const int* gpu_type_of_variable,
    float* fitness_total);
};
