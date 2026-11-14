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
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>
class Parameters;

struct TrainingReport
{
  int generation;
  int batch;
  float total;
  float l1;
  float l2;
  float* parameters;
  std::string checkpoint;
};

class Potential
{
public:
  virtual ~Potential() = default;
  virtual Parameters& get_parameters() = 0;
  virtual void initialize_parameters(
    std::mt19937& rng, std::vector<float>& mu, std::vector<float>& sigma) = 0;
  virtual void parameter_groups(std::vector<int>& groups) = 0;
  virtual void read_model(float* parameters) = 0;
  virtual void write_restart(
    const char* filename, const std::vector<float>& mu, const std::vector<float>& sigma) = 0;
  virtual int prepare_data() = 0;
  virtual void prepare_training() = 0;
  virtual void set_population_mean(const float* gpu_mean) {}
  // Evaluate candidate parameter vectors stored contiguously on the host. The model
  // owns intermediate losses; accumulate_fitness adds them after common regularization.
  virtual void evaluate(int batch, const float* population) = 0;
  virtual void accumulate_fitness(float* fitness_total) = 0;
  virtual void print_training_header() = 0;
  virtual void report_error(const TrainingReport& report) = 0;
  virtual void predict(float* parameters) = 0;
};

std::unique_ptr<Potential> create_potential();
