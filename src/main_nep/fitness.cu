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
Get the fitness
------------------------------------------------------------------------------*/

#include "fitness.cuh"
#include "parameters.cuh"
#include "utilities/error.cuh"
#include <cmath>
#include <ctime>

Fitness::Fitness(Potential& potential) : potential(potential)
{
  Parameters& para = potential.get_parameters();
  num_batches = potential.prepare_data();
  if (para.prediction == 0) {
    fitness_L1.resize(para.population_size * (para.num_types + 1));
    fitness_L2.resize(para.population_size * (para.num_types + 1));
    cost_L1reg.resize(para.population_size);
    cost_L2reg.resize(para.population_size);
    gpuSetDevice(0);
    gpu_cost_L1reg.resize(para.population_size);
    gpu_cost_L2reg.resize(para.population_size);
  }
}

void Fitness::print_training_header()
{
  potential.print_training_header();
}

void Fitness::compute(
  const int generation,
  Parameters& para,
  const float* population,
  const float* gpu_population,
  const int* gpu_type_of_variable,
  float* fitness_total)
{
  if (generation == 0) {
    potential.prepare_training();
  } else {
    potential.evaluate(generation % num_batches, population);
  }
  regularize(para, gpu_population, gpu_type_of_variable, fitness_total);
  potential.accumulate_fitness(fitness_total);
}

static __global__ void gpu_find_L1_L2(
  const int number_of_variables,
  const int g_num_types,
  const int g_type,
  const int* g_type_of_variable,
  const float* g_population,
  float* gpu_cost_L1reg,
  float* gpu_cost_L2reg)
{
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  __shared__ float s_cost_L1reg[1024];
  __shared__ float s_cost_L2reg[1024];
  s_cost_L1reg[tid] = 0.0f;
  s_cost_L2reg[tid] = 0.0f;
  for (int v = tid; v < number_of_variables; v += blockDim.x) {
    const float para = g_population[bid * number_of_variables + v];
    if ((g_type_of_variable[v] == g_type) && (g_type != g_num_types) ||
        (g_type_of_variable[v] != g_type) && (g_type == g_num_types))  {
      s_cost_L1reg[tid] += abs(para);
      s_cost_L2reg[tid] += para * para;
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_cost_L1reg[tid] += s_cost_L1reg[tid + offset];
      s_cost_L2reg[tid] += s_cost_L2reg[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    gpu_cost_L1reg[bid] = s_cost_L1reg[0];
    gpu_cost_L2reg[bid] = s_cost_L2reg[0];
  }
}

void Fitness::regularize(
  Parameters& para,
  const float* gpu_population,
  const int* gpu_type_of_variable,
  float* fitness_total)
{
  const int population_size = para.population_size;
  const int number_of_variables = para.number_of_variables;
  gpuSetDevice(0); // normally use GPU-0

  for (int t = 0; t <= para.num_types; ++t) {
    float num_variables = float(para.number_of_variables) / para.num_types;
    if (t == para.num_types) {
      num_variables = para.number_of_variables;
    }

    gpu_find_L1_L2<<<population_size, 1024>>>(
      number_of_variables,
      para.num_types,
      t,
      gpu_type_of_variable,
      gpu_population,
      gpu_cost_L1reg.data(),
      gpu_cost_L2reg.data());
    GPU_CHECK_KERNEL

    gpu_cost_L1reg.copy_to_host(cost_L1reg.data());
    gpu_cost_L2reg.copy_to_host(cost_L2reg.data());

    for (int p = 0; p < population_size; ++p) {
      float cost_L1 = para.lambda_1 * cost_L1reg[p] / num_variables;
      float cost_L2 = para.lambda_2 * sqrt(cost_L2reg[p] / num_variables);
      fitness_total[p + t * population_size] = cost_L1 + cost_L2;
      fitness_L1[p + t * population_size] = cost_L1;
      fitness_L2[p + t * population_size] = cost_L2;
    }
  }
}

void Fitness::get_save_potential_label(Parameters& para, const int generation, std::string& label) {
    if (para.save_potential_format == 1) {
      time_t rawtime;
      time(&rawtime);
      struct tm* timeinfo = localtime(&rawtime);
      char buffer[200];
      strftime(buffer, sizeof(buffer), "nep_y%Y_m%m_d%d_h%H_m%M_s%S_generation", timeinfo);
      label = std::string(buffer) + std::to_string(generation + 1);
    } else {
      label = "nep_gen" + std::to_string(generation + 1);
    }
}

void Fitness::report_error(
  Parameters& para,
  const int generation,
  const float loss_total,
  const int best_index,
  float* elite,
  const std::string& checkpoint_base)
{
  if ((generation + 1) % para.output_interval == 0) {
    int index = para.num_types * para.population_size + best_index;
    TrainingReport report = {generation, generation % num_batches,
                             loss_total, fitness_L1[index], fitness_L2[index], elite, ""};
    if (!checkpoint_base.empty()) {
      report.checkpoint = checkpoint_base + ".txt";
    }
    potential.report_error(report);
  }
  if ((generation + 1) % 1000 == 0) {
    predict(elite);
  }
}

void Fitness::predict(float* elite)
{
  potential.predict(elite);
}
