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

/*----------------------------------------------------------------------------80
Use the separable natural evolution strategy (SNES) to fit potential parameters.

Reference:

T. Schaul, T. Glasmachers, and J. Schmidhuber,
High Dimensions and Heavy Tails for Natural Evolution Strategies,
https://doi.org/10.1145/2001576.2001692
------------------------------------------------------------------------------*/

#include "fitness.cuh"
#include "parameters.cuh"
#include "snes.cuh"
#include "utilities/error.cuh"
#include <chrono>
#include <cmath>

SNES::SNES(Parameters& para, Fitness* fitness_function)
{
  maximum_generation = para.maximum_generation;
  number_of_variables = para.number_of_variables;
  population_size = para.population_size;
  eta_sigma = (3.0f + std::log(number_of_variables * 1.0f)) /
              (5.0f * sqrt(number_of_variables * 1.0f)) / 2.0f;
  fitness.resize(population_size * 6);
  fitness_copy.resize(population_size * 6);
  index.resize(population_size);
  population.resize(population_size * number_of_variables);
  population_copy.resize(population_size * number_of_variables);
  s.resize(population_size * number_of_variables);
  s_copy.resize(population_size * number_of_variables);
  mu.resize(number_of_variables);
  sigma.resize(number_of_variables);
  utility.resize(population_size);
  initialize_rng();
  initialize_mu_and_sigma(para);
  calculate_utility();
  compute(para, fitness_function);
}

void SNES::initialize_rng()
{
#ifdef DEBUG
  rng = std::mt19937(12345678);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
};

void SNES::initialize_mu_and_sigma(Parameters& para)
{
  FILE* fid_restart = fopen("nep.restart", "r");
  if (fid_restart == NULL) {
    std::uniform_real_distribution<float> r1(0, 1);
    for (int n = 0; n < number_of_variables; ++n) {
      mu[n] = r1(rng) - 0.5f;
      sigma[n] = eta_sigma;
    }
  } else {
    for (int n = 0; n < number_of_variables; ++n) {
      int count = fscanf(fid_restart, "%f%f", &mu[n], &sigma[n]);
      PRINT_SCANF_ERROR(count, 2, "Reading error for nep.restart.");
    }
    fclose(fid_restart);
  }
}

void SNES::calculate_utility()
{
  float utility_sum = 0.0f;
  for (int n = 0; n < population_size; ++n) {
    utility[n] = std::max(0.0f, std::log(population_size * 0.5f + 1.0f) - std::log(n + 1.0f));
    utility_sum += utility[n];
  }
  for (int n = 0; n < population_size; ++n) {
    utility[n] = utility[n] / utility_sum - 1.0f / population_size;
  }
}

void SNES::compute(Parameters& para, Fitness* fitness_function)
{

  print_line_1();
  printf("Started training.\n");
  print_line_2();

  printf(
    "%-8s%-11s%-11s%-11s%-13s%-13s%-13s%-13s%-13s%-13s\n", "Step", "Total-Loss", "L1Reg-Loss",
    "L2Reg-Loss", "RMSE-E-Train", "RMSE-F-Train", "RMSE-V-Train", "RMSE-E-Test", "RMSE-F-Test",
    "RMSE-V-Test");

  for (int n = 0; n < maximum_generation; ++n) {
    create_population(para);
    fitness_function->compute(n, para, population.data(), fitness.data() + 3 * population_size);
    regularize(para);
    sort_population();
    fitness_function->report_error(
      para, n, fitness[0 + 0 * population_size], fitness[0 + 1 * population_size],
      fitness[0 + 2 * population_size], population.data());
    update_mu_and_sigma();
    if (0 == (n + 1) % 100) {
      output_mu_and_sigma(para);
    }
  }
}

void SNES::create_population(Parameters& para)
{
  std::normal_distribution<float> r1(0, 1);
  for (int p = 0; p < population_size; ++p) {
    for (int v = 0; v < number_of_variables; ++v) {
      int pv = p * number_of_variables + v;
      s[pv] = r1(rng);
      population[pv] = sigma[v] * s[pv] + mu[v];
      // avoid zero
      if (v >= para.number_of_variables_ann) {
        if (population[pv] > 0) {
          population[pv] += 0.1f;
        } else {
          population[pv] -= 0.1f;
        }
      }
    }
  }
}

void SNES::regularize(Parameters& para)
{
  for (int p = 0; p < population_size; ++p) {
    float cost_L1 = 0.0f, cost_L2 = 0.0f;
    for (int v = 0; v < number_of_variables; ++v) {
      int pv = p * number_of_variables + v;
      cost_L1 += std::abs(population[pv]);
      cost_L2 += population[pv] * population[pv];
    }
    cost_L1 *= para.lambda_1 / number_of_variables;
    cost_L2 = para.lambda_2 * sqrt(cost_L2 / number_of_variables);
    fitness[p] = cost_L1 + cost_L2 + fitness[p + 3 * population_size] +
                 fitness[p + 4 * population_size] + fitness[p + 5 * population_size];
    fitness[p + 1 * population_size] = cost_L1;
    fitness[p + 2 * population_size] = cost_L2;
  }
}

static void insertion_sort(float array[], int index[], int n)
{
  for (int i = 1; i < n; i++) {
    float key = array[i];
    int j = i - 1;
    while (j >= 0 && array[j] > key) {
      array[j + 1] = array[j];
      index[j + 1] = index[j];
      --j;
    }
    array[j + 1] = key;
    index[j + 1] = i;
  }
}

void SNES::sort_population()
{
  for (int n = 0; n < population_size; ++n) {
    index[n] = n;
  }
  insertion_sort(fitness.data(), index.data(), population_size);
  for (int n = 0; n < population_size * number_of_variables; ++n) {
    s_copy[n] = s[n];
    population_copy[n] = population[n];
  }
  for (int n = 0; n < population_size; ++n) {
    for (int k = 1; k < 6; ++k) {
      fitness_copy[n + k * population_size] = fitness[n + k * population_size];
    }
  }
  for (int n = 0; n < population_size; ++n) {
    int n1 = n * number_of_variables;
    int n2 = index[n] * number_of_variables;
    for (int m = 0; m < number_of_variables; ++m) {
      s[n1 + m] = s_copy[n2 + m];
      population[n1 + m] = population_copy[n2 + m];
    }
    for (int k = 1; k < 6; ++k) {
      fitness[n + k * population_size] = fitness_copy[index[n] + k * population_size];
    }
  }
}

void SNES::update_mu_and_sigma()
{
  for (int v = 0; v < number_of_variables; ++v) {
    float gradient_mu = 0.0f, gradient_sigma = 0.0f;
    for (int p = 0; p < population_size; ++p) {
      int pv = p * number_of_variables + v;
      gradient_mu += s[pv] * utility[p];
      gradient_sigma += (s[pv] * s[pv] - 1.0f) * utility[p];
    }
    mu[v] += sigma[v] * gradient_mu;
    sigma[v] *= std::exp(eta_sigma * gradient_sigma);
  }
}

void SNES::output_mu_and_sigma(Parameters& para)
{
  FILE* fid_restart = my_fopen("nep.restart", "w");
  for (int n = 0; n < number_of_variables; ++n) {
    fprintf(fid_restart, "%15.7e %15.7e\n", mu[n], sigma[n]);
  }
  fclose(fid_restart);
}
