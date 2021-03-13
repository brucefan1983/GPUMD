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

#include "error.cuh"
#include "fitness.cuh"
#include "snes.cuh"
#include <chrono>
#include <cmath>

SNES::SNES(char* input_dir, Fitness* fitness_function)
{
  number_of_variables = fitness_function->number_of_variables;
  population_size = 4 + int(std::floor(3.0f * std::log(number_of_variables * 1.0f)));
  eta_sigma = (3.0f + std::log(number_of_variables * 1.0f)) /
              (5.0f * sqrt(number_of_variables * 1.0f)) / 2.0f;
  fitness.resize(population_size);
  index.resize(population_size);
  population.resize(population_size * number_of_variables);
  population_copy.resize(population_size * number_of_variables);
  s.resize(population_size * number_of_variables);
  s_copy.resize(population_size * number_of_variables);
  mu.resize(number_of_variables);
  sigma.resize(number_of_variables);
  utility.resize(population_size);
  initialize_rng();
  initialize_mu_and_sigma();
  calculate_utility();
  compute(input_dir, fitness_function);
}

void SNES::initialize_rng()
{
#ifdef DEBUG
  rng = std::mt19937(12345678);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
};

void SNES::initialize_mu_and_sigma()
{
  std::uniform_real_distribution<float> r1(0, 1);
  for (int n = 0; n < number_of_variables; ++n) {
    mu[n] = r1(rng) * 2.0f - 1.0f;
    sigma[n] = 1.0f;
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

void SNES::compute(char* input_dir, Fitness* fitness_function)
{
  print_line_1();
  printf("Started training.\n");
  print_line_2();
  char file[200];
  strcpy(file, input_dir);
  strcat(file, "/ga.out");
  FILE* fid = my_fopen(file, "w");
  for (int n = 0; n < maximum_generation; ++n) {
    create_population();
    fitness_function->compute(population_size, population.data(), fitness.data());
    regularize();
    sort_population();
    output(n, fid);
    update_mu_and_sigma();
  }
  fclose(fid);
  fitness_function->predict(input_dir, population.data());
}

void SNES::create_population()
{
  std::normal_distribution<float> r1(0, 1);
  for (int p = 0; p < population_size; ++p) {
    for (int v = 0; v < number_of_variables; ++v) {
      int pv = p * number_of_variables + v;
      s[pv] = r1(rng);
      population[pv] = sigma[v] * s[pv] + mu[v];
    }
  }
}

void SNES::regularize()
{
  for (int p = 0; p < population_size; ++p) {
    for (int v = 0; v < number_of_variables; ++v) {
      int pv = p * number_of_variables + v;
      fitness[p] += 1.0e-5f * (0.5f * population[pv] * population[pv] + std::abs(population[pv]));
    }
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
    int n1 = n * number_of_variables;
    int n2 = index[n] * number_of_variables;
    for (int m = 0; m < number_of_variables; ++m) {
      s[n1 + m] = s_copy[n2 + m];
      population[n1 + m] = population_copy[n2 + m];
    }
  }
}

void SNES::output(int generation, FILE* fid)
{
  if (0 == (generation + 1) % 100) {
    fprintf(fid, "%d %g ", generation + 1, fitness[0]); // to file
    printf("%d %g\n", generation + 1, fitness[0]);      // to screen
    for (int m = 0; m < number_of_variables; ++m) {
      fprintf(fid, "%g ", population[m]); // to file
    }
    fprintf(fid, "\n"); // to file
    fflush(fid);        // to file
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
