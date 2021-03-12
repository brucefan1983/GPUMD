/*
    Copyright 2019 Zheyong Fan
    This file is part of GPUGA.
    GPUGA is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUGA is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUGA.  If not, see <http://www.gnu.org/licenses/>.
*/

/*----------------------------------------------------------------------------80
Use the genetic algorithm to fit potential parameters.
------------------------------------------------------------------------------*/

#include "error.cuh"
#include "fitness.cuh"
#include "ga.cuh"
#include <chrono>
#include <cmath>
#include <errno.h>

GA::GA(char* input_dir, Fitness* fitness_function)
{
  // parameters
  number_of_variables = fitness_function->number_of_variables;
  parameters_min.resize(number_of_variables);
  parameters_max.resize(number_of_variables);
  for (int n = 0; n < number_of_variables; ++n) {
    parameters_min[n] = fitness_function->parameters_min[n];
    parameters_max[n] = fitness_function->parameters_max[n];
  }

  // memory
  fitness.resize(population_size);
  index.resize(population_size);
  cumulative_probabilities.resize(parent_number);
  population.resize(population_size * number_of_variables);
  population_copy.resize(population_size * number_of_variables);
  // constants used for slecting parents
  float numerator = 0.0;
  float denominator = (1.0 + parent_number) * parent_number / 2.0;
  for (int n = 0; n < parent_number; ++n) {
    numerator += parent_number - n;
    cumulative_probabilities[n] = numerator / denominator;
  }
#ifdef DEBUG
  rng = std::mt19937(12345678);
#else
  rng = std::mt19937(std::chrono::system_clock::now().time_since_epoch().count());
#endif
  // initial population
  std::uniform_real_distribution<float> r1(0, 1);
  for (int n = 0; n < population_size * number_of_variables; ++n) {
    population[n] = r1(rng);
  }

  // for SNES
  population_size_auto = 4 + int(floor(3.0f * std::log(number_of_variables * 1.0f)));
  eta_sigma = (3.0f + std::log(number_of_variables * 1.0f)) /
              (5.0f * sqrt(number_of_variables * 1.0f)) / 2.0f;
  printf("number_of_variables = %d\n", number_of_variables);
  printf("population_size_auto = %d\n", population_size_auto);
  printf("eta_sigma = %f\n", eta_sigma);
  mu.resize(number_of_variables);
  sigma.resize(number_of_variables);
  utility.resize(population_size_auto);
  for (int n = 0; n < number_of_variables; ++n) {
    // mu[n] = r1(rng) * 2.0f - 1.0f;
    sigma[n] = 1.0f;
  }
  float utility_sum = 0.0f;
  for (int n = 0; n < population_size_auto; ++n) {
    utility[n] = std::max(0.0f, std::log(population_size_auto * 0.5f + 1.0f) - std::log(n + 1.0f));
    utility_sum += utility[n];
  }
  for (int n = 0; n < population_size_auto; ++n) {
    utility[n] = utility[n] / utility_sum - 1.0f / population_size_auto;
  }
  s.resize(population_size_auto * number_of_variables);

  // run the GA
  compute(input_dir, fitness_function);
}

void GA::compute(char* input_dir, Fitness* fitness_function)
{
  print_line_1();
  printf("Started GA evolution.\n");
  print_line_2();
  char file[200];
  strcpy(file, input_dir);
  strcat(file, "/ga.out");
  FILE* fid = my_fopen(file, "w");
  for (int n = 0; n < maximum_generation; ++n) {
    fitness_function->compute(population_size, population.data(), fitness.data());
    sort_population(n);
    output(n, fid);
    crossover();
    mutation(n);
  }
  fclose(fid);

  fitness_function->predict(input_dir, population.data());
}

void GA::compute_snes(char* input_dir, Fitness* fitness_function)
{
  print_line_1();
  printf("Started training.\n");
  print_line_2();
  char file[200];
  strcpy(file, input_dir);
  strcat(file, "/ga.out");
  FILE* fid = my_fopen(file, "w");
  for (int n = 0; n < maximum_generation; ++n) {
    std::normal_distribution<float> r1(0, 1);
    for (int p = 0; p < population_size_auto; ++p) {
      for (int v = 0; v < number_of_variables; ++v) {
        int pv = p * number_of_variables + v;
        s[pv] = r1(rng);
        population[pv] = sigma[v] * s[pv] + mu[v];
      }
    }
    fitness_function->compute(population_size_auto, population.data(), fitness.data());
    sort_population(n);
    output(n, fid);
    for (int v = 0; v < number_of_variables; ++v) {
      float gradient_mu = 0.0f, gradient_sigma = 0.0f;
      for (int p = 0; p < population_size_auto; ++p) {
        int pv = p * number_of_variables + v;
        gradient_mu += s[pv] * utility[p];
        gradient_sigma += (s[pv] * s[pv] - 1.0f) * utility[p];
      }
      mu[v] += sigma[v] * gradient_mu;
      sigma[v] *= std::exp(eta_sigma * gradient_sigma);
    }
  }
  fclose(fid);
  fitness_function->predict(input_dir, population.data());
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

void GA::sort_population(int generation)
{
  for (int n = 0; n < population_size; ++n) {
    index[n] = n;
  }
  insertion_sort(fitness.data(), index.data(), population_size);
  for (int n = 0; n < population_size * number_of_variables; ++n) {
    population_copy[n] = population[n];
  }
  for (int n = 0; n < population_size; ++n) {
    int n1 = n * number_of_variables;
    int n2 = index[n] * number_of_variables;
    for (int m = 0; m < number_of_variables; ++m) {
      population[n1 + m] = population_copy[n2 + m];
    }
  }
}

void GA::output(int generation, FILE* fid)
{
  // to file
  fprintf(fid, "%d %g ", generation, fitness[0]);
  for (int m = 0; m < number_of_variables; ++m) {
    float a = parameters_min[m];
    float b = parameters_max[m] - a;
    fprintf(fid, "%g ", a + b * population[m]);
  }
  fprintf(fid, "\n");
  fflush(fid);
  // to screen
  if (0 == (generation + 1) % 10) {
    printf("%d %g ", generation + 1, fitness[0]);
    for (int m = 0; m < number_of_variables; ++m) {
      float a = parameters_min[m];
      float b = parameters_max[m] - a;
      printf("%g ", a + b * population[m]);
    }
    printf("\n");
  }
}

void GA::output_snes(int generation, FILE* fid)
{
  // to file
  fprintf(fid, "%d %g ", generation, fitness[0]);
  for (int m = 0; m < number_of_variables; ++m) {
    fprintf(fid, "%g ", population[m]);
  }
  fprintf(fid, "\n");
  fflush(fid);
  // to screen
  if (0 == (generation + 1) % 10) {
    printf("%d %g ", generation + 1, fitness[0]);
    for (int m = 0; m < number_of_variables; ++m) {
      printf("%g ", population[m]);
    }
    printf("\n");
  }
}

void GA::crossover(void)
{
  for (int m = 0; m < child_number; m += 2) {
    int parent_1 = get_a_parent();
    int parent_2 = get_a_parent();
    while (parent_2 == parent_1) {
      parent_2 = get_a_parent();
    }
    std::uniform_int_distribution<int> r1(1, number_of_variables - 1);
    int crossover_point = r1(rng);
    int child_1 = parent_number + m;
    int child_2 = child_1 + 1;
    for (int n = 0; n < crossover_point; ++n) {
      population[child_1 * number_of_variables + n] =
        population[parent_1 * number_of_variables + n];
      population[child_2 * number_of_variables + n] =
        population[parent_2 * number_of_variables + n];
    }
    for (int n = crossover_point; n < number_of_variables; ++n) {
      population[child_1 * number_of_variables + n] =
        population[parent_2 * number_of_variables + n];
      population[child_2 * number_of_variables + n] =
        population[parent_1 * number_of_variables + n];
    }
  }
}

void GA::mutation(int n)
{
  float rate_n = mutation_rate * (1.0 - float(n) / maximum_generation);
  int m = population_size * number_of_variables;
  int number_of_mutations = round(m * rate_n);
  for (int n = 0; n < number_of_mutations; ++n) {
    std::uniform_int_distribution<int> r1(number_of_variables, m - 1);
    std::uniform_real_distribution<float> r2(0, 1);
    population[r1(rng)] = r2(rng);
  }
}

int GA::get_a_parent(void)
{
  int parent = 0;
  std::uniform_real_distribution<float> r1(0, 1);
  float reference_value = r1(rng);
  for (int n = 0; n < parent_number; ++n) {
    if (cumulative_probabilities[n] > reference_value) {
      parent = n;
      break;
    }
  }
  return parent;
}
