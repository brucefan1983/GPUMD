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
#include <iostream>

SNES::SNES(Parameters& para, Fitness* fitness_function)
{
  maximum_generation = para.maximum_generation;
  number_of_variables = para.number_of_variables;
  population_size = para.population_size;
  eta_sigma = (3.0f + std::log(number_of_variables * 1.0f)) /
              (5.0f * sqrt(number_of_variables * 1.0f)) / 2.0f;
  fitness.resize(population_size * 6 * (para.num_types + 1));
  index.resize(population_size * (para.num_types + 1));
  population.resize(population_size * number_of_variables);
  s.resize(population_size * number_of_variables);
  mu.resize(number_of_variables);
  sigma.resize(number_of_variables);
  utility.resize(population_size);
  type_of_variable.resize(number_of_variables, para.num_types);
  initialize_rng();
  initialize_mu_and_sigma(para);
  calculate_utility();
  find_type_of_variable(para);
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
      sigma[n] = 0.1f;
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

void SNES::find_type_of_variable(Parameters& para)
{
  if (para.version == 4) {
    int offset = 0;
    int num_ann = (para.train_mode == 2) ? 2 : 1;
    for (int ann = 0; ann < num_ann; ++ann) {
      for (int t = 0; t < para.num_types; ++t) {
        for (int n = 0; n < (para.dim + 2) * para.num_neurons1; ++n) {
          type_of_variable[n + offset] = t;
        }
        offset += (para.dim + 2) * para.num_neurons1;
      }
      ++offset; // the bias
    }
    for (int n = 0; n <= para.n_max_radial; ++n) {
      for (int k = 0; k <= para.basis_size_radial; ++k) {
        int nk = n * (para.basis_size_radial + 1) + k;
        for (int t1 = 0; t1 < para.num_types; ++t1) {
          for (int t2 = 0; t2 < para.num_types; ++t2) {
            int t12 = t1 * para.num_types + t2;
            type_of_variable[nk * para.num_types * para.num_types + t12 + offset] = t1;
          }
        }
      }
    }
    offset +=
      (para.n_max_radial + 1) * (para.basis_size_radial + 1) * para.num_types * para.num_types;
    for (int n = 0; n <= para.n_max_angular; ++n) {
      for (int k = 0; k <= para.basis_size_angular; ++k) {
        int nk = n * (para.basis_size_angular + 1) + k;
        for (int t1 = 0; t1 < para.num_types; ++t1) {
          for (int t2 = 0; t2 < para.num_types; ++t2) {
            int t12 = t1 * para.num_types + t2;
            type_of_variable[nk * para.num_types * para.num_types + t12 + offset] = t1;
          }
        }
      }
    }
  }
}

void SNES::compute(Parameters& para, Fitness* fitness_function)
{

  print_line_1();
  if (para.prediction == 0) {
    printf("Started training.\n");
  } else {
    printf("Started predcting.\n");
  }

  print_line_2();

  if (para.prediction == 0) {

    if (para.train_mode == 0) {
      printf(
        "%-8s%-11s%-11s%-11s%-13s%-13s%-13s%-13s%-13s%-13s\n", "Step", "Total-Loss", "L1Reg-Loss",
        "L2Reg-Loss", "RMSE-E-Train", "RMSE-F-Train", "RMSE-V-Train", "RMSE-E-Test", "RMSE-F-Test",
        "RMSE-V-Test");
    } else {
      printf(
        "%-8s%-11s%-11s%-11s%-13s%-13s\n", "Step", "Total-Loss", "L1Reg-Loss", "L2Reg-Loss",
        "RMSE-P-Train", "RMSE-P-Test");
    }
  }

  if (para.prediction == 0) {
    for (int n = 0; n < maximum_generation; ++n) {
      create_population(para);
      fitness_function->compute(n, para, population.data(), fitness.data());

      regularize(para);
      sort_population(para);

      int best_index = index[para.num_types * population_size];
      float fitness_total = fitness[0 + (6 * para.num_types + 0) * population_size];
      float fitness_L1 = fitness[best_index + (6 * para.num_types + 1) * population_size];
      float fitness_L2 = fitness[best_index + (6 * para.num_types + 2) * population_size];
      fitness_function->report_error(
        para, n, fitness_total, fitness_L1, fitness_L2,
        population.data() + number_of_variables * best_index);

      update_mu_and_sigma();
      if (0 == (n + 1) % 100) {
        output_mu_and_sigma(para);
      }
    }
  } else {
    std::ifstream input("nep.txt");
    if (!input.is_open()) {
      PRINT_INPUT_ERROR("Failed to open nep.txt.");
    }
    std::vector<std::string> tokens;
    for (int n = 0; n < 6; ++n) {
      tokens = get_tokens(input);
    }
    for (int n = 0; n < number_of_variables; ++n) {
      tokens = get_tokens(input);
      population[n] = get_float_from_token(tokens[0], __FILE__, __LINE__);
    }
    for (int d = 0; d < para.dim; ++d) {
      tokens = get_tokens(input);
      para.q_scaler_cpu[d] = get_float_from_token(tokens[0], __FILE__, __LINE__);
    }
    para.q_scaler_gpu[0].copy_from_host(para.q_scaler_cpu.data());
    fitness_function->predict(para, population.data());
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
    }
  }
}

void SNES::regularize(Parameters& para)
{
  float lambda_1 = para.lambda_1;
  float lambda_2 = para.lambda_2;
  if (para.lambda_1 < 0.0f || para.lambda_2 < 0.0f) {
    float auto_reg = 1.0e30f;
    for (int p = 0; p < population_size; ++p) {
      float temp = fitness[p + (6 * para.num_types + 3) * population_size] +
                   fitness[p + (6 * para.num_types + 4) * population_size] +
                   fitness[p + (6 * para.num_types + 5) * population_size];
      if (auto_reg > temp) {
        auto_reg = temp;
      }
    }
    if (para.lambda_1 < 0.0f) {
      lambda_1 = auto_reg;
    }
    if (para.lambda_2 < 0.0f) {
      lambda_2 = auto_reg;
    }
  }

  for (int p = 0; p < population_size; ++p) {
    float cost_L1 = 0.0f, cost_L2 = 0.0f;
    for (int v = 0; v < number_of_variables; ++v) {
      int pv = p * number_of_variables + v;
      cost_L1 += std::abs(population[pv]);
      cost_L2 += population[pv] * population[pv];
    }

    cost_L1 *= lambda_1 / number_of_variables;
    cost_L2 = lambda_2 * sqrt(cost_L2 / number_of_variables);

    for (int t = 0; t <= para.num_types; ++t) {
      fitness[p + (6 * t + 0) * population_size] =
        cost_L1 + cost_L2 + fitness[p + (6 * t + 3) * population_size] +
        fitness[p + (6 * t + 4) * population_size] + fitness[p + (6 * t + 5) * population_size];
      fitness[p + (6 * t + 1) * population_size] = cost_L1;
      fitness[p + (6 * t + 2) * population_size] = cost_L2;
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

void SNES::sort_population(Parameters& para)
{
  for (int t = 0; t < para.num_types + 1; ++t) {
    for (int n = 0; n < population_size; ++n) {
      index[t * population_size + n] = n;
    }

    insertion_sort(
      fitness.data() + t * population_size * 6, index.data() + t * population_size,
      population_size);
  }
}

void SNES::update_mu_and_sigma()
{
  for (int v = 0; v < number_of_variables; ++v) {
    int type = type_of_variable[v];
    float gradient_mu = 0.0f, gradient_sigma = 0.0f;
    for (int p = 0; p < population_size; ++p) {
      int pv = index[type * population_size + p] * number_of_variables + v;
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
