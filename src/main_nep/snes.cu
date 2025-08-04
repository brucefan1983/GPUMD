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
#include "utilities/gpu_macro.cuh"
#include <chrono>
#include <cmath>
#include <iostream>
#include <cstring>

static __global__ void initialize_curand_states(gpurandState* state, int N, int seed)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    gpurand_init(seed, n, 0, &state[n]);
  }
}

SNES::SNES(Parameters& para, Fitness* fitness_function)
{
  maximum_generation = para.maximum_generation;
  number_of_variables = para.number_of_variables;
  population_size = para.population_size;
  const int N = population_size * number_of_variables;
  int num = number_of_variables;
  if (para.version != 3) {
    num /= para.num_types;
  }
  eta_sigma = (3.0f + std::log(num * 1.0f)) / (5.0f * sqrt(num * 1.0f)) / 2.0f;
  fitness.resize(population_size * 7 * (para.num_types + 1));
  index.resize(population_size * (para.num_types + 1));
  population.resize(N);
  mu.resize(number_of_variables);
  sigma.resize(number_of_variables);
  cost_L1reg.resize(population_size);
  cost_L2reg.resize(population_size);
  utility.resize(population_size);
  type_of_variable.resize(number_of_variables, para.num_types);
  initialize_rng();

  gpuSetDevice(0); // normally use GPU-0
  gpu_type_of_variable.resize(number_of_variables);
  gpu_index.resize(population_size * (para.num_types + 1));
  gpu_utility.resize(number_of_variables);
  gpu_sigma.resize(number_of_variables);
  gpu_mu.resize(number_of_variables);
  gpu_cost_L1reg.resize(population_size);
  gpu_cost_L2reg.resize(population_size);
  gpu_s.resize(N);
  gpu_population.resize(N);
  curand_states.resize(N);
  initialize_curand_states<<<(N - 1) / 128 + 1, 128>>>(curand_states.data(), N, 1234567);
  GPU_CHECK_KERNEL

  if (para.fine_tune) {
    initialize_mu_and_sigma_fine_tune(para);
  } else {
    initialize_mu_and_sigma(para);
  }
  
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
      mu[n] = (r1(rng) - 0.5f) * 2.0f;
      sigma[n] = para.sigma0;
    }
    // make sure the initial charges are zero
    if (para.charge_mode) {
      const int num_full = (para.dim + 3) * para.num_neurons1;
      const int num_part = (para.dim + 2) * para.num_neurons1;
      for (int t = 0; t < para.num_types; ++t) {
        for (int n = num_full * t + num_part; n < num_full * (t + 1); ++n) {
          mu[n] = 0.0f;
        }
      }
      mu[num_full * para.num_types] = 2.0f; // make sure initial sqrt(epsilon_inf) > 0
    }
  } else {
    for (int n = 0; n < number_of_variables; ++n) {
      int count = fscanf(fid_restart, "%f%f", &mu[n], &sigma[n]);
      PRINT_SCANF_ERROR(count, 2, "Reading error for nep.restart.");
    }
    fclose(fid_restart);
  }
  gpuSetDevice(0); // normally use GPU-0
  gpu_mu.copy_from_host(mu.data());
  gpu_sigma.copy_from_host(sigma.data());
}

void SNES::initialize_mu_and_sigma_fine_tune(Parameters& para)
{
  // This map is needed because the foundation model misses 5 elements between H-Pu
  const int element_map[94] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
    20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,
    40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,
    60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,
    80,81,82,0,0,0,0,0,83,84,85,86,87,88
  };
  // read in the whole foundation file first
  const int NUM89 = 89;
  const int num_ann_per_element = (para.dim + (para.charge_mode ? 3 : 2)) * para.num_neurons1;
  const int num_ann = NUM89 * num_ann_per_element + (para.charge_mode ? 2 : 1);
  const int num_cnk_radial = NUM89 * NUM89 * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
  const int num_cnk_angular = NUM89 * NUM89 * (para.n_max_angular + 1) * (para.basis_size_angular + 1);
  const int num_tot = num_ann + num_cnk_radial + num_cnk_angular;
  std::vector<float> restart_mu(num_tot);
  std::vector<float> restart_sigma(num_tot);

  std::ifstream input(para.fine_tune_nep_restart);
  if (!input.is_open()) {
    std::cout << "Cannot open the foundation model file " << para.fine_tune_nep_restart << std::endl;
    exit(1);
  }
  std::vector<std::string> tokens;
    
  for (int n = 0; n < num_tot; ++n) {
    tokens = get_tokens(input);
    if (tokens.size() != 2) {
      std::cout << "Foundation model file should have two columns.\n";
      exit(1);
    }
    restart_mu[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    restart_sigma[n] = get_double_from_token(tokens[1], __FILE__, __LINE__);
  }

  // get the required part
  int count = 0;
  for (int i = 0; i < para.num_types; ++ i) {
    int element_index = element_map[para.atomic_numbers[i] - 1];
    for (int j = 0; j < num_ann_per_element; ++j) {
      mu[count] = restart_mu[element_index * num_ann_per_element + j];
      sigma[count] = restart_sigma[element_index * num_ann_per_element + j];
      ++count;
    }
  }
  ++count; // the global bias

  // radial descriptors
  for (int n = 0; n <= para.n_max_radial; ++n) {
    for (int k = 0; k <= para.basis_size_radial; ++k) {
      int nk = n * (para.basis_size_radial + 1) + k;
      for (int t1 = 0; t1 < para.num_types; ++t1) {
        for (int t2 = 0; t2 < para.num_types; ++t2) {
          int element_index_1 = element_map[para.atomic_numbers[t1] - 1];
          int element_index_2 = element_map[para.atomic_numbers[t2] - 1];
          int t12 = element_index_1 * NUM89 + element_index_2;
          mu[count] = restart_mu[nk * NUM89 * NUM89 + t12 + num_ann];
#ifdef FINE_TUNE_DESCRIPTOR
          sigma[count] = restart_sigma[nk * NUM89 * NUM89 + t12 + num_ann];
#else
          sigma[count] = 0.0f * restart_sigma[nk * NUM89 * NUM89 + t12 + num_ann];
#endif
          ++count;
        }
      }
    }
  }

  // angular descriptors
  for (int n = 0; n <= para.n_max_angular; ++n) {
    for (int k = 0; k <= para.basis_size_angular; ++k) {
      int nk = n * (para.basis_size_angular + 1) + k;
      for (int t1 = 0; t1 < para.num_types; ++t1) {
        for (int t2 = 0; t2 < para.num_types; ++t2) {
          int element_index_1 = element_map[para.atomic_numbers[t1] - 1];
          int element_index_2 = element_map[para.atomic_numbers[t2] - 1];
          int t12 = element_index_1 * NUM89 + element_index_2;
          mu[count] = restart_mu[nk * NUM89 * NUM89 + t12 + num_ann + num_cnk_radial];
#ifdef FINE_TUNE_DESCRIPTOR
          sigma[count] = restart_sigma[nk * NUM89 * NUM89 + t12 + num_ann + num_cnk_radial];
#else
          sigma[count] = 0.0f * restart_sigma[nk * NUM89 * NUM89 + t12 + num_ann + num_cnk_radial];
#endif
          ++count;
        }
      }
    }
  }

  input.close();
  gpuSetDevice(0); // normally use GPU-0
  gpu_mu.copy_from_host(mu.data());
  gpu_sigma.copy_from_host(sigma.data());
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
  int num_para_ann_per_type = (para.dim + 2) * para.num_neurons1;
  if (para.charge_mode) {
    num_para_ann_per_type += para.num_neurons1;
  }

  int offset = 0;

  // NN part
  if (para.version != 3) {
    int num_ann = (para.train_mode == 2) ? 2 : 1;
    for (int ann = 0; ann < num_ann; ++ann) {
      for (int t = 0; t < para.num_types; ++t) {
        for (int n = 0; n < num_para_ann_per_type; ++n) {
          type_of_variable[n + offset] = t;
        }
        offset += num_para_ann_per_type;
      }
      offset += para.charge_mode ? 2 : 1; // the bias
    }
  } else {
    offset += num_para_ann_per_type + 1;
  }

  // descriptor part
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

void SNES::compute(Parameters& para, Fitness* fitness_function)
{

  print_line_1();
  if (para.prediction == 0) {
    printf("Started training.\n");
  } else {
    printf("Started predicting.\n");
  }

  print_line_2();

  if (para.prediction == 0) {

    if (para.train_mode == 0 || para.train_mode == 3) {
      printf(
        "%-8s%-11s%-11s%-11s%-13s%-13s%-13s%-13s%-13s%-13s\n",
        "Step",
        "Total-Loss",
        "L1Reg-Loss",
        "L2Reg-Loss",
        "RMSE-E-Train",
        "RMSE-F-Train",
        "RMSE-V-Train",
        "RMSE-E-Test",
        "RMSE-F-Test",
        "RMSE-V-Test");
    } else {
      printf(
        "%-8s%-11s%-11s%-11s%-13s%-13s\n",
        "Step",
        "Total-Loss",
        "L1Reg-Loss",
        "L2Reg-Loss",
        "RMSE-P-Train",
        "RMSE-P-Test");
    }
  }

  if (para.prediction == 0) {
    for (int n = 0; n < maximum_generation; ++n) {
      create_population(para);
      fitness_function->compute(n, para, population.data(), fitness.data());

      if (para.version != 3) {
        regularize_NEP4(para);
      } else {
        regularize(para);
      }

      sort_population(para);

      int best_index = index[para.num_types * population_size];
      float fitness_total = fitness[0 + (7 * para.num_types + 0) * population_size];
      float fitness_L1 = fitness[best_index + (7 * para.num_types + 1) * population_size];
      float fitness_L2 = fitness[best_index + (7 * para.num_types + 2) * population_size];
      fitness_function->report_error(
        para,
        n,
        fitness_total,
        fitness_L1,
        fitness_L2,
        population.data() + number_of_variables * best_index);

      update_mu_and_sigma(para);
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
    tokens = get_tokens(input);
    int num_lines_to_be_skipped = 5;
    if (
      tokens[0] == "nep3_zbl" || 
      tokens[0] == "nep4_zbl" || 
      tokens[0] == "nep3_zbl_temperature" ||
      tokens[0] == "nep4_zbl_temperature" || 
      tokens[0] == "nep3_zbl_charge1" ||
      tokens[0] == "nep3_zbl_charge2" ||
      tokens[0] == "nep3_zbl_charge3" ||
      tokens[0] == "nep4_zbl_charge1" ||
      tokens[0] == "nep4_zbl_charge2" ||
      tokens[0] == "nep4_zbl_charge3") {
      num_lines_to_be_skipped = 6;
    }

    for (int n = 0; n < num_lines_to_be_skipped; ++n) {
      tokens = get_tokens(input);
    }
    for (int n = 0; n < number_of_variables; ++n) {
      tokens = get_tokens(input);
      population[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    for (int d = 0; d < para.dim; ++d) {
      tokens = get_tokens(input);
      para.q_scaler_cpu[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    para.q_scaler_gpu[0].copy_from_host(para.q_scaler_cpu.data());
    fitness_function->predict(para, population.data());
  }
}

static __global__ void gpu_create_population(
  const int N,
  const int number_of_variables,
  const float* g_mu,
  const float* g_sigma,
  gpurandState* g_state,
  float* g_s,
  float* g_population)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    int v = n % number_of_variables;
    gpurandState state = g_state[n];
    float s = gpurand_normal(&state);
    g_s[n] = s;
    g_population[n] = g_sigma[v] * s + g_mu[v];
    g_state[n] = state;
  }
}

void SNES::create_population(Parameters& para)
{
  gpuSetDevice(0); // normally use GPU-0
  const int N = population_size * number_of_variables;
  gpu_create_population<<<(N - 1) / 128 + 1, 128>>>(
    N,
    number_of_variables,
    gpu_mu.data(),
    gpu_sigma.data(),
    curand_states.data(),
    gpu_s.data(),
    gpu_population.data());
  GPU_CHECK_KERNEL
  gpu_population.copy_to_host(population.data());
}

static __global__ void gpu_find_L1_L2_NEP4(
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
    if (g_type_of_variable[v] == g_type || g_type == g_num_types) {
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

void SNES::regularize_NEP4(Parameters& para)
{
  gpuSetDevice(0); // normally use GPU-0

  for (int t = 0; t <= para.num_types; ++t) {
    float num_variables = float(para.number_of_variables) / para.num_types;
    if (t == para.num_types) {
      num_variables = para.number_of_variables;
    }

    gpu_find_L1_L2_NEP4<<<population_size, 1024>>>(
      number_of_variables,
      para.num_types,
      t,
      gpu_type_of_variable.data(),
      gpu_population.data(),
      gpu_cost_L1reg.data(),
      gpu_cost_L2reg.data());
    GPU_CHECK_KERNEL

    gpu_cost_L1reg.copy_to_host(cost_L1reg.data());
    gpu_cost_L2reg.copy_to_host(cost_L2reg.data());

    for (int p = 0; p < population_size; ++p) {
      float cost_L1 = para.lambda_1 * cost_L1reg[p] / num_variables;
      float cost_L2 = para.lambda_2 * sqrt(cost_L2reg[p] / num_variables);
      fitness[p + (7 * t + 0) * population_size] =
        cost_L1 + cost_L2 + fitness[p + (7 * t + 3) * population_size] +
        fitness[p + (7 * t + 4) * population_size] + fitness[p + (7 * t + 5) * population_size] +
        fitness[p + (7 * t + 6) * population_size];
      fitness[p + (7 * t + 1) * population_size] = cost_L1;
      fitness[p + (7 * t + 2) * population_size] = cost_L2;
    }
  }
}

static __global__ void gpu_find_L1_L2(
  const int number_of_variables,
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
    s_cost_L1reg[tid] += abs(para);
    s_cost_L2reg[tid] += para * para;
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

void SNES::regularize(Parameters& para)
{
  gpuSetDevice(0); // normally use GPU-0
  gpu_find_L1_L2<<<population_size, 1024>>>(
    number_of_variables, gpu_population.data(), gpu_cost_L1reg.data(), gpu_cost_L2reg.data());
  GPU_CHECK_KERNEL
  gpu_cost_L1reg.copy_to_host(cost_L1reg.data());
  gpu_cost_L2reg.copy_to_host(cost_L2reg.data());

  for (int p = 0; p < population_size; ++p) {
    float cost_L1 = para.lambda_1 * cost_L1reg[p] / number_of_variables;
    float cost_L2 = para.lambda_2 * sqrt(cost_L2reg[p] / number_of_variables);

    for (int t = 0; t <= para.num_types; ++t) {
      fitness[p + (7 * t + 0) * population_size] =
        cost_L1 + cost_L2 + fitness[p + (7 * t + 3) * population_size] +
        fitness[p + (7 * t + 4) * population_size] + fitness[p + (7 * t + 5) * population_size] +
        fitness[p + (7 * t + 6) * population_size];
      fitness[p + (7 * t + 1) * population_size] = cost_L1;
      fitness[p + (7 * t + 2) * population_size] = cost_L2;
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
      fitness.data() + t * population_size * 7,
      index.data() + t * population_size,
      population_size);
  }
}

static __global__ void gpu_update_mu_and_sigma(
  const int population_size,
  const int number_of_variables,
  const float eta_sigma,
  const float simga0,
  const int* g_type_of_variable,
  const int* g_index,
  const float* g_utility,
  const float* g_s,
  float* g_mu,
  float* g_sigma)
{
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < number_of_variables) {
    const int type = g_type_of_variable[v];
    float gradient_mu = 0.0f, gradient_sigma = 0.0f;
    for (int p = 0; p < population_size; ++p) {
      const int pv = g_index[type * population_size + p] * number_of_variables + v;
      const float utility = g_utility[p];
      const float s = g_s[pv];
      gradient_mu += s * utility;
      gradient_sigma += (s * s - 1.0f) * utility;
    }
    const float sigma = g_sigma[v];
    g_mu[v] += sigma * gradient_mu;
    g_sigma[v] = min(simga0, sigma * exp(eta_sigma * gradient_sigma));
  }
}

void SNES::update_mu_and_sigma(Parameters& para)
{
  gpuSetDevice(0); // normally use GPU-0
  gpu_type_of_variable.copy_from_host(type_of_variable.data());
  gpu_index.copy_from_host(index.data());
  gpu_utility.copy_from_host(utility.data());
  gpu_update_mu_and_sigma<<<(number_of_variables - 1) / 128 + 1, 128>>>(
    population_size,
    number_of_variables,
    eta_sigma,
    para.sigma0,
    gpu_type_of_variable.data(),
    gpu_index.data(),
    gpu_utility.data(),
    gpu_s.data(),
    gpu_mu.data(),
    gpu_sigma.data());
  GPU_CHECK_KERNEL;
}

void SNES::output_mu_and_sigma(Parameters& para)
{
  gpuSetDevice(0); // normally use GPU-0
  gpu_mu.copy_to_host(mu.data());
  gpu_sigma.copy_to_host(sigma.data());
  FILE* fid_restart = my_fopen("nep.restart", "w");
  for (int n = 0; n < number_of_variables; ++n) {
    fprintf(fid_restart, "%15.7e %15.7e\n", mu[n], sigma[n]);
  }
  fclose(fid_restart);
}
