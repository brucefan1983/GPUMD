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

#include "parameters.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>

Parameters::Parameters()
{
  is_prediction_set = false;
  is_lambda_1_set = false;
  is_lambda_2_set = false;
  is_batch_set = false;
  is_population_set = false;
  is_seed_set = false;
  is_generation_set = false;
  is_save_potential_set = false;
  is_output_interval_set = false;
  prediction = 0;              // not prediction mode
  lambda_1 = lambda_2 = -1.0f; // automatic regularization
  batch_size = 1000;           // large enough in most cases
  population_size = 50;        // almost optimal
  seed = -1;                   // negative keeps the built-in seeds
  maximum_generation = 100000; // a good starting point
  save_potential = 100000;     // write checkpoint nep.txt files at these intervals
  save_potential_format = 1;   // include time stamp in checkpoint names
  save_potential_restart = 0;  // do not write checkpoint restart files by default
  output_interval = 100;       // write loss.out, nep.txt, nep.restart (and related output) every N generations
  initial_para = 1.0f;
  sigma0 = 0.1f;
  int deviceCount;  
  CHECK(gpuGetDeviceCount(&deviceCount));  
  int population_remainder = population_size % deviceCount;
  if (population_remainder != 0) {
    int population_increase = deviceCount - population_remainder;
    population_size += population_increase;
    printf("Default population size adjusted from 50 to %d for GPU compatibility.\n", population_size);  
  }  
}

bool Parameters::parse_keyword(const char** param, int num_param)
{
  if (strcmp(param[0], "prediction") == 0) {
    parse_prediction(param, num_param);
  } else if (strcmp(param[0], "lambda_1") == 0) {
    parse_lambda_1(param, num_param);
  } else if (strcmp(param[0], "lambda_2") == 0) {
    parse_lambda_2(param, num_param);
  } else if (strcmp(param[0], "batch") == 0) {
    parse_batch(param, num_param);
  } else if (strcmp(param[0], "population") == 0) {
    parse_population(param, num_param);
  } else if (strcmp(param[0], "seed") == 0) {
    parse_seed(param, num_param);
  } else if (strcmp(param[0], "generation") == 0) {
    parse_generation(param, num_param);
  } else if (strcmp(param[0], "save_potential") == 0) {
    parse_save_potential(param, num_param);
  } else if (strcmp(param[0], "output_interval") == 0) {
    parse_output_interval(param, num_param);
  } else if (strcmp(param[0], "initial_para") == 0) {
    parse_initial_para(param, num_param);
  } else if (strcmp(param[0], "sigma0") == 0) {
    parse_sigma0(param, num_param);
  } else {
    return false;
  }
  return true;
}

void Parameters::set_regularization_defaults()
{
  if (!is_lambda_1_set) {
    lambda_1 = sqrt(number_of_variables * 1.0e-6f / num_types);
  }
  if (!is_lambda_2_set) {
    lambda_2 = sqrt(number_of_variables * 1.0e-6f / num_types);
  }
}

void Parameters::parse_prediction(const char** param, int num_param)
{
  is_prediction_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("prediction should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &prediction)) {
    PRINT_INPUT_ERROR("prediction should be an integer.\n");
  }
  if (prediction != 0 && prediction != 1) {
    PRINT_INPUT_ERROR("prediction should = 0 or 1.");
  }
}

void Parameters::parse_lambda_1(const char** param, int num_param)
{
  is_lambda_1_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_1 should have 1 parameter.\n");
  }

  double lambda_1_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_1_tmp)) {
    PRINT_INPUT_ERROR("L1 regularization loss weight should be a number.\n");
  }
  lambda_1 = lambda_1_tmp;

  if (lambda_1 < 0.0f) {
    PRINT_INPUT_ERROR("L1 regularization loss weight should >= 0.");
  }
}

void Parameters::parse_lambda_2(const char** param, int num_param)
{
  is_lambda_2_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("lambda_2 should have 1 parameter.\n");
  }

  double lambda_2_tmp = 0.0;
  if (!is_valid_real(param[1], &lambda_2_tmp)) {
    PRINT_INPUT_ERROR("L2 regularization loss weight should be a number.\n");
  }
  lambda_2 = lambda_2_tmp;

  if (lambda_2 < 0.0f) {
    PRINT_INPUT_ERROR("L2 regularization loss weight should >= 0.");
  }
}

void Parameters::parse_batch(const char** param, int num_param)
{
  is_batch_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("batch should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &batch_size)) {
    PRINT_INPUT_ERROR("batch size should be an integer.\n");
  }
  if (batch_size < 1) {
    PRINT_INPUT_ERROR("batch size should >= 1.");
  }
}

void Parameters::parse_population(const char** param, int num_param)
{
  is_population_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("The population keyword must be followed by a parameter.\n");
  }
  if (!is_valid_int(param[1], &population_size)) {
    PRINT_INPUT_ERROR("population size should be an integer.\n");
  }
  if (population_size < 10) {
    PRINT_INPUT_ERROR("population size should >= 10.");
  } else if (population_size > 200) {
    PRINT_INPUT_ERROR("population size should <= 200.");
  }

  int deviceCount;
  CHECK(gpuGetDeviceCount(&deviceCount));
  int population_remainder = population_size % deviceCount;
  int population_increase;
  if (population_remainder != 0) {
    population_increase = deviceCount - population_remainder;
    population_size += population_increase;
  } else {
    population_increase = 0;
  }
  if (population_increase != 0) {
    printf("The input population size is not divisible by the number of GPUs.\n");
    printf("This causes an inefficient use of resources.\n");
    printf("The population size has therefore been increased to %d.\n", population_size);
  }
}

void Parameters::parse_seed(const char** param, int num_param)
{
  is_seed_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("The seed keyword must be followed by a parameter.\n");
  }
  if (!is_valid_int(param[1], &seed)) {
    PRINT_INPUT_ERROR("seed should be an integer.\n");
  }
  if (seed < 0) {
    PRINT_INPUT_ERROR("seed should be >= 0.\n");
  }
}

void Parameters::parse_generation(const char** param, int num_param)
{
  is_generation_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("generation should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &maximum_generation)) {
    PRINT_INPUT_ERROR("maximum number of generations should be an integer.\n");
  }
  if (maximum_generation < 0) {
    PRINT_INPUT_ERROR("maximum number of generations should >= 0.");
  } else if (maximum_generation > 10000000) {
    PRINT_INPUT_ERROR("maximum number of generations should <= 10000000.");
  }
}

void Parameters::parse_save_potential(const char** param, int num_param)
{
  is_save_potential_set = true;

  if (num_param != 4) {
    PRINT_INPUT_ERROR("save_potential should have 3 parameters.\n");
  }
  if (!is_valid_int(param[1], &save_potential)) {
    PRINT_INPUT_ERROR("save_potential interval should be an integer.\n");
  }
  if (save_potential <= 0) {
    PRINT_INPUT_ERROR("save_potential interval should be > 0.");
  }
  if (!is_valid_int(param[2], &save_potential_format)) {
    PRINT_INPUT_ERROR("save_potential format should be an integer.\n");
  }
  if (save_potential_format != 0 && save_potential_format != 1) {
    PRINT_INPUT_ERROR("save_potential format should be 0 or 1.");
  }  
  if (!is_valid_int(param[3], &save_potential_restart)) {
    PRINT_INPUT_ERROR("save_potential save restart should be an integer.\n");
  }
  if (save_potential_restart != 0 && save_potential_restart != 1) {
    PRINT_INPUT_ERROR("save_potential save restart should be 0 or 1.");
  }
}

void Parameters::parse_output_interval(const char** param, int num_param)
{
  is_output_interval_set = true;

  if (num_param != 2) {
    PRINT_INPUT_ERROR("output_interval should have 1 parameter.\n");
  }
  if (!is_valid_int(param[1], &output_interval)) {
    PRINT_INPUT_ERROR("output_interval should be an integer.\n");
  }
  if (output_interval <= 0) {
    PRINT_INPUT_ERROR("output_interval should be > 0.");
  }
}

void Parameters::parse_initial_para(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("initial_para should have 1 parameter.\n");
  }

  double initial_para_tmp = 0.0;
  if (!is_valid_real(param[1], &initial_para_tmp)) {
    PRINT_INPUT_ERROR("initial_para should be a number.\n");
  }
  initial_para = initial_para_tmp;

  if (initial_para < 0.1f || initial_para > 1.0f) {
    PRINT_INPUT_ERROR("initial_para should be within [0.1, 1].");
  }
}

void Parameters::parse_sigma0(const char** param, int num_param)
{
  if (num_param != 2) {
    PRINT_INPUT_ERROR("sigma0 should have 1 parameter.\n");
  }

  double sigma0_tmp = 0.0;
  if (!is_valid_real(param[1], &sigma0_tmp)) {
    PRINT_INPUT_ERROR("sigma0 should be a number.\n");
  }
  sigma0 = sigma0_tmp;

  if (sigma0 < 0.01f || sigma0 > 0.1f) {
    PRINT_INPUT_ERROR("sigma0 should be within [0.01, 0.1].");
  }
}
