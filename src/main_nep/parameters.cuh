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

// Settings used by the optimizer and scheduler, independent of the model.
class Parameters
{
public:
  Parameters();
  bool parse_keyword(const char** param, int num_param);
  void set_regularization_defaults();

  int batch_size;         // number of configurations in one batch
  int num_types;          // number of atom types
  int population_size;    // population size for SNES
  int seed;               // seed of the SNES random draws; negative keeps the built-in seeds
  int maximum_generation; // maximum number of generations for SNES;
  int save_potential;     // number of generations between writing a checkpoint nep.txt file.
  int save_potential_format;   // format of checkpoint nep.txt file name
  int save_potential_restart;  // if restart files should be written or not. 0=no, 1=yes
  int output_interval;    // number of generations between writing loss.out, nep.txt, nep.restart and related output
  float lambda_1;         // weight parameter for L1 regularization loss
  float lambda_2;         // weight parameter for L2 regularization loss
  int prediction; // 0=no, 1=yes
  float initial_para;
  float sigma0;
  bool is_prediction_set;
  bool is_lambda_1_set;
  bool is_lambda_2_set;
  bool is_batch_set;
  bool is_population_set;
  bool is_seed_set;
  bool is_generation_set;
  bool is_save_potential_set;
  bool is_output_interval_set;
  int number_of_variables;            // total number of parameters (NN and descriptor)

private:
  void parse_prediction(const char** param, int num_param);
  void parse_lambda_1(const char** param, int num_param);
  void parse_lambda_2(const char** param, int num_param);
  void parse_batch(const char** param, int num_param);
  void parse_population(const char** param, int num_param);
  void parse_seed(const char** param, int num_param);
  void parse_generation(const char** param, int num_param);
  void parse_save_potential(const char** param, int num_param);
  void parse_output_interval(const char** param, int num_param);
  void parse_initial_para(const char** param, int num_param);
  void parse_sigma0(const char** param, int num_param);
};
